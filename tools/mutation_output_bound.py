#!/usr/bin/env python3
"""Mutation testing ciblé — lot output-bound-hardening.

Un test qui passe ne prouve rien tant qu'on n'a pas montré qu'il peut ÉCHOUER.
Ce harnais casse volontairement le mécanisme de budget, un défaut à la fois, et
vérifie que la suite le détecte. Un mutant SURVIVANT est un trou de couverture.

Chaque mutation reproduit une régression plausible en revue ou en refactor —
pas une faute arbitraire :

  M1  le défaut du profil n'est plus appliqué      -> retour à l'illimité (la RCA)
  M2  la borne maximale n'est plus vérifiée        -> override arbitraire accepté
  M3  `>` devient `>=` sur la borne                -> refuse la valeur autorisée
  M4  `num_predict` est retiré avant l'envoi       -> Ollama ne reçoit rien
  M5  `done_reason` n'est plus relayé              -> troncature invisible
  M6  tout `done_reason` est écrasé en "stop"      -> troncature maquillée en succès
  M7  le registre accepte un profil sans budget    -> modèle non borné servi
  M8  0 / -1 ne sont plus refusés                  -> sentinelle « illimité » acceptée

Usage (lecture seule sur le dépôt : tout se passe dans une copie temporaire) :
    python3 tools/mutation_output_bound.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (id, fichier, motif recherché, remplacement, description)
MUTATIONS = [
    (
        "M1-no-default",
        "app/main.py",
        "        return generation.default_num_predict, None",
        "        return None, None  # MUTANT: plus de défaut => illimité",
        "le défaut du profil n'est plus appliqué",
    ),
    (
        "M2-no-max-bound",
        "app/main.py",
        "    if raw > generation.max_num_predict:",
        "    if False:  # MUTANT: borne maximale supprimée",
        "la borne maximale n'est plus vérifiée",
    ),
    (
        "M3-boundary-off-by-one",
        "app/main.py",
        "    if raw > generation.max_num_predict:",
        "    if raw >= generation.max_num_predict:  # MUTANT: borne exclusive",
        "`>` devient `>=` (refuse la valeur autorisée)",
    ),
    (
        "M4-drop-from-payload",
        "app/main.py",
        '    options["num_predict"] = effective_num_predict',
        "    pass  # MUTANT: num_predict jamais posé dans les options",
        "`num_predict` n'atteint plus le payload Ollama",
    ),
    (
        "M5-ignore-done-reason",
        "app/main.py",
        '    value = data.get("done_reason")',
        '    value = None  # MUTANT: done_reason ignoré',
        "`done_reason` n'est plus relayé",
    ),
    (
        "M6-force-stop",
        "app/main.py",
        "    if isinstance(value, str) and value.strip():\n        return value.strip()",
        '    if isinstance(value, str) and value.strip():\n        return "stop"  # MUTANT: tout devient stop',
        "tout `done_reason` est écrasé en \"stop\"",
    ),
    (
        "M7-optional-generation-profile",
        "app/registry.py",
        '                     "keep_alive_profile", "generation_profile",',
        '                     "keep_alive_profile",  # MUTANT: budget optionnel',
        "le registre accepte un modèle sans budget déclaré",
    ),
    (
        "M8-allow-unbounded-sentinel",
        "app/main.py",
        "    if raw <= 0:",
        "    if False:  # MUTANT: 0 et -1 acceptés",
        "les sentinelles « illimité » ne sont plus refusées",
    ),
]

TESTS = [
    "tests/test_generation_output_bound.py",
    "tests/test_llm_registry.py",
    "tests/test_runtime_hardening.py",
    "tests/test_generate_stream.py",
    "tests/test_generate.py",
]

IMAGE = "ai_gateway_testrunner:local"


def run_tests(root: Path) -> tuple[bool, str]:
    """True si la suite PASSE dans cette copie."""
    proc = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{root}:/src:ro", "-w", "/src", IMAGE,
            "python", "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider",
        ],
        capture_output=True, text=True,
    )
    return proc.returncode == 0, proc.stdout[-600:]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        shutil.copytree(REPO, base, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "venv", ".venv"
        ))

        print("== contrôle : suite verte sans mutation ==")
        ok, tail = run_tests(base)
        if not ok:
            print("ABANDON : la suite échoue déjà sans mutation.")
            print(tail)
            return 2
        print("   suite verte — les mutants suivants doivent la casser\n")

        killed, survived = [], []
        for mid, relpath, needle, replacement, description in MUTATIONS:
            work = Path(tmp) / mid
            shutil.copytree(base, work)
            target = work / relpath
            source = target.read_text(encoding="utf-8")
            if needle not in source:
                print(f"[{mid}] MOTIF INTROUVABLE dans {relpath} — mutation non appliquée")
                survived.append((mid, description, "motif introuvable"))
                shutil.rmtree(work)
                continue
            target.write_text(source.replace(needle, replacement, 1), encoding="utf-8")

            passed, tail = run_tests(work)
            if passed:
                print(f"[{mid}] SURVIVANT  — {description}")
                survived.append((mid, description, tail.strip().splitlines()[-1:]))
            else:
                print(f"[{mid}] TUÉ        — {description}")
                killed.append(mid)
            shutil.rmtree(work)

        print(f"\n== bilan : {len(killed)} tués / {len(MUTATIONS)} ==")
        for mid, description, detail in survived:
            print(f"  SURVIVANT {mid} : {description} ({detail})")
        return 0 if not survived else 1


if __name__ == "__main__":
    sys.exit(main())
