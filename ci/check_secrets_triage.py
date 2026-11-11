#!/usr/bin/env python3
"""ci/check_secrets_triage.py — porte de sécurité FAIL-CLOSED sur le rapport detect-secrets.

    git ls-files -z | xargs -0 detect-secrets scan [...] > .secrets.json
    python3 ci/check_secrets_triage.py .secrets.json ci/secrets-triage.txt [tracked-files.txt]

Réussit UNIQUEMENT si : le rapport est un JSON detect-secrets valide avec `results` ; chaque signalement
correspond à une exception du fichier de triage par son IDENTITÉ STABLE (chemin | type de détection |
empreinte `hashed_secret` du contenu signalé) ; chaque exception du triage correspond à exactement un
signalement (une exception orpheline — sentinelle modifiée ou retirée — fait échouer : le triage doit
être maintenu) ; le fichier de triage est bien formé ; aucun signalement ne porte sur un fichier hors de
la liste des fichiers suivis (si elle est fournie). Un changement de valeur d'une sentinelle change son
empreinte et invalide l'exception ; un nouveau signalement dans un fichier déjà trié reste bloquant.
Toute erreur (fichier absent, JSON invalide, ligne de triage malformée) → code 1.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def fail(msg: str) -> int:
    print(f"[secret-scan] ÉCHEC : {msg}")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        return fail("usage : check_secrets_triage.py <rapport.json> <triage.txt> [fichiers-suivis.txt]")
    report_path, triage_path = Path(argv[1]), Path(argv[2])
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:  # fichier absent, JSON invalide
        return fail(f"rapport detect-secrets illisible ({exc})")
    if not isinstance(report, dict) or "results" not in report or not isinstance(report["results"], dict):
        return fail("rapport sans clé `results` : le scanner n'a pas produit de rapport valide")
    version = report.get("version", "?")
    findings: list[tuple[str, str, str, int]] = []
    for path, items in report["results"].items():
        for it in items:
            if not isinstance(it, dict) or not {"type", "hashed_secret"} <= it.keys():
                return fail(f"signalement malformé dans {path}")
            findings.append((path.removeprefix("./"), it["type"], it["hashed_secret"], int(it.get("line_number", 0))))

    tracked: set[str] | None = None
    if len(argv) == 4:
        tracked = {l.strip().removeprefix("./") for l in Path(argv[3]).read_text(encoding="utf-8").splitlines() if l.strip()}
        if not tracked:
            return fail("liste des fichiers suivis vide : couverture non prouvée")

    allowed: dict[tuple[str, str, str], str] = {}
    try:
        for n, raw in enumerate(triage_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4 or not all(parts) or not HEX40.match(parts[2]):
                return fail(f"{triage_path}:{n} : attendu `chemin | type | empreinte sha1 (40 hex) | justification`")
            key = (parts[0].removeprefix("./"), parts[1], parts[2])
            if key in allowed:
                return fail(f"{triage_path}:{n} : exception en double")
            allowed[key] = parts[3]
    except OSError as exc:
        return fail(f"fichier de triage illisible ({exc})")

    unknown = [f for f in findings if (f[0], f[1], f[2]) not in allowed]
    matched = {(f[0], f[1], f[2]) for f in findings}
    orphan = [k for k in allowed if k not in matched]
    outside = [f for f in findings if tracked is not None and f[0] not in tracked]

    print(f"[secret-scan] detect-secrets {version} ; fichiers suivis : {len(tracked) if tracked is not None else 'n/a'} ; "
          f"signalements : {len(findings)} ; exceptions triées : {len(allowed)}")
    for f in findings:
        status = "triée" if (f[0], f[1], f[2]) in allowed else "INCONNUE"
        print(f"   {status:8s} {f[0]}:{f[3]} | {f[1]} | {f[2]}")
    rc = 0
    if unknown:
        rc = fail(f"{len(unknown)} signalement(s) non trié(s) — un vrai secret ou une nouvelle sentinelle à examiner")
    if orphan:
        for k in orphan:
            print(f"   orpheline {k[0]} | {k[1]} | {k[2]}")
        rc = fail(f"{len(orphan)} exception(s) de triage sans signalement correspondant (sentinelle modifiée ou retirée) : mettre à jour ci/secrets-triage.txt")
    if outside:
        rc = fail(f"{len(outside)} signalement(s) hors des fichiers suivis par Git : périmètre de scan incorrect")
    if rc == 0:
        print(f"[secret-scan] OK — {len(findings)} signalement(s), tous triés nominativement ; aucun inconnu")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
