"""Lot P4 — `num_ctx` gouverné avec la même discipline que `num_predict`.

Règle unique : LA VALEUR VALIDÉE EST EXACTEMENT LA VALEUR TRANSMISE.

L'ancienne logique passait l'override par `safe_int`, qui confond absent et
invalide et arrondit : `4096.7` était borné sur 4096 mais Ollama recevait
4096.7 ; `True`, `"4096"`, `-5` étaient remplacés en silence par le défaut du
profil ; `0` et 10⁹ partaient sans borne vers un GPU partagé.

Chaque cas accepté vérifie la valeur reçue par le mock Ollama, pas seulement
le code HTTP ; chaque cas refusé vérifie qu'Ollama n'a PAS été appelé et
qu'aucune substitution n'a eu lieu.
"""

import logging

import pytest

from app import main
from app.registry import ContextProfile, RegistryError

OK_PAYLOAD = {"response": "OK", "done": True, "prompt_eval_count": 1, "eval_count": 1}

# Profils exercés : un standard (verifier, via qwen-judge) et long-context
# (via llama-chat, le modèle par défaut). Valeurs lues dans le registre pour
# que le test suive le YAML, pas l'inverse.
STANDARD = ("qwen-judge", "verifier")
LONG = ("llama-chat", "long-context")


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return dict(OK_PAYLOAD)

    monkeypatch.setattr(main.client, "generate", fake_generate)
    return seen


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _event_fields(caplog, name="generate.ok"):
    prefix = f"event={name} "
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(prefix):
            return dict(
                part.split("=", 1) for part in message.split(" ")[1:] if "=" in part
            )
    raise AssertionError(f"événement {name!r} absent")


def _profile(name):
    return main.registry.context_profiles[name]


def _post(clientapp, model, **options):
    body = {"prompt": "x", "model": model}
    if options:
        body["options"] = options
    return clientapp.post("/v1/generate", json=body)


# --------------------------------------------------------------------------- #
# 1. Registre : la borne est obligatoire et effective sur tous les profils
# --------------------------------------------------------------------------- #
def test_every_context_profile_has_an_effective_bound():
    for name, profile in main.registry.context_profiles.items():
        assert isinstance(profile.max_num_ctx, int), name
        assert profile.max_num_ctx >= profile.num_ctx, name


def test_registry_refuses_context_profile_without_bound(tmp_path):
    text = open(main.registry.source_path, encoding="utf-8").read()
    broken = text.replace("    max_num_ctx: 32768\n", "")
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(RegistryError, match="max_num_ctx"):
        main.load_registry(str(path))


# --------------------------------------------------------------------------- #
# 2. Matrice — profil standard ET long-context
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model,profile_name", [STANDARD, LONG])
def test_absent_uses_profile_default(clientapp, captured, logs, model, profile_name):
    r = _post(clientapp, model)
    assert r.status_code == 200
    assert captured["options"]["num_ctx"] == _profile(profile_name).num_ctx
    fields = _event_fields(logs)
    assert fields["effective_num_ctx"] == str(_profile(profile_name).num_ctx)
    assert "num_ctx" not in fields  # rien n'a été demandé


@pytest.mark.parametrize("model,profile_name", [STANDARD, LONG])
def test_valid_override_is_transmitted_verbatim(clientapp, captured, logs, model, profile_name):
    r = _post(clientapp, model, num_ctx=4096)
    assert r.status_code == 200
    sent = captured["options"]["num_ctx"]
    assert sent == 4096 and type(sent) is int
    fields = _event_fields(logs)
    assert fields["num_ctx"] == "4096" and fields["effective_num_ctx"] == "4096"


@pytest.mark.parametrize("model,profile_name", [STANDARD, LONG])
def test_max_is_accepted_inclusive(clientapp, captured, model, profile_name):
    bound = _profile(profile_name).max_num_ctx
    r = _post(clientapp, model, num_ctx=bound)
    assert r.status_code == 200
    assert captured["options"]["num_ctx"] == bound


@pytest.mark.parametrize("model,profile_name", [STANDARD, LONG])
def test_just_below_max_is_accepted(clientapp, captured, model, profile_name):
    bound = _profile(profile_name).max_num_ctx
    r = _post(clientapp, model, num_ctx=bound - 1)
    assert r.status_code == 200
    assert captured["options"]["num_ctx"] == bound - 1


@pytest.mark.parametrize("model,profile_name", [STANDARD, LONG])
def test_above_max_is_rejected(clientapp, captured, model, profile_name):
    bound = _profile(profile_name).max_num_ctx
    r = _post(clientapp, model, num_ctx=bound + 1)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["requested_num_ctx"] == bound + 1
    assert detail["max_num_ctx"] == bound
    assert detail["context_profile"] == profile_name
    assert captured == {}


@pytest.mark.parametrize("value", [True, False, "4096", 4096.7, 4096.0, [4096], {"n": 4096}, None])
@pytest.mark.parametrize("model,profile_name", [STANDARD, LONG])
def test_invalid_type_is_rejected_not_substituted(clientapp, captured, model, profile_name, value):
    """`True` : `isinstance(True, int)` vaut True en Python — un booléen
    franchirait toute borne numérique sans test de type explicite.
    `4096.0` : un float « entier » reste un float, refusé — Ollama recevrait
    un type différent de celui validé."""
    r = _post(clientapp, model, num_ctx=value)
    assert r.status_code == 400
    assert captured == {}  # Ollama pas appelé, donc aucune substitution


@pytest.mark.parametrize("value", [0, -5, -1, -100000])
@pytest.mark.parametrize("model,profile_name", [STANDARD, LONG])
def test_non_positive_is_rejected_not_substituted(clientapp, captured, model, profile_name, value):
    r = _post(clientapp, model, num_ctx=value)
    assert r.status_code == 400
    assert r.json()["detail"]["requested_num_ctx"] == value
    assert captured == {}


def test_huge_value_is_rejected_before_upstream(clientapp, captured):
    r = _post(clientapp, "qwen-large", num_ctx=1_000_000_000)
    assert r.status_code == 400
    assert captured == {}


# --------------------------------------------------------------------------- #
# 3. Alias enregistré : le profil résolu est appliqué de la même façon
# --------------------------------------------------------------------------- #
def test_registered_alias_gets_same_governance(clientapp, captured, logs):
    # `qwen3:14b` est l'alias Ollama de qwen-judge -> profil verifier.
    r = _post(clientapp, "qwen3:14b")
    assert r.status_code == 200
    assert captured["model"] == "qwen3:14b"
    assert captured["options"]["num_ctx"] == _profile("verifier").num_ctx
    assert _event_fields(logs)["context_profile"] == "verifier"

    bound = _profile("verifier").max_num_ctx
    r = _post(clientapp, "qwen3:14b", num_ctx=bound + 1)
    assert r.status_code == 400
    r = _post(clientapp, "qwen3:14b", num_ctx="4096")
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# 4. Invariant central : validé == transmis, même si le dict client diverge
# --------------------------------------------------------------------------- #
def test_transmitted_value_is_the_validated_one(clientapp, captured):
    """Le dict transmis est réécrit depuis la valeur validée. Une régression
    qui validerait une copie et enverrait le dict brut est détectée ici."""
    r = _post(clientapp, "qwen-large", num_ctx=8192, seed=7)
    assert r.status_code == 200
    assert captured["options"] == {"num_ctx": 8192, "seed": 7, "num_predict": 2048}


def test_bound_comes_from_registry_not_code(clientapp, captured, monkeypatch):
    monkeypatch.setitem(
        main.registry.context_profiles,
        "large-general",
        ContextProfile(name="large-general", num_ctx=8192, max_num_ctx=8192),
    )
    assert _post(clientapp, "qwen-large", num_ctx=8192).status_code == 200
    assert _post(clientapp, "qwen-large", num_ctx=8193).status_code == 400


def test_out_of_registry_model_is_untouched(clientapp, captured):
    """Hors registre : relais historique intégral (P3, hors périmètre). Ce
    test fige que P4 n'a PAS étendu la validation à ce chemin."""
    r = _post(clientapp, "unregistered-model:latest", num_ctx="4096")
    assert r.status_code == 200
    assert captured["options"] == {"num_ctx": "4096"}
