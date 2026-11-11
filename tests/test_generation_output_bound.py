"""Lot output-bound-hardening — budget de SORTIE gouverné sur /v1/generate.

Origine : RCA du 2026-09-22. `/v1/generate` n'envoyait aucun `num_predict`,
donc Ollama appliquait son défaut illimité. qwen3:30b-instruct est entré deux
fois dans une boucle non terminante (>11 000 puis 29 250 tokens à ~170 tok/s)
jusqu'au timeout de 180 s de la Gateway.

Invariants testés :

1. aucun `/v1/generate` gouverné ne part sans `num_predict` ;
2. un override explicite passe tel quel, ou est REFUSÉ — jamais clampé ;
3. les valeurs qui signifient « illimité » (0, -1) sont refusées ;
4. le merge des options est déterministe : `num_predict` ne supprime rien ;
5. `done_reason` est relayé verbatim, `length` reste `length` ;
6. l'observabilité expose le budget et la raison d'arrêt, sans jamais
   journaliser le prompt ni le texte produit.
"""

import logging

import pytest

from app import main
from app.errors import UpstreamError, UpstreamTimeout
from app.registry import GenerationProfile, RegistryError

OK_PAYLOAD = {
    "response": "OK",
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 3,
    "eval_count": 2,
    "load_duration": 12_000_000,
}

# Budget gouverné de qwen-large, fixé par ce lot (cf. models_registry.yaml).
QWEN_DEFAULT = 2048
QWEN_MAX = 4096


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
                part.split("=", 1)
                for part in message.split(" ")[1:]
                if "=" in part
            )
    raise AssertionError(f"événement {name!r} absent")


def _qwen(**body):
    return {"prompt": "x", "model": "qwen-large", **body}


# --------------------------------------------------------------------------- #
# 1. Défaut gouverné — le cœur de la correction
# --------------------------------------------------------------------------- #
def test_caller_omits_num_predict_gets_profile_default(clientapp, captured, logs):
    r = clientapp.post("/v1/generate", json=_qwen())
    assert r.status_code == 200
    # C'EST l'invariant du lot : plus aucune génération gouvernée non bornée.
    assert captured["options"]["num_predict"] == QWEN_DEFAULT

    fields = _event_fields(logs)
    assert fields["generation_profile"] == "planner-bounded"
    assert fields["effective_num_predict"] == str(QWEN_DEFAULT)
    # Rien n'a été « demandé » : le champ ne doit pas apparaître.
    assert "requested_num_predict" not in fields


def test_every_governed_llm_model_is_bounded(clientapp, captured):
    """Aucun modèle gouverné ne retombe dans l'illimité — pas seulement qwen."""
    for logical in main.registry.llm_models:
        profile = main.registry.llm_models[logical]
        if not profile.allowed_in(main.settings.environment):
            continue
        captured.clear()
        r = clientapp.post("/v1/generate", json={"prompt": "x", "model": logical})
        if "completion" not in profile.capabilities:
            # Modèle chat seul (oss-reasoning, qwen-reasoning) : jamais relayé sur /v1/generate.
            assert r.status_code == 400 and r.json()["detail"]["code"] == "completion_not_supported", logical
            assert captured == {}
            continue
        assert r.status_code == 200, logical
        num_predict = captured["options"]["num_predict"]
        assert isinstance(num_predict, int) and num_predict > 0, logical


def test_registry_refuses_llm_model_without_generation_profile(tmp_path):
    """Un profil sans budget déclaré fait ÉCHOUER le chargement, il n'est pas
    servi non borné. L'échec est au démarrage, pas à la première requête."""
    yaml_text = (main.registry.source_path and open(main.registry.source_path).read())
    assert yaml_text, "source du registre introuvable"
    broken = yaml_text.replace("    generation_profile: planner-bounded\n", "")
    path = tmp_path / "broken.yaml"
    path.write_text(broken)
    with pytest.raises(RegistryError, match="generation_profile"):
        main.load_registry(str(path))


# --------------------------------------------------------------------------- #
# 2. Override explicite du caller
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [1, 128, 1024, QWEN_DEFAULT, QWEN_MAX])
def test_explicit_num_predict_within_bound_is_propagated(clientapp, captured, value):
    r = clientapp.post(
        "/v1/generate", json=_qwen(options={"num_predict": value})
    )
    assert r.status_code == 200
    # Relayé TEL QUEL : ni clampé, ni remplacé par le défaut.
    assert captured["options"]["num_predict"] == value


def test_explicit_num_predict_is_logged_as_requested(clientapp, captured, logs):
    clientapp.post("/v1/generate", json=_qwen(options={"num_predict": 1024}))
    fields = _event_fields(logs)
    assert fields["requested_num_predict"] == "1024"
    assert fields["effective_num_predict"] == "1024"


def test_num_predict_at_max_is_accepted(clientapp, captured):
    """Borne INCLUSIVE : `> max` refuse, `== max` accepte. Un `>=` mal placé
    refuserait la valeur documentée comme autorisée."""
    r = clientapp.post("/v1/generate", json=_qwen(options={"num_predict": QWEN_MAX}))
    assert r.status_code == 200
    assert captured["options"]["num_predict"] == QWEN_MAX


def test_num_predict_above_max_is_rejected(clientapp, captured):
    r = clientapp.post(
        "/v1/generate", json=_qwen(options={"num_predict": QWEN_MAX + 1})
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["max_num_predict"] == QWEN_MAX
    assert detail["requested_num_predict"] == QWEN_MAX + 1
    # Refus AVANT l'upstream : rien n'est parti chez Ollama.
    assert captured == {}


@pytest.mark.parametrize("value", [0, -1, -2, -1000])
def test_unbounded_or_non_positive_num_predict_is_rejected(clientapp, captured, value):
    """0 et -1 sont les sentinelles Ollama d'une génération illimitée : ce sont
    exactement les valeurs qui ont produit la RCA."""
    r = clientapp.post("/v1/generate", json=_qwen(options={"num_predict": value}))
    assert r.status_code == 400
    assert captured == {}


@pytest.mark.parametrize(
    "value", ["2048", 2048.5, True, False, None, [2048], {"n": 2048}]
)
def test_invalid_type_num_predict_is_rejected(clientapp, captured, value):
    """`True` mérite une mention : `isinstance(True, int)` vaut True en Python,
    donc un booléen franchirait toutes les bornes numériques sans un test de
    type explicite."""
    r = clientapp.post("/v1/generate", json=_qwen(options={"num_predict": value}))
    assert r.status_code == 400
    assert captured == {}


# --------------------------------------------------------------------------- #
# 3. Merge déterministe des options
# --------------------------------------------------------------------------- #
def test_num_predict_does_not_drop_other_options(clientapp, captured):
    r = clientapp.post(
        "/v1/generate",
        json=_qwen(
            options={
                "num_ctx": 8192,
                "top_k": 20,
                "top_p": 0.8,
                "repeat_penalty": 1.0,
                "seed": 42,
            }
        ),
    )
    assert r.status_code == 200
    options = captured["options"]
    # Le budget s'ajoute, il ne remplace pas.
    assert options["num_predict"] == QWEN_DEFAULT
    assert options["num_ctx"] == 8192
    assert options["top_k"] == 20
    assert options["top_p"] == 0.8
    assert options["repeat_penalty"] == 1.0
    assert options["seed"] == 42


def test_num_ctx_and_num_predict_are_independent(clientapp, captured):
    r = clientapp.post(
        "/v1/generate",
        json=_qwen(options={"num_ctx": 8192, "num_predict": 512}),
    )
    assert r.status_code == 200
    assert captured["options"]["num_ctx"] == 8192
    assert captured["options"]["num_predict"] == 512


def test_caller_options_are_not_mutated(clientapp, captured):
    """La Gateway copie les options du client : le dict d'entrée n'est pas
    modifié en place (le plan est construit pour DEUX routes)."""
    payload = _qwen(options={"num_ctx": 8192})
    clientapp.post("/v1/generate", json=payload)
    assert "num_predict" not in payload["options"]


# --------------------------------------------------------------------------- #
# 4. done_reason
# --------------------------------------------------------------------------- #
def test_done_reason_stop_is_preserved(clientapp, monkeypatch, logs):
    monkeypatch.setattr(
        main.client, "generate", lambda **k: dict(OK_PAYLOAD, done_reason="stop")
    )
    r = clientapp.post("/v1/generate", json=_qwen())
    assert r.json()["done_reason"] == "stop"
    assert _event_fields(logs)["done_reason"] == "stop"


def test_done_reason_length_is_preserved_not_normalised(clientapp, monkeypatch, logs):
    """Budget atteint : la réponse est terminée TECHNIQUEMENT mais TRONQUÉE.
    La Gateway ne doit pas la présenter comme sémantiquement complète."""
    monkeypatch.setattr(
        main.client, "generate", lambda **k: dict(OK_PAYLOAD, done_reason="length")
    )
    r = clientapp.post("/v1/generate", json=_qwen())
    body = r.json()
    assert body["done_reason"] == "length"
    assert body["done"] is True  # techniquement terminé
    # Identifiable dans les logs sans relire le texte produit.
    assert _event_fields(logs)["done_reason"] == "length"


def test_done_reason_absent_stays_absent(clientapp, monkeypatch, logs):
    payload = {k: v for k, v in OK_PAYLOAD.items() if k != "done_reason"}
    monkeypatch.setattr(main.client, "generate", lambda **k: dict(payload))
    r = clientapp.post("/v1/generate", json=_qwen())
    assert r.json()["done_reason"] is None
    assert "done_reason" not in _event_fields(logs)


# --------------------------------------------------------------------------- #
# 5. Événement de timeout
# --------------------------------------------------------------------------- #
def test_timeout_emits_qualified_event(clientapp, monkeypatch, logs):
    def boom(**kwargs):
        raise UpstreamTimeout("échec /api/generate: Read timed out. (read timeout=180)")

    monkeypatch.setattr(main.client, "generate", boom)
    r = clientapp.post("/v1/generate", json=_qwen())
    assert r.status_code == 502  # contrat inchangé

    fields = _event_fields(logs, "generate.timeout")
    assert fields["model"] == "qwen3:30b-instruct"
    assert fields["model_logical"] == "qwen-large"
    assert fields["effective_num_predict"] == str(QWEN_DEFAULT)
    assert fields["effective_num_ctx"] == "8192"
    assert "elapsed_ms" in fields


def test_ordinary_upstream_error_is_not_a_timeout(clientapp, monkeypatch, logs):
    """Une panne d'upstream n'est PAS un budget épuisé : confondre les deux
    dans un seul message a coûté une investigation entière."""
    def boom(**kwargs):
        raise UpstreamError("échec /api/generate: connection refused")

    monkeypatch.setattr(main.client, "generate", boom)
    r = clientapp.post("/v1/generate", json=_qwen())
    assert r.status_code == 502
    with pytest.raises(AssertionError):
        _event_fields(logs, "generate.timeout")


# --------------------------------------------------------------------------- #
# 6. Confidentialité des logs
# --------------------------------------------------------------------------- #
def test_logs_never_contain_prompt_or_response(clientapp, monkeypatch, logs):
    secret_prompt = "SECRET_PROMPT_MARKER_12345"
    secret_text = "SECRET_RESPONSE_MARKER_67890"
    monkeypatch.setattr(
        main.client, "generate", lambda **k: dict(OK_PAYLOAD, response=secret_text)
    )
    clientapp.post(
        "/v1/generate", json=_qwen(prompt=secret_prompt, system=secret_text)
    )
    blob = "\n".join(r.getMessage() for r in logs.records)
    assert secret_prompt not in blob
    assert secret_text not in blob
    # La taille du prompt reste journalisée — c'est une métrique, pas du texte.
    assert "prompt_len=" in blob


def test_timeout_logs_never_contain_prompt(clientapp, monkeypatch, logs):
    secret_prompt = "SECRET_TIMEOUT_MARKER_abcdef"

    def boom(**kwargs):
        raise UpstreamTimeout("échec /api/generate: Read timed out.")

    monkeypatch.setattr(main.client, "generate", boom)
    clientapp.post("/v1/generate", json=_qwen(prompt=secret_prompt))
    blob = "\n".join(r.getMessage() for r in logs.records)
    assert secret_prompt not in blob


# --------------------------------------------------------------------------- #
# 7. Contrat du registre
# --------------------------------------------------------------------------- #
def test_qwen_large_profile_values():
    profile = main.registry.llm_models["qwen-large"]
    generation = main.registry.generation_profile_of(profile)
    assert generation.default_num_predict == QWEN_DEFAULT
    assert generation.max_num_predict == QWEN_MAX


def test_generation_profile_rejects_unbounded_declarations():
    from app.registry import _build_generation_profile

    for bad in (0, -1, "2048", 2048.5, True):
        with pytest.raises(RegistryError):
            _build_generation_profile(
                "bad", {"default_num_predict": bad, "max_num_predict": 4096}
            )


def test_generation_profile_rejects_max_below_default():
    from app.registry import _build_generation_profile

    with pytest.raises(RegistryError, match="max_num_predict"):
        _build_generation_profile(
            "bad", {"default_num_predict": 4096, "max_num_predict": 2048}
        )


def test_generation_profiles_exposed_on_health_runtime(clientapp):
    body = clientapp.get("/health/runtime").json()
    profiles = body["generation_profiles"]
    assert profiles["planner-bounded"] == {
        "default_num_predict": QWEN_DEFAULT,
        "max_num_predict": QWEN_MAX,
    }
    entry = [m for m in body["llm_models"] if m["model_logical"] == "qwen-large"][0]
    assert entry["generation_profile"] == "planner-bounded"


def test_generation_profile_exposed_in_public_dict():
    entry = main.registry.llm_models["qwen-large"].to_public_dict()
    assert entry["generation_profile"] == "planner-bounded"


def test_bound_is_taken_from_registry_not_hardcoded(clientapp, captured, monkeypatch):
    """Le plafond appartient à la GOUVERNANCE : changer le profil change le
    comportement. Si la valeur était codée en dur dans le transport, ce test
    échouerait."""
    monkeypatch.setitem(
        main.registry.generation_profiles,
        "planner-bounded",
        GenerationProfile(
            name="planner-bounded", default_num_predict=777, max_num_predict=999
        ),
    )
    r = clientapp.post("/v1/generate", json=_qwen())
    assert r.status_code == 200
    assert captured["options"]["num_predict"] == 777

    r = clientapp.post("/v1/generate", json=_qwen(options={"num_predict": 1000}))
    assert r.status_code == 400
