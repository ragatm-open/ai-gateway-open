"""Lot runtime-hardening — intégration des profils gouvernés dans les routes.

Invariants testés :

1. les défauts gouvernés (num_ctx, keep_alive, think) s'appliquent quand le
   client n'a rien dit, et JAMAIS quand il a parlé — un override n'est pas
   modifié en silence, il passe ou il est refusé en 400 ;
2. la politique think ne s'applique qu'au nom LOGIQUE ;
3. un modèle hors registre garde le comportement historique intégral ;
4. `format` est relayé tel quel (json / schéma), jamais transformé ;
5. l'observabilité (load_state, inflight, compteurs) est correcte, y compris
   sur les chemins d'exception.
"""

import json
import logging

import pytest

from app import main
from app.errors import UpstreamError
from app.metrics import counters, gauges, load_state
from app.ollama_client import EmbedResult, ENDPOINT_EMBED

OK_PAYLOAD = {
    "response": "OK",
    "done": True,
    "prompt_eval_count": 3,
    "eval_count": 2,
    "load_duration": 12_000_000,  # 12 ms -> warm
}


@pytest.fixture
def captured(monkeypatch):
    """Capture les kwargs réellement passés au client Ollama."""
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


# --------------------------------------------------------------------------- #
# 1. Profils de contexte
# --------------------------------------------------------------------------- #
def test_default_model_gets_context_profile(clientapp, captured, logs):
    # Requête historique minimale : le défaut (llama3.1:8b -> llama-chat)
    # reçoit désormais un num_ctx explicite au lieu du défaut runner d'Ollama.
    # 16384 (long-context) et non chat-small : mesuré sur 7 910 requêtes de
    # production, le prompt de chat fait p50=4 966 tokens et max=15 434 —
    # 4096 aurait tronqué la moitié du trafic nominal.
    r = clientapp.post("/v1/generate", json={"prompt": "x"})
    assert r.status_code == 200
    assert captured["options"]["num_ctx"] == 16384

    fields = _event_fields(logs)
    assert fields["model_logical"] == "llama-chat"
    assert fields["context_profile"] == "long-context"
    assert fields["effective_num_ctx"] == "16384"
    # Pas de num_ctx "demandé" : le client n'a rien demandé.
    assert "num_ctx" not in fields


def test_logical_name_resolves_to_ollama_name(clientapp, captured):
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen-judge"})
    assert r.status_code == 200
    assert captured["model"] == "qwen3:14b"
    assert captured["options"]["num_ctx"] == 8192
    assert r.json()["model"] == "qwen3:14b"


def test_client_num_ctx_overrides_profile(clientapp, captured, logs):
    r = clientapp.post(
        "/v1/generate", json={"prompt": "x", "options": {"num_ctx": 16384}}
    )
    assert r.status_code == 200
    # L'override passe tel quel — jamais modifié en silence.
    assert captured["options"]["num_ctx"] == 16384

    fields = _event_fields(logs)
    assert fields["num_ctx"] == "16384"            # demandé
    assert fields["effective_num_ctx"] == "16384"  # appliqué
    assert fields["context_profile"] == "long-context"


def test_client_other_options_are_preserved(clientapp, captured):
    clientapp.post(
        "/v1/generate", json={"prompt": "x", "options": {"num_predict": 5}}
    )
    # Le profil complète les options sans écraser celles du client.
    assert captured["options"]["num_predict"] == 5
    assert captured["options"]["num_ctx"] == 16384


def test_num_ctx_over_profile_bound_is_rejected(clientapp, captured, monkeypatch):
    # Borne posée dynamiquement : les données embarquées n'en fixent aucune
    # (compatibilité), le MÉCANISME doit refuser proprement.
    profile = main.registry.context_profiles["long-context"]
    monkeypatch.setitem(
        main.registry.context_profiles,
        "long-context",
        type(profile)(name="long-context", num_ctx=16384, max_num_ctx=8192),
    )
    r = clientapp.post(
        "/v1/generate", json={"prompt": "x", "options": {"num_ctx": 32768}}
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["provider"] == "ai_gateway"
    assert detail["requested_num_ctx"] == 32768
    assert detail["max_num_ctx"] == 8192
    assert captured == {}  # jamais parti vers Ollama
    assert counters.snapshot()["generate_rejected_num_ctx_total"] == 1


def test_ungoverned_model_keeps_historic_behavior(clientapp, captured, logs):
    # Nom fictif : gemma3:27b est gouverné depuis une version antérieure.
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "unregistered-model:latest"})
    assert r.status_code == 200
    # Aucune option injectée, aucun keep_alive, aucun think.
    assert captured["options"] is None
    assert captured["keep_alive"] is None
    assert captured["think"] is None

    fields = _event_fields(logs)
    assert "context_profile" not in fields
    assert "model_logical" not in fields
    assert "effective_num_ctx" not in fields


# --------------------------------------------------------------------------- #
# 2. keep_alive
# --------------------------------------------------------------------------- #
def test_governed_model_gets_keep_alive_profile(clientapp, captured, logs):
    clientapp.post("/v1/generate", json={"prompt": "x"})
    assert captured["keep_alive"] == "30m"
    assert _event_fields(logs)["keep_alive_profile"] == "resident"


def test_client_keep_alive_overrides_profile(clientapp, captured, logs):
    # 0 = « décharge maintenant », valeur de campagne : elle doit gagner sur le
    # profil resident, et le profil ne doit pas apparaître dans les logs.
    clientapp.post("/v1/generate", json={"prompt": "x", "keep_alive": 0})
    assert captured["keep_alive"] == 0
    assert "keep_alive_profile" not in _event_fields(logs)


def test_transient_profile_applies_to_large_model(clientapp, captured):
    clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen-large"})
    assert captured["model"] == "qwen3:30b-instruct"
    assert captured["keep_alive"] == "2m"
    assert captured["options"]["num_ctx"] == 8192


# --------------------------------------------------------------------------- #
# 3. think — politique du nom logique uniquement
# --------------------------------------------------------------------------- #
def test_logical_judge_defaults_to_think_false(clientapp, captured, logs):
    clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen-judge"})
    assert captured["think"] is False
    fields = _event_fields(logs)
    assert fields["effective_think"] == "False"
    assert "requested_think" not in fields


def test_direct_name_keeps_native_thinking(clientapp, captured, logs):
    # Le MÊME binaire par son nom Ollama : pas de politique imposée.
    clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen3:14b"})
    assert captured["think"] is None
    assert "effective_think" not in _event_fields(logs)


def test_client_think_overrides_judge_policy(clientapp, captured, logs):
    clientapp.post(
        "/v1/generate", json={"prompt": "x", "model": "qwen-judge", "think": True}
    )
    assert captured["think"] is True
    fields = _event_fields(logs)
    assert fields["requested_think"] == "True"
    assert fields["effective_think"] == "True"


# --------------------------------------------------------------------------- #
# 4. format — sorties structurées
# --------------------------------------------------------------------------- #
def test_format_json_is_relayed(clientapp, captured):
    r = clientapp.post("/v1/generate", json={"prompt": "x", "format": "json"})
    assert r.status_code == 200
    assert captured["format"] == "json"


def test_format_schema_is_relayed_verbatim(clientapp, captured):
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "boolean"}},
        "required": ["verdict"],
    }
    r = clientapp.post("/v1/generate", json={"prompt": "x", "format": schema})
    assert r.status_code == 200
    # Relayé STRICTEMENT tel quel : aucune transformation silencieuse.
    assert captured["format"] == schema


def test_format_omitted_is_not_sent(clientapp, captured):
    clientapp.post("/v1/generate", json={"prompt": "x"})
    assert "format" not in captured


def test_format_invalid_string_is_rejected(clientapp, captured):
    r = clientapp.post("/v1/generate", json={"prompt": "x", "format": "xml"})
    assert r.status_code == 400
    assert captured == {}


def test_format_reaches_the_wire(monkeypatch):
    # Au-delà du client simulé : le payload HTTP réel porte bien `format`.
    calls = []

    class _Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return dict(OK_PAYLOAD)

    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None: (calls.append(json), _Resp())[1],
    )
    from app.ollama_client import OllamaClient

    OllamaClient("http://127.0.0.1:11434", timeout=30).generate(
        model="m", prompt="p", format="json"
    )
    assert calls[0]["format"] == "json"
    # Et son absence n'ajoute pas de clé : None-safe.
    OllamaClient("http://127.0.0.1:11434", timeout=30).generate(model="m", prompt="p")
    assert "format" not in calls[1]


def test_format_applies_to_stream_payload(monkeypatch):
    calls = []

    class _Resp:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter([b'{"response":"","done":true}'])

        def close(self):
            pass

    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None, stream=None: (calls.append(json), _Resp())[1],
    )
    from app.ollama_client import OllamaClient

    OllamaClient("http://127.0.0.1:11434", timeout=30).open_generate_stream(
        model="m", prompt="p", format={"type": "object"}
    )
    assert calls[0]["format"] == {"type": "object"}
    assert calls[0]["stream"] is True


# --------------------------------------------------------------------------- #
# 5. Le flux applique les mêmes profils
# --------------------------------------------------------------------------- #
def test_stream_applies_same_governance(clientapp, monkeypatch, logs):
    seen = {}

    class _FakeStream:
        def events(self):
            yield {"response": "ok", "done": False}
            yield dict(OK_PAYLOAD, response="")

        def close(self):
            pass

    def fake_open(**kwargs):
        seen.update(kwargs)
        return _FakeStream()

    monkeypatch.setattr(main.client, "open_generate_stream", fake_open)
    r = clientapp.post(
        "/v1/generate/stream", json={"prompt": "x", "model": "qwen-judge"}
    )
    assert r.status_code == 200
    assert seen["model"] == "qwen3:14b"
    assert seen["options"]["num_ctx"] == 8192
    assert seen["think"] is False
    assert seen["keep_alive"] == "30m"

    fields = _event_fields(logs, "generate_stream.end")
    assert fields["model_logical"] == "qwen-judge"
    assert fields["context_profile"] == "verifier"
    assert fields["effective_num_ctx"] == "8192"
    assert fields["load_state"] == "warm"


def test_environment_allowlist_rejects_forbidden_model(clientapp, captured, monkeypatch):
    profile = main.registry.llm_models["qwen-large"]
    monkeypatch.setattr(profile, "production_allowed", False)
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen-large"})
    assert r.status_code == 400
    assert r.json()["detail"]["model_logical"] == "qwen-large"
    assert captured == {}
    assert counters.snapshot()["generate_rejected_environment_total"] == 1


# --------------------------------------------------------------------------- #
# 6. Observabilité — load_state, inflight, compteurs
# --------------------------------------------------------------------------- #
def test_load_state_derivation():
    assert load_state(0) == "warm"
    assert load_state(12.5) == "warm"
    assert load_state(1000.0) == "warm"
    assert load_state(1000.1) == "cold_or_reload"
    assert load_state(6839.0) == "cold_or_reload"
    assert load_state(None) is None
    assert load_state("beaucoup") is None


def test_generate_logs_load_state(clientapp, captured, logs):
    clientapp.post("/v1/generate", json={"prompt": "x"})
    fields = _event_fields(logs)
    assert fields["load_state"] == "warm"
    # Les valeurs brutes restent journalisées : le label ne les remplace pas.
    assert fields["ollama_load_ms"] == "12.0"


def test_cold_load_is_labelled(clientapp, monkeypatch, logs):
    monkeypatch.setattr(
        main.client,
        "generate",
        lambda **kwargs: dict(OK_PAYLOAD, load_duration=3_600_000_000),
    )
    clientapp.post("/v1/generate", json={"prompt": "x"})
    assert _event_fields(logs)["load_state"] == "cold_or_reload"


def test_missing_load_duration_has_no_label(clientapp, monkeypatch, logs):
    payload = {k: v for k, v in OK_PAYLOAD.items() if k != "load_duration"}
    monkeypatch.setattr(main.client, "generate", lambda **kwargs: payload)
    clientapp.post("/v1/generate", json={"prompt": "x"})
    assert "load_state" not in _event_fields(logs)


def test_inflight_gauge_increments_and_decrements(clientapp, monkeypatch):
    observed = {}

    def fake_generate(**kwargs):
        observed["during"] = gauges.snapshot().get("generate:llama3.1:8b", 0)
        return dict(OK_PAYLOAD)

    monkeypatch.setattr(main.client, "generate", fake_generate)
    clientapp.post("/v1/generate", json={"prompt": "x"})
    assert observed["during"] == 1
    assert gauges.snapshot().get("generate:llama3.1:8b", 0) == 0


def test_inflight_gauge_released_on_upstream_error(clientapp, monkeypatch):
    def boom(**kwargs):
        raise UpstreamError("ollama down")

    monkeypatch.setattr(main.client, "generate", boom)
    r = clientapp.post("/v1/generate", json={"prompt": "x"})
    assert r.status_code == 502
    assert gauges.total() == 0
    assert counters.snapshot()["generate_upstream_errors_total"] == 1


def test_stream_inflight_released_on_open_failure(clientapp, monkeypatch):
    def boom(**kwargs):
        raise UpstreamError("ollama down")

    monkeypatch.setattr(main.client, "open_generate_stream", boom)
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x"})
    assert r.status_code == 502
    assert gauges.total() == 0
    assert counters.snapshot()["generate_stream_errors_total"] == 1


def test_request_counters(clientapp, captured):
    clientapp.post("/v1/generate", json={"prompt": "x"})
    clientapp.post("/v1/generate", json={"prompt": "x"})
    assert counters.snapshot()["generate_requests_total"] == 2


# --------------------------------------------------------------------------- #
# 7. /health/runtime et /v1/models
# --------------------------------------------------------------------------- #
def test_health_runtime_exposes_governance(clientapp):
    r = clientapp.get("/health/runtime")
    assert r.status_code == 200
    body = r.json()

    by_name = {m["model_logical"]: m for m in body["llm_models"]}
    judge = by_name["qwen-judge"]
    assert judge["ollama_name"] == "qwen3:14b"
    assert judge["default_num_ctx"] == 8192
    assert judge["keep_alive"] == "30m"
    assert judge["default_think"] is False
    assert judge["allowed_here"] is True

    assert body["context_profiles"]["chat-small"]["num_ctx"] == 4096
    assert body["context_profiles"]["long-context"]["num_ctx"] == 16384
    assert body["inflight"]["total"] == 0
    # Doctrine : jamais de faux queue_time.
    assert body["queue_time"] == "not_observable"


def test_health_runtime_leaks_nothing_sensitive(clientapp):
    text = clientapp.get("/health/runtime").text
    assert "11434" not in text
    assert "http://" not in text


def test_models_endpoint_carries_llm_metadata(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "list_models",
        lambda timeout=5: [{"name": "llama3.1:8b", "size": 1, "modified_at": "x"}],
    )
    entry = clientapp.get("/v1/models").json()["models"][0]
    assert entry["model_logical"] == "llama-chat"
    assert entry["kind"] == "llm"
    assert entry["resource_class"] == "medium"
    assert entry["context_profile"] == "long-context"
    assert entry["default_num_ctx"] == 16384


# --------------------------------------------------------------------------- #
# 8. Non-régression embeddings (§17 du lot)
# --------------------------------------------------------------------------- #
def test_embeddings_pipeline_untouched_by_llm_governance(clientapp, monkeypatch):
    seen = {}

    def fake_embed(model, inputs, keep_alive=None):
        seen["model"] = model
        seen["inputs"] = inputs
        seen["keep_alive"] = keep_alive
        return EmbedResult([[1.0] + [0.0] * 767 for _ in inputs], ENDPOINT_EMBED)

    monkeypatch.setattr(main.client, "embed", fake_embed)
    r = clientapp.post(
        "/v1/embeddings", json={"texts": ["bonjour"], "task": "query"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline_id"] == main.registry.models["legacy-768"].pipeline_id("query")
    # Préfixe toujours posé par la Gateway ; keep_alive embeddings inchangé
    # (aucun profil LLM ne s'y applique).
    assert seen["inputs"][0].startswith("search_query: ")
    assert seen["keep_alive"] is None


def test_no_sentinel_content_in_logs(clientapp, captured, logs):
    clientapp.post(
        "/v1/generate",
        json={"prompt": "ZZHARDENINGSECRETZZ", "model": "qwen-judge",
              "format": "json"},
    )
    everything = "\n".join(r.getMessage() for r in logs.records)
    assert "ZZHARDENINGSECRETZZ" not in everything
