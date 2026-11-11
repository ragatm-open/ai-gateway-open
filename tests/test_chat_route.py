"""Lot C1 — `POST /v1/chat` (agrégé) : gouvernance, contrat, contenu vide,
done_reason, confidentialité. Registre SYNTHÉTIQUE (tests/chat_support.py) :
aucun modèle réel ne déclare `chat` en C1.
"""

import logging

import pytest
import requests

from app import main
from app.errors import UpstreamError, UpstreamTimeout
from tests.chat_support import (
    BASIC, LLM_BLOCK, install_chat, install_registry, messages, upstream,
)

CONTENT_SENTINEL = "ZZCHATCONTENTSECRETZZ"
THINK_SENTINEL = "ZZCHATTHINKINGSECRETZZ"
USER_SENTINEL = "ZZCHATUSERSECRETZZ"
FORBIDDEN_IN_BODY = ("127.0.0.1", "11434", "/api/", "HTTPConnectionPool", "urllib3", "Traceback")


@pytest.fixture
def reg(monkeypatch, tmp_path):
    return install_registry(monkeypatch, tmp_path)


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    caplog.set_level(logging.INFO, logger="ai_gateway.ollama")
    return caplog


def _post(clientapp, **body):
    payload = {"model": "chat-test", "messages": BASIC, **body}
    return clientapp.post("/v1/chat", json=payload)


def _event(caplog, name):
    prefix = f"event={name} "
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(prefix):
            parts = message.split(" ")[1:]
            return dict(part.split("=", 1) for part in parts if "=" in part)
    return None


def _all_logs(caplog):
    return "\n".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# 1. Transport : /api/chat, messages intacts, multi-tours
# --------------------------------------------------------------------------- #
def test_chat_calls_api_chat_transport_with_messages_intact(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    generate_called = []
    monkeypatch.setattr(main.client, "generate", lambda **kw: generate_called.append(kw))
    r = _post(clientapp)
    assert r.status_code == 200, r.text
    assert generate_called == []                       # jamais /api/generate
    assert captured["model"] == "chat-model:latest"
    assert captured["messages"] == BASIC
    assert captured["options"] == {"num_ctx": 16384, "num_predict": 3072}
    assert captured["keep_alive"] == "30m"
    assert captured["temperature"] == 0.2
    assert captured["think"] is None


def test_multi_turn_messages_are_transmitted_without_loss(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    turns = messages(("system", "S"), ("user", "U1"), ("assistant", "A1"), ("user", "U2"),
                     ("assistant", ""), ("user", "U3"))
    r = _post(clientapp, messages=turns)
    assert r.status_code == 200
    assert captured["messages"] == turns


def test_chat_resolves_physical_name_too(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    for name in ("chat-model", "chat-model:latest"):
        r = _post(clientapp, model=name)
        assert r.status_code == 200, r.text
        assert captured["model"] == "chat-model:latest"


# --------------------------------------------------------------------------- #
# 2. Capability opt-in, registre obligatoire
# --------------------------------------------------------------------------- #
def test_model_without_chat_capability_is_refused_before_ollama(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    for name in ("nochat-test", "nochat-model:7b"):
        r = _post(clientapp, model=name)
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "chat_not_supported"
        assert r.json()["detail"]["capabilities"] == ["completion", "tools"]
    assert captured == {}
    assert main.counters.snapshot()["chat_rejected_capability_total"] == 2


@pytest.mark.parametrize("strict", [False, True])
def test_unregistered_model_is_refused_regardless_of_strict_flag(clientapp, reg, monkeypatch, strict, logs):
    """Aucun chemin legacy pour /v1/chat : strict ou pas, hors registre = 400."""
    monkeypatch.setattr(main.settings, "generation_strict_registry", strict)
    captured = install_chat(monkeypatch)
    r = _post(clientapp, model="inconnu:latest")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "model_not_registered"
    assert r.json()["detail"]["chat_requires_registry"] is True
    assert captured == {}
    snap = main.counters.snapshot()
    assert snap["chat_rejected_unregistered_total"] == 1
    assert snap.get("generate_legacy_model_total", 0) == 0
    assert _event(logs, "chat.model_not_registered") == {"model": "inconnu:latest"}


CHAT_MODELS = {
    # Registre EMBARQUÉ : les seuls modèles `chat`, tous staging-only.
    "oss-reasoning": ["chat", "structured_output"],          # lot C2 / C3
    "qwen-reasoning": ["chat", "structured_output"],         # intégration 2026-09-27 (structured après canaries)
}


def test_only_the_c2_staging_model_has_chat_in_the_embedded_registry(clientapp, monkeypatch):
    """Registre EMBARQUÉ : `oss-reasoning` (lot C2) et `qwen-reasoning`
    (2026-09-27) déclarent `chat` ; tous les autres sont refusés, zéro appel
    Ollama."""
    captured = install_chat(monkeypatch)
    for logical, profile in main.registry.llm_models.items():
        if logical in CHAT_MODELS:
            assert profile.capabilities == CHAT_MODELS[logical], logical
            assert profile.production_allowed is True and profile.staging_allowed is True, logical
            continue
        assert "chat" not in profile.capabilities
        r = _post(clientapp, model=logical)
        assert r.status_code == 400 and r.json()["detail"]["code"] == "chat_not_supported"
    assert captured == {}
    assert not any(p.ollama_name.startswith("glm") for p in main.registry.llm_models.values())


def test_environment_allowlist_is_enforced(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    monkeypatch.setattr(main.settings, "environment", "production")
    r = _post(clientapp, model="chat-staging-only")
    assert r.status_code == 400 and "non autorisé" in r.json()["detail"]["message"]
    assert captured == {}
    assert main.counters.snapshot()["chat_rejected_environment_total"] == 1
    monkeypatch.setattr(main.settings, "environment", "staging")
    assert _post(clientapp, model="chat-staging-only").status_code == 200


# --------------------------------------------------------------------------- #
# 3. Réutilisation des règles gouvernées (num_ctx, num_predict, keep_alive)
# --------------------------------------------------------------------------- #
def test_num_ctx_and_num_predict_reuse_generate_rules(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    r = _post(clientapp, options={"num_ctx": 32768, "num_predict": 8192, "top_p": 0.5})
    assert r.status_code == 200
    assert captured["options"] == {"num_ctx": 32768, "num_predict": 8192, "top_p": 0.5}
    for bad in ({"num_ctx": 32769}, {"num_predict": 8193}, {"num_predict": 0}, {"num_predict": -1},
                {"num_ctx": "4096"}, {"num_predict": True}):
        captured.clear()
        r = _post(clientapp, options=bad)
        assert r.status_code == 400, bad
        assert captured == {}


def test_keep_alive_policy_matches_generate(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    assert _post(clientapp, keep_alive=0).status_code == 200 and captured["keep_alive"] == 0
    r = _post(clientapp, model="chat-locked", keep_alive="-1")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "keep_alive_not_overridable"
    assert main.counters.snapshot()["chat_rejected_keep_alive_total"] == 1
    captured.clear()
    assert _post(clientapp, model="chat-locked").status_code == 200
    assert captured["keep_alive"] == "2m"


def test_caller_options_are_not_mutated(clientapp, reg, monkeypatch):
    install_chat(monkeypatch)
    body = {"model": "chat-test", "messages": BASIC, "options": {"top_p": 0.5}}
    clientapp.post("/v1/chat", json=body)
    assert body["options"] == {"top_p": 0.5}


# --------------------------------------------------------------------------- #
# 4. Contrat de réponse : content / thinking / done_reason / content_empty
# --------------------------------------------------------------------------- #
def test_response_shape_content_only(clientapp, reg, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "Réponse."}))
    body = _post(clientapp).json()
    assert set(body) == {"provider", "upstream_provider", "model", "message", "thinking",
                         "thinking_exposed", "thinking_chars", "reasoning", "done", "done_reason",
                         "content_empty", "tokens", "usage", "cost"}
    assert body["reasoning"] is None
    assert body["message"] == {"role": "assistant", "content": "Réponse."}
    assert body["thinking"] is None and body["thinking_chars"] == 0 and body["thinking_exposed"] is False
    assert body["done_reason"] == "stop" and body["content_empty"] is False and body["done"] is True
    assert body["tokens"] == {"prompt": 12, "completion": 5, "total": 17}
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    assert body["model"] == "chat-model:latest"


def test_response_thinking_and_content_are_never_merged(clientapp, reg, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "Final.",
                                                 "thinking": THINK_SENTINEL}))
    body = _post(clientapp).json()
    assert body["message"]["content"] == "Final."
    assert THINK_SENTINEL not in body["message"]["content"]
    # C1 sans profil de raisonnement : la trace n'est jamais exposée, mais
    # son existence est dite (taille), pas son texte.
    assert body["thinking"] is None and body["thinking_exposed"] is False
    assert body["thinking_chars"] == len(THINK_SENTINEL)
    assert body["content_empty"] is False


def test_case_b_thinking_only_and_length_is_content_empty(clientapp, reg, monkeypatch, logs):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "",
                                                 "thinking": "longue réflexion " * 20},
                                       done_reason="length", eval_count=300))
    r = _post(clientapp)
    assert r.status_code == 200                      # résultat VALIDE, incomplet
    body = r.json()
    assert body["content_empty"] is True
    assert body["done_reason"] == "length"
    assert body["message"]["content"] == ""
    assert body["thinking_chars"] > 0
    assert main.counters.snapshot()["chat_empty_content_total"] == 1
    ev = _event(logs, "chat.empty_content")
    assert ev["done_reason"] == "length" and ev["model"] == "chat-model:latest"
    assert "réflexion" not in _all_logs(logs)


def test_case_a_empty_content_without_thinking(clientapp, reg, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": ""}, done_reason="stop"))
    body = _post(clientapp).json()
    assert body["content_empty"] is True and body["done_reason"] == "stop" and body["thinking_chars"] == 0


def test_whitespace_only_content_is_empty(clientapp, reg, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": " \n\t"}))
    body = _post(clientapp).json()
    assert body["content_empty"] is True and body["message"]["content"] == " \n\t"


def test_unknown_done_reason_is_relayed_verbatim(clientapp, reg, monkeypatch, logs):
    install_chat(monkeypatch, upstream(done_reason="context_shift_weird"))
    body = _post(clientapp).json()
    assert body["done_reason"] == "context_shift_weird"
    assert _event(logs, "chat.ok")["done_reason"] == "context_shift_weird"


def test_absent_done_reason_stays_null(clientapp, reg, monkeypatch, logs):
    data = upstream(); data.pop("done_reason")
    install_chat(monkeypatch, data)
    assert _post(clientapp).json()["done_reason"] is None
    assert "done_reason" not in _event(logs, "chat.ok")


def test_malformed_message_shapes_still_serialize(clientapp, reg, monkeypatch):
    for message in (None, "texte", 42, {"content": 7}, {"thinking": ["x"]}, {}):
        install_chat(monkeypatch, upstream(message=message))
        r = _post(clientapp)
        assert r.status_code == 200, (message, r.text)
        assert r.json()["content_empty"] is True and r.json()["thinking"] is None


def test_missing_token_counts_default_to_zero(clientapp, reg, monkeypatch):
    data = upstream(); data.pop("prompt_eval_count"); data.pop("eval_count")
    install_chat(monkeypatch, data)
    assert _post(clientapp).json()["tokens"] == {"prompt": 0, "completion": 0, "total": 0}


# --------------------------------------------------------------------------- #
# 5. Hors contrat C1 : tools, format, think brut, stream, rôle tool
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("extra", [
    {"tools": [{"type": "function", "function": {"name": "f"}}]},
    {"format": "json"}, {"response_format": {"type": "json"}}, {"json_schema": {}},
    {"think": True}, {"think": "high"}, {"stream": True}, {"prompt": "x"},
])
def test_fields_outside_c1_contract_are_refused_without_upstream_call(clientapp, reg, monkeypatch, extra):
    captured = install_chat(monkeypatch)
    r = _post(clientapp, **extra)
    assert r.status_code == 400, extra
    assert captured == {}


@pytest.mark.parametrize("bad_messages", [
    [], [{"role": "tool", "content": "r"}], [{"role": "user"}], [{"role": "user", "content": "   "}],
    [{"role": "system", "content": ""}], [{"role": "user", "content": "x", "images": ["AAAA"]}],
    [{"role": "user", "content": "x", "tool_calls": []}], [{"role": "user", "content": 5}],
    [{"role": "assistant", "content": "a", "name": "n"}],
])
def test_invalid_messages_are_refused(clientapp, reg, monkeypatch, bad_messages):
    captured = install_chat(monkeypatch)
    r = _post(clientapp, messages=bad_messages)
    assert r.status_code == 400, bad_messages
    assert captured == {}


def test_model_is_required(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    r = clientapp.post("/v1/chat", json={"messages": BASIC})
    assert r.status_code == 400 and captured == {}
    r = clientapp.post("/v1/chat", json={"model": "bad name\n", "messages": BASIC})
    assert r.status_code == 400 and captured == {}


def test_unexpected_tool_calls_are_refused_explicitly(clientapp, reg, monkeypatch, logs):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "",
                                                 "tool_calls": [{"function": {"name": "get_weather", "arguments": {}}}]}))
    r = _post(clientapp)
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "unsupported_upstream_response"
    assert "get_weather" not in r.text and "tool_calls" not in r.text
    snap = main.counters.snapshot()
    assert snap["chat_unexpected_tool_calls_total"] == 1 and snap["chat_upstream_errors_total"] == 1
    assert _event(logs, "chat.unexpected_tool_calls")["model"] == "chat-model:latest"


# --------------------------------------------------------------------------- #
# 6. Pannes upstream (P6) et confidentialité
# --------------------------------------------------------------------------- #
def test_timeout_is_a_neutral_502_with_qualified_event(clientapp, reg, monkeypatch, logs):
    install_chat(monkeypatch, exc=UpstreamTimeout("échec /api/chat: HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out"))
    r = _post(clientapp)
    assert r.status_code == 502
    assert r.json()["detail"]["message"] == "échec du service de génération : délai dépassé"
    for token in FORBIDDEN_IN_BODY:
        assert token not in r.text, token
    snap = main.counters.snapshot()
    assert snap["chat_timeouts_total"] == 1 and snap["chat_upstream_errors_total"] == 1
    ev = _event(logs, "chat.timeout")
    assert ev["model"] == "chat-model:latest" and "timeout_seconds" in ev
    assert "HTTPConnectionPool" in _all_logs(logs)   # le détail reste dans les logs


def test_upstream_error_is_a_neutral_502(clientapp, reg, monkeypatch, logs):
    install_chat(monkeypatch, exc=UpstreamError("échec /api/chat: HTTPConnectionPool(host='127.0.0.1', port=11434): refused"))
    r = _post(clientapp)
    assert r.status_code == 502
    assert r.json()["detail"] == {"status": "error", "provider": "ollama",
                                  "message": "échec du service de génération"}
    assert main.counters.snapshot()["chat_timeouts_total"] == 0
    assert main.gauges.snapshot() == {}


def test_real_transport_failure_never_leaks(clientapp, reg, monkeypatch, logs):
    """Panne injectée au niveau `requests.post` : le vrai chemin client."""
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError(
            "HTTPConnectionPool(host='127.0.0.1', port=11434): Max retries exceeded with url: /api/chat")
    monkeypatch.setattr("app.ollama_client.requests.post", boom)
    r = _post(clientapp)
    assert r.status_code == 502
    for token in FORBIDDEN_IN_BODY:
        assert token not in r.text, token


def test_logs_never_contain_messages_content_or_thinking(clientapp, reg, monkeypatch, logs):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": CONTENT_SENTINEL,
                                                 "thinking": THINK_SENTINEL}))
    r = _post(clientapp, messages=messages(("system", USER_SENTINEL + "-sys"), ("user", USER_SENTINEL)))
    assert r.status_code == 200
    text = _all_logs(logs)
    for s in (CONTENT_SENTINEL, THINK_SENTINEL, USER_SENTINEL):
        assert s not in text, s
    ev = _event(logs, "chat.ok")
    assert ev["messages"] == "2" and ev["content_chars"] == str(len(CONTENT_SENTINEL))
    assert ev["thinking_chars"] == str(len(THINK_SENTINEL)) and ev["content_empty"] == "False"
    assert ev["model_logical"] == "chat-test" and ev["context_profile"] == "long-context"
    assert ev["effective_num_predict"] == "3072" and ev["load_state"] == "warm"


def test_inflight_gauge_is_chat_scoped_and_released(clientapp, reg, monkeypatch):
    seen = {}
    def fake(**kw):
        seen["inflight"] = dict(main.gauges.snapshot())
        return upstream()
    monkeypatch.setattr(main.client, "chat", fake)
    _post(clientapp)
    assert seen["inflight"] == {"chat:chat-model:latest": 1}
    assert main.gauges.snapshot() == {}
    assert main.counters.snapshot()["chat_requests_total"] == 1


def test_runtime_inflight_shows_chat_keys(clientapp, reg, monkeypatch):
    main.gauges.inc("chat:chat-model:latest")
    body = clientapp.get("/health/runtime").json()
    assert body["inflight"]["by_model"] == {"chat:chat-model:latest": 1}
    main.gauges.dec("chat:chat-model:latest")


# --------------------------------------------------------------------------- #
# 7. Isolation : /v1/generate n'est pas touché par un modèle chat
# --------------------------------------------------------------------------- #
def test_generate_still_works_on_synthetic_chat_model(clientapp, reg, monkeypatch):
    captured = {}
    monkeypatch.setattr(main.client, "generate", lambda **kw: (captured.update(kw), {"response": "ok", "done": True})[1])
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "chat-test"})
    assert r.status_code == 200
    assert captured["options"] == {"num_ctx": 16384, "num_predict": 3072}
    assert captured["think"] is None
    assert "messages" not in captured
