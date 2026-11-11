"""Lot C1 — `POST /v1/chat/stream` : événements thinking / content / done,
robustesse P11 (rupture, ligne illisible, flux incomplet, budget), tool_calls
inattendus, fermeture de l'upstream, confidentialité, isolation.
"""

import logging

import pytest
import requests

from app import main
from app.errors import InvalidUpstreamChunk, StreamTimeout, UpstreamError
from tests.chat_support import (
    BASIC, STREAM_DONE, FakeChatStream, chat_event, install_chat, install_chat_stream,
    install_registry, ndjson, messages,
)

THINK_SENTINEL = "ZZSTREAMTHINKSECRETZZ"
CONTENT_SENTINEL = "ZZSTREAMCONTENTSECRETZZ"


@pytest.fixture
def reg(monkeypatch, tmp_path):
    return install_registry(monkeypatch, tmp_path)


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _post(clientapp, **body):
    return clientapp.post("/v1/chat/stream", json={"model": "chat-test", "messages": BASIC, **body})


def _end_event(caplog):
    for record in caplog.records:
        m = record.getMessage()
        if m.startswith("event=chat_stream.end "):
            return dict(p.split("=", 1) for p in m.split(" ")[1:] if "=" in p)
    raise AssertionError("chat_stream.end absent")


NOMINAL = [chat_event(thinking="pense "), chat_event(thinking="encore"),
           chat_event(content="Bon"), chat_event(content="jour"), chat_event(content="", **STREAM_DONE)]


# --------------------------------------------------------------------------- #
# 1. Contrat nominal
# --------------------------------------------------------------------------- #
def test_stream_uses_chat_transport_and_same_plan_as_non_stream(clientapp, reg, monkeypatch):
    captured = install_chat_stream(monkeypatch, FakeChatStream(NOMINAL))
    generate_open = []
    monkeypatch.setattr(main.client, "open_generate_stream", lambda **kw: generate_open.append(kw))
    r = _post(clientapp, options={"num_predict": 100})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert generate_open == []
    assert captured["model"] == "chat-model:latest" and captured["messages"] == BASIC
    assert captured["options"] == {"num_ctx": 16384, "num_predict": 100}
    assert captured["keep_alive"] == "30m" and captured["think"] is None
    non_stream = install_chat(monkeypatch)
    clientapp.post("/v1/chat", json={"model": "chat-test", "messages": BASIC, "options": {"num_predict": 100}})
    assert non_stream == captured


def test_stream_emits_content_then_single_done(clientapp, reg, monkeypatch):
    install_chat_stream(monkeypatch, FakeChatStream(NOMINAL))
    events = ndjson(_post(clientapp))
    assert [e["type"] for e in events] == ["content", "content", "done"]
    assert [e["delta"] for e in events[:2]] == ["Bon", "jour"]
    done = events[-1]
    assert done == {"type": "done", "delta": "", "done": True, "model": "chat-model:latest",
                    "tokens": {"prompt": 12, "completion": 5, "total": 17},
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
                    "done_reason": "stop", "content_empty": False, "thinking_exposed": False,
                    "reasoning": None}


def test_stream_thinking_is_not_emitted_without_exposure_policy(clientapp, reg, monkeypatch, logs):
    install_chat_stream(monkeypatch, FakeChatStream([chat_event(thinking=THINK_SENTINEL),
                                                     chat_event(content="ok"), chat_event(**STREAM_DONE)]))
    r = _post(clientapp)
    assert THINK_SENTINEL not in r.text
    assert [e["type"] for e in ndjson(r)] == ["content", "done"]
    end = _end_event(logs)
    assert end["thinking_chars"] == str(len(THINK_SENTINEL)) and end["termination"] == "completed"
    assert THINK_SENTINEL not in "\n".join(x.getMessage() for x in logs.records)


def test_stream_content_empty_when_only_thinking_and_length(clientapp, reg, monkeypatch, logs):
    install_chat_stream(monkeypatch, FakeChatStream([chat_event(thinking="a"), chat_event(thinking="b"),
                                                     chat_event(**{**STREAM_DONE, "done_reason": "length"})]))
    events = ndjson(_post(clientapp))
    assert events == [{"type": "done", "delta": "", "done": True, "model": "chat-model:latest",
                       "tokens": {"prompt": 12, "completion": 5, "total": 17},
                       "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
                       "done_reason": "length", "content_empty": True, "thinking_exposed": False,
                       "reasoning": None}]
    assert main.counters.snapshot()["chat_empty_content_total"] == 1
    end = _end_event(logs)
    assert end["content_empty"] == "True" and end["done_reason"] == "length"


def test_stream_whitespace_only_content_counts_as_empty(clientapp, reg, monkeypatch):
    install_chat_stream(monkeypatch, FakeChatStream([chat_event(content=" "), chat_event(content="\n"),
                                                     chat_event(**STREAM_DONE)]))
    events = ndjson(_post(clientapp))
    assert [e["type"] for e in events] == ["content", "content", "done"]
    assert events[-1]["content_empty"] is True


def test_stream_unknown_done_reason_is_relayed(clientapp, reg, monkeypatch):
    install_chat_stream(monkeypatch, FakeChatStream([chat_event(content="x"),
                                                     chat_event(**{**STREAM_DONE, "done_reason": "odd"})]))
    assert ndjson(_post(clientapp))[-1]["done_reason"] == "odd"


def test_stream_skips_empty_deltas_and_ignores_role_only_events(clientapp, reg, monkeypatch):
    install_chat_stream(monkeypatch, FakeChatStream([{"message": {"role": "assistant"}, "done": False},
                                                     chat_event(content=""), chat_event(content="x"),
                                                     chat_event(**STREAM_DONE)]))
    assert [e["type"] for e in ndjson(_post(clientapp))] == ["content", "done"]


def test_stream_does_not_accumulate_text(clientapp, reg, monkeypatch):
    """Mémoire constante : aucun attribut du générateur ne conserve le texte."""
    big = [chat_event(content="x" * 1000) for _ in range(50)] + [chat_event(**STREAM_DONE)]
    install_chat_stream(monkeypatch, FakeChatStream(big))
    events = ndjson(_post(clientapp))
    assert len(events) == 51 and events[-1]["content_empty"] is False
    import inspect
    src = inspect.getsource(main._chat_stream_events)
    assert "+= content" not in src and "append(content" not in src and "join(" not in src


# --------------------------------------------------------------------------- #
# 2. Erreurs en cours de flux — événement error, jamais de done
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exc,code,termination,counter", [
    (UpstreamError("échec /api/chat: rupture de transport pendant le flux"), "upstream_error", "upstream_error", "chat_stream_errors_total"),
    (InvalidUpstreamChunk("échec /api/chat: ligne de flux illisible"), "invalid_upstream_response", "invalid_upstream_chunk", "chat_stream_errors_total"),
    (StreamTimeout("échec /api/chat: budget de temps global épuisé pendant le flux"), "upstream_timeout", "timeout", "chat_stream_timeouts_total"),
])
def test_mid_stream_failures_become_error_events(clientapp, reg, monkeypatch, logs, exc, code, termination, counter):
    stream = FakeChatStream([chat_event(content="déb"), chat_event(content="ut")], raises=exc)
    install_chat_stream(monkeypatch, stream)
    r = _post(clientapp)
    assert r.status_code == 200
    events = ndjson(r)
    assert [e["type"] for e in events] == ["content", "content", "error"]
    assert events[-1]["error"]["code"] == code
    assert "127.0.0.1" not in r.text and "/api/" not in r.text
    assert stream.closed
    assert _end_event(logs)["termination"] == termination
    assert main.counters.snapshot()[counter] == 1
    assert main.gauges.snapshot() == {}


def test_real_transport_break_through_chat_stream_is_contained(clientapp, reg, monkeypatch, logs):
    """Le VRAI lecteur (ChatStream) sur une réponse qui casse en cours de
    lecture : rupture convertie en événement NDJSON, pas d'exception serveur."""
    from app.ollama_client import ChatStream
    import time, json as _json

    class _Resp:
        closed = False
        def iter_lines(self):
            yield _json.dumps(chat_event(content="a")).encode()
            raise requests.exceptions.ChunkedEncodingError("HTTPConnectionPool(host='127.0.0.1', port=11434): boom")
        def close(self):
            self.closed = True

    resp = _Resp()
    monkeypatch.setattr(main.client, "open_chat_stream",
                        lambda **kw: ChatStream(resp, deadline=time.monotonic() + 30))
    r = _post(clientapp)
    events = ndjson(r)
    assert [e["type"] for e in events] == ["content", "error"]
    assert events[-1]["error"] == {"code": "upstream_error", "message": "flux interrompu par l'upstream"}
    assert resp.closed and "HTTPConnectionPool" not in r.text


def test_stream_incomplete_without_done_is_an_error(clientapp, reg, monkeypatch, logs):
    stream = FakeChatStream([chat_event(content="a")])
    install_chat_stream(monkeypatch, stream)
    events = ndjson(_post(clientapp))
    assert [e["type"] for e in events] == ["content", "error"]
    assert events[-1]["error"]["code"] == "upstream_error"
    assert stream.closed


def test_stream_unexpected_tool_calls_is_an_explicit_error(clientapp, reg, monkeypatch, logs):
    stream = FakeChatStream([chat_event(content="a"),
                             {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}]}, "done": False},
                             chat_event(**STREAM_DONE)])
    install_chat_stream(monkeypatch, stream)
    r = _post(clientapp)
    events = ndjson(r)
    assert [e["type"] for e in events] == ["content", "error"]
    assert events[-1]["error"]["code"] == "unsupported_upstream_response"
    assert "tool_calls" not in r.text and stream.closed
    snap = main.counters.snapshot()
    assert snap["chat_unexpected_tool_calls_total"] == 1 and snap["chat_stream_errors_total"] == 1
    assert _end_event(logs)["termination"] == "unsupported_upstream_response"


def test_stream_error_body_never_leaks_upstream_details(clientapp, reg, monkeypatch):
    install_chat_stream(monkeypatch, FakeChatStream([], raises=UpstreamError(
        "échec /api/chat: HTTPConnectionPool(host='127.0.0.1', port=11434) ZZBODYZZ", body="ZZBODYZZ")))
    r = _post(clientapp)
    for token in ("ZZBODYZZ", "127.0.0.1", "11434", "/api/chat"):
        assert token not in r.text


# --------------------------------------------------------------------------- #
# 3. Avant le premier octet, gouvernance, fermeture, isolation
# --------------------------------------------------------------------------- #
def test_open_failure_is_a_normal_502_and_releases_gauge(clientapp, reg, monkeypatch):
    def boom(**kw):
        raise UpstreamError("échec /api/chat: HTTPConnectionPool(host='127.0.0.1', port=11434)")
    monkeypatch.setattr(main.client, "open_chat_stream", boom)
    r = _post(clientapp)
    assert r.status_code == 502 and r.json()["detail"]["message"] == "échec du service de génération"
    assert "127.0.0.1" not in r.text
    assert main.gauges.snapshot() == {}
    assert main.counters.snapshot()["chat_stream_errors_total"] == 1


def test_stream_governance_refusals_happen_before_open(clientapp, reg, monkeypatch):
    captured = install_chat_stream(monkeypatch, FakeChatStream(NOMINAL))
    assert _post(clientapp, model="nochat-test").status_code == 400
    assert _post(clientapp, model="inconnu").status_code == 400
    assert _post(clientapp, options={"num_predict": 0}).status_code == 400
    assert _post(clientapp, tools=[]).status_code == 400
    assert _post(clientapp, format="json").status_code == 400
    assert captured == {}


def test_stream_closes_upstream_on_success_and_logs_metadata(clientapp, reg, monkeypatch, logs):
    stream = FakeChatStream(NOMINAL)
    install_chat_stream(monkeypatch, stream)
    _post(clientapp)
    assert stream.closed
    end = _end_event(logs)
    assert end["termination"] == "completed" and end["completion_tokens"] == "5"
    assert end["load_state"] == "warm" and "first_chunk_ms" in end and end["model_logical"] == "chat-test"
    assert main.counters.snapshot()["chat_stream_requests_total"] == 1


def test_stream_logs_never_contain_content(clientapp, reg, monkeypatch, logs):
    install_chat_stream(monkeypatch, FakeChatStream([chat_event(thinking=THINK_SENTINEL),
                                                     chat_event(content=CONTENT_SENTINEL), chat_event(**STREAM_DONE)]))
    _post(clientapp, messages=messages(("user", "ZZUSERZZ")))
    text = "\n".join(r.getMessage() for r in logs.records)
    for s in (THINK_SENTINEL, CONTENT_SENTINEL, "ZZUSERZZ"):
        assert s not in text


def test_stream_client_disconnect_closes_upstream(reg, monkeypatch, logs):
    stream = FakeChatStream(NOMINAL)
    plan = main._plan_chat(main.ChatRequest(model="chat-test", messages=BASIC))
    main.gauges.inc("chat_stream:chat-model:latest")
    gen = main._chat_stream_events(stream, plan, "chat-model:latest", 0.0, 0.0)
    next(gen)
    gen.close()
    assert stream.closed
    assert _end_event(logs)["termination"] == "client_disconnect"
    assert main.gauges.snapshot() == {}


def test_generate_stream_is_untouched(clientapp, reg, monkeypatch):
    from tests.test_generate_stream import _FakeStream, DONE_FIELDS
    monkeypatch.setattr(main.client, "open_generate_stream",
                        lambda **kw: _FakeStream([{"response": "OK"}, DONE_FIELDS]))
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x", "model": "chat-test"})
    events = ndjson(r)
    assert [e["type"] for e in events] == ["delta", "done"]
    assert "content_empty" not in events[-1] and "thinking_exposed" not in events[-1]
