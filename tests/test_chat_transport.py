"""Lot C1 — transport `OllamaClient.chat` / `open_chat_stream` (POST /api/chat).

Ce que ces tests figent : l'endpoint réellement appelé, la forme EXACTE du
payload (messages réduits à role+content, `stream` cohérent, `think` /
`keep_alive` / `temperature` relayés, jamais `tools` ni `format`), la
conversion des pannes en `UpstreamError` / `UpstreamTimeout` sans URL
interne, et le lecteur de flux partagé libellé `/api/chat`.
"""

import json
import time

import pytest
import requests

from app.errors import InvalidUpstreamChunk, StreamTimeout, UpstreamError, UpstreamTimeout
from app.ollama_client import ENDPOINT_CHAT, ChatStream, GenerateStream, OllamaClient

CLIENT = OllamaClient("http://127.0.0.1:11434", timeout=30)
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": ""}, {"role": "user", "content": "u2"}]


class _Resp:
    def __init__(self, payload=None, status=200, text="", lines=None):
        self._payload = payload
        self.status_code = status
        self.text = text
        self._lines = lines or []
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} Server Error")
            err.response = self
            raise err

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def close(self):
        self.closed = True


def _post_recorder(monkeypatch, resp):
    calls = []

    def fake_post(url, json=None, timeout=None, stream=None):
        calls.append({"url": url, "json": json, "timeout": timeout, "stream": stream})
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)
    return calls


# --------------------------------------------------------------------------- #
# 1. Endpoint et payload
# --------------------------------------------------------------------------- #
def test_chat_posts_to_api_chat_not_generate(monkeypatch):
    calls = _post_recorder(monkeypatch, _Resp({"message": {"content": "x"}, "done": True}))
    CLIENT.chat(model="m", messages=MESSAGES)
    assert calls[0]["url"].endswith(ENDPOINT_CHAT)
    assert "/api/generate" not in calls[0]["url"]
    assert calls[0]["stream"] is None            # requête HTTP non streamée
    assert calls[0]["json"]["stream"] is False   # champ Ollama


def test_chat_payload_is_exact(monkeypatch):
    calls = _post_recorder(monkeypatch, _Resp({"message": {"content": "x"}, "done": True}))
    CLIENT.chat(model="m", messages=MESSAGES, temperature=0.1,
                options={"num_ctx": 8192, "num_predict": 64}, think="low", keep_alive="2m")
    body = calls[0]["json"]
    assert body == {
        "model": "m",
        "messages": MESSAGES,
        "stream": False,
        "options": {"num_ctx": 8192, "num_predict": 64, "temperature": 0.1},
        "think": "low",
        "keep_alive": "2m",
    }


def test_chat_payload_omits_absent_fields(monkeypatch):
    calls = _post_recorder(monkeypatch, _Resp({"message": {"content": "x"}, "done": True}))
    CLIENT.chat(model="m", messages=MESSAGES[:2])
    body = calls[0]["json"]
    assert set(body) == {"model", "messages", "stream", "options"}
    assert body["options"] == {}


def test_chat_keep_alive_zero_is_forwarded(monkeypatch):
    calls = _post_recorder(monkeypatch, _Resp({"message": {"content": "x"}, "done": True}))
    CLIENT.chat(model="m", messages=MESSAGES[:2], keep_alive=0)
    assert calls[0]["json"]["keep_alive"] == 0


@pytest.mark.parametrize("think", [True, False, "low", "medium", "high"])
def test_chat_relays_validated_think_values(monkeypatch, think):
    calls = _post_recorder(monkeypatch, _Resp({"message": {"content": "x"}, "done": True}))
    CLIENT.chat(model="m", messages=MESSAGES[:2], think=think)
    assert calls[0]["json"]["think"] == think


def test_chat_messages_are_reduced_to_role_and_content(monkeypatch):
    """Un champ parasite sur un message (images, tool_calls, name) ne part
    JAMAIS vers Ollama : le transport reconstruit chaque message."""
    calls = _post_recorder(monkeypatch, _Resp({"message": {"content": "x"}, "done": True}))
    CLIENT.chat(model="m", messages=[
        {"role": "user", "content": "u", "images": ["AAAA"], "tool_calls": [{"x": 1}], "name": "n"},
    ])
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "u"}]


@pytest.mark.parametrize("bad", [{"role": "tool", "content": "r"}, {"role": "user", "content": 3},
                                 {"role": None, "content": "u"}, {"content": "u"}])
def test_chat_refuses_unrelayable_messages_before_any_network(monkeypatch, bad):
    calls = _post_recorder(monkeypatch, _Resp({"message": {"content": "x"}, "done": True}))
    with pytest.raises(ValueError):
        CLIENT.chat(model="m", messages=[bad])
    assert calls == []


def test_chat_transport_has_no_tools_parameter_and_format_only_aggregated():
    """`tools` (lot C4) n'est un paramètre d'aucun transport chat. `format`
    (lot C3) n'existe que sur le chemin AGRÉGÉ : `open_chat_stream` ne peut
    pas le relayer — la sortie structurée en flux n'existe pas."""
    import inspect
    for fn in (OllamaClient.chat, OllamaClient.open_chat_stream, OllamaClient._build_chat_payload):
        assert "tools" not in set(inspect.signature(fn).parameters), fn.__name__
    assert "format" not in set(inspect.signature(OllamaClient.open_chat_stream).parameters)
    assert "format" in set(inspect.signature(OllamaClient.chat).parameters)


def test_chat_and_stream_build_the_same_payload_but_for_stream():
    a = OllamaClient._build_chat_payload(model="m", messages=MESSAGES, stream=False,
                                         temperature=0.2, options={"num_ctx": 4096}, think=True, keep_alive="1m")
    b = OllamaClient._build_chat_payload(model="m", messages=MESSAGES, stream=True,
                                         temperature=0.2, options={"num_ctx": 4096}, think=True, keep_alive="1m")
    assert a["stream"] is False and b["stream"] is True
    a.pop("stream"); b.pop("stream")
    assert a == b


def test_chat_payload_does_not_mutate_caller_options(monkeypatch):
    _post_recorder(monkeypatch, _Resp({"message": {"content": "x"}, "done": True}))
    options = {"num_ctx": 4096}
    CLIENT.chat(model="m", messages=MESSAGES[:2], temperature=0.3, options=options)
    assert options == {"num_ctx": 4096}


# --------------------------------------------------------------------------- #
# 2. Réponse et pannes
# --------------------------------------------------------------------------- #
def test_chat_returns_raw_body(monkeypatch):
    payload = {"message": {"role": "assistant", "content": "x", "thinking": "t"}, "done": True,
               "done_reason": "length", "eval_count": 3}
    _post_recorder(monkeypatch, _Resp(payload))
    assert CLIENT.chat(model="m", messages=MESSAGES[:2]) == payload


def test_chat_http_error_is_upstream_error(monkeypatch):
    _post_recorder(monkeypatch, _Resp({"error": "boom"}, status=500, text='{"error":"boom"}'))
    with pytest.raises(UpstreamError) as info:
        CLIENT.chat(model="m", messages=MESSAGES[:2])
    assert not isinstance(info.value, UpstreamTimeout)
    assert info.value.body == '{"error":"boom"}'


def test_chat_timeout_is_upstream_timeout(monkeypatch):
    _post_recorder(monkeypatch, requests.exceptions.ReadTimeout("HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out"))
    with pytest.raises(UpstreamTimeout):
        CLIENT.chat(model="m", messages=MESSAGES[:2])


def test_chat_connection_error_is_upstream_error_not_timeout(monkeypatch):
    _post_recorder(monkeypatch, requests.exceptions.ConnectionError("HTTPConnectionPool(host='127.0.0.1', port=11434)"))
    with pytest.raises(UpstreamError) as info:
        CLIENT.chat(model="m", messages=MESSAGES[:2])
    assert not isinstance(info.value, UpstreamTimeout)


def test_chat_unreadable_body_is_fixed_message(monkeypatch):
    _post_recorder(monkeypatch, _Resp(ValueError("Expecting value"), text="<html>SECRET-BODY</html>"))
    with pytest.raises(UpstreamError) as info:
        CLIENT.chat(model="m", messages=MESSAGES[:2])
    assert "SECRET-BODY" not in info.value.message
    assert info.value.message == "échec /api/chat: réponse illisible"


def test_chat_non_object_body_is_refused(monkeypatch):
    _post_recorder(monkeypatch, _Resp(["pas", "un", "objet"]))
    with pytest.raises(UpstreamError):
        CLIENT.chat(model="m", messages=MESSAGES[:2])


# --------------------------------------------------------------------------- #
# 3. Flux
# --------------------------------------------------------------------------- #
def test_open_chat_stream_uses_stream_true_on_api_chat(monkeypatch):
    calls = _post_recorder(monkeypatch, _Resp(lines=[b'{"message":{"content":"a"},"done":true}']))
    stream = CLIENT.open_chat_stream(model="m", messages=MESSAGES[:2], think="high")
    assert isinstance(stream, ChatStream) and isinstance(stream, GenerateStream)
    assert calls[0]["url"].endswith(ENDPOINT_CHAT)
    assert calls[0]["stream"] is True
    assert calls[0]["json"]["stream"] is True
    assert calls[0]["json"]["think"] == "high"
    assert [e for e in stream.events()] == [{"message": {"content": "a"}, "done": True}]


def test_open_chat_stream_http_error_closes_response(monkeypatch):
    resp = _Resp({"error": "x"}, status=503, text="unavailable")
    _post_recorder(monkeypatch, resp)
    with pytest.raises(UpstreamError):
        CLIENT.open_chat_stream(model="m", messages=MESSAGES[:2])
    assert resp.closed


def test_open_chat_stream_timeout_is_upstream_timeout(monkeypatch):
    _post_recorder(monkeypatch, requests.exceptions.ConnectTimeout("HTTPConnectionPool(host='127.0.0.1', port=11434)"))
    with pytest.raises(UpstreamTimeout):
        CLIENT.open_chat_stream(model="m", messages=MESSAGES[:2])


def test_chat_stream_transport_failure_is_labelled_api_chat_without_url():
    """P11 sur /api/chat : la rupture mid-stream devient une UpstreamError au
    message FIXE — libellé chat, sans hôte ni port."""
    resp = _Resp(lines=[b'{"message":{"content":"a"},"done":false}',
                        requests.exceptions.ChunkedEncodingError("HTTPConnectionPool(host='127.0.0.1', port=11434): boom")])
    stream = ChatStream(resp, deadline=time.monotonic() + 30)
    it = stream.events()
    assert next(it) == {"message": {"content": "a"}, "done": False}
    with pytest.raises(UpstreamError) as info:
        next(it)
    assert info.value.message == "échec /api/chat: rupture de transport pendant le flux"
    assert "127.0.0.1" not in info.value.message and "11434" not in info.value.message


def test_chat_stream_parse_errors_stay_invalid_chunk():
    stream = ChatStream(_Resp(lines=[b"{pas du json"]), deadline=time.monotonic() + 30)
    with pytest.raises(InvalidUpstreamChunk) as info:
        next(stream.events())
    assert info.value.message == "échec /api/chat: ligne de flux illisible"
    assert "pas du json" not in info.value.message


def test_chat_stream_deadline_is_enforced():
    stream = ChatStream(_Resp(lines=[b'{"message":{"content":"a"}}', b'{"message":{"content":"b"}}']),
                        deadline=time.monotonic() - 1)
    with pytest.raises(StreamTimeout) as info:
        next(stream.events())
    assert info.value.message.startswith("échec /api/chat:")


def test_generate_stream_messages_are_unchanged_by_the_endpoint_parameter():
    """Régression : le libellé historique de /api/generate reste bit pour bit."""
    stream = GenerateStream(_Resp(lines=[b"{pas du json"]), deadline=time.monotonic() + 30)
    with pytest.raises(InvalidUpstreamChunk) as info:
        next(stream.events())
    assert info.value.message == "échec /api/generate: ligne de flux illisible"
    stream = GenerateStream(_Resp(lines=[b"{}"]), deadline=time.monotonic() - 1)
    with pytest.raises(StreamTimeout) as info:
        next(stream.events())
    assert info.value.message == "échec /api/generate: budget de temps global épuisé pendant le flux"
    resp = _Resp(lines=[requests.exceptions.ConnectionError("x")])
    with pytest.raises(UpstreamError) as info:
        next(GenerateStream(resp, deadline=time.monotonic() + 30).events())
    assert info.value.message == "échec /api/generate: rupture de transport pendant le flux"
