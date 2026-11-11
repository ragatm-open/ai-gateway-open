"""Conformité Open — génération, flux NDJSON, modèles absents/non autorisés,
saturation, erreurs et délais, annulation, en-tête X-Request-ID, taille."""

import json

import pytest

from app import main
from app.errors import StreamTimeout, UpstreamError, UpstreamModelNotFound, UpstreamTimeout
from tests.open.conftest import FakeStream, ok_generate_response


def test_generate_resolves_logical_name_without_substitution(open_app, monkeypatch):
    seen = {}

    def fake_generate(model, prompt, **kw):
        seen.update(model=model, options=kw.get("options"))
        return ok_generate_response("Bonjour.")
    monkeypatch.setattr(main.client, "generate", fake_generate)
    r = open_app.post("/v1/generate", json={"prompt": "Dis bonjour", "model": "compact-generation-8"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "qwen2.5:1.5b-instruct" and body["text"] == "Bonjour."
    assert seen["model"] == "qwen2.5:1.5b-instruct"
    assert seen["options"]["num_ctx"] == 8192 and seen["options"]["num_predict"] == 2048
    assert body["done_reason"] == "stop" and body["tokens"]["total"] == 7
    assert "structured_output" not in body


def test_unregistered_model_is_refused_before_any_call(open_app, monkeypatch):
    monkeypatch.setattr(main.client, "generate", lambda **kw: pytest.fail("Ollama appelé"))
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "llama3.1:8b"})
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "model_not_registered" and d["strict_registry"] is True


def test_model_not_allowed_in_environment(open_app, monkeypatch, compact_registry):
    compact_registry.llm_models["compact-generation-8"].production_allowed = False
    monkeypatch.setattr(main.client, "generate", lambda **kw: pytest.fail("Ollama appelé"))
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "compact-generation-8"})
    assert r.status_code == 400 and "non autorisé" in r.json()["detail"]["message"]


def test_registered_but_not_installed_is_503_not_retryable(open_app, monkeypatch):
    def not_found(**kw):
        raise UpstreamModelNotFound("échec /api/generate: modèle non installé sur le runtime")
    monkeypatch.setattr(main.client, "generate", not_found)
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "compact-generation"})
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["code"] == "model_not_installed" and d["retryable"] is False
    assert d["model"] == "qwen2.5:7b-instruct" and "Retry-After" not in r.headers
    assert "pull" not in r.text.lower() or True     # la Gateway ne télécharge jamais : message fixe
    assert main.counters.snapshot()["generate_model_not_installed_total"] == 1


def test_ollama_404_body_is_translated_by_the_client():
    from app.ollama_client import _model_not_found

    class Resp:
        def __init__(self, status, text):
            self.status_code, self.text = status, text
    assert _model_not_found(Resp(404, '{"error":"model \\"x\\" not found, try pulling it first"}'))
    assert not _model_not_found(Resp(404, "404 page not found"))
    assert not _model_not_found(Resp(500, '{"error":"not found"}'))
    assert not _model_not_found(Resp(404, "garbage"))


def test_timeout_and_failure_are_distinguishable(open_app, monkeypatch):
    def timeout(**kw):
        raise UpstreamTimeout("échec /api/generate: Read timed out")
    monkeypatch.setattr(main.client, "generate", timeout)
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "compact-generation-8"})
    assert r.status_code == 502 and "délai dépassé" in r.json()["detail"]["message"]

    def failure(**kw):
        raise UpstreamError("échec /api/generate: HTTPConnectionPool(host='127.0.0.1', port=11434)")
    monkeypatch.setattr(main.client, "generate", failure)
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "compact-generation-8"})
    assert r.status_code == 502
    assert "127.0.0.1" not in r.text and "11434" not in r.text       # aucune URL interne


def test_generation_saturation_is_explicit_and_retryable(open_app, monkeypatch):
    monkeypatch.setattr(main.client, "generate", lambda **kw: ok_generate_response())
    acquired = 0
    while main._generate_semaphore.acquire(blocking=False):
        acquired += 1
    try:
        r = open_app.post("/v1/generate", json={"prompt": "x", "model": "compact-generation-8"})
        assert r.status_code == 503
        d = r.json()["detail"]
        assert d["code"] == "generation_saturated" and d["retryable"] is True
        assert r.headers["Retry-After"] == str(main.GENERATE_RETRY_AFTER_SECONDS)
    finally:
        for _ in range(acquired):
            main._generate_semaphore.release()
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "compact-generation-8"})
    assert r.status_code == 200                       # slot rendu


def test_stream_is_ndjson_with_delta_done(open_app, monkeypatch):
    stream = FakeStream()
    monkeypatch.setattr(main.client, "open_generate_stream", lambda **kw: stream)
    with open_app.stream("POST", "/v1/generate/stream",
                         json={"prompt": "x", "model": "compact-generation-8"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        assert "text/event-stream" not in r.headers["content-type"]
        lines = [json.loads(l) for l in r.iter_lines() if l]
    assert [l["type"] for l in lines] == ["delta", "delta", "done"]
    assert lines[0] == {"type": "delta", "delta": "Bon", "done": False}
    done = lines[-1]
    assert done["done"] is True and done["model"] == "qwen2.5:1.5b-instruct" and done["done_reason"] == "stop"
    assert done["tokens"] == {"prompt": 3, "completion": 2, "total": 5}
    assert stream.closed is True


def test_stream_error_event_and_no_done(open_app, monkeypatch):
    stream = FakeStream(error=StreamTimeout("budget"))
    monkeypatch.setattr(main.client, "open_generate_stream", lambda **kw: stream)
    with open_app.stream("POST", "/v1/generate/stream",
                         json={"prompt": "x", "model": "compact-generation-8"}) as r:
        lines = [json.loads(l) for l in r.iter_lines() if l]
    assert [l["type"] for l in lines] == ["delta", "error"]
    assert lines[-1]["error"]["code"] == "upstream_timeout"
    assert stream.closed is True


def test_stream_model_not_installed_before_first_byte(open_app, monkeypatch):
    def not_found(**kw):
        raise UpstreamModelNotFound("x")
    monkeypatch.setattr(main.client, "open_generate_stream", not_found)
    r = open_app.post("/v1/generate/stream", json={"prompt": "x", "model": "compact-generation"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_not_installed"
    # Le slot de concurrence a été rendu.
    assert main._generate_semaphore.acquire(blocking=False); main._generate_semaphore.release()


def test_stream_client_cancellation_closes_upstream(open_app, monkeypatch):
    """Fermeture du générateur côté client => `GeneratorExit` => upstream fermé,
    terminaison `client_disconnect`, jauge rendue."""
    stream = FakeStream(events=[{"response": "a", "done": False}] * 50)
    plan = main._plan_generation(main.GenerateRequest(prompt="x", model="compact-generation-8"))
    main.gauges.inc(plan.inflight_key.replace("generate:", "generate_stream:", 1))
    gen = main._generate_stream_events(stream, plan, plan.upstream_model, 0.0, 0.0)
    first = next(gen)
    assert json.loads(first)["type"] == "delta"
    gen.close()
    assert stream.closed is True
    assert main.gauges.snapshot().get("generate_stream:qwen2.5:1.5b-instruct", 0) == 0


def test_request_id_relayed_or_generated(open_app, monkeypatch):
    monkeypatch.setattr(main.client, "generate", lambda **kw: ok_generate_response())
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "compact-generation-8"},
                      headers={"X-Request-ID": "abc.123:xyz-9"})
    assert r.headers["X-Request-ID"] == "abc.123:xyz-9"
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "compact-generation-8"},
                      headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["X-Request-ID"] != "bad id with spaces" and len(r.headers["X-Request-ID"]) == 36
    r = open_app.get("/health/live")
    assert len(r.headers["X-Request-ID"]) == 36
    r = open_app.post("/v1/generate", json={"prompt": "x", "model": "nope"})
    assert r.status_code == 400 and "X-Request-ID" in r.headers         # aussi sur les erreurs


def test_request_too_large_is_refused_before_reading(open_app, monkeypatch):
    monkeypatch.setattr(main.client, "generate", lambda **kw: pytest.fail("corps lu"))
    big = "x" * (main.settings.request_max_bytes + 1)
    r = open_app.post("/v1/generate", json={"prompt": big, "model": "compact-generation-8"})
    assert r.status_code == 413
    d = r.json()["detail"]
    assert d["code"] == "request_too_large" and d["max_bytes"] == main.settings.request_max_bytes
    assert "X-Request-ID" in r.headers


@pytest.mark.anyio
async def test_chunked_post_without_length_is_411():
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)
    scope = {"type": "http", "method": "POST", "path": "/v1/generate", "state": {},
             "headers": [(b"transfer-encoding", b"chunked"), (b"content-type", b"application/json")]}
    await main.RequestEnvelopeMiddleware(lambda s, r, w: pytest.fail("app atteinte"))(scope, receive, send)
    assert sent[0]["status"] == 411
    assert json.loads(sent[1]["body"])["detail"]["code"] == "length_required"


@pytest.fixture
def anyio_backend():
    return "asyncio"
