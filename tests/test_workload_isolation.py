"""Lot workload-isolation — observabilité de la contention et raison d'arrêt.

* `/health/runtime` DÉCLARE la topologie de calcul partagée (prod/staging =
  deux processus, un seul GPU, Ollama et backend reranker communs) et compte
  les embeddings en cours dans `inflight` ;
* `/v1/generate` et `/v1/generate/stream` relaient `done_reason` VERBATIM
  d'Ollama ("stop" | "length"), None si absent — jamais inféré.
"""

import json

from app import main
from app.metrics import gauges
from app.ollama_client import EmbedResult, ENDPOINT_EMBED
from tests.test_generate_stream import _FakeStream, _install_stream


# ----------------------------------------------------------- /health/runtime
def test_runtime_declares_shared_compute(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "production")
    compute = clientapp.get("/health/runtime").json()["compute"]
    assert compute["isolation"] == "none"
    assert compute["shared_with_environments"] == ["staging"]
    assert set(compute["shared_resources"]) == {"gpu", "ollama", "rerank_backend"}
    assert compute["gpu_count"] == 1
    assert compute["sibling_inflight"] == "not_observable_here"

    monkeypatch.setattr(main.settings, "environment", "staging")
    assert clientapp.get("/health/runtime").json()["compute"]["shared_with_environments"] == ["production"]


def test_embeddings_counted_in_inflight(clientapp, monkeypatch):
    observed = {}

    def fake_embed(model, inputs, keep_alive=None):
        observed["during"] = gauges.snapshot().get("embeddings:legacy-768", 0)
        return EmbedResult([[1.0] + [0.0] * 767 for _ in inputs], ENDPOINT_EMBED)

    monkeypatch.setattr(main.client, "embed", fake_embed)
    r = clientapp.post("/v1/embeddings", json={"texts": ["x"], "model": "legacy-768"})
    assert r.status_code == 200
    assert observed["during"] == 1
    assert gauges.snapshot().get("embeddings:legacy-768", 0) == 0
    assert clientapp.get("/health/runtime").json()["inflight"]["total"] == 0


def test_embeddings_inflight_released_on_upstream_error(clientapp, monkeypatch):
    from app.errors import UpstreamError

    def boom(model, inputs, keep_alive=None):
        raise UpstreamError("panne")

    monkeypatch.setattr(main.client, "embed", boom)
    assert clientapp.post("/v1/embeddings", json={"texts": ["x"]}).status_code == 502
    assert gauges.total() == 0


# ----------------------------------------------------------------- done_reason
def _gen(monkeypatch, **extra):
    def fake_generate(model, prompt, system=None, temperature=None, options=None,
                      think=None, keep_alive=None):
        return {"response": "OK", "prompt_eval_count": 1, "eval_count": 1,
                "done": True, **extra}
    monkeypatch.setattr(main.client, "generate", fake_generate)


def test_generate_relays_done_reason_stop_and_length(clientapp, monkeypatch):
    _gen(monkeypatch, done_reason="stop")
    assert clientapp.post("/v1/generate", json={"prompt": "x"}).json()["done_reason"] == "stop"
    _gen(monkeypatch, done_reason="length")
    body = clientapp.post("/v1/generate", json={"prompt": "x"}).json()
    assert body["done_reason"] == "length"
    assert body["done"] is True  # sémantique historique inchangée


def test_generate_done_reason_absent_is_null_not_inferred(clientapp, monkeypatch):
    _gen(monkeypatch)  # upstream ancien : pas de done_reason
    body = clientapp.post("/v1/generate", json={"prompt": "x"}).json()
    assert "done_reason" in body and body["done_reason"] is None
    _gen(monkeypatch, done_reason="   ")
    assert clientapp.post("/v1/generate", json={"prompt": "x"}).json()["done_reason"] is None


def test_stream_done_event_carries_done_reason(clientapp, monkeypatch):
    events = [
        {"response": "Bon", "done": False},
        {"response": "jour", "done": False},
        {"response": "", "done": True, "done_reason": "length",
         "prompt_eval_count": 3, "eval_count": 2},
    ]
    _install_stream(monkeypatch, _FakeStream(events))
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x"})
    lines = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    done = [e for e in lines if e["type"] == "done"]
    assert len(done) == 1 and done[0]["done_reason"] == "length"
    assert [e["delta"] for e in lines if e["type"] == "delta"] == ["Bon", "jour"]


# ------------------------------------------------------------------ queued
def test_rerank_has_no_internal_queue_anymore(clientapp, monkeypatch):
    """Lot P8 : une requête qui trouve le sémaphore pris n'attend plus — elle
    est refusée immédiatement. `queued` reste exposé pour compatibilité et
    vaut toujours 0 ; le refus se lit dans `rerank_saturated_total`."""
    import threading
    from tests.test_rerank import _post

    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "staging")
    release = threading.Event()
    entered = threading.Event()

    def slow_rerank(query, texts):
        entered.set()
        release.wait(timeout=5)
        return [{"index": i, "score": 0.0} for i in range(len(texts))]

    monkeypatch.setattr(main.rerank_client, "rerank", slow_rerank)
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(main, "_rerank_semaphore", sem)

    t1 = threading.Thread(target=lambda: _post(clientapp))
    t1.start(); assert entered.wait(timeout=5)
    r2 = _post(clientapp)                      # ne bloque pas : 503 immédiat
    assert r2.status_code == 503
    assert r2.json()["detail"]["code"] == "rerank_saturated"
    body = clientapp.get("/health/runtime").json()
    assert body["inflight"]["by_model"] == {"rerank:rerank-default": 1}
    assert body["queued"] == {"total": 0, "by_model": {}}
    release.set(); t1.join(timeout=5)
    assert gauges.total() == 0
    after = clientapp.get("/health/runtime").json()
    assert after["inflight"]["total"] == 0 and after["queued"]["total"] == 0
