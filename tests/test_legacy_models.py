"""Lot P3 — instrumentation des générations sur modèle HORS registre.

Avant d'activer un registre strict, il faut SAVOIR si des appels hors registre
subsistent. Chaque relais legacy est donc compté et journalisé — sans jamais
créer de cardinalité libre : un seul compteur global, un seul événement, une
seule clé de jauge canonique `generate:legacy` (le nom arbitraire du client
n'entre plus dans la map des jauges, cf. audit P12).
"""

import logging

import pytest

from app import main
from app.metrics import counters, gauges

UNREG = "unregistered-model:latest"


@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(main.client, "generate",
                        lambda **kw: (seen.update(kw), {"response": "OK", "done": True})[1])
    return seen


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _all(caplog):
    return "\n".join(r.getMessage() for r in caplog.records)


def test_legacy_generation_is_counted_and_logged(clientapp, captured, logs):
    r = clientapp.post("/v1/generate", json={"prompt": "SECRET_PROMPT", "model": UNREG})
    assert r.status_code == 200
    assert captured["model"] == UNREG and captured["options"] is None    # relais historique intact
    assert counters.snapshot()["generate_legacy_model_total"] == 1
    line = [m for m in _all(logs).splitlines() if m.startswith("event=generate.legacy_model ")]
    assert len(line) == 1
    assert f"model={UNREG}" in line[0]
    assert "SECRET_PROMPT" not in _all(logs)


def test_legacy_stream_is_counted_too(clientapp, monkeypatch, logs):
    from tests.test_generate_stream import DONE_FIELDS, _FakeStream, _install_stream
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x", "model": UNREG})
    assert r.status_code == 200
    assert counters.snapshot()["generate_legacy_model_total"] == 1
    assert "event=generate.legacy_model" in _all(logs)


def test_governed_models_are_not_counted_as_legacy(clientapp, captured, logs):
    for name in ("llama-chat", "qwen3:14b", "gemma3:27b", "gemma-large"):
        assert clientapp.post("/v1/generate", json={"prompt": "x", "model": name}).status_code == 200
    assert "generate_legacy_model_total" not in counters.snapshot()
    assert "generate.legacy_model" not in _all(logs)


def test_legacy_inflight_key_is_bounded(clientapp, captured):
    """50 noms arbitraires -> UNE clé de jauge, pas 50."""
    gauges.reset()
    for i in range(50):
        clientapp.post("/v1/generate", json={"prompt": "x", "model": f"nope-{i}:latest"})
    keys = set(gauges._values)
    assert keys == {"generate:legacy"}
    assert gauges.total() == 0
    assert counters.snapshot()["generate_legacy_model_total"] == 50


def test_legacy_inflight_visible_on_runtime_during_call(clientapp, monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()

    def slow(**kw):
        entered.set(); release.wait(timeout=5)
        return {"response": "OK", "done": True}

    monkeypatch.setattr(main.client, "generate", slow)
    t = threading.Thread(target=lambda: clientapp.post("/v1/generate", json={"prompt": "x", "model": UNREG}))
    t.start(); assert entered.wait(timeout=5)
    try:
        body = clientapp.get("/health/runtime").json()
        assert body["inflight"]["by_model"] == {"generate:legacy": 1}
    finally:
        release.set(); t.join(timeout=5)
    assert gauges.total() == 0
