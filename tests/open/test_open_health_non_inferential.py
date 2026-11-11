"""Conformité Open — sondes automatiques STRICTEMENT non inférentielles.

Tests espions : toute méthode des clients Ollama, reranker et ASR lève si elle
est appelée pendant /health/live et /health/runtime.
"""

import os

import pytest

from app import asr_client as asr_transport
from app import main
from app.ollama_client import OllamaClient
from app.rerank_client import RerankBackendClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RuntimeTouched(AssertionError):
    pass


@pytest.fixture
def spies(monkeypatch):
    def trap(name):
        def _trap(*args, **kwargs):
            raise RuntimeTouched(f"appel runtime interdit : {name}")
        return _trap
    for method in ("list_models", "list_running_models", "list_running_models_vram", "generate",
                   "open_generate_stream", "chat", "open_chat_stream", "embed", "ping"):
        monkeypatch.setattr(OllamaClient, method, trap(f"ollama.{method}"))
    for method in ("info", "is_healthy", "rerank"):
        monkeypatch.setattr(RerankBackendClient, method, trap(f"rerank.{method}"))
    for method in ("info", "health"):
        if hasattr(asr_transport.asr_client, method):
            monkeypatch.setattr(type(asr_transport.asr_client), method, trap(f"asr.{method}"))


def test_health_live_never_touches_a_runtime(open_app, spies):
    r = open_app.get("/health/live")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["inferential"] is False and body["registry_loaded"] is True
    assert set(body) == {"status", "provider", "version", "registry_loaded", "registry_version",
                         "uptime_seconds", "inferential"}


def test_health_runtime_never_touches_a_runtime(open_app, spies):
    r = open_app.get("/health/runtime")
    assert r.status_code == 200
    body = r.json()
    assert "capabilities" in body and "limits" in body and "v2h_shadow" not in body


def test_capability_states_use_closed_vocabulary(open_app):
    caps = open_app.get("/health/runtime").json()["capabilities"]
    states = {c["state"] for k, c in caps.items() if k != "inventory"}
    assert states <= {"ENABLED", "DISABLED", "UNAVAILABLE", "UNKNOWN"}
    assert caps["generation"]["state"] == "ENABLED" and caps["embeddings"]["state"] == "ENABLED"
    assert caps["rerank"]["state"] == "ENABLED"
    assert caps["asr"]["state"] == "DISABLED" and caps["vision"]["state"] == "DISABLED"
    for name in ("structured_extract", "semantic_extract", "host_admission"):
        assert caps[name]["state"] == "UNAVAILABLE"
    # Enregistré ≠ installé : le 7 B est au registre, pas sur le runtime.
    assert caps["inventory"]["state"] == "OBSERVED"
    assert caps["generation"]["models"]["compact-generation"]["installed"] is False
    assert caps["generation"]["models"]["compact-generation-8"]["installed"] is True
    assert caps["embeddings"]["models"]["fr-en-retrieval"]["identity_verified"] is True


def test_inventory_unknown_before_any_observation(open_app, compact_registry):
    compact_registry.inventory = None
    compact_registry.inventory_at = None
    caps = open_app.get("/health/runtime").json()["capabilities"]
    assert caps["inventory"]["state"] == "UNKNOWN"
    assert caps["generation"]["models"]["compact-generation"]["installed"] is None


def test_runtime_exposes_explicit_limits(open_app):
    limits = open_app.get("/health/runtime").json()["limits"]
    assert set(limits) >= {"generate_max_concurrency", "embeddings_max_concurrency", "rerank_max_concurrency",
                           "request_max_bytes", "embeddings_max_batch", "timeout_seconds"}


def test_dockerfile_healthcheck_targets_the_live_probe():
    content = open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8").read()
    healthcheck = [l for l in content.splitlines() if l.startswith("HEALTHCHECK")]
    assert healthcheck, "HEALTHCHECK manquant"
    assert "/health/live" in content
    for inferential in ("/health/ready", "/health/embeddings", "/health/rerank"):
        assert inferential not in content.split("HEALTHCHECK", 1)[1].split("CMD [\"uvicorn\"")[0]


def test_inferential_probes_are_explicit_and_separate(open_app, monkeypatch):
    calls = []

    def spy_embed(model, inputs, keep_alive=None):
        calls.append(model)
        return _unit(inputs)
    monkeypatch.setattr(main.probe_client, "embed", spy_embed)
    r = open_app.get("/health/ready")
    assert r.status_code == 200 and r.json()["pipeline_id"].startswith("emb-v2:")
    assert calls == ["granite-embedding:278m"]    # une sonde, volontaire, sur le défaut Compact


def _unit(inputs):
    from app.ollama_client import ENDPOINT_EMBED, EmbedResult
    return EmbedResult([[1.0] + [0.0] * 767 for _ in inputs], ENDPOINT_EMBED)
