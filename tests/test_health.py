import pytest

from app import main
from app.errors import UpstreamError

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast


def test_health_ok(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client, "ping", lambda timeout=5: (True, ["llama3.1:8b", "nomic-embed-text"])
    )
    r = clientapp.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["provider"] == "ollama"
    assert body["ollama_base_url"] == "redacted"
    assert body["default_generate_model"] == "llama3.1:8b"
    assert body["default_embedding_model"] == "nomic-embed-text"
    assert "nomic-embed-text" in body["models"]
    assert "latency_ms" in body


def test_health_upstream_down(clientapp, monkeypatch):
    def boom(timeout=5):
        raise UpstreamError("connexion refusée")

    monkeypatch.setattr(main.client, "ping", boom)
    r = clientapp.get("/health")
    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "error"
    assert r.json()["detail"]["provider"] == "ollama"


def test_models(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "list_models",
        lambda timeout=5: [
            {"name": "llama3.1:8b", "size": 4920753328, "modified_at": "2026-01-01"}
        ],
    )
    r = clientapp.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "ollama"
    assert body["models"][0]["name"] == "llama3.1:8b"
    assert body["models"][0]["size"] == 4920753328
