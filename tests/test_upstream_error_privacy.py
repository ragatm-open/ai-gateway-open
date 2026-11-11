"""Lot P6 — aucune réponse HTTP 5xx ne relaie une exception réseau brute.

Une `requests.RequestException` se décrit avec l'hôte, le port et le chemin
interne (`HTTPConnectionPool(host='127.0.0.1', port=11434): …`). Ce détail
reste INTÉGRALEMENT dans les logs pour l'opérateur ; le client ne reçoit
qu'un message stable et neutre. Les 400 métier (index, tailles, modèle non
autorisé) ne sont pas concernés et gardent leur texte.

Les pannes sont injectées au niveau `requests.post` des clients transport —
exactement le point où l'exception naît — sans aucun réseau (`_no_network`
reste actif : `requests.post` est doublé avant tout socket).
"""

import logging

import pytest
import requests

from app import main
from app.errors import UpstreamError
from tests.test_vision_media import (  # noqa: F401 — fixtures réutilisées
    _post as _vision_post,
    backend,
    vision_ready,
)

# Ce qui ne doit JAMAIS apparaître dans un corps de réponse.
FORBIDDEN = ("127.0.0.1", "localhost", "11434", "8300", "HTTPConnectionPool",
             "/api/", "urllib3", "/rerank backend", "/info reranker")

OLLAMA_CONN = requests.exceptions.ConnectionError(
    "HTTPConnectionPool(host='127.0.0.1', port=11434): Max retries exceeded with url: "
    "/api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection "
    "object>: Failed to establish a new connection: [Errno 111] Connection refused'))"
)
OLLAMA_TIMEOUT = requests.exceptions.ReadTimeout(
    "HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out. (read timeout=180)"
)
RERANK_TIMEOUT = requests.exceptions.ReadTimeout(
    "HTTPConnectionPool(host='127.0.0.1', port=8300): Read timed out. (read timeout=15)"
)


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    caplog.set_level(logging.INFO, logger="ai_gateway.ollama")
    return caplog


def _raise(exc):
    def _post(*args, **kwargs):
        raise exc
    return _post


def _assert_clean(response, logs, detail_marker="HTTPConnectionPool"):
    body = response.text
    for token in FORBIDDEN:
        assert token not in body, f"{token!r} a fui dans la réponse : {body[:200]}"
    # Le détail complet reste dans les logs internes.
    assert detail_marker in "\n".join(r.getMessage() for r in logs.records)


# --------------------------------------------------------------------------- #
# 1. Routes métier — exception injectée au niveau transport
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exc", [OLLAMA_CONN, OLLAMA_TIMEOUT])
def test_generate_502_is_neutral(clientapp, monkeypatch, logs, exc):
    monkeypatch.setattr("app.ollama_client.requests.post", _raise(exc))
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen-large"})
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["provider"] == "ollama" and d["message"].startswith("échec du service de génération")
    _assert_clean(r, logs)


def test_generate_stream_pre_open_502_is_neutral(clientapp, monkeypatch, logs):
    monkeypatch.setattr("app.ollama_client.requests.post", _raise(OLLAMA_CONN))
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x"})
    assert r.status_code == 502
    _assert_clean(r, logs)


@pytest.mark.parametrize("exc", [OLLAMA_CONN, OLLAMA_TIMEOUT])
def test_embeddings_502_is_neutral(clientapp, monkeypatch, logs, exc):
    monkeypatch.setattr("app.ollama_client.requests.post", _raise(exc))
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert r.json()["detail"]["message"] == "échec du service d'embedding"
    _assert_clean(r, logs)


def test_embeddings_budget_exhausted_keeps_its_counter_but_no_detail(clientapp, monkeypatch, logs):
    from app.metrics import counters

    def boom(model, inputs, keep_alive=None):
        raise UpstreamError("échec /api/embeddings: budget de temps global épuisé après 3/10 textes (60s)")

    monkeypatch.setattr(main.client, "embed", boom)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert r.json()["detail"]["message"] == "échec du service d'embedding : budget de temps épuisé"
    assert "/api/" not in r.text
    assert counters.snapshot()["embeddings_fallback_budget_exhausted_total"] == 1


def test_rerank_502_is_neutral(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "staging")
    monkeypatch.setattr("app.rerank_client.requests.post", _raise(RERANK_TIMEOUT))
    r = clientapp.post("/v1/rerank", json={"query": "q", "documents": [{"id": "d", "text": "t"}]})
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["provider"] == "local_cross_encoder"
    assert d["message"] == "échec du service de reranking"
    _assert_clean(r, logs)


def test_vision_502_is_neutral(clientapp, backend, logs):
    # Le backend vision réel lève une UpstreamError portant l'URL Ollama.
    backend.results = [UpstreamError(
        "échec /api/generate: HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out."
    )]
    r = _vision_post(clientapp)
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["code"] == "vision_upstream_error" and d["retryable"] is True
    assert d["message"] == "échec du service d'analyse d'image"
    _assert_clean(r, logs)


# --------------------------------------------------------------------------- #
# 2. Sondes / santé — 503 et payloads de diagnostic
# --------------------------------------------------------------------------- #
def test_health_503_is_neutral(clientapp, monkeypatch, logs):
    monkeypatch.setattr("app.ollama_client.requests.get", _raise(OLLAMA_CONN))
    r = clientapp.get("/health")
    assert r.status_code == 503
    assert r.json()["detail"]["message"] == "service upstream indisponible"
    _assert_clean(r, logs)


def test_models_503_is_neutral(clientapp, monkeypatch, logs):
    monkeypatch.setattr("app.ollama_client.requests.get", _raise(OLLAMA_CONN))
    r = clientapp.get("/v1/models")
    assert r.status_code == 503
    _assert_clean(r, logs)


def test_health_ready_503_is_neutral(clientapp, monkeypatch, logs):
    def boom(model, inputs, keep_alive=None):
        raise UpstreamError("échec /api/embed: HTTPConnectionPool(host='127.0.0.1', port=11434): refused")

    monkeypatch.setattr(main.probe_client, "embed", boom)
    r = clientapp.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["message"] == "service upstream indisponible"
    _assert_clean(r, logs)
    # Le diagnostic 200 partage le même payload : même neutralité.
    r = clientapp.get("/health/embeddings")
    assert r.status_code == 200
    for token in FORBIDDEN:
        assert token not in r.text


def test_health_rerank_503_is_neutral(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr("app.rerank_client.requests.get", _raise(RERANK_TIMEOUT))
    r = clientapp.get("/health/rerank")
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["checks"]["backend_reachable"] is False
    assert d["message"] == "backend de reranking injoignable"
    _assert_clean(r, logs)


def test_health_vision_503_is_neutral(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.settings, "vision_enabled", True)
    monkeypatch.setattr(main.probe_client, "list_running_models",
                        _raise(UpstreamError("échec /api/ps: HTTPConnectionPool(host='127.0.0.1', port=11434): refused")))
    r = clientapp.get("/health/vision")
    assert r.status_code == 503
    assert r.json()["detail"]["message"] == "service upstream indisponible"
    _assert_clean(r, logs)


# --------------------------------------------------------------------------- #
# 3. Les 400 métier utiles gardent leur texte
# --------------------------------------------------------------------------- #
def test_business_400_messages_are_kept(clientapp, monkeypatch):
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"] * 200})
    assert r.status_code == 400 and "batch trop grand" in r.json()["detail"]["message"]
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen-large",
                                              "options": {"num_ctx": 999_999}})
    assert r.status_code == 400 and "dépasse la borne" in r.json()["detail"]["message"]
