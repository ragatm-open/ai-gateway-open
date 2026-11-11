"""Lot P10 — le repli `/api/embeddings` n'est légitime que si `/api/embed`
n'existe pas sur cet Ollama.

Formes réelles relevées sur un Ollama 0.31.1 de l'installation d'origine (lecture seule) :
  route inconnue  -> 404 text/plain `404 page not found`   (sentinelle Gin)
  modèle absent   -> 404 JSON `{"error":"model \\"x\\" not found, try pulling it first"}`
  mauvaise méthode-> 405 text/plain `405 method not allowed`
Seules la première et la troisième prouvent l'absence de l'endpoint.
"""

import pytest
import requests

from app.errors import UpstreamError
from app.ollama_client import ENDPOINT_EMBED, ENDPOINT_EMBEDDINGS, OllamaClient


class _Resp:
    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload if payload is not None else {}
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload


def _client():
    return OllamaClient("http://127.0.0.1:11434", timeout=30)


def _run(monkeypatch, embed_response, n_texts=1):
    """Exécute embed(); retourne (résultat | exception, endpoints appelés)."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(url.rsplit("/api", 1)[-1])
        if url.endswith(ENDPOINT_EMBED):
            if isinstance(embed_response, Exception):
                raise embed_response
            return embed_response
        return _Resp({"embedding": [1.0, 0.0]})

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)
    try:
        return _client().embed(model="m", inputs=[f"t{i}" for i in range(n_texts)]), calls
    except UpstreamError as exc:
        return exc, calls


NO_FALLBACK = [
    ("404 modèle absent", _Resp({"error": 'model "m" not found, try pulling it first'},
                               404, '{"error":"model \\"m\\" not found, try pulling it first"}')),
    ("400 validation", _Resp({"error": "invalid input"}, 400, '{"error":"invalid input"}')),
    ("429", _Resp({}, 429, "too many requests")),
    ("500", _Resp({"error": "runner failed"}, 500, '{"error":"runner failed"}')),
    ("503", _Resp({}, 503, "service unavailable")),
    ("timeout", requests.exceptions.ReadTimeout("read timed out")),
    ("connection error", requests.exceptions.ConnectionError("refused")),
    ("200 sans embeddings", _Resp({}, 200)),
]


@pytest.mark.parametrize("label,response", NO_FALLBACK, ids=[x[0] for x in NO_FALLBACK])
def test_modern_endpoint_errors_never_fall_back(monkeypatch, label, response):
    result, calls = _run(monkeypatch, response, n_texts=3)
    assert isinstance(result, UpstreamError), label
    assert calls == ["/embed"]                      # UN appel, aucun legacy
    assert "/embeddings" not in calls


def test_200_is_served_without_fallback(monkeypatch):
    result, calls = _run(monkeypatch, _Resp({"embeddings": [[1.0, 0.0]]}))
    assert list(result) == [[1.0, 0.0]] and result.upstream_endpoint == ENDPOINT_EMBED
    assert calls == ["/embed"]


@pytest.mark.parametrize("label,response", [
    ("404 route inconnue (Gin)", _Resp({}, 404, "404 page not found")),
    ("405", _Resp({}, 405, "405 method not allowed")),
], ids=["404-endpoint", "405"])
def test_unsupported_endpoint_falls_back(monkeypatch, label, response):
    result, calls = _run(monkeypatch, response, n_texts=2)
    assert result.upstream_endpoint == ENDPOINT_EMBEDDINGS and result.fallback_used is True
    assert calls == ["/embed", "/embeddings", "/embeddings"]


def test_no_batch_amplification_on_500(monkeypatch):
    """N textes + 500 sur /api/embed : UN appel, pas N appels legacy."""
    result, calls = _run(monkeypatch, _Resp({}, 500, "boom"), n_texts=64)
    assert isinstance(result, UpstreamError)
    assert len(calls) == 1 and calls == ["/embed"]
    assert "HTTP 500" in result.message


def test_404_model_not_found_is_reported_as_upstream_error(monkeypatch):
    # le 404 « model not found » est TYPÉ (UpstreamModelNotFound,
    # sous-classe d'UpstreamError) pour rendre `503 model_not_installed` ; il
    # ne déclenche toujours AUCUN repli ni appel supplémentaire.
    from app.errors import UpstreamModelNotFound
    result, calls = _run(monkeypatch, NO_FALLBACK[0][1], n_texts=5)
    assert isinstance(result, UpstreamModelNotFound) and "non installé" in result.message
    assert calls == ["/embed"]
