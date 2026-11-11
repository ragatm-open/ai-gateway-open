"""Repli /api/embeddings : traçabilité, budget de temps, normalisation L2."""

import math

import pytest
import requests

from app import main
from app.metrics import counters
from app.ollama_client import ENDPOINT_EMBED, ENDPOINT_EMBEDDINGS, EmbedResult, OllamaClient


class _Resp:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error", response=self)

    def json(self):
        return self._payload


def _client(timeout=30):
    return OllamaClient("http://127.0.0.1:11434", timeout=timeout)


# --------------------------------------------------------- client bas niveau
def test_nominal_path_reports_embed_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None: _Resp({"embeddings": [[1.0, 0.0]]}),
    )
    result = _client().embed(model="m", inputs=["a"])
    assert isinstance(result, EmbedResult)
    assert result.upstream_endpoint == ENDPOINT_EMBED
    assert result.fallback_used is False
    assert list(result) == [[1.0, 0.0]]


def test_fallback_is_used_and_reported(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(url)
        if url.endswith(ENDPOINT_EMBED):
            # Lot P10 : seule la sentinelle Gin « route inconnue » (Ollama
            # < 0.1.26) ou un 405 justifient le repli — plus un 501 quelconque.
            return _Resp({}, status=404, text="404 page not found")
        return _Resp({"embedding": [3.0, 4.0]})

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)
    result = _client().embed(model="m", inputs=["a", "b"])
    assert result.upstream_endpoint == ENDPOINT_EMBEDDINGS
    assert result.fallback_used is True
    # Un appel /api/embed puis un appel unitaire par texte.
    assert sum(1 for u in calls if u.endswith(ENDPOINT_EMBEDDINGS)) == 2


def test_fallback_result_is_still_a_plain_list(monkeypatch):
    """EmbedResult hérite de list : aucun appelant historique n'est cassé."""
    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None: _Resp({"embeddings": [[1.0]]}),
    )
    result = _client().embed(model="m", inputs=["a"])
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == [1.0]


def test_fallback_global_budget_is_enforced(monkeypatch):
    """Sans budget global, 64 textes × 60 s valaient 64 minutes."""
    clock = {"t": 0.0}
    monkeypatch.setattr("app.ollama_client.time.monotonic", lambda: clock["t"])

    processed = []

    def fake_post(url, json=None, timeout=None):
        if url.endswith(ENDPOINT_EMBED):
            return _Resp({}, status=405, text="405 method not allowed")
        processed.append(json["prompt"])
        clock["t"] += 4.0  # chaque texte consomme 4 s du budget
        return _Resp({"embedding": [1.0, 0.0]})

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)

    from app.errors import UpstreamError

    with pytest.raises(UpstreamError) as excinfo:
        _client(timeout=10).embed(model="m", inputs=[f"t{i}" for i in range(10)])

    assert "budget de temps global" in excinfo.value.message
    # 3 textes encodés (4+4+4 = 12 s > budget de 10 s), puis arrêt net.
    assert len(processed) == 3
    assert "après 3/10 textes" in excinfo.value.message


def test_fallback_stops_before_starting_a_call_without_budget(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr("app.ollama_client.time.monotonic", lambda: clock["t"])
    started = []

    def fake_post(url, json=None, timeout=None):
        if url.endswith(ENDPOINT_EMBED):
            clock["t"] += 60.0  # l'appel batch a déjà consommé tout le budget
            return _Resp({}, status=404, text="404 page not found")
        started.append(url)
        return _Resp({"embedding": [1.0]})

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)
    from app.errors import UpstreamError

    with pytest.raises(UpstreamError):
        _client(timeout=30).embed(model="m", inputs=["a", "b"])
    assert started == []  # aucun traitement après expiration


# ------------------------------------------------------------ route complète
def test_route_normalises_fallback_vectors(clientapp, monkeypatch):
    """Cœur du correctif : /api/embeddings renvoie ‖v‖ ≈ 19,7, /api/embed ‖v‖ = 1.
    Mélanger les deux dans un index vectoriel est muet et faux."""
    raw = EmbedResult([[3.0, 4.0] + [0.0] * 766], ENDPOINT_EMBEDDINGS, fallback_used=True)
    monkeypatch.setattr(main.client, "embed", lambda model, inputs, keep_alive=None: raw)

    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200
    body = r.json()
    assert body["upstream_endpoint"] == ENDPOINT_EMBEDDINGS
    assert body["normalized"] is True
    norm = math.sqrt(sum(x * x for x in body["embeddings"][0]))
    assert abs(norm - 1.0) < 1e-9
    assert body["embeddings"][0][0] == pytest.approx(0.6)
    assert body["embeddings"][0][1] == pytest.approx(0.8)
    assert counters.snapshot()["embeddings_fallback_total"] == 1


def test_route_reports_embed_endpoint_by_default(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult(
            [[1.0] + [0.0] * 767], ENDPOINT_EMBED
        ),
    )
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.json()["upstream_endpoint"] == ENDPOINT_EMBED
    assert counters.snapshot()["embeddings_fallback_total"] == 0


def test_route_refuses_unnormalisable_fallback_vector(clientapp, monkeypatch):
    raw = EmbedResult([[0.0] * 768], ENDPOINT_EMBEDDINGS, fallback_used=True)
    monkeypatch.setattr(main.client, "embed", lambda model, inputs, keep_alive=None: raw)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502


def test_route_does_not_normalise_ungoverned_model(clientapp, monkeypatch):
    """Sans profil, la normalisation attendue est inconnue : on ne touche à
    rien plutôt que de deviner."""
    raw = EmbedResult([[3.0, 4.0]], ENDPOINT_EMBEDDINGS, fallback_used=True)
    monkeypatch.setattr(main.client, "embed", lambda model, inputs, keep_alive=None: raw)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "bge-m3"})
    assert r.status_code == 200
    assert r.json()["embeddings"][0] == [3.0, 4.0]
    assert r.json()["normalized"] is False


def test_budget_exhausted_is_counted_and_surfaced(clientapp, monkeypatch):
    from app.errors import UpstreamError

    def boom(model, inputs, keep_alive=None):
        raise UpstreamError(
            "échec /api/embeddings: budget de temps global épuisé après 3/10 textes (60s)"
        )

    monkeypatch.setattr(main.client, "embed", boom)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert counters.snapshot()["embeddings_fallback_budget_exhausted_total"] == 1
