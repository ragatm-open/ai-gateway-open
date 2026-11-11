"""Lot P8 — une requête qui attend une ressource limitée ne consomme plus un
thread du pool partagé pendant cette attente.

Démontré par la contre-expertise sur un uvicorn réel : 45 rerank simultanés à
concurrence 1 → 40/40 threads AnyIO empruntés, 39 requêtes en file sur le
sémaphore, et `GET /health` (route synchrone comme les autres) répondait en
28 s — contre 10 s de healthcheck Docker. Le backend n'était pas en cause :
l'attente elle-même occupait les threads.

Pourquoi `httpx.ASGITransport` et non `TestClient` : TestClient ouvre un
portail (donc une boucle et un limiteur de threads) PAR requête, ce qui rend
la starvation irreproductible. Ici toutes les requêtes partagent UNE boucle
asyncio, donc UN limiteur AnyIO à 40 tokens — exactement la topologie uvicorn.
Aucun socket : `_no_network` reste actif.
"""

import asyncio
import threading
import time

import httpx
import pytest

from app import main
from app.errors import UpstreamError
from app.vision_admission import VisionAdmission
from app.metrics import counters, gauges
from tests.test_rerank import _docs
from tests.test_vision_media import (  # noqa: F401 — fixtures réutilisées
    _post as _vision_post,
    backend,
    ok_photo_result,
    vision_ready,
)

N_CONCURRENT = 45          # > 40 tokens du pool AnyIO par défaut
HEALTH_BUDGET_S = 1.0      # critère métier : pas de starvation, seuil non fragile

RERANK_BODY = {"query": "quelle est la question ?", "documents": _docs(3)}


@pytest.fixture
def rerank_staging(monkeypatch):
    gauges.reset()
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "staging")
    # `/health` sonde Ollama via `client.ping` (client métier, pas la sonde
    # doublée par conftest) : doublé ici pour que la route reste hermétique.
    monkeypatch.setattr(main.client, "ping", lambda timeout=5: (1, []))
    # Concurrence métier = 1, comme le staging réel.
    monkeypatch.setattr(main, "_rerank_semaphore", threading.BoundedSemaphore(1))
    # Ancien timeout d'attente : ne sert plus qu'à rendre une régression
    # (acquire bloquant) visible en quelques secondes plutôt qu'en 15.
    monkeypatch.setattr(main.settings, "rerank_timeout_seconds", 3)


class _BlockingRerank:
    """Backend mocké : la 1re requête entre et BLOQUE jusqu'à `release`."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def __call__(self, query, texts):
        with self._lock:
            self.calls += 1
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        self.entered.set()
        try:
            assert self.release.wait(timeout=10), "backend mock jamais libéré"
            return [{"index": i, "score": 0.5} for i in range(len(texts))]
        finally:
            with self._lock:
                self._active -= 1


async def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition jamais atteinte"
        await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- #
# 1. TEST CENTRAL — reproduction du scénario P8 (rerank, 45 requêtes)
# --------------------------------------------------------------------------- #
def test_saturated_rerank_does_not_starve_health(rerank_staging, monkeypatch):
    mock = _BlockingRerank()
    monkeypatch.setattr(main.rerank_client, "rerank", mock)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://asgi", timeout=30) as c:
            tasks = [asyncio.create_task(c.post("/v1/rerank", json=RERANK_BODY))
                     for _ in range(N_CONCURRENT)]
            # La 1re requête est dans le backend ; les autres sont arrivées.
            await _wait_for(lambda: mock.entered.is_set())
            await _wait_for(lambda: sum(t.done() for t in tasks) >= N_CONCURRENT - 1)

            t0 = time.perf_counter()
            health = await c.get("/health")
            health_s = time.perf_counter() - t0

            snapshot_during = gauges.snapshot()
            mock.release.set()
            responses = await asyncio.gather(*tasks)
            after = await c.post("/v1/rerank", json=RERANK_BODY)
            return health, health_s, responses, snapshot_during, after

    try:
        health, health_s, responses, snapshot_during, after = asyncio.run(scenario())
    finally:
        mock.release.set()       # jamais de thread bloqué qui survit au test

    # (a) /health disponible pendant la saturation — LE critère du lot.
    assert health.status_code == 200
    assert health_s < HEALTH_BUDGET_S, f"/health a mis {health_s:.2f}s sous saturation"

    # (b) exactement la concurrence métier est entrée dans le backend.
    assert mock.max_active == 1
    assert mock.calls == 1 + 1          # la 1re + celle d'après libération

    accepted = [r for r in responses if r.status_code == 200]
    rejected = [r for r in responses if r.status_code == 503]
    assert len(accepted) == 1
    assert len(rejected) == N_CONCURRENT - 1

    # (c) rejets explicites, immédiats, retryables, sans détail interne.
    for r in rejected:
        d = r.json()["detail"]
        assert d["code"] == "rerank_saturated" and d["retryable"] is True
        assert r.headers["Retry-After"] == str(main.RERANK_RETRY_AFTER_SECONDS)
        assert d["retry_after_seconds"] == main.RERANK_RETRY_AFTER_SECONDS
        assert "127.0.0.1" not in r.text and "8300" not in r.text
    assert counters.snapshot()["rerank_saturated_total"] == N_CONCURRENT - 1

    # (d) aucune file interne : rien n'est compté `queued`, 1 seul inflight.
    assert snapshot_during == {"rerank:rerank-default": 1}

    # (e) la capacité est libérée et une nouvelle requête est acceptée.
    assert after.status_code == 200
    assert gauges.total() == 0


def test_rejection_is_immediate_not_after_timeout(rerank_staging, monkeypatch, clientapp):
    """Un rejet coûte des millisecondes, pas `rerank_timeout_seconds`."""
    mock = _BlockingRerank()
    monkeypatch.setattr(main.rerank_client, "rerank", mock)
    holder = threading.Thread(target=lambda: clientapp.post("/v1/rerank", json=RERANK_BODY))
    holder.start()
    assert mock.entered.wait(timeout=5)
    try:
        t0 = time.perf_counter()
        r = clientapp.post("/v1/rerank", json=RERANK_BODY)
        elapsed = time.perf_counter() - t0
        assert r.status_code == 503
        assert elapsed < 0.5, f"rejet en {elapsed:.2f}s : attente bloquante réintroduite ?"
        assert mock.calls == 1            # backend NON appelé pour la rejetée
    finally:
        mock.release.set()
        holder.join(timeout=5)


# --------------------------------------------------------------------------- #
# 2. Libération garantie — aucune fuite de permit
# --------------------------------------------------------------------------- #
def _raise_upstream(q, t):
    raise UpstreamError("échec /rerank backend: HTTP 500")


@pytest.mark.parametrize("failure", [
    _raise_upstream,                                   # backend en échec -> 502
    lambda q, t: [{"index": 0, "score": 0.1}],         # cardinalité fausse -> 502
])
def test_rerank_permit_released_after_backend_failure(rerank_staging, monkeypatch, clientapp, failure):
    monkeypatch.setattr(main.rerank_client, "rerank", failure)
    r = clientapp.post("/v1/rerank", json=RERANK_BODY)
    assert r.status_code == 502
    assert gauges.total() == 0

    # Le permit est de nouveau disponible : la requête suivante entre.
    monkeypatch.setattr(main.rerank_client, "rerank",
                        lambda q, t: [{"index": i, "score": 0.5} for i in range(len(t))])
    r = clientapp.post("/v1/rerank", json=RERANK_BODY)
    assert r.status_code == 200
    assert "rerank_saturated_total" not in counters.snapshot()


def test_rerank_unsaturated_behaviour_unchanged(rerank_staging, monkeypatch, clientapp):
    """Cas nominal : mêmes scores, même forme, aucun en-tête de saturation."""
    monkeypatch.setattr(main.rerank_client, "rerank",
                        lambda q, t: [{"index": i, "score": 0.9 - i * 0.1} for i in range(len(t))])
    r = clientapp.post("/v1/rerank", json=RERANK_BODY)
    assert r.status_code == 200
    assert "Retry-After" not in r.headers
    assert [x["relevance_score"] for x in r.json()["results"]] == [0.9, 0.8, 0.7]
    assert gauges.total() == 0


# --------------------------------------------------------------------------- #
# 3. VISION — même invariant (backend Ollama TOUJOURS mocké)
# --------------------------------------------------------------------------- #
class _BlockingVision:
    def __init__(self, result):
        self.result = result
        self.entered = threading.Event()
        self.release = threading.Event()
        self.plans = []

    def analyze(self, plan):
        self.plans.append(plan)
        self.entered.set()
        assert self.release.wait(timeout=10), "backend vision jamais libéré"
        return self.result


def test_saturated_vision_rejects_immediately_and_keeps_health(backend, monkeypatch, clientapp):
    gauges.reset()
    monkeypatch.setattr(main.client, "ping", lambda timeout=5: (1, []))
    monkeypatch.setattr(main, "vision_admission", VisionAdmission(1))
    blocking = _BlockingVision(ok_photo_result())
    monkeypatch.setattr(main, "vision_backend", blocking)

    holder = threading.Thread(target=lambda: _vision_post(clientapp))
    holder.start()
    assert blocking.entered.wait(timeout=5)
    try:
        t0 = time.perf_counter()
        r = _vision_post(clientapp, request_id="rid-sat")
        elapsed = time.perf_counter() - t0
        d = r.json()["detail"]
        assert r.status_code == 503 and d["code"] == "vision_overloaded"
        assert d["retryable"] is True and d["request_id"] == "rid-sat"
        assert r.headers["Retry-After"] == str(main.VISION_RETRY_AFTER_SECONDS)
        assert d["retry_after_seconds"] == main.VISION_RETRY_AFTER_SECONDS
        assert elapsed < HEALTH_BUDGET_S, f"rejet vision en {elapsed:.2f}s"
        assert len(blocking.plans) == 1          # backend NON appelé pour la rejetée
        assert gauges.snapshot() == {"vision:vision-large": 1}   # aucune file

        t0 = time.perf_counter()
        h = clientapp.get("/health")
        assert h.status_code == 200 and time.perf_counter() - t0 < HEALTH_BUDGET_S
    finally:
        blocking.release.set()
        holder.join(timeout=5)

    # Capacité libérée : la requête suivante est acceptée.
    monkeypatch.setattr(main, "vision_backend", backend)
    assert _vision_post(clientapp).status_code == 200
    assert gauges.total() == 0
    assert counters.snapshot()["vision_overloaded_total"] == 1


def test_vision_permit_released_after_backend_failure(backend, monkeypatch, clientapp):
    gauges.reset()
    monkeypatch.setattr(main, "vision_admission", VisionAdmission(1))
    backend.results = [UpstreamError("échec /api/generate: HTTP 500"), ok_photo_result()]
    r = _vision_post(clientapp)
    assert r.status_code == 502
    assert gauges.total() == 0
    r = _vision_post(clientapp)                  # le permit n'a pas fui
    assert r.status_code == 200
    assert "Retry-After" not in r.headers


def test_vision_unsaturated_behaviour_unchanged(backend, clientapp):
    r = _vision_post(clientapp)
    assert r.status_code == 200
    assert "Retry-After" not in r.headers
    assert r.json()["visual_kind"] == "photo"
    assert len(backend.plans) == 1
