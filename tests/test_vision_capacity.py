"""Lot CV-CAPACITY — admission vision encapsulée (`VisionAdmission`) et sonde
ADVISORY `GET /v1/vision/capacity`.

Aucun modèle, aucun GPU, aucun réseau : admission synthétique, backend mocké,
`/api/ps` simulé. La sonde ne réserve rien ; le POST reste l'autorité.
"""

import json
import threading
import time

import pytest
from pydantic import ValidationError

from app import main
from app.errors import UpstreamError
from app.metrics import counters, gauges
from app.vision_admission import VisionAdmission, VisionAdmissionError, VisionAdmissionSnapshot
from app.vision_schemas import VisionCapacityResponse
from tests.test_vision_media import (
    _post,
    backend,          # noqa: F401 — fixtures réutilisées
    ok_photo_result,
    vision_ready,     # noqa: F401
)
from tests.test_vision_provenance import DIGEST
from app.vision_prompts import DESCRIBE_PROMPT
from app.vision_provenance import pipeline_identity, vision_pipeline_fingerprint


# ------------------------------------------------------------ VisionAdmission
def _snap(a: VisionAdmission):
    s = a.snapshot()
    return (s.max_concurrency, s.in_flight, s.available_slots)


def test_snapshot_initial_acquired_released():
    a = VisionAdmission(1)
    assert _snap(a) == (1, 0, 1)
    assert a.try_acquire() is True
    assert _snap(a) == (1, 1, 0)
    assert a.try_acquire() is False            # saturé : refus, compteur inchangé
    assert _snap(a) == (1, 1, 0)
    a.release()
    assert _snap(a) == (1, 0, 1)


def test_snapshot_max_concurrency_two():
    a = VisionAdmission(2)
    assert _snap(a) == (2, 0, 2)
    assert a.try_acquire() and _snap(a) == (2, 1, 1)
    assert a.try_acquire() and _snap(a) == (2, 2, 0)
    assert a.try_acquire() is False
    a.release(); assert _snap(a) == (2, 1, 1)
    a.release(); assert _snap(a) == (2, 0, 2)


def test_snapshot_is_immutable_and_derived_under_lock():
    a = VisionAdmission(1)
    s = a.snapshot()
    assert isinstance(s, VisionAdmissionSnapshot)
    with pytest.raises(Exception):
        s.in_flight = 5                        # frozen
    assert VisionAdmission(0).max_concurrency == 1   # plancher comme avant (max(1, n))


def test_release_without_acquire_is_an_explicit_invariant_violation():
    a = VisionAdmission(1)
    with pytest.raises(VisionAdmissionError):
        a.release()
    assert _snap(a) == (1, 0, 1)               # jamais négatif, jamais clampé en silence
    assert a.try_acquire()
    a.release()
    with pytest.raises(VisionAdmissionError):
        a.release()                            # double release
    assert _snap(a) == (1, 0, 1) and a.try_acquire()


def test_no_private_semaphore_introspection_in_code():
    import inspect
    from app import vision_admission
    src = inspect.getsource(vision_admission) + inspect.getsource(main.vision_capacity)
    assert "._value" not in src and "_initial_value" not in src


@pytest.mark.parametrize("max_concurrency", [1, 2, 3])
def test_concurrent_acquisitions_never_exceed_max(max_concurrency):
    a = VisionAdmission(max_concurrency)
    n_threads, rounds = 16, 30
    peak, peak_lock = [0], threading.Lock()
    violations = []

    def worker():
        for _ in range(rounds):
            if a.try_acquire():
                s = a.snapshot()
                with peak_lock:
                    peak[0] = max(peak[0], s.in_flight)
                if s.in_flight > max_concurrency or s.available_slots < 0:
                    violations.append(s)
                time.sleep(0.0005)
                a.release()
            else:
                s = a.snapshot()
                if s.in_flight > max_concurrency:
                    violations.append(s)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    [t.start() for t in threads]; [t.join(timeout=10) for t in threads]
    assert not violations
    assert 1 <= peak[0] <= max_concurrency
    assert _snap(a) == (max_concurrency, 0, max_concurrency)


def test_snapshot_cost_is_negligible():
    a = VisionAdmission(1)
    t = time.perf_counter()
    for _ in range(10_000):
        a.snapshot()
    per_call_ms = (time.perf_counter() - t) / 10_000 * 1000
    assert per_call_ms < 0.1, per_call_ms        # ordre de grandeur : microsecondes


# ----------------------------------------------------- route analyze (503)
def test_overload_503_envelope_unchanged(clientapp, backend, monkeypatch):
    held = VisionAdmission(1); assert held.try_acquire()
    monkeypatch.setattr(main, "vision_admission", held)
    counters.reset()
    r = _post(clientapp, request_id="rid-cap")
    d = r.json()["detail"]
    assert r.status_code == 503
    assert d == {"status": "error", "provider": "ai_gateway",
                 "message": "concurrence vision maximale atteinte",
                 "code": "vision_overloaded", "retryable": True,
                 "request_id": "rid-cap", "retry_after_seconds": 10}
    assert r.headers["Retry-After"] == "10" == str(main.VISION_RETRY_AFTER_SECONDS)
    assert backend.plans == []                      # aucun appel Ollama
    assert counters.snapshot()["vision_overloaded_total"] == 1
    assert _snap(held) == (1, 1, 0)                 # le refus n'a rien modifié


def test_slot_acquired_after_input_validation_and_released_in_finally(clientapp, backend, monkeypatch):
    adm = VisionAdmission(1)
    monkeypatch.setattr(main, "vision_admission", adm)
    seen = []
    original = backend.analyze

    def spy(plan):
        seen.append(_snap(adm))                     # pendant l'inférence : slot pris
        return original(plan)
    backend.analyze = spy
    backend.results = [ok_photo_result()]
    assert _post(clientapp).status_code == 200
    assert seen == [(1, 1, 0)] and _snap(adm) == (1, 0, 1)
    # Rejets AVANT acquisition : jamais comptés en vol.
    assert _post(clientapp, schema_version="3").status_code == 400
    assert _post(clientapp, mime="image/jpeg").status_code == 400     # spoof MIME
    assert _snap(adm) == (1, 0, 1)
    # Exception backend : le finally libère toujours.
    backend.results = [UpstreamError("échec /api/generate: HTTP 500")]
    assert _post(clientapp).status_code == 502
    assert _snap(adm) == (1, 0, 1) and gauges.total() == 0


def test_gauges_still_fed_for_health_runtime(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main, "vision_admission", VisionAdmission(1))
    gauges.reset()
    backend.results = [ok_photo_result()]
    assert _post(clientapp).status_code == 200
    assert gauges.total() == 0
    assert clientapp.get("/health/runtime").json()["inflight"] == {"total": 0, "by_model": {}}


# ---------------------------------------------------- GET /v1/vision/capacity
def _cap(clientapp):
    r = clientapp.get("/v1/vision/capacity")
    assert r.status_code == 200
    assert r.headers["Cache-Control"] == "no-store"
    body = r.json()
    VisionCapacityResponse.model_validate(body)     # schéma FERMÉ
    return body


def test_capacity_available(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main, "vision_admission", VisionAdmission(1))
    monkeypatch.setattr(main.probe_client, "list_running_models",
                        lambda timeout=5: ["mistral-small3.2:24b"])
    body = _cap(clientapp)
    assert body == {
        "provider": "ai_gateway", "contract_version": "1", "scope": "instance",
        "environment": main.settings.environment, "advisory": True,
        "vision": {"enabled": True, "status": "available", "max_concurrency": 1,
                   "in_flight": 0, "available_slots": 1, "retry_after_seconds": 10},
        "model": {"logical": "vision-large", "resident": True, "status": "ok"},
    }
    assert body["vision"]["retry_after_seconds"] == main.VISION_RETRY_AFTER_SECONDS


def test_capacity_saturated_is_200(clientapp, backend, monkeypatch):
    held = VisionAdmission(1); assert held.try_acquire()
    monkeypatch.setattr(main, "vision_admission", held)
    body = _cap(clientapp)
    assert body["vision"]["status"] == "saturated"
    assert body["vision"]["in_flight"] == 1 and body["vision"]["available_slots"] == 0
    held.release()
    assert _cap(clientapp)["vision"]["status"] == "available"


def test_capacity_disabled(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.settings, "vision_enabled", False)
    calls = []
    monkeypatch.setattr(main.probe_client, "list_running_models",
                        lambda timeout=5: calls.append(timeout) or [])
    body = _cap(clientapp)
    assert body["vision"]["enabled"] is False and body["vision"]["status"] == "disabled"
    assert body["vision"]["max_concurrency"] >= 1 and body["vision"]["available_slots"] >= 1
    # résidence : diagnostic toujours lu avec un délai court, jamais /api/generate
    assert calls == [1]


def test_capacity_max_concurrency_matches_settings_and_admission(clientapp, backend):
    body = _cap(clientapp)
    assert body["vision"]["max_concurrency"] == main.vision_admission.max_concurrency
    assert main.vision_admission.max_concurrency == max(1, main.settings.vision_max_concurrency)


def test_capacity_model_telemetry_failure_keeps_200(clientapp, backend, monkeypatch):
    def boom(timeout=5):
        raise UpstreamError("échec /api/ps: timeout")
    monkeypatch.setattr(main.probe_client, "list_running_models", boom)
    body = _cap(clientapp)
    assert body["vision"]["status"] == "available"
    assert body["model"] == {"logical": "vision-large", "resident": None, "status": "unavailable"}


def test_capacity_model_not_resident(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.probe_client, "list_running_models", lambda timeout=5: ["llama3.1:8b"])
    body = _cap(clientapp)
    assert body["model"] == {"logical": "vision-large", "resident": False, "status": "ok"}
    assert body["vision"]["status"] == "available"   # non résident ≠ occupé


def test_capacity_has_no_side_effects(clientapp, backend, monkeypatch):
    adm = VisionAdmission(1)
    monkeypatch.setattr(main, "vision_admission", adm)
    generate_calls = []
    monkeypatch.setattr(main.vision_backend, "analyze",
                        lambda plan: generate_calls.append(plan) or ok_photo_result())
    counters.reset(); gauges.reset()
    before = counters.snapshot()
    for _ in range(5):
        _cap(clientapp)
    assert _snap(adm) == (1, 0, 1)
    assert generate_calls == [] and backend.plans == []
    assert counters.snapshot() == before and gauges.total() == 0


def test_capacity_never_calls_generate_or_loads_model(clientapp, backend, monkeypatch):
    import requests as _requests
    calls = []

    def fake_post(url, *a, **kw):
        calls.append(url)
        raise AssertionError("aucun POST (api/generate) ne doit partir de la sonde")
    monkeypatch.setattr(_requests, "post", fake_post)
    gets = []
    monkeypatch.setattr(main.probe_client, "list_running_models",
                        lambda timeout=5: gets.append(("ps", timeout)) or [])
    _cap(clientapp)
    assert calls == [] and gets == [("ps", 1)]


def test_capacity_exposes_no_sensitive_fields(clientapp, backend, monkeypatch):
    held = VisionAdmission(1); assert held.try_acquire()
    monkeypatch.setattr(main, "vision_admission", held)
    text = json.dumps(_cap(clientapp))
    for forbidden in ("request_id", "pid", "cmd", "client", "prompt", "image", "sha256",
                      "payload", "path", "task", "gpu", "busy", "queue", "oldest", "estimated"):
        assert forbidden not in text, forbidden


def test_capacity_contract_is_strict():
    base = {"provider": "ai_gateway", "contract_version": "1", "scope": "instance",
            "environment": "staging", "advisory": True,
            "vision": {"enabled": True, "status": "available", "max_concurrency": 1,
                       "in_flight": 0, "available_slots": 1, "retry_after_seconds": 10},
            "model": {"logical": "vision-large", "resident": None, "status": "unavailable"}}
    VisionCapacityResponse.model_validate(base)
    bad = [
        {**base, "extra": 1},
        {**base, "scope": "host"},
        {**base, "advisory": False},
        {**base, "contract_version": "2"},
        {**base, "vision": {**base["vision"], "status": "degraded"}},
        {**base, "vision": {**base["vision"], "busy": True}},
        {**base, "vision": {**base["vision"], "in_flight": "0"}},
        {**base, "vision": {**base["vision"], "in_flight": -1}},
        {**base, "model": {**base["model"], "status": "loading"}},
    ]
    for payload in bad:
        with pytest.raises(ValidationError):
            VisionCapacityResponse.model_validate(payload)


def test_capacity_in_openapi_and_analyze_unchanged(clientapp):
    spec = clientapp.get("/openapi.json").json()
    assert "/v1/vision/capacity" in spec["paths"]
    get = spec["paths"]["/v1/vision/capacity"]["get"]
    assert "$ref" in json.dumps(get["responses"]["200"])
    assert "VisionCapacityResponse" in spec["components"]["schemas"]
    # Lot SEMX-4A : "3" (semantic_extract, désactivé) ajouté de façon ADDITIVE.
    assert spec["components"]["schemas"]["VisionAnalyzeRequest"]["properties"]["schema_version"]["enum"] == ["1", "2", "3"]
    assert len(spec["paths"]) == 19 # + GET /health/live (non inférentiel)


# -------------------------------------------- non-régression V1/V2/fingerprint
def test_v1_v2_and_fingerprint_unchanged(clientapp, backend):
    backend.results = [ok_photo_result(), ok_photo_result()]
    v1 = _post(clientapp).json()
    assert v1["schema_version"] == "1" and "fingerprint" not in json.dumps(v1)
    v2 = _post(clientapp, schema_version="2").json()
    assert v2["schema_version"] == "2" and v2["image"]["normalizer_id"] == "vision-image-normalizer/1"
    assert vision_pipeline_fingerprint(pipeline_identity("describe", DIGEST, DESCRIBE_PROMPT)) == \
        "vision-v1:9c68b7db03303789ffb1b34d5ff0da811a8c042cfc912eab89db0fd702dced39"
