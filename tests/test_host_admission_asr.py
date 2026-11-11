"""Lot HOST-WIDE-ADMISSION — ASR : lease hôte composée avec la garde VRAM
locale existante (conservée), tenue jusqu'au premier état terminal observé,
rendue sur refus backend/abandon ; contrôleur injoignable = repli local."""

import pytest

from app import asr_routes, main
from app.asr_client import AsrBackendRejected
from tests.test_asr_routes import JOB, FakeBackend, backend, canonical_result, multipart, staging  # noqa: F401
from tests.test_asr_routes import _post as asr_post
from tests.test_host_admission_gateway import FakeController


@pytest.fixture
def ctrl(monkeypatch):
    fc = FakeController()
    monkeypatch.setattr(main.host_admission, "_reserve", fc.reserve)
    monkeypatch.setattr(main.host_admission, "_release_remote", fc.release)
    monkeypatch.setattr(main.settings, "host_admission_enabled", True)
    monkeypatch.setattr(main.settings, "host_admission_mode", "enforce")
    monkeypatch.setattr(main.settings, "host_admission_asr_enabled", True)
    asr_routes._asr_host_leases.clear()
    return fc


def test_asr_sync_completion_releases_lease(clientapp, backend, ctrl):
    backend.result = canonical_result()
    body, h = multipart({"mode": "auto"})
    r = asr_post(clientapp, body, h)
    assert r.status_code == 200, r.text
    assert ctrl.reserves == [("asr", "asr-whisper-turbo", None)] and ctrl.releases == ["lease1"] and not ctrl.active
    assert backend.created[0]["owner_env"] == "staging"


def test_asr_async_lease_held_until_terminal_observed(clientapp, backend, ctrl):
    backend.result = canonical_result()
    body, h = multipart({"mode": "async"})
    r = asr_post(clientapp, body, h)
    assert r.status_code == 202 and ctrl.active == {"lease1"}            # job en cours : lease tenue
    assert JOB in asr_routes._asr_host_leases
    backend.job_status = "completed"
    g = clientapp.get(f"/v1/audio/transcriptions/{JOB}")
    assert g.status_code == 200 and not ctrl.active and JOB not in asr_routes._asr_host_leases
    clientapp.get(f"/v1/audio/transcriptions/{JOB}")                       # relectures : aucune double libération
    assert ctrl.releases == ["lease1"]


def test_asr_lease_released_when_backend_rejects_job(clientapp, backend, ctrl):
    backend.create_exc = AsrBackendRejected("POST /jobs: HTTP 503", code="resource_saturated", status=503)
    body, h = multipart({})
    r = asr_post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "resource_saturated"
    assert ctrl.releases == ["lease1"] and not ctrl.active


def test_asr_local_vram_guard_still_applies_after_host_lease(clientapp, backend, ctrl, monkeypatch):
    # Garde locale CONSERVÉE : résident Ollama 19 Gio -> budget dépassé -> 503, lease rendue
    monkeypatch.setattr(main.probe_client, "list_running_models_vram",
                        lambda timeout=2: [{"name": "qwen3:30b-instruct", "size_vram": 19 * 1024**3}])
    body, h = multipart({})
    r = asr_post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "resource_saturated"
    assert backend.created == [] and ctrl.releases == ["lease1"] and not ctrl.active


def test_asr_deferred_by_host_before_local_guard_and_backend(clientapp, backend, ctrl):
    ctrl.outcome, ctrl.reason = "DEFERRED", "incompatible_active_workload"
    body, h = multipart({})
    r = asr_post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "host_capacity_deferred"
    assert backend.created == [] and ctrl.active == set()


def test_asr_validation_failure_requests_no_lease(clientapp, backend, ctrl):
    body, h = multipart({"timestamps": "frames"})
    assert asr_post(clientapp, body, h).status_code == 400 and ctrl.reserves == []


def test_asr_controller_unavailable_falls_back_to_local_guards(clientapp, backend, ctrl):
    ctrl.fail = True
    backend.result = canonical_result()
    body, h = multipart({"mode": "auto"})
    assert asr_post(clientapp, body, h).status_code == 200          # classe légère : gardes locales
    assert ctrl.active == set() and backend.created