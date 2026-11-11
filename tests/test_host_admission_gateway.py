"""Lot HOST-WIDE-ADMISSION — intégration côté Gateway : lease demandée après
validation et avant le backend, rendue dans TOUS les chemins de sortie ; 503
typés sur DEFERRED/REJECTED ; contrôleur injoignable = fail-closed pour les
classes lourdes, repli local pour les légères ; mode shadow ; flags OFF =
comportement strictement inchangé. Contrôleur simulé en mémoire."""

import json
import logging

import pytest

from app import main
from app.errors import UpstreamError
from app.host_admission import HostAdmissionClient, HostLease
from app.metrics import counters
from app.ollama_client import ENDPOINT_EMBED, EmbedResult
from tests.test_vision_media import _post, backend, ok_photo_result, vision_ready  # noqa: F401


class FakeController:
    """Simule /v1/admission/reserve et DELETE : décisions scriptées, leases suivies."""
    def __init__(self, outcome="GRANTED", reason="host_capacity_available", retry_after=3, fail=False):
        self.outcome, self.reason, self.retry_after, self.fail = outcome, reason, retry_after, fail
        self.reserves, self.releases, self.active = [], [], set()
        self.n = 0

    def reserve(self, workload_class, model_logical, idempotency_key, retry_family=None):
        if self.fail:
            raise ConnectionRefusedError("controller down")      # NOT_SENT prouvé (lot ORPHAN-LEASE)
        self.reserves.append((workload_class, model_logical, retry_family))
        self.idem_keys = getattr(self, "idem_keys", []) + [idempotency_key]
        if self.outcome != "GRANTED":
            return {"outcome": self.outcome, "reason": self.reason, "retry_after_seconds": self.retry_after}
        self.n += 1; lid = f"lease{self.n}"; self.active.add(lid)
        return {"outcome": "GRANTED", "reason": "host_capacity_available", "lease": {"lease_id": lid}}

    def release(self, lease_id, backend_started=True):
        self.releases.append(lease_id); self.active.discard(lease_id)
        self.release_flags = getattr(self, "release_flags", []) + [backend_started]
        return True                                              # libération confirmée


@pytest.fixture
def ctrl(monkeypatch):
    fc = FakeController()
    monkeypatch.setattr(main.host_admission, "_reserve", fc.reserve)
    monkeypatch.setattr(main.host_admission, "_release_remote", fc.release)
    monkeypatch.setattr(main.settings, "host_admission_enabled", True)
    monkeypatch.setattr(main.settings, "host_admission_mode", "enforce")
    for f in ("vision", "llm", "embedding", "rerank", "asr"):
        monkeypatch.setattr(main.settings, f"host_admission_{f}_enabled", True)
    return fc


def _gen(clientapp, **kw):
    body = {"prompt": "Compte de un a trois.", "model": "llama-chat", "temperature": 0.0}; body.update(kw)
    return clientapp.post("/v1/generate", json=body)


# ------------------------------------------------------------- défaut OFF
def test_default_off_never_contacts_controller(clientapp, backend, monkeypatch):
    calls = []
    monkeypatch.setattr(main.host_admission, "_reserve", lambda *a, **k: calls.append(a) or {"outcome": "GRANTED"})
    assert main.settings.host_admission_enabled is False
    backend.results = [ok_photo_result()]
    assert _post(clientapp).status_code == 200
    assert calls == []
    assert main.host_admission.acquire("vision", "vision-large") is None


def test_rollback_flag_restores_local_semantics(clientapp, backend, ctrl, monkeypatch):
    monkeypatch.setattr(main.settings, "host_admission_enabled", False)
    backend.results = [ok_photo_result()]
    assert _post(clientapp).status_code == 200 and ctrl.reserves == []


# ----------------------------------------------------------------- vision
def test_vision_lease_acquired_after_validation_and_released(clientapp, backend, ctrl):
    backend.results = [ok_photo_result()]
    assert _post(clientapp, request_id="rid-v").status_code == 200
    (cls, model, fam), = ctrl.reserves
    assert (cls, model) == ("vision", "vision-large") and fam.startswith("rf1-") and len(fam) == 36 and "rid-v" not in fam
    assert ctrl.releases == ["lease1"] and not ctrl.active
    # rejets AVANT admission : aucune lease demandée
    assert _post(clientapp, schema_version="3").status_code == 400
    assert _post(clientapp, mime="image/jpeg").status_code == 400
    assert len(ctrl.reserves) == 1


def test_vision_lease_released_on_backend_failure_and_local_guard_refusal(clientapp, backend, ctrl, monkeypatch):
    backend.results = [UpstreamError("échec /api/generate: HTTP 500")]
    assert _post(clientapp).status_code == 502
    assert ctrl.releases == ["lease1"]
    from app.vision_admission import VisionAdmission
    held = VisionAdmission(1); assert held.try_acquire()
    monkeypatch.setattr(main, "vision_admission", held)
    r = _post(clientapp)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "vision_overloaded"
    assert ctrl.releases == ["lease1", "lease2"] and not ctrl.active        # garde locale conservée, lease rendue


def test_vision_deferred_and_rejected_envelopes(clientapp, backend, ctrl):
    ctrl.outcome, ctrl.reason, ctrl.retry_after = "DEFERRED", "workload_slot_saturated", 7
    r = _post(clientapp, request_id="rid-d")
    d = r.json()["detail"]
    assert r.status_code == 503 and d["code"] == "host_capacity_deferred" and d["reason"] == "workload_slot_saturated"
    assert d["retryable"] is True and d["retry_after_seconds"] == 7 and r.headers["Retry-After"] == "7" and d["request_id"] == "rid-d"
    assert backend.plans == []
    ctrl.outcome, ctrl.reason = "REJECTED", "unknown_profile"
    r = _post(clientapp)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "host_capacity_rejected" and r.json()["detail"]["retryable"] is False


def test_controller_unavailable_fails_closed_for_heavy_classes(clientapp, backend, ctrl):
    ctrl.fail = True
    counters.reset()
    r = _post(clientapp)
    d = r.json()["detail"]
    assert r.status_code == 503 and d["code"] == "host_admission_unavailable" and d["retryable"] is True
    assert r.headers["Retry-After"] == "5" and backend.plans == []
    assert counters.snapshot()["host_admission_unavailable_total"] == 1


def test_shadow_mode_observes_without_controlling(clientapp, backend, ctrl, monkeypatch, caplog):
    monkeypatch.setattr(main.settings, "host_admission_mode", "shadow")
    caplog.set_level(logging.INFO)
    ctrl.outcome, ctrl.reason = "DEFERRED", "workload_slot_saturated"
    backend.results = [ok_photo_result()]
    assert _post(clientapp).status_code == 200                      # non contrôlée
    ctrl.outcome = "GRANTED"; backend.results = [ok_photo_result()]
    assert _post(clientapp).status_code == 200
    assert ctrl.releases == ["lease1"] and not ctrl.active          # lease shadow rendue aussitôt
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "host_admission.decision" in text and "shadow=True" in text and "outcome=DEFERRED" in text
    ctrl.fail = True; backend.results = [ok_photo_result()]
    assert _post(clientapp).status_code == 200                      # shadow : indisponible = sans effet


# ------------------------------------------------------- generate / chat / embed / rerank
def test_generate_lease_lifecycle(clientapp, ctrl, monkeypatch):
    monkeypatch.setattr(main.client, "generate", lambda **kw: {"response": "un deux trois", "done": True, "done_reason": "stop",
                                                              "eval_count": 3, "prompt_eval_count": 5})
    r = _gen(clientapp)
    assert r.status_code == 200, r.text
    assert ctrl.reserves[-1][0] == "llm_generation" and ctrl.reserves[-1][1] == "llama-chat" and not ctrl.active
    def boom(**kw): raise UpstreamError("échec /api/generate: HTTP 500")
    monkeypatch.setattr(main.client, "generate", boom)
    assert _gen(clientapp).status_code == 502 and not ctrl.active
    ctrl.outcome, ctrl.reason = "DEFERRED", "model_swap_headroom"
    r = _gen(clientapp); assert r.status_code == 503 and r.json()["detail"]["code"] == "host_capacity_deferred"
    # validation : aucune lease
    n = len(ctrl.reserves); assert _gen(clientapp, prompt="").status_code == 400 and len(ctrl.reserves) == n


def _fake_embed(**kw):
    return EmbedResult([[1.0] + [0.0] * 767 for _ in kw.get("inputs", [])], ENDPOINT_EMBED, fallback_used=False, ollama_metadata={})


def test_embeddings_and_rerank_light_classes_fallback_when_controller_down(clientapp, ctrl, monkeypatch):
    ctrl.fail = True; counters.reset()
    monkeypatch.setattr(main.client, "embed", _fake_embed)
    r = clientapp.post("/v1/embeddings", json={"texts": ["bonjour"]})
    assert r.status_code == 200, r.text
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main.rerank_client, "rerank", lambda q, docs: [{"index": i, "score": float(i)} for i in range(len(docs))])
    r = clientapp.post("/v1/rerank", json={"query": "q", "documents": [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]})
    assert r.status_code == 200, r.text
    assert counters.snapshot()["host_admission_local_fallback_total"] == 2


def test_rerank_lease_released_when_local_semaphore_refuses(clientapp, ctrl, monkeypatch):
    import threading
    held = threading.BoundedSemaphore(1); held.acquire()
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main, "_rerank_semaphore", held)
    r = clientapp.post("/v1/rerank", json={"query": "q", "documents": [{"id": "a", "text": "alpha"}]})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "rerank_saturated"
    assert ctrl.releases == ["lease1"] and not ctrl.active


def test_embeddings_lease_released_on_success_and_failure(clientapp, ctrl, monkeypatch):
    monkeypatch.setattr(main.client, "embed", _fake_embed)
    assert clientapp.post("/v1/embeddings", json={"texts": ["bonjour"]}).status_code == 200
    assert ctrl.reserves[-1][0] == "embedding" and not ctrl.active
    def boom(**kw): raise UpstreamError("échec /api/embed")
    monkeypatch.setattr(main.client, "embed", boom)
    assert clientapp.post("/v1/embeddings", json={"texts": ["bonjour"]}).status_code == 502 and not ctrl.active


# ------------------------------------------------------------------- client pur
def test_client_release_is_idempotent_and_none_safe(monkeypatch):
    c = HostAdmissionClient(); calls = []
    monkeypatch.setattr(c, "_release_remote", lambda lid, backend_started=True: calls.append(lid) or True)
    c.release(None)
    lease = HostLease("abc", "vision", "vision-large")
    c.release(lease); c.release(lease)
    assert calls == ["abc"] and lease.released is True


def test_client_payload_carries_environment_as_attribution_only(monkeypatch):
    c = HostAdmissionClient(); sent = {}
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"outcome": "GRANTED", "reason": "host_capacity_available", "lease": {"lease_id": "L"}}
    def fake_post(url, json=None, timeout=None): sent.update(json); sent["url"] = url; sent["timeout"] = timeout; return R()
    monkeypatch.setattr(c, "_post", fake_post)
    monkeypatch.setattr(main.settings, "host_admission_enabled", True); monkeypatch.setattr(main.settings, "host_admission_vision_enabled", True)
    lease = c.acquire("vision", "vision-large", "rid")
    assert lease.lease_id == "L" and sent["requester_environment"] == main.settings.environment
    assert sent["url"].startswith("http://127.0.0.1:8500") and sent["timeout"] <= 2.0 and sent["contract_version"] == "1"
    assert "prompt" not in json.dumps(sent)
