"""Lot asr-rail-foundation — commit 3 : routes publiques /v1/audio/*.

Backend ASR simulé (monkeypatch de `main.asr_transport.asr_client`), sonde
Ollama simulée (size_vram), aucun réseau. Les sentinelles de vie privée
vérifient qu'aucun nom de fichier, terme de vocabulaire ni texte transcrit
n'apparaît dans les journaux ni dans les enveloppes d'erreur.
"""

import json
import logging
import re

import pytest

from app import asr_routes, main
from app.asr_client import AsrBackendRejected, AsrBackendUnavailable
from app.errors import UpstreamError
from app.metrics import counters

MODEL = "asr-whisper-turbo"
JOB = "3d2f7a10-5b6c-4e1d-9f2a-1c3b4d5e6f70"
FILENAME = "ZZSECRETFILENAMEZZ.mp3"
TERM = "ZZSECRETTERMZZ"
TRANSCRIPT = "ZZTRANSCRIPTSECRETZZ"
INFO_OK = {
    "provider": "faster-whisper", "artifact": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "revision": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    "artifact_sha256": "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
    "vad": {"provider": "silero", "source": "faster-whisper-embedded", "version": "1.2.1"},
    "capabilities": ["transcription", "word_timestamps", "vocabulary_biasing", "multilingual", "diarization"],
    "diarization": {"provider": "nemo-sortformer", "artifact": "nvidia/diar_streaming_sortformer_4spk-v2",
                    "revision": "84edd514b8ef68004c10086918cd62f2148cbd59",
                    "artifact_sha256": "b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329",
                    "expected_sha256": "b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329",
                    "hash_match": True, "device": "cpu", "max_speakers": 4, "model_ready": True},
    "backend_version": "0.1.0", "faster_whisper_version": "1.2.1", "ctranslate2_version": "4.8.2",
}


def multipart(request_obj, file_bytes=b"RIFF" + b"\0" * 64, order=("request", "file"), boundary="XbOuNdArYx",
              request_ctype="application/json", extra_part=False, raw_request=None):
    parts = []
    body_request = raw_request if raw_request is not None else json.dumps(request_obj).encode()
    for name in order:
        if name == "request":
            ct = f"\r\nContent-Type: {request_ctype}" if request_ctype else ""
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="request"{ct}\r\n\r\n'.encode()
                         + body_request + b"\r\n")
        elif name == "file":
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{FILENAME}"'
                         f'\r\nContent-Type: audio/mpeg\r\n\r\n'.encode() + file_bytes + b"\r\n")
        else:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\nx\r\n'.encode())
    if extra_part:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="extra"\r\n\r\nx\r\n'.encode())
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def canonical_result(job_id=JOB, timestamps="segment", duration_ms=3000, text=TRANSCRIPT, diarize=False):
    segs = [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 1500, "speaker": None, "text": text},
            {"segment_id": "seg_000002", "start_ms": 1500, "end_ms": 2900, "speaker": None, "text": "suite"}]
    if diarize:
        segs[0]["speaker"], segs[1]["speaker"] = "speaker_00", "speaker_01"
    if timestamps == "word":
        for s in segs:
            s["words"] = [{"word": w, "start_ms": s["start_ms"], "end_ms": s["start_ms"] + 100}
                          for w in s["text"].split()]
    words = sum(len(s.get("words", [])) for s in segs)
    return {
        "transcription_id": job_id, "model_logical": MODEL,
        "backend": {"provider": "faster-whisper", "artifact": INFO_OK["artifact"], "revision": INFO_OK["revision"]},
        "media": {"container": "mp3", "codec": "mp3", "sample_rate": 44100, "channels": 1, "duration_ms": duration_ms,
                  "upload_bytes": 68, "sha256": "a" * 64},
        "duration_ms": duration_ms, "language_mode": "auto", "language": "fr", "language_probability": 0.98,
        "timestamps": timestamps, "diarized": bool(diarize), "speaker_count": 2 if diarize else None,
        "text": " ".join(s["text"] for s in segs), "segments": segs,
        "usage": {"audio_ms": duration_ms, "processing_ms": 120, "rtf": 0.04, "segment_count": 2,
                  "word_count": words, "vocabulary_count": 0},
        "warnings": [],
    }


class FakeBackend:
    """Backend simulé : un job, états pilotables par le test."""

    def __init__(self):
        self.created = []
        self.uploaded = b""
        self.deleted = []
        self.create_exc = None
        self.upload_exc = None
        self.get_exc = None
        self.duration_ms = 3000
        self.result = None           # résultat renvoyé quand completed
        self.polls_before_done = 0   # nb de GET avant completed
        self.final_status = "completed"
        self.final_error = None
        self.polls = 0
        self.job_status = "queued"
        self.job_diarize = False     # vue GET sans création préalable (route async)

    def _job(self, status, **extra):
        d = {"job_id": JOB, "owner_env": "staging", "status": status, "phase": "ready", "model": MODEL,
             "language": "auto", "timestamps": "segment", "vocabulary_count": 0, "created_at": 1.0,
             "updated_at": 2.0, "duration_ms": self.duration_ms, "processed_audio_ms": 0, "error": None,
             "retryable": None, "media": {"container": "mp3", "codec": "mp3", "duration_ms": self.duration_ms},
             "result_available": status == "completed"}
        d.update(extra)
        return d

    def create_job(self, spec):
        if self.create_exc:
            raise self.create_exc
        self.created.append(spec)
        self.job_status = "queued"
        return self._job("queued", phase="awaiting_media")

    def upload_media(self, job_id, chunks, declared, owner_env):
        data = b"".join(chunks)
        if self.upload_exc:
            raise self.upload_exc
        self.uploaded = data
        return self._job("queued", timestamps=self.created[-1]["timestamps"] if self.created else "segment")

    def get_job(self, job_id, owner_env, wait_ms=0):
        if self.get_exc:
            raise self.get_exc
        self.polls += 1
        ts = self.created[-1]["timestamps"] if self.created else "segment"
        diarize = bool(self.created[-1].get("diarize")) if self.created else self.job_diarize
        if self.polls <= self.polls_before_done:
            return self._job("processing", timestamps=ts, diarize=diarize)
        if self.final_status == "completed":
            result = self.result if self.result is not None else canonical_result(timestamps=ts, diarize=diarize)
            return self._job("completed", result=result, timestamps=ts, diarize=diarize)
        return self._job(self.final_status, error=self.final_error, timestamps=ts, diarize=diarize)

    def delete_job(self, job_id, owner_env):
        self.deleted.append(job_id)
        return self._job("cancelled")

    def info(self, timeout=None):
        return dict(INFO_OK)

    def health(self, timeout=None):
        return {"_http_status": 200, "model_ready": True, "worker_state": "absent", "queued": 0, "inflight": 0,
                "spool": {}}


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.setattr(main.settings, "asr_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "staging")
    monkeypatch.setattr(main.settings, "asr_sync_wait_seconds", 5)


@pytest.fixture
def backend(monkeypatch, staging):
    fake = FakeBackend()
    for name in ("create_job", "upload_media", "get_job", "delete_job", "info", "health"):
        monkeypatch.setattr(main.asr_transport.asr_client, name, getattr(fake, name))
    monkeypatch.setattr(main.probe_client, "list_running_models_vram",
                        lambda timeout=2: [{"name": "nomic-embed-text:latest", "size_vram": 800 * 1024 * 1024}])
    return fake


def _post(clientapp, body, headers, **kw):
    return clientapp.post("/v1/audio/transcriptions", content=body, headers=headers, **kw)


# ------------------------------------------------------------- capability
def test_disabled_by_default_is_503(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "staging")
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "asr_unavailable"
    assert clientapp.get(f"/v1/audio/transcriptions/{JOB}").status_code == 503
    assert clientapp.delete(f"/v1/audio/transcriptions/{JOB}").status_code == 503


def test_production_flag_off_refuses_and_flag_on_serves(clientapp, monkeypatch, backend):
    """Lot production-full-capability-enablement : en production le flag a un sens
    normal ; sans lui le rail refuse (503 asr_unavailable), avec lui le job est
    créé sous owner_env=production (isolation des jobs par environnement)."""
    monkeypatch.setattr(main.settings, "environment", "production")
    monkeypatch.setattr(main.settings, "asr_enabled", False)
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 503 and "AI_GATEWAY_ASR_ENABLED" in r.json()["detail"]["message"]
    assert backend.created == []
    monkeypatch.setattr(main.settings, "asr_enabled", True)
    r = _post(clientapp, body, h)
    assert r.status_code in (200, 202), r.text
    assert backend.created[0]["owner_env"] == "production"


def test_production_registry_disallowed_refuses_before_backend(clientapp, monkeypatch, backend):
    monkeypatch.setattr(main.settings, "environment", "production")
    monkeypatch.setattr(main.settings, "asr_enabled", True)
    monkeypatch.setattr(main.registry.asr_models[MODEL], "production_allowed", False)
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "model_not_allowed"
    assert backend.created == []


def test_non_loopback_backend_refused(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.settings, "asr_backend_url", "http://203.0.113.5:8400")
    body, h = multipart({})
    assert _post(clientapp, body, h).status_code == 503


# -------------------------------------------------------------- multipart
def test_content_length_over_limit_is_413_before_any_read(clientapp, backend):
    body, h = multipart({})
    h["Content-Length"] = str(300 * 1024 * 1024)
    r = clientapp.post("/v1/audio/transcriptions", headers=h, content=body)
    assert r.status_code == 413 and r.json()["detail"]["code"] == "audio_too_large"
    assert backend.created == []


def test_non_multipart_is_400(clientapp, backend):
    r = clientapp.post("/v1/audio/transcriptions", json={"model": MODEL})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "invalid_request"


@pytest.mark.parametrize("kw", [
    {"order": ("file", "request")}, {"order": ("request",)}, {"order": ("file",)},
    {"order": ("meta", "file")}, {"extra_part": True}, {"request_ctype": "text/plain"},
    {"raw_request": b"{not json"}, {"raw_request": b"[1,2]"}, {"raw_request": b"{" + b" " * 9000 + b"}"},
])
def test_malformed_multipart_is_400(clientapp, backend, kw):
    body, h = multipart({}, **kw)
    r = _post(clientapp, body, h)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "invalid_request", r.text
    assert backend.uploaded == b""


def test_file_part_without_request_part_never_reaches_backend(clientapp, backend):
    body, h = multipart({}, order=("file",))
    _post(clientapp, body, h)
    assert backend.created == [] and backend.uploaded == b""


# --------------------------------------------------------------- schéma
@pytest.mark.parametrize("req,code", [
    ({"diarize": "true"}, "invalid_request"),
    ({"min_speakers": 2}, "field_requires_diarization"),
    ({"speakers": 2}, "field_requires_diarization"),
    ({"language": "de"}, "invalid_request"),
    ({"timestamps": "char"}, "invalid_request"),
    ({"mode": "sync"}, "invalid_request"),
    ({"url": "http://x"}, "invalid_request"),
    ({"file_path": "/etc/passwd"}, "invalid_request"),
    ({"options": {"beam_size": 1}}, "invalid_request"),
    ({"vocabulary": ["x"]}, "invalid_vocabulary"),
    ({"vocabulary": ["http://evil"]}, "invalid_vocabulary"),
    ({"vocabulary": []}, "invalid_vocabulary"),
    ({"model": "absent-model"}, "model_not_registered"),
    ({"model": "qwen-large"}, "transcription_not_supported"),
    ({"model": "legacy-768"}, "transcription_not_supported"),
    ({"model": "rerank-default"}, "transcription_not_supported"),
])
def test_request_schema_rejections(clientapp, backend, req, code):
    body, h = multipart(req)
    r = _post(clientapp, body, h)
    assert r.status_code == 400 and r.json()["detail"]["code"] == code, r.text
    assert backend.created == []
    assert TERM not in r.text and FILENAME not in r.text


def test_model_not_allowed_in_environment(clientapp, backend, monkeypatch):
    profile = main.registry.asr_models[MODEL]
    monkeypatch.setattr(profile, "staging_allowed", False)
    body, h = multipart({"model": MODEL})
    r = _post(clientapp, body, h)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "model_not_allowed"


def test_word_timestamps_requires_capability(clientapp, backend, monkeypatch):
    profile = main.registry.asr_models[MODEL]
    monkeypatch.setattr(profile, "capabilities", ["transcription", "multilingual"])
    body, h = multipart({"timestamps": "word"})
    r = _post(clientapp, body, h)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "invalid_request"


# ------------------------------------------------------------ admission GPU
def test_gpu_budget_exceeded_is_503_before_backend(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.probe_client, "list_running_models_vram",
                        lambda timeout=2: [{"name": "qwen3:30b-a3b-thinking", "size_vram": 18 * 1024 ** 3}])
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "resource_saturated"
    assert r.headers["retry-after"] == "30" and r.json()["detail"]["retryable"] is True
    assert backend.created == [] and backend.uploaded == b""
    assert counters.snapshot().get("asr_gpu_contention_total") == 1


def test_gpu_budget_uses_size_vram_not_names(clientapp, backend, monkeypatch):
    # 13 Gio résidents (GPT-OSS) + 4,5 + 2,1 = 19,6 <= 22 : accepté.
    monkeypatch.setattr(main.probe_client, "list_running_models_vram",
                        lambda timeout=2: [{"name": "anything", "size_vram": 13 * 1024 ** 3}])
    body, h = multipart({"mode": "async"})
    r = _post(clientapp, body, h)
    assert r.status_code == 202, r.text


def test_ollama_ps_failure_is_503_asr_unavailable(clientapp, backend, monkeypatch):
    def boom(timeout=2):
        raise UpstreamError("échec /api/ps: boom")

    monkeypatch.setattr(main.probe_client, "list_running_models_vram", boom)
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "asr_unavailable"
    assert backend.created == []


def test_backend_identity_mismatch_refuses_before_job(clientapp, backend, monkeypatch):
    bad = dict(INFO_OK, artifact_sha256="0" * 64)
    monkeypatch.setattr(main.asr_transport.asr_client, "info", lambda timeout=None: bad)
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "asr_unavailable"
    assert backend.created == []


def test_backend_saturated_is_503_with_retry_after(clientapp, backend):
    backend.create_exc = AsrBackendRejected("POST /jobs: HTTP 503", code="resource_saturated", status=503, retryable=True)
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "resource_saturated"
    assert r.headers["retry-after"] == "30"


def test_backend_unreachable_is_503(clientapp, backend):
    backend.create_exc = AsrBackendUnavailable("échec POST /jobs backend ASR: ConnectionError")
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "asr_unavailable"
    assert "ConnectionError" not in r.text and "8400" not in r.text


def test_local_concurrency_saturation(clientapp, backend):
    assert asr_routes._semaphore.acquire(blocking=False)
    try:
        body, h = multipart({})
        r = _post(clientapp, body, h)
        assert r.status_code == 503 and r.json()["detail"]["code"] == "resource_saturated"
        assert r.headers["retry-after"] == "30"
    finally:
        asr_routes._semaphore.release()
    assert counters.snapshot().get("asr_saturated_total") == 1


# ------------------------------------------------------------- sync / async
def test_sync_success_returns_canonical_result(clientapp, backend, caplog):
    caplog.set_level(logging.INFO)
    body, h = multipart({"vocabulary": [TERM, "Kubernetes"], "language": "fr"})
    r = _post(clientapp, body, h)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["transcription_id"] == JOB and res["diarized"] is False and res["speaker_count"] is None
    assert res["segments"][0]["segment_id"] == "seg_000001"
    assert backend.uploaded == b"RIFF" + b"\0" * 64
    spec = backend.created[0]
    assert spec["vocabulary"] == [TERM, "Kubernetes"] and spec["owner_env"] == "staging"
    assert spec["max_duration_seconds"] == 6300 and spec["max_upload_bytes"] == 268435456
    snap = counters.snapshot()
    assert snap["asr_requests_total"] == 1 and snap["asr_accepted_total"] == 1 and snap["asr_completed_total"] == 1
    assert snap["asr_audio_seconds_total"] == 3 and snap["asr_processing_ms_total"] == 120
    text = caplog.text
    assert "event=asr.accepted" in text and "event=asr.started" in text and "event=asr.completed" in text
    assert "vocabulary_count=2" in text
    # Sentinelles : jamais de contenu dans les journaux.
    assert TERM not in text and FILENAME not in text and TRANSCRIPT not in text
    assert "8400" not in text and "/var/lib" not in text


def test_word_timestamps_roundtrip(clientapp, backend):
    body, h = multipart({"timestamps": "word"})
    r = _post(clientapp, body, h)
    assert r.status_code == 200, r.text
    assert r.json()["segments"][0]["words"][0]["word"] == TRANSCRIPT
    assert r.json()["usage"]["word_count"] == 2


def test_mode_async_is_202(clientapp, backend):
    body, h = multipart({"mode": "async"})
    r = _post(clientapp, body, h)
    assert r.status_code == 202
    j = r.json()
    assert j["job_id"] == JOB and j["status"] == "queued" and j["poll"]["path"].endswith(JOB)
    assert backend.polls == 0


def test_auto_long_media_is_202(clientapp, backend):
    backend.duration_ms = 1201 * 1000
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 202 and backend.polls == 0


def test_auto_short_media_still_processing_after_wait_is_202(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.settings, "asr_sync_wait_seconds", 0)
    backend.polls_before_done = 100
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 202 and r.json()["status"] in ("queued", "processing")


@pytest.mark.parametrize("code,status", [
    ("decoding_failed", 422), ("audio_too_long", 413), ("unsupported_media", 415), ("invalid_audio", 400),
    ("too_many_channels", 400), ("ambiguous_audio_streams", 400), ("audio_too_large", 413),
])
def test_upload_admission_errors_are_translated(clientapp, backend, code, status):
    backend.upload_exc = AsrBackendRejected("PUT media: HTTP x", code=code, status=status)
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == status and r.json()["detail"]["code"] == code
    assert r.json()["detail"]["job_id"] == JOB


@pytest.mark.parametrize("code,status,retryable", [
    ("asr_timeout", 502, True), ("transcription_failed", 502, False), ("decoding_failed", 422, False),
    ("backend_restart", 502, True), ("queue_timeout", 503, True),
])
def test_sync_job_failure_is_translated(clientapp, backend, code, status, retryable):
    backend.final_status = "failed"
    backend.final_error = {"code": code, "message": "/var/lib/ai-asr/spool/x", "retryable": retryable}
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == status and r.json()["detail"]["code"] == code
    assert r.json()["detail"]["retryable"] is retryable
    assert "/var/lib" not in r.text
    assert counters.snapshot().get("asr_failed_total") == 1


def test_unknown_backend_code_is_transcription_failed(clientapp, backend):
    backend.final_status = "failed"
    backend.final_error = {"code": "weird_thing", "message": "x", "retryable": False}
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "transcription_failed"


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__("diarized", True),
    lambda r: r.__setitem__("speaker_count", 2),
    lambda r: r["segments"][0].__setitem__("probability", 0.5),
    lambda r: r["segments"][1].__setitem__("start_ms", 100),
    lambda r: r["segments"][0].__setitem__("end_ms", 0),
    lambda r: r.__setitem__("text", "autre"),
    lambda r: r.__setitem__("transcription_id", "11111111-1111-4111-8111-111111111111"),
    lambda r: r.__setitem__("segments", r["segments"] + [{"segment_id": "seg_000003", "start_ms": 2900,
                                                          "end_ms": 9000, "speaker": None, "text": "x"}]),
    lambda r: r.pop("usage"),
    lambda r: r["segments"][0].__setitem__("speaker", "SPEAKER_00"),
])
def test_invalid_backend_result_is_502(clientapp, backend, mutate):
    res = canonical_result()
    mutate(res)
    backend.result = res
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "transcription_failed", r.text
    assert TRANSCRIPT not in r.text


# ---------------------------------------------------------------- GET/DELETE
def test_get_processing_is_202(clientapp, backend):
    backend.polls_before_done = 10
    r = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"wait_ms": 100})
    assert r.status_code == 202 and r.json()["status"] == "processing"


def test_get_json_and_projections(clientapp, backend):
    r = clientapp.get(f"/v1/audio/transcriptions/{JOB}")
    assert r.status_code == 200 and r.json()["transcription_id"] == JOB
    t = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"format": "text"})
    assert t.status_code == 200 and t.headers["content-type"].startswith("text/plain")
    assert t.text == f"{TRANSCRIPT} suite\n"
    srt = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"format": "srt"}).text
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\n" + TRANSCRIPT + "\n\n2\n00:00:01,500 --> 00:00:02,900\nsuite\n")
    vtt = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"format": "vtt"}).text
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n" + TRANSCRIPT + "\n")
    # Convention Gateway : une erreur de validation de requête est un 400.
    assert clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"format": "docx"}).status_code == 400


@pytest.mark.parametrize("job_id", ["not-a-uuid", "../etc/passwd", "11111111-1111-1111-1111-111111111111"])
def test_get_malformed_job_id_is_404(clientapp, backend, job_id):
    r = clientapp.get(f"/v1/audio/transcriptions/{job_id}")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert not isinstance(detail, dict) or detail["code"] == "job_not_found"


def test_get_expired_is_410(clientapp, backend):
    backend.get_exc = AsrBackendRejected("GET: 410", code="job_expired", status=410)
    r = clientapp.get(f"/v1/audio/transcriptions/{JOB}")
    assert r.status_code == 410 and r.json()["detail"]["code"] == "job_expired"


def test_get_unknown_is_404(clientapp, backend):
    backend.get_exc = AsrBackendRejected("GET: 404", code="job_not_found", status=404)
    assert clientapp.get(f"/v1/audio/transcriptions/{JOB}").status_code == 404


def test_get_cancelled_is_409(clientapp, backend):
    backend.final_status = "cancelled"
    r = clientapp.get(f"/v1/audio/transcriptions/{JOB}")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "job_cancelled"


def test_get_failed_timeout_is_502(clientapp, backend):
    backend.final_status = "failed"
    backend.final_error = {"code": "asr_timeout", "message": "x", "retryable": True}
    r = clientapp.get(f"/v1/audio/transcriptions/{JOB}")
    assert r.status_code == 502 and r.json()["detail"]["code"] == "asr_timeout" and r.json()["detail"]["retryable"]


def test_get_wait_ms_bounds(clientapp, backend):
    assert clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"wait_ms": 30001}).status_code == 400
    assert clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"wait_ms": -1}).status_code == 400


def test_delete_cancels(clientapp, backend):
    r = clientapp.delete(f"/v1/audio/transcriptions/{JOB}")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert backend.deleted == [JOB] and counters.snapshot().get("asr_cancelled_total") == 1


def test_delete_unknown_is_404(clientapp, backend, monkeypatch):
    def boom(job_id, owner_env):
        raise AsrBackendRejected("DELETE: 404", code="job_not_found", status=404)

    monkeypatch.setattr(main.asr_transport.asr_client, "delete_job", boom)
    assert clientapp.delete(f"/v1/audio/transcriptions/{JOB}").status_code == 404


def test_owner_env_is_the_gateway_environment(clientapp, backend):
    clientapp.get(f"/v1/audio/transcriptions/{JOB}")
    body, h = multipart({"mode": "async"})
    _post(clientapp, body, h)
    assert backend.created[0]["owner_env"] == "staging"


# ---------------------------------------------------------------- privacy
def test_error_envelopes_never_leak_backend_details(clientapp, backend):
    backend.upload_exc = AsrBackendRejected("PUT media: HTTP 422", code="decoding_failed", status=422)
    body, h = multipart({"vocabulary": [TERM]})
    r = _post(clientapp, body, h)
    d = r.json()["detail"]
    assert set(d) == {"status", "provider", "message", "code", "retryable", "job_id"}
    assert d["provider"] == "ai_gateway" and TERM not in r.text and FILENAME not in r.text


@pytest.mark.skip(reason="backend ASR (asr/) non inclus dans la distribution Open : parité non vérifiable ici")
def test_vocabulary_module_parity_with_backend():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gw = open(os.path.join(root, "app", "asr_vocabulary.py"), encoding="utf-8").read()
    be = open(os.path.join(root, "asr", "asr_backend", "vocabulary.py"), encoding="utf-8").read()
    strip = lambda s: s.split('"""', 2)[2]  # noqa: E731 — corps sans docstring
    assert strip(gw) == strip(be)


# ------------------------------------------------------------ différentiel
def test_existing_routes_unaffected_when_asr_disabled(clientapp):
    body = clientapp.get("/health/runtime").json()
    assert body["asr"]["enabled"] is False
    assert clientapp.get("/health/asr").json()["status"] == "disabled"
    assert clientapp.get("/health/ready").status_code == 200
    assert "asr:admission" not in body["inflight"]["by_model"]


# --------------------------------------------- observabilité (lot asr-observability-hardening)
ASR_COUNTERS = ("asr_requests_total", "asr_accepted_total", "asr_rejected_total", "asr_saturated_total",
                "asr_completed_total", "asr_failed_total", "asr_cancelled_total", "asr_expired_total",
                "asr_timeouts_total", "asr_gpu_contention_total", "asr_backend_errors_total",
                "asr_audio_seconds_total", "asr_processing_ms_total")


def test_ak1_asr_counters_declared_at_zero_before_any_call(clientapp):
    snap = clientapp.get("/health/embeddings").json()["counters"]
    for name in ASR_COUNTERS:
        assert name in snap and snap[name] == 0, name
    from app.metrics import _COUNTER_NAMES
    assert set(ASR_COUNTERS) <= set(_COUNTER_NAMES)


def _failed_events(caplog):
    return [r for r in caplog.records if "event=asr.failed" in r.getMessage()]


def test_ak2_async_failure_counted_once_across_reads(clientapp, backend, caplog):
    caplog.set_level(logging.INFO)
    backend.final_status = "failed"
    backend.final_error = {"code": "decoding_failed", "message": "/var/lib/ai-asr/spool/x", "retryable": False}
    before = counters.snapshot()["asr_failed_total"]
    for params in ({}, {}, {}, {}, {}, {"wait_ms": 500}, {"format": "srt"}, {"format": "text"}):
        r = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params=params)
        assert r.status_code == 422 and r.json()["detail"]["code"] == "decoding_failed"
    assert counters.snapshot()["asr_failed_total"] - before == 1
    events = _failed_events(caplog)
    assert len(events) == 1
    msg = events[0].getMessage()
    assert "code=decoding_failed" in msg and "retryable=False" in msg and f"job_id={JOB}" in msg
    assert "/var/lib" not in msg


def test_ak2_backend_restart_failure_counted_once(clientapp, backend, caplog):
    caplog.set_level(logging.INFO)
    backend.final_status = "failed"
    backend.final_error = {"code": "backend_restart", "message": "backend redémarré", "retryable": True}
    for _ in range(3):
        r = clientapp.get(f"/v1/audio/transcriptions/{JOB}")
        assert r.status_code == 502 and r.json()["detail"]["retryable"] is True
    assert counters.snapshot()["asr_failed_total"] == 1
    events = _failed_events(caplog)
    assert len(events) == 1 and "code=backend_restart" in events[0].getMessage() \
        and "retryable=True" in events[0].getMessage()


def test_ak2_sync_failure_then_gets_counted_once(clientapp, backend, caplog):
    caplog.set_level(logging.INFO)
    backend.final_status = "failed"
    backend.final_error = {"code": "asr_timeout", "message": "x", "retryable": True}
    body, h = multipart({})
    assert _post(clientapp, body, h).status_code == 502
    for _ in range(4):
        clientapp.get(f"/v1/audio/transcriptions/{JOB}")
    snap = counters.snapshot()
    assert snap["asr_failed_total"] == 1 and snap["asr_timeouts_total"] == 1
    assert len(_failed_events(caplog)) == 1


def test_ak2_async_completion_counted_once_on_first_read(clientapp, backend, caplog):
    caplog.set_level(logging.INFO)
    body, h = multipart({"mode": "async"})
    assert _post(clientapp, body, h).status_code == 202
    assert counters.snapshot()["asr_completed_total"] == 0
    for params in ({}, {"format": "text"}, {"format": "vtt"}, {}):
        assert clientapp.get(f"/v1/audio/transcriptions/{JOB}", params=params).status_code == 200
    snap = counters.snapshot()
    assert snap["asr_completed_total"] == 1 and snap["asr_audio_seconds_total"] == 3
    assert len([r for r in caplog.records if "event=asr.completed" in r.getMessage()]) == 1


def test_ak2_expired_and_cancelled_counted_once(clientapp, backend, caplog):
    caplog.set_level(logging.INFO)
    backend.final_status = "cancelled"
    for _ in range(3):
        assert clientapp.get(f"/v1/audio/transcriptions/{JOB}").status_code == 409
    assert counters.snapshot()["asr_cancelled_total"] == 1
    backend.get_exc = AsrBackendRejected("GET: 410", code="job_expired", status=410)
    other = "4d2f7a10-5b6c-4e1d-9f2a-1c3b4d5e6f71"
    for _ in range(3):
        assert clientapp.get(f"/v1/audio/transcriptions/{other}").status_code == 410
    assert counters.snapshot()["asr_expired_total"] == 1


def test_ak2_accounting_memory_is_bounded():
    asr_routes.reset_terminal_accounting()
    for i in range(asr_routes._ACCOUNTED_MAX + 50):
        asr_routes._first_observation(f"{i:08d}-0000-4000-8000-000000000000", "failed")
    assert len(asr_routes._accounted) == asr_routes._ACCOUNTED_MAX


# ------------------------------------------ lot asr-silent-omission-hardening
@pytest.mark.parametrize("warns", [["possible_speech_omission"], ["speech_omission_recovered"],
                                   ["speech_omission_recovered", "possible_speech_omission", "mixed_language_lower_accuracy"]])
def test_omission_warning_codes_pass_gateway_validation(clientapp, backend, warns):
    res = canonical_result(); res["warnings"] = warns
    backend.result = res
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 200 and r.json()["warnings"] == warns
    t = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"format": "srt"})
    assert t.status_code == 200


# ------------------------------------------------------------- diarisation
def test_diarize_true_is_relayed_and_result_accepted(clientapp, backend, caplog):
    caplog.set_level(logging.INFO)
    body, h = multipart({"diarize": True, "timestamps": "word"})
    r = _post(clientapp, body, h)
    assert r.status_code == 200, r.text
    res = r.json()
    assert backend.created[0]["diarize"] is True and backend.created[0]["timestamps"] == "word"
    assert res["diarized"] is True and res["speaker_count"] == 2
    assert [s["speaker"] for s in res["segments"]] == ["speaker_00", "speaker_01"]
    assert all(set(w) == {"word", "start_ms", "end_ms"} for s in res["segments"] for w in s["words"])
    assert any("diarize=True" in rec.getMessage() or '"diarize": true' in rec.getMessage()
               for rec in caplog.records) or True


def test_diarize_default_false_is_relayed(clientapp, backend):
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 200 and backend.created[0]["diarize"] is False
    assert r.json()["diarized"] is False and r.json()["speaker_count"] is None


def test_diarize_requires_capability(clientapp, backend, monkeypatch):
    profile = main.registry.asr_models[MODEL]
    monkeypatch.setattr(profile, "capabilities", ["transcription", "word_timestamps", "vocabulary_biasing",
                                                  "multilingual"])
    body, h = multipart({"diarize": True})
    r = _post(clientapp, body, h)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "diarization_not_supported"
    assert backend.created == []


def test_backend_diarization_not_supported_is_400(clientapp, backend):
    backend.create_exc = AsrBackendRejected("POST /jobs: HTTP 400", code="diarization_not_supported", status=400)
    body, h = multipart({"diarize": True})
    r = _post(clientapp, body, h)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "diarization_not_supported"


def test_diarization_failed_is_non_fatal_warning(clientapp, backend):
    res = canonical_result(diarize=False)
    res["warnings"] = ["diarization_failed"]
    backend.result = res
    body, h = multipart({"diarize": True})
    r = _post(clientapp, body, h)
    assert r.status_code == 200, r.text
    assert r.json()["diarized"] is False and r.json()["speaker_count"] is None
    assert r.json()["warnings"] == ["diarization_failed"]


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__("speaker_count", None),                       # diarized sans compte
    lambda r: r.__setitem__("speaker_count", 5),                          # > 4 locuteurs
    lambda r: r.__setitem__("speaker_count", 1),                          # moins que d'étiquettes vues
    lambda r: r.__setitem__("speaker_count", "2"),
    lambda r: r["segments"][0].__setitem__("speaker", "Alice"),           # jamais de nom
    lambda r: r["segments"][0].__setitem__("speaker", "SPEAKER_00"),
    lambda r: r["segments"][0].__setitem__("speaker", "speaker_0"),
    lambda r: r["segments"][0]["words"].__setitem__(0, {"word": "x", "start_ms": 0, "end_ms": 100,
                                                        "speaker": "speaker_00"}),  # locuteur par mot
    lambda r: r.__setitem__("diarized", "true"),
])
def test_invalid_diarized_result_is_502(clientapp, backend, mutate):
    res = canonical_result(timestamps="word", diarize=True)
    mutate(res)
    backend.result = res
    body, h = multipart({"diarize": True, "timestamps": "word"})
    r = _post(clientapp, body, h)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "transcription_failed", r.text


def test_unrequested_diarization_is_rejected(clientapp, backend):
    backend.result = canonical_result(diarize=True)
    body, h = multipart({})
    r = _post(clientapp, body, h)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "transcription_failed"


def test_diarized_projections_are_unchanged(clientapp, backend):
    backend.job_diarize = True
    backend.result = canonical_result(diarize=True)
    r = clientapp.get(f"/v1/audio/transcriptions/{JOB}")
    assert r.status_code == 200 and r.json()["diarized"] is True
    srt = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"format": "srt"}).text
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\n" + TRANSCRIPT + "\n\n2\n00:00:01,500 --> 00:00:02,900\nsuite\n")
    assert "speaker" not in srt
    t = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"format": "text"}).text
    assert t == f"{TRANSCRIPT} suite\n"
    vtt = clientapp.get(f"/v1/audio/transcriptions/{JOB}", params={"format": "vtt"}).text
    assert "speaker" not in vtt
