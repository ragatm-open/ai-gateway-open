"""Lot asr-rail-foundation — commit 1 : registre `asr_models`/`asr_defaults`,
client backend ASR, /health/asr et section `asr` de /health/runtime.

Aucun réseau : le backend est simulé par monkeypatch sur `main.asr_transport`.
"""

import pytest

from app import main
from app.asr_client import (
    AsrBackendClient,
    AsrBackendError,
    AsrBackendRejected,
    AsrBackendUnavailable,
    backend_url_is_loopback,
)
from app.registry import RegistryError, load_registry

REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
SHA = "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"
ARTIFACT = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"

EMB = """version: 1
models:
  emb:
    provider: ollama
    ollama_name: nomic-embed-text
    digest: "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
    kind: embedding
    dimension: 768
    languages: [en]
    cross_lingual: false
    normalized: true
    query_prefix: "q: "
    document_prefix: "d: "
    max_batch_size: 64
    max_input_chars: 6000
    on_overflow: reject
    production_allowed: true
    staging_allowed: true
"""

ASR = f"""asr_defaults:
  model: asr-t
asr_models:
  asr-t:
    provider: faster-whisper
    artifact: {ARTIFACT}
    revision: "{REVISION}"
    artifact_sha256: "{SHA}"
    compute_type: float16
    vad:
      provider: silero
      source: faster-whisper-embedded
      version: "1.2.1"
    capabilities: [transcription, word_timestamps, vocabulary_biasing, multilingual]
    languages: [fr, en]
    max_duration_seconds: 6300
    sync_max_duration_seconds: 1200
    max_upload_bytes: 268435456
    resource_class: speech-gpu
    staging_allowed: true
    production_allowed: false
"""


def _reg(tmp_path, body):
    p = tmp_path / "r.yaml"
    p.write_text(body, encoding="utf-8")
    return str(p)


# ----------------------------------------------------------------- registre
def test_registry_parses_asr_section(tmp_path):
    reg = load_registry(_reg(tmp_path, EMB + ASR))
    p = reg.resolve_asr(None)
    assert p is not None and p.logical_name == "asr-t"
    assert reg.resolve_asr("asr-t") is p
    assert reg.resolve_asr("absent") is None
    assert p.revision == REVISION and p.artifact_sha256 == SHA
    assert p.vad.version == "1.2.1"
    assert "diarization" not in p.capabilities
    assert p.allowed_in("staging") and not p.allowed_in("production")
    d = p.to_public_dict()
    assert d["model_logical"] == "asr-t" and d["max_upload_bytes"] == 268435456


def test_registry_without_asr_section_is_unchanged(tmp_path):
    reg = load_registry(_reg(tmp_path, EMB))
    assert reg.asr_models == {} and reg.asr_default_model is None
    assert reg.resolve_asr(None) is None


def test_shipped_registry_pins_turbo_exactly():
    reg = load_registry(main.settings.models_registry_path)
    p = reg.resolve_asr(None)
    assert p.logical_name == "asr-whisper-turbo"
    assert p.artifact == ARTIFACT and p.revision == REVISION and p.artifact_sha256 == SHA
    assert p.capabilities == ["transcription", "word_timestamps", "vocabulary_biasing", "multilingual", "diarization"]
    assert p.diarization is not None and p.diarization.provider == "nemo-sortformer"
    assert p.diarization.artifact == "nvidia/diar_streaming_sortformer_4spk-v2"
    assert p.diarization.revision == "84edd514b8ef68004c10086918cd62f2148cbd59"
    assert p.diarization.artifact_sha256 == "b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329"
    assert p.languages == ["fr", "en"]
    assert (p.max_duration_seconds, p.sync_max_duration_seconds, p.max_upload_bytes) == (6300, 1200, 268435456)
    assert p.staging_allowed and p.production_allowed  # lot production-full-capability-enablement


@pytest.mark.parametrize(
    "before,after,fragment",
    [
        (f'revision: "{REVISION}"', 'revision: "main"', "revision mal formée"),
        (f'artifact_sha256: "{SHA}"', 'artifact_sha256: "abc"', "artifact_sha256"),
        ("provider: faster-whisper", "provider: whisperx", "provider non supporté"),
        ("compute_type: float16", "compute_type: int8", "compute_type"),
        ("languages: [fr, en]", "languages: [fr, de]", "languages"),
        ("languages: [fr, en]", "languages: [fr, fr]", "languages"),
        ("capabilities: [transcription, word_timestamps, vocabulary_biasing, multilingual]",
         "capabilities: [transcription, summary]", "inconnue"),
        ("capabilities: [transcription, word_timestamps, vocabulary_biasing, multilingual]",
         "capabilities: [word_timestamps]", "transcription"),
        ("sync_max_duration_seconds: 1200", "sync_max_duration_seconds: 7000", "<= max_duration_seconds"),
        ("max_upload_bytes: 268435456", "max_upload_bytes: 0", "max_upload_bytes"),
        ("resource_class: speech-gpu", "resource_class: cpu", "speech-gpu"),
        ("      provider: silero", "      provider: webrtc", "vad.provider"),
        ("  model: asr-t", "  model: other", "asr_defaults.model inconnu"),
        ("    staging_allowed: true", "    staging_allowed: true\n    extra: 1", "champ inconnu"),
    ],
)
def test_registry_rejects_bad_asr(tmp_path, before, after, fragment):
    body = EMB + ASR
    assert before in body
    with pytest.raises(RegistryError) as excinfo:
        load_registry(_reg(tmp_path, body.replace(before, after)))
    assert fragment in str(excinfo.value)


def test_asr_models_without_defaults_is_rejected(tmp_path):
    body = (EMB + ASR).replace("asr_defaults:\n  model: asr-t\n", "")
    with pytest.raises(RegistryError, match="asr_defaults.model est obligatoire"):
        load_registry(_reg(tmp_path, body))


def test_asr_name_colliding_with_embedding_is_rejected(tmp_path):
    body = (EMB + ASR).replace("asr-t", "emb")
    with pytest.raises(RegistryError, match="nom ambigu"):
        load_registry(_reg(tmp_path, body))


def test_asr_name_colliding_with_llm_is_rejected(tmp_path):
    body = open(main.settings.models_registry_path, encoding="utf-8").read()
    body = body.replace("  asr-whisper-turbo:", "  qwen-judge:", 1).replace("model: asr-whisper-turbo", "model: qwen-judge")
    with pytest.raises(RegistryError, match="nom ambigu"):
        load_registry(_reg(tmp_path, body))


# ------------------------------------------------------------------ client
def test_loopback_detection():
    assert backend_url_is_loopback("http://127.0.0.1:8400")
    assert backend_url_is_loopback("http://localhost:8400")
    assert not backend_url_is_loopback("http://203.0.113.5:8400")
    assert not backend_url_is_loopback("http://[::2]:8400")


def test_client_rejects_malformed_job_id():
    c = AsrBackendClient("http://127.0.0.1:8400", 1, 1)
    with pytest.raises(AsrBackendRejected) as excinfo:
        c.get_job("../etc", "staging")
    assert excinfo.value.code == "job_not_found" and excinfo.value.status == 404


def test_client_network_failure_is_typed_and_never_carries_url(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("http://127.0.0.1:8400 refused")

    monkeypatch.setattr(requests, "get", boom)
    c = AsrBackendClient("http://127.0.0.1:8400", 1, 1)
    with pytest.raises(AsrBackendUnavailable) as excinfo:
        c.info()
    assert "127.0.0.1" not in excinfo.value.message and "8400" not in excinfo.value.message
    assert isinstance(excinfo.value, AsrBackendError)


def test_client_relays_backend_code_not_message(monkeypatch):
    import requests

    class R:
        status_code = 413

        @staticmethod
        def json():
            return {"detail": {"code": "audio_too_long", "message": "/var/lib/ai-asr/spool/x"}}

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    c = AsrBackendClient("http://127.0.0.1:8400", 1, 1)
    with pytest.raises(AsrBackendRejected) as excinfo:
        c.create_job({})
    assert excinfo.value.code == "audio_too_long" and excinfo.value.status == 413
    assert "spool" not in excinfo.value.message


# ------------------------------------------------------------- ollama /api/ps
def test_list_running_models_vram_reads_size_vram(monkeypatch):
    import requests
    from app.ollama_client import OllamaClient

    class R:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"models": [{"name": "a", "size_vram": 1000}, {"name": "b"}, {"name": "c", "size_vram": -1},
                               {"name": "d", "size_vram": True}, "junk"]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    out = OllamaClient("http://127.0.0.1:11434", 5).list_running_models_vram()
    assert out == [{"name": "a", "size_vram": 1000}, {"name": "b", "size_vram": 0},
                   {"name": "c", "size_vram": 0}, {"name": "d", "size_vram": 0}]


# ------------------------------------------------------------- health / runtime
INFO_OK = {
    "provider": "faster-whisper", "artifact": ARTIFACT, "revision": REVISION, "artifact_sha256": SHA,
    "vad": {"provider": "silero", "source": "faster-whisper-embedded", "version": "1.2.1"},
    "capabilities": ["transcription", "word_timestamps", "vocabulary_biasing", "multilingual", "diarization"],
    "diarization": {"provider": "nemo-sortformer", "artifact": "nvidia/diar_streaming_sortformer_4spk-v2",
                    "revision": "84edd514b8ef68004c10086918cd62f2148cbd59",
                    "artifact_sha256": "b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329",
                    "expected_sha256": "b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329",
                    "hash_match": True, "device": "cpu", "max_speakers": 4, "model_ready": True},
    "backend_version": "0.1.0", "faster_whisper_version": "1.2.1", "ctranslate2_version": "4.8.2",
}
HEALTH_OK = {"_http_status": 200, "status": "ok", "worker_state": "idle", "queued": 0, "inflight": 0,
             "model_ready": True, "spool": {"bytes": 0, "capacity_bytes": 2147483648, "jobs": 0}}


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.setattr(main.settings, "asr_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "staging")


@pytest.fixture
def backend(monkeypatch):
    state = {"info": dict(INFO_OK), "health": dict(HEALTH_OK)}
    monkeypatch.setattr(main.asr_transport.asr_client, "info", lambda timeout=None: dict(state["info"]))
    monkeypatch.setattr(main.asr_transport.asr_client, "health", lambda timeout=None: dict(state["health"]))
    return state


def test_health_asr_disabled_by_default(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "staging")
    r = clientapp.get("/health/asr")
    assert r.status_code == 200
    assert r.json()["status"] == "disabled" and r.json()["enabled"] is False
    assert "AI_GATEWAY_ASR_ENABLED" in r.json()["message"]


def test_health_asr_production_flag_absent_stays_disabled(clientapp, monkeypatch, backend):
    """Fail-safe : sans AI_GATEWAY_ASR_ENABLED=true, la production reste OFF."""
    monkeypatch.setattr(main.settings, "asr_enabled", False)
    monkeypatch.setattr(main.settings, "environment", "production")
    r = clientapp.get("/health/asr")
    assert r.status_code == 200 and r.json()["status"] == "disabled"
    assert "AI_GATEWAY_ASR_ENABLED" in r.json()["message"]


def test_health_asr_production_enabled_with_registry_and_backend(clientapp, monkeypatch, backend):
    """Production + flag + registre production_allowed + backend conforme -> ok
    (lot production-full-capability-enablement) ; identité diarisation épinglée vérifiée."""
    monkeypatch.setattr(main.settings, "asr_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "production")
    r = clientapp.get("/health/asr")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["enabled"] is True and body["allowed_here"] is True
    assert body["environment"] == "production" and body["checks"]["diarization_match"] is True
    assert body["diarization"]["hash_match"] is True and body["diarization"]["device"] == "cpu"
    assert body["diarization"]["expected_sha256"] == body["diarization"]["observed_sha256"]


def test_health_asr_production_registry_disallowed(clientapp, monkeypatch, backend):
    monkeypatch.setattr(main.settings, "asr_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "production")
    monkeypatch.setattr(main.registry.asr_models["asr-whisper-turbo"], "production_allowed", False)
    r = clientapp.get("/health/asr")
    assert r.status_code == 200 and r.json()["enabled"] is True and r.json()["allowed_here"] is False


def test_health_asr_production_backend_outside_loopback_disabled(clientapp, monkeypatch, backend):
    monkeypatch.setattr(main.settings, "asr_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "production")
    monkeypatch.setattr(main.settings, "asr_backend_url", "http://203.0.113.5:8400")
    r = clientapp.get("/health/asr")
    assert r.status_code == 200 and r.json()["status"] == "disabled" and "loopback" in r.json()["message"]


@pytest.mark.parametrize("field,value", [
    ("artifact_sha256", "0" * 64),
    ("revision", "f" * 40),
    ("artifact", "nvidia/other-diarizer"),
    ("provider", "pyannote"),
    ("model_ready", False),
])
def test_health_asr_diarization_identity_mismatch_is_503(clientapp, staging, backend, field, value):
    """Hash/identité Sortformer divergents -> rail fail-closed, jamais de substitution silencieuse."""
    backend["info"]["diarization"] = dict(backend["info"]["diarization"], **{field: value})
    r = clientapp.get("/health/asr")
    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["diarization_match"] is False


def test_registry_diarization_identity_required_with_capability(tmp_path):
    with_cap = ASR.replace("capabilities: [transcription, word_timestamps, vocabulary_biasing, multilingual]",
                           "capabilities: [transcription, word_timestamps, vocabulary_biasing, multilingual, diarization]")
    with pytest.raises(RegistryError, match="diarization"):
        load_registry(_reg(tmp_path, EMB + with_cap))
    without_cap = ASR + "    diarization:\n      provider: nemo-sortformer\n      artifact: nvidia/x\n      revision: \"" + "a" * 40 + "\"\n      artifact_sha256: \"" + "b" * 64 + "\"\n"
    with pytest.raises(RegistryError, match="diarization"):
        load_registry(_reg(tmp_path, EMB + without_cap))


def test_health_asr_non_loopback_disabled(clientapp, staging, backend, monkeypatch):
    monkeypatch.setattr(main.settings, "asr_backend_url", "http://203.0.113.5:8400")
    r = clientapp.get("/health/asr")
    assert r.status_code == 200 and "loopback" in r.json()["message"]


def test_health_asr_ok(clientapp, staging, backend):
    r = clientapp.get("/health/asr")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["enabled"] is True
    assert body["backend_url"] == "redacted"
    assert body["model"] == "asr-whisper-turbo"
    assert body["artifact_match"] and body["revision_match"] and body["hash_match"]
    assert body["worker_state"] == "idle" and body["queued"] == 0 and body["inflight"] == 0
    assert body["spool"]["capacity_bytes"] == 2147483648
    assert body["capabilities"] == ["transcription", "word_timestamps", "vocabulary_biasing", "multilingual",
                                    "diarization"]
    assert body["version"]["faster_whisper"] == "1.2.1"
    assert body["allowed_here"] is True
    assert "8400" not in r.text and "/var/lib" not in r.text


@pytest.mark.parametrize("field,value,check", [
    ("revision", "f" * 40, "revision_match"),
    ("artifact_sha256", "0" * 64, "hash_match"),
    ("artifact", "openai/whisper-large-v3", "artifact_match"),
    ("provider", "whisperx", "provider_match"),
    ("capabilities", ["transcription"], "capabilities_match"),
])
def test_health_asr_identity_mismatch_is_503(clientapp, staging, backend, field, value, check):
    backend["info"][field] = value
    r = clientapp.get("/health/asr")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["checks"][check] is False and "divergente" in detail["message"]


def test_health_asr_vad_mismatch_is_503(clientapp, staging, backend):
    backend["info"]["vad"] = {"provider": "silero", "source": "pip", "version": "5.1"}
    r = clientapp.get("/health/asr")
    assert r.status_code == 503 and r.json()["detail"]["checks"]["vad_match"] is False


def test_health_asr_backend_unreachable_is_503(clientapp, staging, monkeypatch):
    def boom(timeout=None):
        raise AsrBackendUnavailable("échec /info backend ASR: ConnectionError")

    monkeypatch.setattr(main.asr_transport.asr_client, "info", boom)
    r = clientapp.get("/health/asr")
    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["backend_reachable"] is False
    assert "ConnectionError" not in r.text


def test_health_asr_worker_not_ready_is_503(clientapp, staging, backend):
    backend["health"]["model_ready"] = False
    r = clientapp.get("/health/asr")
    assert r.status_code == 503 and "dégradé" in r.json()["detail"]["message"]


def test_health_asr_never_touches_ready(clientapp, staging, monkeypatch):
    def boom(timeout=None):
        raise AsrBackendUnavailable("x")

    monkeypatch.setattr(main.asr_transport.asr_client, "info", boom)
    assert clientapp.get("/health/asr").status_code == 503
    assert clientapp.get("/health/ready").status_code == 200


def test_runtime_asr_section_additive(clientapp, monkeypatch):
    body = clientapp.get("/health/runtime").json()
    asr = body["asr"]
    assert asr["enabled"] is False
    assert asr["default_model"] == "asr-whisper-turbo"
    assert [m["model_logical"] for m in asr["models"]] == ["asr-whisper-turbo"]
    assert asr["backend"]["implementation"] == "faster-whisper"
    assert asr["backend"]["image_digest"] == "not_reported"
    assert asr["limits"]["gpu_budget_mib"] == 22528
    # Additif : les sections existantes restent présentes.
    for key in ("rerank", "compute", "inflight", "queued", "llm_models"):
        assert key in body
    assert "8400" not in str(body["asr"])


def test_runtime_asr_enabled_reflects_flag(clientapp, staging):
    assert clientapp.get("/health/runtime").json()["asr"]["enabled"] is True


def test_settings_asr_flag_has_normal_meaning_in_production(monkeypatch):
    """Lot production-full-capability-enablement : plus de forçage OFF en production ;
    le défaut reste OFF (fail-safe) dans tout environnement."""
    from app import settings as smod
    monkeypatch.setenv("AI_GATEWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("AI_GATEWAY_ASR_ENABLED", "true")
    assert smod.Settings().asr_enabled is True
    monkeypatch.delenv("AI_GATEWAY_ASR_ENABLED", raising=False)
    assert smod.Settings().asr_enabled is False
    monkeypatch.setenv("AI_GATEWAY_ENVIRONMENT", "staging")
    assert smod.Settings().asr_enabled is False
