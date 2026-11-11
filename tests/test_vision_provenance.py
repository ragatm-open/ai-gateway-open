"""Lot CV-1 — provenance technique du rail Vision (contrat V2 opt-in).

WHAT : `image.source_sha256` (octets soumis) et `image.normalized_sha256`
(octets exacts transmis au modèle). HOW : `pipeline.fingerprint`
`vision-v1:<64 hex>` d'un payload canonique SÉMANTIQUE (task, digest, prompts,
schéma model-facing, normaliseur, options, retry) ; les réglages
OPÉRATIONNELS (keep_alive, timeout, environnement, locale V1…) n'y entrent
pas. V1 reste octet pour octet inchangé.

Backend toujours mocké : aucun modèle, aucun GPU, aucun réseau. Les vecteurs
du normaliseur (tests/data/vision_normalizer) sont des images synthétiques.
"""

import base64
import hashlib
import json
import logging
import os
from dataclasses import replace

import PIL
import pytest

from app import main
from app.vision_backend import (
    CONTEXT_WRAPPER_VERSION,
    QUALIFIED_FIRST_ATTEMPT,
    QUALIFIED_RETRY_ATTEMPT,
    RETRY_MAX_ATTEMPTS,
    RETRY_TRIGGER,
    OllamaVisionBackend,
    VisionBackendResult,
    VisionInferencePlan,
)
from app.vision_image import NORMALIZED_FORMAT, NORMALIZER_ID, admit_image
from app.vision_prompts import DESCRIBE_PROMPT, SYSTEM_PROMPT, sha256_of
from app.vision_provenance import (
    FINGERPRINT_SCHEMA,
    MODEL_SCHEMA_HASH_ALGORITHM,
    model_schema_sha256,
    pipeline_identity,
    pipeline_provenance,
    vision_pipeline_fingerprint,
)
from app.vision_schemas import (
    VISION_TASKS,
    VisionAnalyzeResponse,
    VisionAnalyzeResponseV2,
    output_schema_for_task,
)
from tests.test_vision_media import (
    _b64,
    _png,
    _post,
    backend,          # noqa: F401 — fixtures réutilisées
    ok_photo_result,
    vision_ready,     # noqa: F401
)

VECTORS_DIR = os.path.join(os.path.dirname(__file__), "data", "vision_normalizer")
DIGEST = "5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b"
V1_FIELDS = {"status", "provider", "upstream_provider", "schema_version", "task", "visual_kind",
             "request_id", "summary", "warnings", "requires_visual_check", "payload", "model",
             "pipeline", "image", "usage", "latency_ms", "truncated"}


def _vectors():
    with open(os.path.join(VECTORS_DIR, "vectors.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _vector_bytes(v):
    with open(os.path.join(VECTORS_DIR, v["file"]), "rb") as fh:
        return fh.read()


# ------------------------------------------------------------ V1 immuable
def test_v1_response_field_set_unchanged(clientapp, backend):
    backend.results = [ok_photo_result()]
    body = _post(clientapp).json()
    assert set(body) == V1_FIELDS
    assert body["schema_version"] == "1"
    assert set(body["image"]) == {"width", "height", "pixels", "bytes", "format"}
    assert set(body["pipeline"]) == {"prompt_profile", "prompt_profile_version", "pipeline_id"}
    assert set(body["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    for forbidden in ("source_sha256", "normalized_sha256", "fingerprint", "attempts"):
        assert forbidden not in json.dumps(body)


def test_v1_model_facing_schema_unchanged_by_v2_models():
    # Les classes de réponse V2 ne doivent JAMAIS fuir dans le schéma imposé
    # au modèle : même SHA que celui figé par MODEL-1 (vérifié aussi par
    # tests/test_vision_model1_evidence.py).
    assert model_schema_sha256("describe") == \
        "bcea8521454921d9a3dbd273420c19c3eee4b5c6833b63a38cee7ef4eb1954cd"
    assert "source_sha256" not in json.dumps(output_schema_for_task("auto"))
    assert "fingerprint" not in json.dumps(output_schema_for_task("auto"))


# ------------------------------------------------------------- V2 opt-in
def test_v2_response_shape_and_values(clientapp, backend):
    backend.results = [ok_photo_result()]
    raw = _png()
    r = _post(clientapp, image_bytes=raw, schema_version="2", request_id="prov-1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == V1_FIELDS                      # mêmes clés de premier niveau
    assert body["schema_version"] == "2"
    VisionAnalyzeResponseV2.model_validate(body)       # schéma FERMÉ V2
    with pytest.raises(Exception):
        VisionAnalyzeResponse.model_validate(body)     # ce n'est PAS une réponse V1
    # payload perceptif identique à V1
    assert body["payload"] == json.loads(ok_photo_result().raw_json)["payload"]
    img = body["image"]
    assert img["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert img["normalized_format"] == "png" == NORMALIZED_FORMAT
    assert img["normalizer_id"] == NORMALIZER_ID == "vision-image-normalizer/1"
    assert img["normalized_sha256"] == hashlib.sha256(backend.plans[0].image_bytes).hexdigest()
    pipe = body["pipeline"]
    assert pipe["pipeline_id"] == f"vision-large@{DIGEST[:8]}/vision_describe_v1@v1"
    assert pipe["fingerprint"].startswith("vision-v1:") and len(pipe["fingerprint"]) == 10 + 64
    assert pipe["system_prompt_sha256"] == sha256_of(SYSTEM_PROMPT)
    # La fixture `vision_ready` substitue des prompts de test à
    # QUALIFIED_PROMPTS : le SHA renvoyé est celui du prompt RÉELLEMENT envoyé.
    # (Le prompt `describe` servi en déploiement est vérifié contre MODEL-1 par
    # tests/test_vision_model1_evidence.py.)
    served_prompt = main.QUALIFIED_PROMPTS["describe"][1]
    assert pipe["task_prompt_sha256"] == sha256_of(served_prompt) == sha256_of(backend.plans[0].task_prompt)
    assert pipe["model_schema_sha256"] == model_schema_sha256("describe")
    assert pipe["model_schema_hash_algorithm"] == "json-sortkeys/1"
    assert pipe["inference"] == {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 2048}
    assert pipe["retry"] == {"max_attempts": 2, "trigger": "structural_invalid_or_truncated",
                             "temperature": 0.35, "seed": 43}
    assert body["model"]["digest"] == DIGEST and "model_digest" not in pipe
    assert body["usage"]["attempts"] == 1


def test_v2_fingerprint_matches_pure_function_and_health(clientapp, backend):
    backend.results = [ok_photo_result()]
    body = _post(clientapp, schema_version="2").json()
    served_prompt = main.QUALIFIED_PROMPTS["describe"][1]   # prompt de test (fixture)
    expected = vision_pipeline_fingerprint(pipeline_identity("describe", DIGEST, served_prompt))
    assert body["pipeline"]["fingerprint"] == expected
    # Avec le prompt qualifié réel, le fingerprint est différent — et c'est
    # celui que le déploiement expose (prompt vérifié par MODEL-1).
    assert expected != vision_pipeline_fingerprint(pipeline_identity("describe", DIGEST, DESCRIBE_PROMPT))
    h = clientapp.get("/health/vision").json()
    entry = h["models"][0]
    assert entry["pipelines"]["describe"]["fingerprint"] == expected
    # la fixture qualifie les six tasks : une entrée par task servie
    assert set(entry["pipelines"]) == set(main.registry.resolve_vision("vision-large").vision_tasks)
    assert set(entry["pipelines"]["describe"]) == {"fingerprint", "system_prompt_sha256",
                                                   "task_prompt_sha256", "model_schema_sha256"}
    assert h["normalizer_id"] == NORMALIZER_ID and h["pillow_version"] == PIL.__version__
    assert h["model_schema_hash_algorithm"] == MODEL_SCHEMA_HASH_ALGORITHM
    assert "source_sha256" not in json.dumps(h) and "normalized_sha256" not in json.dumps(h)


def test_health_pipelines_on_deployment_profile(clientapp):
    """Sans fixture : registre et prompts tels que DÉPLOYÉS. Une seule task
    servie (describe), fingerprint du prompt qualifié réel, et les mêmes
    valeurs que les tests MODEL-1."""
    h = clientapp.get("/health/vision").json()
    entry = next(m for m in h["models"] if m["model_logical"] == "vision-large")
    assert entry["schema_versions"] == ["1", "2"] and entry["tasks"] == ["describe"]
    assert set(entry["pipelines"]) == {"describe"}
    prov = entry["pipelines"]["describe"]
    assert prov["fingerprint"] == vision_pipeline_fingerprint(
        pipeline_identity("describe", DIGEST, DESCRIBE_PROMPT))
    assert prov["system_prompt_sha256"] == sha256_of(SYSTEM_PROMPT)
    assert prov["task_prompt_sha256"] == sha256_of(DESCRIBE_PROMPT)
    assert prov["model_schema_sha256"] == "bcea8521454921d9a3dbd273420c19c3eee4b5c6833b63a38cee7ef4eb1954cd"


def test_unsupported_schema_version_values_rejected(clientapp, backend):
    assert _post(clientapp, schema_version="3").status_code == 400
    assert _post(clientapp, schema_version=2).status_code == 400
    assert backend.plans == []


def test_v2_refused_if_profile_does_not_declare_it(clientapp, backend, monkeypatch):
    profile = main.registry.resolve_vision("vision-large")
    monkeypatch.setattr(profile, "vision_schema_versions", ["1"])
    r = _post(clientapp, schema_version="2")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "vision_schema_version_unsupported"
    assert backend.plans == []
    backend.results = [ok_photo_result()]
    assert _post(clientapp, schema_version="1").status_code == 200


# ---------------------------------------------------------- source hash
def test_source_hash_definition(clientapp, backend):
    raw = _png()
    backend.results = [ok_photo_result(), ok_photo_result()]
    a = _post(clientapp, image_bytes=raw, schema_version="2").json()["image"]
    b = _post(clientapp, image_bytes=raw, schema_version="2").json()["image"]
    assert a["source_sha256"] == b["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert a["source_sha256"] != hashlib.sha256(_b64(raw).encode()).hexdigest()   # jamais le Base64


def test_metadata_only_difference_changes_source_not_normalized():
    import io
    from PIL import Image, PngImagePlugin
    raw = _png()
    img = Image.open(io.BytesIO(raw))
    info = PngImagePlugin.PngInfo(); info.add_text("Comment", "meta")
    out = io.BytesIO(); img.save(out, "PNG", pnginfo=info)
    a, b = admit_image(raw, "image/png"), admit_image(out.getvalue(), "image/png")
    assert a.source_sha256 != b.source_sha256
    assert a.normalized_sha256 == b.normalized_sha256


def test_reencoded_jpeg_has_different_source_identity():
    import io
    from PIL import Image
    img = Image.new("RGB", (64, 48), (10, 20, 30))
    q85, q60 = io.BytesIO(), io.BytesIO()
    img.save(q85, "JPEG", quality=85); img.save(q60, "JPEG", quality=60)
    a, b = admit_image(q85.getvalue(), "image/jpeg"), admit_image(q60.getvalue(), "image/jpeg")
    assert a.source_sha256 != b.source_sha256


# ------------------------------------------------------ normalized hash
def test_normalized_hash_equals_actual_ollama_images_payload(clientapp, backend):
    backend.results = [ok_photo_result()]
    body = _post(clientapp, schema_version="2").json()
    plan: VisionInferencePlan = backend.plans[0]
    payload = OllamaVisionBackend("http://127.0.0.1:11434").build_payload(plan)
    sent = base64.b64decode(payload["images"][0])
    assert hashlib.sha256(sent).hexdigest() == body["image"]["normalized_sha256"]
    assert sent[:8] == b"\x89PNG\r\n\x1a\n"


# ------------------------------------------------------ normalizer vectors
@pytest.mark.parametrize("name", sorted(_vectors()["vectors"]))
def test_normalizer_vectors_byte_for_byte(name):
    """Gel du normaliseur : une montée de Pillow ou un changement de code qui
    modifie UN octet du PNG casse ce test et impose une décision explicite
    (conserver les octets, ou bump vers vision-image-normalizer/2)."""
    v = _vectors()["vectors"][name]
    raw = _vector_bytes(v)
    a = admit_image(raw, v["mime_type"])
    assert a.source_sha256 == v["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert a.normalized_sha256 == v["normalized_sha256"]
    assert (a.width, a.height, a.format) == (v["width"], v["height"], v["input_format"])
    assert len(a.normalized_png) == v["normalized_bytes"]
    assert hashlib.sha256(a.normalized_png).hexdigest() == a.normalized_sha256


def test_normalizer_vectors_cover_required_cases():
    vec = _vectors()
    assert vec["normalizer_id"] == NORMALIZER_ID
    assert {"png_rgb", "png_rgba", "png_gray", "jpeg_rgb", "jpeg_exif6", "webp_rgb"} <= set(vec["vectors"])
    v6, v1 = vec["vectors"]["jpeg_exif6"], vec["vectors"]["jpeg_exif1"]
    assert (v6["width"], v6["height"]) == (v1["height"], v1["width"])   # EXIF 6 transposé
    assert v6["source_sha256"] != v1["source_sha256"]
    assert v1["normalized_sha256"] == vec["vectors"]["jpeg_rgb"]["normalized_sha256"]  # tag 1 = sans tag
    assert all(f["normalized_bytes"] > 0 for f in vec["vectors"].values())


def test_normalizer_deterministic_repeated():
    v = _vectors()["vectors"]["jpeg_exif6"]
    raw = _vector_bytes(v)
    assert len({admit_image(raw, v["mime_type"]).normalized_sha256 for _ in range(5)}) == 1


# ----------------------------------------------------------- fingerprint
def _fp(**kw):
    return vision_pipeline_fingerprint(pipeline_identity("describe", DIGEST, DESCRIBE_PROMPT, **kw))


def test_fingerprint_canonical_payload_exact():
    identity = pipeline_identity("describe", DIGEST, DESCRIBE_PROMPT)
    assert identity == {
        "schema": 1, "task": "describe", "model_digest": DIGEST,
        "system_prompt_sha256": sha256_of(SYSTEM_PROMPT),
        "task_prompt_sha256": sha256_of(DESCRIBE_PROMPT),
        "context_wrapper": "1",
        "model_schema_sha256": model_schema_sha256("describe"),
        "model_schema_hash_algorithm": "json-sortkeys/1",
        "normalizer_id": "vision-image-normalizer/1",
        "inference": {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 2048},
        "retry": {"max_attempts": 2, "trigger": "structural_invalid_or_truncated",
                  "temperature": 0.35, "seed": 43},
    }
    assert FINGERPRINT_SCHEMA == 1 and CONTEXT_WRAPPER_VERSION == "1"
    assert RETRY_MAX_ATTEMPTS == 2 and RETRY_TRIGGER == "structural_invalid_or_truncated"
    # SHA-256 complet du JSON canonique (clés triées, compact) — jamais tronqué.
    from app.registry import canonical_json
    assert _fp() == "vision-v1:" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    assert _fp() == _fp()


def test_fingerprint_ignores_operational_settings():
    base = _fp()
    # keep_alive 0 (production) vs 10 (staging) : option du plan, pas du pipeline
    assert _fp(first_attempt=replace(QUALIFIED_FIRST_ATTEMPT, keep_alive=10)) == base
    assert _fp(retry_attempt=replace(QUALIFIED_RETRY_ATTEMPT, keep_alive=30)) == base
    # locale fr/en, timeout, environnement, request_id : absents du payload
    assert "locale" not in json.dumps(pipeline_identity("describe", DIGEST, DESCRIBE_PROMPT))
    for key in ("keep_alive", "timeout", "environment", "request_id", "schema_version", "pillow"):
        assert key not in json.dumps(pipeline_identity("describe", DIGEST, DESCRIBE_PROMPT))


@pytest.mark.parametrize("mutation", [
    dict(system_prompt=SYSTEM_PROMPT + " "),
    dict(normalizer_id="vision-image-normalizer/2"),
    dict(context_wrapper="2"),
    dict(retry_max_attempts=3),
    dict(retry_trigger="any_failure"),
    dict(first_attempt=replace(QUALIFIED_FIRST_ATTEMPT, temperature=0.1)),
    dict(first_attempt=replace(QUALIFIED_FIRST_ATTEMPT, seed=7)),
    dict(first_attempt=replace(QUALIFIED_FIRST_ATTEMPT, num_ctx=4096)),
    dict(first_attempt=replace(QUALIFIED_FIRST_ATTEMPT, num_predict=1024)),
    dict(retry_attempt=replace(QUALIFIED_RETRY_ATTEMPT, temperature=0.5)),
    dict(retry_attempt=replace(QUALIFIED_RETRY_ATTEMPT, seed=99)),
    dict(schema_sha256="0" * 64),
])
def test_fingerprint_changes_on_semantic_mutation(mutation):
    assert _fp(**mutation) != _fp()


def test_fingerprint_changes_on_digest_prompt_and_task():
    base = _fp()
    assert vision_pipeline_fingerprint(pipeline_identity("describe", "ab" * 32, DESCRIBE_PROMPT)) != base
    assert vision_pipeline_fingerprint(pipeline_identity("describe", DIGEST, DESCRIBE_PROMPT + ".")) != base
    assert vision_pipeline_fingerprint(pipeline_identity("table", DIGEST, DESCRIBE_PROMPT)) != base
    # toutes les tasks ont un fingerprint distinct (schéma model-facing distinct)
    fps = {vision_pipeline_fingerprint(pipeline_identity(t, DIGEST, DESCRIBE_PROMPT)) for t in VISION_TASKS}
    assert len(fps) == len(VISION_TASKS)


def test_fingerprint_identical_production_vs_staging(clientapp, backend, monkeypatch):
    backend.results = [ok_photo_result(), ok_photo_result()]
    monkeypatch.setattr(main.settings, "vision_keep_alive_seconds", 0)
    monkeypatch.setattr(main.settings, "vision_timeout_seconds", 60)
    fp_prod = _post(clientapp, schema_version="2").json()["pipeline"]["fingerprint"]
    monkeypatch.setattr(main.settings, "vision_keep_alive_seconds", 10)
    monkeypatch.setattr(main.settings, "vision_timeout_seconds", 30)
    fp_staging = _post(clientapp, schema_version="2").json()["pipeline"]["fingerprint"]
    assert fp_prod == fp_staging
    assert backend.plans[0].options.keep_alive == 0 and backend.plans[1].options.keep_alive == 10


def test_fingerprint_identical_fr_vs_en(clientapp, backend):
    backend.results = [ok_photo_result(), ok_photo_result()]
    fr = _post(clientapp, schema_version="2", locale="fr").json()["pipeline"]["fingerprint"]
    en = _post(clientapp, schema_version="2", locale="en").json()["pipeline"]["fingerprint"]
    assert fr == en


def test_pipeline_provenance_public_fields():
    prov = pipeline_provenance("describe", DIGEST, DESCRIBE_PROMPT)
    assert set(prov) == {"fingerprint", "system_prompt_sha256", "task_prompt_sha256",
                         "model_schema_sha256", "model_schema_hash_algorithm", "inference", "retry"}
    assert "model_digest" not in prov


# --------------------------------------------------------------- attempts
def _invalid_then_valid():
    bad = {"visual_kind": "photo", "summary": "x", "warnings": [],
           "payload": {"description": "d", "objects": [{"label": "o", "certainty": "high"}]}}
    return [VisionBackendResult(raw_json=json.dumps(bad), prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ok_photo_result()]


def test_attempts_one_on_first_success(clientapp, backend):
    backend.results = [ok_photo_result()]
    assert _post(clientapp, schema_version="2").json()["usage"]["attempts"] == 1


def test_attempts_two_after_structural_retry(clientapp, backend):
    backend.results = _invalid_then_valid()
    body = _post(clientapp, schema_version="2").json()
    assert body["usage"]["attempts"] == 2 and len(backend.plans) == 2


def test_attempts_two_after_truncated_retry(clientapp, backend):
    truncated = VisionBackendResult(raw_json=ok_photo_result().raw_json, prompt_tokens=1,
                                    completion_tokens=2048, total_tokens=2049, truncated=True)
    backend.results = [truncated, ok_photo_result()]
    body = _post(clientapp, schema_version="2").json()
    assert body["usage"]["attempts"] == 2


def test_v1_unchanged_after_retry(clientapp, backend):
    backend.results = _invalid_then_valid()
    body = _post(clientapp).json()
    assert set(body["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}


# ------------------------------------------------------- erreurs / privacy
def test_error_envelopes_carry_no_provenance(clientapp, backend):
    bad = _invalid_then_valid()[0]
    backend.results = [bad, bad]
    r = _post(clientapp, schema_version="2")
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert set(detail) == {"status", "provider", "message", "code", "retryable", "request_id"}
    spoof = _post(clientapp, mime="image/jpeg", schema_version="2")   # PNG déclaré JPEG
    assert spoof.status_code == 400 and "sha256" not in spoof.text
    assert set(spoof.json()["detail"]) == {"status", "provider", "message", "code", "retryable", "request_id"}


def test_image_hashes_never_logged(clientapp, backend, caplog):
    caplog.set_level(logging.DEBUG)
    raw = _png()
    backend.results = [ok_photo_result(), _invalid_then_valid()[0], _invalid_then_valid()[0]]
    ok = _post(clientapp, image_bytes=raw, schema_version="2").json()
    assert _post(clientapp, image_bytes=raw, schema_version="2").status_code == 502
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    for h in (ok["image"]["source_sha256"], ok["image"]["normalized_sha256"]):
        assert h not in text and h[:8] not in text and h[:12] not in text
    assert "vision.ok" in text   # les événements existent toujours
