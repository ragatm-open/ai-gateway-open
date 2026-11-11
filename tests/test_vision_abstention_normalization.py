"""Lot AI-GATEWAY-VISION-ABSTENTION-NORMALIZATION-1A — abstention `unknown`
non ambiguë normalisée, tout le reste inchangé (fail-closed).

Classe normalisée, et elle seule : task `describe`, `visual_kind: unknown`,
payload absent/null, AUCUN warning, `summary` absent ou blanc. Backend
TOUJOURS mocké : aucun modèle, aucun GPU, aucun réseau.
"""

import json
import logging

import pytest
from pydantic import ValidationError

from app import main, vision_prompts
from app.metrics import counters
from app.vision_backend import (
    QUALIFIED_FIRST_ATTEMPT,
    QUALIFIED_RETRY_ATTEMPT,
    RETRY_MAX_ATTEMPTS,
    RETRY_TRIGGER,
    VisionBackendResult,
)
from app.vision_invariants import (
    ABSTENTION_NORMALIZATION_REASON,
    VisionInvariantError,
    derive_requires_visual_check,
    normalize_unambiguous_unknown,
    validate_semantic_output,
)
from app.vision_provenance import (
    model_schema_sha256,
    pipeline_identity,
    vision_pipeline_fingerprint,
)
from app.vision_schemas import VISION_TASKS, RawModelOutput
from tests.test_vision_media import (
    _post,
    backend,          # noqa: F401 — fixtures réutilisées
    ok_photo_result,
    vision_ready,     # noqa: F401
)

REQUIRED = "insufficient_visual_evidence"
# Références gelées (MODEL-1 / CV-1) : recalculées ci-dessous, jamais forcées.
SYSTEM_PROMPT_SHA256 = "048ef7108332ecc3d52981dc430978d2ff9c892125576497eebec74c14115a87"
DESCRIBE_PROMPT_SHA256 = "3ff92ee7123743f30a5eab621a0c071029aefc52704e0e07ba836766f2b54bb0"
DESCRIBE_SCHEMA_SHA256 = "bcea8521454921d9a3dbd273420c19c3eee4b5c6833b63a38cee7ef4eb1954cd"
VISION_LARGE_DIGEST = "5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b"
DESCRIBE_FINGERPRINT = "vision-v1:9c68b7db03303789ffb1b34d5ff0da811a8c042cfc912eab89db0fd702dced39"


def _raw(**fields):
    return RawModelOutput.model_validate({"visual_kind": "unknown", **fields})


def _backend_json(obj):
    return VisionBackendResult(raw_json=json.dumps(obj), prompt_tokens=10,
                               completion_tokens=5, total_tokens=15)


B005_CLASS = {"visual_kind": "unknown", "payload": None, "warnings": []}


# ------------------------------------------------------------ fonction pure
@pytest.mark.parametrize("fields", [
    {},                                          # payload et warnings absents
    {"payload": None},
    {"warnings": []},
    {"payload": None, "warnings": []},           # classe B-005
    {"payload": None, "warnings": [], "summary": None},
    {"payload": None, "warnings": [], "summary": ""},
    {"payload": None, "warnings": [], "summary": "  \n"},
])
def test_unambiguous_unknown_is_normalized(fields):
    raw = _raw(**fields)
    out, reason = normalize_unambiguous_unknown("describe", raw)
    assert reason == ABSTENTION_NORMALIZATION_REASON
    assert out.warnings == [REQUIRED]
    assert out.payload is None and out.visual_kind == "unknown"
    assert out.summary == raw.summary
    assert validate_semantic_output("describe", out) is None
    assert derive_requires_visual_check(out.visual_kind, None, out.warnings) is True


def test_already_valid_unknown_is_returned_unchanged():
    raw = _raw(payload=None, warnings=[REQUIRED])
    out, reason = normalize_unambiguous_unknown("describe", raw)
    assert out is raw and reason is None
    assert out.warnings == [REQUIRED]


@pytest.mark.parametrize("fields", [
    {"warnings": ["low_resolution"]},
    {"warnings": ["unreadable_region"]},
    {"warnings": ["low_resolution", "ambiguous_structure"]},
    {"payload": {}},
    {"payload": {"description": "x"}},
    {"warnings": [], "payload": {}},
    {"summary": "Page blanche."},
    {"summary": "Un graphique en barres montrant 3 séries."},
])
def test_ambiguous_unknown_is_not_normalized_and_stays_rejected(fields):
    raw = _raw(**fields)
    out, reason = normalize_unambiguous_unknown("describe", raw)
    assert out is raw and reason is None
    with pytest.raises(VisionInvariantError):
        validate_semantic_output("describe", out)


@pytest.mark.parametrize("task", sorted(set(VISION_TASKS) - {"describe"}))
def test_other_tasks_are_out_of_scope(task):
    raw = _raw(payload=None, warnings=[])
    out, reason = normalize_unambiguous_unknown(task, raw)
    assert out is raw and reason is None
    with pytest.raises(VisionInvariantError):
        validate_semantic_output(task, out)


@pytest.mark.parametrize("kind", ["photo", "table", "diagram", "chart",
                                  "technical_drawing", "mixed", "decorative"])
def test_other_visual_kinds_are_never_touched(kind):
    raw = RawModelOutput.model_validate({"visual_kind": kind, "warnings": []})
    for task in VISION_TASKS:
        out, reason = normalize_unambiguous_unknown(task, raw)
        assert out is raw and reason is None


@pytest.mark.parametrize("bad", [
    {"warnings": None},
    {"warnings": REQUIRED},
    {"warnings": ["pas_un_warning"]},
    {"warnings": [None]},
    {"payload": None, "champ_inconnu": 1},
    {"payload": "null"},
    {"payload": []},
    {"summary": 3},
])
def test_structurally_invalid_outputs_never_reach_normalization(bad):
    with pytest.raises(ValidationError):
        _raw(**bad)


def test_idempotent():
    once, r1 = normalize_unambiguous_unknown("describe", _raw(payload=None, warnings=[]))
    twice, r2 = normalize_unambiguous_unknown("describe", once)
    assert r1 == ABSTENTION_NORMALIZATION_REASON and r2 is None
    assert twice is once and twice.warnings == [REQUIRED]   # jamais de doublon


def test_source_object_is_not_mutated():
    raw = _raw(payload=None, warnings=[])
    before = raw.model_dump()
    warnings_ref = raw.warnings
    out, _ = normalize_unambiguous_unknown("describe", raw)
    assert out is not raw
    assert raw.model_dump() == before and raw.warnings is warnings_ref == []
    assert out.warnings is not warnings_ref


# -------------------------------------------------------------- route (mock)
@pytest.fixture
def real_describe_prompt(monkeypatch, vision_ready):
    """Prompt describe RÉEL (le fixture commun le remplace par un texte de test)
    pour que la provenance V2 rendue soit celle du déploiement qualifié."""
    monkeypatch.setattr(main, "QUALIFIED_PROMPTS", dict(vision_prompts.QUALIFIED_PROMPTS))


def _events(caplog, name):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith(f"event={name} ")]


@pytest.mark.parametrize("schema_version", ["1", "2"])
def test_t1_b005_class_is_served_as_valid_abstention(clientapp, backend, caplog,
                                                     real_describe_prompt, schema_version):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    backend.results = [_backend_json(B005_CLASS)]
    r = _post(clientapp, request_id="rid-b005", schema_version=schema_version)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["task"] == "describe"
    assert body["visual_kind"] == "unknown" and body["payload"] is None
    assert body["warnings"] == [REQUIRED]
    assert body["requires_visual_check"] is True
    assert body["schema_version"] == schema_version
    assert len(backend.plans) == 1                       # aucune seconde inférence
    if schema_version == "2":
        assert body["usage"]["attempts"] == 1
        assert body["pipeline"]["fingerprint"] == DESCRIBE_FINGERPRINT
    snap = counters.snapshot()
    assert snap["vision_abstention_normalized_total"] == 1
    assert snap["vision_ok_total"] == 1 and snap["vision_invalid_output_total"] == 0
    (line,) = _events(caplog, "vision.abstention_normalized")
    fields = dict(p.split("=", 1) for p in line.split(" ")[1:] if "=" in p)
    assert fields == {"attempt": "1", "gateway_build": fields["gateway_build"],
                      "model_logical": "vision-large",
                      "normalization_reason": ABSTENTION_NORMALIZATION_REASON,
                      "request_id": "rid-b005", "task": "describe"}


def test_t1_normalized_on_retry_counts_once(clientapp, backend, caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    backend.results = [VisionBackendResult(raw_json="{tronqué"), _backend_json(B005_CLASS)]
    r = _post(clientapp)
    assert r.status_code == 200 and r.json()["warnings"] == [REQUIRED]
    assert len(backend.plans) == 2
    assert counters.snapshot()["vision_abstention_normalized_total"] == 1
    (line,) = _events(caplog, "vision.abstention_normalized")
    assert " attempt=2 " in f"{line} "


def test_t2_already_valid_abstention_emits_nothing(clientapp, backend, caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    backend.results = [_backend_json({**B005_CLASS, "warnings": [REQUIRED]})]
    r = _post(clientapp)
    assert r.status_code == 200
    body = r.json()
    assert body["warnings"] == [REQUIRED] and body["requires_visual_check"] is True
    assert counters.snapshot()["vision_abstention_normalized_total"] == 0
    assert _events(caplog, "vision.abstention_normalized") == []


@pytest.mark.parametrize("bad", [
    {**B005_CLASS, "warnings": ["low_resolution"]},              # T3 ambiguë
    {**B005_CLASS, "payload": {}},                               # T4 payload
    {**B005_CLASS, "payload": {"description": "x", "setting": None, "objects": [],
                               "observable_relations": [], "visible_text": []}},
    {**B005_CLASS, "summary": "Deux objets posés sur une table."},  # résumé factuel
    {**B005_CLASS, "warnings": None},                            # T6 mal typé
    {**B005_CLASS, "warnings": REQUIRED},
    {**B005_CLASS, "warnings": ["pas_un_warning"]},
    {**B005_CLASS, "extra": True},
    {**B005_CLASS, "visual_kind": "table"},                      # genre interdit (describe)
    {**B005_CLASS, "visual_kind": "decorative"},
])
def test_t3_t4_t6_rejections_keep_historical_retry_then_502(clientapp, backend, caplog, bad):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    backend.results = [_backend_json(bad), _backend_json(bad)]
    r = _post(clientapp)
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["code"] == "vision_schema_validation_failed" and d["retryable"] is False
    assert len(backend.plans) == 2                       # retry historique, jamais > 2
    assert counters.snapshot()["vision_abstention_normalized_total"] == 0
    assert _events(caplog, "vision.abstention_normalized") == []


def test_t3_ambiguous_then_unambiguous_is_accepted_on_retry(clientapp, backend):
    backend.results = [_backend_json({**B005_CLASS, "warnings": ["low_resolution"]}),
                       _backend_json(B005_CLASS)]
    r = _post(clientapp)
    assert r.status_code == 200 and r.json()["warnings"] == [REQUIRED]
    assert len(backend.plans) == 2


def test_t6_invalid_json_and_truncation_are_not_normalized(clientapp, backend):
    backend.results = [VisionBackendResult(raw_json='{"visual_kind": "unknown"'),
                       VisionBackendResult(raw_json='{"visual_kind": "unknown"')]
    r = _post(clientapp)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "vision_invalid_output"
    t = VisionBackendResult(raw_json=json.dumps(B005_CLASS), truncated=True)
    backend.results = [t, t]
    r = _post(clientapp)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "vision_truncated_output"
    assert counters.snapshot()["vision_abstention_normalized_total"] == 0


def test_out_of_scope_task_keeps_historical_rejection(clientapp, backend):
    backend.results = [_backend_json(B005_CLASS), _backend_json(B005_CLASS)]
    r = _post(clientapp, task="table")
    assert r.status_code == 502
    assert r.json()["detail"]["message"].startswith(
        "sortie modèle invalide : unknown exige le warning insufficient_visual_evidence")


@pytest.mark.parametrize("schema_version", ["1", "2"])
def test_t5_photo_response_is_unchanged(clientapp, backend, caplog,
                                        real_describe_prompt, schema_version):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    backend.results = [ok_photo_result()]
    body = _post(clientapp, request_id="rid-photo", schema_version=schema_version).json()
    expected = json.loads(ok_photo_result().raw_json)
    assert body["visual_kind"] == "photo" and body["payload"] == expected["payload"]
    assert body["warnings"] == [] and body["requires_visual_check"] is False
    assert body["summary"] == expected["summary"]
    if schema_version == "2":
        assert body["pipeline"]["fingerprint"] == DESCRIBE_FINGERPRINT
    assert _events(caplog, "vision.abstention_normalized") == []
    assert counters.snapshot()["vision_abstention_normalized_total"] == 0


def test_t7_model_identity_prompts_schema_options_fingerprint_unchanged(
        clientapp, backend, real_describe_prompt):
    assert vision_prompts.sha256_of(vision_prompts.SYSTEM_PROMPT) == SYSTEM_PROMPT_SHA256
    assert vision_prompts.sha256_of(vision_prompts.DESCRIBE_PROMPT) == DESCRIBE_PROMPT_SHA256
    assert model_schema_sha256("describe") == DESCRIBE_SCHEMA_SHA256
    assert (QUALIFIED_FIRST_ATTEMPT.temperature, QUALIFIED_FIRST_ATTEMPT.seed,
            QUALIFIED_FIRST_ATTEMPT.num_ctx, QUALIFIED_FIRST_ATTEMPT.num_predict) == (0.0, 42, 8192, 2048)
    assert (QUALIFIED_RETRY_ATTEMPT.temperature, QUALIFIED_RETRY_ATTEMPT.seed) == (0.35, 43)
    assert (RETRY_MAX_ATTEMPTS, RETRY_TRIGGER) == (2, "structural_invalid_or_truncated")
    identity = pipeline_identity("describe", VISION_LARGE_DIGEST, vision_prompts.DESCRIBE_PROMPT)
    assert vision_pipeline_fingerprint(identity) == DESCRIBE_FINGERPRINT
    # et le plan réellement envoyé au backend pour une abstention normalisée
    backend.results = [_backend_json(B005_CLASS)]
    body = _post(clientapp, schema_version="2").json()
    (plan,) = backend.plans
    assert plan.model_concrete == "mistral-small3.2:24b"
    assert plan.model_logical == "vision-large"
    assert plan.system_prompt == vision_prompts.SYSTEM_PROMPT
    assert plan.task_prompt == vision_prompts.DESCRIBE_PROMPT
    assert plan.output_json_schema == main.output_schema_for_task("describe")
    assert (plan.options.temperature, plan.options.seed, plan.options.num_ctx,
            plan.options.num_predict) == (0.0, 42, 8192, 2048)
    assert body["model"] == {"logical": "vision-large", "concrete": "mistral-small3.2:24b",
                             "digest": VISION_LARGE_DIGEST}
    assert body["pipeline"]["model_schema_sha256"] == DESCRIBE_SCHEMA_SHA256
    assert body["pipeline"]["fingerprint"] == DESCRIBE_FINGERPRINT


def test_t8_normalization_event_carries_no_content(clientapp, backend, caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    ctx = {"caption": "ZZCAPSECRETZZ", "alt_text": "ZZALTSECRETZZ",
           "nearby_text": "ZZNEARSECRETZZ"}
    backend.results = [_backend_json({**B005_CLASS, "summary": " "})]
    r = _post(clientapp, context=ctx)
    assert r.status_code == 200
    (line,) = _events(caplog, "vision.abstention_normalized")
    for forbidden in ("ZZ", "summary", "payload", "image", "data", "prompt"):
        assert forbidden not in line
