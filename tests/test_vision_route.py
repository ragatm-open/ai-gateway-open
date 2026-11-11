"""Lot vision-1a — route /v1/vision/analyze : goldens, gouvernance, retry,
overload, timeout, santé, privacy. Backend TOUJOURS mocké (aucun modèle réel,
aucune exécution GPU, aucun réseau)."""

import json
import logging
import threading

import pytest

from app import main
from app.errors import UpstreamError
from app.metrics import counters, gauges
from app.vision_admission import VisionAdmission
from app.vision_backend import VisionBackendResult, VisionBackendTimeout
from tests.test_vision_media import (
    FULL_TASKS,
    RecordingBackend,
    _b64,
    _png,
    _post,
    backend,          # noqa: F401 — fixtures réutilisées
    ok_photo_result,
    vision_ready,     # noqa: F401
)


def _result(kind, payload, summary="Résumé.", warnings=(), usage=(10, 5, 15),
            truncated=False):
    raw = {"visual_kind": kind, "summary": summary,
           "warnings": list(warnings), "payload": payload}
    p, c, t = usage if usage else (None, None, None)
    return VisionBackendResult(raw_json=json.dumps(raw), prompt_tokens=p,
                               completion_tokens=c, total_tokens=t,
                               truncated=truncated)


GOLDEN_TABLE = {
    "n_rows": 4, "n_cols": 3, "caption": "Ventes 2025 (k€)", "notes": [],
    "cells": [
        {"row": 0, "col": 0, "row_span": 2, "col_span": 1, "role": "corner",
         "state": "blank", "text": None, "certainty": "exact"},
        {"row": 0, "col": 1, "row_span": 1, "col_span": 2, "role": "column_header",
         "state": "filled", "text": "Ventes", "certainty": "exact"},
        {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "role": "column_header",
         "state": "filled", "text": "T1", "certainty": "exact"},
        {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "role": "column_header",
         "state": "filled", "text": "T2", "certainty": "exact"},
        {"row": 2, "col": 0, "row_span": 1, "col_span": 1, "role": "row_header",
         "state": "filled", "text": "Nord", "certainty": "exact"},
        {"row": 2, "col": 1, "row_span": 1, "col_span": 1, "role": "data",
         "state": "filled", "text": "120", "certainty": "exact"},
        {"row": 2, "col": 2, "row_span": 1, "col_span": 1, "role": "data",
         "state": "blank", "text": None, "certainty": "exact"},
        {"row": 3, "col": 0, "row_span": 1, "col_span": 1, "role": "row_header",
         "state": "filled", "text": "Sud", "certainty": "exact"},
        {"row": 3, "col": 1, "row_span": 1, "col_span": 1, "role": "data",
         "state": "filled", "text": "95", "certainty": "exact"},
        {"row": 3, "col": 2, "row_span": 1, "col_span": 1, "role": "data",
         "state": "unreadable", "text": None, "certainty": "unreadable"},
    ],
}

GOLDEN_DIAGRAM = {
    "diagram_type": "flowchart",
    "nodes": [
        {"id": "n1", "label": "Capteur", "type": "input", "certainty": "exact"},
        {"id": "n2", "label": "Contrôleur", "type": "process", "certainty": "exact"},
        {"id": "n3", "label": "Vanne", "type": "actuator", "certainty": "exact"}],
    "edges": [
        {"source_id": "n1", "target_id": "n2", "direction": "forward",
         "relation": None, "label": "4-20 mA", "certainty": "exact"},
        {"source_id": "n2", "target_id": "n3", "direction": "forward",
         "relation": "commande", "label": "commande", "certainty": "exact"}],
    "groups": [{"id": "g1", "label": "Boucle", "members": ["n2", "n3"]}],
    "annotations": [{"text": "Rev. C", "certainty": "exact"}],
}

GOLDEN_CHART = {
    "chart_type": "bar",
    "axes": [
        {"id": "ax", "orientation": "x", "label": "Trimestre", "unit": None,
         "scale": "categorical", "range": None, "certainty": "exact"},
        {"id": "ay", "orientation": "y", "label": "CA", "unit": "k€",
         "scale": "linear",
         "range": {"min": 0, "max": 200, "certainty": "approximate"},
         "certainty": "exact"}],
    "series": [{"id": "s1", "label": "CA 2025", "axis_ids": ["ax", "ay"],
                "points": [
        {"label": "T1", "x": "T1", "value": 120, "value_text": "120",
         "precision": "exact_labeled"},
        {"label": "T2", "x": "T2", "value": 95, "value_text": None,
         "precision": "approximate_visual"},
        {"label": "T3", "x": "T3", "value": None, "value_text": None,
         "precision": "unreadable"}]}],
    "trend_summary": "Baisse de T1 à T2 ; T3 illisible.",
}

GOLDEN_DRAWING = {
    "drawing_type": "plan mécanique",
    "elements": [
        {"id": "e1", "kind": "perçage", "label": "A", "certainty": "exact"},
        {"id": "e2", "kind": "plaque", "label": None, "certainty": "exact"}],
    "explicit_dimensions": [
        {"value_text": "Ø45", "unit": "mm", "value": 45, "between": ["e1"],
         "certainty": "exact"}],
    "symbols": [], "zones": [{"id": "z1", "label": "Détail B", "members": ["e1"]}],
    "relations": [{"source_id": "e1", "target_id": "e2", "relation": "inside",
                   "label": None, "certainty": "exact"}],
    "legend": [], "annotations": [{"text": "NE PAS PEINDRE", "certainty": "exact"}],
}


# --------------------------------------------------------------- goldens
def test_golden_photo_envelope(clientapp, backend):
    r = _post(clientapp, request_id="g-photo-1")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "provider", "upstream_provider",
                         "schema_version", "task", "visual_kind", "request_id",
                         "summary", "warnings", "requires_visual_check",
                         "payload", "model", "pipeline", "image", "usage",
                         "latency_ms", "truncated"}   # schéma FERMÉ V1
    assert body["status"] == "ok" and body["provider"] == "ai_gateway"
    assert body["upstream_provider"] == "ollama"
    assert body["task"] == "describe" and body["visual_kind"] == "photo"
    assert body["request_id"] == "g-photo-1"
    assert body["model"] == {"logical": "vision-large",
                             "concrete": "mistral-small3.2:24b",
                             "digest": main.registry.resolve_vision("vision-large").digest}
    d8 = body["model"]["digest"][:8]
    assert body["pipeline"] == {
        "prompt_profile": "vision_describe_v1", "prompt_profile_version": "1",
        "pipeline_id": f"vision-large@{d8}/vision_describe_v1@v1"}
    assert body["truncated"] is False
    assert body["requires_visual_check"] is False
    assert isinstance(body["latency_ms"], int)
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 5,
                             "total_tokens": 15}


def test_null_usage_is_preserved_not_zeroed(clientapp, backend):
    backend.results = [_result("photo", ok_photo_json()["payload"], usage=None)]
    body = _post(clientapp).json()
    assert body["usage"] == {"prompt_tokens": None, "completion_tokens": None,
                             "total_tokens": None}


def ok_photo_json():
    return json.loads(ok_photo_result().raw_json)


def test_golden_table(clientapp, backend):
    backend.results = [_result("table", GOLDEN_TABLE,
                               warnings=["unreadable_region"])]
    r = _post(clientapp, task="table")
    body = r.json()
    assert r.status_code == 200 and body["visual_kind"] == "table"
    assert body["warnings"] == ["unreadable_region"]
    assert body["requires_visual_check"] is True
    cells = body["payload"]["cells"]
    assert any(c["state"] == "blank" and c["text"] is None for c in cells)
    assert any(c["state"] == "unreadable" for c in cells)
    assert any(c["col_span"] == 2 for c in cells)   # fusion émise une fois


@pytest.mark.parametrize("mutate", [
    lambda t: t["cells"].pop(),                                     # trou
    lambda t: t["cells"].append(dict(t["cells"][5], row=2, col=1)),  # chevauchement
    lambda t: t["cells"][1].update(col_span=3),                     # span hors grille
    lambda t: t["cells"][5].update(text=None),                      # filled+null
    lambda t: t["cells"][6].update(text="x"),                       # blank+texte
    lambda t: t["cells"][9].update(text="x"),                       # unreadable+texte
    lambda t: t["cells"][6].update(certainty="ambiguous"),          # blank!=exact
])
def test_table_invariant_violations_are_502(clientapp, backend, mutate):
    import copy
    bad = copy.deepcopy(GOLDEN_TABLE); mutate(bad)
    backend.results = [_result("table", bad), _result("table", bad)]
    r = _post(clientapp, task="table")
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["code"] == "vision_schema_validation_failed"
    assert d["retryable"] is False
    assert len(backend.plans) == 2   # retry interne unique


def test_golden_diagram_and_dangling_edge(clientapp, backend):
    backend.results = [_result("diagram", GOLDEN_DIAGRAM)]
    r = _post(clientapp, task="diagram")
    assert r.status_code == 200
    assert r.json()["requires_visual_check"] is False
    import copy
    bad = copy.deepcopy(GOLDEN_DIAGRAM)
    bad["edges"][0]["target_id"] = "fantome"
    backend.results = [_result("diagram", bad), _result("diagram", GOLDEN_DIAGRAM)]
    backend.plans.clear()
    r = _post(clientapp, task="diagram")
    assert r.status_code == 200 and len(backend.plans) == 2  # retry puis succès


def test_golden_chart_value_provenance(clientapp, backend):
    backend.results = [_result("chart", GOLDEN_CHART)]
    body = _post(clientapp, task="chart").json()
    pts = body["payload"]["series"][0]["points"]
    assert [p["precision"] for p in pts] == ["exact_labeled",
                                             "approximate_visual", "unreadable"]
    assert pts[2]["value"] is None
    assert body["requires_visual_check"] is True  # unreadable dans le payload


def test_chart_unreadable_with_value_is_rejected(clientapp, backend):
    import copy
    bad = copy.deepcopy(GOLDEN_CHART)
    bad["series"][0]["points"][2]["value"] = 42   # valeur inventée
    backend.results = [_result("chart", bad), _result("chart", bad)]
    assert _post(clientapp, task="chart").status_code == 502


def test_golden_technical_drawing_and_dangling_dimension(clientapp, backend):
    backend.results = [_result("technical_drawing", GOLDEN_DRAWING)]
    r = _post(clientapp, task="technical_drawing")
    assert r.status_code == 200
    assert r.json()["payload"]["explicit_dimensions"][0]["value_text"] == "Ø45"
    import copy
    bad = copy.deepcopy(GOLDEN_DRAWING)
    bad["explicit_dimensions"][0]["between"] = ["e999"]
    backend.results = [_result("technical_drawing", bad),
                       _result("technical_drawing", bad)]
    assert _post(clientapp, task="technical_drawing").status_code == 502


def test_golden_mixed_and_recursion_forbidden(clientapp, backend):
    mixed = {"regions": [
        {"bbox": [0.05, 0.08, 0.95, 0.52], "visual_kind": "chart",
         "payload": GOLDEN_CHART, "warnings": [], "requires_visual_check": False},
        {"bbox": [0.05, 0.58, 0.95, 0.93], "visual_kind": "table",
         "payload": GOLDEN_TABLE, "warnings": [], "requires_visual_check": False}]}
    backend.results = [_result("mixed", mixed)]
    r = _post(clientapp, task="auto")
    assert r.status_code == 200 and r.json()["visual_kind"] == "mixed"
    nested = {"regions": [
        {"bbox": [0.1, 0.1, 0.5, 0.5], "visual_kind": "mixed",
         "payload": {"regions": []}, "warnings": [], "requires_visual_check": False},
        {"bbox": [0.5, 0.5, 0.9, 0.9], "visual_kind": "text",
         "payload": {"lines": [{"text": "a", "certainty": "exact"}]},
         "warnings": [], "requires_visual_check": False}]}
    backend.results = [_result("mixed", nested), _result("mixed", nested)]
    assert _post(clientapp, task="auto").status_code == 502


@pytest.mark.parametrize("bbox", [
    [-0.1, 0, 0.5, 0.5], [0, 0, 1.2, 0.5], [0.5, 0.1, 0.2, 0.5],
    [0.1, 0.3, 0.1, 0.5], [0.1, 0.3, 0.6, 0.3],
])
def test_mixed_bbox_violations(clientapp, backend, bbox):
    mixed = {"regions": [
        {"bbox": bbox, "visual_kind": "text",
         "payload": {"lines": [{"text": "a", "certainty": "exact"}]},
         "warnings": [], "requires_visual_check": False},
        {"bbox": [0.6, 0.6, 0.9, 0.9], "visual_kind": "decorative",
         "payload": None, "warnings": [], "requires_visual_check": False}]}
    backend.results = [_result("mixed", mixed), _result("mixed", mixed)]
    assert _post(clientapp, task="auto").status_code == 502


def test_unknown_semantics(clientapp, backend):
    backend.results = [_result("unknown", None, summary=None,
                               warnings=["insufficient_visual_evidence"])]
    body = _post(clientapp, task="table").json()
    assert body["visual_kind"] == "unknown" and body["payload"] is None
    assert body["warnings"] == ["insufficient_visual_evidence"]
    assert body["requires_visual_check"] is True


def test_unknown_without_required_warning_is_rejected(clientapp, backend):
    # Hors de la classe normalisée (lot ABSTENTION-NORMALIZATION-1A) : task
    # table ET résumé non vide. Classe normalisée et ses frontières :
    # tests/test_vision_abstention_normalization.py.
    backend.results = [_result("unknown", None), _result("unknown", None)]
    assert _post(clientapp, task="table").status_code == 502


def test_decorative_semantics(clientapp, backend):
    backend.results = [_result("decorative", None, summary="Bandeau décoratif.")]
    body = _post(clientapp, task="auto").json()
    assert body["visual_kind"] == "decorative" and body["payload"] is None
    assert body["warnings"] == [] and body["requires_visual_check"] is False


def test_task_kind_matrix_enforced(clientapp, backend):
    # task=table ne peut pas rendre photo — matrice gelée.
    backend.results = [ok_photo_result(), ok_photo_result()]
    assert _post(clientapp, task="table").status_code == 502
    # et l'union de sortie par task est correcte (§112-113). Depuis le lot
    # vision-1a-r1 le schéma model-facing est une union DISCRIMINÉE : les
    # genres admis se lisent sur les variantes, pas sur un enum isolé — et le
    # payload de chaque genre y est structurellement contraint.
    from app.vision_schemas import ALLOWED_KINDS_BY_TASK, RESULT_MODEL_BY_KIND

    def kinds_of(task):
        defs = main.output_schema_for_task(task).get("$defs", {})
        return {kind for kind, model in RESULT_MODEL_BY_KIND.items()
                if model.__name__ in defs}

    assert kinds_of("table") == ALLOWED_KINDS_BY_TASK["table"] == {"table", "unknown"}
    assert kinds_of("auto") == {"photo", "table", "diagram", "chart",
                                "technical_drawing", "mixed", "decorative",
                                "unknown"}
    assert "n_rows" in json.dumps(main.output_schema_for_task("table"))


def test_warnings_deduplicated_and_sorted(clientapp, backend):
    backend.results = [_result("photo", ok_photo_json()["payload"],
                               warnings=["unreadable_region", "low_resolution",
                                         "unreadable_region"])]
    assert _post(clientapp).json()["warnings"] == ["low_resolution",
                                                   "unreadable_region"]


# ------------------------------------------------------------ gouvernance
def test_feature_off_by_default(clientapp, monkeypatch):
    monkeypatch.setattr(main.probe_client, "list_running_models", lambda timeout=5: [])
    r = clientapp.post("/v1/vision/analyze", json={
        "schema_version": "1", "task": "auto",
        "image": {"data": _b64(_png()), "mime_type": "image/png"}})
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["code"] == "vision_not_enabled" and d["retryable"] is False
    h = clientapp.get("/health/vision")
    assert h.status_code == 200 and h.json()["status"] == "disabled"
    assert main.settings.vision_enabled is False  # défaut du dépôt


def test_unknown_model_rejected(clientapp, backend):
    r = _post(clientapp, model="inexistant")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "vision_model_not_allowed"


def test_text_only_model_rejected_no_fallback(clientapp, backend):
    r = _post(clientapp, model="llama-chat")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "vision_model_not_allowed"
    assert backend.plans == []


def test_production_enabled_serves_describe_with_keep_alive_zero(clientapp, backend, monkeypatch, vision_ready):
    """Lot production-full-capability-enablement : en production, le flag
    AI_GATEWAY_VISION_ENABLED sert `describe` ; la résidence reste forcée à 0."""
    monkeypatch.setattr(main.settings, "environment", "production")
    monkeypatch.setattr(main.settings, "vision_enabled", True)
    monkeypatch.setattr(main.settings, "vision_keep_alive_seconds", 0)
    r = _post(clientapp)
    assert r.status_code == 200, r.text
    assert backend.plans and all(getattr(p, "keep_alive_seconds", 0) == 0 for p in backend.plans)
    h = clientapp.get("/health/vision").json()
    assert h["enabled"] is True and h["environment"] == "production" and h["keep_alive_seconds"] == 0
    monkeypatch.setattr(main.settings, "vision_enabled", False)
    assert _post(clientapp).status_code == 503


def test_environment_disallowed_model_rejected(clientapp, backend, monkeypatch,
                                               vision_ready):
    monkeypatch.setattr(vision_ready, "production_allowed", False)
    r = _post(clientapp, model="vision-large")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "vision_model_not_allowed"


def test_task_not_qualified_rejected(clientapp, backend, monkeypatch, vision_ready):
    monkeypatch.setattr(vision_ready, "vision_tasks", ["describe"])
    r = _post(clientapp, task="chart")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "vision_task_not_qualified"


def test_repo_registry_exposes_only_qualified_tasks():
    """Le registre livré n'expose QUE ce que MODEL-1 a qualifié : `describe`.
    auto/table/diagram/chart/technical_drawing existent au schéma V1 mais ont
    échoué leurs hard gates — elles restent fail-closed."""
    profile = main.registry.resolve_vision("vision-large")
    assert profile.vision_tasks == ["describe"]
    assert profile.ollama_name == "mistral-small3.2:24b"
    assert profile.vision_max_image_pixels == 2_000_000
    from app.vision_prompts import QUALIFIED_PROMPTS
    assert set(QUALIFIED_PROMPTS) == {"describe"}


def test_digest_mismatch_unavailable(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.probe_client, "list_models",
                        lambda timeout=5: [{"name": "mistral-small3.2:24b",
                                            "digest": "f" * 64}])
    r = _post(clientapp)
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "vision_model_unavailable"


def test_missing_default_model_unavailable(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.registry, "vision_default_model", None)
    r = _post(clientapp)   # model omis
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "vision_model_unavailable"


# ------------------------------------------------------------ retry/erreurs
def test_retry_invalid_json_then_valid(clientapp, backend):
    backend.results = [VisionBackendResult(raw_json="{pas du json"),
                       ok_photo_result()]
    assert _post(clientapp).status_code == 200
    assert len(backend.plans) == 2


def test_retry_invalid_json_twice_502(clientapp, backend):
    backend.results = [VisionBackendResult(raw_json="{"),
                       VisionBackendResult(raw_json="[1,2")]
    r = _post(clientapp)
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["code"] == "vision_invalid_output" and d["retryable"] is False
    assert len(backend.plans) == 2   # jamais > 2


def test_truncated_twice_never_partial_success(clientapp, backend):
    t = _result("photo", ok_photo_json()["payload"], truncated=True)
    backend.results = [t, t]
    r = _post(clientapp)
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "vision_truncated_output"
    assert len(backend.plans) == 2


def test_upstream_error_no_hidden_retry(clientapp, backend):
    backend.results = [UpstreamError("connexion refusée")]
    r = _post(clientapp, request_id="rid-up")
    d = r.json()["detail"]
    assert r.status_code == 502 and d["code"] == "vision_upstream_error"
    assert d["retryable"] is True and d["request_id"] == "rid-up"
    assert len(backend.plans) == 1   # AUCUN retry caché sur échec transport


def test_timeout_504(clientapp, backend):
    backend.results = [VisionBackendTimeout()]
    r = _post(clientapp)
    d = r.json()["detail"]
    assert r.status_code == 504 and d["code"] == "vision_timeout"
    assert d["retryable"] is True


def test_overload_503_without_backend_call(clientapp, backend, monkeypatch):
    held = VisionAdmission(1); assert held.try_acquire()     # slot occupé (synthétique)
    monkeypatch.setattr(main, "vision_admission", held)
    monkeypatch.setattr(main.settings, "vision_timeout_seconds", 0.05)
    r = _post(clientapp, request_id="rid-ov")
    d = r.json()["detail"]
    assert r.status_code == 503 and d["code"] == "vision_overloaded"
    assert d["retryable"] is True and d["request_id"] == "rid-ov"
    assert backend.plans == []
    assert gauges.total() == 0


# ------------------------------------------------------------------ santé
def test_health_vision_states(clientapp, backend, monkeypatch, vision_ready):
    monkeypatch.setattr(main.probe_client, "list_running_models",
                        lambda timeout=5: [])
    body = clientapp.get("/health/vision").json()
    assert body["status"] == "ok" and body["enabled"] is True
    entry = body["models"][0]
    assert entry["installed"] is True and entry["digest_match"] is True
    assert entry["resident"] is False   # installé != résident, PAS unhealthy
    monkeypatch.setattr(main.probe_client, "list_running_models",
                        lambda timeout=5: ["mistral-small3.2:24b"])
    assert clientapp.get("/health/vision").json()["models"][0]["resident"] is True
    monkeypatch.setattr(main.probe_client, "list_models",
                        lambda timeout=5: (_ for _ in ()).throw(UpstreamError("down")))
    assert clientapp.get("/health/vision").status_code == 503


def test_health_vision_never_calls_backend(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.probe_client, "list_running_models",
                        lambda timeout=5: [])
    clientapp.get("/health/vision")
    assert backend.plans == []


def test_health_ready_never_touches_vision(clientapp, backend):
    assert clientapp.get("/health/ready").status_code == 200
    assert backend.plans == []


def test_models_endpoint_additive_vision_metadata(clientapp, monkeypatch):
    monkeypatch.setattr(main.client, "list_models", lambda timeout=5: [
        {"name": "mistral-small3.2:24b", "size": 1, "modified_at": "x"},
        {"name": "llama3.1:8b", "size": 1, "modified_at": "x"}])
    models = {m["name"]: m for m in clientapp.get("/v1/models").json()["models"]}
    vis = models["mistral-small3.2:24b"]
    assert vis["modalities"] == ["text", "image"]
    assert vis["vision"] == {"tasks": ["describe"], "schema_versions": ["1", "2"],
                             "max_images": 1, "max_image_pixels": 2_000_000}
    assert "vision" not in models["llama3.1:8b"] and "modalities" not in models["llama3.1:8b"]


# ---------------------------------------------------------------- privacy
def test_privacy_sentinels_absent_from_logs_and_errors(clientapp, backend,
                                                       caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    img_sentinel = b"ZZVISIONIMGSECRETZZ" + _png()
    ctx = {"caption": "ZZCAPSECRETZZ", "alt_text": "ZZALTSECRETZZ",
           "nearby_text": "ZZNEARSECRETZZ"}
    # succès
    r = _post(clientapp, image_bytes=_png(), context=ctx)
    assert r.status_code == 200
    # échec de validation (base64 du sentinel binaire mais mime spoofé)
    backend.results = [ok_photo_result()]
    _post(clientapp, image_bytes=img_sentinel, context=ctx)
    # sortie modèle invalide contenant un secret : jamais relayée
    backend.results = [VisionBackendResult(raw_json='{"ZZRAWSECRETZZ": 1'),
                       VisionBackendResult(raw_json='{"ZZRAWSECRETZZ": 1')]
    r = _post(clientapp, context=ctx)
    assert "ZZRAWSECRETZZ" not in r.text
    # timeout
    backend.results = [VisionBackendTimeout()]
    _post(clientapp, context=ctx)
    logs = "\n".join(rec.getMessage() for rec in caplog.records)
    for sentinel in ("ZZCAPSECRETZZ", "ZZALTSECRETZZ", "ZZNEARSECRETZZ",
                     "ZZRAWSECRETZZ", "ZZVISIONIMGSECRETZZ",
                     _b64(img_sentinel)[:40]):
        assert sentinel not in logs, sentinel


def test_vision_counters_and_events(clientapp, backend, caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    _post(clientapp, request_id="rid-ok")
    snap = counters.snapshot()
    assert snap["vision_requests_total"] == 1 and snap["vision_ok_total"] == 1
    line = next(r.getMessage() for r in caplog.records
                if r.getMessage().startswith("event=vision.ok "))
    fields = dict(p.split("=", 1) for p in line.split(" ")[1:] if "=" in p)
    assert fields["model_logical"] == "vision-large"
    assert fields["request_id"] == "rid-ok"
    assert "payload" not in fields and "summary" not in fields
