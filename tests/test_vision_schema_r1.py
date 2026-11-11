"""Lot AI-GATEWAY-VISION-1A-R1 — le schéma envoyé au modèle doit contraindre
la FORME du payload, pas seulement `visual_kind`.

Défaut constaté en reconnaissance GPU réelle (MODEL-1, 2026-09-15) : le
schéma transmis à Ollama laissait `payload` libre, le modèle a émis
`certainty: "high"` (hors enum V1) et la Gateway a dû rejeter en 502. Ces
tests inspectent le SCHÉMA GÉNÉRÉ lui-même, pas le récit de la recon.
"""

import json

import pytest

from app import main
from tests.test_vision_media import _post, backend, vision_ready  # noqa: F401

TASKS = ("auto", "describe", "table", "diagram", "chart", "technical_drawing")


def _dump(task):
    return json.dumps(main.output_schema_for_task(task), sort_keys=True)


# ------------------------------------------------- défaut : payload libre
@pytest.mark.parametrize("task,marker", [
    ("table", "n_rows"),
    ("describe", "observable_relations"),
    ("diagram", "source_id"),
    ("chart", "exact_labeled"),
    ("technical_drawing", "value_text"),
])
def test_task_schema_embeds_its_payload_definition(task, marker):
    """Le schéma model-facing doit porter la définition du payload du genre."""
    assert marker in _dump(task), f"{task}: payload non contraint (marqueur {marker} absent)"


def test_certainty_enum_is_transmitted_for_table():
    """Régression directe de la recon : `high` ne doit pas pouvoir passer."""
    dumped = _dump("table")
    assert '"exact"' in dumped and '"unreadable"' in dumped
    adapter = main.vision_result_adapter("table")
    good = {"visual_kind": "table", "summary": None, "warnings": [], "payload": {
        "n_rows": 1, "n_cols": 1, "caption": None, "notes": [],
        "cells": [{"row": 0, "col": 0, "row_span": 1, "col_span": 1,
                   "role": "data", "state": "filled", "text": "x",
                   "certainty": "exact"}]}}
    adapter.validate_python(good)                       # accepté
    bad = json.loads(json.dumps(good))
    bad["payload"]["cells"][0]["certainty"] = "high"    # valeur observée en recon
    with pytest.raises(Exception):
        adapter.validate_python(bad)


# ------------------------------------------- enums réellement transmis
@pytest.mark.parametrize("task,values", [
    ("table", ["column_header", "row_header", "corner", "filled", "blank",
               "unreadable", "exact", "approximate", "ambiguous"]),
    ("describe", ["exact", "approximate", "ambiguous", "unreadable"]),
    ("chart", ["exact_labeled", "approximate_visual", "unreadable",
               "stacked_bar", "categorical", "temporal"]),
    ("diagram", ["forward", "bidirectional", "undirected", "unknown"]),
    ("technical_drawing", ["connected_to", "inside", "adjacent_to", "feeds",
                           "enters", "exits", "other"]),
])
def test_task_schema_transmits_frozen_enums(task, values):
    dumped = _dump(task)
    for value in values:
        assert f'"{value}"' in dumped, f"{task}: {value} absent du schéma"


@pytest.mark.parametrize("task,kind,field,bad", [
    ("table", "table", "certainty", "high"),
    ("describe", "photo", "certainty", "high"),
    ("chart", "chart", "precision", "exact"),        # vocabulaire chart distinct
    ("diagram", "diagram", "direction", "backward"),  # retiré du contrat
    ("technical_drawing", "technical_drawing", "relation", "bolted_to"),
])
def test_invalid_enum_values_are_rejected(task, kind, field, bad):
    adapter = main.vision_result_adapter(task)
    payload = _minimal_payload(kind)
    _set_deep(payload, field, bad)
    with pytest.raises(Exception):
        adapter.validate_python({"visual_kind": kind, "summary": None,
                                 "warnings": [], "payload": payload})


def _minimal_payload(kind):
    return json.loads(json.dumps({
        "table": {"n_rows": 1, "n_cols": 1, "caption": None, "notes": [],
                  "cells": [{"row": 0, "col": 0, "row_span": 1, "col_span": 1,
                             "role": "data", "state": "filled", "text": "x",
                             "certainty": "exact"}]},
        "photo": {"description": "d", "setting": None,
                  "objects": [{"label": "o", "count": 1, "attributes": [],
                               "certainty": "exact"}],
                  "observable_relations": [], "visible_text": []},
        "chart": {"chart_type": "bar", "axes": [], "trend_summary": None,
                  "series": [{"id": "s", "label": None, "axis_ids": None,
                              "points": [{"label": "a", "x": "a", "value": 1.0,
                                          "value_text": "1",
                                          "precision": "exact_labeled"}]}]},
        "diagram": {"diagram_type": None, "groups": [], "annotations": [],
                    "nodes": [{"id": "n1", "label": None, "type": None,
                               "certainty": "exact"},
                              {"id": "n2", "label": None, "type": None,
                               "certainty": "exact"}],
                    "edges": [{"source_id": "n1", "target_id": "n2",
                               "direction": "forward", "relation": None,
                               "label": None, "certainty": "exact"}]},
        "technical_drawing": {"drawing_type": None, "explicit_dimensions": [],
                              "symbols": [], "zones": [], "legend": [],
                              "annotations": [],
                              "elements": [{"id": "e1", "kind": "k",
                                            "label": None, "certainty": "exact"},
                                           {"id": "e2", "kind": "k",
                                            "label": None, "certainty": "exact"}],
                              "relations": [{"source_id": "e1", "target_id": "e2",
                                             "relation": "inside", "label": None,
                                             "certainty": "exact"}]},
    }[kind]))


def _set_deep(obj, field, value):
    """Remplace la première occurrence de `field` trouvée en profondeur."""
    if isinstance(obj, dict):
        if field in obj:
            obj[field] = value
            return True
        return any(_set_deep(v, field, value) for v in obj.values())
    if isinstance(obj, list):
        return any(_set_deep(v, field, value) for v in obj)
    return False


# ------------------------------------------------ kinds sans payload
@pytest.mark.parametrize("kind", ["unknown", "decorative"])
def test_null_kinds_reject_any_payload(kind):
    adapter = main.vision_result_adapter("auto")
    adapter.validate_python({"visual_kind": kind, "summary": None,
                             "warnings": [], "payload": None})       # valide
    with pytest.raises(Exception):
        adapter.validate_python({"visual_kind": kind, "summary": None,
                                 "warnings": [], "payload": {}})     # invalide


def test_kind_cannot_validate_with_foreign_payload():
    adapter = main.vision_result_adapter("auto")
    with pytest.raises(Exception):
        adapter.validate_python({"visual_kind": "table", "summary": None,
                                 "warnings": [], "payload": _minimal_payload("photo")})


# ------------------------------------------------------------- mixed
def test_mixed_regions_are_discriminated_and_non_recursive():
    adapter = main.vision_result_adapter("auto")
    region = lambda kind, payload, x: {
        "bbox": [x, 0.1, x + 0.3, 0.9], "visual_kind": kind,
        "payload": payload, "warnings": [], "requires_visual_check": False}
    ok = {"visual_kind": "mixed", "summary": None, "warnings": [], "payload": {
        "regions": [region("chart", _minimal_payload("chart"), 0.05),
                    region("table", _minimal_payload("table"), 0.5)]}}
    adapter.validate_python(ok)
    # région mixed imbriquée : structurellement impossible
    nested = {"visual_kind": "mixed", "summary": None, "warnings": [], "payload": {
        "regions": [region("mixed", {"regions": []}, 0.05),
                    region("table", _minimal_payload("table"), 0.5)]}}
    with pytest.raises(Exception):
        adapter.validate_python(nested)
    # payload étranger dans une région
    wrong = {"visual_kind": "mixed", "summary": None, "warnings": [], "payload": {
        "regions": [region("chart", _minimal_payload("table"), 0.05),
                    region("table", _minimal_payload("table"), 0.5)]}}
    with pytest.raises(Exception):
        adapter.validate_python(wrong)
    assert '"mixed"' not in json.dumps(
        main.output_schema_for_task("auto")["$defs"].get("MixedPayloadOut", {}))


def test_auto_schema_carries_every_frozen_variant():
    defs = main.output_schema_for_task("auto").get("$defs", {})
    for model in ("PhotoResultOut", "TableResultOut", "DiagramResultOut",
                  "ChartResultOut", "TechnicalDrawingResultOut",
                  "MixedResultOut", "DecorativeResultOut", "UnknownResultOut"):
        assert model in defs, model


@pytest.mark.parametrize("task,allowed", [
    ("describe", {"photo", "unknown"}),
    ("table", {"table", "unknown"}),
    ("diagram", {"diagram", "unknown"}),
    ("chart", {"chart", "unknown"}),
    ("technical_drawing", {"technical_drawing", "unknown"}),
    ("auto", {"photo", "table", "diagram", "chart", "technical_drawing",
              "mixed", "decorative", "unknown"}),
])
def test_task_kind_matrix_unchanged_by_r1(task, allowed):
    """La matrice gelée n'est ni élargie ni rétrécie par la correction."""
    from app.vision_schemas import ALLOWED_KINDS_BY_TASK
    assert ALLOWED_KINDS_BY_TASK[task] == allowed
    adapter = main.vision_result_adapter(task)
    for kind in allowed:
        payload = None if kind in ("unknown", "decorative") else (
            {"regions": [
                {"bbox": [0.05, 0.1, 0.4, 0.9], "visual_kind": "chart",
                 "payload": _minimal_payload("chart"), "warnings": [],
                 "requires_visual_check": False},
                {"bbox": [0.5, 0.1, 0.9, 0.9], "visual_kind": "table",
                 "payload": _minimal_payload("table"), "warnings": [],
                 "requires_visual_check": False}]}
            if kind == "mixed" else _minimal_payload(kind))
        adapter.validate_python({"visual_kind": kind, "summary": None,
                                 "warnings": [], "payload": payload})
    for kind in set(main.output_schema_for_task.__globals__["RESULT_MODEL_BY_KIND"]) - allowed:
        with pytest.raises(Exception):
            adapter.validate_python({"visual_kind": kind, "summary": None,
                                     "warnings": [], "payload": None})


# ------------------------------------------------- audit objets libres
def test_no_free_form_semantic_object_remains():
    """Aucun noeud sémantique ne doit accepter un objet arbitraire."""
    for task in TASKS:
        schema = main.output_schema_for_task(task)
        free = []

        def walk(node, path="$"):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" not in node \
                        and "$ref" not in node and "additionalProperties" not in node:
                    free.append(path)
                if node.get("additionalProperties") is True:
                    free.append(path + "[additionalProperties:true]")
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, f"{path}[{i}]")

        walk(schema)
        assert free == [], f"{task}: objets libres restants {free}"


def test_semantic_objects_forbid_extra_fields():
    defs = main.output_schema_for_task("auto").get("$defs", {})
    for name in ("TablePayload", "TableCell", "PhotoPayload", "PhotoObject",
                 "DiagramNode", "DiagramEdge", "ChartPayload", "ChartPoint",
                 "TechnicalDrawingPayload", "MixedPayloadOut"):
        assert defs[name].get("additionalProperties") is False, name


# ------------------------------------------------ stabilité et identité
def test_schema_is_serializable_and_stable():
    for task in TASKS:
        dumps = {json.dumps(main.output_schema_for_task(task), sort_keys=True)
                 for _ in range(3)}
        assert len(dumps) == 1, f"{task}: schéma instable"


def test_gateway_validation_still_rejects_bad_certainty():
    """Défense en profondeur : la validation postérieure reste seule maîtresse."""
    from app.vision_schemas import RawModelOutput
    from app.vision_invariants import validate_semantic_output
    bad = _minimal_payload("table")
    bad["cells"][0]["certainty"] = "high"
    raw = RawModelOutput.model_validate(
        {"visual_kind": "table", "warnings": [], "payload": bad})
    with pytest.raises(Exception):
        validate_semantic_output("table", raw)


# --------------------------------------- pipeline complet (scénario recon)
def test_recon_scenario_end_to_end(clientapp, backend):
    """Reproduit la recon GPU : 1re sortie `certainty:"high"` -> rejetée,
    retry interne unique -> 2e sortie conforme -> 200. Prouve que la
    validation postérieure reste seule maîtresse, schéma renforcé ou non."""
    from app.vision_backend import VisionBackendResult

    payload = _minimal_payload("table")
    bad = json.loads(json.dumps(payload))
    bad["cells"][0]["certainty"] = "high"
    backend.results = [
        VisionBackendResult(raw_json=json.dumps(
            {"visual_kind": "table", "summary": "s", "warnings": [], "payload": bad})),
        VisionBackendResult(raw_json=json.dumps(
            {"visual_kind": "table", "summary": "s", "warnings": [], "payload": payload}),
            prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ]
    r = _post(clientapp, task="table")
    assert r.status_code == 200
    assert r.json()["payload"]["cells"][0]["certainty"] == "exact"
    assert len(backend.plans) == 2            # exactement un retry interne
    # le backend a bien reçu le schéma CONTRAINT
    assert "n_rows" in json.dumps(backend.plans[0].output_json_schema)


def test_backend_plan_carries_constrained_schema(clientapp, backend):
    _post(clientapp, task="describe")
    schema = json.dumps(backend.plans[-1].output_json_schema)
    assert "observable_relations" in schema and '"unreadable"' in schema
