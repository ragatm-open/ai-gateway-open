"""Lot CV-0 — correctness du contrat Vision V1, SANS changement perceptif.

1. `requires_visual_check` : les signaux portés par une région `mixed`
   (warnings de région, `requires_visual_check` de région) remontent au
   top-level ; les déclencheurs existants sont conservés.
2. Warnings : ceux des régions sont agrégés aux warnings top-level, dans un
   ordre déterministe, sans doublon.
3. `describe` : réponse nominale strictement inchangée (le fix est latent).
4. `locale` : RÉSERVÉ, non effectif en V1 — gelé par test.

Backend toujours mocké : aucun modèle, aucun GPU, aucun réseau.
"""

import json

import pytest

from app.vision_backend import (
    QUALIFIED_FIRST_ATTEMPT,
    OllamaVisionBackend,
    VisionInferencePlan,
)
from app.vision_invariants import (
    collect_warnings,
    derive_requires_visual_check,
    validate_semantic_output,
)
from app.vision_prompts import DESCRIBE_PROMPT, SYSTEM_PROMPT
from app.vision_schemas import RawModelOutput, output_schema_for_task
from tests.test_vision_media import (
    _post,
    backend,          # noqa: F401 — fixtures réutilisées
    ok_photo_result,
    vision_ready,     # noqa: F401
)
from tests.test_vision_route import GOLDEN_CHART, GOLDEN_TABLE, _result

_TEXT = {"lines": [{"text": "A", "certainty": "exact"}]}


def _mixed(region_warnings=(), region_rvc=False, top_warnings=(), text=_TEXT):
    """Deux régions, tout `certainty` exact : seuls les signaux de niveau
    région varient. Région 1 = chart, région 2 = texte."""
    return {
        "visual_kind": "mixed", "summary": None, "warnings": list(top_warnings),
        "payload": {"regions": [
            {"bbox": [0.05, 0.05, 0.5, 0.5], "visual_kind": "chart",
             "payload": GOLDEN_CHART_EXACT, "warnings": list(region_warnings),
             "requires_visual_check": region_rvc},
            {"bbox": [0.5, 0.5, 0.95, 0.95], "visual_kind": "text",
             "payload": text, "warnings": [], "requires_visual_check": False}]},
    }


# Chart golden sans aucun point `unreadable` : le contrôle négatif doit être
# faux POUR LA BONNE RAISON (aucun signal), pas masqué par un autre déclencheur.
GOLDEN_CHART_EXACT = json.loads(json.dumps(GOLDEN_CHART))
GOLDEN_CHART_EXACT["series"][0]["points"] = [
    {"label": "T1", "x": "T1", "value": 120, "value_text": "120", "precision": "exact_labeled"}]
GOLDEN_CHART_EXACT["axes"][1]["range"]["certainty"] = "exact"


def _derive(raw_dict):
    raw = RawModelOutput.model_validate(raw_dict)
    payload = validate_semantic_output("auto", raw)
    warnings = collect_warnings(sorted(set(raw.warnings)), payload)
    return derive_requires_visual_check(raw.visual_kind, payload, warnings), warnings


# ------------------------------------------------ requires_visual_check
def test_mixed_region_warning_propagates_to_top_level():
    rvc, warnings = _derive(_mixed(region_warnings=["low_resolution"]))
    assert rvc is True and warnings == ["low_resolution"]


def test_mixed_region_requires_visual_check_propagates_to_top_level():
    rvc, warnings = _derive(_mixed(region_rvc=True))
    assert rvc is True and warnings == []


def test_mixed_negative_control_all_exact_no_signal():
    rvc, warnings = _derive(_mixed())
    assert rvc is False and warnings == []


def test_existing_trigger_nested_certainty_ambiguous_unchanged():
    text = {"lines": [{"text": "A", "certainty": "ambiguous"}]}
    rvc, warnings = _derive(_mixed(text=text))
    assert rvc is True and warnings == []


def test_existing_trigger_top_level_warning_unchanged():
    rvc, warnings = _derive(_mixed(top_warnings=["ambiguous_structure"]))
    assert rvc is True and warnings == ["ambiguous_structure"]


def test_existing_triggers_unknown_and_direction_unchanged():
    assert derive_requires_visual_check("unknown", None, []) is True
    from app.vision_schemas import DiagramPayload
    diagram = DiagramPayload.model_validate({
        "nodes": [{"id": "a", "certainty": "exact"}, {"id": "b", "certainty": "exact"}],
        "edges": [{"source_id": "a", "target_id": "b", "direction": "unknown",
                   "certainty": "exact"}]})
    assert derive_requires_visual_check("diagram", diagram, []) is True


# ----------------------------------------------------- agrégation warnings
def test_collect_warnings_order_and_deduplication_pure():
    class _Payload:
        def model_dump(self):
            return {"regions": [
                {"warnings": ["low_resolution", "global_a"], "payload": {"x": 1}},
                {"warnings": ["cropped"], "payload": None}]}

    assert collect_warnings(["global_a"], _Payload()) == ["global_a", "low_resolution", "cropped"]
    # top-level conservé dans l'ordre reçu, régions dans l'ordre du document,
    # doublons ignorés à la première occurrence ; sans payload : inchangé.
    assert collect_warnings(["b", "a"], None) == ["b", "a"]
    assert collect_warnings([], _Payload()) == ["low_resolution", "global_a", "cropped"]


def test_mixed_region_warnings_aggregated_in_response(clientapp, backend):
    mixed = _mixed(region_warnings=["unreadable_region", "low_resolution"],
                   top_warnings=["low_resolution"])
    # région 2 porte aussi un warning, après la région 1
    mixed["payload"]["regions"][1]["warnings"] = ["ambiguous_structure"]
    backend.results = [_result("mixed", mixed["payload"],
                               warnings=mixed["warnings"])]
    body = _post(clientapp, task="auto").json()
    assert body["visual_kind"] == "mixed"
    assert body["warnings"] == ["low_resolution", "unreadable_region", "ambiguous_structure"]
    assert body["requires_visual_check"] is True


def test_mixed_region_rvc_true_in_response_without_any_warning(clientapp, backend):
    mixed = _mixed(region_rvc=True)
    backend.results = [_result("mixed", mixed["payload"])]
    body = _post(clientapp, task="auto").json()
    assert body["warnings"] == [] and body["requires_visual_check"] is True


def test_mixed_negative_control_in_response(clientapp, backend):
    mixed = _mixed()
    backend.results = [_result("mixed", mixed["payload"])]
    body = _post(clientapp, task="auto").json()
    assert body["warnings"] == [] and body["requires_visual_check"] is False


# ------------------------------------------------ describe : non-régression
def test_describe_nominal_response_unchanged_by_mixed_fix(clientapp, backend):
    backend.results = [ok_photo_result()]
    body = _post(clientapp).json()
    assert body["warnings"] == [] and body["requires_visual_check"] is False
    assert body["payload"] == json.loads(ok_photo_result().raw_json)["payload"]
    # Avec warnings top-level : toujours dédoublonnés et triés, rien d'ajouté.
    backend.results = [ok_photo_result(warnings=["unreadable_region", "low_resolution",
                                                 "unreadable_region"])]
    body = _post(clientapp).json()
    assert body["warnings"] == ["low_resolution", "unreadable_region"]
    assert body["requires_visual_check"] is True


def test_describe_model_facing_schema_has_no_region_signals():
    # Le schéma `describe` n'expose aucune région : le fix ne peut pas y
    # changer une seule sortie. (L'identité du schéma avec MODEL-1 est
    # vérifiée par tests/test_vision_model1_evidence.py.)
    schema = json.dumps(output_schema_for_task("describe"))
    assert "regions" not in schema and "requires_visual_check" not in schema


# ------------------------------------------------------------------ locale
def _plan(locale):
    return VisionInferencePlan(
        image_bytes=b"\x89PNG", image_mime="image/png", task="describe",
        prompt_profile="vision_describe_v1", prompt_profile_version="1",
        output_json_schema=output_schema_for_task("describe"), locale=locale,
        caption="légende", alt_text=None, nearby_text="texte voisin",
        model_logical="vision-large", model_concrete="mistral-small3.2:24b",
        remaining_timeout_seconds=30.0, system_prompt=SYSTEM_PROMPT,
        task_prompt=DESCRIBE_PROMPT, options=QUALIFIED_FIRST_ATTEMPT)


@pytest.mark.parametrize("other", ["en", "fr"])
def test_locale_is_reserved_and_non_effective_in_v1(other):
    """GEL VOLONTAIRE (lot CV-0). En Vision V1 `locale` est accepté pour la
    compatibilité ascendante mais n'influence RIEN de ce que le modèle
    reçoit : le prompt qualifié par MODEL-1 est français et unique. Toute
    différence fr/en dans ce payload serait un changement perceptif exigeant
    une nouvelle qualification et une nouvelle version de prompt — ce test
    doit alors être modifié EN CONNAISSANCE DE CAUSE, jamais contourné."""
    transport = OllamaVisionBackend("http://127.0.0.1:11434")
    payload_fr = transport.build_payload(_plan("fr"))
    payload_other = transport.build_payload(_plan(other))
    assert payload_fr == payload_other
    assert "locale" not in json.dumps(payload_fr)


def test_locale_still_accepted_by_public_contract(clientapp, backend):
    backend.results = [ok_photo_result(), ok_photo_result()]
    assert _post(clientapp, locale="en").status_code == 200
    assert _post(clientapp, locale="fr").status_code == 200
    assert backend.plans[0].locale == "en" and backend.plans[1].locale == "fr"
