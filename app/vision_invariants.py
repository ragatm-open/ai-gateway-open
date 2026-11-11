"""Invariants déterministes du contrat Vision V1 (étage 4 du pipeline gelé).

Valident ce que Pydantic seul ne peut pas prouver : couverture exacte de
grille, références de graphe, provenance des valeurs de chart, bbox, cohérence
task/visual_kind, sémantique unknown/decorative. Toute violation lève
`VisionInvariantError` — traitée comme sortie modèle invalide (retry interne
unique puis 502 vision_schema_validation_failed), JAMAIS normalisée en succès.

Seule exception, bornée et appliquée AVANT ces invariants (lot
AI-GATEWAY-VISION-ABSTENTION-NORMALIZATION-1A) : `normalize_unambiguous_unknown`
complète le warning obligatoire d'une abstention `describe` sans aucun autre
contenu. Elle ne relâche aucun invariant : la sortie normalisée les traverse
tous, inchangés.
"""

import math
from typing import Any, Dict, List, Optional

from .vision_schemas import (
    ALLOWED_KINDS_BY_TASK,
    PAYLOAD_BY_KIND,
    REGION_PAYLOAD_BY_KIND,
    RawModelOutput,
)


class VisionInvariantError(ValueError):
    """Message SANS contenu (jamais de texte extrait, de cellule, de label)."""


def _fail(message: str) -> None:
    raise VisionInvariantError(message)


# ------------------------------------------------------------------- table
def check_table(payload) -> None:
    covered: Dict[tuple, int] = {}
    for i, cell in enumerate(payload.cells):
        if cell.row + cell.row_span > payload.n_rows or \
                cell.col + cell.col_span > payload.n_cols:
            _fail(f"cellule {i} : span hors grille")
        if cell.state == "filled" and cell.text is None:
            _fail(f"cellule {i} : filled exige un texte")
        if cell.state != "filled" and cell.text is not None:
            _fail(f"cellule {i} : {cell.state} exige text=null")
        if cell.state == "blank" and cell.certainty != "exact":
            _fail(f"cellule {i} : blank exige certainty=exact")
        if cell.state == "unreadable" and cell.certainty != "unreadable":
            _fail(f"cellule {i} : unreadable exige certainty=unreadable")
        for r in range(cell.row, cell.row + cell.row_span):
            for c in range(cell.col, cell.col + cell.col_span):
                if (r, c) in covered:
                    _fail(f"grille : ({r},{c}) couverte deux fois")
                covered[(r, c)] = i
    expected = payload.n_rows * payload.n_cols
    if len(covered) != expected:
        _fail(f"grille : {len(covered)}/{expected} coordonnées couvertes (trou)")


# ----------------------------------------------------------------- diagram
def check_diagram(payload) -> None:
    node_ids = [n.id for n in payload.nodes]
    if len(node_ids) != len(set(node_ids)):
        _fail("diagram : IDs de nœuds non uniques")
    known = set(node_ids)
    group_ids = [g.id for g in payload.groups]
    if len(group_ids) != len(set(group_ids)):
        _fail("diagram : IDs de groupes non uniques")
    for i, e in enumerate(payload.edges):
        if e.source_id not in known:
            _fail(f"diagram : edge {i} source_id inconnu")
        if e.target_id not in known:
            _fail(f"diagram : edge {i} target_id inconnu")
    for g in payload.groups:
        for m in g.members:
            if m not in known:
                _fail(f"diagram : membre de groupe inconnu ({g.id})")


# ------------------------------------------------------------------- chart
def check_chart(payload) -> None:
    axis_ids = [a.id for a in payload.axes]
    if len(axis_ids) != len(set(axis_ids)):
        _fail("chart : IDs d'axes non uniques")
    series_ids = [s.id for s in payload.series]
    if len(series_ids) != len(set(series_ids)):
        _fail("chart : IDs de séries non uniques")
    known_axes = set(axis_ids)
    for a in payload.axes:
        if a.range is not None:
            for bound in (a.range.min, a.range.max):
                if bound is not None and not math.isfinite(bound):
                    _fail("chart : borne d'axe non finie")
    for s in payload.series:
        for ax in (s.axis_ids or []):
            if ax not in known_axes:
                _fail(f"chart : série {s.id} référence un axe inconnu")
        for j, p in enumerate(s.points):
            if p.value is not None and not math.isfinite(p.value):
                _fail(f"chart : valeur non finie (série {s.id}, point {j})")
            if p.precision == "unreadable" and p.value is not None:
                _fail(f"chart : unreadable exige value=null (série {s.id}, point {j})")
            if p.precision == "exact_labeled" and p.value_text is None:
                _fail(f"chart : exact_labeled exige value_text (série {s.id}, point {j})")
            if isinstance(p.x, float) and not math.isfinite(p.x):
                _fail(f"chart : x non fini (série {s.id}, point {j})")


# ------------------------------------------------------- technical drawing
def check_technical_drawing(payload) -> None:
    ids = [e.id for e in payload.elements]
    if len(ids) != len(set(ids)):
        _fail("drawing : IDs d'éléments non uniques")
    known = set(ids)
    for i, d in enumerate(payload.explicit_dimensions):
        for ref in (d.between or []):
            if ref not in known:
                _fail(f"drawing : dimension {i} référence un élément inconnu")
        if d.value is not None and not math.isfinite(d.value):
            _fail(f"drawing : dimension {i} valeur normalisée non finie")
    zone_ids = [z.id for z in payload.zones]
    if len(zone_ids) != len(set(zone_ids)):
        _fail("drawing : IDs de zones non uniques")
    for z in payload.zones:
        for m in z.members:
            if m not in known:
                _fail(f"drawing : membre de zone inconnu ({z.id})")
    for i, r in enumerate(payload.relations):
        if r.source_id not in known or r.target_id not in known:
            _fail(f"drawing : relation {i} référence pendante")
        if r.relation == "other" and not (r.label or "").strip():
            _fail(f"drawing : relation {i} other exige un label")


# -------------------------------------------------------------------- bbox
def check_bbox(bbox: List[float]) -> None:
    x_min, y_min, x_max, y_max = bbox
    if not all(math.isfinite(v) for v in bbox):
        _fail("bbox : coordonnée non finie")
    if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
        _fail("bbox : hors convention [0,1] top-left ou aire nulle")


# ------------------------------------------------------------------- mixed
def check_mixed(payload) -> None:
    for i, region in enumerate(payload.regions):
        check_bbox(region.bbox)
        cls = REGION_PAYLOAD_BY_KIND[region.visual_kind]   # "mixed" absent => récursion impossible par construction
        if cls is None:
            if region.payload is not None:
                _fail(f"mixed : région {i} {region.visual_kind} exige payload=null")
            continue
        if region.payload is None:
            _fail(f"mixed : région {i} {region.visual_kind} exige un payload")
        sub = cls.model_validate(region.payload)   # ValidationError -> schéma invalide
        _CHECKS.get(region.visual_kind, lambda _p: None)(sub)


_CHECKS = {
    "table": check_table,
    "diagram": check_diagram,
    "chart": check_chart,
    "technical_drawing": check_technical_drawing,
}


# ------------------------------------------------ abstention non ambiguë
# Lot AI-GATEWAY-VISION-ABSTENTION-NORMALIZATION-1A. Le schéma model-facing
# (`UnknownResultOut`, gelé : il entre dans le fingerprint) admet `unknown`
# avec `warnings: []`, que l'invariant ci-dessous rejette — une abstention
# légitime finissait en 502 après retry (B-005, page blanche).
# Périmètre : la seule task qualifiée, où le défaut a été observé.
ABSTENTION_NORMALIZATION_TASKS = frozenset({"describe"})
ABSTENTION_NORMALIZATION_REASON = "unknown_missing_required_warning"


def normalize_unambiguous_unknown(task: str, raw: RawModelOutput):
    """Fonction PURE, appelée après Pydantic et avant `validate_semantic_output`.

    Rend `(raw, None)` inchangé, ou `(copie, raison)` lorsque la sortie est une
    abstention NON AMBIGUË : task qualifiée, `unknown`, payload absent/null,
    AUCUN warning, `summary` absent ou blanc. Seul ajout :
    `insufficient_visual_evidence`. Jamais d'effacement : un payload, un autre
    warning ou un résumé non vide (affirmations possibles, non qualifiables
    de façon déterministe) laissent la sortie à l'invariant historique.
    """
    if task not in ABSTENTION_NORMALIZATION_TASKS:
        return raw, None
    if raw.visual_kind != "unknown" or raw.payload is not None or raw.warnings:
        return raw, None
    if raw.summary is not None and raw.summary.strip():
        return raw, None
    normalized = raw.model_copy(update={"warnings": ["insufficient_visual_evidence"]})
    return normalized, ABSTENTION_NORMALIZATION_REASON


# ------------------------------------------------------------ sortie brute
def validate_semantic_output(task: str, raw: RawModelOutput):
    """Étages 3-4 : valide la sortie sémantique et RETOURNE le payload typé.

    Rend (payload_model|None). Lève ValidationError (schéma) ou
    VisionInvariantError (invariant).
    """
    kind = raw.visual_kind
    if kind not in ALLOWED_KINDS_BY_TASK[task]:
        _fail(f"visual_kind {kind} non admis pour task {task}")

    if kind == "unknown":
        if raw.payload is not None:
            _fail("unknown exige payload=null")
        if "insufficient_visual_evidence" not in raw.warnings:
            _fail("unknown exige le warning insufficient_visual_evidence")
        return None
    if kind == "decorative":
        if raw.payload is not None:
            _fail("decorative exige payload=null")
        if raw.warnings:
            _fail("decorative exige warnings=[]")
        return None

    cls = PAYLOAD_BY_KIND[kind]
    if raw.payload is None:
        _fail(f"{kind} exige un payload")
    payload = cls.model_validate(raw.payload)
    if kind == "mixed":
        check_mixed(payload)
    else:
        _CHECKS.get(kind, lambda _p: None)(payload)
    return payload


# ------------------------------------------------- requires_visual_check
def _scan(value: Any) -> bool:
    if isinstance(value, dict):
        for key, v in value.items():
            if key == "certainty" and v in ("ambiguous", "unreadable"):
                return True
            # `precision` est le vocabulaire d'incertitude des points de chart
            # (§79 du contrat) : unreadable y déclenche aussi la vérification.
            if key == "precision" and v == "unreadable":
                return True
            if key == "direction" and v == "unknown":
                return True
            # Lot CV-0 : signaux portés par une RÉGION `mixed`. Un warning de
            # région ou un `requires_visual_check` de région est un signal de
            # revue au même titre qu'un warning top-level ; avant ce lot ils
            # étaient parcourus sans effet (région `low_resolution` avec
            # certainty exact => False, à tort).
            if key == "warnings" and isinstance(v, list) and v:
                return True
            if key == "requires_visual_check" and v is True:
                return True
            if _scan(v):
                return True
    elif isinstance(value, list):
        return any(_scan(v) for v in value)
    return False


def derive_requires_visual_check(visual_kind: str, payload, warnings: List[str]) -> bool:
    """Dérivation DÉTERMINISTE côté Gateway (contrat §70-71) — le booléen du
    modèle n'est jamais cru sur parole. Déclencheurs gelés : kind unknown,
    tout warning (top-level ou de région), tout certainty
    ambiguous/unreadable, direction=unknown, requires_visual_check de région."""
    if visual_kind == "unknown":
        return True
    if warnings:
        return True
    if payload is None:
        return False
    return _scan(payload.model_dump())


# ------------------------------------------------------ warnings agrégés
def _nested_warnings(value: Any) -> List[str]:
    """Warnings portés par les sous-structures (régions `mixed`), dans l'ordre
    du document : dictionnaires parcourus dans l'ordre d'insertion, listes
    dans l'ordre des éléments. Aucune clé `warnings` n'existe ailleurs dans
    les payloads V1 ; le parcours est générique pour ne pas dépendre de la
    forme exacte de `MixedPayload`."""
    found: List[str] = []
    if isinstance(value, dict):
        for key, v in value.items():
            if key == "warnings" and isinstance(v, list):
                found.extend(w for w in v if isinstance(w, str))
            else:
                found.extend(_nested_warnings(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(_nested_warnings(v))
    return found


def collect_warnings(top_level: List[str], payload) -> List[str]:
    """Warnings de la réponse (lot CV-0) : les warnings top-level, DANS L'ORDRE
    REÇU, puis ceux des régions dans l'ordre du document, en ignorant toute
    occurrence déjà présente. Déterministe, sans réordonnancement ; aucune
    chaîne n'est inventée ni renommée. Avant ce lot les warnings de région
    étaient perdus (seuls les top-level sortaient)."""
    out: List[str] = []
    seen = set()
    nested = _nested_warnings(payload.model_dump()) if payload is not None else []
    for w in list(top_level) + nested:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out
