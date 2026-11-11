"""Schémas du contrat Vision V1 (lot vision-1a).

SOURCE AUTORITAIRE : AI-GATEWAY-VISION-CONTRACT-0 (gelé). Aucun renommage,
aucun champ ajouté/retiré sans nouvelle schema_version. Les modèles de
REQUÊTE sont fail-closed (`extra="forbid"`) ; les payloads de RÉPONSE sont
également stricts : une clé inconnue produite par le modèle est une sortie
invalide (502), jamais un champ ignoré en silence.

La validation ici est STRUCTURELLE (types, enums, bornes locales). Les
invariants inter-champs (couverture de grille, références de graphe, bbox…)
vivent dans `vision_invariants.py` — étage 4 du pipeline gelé.
"""

from functools import lru_cache
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from .settings import settings
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

SCHEMA_VERSION = "1"
# Versions du contrat PUBLIC (requête/réponse HTTP). "2" (lot CV-1) = même
# perception + provenance technique ; "1" reste servi strictement inchangé.
# Distinct du schéma MODEL-FACING (`output_schema_for_task`), identique pour
# les deux versions — son identité est `pipeline.model_schema_sha256`.
SCHEMA_VERSIONS = ("1", "2")
# Vocabulaire de la REQUÊTE publique (lot SEMX-4A) : "3" = contrat sémantique
# `semantic_extract`, apparié EXCLUSIVEMENT à cette task (contrôlé en tête de
# route). Représentable n'est pas servable : flag, registre et prompt qualifié
# restent des verrous indépendants. SCHEMA_VERSIONS (V1/V2) est inchangé.
PUBLIC_SCHEMA_VERSIONS = ("1", "2", "3")
SEMANTIC_TASK = "semantic_extract"
SEMANTIC_SCHEMA_VERSION = "3"

# Enums gelés — ne JAMAIS étendre sans nouvelle schema_version.
VISION_TASKS = ("auto", "describe", "table", "diagram", "chart", "technical_drawing")
VISUAL_KINDS = ("photo", "table", "diagram", "chart", "technical_drawing",
                "mixed", "decorative", "unknown")
REGION_KINDS = ("photo", "table", "diagram", "chart", "technical_drawing",
                "text", "decorative", "unknown")  # jamais "mixed" (récursion interdite)
CERTAINTIES = ("exact", "approximate", "ambiguous", "unreadable")
WARNING_CODES = ("low_resolution", "unreadable_region", "ambiguous_structure",
                 "incomplete_axes", "insufficient_visual_evidence")

VisionTask = Literal["auto", "describe", "table", "diagram", "chart", "technical_drawing"]
# Type PUBLIC de la requête (lot SEMX-4A) : les six tasks V1 + semantic_extract.
# `VisionTask` reste le type INTERNE du rail V1/V2 (réponses, MODEL-1, schémas
# model-facing) et n'est jamais élargi : aucun schéma ni fingerprint describe
# ne dépend de ce type public.
PUBLIC_VISION_TASKS = VISION_TASKS + (SEMANTIC_TASK,)
PublicVisionTask = Literal["auto", "describe", "table", "diagram", "chart", "technical_drawing",
                           "semantic_extract"]
VisualKind = Literal["photo", "table", "diagram", "chart", "technical_drawing",
                     "mixed", "decorative", "unknown"]
RegionKind = Literal["photo", "table", "diagram", "chart", "technical_drawing",
                     "text", "decorative", "unknown"]
Certainty = Literal["exact", "approximate", "ambiguous", "unreadable"]
WarningCode = Literal["low_resolution", "unreadable_region", "ambiguous_structure",
                      "incomplete_axes", "insufficient_visual_evidence"]

# Mapping gelé task -> profil de prompt (CONTRACT-0 § N).
PROMPT_PROFILES = {
    "auto": ("vision_auto_v1", "1"),
    "describe": ("vision_describe_v1", "1"),
    "table": ("vision_table_v1", "1"),
    "diagram": ("vision_diagram_v1", "1"),
    "chart": ("vision_chart_v1", "1"),
    "technical_drawing": ("vision_technical_drawing_v1", "1"),
}

# Matrice gelée task -> visual_kind admis en sortie (CONTRACT-0 §15-§18) :
# `auto` peut produire tous les genres ; une task explicite produit SON genre
# ou `unknown` (le genre demandé n'est jamais forcé ; mixed/decorative sont
# détectés par auto uniquement).
ALLOWED_KINDS_BY_TASK = {
    "auto": set(VISUAL_KINDS),
    "describe": {"photo", "unknown"},
    "table": {"table", "unknown"},
    "diagram": {"diagram", "unknown"},
    "chart": {"chart", "unknown"},
    "technical_drawing": {"technical_drawing", "unknown"},
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Requête
# --------------------------------------------------------------------------- #
# Lot P7 — syntaxe Base64 stricte (alphabet standard, longueur multiple de
# 4, padding `=` terminal uniquement) vérifiée par expression régulière : un
# balayage linéaire SANS allocation du résultat, là où `b64decode` décodait
# toute la chaîne pour la jeter. Équivalent à `b64decode(validate=True)` pour
# l'admission ; le décodage lui-même n'a lieu qu'UNE fois, dans la route.
_BASE64_STRICT_RE = re.compile(
    r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
)


class VisionImage(_Strict):
    # `max_length` : borne dérivée de la taille décodée maximale
    # (4 Mio -> 5 592 408 caractères), appliquée par Pydantic AVANT le
    # validateur — une chaîne trop longue est refusée sans être décodée.
    data: str = Field(..., min_length=1, max_length=settings.vision_max_image_base64_chars)
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]

    @field_validator("data")
    @classmethod
    def data_is_base64(cls, v: str) -> str:
        if not _BASE64_STRICT_RE.match(v) or len(v) % 4 != 0:
            raise ValueError("base64 invalide")
        return v


class VisionContext(_Strict):
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    nearby_text: Optional[str] = None


class VisionAnalyzeRequest(_Strict):
    schema_version: Literal["1", "2", "3"]
    task: PublicVisionTask
    image: VisionImage
    model: Optional[str] = None
    locale: Literal["fr", "en"] = "fr"
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128)

    @field_validator("model")
    @classmethod
    def _model_name(cls, v: Optional[str]) -> Optional[str]:
        from .schemas import validate_model_name   # même alphabet que les autres routes (P13)
        return validate_model_name(v)
    context: Optional[VisionContext] = None
    # PAS de options/temperature/seed : réglages d'inférence = profils Gateway.
    # PAS d'URL ni de chemin : l'appelant fournit les octets (frontière SSRF).


# --------------------------------------------------------------------------- #
# Payloads — communs
# --------------------------------------------------------------------------- #
class VisibleText(_Strict):
    text: str = Field(..., min_length=1)
    certainty: Certainty


# ------------------------------------------------------------------- photo
class PhotoObject(_Strict):
    label: str = Field(..., min_length=1)
    count: Optional[int] = Field(default=None, ge=0)
    attributes: List[str] = Field(default_factory=list)
    certainty: Certainty


class PhotoRelation(_Strict):
    subject: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    certainty: Certainty


class PhotoPayload(_Strict):
    description: str = Field(..., min_length=1)
    setting: Optional[str] = None
    objects: List[PhotoObject] = Field(default_factory=list)
    observable_relations: List[PhotoRelation] = Field(default_factory=list)
    visible_text: List[VisibleText] = Field(default_factory=list)


# ------------------------------------------------------------------- table
class TableCell(_Strict):
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    row_span: int = Field(..., ge=1)
    col_span: int = Field(..., ge=1)
    role: Literal["column_header", "row_header", "data", "corner", "note", "unknown"]
    state: Literal["filled", "blank", "unreadable"]
    text: Optional[str] = None
    certainty: Certainty


class TablePayload(_Strict):
    n_rows: int = Field(..., gt=0)
    n_cols: int = Field(..., gt=0)
    caption: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    cells: List[TableCell] = Field(..., min_length=1)


# ----------------------------------------------------------------- diagram
class DiagramNode(_Strict):
    id: str = Field(..., min_length=1)
    label: Optional[str] = None
    type: Optional[str] = None
    certainty: Certainty


class DiagramEdge(_Strict):
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    # `backward` volontairement ABSENT (contrat §47) : une flèche B->A s'émet
    # source_id=B, target_id=A, direction=forward — représentation canonique.
    direction: Literal["forward", "bidirectional", "undirected", "unknown"]
    relation: Optional[str] = None
    label: Optional[str] = None
    certainty: Certainty


class DiagramGroup(_Strict):
    id: str = Field(..., min_length=1)
    label: Optional[str] = None
    members: List[str] = Field(default_factory=list)


class DiagramAnnotation(_Strict):
    text: str = Field(..., min_length=1)
    certainty: Certainty


class DiagramPayload(_Strict):
    diagram_type: Optional[str] = None
    nodes: List[DiagramNode] = Field(..., min_length=1)
    edges: List[DiagramEdge] = Field(default_factory=list)
    groups: List[DiagramGroup] = Field(default_factory=list)
    annotations: List[DiagramAnnotation] = Field(default_factory=list)


# ------------------------------------------------------------------- chart
class ChartAxisRange(_Strict):
    min: Optional[float] = None
    max: Optional[float] = None
    certainty: Certainty


class ChartAxis(_Strict):
    id: str = Field(..., min_length=1)
    orientation: Literal["x", "y", "other"]
    label: Optional[str] = None
    unit: Optional[str] = None
    scale: Literal["linear", "log", "categorical", "temporal", "unknown"]
    range: Optional[ChartAxisRange] = None
    certainty: Certainty


class ChartPoint(_Strict):
    label: Optional[str] = None
    x: Union[float, str, None] = None
    value: Optional[float] = None
    value_text: Optional[str] = None
    precision: Literal["exact_labeled", "approximate_visual", "unreadable"]


class ChartSeries(_Strict):
    id: str = Field(..., min_length=1)
    label: Optional[str] = None
    axis_ids: Optional[List[str]] = None
    points: List[ChartPoint] = Field(..., min_length=1)


class ChartPayload(_Strict):
    chart_type: Literal["bar", "line", "pie", "scatter", "area",
                        "stacked_bar", "other", "unknown"]
    axes: List[ChartAxis] = Field(default_factory=list)
    series: List[ChartSeries] = Field(..., min_length=1)
    trend_summary: Optional[str] = None


# ------------------------------------------------------- technical drawing
class DrawingElement(_Strict):
    id: str = Field(..., min_length=1)
    kind: str = Field(..., min_length=1)   # lecture visuelle, PAS une classe CAD
    label: Optional[str] = None
    certainty: Certainty


class ExplicitDimension(_Strict):
    value_text: str = Field(..., min_length=1)   # verbatim OBLIGATOIRE
    unit: Optional[str] = None
    value: Optional[float] = None                # normalisation optionnelle
    between: Optional[List[str]] = None
    certainty: Certainty


class DrawingSymbol(_Strict):
    label: str = Field(..., min_length=1)
    meaning: Optional[str] = None
    certainty: Certainty


class DrawingZone(_Strict):
    id: str = Field(..., min_length=1)
    label: Optional[str] = None
    members: List[str] = Field(default_factory=list)


class DrawingRelation(_Strict):
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    relation: Literal["connected_to", "inside", "adjacent_to", "above", "below",
                      "left_of", "right_of", "feeds", "enters", "exits", "other"]
    label: Optional[str] = None
    certainty: Certainty


class DrawingLegendItem(_Strict):
    key: str = Field(..., min_length=1)
    meaning: Optional[str] = None
    certainty: Certainty


class TechnicalDrawingPayload(_Strict):
    drawing_type: Optional[str] = None
    elements: List[DrawingElement] = Field(..., min_length=1)
    explicit_dimensions: List[ExplicitDimension] = Field(default_factory=list)
    symbols: List[DrawingSymbol] = Field(default_factory=list)
    zones: List[DrawingZone] = Field(default_factory=list)
    relations: List[DrawingRelation] = Field(default_factory=list)
    legend: List[DrawingLegendItem] = Field(default_factory=list)
    annotations: List[DiagramAnnotation] = Field(default_factory=list)


# ------------------------------------------------------------------- mixed
class TextRegionPayload(_Strict):
    lines: List[VisibleText] = Field(..., min_length=1)


class MixedRegion(_Strict):
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    visual_kind: RegionKind
    payload: Optional[Dict[str, Any]] = None   # re-validé par kind (invariants)
    warnings: List[WarningCode] = Field(default_factory=list)
    requires_visual_check: bool = False


class MixedPayload(_Strict):
    regions: List[MixedRegion] = Field(..., min_length=2)


# Union discriminée gelée (CONTRACT-0 §144) : visual_kind -> classe payload.
# decorative/unknown -> payload null (imposé par les invariants).
PAYLOAD_BY_KIND = {
    "photo": PhotoPayload,
    "table": TablePayload,
    "diagram": DiagramPayload,
    "chart": ChartPayload,
    "technical_drawing": TechnicalDrawingPayload,
    "mixed": MixedPayload,
    "decorative": None,
    "unknown": None,
}
REGION_PAYLOAD_BY_KIND = {
    "photo": PhotoPayload,
    "table": TablePayload,
    "diagram": DiagramPayload,
    "chart": ChartPayload,
    "technical_drawing": TechnicalDrawingPayload,
    "text": TextRegionPayload,
    "decorative": None,
    "unknown": None,
}


# --------------------------------------------------------------------------- #
# Sortie sémantique BRUTE du modèle (étage 3 du pipeline) — ce que le backend
# (mock en 1A, Ollama en 1B) doit produire sous contrainte de schéma.
# --------------------------------------------------------------------------- #
class RawModelOutput(_Strict):
    visual_kind: VisualKind
    summary: Optional[str] = None
    warnings: List[WarningCode] = Field(default_factory=list)
    payload: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Réponse V1 — schéma FERMÉ (aucun champ non déclaré ne sort).
# --------------------------------------------------------------------------- #
class VisionModelIdentity(_Strict):
    logical: str
    concrete: str
    digest: str            # digest gouverné ou sentinelle "not_reported"


class VisionPipelineIdentity(_Strict):
    prompt_profile: str
    prompt_profile_version: str
    pipeline_id: str


class VisionImageDiagnostics(_Strict):
    width: int
    height: int
    pixels: int
    bytes: int
    format: Literal["png", "jpeg", "webp"]


class VisionUsage(_Strict):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class VisionAnalyzeResponse(_Strict):
    status: Literal["ok"] = "ok"
    provider: Literal["ai_gateway"] = "ai_gateway"
    upstream_provider: Literal["ollama"] = "ollama"
    schema_version: Literal["1"] = "1"
    task: VisionTask
    visual_kind: VisualKind
    request_id: Optional[str] = None
    summary: Optional[str] = None
    warnings: List[WarningCode] = Field(default_factory=list)
    requires_visual_check: bool
    payload: Optional[Dict[str, Any]] = None
    model: VisionModelIdentity
    pipeline: VisionPipelineIdentity
    image: VisionImageDiagnostics
    usage: VisionUsage
    latency_ms: int
    truncated: Literal[False] = False   # V1 : toute troncature est une ERREUR


# --------------------------------------------------------------------------- #
# Réponse V2 (lot CV-1) — V1 + provenance TECHNIQUE, schéma FERMÉ lui aussi.
#
# WHAT (image.*_sha256) et HOW (pipeline.fingerprint) sont deux champs
# distincts : un client raisonne « même image / autre pipeline » et « autre
# image / même pipeline » sans décomposer un hash opaque. Aucune provenance
# documentaire (document, page, crop) : elle appartient à l'appelant.
# --------------------------------------------------------------------------- #
_SHA256_RE = r"^[0-9a-f]{64}$"


class VisionInferenceIdentity(_Strict):
    temperature: float
    seed: int
    num_ctx: int
    num_predict: int


class VisionRetryIdentity(_Strict):
    max_attempts: int
    trigger: str
    temperature: float
    seed: int


class VisionPipelineIdentityV2(VisionPipelineIdentity):
    fingerprint: str = Field(..., pattern=r"^vision-v1:[0-9a-f]{64}$")
    system_prompt_sha256: str = Field(..., pattern=_SHA256_RE)
    task_prompt_sha256: str = Field(..., pattern=_SHA256_RE)
    model_schema_sha256: str = Field(..., pattern=_SHA256_RE)
    model_schema_hash_algorithm: str
    inference: VisionInferenceIdentity
    retry: VisionRetryIdentity


class VisionImageDiagnosticsV2(VisionImageDiagnostics):
    source_sha256: str = Field(..., pattern=_SHA256_RE)
    normalized_sha256: str = Field(..., pattern=_SHA256_RE)
    normalized_format: Literal["png"]
    normalizer_id: str


class VisionUsageV2(VisionUsage):
    attempts: Literal[1, 2]   # appels backend réellement consommés (diagnostic)


class VisionAnalyzeResponseV2(VisionAnalyzeResponse):
    schema_version: Literal["2"] = "2"
    pipeline: VisionPipelineIdentityV2
    image: VisionImageDiagnosticsV2
    usage: VisionUsageV2


# --------------------------------------------------------------------------- #
# Sonde de capacité (lot CV-CAPACITY) — `GET /v1/vision/capacity`, schéma
# FERMÉ, contrat `contract_version "1"` INDÉPENDANT du `schema_version`
# perceptif. ADVISORY : rien n'est réservé ; le POST reste la seule autorité
# d'admission et un 503 après une sonde « libre » est un comportement valide.
# --------------------------------------------------------------------------- #
class _StrictTyped(BaseModel):
    """Fermé ET typé strictement (aucune coercition "0" -> 0) : un client qui
    consomme la sonde ne doit jamais lire une valeur reinterprétée."""
    model_config = ConfigDict(extra="forbid", strict=True)


class VisionCapacity(_StrictTyped):
    enabled: bool
    # disabled = flag OFF ; available = au moins un slot libre ; saturated =
    # aucun slot libre. Pas de `degraded` : aucune source objective en V1.
    status: Literal["available", "saturated", "disabled"]
    max_concurrency: int = Field(..., ge=1)
    in_flight: int = Field(..., ge=0)
    available_slots: int = Field(..., ge=0)
    # Même constante que l'en-tête Retry-After du 503 : « attends au moins
    # cette durée avant de re-sonder » — ni une ETA ni une promesse.
    retry_after_seconds: int = Field(..., ge=1)


class VisionCapacityModel(_StrictTyped):
    """Diagnostic de résidence (lecture `/api/ps`, jamais un chargement).
    `resident` null et `status` unavailable quand Ollama ne répond pas dans
    le délai court : la capacité d'admission reste servie."""

    logical: str
    resident: Optional[bool]
    status: Literal["ok", "unavailable"]


class VisionCapacityResponse(_StrictTyped):
    provider: Literal["ai_gateway"] = "ai_gateway"
    contract_version: Literal["1"] = "1"
    scope: Literal["instance"] = "instance"   # jamais "host" : GPU/Ollama partagés non couverts
    environment: str
    advisory: Literal[True] = True
    vision: VisionCapacity
    model: VisionCapacityModel


# --------------------------------------------------------------------------- #
# Schéma MODEL-FACING (lot vision-1a-r1) — union discriminée par `visual_kind`.
#
# Ce que le modèle REÇOIT comme contrainte de sortie structurée, distinct de
# ce que la Gateway VALIDE ensuite (RawModelOutput + payload + invariants :
# défense en profondeur, cf. étages 2-4). La reconnaissance GPU du 2026-09-15
# a prouvé la nécessité : avec un `payload` libre, le modèle a émis
# `certainty: "high"` (hors enum V1) et la réponse a dû être rejetée en 502 ;
# avec le payload du genre imbriqué, le MÊME modèle produit du V1 valide.
#
# La relation kind -> payload est donc portée STRUCTURELLEMENT : un
# `visual_kind: "table"` ne peut pas valider avec un PhotoPayload ni avec un
# objet arbitraire.
# --------------------------------------------------------------------------- #
class _RegionOutBase(_Strict):
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    warnings: List[WarningCode] = Field(default_factory=list)
    requires_visual_check: bool = False


class PhotoRegionOut(_RegionOutBase):
    visual_kind: Literal["photo"]
    payload: PhotoPayload


class TableRegionOut(_RegionOutBase):
    visual_kind: Literal["table"]
    payload: TablePayload


class DiagramRegionOut(_RegionOutBase):
    visual_kind: Literal["diagram"]
    payload: DiagramPayload


class ChartRegionOut(_RegionOutBase):
    visual_kind: Literal["chart"]
    payload: ChartPayload


class TechnicalDrawingRegionOut(_RegionOutBase):
    visual_kind: Literal["technical_drawing"]
    payload: TechnicalDrawingPayload


class TextRegionOut(_RegionOutBase):
    visual_kind: Literal["text"]
    payload: TextRegionPayload


class DecorativeRegionOut(_RegionOutBase):
    visual_kind: Literal["decorative"]
    payload: None = None


class UnknownRegionOut(_RegionOutBase):
    visual_kind: Literal["unknown"]
    payload: None = None


# `mixed` est ABSENT de l'union des régions : la récursion interdite par le
# contrat devient structurellement impossible, pas seulement rejetée après coup.
RegionOut = Annotated[
    Union[PhotoRegionOut, TableRegionOut, DiagramRegionOut, ChartRegionOut,
          TechnicalDrawingRegionOut, TextRegionOut, DecorativeRegionOut,
          UnknownRegionOut],
    Field(discriminator="visual_kind"),
]


class MixedPayloadOut(_Strict):
    regions: List[RegionOut] = Field(..., min_length=2)


class _ResultOutBase(_Strict):
    summary: Optional[str] = None
    warnings: List[WarningCode] = Field(default_factory=list)


class PhotoResultOut(_ResultOutBase):
    visual_kind: Literal["photo"]
    payload: PhotoPayload


class TableResultOut(_ResultOutBase):
    visual_kind: Literal["table"]
    payload: TablePayload


class DiagramResultOut(_ResultOutBase):
    visual_kind: Literal["diagram"]
    payload: DiagramPayload


class ChartResultOut(_ResultOutBase):
    visual_kind: Literal["chart"]
    payload: ChartPayload


class TechnicalDrawingResultOut(_ResultOutBase):
    visual_kind: Literal["technical_drawing"]
    payload: TechnicalDrawingPayload


class MixedResultOut(_ResultOutBase):
    visual_kind: Literal["mixed"]
    payload: MixedPayloadOut


class DecorativeResultOut(_ResultOutBase):
    visual_kind: Literal["decorative"]
    payload: None = None


class UnknownResultOut(_ResultOutBase):
    visual_kind: Literal["unknown"]
    payload: None = None


RESULT_MODEL_BY_KIND = {
    "photo": PhotoResultOut,
    "table": TableResultOut,
    "diagram": DiagramResultOut,
    "chart": ChartResultOut,
    "technical_drawing": TechnicalDrawingResultOut,
    "mixed": MixedResultOut,
    "decorative": DecorativeResultOut,
    "unknown": UnknownResultOut,
}


@lru_cache(maxsize=None)
def vision_result_adapter(task: str) -> TypeAdapter:
    """Adaptateur de l'union discriminée admise pour cette task.

    Les genres proviennent de l'UNIQUE source gelée `ALLOWED_KINDS_BY_TASK` :
    R1 ne les élargit pas pour simplifier la génération du schéma.
    """
    kinds = sorted(ALLOWED_KINDS_BY_TASK[task])
    models = tuple(RESULT_MODEL_BY_KIND[kind] for kind in kinds)
    if len(models) == 1:
        return TypeAdapter(models[0])
    return TypeAdapter(
        Annotated[Union[models], Field(discriminator="visual_kind")]
    )


def output_schema_for_task(task: str) -> Dict[str, Any]:
    """JSON Schema envoyé au backend comme contrainte de sortie structurée.

    Contraint la FORME du payload selon le genre, pas seulement `visual_kind`
    (défaut corrigé au lot r1). Ne remplace jamais la validation postérieure.
    """
    return vision_result_adapter(task).json_schema()
