"""Modèles Pydantic request/response.

Validation stricte des payloads entrants ; refus explicite des cas invalides
(liste vide, textes non string, prompt vide).
"""

import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

# Nom de modèle demandé par un client. Alphabet des noms Ollama et
# HuggingFace réellement en usage — `qwen3:30b-instruct`, `nomic-embed-text:latest`,
# `BAAI/bge-reranker-v2-m3`, noms logiques `fr-en-retrieval` — et rien d'autre :
# ni espace, ni caractère de contrôle. Un `\n` dans `model` fabriquait une
# fausse ligne `event=…` dans les logs (démontré). Refusé AVANT tout backend.
# Vérifié contre tous les noms du registre, des tests et de la documentation.
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
MODEL_NAME_MAX_CHARS = 256


def validate_model_name(value: Optional[str]) -> Optional[str]:
    """Champ `model` d'une requête : absent/vide = défaut historique ; sinon
    strictement l'alphabet ci-dessus. La chaîne fautive n'est PAS reproduite
    dans l'erreur : elle est précisément ce qu'on refuse de voir circuler."""
    if value is None or value == "":
        return value
    if len(value) > MODEL_NAME_MAX_CHARS or not MODEL_NAME_RE.match(value):
        raise ValueError(
            "nom de modèle invalide : lettres, chiffres, '.', '_', ':', '/', '-' "
            f"uniquement, {MODEL_NAME_MAX_CHARS} caractères max"
        )
    return value

# Durée de rétention d'un modèle en VRAM côté Ollama. Accepte une durée
# ("5m", "30s"), un nombre de secondes, 0 (décharge immédiatement) ou -1
# (garde indéfiniment). Champ top-level Ollama, hors `options`.
KeepAlive = Union[str, int, float]


# --------------------------------------------------------------------------- #
# Génération
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    # Fail-closed (lot vision-1a, contrat CONTRACT-0 §5) : un champ inconnu —
    # notamment `images` — est refusé en 400 au lieu d'être ignoré en silence.
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)
    model: Optional[str] = None
    temperature: Optional[float] = 0.2
    system: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


    @field_validator("model")
    @classmethod
    def _model_name(cls, v: Optional[str]) -> Optional[str]:
        return validate_model_name(v)
    # Contrôle du mode raisonnement des modèles hybrides (ex. qwen3).
    # think=False force une réponse directe sans bloc <think>. None => défaut
    # du modèle. C'est un champ top-level Ollama (hors `options`).
    think: Optional[bool] = None
    # Rétention du modèle en VRAM. Utile en campagne de comparaison : passer 0
    # libère la VRAM dès la fin de la requête au lieu d'attendre 5 minutes,
    # ce qui raccourcit d'autant l'éviction du modèle de production.
    keep_alive: Optional[KeepAlive] = None
    # Sortie structurée Ollama (lot runtime-hardening). Champ top-level Ollama :
    # "json" force un objet JSON valide ; un dict est un JSON Schema, relayé
    # STRICTEMENT tel quel — la Gateway ne transforme jamais un schéma en
    # silence. Toute autre chaîne que "json" est refusée en 400.
    # RELAIS BRUT HISTORIQUE : ni validation du schéma, ni revalidation de
    # la sortie — pour cela, voir `response_format` ci-dessous.
    format: Optional[Union[Literal["json"], Dict[str, Any]]] = None
    # Sortie structurée GOUVERNÉE sur le rail completion (lot G2) : même
    # contrat que /v1/chat (`{type: "json_schema", schema}`), schéma validé
    # AVANT Ollama, relayé dans `format` de /api/generate, sortie analysée
    # strictement et revalidée APRÈS. Exige la capability `structured_output`.
    # Exclusif de `format` (400 `generate_format_conflict` si les deux).
    response_format: Optional["ChatResponseFormat"] = None

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("prompt ne doit pas être vide")
        return v


class TokenCounts(BaseModel):
    prompt: int
    completion: int
    total: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    text: str
    provider: str = "ai_gateway"
    upstream_provider: str = "ollama"
    model: str
    tokens: TokenCounts
    usage: Usage
    cost: float = 0.0
    done: bool = True
    # Raison d'arrêt relayée VERBATIM d'Ollama (`done_reason` : "stop" = fin
    # naturelle, "length" = plafond num_predict atteint). None si l'upstream ne
    # la fournit pas : on ne déduit jamais une raison de completion_tokens.
    done_reason: Optional[str] = None
    # Lot G2 — présents UNIQUEMENT en mode structuré (`response_format`).
    # Sur un appel historique ils sont RETIRÉS de la sérialisation (voir
    # `_drop_structured_fields_when_absent`) : le corps de réponse legacy
    # reste identique octet pour octet.
    structured_output: Optional[Dict[str, Any]] = None
    response_schema_sha256: Optional[str] = None

    @model_serializer(mode="wrap")
    def _drop_structured_fields_when_absent(self, handler):
        data = handler(self)
        if self.response_schema_sha256 is None and isinstance(data, dict):
            data.pop("structured_output", None)
            data.pop("response_schema_sha256", None)
        return data


# --------------------------------------------------------------------------- #
# Génération en flux (NDJSON) — POST /v1/generate/stream
# --------------------------------------------------------------------------- #
# Ces modèles ne servent PAS de `response_model` : un flux NDJSON n'est pas une
# réponse JSON unique. Ils existent pour que les lignes émises soient
# construites, documentées et testées à partir d'une définition unique, plutôt
# qu'assemblées à la main à trois endroits du code.
#
# Le contrat est volontairement minimal : le client reçoit du texte et, à la
# fin, ses comptes de tokens. Les métriques de performance détaillées restent
# dans les logs — elles concernent l'exploitant, pas l'appelant.


class GenerateStreamDelta(BaseModel):
    """Fragment de texte. Émis uniquement si le fragment est non vide."""

    type: Literal["delta"] = "delta"
    delta: str
    done: Literal[False] = False


class GenerateStreamDone(BaseModel):
    """Événement final d'un flux abouti. Exactement un par flux réussi.

    `tokens` et `usage` portent la même sémantique qu'en non-stream, aux mêmes
    valeurs : un compte absent vaut 0.
    """

    type: Literal["done"] = "done"
    delta: Literal[""] = ""
    done: Literal[True] = True
    model: str
    tokens: TokenCounts
    usage: Usage
    # Même sémantique qu'en non-stream : valeur Ollama verbatim ou None.
    done_reason: Optional[str] = None


class StreamErrorDetail(BaseModel):
    code: str
    message: str


class GenerateStreamError(BaseModel):
    """Rupture survenue APRÈS l'envoi des en-têtes HTTP.

    Une panne détectée avant le premier octet reste un code HTTP classique
    (502) : cet événement ne concerne que ce qui casse en cours de flux, quand
    le statut de la réponse est déjà parti. Un flux en erreur ne produit jamais
    d'événement `done`.
    """

    type: Literal["error"] = "error"
    error: StreamErrorDetail


# --------------------------------------------------------------------------- #
# Chat — POST /v1/chat et /v1/chat/stream (lot C1)
# --------------------------------------------------------------------------- #
# Contrat ADDITIF, séparé de `/v1/generate` : transport `/api/chat`, messages
# multi-tours, `thinking` distinct du `content`, `done_reason` et
# `content_empty` explicites. Volontairement ABSENTS du contrat C1 (fail-closed
# par `extra="forbid"`) : `tools`, `format`, `think` brut, `stream`, rôle
# `tool`. Ils arrivent par leurs propres lots (C3 structured output, C4 tools).
CHAT_MESSAGE_ROLES = ("system", "user", "assistant")


class ChatMessage(BaseModel):
    # Fail-closed : `images`, `tool_calls`, `name`… sont refusés en 400. Seuls
    # `role` et `content` existent, et seuls eux sont relayés à Ollama.
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str

    @model_validator(mode="after")
    def _content_policy(self) -> "ChatMessage":
        # Un tour `system` ou `user` vide n'a pas de sens et tromperait le
        # modèle ; un tour `assistant` vide est légitime — c'est exactement ce
        # qu'un client rejoue après une réponse `content_empty: true`.
        if self.role != "assistant" and not self.content.strip():
            raise ValueError(f"le contenu d'un message {self.role} ne doit pas être vide")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # OBLIGATOIRE, contrairement à `/v1/generate` : `/v1/chat` n'a pas de
    # défaut d'environnement. La capability `chat` est opt-in par modèle ;
    # un défaut implicite recréerait un chemin non gouverné.
    model: str = Field(..., min_length=1)
    messages: List[ChatMessage] = Field(..., min_length=1)
    temperature: Optional[float] = 0.2
    # `num_ctx` / `num_predict` : mêmes bornes gouvernées que `/v1/generate`
    # (profils de contexte et de génération du registre). ⚠️ `num_predict`
    # est le budget TOTAL de sortie Ollama : réflexion + contenu.
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[KeepAlive] = None
    # Mode de raisonnement ABSTRAIT (lot C1), vocabulaire fermé. Validé contre
    # le `reasoning_profile` du modèle puis traduit en `think` Ollama par la
    # Gateway ; une valeur hors profil est un 400 avant tout appel. Absent =
    # défaut du profil. `think` brut n'est PAS un champ de ce contrat.
    reasoning: Optional[Literal["none", "off", "on", "low", "medium", "high"]] = None
    # Sortie structurée gouvernée (lot C3). Abstraction Gateway : le `format`
    # Ollama n'est JAMAIS un champ client. Un seul mode, `json_schema` ; le
    # schéma est validé (sous-ensemble, bornes) AVANT tout appel et la
    # réponse est revalidée APRÈS. Exige la capability `structured_output`.
    response_format: Optional["ChatResponseFormat"] = None

    @field_validator("model")
    @classmethod
    def _model_name(cls, v: str) -> str:
        return validate_model_name(v)


class ChatResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    # Objet JSON brut : la validation fine (sous-ensemble, bornes) vit dans
    # app/structured.py, avec des codes d'erreur stables.
    schema_: Dict[str, Any] = Field(..., alias="schema")


class ChatAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    # Contenu FINAL du modèle, jamais fusionné avec la réflexion. Chaîne vide
    # possible : `content_empty` le dit explicitement.
    content: str


class ChatResponse(BaseModel):
    provider: str = "ai_gateway"
    upstream_provider: str = "ollama"
    model: str
    message: ChatAssistantMessage
    # Trace de raisonnement du modèle, distincte du contenu. `null` si le
    # modèle n'en a pas produit OU si la politique du modèle ne l'expose pas
    # (`thinking_exposed` lève l'ambiguïté ; `thinking_chars` dit s'il y en a
    # eu, sans en révéler le texte).
    thinking: Optional[str] = None
    thinking_exposed: bool = False
    thinking_chars: int = 0
    # Mode de raisonnement EFFECTIVEMENT appliqué (valeur bornée du même
    # vocabulaire que la requête), ou null si le modèle n'en pilote aucun.
    reasoning: Optional[str] = None
    done: bool = True
    # Raison d'arrêt VERBATIM d'Ollama ("stop", "length", …) ou null si
    # absente. Jamais normalisée, jamais inférée.
    done_reason: Optional[str] = None
    # True si le contenu assistant final est vide (ou blanc). Cas critique :
    # `thinking` non vide + `content` vide + `done_reason: "length"` = budget
    # de sortie consommé par la réflexion. Résultat VALIDE du backend, mais
    # incomplet — signalé, jamais remplacé par un texte de substitution.
    content_empty: bool
    tokens: TokenCounts
    usage: Usage
    cost: float = 0.0
    # Lot C3 — présent UNIQUEMENT quand la requête portait `response_format`
    # (le champ est retiré de la sérialisation sinon : contrat C2 intact).
    # Document déjà analysé strictement et validé contre le schéma demandé ;
    # `message.content` reste le texte brut du modèle.
    structured_output: Optional[Dict[str, Any]] = None


class ChatStreamThinking(BaseModel):
    """Fragment de réflexion. Émis uniquement si non vide ET si la politique
    du modèle expose la réflexion."""

    type: Literal["thinking"] = "thinking"
    delta: str
    done: Literal[False] = False


class ChatStreamContent(BaseModel):
    """Fragment de contenu final. Émis uniquement si non vide."""

    type: Literal["content"] = "content"
    delta: str
    done: Literal[False] = False


class ChatStreamDone(BaseModel):
    """Événement final d'un flux chat abouti — exactement un par flux réussi."""

    type: Literal["done"] = "done"
    delta: Literal[""] = ""
    done: Literal[True] = True
    model: str
    tokens: TokenCounts
    usage: Usage
    done_reason: Optional[str] = None
    # Même sémantique qu'en non-stream : aucun fragment de contenu non blanc
    # n'a été émis pendant le flux.
    content_empty: bool
    thinking_exposed: bool = False
    reasoning: Optional[str] = None


# --------------------------------------------------------------------------- #
# Reranking — POST /v1/rerank (lot rerank-sidecar-staging)
# --------------------------------------------------------------------------- #
class RerankDocument(BaseModel):
    """Passage à scorer. `id` est un identifiant OPAQUE du client (ses chunks,
    ses conventions) : renvoyé tel quel, jamais interprété, jamais journalisé."""
    # Fail-closed (lot vision-1a, contrat CONTRACT-0 §5) : un champ inconnu —
    # notamment `images` — est refusé en 400 au lieu d'être ignoré en silence.
    model_config = ConfigDict(extra="forbid")


    id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1)

    @field_validator("id", "text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ne doit pas être vide")
        return v


class RerankRequest(BaseModel):
    # Fail-closed (lot vision-1a, contrat CONTRACT-0 §5) : un champ inconnu —
    # notamment `images` — est refusé en 400 au lieu d'être ignoré en silence.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    documents: List[RerankDocument] = Field(..., min_length=1)
    model: Optional[str] = None
    # Absent => tous les documents, reclassés. Borne haute contrôlée en route
    # (top_n <= n_documents), la borne basse par Pydantic.
    top_n: Optional[int] = Field(default=None, ge=1)


    @field_validator("model")
    @classmethod
    def _model_name(cls, v: Optional[str]) -> Optional[str]:
        return validate_model_name(v)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query ne doit pas être vide")
        return v


class RerankScoreSemantics(BaseModel):
    """Sémantique EXPLICITE du score : un cross-encoder produit des logits
    ordinaux. Jamais de champ « probability »/« confidence » tant que
    `calibrated` est false."""

    higher_is_better: bool = True
    calibrated: bool = False


class RerankResult(BaseModel):
    index: int
    id: str
    relevance_score: float


class RerankUsage(BaseModel):
    documents: int
    returned: int


class RerankResponse(BaseModel):
    provider: str = "ai_gateway"
    upstream_provider: str = "local_cross_encoder"
    model: str
    model_logical: str
    model_revision: str
    # Tri décroissant par score, départage DÉTERMINISTE par index croissant.
    results: List[RerankResult]
    usage: RerankUsage
    score_semantics: RerankScoreSemantics


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class EmbeddingsRequest(BaseModel):
    # Fail-closed (lot vision-1a, contrat CONTRACT-0 §5) : un champ inconnu —
    # notamment `images` — est refusé en 400 au lieu d'être ignoré en silence.
    model_config = ConfigDict(extra="forbid")

    texts: List[str] = Field(..., min_length=1)
    model: Optional[str] = None
    keep_alive: Optional[KeepAlive] = None


    @field_validator("model")
    @classmethod
    def _model_name(cls, v: Optional[str]) -> Optional[str]:
        return validate_model_name(v)
    # Nature du texte encodé. Optionnel : son absence vaut "document", ce qui
    # reproduit exactement le comportement historique pour les clients qui
    # ignorent ce champ. La Gateway est le SEUL composant à appliquer les
    # préfixes du profil de modèle (cf. app/registry.py) — un préfixe posé en
    # double coûte 10 points de Top-1 en cross-lingue (mesuré).
    task: Optional[Literal["query", "document"]] = None

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("texts doit être une liste non vide")
        for i, item in enumerate(v):
            if not isinstance(item, str):
                raise ValueError(f"texts[{i}] doit être une chaîne de caractères")
            if item.strip() == "":
                raise ValueError(f"texts[{i}] ne doit pas être vide")
        return v


class EmbeddingsUsage(BaseModel):
    total_texts: int


class EmbeddingsResponse(BaseModel):
    """Réponse d'embeddings.

    Les sept premiers champs sont ceux du contrat historique : ils gardent leur
    nom, leur type et leur sens. Tout ce qui suit est **strictement additif** —
    un client qui les ignore n'observe aucun changement.
    """

    provider: str = "ai_gateway"
    upstream_provider: str = "ollama"
    model: str
    dimension: int
    embeddings: List[List[float]]
    usage: EmbeddingsUsage
    cost: float = 0.0

    # --- Ajouts lot 1 : identité, traçabilité, observabilité ---------------- #
    # `governed=False` signale un modèle absent du registre (ou déclassé) : la
    # dimension est alors seulement *observée*, pas garantie, et l'identité de
    # pipeline est nulle plutôt qu'inventée.
    governed: bool = False
    model_logical: Optional[str] = None
    # Lot P2 — `model_digest` est le digest RÉELLEMENT OBSERVÉ sur l'artefact
    # Ollama au dernier contrôle, ou null ; il n'est plus jamais rempli avec le
    # digest attendu. Celui-ci vit dans `expected_digest` (déclaratif).
    model_digest: Optional[str] = None
    expected_digest: Optional[str] = None
    # Preuve applicable à l'état courant : observé ET conforme. Distinct de
    # `governed` (autorisé par la politique, fail-open sur upstream injoignable).
    identity_verified: bool = False
    # Dernière observation conforme (ISO 8601 UTC) ou null — dernière preuve
    # connue, pas une preuve courante.
    verified_at: Optional[str] = None
    pipeline_id: Optional[str] = None
    task: str = "document"
    normalized: bool = False
    # Un booléen par texte, dans l'ordre. Avec `on_overflow: reject` (défaut du
    # registre), une réponse 200 ne peut contenir que des `false` : un texte
    # trop long est refusé en 400 au lieu d'être tronqué en silence.
    truncated: List[bool] = Field(default_factory=list)
    upstream_endpoint: str = "/api/embed"
