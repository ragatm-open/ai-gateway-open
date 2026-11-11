"""Registre versionné des modèles d'embedding.

Rôle : donner à la Gateway une identité *gouvernée* pour chaque modèle servi —
dimension attendue, digest de l'artefact, préfixes query/document, fenêtre
d'entrée, autorisation par environnement — au lieu de relayer aveuglément le
champ `model` du client vers Ollama.

Le fichier de description est `app/models_registry.yaml` (versionné avec le
code). Ce module le charge, le valide strictement, résout un nom demandé vers un
profil et vérifie que le digest annoncé correspond à celui réellement installé.

Politique de statut, volontairement asymétrique :

    ready       digest vérifié et conforme            -> gouverné, identité VÉRIFIÉE
    unknown     vérification pas encore faite         -> gouverné, identité non vérifiée
    unavailable Ollama injoignable / modèle absent    -> gouverné, identité non vérifiée
    degraded    digest PROUVÉ différent de l'attendu  -> NON gouverné, identité non vérifiée

Depuis une version antérieure, `governed` (autorisé par la politique) et `identity_verified`
(preuve applicable à l'état courant) sont deux propriétés distinctes ; la
réponse expose `expected_digest` (déclaré) et `model_digest` (observé ou
null), jamais l'un à la place de l'autre.

On échoue « ouvert » sur un incident de transport (refuser le service parce
qu'Ollama a hoqueté serait pire que le mal) et « fermé » sur une preuve de
divergence : dans ce cas l'artefact n'est plus celui décrit par le registre, et
continuer à estampiller les vecteurs d'un `pipeline_id` serait un mensonge.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
import re
import threading
import time
from dataclasses import dataclass, field, field
from typing import Any, Dict, List, Optional

import yaml

from .reasoning import IMPLICIT_NONE, ReasoningProfile, build_reasoning_profile

logger = logging.getLogger("ai_gateway.registry")

# Versions de schéma comprises par ce module.
SUPPORTED_VERSIONS = {1}

# --------------------------------------------------------------------------- #
# Identité canonique d'un pipeline d'embedding
# --------------------------------------------------------------------------- #
# Le `pipeline_id` est ce qu'un indexeur persiste par vecteur pour ne jamais
# comparer deux vecteurs issus de transformations différentes. Il DOIT donc
# refléter la transformation effective — et rien d'autre.
#
# L'ancienne forme `<logical>@<digest[:8]>/<task>-v1` était à la fois
# sous-sensible (un préfixe, la normalisation ou la politique de troncature
# changeaient sans que l'identifiant bouge — seule une constante manuelle
# pouvait le faire) et sur-sensible (renommer le nom logique changeait
# l'identifiant alors que les vecteurs étaient identiques).
#
# Nouvelle forme : `emb-v<schema>:<32 hex>` = SHA-256 (tronqué à 128 bits) d'un
# payload canonique — JSON à clés triées, séparateurs compacts, UTF-8 — qui ne
# porte QUE ce qui façonne le vecteur :
#   model_digest    digest COMPLET de l'artefact (plus 8 caractères)
#   task            query / document — rôle sémantique, même à préfixe égal
#   prefix          le préfixe effectif de CETTE task
#   overflow        `reject` (filtre d'admission : n'altère jamais un texte
#                   accepté, la borne n'entre donc PAS) ou
#                   `truncate` + `max_chars` (la borne façonne le texte)
#   normalized      garantie de norme (normalisation appliquée sur le repli)
#   implementation  PREPROCESSING_VERSION, cf. ci-dessous
#   schema          version du payload lui-même
# En sont EXCLUS : nom logique, ollama_name, langues, cross_lingual, taille de
# lot, allowlists d'environnement, dimension (conséquence du digest).
#
# Deux conséquences documentées : un renommage logique ne change PAS
# l'identifiant ; tout changement de préfixe / normalisation / troncature le
# change AUTOMATIQUEMENT, sans bump manuel.
EMBEDDING_PIPELINE_SCHEMA_VERSION = 2

# Version du CODE de préparation (`_prepare_texts`, garde anti double préfixe,
# normalisation du repli). Le YAML ne peut pas capturer un changement tel que
# `prefix + text` -> `prefix + normalize_whitespace(text)` : cette constante
# le doit. À incrémenter UNIQUEMENT quand un changement de code modifie la
# transformation effective — jamais pour un refactor à sortie identique.
PREPROCESSING_VERSION = "1"

# Longueur exposée : 32 hex = 128 bits. Collision accidentelle négligeable
# (bien au-delà de ce que quelques dizaines de pipelines pourront jamais
# produire) tout en restant lisible dans un log ou une métadonnée de chunk.
PIPELINE_ID_HEX_CHARS = 32


def canonical_json(payload: Dict[str, Any]) -> str:
    """Sérialisation déterministe : clés triées récursivement, séparateurs
    compacts, UTF-8 brut. Aucune dépendance à l'ordre d'insertion."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def embedding_pipeline_identity(
    *,
    digest: str,
    task: str,
    prefix: str,
    normalized: bool,
    on_overflow: str,
    max_input_chars: int,
    implementation_version: str = PREPROCESSING_VERSION,
    schema_version: int = EMBEDDING_PIPELINE_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Payload canonique de la transformation — voir le commentaire de section."""
    if on_overflow == "truncate_and_flag":
        overflow: Dict[str, Any] = {"policy": "truncate", "max_chars": int(max_input_chars)}
    else:
        overflow = {"policy": "reject"}
    return {
        "schema": int(schema_version),
        "model_digest": digest,
        "task": task,
        "prefix": prefix,
        "overflow": overflow,
        "normalized": bool(normalized),
        "implementation": str(implementation_version),
    }


def embedding_pipeline_id(identity: Dict[str, Any]) -> str:
    """`emb-v<schema>:<hex>` — SHA-256 du payload canonique, tronqué."""
    material = canonical_json(identity).encode("utf-8")
    return f"emb-v{identity['schema']}:{hashlib.sha256(material).hexdigest()[:PIPELINE_ID_HEX_CHARS]}"

# Tâches supportées par le contrat /v1/embeddings.
TASKS = ("query", "document")
DEFAULT_TASK = "document"

# Politiques de dépassement de `max_input_chars`.
ON_OVERFLOW_VALUES = ("reject", "truncate_and_flag")

# Classes de ressources des profils LLM. Seuils justifiés par les mesures VRAM
# sur la RTX 3090 (24 Gio) du 2026-08-17 : small ≤ ~1 Gio (embeddings),
# medium ≤ ~10 Gio à contexte gouverné (llama@4096 = 4,91 ; qwen14@8192 = 9,75),
# large > 12 Gio (qwen30@8192 = 17,95 ; mistral@32k = 18,3) — un modèle `large`
# ne peut PAS cohabiter avec un autre LLM sur ce GPU.
RESOURCE_CLASSES = ("small", "medium", "large")

# Statuts possibles d'un profil.
STATUS_READY = "ready"
STATUS_UNKNOWN = "unknown"
STATUS_DEGRADED = "degraded"
STATUS_UNAVAILABLE = "unavailable"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_FIELDS = (
    "provider",
    "ollama_name",
    "digest",
    "kind",
    "dimension",
    "max_batch_size",
    "max_input_chars",
    "on_overflow",
    "production_allowed",
    "staging_allowed",
)


class RegistryError(Exception):
    """Registre invalide. Le message ne contient jamais de secret."""


# --------------------------------------------------------------------------- #
# Chargement YAML strict
# --------------------------------------------------------------------------- #
class _NoDuplicateLoader(yaml.SafeLoader):
    """SafeLoader qui refuse les clés dupliquées.

    PyYAML accepte par défaut `a: 1` deux fois et garde silencieusement la
    dernière valeur. Pour un registre de modèles, c'est exactement le genre de
    faute qui passe la revue et surprend en production.
    """


def _no_duplicates(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RegistryError(f"clé dupliquée dans le registre : {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicateLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


# --------------------------------------------------------------------------- #
# Métadonnées Compact — communes aux trois familles de profils
# --------------------------------------------------------------------------- #
# Ce qu'un registre Open doit dire d'un modèle AVANT qu'il soit servi : son
# rôle dans le profil Compact, sa licence déclarée, son état de
# qualification, son budget mémoire ESTIMÉ et sa version. Tout est déclaratif
# et optionnel ; rien ici n'est jamais inventé par le code — un champ absent
# est rendu `null`, jamais rempli par une valeur plausible.
QUALIFICATION_STATES = ("candidate", "experimental", "qualified", "deprecated")


@dataclass(frozen=True)
class ProfileMetadata:
    compact_role: Optional[str] = None
    license: Optional[str] = None
    qualification: str = "candidate"
    memory_budget_mib: Optional[int] = None
    version: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "compact_role": self.compact_role,
            "license": self.license,
            "qualification": self.qualification,
            "memory_budget_mib": self.memory_budget_mib,
            "version": self.version,
        }


def _build_metadata(raw: Dict[str, Any], where: str) -> ProfileMetadata:
    def text(name: str) -> Optional[str]:
        value = raw.get(name)
        if value is None:
            return None
        _require(isinstance(value, str) and value.strip() != "", f"{where}.{name} doit être une chaîne non vide")
        return value.strip()
    qualification = raw.get("qualification", "candidate")
    _require(qualification in QUALIFICATION_STATES,
             f"{where}.qualification inconnue : {qualification!r} (admises : {', '.join(QUALIFICATION_STATES)})")
    memory = raw.get("memory_budget_mib")
    if memory is not None:
        memory = _as_positive_int(memory, "memory_budget_mib", where)
    return ProfileMetadata(
        compact_role=text("compact_role"), license=text("license"), qualification=qualification,
        memory_budget_mib=memory, version=text("version"),
    )


# --------------------------------------------------------------------------- #
# Profil
# --------------------------------------------------------------------------- #
@dataclass
class ModelProfile:
    logical_name: str
    provider: str
    ollama_name: str
    digest: str
    kind: str
    dimension: int
    languages: List[str]
    cross_lingual: bool
    normalized: bool
    query_prefix: str
    document_prefix: str
    max_batch_size: int
    max_input_chars: int
    on_overflow: str
    production_allowed: bool
    staging_allowed: bool

    # État runtime (renseigné par verify_digests, jamais par le YAML).
    status: str = STATUS_UNKNOWN
    observed_digest: Optional[str] = None
    status_detail: Optional[str] = None
    last_checked: Optional[float] = None
    # Lot P2 — horodatage (epoch) de la DERNIÈRE observation conforme
    # (observed == digest). Conservé à travers `unavailable`/`degraded` : c'est
    # la dernière preuve connue, PAS une preuve courante — cf. identity_verified.
    verified_at: Optional[float] = None
    # Métadonnées Compact, déclaratives.
    metadata: ProfileMetadata = field(default_factory=ProfileMetadata)

    # ------------------------------------------------------------------ infos
    @property
    def short_digest(self) -> str:
        return self.digest[:8]

    @property
    def governed(self) -> bool:
        """AUTORISÉ par la politique du registre : décrit ici et aucune
        divergence prouvée. Ne dit RIEN sur une vérification courante — c'est
        `identity_verified`. Un profil `degraded` cesse d'être
        gouverné (cf. docstring module)."""
        return self.status != STATUS_DEGRADED

    @property
    def identity_verified(self) -> bool:
        """Preuve APPLICABLE à l'état courant : au dernier contrôle, un digest a
        été observé et il est conforme. Fonction pure de l'état — `ready` n'est
        posé que par une observation conforme et toute autre issue (upstream
        injoignable, modèle absent, digest non rapporté, divergent) le retire.
        Aucun TTL parallèle : `last_checked` dit quand, l'état dit quoi."""
        return self.status == STATUS_READY and self.observed_digest == self.digest

    @property
    def verified_at_iso(self) -> Optional[str]:
        if self.verified_at is None:
            return None
        return datetime.fromtimestamp(self.verified_at, tz=timezone.utc).isoformat()

    def prefix_for(self, task: str) -> str:
        return self.query_prefix if task == "query" else self.document_prefix

    def pipeline_identity(self, task: str) -> Dict[str, Any]:
        """Payload canonique de la transformation pour cette task.
        Le nom logique n'y entre pas : c'est une étiquette, pas une étape."""
        identity = embedding_pipeline_identity(
            digest=self.digest,
            task=task,
            prefix=self.prefix_for(task),
            normalized=self.normalized,
            on_overflow=self.on_overflow,
            max_input_chars=self.max_input_chars,
        )
        return identity

    def pipeline_id(self, task: str) -> str:
        """Identité déterministe du pipeline ayant produit un vecteur.

        Forme : ``emb-v2:<32 hex>`` — SHA-256 du payload canonique. Deux
        profils à transformation identique partagent l'identifiant quel que
        soit leur nom ; toute différence de transformation le change.
        """
        return embedding_pipeline_id(self.pipeline_identity(task))

    def allowed_in(self, environment: str) -> bool:
        if environment == "production":
            return self.production_allowed
        if environment == "staging":
            return self.staging_allowed
        # Environnement inconnu : on n'invente pas d'autorisation.
        return False

    def matches_ollama_name(self, name: str) -> bool:
        """`nomic-embed-text` et `nomic-embed-text:latest` désignent le même
        artefact côté Ollama ; le registre n'a pas à porter les deux formes."""
        return name in self._name_aliases()

    def _name_aliases(self) -> set:
        base = self.ollama_name
        aliases = {base}
        if ":" in base:
            aliases.add(base.split(":", 1)[0])
        else:
            aliases.add(f"{base}:latest")
        return aliases

    def to_public_dict(self) -> Dict[str, Any]:
        """Vue exposable : ni chemin de blob, ni détail interne d'exception."""
        return {
            "model_logical": self.logical_name,
            "ollama_name": self.ollama_name,
            "kind": self.kind,
            "dimension": self.dimension,
            "languages": list(self.languages),
            "cross_lingual": self.cross_lingual,
            "normalized": self.normalized,
            "max_batch_size": self.max_batch_size,
            "max_input_chars": self.max_input_chars,
            "on_overflow": self.on_overflow,
            "production_allowed": self.production_allowed,
            "staging_allowed": self.staging_allowed,
            "expected_digest": self.digest,
            "observed_digest": self.observed_digest,
            "registry_status": self.status,
            # Lot P2 : la politique (`registry_status`/governed) et la preuve
            # d'identité sont deux informations distinctes.
            "identity_verified": self.identity_verified,
            "verified_at": self.verified_at_iso,
            "compact": self.metadata.to_public_dict(),
        }


# --------------------------------------------------------------------------- #
# Profils LLM (lot runtime-hardening)
# --------------------------------------------------------------------------- #
# Philosophie : rendre EXPLICITES des décisions aujourd'hui implicites.
# Sans profil, l'absence de `num_ctx` laisse Ollama allouer son défaut de
# runner (32768 observé sur 0.31.1) — soit +3,7 Gio de KV inutiles sur
# llama3.1:8b — et l'absence de `keep_alive` laisse le TTL à 5 min, trop court
# pour un juge sollicité par vagues. Ces profils ne sont PAS un scheduler : ils
# fournissent des valeurs par défaut stables, que le client peut toujours
# surcharger explicitement.


@dataclass(frozen=True)
class ContextProfile:
    """Profil de contexte STABLE. La stabilité est la propriété clé : changer
    `num_ctx` entre deux requêtes force Ollama à recharger le runner
    (3 à 6 s mesurées), même modèle déjà résident."""

    name: str
    num_ctx: int
    # Borne supérieure (INCLUSIVE) d'un `num_ctx` explicite du client.
    # Obligatoire depuis une version antérieure : sans borne, un client pouvait demander
    # n'importe quelle fenêtre — et le KV cache est alloué sur un GPU partagé
    # avec la production. Au-delà : refus 400, jamais un clamp silencieux.
    max_num_ctx: int


@dataclass(frozen=True)
class KeepAliveProfile:
    """TTL d'inactivité INTENTIONNEL, transmis à Ollama à chaque requête.

    ⚠️ Ce n'est PAS une réservation VRAM : sous pression mémoire, Ollama évince
    un modèle quel que soit son keep_alive (mesuré — un modèle épinglé `-1` a
    été évincé par le chargement suivant). `keep_alive` gouverne uniquement le
    déchargement pour inactivité.
    """

    name: str
    # None = ne rien envoyer, Ollama applique son défaut (5 min).
    keep_alive: Optional[str] = None
    # Lot P3 — le client peut-il surcharger `keep_alive` sur ce profil ?
    # True (défaut, comportement historique) : la valeur explicite du client
    # passe telle quelle, `-1` compris. False : toute valeur client est
    # refusée en 400 — la politique de rétention appartient au profil. Posé
    # sur le profil de gemma3:27b, dont le chemin legacy laissait un client
    # épingler 17 Gio de VRAM sur un GPU partagé avec la production.
    client_override: bool = True


@dataclass(frozen=True)
class GenerationProfile:
    """Budget de SORTIE gouverné (lot output-bound-hardening).

    Raison d'être — RCA du 2026-09-22. Sans `num_predict`, Ollama applique son
    défaut illimité : qwen3:30b-instruct est entré deux fois dans une boucle
    non terminante et a produit >11 000 puis 29 250 tokens à ~170 tok/s
    jusqu'au timeout de 180 s de la Gateway. Aucun garde-fou n'existait :
    `repeat_penalty=1.0`, `dry=0` et `--context-shift` côté runner laissent la
    génération tourner indéfiniment, la fenêtre de contexte ne se remplissant
    jamais.

    Le plafond est ici, dans la GOUVERNANCE, et non dans le transport : un
    `num_predict` codé en dur dans ollama_client serait invisible en revue,
    identique pour tous les modèles, et impossible à justifier par profil.

    ⚠️ Un budget n'est PAS un timeout. Le timeout reste un filet de sécurité
    pour un upstream réellement bloqué ; le budget borne la longueur de sortie.
    L'augmenter ne corrige rien à une génération non bornée — la RCA l'a établi.
    """

    name: str
    # Appliqué quand le caller ne fournit aucun `num_predict`.
    default_num_predict: int
    # Borne ABSOLUE d'un override explicite. Au-delà : refus 400, jamais un
    # clamp silencieux — un clamp changerait le contrat sans que le caller
    # l'apprenne, et il découvrirait la troncature dans le texte produit.
    max_num_predict: int


@dataclass
class LLMProfile:
    """Profil gouverné d'un modèle de génération.

    Beaucoup plus léger que ModelProfile (embeddings) : pas de digest épinglé
    ni de statut runtime — la génération n'a pas d'identité de pipeline à
    protéger. Le profil porte les propriétés OPÉRATIONNELLES : contexte,
    rétention VRAM, politique de raisonnement, classe de ressources.
    """

    logical_name: str
    ollama_name: str
    capabilities: List[str]
    resource_class: str
    context_profile: str
    keep_alive_profile: str
    generation_profile: str
    production_allowed: bool
    staging_allowed: bool
    # Politique de raisonnement appliquée UNIQUEMENT quand le client demande le
    # nom logique (cf. resolve_llm) : demander `qwen-judge` implique
    # think=false ; demander `qwen3:14b` conserve le comportement du modèle.
    default_think: Optional[bool] = None
    structured_output: bool = False
    # Vision (lot vision-1a) — additif. `modalities` est la source AUTORITAIRE
    # du contrôle de modalité (le champ historique `capabilities` reste
    # informatif ; la cohérence des deux est validée au chargement).
    modalities: List[str] = field(default_factory=lambda: ["text"])
    # Digest Ollama épinglé — OBLIGATOIRE pour un profil à modalité image
    # (l'identité vision est gouvernée comme celle des embeddings).
    digest: Optional[str] = None
    vision_schema_versions: List[str] = field(default_factory=list)
    # Tasks QUALIFIÉES (MODEL-1). Vide = rien de qualifié => refus contrôlé.
    vision_tasks: List[str] = field(default_factory=list)
    vision_max_images: int = 1
    vision_max_image_pixels: Optional[int] = None
    # Raisonnement gouverné (lot C1) : nom d'un `reasoning_profiles`, ou None
    # = profil implicite `none` (aucun mode pilotable). OBLIGATOIRE dès que la
    # capability `chat` est déclarée : un modèle chat sans politique de
    # raisonnement explicite ne charge pas.
    reasoning_profile: Optional[str] = None
    # Métadonnées Compact, déclaratives.
    metadata: ProfileMetadata = field(default_factory=ProfileMetadata)

    @property
    def supports_image(self) -> bool:
        return "image" in self.modalities

    def allowed_in(self, environment: str) -> bool:
        if environment == "production":
            return self.production_allowed
        if environment == "staging":
            return self.staging_allowed
        return False

    def _name_aliases(self) -> set:
        """Équivalences de nom, plus strictes que côté embeddings.

        Chez Ollama, `x` nu désigne `x:latest` — et RIEN d'autre. `qwen3` ne
        désigne donc ni `qwen3:14b` ni `qwen3:30b-instruct` : ajouter l'alias
        nu pour un tag explicite créerait de fausses ambiguïtés entre les
        membres d'une même famille.
        """
        base = self.ollama_name
        aliases = {base}
        if ":" in base:
            name, tag = base.split(":", 1)
            if tag == "latest":
                aliases.add(name)
        else:
            aliases.add(f"{base}:latest")
        return aliases

    def matches_ollama_name(self, name: str) -> bool:
        return name in self._name_aliases()

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "model_logical": self.logical_name,
            "ollama_name": self.ollama_name,
            "capabilities": list(self.capabilities),
            "resource_class": self.resource_class,
            "context_profile": self.context_profile,
            "keep_alive_profile": self.keep_alive_profile,
            "generation_profile": self.generation_profile,
            "default_think": self.default_think,
            "structured_output": self.structured_output,
            "production_allowed": self.production_allowed,
            "staging_allowed": self.staging_allowed,
            "reasoning_profile": self.reasoning_profile,
            # Digest DÉCLARÉ (optionnel hors vision) : jamais inventé.
            "expected_digest": self.digest,
            "compact": self.metadata.to_public_dict(),
        }

    def to_public_dict_with_vision(self) -> Dict[str, Any]:
        """Dict public + métadonnées vision STATIQUES (contrat §98) — l'état
        dynamique (enabled/resident) appartient à /health/vision."""
        entry = self.to_public_dict()
        if self.supports_image:
            entry["modalities"] = list(self.modalities)
            entry["vision"] = {
                "tasks": list(self.vision_tasks),
                "schema_versions": list(self.vision_schema_versions),
                "max_images": self.vision_max_images,
            }
            if self.vision_max_image_pixels is not None:
                entry["vision"]["max_image_pixels"] = self.vision_max_image_pixels
        return entry


@dataclass(frozen=True)
class ResolvedLLM:
    """Résultat de `resolve_llm` : le profil ET la façon dont il a été nommé.

    `via_logical` conditionne `default_think` : la politique de raisonnement
    est un choix d'usage (le rôle « juge »), pas une propriété du binaire —
    elle ne s'applique donc qu'au nom logique.
    """

    profile: LLMProfile
    via_logical: bool


# --------------------------------------------------------------------------- #
# Profils reranker (lot rerank-sidecar-staging)
# --------------------------------------------------------------------------- #
# Un reranker n'est PAS un modèle Ollama : il vit dans un backend local dédié
# (cross-encoder, bind loopback) et n'a ni dimension, ni pipeline_id, ni
# préfixes — son identité se réduit à artefact + révision. D'où un profil
# distinct plutôt qu'une généralisation risquée du profil d'embedding.

RERANKER_PROVIDERS = ("local_cross_encoder",)

# Révision git HuggingFace : 40 caractères hexadécimaux, immuable — jamais un
# tag mobile (`main`).
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class RerankerProfile:
    """Profil gouverné d'un modèle de reranking.

    Les scores d'un cross-encoder sont des LOGITS ORDINAUX, pas des
    probabilités : `higher_is_better`/`calibrated` font partie du contrat pour
    que personne n'ait à le deviner.
    """

    logical_name: str
    provider: str
    artifact: str
    revision: str
    max_documents_per_request: int
    max_query_chars: int
    max_document_chars: int
    max_total_chars: int
    on_overflow: str
    production_allowed: bool
    staging_allowed: bool
    higher_is_better: bool = True
    calibrated: bool = False
    metadata: ProfileMetadata = field(default_factory=ProfileMetadata)

    @property
    def short_revision(self) -> str:
        return self.revision[:8]

    def allowed_in(self, environment: str) -> bool:
        if environment == "production":
            return self.production_allowed
        if environment == "staging":
            return self.staging_allowed
        return False

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "model_logical": self.logical_name,
            "compact": self.metadata.to_public_dict(),
            "provider": self.provider,
            "artifact": self.artifact,
            "revision": self.revision,
            "max_documents_per_request": self.max_documents_per_request,
            "max_query_chars": self.max_query_chars,
            "max_document_chars": self.max_document_chars,
            "max_total_chars": self.max_total_chars,
            "on_overflow": self.on_overflow,
            "higher_is_better": self.higher_is_better,
            "calibrated": self.calibrated,
            "production_allowed": self.production_allowed,
            "staging_allowed": self.staging_allowed,
        }


# --------------------------------------------------------------------------- #
# ASR (lot ASR-RAIL-FOUNDATION) — section `asr_models`, séparée des LLM.
# --------------------------------------------------------------------------- #
ASR_PROVIDERS = ("faster-whisper",)
ASR_CAPABILITIES = ("transcription", "word_timestamps", "vocabulary_biasing", "multilingual", "diarization")
ASR_LANGUAGES = ("fr", "en")
ASR_COMPUTE_TYPES = ("float16", "int8_float16")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AsrVadIdentity:
    provider: str
    source: str
    version: str

    def to_public_dict(self) -> Dict[str, Any]:
        return {"provider": self.provider, "source": self.source, "version": self.version}


@dataclass(frozen=True)
class AsrDiarizationIdentity:
    """Identité ÉPINGLÉE du modèle de diarisation servi par le backend
    (Sortformer) : `/health/asr` compare provider, artefact, révision et hash
    .nemo mesuré à ces valeurs (fail-closed, lot production-full-capability-enablement)."""
    provider: str
    artifact: str
    revision: str
    artifact_sha256: str

    def to_public_dict(self) -> Dict[str, Any]:
        return {"provider": self.provider, "artifact": self.artifact, "revision": self.revision,
                "artifact_sha256": self.artifact_sha256}


@dataclass
class AsrModelProfile:
    """Profil gouverné d'un modèle de transcription (backend ASR dédié).

    Identité ÉPINGLÉE trois fois (artefact, révision, hash du fichier
    modèle) : `/health/asr` refuse de servir un backend dont `/info` diverge.
    Les bornes (durée, octets, seuil sync) sont des décisions de gouvernance
    fondées sur la recon du 2026-09-28, pas des limites natives du modèle.
    """

    logical_name: str
    provider: str
    artifact: str
    revision: str
    artifact_sha256: str
    compute_type: str
    vad: AsrVadIdentity
    capabilities: List[str]
    languages: List[str]
    max_duration_seconds: int
    sync_max_duration_seconds: int
    max_upload_bytes: int
    resource_class: str
    production_allowed: bool
    staging_allowed: bool
    diarization: Optional[AsrDiarizationIdentity] = None

    @property
    def short_revision(self) -> str:
        return self.revision[:8]

    def allowed_in(self, environment: str) -> bool:
        if environment == "production":
            return self.production_allowed
        if environment == "staging":
            return self.staging_allowed
        return False

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "model_logical": self.logical_name,
            "diarization": self.diarization.to_public_dict() if self.diarization is not None else None,
            "provider": self.provider,
            "artifact": self.artifact,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "compute_type": self.compute_type,
            "vad": self.vad.to_public_dict(),
            "capabilities": list(self.capabilities),
            "languages": list(self.languages),
            "max_duration_seconds": self.max_duration_seconds,
            "sync_max_duration_seconds": self.sync_max_duration_seconds,
            "max_upload_bytes": self.max_upload_bytes,
            "resource_class": self.resource_class,
            "production_allowed": self.production_allowed,
            "staging_allowed": self.staging_allowed,
        }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _as_bool(value: Any, name: str, logical: str) -> bool:
    _require(isinstance(value, bool), f"{logical}.{name} doit être un booléen")
    return value


def _as_positive_int(value: Any, name: str, logical: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{logical}.{name} doit être un entier",
    )
    _require(value > 0, f"{logical}.{name} doit être strictement positif (reçu {value})")
    return value


def _as_prefix(value: Any, name: str, logical: str) -> str:
    # Un préfixe absent est licite (bge-m3, granite n'en veulent aucun) ; un
    # préfixe non textuel ne l'est pas.
    if value is None:
        return ""
    _require(isinstance(value, str), f"{logical}.{name} doit être une chaîne")
    return value


def _build_profile(logical: str, raw: Any) -> ModelProfile:
    _require(isinstance(raw, dict), f"le modèle {logical!r} doit être un mapping")
    _require(isinstance(logical, str) and logical.strip() != "",
             "un nom logique de modèle ne peut pas être vide")

    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    _require(not missing, f"{logical}: champs obligatoires manquants : {', '.join(missing)}")

    provider = raw["provider"]
    _require(provider == "ollama", f"{logical}.provider non supporté : {provider!r}")

    ollama_name = raw["ollama_name"]
    _require(
        isinstance(ollama_name, str) and ollama_name.strip() != "",
        f"{logical}.ollama_name est vide",
    )

    digest = raw["digest"]
    _require(isinstance(digest, str), f"{logical}.digest doit être une chaîne")
    digest = digest.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:"):]
    _require(
        bool(_DIGEST_RE.match(digest)),
        f"{logical}.digest mal formé : 64 caractères hexadécimaux attendus",
    )

    kind = raw["kind"]
    _require(kind == "embedding", f"{logical}.kind doit valoir 'embedding' (reçu {kind!r})")

    languages = raw.get("languages") or []
    _require(isinstance(languages, list), f"{logical}.languages doit être une liste")
    _require(
        all(isinstance(x, str) and x.strip() for x in languages),
        f"{logical}.languages doit ne contenir que des codes de langue textuels",
    )

    on_overflow = raw["on_overflow"]
    _require(
        on_overflow in ON_OVERFLOW_VALUES,
        f"{logical}.on_overflow inconnu : {on_overflow!r} "
        f"(valeurs admises : {', '.join(ON_OVERFLOW_VALUES)})",
    )

    return ModelProfile(
        logical_name=logical,
        provider=provider,
        ollama_name=ollama_name,
        digest=digest,
        kind=kind,
        dimension=_as_positive_int(raw["dimension"], "dimension", logical),
        languages=list(languages),
        cross_lingual=_as_bool(raw.get("cross_lingual", False), "cross_lingual", logical),
        normalized=_as_bool(raw.get("normalized", False), "normalized", logical),
        query_prefix=_as_prefix(raw.get("query_prefix", ""), "query_prefix", logical),
        document_prefix=_as_prefix(raw.get("document_prefix", ""), "document_prefix", logical),
        max_batch_size=_as_positive_int(raw["max_batch_size"], "max_batch_size", logical),
        max_input_chars=_as_positive_int(raw["max_input_chars"], "max_input_chars", logical),
        on_overflow=on_overflow,
        production_allowed=_as_bool(raw["production_allowed"], "production_allowed", logical),
        staging_allowed=_as_bool(raw["staging_allowed"], "staging_allowed", logical),
        metadata=_build_metadata(raw, logical),
    )


def _build_context_profile(name: str, raw: Any) -> ContextProfile:
    _require(isinstance(raw, dict), f"context_profiles.{name} doit être un mapping")
    _require("num_ctx" in raw, f"context_profiles.{name}: champ num_ctx manquant")
    num_ctx = _as_positive_int(raw["num_ctx"], "num_ctx", f"context_profiles.{name}")
    # Borne OBLIGATOIRE : un profil sans plafond laisserait un
    # override client dimensionner librement le KV cache. Même logique que
    # `generation_profile` — le registre refuse de charger plutôt que de
    # servir un profil non borné.
    _require("max_num_ctx" in raw, f"context_profiles.{name}: champ max_num_ctx manquant")
    max_num_ctx = _as_positive_int(
        raw["max_num_ctx"], "max_num_ctx", f"context_profiles.{name}"
    )
    _require(
        max_num_ctx >= num_ctx,
        f"context_profiles.{name}: max_num_ctx ({max_num_ctx}) < num_ctx ({num_ctx})",
    )
    return ContextProfile(name=name, num_ctx=num_ctx, max_num_ctx=max_num_ctx)


def _build_keep_alive_profile(name: str, raw: Any) -> KeepAliveProfile:
    _require(isinstance(raw, dict), f"keep_alive_profiles.{name} doit être un mapping")
    _require(
        "keep_alive" in raw, f"keep_alive_profiles.{name}: champ keep_alive manquant"
    )
    value = raw["keep_alive"]
    if value is not None:
        # Durée Ollama ("30m", "90s") ou nombre de secondes. On accepte les
        # deux formes mais on normalise en chaîne, la forme la plus lisible
        # dans les logs. 0 et -1 sont volontairement refusés en PROFIL : un
        # déchargement immédiat ou un épinglage sont des gestes ponctuels de
        # campagne, pas une politique par défaut.
        _require(
            isinstance(value, (str, int)) and not isinstance(value, bool),
            f"keep_alive_profiles.{name}: durée invalide",
        )
        if isinstance(value, int):
            _require(
                value > 0,
                f"keep_alive_profiles.{name}: un profil ne peut pas porter "
                f"{value} (0/-1 sont des gestes de campagne, pas une politique)",
            )
            value = f"{value}s"
        _require(value.strip() != "", f"keep_alive_profiles.{name}: durée vide")
    client_override = _as_bool(
        raw.get("client_override", True), "client_override", f"keep_alive_profiles.{name}"
    )
    return KeepAliveProfile(name=name, keep_alive=value, client_override=client_override)


def _build_generation_profile(name: str, raw: Any) -> GenerationProfile:
    _require(isinstance(raw, dict), f"generation_profiles.{name} doit être un mapping")
    for required in ("default_num_predict", "max_num_predict"):
        _require(
            required in raw, f"generation_profiles.{name}: champ {required} manquant"
        )
    where = f"generation_profiles.{name}"
    # _as_positive_int refuse déjà 0, les négatifs, les booléens et les
    # non-entiers. C'est exactement le contrat voulu : un profil ne peut PAS
    # déclarer un budget nul, négatif, ou la sentinelle -1 « illimité » —
    # ce sont précisément les valeurs qui ont produit la RCA.
    default_num_predict = _as_positive_int(
        raw["default_num_predict"], "default_num_predict", where
    )
    max_num_predict = _as_positive_int(
        raw["max_num_predict"], "max_num_predict", where
    )
    _require(
        max_num_predict >= default_num_predict,
        f"{where}: max_num_predict ({max_num_predict}) < default_num_predict "
        f"({default_num_predict})",
    )
    return GenerationProfile(
        name=name,
        default_num_predict=default_num_predict,
        max_num_predict=max_num_predict,
    )


def _build_llm_profile(
    logical: str,
    raw: Any,
    context_profiles: Dict[str, ContextProfile],
    keep_alive_profiles: Dict[str, KeepAliveProfile],
    generation_profiles: Dict[str, GenerationProfile],
    reasoning_profiles: Optional[Dict[str, ReasoningProfile]] = None,
) -> LLMProfile:
    _require(isinstance(raw, dict), f"llm_models.{logical} doit être un mapping")
    _require(
        isinstance(logical, str) and logical.strip() != "",
        "un nom logique de modèle LLM ne peut pas être vide",
    )
    # `generation_profile` est OBLIGATOIRE : un modèle de génération gouverné
    # sans budget de sortie déclaré retomberait dans l'illimité d'Ollama, soit
    # exactement la cause racine. Le registre refuse donc de charger plutôt que
    # de servir un modèle non borné (échec au démarrage, pas à la 1re requête).
    for required in ("ollama_name", "resource_class", "context_profile",
                     "keep_alive_profile", "generation_profile",
                     "production_allowed", "staging_allowed"):
        _require(required in raw, f"llm_models.{logical}: champ {required} manquant")

    ollama_name = raw["ollama_name"]
    _require(
        isinstance(ollama_name, str) and ollama_name.strip() != "",
        f"llm_models.{logical}.ollama_name est vide",
    )

    resource_class = raw["resource_class"]
    _require(
        resource_class in RESOURCE_CLASSES,
        f"llm_models.{logical}.resource_class inconnu : {resource_class!r} "
        f"(admis : {', '.join(RESOURCE_CLASSES)})",
    )

    context_profile = raw["context_profile"]
    _require(
        context_profile in context_profiles,
        f"llm_models.{logical}.context_profile inconnu : {context_profile!r}",
    )
    keep_alive_profile = raw["keep_alive_profile"]
    _require(
        keep_alive_profile in keep_alive_profiles,
        f"llm_models.{logical}.keep_alive_profile inconnu : {keep_alive_profile!r}",
    )
    generation_profile = raw["generation_profile"]
    _require(
        generation_profile in generation_profiles,
        f"llm_models.{logical}.generation_profile inconnu : {generation_profile!r}",
    )

    capabilities = raw.get("capabilities") or []
    _require(isinstance(capabilities, list), f"llm_models.{logical}.capabilities doit être une liste")
    _require(
        all(isinstance(x, str) and x.strip() for x in capabilities),
        f"llm_models.{logical}.capabilities ne doit contenir que des chaînes",
    )

    default_think = raw.get("default_think")
    if default_think is not None:
        _require(
            isinstance(default_think, bool),
            f"llm_models.{logical}.default_think doit être un booléen",
        )

    # Raisonnement gouverné (lot C1). La capability `chat` est opt-in et
    # EXIGE une politique de raisonnement explicite : c'est le contrat de
    # `/v1/chat` (mode client validé contre le profil, jamais relayé brut).
    reasoning_profile = raw.get("reasoning_profile")
    if reasoning_profile is not None:
        _require(
            isinstance(reasoning_profile, str) and reasoning_profile in (reasoning_profiles or {}),
            f"llm_models.{logical}.reasoning_profile inconnu : {reasoning_profile!r}",
        )
    _require(
        "chat" not in capabilities or reasoning_profile is not None,
        f"llm_models.{logical} : la capability chat exige un reasoning_profile explicite",
    )

    # Vision (lot vision-1a). `modalities` est la source autoritaire du
    # contrôle de modalité ; cohérence gelée avec `capabilities` : la
    # capability historique "vision" et la modalité "image" vont ensemble.
    modalities = raw.get("modalities") or ["text"]
    _require(
        isinstance(modalities, list) and modalities
        and all(m in ("text", "image") for m in modalities),
        f"llm_models.{logical}.modalities doit être une liste non vide parmi text/image",
    )
    _require(
        ("vision" in capabilities) == ("image" in modalities),
        f"llm_models.{logical} : capability 'vision' et modalité 'image' doivent "
        f"être déclarées ensemble (registre incohérent)",
    )
    digest = raw.get("digest")
    if digest is not None:
        _require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"llm_models.{logical}.digest doit être 64 caractères hexadécimaux",
        )
    vision_raw = raw.get("vision")
    vision_schema_versions: list = []
    vision_tasks: list = []
    vision_max_images = 1
    vision_max_image_pixels = None
    if "image" in modalities:
        _require(vision_raw is not None,
                 f"llm_models.{logical} : modalité image exige un bloc vision")
        _require(digest is not None,
                 f"llm_models.{logical} : modalité image exige un digest épinglé")
        _require(isinstance(vision_raw, dict),
                 f"llm_models.{logical}.vision doit être un mapping")
        vision_schema_versions = vision_raw.get("schema_versions") or []
        _require(
            isinstance(vision_schema_versions, list) and vision_schema_versions
            and all(isinstance(v, str) for v in vision_schema_versions),
            f"llm_models.{logical}.vision.schema_versions doit être une liste de chaînes",
        )
        vision_tasks = vision_raw.get("tasks")
        _require(isinstance(vision_tasks, list),
                 f"llm_models.{logical}.vision.tasks doit être une liste (vide admise)")
        # Lot SEMX-4A : `semantic_extract` (contrat v3) est un vocabulaire
        # ADMIS, pas une activation — aucun profil ne le déclare par défaut.
        _VISION_TASKS = ("auto", "describe", "table", "diagram", "chart", "technical_drawing",
                         "semantic_extract")
        _require(all(t in _VISION_TASKS for t in vision_tasks),
                 f"llm_models.{logical}.vision.tasks contient une task hors contrat")
        # Appariement contractuel : la version "3" n'existe QUE pour
        # semantic_extract, et réciproquement. Un registre incohérent refuse
        # de charger plutôt que de servir une moitié de contrat.
        _require(("semantic_extract" in vision_tasks) == ("3" in vision_schema_versions),
                 f"llm_models.{logical}.vision : semantic_extract et schema_version \"3\" "
                 "vont ensemble")
        vision_max_images = vision_raw.get("max_images", 1)
        _require(vision_max_images == 1,
                 f"llm_models.{logical}.vision.max_images doit valoir 1 en V1")
        vision_max_image_pixels = vision_raw.get("max_image_pixels")
        if vision_max_image_pixels is not None:
            _require(isinstance(vision_max_image_pixels, int) and vision_max_image_pixels > 0,
                     f"llm_models.{logical}.vision.max_image_pixels doit être un entier > 0")
    else:
        _require(vision_raw is None,
                 f"llm_models.{logical} : bloc vision sans modalité image")

    return LLMProfile(
        logical_name=logical,
        ollama_name=ollama_name,
        capabilities=list(capabilities),
        resource_class=resource_class,
        context_profile=context_profile,
        keep_alive_profile=keep_alive_profile,
        generation_profile=generation_profile,
        production_allowed=_as_bool(raw["production_allowed"], "production_allowed", logical),
        staging_allowed=_as_bool(raw["staging_allowed"], "staging_allowed", logical),
        default_think=default_think,
        structured_output=_as_bool(
            raw.get("structured_output", False), "structured_output", logical
        ),
        modalities=list(modalities),
        digest=digest,
        vision_schema_versions=list(vision_schema_versions),
        vision_tasks=list(vision_tasks),
        vision_max_images=vision_max_images,
        vision_max_image_pixels=vision_max_image_pixels,
        reasoning_profile=reasoning_profile,
        metadata=_build_metadata(raw, f"llm_models.{logical}"),
    )


def _build_reranker_profile(logical: str, raw: Any) -> RerankerProfile:
    _require(isinstance(raw, dict), f"reranker_models.{logical} doit être un mapping")
    _require(
        isinstance(logical, str) and logical.strip() != "",
        "un nom logique de reranker ne peut pas être vide",
    )
    for required in ("provider", "artifact", "revision", "max_documents_per_request",
                     "max_query_chars", "max_document_chars", "max_total_chars",
                     "on_overflow", "production_allowed", "staging_allowed"):
        _require(required in raw, f"reranker_models.{logical}: champ {required} manquant")

    provider = raw["provider"]
    _require(
        provider in RERANKER_PROVIDERS,
        f"reranker_models.{logical}.provider non supporté : {provider!r}",
    )

    artifact = raw["artifact"]
    _require(
        isinstance(artifact, str) and artifact.strip() != "",
        f"reranker_models.{logical}.artifact est vide",
    )

    revision = raw["revision"]
    _require(isinstance(revision, str), f"reranker_models.{logical}.revision doit être une chaîne")
    revision = revision.strip().lower()
    _require(
        bool(_REVISION_RE.match(revision)),
        f"reranker_models.{logical}.revision mal formée : 40 caractères "
        "hexadécimaux attendus (SHA de commit HuggingFace, pas un tag)",
    )

    on_overflow = raw["on_overflow"]
    # Volontairement plus strict que les embeddings : aucune politique de
    # troncature n'existe pour le reranking — un passage amputé change le score
    # sans le dire.
    _require(
        on_overflow == "reject",
        f"reranker_models.{logical}.on_overflow doit valoir 'reject'",
    )

    return RerankerProfile(
        logical_name=logical,
        provider=provider,
        artifact=artifact,
        revision=revision,
        max_documents_per_request=_as_positive_int(
            raw["max_documents_per_request"], "max_documents_per_request", logical
        ),
        max_query_chars=_as_positive_int(raw["max_query_chars"], "max_query_chars", logical),
        max_document_chars=_as_positive_int(
            raw["max_document_chars"], "max_document_chars", logical
        ),
        max_total_chars=_as_positive_int(raw["max_total_chars"], "max_total_chars", logical),
        on_overflow=on_overflow,
        production_allowed=_as_bool(raw["production_allowed"], "production_allowed", logical),
        staging_allowed=_as_bool(raw["staging_allowed"], "staging_allowed", logical),
        higher_is_better=_as_bool(
            raw.get("higher_is_better", True), "higher_is_better", logical
        ),
        calibrated=_as_bool(raw.get("calibrated", False), "calibrated", logical),
        metadata=_build_metadata(raw, f"reranker_models.{logical}"),
    )


def _build_asr_profile(logical: str, raw: Any) -> AsrModelProfile:
    where = f"asr_models.{logical}"
    _require(isinstance(raw, dict), f"{where} doit être un mapping")
    _require(isinstance(logical, str) and logical.strip() != "", "un nom logique ASR ne peut pas être vide")
    required = ("provider", "artifact", "revision", "artifact_sha256", "compute_type", "vad", "capabilities",
                "languages", "max_duration_seconds", "sync_max_duration_seconds", "max_upload_bytes",
                "resource_class", "production_allowed", "staging_allowed")
    optional = ("diarization",)
    for key in required:
        _require(key in raw, f"{where}: champ {key} manquant")
    for key in raw:
        _require(key in required or key in optional, f"{where}: champ inconnu {key!r}")
    _require(raw["provider"] in ASR_PROVIDERS, f"{where}.provider non supporté : {raw['provider']!r}")
    artifact = raw["artifact"]
    _require(isinstance(artifact, str) and artifact.strip() != "" and "/" in artifact, f"{where}.artifact doit être 'organisation/nom'")
    revision = raw["revision"]
    _require(isinstance(revision, str) and bool(_REVISION_RE.match(revision.strip().lower())),
             f"{where}.revision mal formée : 40 caractères hexadécimaux attendus")
    sha = raw["artifact_sha256"]
    _require(isinstance(sha, str) and bool(_SHA256_RE.match(sha.strip().lower())),
             f"{where}.artifact_sha256 mal formé : 64 caractères hexadécimaux attendus")
    _require(raw["compute_type"] in ASR_COMPUTE_TYPES, f"{where}.compute_type non supporté : {raw['compute_type']!r}")
    vad = raw["vad"]
    _require(isinstance(vad, dict) and set(vad) == {"provider", "source", "version"}, f"{where}.vad doit contenir provider, source, version")
    _require(all(isinstance(vad[k], str) and vad[k].strip() for k in vad), f"{where}.vad : champs non vides attendus")
    _require(vad["provider"] == "silero", f"{where}.vad.provider non supporté : {vad['provider']!r}")
    caps = raw["capabilities"]
    _require(isinstance(caps, list) and caps and all(isinstance(c, str) for c in caps), f"{where}.capabilities doit être une liste non vide")
    _require(len(set(caps)) == len(caps), f"{where}.capabilities contient un doublon")
    for c in caps:
        _require(c in ASR_CAPABILITIES, f"{where}.capabilities : {c!r} inconnue (admises : {', '.join(ASR_CAPABILITIES)})")
    _require("transcription" in caps, f"{where}.capabilities doit contenir 'transcription'")
    langs = raw["languages"]
    _require(isinstance(langs, list) and langs and all(l in ASR_LANGUAGES for l in langs) and len(set(langs)) == len(langs),
             f"{where}.languages doit être une liste sans doublon parmi {', '.join(ASR_LANGUAGES)}")
    max_dur = _as_positive_int(raw["max_duration_seconds"], "max_duration_seconds", logical)
    sync_dur = _as_positive_int(raw["sync_max_duration_seconds"], "sync_max_duration_seconds", logical)
    _require(sync_dur <= max_dur, f"{where}.sync_max_duration_seconds doit être <= max_duration_seconds")
    max_bytes = _as_positive_int(raw["max_upload_bytes"], "max_upload_bytes", logical)
    _require(raw["resource_class"] == "speech-gpu", f"{where}.resource_class doit valoir 'speech-gpu'")
    diar_identity: Optional[AsrDiarizationIdentity] = None
    if "diarization" in caps:
        # La capability `diarization` exige une identité épinglée du modèle
        # de diarisation : sans elle, le registre est refusé (fail-closed).
        d = raw.get("diarization")
        _require(isinstance(d, dict) and set(d) == {"provider", "artifact", "revision", "artifact_sha256"},
                 f"{where}.diarization doit contenir provider, artifact, revision, artifact_sha256")
        _require(d["provider"] == "nemo-sortformer", f"{where}.diarization.provider non supporté : {d['provider']!r}")
        _require(isinstance(d["artifact"], str) and "/" in d["artifact"], f"{where}.diarization.artifact doit être 'organisation/nom'")
        _require(isinstance(d["revision"], str) and bool(_REVISION_RE.match(d["revision"].strip().lower())),
                 f"{where}.diarization.revision mal formée : 40 caractères hexadécimaux attendus")
        _require(isinstance(d["artifact_sha256"], str) and bool(_SHA256_RE.match(d["artifact_sha256"].strip().lower())),
                 f"{where}.diarization.artifact_sha256 mal formé : 64 caractères hexadécimaux attendus")
        diar_identity = AsrDiarizationIdentity(provider=d["provider"], artifact=d["artifact"].strip(),
                                               revision=d["revision"].strip().lower(),
                                               artifact_sha256=d["artifact_sha256"].strip().lower())
    else:
        _require("diarization" not in raw, f"{where}.diarization n'a de sens qu'avec la capability 'diarization'")
    return AsrModelProfile(
        logical_name=logical, provider=raw["provider"], artifact=artifact.strip(), revision=revision.strip().lower(),
        artifact_sha256=sha.strip().lower(), compute_type=raw["compute_type"],
        vad=AsrVadIdentity(provider=vad["provider"], source=vad["source"], version=vad["version"]),
        capabilities=list(caps), languages=list(langs), max_duration_seconds=max_dur,
        sync_max_duration_seconds=sync_dur, max_upload_bytes=max_bytes, resource_class=raw["resource_class"],
        production_allowed=_as_bool(raw["production_allowed"], "production_allowed", logical),
        staging_allowed=_as_bool(raw["staging_allowed"], "staging_allowed", logical),
        diarization=diar_identity,
    )


# --------------------------------------------------------------------------- #
# Registre
# --------------------------------------------------------------------------- #
@dataclass
class ModelRegistry:
    version: int
    models: Dict[str, ModelProfile] = field(default_factory=dict)
    source_path: Optional[str] = None
    # Gouvernance LLM (lot runtime-hardening). Sections OPTIONNELLES du même
    # YAML : un registre sans elles reste valide et le service se comporte
    # exactement comme avant.
    context_profiles: Dict[str, ContextProfile] = field(default_factory=dict)
    keep_alive_profiles: Dict[str, KeepAliveProfile] = field(default_factory=dict)
    generation_profiles: Dict[str, GenerationProfile] = field(default_factory=dict)
    # Raisonnement gouverné (lot C1), section optionnelle.
    reasoning_profiles: Dict[str, ReasoningProfile] = field(default_factory=dict)
    llm_models: Dict[str, LLMProfile] = field(default_factory=dict)
    reranker_models: Dict[str, RerankerProfile] = field(default_factory=dict)
    # Vision (lot vision-1a) : modèle vision par défaut, résolu PAR LE
    # REGISTRE (jamais une env var ad-hoc, jamais un premier-modèle-venu).
    vision_default_model: Optional[str] = None
    # ASR (lot ASR-RAIL-FOUNDATION) : modèles de transcription + défaut.
    asr_models: Dict[str, AsrModelProfile] = field(default_factory=dict)
    asr_default_model: Optional[str] = None
    # Inventaire des modèles INSTALLÉS sur l'upstream : noms
    # rapportés par /api/tags au dernier contrôle, ou None tant qu'aucun
    # contrôle n'a abouti. Lecture NON inférentielle : « enregistré » (ici)
    # et « installé » (inventaire) sont deux états distincts, et aucun des
    # deux ne vaut « capacité testée ».
    inventory: Optional[set] = None
    inventory_at: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def llm_installed(self, profile: "LLMProfile") -> Optional[bool]:
        """True/False d'après le dernier inventaire ; None = inconnu."""
        if self.inventory is None:
            return None
        return any(alias in self.inventory for alias in profile._name_aliases())

    def inventory_state(self) -> str:
        return "UNKNOWN" if self.inventory is None else "OBSERVED"

    def resolve_asr(self, name: Optional[str]) -> Optional[AsrModelProfile]:
        """Nom logique seul (aucun alias, aucun nom d'artefact) ; None → défaut."""
        if name is None:
            name = self.asr_default_model
        if not name:
            return None
        return self.asr_models.get(name)

    # ---------------------------------------------------------------- lecture
    def resolve(self, name: str) -> Optional[ModelProfile]:
        """Résout un nom demandé par le client.

        1. nom logique du registre ;
        2. sinon `ollama_name` (avec l'équivalence `x` / `x:latest`) ;
        3. sinon None — l'appelant décide (mode legacy ou refus strict).
        """
        if not name:
            return None
        if name in self.models:
            return self.models[name]
        for profile in self.models.values():
            if profile.matches_ollama_name(name):
                return profile
        return None

    def embedding_models(self) -> List[ModelProfile]:
        return [p for p in self.models.values() if p.kind == "embedding"]

    def vision_profiles(self) -> List[LLMProfile]:
        return [p for p in self.llm_models.values() if p.supports_image]

    def resolve_vision(self, name: str) -> Optional[LLMProfile]:
        """Résout un nom LOGIQUE de profil vision. Le nom Ollama concret n'est
        volontairement PAS accepté ici : la sémantique de requête gelée est
        « profil logique du registre » (contrat §20)."""
        profile = self.llm_models.get(name)
        if profile is not None and profile.supports_image:
            return profile
        return None

    def resolve_llm(self, name: str) -> Optional[ResolvedLLM]:
        """Résout un nom de modèle de GÉNÉRATION vers son profil gouverné.

        1. nom logique (`qwen-judge`) → profil, `via_logical=True` ;
        2. nom Ollama (`qwen3:14b`, équivalence `x` ≡ `x:latest`) → profil,
           `via_logical=False` : les défauts opérationnels (contexte,
           keep_alive) s'appliquent, mais PAS `default_think` ;
        3. sinon None — le modèle est relayé tel quel, comportement historique
           intégral (aucun profil, aucune restriction).
        """
        if not name:
            return None
        if name in self.llm_models:
            return ResolvedLLM(profile=self.llm_models[name], via_logical=True)
        for profile in self.llm_models.values():
            if profile.matches_ollama_name(name):
                return ResolvedLLM(profile=profile, via_logical=False)
        return None

    def resolve_reranker(self, name: str) -> Optional[RerankerProfile]:
        """Résout un nom logique de reranker. Strictement le nom logique :
        contrairement aux modèles Ollama, il n'existe aucun alias naturel."""
        if not name:
            return None
        return self.reranker_models.get(name)

    def context_profile_of(self, profile: LLMProfile) -> ContextProfile:
        return self.context_profiles[profile.context_profile]

    def generation_profile_of(self, profile: LLMProfile) -> GenerationProfile:
        """Budget de sortie du modèle. Jamais None : `generation_profile` est
        obligatoire et sa référence est validée au chargement."""
        return self.generation_profiles[profile.generation_profile]

    def keep_alive_of(self, profile: LLMProfile) -> KeepAliveProfile:
        return self.keep_alive_profiles[profile.keep_alive_profile]

    def reasoning_profile_of(self, profile: LLMProfile) -> ReasoningProfile:
        """Politique de raisonnement du modèle (lot C1). Un modèle sans
        `reasoning_profile` porte le profil implicite `none` : aucun mode
        pilotable, aucun `think` relayé."""
        if profile.reasoning_profile is None:
            return IMPLICIT_NONE
        return self.reasoning_profiles[profile.reasoning_profile]

    # ------------------------------------------------------- vérif du digest
    def verify_digests(self, client, timeout: int = 5) -> None:
        """Compare le digest attendu de chaque modèle à celui installé.

        Ne lève jamais : un incident de transport laisse les profils gouvernés
        (statut `unavailable`), seule une divergence prouvée déclasse un modèle.
        """
        try:
            installed = client.list_models(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - on ne veut aucune exception ici
            with self._lock:
                for profile in self.models.values():
                    profile.status = STATUS_UNAVAILABLE
                    profile.observed_digest = None
                    profile.status_detail = "upstream injoignable"
                    profile.last_checked = time.time()
            logger.warning(
                "registry: vérification des digests impossible (%s)", type(exc).__name__
            )
            return

        by_name: Dict[str, Dict[str, Any]] = {}
        for entry in installed:
            name = entry.get("name")
            if name:
                by_name[name] = entry

        now = time.time()
        with self._lock:
            self.inventory = set(by_name)
            self.inventory_at = now
            for profile in self.models.values():
                entry = None
                for alias in profile._name_aliases():
                    if alias in by_name:
                        entry = by_name[alias]
                        break

                profile.last_checked = now
                if entry is None:
                    profile.status = STATUS_UNAVAILABLE
                    profile.observed_digest = None
                    profile.status_detail = "modèle absent de l'upstream"
                    logger.warning(
                        "registry: modèle absent logical=%s ollama_name=%s",
                        profile.logical_name,
                        profile.ollama_name,
                    )
                    continue

                observed = (entry.get("digest") or "").strip().lower()
                if observed.startswith("sha256:"):
                    observed = observed[len("sha256:"):]
                profile.observed_digest = observed or None

                if not observed:
                    profile.status = STATUS_UNAVAILABLE
                    profile.status_detail = "digest non rapporté par l'upstream"
                elif observed == profile.digest:
                    profile.status = STATUS_READY
                    profile.status_detail = None
                    profile.verified_at = now          # seule issue qui prouve
                else:
                    profile.status = STATUS_DEGRADED
                    profile.status_detail = "digest divergent du registre"
                    logger.error(
                        "registry: DIGEST DIVERGENT logical=%s ollama_name=%s "
                        "expected=%s observed=%s action=retire_de_l_allowlist",
                        profile.logical_name,
                        profile.ollama_name,
                        profile.short_digest,
                        observed[:8],
                    )

    def status_of(self, logical_name: str) -> str:
        profile = self.models.get(logical_name)
        return profile.status if profile else STATUS_UNAVAILABLE


# --------------------------------------------------------------------------- #
# Fabrique
# --------------------------------------------------------------------------- #
def load_registry(path: str) -> ModelRegistry:
    """Charge et valide le registre. Lève RegistryError si invalide."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.load(handle, Loader=_NoDuplicateLoader)
    except RegistryError:
        raise
    except FileNotFoundError as exc:
        raise RegistryError(f"registre introuvable : {path}") from exc
    except yaml.YAMLError as exc:
        # On ne propage pas l'objet d'exception brut : il peut contenir des
        # extraits du fichier. Seule la position est utile.
        mark = getattr(exc, "problem_mark", None)
        where = f" (ligne {mark.line + 1})" if mark is not None else ""
        raise RegistryError(f"registre YAML illisible{where}") from exc

    _require(isinstance(raw, dict), "le registre doit être un mapping YAML")

    version = raw.get("version")
    _require(version is not None, "le registre doit déclarer un champ 'version'")
    _require(
        version in SUPPORTED_VERSIONS,
        f"version de registre inconnue : {version!r} "
        f"(supportées : {sorted(SUPPORTED_VERSIONS)})",
    )

    models_raw = raw.get("models")
    _require(isinstance(models_raw, dict) and models_raw, "le registre ne déclare aucun modèle")

    models: Dict[str, ModelProfile] = {}
    for logical, entry in models_raw.items():
        profile = _build_profile(logical, entry)
        models[logical] = profile

    # Deux noms logiques ne doivent pas viser le même artefact : le pipeline_id
    # deviendrait ambigu et la résolution par ollama_name non déterministe.
    seen_aliases: Dict[str, str] = {}
    for profile in models.values():
        for alias in profile._name_aliases():
            if alias in seen_aliases and seen_aliases[alias] != profile.logical_name:
                raise RegistryError(
                    f"ollama_name ambigu : {alias!r} est visé par "
                    f"{seen_aliases[alias]!r} et {profile.logical_name!r}"
                )
            seen_aliases[alias] = profile.logical_name

    # ------------------------- sections LLM, toutes optionnelles ------------ #
    context_profiles: Dict[str, ContextProfile] = {}
    raw_ctx = raw.get("context_profiles") or {}
    _require(isinstance(raw_ctx, dict), "context_profiles doit être un mapping")
    for name, entry in raw_ctx.items():
        context_profiles[name] = _build_context_profile(name, entry)

    generation_profiles: Dict[str, GenerationProfile] = {}
    raw_gen = raw.get("generation_profiles") or {}
    _require(isinstance(raw_gen, dict), "generation_profiles doit être un mapping")
    for name, entry in raw_gen.items():
        generation_profiles[name] = _build_generation_profile(name, entry)

    keep_alive_profiles: Dict[str, KeepAliveProfile] = {}
    raw_ka = raw.get("keep_alive_profiles") or {}
    _require(isinstance(raw_ka, dict), "keep_alive_profiles doit être un mapping")
    for name, entry in raw_ka.items():
        keep_alive_profiles[name] = _build_keep_alive_profile(name, entry)

    reasoning_profiles: Dict[str, ReasoningProfile] = {}
    raw_reasoning = raw.get("reasoning_profiles") or {}
    _require(isinstance(raw_reasoning, dict), "reasoning_profiles doit être un mapping")
    for name, entry in raw_reasoning.items():
        _require(isinstance(name, str) and name.strip() != "",
                 "un nom de reasoning_profile ne peut pas être vide")
        try:
            reasoning_profiles[name] = build_reasoning_profile(name, entry)
        except ValueError as exc:
            raise RegistryError(str(exc)) from exc

    llm_models: Dict[str, LLMProfile] = {}
    raw_llm = raw.get("llm_models") or {}
    _require(isinstance(raw_llm, dict), "llm_models doit être un mapping")
    for logical, entry in raw_llm.items():
        llm_models[logical] = _build_llm_profile(
            logical, entry, context_profiles, keep_alive_profiles,
            generation_profiles, reasoning_profiles,
        )

    # Unicité des alias LLM entre eux, et disjonction avec les embeddings :
    # un même nom Ollama résolu tantôt en profil d'embedding tantôt en profil
    # LLM rendrait la gouvernance dépendante de l'endpoint appelé.
    llm_aliases: Dict[str, str] = {}
    for profile in llm_models.values():
        for alias in profile._name_aliases() | {profile.logical_name}:
            if alias in llm_aliases and llm_aliases[alias] != profile.logical_name:
                raise RegistryError(
                    f"nom LLM ambigu : {alias!r} est visé par "
                    f"{llm_aliases[alias]!r} et {profile.logical_name!r}"
                )
            llm_aliases[alias] = profile.logical_name
            if alias in seen_aliases:
                raise RegistryError(
                    f"nom ambigu entre registres : {alias!r} est à la fois un "
                    f"modèle d'embedding ({seen_aliases[alias]!r}) et un LLM "
                    f"({profile.logical_name!r})"
                )

    reranker_models: Dict[str, RerankerProfile] = {}
    raw_rerank = raw.get("reranker_models") or {}
    _require(isinstance(raw_rerank, dict), "reranker_models doit être un mapping")
    for logical, entry in raw_rerank.items():
        reranker_models[logical] = _build_reranker_profile(logical, entry)

    # Unicité inter-registres : un nom logique de reranker ne doit entrer en
    # collision ni avec un alias d'embedding ni avec un nom/alias LLM — la
    # gouvernance ne doit jamais dépendre de l'endpoint appelé.
    for logical in reranker_models:
        if logical in models or logical in seen_aliases:
            raise RegistryError(
                f"nom ambigu entre registres : {logical!r} est à la fois un "
                "modèle d'embedding (nom logique ou alias) et un reranker"
            )
        if logical in llm_aliases:
            raise RegistryError(
                f"nom ambigu entre registres : {logical!r} est à la fois un "
                f"LLM ({llm_aliases[logical]!r}) et un reranker"
            )

    vision_defaults = raw.get("vision_defaults")
    vision_default_model = None
    if vision_defaults is not None:
        if not isinstance(vision_defaults, dict):
            raise RegistryError("vision_defaults doit être un mapping")
        vision_default_model = vision_defaults.get("default_model")
        if vision_default_model is not None:
            if vision_default_model not in llm_models:
                raise RegistryError(
                    f"vision_defaults.default_model inconnu : {vision_default_model!r}"
                )
            if "image" not in llm_models[vision_default_model].modalities:
                raise RegistryError(
                    f"vision_defaults.default_model {vision_default_model!r} "
                    "n'a pas la modalité image"
                )

    asr_models: Dict[str, AsrModelProfile] = {}
    raw_asr = raw.get("asr_models") or {}
    _require(isinstance(raw_asr, dict), "asr_models doit être un mapping")
    for logical, entry in raw_asr.items():
        asr_models[logical] = _build_asr_profile(logical, entry)
    for logical in asr_models:
        if logical in models or logical in seen_aliases or logical in llm_aliases or logical in reranker_models:
            raise RegistryError(f"nom ambigu entre registres : {logical!r} est déjà un embedding, un LLM ou un reranker")
    asr_default_model = None
    asr_defaults = raw.get("asr_defaults")
    if asr_defaults is not None:
        _require(isinstance(asr_defaults, dict) and set(asr_defaults) <= {"model"}, "asr_defaults ne peut contenir que `model`")
        asr_default_model = asr_defaults.get("model")
        _require(asr_default_model in asr_models, f"asr_defaults.model inconnu : {asr_default_model!r}")
    _require(not asr_models or asr_default_model is not None, "asr_defaults.model est obligatoire dès qu'un asr_models existe")

    registry = ModelRegistry(
        version=version,
        models=models,
        source_path=path,
        context_profiles=context_profiles,
        keep_alive_profiles=keep_alive_profiles,
        generation_profiles=generation_profiles,
        reasoning_profiles=reasoning_profiles,
        llm_models=llm_models,
        reranker_models=reranker_models,
        vision_default_model=vision_default_model,
        asr_models=asr_models,
        asr_default_model=asr_default_model,
    )
    logger.info(
        "registry: chargé version=%s modèles=%d llm=%d rerankers=%d asr=%d profils_ctx=%d "
        "profils_gen=%d source=%s",
        version,
        len(models),
        len(llm_models),
        len(reranker_models),
        len(asr_models),
        len(context_profiles),
        len(generation_profiles),
        path,
    )
    return registry
