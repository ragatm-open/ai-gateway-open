"""Helpers d'erreurs propres et observables.

Toutes les erreurs exposées suivent le format :

    {"detail": {"status": "error", "provider": "...", "message": "..."}}
"""

from typing import Any, Optional

from fastapi import HTTPException


class UpstreamError(Exception):
    """Erreur en provenance de l'upstream Ollama."""

    def __init__(self, message: str, body: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.body = body


# --------------------------------------------------------------------------- #
# Erreurs propres au streaming
# --------------------------------------------------------------------------- #
# Sous-classes d'UpstreamError : un `except UpstreamError` existant continue de
# les intercepter. Elles n'existent que pour permettre à la route de qualifier
# la *terminaison* du flux dans ses logs — un budget épuisé et une ligne
# illisible ne se diagnostiquent pas de la même façon.


class StreamTimeout(UpstreamError):
    """Le budget de temps mural du flux est épuisé.

    À distinguer d'un timeout de lecture : Ollama peut très bien émettre
    régulièrement quelques octets et dépasser malgré tout le plafond global.
    """


class UpstreamTimeout(UpstreamError):
    """Le budget de lecture de l'appel NON streamé est épuisé.

    Sous-classe d'UpstreamError : tout `except UpstreamError` existant continue
    de l'intercepter et le contrat 502 reste inchangé. Elle n'existe que pour
    permettre à la route de QUALIFIER la terminaison dans ses logs.

    Distinction essentielle depuis la RCA du 2026-09-22 : un timeout n'est pas
    une panne d'Ollama. Les deux incidents qwen ont vu le modèle générer sans
    interruption à ~170 tok/s pendant 180 s — la Gateway attendait une réponse
    qui ne venait pas parce que la génération n'était pas bornée, pas parce que
    l'upstream était bloqué. Confondre les deux dans un seul `generate: échec
    upstream` a coûté une investigation entière.
    """


class InvalidUpstreamChunk(UpstreamError):
    """Une ligne du flux Ollama n'est pas du JSON exploitable.

    Le message ne contient JAMAIS la ligne fautive : elle porte potentiellement
    du texte généré, donc du contenu utilisateur.
    """


class IncompleteUpstreamStream(UpstreamError):
    """Le flux s'est terminé sans jamais annoncer `done: true`.

    Traité comme une erreur et non comme un succès : la génération a été
    interrompue en amont, le client doit pouvoir le savoir.
    """


class UpstreamModelNotFound(UpstreamError):
    """Le runtime répond qu'il ne possède PAS le modèle demandé.

    Ollama rend `404 {"error": "model \"x\" not found, try pulling it first"}`.
    Distinguée d'une panne : la Gateway ne télécharge JAMAIS un modèle à la
    place de l'exploitant — l'état est rendu explicitement au client
    (`503 model_not_installed`, non réessayable) au lieu d'un 502 générique.
    """


class UnsupportedUpstreamResponse(UpstreamError):
    """L'upstream a répondu une forme que le contrat public ne couvre pas.

    Lot C1 : `/api/chat` peut émettre `tool_calls` alors qu'aucun outil n'a
    été transmis (hallucination d'appel d'outil). Plutôt que d'exposer en
    silence un champ non documenté — ou de le laisser tomber en faisant
    passer une réponse vide pour un succès — la route le refuse
    explicitement. Le contrat tools appartient au lot C4.
    """


# --------------------------------------------------------------------------- #
# Messages PUBLICS des pannes upstream
# --------------------------------------------------------------------------- #
# Une exception réseau (`requests`) se décrit avec l'hôte, le port et le
# chemin interne : "HTTPConnectionPool(host='127.0.0.1', port=11434): …". Ce
# texte est utile à l'opérateur — il reste INTÉGRALEMENT dans les logs — mais
# n'a rien à faire dans une réponse HTTP : la doc promet « aucune URL interne
# dans les erreurs ». Les routes n'exposent donc que ces chaînes STABLES ; le
# champ `provider` dit déjà quel service a failli.
PUBLIC_UPSTREAM_UNAVAILABLE = "service upstream indisponible"
PUBLIC_GENERATION_FAILED = "échec du service de génération"
PUBLIC_GENERATION_TIMEOUT = "échec du service de génération : délai dépassé"
PUBLIC_EMBEDDING_FAILED = "échec du service d'embedding"
PUBLIC_EMBEDDING_BUDGET_EXHAUSTED = "échec du service d'embedding : budget de temps épuisé"
PUBLIC_RERANK_FAILED = "échec du service de reranking"
PUBLIC_RERANK_BACKEND_UNREACHABLE = "backend de reranking injoignable"
PUBLIC_RERANK_PROBE_FAILED = "sonde de scoring en échec"
PUBLIC_VISION_FAILED = "échec du service d'analyse d'image"
# modèle demandé absent du runtime local : aucun téléchargement
# implicite, l'exploitant doit l'installer (`ollama pull`).
PUBLIC_MODEL_NOT_INSTALLED = "modèle non installé sur le runtime local"
# Lot C1 — réponse `/api/chat` hors contrat (tool_calls sans outil déclaré).
PUBLIC_CHAT_UNSUPPORTED_RESPONSE = "réponse du service de génération non supportée"
# Lot C3 — sortie structurée : la réponse du modèle ne satisfait pas le
# contrat demandé. Messages FIXES : ni contenu généré, ni schéma.
PUBLIC_STRUCTURED_EMPTY = "sortie structurée : le modèle n'a produit aucun contenu"
PUBLIC_STRUCTURED_INVALID = "sortie structurée : le contenu n'est pas un document JSON strict"
PUBLIC_STRUCTURED_MISMATCH = "sortie structurée : document JSON non conforme au schéma demandé"


def http_error(
    status_code: int,
    message: str,
    provider: str = "ollama",
    extra: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> HTTPException:
    detail: dict[str, Any] = {
        "status": "error",
        "provider": provider,
        "message": message,
    }
    if extra:
        detail.update(extra)
    # `headers` : porte `Retry-After` sur une saturation. FastAPI les
    # relaie tels quels dans son handler HTTPException par défaut.
    return HTTPException(status_code=status_code, detail=detail, headers=headers)
