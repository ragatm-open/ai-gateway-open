"""Routes FastAPI de l'AI Gateway Open.

Endpoints :
    GET  /health/live        – processus vivant, registre chargé (JAMAIS d'appel upstream)
    GET  /health/runtime     – gouvernance, états de capacités, concurrence (JAMAIS d'appel upstream)
    GET  /health             – liveness + inventaire Ollama (/api/tags, non inférentiel)
    GET  /health/ready       – readiness : embedding réellement calculé et validé (sonde INFÉRENTIELLE explicite)
    GET  /health/embeddings  – diagnostic détaillé des modèles gouvernés (sonde INFÉRENTIELLE explicite)
    GET  /v1/models          – liste des modèles installés (+ identité du registre)
    POST /v1/generate        – génération texte (réponse agrégée)
    POST /v1/generate/stream – génération texte en flux NDJSON
    POST /v1/chat            – chat multi-tours gouverné (lot C1, /api/chat)
    POST /v1/chat/stream     – chat en flux NDJSON (thinking / content / done)
    POST /v1/embeddings      – embeddings pour indexation RAG
    POST /v1/rerank          – reranking query/passages (capability optionnelle)
    GET  /health/rerank      – diagnostic de la capability reranking (sonde INFÉRENTIELLE explicite)

Profil Open : runtime local uniquement (Ollama + backend de
reranking sur la boucle locale), aucun fournisseur distant, aucun repli cloud,
aucun téléchargement de modèle pendant une requête ; un modèle absent du
runtime est rendu `503 model_not_installed`. Chaque réponse porte
`X-Request-ID` (relayé s'il est valide, généré sinon).

Sécurité : la base URL Ollama n'est jamais exposée (masquée `redacted`).
Aucune donnée utilisateur (prompt, textes, vecteurs) n'est journalisée.

Lot 1 — le service d'embedding devient *gouverné* : chaque modèle servi est
décrit par un profil du registre (dimension, digest, préfixes, fenêtre
d'entrée), chaque réponse porte l'identité du pipeline qui l'a produite, et
aucun vecteur inexploitable ne sort de la Gateway.
"""

import base64
import binascii
import json
import logging
import math
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

from datetime import datetime, timezone

import PIL
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from . import __version__
from .errors import (
    PUBLIC_EMBEDDING_BUDGET_EXHAUSTED,
    PUBLIC_EMBEDDING_FAILED,
    PUBLIC_GENERATION_FAILED,
    PUBLIC_GENERATION_TIMEOUT,
    PUBLIC_RERANK_BACKEND_UNREACHABLE,
    PUBLIC_RERANK_FAILED,
    PUBLIC_RERANK_PROBE_FAILED,
    PUBLIC_UPSTREAM_UNAVAILABLE,
    PUBLIC_CHAT_UNSUPPORTED_RESPONSE,
    PUBLIC_STRUCTURED_EMPTY,
    PUBLIC_STRUCTURED_INVALID,
    PUBLIC_STRUCTURED_MISMATCH,
    PUBLIC_VISION_FAILED,
    PUBLIC_MODEL_NOT_INSTALLED,
    IncompleteUpstreamStream,
    InvalidUpstreamChunk,
    StreamTimeout,
    UnsupportedUpstreamResponse,
    UpstreamError,
    UpstreamModelNotFound,
    UpstreamTimeout,
    http_error,
)
from .metrics import (
    HOST_ADMISSION_REAPER_PENDING_GAUGE,
    business_inflight,
    ERROR_CLIENT,
    ERROR_INTERNAL,
    ERROR_UPSTREAM,
    count_error,
    counters,
    embedding_throughput,
    gauges,
    load_state,
    log_event,
    normalize_embedding_metrics,
    normalize_generation_metrics,
    safe_int,
)
from .ollama_client import (
    ENDPOINT_EMBED,
    ENDPOINT_EMBEDDINGS,
    ChatStream,
    GenerateStream,
    OllamaClient,
    client,
)
from .reasoning import ReasoningModeError, resolve_reasoning
from .structured import (
    ERR_OUTPUT_EMPTY,
    ERR_OUTPUT_INVALID,
    ERR_OUTPUT_MISMATCH,
    SchemaError,
    SchemaStats,
    StructuredOutputError,
    schema_sha256,
    validate_client_schema,
    validate_structured_output,
)
from .rerank_client import BACKEND_IMPLEMENTATION, backend_url_is_loopback, rerank_client
from . import asr_client as asr_transport
from .asr_client import AsrBackendError, AsrBackendUnavailable
from . import asr_routes
from .registry import (
    DEFAULT_TASK,
    STATUS_DEGRADED,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    ContextProfile,
    GenerationProfile,
    ModelProfile,
    RegistryError,
    load_registry,
)
from .schemas import (
    ChatAssistantMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamContent,
    ChatStreamDone,
    ChatStreamThinking,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    GenerateRequest,
    GenerateResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankScoreSemantics,
    RerankUsage,
    GenerateStreamDelta,
    GenerateStreamDone,
    GenerateStreamError,
    StreamErrorDetail,
    TokenCounts,
    Usage,
)
from .settings import settings
from .vision_backend import (
    QUALIFIED_FIRST_ATTEMPT,
    QUALIFIED_RETRY_ATTEMPT,
    VisionBackendTimeout,
    VisionInferencePlan,
    vision_backend,
)
from .vision_prompts import QUALIFIED_PROMPTS, SYSTEM_PROMPT
from .vision_admission import VisionAdmission
from .host_admission import HostAdmissionError, host_admission, host_admission_http_error
from .vision_image import NORMALIZED_FORMAT, NORMALIZER_ID, ImageAdmissionError, admit_image
from .vision_provenance import MODEL_SCHEMA_HASH_ALGORITHM, pipeline_provenance
from .vision_invariants import (
    VisionInvariantError,
    collect_warnings,
    derive_requires_visual_check,
    normalize_unambiguous_unknown,
    validate_semantic_output,
)
from .vision_schemas import (
    ALLOWED_KINDS_BY_TASK,
    PROMPT_PROFILES,
    output_schema_for_task,
    vision_result_adapter,
    SEMANTIC_SCHEMA_VERSION,
    SEMANTIC_TASK,
    RawModelOutput,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionAnalyzeResponseV2,
    VisionCapacity,
    VisionCapacityModel,
    VisionCapacityResponse,
    VisionContext,
    VisionImageDiagnostics,
    VisionImageDiagnosticsV2,
    VisionModelIdentity,
    VisionPipelineIdentity,
    VisionPipelineIdentityV2,
    VisionUsage,
    VisionUsageV2,
)
from .vectors import (
    NORM_MAX,
    NORM_MIN,
    VectorError,
    l2_norm,
    l2_normalize,
    norm_in_tolerance,
    validate_vectors,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ai_gateway")

app = FastAPI(title="AI Gateway Open", version=__version__)

# --------------------------------------------------------------------------- #
# Middleware ASGI : corrélation X-Request-ID + borne de taille.
# --------------------------------------------------------------------------- #
# Middleware ASGI PUR (pas BaseHTTPMiddleware) : il n'intercepte pas le corps
# et ne modifie pas la sémantique d'annulation des flux NDJSON (le générateur
# async de la route reste celui que Starlette annule au départ du client).
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
# Routes qui bornent ELLES-MÊMES leur corps : multipart ASR (max_upload_bytes)
# et vision (vision_max_image_base64_chars contrôlé par Pydantic avant décodage).
_BODY_LIMIT_EXEMPT_PREFIXES = ("/v1/audio/", "/v1/vision/")
_PROCESS_STARTED = time.monotonic()


def request_id_or_new(value: Optional[str]) -> str:
    """Identifiant RELAYÉ s'il respecte l'alphabet borné, GÉNÉRÉ sinon.

    Un identifiant invalide (vide, trop long, caractère de contrôle, espace)
    n'est jamais relayé ni journalisé : il est remplacé — un `\n` dans un
    en-tête relayé fabriquerait une fausse ligne de log (doctrine P13).
    """
    if value is not None and _REQUEST_ID_RE.match(value):
        return value
    return str(uuid.uuid4())


def _error_response_bytes(status: int, message: str, code: str, **extra: Any) -> bytes:
    detail: Dict[str, Any] = {"status": "error", "provider": "ai_gateway", "message": message, "code": code}
    detail.update(extra)
    return json.dumps({"detail": detail}, ensure_ascii=False).encode("utf-8")


class RequestEnvelopeMiddleware:
    """Pose `X-Request-ID` sur toute réponse et refuse les corps trop grands.

    Borne de taille : `Content-Length` > `AI_GATEWAY_REQUEST_MAX_BYTES` =>
    413 `request_too_large` AVANT toute lecture de corps ; un POST JSON sans
    `Content-Length` (transfert par morceaux) => 411 `length_required`. Les
    routes multipart ASR sont exemptées (elles bornent elles-mêmes l'upload).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        request_id = request_id_or_new(headers.get(REQUEST_ID_HEADER.lower()))
        scope.setdefault("state", {})["request_id"] = request_id
        header_bytes = (REQUEST_ID_HEADER.encode("latin-1"), request_id.encode("latin-1"))

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                raw = [h for h in message.get("headers", []) if h[0].lower() != b"x-request-id"]
                raw.append(header_bytes)
                message = {**message, "headers": raw}
            await send(message)

        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()
        if method in ("POST", "PUT", "PATCH") and not path.startswith(_BODY_LIMIT_EXEMPT_PREFIXES):
            length_raw = headers.get("content-length")
            reject: Optional[Tuple[int, bytes]] = None
            if length_raw is None:
                if "chunked" in headers.get("transfer-encoding", "").lower():
                    reject = (411, _error_response_bytes(
                        411, "Content-Length requis", "length_required"))
            else:
                try:
                    length = int(length_raw)
                except ValueError:
                    length = -1
                if length < 0:
                    reject = (400, _error_response_bytes(400, "Content-Length invalide", "invalid_content_length"))
                elif length > settings.request_max_bytes:
                    reject = (413, _error_response_bytes(
                        413, "corps de requête trop volumineux", "request_too_large",
                        max_bytes=settings.request_max_bytes))
            if reject is not None:
                status, body = reject
                await send_with_id({"type": "http.response.start", "status": status, "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send_with_id)


app.add_middleware(RequestEnvelopeMiddleware)

# --------------------------------------------------------------------------- #
# Registre
# --------------------------------------------------------------------------- #
# Chargement au démarrage, volontairement bloquant : le registre fait partie de
# l'image (app/models_registry.yaml). S'il est illisible, l'image est cassée et
# doit être détectée en staging, pas servir des embeddings non gouvernés en
# silence. Une divergence de digest, elle, ne fait PAS tomber le service
# (cf. app/registry.py) : elle déclasse le seul modèle concerné.
try:
    registry = load_registry(settings.models_registry_path)
except RegistryError as exc:
    logger.error("registre de modèles invalide : %s", exc)
    raise RuntimeError(f"registre de modèles invalide : {exc}") from exc

# Contrôles de démarrage FAIL-CLOSED : les défauts d'environnement
# doivent être cohérents avec le registre AVANT de servir — une erreur de
# configuration se voit au démarrage, pas à la première requête.
_default_embedding_profile = registry.resolve(settings.embedding_model)
if settings.embeddings_strict_registry and _default_embedding_profile is None:
    raise RuntimeError(
        f"configuration invalide : OLLAMA_EMBEDDING_MODEL={settings.embedding_model!r} absent du "
        "registre alors que AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY est actif"
    )
if (_default_embedding_profile is not None
        and _default_embedding_profile.dimension != settings.embeddings_dimension):
    raise RuntimeError(
        f"configuration invalide : le modèle d'embedding par défaut {settings.embedding_model!r} produit "
        f"{_default_embedding_profile.dimension} dimensions, l'index attend "
        f"AI_GATEWAY_EMBEDDINGS_DIMENSION={settings.embeddings_dimension}"
    )
if settings.generation_strict_registry and registry.resolve_llm(settings.generate_model) is None:
    raise RuntimeError(
        f"configuration invalide : OLLAMA_GENERATE_MODEL={settings.generate_model!r} absent du "
        "registre alors que AI_GATEWAY_GENERATION_STRICT_REGISTRY est actif"
    )

# Client dédié aux sondes de santé : timeout court (5 s) indépendant du timeout
# métier (AI_GATEWAY_TIMEOUT_SECONDS).
probe_client = OllamaClient(settings.ollama_base_url, settings.readiness_timeout_seconds)

_verify_lock = threading.Lock()
_verify_state = {"last": 0.0}

_readiness_lock = threading.Lock()
_readiness_cache: Dict[str, Any] = {"expires": 0.0, "status": 503, "payload": None}

PROBE_TEXT = "controle de sante de la passerelle"


def _ensure_registry_verified(force: bool = False) -> None:
    """Vérifie les digests au plus une fois par intervalle configuré.

    Ne lève jamais et ne bloque jamais une requête : si une autre vérification
    est déjà en cours, on passe son tour.
    """
    now = time.monotonic()
    if not force and _verify_state["last"] and (
        now - _verify_state["last"] < settings.registry_verify_interval_seconds
    ):
        return
    if not _verify_lock.acquire(blocking=False):
        return
    try:
        before = {p.logical_name: p.status for p in registry.models.values()}
        registry.verify_digests(probe_client, timeout=settings.readiness_timeout_seconds)
        for name, profile in registry.models.items():
            if profile.status == STATUS_DEGRADED and before.get(name) != STATUS_DEGRADED:
                counters.inc("registry_digest_mismatch_total")
        _verify_state["last"] = time.monotonic()
    finally:
        _verify_lock.release()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Vérification initiale des digests, best effort : un upstream indisponible
    # au démarrage ne doit pas empêcher le process de servir /health.
    try:
        _ensure_registry_verified(force=True)
    except Exception:  # noqa: BLE001
        logger.warning("startup: vérification initiale des digests impossible")
    yield


app.router.lifespan_context = _lifespan


# --------------------------------------------------------------------------- #
# Handlers d'erreurs -> format {"detail": {...}}
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def _validation_handler(_request, exc: RequestValidationError) -> JSONResponse:
    # 400 sur payload invalide (au lieu du 422 par défaut) pour RagATM.
    if _request.url.path == "/v1/embeddings":
        count_error(ERROR_CLIENT)
    errors = jsonable_encoder(exc.errors())
    if _request.url.path.startswith("/v1/vision"):
        # Privacy (lot vision-1a, contrat §79) : jamais d'écho de l'input —
        # un base64 d'image ou un contexte invalides ne ressortent ni dans la
        # réponse ni dans les logs. Type + localisation suffisent au client.
        errors = [
            {k: v for k, v in error.items() if k in ("type", "loc", "msg")}
            for error in errors
        ]
    return JSONResponse(
        status_code=400,
        content={
            "detail": {
                "status": "error",
                "provider": "ai_gateway",
                "message": "payload invalide",
                # jsonable_encoder : certains validateurs custom placent l'objet
                # ValueError dans `ctx`, non sérialisable par json.dumps -> 500.
                # On sérialise proprement pour garantir un vrai 400.
                "errors": errors,
            }
        },
    )


@app.exception_handler(Exception)
async def _unexpected_handler(_request, exc: Exception) -> JSONResponse:
    """Filet pour les exceptions non prévues : 500 dans l'enveloppe habituelle.

    La trace part dans les logs, jamais dans la réponse — elle contiendrait des
    chemins internes et, potentiellement, des fragments de texte utilisateur.
    """
    if _request.url.path == "/v1/embeddings":
        count_error(ERROR_INTERNAL)
    logger.exception("erreur interne non prévue sur %s", _request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "status": "error",
                "provider": "ai_gateway",
                "message": f"erreur interne ({type(exc).__name__})",
            }
        },
    )


# --------------------------------------------------------------------------- #
# Health — vivacité STRICTEMENT non inférentielle
# --------------------------------------------------------------------------- #
@app.get("/health/live")
def health_live() -> Dict[str, Any]:
    """Processus vivant et registre chargé. AUCUN appel réseau, AUCUNE
    inférence, AUCUN chargement : c'est la sonde destinée au healthcheck
    Docker et au monitoring automatique. Les sondes inférentielles
    (/health/ready, /health/embeddings, /health/rerank) sont explicites et ne
    doivent jamais être branchées sur un battement automatique."""
    return {
        "status": "ok",
        "provider": "ai_gateway",
        "version": __version__,
        "registry_loaded": True,
        "registry_version": registry.version,
        "uptime_seconds": int(time.monotonic() - _PROCESS_STARTED),
        "inferential": False,
    }


# --------------------------------------------------------------------------- #
# Health — liveness (contrat historique inchangé)
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, Any]:
    started = time.time()
    try:
        _, names = client.ping(timeout=5)
    except UpstreamError as exc:
        logger.warning("health: upstream indisponible: %s", exc.message)
        raise http_error(503, PUBLIC_UPSTREAM_UNAVAILABLE, provider="ollama")

    default_profile = registry.resolve(settings.embedding_model)
    return {
        "status": "ok",
        "provider": "ollama",
        "ollama_base_url": "redacted",
        "default_generate_model": settings.generate_model,
        "default_embedding_model": settings.embedding_model,
        "models": names,
        "latency_ms": int((time.time() - started) * 1000),
        # Ajout additif : état du registre, sans appel upstream supplémentaire
        # (les statuts sont ceux du dernier contrôle en cache).
        "registry": {
            "version": registry.version,
            "models": len(registry.models),
            "default_embedding_logical": (
                default_profile.logical_name if default_profile else None
            ),
            "default_embedding_status": (
                default_profile.status if default_profile else STATUS_UNAVAILABLE
            ),
            "default_embedding_identity_verified": (
                default_profile.identity_verified if default_profile else False
            ),
            "strict": settings.embeddings_strict_registry,
            "environment": settings.environment,
        },
    }


# --------------------------------------------------------------------------- #
# Health — readiness (sonde d'embedding réelle, mise en cache)
# --------------------------------------------------------------------------- #
def _probe_embedding(profile: ModelProfile, task: str = DEFAULT_TASK) -> Dict[str, Any]:
    """Encode un texte de sonde et contrôle le vecteur obtenu.

    Retourne un dict de diagnostic ; ne lève pas. Aucun texte métier n'est
    utilisé : la sonde encode une constante interne.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "dimension": None,
        "norm": None,
        "latency_ms": None,
        "upstream_endpoint": None,
        "error": None,
    }
    text = profile.prefix_for(task) + PROBE_TEXT
    started = time.time()
    try:
        vectors = probe_client.embed(model=profile.ollama_name, inputs=[text])
    except UpstreamError as exc:
        # Détail (chemin, port) pour l'opérateur dans les logs ; le payload de
        # diagnostic, lui, est une réponse HTTP — message neutre.
        logger.warning("health/embeddings: sonde en échec: %s", exc.message)
        result["error"] = PUBLIC_UPSTREAM_UNAVAILABLE
        return result
    except Exception as exc:  # noqa: BLE001 - la sonde ne doit jamais propager
        result["error"] = f"erreur upstream ({type(exc).__name__})"
        return result

    result["latency_ms"] = int((time.time() - started) * 1000)
    result["upstream_endpoint"] = getattr(vectors, "upstream_endpoint", ENDPOINT_EMBED)

    try:
        dimension = validate_vectors(
            vectors,
            expected_count=1,
            expected_dimension=profile.dimension,
            require_normalized=False,
        )
    except VectorError as exc:
        result["error"] = str(exc)
        return result

    vector = vectors[0]
    if result["upstream_endpoint"] == ENDPOINT_EMBEDDINGS and profile.normalized:
        try:
            vector = l2_normalize(vector)
        except VectorError as exc:
            result["error"] = str(exc)
            return result

    norm = l2_norm(vector)
    result["dimension"] = dimension
    result["norm"] = round(norm, 6)

    if profile.normalized and not norm_in_tolerance(norm):
        result["error"] = (
            f"norme {norm:.6f} hors tolérance [{NORM_MIN}, {NORM_MAX}]"
        )
        return result

    result["ok"] = True
    return result


def _run_readiness() -> Tuple[int, Dict[str, Any]]:
    """Sonde complète du modèle d'embedding par défaut."""
    counters.inc("readiness_probe_total")
    started = time.time()
    checks: Dict[str, Any] = {}
    payload: Dict[str, Any] = {
        "status": "error",
        "provider": "ollama",
        "ollama_base_url": "redacted",
        "default_embedding_model": settings.embedding_model,
        "environment": settings.environment,
        "checks": checks,
    }

    profile = registry.resolve(settings.embedding_model)
    checks["registry_profile"] = profile is not None
    if profile is None:
        payload["message"] = (
            f"modèle d'embedding par défaut absent du registre : {settings.embedding_model}"
        )
        counters.inc("readiness_probe_failed_total")
        return 503, payload

    payload["model_logical"] = profile.logical_name
    payload["ollama_name"] = profile.ollama_name
    payload["expected_digest"] = profile.digest

    # Digest : revérifié à chaque sonde (le cache de readiness borne déjà la
    # fréquence à une fois par minute).
    _ensure_registry_verified(force=True)
    payload["observed_digest"] = profile.observed_digest
    payload["registry_status"] = profile.status
    # Lot P2 : « service prêt ? » et « identité prouvée ? » sont deux questions.
    payload["identity_verified"] = profile.identity_verified
    payload["verified_at"] = profile.verified_at_iso

    checks["upstream_reachable"] = profile.status != STATUS_UNAVAILABLE
    checks["model_installed"] = profile.status in (STATUS_READY, STATUS_DEGRADED)
    checks["digest_match"] = profile.status == STATUS_READY
    checks["environment_allowed"] = profile.allowed_in(settings.environment)

    if not checks["model_installed"]:
        payload["message"] = "modèle d'embedding par défaut indisponible sur l'upstream"
        counters.inc("readiness_probe_failed_total")
        return 503, payload

    if profile.status == STATUS_DEGRADED:
        payload["message"] = "digest du modèle par défaut divergent du registre"
        counters.inc("readiness_probe_failed_total")
        return 503, payload

    if not checks["environment_allowed"]:
        payload["message"] = (
            f"modèle par défaut non autorisé dans l'environnement {settings.environment}"
        )
        counters.inc("readiness_probe_failed_total")
        return 503, payload

    probe = _probe_embedding(profile)
    checks["embedding_call"] = probe["error"] is None
    checks["dimension"] = probe["dimension"] == profile.dimension
    checks["finite_values"] = probe["error"] is None
    checks["norm"] = probe["norm"] is not None and (
        not profile.normalized or norm_in_tolerance(probe["norm"])
    )
    payload["dimension"] = probe["dimension"]
    payload["norm"] = probe["norm"]
    payload["embedding_latency_ms"] = probe["latency_ms"]
    payload["upstream_endpoint"] = probe["upstream_endpoint"]
    payload["latency_ms"] = int((time.time() - started) * 1000)

    if not probe["ok"]:
        payload["message"] = probe["error"] or "sonde d'embedding en échec"
        counters.inc("readiness_probe_failed_total")
        return 503, payload

    payload["status"] = "ok"
    payload["pipeline_id"] = profile.pipeline_id(DEFAULT_TASK)
    return 200, payload


@app.get("/health/ready")
def health_ready() -> JSONResponse:
    """Readiness fonctionnelle : un embedding est réellement calculé et validé.

    Résultat mis en cache `AI_GATEWAY_READINESS_CACHE_SECONDS` (60 s) et
    protégé par un verrou : deux sondes simultanées n'en déclenchent qu'une.
    """
    now = time.monotonic()
    with _readiness_lock:
        cached = _readiness_cache["payload"] is not None and _readiness_cache["expires"] > now
        if not cached:
            status, payload = _run_readiness()
            _readiness_cache["status"] = status
            _readiness_cache["payload"] = payload
            _readiness_cache["expires"] = time.monotonic() + settings.readiness_cache_seconds
        status = _readiness_cache["status"]
        payload = dict(_readiness_cache["payload"])
        payload["cached"] = cached
        payload["cache_ttl_seconds"] = settings.readiness_cache_seconds

    if status != 200:
        return JSONResponse(status_code=status, content={"detail": payload})
    return JSONResponse(status_code=status, content=payload)


def _readiness_cache_clear() -> None:
    """Invalide le cache de readiness. Utilisé par les tests et par
    /health/embeddings, qui vient de mesurer la même chose."""
    with _readiness_lock:
        _readiness_cache["expires"] = 0.0
        _readiness_cache["payload"] = None


# --------------------------------------------------------------------------- #
# Health — diagnostic détaillé
# --------------------------------------------------------------------------- #
@app.get("/health/embeddings")
def health_embeddings() -> Dict[str, Any]:
    """Diagnostic par modèle gouverné.

    Route de diagnostic : elle encode un texte de sonde par modèle du registre.
    Elle n'est pas destinée à être appelée en boucle. Le réseau reste le seul
    contrôle d'accès de ce service (UFW, bind local) : aucun secret, aucun
    chemin de blob et aucune trace d'exception brute n'est renvoyé ici.
    """
    _ensure_registry_verified(force=True)

    models: List[Dict[str, Any]] = []
    for profile in registry.embedding_models():
        entry = profile.to_public_dict()
        entry["allowed_here"] = profile.allowed_in(settings.environment)
        entry["is_default"] = registry.resolve(settings.embedding_model) is profile
        entry["pipeline_id_query"] = profile.pipeline_id("query")
        entry["pipeline_id_document"] = profile.pipeline_id(DEFAULT_TASK)

        if profile.status == STATUS_UNAVAILABLE:
            entry["state"] = STATUS_UNAVAILABLE
            entry["error"] = profile.status_detail
            entry.update(
                {"measured_dimension": None, "measured_norm": None, "latency_ms": None,
                 "upstream_endpoint": None}
            )
            models.append(entry)
            continue

        probe = _probe_embedding(profile)
        entry["measured_dimension"] = probe["dimension"]
        entry["measured_norm"] = probe["norm"]
        entry["latency_ms"] = probe["latency_ms"]
        entry["upstream_endpoint"] = probe["upstream_endpoint"]
        entry["error"] = probe["error"]

        if profile.status == STATUS_DEGRADED:
            entry["state"] = STATUS_DEGRADED
        elif probe["ok"]:
            entry["state"] = STATUS_READY
        else:
            entry["state"] = STATUS_UNAVAILABLE
        models.append(entry)

    return {
        "provider": "ollama",
        "ollama_base_url": "redacted",
        "environment": settings.environment,
        "strict_registry": settings.embeddings_strict_registry,
        "registry_version": registry.version,
        "default_embedding_model": settings.embedding_model,
        "models": models,
        "counters": counters.snapshot(),
    }


# --------------------------------------------------------------------------- #
# États de capacités — NON inférentiels
# --------------------------------------------------------------------------- #
# Vocabulaire fermé :
#   ENABLED      configurée ET au moins un modèle enregistré autorisé ici
#   DISABLED     désactivée par configuration (flag OFF) ou sans modèle au registre
#   UNAVAILABLE  composant NON inclus dans cette distribution
#   UNKNOWN      réservé aux faits non encore observés (inventaire upstream)
# « enregistré » (registre), « installé » (inventaire /api/tags, en cache) et
# « capacité testée » (sonde inférentielle EXPLICITE) sont trois états
# distincts ; cette section ne déclenche aucun des deux derniers.
CAPABILITY_ENABLED = "ENABLED"
CAPABILITY_DISABLED = "DISABLED"
CAPABILITY_UNAVAILABLE = "UNAVAILABLE"
CAPABILITY_UNKNOWN = "UNKNOWN"


def _capability_states() -> Dict[str, Any]:
    env = settings.environment
    inventory_known = registry.inventory is not None
    generation = {
        name: {"installed": registry.llm_installed(p)}
        for name, p in registry.llm_models.items()
        if COMPLETION_CAPABILITY in p.capabilities and p.allowed_in(env)
    }
    chat_models = [name for name, p in registry.llm_models.items()
                   if CHAT_CAPABILITY in p.capabilities and p.allowed_in(env)]
    embeddings = {
        name: {
            "registry_status": p.status,
            "installed": (p.status in (STATUS_READY, STATUS_DEGRADED)) if inventory_known else None,
            "identity_verified": p.identity_verified,
        }
        for name, p in registry.models.items() if p.allowed_in(env)
    }
    rerank_enabled = _rerank_capability_blocker() is None and bool(
        [p for p in registry.reranker_models.values() if p.allowed_in(env)])
    asr_enabled = _asr_capability_blocker() is None
    vision_models = [p for p in registry.vision_profiles() if p.allowed_in(env)]
    return {
        "inventory": {
            "state": registry.inventory_state(),
            "observed_at": (datetime.fromtimestamp(registry.inventory_at, tz=timezone.utc).isoformat()
                            if registry.inventory_at else None),
        },
        "generation": {"state": CAPABILITY_ENABLED if generation else CAPABILITY_DISABLED,
                       "default_model": settings.generate_model, "models": generation,
                       "probe": None},
        "chat": {"state": CAPABILITY_ENABLED if chat_models else CAPABILITY_DISABLED,
                 "models": chat_models, "probe": None},
        "embeddings": {"state": CAPABILITY_ENABLED if embeddings else CAPABILITY_DISABLED,
                       "default_model": settings.embedding_model, "models": embeddings,
                       "probe": "/health/ready"},
        "rerank": {"state": CAPABILITY_ENABLED if rerank_enabled else CAPABILITY_DISABLED,
                   "probe": "/health/rerank"},
        "asr": {"state": CAPABILITY_ENABLED if asr_enabled else CAPABILITY_DISABLED,
                "probe": "/health/asr"},
        "vision": {"state": CAPABILITY_ENABLED if settings.vision_enabled and vision_models else CAPABILITY_DISABLED,
                   "probe": "/health/vision"},
        "structured_extract": {"state": CAPABILITY_UNAVAILABLE, "reason": "module_not_included"},
        "semantic_extract": {"state": CAPABILITY_UNAVAILABLE, "reason": "module_not_included"},
        "host_admission": {"state": CAPABILITY_UNAVAILABLE, "reason": "controller_not_included"},
    }


# --------------------------------------------------------------------------- #
# Health — diagnostic runtime LLM (lot runtime-hardening)
# --------------------------------------------------------------------------- #
@app.get("/health/runtime")
def health_runtime() -> Dict[str, Any]:
    """État de la gouvernance LLM et de la concurrence en cours.

    Aucun appel upstream : tout vient du registre chargé et des jauges du
    process — la route est appelable en boucle sans coût. Aucune donnée
    sensible : pas d'URL, pas de contenu, uniquement des profils déclarés et
    des compteurs.

    Doctrine reprise du lot observabilité : le temps d'attente en file
    d'Ollama n'est PAS observable. Cette route ne publie donc aucun
    `queue_time` — et `total_ms - ollama_total_ms` n'en est pas un.
    """
    gauge_snapshot = gauges.snapshot()       # un seul instantané pour `inflight` / `host_admission`
    llm_models = []
    for profile in registry.llm_models.values():
        entry = profile.to_public_dict()
        context = registry.context_profile_of(profile)
        keep_alive = registry.keep_alive_of(profile)
        entry["default_num_ctx"] = context.num_ctx
        entry["max_num_ctx"] = context.max_num_ctx
        entry["keep_alive"] = keep_alive.keep_alive
        entry["keep_alive_client_override"] = keep_alive.client_override
        entry["allowed_here"] = profile.allowed_in(settings.environment)
        entry["chat"] = CHAT_CAPABILITY in profile.capabilities
        entry["reasoning"] = registry.reasoning_profile_of(profile).to_public_dict()
        llm_models.append(entry)

    return {
        "provider": "ai_gateway",
        "version": __version__,
        "environment": settings.environment,
        # Provenance de build (lot build-provenance-observability) : permet à
        # un client autorisé d'attester QUEL build sert la requête, sans SSH
        # ni accès Docker. `git_sha` est figé dans l'image au build ; une
        # valeur absente/invalide rend `not_reported`, jamais une invention.
        # `image_digest` : les images Gateway sont construites localement et
        # jamais poussées dans un registre — elles n'ont pas de digest de
        # distribution. On ne fabrique pas un pseudo-digest : not_reported.
        "build": {
            "git_sha": settings.build_sha or "not_reported",
            "version": __version__,
            "image_digest": "not_reported",
        },
        "registry_version": registry.version,
        # Lot P3 : lisible sans ouvrir l'environnement du conteneur.
        "generation_strict_registry": settings.generation_strict_registry,
        "llm_models": llm_models,
        "context_profiles": {
            name: {"num_ctx": p.num_ctx, "max_num_ctx": p.max_num_ctx}
            for name, p in registry.context_profiles.items()
        },
        "keep_alive_profiles": {
            name: {"keep_alive": p.keep_alive, "client_override": p.client_override}
            for name, p in registry.keep_alive_profiles.items()
        },
        # Budgets de sortie (lot output-bound-hardening) — exposés au même
        # titre que les deux autres familles de profils : un opérateur doit
        # pouvoir lire le plafond en vigueur sans ouvrir le registre.
        "generation_profiles": {
            name: {
                "default_num_predict": p.default_num_predict,
                "max_num_predict": p.max_num_predict,
            }
            for name, p in registry.generation_profiles.items()
        },
        # Raisonnement gouverné (lot C1) : politiques déclarées, lisibles sans
        # ouvrir le registre. Un modèle sans profil porte l'implicite `none`.
        "reasoning_profiles": {
            name: p.to_public_dict() for name, p in registry.reasoning_profiles.items()
        },
        # `inflight` = requêtes MÉTIER en cours (filtre positif, cf. app/metrics.py) :
        # 0 signifie « aucune inférence/requête métier », même si une file de
        # récupération interne existe (préflight de promotion, RagATM ensure_quiet).
        "inflight": {
            "total": sum(business_inflight(gauge_snapshot).values()),
            "by_model": business_inflight(gauge_snapshot),
        },
        # Admission hôte : file de RÉCUPÉRATION interne (aborts/libérations non
        # confirmés, lot ORPHAN-LEASE) — jamais comptée dans `inflight`.
        "host_admission": {
            "reaper_pending": int(gauge_snapshot.get(HOST_ADMISSION_REAPER_PENDING_GAUGE, 0)),
        },
        # Depuis une version antérieure il n'existe PLUS de file applicative : une requête
        # rerank/vision à capacité pleine est refusée immédiatement (503 +
        # Retry-After) au lieu d'attendre dans un thread partagé. La clé est
        # conservée pour les consommateurs qui la lisent et vaut toujours 0 ;
        # les refus se comptent dans `rerank_saturated_total` /
        # `vision_overloaded_total`. Generate/embeddings attendent dans
        # Ollama, non observable (cf. queue_time).
        "queued": {
            "total": sum(v for k, v in gauges.snapshot().items() if k.startswith("queued:")),
            "by_model": {k[len("queued:"):]: v for k, v in gauges.snapshot().items() if k.startswith("queued:")},
        },
        # Topologie de calcul (lot workload-isolation) — DÉCLARÉE, pas
        # mesurée : si plusieurs instances Gateway (ex. production/staging)
        # partagent le même GPU, le même Ollama et le même backend reranker,
        # une charge sur l'une dégrade l'autre ; un consommateur qui lance une
        # campagne doit lire l'inflight de l'instance voisine avant de démarrer.
        "compute": {
            "isolation": "none",
            "shared_with_environments": sorted(
                {"production", "staging"} - {settings.environment}
            ),
            "shared_resources": ["gpu", "ollama", "rerank_backend"],
            "gpu_count": 1,
            "sibling_inflight": "not_observable_here",
        },
        # Capability reranking : configuration STATIQUE uniquement — l'état
        # réel du backend se sonde sur /health/rerank (cette route reste sans
        # appel upstream, par contrat).
        "rerank": {
            "enabled": settings.rerank_enabled
            and backend_url_is_loopback(settings.rerank_backend_url),
            "default_model": settings.rerank_model,
            "models": [p.to_public_dict() for p in registry.reranker_models.values()],
            # Provenance backend (additif) : identité de l'artefact qui sert
            # réellement le scoring. Le digest vient du déploiement (fichier
            # .env, même valeur que la ligne image@digest du sidecar) — pas
            # d'URL, pas de port : l'adresse du backend reste `redacted`.
            "backend": {
                "implementation": BACKEND_IMPLEMENTATION,
                "image_digest": settings.rerank_backend_image_digest
                or "not_reported",
            },
        },
        # Capability ASR (lot asr-rail-foundation) : configuration STATIQUE
        # uniquement — l'état réel du backend se sonde sur /health/asr. Section
        # ADDITIVE : aucune clé existante ne change de forme.
        "asr": {
            "enabled": _asr_capability_blocker() is None,
            "default_model": registry.asr_default_model,
            "models": [p.to_public_dict() for p in registry.asr_models.values()],
            "limits": {
                "sync_wait_seconds": settings.asr_sync_wait_seconds,
                "max_concurrency": settings.asr_max_concurrency,
                "gpu_budget_mib": settings.asr_gpu_budget_mib,
                "gpu_reserve_mib": settings.asr_gpu_reserve_mib,
                "gpu_baseline_mib": settings.asr_gpu_baseline_mib,
            },
            "backend": {
                "implementation": asr_transport.BACKEND_IMPLEMENTATION,
                "image_digest": settings.asr_backend_image_digest or "not_reported",
            },
        },
        "queue_time": "not_observable",
        # états de capacités (vocabulaire fermé) et limites de
        # concurrence EXPLICITES, sans aucun appel upstream.
        "capabilities": _capability_states(),
        "limits": {
            "generate_max_concurrency": settings.generate_max_concurrency,
            "embeddings_max_concurrency": settings.embeddings_max_concurrency,
            "rerank_max_concurrency": settings.rerank_max_concurrency,
            "vision_max_concurrency": settings.vision_max_concurrency,
            "request_max_bytes": settings.request_max_bytes,
            "embeddings_max_batch": settings.embeddings_max_batch,
            "timeout_seconds": settings.timeout_seconds,
        },
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    try:
        models = client.list_models(timeout=5)
    except UpstreamError as exc:
        logger.warning("models: upstream indisponible: %s", exc.message)
        raise http_error(503, PUBLIC_UPSTREAM_UNAVAILABLE, provider="ollama")

    out: List[Dict[str, Any]] = []
    for m in models:
        name = m.get("name")
        # Champs historiques d'abord : leur nom et leur sens ne changent pas.
        entry: Dict[str, Any] = {
            "name": name,
            "size": m.get("size"),
            "modified_at": m.get("modified_at"),
        }
        # Ajouts additifs. `details` d'Ollama ne contient pas de chemin de blob
        # (contrairement à /api/show) : on ne recopie que des champs connus.
        digest = m.get("digest")
        if digest:
            entry["digest"] = digest
        details = m.get("details") or {}
        if details:
            entry["details"] = {
                key: details.get(key)
                for key in ("family", "parameter_size", "quantization_level",
                            "context_length", "embedding_length")
                if details.get(key) is not None
            }
        if m.get("capabilities"):
            entry["capabilities"] = m["capabilities"]

        profile = registry.resolve(name) if name else None
        if profile is not None:
            entry.update(
                {
                    "model_logical": profile.logical_name,
                    "kind": profile.kind,
                    "dimension": profile.dimension,
                    "registry_status": profile.status,
                    # métadonnées Compact déclaratives (additif).
                    "compact": profile.metadata.to_public_dict(),
                }
            )
        # Gouvernance LLM (additif) : un modèle de génération décrit au
        # registre expose son profil opérationnel.
        llm = registry.resolve_llm(name) if name else None
        if llm is not None:
            context = registry.context_profile_of(llm.profile)
            entry.update(
                {
                    "model_logical": llm.profile.logical_name,
                    "kind": "llm",
                    "resource_class": llm.profile.resource_class,
                    "context_profile": context.name,
                    "default_num_ctx": context.num_ctx,
                    "keep_alive_profile": llm.profile.keep_alive_profile,
                    "compact": llm.profile.metadata.to_public_dict(),
                }
            )
            # Vision (lot vision-1a) : métadonnées STATIQUES additives — l'état
            # dynamique (enabled/resident) vit sur /health/vision, pas ici.
            if llm.profile.supports_image:
                entry["modalities"] = list(llm.profile.modalities)
                entry["vision"] = {
                    "tasks": list(llm.profile.vision_tasks),
                    "schema_versions": list(llm.profile.vision_schema_versions),
                    "max_images": llm.profile.vision_max_images,
                }
                if llm.profile.vision_max_image_pixels is not None:
                    entry["vision"]["max_image_pixels"] = llm.profile.vision_max_image_pixels
        out.append(entry)

    return {"provider": "ollama", "models": out}


# --------------------------------------------------------------------------- #
# Generate — résolution gouvernée (lot runtime-hardening)
# --------------------------------------------------------------------------- #
@dataclass
class GenerationPlan:
    """Paramètres effectifs d'une génération, après application du registre.

    Une seule fonction produit ce plan pour les DEUX routes (agrégée et flux) :
    les défauts gouvernés ne peuvent pas diverger entre les deux chemins.
    Règle de priorité, identique pour chaque champ :

        valeur explicite du client  >  profil du registre  >  rien (historique)

    Un override client n'est JAMAIS modifié en silence : soit il passe tel
    quel, soit il est refusé en 400 explicite (num_ctx hors borne du profil).
    """

    upstream_model: str
    options: Optional[Dict[str, Any]]
    think: Optional[bool]
    keep_alive: Optional[Any]
    log_fields: Dict[str, Any]
    inflight_key: str
    # Lot G2 — sortie structurée gouvernée sur le rail completion : schéma
    # VALIDÉ relayé dans `format` (à la place de `req.format`), ou None =
    # comportement historique intégral. Champs à défaut : un plan legacy est
    # construit exactement comme avant.
    response_schema: Optional[Dict[str, Any]] = None
    schema_stats: Optional[SchemaStats] = None
    response_schema_sha256: Optional[str] = None


def _resolve_num_ctx(
    raw_options: Optional[Dict[str, Any]], context: "ContextProfile"
) -> tuple[int, Optional[int]]:
    """(effectif, demandé) — la fenêtre de contexte appliquée à cette requête.

    Même discipline que `_resolve_num_predict`, pour la même raison : la
    valeur VALIDÉE doit être exactement la valeur TRANSMISE. L'ancienne
    logique passait l'override par `safe_int`, qui confond absent et invalide
    et arrondit — un client envoyant `4096.7` voyait la borne évaluée sur 4096
    et Ollama recevoir 4096.7 ; `True`, `"4096"` ou `-5` étaient remplacés en
    silence par le défaut du profil. Ici :

        absent   -> num_ctx du profil (défaut inchangé)
        valide   -> relayé tel quel, borné par max_num_ctx (inclusif)
        invalide -> 400, jamais un défaut de repli ni un arrondi
    """
    options = raw_options or {}
    if "num_ctx" not in options:
        return context.num_ctx, None

    raw = options["num_ctx"]
    if isinstance(raw, bool) or not isinstance(raw, int):
        counters.inc("generate_rejected_num_ctx_total")
        raise http_error(
            400,
            "num_ctx doit être un entier",
            provider="ai_gateway",
            extra={"context_profile": context.name},
        )
    if raw <= 0:
        counters.inc("generate_rejected_num_ctx_total")
        raise http_error(
            400,
            f"num_ctx {raw} invalide : la fenêtre de contexte doit être "
            f"strictement positive (profil {context.name})",
            provider="ai_gateway",
            extra={
                "context_profile": context.name,
                "requested_num_ctx": raw,
                "max_num_ctx": context.max_num_ctx,
            },
        )
    if raw > context.max_num_ctx:
        counters.inc("generate_rejected_num_ctx_total")
        raise http_error(
            400,
            f"num_ctx {raw} dépasse la borne "
            f"{context.max_num_ctx} du profil {context.name}",
            provider="ai_gateway",
            extra={
                "context_profile": context.name,
                "requested_num_ctx": raw,
                "max_num_ctx": context.max_num_ctx,
            },
        )
    return raw, raw


def _resolve_num_predict(
    raw_options: Optional[Dict[str, Any]], generation: "GenerationProfile"
) -> tuple[int, Optional[int]]:
    """(effectif, demandé) — le budget de sortie appliqué à cette requête.

    Pourquoi ne PAS réutiliser `safe_int` ici : il renvoie None aussi bien pour
    « absent » que pour « invalide ». Appliquer le défaut sur une valeur
    invalide reviendrait à corriger en silence l'erreur du caller, exactement
    ce que ce lot interdit. La distinction est donc faite sur la valeur BRUTE :

        absent   -> défaut du profil
        valide   -> relayé tel quel
        invalide -> 400, jamais un clamp ni un défaut de repli
    """
    options = raw_options or {}
    if "num_predict" not in options:
        return generation.default_num_predict, None

    raw = options["num_predict"]
    # `isinstance(True, int)` vaut True en Python : le booléen se déguise en
    # entier et passerait toutes les bornes numériques. Testé en premier.
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise http_error(
            400,
            "num_predict doit être un entier",
            provider="ai_gateway",
            extra={"generation_profile": generation.name},
        )
    # `0` et `-1` sont les sentinelles Ollama d'une génération ILLIMITÉE —
    # exactement ce qui a produit la RCA du 2026-09-22. Tout le domaine <= 0
    # est refusé : aucune valeur n'y a de sens comme budget.
    if raw <= 0:
        raise http_error(
            400,
            f"num_predict {raw} demande une génération non bornée — refusé "
            f"(profil {generation.name})",
            provider="ai_gateway",
            extra={
                "generation_profile": generation.name,
                "requested_num_predict": raw,
                "max_num_predict": generation.max_num_predict,
            },
        )
    if raw > generation.max_num_predict:
        # Refus explicite plutôt que clamp : un clamp silencieux ferait
        # découvrir la troncature dans le texte produit, pas dans la réponse.
        raise http_error(
            400,
            f"num_predict {raw} dépasse la borne "
            f"{generation.max_num_predict} du profil {generation.name}",
            provider="ai_gateway",
            extra={
                "generation_profile": generation.name,
                "requested_num_predict": raw,
                "max_num_predict": generation.max_num_predict,
            },
        )
    return raw, raw


COMPLETION_CAPABILITY = "completion"
THINKING_CAPABILITY = "thinking"

# Concurrence bornée EXPLICITE de la génération (generate + chat, même classe
# de travail GPU) et des embeddings. Acquisition NON bloquante :
# à capacité pleine, 503 + Retry-After immédiats (doctrine P8), jamais une
# attente dans un thread partagé. `Retry-After` est un délai de politesse
# fixe, pas une prédiction.
_generate_semaphore = threading.BoundedSemaphore(max(1, settings.generate_max_concurrency))
_embeddings_semaphore = threading.BoundedSemaphore(max(1, settings.embeddings_max_concurrency))
GENERATE_RETRY_AFTER_SECONDS = 2
EMBEDDINGS_RETRY_AFTER_SECONDS = 1


def _saturated_error(code: str, message: str, retry_after: int):
    return http_error(
        503, message, provider="ai_gateway",
        extra={"code": code, "retryable": True, "retry_after_seconds": retry_after},
        headers={"Retry-After": str(retry_after)},
    )


def _model_not_installed_error(model: str, rail: str):
    """503 typé, NON réessayable : la Gateway ne télécharge jamais un modèle."""
    counters.inc(f"{rail}_model_not_installed_total")
    log_event(logger, logging.WARNING, f"{rail}.model_not_installed", model=model)
    return http_error(
        503, PUBLIC_MODEL_NOT_INSTALLED, provider="ollama",
        extra={"code": "model_not_installed", "retryable": False, "model": model},
    )

# --------------------------------------------------------------------------- #
# Sortie structurée gouvernée — helpers partagés chat / generate (lot G2)
# --------------------------------------------------------------------------- #
_STRUCTURED_PUBLIC_MESSAGE = {
    ERR_OUTPUT_EMPTY: PUBLIC_STRUCTURED_EMPTY,
    ERR_OUTPUT_INVALID: PUBLIC_STRUCTURED_INVALID,
    ERR_OUTPUT_MISMATCH: PUBLIC_STRUCTURED_MISMATCH,
}
_STRUCTURED_COUNTER_SUFFIX = {
    ERR_OUTPUT_EMPTY: "empty_total",
    ERR_OUTPUT_INVALID: "invalid_json_total",
    ERR_OUTPUT_MISMATCH: "schema_mismatch_total",
}


def _schema_rejected_error(exc: SchemaError, model_logical: str) -> Any:
    """400 d'un schéma client refusé — même forme sur les deux rails."""
    return http_error(
        400, exc.message, provider="ai_gateway",
        extra={"code": exc.code, "model_logical": model_logical, **exc.extra},
    )


def _structured_invalid_error(exc: StructuredOutputError, reason: Optional[str]) -> Any:
    """502 d'une sortie du modèle hors contrat — message FIXE, jamais le contenu."""
    return http_error(
        502, _STRUCTURED_PUBLIC_MESSAGE[exc.code], provider="ollama",
        extra={"code": exc.code, **({"done_reason": reason} if reason else {})},
    )


def _count_structured_invalid(rail: str, code: str) -> None:
    """`rail` ∈ {chat, generate} : agrégat + ventilation par cause."""
    counters.inc(f"{rail}_structured_invalid_upstream_total")
    counters.inc(f"{rail}_structured_{_STRUCTURED_COUNTER_SUFFIX[code]}")


def completion_structured_output_blocker(profile, reasoning_profile, effective_think) -> Optional[str]:
    """Pourquoi la sortie structurée est IMPOSSIBLE sur le rail completion
    pour ce modèle, ou None si elle est admissible. Règle GÉNÉRIQUE (aucun
    nom de modèle) : sur `/api/generate` + `format`, un canal de réflexion
    actif corrompt la sortie contrainte (mesuré le 2026-09-26 : texte de
    raisonnement déformé dans `response`). Sont donc refusés :

    - un profil de raisonnement non désactivable (`disable_supported: false`) ;
    - un modèle qui déclare `thinking` sans que la requête effective ne
      l'éteigne (`think` effectif ≠ False) — le rail completion ne pilote
      que `think: true/false`, jamais les niveaux.
    """
    if not reasoning_profile.disable_supported:
        return "reasoning_not_disableable"
    if THINKING_CAPABILITY in profile.capabilities and effective_think is not False:
        return "thinking_channel_active"
    return None


def _plan_generation(req: GenerateRequest) -> GenerationPlan:
    model = req.model or settings.generate_model
    resolved = registry.resolve_llm(model)

    log_fields: Dict[str, Any] = {}
    if req.think is not None:
        log_fields["requested_think"] = req.think

    # Lot G2 — `format` (relais brut) et `response_format` (gouverné) sont
    # exclusifs : aucun des deux ne doit pouvoir écraser l'autre en silence.
    if req.response_format is not None and req.format is not None:
        counters.inc("generate_structured_conflict_rejected_total")
        raise http_error(
            400,
            "format et response_format sont exclusifs : choisir le relais brut "
            "(format) ou la sortie structurée gouvernée (response_format)",
            provider="ai_gateway",
            extra={"code": "generate_format_conflict"},
        )

    if resolved is None:
        if req.response_format is not None:
            # Une sortie gouvernée exige un modèle ENREGISTRÉ porteur de la
            # capability : le chemin legacy ne peut jamais la servir.
            counters.inc("generate_structured_requests_total")
            counters.inc("generate_structured_capability_rejected_total")
            log_event(logger, logging.WARNING, "generate.structured_output_not_supported", model=model)
            raise http_error(
                400,
                f"modèle {model} absent du registre : response_format exige un "
                f"modèle enregistré déclarant la capability {STRUCTURED_OUTPUT_CAPABILITY}",
                provider="ai_gateway",
                extra={"code": "structured_output_not_supported", "model_logical": None},
            )
        if settings.generation_strict_registry:
            # Lot P3 — registre strict : seuls les modèles enregistrés (nom
            # logique, alias ou nom physique) sont servis. Refus AVANT Ollama,
            # compté à part : ce n'est pas une génération legacy relayée.
            counters.inc("generate_rejected_unregistered_total")
            log_event(logger, logging.WARNING, "generate.model_not_registered", model=model)
            raise http_error(
                400,
                f"modèle {model} absent du registre (registre strict activé)",
                provider="ai_gateway",
                extra={"code": "model_not_registered", "strict_registry": True},
            )
        # Modèle hors registre : relais intégral, comportement historique.
        # `safe_int` ne sert ICI qu'à journaliser une valeur lisible — rien
        # n'est validé ni modifié sur ce chemin, les options partent brutes.
        requested_num_ctx = safe_int((req.options or {}).get("num_ctx"))
        if requested_num_ctx is not None:
            # `num_ctx` est le nom HISTORIQUE du champ « demandé par le
            # client » : conservé pour ne pas casser les analyses existantes.
            log_fields["num_ctx"] = requested_num_ctx
            log_fields["effective_num_ctx"] = requested_num_ctx
        if req.think is not None:
            log_fields["effective_think"] = req.think
        # Lot P3 — instrumentation : chaque relais legacy est compté et
        # journalisé (nom déjà validé par le schéma, donc mono-ligne), sans
        # cardinalité libre — un compteur global et UNE clé de jauge
        # canonique, le nom arbitraire du client n'entre plus dans la map.
        counters.inc("generate_legacy_model_total")
        log_event(logger, logging.WARNING, "generate.legacy_model", model=model)
        return GenerationPlan(
            upstream_model=model,
            options=req.options,
            think=req.think,
            keep_alive=req.keep_alive,
            log_fields=log_fields,
            inflight_key="generate:legacy",
        )

    profile = resolved.profile
    log_fields["model_logical"] = profile.logical_name

    # Lot C2 — capability `completion` OPT-IN pour un modèle enregistré, au
    # même titre que `chat` sur /v1/chat. Tous les modèles gouvernés avant
    # C2 la déclarent : leur comportement est inchangé (démontré par le
    # différentiel C1). Un modèle enregistré SANS `completion` — gpt-oss:20b,
    # dont /api/generate ne sait ni piloter les niveaux de raisonnement ni
    # servir `format` — est refusé ici, AVANT Ollama, plutôt que servi avec
    # une gouvernance insuffisante. Le chemin legacy (hors registre) n'est
    # pas concerné : il n'a pas de profil, donc pas de capabilities.
    if COMPLETION_CAPABILITY not in profile.capabilities:
        counters.inc("generate_rejected_capability_total")
        log_event(
            logger, logging.WARNING, "generate.capability_not_supported",
            model=profile.ollama_name, model_logical=profile.logical_name,
        )
        raise http_error(
            400,
            f"le modèle {profile.logical_name} ne déclare pas la capability "
            f"{COMPLETION_CAPABILITY} : il n'est pas servi par /v1/generate",
            provider="ai_gateway",
            extra={
                "code": "completion_not_supported",
                "model_logical": profile.logical_name,
                "capabilities": list(profile.capabilities),
            },
        )

    # Autorisation d'environnement — même logique que les embeddings : une
    # décision de gouvernance ne dépend pas de l'endpoint appelé.
    if not profile.allowed_in(settings.environment):
        counters.inc("generate_rejected_environment_total")
        raise http_error(
            400,
            f"modèle {profile.logical_name} non autorisé dans l'environnement "
            f"{settings.environment}",
            provider="ai_gateway",
            extra={
                "model_logical": profile.logical_name,
                "environment": settings.environment,
            },
        )

    context = registry.context_profile_of(profile)
    options = dict(req.options or {})
    # Fenêtre de contexte — un lot antérieur. `options["num_ctx"]` est TOUJOURS réécrit
    # depuis la valeur validée : le dict transmis ne peut pas diverger de la
    # décision (c'était le défaut : borne évaluée sur une copie arrondie,
    # dict brut envoyé à Ollama). Sans override, le défaut du profil évite
    # au runner d'allouer son contexte par défaut (32768 observé — 8,57 Gio
    # pour llama3.1:8b là où 4096 en coûte 4,91).
    effective_num_ctx, requested_num_ctx = _resolve_num_ctx(req.options, context)
    options["num_ctx"] = effective_num_ctx
    if requested_num_ctx is not None:
        log_fields["num_ctx"] = requested_num_ctx
    log_fields["context_profile"] = context.name
    log_fields["effective_num_ctx"] = effective_num_ctx

    # Budget de sortie — cœur du lot output-bound-hardening. Le merge est
    # DÉTERMINISTE : `options` porte déjà les clés du client (sampling, num_ctx
    # …) et on ne remplace que `num_predict`, jamais le reste.
    generation = registry.generation_profile_of(profile)
    effective_num_predict, requested_num_predict = _resolve_num_predict(
        req.options, generation
    )
    options["num_predict"] = effective_num_predict
    log_fields["generation_profile"] = generation.name
    log_fields["effective_num_predict"] = effective_num_predict
    if requested_num_predict is not None:
        log_fields["requested_num_predict"] = requested_num_predict

    # think : la politique du profil ne s'applique QUE via le nom logique.
    # Demander `qwen-judge` implique le rôle (think=false) ; demander
    # `qwen3:14b` conserve le comportement natif du modèle.
    think = req.think
    if think is None and resolved.via_logical:
        think = profile.default_think
    if think is not None:
        log_fields["effective_think"] = think

    retention = registry.keep_alive_of(profile)
    keep_alive = req.keep_alive
    if keep_alive is not None and not retention.client_override:
        # Lot P3 : profil verrouillé — la rétention VRAM est une décision de
        # gouvernance, pas un paramètre client. Refus explicite (jamais un
        # remplacement silencieux), avant tout appel Ollama.
        counters.inc("generate_rejected_keep_alive_total")
        raise http_error(
            400,
            f"keep_alive n'est pas surchargeable pour le modèle "
            f"{profile.logical_name} (profil {retention.name})",
            provider="ai_gateway",
            extra={
                "code": "keep_alive_not_overridable",
                "model_logical": profile.logical_name,
                "keep_alive_profile": retention.name,
            },
        )
    if keep_alive is None:
        keep_alive = retention.keep_alive
        if keep_alive is not None:
            log_fields["keep_alive_profile"] = profile.keep_alive_profile

    # Lot G2 — sortie structurée gouvernée sur le rail completion. Tout ce
    # bloc est CONDITIONNEL à `response_format` : absent, le plan est celui
    # d'avant, champ par champ.
    response_schema: Optional[Dict[str, Any]] = None
    schema_stats: Optional[SchemaStats] = None
    response_schema_sha256: Optional[str] = None
    if req.response_format is not None:
        counters.inc("generate_structured_requests_total")
        if STRUCTURED_OUTPUT_CAPABILITY not in profile.capabilities:
            counters.inc("generate_structured_capability_rejected_total")
            log_event(
                logger, logging.WARNING, "generate.structured_output_not_supported",
                model=profile.ollama_name, model_logical=profile.logical_name,
            )
            raise http_error(
                400,
                f"le modèle {profile.logical_name} ne déclare pas la capability "
                f"{STRUCTURED_OUTPUT_CAPABILITY}",
                provider="ai_gateway",
                extra={
                    "code": "structured_output_not_supported",
                    "model_logical": profile.logical_name,
                    "capabilities": list(profile.capabilities),
                },
            )
        blocker = completion_structured_output_blocker(
            profile, registry.reasoning_profile_of(profile), think
        )
        if blocker is not None:
            counters.inc("generate_structured_reasoning_rejected_total")
            log_event(
                logger, logging.WARNING, "generate.structured_output_reasoning_incompatible",
                model=profile.ollama_name, model_logical=profile.logical_name, reason=blocker,
            )
            raise http_error(
                400,
                f"sortie structurée impossible sur le rail completion pour "
                f"{profile.logical_name} : canal de raisonnement actif ({blocker})",
                provider="ai_gateway",
                extra={
                    "code": "structured_output_reasoning_incompatible",
                    "model_logical": profile.logical_name,
                    "reason": blocker,
                },
            )
        try:
            schema_stats = validate_client_schema(req.response_format.schema_)
        except SchemaError as exc:
            counters.inc("generate_structured_schema_rejected_total")
            log_event(
                logger, logging.WARNING, "generate.schema_rejected",
                model=profile.ollama_name, model_logical=profile.logical_name,
                code=exc.code, **{k: v for k, v in exc.extra.items() if k in ("limit", "value", "max", "keyword")},
            )
            raise _schema_rejected_error(exc, profile.logical_name)
        response_schema = req.response_format.schema_
        response_schema_sha256 = schema_sha256(response_schema)
        log_fields["structured_output"] = True
        log_fields["schema_bytes"] = schema_stats.bytes
        log_fields["schema_depth"] = schema_stats.depth
        log_fields["schema_nodes"] = schema_stats.nodes
        log_fields["response_schema_sha256"] = response_schema_sha256

    return GenerationPlan(
        upstream_model=profile.ollama_name,
        options=options,
        think=think,
        keep_alive=keep_alive,
        log_fields=log_fields,
        inflight_key=f"generate:{profile.ollama_name}",
        response_schema=response_schema,
        schema_stats=schema_stats,
        response_schema_sha256=response_schema_sha256,
    )


def _done_reason(data: Dict[str, Any]) -> Optional[str]:
    """Raison d'arrêt telle que fournie par Ollama (`done_reason`), ou None.

    Relais VERBATIM ("stop", "length", …) : la Gateway n'infère rien de
    `eval_count` vs `num_predict` — un client qui veut savoir si le texte a été
    tronqué lit ce champ, et un upstream qui ne le fournit pas donne None.
    """
    value = data.get("done_reason")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    started = time.time()
    counters.inc("generate_requests_total")
    plan = _plan_generation(req)
    model = plan.upstream_model
    # Ligne historique conservée telle quelle : `prompt_len` reste la seule
    # information de taille disponible AVANT l'appel, donc la seule exploitable
    # quand l'upstream échoue. `prompt_tokens` ne la remplace pas.
    logger.info("generate model=%s prompt_len=%d", model, len(req.prompt))

    # Lot G2 : en mode structuré, `format` porte le schéma VALIDÉ ; sinon le
    # relais brut historique de `req.format` (exclusifs, vérifié au plan).
    upstream_format = plan.response_schema if plan.response_schema is not None else req.format

    # Admission HÔTE (lot host-wide-admission) : lease demandée APRÈS toute
    # validation, immédiatement avant le travail backend ; rendue au `finally`.
    # Inactive sans flag (None) ; DEFERRED/REJECTED/indisponible -> 503 typé.
    try:
        host_lease = host_admission.acquire("llm_generation", plan.log_fields.get("model_logical") or model)
    except HostAdmissionError as exc:
        raise host_admission_http_error(exc)
    if not _generate_semaphore.acquire(blocking=False):
        host_admission.release(host_lease)
        counters.inc("generate_saturated_total")
        raise _saturated_error("generation_saturated", "génération saturée : concurrence maximale atteinte",
                               GENERATE_RETRY_AFTER_SECONDS)
    host_admission.mark_started(host_lease)   # lot ORPHAN-LEASE : backend engagé à partir d'ici
    gauges.inc(plan.inflight_key)
    upstream_started = time.time()
    try:
        # `format` n'est passé que s'il est fourni : les doubles de test et
        # d'éventuels clients alternatifs gardent leur signature historique.
        data = client.generate(
            model=model,
            prompt=req.prompt,
            system=req.system,
            temperature=req.temperature,
            options=plan.options,
            think=plan.think,
            keep_alive=plan.keep_alive,
            **({"format": upstream_format} if upstream_format is not None else {}),
        )
    except UpstreamTimeout as exc:
        # Terminaison QUALIFIÉE. Le budget de lecture est épuisé : la Gateway
        # n'a rien reçu, ce qui ne dit PAS qu'Ollama est bloqué — lors de la
        # RCA, il générait sans interruption. Les champs journalisés sont ceux
        # qui permettent de trancher sans relire les logs du runner.
        counters.inc("generate_upstream_errors_total")
        counters.inc("generate_timeouts_total")
        log_event(
            logger,
            logging.WARNING,
            "generate.timeout",
            model=model,
            elapsed_ms=int((time.time() - upstream_started) * 1000),
            timeout_seconds=client.timeout,
            **plan.log_fields,
        )
        logger.warning("generate: échec upstream: %s", exc.message)
        raise http_error(502, PUBLIC_GENERATION_TIMEOUT, provider="ollama")
    except UpstreamModelNotFound:
        raise _model_not_installed_error(model, "generate")
    except UpstreamError as exc:
        counters.inc("generate_upstream_errors_total")
        logger.warning("generate: échec upstream: %s", exc.message)
        raise http_error(502, PUBLIC_GENERATION_FAILED, provider="ollama")
    finally:
        gauges.dec(plan.inflight_key)
        _generate_semaphore.release()
        host_admission.release(host_lease)
    upstream_ms = int((time.time() - upstream_started) * 1000)

    # Contrat de réponse : inchangé. Un compte absent y vaut 0 depuis l'origine,
    # `TokenCounts` n'acceptant pas None. Les métriques journalisées, elles,
    # OMETTENT ce qui est absent — « non mesuré » et « zéro token » ne sont pas
    # la même information pour qui analyse les performances.
    prompt_tokens = data.get("prompt_eval_count", 0) or 0
    completion_tokens = data.get("eval_count", 0) or 0
    total_tokens = prompt_tokens + completion_tokens

    metrics = normalize_generation_metrics(data)
    state = load_state(metrics.get("ollama_load_ms"))
    # `done_reason` dans le LOG et pas seulement dans la réponse : une sortie
    # tronquée par le budget ("length") doit être identifiable a posteriori,
    # sans rejouer l'appel ni lire le texte produit.
    reason = _done_reason(data)

    # Lot G2 — le contrat structuré n'est tenu que si `response` est un
    # document JSON STRICT conforme au schéma (mêmes helpers que /v1/chat).
    # Un document complet et conforme sous `done_reason: length` est accepté ;
    # vide, tronqué ou non conforme → 502 contrôlé, jamais un 200 maquillé.
    structured_output: Optional[Dict[str, Any]] = None
    if plan.response_schema is not None:
        try:
            structured_output = validate_structured_output(data.get("response", ""), plan.response_schema)
        except StructuredOutputError as exc:
            _count_structured_invalid("generate", exc.code)
            counters.inc("generate_upstream_errors_total")
            log_event(
                logger, logging.WARNING, "generate.structured_invalid", model=model,
                code=exc.code, content_chars=len(data.get("response") or "") if isinstance(data.get("response"), str) else 0,
                upstream_ms=upstream_ms, total_ms=int((time.time() - started) * 1000),
                **metrics, **({"done_reason": reason} if reason else {}), **plan.log_fields,
            )
            raise _structured_invalid_error(exc, reason)
        counters.inc("generate_structured_ok_total")
        plan.log_fields["structured_valid"] = True

    log_event(
        logger,
        logging.INFO,
        "generate.ok",
        model=model,
        upstream_ms=upstream_ms,
        total_ms=int((time.time() - started) * 1000),
        **metrics,
        **({"load_state": state} if state else {}),
        **({"done_reason": reason} if reason else {}),
        **plan.log_fields,
    )

    return GenerateResponse(
        text=data.get("response", ""),
        model=model,
        tokens=TokenCounts(
            prompt=prompt_tokens,
            completion=completion_tokens,
            total=total_tokens,
        ),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        cost=0.0,
        done=bool(data.get("done", True)),
        done_reason=_done_reason(data),
        structured_output=structured_output,
        response_schema_sha256=plan.response_schema_sha256,
    )


# --------------------------------------------------------------------------- #
# Generate — flux NDJSON
# --------------------------------------------------------------------------- #
# Endpoint SÉPARÉ, et non un drapeau `stream` sur /v1/generate. Deux raisons :
# le contrat historique (réponse agrégée) reste intact bit pour bit, et un
# client qui envoyait déjà `stream: true` — silencieusement ignoré par Pydantic
# — ne change pas de comportement du jour au lendemain. Le mode vient du
# chemin, pas d'un champ ambigu.

MEDIA_TYPE_NDJSON = "application/x-ndjson"

# Terminaisons possibles d'un flux, journalisées dans `generate_stream.end`.
TERMINATION_COMPLETED = "completed"
TERMINATION_CLIENT_DISCONNECT = "client_disconnect"
TERMINATION_TIMEOUT = "timeout"
TERMINATION_UPSTREAM_ERROR = "upstream_error"
TERMINATION_INVALID_CHUNK = "invalid_upstream_chunk"


def _ndjson(event: BaseModel) -> bytes:
    """Sérialise un événement en une ligne NDJSON complète."""
    return event.model_dump_json().encode("utf-8") + b"\n"


def _stream_error(code: str, message: str) -> bytes:
    return _ndjson(
        GenerateStreamError(error=StreamErrorDetail(code=code, message=message))
    )


def _generate_stream_events(
    stream: GenerateStream,
    plan: GenerationPlan,
    model: str,
    started: float,
    started_monotonic: float,
):
    """Relaie le flux Ollama, normalisé au contrat Gateway.

    Générateur volontairement sans état accumulé : aucun fragment n'est
    conservé, aucun texte n'est reconstitué. La mémoire utilisée est constante,
    quelle que soit la longueur de la génération — la Gateway n'a pas besoin de
    la réponse complète, seul le client en a l'usage.
    """
    termination = TERMINATION_UPSTREAM_ERROR
    first_chunk_ms: Optional[int] = None
    final_event: Optional[Dict[str, Any]] = None

    try:
        for event in stream.events():
            delta = event.get("response")
            if not isinstance(delta, str):
                delta = ""

            # Un fragment vide (métadonnée seule, ou événement final) ne produit
            # pas de ligne `delta` et ne déclenche pas la mesure de premier
            # fragment : le client attend du texte, pas des accusés de réception.
            if delta:
                if first_chunk_ms is None:
                    first_chunk_ms = int(
                        (time.monotonic() - started_monotonic) * 1000
                    )
                yield _ndjson(GenerateStreamDelta(delta=delta))

            if event.get("done"):
                final_event = event
                break

        if final_event is None:
            # Socket fermée sans `done: true` : la génération a été interrompue
            # en amont. On ne fabrique surtout pas un `done` de complaisance.
            raise IncompleteUpstreamStream(
                "échec /api/generate: flux interrompu avant la fin de génération"
            )

        # Comptage identique à /v1/generate — une seule interprétation des
        # tokens dans tout le service.
        prompt_tokens = final_event.get("prompt_eval_count", 0) or 0
        completion_tokens = final_event.get("eval_count", 0) or 0
        yield _ndjson(
            GenerateStreamDone(
                model=model,
                tokens=TokenCounts(
                    prompt=prompt_tokens,
                    completion=completion_tokens,
                    total=prompt_tokens + completion_tokens,
                ),
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                done_reason=_done_reason(final_event),
            )
        )
        termination = TERMINATION_COMPLETED

    except GeneratorExit:
        # Le consommateur a fermé le générateur : client parti. Rien à émettre,
        # plus personne n'écoute — mais le `finally` ferme bien l'upstream.
        termination = TERMINATION_CLIENT_DISCONNECT
        raise
    except StreamTimeout:
        termination = TERMINATION_TIMEOUT
        yield _stream_error(
            "upstream_timeout", "budget de temps de la passerelle épuisé"
        )
    except InvalidUpstreamChunk:
        termination = TERMINATION_INVALID_CHUNK
        yield _stream_error(
            "invalid_upstream_response", "réponse de flux illisible"
        )
    except UpstreamError:
        # Couvre IncompleteUpstreamStream et toute rupture de transport.
        # `exc.message` n'est pas relayé : il peut porter un fragment de corps
        # upstream. Le client reçoit un code stable, l'opérateur a les logs.
        termination = TERMINATION_UPSTREAM_ERROR
        yield _stream_error("upstream_error", "flux interrompu par l'upstream")
    finally:
        stream.close()
        gauges.dec(plan.inflight_key.replace("generate:", "generate_stream:", 1))
        if termination == TERMINATION_TIMEOUT:
            counters.inc("generate_stream_timeouts_total")
        if termination in (TERMINATION_UPSTREAM_ERROR, TERMINATION_INVALID_CHUNK):
            counters.inc("generate_stream_errors_total")
        metrics = normalize_generation_metrics(final_event)
        state = load_state(metrics.get("ollama_load_ms"))
        extra: Dict[str, Any] = {}
        if first_chunk_ms is not None:
            extra["first_chunk_ms"] = first_chunk_ms
        if state:
            extra["load_state"] = state
        log_event(
            logger,
            logging.INFO,
            "generate_stream.end",
            model=model,
            termination=termination,
            total_ms=int((time.time() - started) * 1000),
            **metrics,
            **extra,
            **plan.log_fields,
        )


_ITERATION_END = object()


async def _with_host_lease(events, host_lease, semaphore=None):
    """Rend la lease hôte — et le slot de concurrence Gateway
    d'un FLUX quand il se termine : fin normale, erreur ou départ du client
    (l'annulation ferme ce générateur async, cf. `_drain_to_client`). Sans
    lease (None), transparent pour la partie hôte."""
    try:
        async for item in events:
            yield item
    finally:
        if semaphore is not None:
            semaphore.release()
        host_admission.release(host_lease)


async def _drain_to_client(lines):
    """Pont entre le générateur SYNCHRONE ci-dessus et le monde asynchrone.

    Cette indirection n'est pas cosmétique, elle est le mécanisme d'annulation.
    Starlette itère un générateur *synchrone* dans un threadpool et ne le ferme
    jamais quand le client s'en va : le `finally` du générateur ne s'exécuterait
    pas, l'upstream resterait ouvert et Ollama continuerait d'occuper le GPU
    pour une réponse que plus personne n'attend (constaté en conditions
    réelles : 12 s de génération après le départ du client).

    Avec un générateur *asynchrone*, la déconnexion annule la tâche pendant
    l'`await` ci-dessous, ce qui déclenche le `finally` — donc la fermeture de
    l'upstream et la journalisation de la terminaison.

    Chaque ligne upstream est lue dans un thread pour ne pas bloquer la boucle
    d'événements. L'annulation attend délibérément la fin de la lecture en
    cours (au lieu d'abandonner le thread) : pendant une génération, une ligne
    arrive toutes les quelques millisecondes, et abandonner le thread ferait
    courir le risque qu'il journalise une seconde terminaison en concurrence.
    """
    try:
        while True:
            chunk = await run_in_threadpool(next, lines, _ITERATION_END)
            if chunk is _ITERATION_END:
                return
            yield chunk
    finally:
        # Ferme le générateur synchrone : `GeneratorExit` y remonte, son
        # `finally` ferme l'upstream et journalise la terminaison.
        lines.close()


@app.post("/v1/generate/stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    """Génération en flux NDJSON.

    Une ligne JSON par événement : `{"type":"delta",...}` pour chaque fragment,
    puis exactement un `{"type":"done",...}` portant les comptes de tokens. Une
    rupture en cours de flux produit `{"type":"error",...}` et jamais de `done`.
    """
    started = time.time()
    started_monotonic = time.monotonic()
    counters.inc("generate_stream_requests_total")
    if req.response_format is not None:
        # Lot G2 : pas de sortie structurée en flux (document complet requis
        # pour valider) — refus AVANT toute résolution, comme /v1/chat/stream.
        counters.inc("generate_structured_streaming_rejected_total")
        raise http_error(
            400,
            "la sortie structurée n'est pas disponible en flux : utiliser /v1/generate",
            provider="ai_gateway",
            extra={"code": "structured_output_streaming_not_supported"},
        )
    plan = _plan_generation(req)
    model = plan.upstream_model
    logger.info("generate_stream model=%s prompt_len=%d", model, len(req.prompt))

    # Ouverture bloquante déportée dans un thread : tout ce qui échoue ici
    # (connexion refusée, statut HTTP d'erreur) survient AVANT le premier octet
    # de réponse et reste donc un 502 classique, cohérent avec /v1/generate.
    inflight_key = plan.inflight_key.replace("generate:", "generate_stream:", 1)
    try:
        host_lease = await run_in_threadpool(
            host_admission.acquire, "llm_generation", plan.log_fields.get("model_logical") or model)
    except HostAdmissionError as exc:
        raise host_admission_http_error(exc)
    if not _generate_semaphore.acquire(blocking=False):
        host_admission.release(host_lease)
        counters.inc("generate_saturated_total")
        raise _saturated_error("generation_saturated", "génération saturée : concurrence maximale atteinte",
                               GENERATE_RETRY_AFTER_SECONDS)
    host_admission.mark_started(host_lease)   # lot ORPHAN-LEASE : backend engagé à partir d'ici
    gauges.inc(inflight_key)
    try:
        stream = await run_in_threadpool(
            client.open_generate_stream,
            model=model,
            prompt=req.prompt,
            system=req.system,
            temperature=req.temperature,
            options=plan.options,
            think=plan.think,
            keep_alive=plan.keep_alive,
            **({"format": req.format} if req.format is not None else {}),
        )
    except UpstreamModelNotFound:
        gauges.dec(inflight_key)
        _generate_semaphore.release()
        host_admission.release(host_lease)
        raise _model_not_installed_error(model, "generate")
    except UpstreamError as exc:
        # La jauge du flux est normalement rendue par le finally du générateur ;
        # ici le flux n'a jamais existé, on la rend nous-mêmes.
        gauges.dec(inflight_key)
        _generate_semaphore.release()
        host_admission.release(host_lease)
        counters.inc("generate_stream_errors_total")
        logger.warning("generate_stream: échec upstream: %s", exc.message)
        raise http_error(502, PUBLIC_GENERATION_FAILED, provider="ollama")

    return StreamingResponse(
        _with_host_lease(_drain_to_client(
            _generate_stream_events(stream, plan, model, started, started_monotonic)
        ), host_lease, _generate_semaphore),
        media_type=MEDIA_TYPE_NDJSON,
        # Un flux relu depuis un cache n'aurait aucun sens ; l'en-tête décourage
        # aussi la mise en tampon par un intermédiaire.
        headers={"Cache-Control": "no-cache"},
    )


# --------------------------------------------------------------------------- #
# Chat — transport /api/chat gouverné (lot C1)
# --------------------------------------------------------------------------- #
# Route ADDITIVE et transport SÉPARÉ : `/v1/generate` continue de parler
# `/api/generate` avec `_plan_generation`, intouché. `/v1/chat` parle
# `/api/chat` avec son propre plan, qui RÉUTILISE les mêmes règles gouvernées
# (`_resolve_num_ctx`, `_resolve_num_predict`, profils de contexte, de
# génération et de rétention, allowlist d'environnement) sans les dupliquer ni
# les modifier.
#
# Deux différences de politique, voulues et documentées :
#   · la capability `chat` est OPT-IN par modèle du registre ; un modèle qui ne
#     la déclare pas est refusé en 400 AVANT tout appel Ollama ;
#   · il n'existe AUCUN chemin legacy : un modèle hors registre est refusé
#     quel que soit `AI_GATEWAY_GENERATION_STRICT_REGISTRY`. Le relais
#     historique de `/v1/generate` est une compatibilité héritée ; une API
#     plus riche (messages, raisonnement, bientôt outils) ne recrée pas un
#     chemin non gouverné.
#
# Concurrence (doctrine P8) : même backend GPU que `/v1/generate`, et comme
# lui aucune sémaphore applicative — Ollama sérialise lui-même (NUM_PARALLEL
# effectif 1, mesuré). Une file bloquante ici doublerait la charge de threads
# sans rien protéger. Seule la jauge `chat:<modèle>` rend l'inflight visible.


@dataclass
class ChatPlan:
    """Paramètres effectifs d'un appel `/api/chat`, après gouvernance."""

    upstream_model: str
    options: Dict[str, Any]
    # Valeur Ollama `think` déjà VALIDÉE par le profil (bool, niveau ou None).
    # Aucune valeur client brute ne transite jusqu'ici.
    think: Optional[Any]
    keep_alive: Optional[Any]
    # Politique d'exposition de la trace de réflexion (profil de raisonnement ;
    # sans profil, jamais exposée).
    expose_thinking: bool
    # Mode de raisonnement Gateway EFFECTIF (vocabulaire borné) ou None.
    reasoning: Optional[str]
    log_fields: Dict[str, Any]
    inflight_key: str
    # Lot C3 — schéma VALIDÉ relayé dans `format`, ou None (chat normal).
    response_schema: Optional[Dict[str, Any]] = None
    schema_stats: Optional[SchemaStats] = None


CHAT_CAPABILITY = "chat"
STRUCTURED_OUTPUT_CAPABILITY = "structured_output"


def _plan_chat(req: ChatRequest) -> ChatPlan:
    model = req.model
    resolved = registry.resolve_llm(model)
    if resolved is None:
        counters.inc("chat_rejected_unregistered_total")
        log_event(logger, logging.WARNING, "chat.model_not_registered", model=model)
        raise http_error(
            400,
            f"modèle {model} absent du registre : /v1/chat exige un modèle "
            f"enregistré déclarant la capability {CHAT_CAPABILITY}",
            provider="ai_gateway",
            extra={"code": "model_not_registered", "chat_requires_registry": True},
        )
    profile = resolved.profile
    log_fields: Dict[str, Any] = {"model_logical": profile.logical_name}

    if CHAT_CAPABILITY not in profile.capabilities:
        counters.inc("chat_rejected_capability_total")
        log_event(
            logger, logging.WARNING, "chat.capability_not_supported",
            model=profile.ollama_name, model_logical=profile.logical_name,
        )
        raise http_error(
            400,
            f"le modèle {profile.logical_name} ne déclare pas la capability "
            f"{CHAT_CAPABILITY}",
            provider="ai_gateway",
            extra={
                "code": "chat_not_supported",
                "model_logical": profile.logical_name,
                "capabilities": list(profile.capabilities),
            },
        )

    if not profile.allowed_in(settings.environment):
        counters.inc("chat_rejected_environment_total")
        raise http_error(
            400,
            f"modèle {profile.logical_name} non autorisé dans l'environnement "
            f"{settings.environment}",
            provider="ai_gateway",
            extra={
                "model_logical": profile.logical_name,
                "environment": settings.environment,
            },
        )

    # Sortie structurée (lot C3) — capability OPT-IN, décidée par le
    # registre (jamais par un nom de modèle), puis schéma validé contre le
    # sous-ensemble et les bornes. Rien ne part vers Ollama avant ces deux
    # contrôles. Le schéma lui-même n'est jamais journalisé : seules ses
    # dimensions le sont.
    response_schema: Optional[Dict[str, Any]] = None
    schema_stats: Optional[SchemaStats] = None
    if req.response_format is not None:
        counters.inc("chat_structured_requests_total")
        if STRUCTURED_OUTPUT_CAPABILITY not in profile.capabilities:
            counters.inc("chat_structured_capability_rejected_total")
            log_event(
                logger, logging.WARNING, "chat.structured_output_not_supported",
                model=profile.ollama_name, model_logical=profile.logical_name,
            )
            raise http_error(
                400,
                f"le modèle {profile.logical_name} ne déclare pas la capability "
                f"{STRUCTURED_OUTPUT_CAPABILITY}",
                provider="ai_gateway",
                extra={
                    "code": "structured_output_not_supported",
                    "model_logical": profile.logical_name,
                    "capabilities": list(profile.capabilities),
                },
            )
        try:
            schema_stats = validate_client_schema(req.response_format.schema_)
        except SchemaError as exc:
            counters.inc("chat_structured_schema_rejected_total")
            log_event(
                logger, logging.WARNING, "chat.schema_rejected",
                model=profile.ollama_name, model_logical=profile.logical_name,
                code=exc.code, **{k: v for k, v in exc.extra.items() if k in ("limit", "value", "max", "keyword")},
            )
            raise _schema_rejected_error(exc, profile.logical_name)
        response_schema = req.response_format.schema_
        log_fields["structured_output"] = True
        log_fields["schema_bytes"] = schema_stats.bytes
        log_fields["schema_depth"] = schema_stats.depth
        log_fields["schema_nodes"] = schema_stats.nodes

    context = registry.context_profile_of(profile)
    options = dict(req.options or {})
    effective_num_ctx, requested_num_ctx = _resolve_num_ctx(req.options, context)
    options["num_ctx"] = effective_num_ctx
    if requested_num_ctx is not None:
        log_fields["num_ctx"] = requested_num_ctx
    log_fields["context_profile"] = context.name
    log_fields["effective_num_ctx"] = effective_num_ctx

    generation = registry.generation_profile_of(profile)
    effective_num_predict, requested_num_predict = _resolve_num_predict(
        req.options, generation
    )
    options["num_predict"] = effective_num_predict
    log_fields["generation_profile"] = generation.name
    log_fields["effective_num_predict"] = effective_num_predict
    if requested_num_predict is not None:
        log_fields["requested_num_predict"] = requested_num_predict

    # Raisonnement — cœur du lot C1. La demande client (abstraite) est
    # confrontée au profil du modèle ; seule la valeur `think` qui en découle
    # part vers Ollama. Une demande non honorable est un 400 ici, jamais un
    # `think: false` envoyé « en espérant » (modèle non désactivable).
    reasoning_profile = registry.reasoning_profile_of(profile)
    try:
        decision = resolve_reasoning(reasoning_profile, req.reasoning)
    except ReasoningModeError as exc:
        counters.inc("chat_rejected_reasoning_total")
        log_event(
            logger, logging.WARNING, "chat.reasoning_rejected",
            model=profile.ollama_name, model_logical=profile.logical_name,
            reasoning_profile=reasoning_profile.name, code=exc.code,
            **({"requested": req.reasoning} if req.reasoning is not None else {}),
        )
        raise http_error(
            400, exc.message, provider="ai_gateway",
            extra={"code": exc.code, "model_logical": profile.logical_name, **exc.extra},
        )
    log_fields["reasoning_profile"] = reasoning_profile.name
    if decision.mode is not None:
        log_fields["reasoning"] = decision.mode
    if decision.think is not None:
        log_fields["effective_think"] = decision.think

    retention = registry.keep_alive_of(profile)
    keep_alive = req.keep_alive
    if keep_alive is not None and not retention.client_override:
        counters.inc("chat_rejected_keep_alive_total")
        raise http_error(
            400,
            f"keep_alive n'est pas surchargeable pour le modèle "
            f"{profile.logical_name} (profil {retention.name})",
            provider="ai_gateway",
            extra={
                "code": "keep_alive_not_overridable",
                "model_logical": profile.logical_name,
                "keep_alive_profile": retention.name,
            },
        )
    if keep_alive is None:
        keep_alive = retention.keep_alive
        if keep_alive is not None:
            log_fields["keep_alive_profile"] = profile.keep_alive_profile

    return ChatPlan(
        upstream_model=profile.ollama_name,
        options=options,
        think=decision.think,
        keep_alive=keep_alive,
        expose_thinking=reasoning_profile.expose_thinking,
        reasoning=decision.mode,
        log_fields=log_fields,
        inflight_key=f"chat:{profile.ollama_name}",
        response_schema=response_schema,
        schema_stats=schema_stats,
    )


def _chat_messages(req: ChatRequest) -> List[Dict[str, str]]:
    """Messages relayés : STRICTEMENT `role` + `content` validés par le schéma."""
    return [{"role": m.role, "content": m.content} for m in req.messages]


def _chat_message_fields(data: Dict[str, Any]) -> Tuple[str, Optional[str], bool]:
    """(content, thinking, has_tool_calls) depuis une réponse `/api/chat`.

    Tolérant sur la forme (runtime plus ancien ou plus récent) : un champ
    absent ou d'un autre type vaut « absent », jamais une exception de
    sérialisation. `thinking` et `content` ne sont JAMAIS fusionnés.
    """
    message = data.get("message")
    if not isinstance(message, dict):
        message = {}
    content = message.get("content")
    if not isinstance(content, str):
        content = ""
    thinking = message.get("thinking")
    if not isinstance(thinking, str) or thinking == "":
        thinking = None
    has_tool_calls = bool(message.get("tool_calls"))
    return content, thinking, has_tool_calls


def _content_is_empty(content: str) -> bool:
    """Définition UNIQUE de `content_empty` : vide ou uniquement blanc."""
    return content.strip() == ""


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> JSONResponse:
    started = time.time()
    counters.inc("chat_requests_total")
    plan = _plan_chat(req)
    model = plan.upstream_model
    # Tailles seulement : ni rôle par rôle, ni contenu.
    logger.info("chat model=%s messages=%d", model, len(req.messages))

    try:
        host_lease = host_admission.acquire("llm_generation", plan.log_fields.get("model_logical") or model)
    except HostAdmissionError as exc:
        raise host_admission_http_error(exc)
    if not _generate_semaphore.acquire(blocking=False):
        host_admission.release(host_lease)
        counters.inc("generate_saturated_total")
        raise _saturated_error("generation_saturated", "génération saturée : concurrence maximale atteinte",
                               GENERATE_RETRY_AFTER_SECONDS)
    host_admission.mark_started(host_lease)   # lot ORPHAN-LEASE : backend engagé à partir d'ici
    gauges.inc(plan.inflight_key)
    upstream_started = time.time()
    try:
        data = client.chat(
            model=model,
            messages=_chat_messages(req),
            temperature=req.temperature,
            options=plan.options,
            think=plan.think,
            keep_alive=plan.keep_alive,
            **({"format": plan.response_schema} if plan.response_schema is not None else {}),
        )
    except UpstreamTimeout as exc:
        counters.inc("chat_upstream_errors_total")
        counters.inc("chat_timeouts_total")
        log_event(
            logger, logging.WARNING, "chat.timeout", model=model,
            elapsed_ms=int((time.time() - upstream_started) * 1000),
            timeout_seconds=client.timeout, **plan.log_fields,
        )
        logger.warning("chat: échec upstream: %s", exc.message)
        raise http_error(502, PUBLIC_GENERATION_TIMEOUT, provider="ollama")
    except UpstreamModelNotFound:
        raise _model_not_installed_error(model, "chat")
    except UpstreamError as exc:
        counters.inc("chat_upstream_errors_total")
        logger.warning("chat: échec upstream: %s", exc.message)
        raise http_error(502, PUBLIC_GENERATION_FAILED, provider="ollama")
    finally:
        gauges.dec(plan.inflight_key)
        _generate_semaphore.release()
        host_admission.release(host_lease)
    upstream_ms = int((time.time() - upstream_started) * 1000)

    content, thinking, has_tool_calls = _chat_message_fields(data)
    if has_tool_calls:
        # Aucun outil n'a été transmis : un `tool_calls` est une réponse hors
        # contrat. Refus explicite plutôt qu'un contrat public accidentel ou
        # un « succès » au contenu vide.
        counters.inc("chat_unexpected_tool_calls_total")
        counters.inc("chat_upstream_errors_total")
        log_event(logger, logging.WARNING, "chat.unexpected_tool_calls",
                  model=model, **plan.log_fields)
        raise http_error(
            502, PUBLIC_CHAT_UNSUPPORTED_RESPONSE, provider="ollama",
            extra={"code": "unsupported_upstream_response"},
        )

    prompt_tokens = data.get("prompt_eval_count", 0) or 0
    completion_tokens = data.get("eval_count", 0) or 0
    total_tokens = prompt_tokens + completion_tokens
    reason = _done_reason(data)
    content_empty = _content_is_empty(content)
    thinking_chars = len(thinking) if thinking else 0

    # Sortie structurée (lot C3) : le contrat demandé n'est satisfait que si
    # `message.content` — et lui seul, jamais la réflexion — est un document
    # JSON STRICT conforme au schéma. Sinon 502 contrôlé : un contenu vide
    # ou tronqué (budget épuisé par la réflexion) n'est pas un « succès à
    # content_empty » ici, c'est un contrat non tenu. Un document complet et
    # conforme sous `done_reason: length` est accepté tel quel, la raison
    # d'arrêt restant visible pour le client.
    structured_output: Optional[Dict[str, Any]] = None
    if plan.response_schema is not None:
        try:
            structured_output = validate_structured_output(content, plan.response_schema)
        except StructuredOutputError as exc:
            _count_structured_invalid("chat", exc.code)
            counters.inc("chat_upstream_errors_total")
            log_event(
                logger, logging.WARNING, "chat.structured_invalid", model=model,
                code=exc.code, content_chars=len(content), thinking_chars=thinking_chars,
                **({"done_reason": reason} if reason else {}), **plan.log_fields,
            )
            raise _structured_invalid_error(exc, reason)
        counters.inc("chat_structured_ok_total")
        plan.log_fields["structured_valid"] = True

    if content_empty:
        counters.inc("chat_empty_content_total")
        log_event(
            logger, logging.WARNING, "chat.empty_content", model=model,
            thinking_chars=thinking_chars,
            **({"done_reason": reason} if reason else {}), **plan.log_fields,
        )

    metrics = normalize_generation_metrics(data)
    state = load_state(metrics.get("ollama_load_ms"))
    # Journal : tailles et faits, jamais le contenu ni la réflexion.
    log_event(
        logger, logging.INFO, "chat.ok", model=model,
        upstream_ms=upstream_ms, total_ms=int((time.time() - started) * 1000),
        messages=len(req.messages), content_chars=len(content),
        thinking_chars=thinking_chars, content_empty=content_empty,
        **metrics,
        **({"load_state": state} if state else {}),
        **({"done_reason": reason} if reason else {}),
        **plan.log_fields,
    )

    response = ChatResponse(
        model=model,
        message=ChatAssistantMessage(content=content),
        thinking=thinking if plan.expose_thinking else None,
        thinking_exposed=plan.expose_thinking,
        thinking_chars=thinking_chars,
        reasoning=plan.reasoning,
        done=bool(data.get("done", True)),
        done_reason=reason,
        content_empty=content_empty,
        tokens=TokenCounts(prompt=prompt_tokens, completion=completion_tokens, total=total_tokens),
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens),
        cost=0.0,
        structured_output=structured_output,
    )
    # `structured_output` n'existe dans la réponse QUE si la requête portait
    # `response_format` : le contrat C2 d'un chat normal reste identique.
    exclude = None if plan.response_schema is not None else {"structured_output"}
    return JSONResponse(jsonable_encoder(response, exclude=exclude))


# Terminaison supplémentaire propre au chat (lot C1).
TERMINATION_UNSUPPORTED_RESPONSE = "unsupported_upstream_response"


def _chat_stream_events(
    stream: ChatStream,
    plan: ChatPlan,
    model: str,
    started: float,
    started_monotonic: float,
):
    """Relaie un flux `/api/chat` au contrat Gateway (thinking / content / done).

    Même discipline que `_generate_stream_events` : mémoire constante (aucun
    texte accumulé — seul un booléen « du contenu non blanc a été émis »
    sert à `content_empty`), upstream fermé dans tous les cas, terminaison
    journalisée sans contenu. Le parser est PROPRE au chat : la forme Ollama
    (`message.content`, `message.thinking`) diffère de `/api/generate`.
    """
    termination = TERMINATION_UPSTREAM_ERROR
    first_chunk_ms: Optional[int] = None
    final_event: Optional[Dict[str, Any]] = None
    saw_content = False
    thinking_chars = 0

    try:
        for event in stream.events():
            content, thinking, has_tool_calls = _chat_message_fields(event)
            if has_tool_calls:
                raise UnsupportedUpstreamResponse(
                    "échec /api/chat: tool_calls reçus sans outil déclaré"
                )
            if thinking:
                thinking_chars += len(thinking)
                if plan.expose_thinking:
                    yield _ndjson(ChatStreamThinking(delta=thinking))
            if content:
                if not saw_content and content.strip():
                    saw_content = True
                if first_chunk_ms is None:
                    first_chunk_ms = int((time.monotonic() - started_monotonic) * 1000)
                yield _ndjson(ChatStreamContent(delta=content))
            if event.get("done"):
                final_event = event
                break

        if final_event is None:
            raise IncompleteUpstreamStream(
                "échec /api/chat: flux interrompu avant la fin de génération"
            )

        prompt_tokens = final_event.get("prompt_eval_count", 0) or 0
        completion_tokens = final_event.get("eval_count", 0) or 0
        content_empty = not saw_content
        if content_empty:
            counters.inc("chat_empty_content_total")
        yield _ndjson(
            ChatStreamDone(
                model=model,
                tokens=TokenCounts(prompt=prompt_tokens, completion=completion_tokens,
                                   total=prompt_tokens + completion_tokens),
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens),
                done_reason=_done_reason(final_event),
                content_empty=content_empty,
                thinking_exposed=plan.expose_thinking,
                reasoning=plan.reasoning,
            )
        )
        termination = TERMINATION_COMPLETED

    except GeneratorExit:
        termination = TERMINATION_CLIENT_DISCONNECT
        raise
    except StreamTimeout:
        termination = TERMINATION_TIMEOUT
        yield _stream_error("upstream_timeout", "budget de temps de la passerelle épuisé")
    except InvalidUpstreamChunk:
        termination = TERMINATION_INVALID_CHUNK
        yield _stream_error("invalid_upstream_response", "réponse de flux illisible")
    except UnsupportedUpstreamResponse:
        termination = TERMINATION_UNSUPPORTED_RESPONSE
        counters.inc("chat_unexpected_tool_calls_total")
        yield _stream_error("unsupported_upstream_response", PUBLIC_CHAT_UNSUPPORTED_RESPONSE)
    except UpstreamError:
        termination = TERMINATION_UPSTREAM_ERROR
        yield _stream_error("upstream_error", "flux interrompu par l'upstream")
    finally:
        stream.close()
        gauges.dec(plan.inflight_key.replace("chat:", "chat_stream:", 1))
        if termination == TERMINATION_TIMEOUT:
            counters.inc("chat_stream_timeouts_total")
        if termination in (TERMINATION_UPSTREAM_ERROR, TERMINATION_INVALID_CHUNK,
                           TERMINATION_UNSUPPORTED_RESPONSE):
            counters.inc("chat_stream_errors_total")
        metrics = normalize_generation_metrics(final_event)
        state = load_state(metrics.get("ollama_load_ms"))
        extra: Dict[str, Any] = {"thinking_chars": thinking_chars, "content_empty": not saw_content}
        if first_chunk_ms is not None:
            extra["first_chunk_ms"] = first_chunk_ms
        if state:
            extra["load_state"] = state
        if final_event is not None and _done_reason(final_event):
            extra["done_reason"] = _done_reason(final_event)
        log_event(
            logger, logging.INFO, "chat_stream.end", model=model,
            termination=termination, total_ms=int((time.time() - started) * 1000),
            **metrics, **extra, **plan.log_fields,
        )


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Chat en flux NDJSON : `{"type":"thinking"}` (si exposé),
    `{"type":"content"}`, puis exactement un `{"type":"done"}` portant
    `done_reason` et `content_empty`. Une rupture en cours de flux produit
    `{"type":"error"}` et jamais de `done`."""
    started = time.time()
    started_monotonic = time.monotonic()
    counters.inc("chat_stream_requests_total")
    if req.response_format is not None:
        # Lot C3 : la conformité d'un document ne se garantit qu'une fois le
        # document complet — incompatible avec un relais fragment par
        # fragment. Refus AVANT résolution du modèle et avant tout appel.
        counters.inc("chat_structured_streaming_rejected_total")
        raise http_error(
            400,
            "la sortie structurée n'est pas disponible en flux : utiliser /v1/chat",
            provider="ai_gateway",
            extra={"code": "structured_output_streaming_not_supported"},
        )
    plan = _plan_chat(req)
    model = plan.upstream_model
    logger.info("chat_stream model=%s messages=%d", model, len(req.messages))

    inflight_key = plan.inflight_key.replace("chat:", "chat_stream:", 1)
    try:
        host_lease = await run_in_threadpool(
            host_admission.acquire, "llm_generation", plan.log_fields.get("model_logical") or model)
    except HostAdmissionError as exc:
        raise host_admission_http_error(exc)
    if not _generate_semaphore.acquire(blocking=False):
        host_admission.release(host_lease)
        counters.inc("generate_saturated_total")
        raise _saturated_error("generation_saturated", "génération saturée : concurrence maximale atteinte",
                               GENERATE_RETRY_AFTER_SECONDS)
    host_admission.mark_started(host_lease)   # lot ORPHAN-LEASE : backend engagé à partir d'ici
    gauges.inc(inflight_key)
    try:
        stream = await run_in_threadpool(
            client.open_chat_stream,
            model=model,
            messages=_chat_messages(req),
            temperature=req.temperature,
            options=plan.options,
            think=plan.think,
            keep_alive=plan.keep_alive,
        )
    except UpstreamModelNotFound:
        gauges.dec(inflight_key)
        _generate_semaphore.release()
        host_admission.release(host_lease)
        raise _model_not_installed_error(model, "chat")
    except UpstreamError as exc:
        gauges.dec(inflight_key)
        _generate_semaphore.release()
        host_admission.release(host_lease)
        counters.inc("chat_stream_errors_total")
        logger.warning("chat_stream: échec upstream: %s", exc.message)
        raise http_error(502, PUBLIC_GENERATION_FAILED, provider="ollama")

    return StreamingResponse(
        _with_host_lease(_drain_to_client(
            _chat_stream_events(stream, plan, model, started, started_monotonic)
        ), host_lease, _generate_semaphore),
        media_type=MEDIA_TYPE_NDJSON,
        headers={"Cache-Control": "no-cache"},
    )


# --------------------------------------------------------------------------- #
# Rerank — capability optionnelle (lot rerank-sidecar-staging)
# --------------------------------------------------------------------------- #
# Capability DÉSACTIVÉE par défaut : sans AI_GATEWAY_RERANK_ENABLED=1, la route
# répond 503 explicite et rien d'autre ne change — generate, embeddings et la
# readiness historique n'ont AUCUNE dépendance vers le backend reranker.
# Le backend est partagé prod/staging (une seule copie GPU). L'allowlist
# d'environnement du registre reste non contournable ; depuis le lot
# rerank-production-capability-readiness elle autorise la production
# (la frontière OFF restante est côté consommateur RagATM).

# Concurrence bornée : un sémaphore, pas un scheduler. Deux requêtes passent,
# les suivantes attendent leur tour (borné par le timeout), l'empilement
# d'activations GPU reste contrôlé.
_rerank_semaphore = threading.BoundedSemaphore(max(1, settings.rerank_max_concurrency))

# Lot P8 — saturation : `Retry-After` FIXE et documenté. Ce n'est pas une
# prédiction de la durée d'attente backend (inconnaissable ici) mais un délai
# de politesse avant retry client. Rerank : un lot se joue en centaines de ms
# (mesuré) — 1 s. Vision : un describe coûte 9,5 s p50 (mesuré) — 10 s.
RERANK_RETRY_AFTER_SECONDS = 1
VISION_RETRY_AFTER_SECONDS = 10

RERANK_PROBE_QUERY = "controle de sante du reranker"
RERANK_PROBE_TEXT = "texte de sonde interne de la passerelle"


def _rerank_capability_blocker() -> Optional[str]:
    """Raison pour laquelle la capability ne peut pas servir, ou None.

    L'URL non-loopback est traitée comme une capability indisponible : le
    backend n'a ni authentification ni gouvernance propres, le servir à
    travers le LAN ouvrirait un chemin de scoring non gouverné.
    """
    if not settings.rerank_enabled:
        return "capability reranking désactivée (AI_GATEWAY_RERANK_ENABLED)"
    if not backend_url_is_loopback(settings.rerank_backend_url):
        return "backend reranker hors loopback : capability refusée"
    return None


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    started = time.time()
    counters.inc("rerank_requests_total")

    blocker = _rerank_capability_blocker()
    if blocker is not None:
        raise http_error(503, blocker, provider="ai_gateway")

    requested_model = req.model or settings.rerank_model
    profile = registry.resolve_reranker(requested_model)
    if profile is None:
        counters.inc("rerank_client_errors_total")
        raise http_error(
            400,
            f"reranker inconnu : {requested_model}",
            provider="ai_gateway",
        )

    # Gouvernance d'environnement — même sémantique, même non-contournement
    # que les embeddings : production_allowed=false est une décision, pas un
    # défaut de configuration.
    if not profile.allowed_in(settings.environment):
        counters.inc("rerank_client_errors_total")
        raise http_error(
            400,
            f"reranker {profile.logical_name} non autorisé dans l'environnement "
            f"{settings.environment}",
            provider="ai_gateway",
            extra={
                "model_logical": profile.logical_name,
                "environment": settings.environment,
            },
        )

    n_documents = len(req.documents)
    if req.top_n is not None and req.top_n > n_documents:
        counters.inc("rerank_client_errors_total")
        raise http_error(
            400,
            f"top_n ({req.top_n}) > nombre de documents ({n_documents})",
            provider="ai_gateway",
        )

    # Limites gouvernées, fail-closed. Les messages citent index et tailles,
    # JAMAIS le contenu ni l'id client (l'id peut être signifiant).
    def _oversize(message: str, **extra: Any):
        counters.inc("rerank_rejected_oversize_total")
        counters.inc("rerank_client_errors_total")
        return http_error(400, message, provider="ai_gateway", extra=extra or None)

    if n_documents > profile.max_documents_per_request:
        raise _oversize(
            f"trop de documents ({n_documents} > "
            f"{profile.max_documents_per_request})",
        )
    if len(req.query) > profile.max_query_chars:
        raise _oversize(
            f"query trop longue ({len(req.query)} caractères > "
            f"{profile.max_query_chars})",
        )
    total_chars = len(req.query)
    for position, document in enumerate(req.documents):
        length = len(document.text)
        total_chars += length
        if length > profile.max_document_chars:
            raise _oversize(
                f"document {position} trop long ({length} caractères > "
                f"{profile.max_document_chars})",
                index=position,
                length=length,
                max_document_chars=profile.max_document_chars,
            )
    if total_chars > profile.max_total_chars:
        raise _oversize(
            f"requête trop volumineuse ({total_chars} caractères > "
            f"{profile.max_total_chars})",
            total_chars=total_chars,
            max_total_chars=profile.max_total_chars,
        )

    # Concurrence bornée — acquisition NON BLOQUANTE. L'ancienne
    # attente `acquire(timeout=15)` se faisait DANS un thread du pool AnyIO
    # partagé (40) : 45 rerank simultanés à concurrence 1 immobilisaient 40
    # threads pendant 15 s, et /health — route synchrone comme les autres —
    # attendait 28 s (mesuré) contre 10 s de healthcheck Docker. Une requête
    # qui attend une ressource limitée ne doit pas consommer un thread
    # partagé pendant cette attente : à capacité pleine, elle est refusée
    # immédiatement avec `Retry-After`, et c'est au client de réessayer.
    # `acquire(blocking=False)` est atomique — pas de « test puis acquiert ».
    # Corollaire : plus aucune file interne, donc plus de jauge `queued:`.
    # Admission HÔTE d'abord (lease rendue si la garde locale refuse ensuite),
    # puis la garde locale existante, conservée comme dernière défense.
    try:
        host_lease = host_admission.acquire("rerank", profile.logical_name)
    except HostAdmissionError as exc:
        raise host_admission_http_error(exc)
    if not _rerank_semaphore.acquire(blocking=False):
        host_admission.release(host_lease)
        counters.inc("rerank_saturated_total")
        raise http_error(
            503,
            "reranker saturé : concurrence maximale atteinte",
            provider="ai_gateway",
            extra={
                "code": "rerank_saturated",
                "retryable": True,
                "retry_after_seconds": RERANK_RETRY_AFTER_SECONDS,
            },
            headers={"Retry-After": str(RERANK_RETRY_AFTER_SECONDS)},
        )
    host_admission.mark_started(host_lease)   # lot ORPHAN-LEASE : backend engagé à partir d'ici
    gauges.inc(f"rerank:{profile.logical_name}")
    upstream_started = time.time()
    try:
        scored = rerank_client.rerank(
            req.query, [document.text for document in req.documents]
        )
    except UpstreamError as exc:
        counters.inc("rerank_upstream_errors_total")
        logger.warning("rerank: échec backend: %s", exc.message)
        raise http_error(502, PUBLIC_RERANK_FAILED, provider="local_cross_encoder")
    finally:
        gauges.dec(f"rerank:{profile.logical_name}")
        _rerank_semaphore.release()
        host_admission.release(host_lease)
    upstream_ms = int((time.time() - upstream_started) * 1000)

    # Cardinalité et unicité : chaque document exactement un score. Un backend
    # qui répond autre chose est en faute — 502, jamais un résultat partiel.
    if len(scored) != n_documents:
        counters.inc("rerank_upstream_errors_total")
        raise http_error(
            502,
            f"nombre de scores ({len(scored)}) != nombre de documents "
            f"({n_documents})",
            provider="local_cross_encoder",
        )
    seen_indices = set()
    for entry in scored:
        index = entry["index"]
        # Lot P17 : un score non fini (NaN, ±Inf) ne doit JAMAIS atteindre le
        # tri ni la sérialisation JSON — `json.dumps` lève sur NaN et la
        # réponse partait en 500 non maîtrisé. Le client transport valide déjà
        # la finitude, mais la route est le point de passage OBLIGÉ quel que
        # soit le backend branché : c'est ici que l'invariant est tenu. Un
        # seul score invalide rend TOUTE la réponse fautive — jamais de
        # filtrage, de clamp ni de remplacement par 0.
        score = entry.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) \
                or not math.isfinite(score):
            counters.inc("rerank_upstream_errors_total")
            log_event(logger, logging.WARNING, "rerank.invalid_score",
                      model_logical=profile.logical_name, index=index)
            raise http_error(
                502,
                "score de reranking invalide (valeur non finie)",
                provider="local_cross_encoder",
            )
        if index < 0 or index >= n_documents or index in seen_indices:
            counters.inc("rerank_upstream_errors_total")
            raise http_error(
                502,
                "indices de scores invalides ou dupliqués",
                provider="local_cross_encoder",
            )
        seen_indices.add(index)

    # Tri décroissant, départage DÉTERMINISTE par index d'origine croissant :
    # deux exécutions identiques rendent exactement le même ordre.
    ranked = sorted(scored, key=lambda entry: (-entry["score"], entry["index"]))
    if req.top_n is not None:
        ranked = ranked[: req.top_n]

    counters.inc("rerank_pairs_total", n_documents)
    log_event(
        logger,
        logging.INFO,
        "rerank.ok",
        model_logical=profile.logical_name,
        n_documents=n_documents,
        returned=len(ranked),
        query_chars=len(req.query),
        total_document_chars=total_chars - len(req.query),
        upstream_ms=upstream_ms,
        total_ms=int((time.time() - started) * 1000),
        **(
            {"pairs_per_second": round(n_documents / (upstream_ms / 1000.0), 2)}
            if upstream_ms > 0
            else {}
        ),
    )

    return RerankResponse(
        model=profile.artifact,
        model_logical=profile.logical_name,
        model_revision=profile.revision,
        results=[
            RerankResult(
                index=entry["index"],
                id=req.documents[entry["index"]].id,
                relevance_score=entry["score"],
            )
            for entry in ranked
        ],
        usage=RerankUsage(documents=n_documents, returned=len(ranked)),
        score_semantics=RerankScoreSemantics(
            higher_is_better=profile.higher_is_better,
            calibrated=profile.calibrated,
        ),
    )


@app.get("/health/rerank")
def health_rerank() -> JSONResponse:
    """Diagnostic de la capability reranking.

    Capability désactivée => 200 avec status `disabled` : c'est un état de
    CONFIGURATION voulu, pas une panne. Backend absent/divergent => 503.
    Cette route sonde réellement le backend (info + scoring d'une paire
    interne constante) : ne pas l'appeler en boucle.

    Indépendance totale de `/health/ready` : un reranker en panne ne rend
    jamais le conteneur unhealthy pour generate/embeddings.
    """
    checks: Dict[str, Any] = {}
    payload: Dict[str, Any] = {
        "status": "error",
        "provider": "ai_gateway",
        "backend_url": "redacted",
        "environment": settings.environment,
        "checks": checks,
    }

    blocker = _rerank_capability_blocker()
    if blocker is not None:
        payload["status"] = "disabled"
        payload["message"] = blocker
        return JSONResponse(status_code=200, content=payload)

    profile = registry.resolve_reranker(settings.rerank_model)
    checks["registry_profile"] = profile is not None
    if profile is None:
        payload["message"] = (
            f"reranker par défaut absent du registre : {settings.rerank_model}"
        )
        return JSONResponse(status_code=503, content={"detail": payload})

    payload["model_logical"] = profile.logical_name
    payload["artifact"] = profile.artifact
    payload["expected_revision"] = profile.revision
    payload["allowed_here"] = profile.allowed_in(settings.environment)

    try:
        info = rerank_client.info(timeout=5)
    except UpstreamError as exc:
        checks["backend_reachable"] = False
        logger.warning("health/rerank: backend injoignable: %s", exc.message)
        payload["message"] = PUBLIC_RERANK_BACKEND_UNREACHABLE
        return JSONResponse(status_code=503, content={"detail": payload})
    checks["backend_reachable"] = True

    observed_artifact = info.get("model_id")
    observed_revision = (info.get("model_sha") or "").strip().lower()
    payload["observed_revision"] = observed_revision or None
    checks["artifact_match"] = observed_artifact == profile.artifact
    checks["revision_match"] = observed_revision == profile.revision
    if not checks["artifact_match"] or not checks["revision_match"]:
        # Divergence PROUVÉE : même politique que les digests d'embeddings —
        # on ne sert pas un modèle qui n'est pas celui décrit par le registre.
        payload["message"] = "artefact ou révision divergents du registre"
        return JSONResponse(status_code=503, content={"detail": payload})

    started = time.time()
    try:
        scored = rerank_client.rerank(RERANK_PROBE_QUERY, [RERANK_PROBE_TEXT])
        checks["scoring"] = len(scored) == 1
    except UpstreamError as exc:
        checks["scoring"] = False
        logger.warning("health/rerank: sonde de scoring en échec: %s", exc.message)
        payload["message"] = PUBLIC_RERANK_PROBE_FAILED
        return JSONResponse(status_code=503, content={"detail": payload})
    payload["probe_latency_ms"] = int((time.time() - started) * 1000)

    if not checks["scoring"]:
        payload["message"] = "sonde de scoring en échec"
        return JSONResponse(status_code=503, content={"detail": payload})

    payload["status"] = "ok"
    return JSONResponse(status_code=200, content=payload)


# --------------------------------------------------------------------------- #
# ASR — capability (lot asr-rail-foundation). Les routes /v1/audio/* vivent
# dans app/asr_routes.py ; ici seulement le blocage de capability, la
# vérification d'identité backend et /health/asr (contrat §69-§72).
# --------------------------------------------------------------------------- #
def _asr_capability_blocker() -> Optional[str]:
    """Raison pour laquelle la capability ASR ne peut pas servir, ou None.

    Hors loopback = capability indisponible (le backend n'a ni auth ni
    gouvernance propres). Le flag AI_GATEWAY_ASR_ENABLED (défaut OFF) vaut
    dans tout environnement ; production_allowed est vérifié par route.
    """
    if not settings.asr_enabled:
        return "capability ASR désactivée (AI_GATEWAY_ASR_ENABLED)"
    if not asr_transport.backend_url_is_loopback(settings.asr_backend_url):
        return "backend ASR hors loopback : capability refusée"
    if registry.asr_default_model is None:
        return "aucun modèle ASR au registre"
    return None


def _asr_identity_checks(profile: Any, info: Dict[str, Any]) -> Dict[str, bool]:
    """Compare /info du backend au profil du registre — identité épinglée
    trois fois (artefact, révision, hash du fichier modèle) + VAD + provider.
    Toute divergence = capability NON servie (fail-closed, §70)."""
    observed_rev = str(info.get("revision") or "").strip().lower()
    observed_sha = str(info.get("artifact_sha256") or "").strip().lower()
    vad = info.get("vad") if isinstance(info.get("vad"), dict) else {}
    checks = {
        "provider_match": info.get("provider") == profile.provider,
        "artifact_match": info.get("artifact") == profile.artifact,
        "revision_match": observed_rev == profile.revision,
        "hash_match": observed_sha == profile.artifact_sha256,
        "vad_match": (
            vad.get("provider") == profile.vad.provider
            and vad.get("source") == profile.vad.source
            and str(vad.get("version") or "") == profile.vad.version
        ),
        "capabilities_match": isinstance(info.get("capabilities"), list)
        and set(profile.capabilities) <= set(str(c) for c in info.get("capabilities")),
    }
    if profile.diarization is not None:
        # Identité Sortformer ÉPINGLÉE (lot production-full-capability-enablement) :
        # provider, artefact, révision, hash .nemo mesuré == attendu, prêt.
        diar = info.get("diarization") if isinstance(info.get("diarization"), dict) else {}
        checks["diarization_match"] = (
            diar.get("provider") == profile.diarization.provider
            and diar.get("artifact") == profile.diarization.artifact
            and str(diar.get("revision") or "").strip().lower() == profile.diarization.revision
            and str(diar.get("artifact_sha256") or "").strip().lower() == profile.diarization.artifact_sha256
            and diar.get("model_ready") is True
        )
    return checks


def asr_backend_verified(profile: Any) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Sonde /info et renvoie (ok, checks, message_public). Utilisé par
    /health/asr ET par l'admission des routes (pas de résultat en cache :
    une divergence doit être vue à la requête suivante)."""
    checks: Dict[str, Any] = {}
    try:
        info = asr_transport.asr_client.info()
    except AsrBackendError as exc:
        checks["backend_reachable"] = False
        logger.warning("health/asr: backend injoignable: %s", exc.message)
        return False, checks, "backend ASR injoignable"
    checks["backend_reachable"] = True
    checks.update(_asr_identity_checks(profile, info))
    checks["_info"] = info
    if not all(v for k, v in checks.items() if k != "_info"):
        return False, checks, "identité du backend ASR divergente du registre"
    return True, checks, None


@app.get("/health/asr")
def health_asr() -> JSONResponse:
    """Diagnostic de la capability ASR (contrat §69-§72).

    Capability désactivée => 200 `disabled` (configuration voulue). Backend
    absent ou identité divergente => 503, et la capability n'est PAS servie.
    Ne charge JAMAIS le modèle : /health et /info backend sont sans effet
    sur le worker. Indépendant de /health/ready.
    """
    checks: Dict[str, Any] = {}
    payload: Dict[str, Any] = {
        "status": "error",
        "provider": "ai_gateway",
        "backend_url": "redacted",
        "environment": settings.environment,
        "enabled": False,
        "checks": checks,
    }
    blocker = _asr_capability_blocker()
    if blocker is not None:
        payload["status"] = "disabled"
        payload["message"] = blocker
        return JSONResponse(status_code=200, content=payload)
    payload["enabled"] = True

    profile = registry.resolve_asr(None)
    checks["registry_profile"] = profile is not None
    if profile is None:
        payload["message"] = "modèle ASR par défaut absent du registre"
        return JSONResponse(status_code=503, content={"detail": payload})
    payload["model"] = profile.logical_name
    payload["artifact"] = profile.artifact
    payload["expected_revision"] = profile.revision
    payload["allowed_here"] = profile.allowed_in(settings.environment)
    payload["capabilities"] = list(profile.capabilities)

    ok, id_checks, message = asr_backend_verified(profile)
    info = id_checks.pop("_info", {}) or {}
    checks.update(id_checks)
    payload["backend_reachable"] = bool(checks.get("backend_reachable"))
    if not checks.get("backend_reachable"):
        payload["message"] = message
        return JSONResponse(status_code=503, content={"detail": payload})
    payload["version"] = {
        "backend": str(info.get("backend_version") or "unknown"),
        "faster_whisper": str(info.get("faster_whisper_version") or "unknown"),
        "ctranslate2": str(info.get("ctranslate2_version") or "unknown"),
    }
    payload["observed_revision"] = str(info.get("revision") or "") or None
    payload["artifact_match"] = checks.get("artifact_match", False)
    payload["revision_match"] = checks.get("revision_match", False)
    payload["hash_match"] = checks.get("hash_match", False)
    if profile.diarization is not None:
        diar_info = info.get("diarization") if isinstance(info.get("diarization"), dict) else {}
        payload["diarization"] = {
            "artifact": profile.diarization.artifact,
            "expected_revision": profile.diarization.revision,
            "expected_sha256": profile.diarization.artifact_sha256,
            "observed_sha256": diar_info.get("artifact_sha256"),
            "model_ready": diar_info.get("model_ready"),
            "device": diar_info.get("device"),
            "hash_match": bool(checks.get("diarization_match", False)),
        }
    if not ok:
        payload["message"] = message
        return JSONResponse(status_code=503, content={"detail": payload})

    try:
        health = asr_transport.asr_client.health()
    except AsrBackendError as exc:
        checks["backend_health"] = False
        logger.warning("health/asr: /health backend en échec: %s", exc.message)
        payload["message"] = "backend ASR injoignable"
        return JSONResponse(status_code=503, content={"detail": payload})
    checks["backend_health"] = health.get("_http_status") == 200
    payload["worker_state"] = str(health.get("worker_state") or "unknown")
    payload["queued"] = health.get("queued")
    payload["inflight"] = health.get("inflight")
    spool = health.get("spool") if isinstance(health.get("spool"), dict) else {}
    payload["spool"] = {
        "bytes": spool.get("bytes"),
        "capacity_bytes": spool.get("capacity_bytes"),
        "jobs": spool.get("jobs"),
    }
    checks["model_ready"] = bool(health.get("model_ready"))
    if not checks["backend_health"] or not checks["model_ready"]:
        payload["message"] = "backend ASR dégradé"
        return JSONResponse(status_code=503, content={"detail": payload})
    payload["status"] = "ok"
    return JSONResponse(status_code=200, content=payload)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _prepare_texts(
    texts: List[str], profile: Optional[ModelProfile], task: str
) -> List[str]:
    """Applique le préfixe du profil et refuse les entrées trop longues.

    Garde contre le double préfixage : si le texte commence DÉJÀ exactement par
    le préfixe attendu, on ne le pose pas une seconde fois. On ne retire jamais
    un préfixe fourni par le client et on ne tente aucune reconnaissance
    approximative — seule l'égalité exacte est traitée.
    """
    prefix = profile.prefix_for(task) if profile is not None else ""
    limit = profile.max_input_chars if profile is not None else None
    on_overflow = profile.on_overflow if profile is not None else "reject"

    prepared: List[str] = []
    for index, text in enumerate(texts):
        if prefix and text.startswith(prefix):
            counters.inc("embeddings_double_prefix_avoided_total")
            log_event(
                logger,
                logging.WARNING,
                "embeddings.double_prefix_avoided",
                index=index,
                task=task,
                model_logical=profile.logical_name if profile else "-",
            )
            candidate = text
        else:
            candidate = prefix + text

        if limit is not None and len(candidate) > limit:
            counters.inc("embeddings_overflow_total")
            if on_overflow == "reject":
                count_error(ERROR_CLIENT)
                # Ni le texte ni un extrait : seulement l'index et les tailles.
                raise http_error(
                    400,
                    f"texte {index} trop long ({len(candidate)} caractères > "
                    f"{limit}) pour le modèle {profile.logical_name}",
                    provider="ai_gateway",
                    extra={
                        "index": index,
                        "length": len(candidate),
                        "max_input_chars": limit,
                        "on_overflow": on_overflow,
                    },
                )
            # truncate_and_flag : implémenté pour ne pas laisser le registre
            # décrire une politique inapplicable, mais AUCUN modèle ne l'active
            # dans ce lot (cf. app/models_registry.yaml).
            candidate = candidate[:limit]

        prepared.append(candidate)
    return prepared


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
def embeddings(req: EmbeddingsRequest) -> EmbeddingsResponse:
    started = time.time()
    counters.inc("embeddings_requests_total")

    requested_model = req.model or settings.embedding_model
    task = req.task or DEFAULT_TASK

    _ensure_registry_verified()
    profile = registry.resolve(requested_model)

    # 1) Autorisation d'environnement — évaluée dès que le modèle est DÉCRIT par
    #    le registre, y compris s'il est déclassé. Un `production_allowed: false`
    #    explicite est une décision de gouvernance : elle ne doit pas pouvoir
    #    être contournée en faisant diverger le digest pour retomber en mode
    #    legacy, ni en désactivant le mode strict.
    if profile is not None and not profile.allowed_in(settings.environment):
        counters.inc("embeddings_registry_rejected_total")
        count_error(ERROR_CLIENT)
        raise http_error(
            400,
            f"modèle {profile.logical_name} non autorisé dans l'environnement "
            f"{settings.environment}",
            provider="ai_gateway",
            extra={
                "model_logical": profile.logical_name,
                "environment": settings.environment,
            },
        )

    # 2) Un profil au digest divergent cesse d'être gouverné : on ne va pas
    #    estampiller un pipeline_id sur des vecteurs produits par un artefact
    #    dont on vient de prouver qu'il n'est plus celui décrit par le registre.
    governed = profile is not None and profile.governed
    if profile is not None and not governed:
        counters.inc("embeddings_degraded_model_total")
        log_event(
            logger,
            logging.ERROR,
            "embeddings.model_degraded",
            model_logical=profile.logical_name,
            reason="digest_mismatch",
        )

    if governed and not profile.identity_verified:
        # Lot P2 : servi (fail-open, doctrine inchangée) mais SANS preuve
        # d'identité courante — rendu visible plutôt que masqué.
        counters.inc("embeddings_identity_unverified_total")
        log_event(
            logger,
            logging.WARNING,
            "embeddings.identity_unverified",
            model_logical=profile.logical_name,
            registry_status=profile.status,
            expected_digest=profile.short_digest,
            verified_at=profile.verified_at_iso or "-",
        )

    if not governed:
        if settings.embeddings_strict_registry:
            counters.inc("embeddings_registry_rejected_total")
            count_error(ERROR_CLIENT)
            raise http_error(
                400,
                f"modèle non gouverné : {requested_model}",
                provider="ai_gateway",
                extra={"strict_registry": True},
            )
        counters.inc("embeddings_legacy_model_total")
        log_event(
            logger,
            logging.WARNING,
            "embeddings.legacy_model",
            model=requested_model,
            reason="absent_du_registre" if profile is None else "degraded",
        )

    # Contrainte d'index Compact : un modèle gouverné d'une autre
    # dimension que celle de l'index configuré est REFUSÉ avant tout appel —
    # mélanger deux dimensions dans un index est une corruption silencieuse.
    if governed and profile.dimension != settings.embeddings_dimension:
        counters.inc("embeddings_dimension_incompatible_total")
        count_error(ERROR_CLIENT)
        raise http_error(
            400,
            f"modèle {profile.logical_name} incompatible : {profile.dimension} dimensions, "
            f"l'index attend {settings.embeddings_dimension}",
            provider="ai_gateway",
            extra={
                "code": "embedding_dimension_incompatible",
                "model_logical": profile.logical_name,
                "model_dimension": profile.dimension,
                "expected_dimension": settings.embeddings_dimension,
            },
        )

    # Nom réellement envoyé à Ollama. Un client qui a demandé un alias Ollama
    # (`nomic-embed-text:latest`) le retrouve inchangé dans la réponse ; un
    # client qui a demandé un nom logique reçoit le nom Ollama canonique.
    if governed:
        upstream_model = (
            requested_model
            if profile.matches_ollama_name(requested_model)
            else profile.ollama_name
        )
    else:
        upstream_model = requested_model

    # Limite de batch : le plus contraignant entre l'environnement et le profil.
    max_batch = settings.embeddings_max_batch
    if governed:
        max_batch = min(max_batch, profile.max_batch_size)
    if len(req.texts) > max_batch:
        count_error(ERROR_CLIENT)
        raise http_error(
            400,
            f"batch trop grand ({len(req.texts)} > {max_batch})",
            provider="ai_gateway",
        )

    prepared = _prepare_texts(req.texts, profile if governed else None, task)

    logger.info("embeddings model=%s n_texts=%d", upstream_model, len(req.texts))
    upstream_started = time.time()
    # Jauge inflight (lot workload-isolation) : un batch d'indexation occupe
    # le runner Ollama partagé pendant toute sa durée — c'est LA contention
    # mesurée entre environnements. La rendre visible sur /health/runtime
    # permet à un consommateur de coordonner ses campagnes.
    inflight_key = f"embeddings:{profile.logical_name if profile is not None else upstream_model}"
    try:
        host_lease = host_admission.acquire(
            "embedding", profile.logical_name if profile is not None else upstream_model)
    except HostAdmissionError as exc:
        raise host_admission_http_error(exc)
    if not _embeddings_semaphore.acquire(blocking=False):
        host_admission.release(host_lease)
        counters.inc("embeddings_saturated_total")
        raise _saturated_error("embeddings_saturated", "embeddings saturés : concurrence maximale atteinte",
                               EMBEDDINGS_RETRY_AFTER_SECONDS)
    host_admission.mark_started(host_lease)   # lot ORPHAN-LEASE : backend engagé à partir d'ici
    gauges.inc(inflight_key)
    try:
        vectors = client.embed(
            model=upstream_model, inputs=prepared, keep_alive=req.keep_alive
        )
    except UpstreamModelNotFound:
        count_error(ERROR_UPSTREAM)
        raise _model_not_installed_error(upstream_model, "embeddings")
    except UpstreamError as exc:
        count_error(ERROR_UPSTREAM)
        logger.warning("embeddings: échec upstream: %s", exc.message)
        if "budget de temps global" in exc.message:
            counters.inc("embeddings_fallback_budget_exhausted_total")
            raise http_error(502, PUBLIC_EMBEDDING_BUDGET_EXHAUSTED, provider="ollama")
        raise http_error(502, PUBLIC_EMBEDDING_FAILED, provider="ollama")
    finally:
        gauges.dec(inflight_key)
        _embeddings_semaphore.release()
        host_admission.release(host_lease)
    upstream_ms = int((time.time() - upstream_started) * 1000)

    endpoint = getattr(vectors, "upstream_endpoint", ENDPOINT_EMBED)
    if endpoint == ENDPOINT_EMBEDDINGS:
        counters.inc("embeddings_fallback_total")

    # 1) Cardinalité — message historique conservé.
    if len(vectors) != len(req.texts):
        count_error(ERROR_UPSTREAM)
        raise http_error(
            502,
            f"nombre de vecteurs ({len(vectors)}) != nombre de textes ({len(req.texts)})",
            provider="ollama",
        )

    # 2) Dimension — désormais contrôlée pour TOUS les modèles gouvernés, plus
    #    seulement pour le modèle par défaut.
    first = vectors[0] if vectors else None
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        counters.inc("embeddings_invalid_vector_total")
        count_error(ERROR_UPSTREAM)
        raise http_error(502, "vecteur 0 vide", provider="ollama")
    dimension = len(first)

    expected_dimension = profile.dimension if governed else None
    if expected_dimension is not None and dimension != expected_dimension:
        counters.inc("embeddings_dimension_errors_total")
        count_error(ERROR_UPSTREAM)
        raise http_error(
            502,
            f"dimension inattendue ({dimension} != {expected_dimension})",
            provider="ollama",
        )

    # 3) Homogénéité, valeurs finies, vecteurs non nuls.
    try:
        validate_vectors(
            vectors,
            expected_count=len(req.texts),
            expected_dimension=expected_dimension,
            require_normalized=False,
        )
    except VectorError as exc:
        counters.inc("embeddings_invalid_vector_total")
        count_error(ERROR_UPSTREAM)
        raise http_error(502, str(exc), provider="ollama")

    # 4) Normalisation défensive du repli historique.
    #    /api/embeddings renvoie des vecteurs NON normalisés (‖v‖ ≈ 19 à 26)
    #    là où /api/embed renvoie ‖v‖ = 1. Sans cette étape, un basculement en
    #    repli injecterait deux échelles dans le même index vectoriel.
    out_vectors: List[List[float]] = [list(v) for v in vectors]
    normalized = False
    if governed and profile.normalized:
        if endpoint == ENDPOINT_EMBEDDINGS:
            try:
                out_vectors = [l2_normalize(v) for v in out_vectors]
            except VectorError as exc:
                counters.inc("embeddings_invalid_vector_total")
                count_error(ERROR_UPSTREAM)
                raise http_error(502, str(exc), provider="ollama")
            log_event(
                logger,
                logging.WARNING,
                "embeddings.fallback_normalized",
                model_logical=profile.logical_name,
                n_texts=len(out_vectors),
            )
        # Contrôle final, quel que soit le chemin emprunté.
        for index, vector in enumerate(out_vectors):
            norm = l2_norm(vector)
            if not norm_in_tolerance(norm):
                counters.inc("embeddings_invalid_vector_total")
                count_error(ERROR_UPSTREAM)
                raise http_error(
                    502,
                    f"vecteur {index} de norme {norm:.6f} hors tolérance "
                    f"[{NORM_MIN}, {NORM_MAX}] pour un modèle déclaré normalisé",
                    provider="ollama",
                )
        normalized = True
    elif endpoint == ENDPOINT_EMBEDDINGS:
        # Modèle non gouverné : la normalisation attendue est inconnue, on ne
        # touche donc à rien — mais on le dit.
        log_event(
            logger,
            logging.WARNING,
            "embeddings.fallback_not_normalized",
            model=upstream_model,
            reason="profil_absent",
        )

    counters.inc("embeddings_texts_total", len(req.texts))
    # Métriques de performance (lot 3), toutes additives :
    #  · `normalize_embedding_metrics` — durées mesurées par OLLAMA, absentes sur
    #    le chemin de repli `/api/embeddings` qui n'en publie aucune ;
    #  · `embedding_throughput` — débits dérivés de la mesure MURALE Gateway,
    #    donc toujours disponibles, quel que soit l'endpoint emprunté.
    # `upstream_endpoint`, déjà présent, reste la clé qui permet de distinguer
    # les deux régimes lors de l'analyse.
    log_event(
        logger,
        logging.INFO,
        "embeddings.ok",
        model=upstream_model,
        model_logical=profile.logical_name if governed else "-",
        governed=governed,
        task=task,
        n_texts=len(req.texts),
        dimension=dimension,
        upstream_endpoint=endpoint,
        upstream_ms=upstream_ms,
        total_ms=int((time.time() - started) * 1000),
        **normalize_embedding_metrics(getattr(vectors, "ollama_metadata", None)),
        **embedding_throughput(len(req.texts), upstream_ms),
    )

    return EmbeddingsResponse(
        model=upstream_model,
        dimension=dimension,
        embeddings=out_vectors,
        usage=EmbeddingsUsage(total_texts=len(req.texts)),
        cost=0.0,
        governed=governed,
        model_logical=profile.logical_name if governed else None,
        # Lot P2 : déclaré et observé sont deux champs. `model_digest` ne porte
        # que ce qui a été VU (null sinon, divergent si divergent) ; l'attendu
        # est dans `expected_digest`. Exposés dès que le registre DÉCRIT le
        # modèle, gouverné ou déclassé — jamais inventés hors registre.
        expected_digest=profile.digest if profile is not None else None,
        model_digest=profile.observed_digest if profile is not None else None,
        identity_verified=profile.identity_verified if profile is not None else False,
        verified_at=profile.verified_at_iso if profile is not None else None,
        # `pipeline_id` décrit la transformation DÉCLARÉE (P1) ; il ne dépend
        # pas de l'état runtime — `identity_verified` dit si le modèle observé
        # correspond à cette déclaration.
        pipeline_id=profile.pipeline_id(task) if governed else None,
        task=task,
        normalized=normalized,
        # `on_overflow: reject` garantit qu'aucune réponse 200 ne contient un
        # texte tronqué : le champ existe pour que les clients puissent déjà
        # s'y appuyer si une politique de troncature était un jour activée.
        truncated=[False] * len(req.texts),
        upstream_endpoint=endpoint,
    )


# --------------------------------------------------------------------------- #
# Vision (lot vision-1a) — contrat AI-GATEWAY-VISION-CONTRACT-0, gelé.
#
# Pipeline : validation stricte -> flag -> gouvernance -> admission image ->
# sémaphore borné -> backend (mock en 1A) -> parse JSON -> schéma ->
# invariants -> réponse normalisée. Budget mural UNIQUE pour le tout.
# Aucun contenu (image, contexte, payload) dans les logs ni les erreurs.
# --------------------------------------------------------------------------- #
# Admission vision (lot CV-CAPACITY) : UN objet possède le sémaphore et le
# compteur en vol, lisible sans effet de bord par GET /v1/vision/capacity.
# Même source que `settings.vision_max_concurrency` : aucune divergence possible
# entre la capacité annoncée et celle appliquée. Portée : ce processus (= cette
# instance, un seul worker uvicorn) — jamais l'hôte.
vision_admission = VisionAdmission(settings.vision_max_concurrency)


def deadline_for(task: str) -> int:
    """Budget mural total (s) de la task (lot SEMX-4C). L'identité de la TASK
    décide, jamais la seule schema_version : semantic_extract ->
    AI_GATEWAY_VISION_SEMANTIC_TIMEOUT_SECONDS (150, lot SEMX-R2I) ; toute autre task (describe
    et rail V1/V2) -> AI_GATEWAY_VISION_TIMEOUT_SECONDS (60), inchangé."""
    return settings.vision_timeout_seconds


def _vision_error(status: int, code: str, message: str, retryable: bool,
                  request_id: Optional[str], provider: str = "ai_gateway"):
    return http_error(status, message, provider=provider, extra={
        "code": code, "retryable": retryable, "request_id": request_id,
    })


def _vision_digest_observed(profile) -> Optional[str]:
    """Digest réellement installé, via /api/tags (JAMAIS de chargement)."""
    for entry in probe_client.list_models(timeout=5):
        if profile.matches_ollama_name(entry.get("name", "")):
            return entry.get("digest")
    return None


def _structural_failure_after_deadline(failure, profile, request_id: Optional[str]):
    """Erreur rendue quand le délai total s'épuise PENDANT le retry interne déclenché par un échec structurel de la
    1re tentative (lot AI-GATEWAY-VISUAL-22A). Mêmes codes, statut (502) et `retryable: false` que l'échec structurel
    après nouvelle tentative (contrat existant, doc/04-api.md) ; seule la cause loggée diffère."""
    counters.inc("vision_invalid_output_total")
    log_event(logger, logging.WARNING, "vision.invalid_output",
              code=failure[0], cause="deadline_exhausted_during_retry",
              model_logical=profile.logical_name, request_id=request_id or "-")
    return _vision_error(502, failure[0],
                         f"{failure[1]} (nouvelle tentative interrompue par le délai total)",
                         retryable=False, request_id=request_id, provider="ollama")


def _resolve_vision_profile(requested: Optional[str], request_id: Optional[str]):
    name = requested or registry.vision_default_model
    if name is None:
        raise _vision_error(503, "vision_model_unavailable",
                            "aucun modèle vision par défaut configuré au registre",
                            retryable=False, request_id=request_id)
    if requested is not None:
        candidate = registry.llm_models.get(requested)
        if candidate is None or not candidate.supports_image:
            # Profil inconnu OU texte-seul : refus — jamais de repli texte.
            counters.inc("vision_client_errors_total")
            raise _vision_error(400, "vision_model_not_allowed",
                                f"modèle vision inconnu ou sans modalité image : {requested}",
                                retryable=False, request_id=request_id)
        profile = candidate
    else:
        profile = registry.resolve_vision(name)
        if profile is None:
            raise _vision_error(503, "vision_model_unavailable",
                                f"modèle vision par défaut invalide : {name}",
                                retryable=False, request_id=request_id)
    if not profile.allowed_in(settings.environment):
        counters.inc("vision_client_errors_total")
        raise _vision_error(400, "vision_model_not_allowed",
                            f"modèle {profile.logical_name} non autorisé dans "
                            f"l'environnement {settings.environment}",
                            retryable=False, request_id=request_id)
    return profile


# Documentation OpenAPI du 200 (lot SEMX-4A) : union V1/V2/V3 discriminée par
# `schema_version`. Purement descriptive : la route rend un JSONResponse déjà
# construit par la classe fermée de chaque version, rien n'est re-sérialisé.
VisionAnalyzeResponseAny = Annotated[
    Union[VisionAnalyzeResponse, VisionAnalyzeResponseV2],
    Field(discriminator="schema_version"),
]


@app.post("/v1/vision/analyze",
          responses={200: {"model": VisionAnalyzeResponseAny,
                           "description": "Résultat V1 ou V2 selon schema_version"}})
def vision_analyze(req: VisionAnalyzeRequest) -> JSONResponse:
    started = time.time()
    counters.inc("vision_requests_total")
    request_id = req.request_id

    def client_reject(status: int, code: str, message: str):
        counters.inc("vision_client_errors_total")
        if code == "image_too_large":
            counters.inc("vision_rejected_oversize_total")
        log_event(logger, logging.WARNING, "vision.rejected",
                  code=code, task=req.task, request_id=request_id or "-")
        return _vision_error(status, code, message, retryable=False,
                             request_id=request_id)

    # Appariement task <-> schema_version (lot SEMX-4A) : PREMIÈRE décision,
    # avant tout flag, toute gouvernance, tout décodage d'image et toute
    # admission. "3" n'existe que pour semantic_extract, et réciproquement.
    is_semantic = req.task == SEMANTIC_TASK
    if is_semantic != (req.schema_version == SEMANTIC_SCHEMA_VERSION):
        raise client_reject(400, "vision_schema_version_unsupported",
                            f"schema_version {req.schema_version} incompatible avec "
                            f"la task {req.task}")

    if not settings.vision_enabled:
        counters.inc("vision_client_errors_total")
        raise _vision_error(503, "vision_not_enabled",
                            "capability vision désactivée (AI_GATEWAY_VISION_ENABLED)",
                            retryable=False, request_id=request_id)
    # Version Open : la task `semantic_extract` (contrat v3) repose sur le
    # module `vision_semantic_*`, NON inclus dans cette distribution. Elle est
    # annoncée indisponible — jamais servie, jamais simulée — avant la
    # résolution du profil. N'affecte jamais `describe`.
    if is_semantic:
        counters.inc("vision_client_errors_total")
        log_event(logger, logging.WARNING, "vision.rejected",
                  code="vision_task_unavailable", task=req.task, request_id=request_id or "-")
        raise _vision_error(503, "vision_task_unavailable",
                            "task semantic_extract non incluse dans cette distribution",
                            retryable=False, request_id=request_id)

    # Gouvernance : profil logique, environnement, modalité, task qualifiée.
    profile = _resolve_vision_profile(req.model, request_id)
    if req.task not in profile.vision_tasks:
        raise client_reject(400, "vision_task_not_qualified",
                            f"task {req.task} non qualifiée pour "
                            f"{profile.logical_name} (MODEL-1)")
    # Double verrou : le registre autorise la task ET un profil de prompt
    # QUALIFIÉ existe pour elle. Un registre incohérent (task autorisée sans
    # prompt mesuré) échoue ici plutôt que d'inventer un prompt.
    qualified = QUALIFIED_PROMPTS.get(req.task)
    if qualified is None:
        raise client_reject(400, "vision_task_not_qualified",
                            f"task {req.task} sans profil de prompt qualifié")
    # Version du contrat PUBLIC demandée par le client ("1" ou "2", lot CV-1) :
    # le profil doit la déclarer. Le schéma MODEL-FACING est le même pour les
    # deux — seule la forme de la réponse change.
    if req.schema_version not in profile.vision_schema_versions:
        raise client_reject(400, "vision_schema_version_unsupported",
                            f"schema_version {req.schema_version} non supportée par "
                            f"{profile.logical_name}")
    if profile.digest is None:
        raise _vision_error(503, "vision_model_unavailable",
                            "profil vision sans digest gouverné",
                            retryable=False, request_id=request_id)
    try:
        observed = _vision_digest_observed(profile)
    except UpstreamError as exc:
        counters.inc("vision_upstream_errors_total")
        logger.warning("vision: échec upstream: %s", exc.message)
        raise _vision_error(502, "vision_upstream_error", PUBLIC_VISION_FAILED,
                            retryable=True, request_id=request_id, provider="ollama")
    if observed != profile.digest:
        raise _vision_error(503, "vision_model_unavailable",
                            f"modèle {profile.logical_name} absent ou digest divergent",
                            retryable=False, request_id=request_id)

    # Bornes de contexte (provisoires — MODEL-1 TO QUALIFY).
    context = req.context or VisionContext()
    fields = [context.caption, context.alt_text, context.nearby_text]
    if any(f is not None and len(f) > settings.vision_max_context_field_chars
           for f in fields) or \
            sum(len(f) for f in fields if f) > settings.vision_max_context_total_chars:
        # Dépassement d'ENTRÉE = erreur de validation (contrat § AI : 400,
        # enveloppe existante). Bornes QUALIFIÉES : 1000 car. par champ,
        # 2000 au total. On REFUSE plutôt que de tronquer : une troncature
        # silencieuse changerait le contexte d'interprétation sans le dire.
        counters.inc("vision_client_errors_total")
        raise http_error(400, "contexte trop long", provider="ai_gateway")

    # Admission image : le décodage local fait foi. Lot P7 : c'est le SEUL
    # décodage de la requête (le schéma a vérifié longueur et syntaxe sans
    # décoder) ; une chaîne syntaxiquement plausible mais indécodable est une
    # image non décodable, pas une erreur serveur.
    try:
        raw_bytes = base64.b64decode(req.image.data, validate=True)
    except (binascii.Error, ValueError):
        raise client_reject(400, "image_decode_failed", "base64 invalide")
    try:
        admitted = admit_image(raw_bytes, req.image.mime_type)
    except ImageAdmissionError as exc:
        raise client_reject(exc.http_status, exc.code, exc.message)

    deadline = started + max(1e-3, float(deadline_for(req.task)))

    # Concurrence bornée — acquisition NON BLOQUANTE (un lot antérieur, même raison que
    # rerank : l'attente se faisait dans un thread du pool partagé, jusqu'à
    # 60 s ici). À capacité pleine : 503 immédiat + Retry-After, code
    # `vision_overloaded` conservé (contrat existant). Plus de jauge `queued:`.
    # Admission HÔTE (lot host-wide-admission) : UN slot vision pour tout
    # l'hôte (prod + staging), demandé après l'admission image et avant la
    # garde locale `VisionAdmission`, conservée comme dernière défense.
    try:
        # Lot SEMX-4C : même classe physique « vision », même modèle logique ;
        # seul semantic_extract DEMANDE un délai explicite (TTL = délai + 10).
        # describe n'envoie rien : appel et corps de réservation inchangés.
        # Lot GPU-24A : attente BORNÉE (AI_GATEWAY_VISION_HOST_ADMISSION_WAIT_SECONDS,
        # défaut 0 = appel historique) sur un DEFERRED transitoire — worker StructX
        # encore résident (<= son TTL), VRAM en cours de libération — dans le délai
        # total de la route ; un seul attenteur vision par processus.
        host_lease, host_waited_ms, _ = host_admission.acquire_bounded(
            "vision", profile.logical_name, request_id,
            timeout_seconds=None,
            wait_seconds=settings.vision_host_admission_wait_seconds, deadline=deadline)
    except HostAdmissionError as exc:
        raise host_admission_http_error(exc, request_id)
    if not vision_admission.try_acquire():
        host_admission.release(host_lease)
        counters.inc("vision_overloaded_total")
        log_event(logger, logging.WARNING, "vision.overloaded",
                  model_logical=profile.logical_name, request_id=request_id or "-")
        error = _vision_error(503, "vision_overloaded",
                              "concurrence vision maximale atteinte",
                              retryable=True, request_id=request_id)
        error.detail["retry_after_seconds"] = VISION_RETRY_AFTER_SECONDS
        error.headers = {"Retry-After": str(VISION_RETRY_AFTER_SECONDS)}
        raise error
    host_admission.mark_started(host_lease)   # lot ORPHAN-LEASE : backend engagé à partir d'ici

    prompt_profile, prompt_version = PROMPT_PROFILES[req.task]
    plan_base = dict(
        image_bytes=admitted.normalized_png,
        image_mime=admitted.normalized_mime,
        task=req.task,
        prompt_profile=prompt_profile,
        prompt_profile_version=prompt_version,
        output_json_schema=output_schema_for_task(req.task),
        locale=req.locale,
        caption=context.caption,
        alt_text=context.alt_text,
        nearby_text=context.nearby_text,
        model_logical=profile.logical_name,
        model_concrete=profile.ollama_name,
        system_prompt=SYSTEM_PROMPT,
        task_prompt=qualified[1],
    )

    gauges.inc(f"vision:{profile.logical_name}")
    try:
        payload_model = None
        raw = None
        attempts_used = 0
        normalization = None
        # Lot AI-GATEWAY-VISUAL-22A : échec STRUCTUREL de la 1re tentative (troncature, JSON, schéma). Si le délai
        # total s'épuise pendant le retry interne qu'il a déclenché, le client reçoit CET échec (502, non
        # réessayable, codes existants) et non un 504 `vision_timeout` réessayable : la 1re tentative est
        # déterministe (temperature 0, seed fixe), un nouvel appel client reproduirait la même troncature.
        # Mesuré en staging : 46/47 des 504 étaient de ce type. Politique de retry (fingerprint) INCHANGÉE.
        prior_failure = None
        # Retry INTERNE unique, réservé aux sorties structurées invalides
        # (parse/schéma/invariants/troncature). Jamais pour un échec transport.
        # (Politique nommée RETRY_MAX_ATTEMPTS / RETRY_TRIGGER dans
        # vision_backend : elle entre dans le fingerprint du pipeline.)
        for attempt in (1, 2):
            attempts_used = attempt
            remaining = deadline - time.time()
            if remaining <= 0 and prior_failure is not None:
                raise _structural_failure_after_deadline(prior_failure, profile, request_id)
            if remaining <= 0:
                counters.inc("vision_timeouts_total")
                log_event(logger, logging.WARNING, "vision.timeout",
                          model_logical=profile.logical_name,
                          request_id=request_id or "-")
                raise _vision_error(504, "vision_timeout",
                                    "budget total vision dépassé",
                                    retryable=True, request_id=request_id,
                                    provider="ollama")
            # Options du profil qualifié : 1re tentative déterministe, retry
            # à température variée (un retry seed-only ne changerait rien à
            # temperature=0 — mesuré par MODEL-1).
            options = QUALIFIED_FIRST_ATTEMPT if attempt == 1 else QUALIFIED_RETRY_ATTEMPT
            # Résidence courte (lot vision-residency-02) : SEUL `keep_alive`
            # varie. Le profil qualifié lui-même n'est pas modifié — on en
            # dérive une copie, si bien qu'aucune autre option d'inférence
            # (température, seed, num_ctx, num_predict) ne peut dériver.
            if settings.vision_keep_alive_seconds:
                options = replace(options,
                                  keep_alive=settings.vision_keep_alive_seconds)
            plan = VisionInferencePlan(remaining_timeout_seconds=remaining,
                                       options=options, **plan_base)
            try:
                result = vision_backend.analyze(plan)
            except VisionBackendTimeout:
                if prior_failure is not None:
                    raise _structural_failure_after_deadline(prior_failure, profile, request_id)
                counters.inc("vision_timeouts_total")
                log_event(logger, logging.WARNING, "vision.timeout",
                          model_logical=profile.logical_name,
                          request_id=request_id or "-")
                raise _vision_error(504, "vision_timeout",
                                    "budget total vision dépassé",
                                    retryable=True, request_id=request_id,
                                    provider="ollama")
            except UpstreamError as exc:
                counters.inc("vision_upstream_errors_total")
                log_event(logger, logging.WARNING, "vision.upstream_error",
                          model_logical=profile.logical_name,
                          request_id=request_id or "-")
                logger.warning("vision: échec upstream: %s", exc.message)
                raise _vision_error(502, "vision_upstream_error", PUBLIC_VISION_FAILED,
                                    retryable=True, request_id=request_id,
                                    provider="ollama")
            failure = None
            if result.truncated:
                failure = ("vision_truncated_output",
                           "sortie modèle tronquée : résultat rejeté")
            else:
                try:
                    parsed = json.loads(result.raw_json)
                    raw = RawModelOutput.model_validate(parsed)
                    # Lot ABSTENTION-NORMALIZATION-1A : abstention describe
                    # sans warning complétée AVANT les invariants, qui
                    # s'appliquent ensuite inchangés. Raison retenue
                    # seulement si la validation réussit.
                    raw, normalization = normalize_unambiguous_unknown(req.task, raw)
                    payload_model = validate_semantic_output(req.task, raw)
                except (json.JSONDecodeError, TypeError):
                    failure = ("vision_invalid_output",
                               "sortie modèle non parsable en JSON")
                except (PydanticValidationError, VisionInvariantError) as exc:
                    detail = exc.args[0] if isinstance(exc, VisionInvariantError) else "schéma non conforme"
                    failure = ("vision_schema_validation_failed",
                               f"sortie modèle invalide : {detail}")
            if failure is None:
                break
            prior_failure = failure
            if attempt == 2:
                counters.inc("vision_invalid_output_total")
                log_event(logger, logging.WARNING, "vision.invalid_output",
                          code=failure[0], model_logical=profile.logical_name,
                          request_id=request_id or "-")
                raise _vision_error(502, failure[0],
                                    f"{failure[1]} (après nouvelle tentative)",
                                    retryable=False, request_id=request_id,
                                    provider="ollama")
    finally:
        gauges.dec(f"vision:{profile.logical_name}")
        vision_admission.release()
        host_admission.release(host_lease)

    # Warnings top-level : dédoublonnés et triés (contrat existant, inchangé) ;
    # lot CV-0 : les warnings des régions `mixed` sont ensuite ajoutés dans
    # l'ordre du document, sans doublon — ils étaient perdus auparavant. Sans
    # région (describe), la liste est strictement celle d'avant.
    warnings_list = collect_warnings(sorted(set(raw.warnings)), payload_model)
    rvc = derive_requires_visual_check(raw.visual_kind, payload_model, warnings_list)
    digest8 = profile.digest[:8]
    # Champs communs V1/V2. `pipeline_id` reste l'identifiant LISIBLE
    # (digest 8) ; l'identité cryptographique V2 est `pipeline.fingerprint`.
    pipeline_fields = dict(
        prompt_profile=prompt_profile,
        prompt_profile_version=prompt_version,
        pipeline_id=f"{profile.logical_name}@{digest8}/{prompt_profile}@v{prompt_version}",
    )
    image_fields = dict(width=admitted.width, height=admitted.height,
                        pixels=admitted.pixels, bytes=admitted.bytes,
                        format=admitted.format)
    usage_fields = dict(prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.total_tokens)
    common = dict(
        task=req.task,
        visual_kind=raw.visual_kind,
        request_id=request_id,
        summary=raw.summary,
        warnings=warnings_list,
        requires_visual_check=rvc,
        payload=payload_model.model_dump() if payload_model is not None else None,
        model=VisionModelIdentity(logical=profile.logical_name,
                                  concrete=profile.ollama_name,
                                  digest=profile.digest),
        latency_ms=int((time.time() - started) * 1000),
    )
    if req.schema_version == "2":
        # Lot CV-1 : provenance TECHNIQUE (WHAT : hashes image ; HOW :
        # fingerprint + composants). Aucune provenance documentaire.
        response = VisionAnalyzeResponseV2(
            pipeline=VisionPipelineIdentityV2(
                **pipeline_fields,
                **pipeline_provenance(req.task, profile.digest, qualified[1])),
            image=VisionImageDiagnosticsV2(
                **image_fields,
                source_sha256=admitted.source_sha256,
                normalized_sha256=admitted.normalized_sha256,
                normalized_format=NORMALIZED_FORMAT,
                normalizer_id=NORMALIZER_ID),
            usage=VisionUsageV2(**usage_fields, attempts=attempts_used),
            **common,
        )
    else:
        # V1 : enveloppe FERMÉE strictement inchangée (contrat gelé).
        response = VisionAnalyzeResponse(
            pipeline=VisionPipelineIdentity(**pipeline_fields),
            image=VisionImageDiagnostics(**image_fields),
            usage=VisionUsage(**usage_fields),
            **common,
        )
    if normalization is not None:
        # Une fois par RÉPONSE 200 servie après normalisation (la tentative
        # qui réussit est la seule dont la raison est conservée).
        counters.inc("vision_abstention_normalized_total")
        log_event(logger, logging.INFO, "vision.abstention_normalized",
                  task=req.task, normalization_reason=normalization,
                  attempt=attempts_used, model_logical=profile.logical_name,
                  gateway_build=settings.build_sha or "not_reported",
                  request_id=request_id or "-")
    counters.inc("vision_ok_total")
    log_event(logger, logging.INFO, "vision.ok",
              task=req.task, visual_kind=raw.visual_kind,
              model_logical=profile.logical_name,
              image_bytes=admitted.bytes, width=admitted.width,
              height=admitted.height, pixels=admitted.pixels,
              warnings=len(warnings_list),
              output_bytes=len(result.raw_json),
              latency_ms=response.latency_ms,
              host_wait_ms=host_waited_ms,            # lot GPU-24A : attente d'admission hôte (0 sans attente)
              request_id=request_id or "-")
    return JSONResponse(status_code=200, content=response.model_dump())


@app.get("/v1/vision/capacity", response_model=VisionCapacityResponse)
def vision_capacity() -> JSONResponse:
    """Sonde de capacité ADVISORY (lot CV-CAPACITY) : « est-il raisonnable
    d'envoyer une requête vision à CETTE instance maintenant ? ».

    Lecture seule : n'acquiert aucun slot, n'appelle jamais /api/generate, ne
    charge aucun modèle, ne touche ni compteurs ni jauges. Toujours 200 —
    `saturated` n'est pas une erreur de la sonde. Le POST /v1/vision/analyze
    reste la seule autorité d'admission : un 503 `vision_overloaded` après une
    sonde « disponible » est un comportement valide (course inévitable).
    `retry_after_seconds` est la constante du 503, pas une estimation.
    """
    snap = vision_admission.snapshot()
    if not settings.vision_enabled:
        status = "disabled"
    elif snap.available_slots > 0:
        status = "available"
    else:
        status = "saturated"
    logical = registry.vision_default_model or "unconfigured"
    profile = registry.resolve_vision(logical) if registry.vision_default_model else None
    # Diagnostic de résidence : /api/ps, délai court, jamais bloquant pour
    # la capacité ; en échec -> null/unavailable, la réponse reste 200.
    resident: Optional[bool] = None
    model_status = "unavailable"
    if profile is not None:
        try:
            names = probe_client.list_running_models(timeout=1)
            resident = any(profile.matches_ollama_name(n) for n in names)
            model_status = "ok"
        except UpstreamError:
            resident, model_status = None, "unavailable"
    body = VisionCapacityResponse(
        environment=settings.environment,
        vision=VisionCapacity(
            enabled=settings.vision_enabled, status=status,
            max_concurrency=snap.max_concurrency, in_flight=snap.in_flight,
            available_slots=snap.available_slots,
            retry_after_seconds=VISION_RETRY_AFTER_SECONDS),
        model=VisionCapacityModel(logical=logical, resident=resident, status=model_status),
    )
    # Valeur volatile : aucun cache intermédiaire ne doit la conserver.
    return JSONResponse(status_code=200, content=body.model_dump(),
                        headers={"Cache-Control": "no-store"})


@app.get("/health/vision")
def health_vision() -> JSONResponse:
    """État de la capability vision. AUCUNE inférence, AUCUN chargement :
    registre + /api/tags (installé, digest) + /api/ps (résident) seulement.
    installed=True + resident=False = disponible, PAS unhealthy."""
    body: Dict[str, Any] = {
        "status": "ok",
        "provider": "ai_gateway",
        "enabled": settings.vision_enabled,
        "environment": settings.environment,
        # Valeur EFFECTIVE après garde production, pas la variable déclarée :
        # c'est elle qui part dans le payload Ollama (lot residency-02 §7).
        "keep_alive_seconds": settings.vision_keep_alive_seconds,
        # Additif (lot AI-GATEWAY-GPU-24A) : attente bornée en Gateway sur DEFERRED transitoire (0 = aucune).
        "host_admission_wait_seconds": settings.vision_host_admission_wait_seconds,
        "default_model": registry.vision_default_model,
        # Build servant la réponse (même source que /health/runtime) — HORS de
        # toute identité de pipeline.
        "gateway_build": settings.build_sha or "not_reported",
        # Version Open : la task sémantique v3 n'est pas incluse (module exclu).
        "semantic_enabled": False,
        "semantic_available": False,
        "models": [],
    }
    if not settings.vision_enabled:
        body["status"] = "disabled"
    try:
        installed = {m.get("name"): m.get("digest")
                     for m in probe_client.list_models(timeout=5)}
        resident_names = set(probe_client.list_running_models(timeout=5))
    except UpstreamError as exc:
        body["status"] = "error"
        logger.warning("health/vision: upstream indisponible: %s", exc.message)
        body["message"] = PUBLIC_UPSTREAM_UNAVAILABLE
        return JSONResponse(status_code=503, content={"detail": body})
    # Provenance STATIQUE du pipeline (lot CV-1) : permet de vérifier, sans
    # requête ni inférence, que production et staging servent le même pipeline
    # qualifié. Aucune donnée de requête, aucun hash d'image.
    body["normalizer_id"] = NORMALIZER_ID
    body["pillow_version"] = PIL.__version__   # diagnostic, HORS fingerprint
    body["model_schema_hash_algorithm"] = MODEL_SCHEMA_HASH_ALGORITHM
    for profile in registry.vision_profiles():
        observed = next((d for n, d in installed.items()
                         if n and profile.matches_ollama_name(n)), None)
        pipelines = {}
        if profile.digest is not None:
            for task in profile.vision_tasks:
                if task == SEMANTIC_TASK:
                    continue   # identité v3 distincte : bloc `semantic_v3` ci-dessous
                qualified = QUALIFIED_PROMPTS.get(task)
                if qualified is None:
                    continue   # task au registre sans prompt qualifié : non servie
                prov = pipeline_provenance(task, profile.digest, qualified[1])
                pipelines[task] = {k: prov[k] for k in (
                    "fingerprint", "system_prompt_sha256", "task_prompt_sha256",
                    "model_schema_sha256")}
        body["models"].append({
            "model_logical": profile.logical_name,
            "concrete": profile.ollama_name,
            "configured": True,
            "allowed_here": profile.allowed_in(settings.environment),
            "installed": observed is not None,
            "digest_match": observed == profile.digest,
            "resident": any(profile.matches_ollama_name(n) for n in resident_names),
            "tasks": list(profile.vision_tasks),
            "schema_versions": list(profile.vision_schema_versions),
            "pipelines": pipelines,
            "semantic_v3": {"status": "unavailable", "reason": "module_not_included"},
        })
    return JSONResponse(status_code=200, content=body)


# --------------------------------------------------------------------------- #
# Rail ASR (lot asr-rail-foundation) : routes publiques dans app/asr_routes.py,
# branchées ici avec le registre, la sonde Ollama et la vérification
# d'identité backend (fail-closed) — jamais servies en production.
# --------------------------------------------------------------------------- #
asr_routes.configure(registry, probe_client, asr_backend_verified)
app.include_router(asr_routes.router)

# Version Open : l'extraction structurée PP-StructureV3 (`app/structx_route.py`,
# backend `structx/`) n'est PAS incluse dans cette distribution. Aucune route
# `/v1/vision/structured-extract` n'est déclarée : un client reçoit 404.
