"""Routes publiques du rail ASR (lot asr-rail-foundation, contrat §16-§24,
§57-§62, §76-§92).

  POST   /v1/audio/transcriptions            multipart `request` puis `file`
  GET    /v1/audio/transcriptions/{job_id}   ?format=json|text|srt|vtt&wait_ms
  DELETE /v1/audio/transcriptions/{job_id}

Doctrine : la Gateway ne touche JAMAIS l'audio (relais en flux vers le
backend, octets comptés, aucun spool local), refuse AVANT tout octet tout
ce qui peut l'être (capability, modèle, schéma, vocabulaire, Content-Length,
concurrence locale, budget GPU, quota backend), et traduit les codes du
backend dans SA taxonomie sans jamais relayer un message ou une URL backend.

Journalisation : identifiants, tailles, durées et codes uniquement — jamais
de contenu, de vocabulaire, de nom de fichier ni de chemin.
"""

import asyncio
import json
import logging
import queue
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.requests import ClientDisconnect

from . import asr_client as asr_transport
from .asr_client import AsrBackendError, AsrBackendRejected, AsrBackendUnavailable
from .asr_projections import MEDIA_TYPES, PROJECTIONS
from .asr_validation import MAX_RESULT_JSON_BYTES, ResponseInvalid, validate_result
from .asr_vocabulary import VocabularyError, normalize_vocabulary
from .errors import UpstreamError, http_error
from .host_admission import HostAdmissionError, host_admission, host_admission_http_error
from .metrics import counters, gauges, log_event
from .settings import settings

logger = logging.getLogger("ai_gateway.asr")
router = APIRouter()

RETRY_AFTER_SECONDS = 30
_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_MULTIPART_OVERHEAD = 64 * 1024
_DIARIZATION_FIELDS = ("min_speakers", "max_speakers", "speakers", "speaker_count", "num_speakers")

# Dépendances injectées par main.py (registre, sonde Ollama, vérification
# d'identité) — évite un import circulaire.
_deps: Dict[str, Any] = {}


def configure(registry: Any, probe_client: Any, backend_verified: Any) -> None:
    _deps["registry"] = registry
    _deps["probe_client"] = probe_client
    _deps["backend_verified"] = backend_verified


# Concurrence LOCALE d'admission (téléversement + attente synchrone) : une
# requête à la fois par instance ; la file bornée vit dans le backend.
_semaphore = threading.BoundedSemaphore(max(1, settings.asr_max_concurrency))


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: Optional[str] = Field(default=None, min_length=1, max_length=64)
    language: str = Field(default="auto", pattern=r"^(auto|fr|en)$")
    timestamps: str = Field(default="segment", pattern=r"^(segment|word)$")
    vocabulary: Optional[List[str]] = None
    diarize: bool = False
    mode: str = Field(default="auto", pattern=r"^(auto|async)$")


# --------------------------------------------------------------- erreurs
def _error(status: int, code: str, message: str, retryable: bool = False, job_id: Optional[str] = None,
           headers: Optional[Dict[str, str]] = None):
    extra: Dict[str, Any] = {"code": code, "retryable": retryable}
    if job_id is not None:
        extra["job_id"] = job_id
    return http_error(status, message, provider="ai_gateway", extra=extra, headers=headers)


def _saturated(code: str = "resource_saturated", message: str = "capacité ASR atteinte", job_id: Optional[str] = None):
    counters.inc("asr_saturated_total")
    return _error(503, code, message, retryable=True, job_id=job_id,
                  headers={"Retry-After": str(RETRY_AFTER_SECONDS)})


# Traduction des codes backend (job.error.code / enveloppe) -> taxonomie
# publique (§79). Toute valeur inconnue devient transcription_failed.
_BACKEND_CODE_MAP: Dict[str, Tuple[int, str, bool]] = {
    "invalid_request": (400, "invalid_request", False),
    "invalid_vocabulary": (400, "invalid_vocabulary", False),
    "diarization_not_supported": (400, "diarization_not_supported", False),
    "invalid_audio": (400, "invalid_audio", False),
    "ambiguous_audio_streams": (400, "ambiguous_audio_streams", False),
    "too_many_channels": (400, "too_many_channels", False),
    "audio_too_large": (413, "audio_too_large", False),
    "audio_too_long": (413, "audio_too_long", False),
    "unsupported_media": (415, "unsupported_media", False),
    "decoding_failed": (422, "decoding_failed", False),
    "resource_saturated": (503, "resource_saturated", True),
    "asr_unavailable": (503, "asr_unavailable", True),
    "asr_timeout": (502, "asr_timeout", True),
    "transcription_failed": (502, "transcription_failed", False),
    "backend_restart": (502, "backend_restart", True),
    "queue_timeout": (503, "queue_timeout", True),
    "job_not_found": (404, "job_not_found", False),
    "job_expired": (410, "job_expired", False),
    "job_cancelled": (409, "job_cancelled", False),
    "job_state": (409, "job_cancelled", False),
}
_PUBLIC_MESSAGES = {
    "invalid_audio": "fichier audio illisible", "ambiguous_audio_streams": "plusieurs flux audio : un seul attendu",
    "too_many_channels": "plus de deux canaux audio", "audio_too_large": "média au-delà de la taille maximale",
    "audio_too_long": "média au-delà de la durée maximale", "unsupported_media": "conteneur ou codec non supporté",
    "decoding_failed": "décodage audio en échec", "resource_saturated": "capacité ASR atteinte",
    "asr_unavailable": "backend ASR indisponible", "asr_timeout": "délai de transcription dépassé",
    "transcription_failed": "transcription en échec", "backend_restart": "backend ASR redémarré pendant le job",
    "queue_timeout": "délai d'attente en file dépassé", "job_not_found": "job inconnu",
    "job_expired": "résultat expiré", "job_cancelled": "job annulé", "invalid_request": "requête invalide",
    "invalid_vocabulary": "vocabulaire invalide",
}


def _translate(code: Optional[str], job_id: Optional[str] = None):
    status, public, retryable = _BACKEND_CODE_MAP.get(code or "", (502, "transcription_failed", False))
    headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if status == 503 else None
    if status == 503:
        counters.inc("asr_saturated_total")
    return _error(status, public, _PUBLIC_MESSAGES.get(public, "transcription en échec"), retryable=retryable,
                  job_id=job_id, headers=headers)


def _from_backend_error(exc: AsrBackendError, job_id: Optional[str] = None):
    if isinstance(exc, AsrBackendUnavailable):
        counters.inc("asr_backend_errors_total")
        log_event(logger, logging.WARNING, "asr.backend_degraded", reason=exc.message, job_id=job_id)
        return _error(503, "asr_unavailable", "backend ASR indisponible", retryable=True, job_id=job_id,
                      headers={"Retry-After": str(RETRY_AFTER_SECONDS)})
    if isinstance(exc, AsrBackendRejected):
        return _translate(exc.code, job_id)
    counters.inc("asr_backend_errors_total")
    return _error(502, "transcription_failed", "réponse backend ASR invalide", job_id=job_id)


# ------------------------------------------------------------- admission
def capability_blocker() -> Optional[str]:
    registry = _deps["registry"]
    if not settings.asr_enabled:
        return "capability ASR désactivée (AI_GATEWAY_ASR_ENABLED)"
    if not asr_transport.backend_url_is_loopback(settings.asr_backend_url):
        return "backend ASR hors loopback : capability refusée"
    if registry.asr_default_model is None:
        return "aucun modèle ASR au registre"
    return None


def _require_capability() -> None:
    blocker = capability_blocker()
    if blocker is not None:
        counters.inc("asr_rejected_total")
        raise _error(503, "asr_unavailable", blocker, retryable=False)


def _resolve_profile(requested: Optional[str]):
    registry = _deps["registry"]
    if requested is None:
        profile = registry.resolve_asr(None)
    else:
        profile = registry.asr_models.get(requested)
        if profile is None:
            if requested in registry.models or requested in registry.llm_models or requested in registry.reranker_models:
                raise _error(400, "transcription_not_supported",
                             f"le modèle {requested} n'est pas un modèle de transcription")
            raise _error(400, "model_not_registered", f"modèle ASR inconnu : {requested}")
    if profile is None:
        raise _error(503, "asr_unavailable", "aucun modèle ASR par défaut", retryable=False)
    if not profile.allowed_in(settings.environment):
        raise _error(400, "model_not_allowed",
                     f"modèle {profile.logical_name} non autorisé dans l'environnement {settings.environment}")
    return profile


def _parse_request_part(raw: bytes) -> Tuple[TranscriptionRequest, List[str]]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise _error(400, "invalid_request", "partie `request` : JSON illisible")
    if isinstance(payload, dict) and any(k in payload for k in _DIARIZATION_FIELDS):
        # Diarisation V1 : aucun paramètre public (nombre de locuteurs,
        # périphérique…) — `diarize` est le seul commutateur admis.
        raise _error(400, "field_requires_diarization", "champ de diarisation non admis (aucun paramètre public)")
    try:
        req = TranscriptionRequest.model_validate(payload)
    except ValidationError as exc:
        fields = sorted({str(e["loc"][0]) for e in exc.errors() if e.get("loc")})
        raise _error(400, "invalid_request", f"partie `request` invalide : {', '.join(fields) or 'schéma'}")
    try:
        terms, _ = normalize_vocabulary(req.vocabulary)
    except VocabularyError as exc:
        raise _error(400, "invalid_vocabulary", exc.reason)
    return req, terms


def _check_profile_features(profile: Any, req: TranscriptionRequest, terms: List[str]) -> None:
    caps = set(profile.capabilities)
    if req.timestamps == "word" and "word_timestamps" not in caps:
        raise _error(400, "invalid_request", "timestamps=word non supporté par ce modèle")
    if terms and "vocabulary_biasing" not in caps:
        raise _error(400, "invalid_request", "vocabulary non supporté par ce modèle")
    if req.language == "auto" and "multilingual" not in caps:
        raise _error(400, "invalid_request", "language=auto non supporté par ce modèle")
    if req.language != "auto" and req.language not in profile.languages:
        raise _error(400, "invalid_request", f"langue {req.language} non supportée par ce modèle")
    if req.diarize and "diarization" not in caps:
        raise _error(400, "diarization_not_supported", "diarisation non supportée par ce modèle")


def _gpu_admission(profile: Any) -> None:
    """Budget VRAM (§57-§62) : résident Ollama (size_vram RETOURNÉS) + réserve
    ASR + base <= budget. Jamais de déchargement, jamais de nom de modèle
    codé en dur. /api/ps injoignable => 503 asr_unavailable."""
    probe_client = _deps["probe_client"]
    try:
        resident = probe_client.list_running_models_vram(timeout=settings.asr_ps_timeout_seconds)
    except UpstreamError as exc:
        counters.inc("asr_backend_errors_total")
        log_event(logger, logging.WARNING, "asr.backend_degraded", reason="ollama_ps_unreachable")
        raise _error(503, "asr_unavailable", "état GPU non observable (Ollama /api/ps)", retryable=True,
                     headers={"Retry-After": str(RETRY_AFTER_SECONDS)})
    resident_mib = sum(int(m.get("size_vram", 0)) for m in resident) // (1024 * 1024)
    needed = resident_mib + settings.asr_gpu_reserve_mib + settings.asr_gpu_baseline_mib
    if needed > settings.asr_gpu_budget_mib:
        counters.inc("asr_gpu_contention_total")
        log_event(logger, logging.INFO, "asr.saturated", reason="gpu_budget", resident_mib=resident_mib,
                  needed_mib=needed, budget_mib=settings.asr_gpu_budget_mib, resident_count=len(resident))
        raise _saturated("resource_saturated", "budget GPU insuffisant pour la transcription")


# ------------------------------------------------------------ multipart
class _MultipartRelay:
    """Analyse en flux du multipart : `request` (JSON <= 8 Kio) puis `file`
    poussé morceau par morceau vers une file consommée par le thread de
    relais. Aucune copie complète du média côté Gateway."""

    def __init__(self, boundary: bytes, on_request) -> None:
        # `on_request(bytes) -> int` : valide la partie `request` DÈS sa fin
        # (avant tout octet de `file`) et retourne le plafond d'octets du
        # média pour le modèle résolu.
        self.on_request = on_request
        self.max_file_bytes = 0
        self.part_index = 0
        self.current_name: Optional[str] = None
        self.current_ctype: Optional[str] = None
        self._hf: List[bytes] = []
        self._hv: List[bytes] = []
        self._headers: Dict[bytes, bytes] = {}
        self.request_bytes = bytearray()
        self.request_done = False
        self.file_started = False
        self.file_done = False
        self.file_bytes = 0
        self.chunks: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=256)
        self.error: Optional[Exception] = None
        self.parser = MultipartParser(boundary, {
            "on_part_begin": self._part_begin, "on_part_data": self._part_data, "on_part_end": self._part_end,
            "on_header_field": self._hfield, "on_header_value": self._hvalue, "on_header_end": self._hend,
            "on_headers_finished": self._headers_finished, "on_end": lambda: None,
        })

    # -- en-têtes de partie
    def _part_begin(self) -> None:
        self._headers = {}
        self.current_name = None
        self.current_ctype = None

    def _hfield(self, data: bytes, start: int, end: int) -> None:
        self._hf.append(data[start:end])

    def _hvalue(self, data: bytes, start: int, end: int) -> None:
        self._hv.append(data[start:end])

    def _hend(self) -> None:
        self._headers[b"".join(self._hf).lower()] = b"".join(self._hv)
        self._hf, self._hv = [], []

    def _headers_finished(self) -> None:
        self.part_index += 1
        disposition = self._headers.get(b"content-disposition", b"")
        _, params = parse_options_header(disposition)
        name = params.get(b"name", b"").decode("utf-8", "replace")
        self.current_name = name
        self.current_ctype = self._headers.get(b"content-type", b"").decode("utf-8", "replace").split(";")[0].strip().lower()
        if self.part_index == 1:
            if name != "request":
                raise _error(400, "invalid_request", "la première partie doit être `request`")
            if self.current_ctype and self.current_ctype != "application/json":
                raise _error(400, "invalid_request", "partie `request` : application/json attendu")
        elif self.part_index == 2:
            if name != "file":
                raise _error(400, "invalid_request", "la seconde partie doit être `file`")
            self.file_started = True
        else:
            raise _error(400, "invalid_request", "parties multipart excédentaires")

    # -- données
    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if self.part_index == 1:
            self.request_bytes.extend(data[start:end])
            if len(self.request_bytes) > settings.asr_max_request_part_bytes:
                raise _error(400, "invalid_request", "partie `request` au-delà de 8 Kio")
        elif self.part_index == 2:
            piece = bytes(data[start:end])
            self.file_bytes += len(piece)
            if self.file_bytes > self.max_file_bytes:
                raise _error(413, "audio_too_large", "média au-delà de la taille maximale")
            self.chunks.put(piece, timeout=settings.asr_upload_timeout_seconds)

    def _part_end(self) -> None:
        if self.part_index == 1:
            self.request_done = True
            self.max_file_bytes = int(self.on_request(bytes(self.request_bytes)))
        elif self.part_index == 2:
            self.file_done = True
            self.chunks.put(None)

    def write(self, data: bytes) -> None:
        self.parser.write(data)

    def finalize(self) -> None:
        self.parser.finalize()

    def abort(self) -> None:
        """Interrompt le relais : le thread consommateur lève, la connexion
        vers le backend casse, le backend voit un téléversement incomplet et
        supprime le job (§38). Jamais une fin « propre » sur un flux tronqué."""
        self.chunks.put(_ABORT)

    def iter_file(self):
        while True:
            piece = self.chunks.get()
            if piece is _ABORT:
                raise RuntimeError("upload aborted")
            if piece is None:
                return
            yield piece


_ABORT = object()


def _boundary(request: Request) -> bytes:
    ctype = request.headers.get("content-type", "")
    main, params = parse_options_header(ctype.encode("latin-1", "replace"))
    if main != b"multipart/form-data" or not params.get(b"boundary"):
        raise _error(400, "invalid_request", "multipart/form-data attendu (parties `request` puis `file`)")
    return params[b"boundary"]


# ------------------------------------------------------------ handlers
@router.post("/v1/audio/transcriptions")
async def create_transcription(request: Request):
    started = time.time()
    counters.inc("asr_requests_total")
    _require_capability()
    boundary = _boundary(request)
    default_profile = _deps["registry"].resolve_asr(None)
    max_upload = default_profile.max_upload_bytes if default_profile else 0
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared_total = int(length)
        except ValueError:
            raise _error(400, "invalid_request", "Content-Length illisible")
        if declared_total > max_upload + _MULTIPART_OVERHEAD:
            counters.inc("asr_rejected_total")
            log_event(logger, logging.INFO, "asr.rejected", code="audio_too_large", declared_bytes=declared_total)
            raise _error(413, "audio_too_large", "média au-delà de la taille maximale")
    if not _semaphore.acquire(blocking=False):
        log_event(logger, logging.INFO, "asr.saturated", reason="local_concurrency")
        raise _saturated()
    gauges.inc("asr:admission")
    try:
        return await _handle_upload(request, boundary, started)
    finally:
        gauges.dec("asr:admission")
        _semaphore.release()


async def _handle_upload(request: Request, boundary: bytes, started: float):
    state: Dict[str, Any] = {"req": None, "terms": [], "profile": None}

    def _on_request(raw: bytes) -> int:
        req_, terms_ = _parse_request_part(raw)
        profile_ = _resolve_profile(req_.model)
        _check_profile_features(profile_, req_, terms_)
        state.update(req=req_, terms=terms_, profile=profile_)
        return profile_.max_upload_bytes

    relay = _MultipartRelay(boundary, _on_request)
    stream = request.stream()
    profile = None
    req: Optional[TranscriptionRequest] = None
    terms: List[str] = []
    job_id: Optional[str] = None
    upload_task = None
    job: Optional[Dict[str, Any]] = None
    host_lease = None

    def _reject(code: str, **fields: Any) -> None:
        counters.inc("asr_rejected_total")
        log_event(logger, logging.INFO, "asr.rejected", code=code, **fields)

    try:
        async for chunk in stream:
            relay.write(chunk)
            if state["req"] is not None and job_id is None:
                req, terms, profile = state["req"], state["terms"], state["profile"]
                # Admission GPU + réservation backend AVANT de relayer le
                # premier octet du média (§57 : le budget se vérifie avant
                # d'accepter) — les octets déjà analysés attendent en file.
                # Admission HÔTE (lot host-wide-admission) AVANT la garde VRAM locale,
                # qui reste en place (ASR_EXISTING_LOCAL_VRAM_GUARD_REMOVED = NO).
                try:
                    host_lease = await run_in_threadpool(host_admission.acquire, "asr", profile.logical_name)
                except HostAdmissionError as exc:
                    raise host_admission_http_error(exc)
                await run_in_threadpool(_gpu_admission, profile)
                ok, checks, message = _deps["backend_verified"](profile)
                if not ok:
                    _reject("asr_unavailable", reason="identity_or_unreachable")
                    raise _error(503, "asr_unavailable", message or "backend ASR indisponible", retryable=True,
                                 headers={"Retry-After": str(RETRY_AFTER_SECONDS)})
                spec = {"owner_env": settings.environment, "model": profile.logical_name, "language": req.language,
                        "timestamps": req.timestamps, "diarize": req.diarize, "vocabulary": terms or None,
                        "declared_bytes": None, "max_duration_seconds": profile.max_duration_seconds,
                        "max_upload_bytes": profile.max_upload_bytes}
                host_admission.mark_started(host_lease)   # lot ORPHAN-LEASE : backend engagé
                try:
                    job = await run_in_threadpool(asr_transport.asr_client.create_job, spec)
                except AsrBackendError as exc:
                    host_admission.release(host_lease)
                    _reject(getattr(exc, "code", None) or "asr_unavailable")
                    raise _from_backend_error(exc)
                job_id = job["job_id"]
                _track_asr_lease(job_id, host_lease)
                counters.inc("asr_accepted_total")
                log_event(logger, logging.INFO, "asr.accepted", job_id=job_id, model=profile.logical_name,
                          language=req.language, timestamps=req.timestamps, vocabulary_count=len(terms),
                          diarize=req.diarize, mode=req.mode, environment=settings.environment)
                upload_task = asyncio.get_running_loop().run_in_executor(
                    None, asr_transport.asr_client.upload_media, job_id, relay.iter_file(), None,
                    settings.environment)
        relay.finalize()
    except ClientDisconnect:
        relay.abort()
        if job_id is not None:
            await _abandon(job_id, upload_task)
        else:
            host_admission.release(host_lease)
        log_event(logger, logging.INFO, "asr.rejected", code="upload_interrupted", job_id=job_id)
        counters.inc("asr_rejected_total")
        raise _error(400, "invalid_request", "téléversement interrompu")
    except Exception as exc:
        relay.abort()
        if job_id is not None:
            await _abandon(job_id, upload_task)
        else:
            host_admission.release(host_lease)
        if not hasattr(exc, "status_code"):
            counters.inc("asr_rejected_total")
            log_event(logger, logging.INFO, "asr.rejected", code="invalid_request", reason=type(exc).__name__)
            raise _error(400, "invalid_request", "multipart illisible")
        _reject(getattr(exc, "detail", {}).get("code", "invalid_request") if isinstance(getattr(exc, "detail", None), dict) else "invalid_request")
        raise
    if req is None or not relay.request_done:
        host_admission.release(host_lease)
        _reject("invalid_request", reason="missing_request_part")
        raise _error(400, "invalid_request", "partie `request` absente")
    if not relay.file_done or relay.file_bytes == 0:
        relay.abort()
        if job_id is not None:
            await _abandon(job_id, upload_task)
        _reject("invalid_request", reason="missing_file_part")
        raise _error(400, "invalid_request", "partie `file` absente ou vide")
    assert job_id is not None and upload_task is not None and profile is not None
    try:
        job = await upload_task
    except AsrBackendError as exc:
        _reject(getattr(exc, "code", None) or "backend", job_id=job_id, upload_bytes=relay.file_bytes)
        raise _from_backend_error(exc, job_id)
    media = job.get("media") or {}
    duration_ms = int(job.get("duration_ms") or media.get("duration_ms") or 0)
    log_event(logger, logging.INFO, "asr.started", job_id=job_id, upload_bytes=relay.file_bytes,
              duration_ms=duration_ms, container=media.get("container"), codec=media.get("codec"))

    sync = req.mode == "auto" and duration_ms <= profile.sync_max_duration_seconds * 1000
    if sync:
        job = await run_in_threadpool(_wait_sync, job_id, job)
        if job["status"] == "completed":
            result = _validated_result(job, profile.logical_name, req.timestamps, req.diarize)
            _log_terminal(job, started)
            return JSONResponse(status_code=200, content=result)
        if job["status"] in ("failed", "cancelled"):
            _log_terminal(job, started)
            raise _job_failure(job)
    return JSONResponse(status_code=202, content=_job_status(job, req.mode))


async def _abandon(job_id: str, upload_task) -> None:
    _release_asr_lease(job_id)
    if upload_task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(upload_task), timeout=settings.asr_control_timeout_seconds)
        except Exception:
            pass
    try:
        await run_in_threadpool(asr_transport.asr_client.delete_job, job_id, settings.environment)
    except AsrBackendError:
        pass


def _wait_sync(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    deadline = time.monotonic() + settings.asr_sync_wait_seconds
    while job["status"] in ("queued", "processing"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        wait_ms = int(min(remaining, 30) * 1000)
        try:
            job = asr_transport.asr_client.get_job(job_id, settings.environment, wait_ms=wait_ms)
        except AsrBackendError as exc:
            raise _from_backend_error(exc, job_id)
    return job


def _validated_result(job: Dict[str, Any], model: str, timestamps: str, diarize: bool = False) -> Dict[str, Any]:
    result = job.get("result")
    try:
        encoded = json.dumps(result, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > MAX_RESULT_JSON_BYTES:
            raise ResponseInvalid("output_bounds_exceeded")
        return validate_result(result, job["job_id"], model, timestamps, diarize)
    except (ResponseInvalid, TypeError, ValueError) as exc:
        reason = getattr(exc, "reason", "invalid")
        if _first_observation(job["job_id"], "failed"):
            counters.inc("asr_failed_total")
            log_event(logger, logging.WARNING, "asr.failed", job_id=job["job_id"], code="transcription_failed",
                      retryable=False, reason=reason, status="failed")
        raise _error(502, "transcription_failed", "réponse backend ASR invalide", job_id=job["job_id"])


def _job_failure(job: Dict[str, Any]):
    if job["status"] == "cancelled":
        return _error(409, "job_cancelled", "job annulé", job_id=job["job_id"])
    err = job.get("error") or {}
    return _translate(err.get("code"), job["job_id"])


# Comptabilité des états terminaux (AK-2). La TRANSITION a lieu dans le
# backend (point unique : store.fail/complete/cancel/expire, compteurs
# jobs_*_total de son /health). La Gateway ne voit un job que lorsqu'elle le
# lit : elle compte et journalise à la PREMIÈRE observation d'un état
# terminal, quelle que soit la voie (attente synchrone, GET, DELETE), et
# jamais aux lectures suivantes. Mémoire bornée : au plus 4096 job_id.
_ACCOUNTED_MAX = 4096
_accounted: "OrderedDict[str, str]" = OrderedDict()
_accounted_lock = threading.Lock()

# Leases HÔTE des jobs ASR (lot host-wide-admission) : la lease couvre le job
# jusqu'au premier état terminal observé (attente sync, GET, DELETE) ou à
# l'abandon ; une lease jamais rendue expire par TTL côté contrôleur. Mémoire
# bornée comme la comptabilité terminale.
_asr_host_leases: "OrderedDict[str, Any]" = OrderedDict()


def _track_asr_lease(job_id: str, lease: Any) -> None:
    if lease is None:
        return
    with _accounted_lock:
        _asr_host_leases[job_id] = lease
        while len(_asr_host_leases) > _ACCOUNTED_MAX:
            _, old = _asr_host_leases.popitem(last=False)
            host_admission.release(old)


def _release_asr_lease(job_id: Optional[str]) -> None:
    if job_id is None:
        return
    with _accounted_lock:
        lease = _asr_host_leases.pop(job_id, None)
    host_admission.release(lease)


def reset_terminal_accounting() -> None:
    """Réservé aux tests."""
    with _accounted_lock:
        _accounted.clear()


def _first_observation(job_id: str, status: str) -> bool:
    with _accounted_lock:
        if job_id in _accounted:
            return False
        _accounted[job_id] = status
        while len(_accounted) > _ACCOUNTED_MAX:
            _accounted.popitem(last=False)
        return True


def _observe_terminal(job: Dict[str, Any], started: Optional[float] = None) -> None:
    """Compteur + événement, exactement une fois par job terminal observé."""
    status = job["status"]
    if status not in ("completed", "failed", "cancelled", "expired"):
        return
    _release_asr_lease(job["job_id"])
    if not _first_observation(job["job_id"], status):
        return
    err = job.get("error") or {}
    result = job.get("result") or {}
    usage = result.get("usage") or {}
    fields: Dict[str, Any] = {"job_id": job["job_id"], "status": status, "duration_ms": job.get("duration_ms")}
    if started is not None:
        fields["latency_ms"] = int((time.time() - started) * 1000)
    if status == "completed":
        counters.inc("asr_completed_total")
        counters.inc("asr_audio_seconds_total", int((usage.get("audio_ms") or 0) // 1000))
        counters.inc("asr_processing_ms_total", int(usage.get("processing_ms") or 0))
        log_event(logger, logging.INFO, "asr.completed", processing_ms=usage.get("processing_ms"),
                  rtf=usage.get("rtf"), segment_count=usage.get("segment_count"),
                  word_count=usage.get("word_count"), warnings=len(result.get("warnings") or []), **fields)
        if "duration_mismatch" in (result.get("warnings") or []):
            log_event(logger, logging.INFO, "asr.duration_mismatch", job_id=job["job_id"])
    elif status == "failed":
        counters.inc("asr_failed_total")
        if err.get("code") == "asr_timeout":
            counters.inc("asr_timeouts_total")
        log_event(logger, logging.WARNING, "asr.failed", code=err.get("code"), retryable=bool(err.get("retryable")),
                  **fields)
    elif status == "cancelled":
        counters.inc("asr_cancelled_total")
        log_event(logger, logging.INFO, "asr.cancelled", **fields)
    elif status == "expired":
        counters.inc("asr_expired_total")
        log_event(logger, logging.INFO, "asr.expired", **fields)


def _log_terminal(job: Dict[str, Any], started: float) -> None:
    _observe_terminal(job, started)


def _job_status(job: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
    err = job.get("error") or {}
    return {
        "transcription_id": job["job_id"],
        "job_id": job["job_id"],
        "status": job["status"],
        "model_logical": job.get("model"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "duration_ms": job.get("duration_ms"),
        "processed_audio_ms": job.get("processed_audio_ms"),
        "result_available": bool(job.get("result_available")),
        "error": ({"code": _BACKEND_CODE_MAP.get(err.get("code", ""), (0, "transcription_failed", False))[1],
                   "retryable": bool(err.get("retryable"))} if err else None),
        "poll": {"path": f"/v1/audio/transcriptions/{job['job_id']}", "wait_ms_max": 30000},
    }


@router.get("/v1/audio/transcriptions/{job_id}")
async def get_transcription(job_id: str, format: str = Query(default="json", pattern=r"^(json|text|srt|vtt)$"),
                            wait_ms: int = Query(default=0, ge=0, le=30000)):
    _require_capability()
    if not _JOB_ID_RE.match(job_id):
        raise _error(404, "job_not_found", "job inconnu")
    try:
        job = await run_in_threadpool(asr_transport.asr_client.get_job, job_id, settings.environment, wait_ms)
    except AsrBackendError as exc:
        if getattr(exc, "code", None) == "job_expired":
            _observe_terminal({"job_id": job_id, "status": "expired"})
        raise _from_backend_error(exc, job_id)
    status = job["status"]
    if status in ("queued", "processing"):
        return JSONResponse(status_code=202, content=_job_status(job))
    _observe_terminal(job)
    if status == "expired":
        raise _error(410, "job_expired", "résultat expiré", job_id=job_id)
    if status in ("failed", "cancelled"):
        raise _job_failure(job)
    profile = _deps["registry"].asr_models.get(job.get("model") or "")
    timestamps = job.get("timestamps") or "segment"
    result = _validated_result(job, job.get("model"), timestamps, job.get("diarize") is True)
    if format == "json":
        return JSONResponse(status_code=200, content=result)
    return PlainTextResponse(PROJECTIONS[format](result), media_type=MEDIA_TYPES[format])


@router.delete("/v1/audio/transcriptions/{job_id}")
async def delete_transcription(job_id: str):
    _require_capability()
    if not _JOB_ID_RE.match(job_id):
        raise _error(404, "job_not_found", "job inconnu")
    try:
        job = await run_in_threadpool(asr_transport.asr_client.delete_job, job_id, settings.environment)
    except AsrBackendError as exc:
        raise _from_backend_error(exc, job_id)
    status = job.get("status")
    if status == "cancelled":
        _observe_terminal(job)
    return JSONResponse(status_code=200, content={"transcription_id": job_id, "job_id": job_id,
                                                  "status": "cancelled" if status == "cancelled" else "deleted"})
