"""Client HTTP vers le backend ASR local (faster-whisper, loopback).

Le backend est un service dédié (asr/, NON inclus dans cette distribution,
attendu sur 127.0.0.1:8400) qui possède le spool, le décodage et le moteur de jobs. La
Gateway ne touche jamais l'audio : elle RELAIE le flux multipart vers
`PUT /jobs/{id}/media` sans le spooler, et lit l'état du job.

Même doctrine que `rerank_client.py` : PUR TRANSPORT. Ce module ne projette
pas, ne valide pas les invariants de timestamps, ne juge pas l'admission ;
il convertit les erreurs réseau en exceptions typées et ne relaie JAMAIS le
corps d'erreur brut du backend (il pourrait citer un chemin de spool). Les
codes d'erreur applicatifs du backend (`code` de son enveloppe) sont relayés
sous forme d'identifiant court, jamais son `message` ni son URL.

Sécurité : l'URL du backend ne sort jamais de ce module (ni réponse ni log —
politique `redacted`, identique à l'URL Ollama et au backend reranker).
"""

import logging
import re
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import requests

from .settings import settings

logger = logging.getLogger("ai_gateway.asr")

# Implémentation concrète du backend (FAIT du code, exposé par /health/runtime).
BACKEND_IMPLEMENTATION = "faster-whisper"

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_CODE_RE = re.compile(r"^[a-z_]{1,48}$")


def backend_url_is_loopback(url: str) -> bool:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


class AsrBackendError(Exception):
    """Erreur de transport ou de forme côté backend ASR (jamais le corps brut)."""

    def __init__(self, message: str, code: Optional[str] = None, status: Optional[int] = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.retryable = retryable


class AsrBackendUnavailable(AsrBackendError):
    """Backend injoignable / timeout de contrôle : 503 asr_unavailable côté route."""


class AsrBackendRejected(AsrBackendError):
    """Refus applicatif du backend (4xx/5xx avec `code` lisible) : la route
    traduit le `code` dans SA taxonomie ; le `message` backend n'est pas relayé."""


def _extract_code(resp: requests.Response) -> Optional[str]:
    try:
        body = resp.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            code = detail.get("code")
            if isinstance(code, str) and _CODE_RE.match(code):
                return code
    return None


def _raise_for_status(resp: requests.Response, what: str) -> None:
    if resp.status_code < 400:
        return
    code = _extract_code(resp)
    retry = resp.status_code in (503, 502)
    raise AsrBackendRejected(f"{what}: HTTP {resp.status_code}", code=code, status=resp.status_code,
                             retryable=retry)


class AsrBackendClient:
    def __init__(self, base_url: str, control_timeout: int, upload_timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.control_timeout = control_timeout
        self.upload_timeout = upload_timeout

    # ------------------------------------------------------------- health
    def is_healthy(self, timeout: Optional[int] = None) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout or self.control_timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def health(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Payload /health du backend (état worker, files, spool). Ne charge
        jamais le modèle (contrat §71)."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout or self.control_timeout)
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise AsrBackendUnavailable(f"échec /health backend ASR: {type(exc).__name__}") from exc
        if not isinstance(data, dict):
            raise AsrBackendUnavailable("réponse /health backend ASR inattendue")
        data["_http_status"] = resp.status_code
        return data

    def info(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Identité servie par le backend : provider, artifact, revision,
        artifact_sha256, vad, versions, capabilities. Matière première de la
        vérification /health/asr (équivalent de `verify_digests`)."""
        try:
            resp = requests.get(f"{self.base_url}/info", timeout=timeout or self.control_timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise AsrBackendUnavailable(f"échec /info backend ASR: {type(exc).__name__}") from exc
        if not isinstance(data, dict):
            raise AsrBackendUnavailable("réponse /info backend ASR inattendue")
        return data

    # --------------------------------------------------------------- jobs
    def create_job(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """POST /jobs — réserve un job (quota file/spool vérifié AVANT tout
        octet). Retourne le job (job_id uuid4 + status queued)."""
        try:
            resp = requests.post(f"{self.base_url}/jobs", json=spec, timeout=self.control_timeout)
        except requests.RequestException as exc:
            raise AsrBackendUnavailable(f"échec POST /jobs backend ASR: {type(exc).__name__}") from exc
        _raise_for_status(resp, "POST /jobs")
        return self._job_payload(resp)

    def upload_media(self, job_id: str, chunks: Iterable[bytes], declared_bytes: Optional[int],
                     owner_env: str) -> Dict[str, Any]:
        """PUT /jobs/{id}/media — relais EN FLUX (chunked). Le générateur
        `chunks` est consommé au fil de l'eau : aucune copie disque ni
        mémoire complète côté Gateway. Retourne le job après admission média
        (probe, allowlists, durée)."""
        self._check_job_id(job_id)
        headers = {"Content-Type": "application/octet-stream", "X-ASR-Owner-Env": owner_env}
        if declared_bytes is not None:
            headers["X-ASR-Declared-Bytes"] = str(declared_bytes)
        try:
            resp = requests.put(f"{self.base_url}/jobs/{job_id}/media", data=chunks, headers=headers,
                                timeout=(self.control_timeout, self.upload_timeout))
        except requests.RequestException as exc:
            raise AsrBackendUnavailable(f"échec PUT media backend ASR: {type(exc).__name__}") from exc
        _raise_for_status(resp, "PUT media")
        return self._job_payload(resp)

    def get_job(self, job_id: str, owner_env: str, wait_ms: int = 0) -> Dict[str, Any]:
        self._check_job_id(job_id)
        wait_ms = max(0, min(int(wait_ms), 30000))
        timeout = self.control_timeout + wait_ms / 1000.0 + 5
        try:
            resp = requests.get(f"{self.base_url}/jobs/{job_id}", params={"wait_ms": wait_ms},
                                headers={"X-ASR-Owner-Env": owner_env}, timeout=timeout)
        except requests.RequestException as exc:
            raise AsrBackendUnavailable(f"échec GET /jobs backend ASR: {type(exc).__name__}") from exc
        _raise_for_status(resp, "GET /jobs")
        return self._job_payload(resp)

    def delete_job(self, job_id: str, owner_env: str) -> Dict[str, Any]:
        self._check_job_id(job_id)
        try:
            resp = requests.delete(f"{self.base_url}/jobs/{job_id}", headers={"X-ASR-Owner-Env": owner_env},
                                   timeout=self.control_timeout)
        except requests.RequestException as exc:
            raise AsrBackendUnavailable(f"échec DELETE /jobs backend ASR: {type(exc).__name__}") from exc
        _raise_for_status(resp, "DELETE /jobs")
        return self._job_payload(resp)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _check_job_id(job_id: str) -> None:
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            raise AsrBackendRejected("job_id mal formé", code="job_not_found", status=404)

    @staticmethod
    def _job_payload(resp: requests.Response) -> Dict[str, Any]:
        """Validation de FORME uniquement : un mapping avec `job_id` et
        `status` lisibles. Le contenu (résultat) est validé par la route."""
        try:
            data = resp.json()
        except ValueError as exc:
            raise AsrBackendError("réponse job backend ASR illisible") from exc
        if not isinstance(data, dict):
            raise AsrBackendError("réponse job backend ASR inattendue")
        job_id = data.get("job_id")
        status = data.get("status")
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id) or not isinstance(status, str):
            raise AsrBackendError("réponse job backend ASR incomplète")
        return data


asr_client = AsrBackendClient(
    settings.asr_backend_url, settings.asr_control_timeout_seconds, settings.asr_upload_timeout_seconds
)
