"""Client Gateway du contrôleur d'admission HÔTE (lot AI-GATEWAY-HOST-WIDE-ADMISSION).

Les deux Gateways (production :8100, staging :8200) sont CLIENTES d'une
autorité unique sur l'hôte ; aucune d'elles ne détient l'état d'admission
global. L'environnement n'est transmis que pour l'attribution.

Modes (settings) :
  désactivé (défaut)  : `acquire()` rend None, comportement d'avant ce lot
  shadow              : le contrôleur décide, la décision est journalisée et
                        la lease immédiatement rendue ; la requête n'est PAS
                        contrôlée (qualification en staging)
  enforce             : GRANTED -> lease tenue jusqu'au `finally` de la route ;
                        DEFERRED/REJECTED -> 503 typé ; contrôleur injoignable
                        -> classes LOURDES (vision, llm_generation) 503
                        `host_admission_unavailable` (jamais de fail-open vers
                        une double admission inter-Gateways), classes légères
                        (embedding, rerank, asr : gardes locales propres)
                        repli local compté.

Les gardes locales existantes (VisionAdmission, sémaphore rerank, budget VRAM
ASR, file backend) RESTENT en place : le contrôleur est une couche de
coordination au-dessus, pas un remplacement.
"""

import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from collections import OrderedDict

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import http_error
from .metrics import counters, gauges, log_event
from .settings import settings

logger = logging.getLogger("ai_gateway.host_admission")

# Lot GPU-24A : `structx` (worker PP-StructureV3, 5,7-6,4 Gio) est LOURDE au sens du repli : contrôleur injoignable =>
# 503 host_admission_unavailable, jamais une admission locale parallèle à celle de la Vision.
HEAVY_CLASSES = ("vision", "llm_generation", "structx")
CLASS_FLAG = {"vision": "host_admission_vision_enabled", "llm_generation": "host_admission_llm_enabled",
              "embedding": "host_admission_embedding_enabled", "rerank": "host_admission_rerank_enabled",
              "asr": "host_admission_asr_enabled", "structx": "host_admission_structx_enabled"}
UNAVAILABLE_RETRY_AFTER_SECONDS = 5

# Lot ORPHAN-LEASE (CANCEL_WITH_FENCING). Un délai de transport n'est PAS un
# « non » : après l'envoi, la Gateway ignore si le contrôleur a accordé.
#   NOT_SENT  : la requête n'a pas pu entrer dans le contrôleur (connexion
#               refusée / délai de CONNEXION / 4xx de validation) => aucune lease
#   AMBIGUOUS : délai de LECTURE, coupure après envoi, 5xx, réponse illisible
#               => abort PAR CLÉ (fencing) avant toute conclusion
NOT_SENT, AMBIGUOUS = "NOT_SENT", "AMBIGUOUS"
ABORT_OUTCOMES = ("lease_released", "nothing_held", "already_aborted", "already_closed")
# Borne de vie d'un élément du reaper = TTL de lease du contrôleur (timeout de
# route + 10 s, profils calibrés) + marge : au-delà, la lease a de toute façon expiré.
LEASE_TTL_BOUND_SECONDS = {"vision": 70, "llm_generation": 190, "embedding": 70, "rerank": 70, "asr": 910, "structx": 160}
REAPER_MARGIN_SECONDS = 10
# Lot SEMX-4C : marge du contrôleur entre délai DEMANDÉ et TTL de lease
# (scheduler.LEASE_TTL_MARGIN_SECONDS). Une réservation à délai explicite T
# borne son élément de reaper à max(borne de classe, T + 10) — jamais une
# extension globale des autres leases.
REQUEST_TTL_MARGIN_SECONDS = 10


def lease_ttl_bound(workload_class: str, timeout_seconds: Optional[int] = None) -> int:
    base = LEASE_TTL_BOUND_SECONDS.get(workload_class, max(LEASE_TTL_BOUND_SECONDS.values()))
    if timeout_seconds is None:
        return base
    return max(base, int(timeout_seconds) + REQUEST_TTL_MARGIN_SECONDS)
_INLINE_BACKOFF_SECONDS = (0.0, 0.02, 0.05, 0.1, 0.2)


class _ControllerHTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"http_{status}")
        self.status = status


def classify_transport_failure(exc: BaseException) -> str:
    """NOT_SENT uniquement quand l'absence de traitement est PROUVÉE ; tout le
    reste est AMBIGUOUS (le contrôleur a peut-être accordé)."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return NOT_SENT
    if isinstance(exc, ConnectionRefusedError):
        return NOT_SENT
    if isinstance(exc, (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema,
                        requests.exceptions.InvalidSchema)):
        return NOT_SENT
    if isinstance(exc, _ControllerHTTPError) and 400 <= exc.status < 500:
        return NOT_SENT                      # rejet de validation AVANT le planificateur
    if isinstance(exc, requests.exceptions.ConnectionError) and not isinstance(exc, requests.exceptions.ReadTimeout):
        reason = getattr(exc.args[0], "reason", None) if exc.args else None
        if isinstance(reason, (urllib3.exceptions.NewConnectionError, urllib3.exceptions.ConnectTimeoutError)):
            return NOT_SENT                  # connexion jamais établie : rien n'a été envoyé
    return AMBIGUOUS


def _no_retry_session() -> requests.Session:
    """Transport EXPLICITE : aucune relance implicite (urllib3) — le protocole de
    récupération possède seul les nouvelles tentatives ; pas de réutilisation de
    connexion (comportement inchangé)."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=0, connect=0, read=0, redirect=0, status=0, other=0,
                                            raise_on_status=False))
    session.mount("http://", adapter)
    return session


def _http_post(url, json=None, timeout=None):
    with _no_retry_session() as session:
        return session.post(url, json=json, timeout=timeout)


def _http_delete(url, params=None, timeout=None):
    with _no_retry_session() as session:
        return session.delete(url, params=params, timeout=timeout)


def _labeled(name: str, cls: str) -> str:
    return f"{name}{{class={cls}}}"

# Famille de retry (lot PENDING-AGING, LIVENESS_BOUND_RETRY_FAMILY) : seules les
# classes où l'APPELANT fournit une identité logique stable la propagent. Vision :
# `request_id` appelant (RagATM : `rgv1-…`, stable entre retries Celery). Aucune
# identité n'est inventée pour les autres routes (DEFERRED sans état).
RETRY_FAMILY_CLASSES = ("vision", "structx")
# Lot GPU-24A : raisons de DEFERRED TRANSITOIRES par construction (une ressource se libère d'elle-même en secondes :
# keep_alive vision, TTL du worker StructX, fin d'un job) sur lesquelles une attente bornée a un sens. Jamais sur un
# état dégradé (télémétrie, reconstruction 120 s) : refus immédiat, le client réessaie plus tard.
WAITABLE_REASONS = frozenset({"insufficient_vram", "model_swap_headroom", "incompatible_active_workload",
                              "workload_slot_saturated", "residency_preference", "aging_reservation", "priority_yield"})
RETRY_FAMILY_VERSION = "rf1"
# Clé HMAC aléatoire PAR PROCESSUS Gateway (jamais écrite, jamais journalisée) :
# famille déterministe pour la vie du processus (la seule portée utile : une
# attente vit ≤ quelques dizaines de secondes), non réversible même pour un
# request_id à faible entropie ; un redémarrage Gateway remet l'âge à zéro.
_FAMILY_KEY = secrets.token_bytes(32)


def derive_retry_family(requester_instance: str, workload_class: str, request_id: str) -> str:
    """`rf1-` + 32 hex de HMAC-SHA256(clé processus, JSON canonique
    ["aigw-retry-family/1", instance, classe, request_id])."""
    canonical = json.dumps(["aigw-retry-family/1", requester_instance, workload_class, request_id],
                           ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{RETRY_FAMILY_VERSION}-" + hmac.new(_FAMILY_KEY, canonical, hashlib.sha256).hexdigest()[:32]


class HostAdmissionError(Exception):
    reason: str = "host_admission_error"


class HostAdmissionDeferred(HostAdmissionError):
    def __init__(self, reason: str, retry_after_seconds: int) -> None:
        super().__init__(reason)
        self.reason, self.retry_after_seconds = reason, max(1, int(retry_after_seconds or 1))


class HostAdmissionRejected(HostAdmissionError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class HostAdmissionUnavailable(HostAdmissionError):
    reason = "controller_unavailable"


@dataclass
class HostLease:
    lease_id: str
    workload_class: str
    model_logical: str
    shadow: bool = False
    released: bool = False
    # Vrai SEULEMENT quand la route a effectivement engagé le backend : une
    # libération avant (garde locale, validation) ne crée aucune traîne.
    backend_started: bool = False
    # Délai DEMANDÉ au contrôleur (lot SEMX-4C) ; None = borne de classe.
    timeout_seconds: Optional[int] = None


class HostAdmissionClient:
    def __init__(self) -> None:
        self._post = _http_post
        self._delete = _http_delete
        self._sleep = time.sleep
        self._lock = threading.Lock()
        # Reaper : aborts et libérations NON confirmés (identifiants opaques seulement)
        self._reaper_items: "OrderedDict[tuple, dict]" = OrderedDict()
        self._reaper_thread: Optional[threading.Thread] = None
        self._reaper_wake = threading.Event()
        self.autostart_reaper = True
        # Lot GPU-24A : UN attenteur par classe et par processus (jamais de file dans le pool de threads).
        self._waiters: dict = {}
        self._time = time.time

    # ---------------------------------------------------------- policy
    def class_enabled(self, workload_class: str) -> bool:
        return bool(settings.host_admission_enabled) and bool(getattr(settings, CLASS_FLAG[workload_class], False))

    @property
    def shadow(self) -> bool:
        return settings.host_admission_mode == "shadow"

    # --------------------------------------------------------- transport
    def _reserve(self, workload_class: str, model_logical: str, idempotency_key: str,
                 retry_family: Optional[str] = None, timeout_seconds: Optional[int] = None) -> dict:
        # Le request_id appelant ne franchit PAS la frontière contrôleur : seule la
        # famille opaque (si l'appelant en fournit une base stable) la franchit.
        body = {"contract_version": "1", "workload_class": workload_class, "model_logical": model_logical,
                "idempotency_key": idempotency_key, "requester_instance": settings.host_admission_instance,
                "requester_environment": settings.environment, "request_id": "-"}
        if retry_family:
            body["retry_family"] = retry_family
        # Lot SEMX-4C : clé ABSENTE quand aucun délai n'est demandé (jamais
        # `null`) — le corps describe reste octet pour octet celui d'avant.
        if timeout_seconds is not None:
            body["timeout_seconds"] = int(timeout_seconds)
        resp = self._post(f"{settings.host_admission_url}/v1/admission/reserve", json=body,
                          timeout=settings.host_admission_timeout_ms / 1000.0)
        if resp.status_code >= 300:
            raise _ControllerHTTPError(resp.status_code)
        data = resp.json()
        if not isinstance(data, dict) or data.get("outcome") not in ("GRANTED", "DEFERRED", "REJECTED"):
            raise ValueError("réponse admission inattendue")   # après envoi : AMBIGUOUS
        return data

    def _abort(self, idempotency_key: str) -> str:
        """Abort PAR CLÉ (fencing) : rend l'issue confirmée ou lève."""
        body = {"contract_version": "1", "requester_instance": settings.host_admission_instance,
                "idempotency_key": idempotency_key}
        resp = self._post(f"{settings.host_admission_url}/v1/admission/abort", json=body,
                          timeout=settings.host_admission_recovery_timeout_ms / 1000.0)
        if resp.status_code != 200:
            raise _ControllerHTTPError(resp.status_code)
        outcome = (resp.json() or {}).get("outcome")
        if outcome not in ABORT_OUTCOMES:
            raise ValueError("réponse abort inattendue")
        return outcome

    def _release_remote(self, lease_id: str, backend_started: bool = True) -> bool:
        """Libération idempotente par lease_id ; True si CONFIRMÉE (libérée ou doublon)."""
        try:
            resp = self._delete(f"{settings.host_admission_url}/v1/admission/leases/{lease_id}",
                                params={"backend_started": "true" if backend_started else "false"},
                                timeout=settings.host_admission_recovery_timeout_ms / 1000.0)
            if 200 <= resp.status_code < 300:
                return True
            raise _ControllerHTTPError(resp.status_code)
        except Exception as exc:  # noqa: BLE001 — confiée au reaper, plus jamais avalée
            counters.inc("host_admission_release_failed_total")
            logger.warning("host_admission: libération non confirmée, confiée au reaper : %s", type(exc).__name__)
            return False

    # ------------------------------------------------------------ public
    def acquire(self, workload_class: str, model_logical: str, request_id: Optional[str] = None,
                timeout_seconds: Optional[int] = None) -> Optional[HostLease]:
        """Rend une lease (ou None si la classe n'est pas gouvernée / repli).
        Lève HostAdmissionDeferred / Rejected / Unavailable en mode enforce.
        `timeout_seconds` (lot SEMX-4C) : délai explicite demandé au contrôleur
        (TTL = délai + 10) ; None = appel STRICTEMENT identique à l'historique."""
        if not self.class_enabled(workload_class):
            return None
        idem = uuid.uuid4().hex          # UNE transaction d'admission (identité de lease, de rejeu ET d'abort)
        family = (derive_retry_family(settings.host_admission_instance, workload_class, request_id)
                  if request_id and workload_class in RETRY_FAMILY_CLASSES else None)
        t0 = time.perf_counter()
        try:
            if timeout_seconds is None:
                data = self._reserve(workload_class, model_logical, idem, family)
            else:
                data = self._reserve(workload_class, model_logical, idem, family, timeout_seconds=timeout_seconds)
        except Exception as exc:  # noqa: BLE001 — classé : NOT_SENT vs AMBIGUOUS
            kind = classify_transport_failure(exc)
            counters.inc("host_admission_unavailable_total")
            log_event(logger, logging.WARNING, "host_admission.unavailable", workload_class=workload_class,
                      model_logical=model_logical, request_id=request_id or "-", reason=type(exc).__name__,
                      transport=kind, key=idem[:8])
            if kind == AMBIGUOUS:
                return self._recover_ambiguous(workload_class, idem, timeout_seconds)
            counters.inc(_labeled("reserve_not_sent_total", workload_class))
            if self.shadow:
                return None
            if workload_class in HEAVY_CLASSES:
                raise HostAdmissionUnavailable() from exc
            counters.inc("host_admission_local_fallback_total")
            return None
        decision_ms = round((time.perf_counter() - t0) * 1000, 2)
        outcome, reason = data["outcome"], data.get("reason", "-")
        lease_data = data.get("lease") or {}
        counters.inc(f"host_admission_{outcome.lower()}_total")
        log_event(logger, logging.INFO, "host_admission.decision", outcome=outcome, reason=reason,
                  workload_class=workload_class, model_logical=model_logical, request_id=request_id or "-",
                  shadow=self.shadow, decision_ms=decision_ms, lease=(lease_data.get("lease_id") or "-")[:8],
                  family=(family[:12] if family else "-"))
        if self.shadow:
            # Qualification : décision observée, requête non contrôlée, lease rendue aussitôt.
            # La requête VA s'exécuter : backend_started=True (comptabilité de transition réaliste).
            if outcome == "GRANTED" and lease_data.get("lease_id"):
                self._release_or_reap(lease_data["lease_id"], workload_class, backend_started=True,
                                      timeout_seconds=timeout_seconds)
            return None
        if outcome == "GRANTED":
            return HostLease(lease_id=lease_data["lease_id"], workload_class=workload_class, model_logical=model_logical,
                             timeout_seconds=timeout_seconds)
        if outcome == "DEFERRED":
            raise HostAdmissionDeferred(reason, int(data.get("retry_after_seconds") or 3))
        raise HostAdmissionRejected(reason)

    def acquire_bounded(self, workload_class: str, model_logical: str, request_id: Optional[str] = None,
                        timeout_seconds: Optional[int] = None, wait_seconds: int = 0,
                        deadline: Optional[float] = None):
        """`acquire()` + attente BORNÉE (lot GPU-24A) sur un DEFERRED transitoire : la demande est re-réservée après
        `retry_after_seconds` (nouvelle transaction, MÊME famille de retry => même ancienneté au contrôleur) tant que
        le budget `wait_seconds` et l'échéance murale `deadline` (marge 1 s) le permettent. UN seul attenteur par
        classe et par processus : un second appelant reçoit le DEFERRED immédiatement. Rend
        (lease | None, waited_ms, attentes). `wait_seconds` 0, classe non gouvernée ou mode shadow : strictement
        `acquire()` (appel historique)."""
        if wait_seconds <= 0 or not self.class_enabled(workload_class) or self.shadow:
            return self.acquire(workload_class, model_logical, request_id, timeout_seconds), 0, 0
        waiter = self._waiters.setdefault(workload_class, threading.BoundedSemaphore(1))
        if not waiter.acquire(blocking=False):
            counters.inc(_labeled("host_admission_wait_refused_total", workload_class))
            return self.acquire(workload_class, model_logical, request_id, timeout_seconds), 0, 0
        waited_ms, waits = 0, 0
        try:
            while True:
                try:
                    lease = self.acquire(workload_class, model_logical, request_id, timeout_seconds)
                except HostAdmissionDeferred as exc:
                    pause = float(exc.retry_after_seconds)
                    budget_left = float(wait_seconds) - waited_ms / 1000.0
                    wall_left = (deadline - self._time()) if deadline is not None else float("inf")
                    if exc.reason not in WAITABLE_REASONS or pause > budget_left or pause >= wall_left - 1.0:
                        if waits:
                            counters.inc(_labeled("host_admission_wait_exhausted_total", workload_class))
                            log_event(logger, logging.WARNING, "host_admission.wait", outcome="exhausted",
                                      reason=exc.reason, workload_class=workload_class, waits=waits,
                                      waited_ms=waited_ms, request_id=request_id or "-")
                        raise
                    counters.inc(_labeled("host_admission_wait_total", workload_class))
                    self._sleep(pause)
                    waited_ms += int(pause * 1000)
                    waits += 1
                    continue
                if waits:
                    counters.inc(_labeled("host_admission_wait_recovered_total", workload_class))
                    log_event(logger, logging.INFO, "host_admission.wait", outcome="granted",
                              workload_class=workload_class, waits=waits, waited_ms=waited_ms,
                              request_id=request_id or "-")
                return lease, waited_ms, waits
        finally:
            waiter.release()

    def _recover_ambiguous(self, workload_class: str, idem: str,
                           timeout_seconds: Optional[int] = None) -> Optional[HostLease]:
        """Issue inconnue : abort par clé AVANT toute conclusion. Lourd : jamais de
        backend (503). Léger : repli local SEULEMENT si l'abort est confirmé.
        Shadow : une tentative, puis le reaper ; la requête n'est jamais contrôlée."""
        counters.inc(_labeled("reserve_transport_timeout_total", workload_class))
        counters.inc(_labeled("admission_unknown_outcome_total", workload_class))
        attempts = 1 if self.shadow else settings.host_admission_recovery_attempts
        outcome = self._abort_inline(workload_class, idem, attempts)
        if outcome is None:
            counters.inc(_labeled("reserve_recovery_failure_total", workload_class))
            self._enqueue("abort", idem, workload_class, timeout_seconds=timeout_seconds)
        if self.shadow:
            return None
        if outcome is not None and workload_class not in HEAVY_CLASSES:
            counters.inc("host_admission_local_fallback_total")   # aucune lease ne subsiste : repli sûr
            return None
        raise HostAdmissionUnavailable()

    def _abort_inline(self, workload_class: str, idem: str, attempts: int) -> Optional[str]:
        for i in range(max(1, attempts)):
            if i:
                self._sleep(_INLINE_BACKOFF_SECONDS[min(i, len(_INLINE_BACKOFF_SECONDS) - 1)])
            counters.inc(_labeled("reserve_recovery_attempt_total", workload_class))
            try:
                outcome = self._abort(idem)
            except Exception as exc:  # noqa: BLE001
                log_event(logger, logging.WARNING, "host_admission.recovery", stage="abort", outcome="unconfirmed",
                          workload_class=workload_class, key=idem[:8], reason=type(exc).__name__)
                continue
            counters.inc(f"reserve_recovery_success_total{{class={workload_class}}}{{outcome={outcome}}}")
            log_event(logger, logging.INFO, "host_admission.recovery", stage="abort", outcome=outcome,
                      workload_class=workload_class, key=idem[:8])
            return outcome
        return None

    def mark_started(self, lease: Optional[HostLease]) -> None:
        """À appeler IMMÉDIATEMENT avant d'engager le backend (None-safe)."""
        if lease is not None:
            lease.backend_started = True

    def release(self, lease: Optional[HostLease], backend_started: Optional[bool] = None) -> None:
        """Idempotent et None-safe : appelé dans TOUS les `finally` de route. Une
        libération non confirmée est confiée au reaper (plus jamais avalée)."""
        if lease is None or lease.released:
            return
        lease.released = True
        started = lease.backend_started if backend_started is None else backend_started
        self._release_or_reap(lease.lease_id, lease.workload_class, backend_started=started,
                              timeout_seconds=lease.timeout_seconds)

    def _release_or_reap(self, lease_id: str, workload_class: str, backend_started: bool,
                         timeout_seconds: Optional[int] = None) -> None:
        if not self._release_remote(lease_id, backend_started):
            self._enqueue("release", lease_id, workload_class, backend_started, timeout_seconds=timeout_seconds)

    # ------------------------------------------------------------- reaper
    def reaper_pending(self) -> int:
        with self._lock:
            return len(self._reaper_items)

    def _enqueue(self, kind: str, ident: str, workload_class: str, backend_started: bool = False,
                 timeout_seconds: Optional[int] = None) -> None:
        deadline = time.monotonic() + lease_ttl_bound(workload_class, timeout_seconds) + REAPER_MARGIN_SECONDS
        with self._lock:
            key = (kind, ident)
            if key not in self._reaper_items:
                gauges.inc("host_admission_reaper_pending")
            self._reaper_items[key] = {"kind": kind, "ident": ident, "class": workload_class,
                                       "backend_started": backend_started, "deadline": deadline}
            while len(self._reaper_items) > settings.host_admission_reaper_max_items:
                self._reaper_items.popitem(last=False)
                gauges.dec("host_admission_reaper_pending")
                counters.inc("host_admission_reaper_dropped_total")
            start = self.autostart_reaper and (self._reaper_thread is None or not self._reaper_thread.is_alive())
            if start:
                self._reaper_thread = threading.Thread(target=self._reaper_loop, name="host-admission-reaper",
                                                       daemon=True)
                self._reaper_thread.start()

    def run_reaper_once(self, now: Optional[float] = None) -> int:
        """Un passage : retente chaque abort/libération non confirmé ; retire ce
        qui est confirmé ou échu (TTL de lease dépassée). Rend le reste."""
        now = time.monotonic() if now is None else now
        with self._lock:
            items = list(self._reaper_items.items())
        for key, item in items:
            done = False
            if now > item["deadline"]:
                counters.inc("host_admission_reaper_expired_total")
                done = True
            elif item["kind"] == "abort":
                try:
                    outcome = self._abort(item["ident"])
                    counters.inc(f"reserve_recovery_success_total{{class={item['class']}}}{{outcome={outcome}}}")
                    done = True
                except Exception:  # noqa: BLE001 — contrôleur toujours injoignable : on réessaiera
                    pass
            else:
                counters.inc("release_retry_total")
                done = self._release_remote(item["ident"], item["backend_started"])
            if done:
                with self._lock:
                    if self._reaper_items.pop(key, None) is not None:
                        gauges.dec("host_admission_reaper_pending")
        return self.reaper_pending()

    def _reaper_loop(self) -> None:
        interval = settings.host_admission_reaper_interval_ms / 1000.0
        while True:
            self._reaper_wake.wait(interval)       # cadence ~1 s, jamais de boucle active
            self._reaper_wake.clear()
            if self.run_reaper_once() == 0:
                with self._lock:
                    if not self._reaper_items:
                        self._reaper_thread = None
                        return


def host_admission_http_error(exc: HostAdmissionError, request_id: Optional[str] = None):
    """Enveloppe 503 typée, cohérente avec les refus de capacité existants."""
    extra = {"retryable": True}
    if request_id is not None:
        extra["request_id"] = request_id
    if isinstance(exc, HostAdmissionDeferred):
        extra.update(code="host_capacity_deferred", reason=exc.reason, retry_after_seconds=exc.retry_after_seconds)
        return http_error(503, "capacité hôte indisponible pour cette charge, réessayer", provider="ai_gateway",
                          extra=extra, headers={"Retry-After": str(exc.retry_after_seconds)})
    if isinstance(exc, HostAdmissionRejected):
        extra.update(code="host_capacity_rejected", reason=exc.reason, retryable=False)
        return http_error(503, "charge non admissible par le pool hôte", provider="ai_gateway", extra=extra)
    extra.update(code="host_admission_unavailable", reason="controller_unavailable",
                 retry_after_seconds=UNAVAILABLE_RETRY_AFTER_SECONDS)
    return http_error(503, "contrôleur d'admission hôte injoignable", provider="ai_gateway", extra=extra,
                      headers={"Retry-After": str(UNAVAILABLE_RETRY_AFTER_SECONDS)})


host_admission = HostAdmissionClient()
