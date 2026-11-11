"""Compteurs internes, journalisation structurée et métriques de performance.

Le dépôt n'embarque pas (encore) de pile de métriques : ce module fournit des
compteurs en mémoire, sûrs vis-à-vis des threads, exposés par
`GET /health/embeddings`, un helper de log structuré `log_event`, et la
normalisation des métriques de performance renvoyées par Ollama (lot 3).

Règle absolue, héritée du contrat de sécurité existant : on journalise des
**faits mesurables** (modèle, tâche, tailles, latences, endpoint) et jamais le
contenu — ni texte, ni prompt, ni vecteur, ni URL upstream. Les normaliseurs
ci-dessous ne lisent que des champs numériques d'une liste fermée : ils ne
peuvent pas recopier un fragment de texte, même si l'upstream en plaçait un
dans sa réponse.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, Optional

# Compteurs suivis. Déclarés explicitement pour que `/health/embeddings`
# retourne une forme stable même quand rien ne s'est encore produit.
_COUNTER_NAMES = (
    "embeddings_requests_total",
    "embeddings_texts_total",
    # Erreurs, classées par responsabilité. `embeddings_errors_total` est
    # conservé comme AGRÉGAT des trois suivants, pour compatibilité avec ce qui
    # aurait déjà été branché dessus ; il ne doit plus servir à alerter, car il
    # mélangeait un 400 « batch trop grand » (faute du client) avec un 502
    # « Ollama est tombé » (incident d'exploitation).
    "embeddings_errors_total",
    "embeddings_client_errors_total",
    "embeddings_upstream_errors_total",
    "embeddings_internal_errors_total",
    # Anomalie upstream qui ne fait PAS échouer la requête : le modèle demandé
    # est décrit par le registre mais son digest a divergé, la réponse est donc
    # servie en mode non gouverné.
    "embeddings_degraded_model_total",
    "embeddings_fallback_total",
    "embeddings_dimension_errors_total",
    "embeddings_invalid_vector_total",
    "embeddings_overflow_total",
    "embeddings_double_prefix_avoided_total",
    "embeddings_legacy_model_total",
    "embeddings_registry_rejected_total",
    "embeddings_fallback_budget_exhausted_total",
    "registry_digest_mismatch_total",
    # Vision (lot vision-1a). Sans labels, comme tout ce module.
    "vision_requests_total",
    "vision_ok_total",
    "vision_client_errors_total",
    "vision_upstream_errors_total",
    "vision_invalid_output_total",
    "vision_timeouts_total",
    "vision_overloaded_total",
    "vision_rejected_oversize_total",
    # Réponses 200 servies après normalisation d'une abstention describe
    # sans warning (lot ABSTENTION-NORMALIZATION-1A) ; incluses aussi dans
    # vision_ok_total.
    "vision_abstention_normalized_total",
    "readiness_probe_total",
    "readiness_probe_failed_total",
    # Génération (lot runtime-hardening). Sans labels, comme tout ce module :
    # le détail par modèle vit dans les lignes de log, pas en dimensions de
    # compteurs — la cardinalité reste nulle par construction.
    "generate_requests_total",
    "generate_upstream_errors_total",
    "generate_stream_requests_total",
    "generate_stream_errors_total",
    "generate_stream_timeouts_total",
    "generate_rejected_environment_total",
    "generate_rejected_num_ctx_total",
    # Lot C2 : modèle enregistré sans capability `completion` demandé sur
    # /v1/generate (refus avant Ollama).
    "generate_rejected_capability_total",
    # modèle demandé absent du runtime local (503 model_not_installed,
    # aucun téléchargement implicite) et saturation de la concurrence bornée.
    "generate_model_not_installed_total",
    "generate_saturated_total",
    "chat_model_not_installed_total",
    "embeddings_model_not_installed_total",
    "embeddings_saturated_total",
    "embeddings_dimension_incompatible_total",
    # Sortie structurée gouvernée sur le rail completion (lot G2). Mêmes
    # familles que chat_structured_*, compteurs DISTINCTS.
    "generate_structured_requests_total",
    "generate_structured_ok_total",
    "generate_structured_capability_rejected_total",
    "generate_structured_reasoning_rejected_total",
    "generate_structured_schema_rejected_total",
    "generate_structured_conflict_rejected_total",
    "generate_structured_streaming_rejected_total",
    "generate_structured_invalid_upstream_total",
    "generate_structured_empty_total",
    "generate_structured_invalid_json_total",
    "generate_structured_schema_mismatch_total",
    # Reranking (lot rerank-sidecar-staging). Même philosophie sans labels ;
    # le détail par requête vit dans l'événement `rerank.ok`.
    "rerank_requests_total",
    "rerank_pairs_total",
    "rerank_client_errors_total",
    "rerank_upstream_errors_total",
    "rerank_internal_errors_total",
    "rerank_rejected_oversize_total",
    # Chat (lot C1). Sans labels, comme le reste : le mode de raisonnement et
    # le modèle vivent dans les lignes de log (`chat.ok`), jamais en
    # dimensions de compteur.
    "chat_requests_total",
    "chat_upstream_errors_total",
    "chat_timeouts_total",
    # Réponse 200 dont le contenu assistant est vide (typiquement : budget
    # `num_predict` consommé par la réflexion, done_reason=length). Résultat
    # valide du backend, incomplet fonctionnellement — compté, jamais maquillé.
    "chat_empty_content_total",
    "chat_unexpected_tool_calls_total",
    "chat_rejected_unregistered_total",
    "chat_rejected_capability_total",
    "chat_rejected_environment_total",
    "chat_rejected_keep_alive_total",
    "chat_rejected_reasoning_total",
    "chat_stream_requests_total",
    "chat_stream_errors_total",
    "chat_stream_timeouts_total",
    # Sorties structurées (lot C3). Sans labels : le code de refus vit dans
    # l'événement journalisé (vocabulaire fermé), jamais en dimension.
    "chat_structured_requests_total",
    "chat_structured_ok_total",
    "chat_structured_capability_rejected_total",
    "chat_structured_schema_rejected_total",
    "chat_structured_streaming_rejected_total",
    # Réponse upstream hors contrat (agrégat) et sa ventilation.
    "chat_structured_invalid_upstream_total",
    "chat_structured_empty_total",
    "chat_structured_invalid_json_total",
    "chat_structured_schema_mismatch_total",
    # Rail ASR (lot asr-rail-foundation, déclarés par asr-observability-
    # hardening / AK-1) : présents à zéro dès le démarrage, sans attendre un
    # premier incrément. Sans labels : job_id, modèle et code vivent dans les
    # événements `asr.*`. Les compteurs du BACKEND (worker_exited_idle_total,
    # recovery_dropped_total, jobs_*_total) se lisent sur son /health.
    "asr_requests_total",
    "asr_accepted_total",
    "asr_rejected_total",
    "asr_saturated_total",
    "asr_completed_total",
    "asr_failed_total",
    "asr_cancelled_total",
    "asr_expired_total",
    "asr_timeouts_total",
    "asr_gpu_contention_total",
    "asr_backend_errors_total",
    "asr_audio_seconds_total",
    "asr_processing_ms_total",
    # Extraction structurée chart/table (lot PPSTRUCTUREV3-STAGING-INTEGRATION-1).
    # Sans labels : task et code vivent dans les événements `structured_extract.*`.
    # Les compteurs du BACKEND (worker, gpu_busy, timeouts) se lisent sur son /health.
    "structured_extract_requests_total",
    "structured_extract_ok_total",
    "structured_extract_rejected_total",
    "structured_extract_overloaded_total",
    "structured_extract_gpu_busy_total",
    "structured_extract_timeouts_total",
    "structured_extract_backend_errors_total",
    # Attente bornée sur refus de capacité transitoire (lot AI-GATEWAY-VISUAL-22A).
    "structured_extract_capacity_waits_total",
    "structured_extract_capacity_recovered_total",
    # Lot GPU-24A : refus (final) par l'admission hôte, classe structx (503 host_capacity_deferred/rejected/unavailable).
    "structured_extract_host_deferred_total",
)


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, int] = {name: 0 for name in _COUNTER_NAMES}

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0) + amount

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        """Réservé aux tests."""
        with self._lock:
            self._values = {name: 0 for name in _COUNTER_NAMES}


counters = Counters()


class Gauges:
    """Jauges en mémoire, thread-safe — l'inflight par modèle notamment.

    Même philosophie que `Counters` : pas de backend, pas de labels de
    cardinalité libre. Les clés sont `<endpoint>:<modèle>`, bornées par le
    nombre de modèles installés. `dec` plancher à 0 : une jauge négative
    signalerait un bug d'appariement inc/dec, pas une réalité.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, int] = {}

    def inc(self, name: str) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0) + 1

    def dec(self, name: str) -> None:
        with self._lock:
            self._values[name] = max(0, self._values.get(name, 0) - 1)

    def snapshot(self) -> Dict[str, int]:
        """Jauges non nulles uniquement : la forme reste petite et lisible."""
        with self._lock:
            return {k: v for k, v in self._values.items() if v > 0}

    def total(self) -> int:
        with self._lock:
            return sum(v for v in self._values.values() if v > 0)

    def reset(self) -> None:
        """Réservé aux tests."""
        with self._lock:
            self._values = {}


gauges = Gauges()

# Contrat `inflight` de /health/runtime (lot RUNTIME-INFLIGHT) : requêtes
# MÉTIER/modèle en cours de service, clés `<route>:<modèle>`. Filtre POSITIF :
# seules les jauges enregistrées ici comptent — toute jauge interne (file de
# récupération, maintenance) est exclue par construction, y compris les futures.
BUSINESS_INFLIGHT_PREFIXES = (
    "generate:", "generate_stream:", "chat:", "chat_stream:",
    "embeddings:", "rerank:", "vision:", "asr:",
)
# File de RÉCUPÉRATION interne du client d'admission hôte (aborts et libérations
# non confirmés) : observable à part, jamais comptée comme requête en cours.
HOST_ADMISSION_REAPER_PENDING_GAUGE = "host_admission_reaper_pending"


def is_business_inflight_gauge(name: str) -> bool:
    return name.startswith(BUSINESS_INFLIGHT_PREFIXES)


def business_inflight(snapshot: Dict[str, int]) -> Dict[str, int]:
    """Sous-ensemble métier d'un instantané de jauges (`inflight.by_model`)."""
    return {k: v for k, v in snapshot.items() if is_business_inflight_gauge(k)}


# --------------------------------------------------------------------------- #
# Dérivation cold/warm (lot runtime-hardening)
# --------------------------------------------------------------------------- #
# Seuil séparant les deux populations OBSERVÉES sur l'installation d'origine : un modèle
# résident rapporte un `load_duration` résiduel de 15 à 210 ms, un chargement
# ou rechargement réel coûte 1,8 à 18 s. 1000 ms passe largement entre les
# deux. Le label est une COMMODITÉ de lecture : il ne prétend pas distinguer
# cold initial / éviction VRAM / reload num_ctx (donnée insuffisante) — les
# valeurs brutes restent journalisées pour l'analyse fine.
LOAD_STATE_WARM_THRESHOLD_MS = 1000.0

LOAD_STATE_WARM = "warm"
LOAD_STATE_COLD_OR_RELOAD = "cold_or_reload"


def load_state(ollama_load_ms: Any) -> Optional[str]:
    """Label warm/cold_or_reload depuis `ollama_load_ms`, ou None si absent."""
    value = _finite_number(ollama_load_ms)
    if value is None:
        return None
    return (
        LOAD_STATE_WARM
        if value <= LOAD_STATE_WARM_THRESHOLD_MS
        else LOAD_STATE_COLD_OR_RELOAD
    )

# Classes d'erreur exposées par `count_error`.
ERROR_CLIENT = "client"
ERROR_UPSTREAM = "upstream"
ERROR_INTERNAL = "internal"


def count_error(kind: str) -> None:
    """Incrémente le compteur de la classe ET l'agrégat historique.

    client   – la requête est fautive (400) : batch trop grand, texte vide ou
               trop long, `task` invalide, modèle interdit ou non gouverné en
               mode strict.
    upstream – Ollama ou sa réponse sont en cause (502) : indisponibilité,
               timeout, cardinalité, dimension, valeur non finie, vecteur nul,
               norme hors tolérance, budget de repli épuisé.
    internal – bug de la Gateway (500) : exception inattendue.
    """
    counters.inc(f"embeddings_{kind}_errors_total")
    counters.inc("embeddings_errors_total")


def _render_log_value(value: Any) -> str:
    """Rendu d'une valeur de champ : inchangé pour un token simple, cité sinon.

    Lot P13 — UNE invocation de `log_event` = UNE ligne. Une valeur portant
    un saut de ligne fabriquait une seconde ligne `event=…` (forgeage,
    démontré via `model`). Un token simple (nombre, nom, identifiant) garde
    exactement son rendu historique, donc les greps et parseurs
    `clé=valeur` existants ne voient aucune différence ; tout ce qui contient
    un blanc, un caractère de contrôle ou est vide passe par `repr`, qui
    tient sur une ligne et rend l'anomalie visible plutôt que masquée.
    """
    text = str(value)
    if text and all(not ch.isspace() and ch.isprintable() for ch in text):
        return text
    return repr(text)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Émet une ligne `event=... clé=valeur` triée, donc testable.

    Aucun champ de contenu n'est accepté par convention : les appelants ne
    passent que des identifiants, des tailles et des durées. Quoi qu'il
    arrive, la sortie tient sur UNE ligne (cf. `_render_log_value`).
    """
    parts = " ".join(
        f"{key}={_render_log_value(value)}" for key, value in sorted(fields.items())
    )
    logger.log(level, "event=%s %s", event, parts)


# --------------------------------------------------------------------------- #
# Métriques de performance upstream (lot 3)
# --------------------------------------------------------------------------- #
# Ollama exprime ses durées en NANOSECONDES ; la Gateway journalise des
# MILLISECONDES. La conversion est centralisée ici pour qu'il n'existe qu'un
# seul endroit où l'unité change — et un seul à relire le jour où l'endpoint de
# streaming réutilisera ces normaliseurs sur le dernier événement d'un flux
# (les statistiques y portent exactement les mêmes noms de champs).
#
# Invariant : une métrique d'observabilité ne doit JAMAIS faire échouer une
# réponse par ailleurs valide. Toutes les fonctions ci-dessous sont totales —
# elles retournent None (ou un dict vide) plutôt que de lever, quelle que soit
# la forme des données reçues d'un runtime Ollama plus ancien ou plus récent.
NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000


def _finite_number(value: Any) -> Optional[float]:
    """Convertit en float exploitable, ou None.

    Rejette : None, les booléens (`isinstance(True, int)` vaut True en Python),
    les chaînes et objets divers, NaN/Infinity, et les valeurs négatives — une
    durée ou un compte de tokens négatif est une donnée corrompue, pas une
    mesure.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def safe_int(value: Any) -> Optional[int]:
    """Compte entier exploitable (tokens, `num_ctx`), ou None."""
    number = _finite_number(value)
    return None if number is None else int(number)


def duration_ms(nanoseconds: Any) -> Optional[float]:
    """Durée Ollama (ns) convertie en millisecondes, ou None si inexploitable."""
    number = _finite_number(nanoseconds)
    return None if number is None else round(number / NS_PER_MS, 3)


def rate_per_second(count: Any, duration_ns: Any) -> Optional[float]:
    """Débit `count / durée`, la durée étant en nanosecondes.

    Retourne None si l'un des deux termes manque ou si la durée est nulle : on
    préfère l'absence de métrique à un `Infinity` qui polluerait les logs.
    """
    items = _finite_number(count)
    elapsed_ns = _finite_number(duration_ns)
    if items is None or elapsed_ns is None or elapsed_ns <= 0:
        return None
    return round(items / (elapsed_ns / NS_PER_SECOND), 2)


def normalize_generation_metrics(raw: Any) -> Dict[str, Any]:
    """Métriques de génération, depuis la réponse brute d'Ollama.

    `raw` est le corps JSON de `/api/generate` en mode non-stream — et, plus
    tard, le dernier événement d'un flux : Ollama y place les mêmes champs.
    C'est la raison d'être de cette fonction, volontairement ignorante du
    transport.

    Seules les clés présentes ET exploitables figurent dans le résultat : une
    métrique absente est omise, jamais rendue comme `None` dans la ligne de log.
    Un champ non renvoyé par un runtime plus ancien disparaît donc simplement.
    """
    if not isinstance(raw, dict):
        return {}
    metrics = {
        "ollama_total_ms": duration_ms(raw.get("total_duration")),
        "ollama_load_ms": duration_ms(raw.get("load_duration")),
        "ollama_prompt_eval_ms": duration_ms(raw.get("prompt_eval_duration")),
        "ollama_eval_ms": duration_ms(raw.get("eval_duration")),
        "prompt_tokens": safe_int(raw.get("prompt_eval_count")),
        "completion_tokens": safe_int(raw.get("eval_count")),
        "prompt_tokens_per_second": rate_per_second(
            raw.get("prompt_eval_count"), raw.get("prompt_eval_duration")
        ),
        "completion_tokens_per_second": rate_per_second(
            raw.get("eval_count"), raw.get("eval_duration")
        ),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def normalize_embedding_metrics(raw: Any) -> Dict[str, Any]:
    """Métriques d'embedding, depuis les métadonnées brutes d'`/api/embed`.

    Le chemin de repli `/api/embeddings` ne fournit aucune de ces valeurs : il
    passe alors un dict vide et le résultat l'est aussi. On ne reconstitue rien
    par sommation des appels unitaires — ce serait une mesure inventée.
    """
    if not isinstance(raw, dict):
        return {}
    metrics = {
        "ollama_total_ms": duration_ms(raw.get("total_duration")),
        "ollama_load_ms": duration_ms(raw.get("load_duration")),
        "ollama_prompt_eval_tokens": safe_int(raw.get("prompt_eval_count")),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def embedding_throughput(n_texts: Any, upstream_ms: Any) -> Dict[str, Any]:
    """Débits dérivés de la mesure MURALE Gateway, pas des durées Ollama.

    `ms_per_text` reste calculable quand la mesure murale est nulle (upstream
    simulé, lot très rapide) ; `texts_per_second` ne l'est pas et est alors
    omis plutôt que porté à l'infini.
    """
    count = _finite_number(n_texts)
    elapsed_ms = _finite_number(upstream_ms)
    if not count or elapsed_ms is None:
        return {}
    metrics: Dict[str, Any] = {"ms_per_text": round(elapsed_ms / count, 3)}
    if elapsed_ms > 0:
        metrics["texts_per_second"] = round(count / (elapsed_ms / 1000.0), 2)
    return metrics
