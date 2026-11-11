"""Lecture de la configuration depuis l'environnement — profil Open.

Aucune valeur sensible n'est journalisée. La base URL Ollama est masquée
(`redacted`) dans les réponses exposées par la Gateway.

Doctrine Open : configuration FAIL-CLOSED. Une variable absente
vaut le défaut autonome documenté ; une variable illisible ou hors domaine
empêche le démarrage ; une capacité dont le composant n'est pas inclus dans
cette distribution ne peut pas être activée (refus explicite au démarrage).
Les défauts ne portent aucun nom de machine ni adresse privée : seuls des
hôtes de boucle locale apparaissent.
"""

import os
import re

# Provenance (lot build-provenance-observability) : formats stricts, tout le
# reste est traité comme « non rapporté ». On n'expose JAMAIS une valeur
# approximative — un SHA tronqué ou un digest mal formé ne prouve rien.
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _get_validated(name: str, pattern: "re.Pattern[str]") -> "str | None":
    """Valeur d'environnement validée par regex, sinon None (= non rapportée).

    None plutôt qu'une chaîne vide ou la valeur brute : la surface HTTP rend
    `not_reported` et personne ne peut confondre une provenance invalide avec
    une provenance attestée.
    """
    raw = (os.getenv(name) or "").strip().lower()
    if raw and pattern.match(raw):
        return raw
    return None


class ConfigurationError(RuntimeError):
    """Configuration invalide connue au démarrage.

    Levée par `Settings()`, donc à l'import de `app.settings` : uvicorn ne
    démarre pas. Avant, `PORT=abc` était avalé (défaut silencieux),
    `TIMEOUT_SECONDS=0` passait et explosait dans `requests` à la première
    requête, `MAX_BATCH=-3` rendait 100 % de 400. Le message nomme TOUJOURS la
    variable fautive. Une variable absente ou vide vaut « défaut » ; une
    variable explicitement illisible ou hors domaine vaut « refus ».
    """


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _get_int(
    name: str,
    default: int,
    *,
    minimum: "int | None" = None,
    maximum: "int | None" = None,
    allowed: "tuple[int, ...] | None" = None,
) -> int:
    """Entier d'environnement, validé au démarrage.

    Absent/vide -> défaut. Illisible, hors [minimum, maximum] ou hors
    `allowed` -> ConfigurationError nommant la variable. Aucune valeur
    explicite n'est jamais remplacée en silence.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    text = raw.strip()
    try:
        value = int(text)
    except ValueError:
        raise ConfigurationError(
            f"configuration invalide : {name} doit être un entier (reçu {text!r})"
        ) from None
    if allowed is not None and value not in allowed:
        raise ConfigurationError(
            f"configuration invalide : {name} doit valoir l'une de "
            f"{', '.join(str(a) for a in allowed)} (reçu {value})"
        )
    if minimum is not None and value < minimum:
        raise ConfigurationError(
            f"configuration invalide : {name} doit être >= {minimum} (reçu {value})"
        )
    if maximum is not None and value > maximum:
        raise ConfigurationError(
            f"configuration invalide : {name} doit être <= {maximum} (reçu {value})"
        )
    return value


def _get_bool(name: str, default: bool) -> bool:
    """Drapeau d'environnement. Convention inchangée (1/true/yes/on,
    0/false/no/off, insensible à la casse) ; une valeur hors convention ne
    retombe plus sur le défaut : elle empêche le démarrage."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    text = raw.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigurationError(
        f"configuration invalide : {name} doit être un booléen "
        f"(1/0, true/false, yes/no, on/off ; reçu {raw.strip()!r})"
    )


# Drapeaux d'activation de composants EXCLUS de la distribution Open :
# extraction structurée PP-StructureV3 (structx/), moteur V2-H et son shadow,
# contrat vision sémantique v3. Vrai => refus au démarrage (voir Settings).
_EXCLUDED_CAPABILITY_FLAGS = (
    "AI_GATEWAY_STRUCTURED_EXTRACT_ENABLED",
    "AI_GATEWAY_STRUCTURED_EXTRACT_PRODUCTION_ALLOWED",
    "AI_GATEWAY_V2H_ENABLED",
    "AI_GATEWAY_V2H_SHADOW_ENABLED",
    "AI_GATEWAY_VISION_SEMANTIC_ENABLED",
)


class Settings:
    def __init__(self) -> None:
        # Réseau Gateway
        self.host: str = os.getenv("AI_GATEWAY_HOST", "0.0.0.0")
        # Bornes : 1..65535 pour un port ; un timeout ou une taille
        # de lot à 0 n'a aucun sens et cassait à la première requête.
        self.port: int = _get_int("AI_GATEWAY_PORT", 8100, minimum=1, maximum=65535)
        self.timeout_seconds: int = _get_int("AI_GATEWAY_TIMEOUT_SECONDS", 60, minimum=1)

        # Upstream Ollama
        self.ollama_base_url: str = os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        ).rstrip("/")

        # Modèles par défaut — noms LOGIQUES du registre Compact
        # (app/models_registry.yaml). Compatibilité ascendante : OLLAMA_MODEL
        # est accepté comme fallback pour OLLAMA_GENERATE_MODEL.
        self.generate_model: str = (
            os.getenv("OLLAMA_GENERATE_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or "compact-generation"
        )
        self.embedding_model: str = os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "fr-en-retrieval"
        )

        # Embeddings
        self.embeddings_dimension: int = _get_int(
            "AI_GATEWAY_EMBEDDINGS_DIMENSION", 768, minimum=1
        )
        self.embeddings_max_batch: int = _get_int(
            "AI_GATEWAY_EMBEDDINGS_MAX_BATCH", 64, minimum=1
        )

        # Limites de ressources et de concurrence EXPLICITES.
        # Acquisition non bloquante : à capacité pleine, 503 + Retry-After,
        # jamais une file dans un thread partagé (doctrine P8). Un runtime
        # Ollama local sérialise lui-même ses générations : la borne Gateway
        # protège le processus et rend la saturation OBSERVABLE au client.
        self.generate_max_concurrency: int = _get_int(
            "AI_GATEWAY_GENERATE_MAX_CONCURRENCY", 2, minimum=1, maximum=64
        )
        self.embeddings_max_concurrency: int = _get_int(
            "AI_GATEWAY_EMBEDDINGS_MAX_CONCURRENCY", 2, minimum=1, maximum=64
        )
        # Taille maximale d'un corps de requête JSON (octets). Les routes
        # multipart ASR portent leurs propres bornes (max_upload_bytes).
        self.request_max_bytes: int = _get_int(
            "AI_GATEWAY_REQUEST_MAX_BYTES", 2 * 1024 * 1024, minimum=64 * 1024
        )

        # Registre de modèles (lot 1). Le fichier vit dans le code, pas dans
        # l'environnement : seul son chemin est surchargeable, pour les tests.
        self.models_registry_path: str = os.getenv(
            "AI_GATEWAY_MODELS_REGISTRY",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_registry.yaml"),
        )

        # Registres STRICTS par défaut (profil Open, fail-closed) : un modèle
        # absent du registre est refusé en 400 `model_not_registered` AVANT
        # tout appel Ollama. À 0, le relais historique « legacy non gouverné »
        # (compté `*_legacy_model_total`, WARNING) est rétabli — réglage de
        # compatibilité, jamais le défaut d'une installation Open.
        self.embeddings_strict_registry: bool = _get_bool(
            "AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY", True
        )
        self.generation_strict_registry: bool = _get_bool(
            "AI_GATEWAY_GENERATION_STRICT_REGISTRY", True
        )

        # Environnement logique, confronté aux drapeaux production_allowed /
        # staging_allowed du registre. Défaut prudent : "production".
        self.environment: str = os.getenv("AI_GATEWAY_ENVIRONMENT", "production").strip().lower()

        # Readiness : sonde réelle, mise en cache pour ne pas encoder un texte
        # à chaque battement du healthcheck Docker (30 s).
        # Cache 0 = pas de cache (légitime) ; timeout de sonde >= 1 s.
        self.readiness_cache_seconds: int = _get_int("AI_GATEWAY_READINESS_CACHE_SECONDS", 60, minimum=0)
        self.readiness_timeout_seconds: int = _get_int("AI_GATEWAY_READINESS_TIMEOUT_SECONDS", 5, minimum=1)

        # Fréquence de revérification des digests (hors readiness).
        self.registry_verify_interval_seconds: int = _get_int(
            "AI_GATEWAY_REGISTRY_VERIFY_INTERVAL_SECONDS", 300, minimum=0
        )

        # Reranking (lot rerank-sidecar-staging). Capability OPTIONNELLE,
        # désactivée par défaut : sans configuration explicite, /v1/rerank
        # répond 503 et rien d'autre ne change.
        self.rerank_enabled: bool = _get_bool("AI_GATEWAY_RERANK_ENABLED", False)
        # Backend local dédié (cross-encoder). DOIT rester en loopback : le
        # backend n'a ni authentification ni gouvernance propres — la route
        # refuse de servir si cette URL sort de 127.0.0.1/localhost.
        self.rerank_backend_url: str = os.getenv(
            "AI_GATEWAY_RERANK_BACKEND_URL", "http://127.0.0.1:8300"
        ).rstrip("/")
        # Timeout DÉDIÉ, volontairement découplé du timeout génération : un
        # rerank de 64 paires se joue en centaines de millisecondes, pas en
        # minutes. 15 s couvrent largement un backend en train de se réchauffer.
        self.rerank_timeout_seconds: int = _get_int("AI_GATEWAY_RERANK_TIMEOUT_SECONDS", 15, minimum=1)
        # Modèle logique par défaut (registre reranker_models).
        self.rerank_model: str = os.getenv("AI_GATEWAY_RERANK_MODEL", "rerank-default")
        # Concurrence bornée côté Gateway (sémaphore, pas de scheduler).
        self.rerank_max_concurrency: int = _get_int("AI_GATEWAY_RERANK_MAX_CONCURRENCY", 2, minimum=1)

        # Provenance de build (lot build-provenance-observability).
        # AI_GATEWAY_BUILD_SHA est INJECTÉ AU BUILD de l'image (ARG → ENV du
        # Dockerfile, valeur fournie par la commande de déploiement) — jamais
        # lu d'un `git` runtime ni d'un dépôt monté. Pour une image donnée, la
        # valeur est donc immuable. Absent/invalide => None => `not_reported`.
        self.build_sha: "str | None" = _get_validated(
            "AI_GATEWAY_BUILD_SHA", _GIT_SHA_RE
        )
        # Digest de l'image du backend reranker. Source de vérité UNIQUE :
        # le fichier .env versionné, consommé à la fois par l'interpolation
        # de docker-compose.reranker.yml (ligne image@digest) et par les
        # env_file des deux Gateways — la valeur ne vit qu'à un seul endroit.
        self.rerank_backend_image_digest: "str | None" = _get_validated(
            "AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST", _IMAGE_DIGEST_RE
        )

        # ASR (lot asr-rail-foundation). Capability OPTIONNELLE, désactivée
        # par défaut : sans flag, /v1/audio/transcriptions répond 503
        # asr_unavailable et rien d'autre ne change. Backend dédié sur la
        # boucle locale : hors loopback, la capability est refusée (aucune
        # auth propre côté backend). Version Open : le backend ASR n'est PAS
        # inclus dans cette distribution ; le flag reste accepté pour une
        # installation qui fournit son propre backend compatible.
        self.asr_enabled: bool = _get_bool("AI_GATEWAY_ASR_ENABLED", False)
        self.asr_backend_url: str = os.getenv("AI_GATEWAY_ASR_BACKEND_URL", "http://127.0.0.1:8400")
        # Digest de l'image du backend ASR (même mécanique que le reranker :
        # une seule valeur, dans .env, lue par compose ET par la Gateway).
        self.asr_backend_image_digest: "str | None" = _get_validated(
            "AI_GATEWAY_ASR_BACKEND_IMAGE_DIGEST", _IMAGE_DIGEST_RE
        )
        # Budgets de transport (contrat §36/§37) : appels de contrôle courts,
        # relais d'upload long (256 Mio sur loopback), attente synchrone
        # bornée — au-delà, la requête devient un job différé (202).
        self.asr_control_timeout_seconds: int = _get_int("AI_GATEWAY_ASR_CONTROL_TIMEOUT_SECONDS", 10, minimum=1, maximum=60)
        self.asr_upload_timeout_seconds: int = _get_int("AI_GATEWAY_ASR_UPLOAD_TIMEOUT_SECONDS", 300, minimum=10, maximum=900)
        self.asr_sync_wait_seconds: int = _get_int("AI_GATEWAY_ASR_SYNC_WAIT_SECONDS", 120, minimum=5, maximum=300)
        self.asr_ps_timeout_seconds: int = _get_int("AI_GATEWAY_ASR_PS_TIMEOUT_SECONDS", 2, minimum=1, maximum=10)
        # Admission GPU (contrat §57-§62) : résident Ollama (size_vram) +
        # réserve ASR + base système <= budget. Valeurs QUALIFIÉES par la recon
        # (4,5 Gio mesurés en séquentiel, 2,1 Gio de base hors Ollama). Le
        # budget total (22 Gio) laisse ~2 Gio de marge sur une RTX 24 Gio.
        self.asr_gpu_budget_mib: int = _get_int("AI_GATEWAY_ASR_GPU_BUDGET_MIB", 22528, minimum=1024)
        self.asr_gpu_reserve_mib: int = _get_int("AI_GATEWAY_ASR_GPU_RESERVE_MIB", 4608, minimum=256)
        self.asr_gpu_baseline_mib: int = _get_int("AI_GATEWAY_ASR_GPU_BASELINE_MIB", 2150, minimum=0)
        # Concurrence Gateway : UNE transcription admise à la fois par
        # instance (sémaphore non bloquant, 503 resource_saturated sinon).
        self.asr_max_concurrency: int = _get_int("AI_GATEWAY_ASR_MAX_CONCURRENCY", 1, minimum=1, maximum=1)

        # Admission HÔTE (lot AI-GATEWAY-HOST-WIDE-ADMISSION). Un contrôleur
        # unique sur l'hôte (loopback 8500) arbitre le pool GPU partagé entre
        # production et staging — sans quota ni partition par environnement.
        # TOUT est désactivé par défaut : sans flag, aucun comportement ne
        # change (lot livré inactif ; activation = lot de promotion séparé).
        # `mode` : enforce (la décision contrôle la requête) | shadow (décision
        # journalisée seulement, requête non contrôlée — qualification staging).
        # Les flags par classe permettent un déploiement par phases.
        self.host_admission_enabled: bool = _get_bool("AI_GATEWAY_HOST_ADMISSION_ENABLED", False)
        self.host_admission_mode: str = os.getenv("AI_GATEWAY_HOST_ADMISSION_MODE", "enforce").strip().lower()
        if self.host_admission_mode not in ("enforce", "shadow"):
            raise ConfigurationError("configuration invalide : AI_GATEWAY_HOST_ADMISSION_MODE doit valoir enforce ou shadow")
        self.host_admission_url: str = os.getenv("AI_GATEWAY_HOST_ADMISSION_URL", "http://127.0.0.1:8500").rstrip("/")
        if not (self.host_admission_url.startswith("http://127.0.0.1") or self.host_admission_url.startswith("http://localhost")):
            raise ConfigurationError("configuration invalide : AI_GATEWAY_HOST_ADMISSION_URL doit rester en boucle locale")
        # Délai NORMAL du reserve (lot ORPHAN-LEASE : 250 ms, réglage de performance
        # seulement — la correction repose sur l'abort par clé, pas sur ce délai).
        self.host_admission_timeout_ms: int = _get_int("AI_GATEWAY_HOST_ADMISSION_TIMEOUT_MS", 250, minimum=5, maximum=2000)
        self.host_admission_recovery_timeout_ms: int = _get_int("AI_GATEWAY_HOST_ADMISSION_RECOVERY_TIMEOUT_MS", 250,
                                                                minimum=5, maximum=2000)
        self.host_admission_recovery_attempts: int = _get_int("AI_GATEWAY_HOST_ADMISSION_RECOVERY_ATTEMPTS", 3,
                                                              minimum=1, maximum=5)
        self.host_admission_reaper_interval_ms: int = _get_int("AI_GATEWAY_HOST_ADMISSION_REAPER_INTERVAL_MS", 1000,
                                                               minimum=200, maximum=10000)
        self.host_admission_reaper_max_items: int = _get_int("AI_GATEWAY_HOST_ADMISSION_REAPER_MAX_ITEMS", 4096,
                                                             minimum=16, maximum=65536)
        self.host_admission_vision_enabled: bool = _get_bool("AI_GATEWAY_HOST_ADMISSION_VISION_ENABLED", False)
        self.host_admission_llm_enabled: bool = _get_bool("AI_GATEWAY_HOST_ADMISSION_LLM_ENABLED", False)
        self.host_admission_embedding_enabled: bool = _get_bool("AI_GATEWAY_HOST_ADMISSION_EMBEDDING_ENABLED", False)
        self.host_admission_rerank_enabled: bool = _get_bool("AI_GATEWAY_HOST_ADMISSION_RERANK_ENABLED", False)
        self.host_admission_asr_enabled: bool = _get_bool("AI_GATEWAY_HOST_ADMISSION_ASR_ENABLED", False)
        # Lot AI-GATEWAY-GPU-24A : classe `structx` (worker PP-StructureV3 du backend structx/) gouvernée par le MÊME
        # contrôleur que la Vision => describe et StructX sérialisés par une décision commune (plus deux politiques
        # indépendantes). Défaut OFF : la route structured-extract garde strictement son comportement antérieur.
        self.host_admission_structx_enabled: bool = _get_bool("AI_GATEWAY_HOST_ADMISSION_STRUCTX_ENABLED", False)
        # Version Open : le CONTRÔLEUR d'admission hôte (paquet `admission/`)
        # n'est pas inclus dans cette distribution. Le client reste présent,
        # inerte (désactivé) ; demander son activation est une erreur de
        # configuration explicite — jamais un service fictif, jamais un
        # fail-open vers une admission locale parallèle.
        if self.host_admission_enabled:
            raise ConfigurationError(
                "configuration invalide : AI_GATEWAY_HOST_ADMISSION_ENABLED — le contrôleur "
                "d'admission hôte n'est pas inclus dans cette distribution (capacité UNAVAILABLE)"
            )
        # Identité d'instance transmise au contrôleur (attribution, jamais un droit).
        self.host_admission_instance: str = f"{self.environment}:{os.getenv('AI_GATEWAY_PORT', '8100').strip()}"
        # Taille max de la partie `request` du multipart (contrat §21).
        self.asr_max_request_part_bytes: int = 8 * 1024
        # Le rail ASR est activé par AI_GATEWAY_ASR_ENABLED dans TOUT
        # environnement (lot production-full-capability-enablement) ; le
        # défaut reste OFF (fail-safe) et le registre (production_allowed) +
        # l'identité du backend (/info) restent des verrous indépendants.

        # Vision (lot vision-1a). Capability OPTIONNELLE, désactivée par
        # défaut : sans flag explicite, /v1/vision/analyze répond 503
        # vision_not_enabled et rien d'autre ne change. L'activation est une
        # décision d'infrastructure/environnement, jamais utilisateur.
        self.vision_enabled: bool = _get_bool("AI_GATEWAY_VISION_ENABLED", False)
        # Budget mural TOTAL (admission + attente + inférence + éventuel retry
        # structurel + validation). QUALIFIÉ par MODEL-1 : pire cas mesuré
        # 15,3 s + chargement à froid 10,1 s + un retry, avec marge.
        # Réglage d'EXPLOITATION (§104) : surchargeable dans le domaine
        # qualifié 5..600 s. Lot P14 : hors domaine = refus au démarrage, plus
        # un retour silencieux au défaut (domaine INCHANGÉ).
        self.vision_timeout_seconds: int = _get_int(
            "AI_GATEWAY_VISION_TIMEOUT_SECONDS", 60, minimum=5, maximum=600
        )
        # Concurrence bornée. QUALIFIÉE à 1 : le modèle vision ne peut pas
        # co-résider avec le LLM de chat de production (mesuré). Domaine 1..4.
        self.vision_max_concurrency: int = _get_int(
            "AI_GATEWAY_VISION_MAX_CONCURRENCY", 1, minimum=1, maximum=4
        )
        # Résidence du modèle vision (lot vision-residency-02). Valeur relayée
        # telle quelle à Ollama dans `keep_alive`. DÉFAUT 0 = déchargement
        # immédiat : une installation qui ne configure rien garde EXACTEMENT le
        # comportement qualifié par MODEL-1/1B.
        #
        # ALLOWLIST, pas un intervalle : seules 0, 10 et 30 secondes sont
        # qualifiables (lot RESIDENCY-02 §9). Lot P14 : toute autre valeur
        # y compris -1, 60 ou une chaîne illisible — empêche le démarrage au
        # lieu de retomber sur 0. Une résidence permanente reste inatteignable
        # par configuration ; une faute de frappe est vue, pas masquée.
        self.vision_keep_alive_seconds: int = _get_int(
            "AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS", 0, allowed=(0, 10, 30)
        )
        # GARDE PRODUCTION (§7) : la résidence n'est pas qualifiée en
        # production, et staging/production partagent le MÊME GPU. En
        # production la valeur est forcée à 0, quelle que soit la variable
        # d'environnement — l'erreur de configuration devient inopérante,
        # elle n'est pas seulement déconseillée.
        if self.environment == "production":
            self.vision_keep_alive_seconds = 0
        # Lot AI-GATEWAY-GPU-24A : attente BORNÉE, en Gateway, sur un DEFERRED transitoire de l'admission hôte (worker
        # StructX encore résident <= son TTL, VRAM en cours de libération, job StructX en cours) avant de rendre le 503
        # host_capacity_deferred. La demande est re-réservée après `Retry-After` avec la même famille de retry (même
        # ancienneté au contrôleur). 0 = refus immédiat (comportement antérieur ; production). Plafond 30 s : le délai total
        # de la route (60 s) garde au moins 30 s pour l'inférence. Sans admission hôte, sans effet.
        self.vision_host_admission_wait_seconds: int = _get_int(
            "AI_GATEWAY_VISION_HOST_ADMISSION_WAIT_SECONDS", 0, minimum=0, maximum=30
        )

        # Bornes d'admission image — QUALIFIÉES par MODEL-1 : au-delà de ~2 MP
        # le modèle redimensionne en interne (tokens de prompt constants de
        # 1 à 4 MP), augmenter n'apporte rien et coûte mémoire et latence.
        self.vision_max_image_bytes: int = 4 * 1024 * 1024        # 4 MiB
        # Lot P7 — borne sur la CHAÎNE Base64, contrôlée par Pydantic AVANT
        # tout décodage : Base64 code 3 octets en 4 caractères, padding
        # compris, donc len = 4 * ceil(n / 3). Pour 4 Mio : 5 592 408. Toute
        # chaîne plus longue ne peut pas décoder en <= 4 Mio ; une chaîne à la
        # borne peut décoder jusqu'à 4 Mio + 2 octets, tranchés ensuite en 413.
        self.vision_max_image_base64_chars: int = 4 * ((self.vision_max_image_bytes + 2) // 3)
        self.vision_max_image_pixels: int = 2_000_000             # 2 MP
        # Bornes de contexte (par champ / agrégat) — QUALIFIÉES : couvrent une
        # légende, un texte alternatif et un paragraphe voisin, sans ouvrir la
        # porte à l'injection d'un document entier.
        self.vision_max_context_field_chars: int = 1000
        self.vision_max_context_total_chars: int = 2000

        # Version Open : capacités dont le composant n'est PAS inclus. Toute
        # tentative d'activation par l'environnement est un refus explicite au
        # démarrage (fail-closed) — ces variables n'ont aucun effet « OFF ».
        for name in _EXCLUDED_CAPABILITY_FLAGS:
            if (os.getenv(name) or "").strip().lower() in _TRUE:
                raise ConfigurationError(
                    f"configuration invalide : {name} — composant non inclus dans cette "
                    "distribution (capacité UNAVAILABLE)"
                )

        # Logs
        self.log_level: str = os.getenv("AI_GATEWAY_LOG_LEVEL", "INFO").upper()


settings = Settings()
