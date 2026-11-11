# Configuration — référentiel des variables

Toutes les variables sont lues à l'import de `app.settings` ; une valeur
**absente ou vide vaut le défaut**, une valeur **illisible ou hors domaine
empêche le démarrage** (`ConfigurationError` nommant la variable). Booléens :
`1/0`, `true/false`, `yes/no`, `on/off`. Aucune valeur par défaut ne porte de
nom de machine ni d'adresse privée.

## Réseau et délais

| Variable | Défaut | Domaine | Rôle |
|---|---|---|---|
| `AI_GATEWAY_HOST` | `0.0.0.0` | — | bind dans le conteneur (le port est publié sur `127.0.0.1` par Compose) |
| `AI_GATEWAY_PORT` | `8100` | 1..65535 | port d'écoute |
| `AI_GATEWAY_TIMEOUT_SECONDS` | `60` | ≥ 1 | budget mural des appels Ollama (génération, flux, embeddings) |
| `AI_GATEWAY_LOG_LEVEL` | `INFO` | niveaux `logging` | verbosité ; jamais de contenu dans les journaux |
| `AI_GATEWAY_ENVIRONMENT` | `production` | `production` \| `staging` | confronté aux allowlists `production_allowed` / `staging_allowed` du registre ; toute autre valeur refuse tous les modèles |

## Runtime Ollama et modèles par défaut

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | runtime local (jamais exposé : `redacted`) |
| `OLLAMA_GENERATE_MODEL` (ou `OLLAMA_MODEL`) | `compact-generation` | modèle de génération par défaut (nom logique) |
| `OLLAMA_EMBEDDING_MODEL` | `fr-en-retrieval` | modèle d'embeddings par défaut (nom logique) |
| `AI_GATEWAY_MODELS_REGISTRY` | `app/models_registry.yaml` | chemin du registre (surchargeable, ex. tests) |

## Registres stricts (profil Open : **activés**)

| Variable | Défaut | Rôle |
|---|---|---|
| `AI_GATEWAY_GENERATION_STRICT_REGISTRY` | `1` | modèle de génération hors registre ⇒ `400 model_not_registered` ; `0` rétablit le relais legacy (déconseillé) |
| `AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY` | `1` | idem embeddings ⇒ `400` ; `0` sert en mode non gouverné (`governed: false`) |

Au démarrage, le modèle par défaut de chaque rail doit exister au registre
quand le registre strict est actif.

## Embeddings

| Variable | Défaut | Domaine | Rôle |
|---|---|---|---|
| `AI_GATEWAY_EMBEDDINGS_DIMENSION` | `768` | ≥ 1 | dimension de l'index ; un profil d'une autre dimension est refusé (`400 embedding_dimension_incompatible`) et le défaut doit la respecter au démarrage |
| `AI_GATEWAY_EMBEDDINGS_MAX_BATCH` | `64` | ≥ 1 | textes par requête (le profil peut être plus strict) |

## Limites explicites

| Variable | Défaut | Domaine | Rôle |
|---|---|---|---|
| `AI_GATEWAY_GENERATE_MAX_CONCURRENCY` | `2` | 1..64 | générations + chats simultanés ; au-delà `503 generation_saturated`, `Retry-After: 2` |
| `AI_GATEWAY_EMBEDDINGS_MAX_CONCURRENCY` | `2` | 1..64 | lots d'embeddings simultanés ; au-delà `503 embeddings_saturated`, `Retry-After: 1` |
| `AI_GATEWAY_REQUEST_MAX_BYTES` | `2097152` | ≥ 65536 | `Content-Length` maximal des corps JSON ⇒ `413 request_too_large` ; routes `/v1/audio/*` et `/v1/vision/*` exemptées (bornes propres) |
| `AI_GATEWAY_RERANK_MAX_CONCURRENCY` | `2` | ≥ 1 | reranks simultanés ⇒ `503 rerank_saturated`, `Retry-After: 1` |
| `AI_GATEWAY_VISION_MAX_CONCURRENCY` | `1` | 1..4 | analyses vision simultanées (capacité OFF par défaut) |

## Reranking (backend local TEI)

| Variable | Défaut | Rôle |
|---|---|---|
| `AI_GATEWAY_RERANK_ENABLED` | `0` (`1` dans `.env.example`) | capacité ; OFF ⇒ `503` explicite |
| `AI_GATEWAY_RERANK_BACKEND_URL` | `http://127.0.0.1:8300` | **boucle locale obligatoire** (sinon capacité refusée) |
| `AI_GATEWAY_RERANK_MODEL` | `rerank-default` | profil par défaut |
| `AI_GATEWAY_RERANK_TIMEOUT_SECONDS` | `15` | délai dédié |
| `AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST` | (depuis `.env`, valeur fournie par `.env.example`) | digest `sha256:` de l'image TEI, exposé sur `/health/runtime` ; invalide ⇒ `not_reported` |

## Sondes

| Variable | Défaut | Rôle |
|---|---|---|
| `AI_GATEWAY_READINESS_CACHE_SECONDS` | `60` | cache de `/health/ready` (0 = aucun) |
| `AI_GATEWAY_READINESS_TIMEOUT_SECONDS` | `5` | délai des sondes |
| `AI_GATEWAY_REGISTRY_VERIFY_INTERVAL_SECONDS` | `300` | revérification des digests hors readiness |

## Capacités optionnelles (OFF ; backends non inclus)

| Variable | Défaut | Rôle |
|---|---|---|
| `AI_GATEWAY_ASR_ENABLED` | `0` | rail ASR ; exige un backend compatible sur `AI_GATEWAY_ASR_BACKEND_URL` (boucle locale, défaut `http://127.0.0.1:8400`) et un `asr_models` au registre |
| `AI_GATEWAY_ASR_*` (délais, budgets GPU, concurrence 1) | voir `app/settings.py` | sans effet tant que le rail est OFF |
| `AI_GATEWAY_VISION_ENABLED` | `0` | rail vision ; exige un profil à modalité image et `vision_defaults` au registre |
| `AI_GATEWAY_VISION_TIMEOUT_SECONDS` | `60` (5..600) | budget mural vision |
| `AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS` | `0` (0, 10, 30) | résidence du modèle vision ; forcée à 0 en `production` |
| `AI_GATEWAY_VISION_HOST_ADMISSION_WAIT_SECONDS` | `0` (0..30) | sans effet (contrôleur non inclus) |

## Variables **refusées** (composants non inclus)

Vraies ⇒ échec au démarrage : `AI_GATEWAY_HOST_ADMISSION_ENABLED`,
`AI_GATEWAY_STRUCTURED_EXTRACT_ENABLED`,
`AI_GATEWAY_STRUCTURED_EXTRACT_PRODUCTION_ALLOWED`, `AI_GATEWAY_V2H_ENABLED`,
`AI_GATEWAY_V2H_SHADOW_ENABLED`, `AI_GATEWAY_VISION_SEMANTIC_ENABLED`.

## Provenance

| Variable | Rôle |
|---|---|
| `AI_GATEWAY_BUILD_SHA` | SHA git (40 hex) injecté au build de l'image, exposé `build.git_sha` ; invalide ⇒ `not_reported` |

## Références d'implémentation

`app/settings.py` (`Settings`, `_EXCLUDED_CAPABILITY_FLAGS`), `.env.example`,
`tests/test_settings_validation.py`, `tests/open/test_open_configuration.py`.
