# gateway-api — journal du contrat

## 1.0.0-candidate.2 — 2026-11-09

`CONTRACT_STATUS = CANDIDATE_FOR_RATIFICATION` — toujours **non ratifié**.
Empreinte SHA-256 du fichier : `b4f80689c5be8f10dfdc7cf9b08144f5101fe6cf071e20b9f83070ae7f1e98c7`
(candidate.1 : `2d0f23409f91bb3297330793652b9d4cf6ac1cc42e1576f636cec1b9ab46b179`).

Régénération **sans changement fonctionnel** après la mise à jour des dépendances de
sécurité de la Gateway (FastAPI 0.115.6 → 0.162.2, Starlette 0.41.3 → 1.10.0,
pydantic 2.10.4 → 2.14.3, Pillow, python-multipart, requests, certifi — 71 avis de
vulnérabilité fermés). Les 19 routes, les
schémas de requête/réponse, les codes d'erreur, les en-têtes et les événements NDJSON
sont **identiques**. Quinze différences de rendu, toutes additives et sans effet sur la
validation (le générateur de schémas de pydantic 2.14 les explicite) :

- `additionalProperties: true` posé explicitement sur treize schémas d'objets libres
  (`options`, `format`, `structured_output`, `payload` des réponses vision, corps des
  sondes `/health*` et de `/v1/models`) — en OpenAPI, l'absence de la clé avait déjà
  cette signification ;
- `ValidationError` (erreur 422 standard de FastAPI) décrit deux champs
  supplémentaires `ctx` et `input`, présents dans les réponses depuis pydantic v2.

Impact : aucun pour RAGatm Open (ses schémas de requête sont les siens) ni pour
AI Gateway Pool Open (son instantané `openapi-routes.json` ne porte que les routes,
inchangées ; provenance mise à jour). Décision à confirmer par le responsable du
programme dans le dossier de ratification (dossier de qualification interne).

## 1.0.0-candidate.1 — 2026-11-09

`CONTRACT_STATUS = CANDIDATE_FOR_RATIFICATION` — ratification attendue par
Aucun contrat n'est déclaré ratifié par ce lot.

Dérivé de l'API réellement servie par AI Gateway Open (`tools/build_openapi.py`
depuis `app.openapi()`), 19 routes, 45 schémas.

### Routes obligatoires (profil Compact)

`GET /health/live`, `GET /health/runtime`, `GET /v1/models`,
`POST /v1/generate`, `POST /v1/generate/stream`, `POST /v1/embeddings`,
`POST /v1/rerank`.

### Conservé inchangé par rapport à l'API d'origine (21 routes historiques)

- Enveloppe d'erreur `{"detail": {status, provider, message, code?, retryable?}}`.
- Événements NDJSON de `/v1/generate/stream` : `delta`, `done`, `error`
  (et `thinking`/`content`/`done`/`error` pour `/v1/chat/stream`) ; média
  `application/x-ndjson` — ce n'est **pas** du SSE.
- Schémas de requête/réponse de generate, chat, embeddings, rerank, vision, ASR.
- Identité d'embeddings : `pipeline_id` (`emb-v2:…`), `expected_digest`,
  `model_digest`, `identity_verified`, `verified_at`, `normalized`, `truncated`.
- Sondes `/health`, `/health/ready`, `/health/embeddings`, `/health/rerank`,
  `/health/asr`, `/health/vision`, `/v1/vision/capacity`.

### Ajouts (additifs)

| Ajout | Détail | Impact RAGatm | Impact Pool |
|---|---|---|---|
| `GET /health/live` | sonde strictement non inférentielle | utiliser pour l'attente de disponibilité | sonde `HEALTH_PROBE` recommandée |
| En-tête `X-Request-ID` | relayé (motif borné) ou généré, sur toute réponse | lire l'en-tête pour corréler | relayer celui validé/généré (AIGP-NEW-04) |
| `503 model_not_installed` (`retryable: false`) | modèle enregistré mais absent du runtime ; aucun téléchargement | afficher une erreur d'installation, pas de retry | ne pas basculer de nœud automatiquement |
| `503 generation_saturated`, `503 embeddings_saturated` + `Retry-After` | concurrence bornée explicite | retry après `Retry-After` | saturation observable par nœud |
| `413 request_too_large`, `411 length_required`, `400 invalid_content_length` | borne de taille (`AI_GATEWAY_REQUEST_MAX_BYTES`), `Content-Length` requis sur POST JSON hors `/v1/audio/*`, `/v1/vision/*` | borner les lots | le relais conserve `Content-Length` (déjà le cas) |
| `400 embedding_dimension_incompatible` | profil d'une autre dimension que l'index | protège l'index 768 | — |
| `/health/runtime.capabilities`, `.limits` | états `ENABLED/DISABLED/UNAVAILABLE/UNKNOWN`, `installed` par modèle, limites | tableau de bord | découverte de capacités |
| `compact` et `expected_digest` sur `/v1/models` et `/health/runtime` | métadonnées déclaratives (rôle, licence, qualification, budget mémoire, version) | affichage | — |
| `503 vision_task_unavailable` | `semantic_extract` non inclus | — | — |

### Retraits / changements de défaut (à ratifier)

| Changement | Détail | Impact |
|---|---|---|
| Registres stricts par défaut | `AI_GATEWAY_GENERATION_STRICT_REGISTRY=1`, `AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY=1` | un nom hors registre ⇒ `400 model_not_registered` (`0` rétablit le relais legacy) |
| `/health/runtime.v2h_shadow` | section retirée (composant non inclus) | consommateurs : ne plus lire la clé |
| `/v1/vision/structured-extract` | route absente (`404`) | capacité `UNAVAILABLE` |
| `schema_version: "3"` (vision) | toujours `503 vision_task_unavailable` | — |
| Défauts de modèles | `compact-generation`, `fr-en-retrieval` (noms logiques Compact) | configurer `OLLAMA_*_MODEL` par nom logique |

### Différences avec les 21 routes historiques (pour le Pool)

- +1 route : `GET /health/live` ⇒ 19 routes servies ici ; les 2 routes
  historiques manquantes (`/v1/vision/structured-extract` et le routeur
  StructX) ne sont pas incluses.
- Comportement de saturation étendu à generate/chat/embeddings (déjà présent
  sur rerank/vision/asr).
- Affinité ASR (`job_id`) : inchangée, capacité OFF par défaut.

### Points ouverts lors de la qualification

1. Statut HTTP de `model_not_installed` (503 retenu, non réessayable) vs 424.
2. Faut-il un `request_id` dans l'enveloppe d'erreur en plus de l'en-tête ?
3. Authentification inter-services (hors contrat 1.0).
4. Règle « backend reranker en boucle locale » vs réseau interne Compose
   (résolue ici par `network_mode: service:ai-gateway`).
