# API — contrat `gateway-api/1.0` (candidat)

Spécification machine : [`../gateway-api/1.0.yaml`](../gateway-api/1.0.yaml)
(OpenAPI 3.1, dérivée des schémas Pydantic réels par `tools/build_openapi.py`).
État : **`CANDIDATE_FOR_RATIFICATION`** jusqu'à la décision. Cette
page résume la sémantique ; en cas de doute, le YAML et le code font foi.

## Conventions

- Corps JSON, validation stricte : tout champ inconnu ⇒ `400` (`"message": "payload invalide"`, liste `errors`).
- **Enveloppe d'erreur** unique : `{"detail": {"status": "error", "provider": "ai_gateway|ollama|local_cross_encoder", "message": "...", "code"?: "...", "retryable"?: bool, ...}}`. `message` est fixe et public (jamais d'URL interne, jamais de contenu) ; `code` est l'identifiant stable quand il existe.
- **`X-Request-ID`** : relayé s'il respecte `^[A-Za-z0-9._:-]{1,128}$`, sinon généré (UUID v4) ; toujours présent dans la réponse, y compris sur les erreurs du middleware.
- **`Retry-After`** (secondes, entier) sur toute saturation `503` réessayable.
- Taille : `Content-Length` > `AI_GATEWAY_REQUEST_MAX_BYTES` ⇒ `413 request_too_large` ; corps sans longueur (`chunked`) sur un POST JSON ⇒ `411 length_required`. Exemptées : `/v1/audio/*`, `/v1/vision/*` (bornes propres).
- Délais : `AI_GATEWAY_TIMEOUT_SECONDS` borne les appels Ollama (⇒ `502` « délai dépassé » en agrégé, événement `error` `upstream_timeout` en flux) ; `AI_GATEWAY_RERANK_TIMEOUT_SECONDS` pour le reranker.
- **Annulation client** : la fermeture de la connexion pendant un flux ferme l'appel upstream (la génération s'arrête côté runtime) ; terminaison journalisée `client_disconnect`.
- Le nom de modèle demandé n'est **jamais substitué** : il est résolu (nom logique → tag Ollama) ou refusé.

## Santé (voir aussi [22-sante-journaux-diagnostic.md](22-sante-journaux-diagnostic.md))

| Route | Inférence | Réponse |
|---|---|---|
| `GET /health/live` | **aucune**, aucun réseau | `{status: ok, provider, version, registry_loaded, registry_version, uptime_seconds, inferential: false}` |
| `GET /health/runtime` | aucune | gouvernance (`llm_models`, profils), `capabilities` (états `ENABLED/DISABLED/UNAVAILABLE/UNKNOWN`, `installed` par modèle), `limits`, `inflight`, `build`, `rerank`, `asr` |
| `GET /health` | aucune (`/api/tags`) | contrat historique : inventaire + registre ; `503` si Ollama injoignable |
| `GET /health/ready` | **oui** (embedding réel) | `200` prêt / `503` ; en cache 60 s |
| `GET /health/embeddings` | **oui** (un embedding par profil) | diagnostic par modèle + compteurs |
| `GET /health/rerank` | **oui** (scoring réel) | `200 ok` / `200 disabled` / `503` |
| `GET /health/asr`, `GET /health/vision` | `/info` backend, `/api/ps` | `disabled` par défaut |

## Catalogue — `GET /v1/models`

Inventaire réel du runtime (`/api/tags`) enrichi de l'identité du registre :
`name`, `size`, `modified_at`, `digest`, `details`, et, pour un modèle décrit,
`model_logical`, `kind` (`embedding`|`llm`), `dimension` / profils, `registry_status`.
Un modèle enregistré **non installé n'apparaît pas** ici (il est visible
`installed: false` dans `/health/runtime`). `503` si Ollama injoignable.

## Génération — `POST /v1/generate`

Requête : `prompt` (obligatoire), `model?` (logique ou tag), `temperature?`
(défaut 0.2), `system?`, `options?` (`num_ctx`, `num_predict` bornés par le
profil ; hors borne ⇒ `400`, jamais un clamp), `think?`, `keep_alive?`
(refusé `400 keep_alive_not_overridable` si le profil le verrouille),
`format?` (`"json"` ou JSON Schema relayé brut) **ou** `response_format?`
(`{type: json_schema, schema}` validé et revalidé ; conflit ⇒ `400 generate_format_conflict`).

Réponse `200` : `text`, `provider: ai_gateway`, `upstream_provider: ollama`,
`model` (tag Ollama servi), `tokens{prompt, completion, total}`,
`usage{...}`, `cost: 0`, `done`, `done_reason` (verbatim ou `null`) ;
en mode structuré : `structured_output`, `response_schema_sha256`.

Erreurs : `400 model_not_registered`, `400 completion_not_supported`,
`400 structured_output_not_supported`, `400 structured_output_reasoning_incompatible`,
`400` (environnement, bornes, schéma refusé), `503 model_not_installed`
(`retryable: false`), `503 generation_saturated` (+`Retry-After`),
`502` échec/délai upstream, `502 output_empty|output_invalid|output_mismatch`
(structuré).

## Génération en flux — `POST /v1/generate/stream`

Même requête (`response_format` refusé `400 structured_output_streaming_not_supported`).
Réponse `200 application/x-ndjson`, une ligne JSON par événement — ce flux
**n'est pas du SSE** :

```json
{"type":"delta","delta":"Bon","done":false}
{"type":"done","delta":"","done":true,"model":"qwen2.5:1.5b-instruct","tokens":{...},"usage":{...},"done_reason":"stop"}
{"type":"error","error":{"code":"upstream_timeout|invalid_upstream_response|upstream_error","message":"..."}}
```

Exactement un `done` par flux réussi ; un flux en erreur n'a jamais de `done`.
Les pannes détectées **avant** le premier octet restent des statuts HTTP
(`400`, `502`, `503`).

## Chat — `POST /v1/chat`, `POST /v1/chat/stream` (*implémenté, aucun modèle Compact ne le déclare*)

`model` obligatoire et enregistré avec la capability `chat` (sinon `400
model_not_registered` / `400 chat_not_supported`), `messages[]`
(`system|user|assistant`), `reasoning?` validé contre le profil. Flux :
événements `thinking` (si exposé), `content`, `done` (`content_empty`,
`done_reason`), `error`.

## Embeddings — `POST /v1/embeddings`

Requête : `texts[]` (non vides, ≤ `max_batch`), `model?`, `task?`
(`query`|`document`, défaut `document`), `keep_alive?`.

Réponse `200` : `model`, `dimension`, `embeddings[][]`, `usage.total_texts`,
`governed`, `model_logical`, `expected_digest`, `model_digest` (observé ou
`null`), `identity_verified`, `verified_at`, **`pipeline_id`**
(`emb-v2:<32 hex>`, dérivé de la transformation effective), `task`,
`normalized`, `truncated[]` (toujours `false` en `reject`), `upstream_endpoint`.

Erreurs : `400 model_not_registered`-équivalent (« modèle non gouverné », registre strict),
`400` texte trop long / batch trop grand / environnement, `400
embedding_dimension_incompatible`, `503 model_not_installed`, `503
embeddings_saturated`, `502` (échec, budget épuisé, cardinalité, dimension,
vecteur invalide, norme hors tolérance).

## Reranking — `POST /v1/rerank`

Requête : `query`, `documents[{id, text}]` (ids opaques, ≤ 64), `model?`,
`top_n?`. Réponse `200` : `model` (artefact), `model_logical`,
`model_revision`, `results[{index, id, relevance_score}]` triés (score
décroissant, départage par index), `usage{documents, returned}`,
`score_semantics{higher_is_better: true, calibrated: false}` — les scores
sont des **logits ordinaux**, pas des probabilités.

Erreurs : `503` capacité désactivée ou backend hors boucle locale, `400`
reranker inconnu / bornes (`rerank_rejected_oversize`), `503 rerank_saturated`
(+`Retry-After`), `502` backend.

## Vision et ASR (*OFF par défaut, backends non inclus*)

`POST /v1/vision/analyze` ⇒ `503 vision_not_enabled` ; la task
`semantic_extract` ⇒ `503 vision_task_unavailable` (module non inclus) ;
`GET /v1/vision/capacity` ⇒ `status: disabled` ; `POST /v1/audio/transcriptions`
⇒ `503 asr_unavailable`. Les schémas restent définis dans le YAML, marqués
`x-gateway-capability.default_state: DISABLED`.

## Codes de retour — synthèse

| HTTP | Sens | `retryable` |
|---|---|---|
| 400 | requête ou gouvernance refusée | non |
| 411 / 413 | longueur requise / corps trop grand | non |
| 500 | erreur interne (enveloppe standard) | — |
| 502 | upstream en échec, délai, sortie hors contrat | selon le cas |
| 503 | capacité désactivée, modèle non installé (`false`), saturation (`true` + `Retry-After`), upstream indisponible | voir `code` |

## Références d'implémentation

`app/main.py`, `app/schemas.py`, `app/errors.py`, `gateway-api/1.0.yaml`,
`tests/open/test_open_openapi_conformance.py`.
