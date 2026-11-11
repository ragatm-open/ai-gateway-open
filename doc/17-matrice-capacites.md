# Matrice des capacités — AI Gateway Open

| Capacité | Routes | État dans cette distribution | Preuve / condition |
|---|---|---|---|
| Génération | `POST /v1/generate` | **implémenté, activé** | inférence réelle (CPU isolé) sur `compact-generation-8` ; `compact-generation` candidat non testé |
| Génération en flux (NDJSON) | `POST /v1/generate/stream` | **implémenté, activé** | idem ; annulation client fermant l'upstream |
| Embeddings | `POST /v1/embeddings` | **implémenté, activé** | identité vérifiée (digest observé = attendu), `pipeline_id`, dimension 768 |
| Reranking | `POST /v1/rerank` | **implémenté, activable par flag** (`.env.example` : ON) | TEI CPU isolé, identité `/info` vérifiée |
| Catalogue | `GET /v1/models` | **implémenté** | inventaire `/api/tags` + registre |
| Identité des modèles | registre, `/health/runtime`, `/v1/embeddings` | **implémenté** | digests relevés / `expected_digest` null si non relevé |
| Erreurs cohérentes | toutes | **implémenté** | enveloppe unique, `code`, `retryable`, `Retry-After`, `X-Request-ID` |
| Healthchecks non inférentiels | `GET /health/live`, `GET /health/runtime` | **implémenté** | tests espions : aucun appel runtime |
| Sondes inférentielles explicites | `/health/ready`, `/health/embeddings`, `/health/rerank` | **implémenté** | hors monitoring automatique |
| Chat gouverné | `POST /v1/chat(/stream)` | implémenté, **aucun modèle Compact ne le déclare** | `400 chat_not_supported` / `model_not_registered` |
| Sortie structurée (JSON Schema) | `response_format` sur generate/chat | implémenté | validation avant/après ; non qualifiée sur les modèles Compact |
| Vision (`describe`) | `POST /v1/vision/analyze`, `GET /v1/vision/capacity`, `/health/vision` | **OFF** (`DISABLED`) — aucun modèle vision au registre Compact | `503 vision_not_enabled` ; activable seulement avec un profil image + flag |
| Vision sémantique v3 (`semantic_extract`) | — | **non inclus** (`UNAVAILABLE`) | `503 vision_task_unavailable` |
| Extraction structurée PP-StructureV3 | `/v1/vision/structured-extract` | **non inclus** (`UNAVAILABLE`) | route absente (`404`) ; flag refusé au démarrage |
| ASR (transcription, diarisation) | `/v1/audio/transcriptions*`, `/health/asr` | **OFF** (`DISABLED`), backend non inclus | `503 asr_unavailable` ; code conservé pour un backend compatible fourni par l'exploitant |
| V2-H / shadow | — | **non inclus** | flags refusés au démarrage |
| Contrôleur d'admission hôte | — | **non inclus** ; client inerte | flag refusé au démarrage ; `capabilities.host_admission: UNAVAILABLE` |
| Fournisseur distant / repli cloud | — | **absent par construction** | test `tests/open/test_open_no_external_provider.py` |

Vocabulaire des états exposés : `ENABLED`, `DISABLED`, `UNAVAILABLE`,
`UNKNOWN` (`/health/runtime.capabilities`).

## Références d'implémentation

`app/main.py` (`_capability_states`), `app/settings.py`
(`_EXCLUDED_CAPABILITY_FLAGS`), `tests/open/`.
