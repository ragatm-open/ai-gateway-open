# Écarts avec l'installation d'origine et limites connues

## Retiré ou non inclus

| Élément | Décision | Effet |
|---|---|---|
| Vision sémantique v3 (`vision_semantic_*`) | non inclus | `semantic_extract` ⇒ `503 vision_task_unavailable` ; union OpenAPI V1/V2 seulement |
| Extraction structurée PP-StructureV3 (`structx_route`, backend `structx/`) | non inclus | route absente ; flags refusés au démarrage |
| V2-H et shadow V2-H (`v2h_*`) | non inclus | section `v2h_shadow` retirée de `/health/runtime` ; flags refusés |
| Contrôleur d'admission hôte (`admission/`) | non inclus ; client `host_admission.py` conservé inerte | `AI_GATEWAY_HOST_ADMISSION_ENABLED=1` refusé |
| Backends ASR et vision | non inclus | rails présents, OFF, sans modèle au registre Compact |
| Registre de production (14–30 B, mesures privées) | remplacé par le registre Compact | profils candidats/expérimentaux |
| Compose/ops privés | remplacés par une pile autonome | `docker-compose.yml`, `.env.example` |

## Changements de contrat (candidats, à ratifier)

| Changement | Avant | Après | Impact RAGatm / Pool |
|---|---|---|---|
| Registres stricts par défaut | OFF (relais legacy) | **ON** | un nom hors registre ⇒ `400` ; configurer des noms logiques |
| Modèle absent du runtime | `502` générique | `503 model_not_installed`, `retryable: false` | distinguable d'une panne ; aucun retry automatique |
| Concurrence génération/embeddings | aucune borne Gateway | `503 generation_saturated` / `embeddings_saturated` + `Retry-After` | Pool : saturation observable |
| Taille de requête | non bornée (hors vision) | `413 request_too_large`, `411 length_required` | `Content-Length` requis sur les POST JSON |
| `X-Request-ID` | absent | relayé/généré sur chaque réponse | corrélation Pool/consommateur |
| `/health/live` | absent | nouvelle route non inférentielle | sonde Docker |
| `/health/runtime` | — | + `capabilities`, + `limits`, − `v2h_shadow` | additif sauf `v2h_shadow` |
| `/v1/models`, `/health/runtime` | — | + `compact` (métadonnées), + `expected_digest` LLM | additif |
| Dimension d'embeddings | contrôle sur la réponse | refus préalable `400 embedding_dimension_incompatible` | protège l'index |

Événements NDJSON (`delta`, `done`, `error`) et enveloppe d'erreur :
**inchangés**.

## Limites connues

- Aucune mesure GPU : budgets 8/12 Go non vérifiés ; `compact-generation`
  (7 B) sans digest relevé et jamais exécuté dans Open.
- Qualité RAG des candidats Compact non évaluée.
- Pas d'authentification (réseau interne requis).
- Les tests hérités tournent sur un registre de fixture, pas sur le registre
  Compact (couvert par `tests/open/`).
- La capability `chat` n'est portée par aucun modèle Compact : `/v1/chat`
  répond `400` tant qu'un profil ne la déclare pas avec un `reasoning_profile`.
- Latence du reranker CPU vs budget du consommateur non mesurée.

## Références d'implémentation

`app/main.py`, `app/settings.py`, `gateway-api/CHANGELOG.md`.
