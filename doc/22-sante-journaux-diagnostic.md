# Santé, journaux, diagnostic

## Séparation des états

| Question | Route | Inférence | Charge un modèle ? |
|---|---|---|---|
| Le processus répond-il ? | `/health/live` | non | non |
| Le logiciel est-il prêt (registre chargé, configuration valide) ? | `/health/live`, `/health/runtime` | non | non |
| Quels modèles sont **enregistrés** ? | `/health/runtime` | non | non |
| Quels modèles sont **installés** ? | `/health/runtime` (`installed`, inventaire en cache), `/v1/models` (`/api/tags`) | non | non |
| La capacité fonctionne-t-elle **réellement** ? | `/health/ready` (embedding), `/health/embeddings`, `/health/rerank` (scoring) | **oui** | oui (embeddings / reranker) |
| Inférence volontaire | `/v1/*` | oui | oui |

La sonde Docker (`HEALTHCHECK` du `Dockerfile`) vise `/health/live`. Des
tests espions (`tests/open/test_open_health_non_inferential.py`) garantissent
que `/health/live` et `/health/runtime` n'appellent aucune méthode des clients
Ollama, reranker ou ASR.

## États de capacité

`ENABLED` configurée et au moins un modèle autorisé ; `DISABLED` flag OFF ou
registre vide ; `UNAVAILABLE` composant non inclus ; `UNKNOWN` fait non encore
observé (inventaire). `installed` : `true`/`false` après inventaire, `null`
avant.

## Diagnostic rapide

| Symptôme | Lecture | Action |
|---|---|---|
| `/health/live` KO | processus mort ou configuration invalide (journal de démarrage nomme la variable) | corriger `.env`, redémarrer |
| `503 model_not_installed` | modèle enregistré, absent du runtime | `ollama pull` du tag du profil |
| `400 model_not_registered` | registre strict, nom inconnu | utiliser un nom logique du registre ou déclarer le profil |
| `identity_verified: false` avec `registry_status: unavailable` | Ollama injoignable ou modèle absent | voir `/health` |
| `registry_status: degraded` | digest observé ≠ attendu | décision registre + réindexation |
| `503 generation_saturated` / `embeddings_saturated` / `rerank_saturated` | concurrence atteinte | réessayer après `Retry-After` |
| `413 request_too_large` | corps > `AI_GATEWAY_REQUEST_MAX_BYTES` | réduire le lot |
| `/health/rerank` `503 backend injoignable` | TEI arrêté ou hors boucle locale | vérifier le conteneur `reranker` |

## Journaux

Une ligne par événement, `event=… clé=valeur`, aucune donnée utilisateur.
`X-Request-ID` n'est pas journalisé par défaut (il voyage dans l'en-tête de
réponse) ; un consommateur corrèle par son propre identifiant.

## Références d'implémentation

`app/main.py`, `Dockerfile`, `tests/open/test_open_health_non_inferential.py`.
