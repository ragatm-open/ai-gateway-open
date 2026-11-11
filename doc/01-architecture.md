# Architecture

## Vue d'ensemble

```text
                 ┌──────────────────────────────────────────────────┐
RAGatm Open ───► │ AI Gateway Open (FastAPI, un processus uvicorn)  │
 (ou Pool Open)  │  · validation stricte (Pydantic, extra=forbid)   │
                 │  · registre Compact (identité, bornes, allowlist)│
                 │  · concurrence bornée, erreurs typées            │
                 │  · X-Request-ID, borne de taille de requête      │
                 └───────┬───────────────────────────┬──────────────┘
                         │ HTTP /api/*               │ HTTP /rerank, /info
                         ▼                           ▼
                 Ollama local                 Reranker local (TEI)
                 génération, embeddings       cross-encoder, boucle locale
```

Trois composants, tous locaux. La Gateway **ne contient aucun modèle** et
**n'en télécharge jamais** : elle relaie vers des runtimes que l'exploitant a
approvisionnés. Aucun fournisseur de LLM distant, aucun repli cloud : une
panne ou un modèle absent est rendu au client par une erreur typée, jamais
masqué par un autre fournisseur.

## Composants

| Composant | Rôle | Frontière |
|---|---|---|
| `app/main.py` | routes HTTP, plans de génération gouvernés, états de capacités | un processus, un worker ; aucune file applicative |
| `app/settings.py` | configuration fail-closed lue à l'import | une variable invalide empêche le démarrage |
| `app/registry.py` + `models_registry.yaml` | registre Compact : profils d'embeddings (digest, dimension, préfixes, fenêtre), LLM (contexte, budget de sortie, rétention), reranker (artefact, révision) ; métadonnées Compact (rôle, licence, qualification, budget mémoire) | un modèle non décrit est refusé (registres stricts) |
| `app/ollama_client.py` | transport `/api/tags`, `/api/generate`, `/api/chat`, `/api/embed` (+ repli `/api/embeddings` borné) | URL jamais exposée ; `404 model not found` typé |
| `app/rerank_client.py` | transport TEI `/info`, `/rerank` (logits bruts) | URL obligatoirement en boucle locale |
| `app/asr_*`, `app/vision_*` | rails optionnels, **OFF** par défaut, backends non inclus | refus typé `503` quand désactivés |
| `app/host_admission.py` | client d'un contrôleur d'admission hôte **non inclus** ; inerte | activation refusée au démarrage |

## Flux d'une requête de génération

1. Validation Pydantic (champ inconnu ⇒ `400`), borne de taille (`413`), `X-Request-ID` posé.
2. Résolution du modèle dans le registre (nom logique ou tag Ollama) ; absent ⇒ `400 model_not_registered`.
3. Plan gouverné : `num_ctx` et `num_predict` du profil (ou surcharge client bornée, sinon `400`), `keep_alive` du profil, allowlist d'environnement.
4. Slot de concurrence (non bloquant) ⇒ sinon `503 generation_saturated` + `Retry-After`.
5. Appel Ollama ; modèle absent du runtime ⇒ `503 model_not_installed` (non réessayable) ; délai ⇒ `502` ; panne ⇒ `502`.
6. Réponse agrégée, ou flux NDJSON `delta*` → `done` (ou `error`).

## Flux d'embeddings

Même gouvernance, plus : préfixe de tâche du profil (`query`/`document`),
refus des textes trop longs (`reject`), contrôle de dimension (`768` en Compact
v1 — un profil d'une autre dimension est refusé `400
embedding_dimension_incompatible`), norme vérifiée pour un modèle déclaré
normalisé, et identité : `pipeline_id` (dérivé de la transformation
effective), `expected_digest`, `model_digest` observé, `identity_verified`.

## Trois états distincts, jamais confondus

| État | Source | Route |
|---|---|---|
| **enregistré** | registre (`models_registry.yaml`) | `/health/runtime` |
| **installé** | inventaire `/api/tags` en cache (non inférentiel) | `/health/runtime` → `capabilities.*.models.*.installed`, `/v1/models` |
| **capacité testée** | sonde inférentielle **explicite** | `/health/ready`, `/health/embeddings`, `/health/rerank` |

`/health/live` (sonde Docker) ne touche à rien : ni réseau, ni inférence.

## Compatibilité Pool

La Gateway est consommable directement ou via AI Gateway Pool Open : erreurs
distinguables par `code`, `X-Request-ID` relayé ou généré, modèle jamais
substitué (le nom demandé est résolu, jamais remplacé en silence), aucun en-tête
`X-AIGP-*` émis par la Gateway.

## Références d'implémentation

`app/main.py` (`RequestEnvelopeMiddleware`, `_plan_generation`, `_capability_states`),
`app/registry.py` (`ModelRegistry.verify_digests`, `llm_installed`),
`app/ollama_client.py` (`_model_not_found`).
