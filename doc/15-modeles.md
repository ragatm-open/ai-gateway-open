# Registre Compact des modèles

Fichier : `app/models_registry.yaml` (fait partie du code). Chaque profil
porte, en plus de ses propriétés techniques, des **métadonnées Compact**
exposées sur `/health/runtime` et `/v1/models` : `compact_role`, `license`,
`qualification`, `memory_budget_mib`, `version`.

## États de qualification

| État | Sens |
|---|---|
| `candidate` | identifié, aucune mesure Open ; digest relevé ou non |
| `experimental` | identité vérifiée et inférence réelle servie dans un environnement isolé, qualité/ressources non mesurées |
| `qualified` | campagne de qualification Compact passée (aucun profil à ce lot) |
| `deprecated` | à ne plus sélectionner |

Un modèle **n'est jamais déclaré disponible** par le registre : la
disponibilité (installé) vient de l'inventaire runtime et l'identité
(`identity_verified`) de la comparaison digest attendu / observé.

## Profils

| Nom logique | Rôle | Artefact | Digest | Licence | Qualification | Budget mémoire estimé |
|---|---|---|---|---|---|---|
| `fr-en-retrieval` | embeddings FR/EN, **défaut** | `granite-embedding:278m`, 768 dims, 512 tokens (1 200 car., `reject`) | relevé (identique à l'origine) | Apache-2.0 | experimental | ~600 Mio |
| `legacy-768` | embeddings EN (alternative) | `nomic-embed-text`, 768 dims, 2 048 tokens (6 000 car.) | relevé par l'installation d'origine, non re-relevé | Apache-2.0 | candidate | ~650 Mio |
| `compact-generation` | génération Compact-12, **défaut** | `qwen2.5:7b-instruct` (Q4_K_M) | **aucun** (tag non tiré) | Apache-2.0 | candidate | ~6 Gio @8192 (**NOT_VERIFIED**) |
| `compact-generation-8` | génération Compact-8, expérimentale | `qwen2.5:1.5b-instruct` (Q4_K_M) | relevé | Apache-2.0 | experimental | ~2 Gio @8192 (**NOT_VERIFIED** GPU ; RSS CPU mesuré dans les preuves) |
| `rerank-default` | reranking local | `BAAI/bge-reranker-v2-m3` rév. `953dc6f6…` (TEI) | révision épinglée, vérifiée `/info` | Apache-2.0 | experimental | ~1,5 Gio |

Les modèles de production de l'installation d'origine (14–30 B) ne figurent
pas au registre Compact. Alternatives identifiées mais **non déclarées**
(licence ou qualification à établir) : `qwen3:4b`, `granite3.3:2b` (Compact-8),
`llama3.1:8b` (licence Llama Community, non OSI). `qwen2.5:3b-instruct` a été
écarté : licence Qwen Research, non Apache-2.0.

## Profils de gouvernance

| Profil | Valeurs | Usage |
|---|---|---|
| `context_profiles.compact-rag` | `num_ctx 8192`, max `16384` | chat RAG (prompt p95 ≈ 7 100 tokens mesuré à l'origine) |
| `context_profiles.compact-small` | `8192` / `16384` | petits modèles |
| `generation_profiles.compact-answer` | `num_predict 2048`, max `4096` | réponses ; jamais illimité |
| `keep_alive_profiles.resident` / `transient` | `30m` / `2m` | rétention VRAM (pas une réservation) |

Aucun `reasoning_profiles` : les candidats Compact sont des modèles instruct ;
la capability `chat` n'est déclarée par aucun profil.

## Identité d'embeddings

`pipeline_id = emb-v2:<sha256[:32]>` du payload canonique `{schema: 2,
model_digest, task, prefix, overflow, normalized, implementation: "1"}`. Il
dépend de la **transformation effective**, pas du nom logique : changer un
digest, un préfixe, la normalisation ou la politique de troncature le change
et impose une réindexation ; renommer un profil ne le change pas. Contrainte
Compact v1 : **768 dimensions** ; tout profil d'une autre dimension est refusé
tant qu'une migration d'index n'est pas décidée.

| Profil | `pipeline_id` document | `pipeline_id` query |
|---|---|---|
| `fr-en-retrieval` | `emb-v2:c8215aca68cbf785bf977a46ead4b23c` | voir `/health/embeddings` |

## Ressources

Aucune mesure GPU n'a été réalisée lors de la qualification (pas de GPU 8/12 Go isolé
disponible) : les budgets mémoire sont des estimations déclarées. Le profil
Compact **n'affirme pas** le support 8 Go ou 12 Go avant la qualification interne.

## Références d'implémentation

`app/models_registry.yaml`, `app/registry.py` (`ProfileMetadata`,
`embedding_pipeline_identity`), `tests/open/test_open_registry.py`.
