# Mentions de tiers — AI Gateway Open

Le code original de ce dépôt est sous **Apache-2.0** (`LICENSE`, `NOTICE`) ; les composants ci-dessous conservent leurs licences.

Ce fichier recense les composants tiers utilisés ou référencés par cette
distribution. Il ne constitue pas une analyse juridique : seules les
métadonnées de licence publiées par les éditeurs (fiches de modèles, métadonnées
de paquets) sont reprises. **Aucun poids de modèle n'est redistribué** par AI
Gateway Open : les modèles sont installés par l'exploitant depuis les
bibliothèques publiques (Ollama, Hugging Face) et relèvent de leurs licences.

## Modèles du registre Compact (`app/models_registry.yaml`)

| Rôle | Artefact | Licence déclarée (fiche éditeur) | État de qualification |
|---|---|---|---|
| Embeddings FR/EN (défaut) | `granite-embedding:278m` (IBM Granite Embedding 278M) | Apache-2.0 | experimental |
| Embeddings EN (alternative) | `nomic-embed-text` (Nomic Embed Text v1.5) | Apache-2.0 | candidate |
| Génération Compact-12 | `qwen2.5:7b-instruct` (Alibaba Qwen2.5) | Apache-2.0 | candidate |
| Génération Compact-8 (expérimentale) | `qwen2.5:1.5b-instruct` (Alibaba Qwen2.5) | Apache-2.0 | experimental |
| Reranking | `BAAI/bge-reranker-v2-m3` (révision `953dc6f6…`) | Apache-2.0 | experimental |

Note de sélection : la variante `qwen2.5:3b-instruct` n'a **pas** été retenue
comme candidat Compact-8 parce que sa licence (Qwen Research License) n'est pas
Apache-2.0 ; vérifier la fiche de tout modèle ajouté avant de le déclarer.

## Runtimes locaux

- **Ollama** (`ollama/ollama`) — MIT. Runtime de génération et d'embeddings.
- **Text Embeddings Inference** (`ghcr.io/huggingface/text-embeddings-inference`)
  — Apache-2.0. Backend de reranking (cross-encoder), variante CPU épinglée par
  digest dans `.env.example` (copié dans `.env`).

## Bibliothèques Python (dépendances directes, `requirements.in`)

| Paquet | Licence |
|---|---|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| Requests | Apache-2.0 |
| Pydantic | MIT |
| PyYAML | MIT |
| Pillow | MIT-CMU |
| jsonschema | MIT |
| python-multipart | Apache-2.0 |

L'inventaire complet (dépendances transitives, versions, hashes) est dans
`requirements.txt` et dans les SBOM générés par `tools/sbom.sh`.

## Composants NON inclus dans cette distribution

Les backends ASR (faster-whisper, diarisation Sortformer), d'extraction
structurée (PP-StructureV3) et V2-H de l'installation d'origine ne font pas
partie d'AI Gateway Open. Leurs mentions de tiers appartiennent à leurs
dépôts respectifs ; les routes correspondantes sont annoncées `DISABLED` ou
`UNAVAILABLE` (voir `doc/17-matrice-capacites.md`).

## Composants soumis à une obligation particulière

| Composant | Licence | Disposition |
|---|---|---|
| certifi | MPL-2.0 | fichier de certificats non modifié (transitif de requests) |
| `python:3.12-slim` (image de base) | PSF-2.0 + licences Debian | référencée par digest, non modifiée |

## Inventaire complet des bibliothèques Python (`requirements.txt` = runtime ; `requirements-dev.txt` = développement)

| Paquet | Version | Licence (métadonnées publiées) | Portée |
|---|---|---|---|
| annotated-doc | 0.0.5 | MIT | runtime |
| annotated-types | 0.8.0 | MIT | runtime |
| anyio | 4.16.0 | MIT | runtime |
| attrs | 26.1.0 | MIT | runtime |
| certifi | 2026.10.23 | MPL-2.0 | runtime |
| charset-normalizer | 3.5.2 | MIT | runtime |
| click | 8.5.1 | BSD-3-Clause | runtime |
| fastapi | 0.162.2 | MIT | runtime |
| h11 | 0.16.0 | MIT | runtime |
| httpcore | 1.0.9 | BSD-3-Clause | développement / CI |
| httptools | 0.9.0 | MIT | runtime |
| httpx | 0.28.1 | BSD-3-Clause | développement / CI |
| idna | 3.20 | BSD-3-Clause | runtime |
| iniconfig | 2.3.1 | MIT | développement / CI |
| jsonschema | 4.26.0 | MIT | runtime |
| jsonschema-specifications | 2025.9.1 | MIT | runtime |
| opentelemetry-api | 1.45.2 | Apache-2.0 | runtime |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | développement / CI |
| pillow | 13.0.0 | MIT-CMU | runtime |
| pluggy | 1.7.0 | MIT | développement / CI |
| pydantic | 2.14.3 | MIT | runtime |
| pydantic_core | 2.50.4 | MIT | runtime |
| pytest | 8.3.4 | MIT | développement / CI |
| python-dotenv | 1.2.4 | BSD-3-Clause | runtime |
| python-multipart | 0.0.33 | Apache-2.0 | runtime |
| PyYAML | 6.0.2 | MIT | runtime |
| referencing | 0.37.0 | MIT | runtime |
| requests | 2.34.2 | Apache-2.0 | runtime |
| rpds-py | 2026.9.1 | MIT | runtime |
| starlette | 1.10.0 | BSD-3-Clause | runtime |
| typing-inspection | 0.4.5 | MIT | runtime |
| typing_extensions | 4.16.0 | PSF-2.0 | runtime |
| urllib3 | 2.8.0 | MIT | runtime |
| uvicorn | 0.54.0 | BSD-3-Clause | développement / CI |
| uvloop | 0.23.0 | MIT License | runtime |
| watchfiles | 1.3.0 | MIT | runtime |
| websockets | 17.4 | BSD-3-Clause | runtime |
