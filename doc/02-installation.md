# Installation

## Prérequis

- Docker Engine avec Compose v2, ou Python 3.12 pour un lancement hors Docker.
- Espace disque pour les modèles : ~0,6 Go (`granite-embedding:278m`),
  ~4,7 Go (`qwen2.5:7b-instruct`), ~2,3 Go (`bge-reranker-v2-m3`).
- GPU optionnel. Les budgets « 8 Go » / « 12 Go » du profil Compact sont des
  **estimations non mesurées** à ce lot (voir [15-modeles.md](15-modeles.md)).

## Pile Docker autonome (recommandé)

```bash
cp .env.example .env
AI_GATEWAY_BUILD_SHA=$(git rev-parse HEAD) docker compose up -d --build
```

Services : `ollama` (réseau interne, volume `ollama_data`), `ai-gateway`
(publié sur `127.0.0.1:8100` uniquement), `reranker` (TEI dans l'espace réseau
du conteneur Gateway, joignable par elle seule sur `127.0.0.1:8300`). Aucun
autre port n'est publié.

### Installer les modèles (à la charge de l'exploitant)

La Gateway ne télécharge jamais un modèle. Installer ceux du registre Compact :

```bash
docker compose exec ollama ollama pull granite-embedding:278m   # embeddings FR/EN (défaut)
docker compose exec ollama ollama pull qwen2.5:7b-instruct      # génération Compact-12
# optionnel :
docker compose exec ollama ollama pull qwen2.5:1.5b-instruct    # Compact-8 expérimental
docker compose exec ollama ollama pull nomic-embed-text         # embeddings EN (alternative)
```

Le reranker TEI télécharge `BAAI/bge-reranker-v2-m3` (révision épinglée) à
son premier démarrage dans le volume `tei_data`. Mémoire : la variante CPU
charge le modèle en float32 (~2,3 Go) et s'échauffe au lot maximal ;
les Compose posent `--max-batch-tokens 4096 --max-batch-requests 8
--auto-truncate` : l'attention est quadratique et les valeurs 16384 (défaut)
et 8192 ont été tuées par un cgroup de 7 Go sur CPU (qualification
). La troncature TEI est inatteignable : la Gateway borne requête
(2 000 car.) + document (4 000 car.) sous 4 096 tokens et refuse au-delà.
Sur GPU, relever la valeur et retirer `--auto-truncate`.

### Vérifier

```bash
curl -s http://127.0.0.1:8100/health/live                 # non inférentiel
curl -s http://127.0.0.1:8100/health/runtime | python3 -m json.tool   # états de capacités, installé/non
curl -s http://127.0.0.1:8100/v1/models                   # inventaire + identité
curl -s http://127.0.0.1:8100/health/ready                # sonde INFÉRENTIELLE explicite (embedding réel)
curl -s http://127.0.0.1:8100/health/rerank               # sonde INFÉRENTIELLE explicite (scoring réel)
```

Un modèle enregistré mais non installé apparaît `installed: false` dans
`/health/runtime` et toute requête le visant rend `503 model_not_installed`.

## Hors Docker (développement)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements.txt
docker compose -f docker-compose.reranker.yml up -d        # TEI sur 127.0.0.1:8300
export OLLAMA_BASE_URL=http://127.0.0.1:11434 AI_GATEWAY_RERANK_ENABLED=1
uvicorn app.main:app --host 127.0.0.1 --port 8100
```

Un Ollama local (`ollama serve`) doit posséder les modèles du registre.

## Fail-closed au démarrage

Le processus refuse de démarrer si : une variable est illisible ou hors
domaine ; le modèle par défaut (génération ou embeddings) est absent du
registre alors que le registre strict est actif ; le modèle d'embeddings par
défaut n'a pas la dimension de l'index (`AI_GATEWAY_EMBEDDINGS_DIMENSION`) ;
une capacité **non incluse** est demandée par l'environnement
(`AI_GATEWAY_HOST_ADMISSION_ENABLED`, `AI_GATEWAY_STRUCTURED_EXTRACT_ENABLED`,
`AI_GATEWAY_V2H_*`, `AI_GATEWAY_VISION_SEMANTIC_ENABLED`).

## Références d'implémentation

`docker-compose.yml`, `docker-compose.reranker.yml`, `.env.example`,
`ollama.env`, `Dockerfile` (HEALTHCHECK → `/health/live`), `app/settings.py`,
`app/main.py` (contrôles de démarrage).
