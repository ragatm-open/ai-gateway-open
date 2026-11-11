# AI Gateway Open

Passerelle d'inférence **locale** pour RAGatm Open : génération (agrégée et en
flux NDJSON), embeddings à identité de pipeline vérifiée, reranking par
cross-encoder, catalogue de modèles gouverné, erreurs typées et sondes de santé
non inférentielles. Aucun fournisseur distant, aucun repli cloud, aucun
téléchargement de modèle pendant une requête.

```text
RAGatm Open ──► AI Gateway Open ──► Ollama local        (génération, embeddings)
                       └──────────► Reranker local TEI  (reranking, boucle locale)
```

Statut : contrat `gateway-api/1.0` en état `CANDIDATE_FOR_RATIFICATION`
(voir [gateway-api/CHANGELOG.md](gateway-api/CHANGELOG.md)). Consommateurs : [ragatm-open](https://github.com/ragatm-open/ragatm-open) ; routeur facultatif : [ai-gateway-pool-open](https://github.com/ragatm-open/ai-gateway-pool-open).

## Démarrage rapide (pile autonome)

```bash
git clone https://github.com/ragatm-open/ai-gateway-open.git && cd ai-gateway-open
cp .env.example .env                      # configuration locale (ignorée par git), aucun secret
AI_GATEWAY_BUILD_SHA=$(git rev-parse HEAD) docker compose up -d --build
# installer les modèles du registre Compact (jamais fait par la Gateway) :
docker compose exec ollama ollama pull granite-embedding:278m
docker compose exec ollama ollama pull qwen2.5:7b-instruct
curl -s http://127.0.0.1:8100/health/live
curl -s http://127.0.0.1:8100/health/runtime | python3 -m json.tool
```

Le reranker (TEI) télécharge `BAAI/bge-reranker-v2-m3` à son premier
démarrage dans son volume ; la Gateway, elle, ne télécharge jamais rien.

## Routes

| Route | Rôle | Inférentielle ? |
|---|---|---|
| `GET /health/live` | processus vivant, registre chargé | non (sonde Docker) |
| `GET /health/runtime` | gouvernance, états de capacités, limites | non |
| `GET /v1/models` | modèles installés + identité du registre | non (`/api/tags`) |
| `POST /v1/generate` | génération agrégée | oui |
| `POST /v1/generate/stream` | génération en flux NDJSON (`delta` / `done` / `error`) | oui |
| `POST /v1/embeddings` | embeddings (`pipeline_id`, `identity_verified`) | oui |
| `POST /v1/rerank` | reranking (scores ordinaux) | oui |
| `GET /health/ready`, `/health/embeddings`, `/health/rerank` | sondes **explicites** | oui, jamais en monitoring automatique |
| `POST /v1/chat`, `/v1/chat/stream` | chat gouverné (aucun modèle Compact ne le déclare) | oui |
| `POST /v1/vision/analyze`, `GET /v1/vision/capacity`, `/health/vision` | Vision — `DISABLED` par défaut | — |
| `POST/GET/DELETE /v1/audio/transcriptions` | ASR — `DISABLED` par défaut, backend non inclus | — |

Contrat détaillé : [gateway-api/1.0.yaml](gateway-api/1.0.yaml) et
[doc/04-api.md](doc/04-api.md).

## Documentation

[doc/README.md](doc/README.md) — architecture, installation, configuration,
API, exploitation, sécurité, développement, registre des modèles, matrice des
capacités, santé et diagnostic, écarts et limitations.

## Développement

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements.txt && pip install -r requirements-dev.txt
python -m pytest tests -q
```

La suite héritée s'exécute sur un registre de fixture (`tests/conftest.py`) ;
les tests de conformité Open (`tests/open/`) vérifient le registre Compact réel
et le contrat `gateway-api/1.0`.

## Licence

Code original sous [Apache License 2.0](LICENSE) (voir [`NOTICE`](NOTICE)) ; composants tiers : [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Contribution : [`CONTRIBUTING.md`](CONTRIBUTING.md) ; sécurité : [`SECURITY.md`](SECURITY.md).
