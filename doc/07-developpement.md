# Développement

## Environnement

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements.txt
pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Deux jeux de tests

| Jeu | Registre | Objet |
|---|---|---|
| `tests/` (hérité) | `tests/data/registry_fixture.yaml` via `conftest.py` (registres non stricts, défauts historiques) | invariants de gouvernance, différentiels, identité de pipeline, rails ASR/Vision sur profils synthétiques |
| `tests/open/` (conformité Open) | `app/models_registry.yaml` (réel) et `Settings()` sans variables | défauts Open, registre Compact strict, `/health/live` non inférentiel, ASR/Vision OFF, absence de fournisseur externe, scan de confidentialité, conformité OpenAPI |

Garde-fou commun : toute connexion réseau réelle pendant un test unitaire
lève `ForbiddenNetworkAccess` (`conftest._no_network`). La suite reste verte
sous `--network none`.

## Qualification avec de vrais modèles

`tools/open_live_qualification.py` exécute, contre une Gateway lancée devant
un Ollama et un TEI **isolés**, les épreuves : `/health/live` et
`/health/runtime` sans inférence (espions sur les clients), catalogue,
génération, flux NDJSON, embeddings avec identité vérifiée, reranking ; il
écrit un fichier de preuves JSON (versions, digests, latences, mémoire). Il ne
télécharge rien et refuse une cible qui n'est pas en boucle locale.

## Verrous de dépendances

`requirements.txt` et `requirements-dev.txt` sont **générés**
(`tools/update_requirements.sh`, pip-tools épinglé dans
`tools/toolchain.env`, dans l'image Python du `FROM`). Modifier uniquement les
`.in`. L'installation exige les hashes.

## Image de base

`FROM python:3.12-slim@sha256:…` : tag lisible, digest obligatoire. Pour
renouveler : relever le digest de l'index multi-arch
(`docker buildx imagetools inspect python:3.12-slim`), mettre à jour `FROM` et le
label `org.opencontainers.image.base.digest`, régénérer les verrous.

## SBOM et vérification d'image

```bash
tools/sbom.sh ai-gateway-open:local ./out
tools/verify_image.sh ai-gateway-open:local $(git rev-parse HEAD)
```

## Fixtures synthétiques

`tests/data/vision_normalizer/generate.py` régénère les sept images (formes
Pillow) et leurs attendus ; un changement d'octets sans changement voulu du
normaliseur est une régression. `tests/data/c1_generate_golden.json` est un
golden synthétique du différentiel C1.

## Contrat OpenAPI

`gateway-api/1.0.yaml` est produit par `tools/build_openapi.py` depuis
`app.openapi()` (schémas réels) enrichi des sections contractuelles
(`x-gateway-*` : enveloppe d'erreur, événements NDJSON, en-têtes, états).
`tests/open/test_open_openapi_conformance.py` vérifie que l'API réelle et le
fichier coïncident ; après tout changement d'API : régénérer puis journaliser
dans `gateway-api/CHANGELOG.md`.

## Références d'implémentation

`tools/update_requirements.sh`, `tools/sbom.sh`, `tools/verify_image.sh`,
`tools/toolchain.env`, `tools/build_openapi.py`, `tools/open_live_qualification.py`,
`tests/conftest.py`, `tests/open/`.
