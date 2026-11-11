# Documentation — AI Gateway Open

Passerelle d'inférence locale du programme **RAGatm Open** (profil Compact).
Le code est la source de vérité ; cette documentation décrit ce qui est
réellement implémenté, avec une étiquette d'état par capacité :
*implémenté*, *activé par défaut*, *activable par flag*, *expérimental*,
*candidat*, *non inclus*.

| Page | Contenu |
|---|---|
| [01-architecture.md](01-architecture.md) | composants, flux, frontières (runtime local, boucle locale, aucun fournisseur distant) |
| [02-installation.md](02-installation.md) | pile Docker autonome, installation des modèles, démarrage hors Docker |
| [03-configuration.md](03-configuration.md) | référentiel des variables d'environnement, défauts, domaines, fail-closed |
| [04-api.md](04-api.md) | contrat `gateway-api/1.0` : routes, schémas, erreurs, flux NDJSON, en-têtes |
| [05-exploitation.md](05-exploitation.md) | sondes de santé, journaux, compteurs, procédures |
| [06-securite.md](06-securite.md) | surface d'exposition, confidentialité des contenus, règles de boucle locale |
| [07-developpement.md](07-developpement.md) | tests, verrous de dépendances, image de base, SBOM, régénération des fixtures |
| [15-modeles.md](15-modeles.md) | registre Compact : profils, digests, licences, états de qualification, identité d'embeddings |
| [17-matrice-capacites.md](17-matrice-capacites.md) | matrice des capacités et de leur état dans cette distribution |
| [22-sante-journaux-diagnostic.md](22-sante-journaux-diagnostic.md) | sondes inférentielles vs non inférentielles, diagnostic |
| [23-ecarts-limitations.md](23-ecarts-limitations.md) | écarts avec l'installation d'origine, limites connues, points ouverts |

Contrat API : [`../gateway-api/1.0.yaml`](../gateway-api/1.0.yaml) — état
`CANDIDATE_FOR_RATIFICATION` (à rendre), journal dans
[`../gateway-api/CHANGELOG.md`](../gateway-api/CHANGELOG.md).

## Références d'implémentation

- `app/main.py` — routes, middleware d'enveloppe (X-Request-ID, borne de taille), états de capacités ;
- `app/settings.py` — configuration fail-closed du profil Open ;
- `app/registry.py`, `app/models_registry.yaml` — registre Compact, identité de pipeline ;
- `app/ollama_client.py`, `app/rerank_client.py` — transports vers les runtimes locaux ;
- `tests/open/` — tests de conformité Open ; `tests/` — suite héritée (registre de fixture).
