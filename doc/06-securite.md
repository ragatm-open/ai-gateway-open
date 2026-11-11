# Sécurité

## Surface d'exposition

- La Gateway **n'a pas d'authentification** : elle est conçue pour n'être
  joignable que depuis le réseau interne de la pile (Compose publie le port sur
  `127.0.0.1` uniquement). Un jeton inter-services est une décision
  d'architecture du programme Open (hors de ce lot).
- Aucun port d'Ollama ni de TEI n'est publié ; le reranker partage l'espace
  réseau du conteneur Gateway et la Gateway refuse un backend reranker ou ASR
  **hors boucle locale** (ces backends n'ont ni auth ni gouvernance propres).
- Aucun fournisseur distant, aucun repli cloud, aucun téléchargement pendant
  une requête : un modèle absent est une erreur `503 model_not_installed`.

## Confidentialité des contenus

- Aucun prompt, message, texte à encoder, document à reclasser, vecteur ou
  image n'est journalisé ; les journaux portent des **faits mesurables**
  (modèle, tailles, durées, codes) sur une ligne (`log_event`), avec citation
  des valeurs non triviales pour empêcher la fabrication de fausses lignes.
- Les messages d'erreur publics sont **fixes** : jamais d'URL interne, jamais
  de corps upstream (qui peut citer le texte fautif), jamais de nom de fichier.
- Les noms de modèles sont validés par un alphabet borné (`^[A-Za-z0-9._:/-]+$`,
  256 caractères) avant tout backend ou journal ; les `X-Request-ID` invalides
  sont remplacés, jamais relayés ni journalisés.
- Les URLs des runtimes sont toujours rendues `redacted`.

## Fail-closed

- Validation stricte (`extra="forbid"`) : un champ inconnu (ex. `images` sur
  `/v1/generate`) est refusé `400`, jamais ignoré.
- Registres stricts par défaut ; allowlist d'environnement non contournable
  (un `production_allowed: false` ne se contourne ni par divergence de digest
  ni par registre non strict).
- Divergence de digest **prouvée** ⇒ modèle déclassé (`degraded`), identité
  non revendiquée ; upstream injoignable ⇒ `unavailable` sans invention.
- Budgets de sortie obligatoires par profil (`num_predict`), bornes de
  contexte, concurrence bornée non bloquante, taille de requête bornée.
- Capacités non incluses : refus explicite au démarrage si on tente de les
  activer ; jamais de service fictif.

## Chaîne d'approvisionnement

Image de base épinglée par digest, dépendances verrouillées avec hashes
(`--require-hashes`), toolchain de génération épinglée (`tools/toolchain.env`),
SBOM reproductible (`tools/sbom.sh`), labels OCI de provenance réels
(`revision`, `base.digest`), aucun `git` dans l'image.

## Hors périmètre de ce lot

Authentification/jeton, chiffrement TLS (reverse proxy de l'hébergeur),
quotas par client, journal d'audit.

## Références d'implémentation

`app/main.py` (`RequestEnvelopeMiddleware`, `_validation_handler`),
`app/errors.py` (messages publics), `app/metrics.py` (`log_event`),
`app/schemas.py` (`validate_model_name`), `app/rerank_client.py`
(`backend_url_is_loopback`), `Dockerfile`.
