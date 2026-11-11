"""Client HTTP vers le backend reranker local (cross-encoder, loopback).

Le backend est un service dédié (TEI — text-embeddings-inference — servant
BAAI/bge-reranker-v2-m3, cf. docker-compose.reranker.yml), PARTAGÉ par les
instances prod et staging de la Gateway : une seule copie du modèle en VRAM,
exactement le patron Ollama.

Même doctrine que `ollama_client.py` : ce module est un PUR TRANSPORT. Il ne
trie pas, ne tronque pas, n'interprète pas les scores ; il convertit les
erreurs réseau en `UpstreamError` et valide seulement que la réponse a la
FORME attendue (liste de {index, score} numériques). Le tri, les limites, la
gouvernance et la journalisation appartiennent à la route.

Sécurité : l'URL du backend ne sort jamais de ce module (jamais dans une
réponse ni un log — même politique `redacted` que l'URL Ollama).
"""

import logging
import math
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from .errors import UpstreamError
from .settings import settings

logger = logging.getLogger("ai_gateway.rerank")

# Implémentation du protocole backend parlé par ce module (/info, /rerank,
# raw_scores). C'est un FAIT du code — le provider logique du registre reste
# `local_cross_encoder`, l'implémentation concrète est TEI. Exposé par
# /health/runtime pour la qualification cross-projet.
BACKEND_IMPLEMENTATION = "tei"

# Hôtes loopback admis pour le backend. Le backend n'a ni authentification ni
# gouvernance propres : l'exposer au-delà de la boucle locale ouvrirait un
# chemin de scoring non gouverné — la route refuse de servir dans ce cas.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def backend_url_is_loopback(url: str) -> bool:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


class RerankBackendClient:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ info
    def info(self, timeout: int = 5) -> Dict[str, Any]:
        """Identité servie par le backend : {model_id, model_sha, ...}.

        C'est la matière première de la vérification de révision — l'équivalent
        reranker de `verify_digests` côté embeddings.
        """
        try:
            resp = requests.get(f"{self.base_url}/info", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise UpstreamError(f"échec /info reranker: {exc}") from exc
        if not isinstance(data, dict):
            raise UpstreamError("réponse /info reranker inattendue")
        return data

    # ---------------------------------------------------------------- health
    def is_healthy(self, timeout: int = 5) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ---------------------------------------------------------------- rerank
    def rerank(self, query: str, texts: List[str]) -> List[Dict[str, Any]]:
        """Score un lot de paires (query, texte) en UN appel backend.

        Le batching tensoriel est la responsabilité du backend (TEI batch
        dynamiquement) : la Gateway n'envoie jamais N requêtes pour N textes.

        `raw_scores=true` : les logits BRUTS du cross-encoder, sans sigmoïde —
        le contrat expose une sémantique ordinale, pas une probabilité.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/rerank",
                json={"query": query, "texts": texts, "raw_scores": True},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            # Le corps d'erreur du backend n'est PAS relayé : il peut citer le
            # texte fautif. Statut seul ; le détail vit dans les logs backend.
            status = exc.response.status_code if exc.response is not None else "?"
            raise UpstreamError(f"échec /rerank backend: HTTP {status}") from exc
        except requests.RequestException as exc:
            raise UpstreamError(f"échec /rerank backend: {exc}") from exc

        # Validation de FORME uniquement (le contrôle de cardinalité/doublons
        # appartient à la route, qui connaît la requête d'origine).
        if not isinstance(data, list):
            raise UpstreamError("réponse /rerank backend inattendue (liste requise)")
        results: List[Dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, dict):
                raise UpstreamError("entrée /rerank backend inattendue")
            index = entry.get("index")
            score = entry.get("score")
            if not isinstance(index, int) or isinstance(index, bool):
                raise UpstreamError("index /rerank backend manquant ou invalide")
            if not isinstance(score, (int, float)) or isinstance(score, bool) \
                    or not math.isfinite(float(score)):
                raise UpstreamError(f"score /rerank backend invalide (index {index})")
            results.append({"index": index, "score": float(score)})
        return results


rerank_client = RerankBackendClient(
    settings.rerank_backend_url, settings.rerank_timeout_seconds
)
