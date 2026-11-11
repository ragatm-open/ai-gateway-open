"""Client HTTP vers l'upstream Ollama (local, 127.0.0.1:11434).

Expose les appels /api/tags, /api/generate — en mode agrégé et en mode flux —,
/api/chat (lot C1, agrégé et flux) et /api/embed. Un fallback vers /api/embeddings est prévu si /api/embed n'est
pas disponible sur l'instance.

Le repli est borné par un **budget de temps global** (lot 1) : il encode un
texte à la fois, et sans budget une requête de 64 textes pouvait consommer
64 × `AI_GATEWAY_TIMEOUT_SECONDS`, soit plus d'une heure en production.

Métriques de performance (lot 3) : ce module reste un pur transport. Il remonte
les durées BRUTES d'Ollama — `generate()` en retournant le corps JSON complet,
`embed()` via `EmbedResult.ollama_metadata` — sans jamais les convertir ni les
interpréter. Conversion, dérivation et journalisation vivent dans
`app/metrics.py` et `app/main.py`.
"""

import json
import logging
import time
from typing import Any, Dict, List, Tuple

import requests

from .errors import (
    IncompleteUpstreamStream,
    InvalidUpstreamChunk,
    StreamTimeout,
    UpstreamError,
    UpstreamModelNotFound,
    UpstreamTimeout,
)
from .settings import settings

logger = logging.getLogger("ai_gateway.ollama")

ENDPOINT_EMBED = "/api/embed"
ENDPOINT_EMBEDDINGS = "/api/embeddings"
ENDPOINT_CHAT = "/api/chat"

# Rôles de message relayés à `/api/chat` (lot C1). Liste FERMÉE : le
# transport ne construit un message qu'à partir de `role` + `content`, jamais
# d'un dict client recopié en bloc — un champ arbitraire (`images`,
# `tool_calls`…) ne peut donc pas atteindre Ollama par ce chemin.
CHAT_ROLES = ("system", "user", "assistant")

# Sentinelle du routeur Gin d'Ollama pour une route inconnue : corps
# `404 page not found` en text/plain. Un modèle absent, lui, rend un 404 JSON
# `{"error": "model \"x\" not found, try pulling it first"}` — relevé sur
# un Ollama 0.31.x. Seule la première forme prouve que
# `/api/embed` n'existe pas (Ollama < 0.1.26) et justifie le repli historique.
_GIN_ROUTE_NOT_FOUND = "page not found"
# Forme Ollama d'un MODÈLE absent : 404 JSON `{"error": "model \"x\" not found,
# try pulling it first"}`. Reconnue sur le corps, jamais sur le
# seul statut : un 404 de route inconnue (Gin) n'est pas un modèle absent.
_MODEL_NOT_FOUND_MARKER = "not found"


def _model_not_found(response: Any) -> bool:
    """True si la réponse prouve que le runtime ne possède pas le modèle."""
    if response is None or getattr(response, "status_code", None) != 404:
        return False
    body = (getattr(response, "text", "") or "")
    if _GIN_ROUTE_NOT_FOUND in body.lower():
        return False
    try:
        data = json.loads(body)
    except ValueError:
        return False
    error = data.get("error") if isinstance(data, dict) else None
    return isinstance(error, str) and _MODEL_NOT_FOUND_MARKER in error.lower()


def _http_error_to_upstream(exc: "requests.HTTPError", endpoint: str) -> UpstreamError:
    """Traduction d'un HTTPError : modèle absent (typé) ou erreur upstream."""
    body = exc.response.text if exc.response is not None else ""
    if _model_not_found(exc.response):
        # Message FIXE : le corps Ollama cite le nom demandé, déjà connu de la
        # route ; rien d'autre n'est relayé.
        return UpstreamModelNotFound(f"échec {endpoint}: modèle non installé sur le runtime")
    return UpstreamError(f"échec {endpoint}: {exc}", body=body)


def _legacy_endpoint_unsupported(response: Any) -> bool:
    """True si la réponse prouve que `/api/embed` lui-même n'est pas servi.

    Politique de refus, volontairement étroite : `405 Method Not Allowed`, ou
    `404` portant la sentinelle Gin d'une route inconnue. Tout le reste — 404
    modèle absent, 400 validation, 429, 5xx — est une erreur de l'endpoint
    moderne, jamais une raison de repli.
    """
    if response is None:
        return False
    status = getattr(response, "status_code", None)
    if status == 405:
        return True
    if status == 404:
        body = (getattr(response, "text", "") or "").lower()
        return _GIN_ROUTE_NOT_FOUND in body
    return False


# Champs de performance lus dans la réponse d'`/api/embed`. Liste FERMÉE et
# strictement numérique : le client ne recopie jamais la réponse upstream en
# bloc, pour qu'aucun champ inattendu ne puisse remonter jusqu'aux logs.
_EMBED_METRIC_FIELDS = ("total_duration", "load_duration", "prompt_eval_count")


def _embed_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extrait les métadonnées de performance d'une réponse `/api/embed`.

    Retourne un dict éventuellement vide : les runtimes Ollama ne renvoient pas
    tous ces champs, et leur absence n'est pas une erreur.
    """
    return {
        field: data[field]
        for field in _EMBED_METRIC_FIELDS
        if data.get(field) is not None
    }


class EmbedResult(list):
    """Liste de vecteurs qui transporte aussi l'endpoint réellement utilisé.

    Sous-classe de `list` volontairement : tout appelant (et tout test) qui
    traite le retour de `embed()` comme une liste continue de fonctionner, sans
    état partagé sur le client — donc sans problème de concurrence.

    `ollama_metadata` porte les durées BRUTES d'Ollama (en nanosecondes), non
    converties : la conversion et la dérivation appartiennent à `app/metrics.py`,
    de sorte qu'il n'existe qu'un seul endroit où l'unité change.
    """

    def __init__(
        self,
        vectors: List[List[float]],
        upstream_endpoint: str = ENDPOINT_EMBED,
        fallback_used: bool = False,
        ollama_metadata: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(vectors)
        self.upstream_endpoint = upstream_endpoint
        self.fallback_used = fallback_used
        # Toujours un dict, jamais None : les appelants n'ont pas à distinguer
        # « pas de métadonnées » de « attribut absent ».
        self.ollama_metadata: Dict[str, Any] = ollama_metadata or {}


class GenerateStream:
    """Flux `/api/generate` ouvert, borné par un budget de temps mural.

    Deux plafonds coexistent, et ils ne protègent pas de la même chose :

    * le **timeout de lecture** de `requests` coupe un upstream devenu muet ;
    * le **budget mural** (`deadline`) coupe un upstream bavard mais trop lent.
      Sans lui, Ollama pourrait émettre quelques octets toutes les secondes et
      tenir la requête bien au-delà de `AI_GATEWAY_TIMEOUT_SECONDS`.

    Le budget est contrôlé après chaque ligne reçue : c'est le seul point où le
    flux nous rend la main sans devoir interrompre une lecture bloquante.

    L'appelant **doit** appeler `close()` — la route le fait dans un `finally`
    qui couvre succès, timeout, erreur de parsing et déconnexion client.
    """

    def __init__(self, response: Any, deadline: float, endpoint: str = "/api/generate") -> None:
        self._response = response
        self._deadline = deadline
        # Lot C1 : le même lecteur sert `/api/chat` (ChatStream). Seul le
        # libellé des messages d'erreur change — la valeur par défaut conserve
        # bit pour bit les messages historiques de `/api/generate`.
        self._endpoint = endpoint

    def events(self):
        """Itère les événements JSON du flux, un par ligne NDJSON d'Ollama.

        Lève `StreamTimeout` si le budget mural est épuisé et
        `InvalidUpstreamChunk` sur une ligne illisible. Les lignes vides
        (keep-alive de transport) sont ignorées silencieusement.
        """
        # Lot P11 : une rupture de transport APRÈS l'ouverture du flux
        # (ChunkedEncodingError, ConnectionError, ReadTimeout…) surgissait de
        # `iter_lines()` en `requests.RequestException` brute — inconnue de la
        # route, qui ne comprend que la famille UpstreamError. Résultat : 200
        # déjà engagé, flux tronqué, aucun événement d'erreur, ExceptionGroup
        # côté serveur. Seul l'appel réseau est encadré : une ligne illisible
        # reste une InvalidUpstreamChunk, jamais maquillée en erreur réseau.
        lines = self._response.iter_lines()
        while True:
            try:
                line = next(lines)
            except StopIteration:
                return
            except requests.RequestException as exc:
                # Message FIXE : pas de `str(exc)`, qui porte l'URL upstream.
                raise UpstreamError(
                    f"échec {self._endpoint}: rupture de transport pendant le flux"
                ) from exc
            if time.monotonic() > self._deadline:
                raise StreamTimeout(
                    f"échec {self._endpoint}: budget de temps global épuisé "
                    "pendant le flux"
                )
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                # La ligne fautive n'est JAMAIS reproduite : elle contient
                # potentiellement du texte généré, donc du contenu utilisateur.
                raise InvalidUpstreamChunk(
                    f"échec {self._endpoint}: ligne de flux illisible"
                ) from exc
            if not isinstance(event, dict):
                raise InvalidUpstreamChunk(
                    f"échec {self._endpoint}: événement de flux inattendu"
                )
            yield event

    def close(self) -> None:
        """Ferme la réponse upstream. Idempotent et silencieux."""
        try:
            self._response.close()
        except Exception:  # noqa: BLE001 - fermer ne doit jamais masquer la cause
            logger.warning("fermeture du flux upstream impossible")


class ChatStream(GenerateStream):
    """Flux `/api/chat` ouvert (lot C1) — même lecteur, mêmes plafonds, mêmes
    garanties P11 que `GenerateStream` ; seul le libellé d'endpoint diffère.
    La FORME des événements diffère (`message.content` / `message.thinking`
    au lieu de `response`) : c'est la route qui l'interprète, pas le
    transport."""

    def __init__(self, response: Any, deadline: float) -> None:
        super().__init__(response, deadline, endpoint=ENDPOINT_CHAT)


class OllamaClient:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ----------------------------------------------------------------- tags
    def list_models(self, timeout: int = 5) -> List[Dict[str, Any]]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("models", []) or []
        except requests.RequestException as exc:
            raise UpstreamError(f"échec /api/tags: {exc}") from exc

    def list_running_models(self, timeout: int = 5) -> List[str]:
        """Noms des modèles RÉSIDENTS (/api/ps) — lecture seule, ne charge
        jamais rien. Ajout lot vision-1a pour /health/vision (résident ≠
        disponible)."""
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise UpstreamError(f"échec /api/ps: {exc}") from exc
        return [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]

    def list_running_models_vram(self, timeout: int = 2) -> List[Dict[str, Any]]:
        """Modèles RÉSIDENTS avec leur empreinte VRAM (/api/ps, lecture seule).

        Ajout lot asr-rail-foundation : l'admission GPU de la transcription
        additionne les `size_vram` RETOURNÉS PAR OLLAMA (aucun nom de modèle
        codé en dur). Chaque entrée : {name, size_vram} avec size_vram entier
        >= 0 (valeur absente ou illisible => 0, jamais une exception : un
        champ manquant n'est pas une panne d'Ollama).
        """
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise UpstreamError(f"échec /api/ps: {exc}") from exc
        out: List[Dict[str, Any]] = []
        for m in (data.get("models", []) if isinstance(data, dict) else []) or []:
            if not isinstance(m, dict):
                continue
            size = m.get("size_vram", 0)
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                size = 0
            out.append({"name": str(m.get("name", "")), "size_vram": size})
        return out

    # ------------------------------------------------------------- generate
    @staticmethod
    def _build_generate_payload(
        model: str,
        prompt: str,
        stream: bool,
        system: str | None = None,
        temperature: float | None = None,
        options: Dict[str, Any] | None = None,
        think: bool | None = None,
        keep_alive: str | int | float | None = None,
        format: str | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Corps de `/api/generate`, identique en stream et en non-stream.

        Seul `stream` diffère entre les deux modes. Factoriser ici garantit
        qu'un futur champ ajouté à la génération vaut pour les deux chemins :
        deux constructeurs parallèles divergeraient à la première évolution.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": dict(options or {}),
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if system:
            payload["system"] = system
        # `think` est un champ top-level Ollama (pas une option de génération).
        if think is not None:
            payload["think"] = think
        # Idem pour `keep_alive`. Test explicite contre None : 0 est falsy mais
        # signifie « décharge le modèle immédiatement » — c'est justement la
        # valeur utile en campagne. Un `if keep_alive:` la perdrait.
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        # `format` : "json" ou JSON Schema, top-level Ollama, relayé tel quel.
        if format is not None:
            payload["format"] = format
        return payload

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        options: Dict[str, Any] | None = None,
        think: bool | None = None,
        keep_alive: str | int | float | None = None,
        format: str | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = self._build_generate_payload(
            model=model,
            prompt=prompt,
            stream=False,
            system=system,
            temperature=temperature,
            options=options,
            think=think,
            keep_alive=keep_alive,
            format=format,
        )

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            raise _http_error_to_upstream(exc, "/api/generate") from exc
        # AVANT RequestException : `requests.Timeout` en est une sous-classe, et
        # l'ordre des `except` décide. Un budget épuisé et un refus de connexion
        # ne se diagnostiquent pas de la même façon.
        except requests.Timeout as exc:
            raise UpstreamTimeout(f"échec /api/generate: {exc}") from exc
        except requests.RequestException as exc:
            raise UpstreamError(f"échec /api/generate: {exc}") from exc

    # ------------------------------------------------------ generate (stream)
    def open_generate_stream(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        options: Dict[str, Any] | None = None,
        think: bool | None = None,
        keep_alive: str | int | float | None = None,
        format: str | Dict[str, Any] | None = None,
    ) -> "GenerateStream":
        """Ouvre un flux `/api/generate` et retourne un handle itérable.

        L'appel est **bloquant jusqu'aux en-têtes de réponse** : une connexion
        refusée ou un statut HTTP d'erreur lève ici, donc AVANT que la route
        n'ait commencé à répondre. C'est ce qui permet de conserver un vrai
        502 pour les pannes détectables à l'ouverture, et de réserver les
        événements d'erreur NDJSON aux ruptures survenant en cours de flux.

        Le budget de temps mural démarre au moment de l'appel : le temps de
        connexion est décompté comme le reste.
        """
        payload = self._build_generate_payload(
            model=model,
            prompt=prompt,
            stream=True,
            system=system,
            temperature=temperature,
            options=options,
            think=think,
            keep_alive=keep_alive,
            format=format,
        )

        deadline = time.monotonic() + self.timeout
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            translated = _http_error_to_upstream(exc, "/api/generate")
            if exc.response is not None:
                exc.response.close()
            raise translated from exc
        except requests.Timeout as exc:
            raise UpstreamTimeout(f"échec /api/generate: {exc}") from exc
        except requests.RequestException as exc:
            raise UpstreamError(f"échec /api/generate: {exc}") from exc

        return GenerateStream(resp, deadline)

    # ----------------------------------------------------------------- chat
    @staticmethod
    def _build_chat_payload(
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: float | None = None,
        options: Dict[str, Any] | None = None,
        think: bool | str | None = None,
        keep_alive: str | int | float | None = None,
        format: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Corps de `/api/chat` (lot C1), identique en stream et non-stream.

        Volontairement SANS `tools` (lot C4). `format` (lot C3) n'accepte
        qu'un OBJET JSON Schema déjà validé par app/structured.py — jamais la
        chaîne `"json"`, jamais une valeur client brute — et n'est exposé que
        par `chat()` : `open_chat_stream()` n'a pas ce paramètre, la sortie
        structurée en flux n'existe pas en C3.

        `think` accepte un booléen (modèles à bascule) ou un niveau
        `"low"/"medium"/"high"` (modèles à paliers) : c'est la sémantique
        d'Ollama ≥ 0.11, que la Gateway ne relaie qu'après validation par le
        profil de raisonnement du modèle (jamais une valeur client brute).
        """
        rendered: List[Dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            if role not in CHAT_ROLES:
                raise ValueError(f"rôle de message non relayable : {role!r}")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("content de message doit être une chaîne")
            rendered.append({"role": role, "content": content})
        payload: Dict[str, Any] = {
            "model": model,
            "messages": rendered,
            "stream": stream,
            "options": dict(options or {}),
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if format is not None:
            if not isinstance(format, dict):
                raise ValueError("format doit être un objet JSON Schema validé")
            payload["format"] = format
        return payload

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        options: Dict[str, Any] | None = None,
        think: bool | str | None = None,
        keep_alive: str | int | float | None = None,
        format: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """`POST /api/chat` agrégé. Retourne le corps JSON BRUT (dict) : la
        route en extrait `message.content`, `message.thinking`, `done_reason`
        et les métriques — le transport n'interprète rien."""
        payload = self._build_chat_payload(
            model=model, messages=messages, stream=False,
            temperature=temperature, options=options, think=think,
            keep_alive=keep_alive, format=format,
        )
        try:
            resp = requests.post(
                f"{self.base_url}{ENDPOINT_CHAT}",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            raise _http_error_to_upstream(exc, ENDPOINT_CHAT) from exc
        except requests.Timeout as exc:
            raise UpstreamTimeout(f"échec {ENDPOINT_CHAT}: {exc}") from exc
        except requests.RequestException as exc:
            raise UpstreamError(f"échec {ENDPOINT_CHAT}: {exc}") from exc
        except ValueError as exc:
            # Corps non JSON : message FIXE, le corps n'est pas reproduit.
            raise UpstreamError(f"échec {ENDPOINT_CHAT}: réponse illisible") from exc
        if not isinstance(data, dict):
            raise UpstreamError(f"échec {ENDPOINT_CHAT}: réponse inattendue")
        return data

    def open_chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        options: Dict[str, Any] | None = None,
        think: bool | str | None = None,
        keep_alive: str | int | float | None = None,
    ) -> "ChatStream":
        """Ouvre un flux `/api/chat`. Même discipline que
        `open_generate_stream` : bloquant jusqu'aux en-têtes (502 classique
        avant le premier octet), budget mural démarré ici."""
        payload = self._build_chat_payload(
            model=model, messages=messages, stream=True,
            temperature=temperature, options=options, think=think,
            keep_alive=keep_alive,
        )
        deadline = time.monotonic() + self.timeout
        try:
            resp = requests.post(
                f"{self.base_url}{ENDPOINT_CHAT}",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            translated = _http_error_to_upstream(exc, ENDPOINT_CHAT)
            if exc.response is not None:
                exc.response.close()
            raise translated from exc
        except requests.Timeout as exc:
            raise UpstreamTimeout(f"échec {ENDPOINT_CHAT}: {exc}") from exc
        except requests.RequestException as exc:
            raise UpstreamError(f"échec {ENDPOINT_CHAT}: {exc}") from exc
        return ChatStream(resp, deadline)

    # ------------------------------------------------------------ embeddings
    def embed(
        self,
        model: str,
        inputs: List[str],
        keep_alive: str | int | float | None = None,
    ) -> "EmbedResult":
        """Retourne les vecteurs, l'endpoint qui les a produits et ses métriques.

        Essaie /api/embed (format moderne, batch natif) puis retombe sur
        /api/embeddings (format historique, un texte à la fois) si nécessaire.
        Seul le chemin moderne renseigne `ollama_metadata` (cf. EmbedResult).

        L'ensemble de l'appel — repli compris — est borné par un budget de
        `self.timeout` secondes. Si le budget expire au milieu du repli, on lève
        une erreur upstream explicite plutôt que de laisser la requête traîner :
        un lot n'est jamais renvoyé à moitié encodé.
        """
        # cf. generate() : test contre None, car 0 est une valeur significative.
        extra: Dict[str, Any] = {}
        if keep_alive is not None:
            extra["keep_alive"] = keep_alive

        deadline = time.monotonic() + self.timeout

        # 1) /api/embed — batch natif
        try:
            resp = requests.post(
                f"{self.base_url}{ENDPOINT_EMBED}",
                json={"model": model, "input": inputs, **extra},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings:
                return EmbedResult(
                    embeddings,
                    ENDPOINT_EMBED,
                    fallback_used=False,
                    ollama_metadata=_embed_metadata(data),
                )
            # Un 200 sans `embeddings` vient d'un /api/embed qui EXISTE : c'est
            # une réponse invalide, pas une absence d'endpoint.
            raise UpstreamError(f"réponse {ENDPOINT_EMBED} sans embeddings")
        except requests.HTTPError as exc:
            if _model_not_found(exc.response):
                raise UpstreamModelNotFound(
                    f"échec {ENDPOINT_EMBED}: modèle non installé sur le runtime"
                ) from exc
            # Lot P10 — le repli n'est légitime QUE si `/api/embed` n'existe pas
            # sur cet Ollama. Avant, tout HTTPError repliait : un 404 « model
            # not found » repartait en N appels unitaires voués au même 404,
            # un 500/OOM multipliait par N la charge sur un upstream déjà en
            # difficulté (observé dans les logs : 404 modèle → repli).
            if not _legacy_endpoint_unsupported(exc.response):
                status = exc.response.status_code if exc.response is not None else "?"
                raise UpstreamError(f"échec {ENDPOINT_EMBED}: HTTP {status}") from exc
            logger.warning(
                "/api/embed non supporté par cet Ollama (HTTP %s), repli /api/embeddings",
                exc.response.status_code if exc.response is not None else "?",
            )
        except requests.RequestException as exc:
            # Timeout, connexion refusée, coupure : l'upstream est en difficulté,
            # pas l'endpoint — jamais de repli (il multiplierait la charge).
            raise UpstreamError(f"échec {ENDPOINT_EMBED}: {exc}") from exc

        # 2) Fallback /api/embeddings — un texte à la fois, sous budget global
        embeddings: List[List[float]] = []
        for index, text in enumerate(inputs):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "fallback /api/embeddings: budget global épuisé "
                    "(traités=%d total=%d budget_s=%d)",
                    index,
                    len(inputs),
                    self.timeout,
                )
                raise UpstreamError(
                    f"échec {ENDPOINT_EMBEDDINGS}: budget de temps global épuisé "
                    f"après {index}/{len(inputs)} textes ({self.timeout}s)"
                )
            try:
                resp = requests.post(
                    f"{self.base_url}{ENDPOINT_EMBEDDINGS}",
                    json={"model": model, "prompt": text, **extra},
                    timeout=min(remaining, self.timeout),
                )
                resp.raise_for_status()
                vec = resp.json().get("embedding")
                if not vec:
                    raise UpstreamError(f"réponse {ENDPOINT_EMBEDDINGS} sans embedding")
                embeddings.append(vec)
            except requests.HTTPError as exc:
                body = exc.response.text if exc.response is not None else ""
                raise UpstreamError(f"échec {ENDPOINT_EMBEDDINGS}: {exc}", body=body) from exc
            except requests.RequestException as exc:
                raise UpstreamError(f"échec {ENDPOINT_EMBEDDINGS}: {exc}") from exc

        # Aucune métadonnée de performance sur ce chemin : `/api/embeddings` n'en
        # renvoie pas, et sommer les durées de N appels unitaires produirait une
        # « durée de lot » qu'Ollama n'a jamais mesurée. L'absence est la réponse
        # honnête ; la requête reste un succès (cf. app/metrics.py).
        return EmbedResult(embeddings, ENDPOINT_EMBEDDINGS, fallback_used=True)

    # ---------------------------------------------------------------- ping
    def ping(self, timeout: int = 5) -> Tuple[bool, List[str]]:
        models = self.list_models(timeout=timeout)
        names = [m.get("name") for m in models if m.get("name")]
        return True, names


client = OllamaClient(settings.ollama_base_url, settings.timeout_seconds)
