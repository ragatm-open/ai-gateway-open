"""Streaming LLM — POST /v1/generate/stream.

Quatre familles d'invariants, dans l'ordre où ils comptent :

1. **Additivité** — `/v1/generate` et `/v1/embeddings` sont intouchés ; le flux
   vit exclusivement derrière son propre chemin.
2. **Vrai flux** — les fragments sortent au fur et à mesure, dans l'ordre, sans
   que la réponse complète soit bufferisée ni reconstituée en mémoire.
3. **Terminaison maîtrisée** — succès, timeout mural, ligne illisible, rupture
   upstream, déconnexion client : dans TOUS les cas l'upstream est fermé et la
   terminaison est journalisée.
4. **Confidentialité** — les fragments traversent le process mais n'entrent
   jamais dans un log.
"""

import json
import logging
import time

import pytest
import requests

from app import main
from app.errors import InvalidUpstreamChunk, StreamTimeout, UpstreamError
from app.ollama_client import GenerateStream, OllamaClient
from app.schemas import GenerateRequest


def _plan(model="m"):
    """Plan minimal pour tester le générateur seul, hors gouvernance."""
    return main.GenerationPlan(
        upstream_model=model, options=None, think=None, keep_alive=None,
        log_fields={}, inflight_key=f"generate:{model}",
    )

PROMPT_SENTINEL = "ZZSTREAMPROMPTSECRETZZ"
SYSTEM_SENTINEL = "ZZSTREAMSYSTEMSECRETZZ"
TEXT_SENTINEL = "ZZSTREAMGENERATEDTEXTZZ"


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
def _line(**fields) -> bytes:
    """Une ligne NDJSON telle qu'Ollama l'émettrait."""
    return json.dumps(fields).encode("utf-8")


DONE_FIELDS = {
    "response": "",
    "done": True,
    "total_duration": 1_500_000_000,
    "load_duration": 300_000_000,
    "prompt_eval_count": 10,
    "prompt_eval_duration": 200_000_000,
    "eval_count": 50,
    "eval_duration": 1_000_000_000,
}


class _FakeUpstreamResponse:
    """Réponse upstream simulée : lignes NDJSON, `close()` observable."""

    def __init__(self, lines, delay=0.0, on_line=None):
        self._lines = list(lines)
        self._delay = delay
        self._on_line = on_line
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            if self._delay:
                time.sleep(self._delay)
            if self._on_line is not None:
                self._on_line(line)
            yield line

    def raise_for_status(self):
        return None

    def close(self):
        self.closed = True


class _FakeStream:
    """Double de `GenerateStream` au niveau de la route."""

    def __init__(self, events, raises=None, on_event=None):
        self._events = list(events)
        self._raises = raises
        self._on_event = on_event
        self.closed = False

    def events(self):
        for event in self._events:
            if self._on_event is not None:
                self._on_event(event)
            yield event
        if self._raises is not None:
            raise self._raises

    def close(self):
        self.closed = True


def _install_stream(monkeypatch, stream):
    """Branche un flux simulé et retourne les paramètres reçus par le client."""
    captured = {}

    def fake_open(**kwargs):
        captured.update(kwargs)
        return stream

    monkeypatch.setattr(main.client, "open_generate_stream", fake_open)
    return captured


def _request(clientapp, **payload):
    body = {"prompt": "x", **payload}
    return clientapp.post("/v1/generate/stream", json=body)


def _events(response):
    """Décode un corps NDJSON en liste d'objets."""
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _event_line(caplog, name="generate_stream.end"):
    prefix = f"event={name} "
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(prefix):
            return message
    raise AssertionError(
        f"événement {name!r} absent : {[r.getMessage() for r in caplog.records]}"
    )


def _fields(line):
    parts = line.split(" ")[1:]
    return dict(part.split("=", 1) for part in parts if "=" in part)


def _all_logs(caplog):
    return "\n".join(record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- #
# 1. Additivité — le contrat historique ne bouge pas
# --------------------------------------------------------------------------- #
def test_stream_endpoint_is_additive(clientapp, monkeypatch):
    # Les deux routes coexistent et répondent chacune dans son format.
    monkeypatch.setattr(
        main.client,
        "generate",
        lambda **kwargs: {"response": "OK", "done": True,
                          "prompt_eval_count": 1, "eval_count": 1},
    )
    _install_stream(monkeypatch, _FakeStream([{"response": "OK"}, DONE_FIELDS]))

    agrege = clientapp.post("/v1/generate", json={"prompt": "x"})
    flux = _request(clientapp)

    assert agrege.status_code == 200
    assert agrege.headers["content-type"].startswith("application/json")
    assert flux.status_code == 200
    assert flux.headers["content-type"].startswith("application/x-ndjson")


def test_legacy_generate_still_forces_stream_false(monkeypatch):
    calls = []

    class _Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "OK", "done": True}

    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None: (calls.append(json), _Resp())[1],
    )
    OllamaClient("http://127.0.0.1:11434", timeout=30).generate(model="m", prompt="p")
    assert calls[0]["stream"] is False


def test_stream_upstream_uses_stream_true(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None, stream=None):
        calls.append({"json": json, "stream": stream})
        return _FakeUpstreamResponse([_line(**DONE_FIELDS)])

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)
    OllamaClient("http://127.0.0.1:11434", timeout=30).open_generate_stream(
        model="m", prompt="p"
    )
    assert calls[0]["json"]["stream"] is True
    assert calls[0]["stream"] is True


def test_stream_does_not_alter_generate_request_schema(clientapp):
    schema = clientapp.get("/openapi.json").json()
    assert "/v1/generate" in schema["paths"]
    assert "/v1/generate/stream" in schema["paths"]
    # `stream` n'a PAS été ajouté au schéma de requête : le mode vient du chemin.
    assert "stream" not in schema["components"]["schemas"]["GenerateRequest"]["properties"]
    # La réponse agrégée pointe toujours sur GenerateResponse.
    agrege = schema["paths"]["/v1/generate"]["post"]["responses"]["200"]
    assert "GenerateResponse" in json.dumps(agrege)


def test_embeddings_untouched_by_streaming(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: [[1.0] + [0.0] * 767 for _ in inputs],
    )
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200
    assert r.json()["dimension"] == 768


# --------------------------------------------------------------------------- #
# 2. Contenu du flux
# --------------------------------------------------------------------------- #
def test_stream_preserves_delta_order(clientapp, monkeypatch):
    fragments = ["Bonjour", " le", " monde", " !"]
    _install_stream(
        monkeypatch,
        _FakeStream([{"response": f, "done": False} for f in fragments] + [DONE_FIELDS]),
    )
    events = _events(_request(clientapp))
    assert [e["delta"] for e in events if e["type"] == "delta"] == fragments


def test_stream_emits_single_done_event(clientapp, monkeypatch):
    _install_stream(
        monkeypatch, _FakeStream([{"response": "a", "done": False}, DONE_FIELDS])
    )
    events = _events(_request(clientapp))
    assert [e["type"] for e in events].count("done") == 1
    # Le `done` est la DERNIÈRE ligne : rien ne le suit.
    assert events[-1]["type"] == "done"
    assert events[-1]["done"] is True
    assert events[-1]["delta"] == ""


def test_stream_done_contains_usage(clientapp, monkeypatch):
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    done = _events(_request(clientapp))[-1]
    assert done["model"] == main.settings.generate_model
    assert done["tokens"] == {"prompt": 10, "completion": 50, "total": 60}
    assert done["usage"] == {
        "prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60,
    }


def test_stream_usage_matches_non_stream_semantics(clientapp, monkeypatch):
    # Comptes absents : 0 des deux côtés, une seule interprétation des tokens.
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, {"done": True}]))
    done = _events(_request(clientapp))[-1]
    assert done["tokens"] == {"prompt": 0, "completion": 0, "total": 0}

    monkeypatch.setattr(
        main.client, "generate", lambda **kwargs: {"response": "a", "done": True}
    )
    agrege = clientapp.post("/v1/generate", json={"prompt": "x"}).json()
    assert agrege["tokens"] == done["tokens"]


def test_stream_skips_empty_deltas(clientapp, monkeypatch):
    # Un événement de métadonnées seul ne doit pas produire de ligne vide.
    _install_stream(
        monkeypatch,
        _FakeStream([{"response": "", "done": False}, {"response": "a"}, DONE_FIELDS]),
    )
    events = _events(_request(clientapp))
    assert [e["type"] for e in events] == ["delta", "done"]


def test_stream_does_not_expose_thinking_channel(clientapp, monkeypatch):
    # `think` reste inchangé : seul `response` est relayé, jamais un éventuel
    # champ de raisonnement séparé.
    _install_stream(
        monkeypatch,
        _FakeStream(
            [{"response": "visible", "thinking": "RAISONNEMENT_INTERNE"}, DONE_FIELDS]
        ),
    )
    body = _request(clientapp, think=True).text
    assert "visible" in body
    assert "RAISONNEMENT_INTERNE" not in body


def test_stream_relays_final_event_text(clientapp, monkeypatch):
    # Si Ollama place du texte sur l'événement final, il n'est pas perdu.
    final = dict(DONE_FIELDS, response="dernier")
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, final]))
    events = _events(_request(clientapp))
    assert [e["delta"] for e in events if e["type"] == "delta"] == ["a", "dernier"]
    assert events[-1]["type"] == "done"


# --------------------------------------------------------------------------- #
# 3. Non-bufferisation — l'invariant central du lot
# --------------------------------------------------------------------------- #
def test_stream_is_not_buffered():
    """Le consommateur reçoit le 1er fragment AVANT que l'upstream ait fini.

    Un endpoint « streaming » qui accumule tout avant d'émettre serait une
    régression fonctionnelle silencieuse : le contrat HTTP serait respecté,
    mais la latence perçue resterait celle du mode agrégé.

    ⚠️ Ce test s'adresse au générateur, PAS au `TestClient` : le transport ASGI
    d'httpx collecte tout le corps avant de rendre la main, si bien qu'un test
    HTTP en process observerait un buffering qui n'existe pas dans
    l'application et ne pourrait rien prouver dans un sens ni dans l'autre. La
    preuve de bout en bout se fait avec `curl -N` (cf. doc/05-exploitation.md).
    """
    journal = []

    def tracer(event):
        journal.append(
            "produit:done" if event.get("done") else f"produit:{event['response']}"
        )

    stream = _FakeStream(
        [{"response": "un"}, {"response": "deux"}, {"response": "trois"}, DONE_FIELDS],
        on_event=tracer,
    )
    generateur = main._generate_stream_events(
        stream, _plan(), "m", time.time(), time.monotonic()
    )

    for ligne in generateur:
        event = json.loads(ligne)
        journal.append(f"consomme:{event.get('delta') or event['type']}")

    # Si la Gateway bufferisait, toutes les entrées « produit: » précéderaient
    # la première « consomme: ». L'entrelacement prouve la paresse du flux.
    assert journal.index("consomme:un") < journal.index("produit:deux")
    assert journal.index("consomme:deux") < journal.index("produit:trois")
    assert journal.index("consomme:trois") < journal.index("produit:done")


def test_stream_reads_one_upstream_event_per_emitted_line():
    """Rien n'est lu en avance : un fragment consommé = un fragment lu."""
    lus = []
    stream = _FakeStream(
        [{"response": f"f{i}"} for i in range(10)] + [DONE_FIELDS],
        on_event=lambda e: lus.append(e),
    )
    generateur = main._generate_stream_events(
        stream, _plan(), "m", time.time(), time.monotonic()
    )

    next(generateur)
    assert len(lus) == 1
    next(generateur)
    assert len(lus) == 2


def test_stream_does_not_accumulate_response_text(clientapp, monkeypatch):
    # Mémoire constante : la Gateway ne reconstitue jamais le texte complet.
    # 2000 fragments doivent traverser sans être conservés.
    stream = _FakeStream(
        [{"response": f"f{i}"} for i in range(2000)] + [DONE_FIELDS]
    )
    _install_stream(monkeypatch, stream)
    events = _events(_request(clientapp))
    assert len([e for e in events if e["type"] == "delta"]) == 2000
    assert events[-1]["type"] == "done"


# --------------------------------------------------------------------------- #
# 4. Erreurs AVANT le premier octet -> HTTP classique
# --------------------------------------------------------------------------- #
def test_upstream_refused_before_stream_is_a_normal_502(clientapp, monkeypatch):
    def boom(**kwargs):
        raise UpstreamError("échec /api/generate: connexion refusée")

    monkeypatch.setattr(main.client, "open_generate_stream", boom)
    r = _request(clientapp)
    assert r.status_code == 502
    assert r.json()["detail"]["provider"] == "ollama"
    # Surtout PAS un flux contenant un événement d'erreur.
    assert r.headers["content-type"].startswith("application/json")


def test_invalid_payload_before_stream_is_a_normal_400(clientapp):
    r = clientapp.post("/v1/generate/stream", json={"prompt": "   "})
    assert r.status_code == 400
    assert r.json()["detail"]["status"] == "error"


def test_open_stream_http_error_closes_response(monkeypatch):
    closed = {"value": False}

    class _ErrResp:
        status_code = 500
        text = "detail interne upstream"

        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error", response=self)

        def close(self):
            closed["value"] = True

    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None, stream=None: _ErrResp(),
    )
    with pytest.raises(UpstreamError):
        OllamaClient("http://127.0.0.1:11434", timeout=30).open_generate_stream(
            model="m", prompt="p"
        )
    assert closed["value"] is True


# --------------------------------------------------------------------------- #
# 5. Erreurs APRÈS le premier octet -> événement NDJSON
# --------------------------------------------------------------------------- #
def test_stream_invalid_json_emits_controlled_error(clientapp, monkeypatch, logs):
    stream = _FakeStream(
        [{"response": "a"}], raises=InvalidUpstreamChunk("ligne illisible")
    )
    _install_stream(monkeypatch, stream)
    events = _events(_request(clientapp))

    assert [e["type"] for e in events] == ["delta", "error"]
    assert events[-1]["error"]["code"] == "invalid_upstream_response"
    # Pas de `done` de complaisance après une erreur.
    assert not any(e["type"] == "done" for e in events)
    assert stream.closed is True
    assert _fields(_event_line(logs))["termination"] == "invalid_upstream_chunk"


def test_stream_upstream_disconnect_before_done_is_error(clientapp, monkeypatch, logs):
    # Le socket se ferme sans jamais annoncer done=true : c'est une erreur, pas
    # un succès tronqué.
    stream = _FakeStream([{"response": "a"}, {"response": "b"}])
    _install_stream(monkeypatch, stream)
    events = _events(_request(clientapp))

    assert [e["type"] for e in events] == ["delta", "delta", "error"]
    assert events[-1]["error"]["code"] == "upstream_error"
    assert stream.closed is True
    assert _fields(_event_line(logs))["termination"] == "upstream_error"


def test_stream_error_after_several_chunks_keeps_deltas(clientapp, monkeypatch):
    stream = _FakeStream(
        [{"response": "A"}, {"response": "B"}], raises=UpstreamError("rupture")
    )
    _install_stream(monkeypatch, stream)
    events = _events(_request(clientapp))
    assert [e.get("delta") for e in events if e["type"] == "delta"] == ["A", "B"]
    assert events[-1]["type"] == "error"


def test_stream_error_never_leaks_upstream_body(clientapp, monkeypatch, logs):
    # Le message upstream peut porter un fragment de corps : il n'est pas relayé.
    stream = _FakeStream(
        [{"response": "a"}],
        raises=UpstreamError(f"upstream a renvoyé {TEXT_SENTINEL}"),
    )
    _install_stream(monkeypatch, stream)
    body = _request(clientapp).text
    assert TEXT_SENTINEL not in body
    assert TEXT_SENTINEL not in _all_logs(logs)


# --------------------------------------------------------------------------- #
# 6. Budget mural et fermeture upstream
# --------------------------------------------------------------------------- #
def test_stream_deadline_already_expired_raises(monkeypatch):
    upstream = _FakeUpstreamResponse([_line(response="a"), _line(**DONE_FIELDS)])
    stream = GenerateStream(upstream, deadline=time.monotonic() - 1)
    with pytest.raises(StreamTimeout):
        list(stream.events())


def test_stream_global_budget_cuts_a_slow_but_chatty_upstream():
    """Le cas qui distingue un budget mural d'un simple read timeout.

    L'upstream émet régulièrement — aucun timeout de lecture ne se déclencherait
    jamais — mais la durée totale dépasse le plafond de la Gateway.
    """
    upstream = _FakeUpstreamResponse(
        [_line(response=f"f{i}") for i in range(20)] + [_line(**DONE_FIELDS)],
        delay=0.01,
    )
    stream = GenerateStream(upstream, deadline=time.monotonic() + 0.05)

    recus = []
    with pytest.raises(StreamTimeout):
        for event in stream.events():
            recus.append(event)

    # Quelques événements sont bien passés avant la coupure : le flux
    # fonctionnait, c'est bien le budget global qui l'a arrêté.
    assert 0 < len(recus) < 21


def test_stream_timeout_closes_upstream(clientapp, monkeypatch, logs):
    stream = _FakeStream([{"response": "a"}], raises=StreamTimeout("budget épuisé"))
    _install_stream(monkeypatch, stream)
    events = _events(_request(clientapp))

    assert events[-1]["type"] == "error"
    assert events[-1]["error"]["code"] == "upstream_timeout"
    assert stream.closed is True
    assert _fields(_event_line(logs))["termination"] == "timeout"


def test_stream_closes_upstream_on_success(clientapp, monkeypatch):
    stream = _FakeStream([{"response": "a"}, DONE_FIELDS])
    _install_stream(monkeypatch, stream)
    _request(clientapp)
    assert stream.closed is True


def test_generate_stream_close_is_tolerant(caplog):
    # Fermer ne doit jamais masquer la cause d'un incident.
    class _Hostile:
        def close(self):
            raise RuntimeError("socket déjà morte")

    GenerateStream(_Hostile(), deadline=time.monotonic() + 10).close()


def test_stream_ignores_blank_upstream_lines():
    upstream = _FakeUpstreamResponse(
        [b"", _line(response="a"), b"", _line(**DONE_FIELDS)]
    )
    stream = GenerateStream(upstream, deadline=time.monotonic() + 10)
    assert len(list(stream.events())) == 2


def test_stream_rejects_non_object_upstream_line():
    upstream = _FakeUpstreamResponse([b'"juste une chaine"'])
    stream = GenerateStream(upstream, deadline=time.monotonic() + 10)
    with pytest.raises(InvalidUpstreamChunk):
        list(stream.events())


def test_stream_invalid_line_never_appears_in_the_error(caplog):
    caplog.set_level(logging.DEBUG)
    upstream = _FakeUpstreamResponse([f'{{"response": {TEXT_SENTINEL}'.encode()])
    stream = GenerateStream(upstream, deadline=time.monotonic() + 10)
    with pytest.raises(InvalidUpstreamChunk) as excinfo:
        list(stream.events())
    assert TEXT_SENTINEL not in str(excinfo.value)
    assert TEXT_SENTINEL not in caplog.text


# --------------------------------------------------------------------------- #
# 7. Déconnexion client
# --------------------------------------------------------------------------- #
def test_stream_client_disconnect_closes_upstream(logs):
    """Fermer le consommateur ferme l'upstream et qualifie la terminaison.

    Test du générateur lui-même : c'est lui qui porte l'invariant, et le
    fermer reproduit exactement ce que fait Starlette quand le client s'en va.
    """
    stream = _FakeStream(
        [{"response": "a"}, {"response": "b"}, {"response": "c"}, DONE_FIELDS]
    )
    generateur = main._generate_stream_events(
        stream, _plan(), "m", time.time(), time.monotonic()
    )

    premier = next(generateur)
    assert b'"delta":"a"' in premier
    assert stream.closed is False

    generateur.close()

    assert stream.closed is True
    assert _fields(_event_line(logs))["termination"] == "client_disconnect"


def test_stream_client_disconnect_stops_reading_upstream(logs):
    lus = []
    stream = _FakeStream(
        [{"response": f"f{i}"} for i in range(100)] + [DONE_FIELDS],
        on_event=lambda e: lus.append(e),
    )
    generateur = main._generate_stream_events(
        stream, _plan(), "m", time.time(), time.monotonic()
    )
    next(generateur)
    generateur.close()

    # La lecture s'arrête net : on n'a pas drainé les 100 fragments.
    assert len(lus) == 1
    assert stream.closed is True


# Pas de test de déconnexion « au niveau HTTP » ici : le transport ASGI d'httpx
# consomme l'application jusqu'au bout avant de rendre la main, si bien
# qu'abandonner la lecture côté TestClient n'interrompt rien. Un tel test
# passerait pour une mauvaise raison — le flux étant déjà terminé — et donnerait
# une fausse assurance. L'invariant réel est porté par les deux tests
# ci-dessus (fermeture du générateur, exactement ce que fait Starlette quand le
# client s'en va) et vérifié en conditions réelles en staging (cf. rapport).


# --------------------------------------------------------------------------- #
# 8. Observabilité
# --------------------------------------------------------------------------- #
def test_stream_logs_completion_metadata(clientapp, monkeypatch, logs):
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    _request(clientapp, model="llama3.1:8b")

    fields = _fields(_event_line(logs))
    assert fields["model"] == "llama3.1:8b"
    assert fields["termination"] == "completed"
    assert fields["ollama_total_ms"] == "1500.0"
    assert fields["ollama_load_ms"] == "300.0"
    assert fields["ollama_prompt_eval_ms"] == "200.0"
    assert fields["ollama_eval_ms"] == "1000.0"
    assert fields["prompt_tokens"] == "10"
    assert fields["completion_tokens"] == "50"
    assert fields["completion_tokens_per_second"] == "50.0"
    assert fields["total_ms"].isdigit()


def test_stream_reuses_generation_metric_normalizer(clientapp, monkeypatch, logs):
    """Preuve directe de réutilisation : le normaliseur du lot précédent est
    appelé, avec l'événement FINAL du flux."""
    appels = []
    vrai = main.normalize_generation_metrics

    def espion(raw):
        appels.append(raw)
        return vrai(raw)

    monkeypatch.setattr(main, "normalize_generation_metrics", espion)
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    _request(clientapp)

    assert appels == [DONE_FIELDS]
    assert _fields(_event_line(logs))["ollama_eval_ms"] == "1000.0"


def test_stream_missing_metrics_still_succeeds(clientapp, monkeypatch, logs):
    _install_stream(
        monkeypatch, _FakeStream([{"response": "a"}, {"done": True}])
    )
    events = _events(_request(clientapp))
    assert events[-1]["type"] == "done"

    fields = _fields(_event_line(logs))
    assert fields["termination"] == "completed"
    assert "ollama_total_ms" not in fields
    assert "completion_tokens_per_second" not in fields
    assert "total_ms" in fields


def test_stream_logs_first_chunk_ms(clientapp, monkeypatch, logs):
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    _request(clientapp)
    assert int(_fields(_event_line(logs))["first_chunk_ms"]) >= 0


def test_first_chunk_ms_waits_for_a_non_empty_delta(logs):
    """Un événement upstream vide ne déclenche pas la mesure.

    Le délai réel entre les deux événements donne une borne inférieure
    vérifiable : si la mesure partait du premier événement (vide), elle serait
    nettement plus petite.
    """
    def evenements():
        yield {"response": "", "done": False}
        time.sleep(0.05)
        yield {"response": "premier texte", "done": False}
        yield DONE_FIELDS

    stream = _FakeStream([])
    stream.events = evenements
    generateur = main._generate_stream_events(
        stream, _plan(), "m", time.time(), time.monotonic()
    )
    list(generateur)

    first_chunk_ms = int(_fields(_event_line(logs))["first_chunk_ms"])
    assert first_chunk_ms >= 45


def test_no_first_chunk_ms_when_no_text_was_emitted(clientapp, monkeypatch, logs):
    _install_stream(monkeypatch, _FakeStream([{"response": ""}, DONE_FIELDS]))
    _request(clientapp)
    assert "first_chunk_ms" not in _fields(_event_line(logs))


def test_stream_num_ctx_logging_matches_non_stream(clientapp, monkeypatch, logs):
    options = {"num_ctx": 8192}
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    _request(clientapp, options=options)
    flux = _fields(_event_line(logs))["num_ctx"]

    logs.clear()
    monkeypatch.setattr(
        main.client,
        "generate",
        lambda **kwargs: {"response": "a", "done": True},
    )
    clientapp.post("/v1/generate", json={"prompt": "x", "options": options})
    agrege = _fields(_event_line(logs, "generate.ok"))["num_ctx"]

    assert flux == agrege == "8192"


def test_stream_rejects_non_numeric_num_ctx_and_never_logs_it(clientapp, monkeypatch, logs):
    # Lot P4 : une valeur non entière n'est plus ignorée en silence (le défaut
    # du profil s'appliquait) mais REFUSÉE avant l'ouverture du flux. L'invariant
    # de confidentialité reste entier : la valeur client n'entre dans aucun log.
    captured = _install_stream(
        monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS])
    )
    r = _request(clientapp, options={"num_ctx": TEXT_SENTINEL})
    assert r.status_code == 400
    assert captured == {}                      # upstream jamais ouvert
    assert "event=generate_stream.end" not in _all_logs(logs)
    assert TEXT_SENTINEL not in _all_logs(logs)


# --------------------------------------------------------------------------- #
# 8 bis. Rupture de transport EN COURS de flux
# --------------------------------------------------------------------------- #
# Avant : `iter_lines()` levait une `requests.RequestException` brute que la
# route ne comprenait pas — 200 déjà engagé, flux tronqué, aucun événement
# d'erreur, ExceptionGroup non gérée. Le client Ollama convertit désormais
# ces erreurs en UpstreamError ; la route fait le reste, sans nouveau code.

_TRIGGER = b"__COUPURE_RESEAU__"


def _broken_upstream(exc):
    """Réponse upstream qui rend un fragment valide puis casse le transport."""
    def on_line(line):
        if line == _TRIGGER:
            raise exc
    return _FakeUpstreamResponse([_line(response="a"), _TRIGGER], on_line=on_line)


@pytest.mark.parametrize("exc", [
    requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead(0 bytes read)"),
    requests.exceptions.ConnectionError("HTTPConnectionPool(host='127.0.0.1', port=11434): connexion coupée"),
    requests.exceptions.ReadTimeout("HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out."),
])
def test_mid_stream_transport_failure_is_reported_in_contract(clientapp, monkeypatch, logs, exc):
    upstream = _broken_upstream(exc)
    stream = GenerateStream(upstream, deadline=time.monotonic() + 30)
    _install_stream(monkeypatch, stream)

    # raise_server_exceptions=True (défaut) : une ExceptionGroup non gérée
    # ferait ÉCHOUER ce test ici même.
    r = _request(clientapp)
    assert r.status_code == 200          # engagé avant la coupure, contrat inchangé
    events = _events(r)

    # Le fragment reçu avant la coupure est bien sorti, puis l'événement
    # d'erreur — et rien d'autre : pas de faux `done`.
    assert [e["delta"] for e in events if e["type"] == "delta"] == ["a"]
    assert [e["type"] for e in events] == ["delta", "error"]
    assert events[-1]["error"]["code"] == "upstream_error"

    # Aucun détail de transport ne fuit : ni URL, ni port, ni texte d'exception.
    assert "127.0.0.1" not in r.text and "11434" not in r.text
    assert "IncompleteRead" not in r.text and "Read timed out" not in r.text

    # Terminaison journalisée et upstream fermé — pas de fuite de connexion.
    assert _fields(_event_line(logs))["termination"] == "upstream_error"
    assert upstream.closed is True


def test_client_converts_transport_failure_but_not_parse_errors():
    """Au niveau du client : RequestException -> UpstreamError (chaînée), et
    une ligne illisible reste InvalidUpstreamChunk — jamais confondues."""
    broken = _broken_upstream(requests.exceptions.ChunkedEncodingError("x"))
    stream = GenerateStream(broken, deadline=time.monotonic() + 30)
    it = stream.events()
    assert next(it) == {"response": "a"}
    with pytest.raises(UpstreamError) as info:
        next(it)
    assert isinstance(info.value.__cause__, requests.exceptions.ChunkedEncodingError)
    assert "127.0.0.1" not in info.value.message
    assert not isinstance(info.value, (StreamTimeout, InvalidUpstreamChunk))

    garbage = GenerateStream(_FakeUpstreamResponse([b"{pas du json"]), deadline=time.monotonic() + 30)
    with pytest.raises(InvalidUpstreamChunk):
        next(garbage.events())


# --------------------------------------------------------------------------- #
# 9. Relais des paramètres de génération
# --------------------------------------------------------------------------- #
def test_stream_keep_alive_forwarding_matches_non_stream(clientapp, monkeypatch):
    captured = _install_stream(
        monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS])
    )
    _request(clientapp, keep_alive=0)
    # 0 est falsy mais signifie « décharge maintenant » : il doit survivre.
    assert captured["keep_alive"] == 0


def test_stream_forwards_all_generation_parameters(clientapp, monkeypatch):
    captured = _install_stream(
        monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS])
    )
    clientapp.post(
        "/v1/generate/stream",
        json={
            "prompt": "p", "model": "llama3.1:8b", "temperature": 0.7,
            "system": "s", "options": {"num_ctx": 4096}, "think": False,
            "keep_alive": "10m",
        },
    )
    # `num_predict` s'ajoute aux options depuis le lot output-bound-hardening :
    # le flux partage le MÊME plan que la route agrégée, donc le même budget
    # gouverné. 3072 = profil chat-bounded de llama-chat (dimensionné sur le
    # trafic de production : p99=1 271, max=2 394 tokens de complétion).
    assert captured == {
        "model": "llama3.1:8b", "prompt": "p", "system": "s",
        "temperature": 0.7,
        "options": {"num_ctx": 4096, "num_predict": 3072},
        "think": False,
        "keep_alive": "10m",
    }


def test_stream_keep_alive_absent_stays_none_for_ungoverned_model(clientapp, monkeypatch):
    # Depuis le lot runtime-hardening, un modèle GOUVERNÉ reçoit le keep_alive
    # de son profil. L'intention historique de ce test — « absence de champ =
    # rien n'est transmis » — reste vraie pour un modèle HORS registre (nom
    # fictif : gemma3:27b est gouverné depuis une version antérieure).
    captured = _install_stream(
        monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS])
    )
    _request(clientapp, model="unregistered-model:latest")
    assert captured["keep_alive"] is None


def test_stream_payload_carries_generation_fields(monkeypatch):
    # Vérifie ce qui part réellement sur le fil, pas seulement l'appel client.
    calls = []

    def fake_post(url, json=None, timeout=None, stream=None):
        calls.append(json)
        return _FakeUpstreamResponse([_line(**DONE_FIELDS)])

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)
    OllamaClient("http://127.0.0.1:11434", timeout=30).open_generate_stream(
        model="m", prompt="p", system="s", temperature=0.5,
        options={"num_ctx": 2048}, think=False, keep_alive=0,
    )
    payload = calls[0]
    assert payload["stream"] is True
    assert payload["system"] == "s"
    assert payload["think"] is False
    assert payload["keep_alive"] == 0
    assert payload["options"] == {"num_ctx": 2048, "temperature": 0.5}


def test_stream_and_non_stream_build_the_same_payload_but_for_stream():
    # Garde-fou anti-divergence : un futur champ ajouté à la génération doit
    # valoir pour les deux modes.
    kwargs = dict(
        model="m", prompt="p", system="s", temperature=0.5,
        options={"num_ctx": 2048}, think=True, keep_alive="10m",
    )
    agrege = OllamaClient._build_generate_payload(stream=False, **kwargs)
    flux = OllamaClient._build_generate_payload(stream=True, **kwargs)
    assert agrege.pop("stream") is False
    assert flux.pop("stream") is True
    assert agrege == flux


# --------------------------------------------------------------------------- #
# 10. Confidentialité
# --------------------------------------------------------------------------- #
def test_stream_logs_do_not_contain_prompt(clientapp, monkeypatch, logs):
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    clientapp.post(
        "/v1/generate/stream",
        json={"prompt": PROMPT_SENTINEL, "system": SYSTEM_SENTINEL},
    )
    everything = _all_logs(logs)
    assert PROMPT_SENTINEL not in everything
    assert SYSTEM_SENTINEL not in everything
    # La longueur reste journalisée, comme en non-stream.
    assert f"prompt_len={len(PROMPT_SENTINEL)}" in everything


def test_stream_logs_do_not_contain_generated_text(clientapp, monkeypatch, logs):
    _install_stream(
        monkeypatch,
        _FakeStream(
            [{"response": TEXT_SENTINEL}, {"response": TEXT_SENTINEL}, DONE_FIELDS]
        ),
    )
    r = _request(clientapp)
    # Les fragments sont bien servis au client…
    assert TEXT_SENTINEL in r.text
    # …et absents de tous les logs.
    assert TEXT_SENTINEL not in _all_logs(logs)


def test_stream_logs_do_not_contain_generated_text_on_disconnect(logs):
    stream = _FakeStream([{"response": TEXT_SENTINEL}] * 10)
    generateur = main._generate_stream_events(
        stream, _plan(), "m", time.time(), time.monotonic()
    )
    next(generateur)
    generateur.close()
    assert TEXT_SENTINEL not in _all_logs(logs)


def test_stream_logs_do_not_contain_ollama_context(clientapp, monkeypatch, logs):
    final = dict(DONE_FIELDS, context=[128009, 271, 5618])
    _install_stream(monkeypatch, _FakeStream([{"response": "a"}, final]))
    _request(clientapp)
    assert "128009" not in _all_logs(logs)


def test_stream_emits_exactly_one_log_line(clientapp, monkeypatch, logs):
    # Une seule ligne finale structurée, pas un événement par fragment.
    _install_stream(
        monkeypatch,
        _FakeStream([{"response": f"f{i}"} for i in range(50)] + [DONE_FIELDS]),
    )
    _request(clientapp)
    lignes = [r for r in logs.records if "event=generate_stream" in r.getMessage()]
    assert len(lignes) == 1
