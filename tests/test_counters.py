"""Lot 2 — classification des erreurs : client / upstream / internal.

`embeddings_errors_total` agrégeait un 400 « batch trop grand » (faute du
client) avec un 502 « Ollama est tombé » (incident d'exploitation), ce qui le
rendait inutilisable pour alerter. Il est conservé comme agrégat.
"""

import pytest

from app import main
from app.errors import UpstreamError
from app.metrics import counters
from app.ollama_client import ENDPOINT_EMBEDDINGS, EmbedResult


def _vec(dim=768, value=1.0):
    return [value] + [0.0] * (dim - 1)


def _stub(monkeypatch, vectors):
    monkeypatch.setattr(
        main.client, "embed", lambda model, inputs, keep_alive=None: vectors
    )


def _c():
    s = counters.snapshot()
    return (
        s["embeddings_client_errors_total"],
        s["embeddings_upstream_errors_total"],
        s["embeddings_internal_errors_total"],
        s["embeddings_errors_total"],
    )


# ------------------------------------------------------------------- client
@pytest.mark.parametrize(
    "payload,stub",
    [
        ({"texts": ["a"] * 65}, [_vec()]),                       # batch trop grand
        ({"texts": [""]}, [_vec()]),                             # texte vide
        ({"texts": ["   "]}, [_vec()]),                          # texte blanc
        ({"texts": []}, [_vec()]),                               # liste vide
        ({"texts": [123]}, [_vec()]),                            # type invalide
        ({"texts": ["a"], "task": "passage"}, [_vec()]),         # task invalide
        ({"texts": ["a" * 6001]}, [_vec()]),                     # dépassement de longueur
    ],
)
def test_client_errors_are_counted_as_client(clientapp, monkeypatch, payload, stub):
    _stub(monkeypatch, stub)
    r = clientapp.post("/v1/embeddings", json=payload)
    assert r.status_code == 400
    client, upstream, internal, total = _c()
    assert (client, upstream, internal) == (1, 0, 0)
    assert total == 1


def test_environment_refusal_is_counted_as_client(clientapp, monkeypatch):
    # fr-en-retrieval est autorisé en production depuis le 2026-08-23 : on
    # force le profil pour exercer le refus d'environnement.
    monkeypatch.setattr(main.registry.resolve("fr-en-retrieval"), "production_allowed", False)
    _stub(monkeypatch, [_vec()])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "fr-en-retrieval"})
    assert r.status_code == 400
    assert _c()[:3] == (1, 0, 0)


def test_ungoverned_model_in_strict_mode_is_a_client_error(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", True)
    _stub(monkeypatch, [_vec()])
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "zz"}).status_code == 400
    assert _c()[:3] == (1, 0, 0)


# ----------------------------------------------------------------- upstream
def test_upstream_unavailable(clientapp, monkeypatch):
    def boom(model, inputs, keep_alive=None):
        raise UpstreamError("échec /api/embed: connexion refusée")

    monkeypatch.setattr(main.client, "embed", boom)
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"]}).status_code == 502
    assert _c()[:3] == (0, 1, 0)


def test_upstream_budget_exhausted(clientapp, monkeypatch):
    def boom(model, inputs, keep_alive=None):
        raise UpstreamError(
            "échec /api/embeddings: budget de temps global épuisé après 3/10 textes (60s)"
        )

    monkeypatch.setattr(main.client, "embed", boom)
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"]}).status_code == 502
    assert _c()[:3] == (0, 1, 0)
    assert counters.snapshot()["embeddings_fallback_budget_exhausted_total"] == 1


@pytest.mark.parametrize(
    "vectors",
    [
        [_vec(), _vec()],                     # cardinalité (1 texte demandé)
        [_vec(512)],                          # mauvaise dimension
        [[float("nan")] + [0.0] * 767],       # NaN
        [[float("inf")] + [0.0] * 767],       # Infinity
        [[0.0] * 768],                        # vecteur nul
        [[]],                                 # vecteur vide
        [_vec(768, value=19.66)],             # norme hors tolérance
    ],
)
def test_bad_vectors_are_counted_as_upstream(clientapp, monkeypatch, vectors):
    _stub(monkeypatch, vectors)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert _c()[:3] == (0, 1, 0)


def test_heterogeneous_dimensions_counted_as_upstream(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec(768), _vec(767)])
    assert clientapp.post("/v1/embeddings", json={"texts": ["a", "b"]}).status_code == 502
    assert _c()[:3] == (0, 1, 0)


def test_unnormalisable_fallback_counted_as_upstream(clientapp, monkeypatch):
    raw = EmbedResult([[0.0] * 768], ENDPOINT_EMBEDDINGS, fallback_used=True)
    monkeypatch.setattr(main.client, "embed", lambda model, inputs, keep_alive=None: raw)
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"]}).status_code == 502
    assert _c()[:3] == (0, 1, 0)


# ----------------------------------------------------------------- internal
def test_unexpected_exception_is_counted_as_internal(clientapp_tolerant, monkeypatch):
    def boom(model, inputs, keep_alive=None):
        raise RuntimeError("bug de programmation")

    monkeypatch.setattr(main.client, "embed", boom)
    r = clientapp_tolerant.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 500
    body = r.json()["detail"]
    assert body["status"] == "error"
    assert body["provider"] == "ai_gateway"
    # Ni la trace, ni le message d'origine ne fuitent.
    assert "bug de programmation" not in r.text
    assert body["message"] == "erreur interne (RuntimeError)"
    assert _c()[:3] == (0, 0, 1)


# ------------------------------------------------------ anomalie non fatale
def test_degraded_model_is_counted_without_failing_the_request(clientapp, monkeypatch):
    """Digest divergent sur le modèle par défaut : la requête passe en mode
    legacy, ce n'est pas une erreur de requête mais bien une anomalie upstream."""
    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [{"name": "nomic-embed-text:latest", "digest": "d" * 64}],
    )
    _stub(monkeypatch, [_vec()])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200
    assert r.json()["governed"] is False
    snap = counters.snapshot()
    assert snap["embeddings_degraded_model_total"] == 1
    assert snap["registry_digest_mismatch_total"] == 1
    assert _c()[:3] == (0, 0, 0)  # aucune requête en échec


# ------------------------------------------------------------------ agrégat
def test_aggregate_is_the_sum_of_the_three_classes(clientapp_tolerant, monkeypatch):
    _stub(monkeypatch, [_vec()])
    clientapp_tolerant.post("/v1/embeddings", json={"texts": ["a"] * 65})          # client
    clientapp_tolerant.post("/v1/embeddings", json={"texts": ["a"], "task": "zz"}) # client

    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: (_ for _ in ()).throw(UpstreamError("ko")),
    )
    clientapp_tolerant.post("/v1/embeddings", json={"texts": ["a"]})               # upstream

    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: (_ for _ in ()).throw(RuntimeError("x")),
    )
    clientapp_tolerant.post("/v1/embeddings", json={"texts": ["a"]})               # internal

    client, upstream, internal, total = _c()
    assert (client, upstream, internal) == (2, 1, 1)
    assert total == client + upstream + internal == 4


def test_successful_request_increments_no_error_counter(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec()])
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"]}).status_code == 200
    assert _c() == (0, 0, 0, 0)


def test_generate_errors_do_not_pollute_embedding_counters(clientapp, monkeypatch):
    """Les compteurs sont ceux du service d'embedding, pas de la génération."""
    r = clientapp.post("/v1/generate", json={"prompt": ""})
    assert r.status_code == 400
    assert _c() == (0, 0, 0, 0)
