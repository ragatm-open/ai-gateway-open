"""Lot P1 — preuve que la TRANSFORMATION n'a pas bougé, seul l'identifiant.

Ce module n'importe AUCUN symbole introduit par le lot : il s'exécute tel quel
contre l'application d'avant (HEAD) et d'après. Sur les deux, il doit passer
à l'identique — c'est la démonstration que les textes envoyés à Ollama, les
options, la normalisation, le batching, l'overflow et la sémantique
query/document sont byte-identiques.
"""

import pytest

from app import main
from app.ollama_client import ENDPOINT_EMBED, ENDPOINT_EMBEDDINGS, EmbedResult
from app.registry import STATUS_READY

UNIT = [1.0] + [0.0] * 767
RAW = [3.0] + [4.0] + [0.0] * 766          # norme 5 : repli non normalisé


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def embed(model, inputs, keep_alive=None):
        seen.update({"model": model, "inputs": list(inputs), "keep_alive": keep_alive})
        return EmbedResult([list(UNIT) for _ in inputs], ENDPOINT_EMBED,
                           fallback_used=False, ollama_metadata={})

    monkeypatch.setattr(main.client, "embed", embed)
    return seen


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "staging")


EXPECTED_INPUTS = {
    # (modèle demandé, task, texte) -> texte exactement envoyé à Ollama
    ("legacy-768", "query", "bonjour"): "search_query: bonjour",
    ("legacy-768", "document", "bonjour"): "search_document: bonjour",
    ("nomic-embed-text", None, "bonjour"): "search_document: bonjour",
    ("legacy-768", "query", "search_query: déjà"): "search_query: déjà",   # garde double préfixe
    ("fr-en-retrieval", "query", "contrôles"): "contrôles",
    ("fr-en-retrieval", "document", "un passage"): "un passage",
}


@pytest.mark.parametrize("key", sorted(EXPECTED_INPUTS, key=str))
def test_text_sent_to_ollama_is_unchanged(clientapp, captured, staging, key):
    model, task, text = key
    body = {"model": model, "texts": [text]}
    if task:
        body["task"] = task
    r = clientapp.post("/v1/embeddings", json=body)
    assert r.status_code == 200
    assert captured["inputs"] == [EXPECTED_INPUTS[key]]
    assert captured["keep_alive"] is None
    resp = r.json()
    assert resp["task"] == (task or "document")
    assert resp["normalized"] is True
    assert resp["dimension"] == 768
    assert resp["truncated"] == [False]
    assert resp["upstream_endpoint"] == "/api/embed"
    assert resp["governed"] is True


def test_batch_and_prefix_unchanged(clientapp, captured):
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b", "c"], "task": "query"})
    assert r.status_code == 200
    assert captured["inputs"] == ["search_query: a", "search_query: b", "search_query: c"]
    assert captured["model"] == "nomic-embed-text"
    assert len(r.json()["embeddings"]) == 3


def test_overflow_reject_unchanged(clientapp, captured):
    r = clientapp.post("/v1/embeddings", json={"texts": ["x" * 6001]})
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["max_input_chars"] == 6000 and d["on_overflow"] == "reject" and d["index"] == 0
    assert captured == {}                                   # jamais envoyé
    r = clientapp.post("/v1/embeddings", json={"texts": ["x" * (6000 - len("search_document: "))]})
    assert r.status_code == 200                              # exactement à la limite : accepté


def test_fallback_normalization_unchanged(clientapp, monkeypatch):
    monkeypatch.setattr(main.client, "embed",
                        lambda model, inputs, keep_alive=None: EmbedResult(
                            [list(RAW) for _ in inputs], ENDPOINT_EMBEDDINGS, fallback_used=True))
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200
    body = r.json()
    assert body["upstream_endpoint"] == "/api/embeddings"
    assert body["normalized"] is True
    assert body["embeddings"][0][:2] == pytest.approx([0.6, 0.8])   # 3/5, 4/5


def test_model_and_digest_unchanged():
    reg = main.registry
    assert set(reg.models) == {"legacy-768", "fr-en-retrieval"}
    a, b = reg.models["legacy-768"], reg.models["fr-en-retrieval"]
    assert (a.ollama_name, a.digest, a.dimension, a.normalized, a.on_overflow, a.max_input_chars) == \
        ("nomic-embed-text", "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f", 768, True, "reject", 6000)
    assert (a.query_prefix, a.document_prefix) == ("search_query: ", "search_document: ")
    assert (b.ollama_name, b.digest, b.dimension, b.normalized, b.on_overflow, b.max_input_chars) == \
        ("granite-embedding:278m", "1a37926bf842899cbf90583e3932f3820548716d9d07661ba622199ffe95c552", 768, True, "reject", 1200)
    assert (b.query_prefix, b.document_prefix) == ("", "")


def test_ungoverned_and_degraded_unchanged(clientapp, captured, monkeypatch):
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", False)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "bge-m3"})
    assert r.status_code == 200 and r.json()["governed"] is False
    assert r.json()["pipeline_id"] is None and r.json()["model_digest"] is None
    assert captured["inputs"] == ["a"]
    p = main.registry.models["legacy-768"]
    p.status = "degraded"
    try:
        r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
        assert r.status_code == 200 and r.json()["governed"] is False
        assert r.json()["pipeline_id"] is None
    finally:
        p.status = STATUS_READY
