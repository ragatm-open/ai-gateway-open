"""Conformité Open — embeddings (identité de pipeline), reranking, catalogue."""

import yaml
import pytest

from app import main
from app.errors import UpstreamError, UpstreamModelNotFound
from app.ollama_client import ENDPOINT_EMBED, EmbedResult
from app.registry import load_registry
from tests.open.conftest import COMPACT_REGISTRY


def _vectors(inputs, dim=768):
    return EmbedResult([[1.0] + [0.0] * (dim - 1) for _ in inputs], ENDPOINT_EMBED, ollama_metadata={})


def test_embeddings_identity_fields(open_app, monkeypatch):
    seen = {}

    def fake_embed(model, inputs, keep_alive=None):
        seen.update(model=model, inputs=list(inputs))
        return _vectors(inputs)
    monkeypatch.setattr(main.client, "embed", fake_embed)
    r = open_app.post("/v1/embeddings", json={"texts": ["bonjour", "hello"], "task": "query"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["model"] == "granite-embedding:278m" and b["model_logical"] == "fr-en-retrieval"
    assert b["governed"] is True and b["identity_verified"] is True
    assert b["expected_digest"] == b["model_digest"] == \
        "1a37926bf842899cbf90583e3932f3820548716d9d07661ba622199ffe95c552"
    assert b["pipeline_id"].startswith("emb-v2:") and len(b["pipeline_id"]) == len("emb-v2:") + 32
    assert b["dimension"] == 768 and b["normalized"] is True and b["truncated"] == [False, False]
    assert seen["inputs"] == ["bonjour", "hello"]        # granite : aucun préfixe


def test_embeddings_unregistered_refused_strictly(open_app, monkeypatch):
    monkeypatch.setattr(main.client, "embed", lambda **kw: pytest.fail("Ollama appelé"))
    r = open_app.post("/v1/embeddings", json={"texts": ["x"], "model": "mxbai-embed-large"})
    assert r.status_code == 400 and r.json()["detail"]["strict_registry"] is True


def test_embeddings_model_not_installed(open_app, monkeypatch):
    def not_found(**kw):
        raise UpstreamModelNotFound("x")
    monkeypatch.setattr(main.client, "embed", not_found)
    r = open_app.post("/v1/embeddings", json={"texts": ["x"], "model": "legacy-768"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_not_installed"
    assert r.json()["detail"]["model"] == "nomic-embed-text"


def test_embeddings_dimension_incompatible_refused(open_app, monkeypatch, tmp_path):
    raw = yaml.safe_load(open(COMPACT_REGISTRY, encoding="utf-8"))
    raw["models"]["big-1024"] = dict(raw["models"]["legacy-768"],
                                     ollama_name="mxbai-embed-large", dimension=1024,
                                     digest="a" * 64, compact_role="embeddings-1024-candidate")
    path = tmp_path / "reg.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(main, "registry", load_registry(str(path)))
    monkeypatch.setattr(main.client, "embed", lambda **kw: pytest.fail("Ollama appelé"))
    r = open_app.post("/v1/embeddings", json={"texts": ["x"], "model": "big-1024"})
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "embedding_dimension_incompatible"
    assert (d["model_dimension"], d["expected_dimension"]) == (1024, 768)


def test_embeddings_saturation(open_app, monkeypatch):
    monkeypatch.setattr(main.client, "embed", lambda **kw: _vectors(kw["inputs"]))
    n = 0
    while main._embeddings_semaphore.acquire(blocking=False):
        n += 1
    try:
        r = open_app.post("/v1/embeddings", json={"texts": ["x"]})
        assert r.status_code == 503 and r.json()["detail"]["code"] == "embeddings_saturated"
        assert r.headers["Retry-After"] == "1"
    finally:
        for _ in range(n):
            main._embeddings_semaphore.release()


def test_embeddings_overflow_rejected_not_truncated(open_app, monkeypatch):
    monkeypatch.setattr(main.client, "embed", lambda **kw: pytest.fail("Ollama appelé"))
    r = open_app.post("/v1/embeddings", json={"texts": ["x" * 1201]})
    assert r.status_code == 400 and r.json()["detail"]["on_overflow"] == "reject"


def test_embeddings_upstream_failure_never_leaks_url(open_app, monkeypatch):
    def boom(**kw):
        raise UpstreamError("échec /api/embed: HTTPConnectionPool(host='127.0.0.1', port=11434)")
    monkeypatch.setattr(main.client, "embed", boom)
    r = open_app.post("/v1/embeddings", json={"texts": ["x"]})
    assert r.status_code == 502 and "11434" not in r.text


# ------------------------------------------------------------------ rerank
def test_rerank_ok_shape(open_app, monkeypatch):
    monkeypatch.setattr(main.rerank_client, "rerank",
                        lambda q, texts: [{"index": 0, "score": -5.2}, {"index": 1, "score": 0.7}])
    r = open_app.post("/v1/rerank", json={"query": "capitale", "documents": [
        {"id": "a", "text": "chat"}, {"id": "b", "text": "Paris est la capitale"}]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["model"] == "BAAI/bge-reranker-v2-m3" and b["model_logical"] == "rerank-default"
    assert b["model_revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert [x["id"] for x in b["results"]] == ["b", "a"]
    assert b["score_semantics"] == {"higher_is_better": True, "calibrated": False}


def test_rerank_disabled_is_typed_503(open_app, monkeypatch):
    monkeypatch.setattr(main.settings, "rerank_enabled", False)
    r = open_app.post("/v1/rerank", json={"query": "q", "documents": [{"id": "a", "text": "t"}]})
    assert r.status_code == 503 and "désactivée" in r.json()["detail"]["message"]
    assert open_app.get("/health/rerank").json()["status"] == "disabled"


def test_rerank_backend_outside_loopback_refused(open_app, monkeypatch):
    monkeypatch.setattr(main.settings, "rerank_backend_url", "http://203.0.113.5:8300")
    monkeypatch.setattr(main.rerank_client, "rerank", lambda q, t: pytest.fail("backend appelé"))
    r = open_app.post("/v1/rerank", json={"query": "q", "documents": [{"id": "a", "text": "t"}]})
    assert r.status_code == 503 and "loopback" in r.json()["detail"]["message"]


def test_rerank_unknown_model_refused(open_app):
    r = open_app.post("/v1/rerank", json={"query": "q", "documents": [{"id": "a", "text": "t"}],
                                           "model": "cohere-rerank"})
    assert r.status_code == 400


# ----------------------------------------------------------------- catalogue
def test_models_catalog_marks_registry_identity(open_app):
    r = open_app.get("/v1/models")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()["models"]}
    emb = by_name["granite-embedding:278m"]
    assert emb["model_logical"] == "fr-en-retrieval" and emb["kind"] == "embedding"
    assert emb["registry_status"] == "ready" and emb["compact"]["license"] == "Apache-2.0"
    llm = by_name["qwen2.5:1.5b-instruct"]
    assert llm["model_logical"] == "compact-generation-8" and llm["kind"] == "llm"
    assert llm["compact"]["qualification"] == "experimental"
    assert "qwen2.5:7b-instruct" not in by_name       # enregistré, NON installé : absent de l'inventaire


def test_models_catalog_503_when_runtime_down(open_app, monkeypatch):
    def boom(timeout=5):
        raise UpstreamError("down")
    monkeypatch.setattr(main.client, "list_models", boom)
    r = open_app.get("/v1/models")
    assert r.status_code == 503 and r.json()["detail"]["provider"] == "ollama"
