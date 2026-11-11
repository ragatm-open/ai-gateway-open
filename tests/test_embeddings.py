from app import main
from app.errors import UpstreamError


def _vec(dim=768):
    # Vecteur unitaire (norme L2 = 1,0 exactement). Depuis le lot 1, un vecteur
    # NUL est refusé en 502 : c'est le comportement voulu (un embedding de norme
    # nulle est inexploitable et corrompt un index vectoriel). Seule la donnée
    # de fixture change ici — aucune assertion de ces tests n'est modifiée.
    return [1.0] + [0.0] * (dim - 1)


def test_embeddings_ok(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: [_vec() for _ in inputs],
    )
    r = clientapp.post(
        "/v1/embeddings",
        json={"texts": ["a", "b"], "model": "nomic-embed-text"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "ai_gateway"
    assert body["upstream_provider"] == "ollama"
    assert body["model"] == "nomic-embed-text"
    assert body["dimension"] == 768
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 768
    assert body["usage"]["total_texts"] == 2
    assert body["cost"] == 0.0


def test_embeddings_empty_list(clientapp):
    r = clientapp.post("/v1/embeddings", json={"texts": []})
    assert r.status_code == 400


def test_embeddings_non_string(clientapp):
    r = clientapp.post("/v1/embeddings", json={"texts": [123]})
    assert r.status_code == 400


def test_embeddings_empty_string(clientapp):
    # Texte vide : le validateur custom levait un ValueError non sérialisable
    # -> 500. jsonable_encoder garantit désormais un vrai 400.
    r = clientapp.post("/v1/embeddings", json={"texts": [""]})
    assert r.status_code == 400
    assert r.json()["detail"]["status"] == "error"


def test_embeddings_batch_too_large(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "embeddings_max_batch", 2)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b", "c"]})
    assert r.status_code == 400


def test_embeddings_wrong_dimension(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: [_vec(512) for _ in inputs],
    )
    r = clientapp.post(
        "/v1/embeddings",
        json={"texts": ["a"], "model": "nomic-embed-text"},
    )
    assert r.status_code == 502


def test_embeddings_count_mismatch(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client, "embed", lambda model, inputs, keep_alive=None: [_vec()]
    )
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b"]})
    assert r.status_code == 502


def test_embeddings_upstream_error(clientapp, monkeypatch):
    def boom(model, inputs, keep_alive=None):
        raise UpstreamError("ollama down")

    monkeypatch.setattr(main.client, "embed", boom)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502


def test_embeddings_keep_alive_forwarded(clientapp, monkeypatch):
    captured = {}

    def fake_embed(model, inputs, keep_alive=None):
        captured["keep_alive"] = keep_alive
        return [_vec() for _ in inputs]

    monkeypatch.setattr(main.client, "embed", fake_embed)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "keep_alive": 0})
    assert r.status_code == 200
    assert captured["keep_alive"] == 0
