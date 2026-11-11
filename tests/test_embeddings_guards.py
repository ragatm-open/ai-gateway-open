"""Garde-fous du lot 1 : longueur d'entrée et validation des vecteurs."""


import pytest

from app import main
from app.metrics import counters


def _vec(dim=768, value=1.0):
    return [value] + [0.0] * (dim - 1)


@pytest.fixture
def profile():
    return main.registry.models["legacy-768"]


def _stub(monkeypatch, vectors):
    monkeypatch.setattr(
        main.client, "embed", lambda model, inputs, keep_alive=None: vectors
    )


# ------------------------------------------------------------------ longueur
def test_length_under_limit(clientapp, monkeypatch, profile):
    _stub(monkeypatch, [_vec()])
    prefix = len(profile.document_prefix)
    text = "a" * (profile.max_input_chars - prefix - 1)
    assert clientapp.post("/v1/embeddings", json={"texts": [text]}).status_code == 200


def test_length_exactly_at_limit(clientapp, monkeypatch, profile):
    _stub(monkeypatch, [_vec()])
    text = "a" * (profile.max_input_chars - len(profile.document_prefix))
    r = clientapp.post("/v1/embeddings", json={"texts": [text]})
    assert r.status_code == 200


def test_length_over_limit_is_rejected(clientapp, monkeypatch, profile):
    _stub(monkeypatch, [_vec()])
    text = "a" * (profile.max_input_chars + 1)
    r = clientapp.post("/v1/embeddings", json={"texts": [text]})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["max_input_chars"] == profile.max_input_chars
    assert detail["length"] == len(text) + len(profile.document_prefix)
    assert counters.snapshot()["embeddings_overflow_total"] == 1


def test_overflow_error_reports_faulty_index(clientapp, monkeypatch, profile):
    _stub(monkeypatch, [_vec(), _vec()])
    long_text = "b" * (profile.max_input_chars + 10)
    r = clientapp.post("/v1/embeddings", json={"texts": ["court", long_text]})
    assert r.status_code == 400
    assert r.json()["detail"]["index"] == 1


def test_overflow_error_never_leaks_the_text(clientapp, monkeypatch, profile):
    _stub(monkeypatch, [_vec()])
    secret = "MOTIFCONFIDENTIEL" * 500
    r = clientapp.post("/v1/embeddings", json={"texts": [secret]})
    assert r.status_code == 400
    assert "MOTIFCONFIDENTIEL" not in r.text


def test_prefix_counts_towards_the_limit(clientapp, monkeypatch, profile):
    """Le préfixe est appliqué AVANT la mesure : c'est bien le texte envoyé à
    Ollama qui est borné, pas celui reçu du client."""
    _stub(monkeypatch, [_vec()])
    text = "a" * profile.max_input_chars
    r = clientapp.post("/v1/embeddings", json={"texts": [text]})
    assert r.status_code == 400


def test_no_length_limit_for_ungoverned_model(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec(1024)])
    r = clientapp.post(
        "/v1/embeddings", json={"texts": ["a" * 20000], "model": "bge-m3"}
    )
    assert r.status_code == 200


# ------------------------------------------------------------------ vecteurs
def test_good_dimension(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec(768)])
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"]}).status_code == 200


def test_wrong_dimension_governed_model(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec(512)])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert "dimension inattendue" in r.json()["detail"]["message"]
    assert counters.snapshot()["embeddings_dimension_errors_total"] == 1


def test_wrong_dimension_on_governed_model_named_by_alias(clientapp, monkeypatch):
    """Le contrôle ne dépend plus d'une égalité stricte avec le modèle par
    défaut : `nomic-embed-text:latest` est gouverné, donc contrôlé."""
    _stub(monkeypatch, [_vec(512)])
    r = clientapp.post(
        "/v1/embeddings", json={"texts": ["a"], "model": "nomic-embed-text:latest"}
    )
    assert r.status_code == 502
    assert "dimension inattendue" in r.json()["detail"]["message"]


def test_dimension_observed_for_ungoverned_model(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec(1024)])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "bge-m3"})
    assert r.status_code == 200
    assert r.json()["dimension"] == 1024
    assert r.json()["governed"] is False


def test_empty_vector(clientapp, monkeypatch):
    _stub(monkeypatch, [[]])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert "vide" in r.json()["detail"]["message"]


def test_zero_vector(clientapp, monkeypatch):
    _stub(monkeypatch, [[0.0] * 768])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert "norme nulle" in r.json()["detail"]["message"]
    assert counters.snapshot()["embeddings_invalid_vector_total"] == 1


def test_nan_vector(clientapp, monkeypatch):
    _stub(monkeypatch, [[float("nan")] + [0.0] * 767])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert "non finie" in r.json()["detail"]["message"]


def test_infinity_vector(clientapp, monkeypatch):
    _stub(monkeypatch, [[float("inf")] + [0.0] * 767])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert "non finie" in r.json()["detail"]["message"]


def test_heterogeneous_dimensions(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec(768), _vec(767)])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b"]})
    assert r.status_code == 502
    assert "vecteur 1 de dimension" in r.json()["detail"]["message"]


def test_cardinality_mismatch(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec()])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b"]})
    assert r.status_code == 502
    assert "nombre de vecteurs" in r.json()["detail"]["message"]


def test_norm_out_of_tolerance_on_nominal_path(clientapp, monkeypatch):
    """Sur /api/embed, un profil `normalized: true` qui renvoie ‖v‖ ≈ 19 est
    une anomalie upstream, pas quelque chose à rattraper en silence."""
    _stub(monkeypatch, [_vec(768, value=19.66)])
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 502
    assert "hors tolérance" in r.json()["detail"]["message"]


def test_norm_within_tolerance_is_accepted(clientapp, monkeypatch):
    _stub(monkeypatch, [_vec(768, value=1.005)])
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"]}).status_code == 200


def test_batch_limit_uses_the_most_restrictive_bound(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "embeddings_max_batch", 2)
    _stub(monkeypatch, [_vec()] * 3)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b", "c"]})
    assert r.status_code == 400
    assert "batch trop grand (3 > 2)" in r.json()["detail"]["message"]
