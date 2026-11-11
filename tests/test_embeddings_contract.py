"""Compatibilité ascendante et nouveaux champs de /v1/embeddings (lot 1)."""

import pytest

from app import main
from app.registry import DEFAULT_TASK

HISTORICAL_FIELDS = (
    "provider",
    "upstream_provider",
    "model",
    "dimension",
    "embeddings",
    "usage",
    "cost",
)


def _vec(dim=768):
    return [1.0] + [0.0] * (dim - 1)


@pytest.fixture
def captured(monkeypatch):
    """Intercepte l'appel upstream et expose les textes réellement envoyés."""
    seen = {}

    def fake_embed(model, inputs, keep_alive=None):
        seen["model"] = model
        seen["inputs"] = list(inputs)
        seen["keep_alive"] = keep_alive
        return [_vec() for _ in inputs]

    monkeypatch.setattr(main.client, "embed", fake_embed)
    return seen


# ------------------------------------------------------- compatibilité API
@pytest.mark.parametrize(
    "payload",
    [
        {"texts": ["texte à encoder"]},
        {"texts": ["texte à encoder"], "model": "nomic-embed-text"},
        {"texts": ["texte à encoder"], "keep_alive": 0},
    ],
)
def test_historical_payloads_still_work(clientapp, captured, payload):
    r = clientapp.post("/v1/embeddings", json=payload)
    assert r.status_code == 200
    body = r.json()
    for field in HISTORICAL_FIELDS:
        assert field in body, f"champ historique disparu : {field}"
    assert body["provider"] == "ai_gateway"
    assert body["upstream_provider"] == "ollama"
    assert body["model"] == "nomic-embed-text"
    assert body["dimension"] == 768
    assert body["usage"]["total_texts"] == 1
    assert body["cost"] == 0.0


def test_new_fields_are_additive(clientapp, captured):
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    body = r.json()
    assert body["governed"] is True
    assert body["model_logical"] == "legacy-768"
    assert body["model_digest"].startswith("0a109f42")
    assert body["pipeline_id"] == main.registry.models["legacy-768"].pipeline_id("document")
    assert body["task"] == DEFAULT_TASK
    assert body["normalized"] is True
    assert body["truncated"] == [False]
    assert body["upstream_endpoint"] == "/api/embed"


def test_logical_name_is_accepted(clientapp, captured):
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "legacy-768"})
    assert r.status_code == 200
    # Le nom logique est traduit vers le nom Ollama avant l'appel upstream.
    assert captured["model"] == "nomic-embed-text"
    assert r.json()["model"] == "nomic-embed-text"


def test_ollama_alias_is_echoed_unchanged(clientapp, captured):
    """Un client qui envoie `:latest` doit le retrouver tel quel."""
    r = clientapp.post(
        "/v1/embeddings", json={"texts": ["a"], "model": "nomic-embed-text:latest"}
    )
    assert r.status_code == 200
    assert captured["model"] == "nomic-embed-text:latest"
    assert r.json()["model"] == "nomic-embed-text:latest"
    assert r.json()["governed"] is True


def test_unknown_extra_field_is_rejected(clientapp, captured):
    """Resserrement INTENTIONNEL (lot vision-1a, contrat CONTRACT-0 §5/§7) :
    les modèles de requête publics sont fail-closed — un champ inconnu est
    refusé en 400 au lieu d'être ignoré en silence (l'ignorance silencieuse
    est exactement le trou `images` de /v1/generate)."""
    r = clientapp.post(
        "/v1/embeddings",
        json={"texts": ["bonjour"], "champ_inconnu": True},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["message"] == "payload invalide"
    assert any(e.get("type") == "extra_forbidden" for e in detail["errors"])

def test_task_absent_uses_document_prefix(clientapp, captured):
    clientapp.post("/v1/embeddings", json={"texts": ["contenu"]})
    assert captured["inputs"] == ["search_document: contenu"]


def test_task_document(clientapp, captured):
    r = clientapp.post("/v1/embeddings", json={"texts": ["contenu"], "task": "document"})
    assert r.status_code == 200
    assert captured["inputs"] == ["search_document: contenu"]
    assert r.json()["task"] == "document"


def test_task_query(clientapp, captured):
    r = clientapp.post("/v1/embeddings", json={"texts": ["contenu"], "task": "query"})
    assert r.status_code == 200
    assert captured["inputs"] == ["search_query: contenu"]
    assert r.json()["task"] == "query"
    assert r.json()["pipeline_id"] == main.registry.models["legacy-768"].pipeline_id("query")


def test_task_invalid_value(clientapp, captured):
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "task": "passage"})
    assert r.status_code == 400
    assert r.json()["detail"]["status"] == "error"


def test_no_double_prefix_when_client_already_prefixed(clientapp, captured):
    r = clientapp.post(
        "/v1/embeddings",
        json={"texts": ["search_query: déjà préfixé"], "task": "query"},
    )
    assert r.status_code == 200
    assert captured["inputs"] == ["search_query: déjà préfixé"]
    from app.metrics import counters

    assert counters.snapshot()["embeddings_double_prefix_avoided_total"] == 1


def test_client_prefix_is_never_stripped(clientapp, captured):
    """On ne retire pas un préfixe fourni par le client, même incohérent."""
    clientapp.post(
        "/v1/embeddings",
        json={"texts": ["search_document: texte"], "task": "query"},
    )
    assert captured["inputs"] == ["search_query: search_document: texte"]


# ----------------------------------------------------------- allowlist
def test_unknown_model_allowed_when_strict_off(clientapp, captured, monkeypatch):
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", False)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "bge-m3"})
    assert r.status_code == 200
    body = r.json()
    assert body["governed"] is False
    assert body["model_logical"] is None
    assert body["pipeline_id"] is None
    assert body["normalized"] is False
    # Aucun préfixe n'est appliqué à un modèle non gouverné.
    assert captured["inputs"] == ["a"]


def test_unknown_model_rejected_when_strict_on(clientapp, captured, monkeypatch):
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", True)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "bge-m3"})
    assert r.status_code == 400
    assert "non gouverné" in r.json()["detail"]["message"]


def test_known_model_still_works_when_strict_on(clientapp, captured, monkeypatch):
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", True)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200


def test_model_not_allowed_in_environment(clientapp, captured, monkeypatch):
    profile = main.registry.models["legacy-768"]
    monkeypatch.setattr(profile, "production_allowed", False)
    monkeypatch.setattr(main.settings, "environment", "production")
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 400
    assert "non autorisé" in r.json()["detail"]["message"]


def test_degraded_model_falls_back_to_legacy_mode(clientapp, captured, monkeypatch):
    # Divergence PROUVÉE : l'upstream annonce un autre digest que le registre.
    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [{"name": "nomic-embed-text:latest", "digest": "d" * 64}],
    )
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200
    body = r.json()
    # Digest divergent : plus de pipeline_id, plus de préfixe, plus de garantie.
    assert body["governed"] is False
    assert body["pipeline_id"] is None
    assert captured["inputs"] == ["a"]
