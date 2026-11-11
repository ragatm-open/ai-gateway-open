"""Readiness fonctionnelle, diagnostic et enrichissement de /v1/models."""

import pytest

from app import main
from app.metrics import counters
from app.ollama_client import ENDPOINT_EMBED, ENDPOINT_EMBEDDINGS, EmbedResult
from app.registry import STATUS_DEGRADED, STATUS_READY

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast


def _unit(dim=768):
    return [1.0] + [0.0] * (dim - 1)


@pytest.fixture
def probe_ok(monkeypatch):
    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult([_unit()], ENDPOINT_EMBED),
    )


# ------------------------------------------------------------------ liveness
def test_health_stays_light_and_compatible(clientapp, monkeypatch):
    """/health ne doit PAS déclencher d'embedding : seulement /api/tags."""
    monkeypatch.setattr(
        main.client, "ping", lambda timeout=5: (True, ["nomic-embed-text:latest"])
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("/health ne doit pas calculer d'embedding")

    monkeypatch.setattr(main.probe_client, "embed", forbidden)

    r = clientapp.get("/health")
    assert r.status_code == 200
    body = r.json()
    for field in (
        "status",
        "provider",
        "ollama_base_url",
        "default_generate_model",
        "default_embedding_model",
        "models",
        "latency_ms",
    ):
        assert field in body
    assert body["ollama_base_url"] == "redacted"
    # Ajout additif.
    assert body["registry"]["default_embedding_logical"] == "legacy-768"


# ----------------------------------------------------------------- readiness
def test_ready_ok(clientapp, probe_ok):
    r = clientapp.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_logical"] == "legacy-768"
    assert body["dimension"] == 768
    assert body["norm"] == pytest.approx(1.0)
    assert body["checks"]["digest_match"] is True
    assert body["checks"]["embedding_call"] is True
    assert body["pipeline_id"] == main.registry.models["legacy-768"].pipeline_id("document")
    assert "redacted" == body["ollama_base_url"]


def test_ready_503_when_model_absent(clientapp, monkeypatch, probe_ok):
    monkeypatch.setattr(main.probe_client, "list_models", lambda timeout=5: [])
    r = clientapp.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["model_installed"] is False


def test_ready_503_when_digest_diverges(clientapp, monkeypatch, probe_ok):
    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [{"name": "nomic-embed-text:latest", "digest": "b" * 64}],
    )
    r = clientapp.get("/health/ready")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["checks"]["digest_match"] is False
    assert detail["registry_status"] == STATUS_DEGRADED
    assert "digest" in detail["message"]


def test_ready_503_on_wrong_dimension(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult([_unit(512)], ENDPOINT_EMBED),
    )
    r = clientapp.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["dimension"] is False


def test_ready_503_on_zero_vector(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult([[0.0] * 768], ENDPOINT_EMBED),
    )
    r = clientapp.get("/health/ready")
    assert r.status_code == 503
    assert "norme nulle" in r.json()["detail"]["message"]


def test_ready_503_when_upstream_unreachable(clientapp, monkeypatch):
    from app.errors import UpstreamError

    def boom(timeout=5):
        raise UpstreamError("connexion refusée")

    monkeypatch.setattr(main.probe_client, "list_models", boom)
    r = clientapp.get("/health/ready")
    assert r.status_code == 503


def test_ready_503_when_default_model_absent_from_registry(clientapp, monkeypatch, probe_ok):
    monkeypatch.setattr(main.settings, "embedding_model", "modele-inconnu")
    r = clientapp.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["registry_profile"] is False


def test_ready_normalises_fallback_before_checking_norm(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult(
            [[3.0, 4.0] + [0.0] * 766], ENDPOINT_EMBEDDINGS
        ),
    )
    r = clientapp.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["norm"] == pytest.approx(1.0)
    assert r.json()["upstream_endpoint"] == ENDPOINT_EMBEDDINGS


# --------------------------------------------------------------------- cache
def test_ready_result_is_cached(clientapp, monkeypatch):
    calls = {"n": 0}

    def counting_embed(model, inputs, keep_alive=None):
        calls["n"] += 1
        return EmbedResult([_unit()], ENDPOINT_EMBED)

    monkeypatch.setattr(main.probe_client, "embed", counting_embed)

    assert clientapp.get("/health/ready").json()["cached"] is False
    for _ in range(5):
        assert clientapp.get("/health/ready").json()["cached"] is True
    assert calls["n"] == 1
    assert counters.snapshot()["readiness_probe_total"] == 1


def test_ready_cache_expires(clientapp, monkeypatch):
    calls = {"n": 0}

    def counting_embed(model, inputs, keep_alive=None):
        calls["n"] += 1
        return EmbedResult([_unit()], ENDPOINT_EMBED)

    monkeypatch.setattr(main.probe_client, "embed", counting_embed)
    monkeypatch.setattr(main.settings, "readiness_cache_seconds", 0)

    clientapp.get("/health/ready")
    clientapp.get("/health/ready")
    assert calls["n"] == 2


def test_ready_concurrent_probes_run_once(clientapp, monkeypatch):
    """Le verrou évite l'effet de troupeau : N sondes simultanées, 1 calcul."""
    import threading

    calls = {"n": 0}
    gate = threading.Event()

    def slow_embed(model, inputs, keep_alive=None):
        calls["n"] += 1
        gate.wait(timeout=2)
        return EmbedResult([_unit()], ENDPOINT_EMBED)

    monkeypatch.setattr(main.probe_client, "embed", slow_embed)

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(clientapp.get("/health/ready").status_code))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(timeout=10)

    assert results == [200] * 6
    assert calls["n"] == 1


# ---------------------------------------------------------------- diagnostic
def test_health_embeddings_diagnostic(clientapp, probe_ok):
    r = clientapp.get("/health/embeddings")
    assert r.status_code == 200
    body = r.json()
    assert body["ollama_base_url"] == "redacted"
    assert body["registry_version"] == 1
    entry = body["models"][0]
    assert entry["model_logical"] == "legacy-768"
    assert entry["ollama_name"] == "nomic-embed-text"
    assert entry["expected_digest"] == entry["observed_digest"]
    assert entry["dimension"] == 768
    assert entry["measured_dimension"] == 768
    assert entry["measured_norm"] == pytest.approx(1.0)
    assert entry["state"] == STATUS_READY
    assert entry["upstream_endpoint"] == ENDPOINT_EMBED
    assert entry["allowed_here"] is True
    assert entry["is_default"] is True
    assert entry["error"] is None
    assert "counters" in body


def test_health_embeddings_reports_degraded(clientapp, monkeypatch, probe_ok):
    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [{"name": "nomic-embed-text:latest", "digest": "c" * 64}],
    )
    r = clientapp.get("/health/embeddings")
    entry = r.json()["models"][0]
    assert entry["state"] == STATUS_DEGRADED
    assert entry["observed_digest"] == "c" * 64


def test_health_embeddings_never_leaks_paths(clientapp, monkeypatch):
    from app.errors import UpstreamError

    def boom(model, inputs, keep_alive=None):
        raise UpstreamError("échec /api/embed: /data/ai/ollama/blobs/sha256-970aa74c")

    monkeypatch.setattr(main.probe_client, "embed", boom)
    r = clientapp.get("/health/embeddings")
    assert r.status_code == 200
    # Le message d'erreur upstream est repris tel quel côté diagnostic ; il ne
    # doit pas contenir l'URL de l'upstream masquée par ailleurs.
    assert "http://127.0.0.1:11434" not in r.text


# ------------------------------------------------------------------- /v1/models
def test_models_exposes_digest_and_details(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "list_models",
        lambda timeout=5: [
            {
                "name": "nomic-embed-text:latest",
                "size": 274302450,
                "modified_at": "2026-07-06T20:07:12Z",
                "digest": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
                "details": {
                    "family": "nomic-bert",
                    "parameter_size": "137M",
                    "quantization_level": "F16",
                    "parent_model": "/data/ai/ollama/blobs/sha256-970aa74c0a90",
                },
                "capabilities": ["embedding"],
            }
        ],
    )
    r = clientapp.get("/v1/models")
    assert r.status_code == 200
    entry = r.json()["models"][0]
    # Champs historiques préservés.
    assert entry["name"] == "nomic-embed-text:latest"
    assert entry["size"] == 274302450
    assert entry["modified_at"] == "2026-07-06T20:07:12Z"
    # Ajouts.
    assert entry["digest"].startswith("0a109f42")
    assert entry["details"]["family"] == "nomic-bert"
    assert entry["model_logical"] == "legacy-768"
    assert entry["kind"] == "embedding"
    assert entry["dimension"] == 768
    assert entry["registry_status"] == STATUS_READY
    # Aucun chemin de blob ne doit fuiter.
    assert "parent_model" not in entry["details"]
    assert "/data/ai/ollama" not in r.text


def test_models_ungoverned_entry_has_no_registry_metadata(clientapp, monkeypatch):
    # Un modèle installé mais absent des DEUX registres (embeddings et LLM).
    # Nom fictif : depuis une version antérieure plus aucun modèle installé n'est hors
    # registre (gemma3:27b est gouverné) — le MÉCANISME reste testé.
    monkeypatch.setattr(
        main.client,
        "list_models",
        lambda timeout=5: [{"name": "unregistered-model:latest", "size": 1, "modified_at": "x"}],
    )
    entry = clientapp.get("/v1/models").json()["models"][0]
    assert "model_logical" not in entry
    assert "registry_status" not in entry
    assert "context_profile" not in entry
