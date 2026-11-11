"""Lot 2 — profil Granite `fr-en-retrieval`, autorisé en staging uniquement."""

import pytest

from app import main
from app.metrics import counters
from app.registry import STATUS_DEGRADED

GRANITE_DIGEST = "1a37926bf842899cbf90583e3932f3820548716d9d07661ba622199ffe95c552"


def _vec(dim=768):
    return [1.0] + [0.0] * (dim - 1)


@pytest.fixture
def profile():
    return main.registry.models["fr-en-retrieval"]


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "staging")


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_embed(model, inputs, keep_alive=None):
        seen["model"] = model
        seen["inputs"] = list(inputs)
        return [_vec() for _ in inputs]

    monkeypatch.setattr(main.client, "embed", fake_embed)
    return seen


# --------------------------------------------------------------------- registre
def test_granite_is_in_the_registry(profile):
    assert profile.provider == "ollama"
    assert profile.ollama_name == "granite-embedding:278m"
    assert profile.kind == "embedding"


def test_granite_digest(profile):
    assert profile.digest == GRANITE_DIGEST
    assert profile.short_digest == "1a37926b"


def test_granite_dimension_is_768(profile):
    """Même dimension que legacy-768 : aucune migration pgvector nécessaire."""
    assert profile.dimension == 768
    assert profile.dimension == main.registry.models["legacy-768"].dimension


def test_granite_languages(profile):
    assert profile.languages == ["fr", "en"]
    assert profile.cross_lingual is True


def test_granite_has_no_prefix(profile):
    """L'éditeur n'en recommande aucun, et en ajouter dégrade (mesuré)."""
    assert profile.query_prefix == ""
    assert profile.document_prefix == ""
    assert profile.prefix_for("query") == ""
    assert profile.prefix_for("document") == ""


def test_granite_input_window_is_conservative(profile):
    """Fenêtre du modèle : 512 tokens. La troncature silencieuse commence dès
    ~1 400 caractères sur du contenu chiffré : 1 200 laisse une marge."""
    assert profile.max_input_chars == 1200
    assert profile.on_overflow == "reject"
    assert profile.max_batch_size == 64


def test_granite_authorisations(profile):
    # production_allowed: true depuis le 2026-08-23 (allowlist seule).
    assert profile.staging_allowed is True
    assert profile.production_allowed is True
    assert profile.allowed_in("staging") is True
    assert profile.allowed_in("production") is True
    assert profile.allowed_in("recette") is False


def test_granite_pipeline_ids(profile):
    # Lot P1 : identité canonique (SHA-256 de la transformation), distincte par task.
    import re
    assert re.match(r"^emb-v2:[0-9a-f]{32}$", profile.pipeline_id("query"))
    assert re.match(r"^emb-v2:[0-9a-f]{32}$", profile.pipeline_id("document"))
    assert profile.pipeline_id("query") != profile.pipeline_id("document")


def test_legacy_profile_untouched():
    legacy = main.registry.models["legacy-768"]
    assert legacy.ollama_name == "nomic-embed-text"
    assert legacy.digest.startswith("0a109f42")
    assert legacy.dimension == 768
    assert legacy.query_prefix == "search_query: "
    assert legacy.max_input_chars == 6000
    assert legacy.production_allowed is True


def test_default_model_is_still_nomic():
    assert main.settings.embedding_model == "nomic-embed-text"
    assert main.registry.resolve(main.settings.embedding_model).logical_name == "legacy-768"


# ------------------------------------------------------------------ staging OK
def test_granite_query_in_staging(clientapp, captured, staging):
    r = clientapp.post(
        "/v1/embeddings",
        json={"model": "fr-en-retrieval", "task": "query", "texts": ["contrôles qualité"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["governed"] is True
    assert body["model_logical"] == "fr-en-retrieval"
    assert body["model"] == "granite-embedding:278m"
    assert body["model_digest"] == GRANITE_DIGEST
    assert body["dimension"] == 768
    assert body["normalized"] is True
    assert body["task"] == "query"
    assert body["truncated"] == [False]
    assert body["upstream_endpoint"] == "/api/embed"
    assert body["pipeline_id"] == main.registry.models["fr-en-retrieval"].pipeline_id("query")
    # Aucun préfixe n'est posé sur Granite.
    assert captured["inputs"] == ["contrôles qualité"]


def test_granite_document_in_staging(clientapp, captured, staging):
    r = clientapp.post(
        "/v1/embeddings",
        json={"model": "fr-en-retrieval", "task": "document", "texts": ["un passage"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "document"
    assert body["pipeline_id"] == main.registry.models["fr-en-retrieval"].pipeline_id("document")
    assert captured["inputs"] == ["un passage"]


def test_granite_without_task_defaults_to_document(clientapp, captured, staging):
    r = clientapp.post("/v1/embeddings", json={"model": "fr-en-retrieval", "texts": ["x"]})
    assert r.json()["task"] == "document"


def test_granite_by_ollama_alias(clientapp, captured, staging):
    r = clientapp.post(
        "/v1/embeddings", json={"model": "granite-embedding:278m", "texts": ["x"]}
    )
    assert r.status_code == 200
    assert r.json()["model_logical"] == "fr-en-retrieval"
    assert captured["model"] == "granite-embedding:278m"


def test_granite_by_ollama_alias_with_latest(clientapp, captured, staging):
    """Si un jour Ollama rapporte `granite-embedding:278m:latest`, la résolution
    ne doit pas se casser : seule la forme sans tag gagne un alias `:latest`."""
    profile = main.registry.models["fr-en-retrieval"]
    assert profile.matches_ollama_name("granite-embedding:278m")
    assert profile.matches_ollama_name("granite-embedding")


# ---------------------------------------------- production : autorisé (2026-08-23)
# Lot AIGW-AUTHORIZE-FR-EN-RETRIEVAL-PROD : allowlist seule. L'environnement de
# test par défaut est production.
def test_granite_allowed_in_production_document_and_query(clientapp, captured):
    for task in ("document", "query"):
        r = clientapp.post(
            "/v1/embeddings",
            json={"model": "fr-en-retrieval", "task": task, "texts": ["x"]},
        )
        assert r.status_code == 200, task
        body = r.json()
        assert body["model_logical"] == "fr-en-retrieval"
        assert body["pipeline_id"] == main.registry.models["fr-en-retrieval"].pipeline_id(task)
        assert len(body["embeddings"][0]) == 768
        assert captured["model"] == "granite-embedding:278m"  # pas de repli legacy-768
    assert counters.snapshot().get("embeddings_registry_rejected_total", 0) == 0


def test_granite_allowed_in_production_by_alias(clientapp, captured):
    r = clientapp.post(
        "/v1/embeddings", json={"model": "granite-embedding:278m", "texts": ["x"]}
    )
    assert r.status_code == 200
    assert r.json()["model_logical"] == "fr-en-retrieval"


def test_granite_contract_unchanged_by_authorization():
    """Autoriser ≠ modifier : seules les autorisations ont bougé."""
    profile = main.registry.resolve("fr-en-retrieval")
    assert profile.production_allowed is True and profile.staging_allowed is True
    assert profile.dimension == 768
    assert profile.max_input_chars == 1200
    assert profile.on_overflow == "reject"
    assert profile.ollama_name == "granite-embedding:278m"
    assert profile.digest == "1a37926bf842899cbf90583e3932f3820548716d9d07661ba622199ffe95c552"
    assert profile.query_prefix == "" and profile.document_prefix == ""
    # legacy-768 reste le défaut de production, inchangé.
    legacy = main.registry.resolve("legacy-768")
    assert legacy.production_allowed is True and legacy.max_input_chars == 6000
    assert main.settings.embedding_model == "nomic-embed-text"


def test_environment_allowlist_mechanism_still_enforced(clientapp, captured, monkeypatch):
    """Le MÉCANISME d'allowlist reste intact (profil forcé non autorisé ⇒ 400),
    non contournable par le mode non strict : autoriser fr-en-retrieval n'a
    pas ouvert tous les modèles."""
    profile = main.registry.resolve("fr-en-retrieval")
    monkeypatch.setattr(profile, "production_allowed", False)
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", False)
    for model in ("fr-en-retrieval", "granite-embedding:278m"):
        r = clientapp.post("/v1/embeddings", json={"model": model, "texts": ["x"]})
        assert r.status_code == 400, model
    detail = r.json()["detail"]
    assert "non autorisé" in detail["message"]
    assert detail["model_logical"] == "fr-en-retrieval"
    assert detail["environment"] == "production"
    assert counters.snapshot()["embeddings_registry_rejected_total"] == 2


def _diverge_digests(monkeypatch):
    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [
            {"name": "nomic-embed-text:latest", "digest": "0" * 63 + "a"},
            {"name": "granite-embedding:278m", "digest": "e" * 64},
        ],
    )


def test_environment_refusal_is_not_bypassed_by_a_degraded_digest(clientapp, captured, monkeypatch):
    """Un profil NON autorisé le reste même si son digest diverge (pas de
    retombée en mode legacy pour contourner l'allowlist) — mécanisme inchangé."""
    profile = main.registry.resolve("fr-en-retrieval")
    monkeypatch.setattr(profile, "production_allowed", False)
    _diverge_digests(monkeypatch)
    r = clientapp.post("/v1/embeddings", json={"model": "fr-en-retrieval", "texts": ["x"]})
    assert r.status_code == 400
    assert "non autorisé" in r.json()["detail"]["message"]


def test_degraded_digest_behaviour_unchanged_for_allowed_model(clientapp, captured, monkeypatch):
    """Comportement `degraded` INCHANGÉ par l'autorisation (lot
    AIGW-AUTHORIZE-FR-EN-RETRIEVAL-PROD §19/§27) : digest divergent ⇒
    STRICT_REGISTRY=0 sert en mode non gouverné SANS pipeline_id ;
    STRICT_REGISTRY=1 refuse en 400. C'est le point à concerter avec RagATM."""
    _diverge_digests(monkeypatch)
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", False)
    r = clientapp.post("/v1/embeddings", json={"model": "fr-en-retrieval", "texts": ["x"]})
    assert r.status_code == 200
    assert r.json().get("pipeline_id") is None
    assert counters.snapshot()["embeddings_degraded_model_total"] >= 1

    monkeypatch.setattr(main.settings, "embeddings_strict_registry", True)
    r = clientapp.post("/v1/embeddings", json={"model": "fr-en-retrieval", "texts": ["x"]})
    assert r.status_code == 400
    assert "non gouverné" in r.json()["detail"]["message"]


def test_granite_refused_in_unknown_environment(clientapp, captured, monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "bac-a-sable")
    r = clientapp.post("/v1/embeddings", json={"model": "fr-en-retrieval", "texts": ["x"]})
    assert r.status_code == 400


def test_legacy_still_served_in_production(clientapp, captured):
    """L'ajout de Granite ne change rien au modèle par défaut."""
    r = clientapp.post("/v1/embeddings", json={"texts": ["x"]})
    assert r.status_code == 200
    assert r.json()["model_logical"] == "legacy-768"


# -------------------------------------------------------------------- longueur
def test_granite_1200_chars_accepted(clientapp, captured, staging, profile):
    r = clientapp.post(
        "/v1/embeddings",
        json={"model": "fr-en-retrieval", "texts": ["a" * profile.max_input_chars]},
    )
    assert r.status_code == 200
    assert r.json()["truncated"] == [False]


def test_granite_1201_chars_rejected(clientapp, captured, staging, profile):
    r = clientapp.post(
        "/v1/embeddings",
        json={"model": "fr-en-retrieval", "texts": ["a" * (profile.max_input_chars + 1)]},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["index"] == 0
    assert detail["length"] == 1201
    assert detail["max_input_chars"] == 1200
    assert detail["on_overflow"] == "reject"


def test_granite_overflow_never_leaks_the_text(clientapp, captured, staging):
    secret = "PHRASECONFIDENTIELLE" * 100
    r = clientapp.post(
        "/v1/embeddings", json={"model": "fr-en-retrieval", "texts": [secret]}
    )
    assert r.status_code == 400
    assert "PHRASECONFIDENTIELLE" not in r.text


def test_granite_length_measured_after_transformation(clientapp, captured, staging, profile):
    """Granite n'a aucun préfixe : la longueur mesurée est exactement celle du
    texte reçu — contrairement à legacy-768, dont le préfixe compte."""
    text = "a" * profile.max_input_chars
    r = clientapp.post("/v1/embeddings", json={"model": "fr-en-retrieval", "texts": [text]})
    assert r.status_code == 200
    assert captured["inputs"] == [text]

    legacy = main.registry.models["legacy-768"]
    r = clientapp.post(
        "/v1/embeddings",
        json={"texts": ["a" * legacy.max_input_chars]},
    )
    assert r.status_code == 400  # le préfixe fait dépasser la limite


def test_granite_batch_limit(clientapp, captured, staging, profile):
    r = clientapp.post(
        "/v1/embeddings",
        json={"model": "fr-en-retrieval", "texts": ["x"] * (profile.max_batch_size + 1)},
    )
    assert r.status_code == 400
    assert "batch trop grand" in r.json()["detail"]["message"]


# ------------------------------------------------------------------ diagnostic
def test_diagnostic_lists_granite_ready_in_staging(clientapp, monkeypatch, staging):
    from app.ollama_client import ENDPOINT_EMBED, EmbedResult

    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult([_vec()], ENDPOINT_EMBED),
    )
    body = clientapp.get("/health/embeddings").json()
    assert body["environment"] == "staging"
    entry = {m["model_logical"]: m for m in body["models"]}["fr-en-retrieval"]
    assert entry["state"] == "ready"
    assert entry["allowed_here"] is True
    assert entry["is_default"] is False
    assert entry["measured_dimension"] == 768
    assert entry["expected_digest"] == entry["observed_digest"]


def test_diagnostic_shows_granite_allowed_in_production(clientapp, monkeypatch):
    from app.ollama_client import ENDPOINT_EMBED, EmbedResult

    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult([_vec()], ENDPOINT_EMBED),
    )
    body = clientapp.get("/health/embeddings").json()
    assert body["environment"] == "production"
    entry = {m["model_logical"]: m for m in body["models"]}["fr-en-retrieval"]
    assert entry["allowed_here"] is True
    assert entry["production_allowed"] is True
    assert entry["staging_allowed"] is True
    assert entry["max_input_chars"] == 1200 and entry["dimension"] == 768
    assert entry["pipeline_id_document"] == main.registry.models["fr-en-retrieval"].pipeline_id("document")


def test_diagnostic_shows_granite_digest_divergence(clientapp, monkeypatch, staging):
    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [
            {"name": "nomic-embed-text:latest", "digest": main.registry.models["legacy-768"].digest},
            {"name": "granite-embedding:278m", "digest": "f" * 64},
        ],
    )
    entry = {
        m["model_logical"]: m for m in clientapp.get("/health/embeddings").json()["models"]
    }["fr-en-retrieval"]
    assert entry["state"] == STATUS_DEGRADED
    assert entry["observed_digest"] == "f" * 64


def test_diagnostic_shows_granite_absent(clientapp, monkeypatch, staging):
    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [
            {"name": "nomic-embed-text:latest", "digest": main.registry.models["legacy-768"].digest}
        ],
    )
    entry = {
        m["model_logical"]: m for m in clientapp.get("/health/embeddings").json()["models"]
    }["fr-en-retrieval"]
    assert entry["state"] == "unavailable"
    assert entry["error"] == "modèle absent de l'upstream"


# --------------------------------------------- readiness insensible à Granite
def test_readiness_stays_green_when_granite_is_absent(clientapp, monkeypatch):
    """Granite n'est PAS le modèle par défaut : son absence ne doit pas rendre
    le service non prêt."""
    from app.ollama_client import ENDPOINT_EMBED, EmbedResult

    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [
            {"name": "nomic-embed-text:latest", "digest": main.registry.models["legacy-768"].digest}
        ],
    )
    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult([_vec()], ENDPOINT_EMBED),
    )
    assert clientapp.get("/health/ready").status_code == 200


def test_readiness_stays_green_when_granite_digest_diverges(clientapp, monkeypatch):
    from app.ollama_client import ENDPOINT_EMBED, EmbedResult

    monkeypatch.setattr(
        main.probe_client,
        "list_models",
        lambda timeout=5: [
            {"name": "nomic-embed-text:latest", "digest": main.registry.models["legacy-768"].digest},
            {"name": "granite-embedding:278m", "digest": "f" * 64},
        ],
    )
    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult([_vec()], ENDPOINT_EMBED),
    )
    assert clientapp.get("/health/ready").status_code == 200
