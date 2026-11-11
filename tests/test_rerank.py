"""Lot rerank-sidecar-staging — capability /v1/rerank.

Invariants :
1. capability OPTIONNELLE : flag off => 503 explicite, rien d'autre ne change ;
2. gouvernance identique aux embeddings : registre, allowlist environnement
   non contournable, limites fail-closed, jamais de troncature silencieuse ;
3. tri déterministe (score décroissant, départage par index croissant) ;
4. backend fautif => 502, jamais un résultat partiel ;
5. isolation totale : backend mort => generate/embeddings/readiness intacts ;
6. privacy : ni query, ni texte, ni id client dans les logs.
"""

import logging
import os

import pytest

from app import main
from app.errors import UpstreamError
from app.metrics import counters, gauges
from app.registry import RegistryError, load_registry

QUERY_SENTINEL = "ZZRERANKQUERYSECRETZZ"
TEXT_SENTINEL = "ZZRERANKPASSAGESECRETZZ"
ID_SENTINEL = "ZZRERANKIDSECRETZZ"

REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@pytest.fixture
def staging(monkeypatch):
    """Active la capability dans l'environnement staging (le défaut de test
    est production, où rerank-default est INTERDIT — comme granite)."""
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "staging")


@pytest.fixture
def backend(monkeypatch):
    """Backend simulé : scores croissants avec l'index (0.0, 1.0, 2.0…)."""
    calls = {}

    def fake_rerank(query, texts):
        calls["query"] = query
        calls["texts"] = list(texts)
        return [{"index": i, "score": float(i)} for i in range(len(texts))]

    monkeypatch.setattr(main.rerank_client, "rerank", fake_rerank)
    return calls


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _docs(n, prefix="d"):
    return [{"id": f"{prefix}{i}", "text": f"passage numero {i}"} for i in range(n)]


def _post(clientapp, **payload):
    body = {"query": "quelle est la question ?", "documents": _docs(3), **payload}
    return clientapp.post("/v1/rerank", json=body)


def _all_logs(caplog):
    return "\n".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# 1. Flag et disponibilité
# --------------------------------------------------------------------------- #
def test_flag_off_returns_explicit_503(clientapp):
    # Défaut du service : capability désactivée.
    r = _post(clientapp)
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["provider"] == "ai_gateway"
    assert "désactivée" in detail["message"]


def test_non_loopback_backend_disables_capability(clientapp, monkeypatch, backend):
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "staging")
    monkeypatch.setattr(
        main.settings, "rerank_backend_url", "http://203.0.113.5:8300"
    )
    r = _post(clientapp)
    assert r.status_code == 503
    assert "loopback" in r.json()["detail"]["message"]


def test_flag_off_leaves_other_capabilities_untouched(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "generate",
        lambda **kwargs: {"response": "OK", "done": True},
    )
    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: [[1.0] + [0.0] * 767 for _ in inputs],
    )
    assert clientapp.post("/v1/generate", json={"prompt": "x"}).status_code == 200
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"]}).status_code == 200
    assert _post(clientapp).status_code == 503


# --------------------------------------------------------------------------- #
# 2. Contrat nominal
# --------------------------------------------------------------------------- #
def test_valid_rerank(clientapp, staging, backend):
    r = _post(clientapp)
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "ai_gateway"
    assert body["upstream_provider"] == "local_cross_encoder"
    assert body["model"] == "BAAI/bge-reranker-v2-m3"
    assert body["model_logical"] == "rerank-default"
    assert body["model_revision"] == REVISION
    assert body["score_semantics"] == {"higher_is_better": True, "calibrated": False}
    assert body["usage"] == {"documents": 3, "returned": 3}
    # UN appel backend pour N documents — jamais N appels.
    assert backend["texts"] == [d["text"] for d in _docs(3)]


def test_ordering_higher_score_first(clientapp, staging, backend):
    body = _post(clientapp).json()
    # Scores simulés croissants avec l'index => ordre inverse attendu.
    assert [entry["index"] for entry in body["results"]] == [2, 1, 0]
    assert [entry["id"] for entry in body["results"]] == ["d2", "d1", "d0"]
    scores = [entry["relevance_score"] for entry in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_stable_tie_break_by_original_index(clientapp, staging, monkeypatch):
    monkeypatch.setattr(
        main.rerank_client,
        "rerank",
        lambda query, texts: [{"index": i, "score": 1.0} for i in range(len(texts))],
    )
    body = _post(clientapp).json()
    # Égalité parfaite : l'ordre d'origine est conservé, déterministe.
    assert [entry["index"] for entry in body["results"]] == [0, 1, 2]


def test_top_n_slices_after_sorting(clientapp, staging, backend):
    body = _post(clientapp, top_n=2).json()
    assert [entry["index"] for entry in body["results"]] == [2, 1]
    assert body["usage"] == {"documents": 3, "returned": 2}


def test_top_n_absent_returns_all(clientapp, staging, backend):
    assert _post(clientapp).json()["usage"]["returned"] == 3


def test_ids_preserved_verbatim(clientapp, staging, backend):
    r = clientapp.post(
        "/v1/rerank",
        json={
            "query": "q",
            "documents": [
                {"id": "chunk::42/v1", "text": "alpha"},
                {"id": "chunk::17/v2", "text": "beta"},
            ],
        },
    )
    assert {e["id"] for e in r.json()["results"]} == {"chunk::42/v1", "chunk::17/v2"}


def test_two_identical_calls_identical_results(clientapp, staging, backend):
    first = _post(clientapp).json()
    second = _post(clientapp).json()
    assert first["results"] == second["results"]


# --------------------------------------------------------------------------- #
# 3. Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "documents": _docs(2)},
        {"query": "   ", "documents": _docs(2)},
        {"query": "q", "documents": []},
        {"query": "q", "documents": [{"id": "a", "text": "  "}]},
        {"query": "q", "documents": [{"id": "", "text": "x"}]},
        {"query": "q", "documents": [{"id": "a" * 129, "text": "x"}]},
        {"query": "q", "documents": _docs(2), "top_n": 0},
        {"query": "q", "documents": _docs(2), "top_n": -1},
    ],
)
def test_invalid_payloads_are_400(clientapp, staging, backend, payload):
    r = clientapp.post("/v1/rerank", json=payload)
    assert r.status_code == 400
    assert r.json()["detail"]["status"] == "error"


def test_top_n_above_document_count_is_400(clientapp, staging, backend):
    r = _post(clientapp, top_n=4)
    assert r.status_code == 400
    assert "top_n" in r.json()["detail"]["message"]


def test_unknown_model_is_400(clientapp, staging, backend):
    r = _post(clientapp, model="rerank-inconnu")
    assert r.status_code == 400
    assert "inconnu" in r.json()["detail"]["message"]


def test_too_many_documents_rejected(clientapp, staging, backend):
    r = clientapp.post(
        "/v1/rerank", json={"query": "q", "documents": _docs(65)}
    )
    assert r.status_code == 400
    assert counters.snapshot()["rerank_rejected_oversize_total"] == 1


def test_oversized_query_rejected(clientapp, staging, backend):
    r = clientapp.post(
        "/v1/rerank", json={"query": "q" * 2001, "documents": _docs(2)}
    )
    assert r.status_code == 400
    assert "query trop longue" in r.json()["detail"]["message"]


def test_oversized_document_rejected_with_index_not_content(clientapp, staging, backend):
    docs = _docs(3)
    docs[1]["text"] = "x" * 4001
    r = clientapp.post("/v1/rerank", json={"query": "q", "documents": docs})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["index"] == 1
    assert detail["length"] == 4001
    assert "x" * 50 not in r.text


def test_oversized_total_rejected(clientapp, staging, backend):
    docs = [{"id": f"d{i}", "text": "y" * 4000} for i in range(41)]
    r = clientapp.post("/v1/rerank", json={"query": "q", "documents": docs})
    assert r.status_code == 400
    assert "volumineuse" in r.json()["detail"]["message"]


# --------------------------------------------------------------------------- #
# 4. Gouvernance registre
# --------------------------------------------------------------------------- #
def test_environment_denied_when_profile_disallows(clientapp, monkeypatch, backend):
    """Le MÉCANISME d'allowlist reste intact : un profil non autorisé dans
    l'environnement est refusé en 400, flag ou pas. Testé en forçant le
    profil (le registre livré autorise la production depuis le lot
    rerank-production-capability-readiness)."""
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    profile = main.registry.resolve_reranker("rerank-default")
    monkeypatch.setattr(profile, "production_allowed", False)
    r = _post(clientapp)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["model_logical"] == "rerank-default"
    assert detail["environment"] == "production"


def test_shipped_registry_serves_production(clientapp, monkeypatch, backend):
    """Politique livrée (lot rerank-production-capability-readiness) : la
    Gateway production SERT /v1/rerank — la frontière d'activation est côté
    RagATM, pas ici. Environnement par défaut des tests = production."""
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    r = _post(clientapp)
    assert r.status_code == 200
    assert r.json()["model_logical"] == "rerank-default"


def test_shipped_registry_pins_exact_revision():
    profile = main.registry.resolve_reranker("rerank-default")
    assert profile is not None
    assert profile.artifact == "BAAI/bge-reranker-v2-m3"
    assert profile.revision == REVISION
    # true depuis le lot rerank-production-capability-readiness (qualité,
    # cohabitation GPU et provenance qualifiées en staging au préalable).
    assert profile.production_allowed is True
    assert profile.staging_allowed is True
    assert profile.on_overflow == "reject"


def _minimal_registry(tmp_path, rerank_extra=""):
    body = f"""
version: 1
models:
  emb:
    provider: ollama
    ollama_name: nomic-embed-text
    digest: "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
    kind: embedding
    dimension: 768
    max_batch_size: 64
    max_input_chars: 6000
    on_overflow: reject
    production_allowed: true
    staging_allowed: true
reranker_models:
  rr:
    provider: local_cross_encoder
    artifact: BAAI/bge-reranker-v2-m3
    revision: "{REVISION}"
    max_documents_per_request: 64
    max_query_chars: 2000
    max_document_chars: 4000
    max_total_chars: 160000
    on_overflow: reject
    production_allowed: false
    staging_allowed: true
{rerank_extra}"""
    path = tmp_path / "r.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_registry_parses_reranker_section(tmp_path):
    reg = load_registry(_minimal_registry(tmp_path))
    profile = reg.resolve_reranker("rr")
    assert profile.revision == REVISION
    assert reg.resolve_reranker("absent") is None


@pytest.mark.parametrize(
    "before,after,fragment",
    [
        (f'revision: "{REVISION}"', 'revision: "main"', "revision mal formée"),
        ("provider: local_cross_encoder", "provider: ollama", "provider non supporté"),
        ("on_overflow: reject", "on_overflow: truncate_and_flag", "reject"),
    ],
)
def test_registry_rejects_bad_reranker(tmp_path, before, after, fragment):
    path = _minimal_registry(tmp_path)
    content = open(path).read().replace(before, after)
    open(path, "w").write(content)
    with pytest.raises(RegistryError) as excinfo:
        load_registry(path)
    assert fragment in str(excinfo.value)


def test_reranker_name_colliding_with_embedding_is_rejected(tmp_path):
    path = _minimal_registry(tmp_path)
    content = open(path).read().replace("  rr:", "  emb:", 1)
    open(path, "w").write(content)
    with pytest.raises(RegistryError):
        load_registry(path)


def test_reranker_name_colliding_with_llm_is_rejected():
    # Le registre embarqué contient qwen-judge (LLM) : un reranker du même nom
    # doit être refusé — vérifié sur la logique de collision inter-registres.
    import textwrap
    import tempfile, os
    from app.registry import load_registry as load

    body = open(main.settings.models_registry_path).read().replace(
        "  rerank-default:", "  qwen-judge:", 1
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        with pytest.raises(RegistryError) as excinfo:
            load(path)
        assert "entre registres" in str(excinfo.value)
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------- #
# 5. Backend fautif
# --------------------------------------------------------------------------- #
def test_backend_error_is_502(clientapp, staging, monkeypatch):
    def boom(query, texts):
        raise UpstreamError("échec /rerank backend: HTTP 500")

    monkeypatch.setattr(main.rerank_client, "rerank", boom)
    r = _post(clientapp)
    assert r.status_code == 502
    assert r.json()["detail"]["provider"] == "local_cross_encoder"
    assert counters.snapshot()["rerank_upstream_errors_total"] == 1


def test_backend_timeout_is_502(clientapp, staging, monkeypatch):
    def boom(query, texts):
        raise UpstreamError("échec /rerank backend: timeout")

    monkeypatch.setattr(main.rerank_client, "rerank", boom)
    assert _post(clientapp).status_code == 502


@pytest.mark.parametrize(
    "scores",
    [
        [],                                                     # cardinalité
        [{"index": 0, "score": 1.0}],                           # incomplet
        [{"index": 0, "score": 1.0}, {"index": 0, "score": 2.0},
         {"index": 1, "score": 3.0}],                           # doublon
        [{"index": 0, "score": 1.0}, {"index": 1, "score": 2.0},
         {"index": 7, "score": 3.0}],                           # hors bornes
    ],
)
def test_bad_backend_cardinality_is_502(clientapp, staging, monkeypatch, scores):
    monkeypatch.setattr(main.rerank_client, "rerank", lambda q, t: scores)
    assert _post(clientapp).status_code == 502


def test_nan_and_inf_scores_are_502(clientapp, staging, monkeypatch):
    # Le client transport valide la finitude : NaN/Inf lèvent UpstreamError.
    from app.rerank_client import RerankBackendClient

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [{"index": 0, "score": float("nan")}]

    monkeypatch.setattr(
        "app.rerank_client.requests.post",
        lambda url, json=None, timeout=None: _Resp(),
    )
    with pytest.raises(UpstreamError):
        RerankBackendClient("http://127.0.0.1:8300", 5).rerank("q", ["a"])


def test_missing_score_is_upstream_error(monkeypatch):
    from app.rerank_client import RerankBackendClient

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [{"index": 0}]

    monkeypatch.setattr(
        "app.rerank_client.requests.post",
        lambda url, json=None, timeout=None: _Resp(),
    )
    with pytest.raises(UpstreamError):
        RerankBackendClient("http://127.0.0.1:8300", 5).rerank("q", ["a"])


# --------------------------------------------------------------------------- #
# 6. Isolation
# --------------------------------------------------------------------------- #
def test_backend_down_leaves_generate_and_embeddings_healthy(
    clientapp, staging, monkeypatch
):
    def boom(query, texts):
        raise UpstreamError("échec /rerank backend: connexion refusée")

    monkeypatch.setattr(main.rerank_client, "rerank", boom)
    monkeypatch.setattr(
        main.client, "generate", lambda **kwargs: {"response": "OK", "done": True}
    )
    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: [[1.0] + [0.0] * 767 for _ in inputs],
    )

    assert _post(clientapp).status_code == 502
    assert clientapp.post("/v1/generate", json={"prompt": "x"}).status_code == 200
    assert clientapp.post("/v1/embeddings", json={"texts": ["a"]}).status_code == 200


def test_health_ready_ignores_rerank_backend(clientapp, staging, monkeypatch):
    # La readiness HISTORIQUE ne doit jamais dépendre du backend reranker.
    monkeypatch.setattr(
        main.probe_client,
        "embed",
        lambda model, inputs: [[1.0] + [0.0] * 767],
    )
    monkeypatch.setattr(
        main.rerank_client,
        "info",
        lambda timeout=5: (_ for _ in ()).throw(UpstreamError("backend mort")),
    )
    r = clientapp.get("/health/ready")
    assert r.status_code == 200


def test_health_rerank_disabled_is_200_with_status(clientapp):
    r = clientapp.get("/health/rerank")
    assert r.status_code == 200
    assert r.json()["status"] == "disabled"


def test_health_rerank_backend_down_is_503(clientapp, staging, monkeypatch):
    monkeypatch.setattr(
        main.rerank_client,
        "info",
        lambda timeout=5: (_ for _ in ()).throw(UpstreamError("injoignable")),
    )
    r = clientapp.get("/health/rerank")
    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["backend_reachable"] is False


def test_health_rerank_revision_mismatch_is_503(clientapp, staging, monkeypatch):
    monkeypatch.setattr(
        main.rerank_client,
        "info",
        lambda timeout=5: {"model_id": "BAAI/bge-reranker-v2-m3",
                           "model_sha": "0" * 40},
    )
    r = clientapp.get("/health/rerank")
    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["revision_match"] is False


def test_health_rerank_ok(clientapp, staging, monkeypatch):
    monkeypatch.setattr(
        main.rerank_client,
        "info",
        lambda timeout=5: {"model_id": "BAAI/bge-reranker-v2-m3",
                           "model_sha": REVISION},
    )
    monkeypatch.setattr(
        main.rerank_client, "rerank", lambda q, t: [{"index": 0, "score": 0.5}]
    )
    r = clientapp.get("/health/rerank")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend_url"] == "redacted"
    assert body["observed_revision"] == REVISION


def test_health_rerank_never_leaks_backend_url(clientapp, staging, monkeypatch):
    monkeypatch.setattr(
        main.rerank_client,
        "info",
        lambda timeout=5: (_ for _ in ()).throw(UpstreamError("injoignable")),
    )
    assert "8300" not in clientapp.get("/health/rerank").text


def test_health_runtime_lists_rerank_capability(clientapp):
    body = clientapp.get("/health/runtime").json()
    assert body["rerank"]["enabled"] is False
    assert body["rerank"]["default_model"] == "rerank-default"
    assert body["rerank"]["models"][0]["artifact"] == "BAAI/bge-reranker-v2-m3"


# --------------------------------------------------------------------------- #
# 7. Observabilité et privacy
# --------------------------------------------------------------------------- #
def test_rerank_ok_event_fields(clientapp, staging, backend, logs):
    _post(clientapp, top_n=2)
    line = next(
        r.getMessage() for r in logs.records
        if r.getMessage().startswith("event=rerank.ok ")
    )
    fields = dict(p.split("=", 1) for p in line.split(" ")[1:] if "=" in p)
    assert fields["model_logical"] == "rerank-default"
    assert fields["n_documents"] == "3"
    assert fields["returned"] == "2"
    assert "query_chars" in fields and "total_document_chars" in fields
    assert "total_ms" in fields and "upstream_ms" in fields


def test_inflight_gauge_released(clientapp, staging, monkeypatch):
    observed = {}

    def fake(query, texts):
        observed["during"] = gauges.snapshot().get("rerank:rerank-default", 0)
        return [{"index": i, "score": 0.0} for i in range(len(texts))]

    monkeypatch.setattr(main.rerank_client, "rerank", fake)
    _post(clientapp)
    assert observed["during"] == 1
    assert gauges.snapshot().get("rerank:rerank-default", 0) == 0


def test_inflight_gauge_released_on_backend_error(clientapp, staging, monkeypatch):
    def boom(query, texts):
        raise UpstreamError("ko")

    monkeypatch.setattr(main.rerank_client, "rerank", boom)
    _post(clientapp)
    assert gauges.total() == 0


def test_pairs_counter(clientapp, staging, backend):
    _post(clientapp)
    assert counters.snapshot()["rerank_pairs_total"] == 3
    assert counters.snapshot()["rerank_requests_total"] == 1


def test_privacy_sentinels_absent_from_logs(clientapp, staging, backend, logs):
    r = clientapp.post(
        "/v1/rerank",
        json={
            "query": QUERY_SENTINEL,
            "documents": [{"id": ID_SENTINEL, "text": TEXT_SENTINEL}],
        },
    )
    assert r.status_code == 200
    # Servis au client…
    assert ID_SENTINEL in r.text
    # …absents de tous les logs.
    everything = _all_logs(logs)
    assert QUERY_SENTINEL not in everything
    assert TEXT_SENTINEL not in everything
    assert ID_SENTINEL not in everything


def test_privacy_sentinels_absent_on_error_paths(clientapp, staging, monkeypatch, logs):
    def boom(query, texts):
        raise UpstreamError("échec /rerank backend: HTTP 500")

    monkeypatch.setattr(main.rerank_client, "rerank", boom)
    clientapp.post(
        "/v1/rerank",
        json={
            "query": QUERY_SENTINEL,
            "documents": [{"id": ID_SENTINEL, "text": TEXT_SENTINEL}],
        },
    )
    everything = _all_logs(logs)
    assert QUERY_SENTINEL not in everything
    assert TEXT_SENTINEL not in everything
    assert ID_SENTINEL not in everything


def test_oversize_error_does_not_leak_content(clientapp, staging, backend, logs):
    r = clientapp.post(
        "/v1/rerank",
        json={"query": "q",
              "documents": [{"id": ID_SENTINEL, "text": TEXT_SENTINEL + "x" * 4001}]},
    )
    assert r.status_code == 400
    assert TEXT_SENTINEL not in r.text
    assert TEXT_SENTINEL not in _all_logs(logs)


# --------------------------------------------------------------------------- #
# 8. Parité de déploiement
# --------------------------------------------------------------------------- #
def test_example_env_file_enables_rerank_capability():
    # Profil Open : la configuration d'exemple distribuée
    # (.env.example) active la capability reranking — le registre l'autorise ;
    # la frontière OFF restante est RAG_RERANKER_ENABLED côté consommateur.
    enabled = None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for line in open(os.path.join(root, ".env.example")).read().splitlines():
        if line.strip().startswith("AI_GATEWAY_RERANK_ENABLED"):
            enabled = line.split("=", 1)[1].split("#")[0].strip().lower()
    assert enabled in ("1", "true", "yes", "on")


def test_reranker_compose_binds_loopback_only():
    content = open("docker-compose.reranker.yml").read()
    assert '"127.0.0.1:8300:80"' in content
    assert "--revision" in content and REVISION in content
