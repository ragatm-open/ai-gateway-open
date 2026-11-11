"""Conformité Open — registre Compact strict : identité, licences, états."""

import re

import yaml

from app.registry import QUALIFICATION_STATES, embedding_pipeline_id, load_registry
from tests.open.conftest import COMPACT_REGISTRY

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_compact_registry_loads_and_declares_expected_profiles(compact_registry):
    r = compact_registry
    assert set(r.models) == {"fr-en-retrieval", "legacy-768"}
    assert set(r.llm_models) == {"compact-generation", "compact-generation-8"}
    assert set(r.reranker_models) == {"rerank-default"}
    assert r.asr_models == {} and r.vision_default_model is None


def test_every_profile_carries_compact_metadata(compact_registry):
    r = compact_registry
    profiles = list(r.models.values()) + list(r.llm_models.values()) + list(r.reranker_models.values())
    for p in profiles:
        meta = p.metadata
        assert meta.compact_role and meta.license and meta.version, p.logical_name
        assert meta.qualification in QUALIFICATION_STATES
        assert meta.memory_budget_mib and meta.memory_budget_mib > 0
        # Aucun profil n'est déclaré qualifié par ce lot : l'état de preuve est honnête.
        assert meta.qualification in ("candidate", "experimental"), p.logical_name


def test_no_private_production_model_in_compact_registry(compact_registry):
    names = {p.ollama_name for p in compact_registry.llm_models.values()}
    for forbidden in ("qwen3:30b", "gemma3:27b", "mistral-small3.2", "gpt-oss", "qwen3:14b", "llama3.1:8b"):
        assert not any(n.startswith(forbidden) for n in names), forbidden
    # Budget mémoire estimé de chaque LLM Compact sous 12 Gio.
    assert all(p.metadata.memory_budget_mib <= 12 * 1024 for p in compact_registry.llm_models.values())


def test_digests_are_observed_or_absent_never_invented(compact_registry):
    r = compact_registry
    # Embeddings : digest OBLIGATOIRE, 64 hex.
    for p in r.models.values():
        assert HEX64.match(p.digest)
    # LLM : le 7 B n'a pas été tiré => AUCUN digest ; le 1.5 B relevé lors de la qualification.
    assert r.llm_models["compact-generation"].digest is None
    assert r.llm_models["compact-generation-8"].digest == \
        "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b"
    assert r.models["fr-en-retrieval"].digest == \
        "1a37926bf842899cbf90583e3932f3820548716d9d07661ba622199ffe95c552"
    assert r.llm_models["compact-generation"].to_public_dict()["expected_digest"] is None


def test_embeddings_respect_compact_v1_dimension(compact_registry):
    assert all(p.dimension == 768 for p in compact_registry.models.values())


def test_pipeline_identity_is_deterministic_and_documented(compact_registry):
    p = compact_registry.models["fr-en-retrieval"]
    assert p.pipeline_id("document") == "emb-v2:c8215aca68cbf785bf977a46ead4b23c"
    assert p.pipeline_id("query") != p.pipeline_id("document") or p.query_prefix == p.document_prefix
    assert embedding_pipeline_id(p.pipeline_identity("document")) == p.pipeline_id("document")
    # Le nom logique n'entre pas dans l'identité.
    assert "fr-en-retrieval" not in str(p.pipeline_identity("document"))


def test_no_chat_capability_and_reasoning_free(compact_registry):
    for p in compact_registry.llm_models.values():
        assert "chat" not in p.capabilities and p.reasoning_profile is None
        assert "completion" in p.capabilities
    assert compact_registry.reasoning_profiles == {}


def test_reranker_identity_pinned(compact_registry):
    rr = compact_registry.reranker_models["rerank-default"]
    assert rr.artifact == "BAAI/bge-reranker-v2-m3"
    assert rr.revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert rr.on_overflow == "reject" and rr.calibrated is False


def test_registry_rejects_unknown_qualification(tmp_path):
    raw = yaml.safe_load(open(COMPACT_REGISTRY, encoding="utf-8"))
    raw["models"]["fr-en-retrieval"]["qualification"] = "ready"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    import pytest
    from app.registry import RegistryError
    with pytest.raises(RegistryError):
        load_registry(str(path))


def test_registry_file_has_no_private_trace():
    text = open(COMPACT_REGISTRY, encoding="utf-8").read().lower()
    for token in (".lan", "192.168.", "/home/", "/opt/"):
        assert token not in text, token
