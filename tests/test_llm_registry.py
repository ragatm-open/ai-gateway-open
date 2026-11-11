"""Lot runtime-hardening — registre LLM, profils de contexte et de keep_alive.

Deux niveaux : le PARSING (validation stricte du YAML, mêmes exigences que le
registre d'embeddings) et la RÉSOLUTION (logique nom logique / nom direct /
hors registre).
"""

import textwrap

import pytest

from app.registry import (
    RegistryError,
    load_registry,
)

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast


def _write(tmp_path, body: str):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


# Registre minimal valide : un embedding (obligatoire) + sections LLM.
VALID = """
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
    context_profiles:
      chat-small:
        num_ctx: 4096
        max_num_ctx: 8192
      bounded:
        num_ctx: 8192
        max_num_ctx: 16384
    keep_alive_profiles:
      resident:
        keep_alive: 30m
      default:
        keep_alive: null
    generation_profiles:
      bounded-out:
        default_num_predict: 2048
        max_num_predict: 4096
    llm_models:
      llama-chat:
        ollama_name: llama3.1:8b
        capabilities: [completion, tools]
        resource_class: medium
        context_profile: chat-small
        keep_alive_profile: resident
        generation_profile: bounded-out
        structured_output: true
        production_allowed: true
        staging_allowed: true
      judge:
        ollama_name: qwen3:14b
        resource_class: medium
        context_profile: bounded
        keep_alive_profile: default
        generation_profile: bounded-out
        default_think: false
        production_allowed: false
        staging_allowed: true
"""


@pytest.fixture
def registry(tmp_path):
    return load_registry(_write(tmp_path, VALID))


# ------------------------------------------------------------------- parsing
def test_llm_registry_parses(registry):
    assert set(registry.llm_models) == {"llama-chat", "judge"}
    assert registry.context_profiles["chat-small"].num_ctx == 4096
    assert registry.context_profiles["bounded"].max_num_ctx == 16384
    assert registry.keep_alive_profiles["resident"].keep_alive == "30m"
    assert registry.keep_alive_profiles["default"].keep_alive is None


def test_llm_sections_are_optional(tmp_path):
    # Un registre sans sections LLM reste valide : comportement historique.
    body = VALID.split("context_profiles:")[0]
    reg = load_registry(_write(tmp_path, body))
    assert reg.llm_models == {}
    assert reg.resolve_llm("llama3.1:8b") is None


def test_llm_profile_public_dict_has_no_surprises(registry):
    entry = registry.llm_models["judge"].to_public_dict()
    assert entry["model_logical"] == "judge"
    assert entry["default_think"] is False
    assert entry["production_allowed"] is False


@pytest.mark.parametrize(
    "mutation,fragment",
    [
        ("context_profile: absent-profile", "context_profile inconnu"),
        ("keep_alive_profile: absent-profile", "keep_alive_profile inconnu"),
        ("resource_class: gigantic", "resource_class inconnu"),
    ],
)
def test_invalid_references_are_rejected(tmp_path, mutation, fragment):
    body = VALID.replace(
        {"context_profile: absent-profile": "context_profile: bounded",
         "keep_alive_profile: absent-profile": "keep_alive_profile: default",
         "resource_class: gigantic": "resource_class: medium"}[mutation],
        mutation,
        1,
    )
    with pytest.raises(RegistryError) as excinfo:
        load_registry(_write(tmp_path, body))
    assert fragment in str(excinfo.value)


def test_context_profile_requires_num_ctx(tmp_path):
    body = VALID.replace("chat-small:\n        num_ctx: 4096", "chat-small: {}")
    with pytest.raises(RegistryError):
        load_registry(_write(tmp_path, body))


def test_max_num_ctx_must_cover_num_ctx(tmp_path):
    body = VALID.replace("max_num_ctx: 16384", "max_num_ctx: 4096")
    with pytest.raises(RegistryError) as excinfo:
        load_registry(_write(tmp_path, body))
    assert "max_num_ctx" in str(excinfo.value)


def test_keep_alive_profile_rejects_campaign_values(tmp_path):
    # 0 (décharge immédiate) et -1 (épinglage) sont des gestes de campagne :
    # un PROFIL par défaut ne doit pas pouvoir les porter.
    body = VALID.replace("keep_alive: 30m", "keep_alive: 0")
    with pytest.raises(RegistryError):
        load_registry(_write(tmp_path, body))


def test_ambiguous_llm_alias_is_rejected(tmp_path):
    body = VALID.replace("ollama_name: qwen3:14b", "ollama_name: llama3.1:8b")
    with pytest.raises(RegistryError) as excinfo:
        load_registry(_write(tmp_path, body))
    assert "ambigu" in str(excinfo.value)


def test_llm_name_colliding_with_embedding_is_rejected(tmp_path):
    body = VALID.replace("ollama_name: qwen3:14b", "ollama_name: nomic-embed-text")
    with pytest.raises(RegistryError) as excinfo:
        load_registry(_write(tmp_path, body))
    assert "entre registres" in str(excinfo.value)


# ---------------------------------------------------------------- résolution
def test_resolve_by_logical_name(registry):
    resolved = registry.resolve_llm("judge")
    assert resolved.profile.ollama_name == "qwen3:14b"
    assert resolved.via_logical is True


def test_resolve_by_direct_ollama_name(registry):
    resolved = registry.resolve_llm("qwen3:14b")
    assert resolved.profile.logical_name == "judge"
    assert resolved.via_logical is False


def test_resolve_latest_alias(tmp_path):
    body = VALID.replace("ollama_name: qwen3:14b", "ollama_name: mymodel:latest")
    reg = load_registry(_write(tmp_path, body))
    assert reg.resolve_llm("mymodel").profile.logical_name == "judge"
    assert reg.resolve_llm("mymodel:latest").profile.logical_name == "judge"


def test_family_base_name_is_not_an_alias(registry):
    # `qwen3` nu désigne `qwen3:latest` chez Ollama — PAS `qwen3:14b`. L'alias
    # nu pour un tag explicite créerait de fausses ambiguïtés de famille.
    assert registry.resolve_llm("qwen3") is None


def test_unknown_model_resolves_to_none(registry):
    assert registry.resolve_llm("gemma3:27b") is None
    assert registry.resolve_llm("") is None


def test_environment_allowlist(registry):
    judge = registry.llm_models["judge"]
    assert judge.allowed_in("staging") is True
    assert judge.allowed_in("production") is False
    assert judge.allowed_in("autre") is False


# ------------------------------------------------- registre embarqué (réel)
def test_shipped_registry_governs_the_default_model():
    from app.main import registry as live

    resolved = live.resolve_llm("llama3.1:8b")
    assert resolved is not None
    assert resolved.profile.logical_name == "llama-chat"
    context = live.context_profile_of(resolved.profile)
    # long-context (16384) : dimensionné sur le trafic réel de production
    # (p50=4 966 tokens, max=15 434) — cf. models_registry.yaml.
    assert context.num_ctx == 16384
    assert live.keep_alive_of(resolved.profile).keep_alive == "30m"


def test_shipped_registry_judge_thinks_only_via_logical_name():
    from app.main import registry as live

    assert live.resolve_llm("qwen-judge").profile.default_think is False
    assert live.resolve_llm("qwen3:14b").via_logical is False


def test_shipped_registry_governs_gemma_since_p3():
    from app.main import registry as live

    resolved = live.resolve_llm("gemma3:27b")
    assert resolved is not None and resolved.profile.logical_name == "gemma-large"
    # Un nom réellement absent reste hors registre.
    assert live.resolve_llm("unregistered-model:latest") is None


def test_shipped_registry_environment_policy():
    # Garde de compatibilité : les modèles antérieurs au lot C2 restent
    # autorisés partout. Les SEULES restrictions d'environnement du registre
    # sont les modèles de raisonnement `oss-reasoning` (gpt-oss:20b, lot C2)
    # et `qwen-reasoning` (qwen3:30b-a3b-thinking-2507, 2026-09-27) : staging
    # seulement.
    from app.main import registry as live

    for profile in live.llm_models.values():
        # Depuis le lot production-full-capability-enablement, les deux modèles
        # de chat (oss-reasoning, qwen-reasoning) sont aussi autorisés en production.
        assert profile.production_allowed, profile.logical_name
        assert profile.staging_allowed, profile.logical_name
