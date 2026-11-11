"""Conformité Open — configuration fail-closed et défauts du profil Compact."""

import pytest

from app.settings import ConfigurationError, Settings


def test_defaults_are_the_open_profile(clean_env):
    s = Settings()
    assert s.generation_strict_registry is True and s.embeddings_strict_registry is True
    assert (s.generate_model, s.embedding_model) == ("compact-generation", "fr-en-retrieval")
    assert s.embeddings_dimension == 768
    assert (s.generate_max_concurrency, s.embeddings_max_concurrency) == (2, 2)
    assert s.request_max_bytes == 2 * 1024 * 1024
    assert s.asr_enabled is False and s.vision_enabled is False and s.rerank_enabled is False
    assert s.host_admission_enabled is False


def test_defaults_name_no_private_host(clean_env):
    s = Settings()
    for url in (s.ollama_base_url, s.rerank_backend_url, s.asr_backend_url, s.host_admission_url):
        assert url.startswith(("http://127.0.0.1", "http://localhost")), url


@pytest.mark.parametrize("name", [
    "AI_GATEWAY_HOST_ADMISSION_ENABLED", "AI_GATEWAY_STRUCTURED_EXTRACT_ENABLED",
    "AI_GATEWAY_STRUCTURED_EXTRACT_PRODUCTION_ALLOWED", "AI_GATEWAY_V2H_ENABLED",
    "AI_GATEWAY_V2H_SHADOW_ENABLED", "AI_GATEWAY_VISION_SEMANTIC_ENABLED",
])
def test_excluded_components_cannot_be_enabled(clean_env, name):
    clean_env.setenv(name, "1")
    with pytest.raises(ConfigurationError) as info:
        Settings()
    assert name in str(info.value)
    clean_env.setenv(name, "0")
    Settings()      # OFF explicite : accepté


@pytest.mark.parametrize("name,value", [
    ("AI_GATEWAY_GENERATE_MAX_CONCURRENCY", "0"), ("AI_GATEWAY_GENERATE_MAX_CONCURRENCY", "65"),
    ("AI_GATEWAY_EMBEDDINGS_MAX_CONCURRENCY", "abc"), ("AI_GATEWAY_REQUEST_MAX_BYTES", "1024"),
])
def test_new_limits_are_validated_at_startup(clean_env, name, value):
    clean_env.setenv(name, value)
    with pytest.raises(ConfigurationError) as info:
        Settings()
    assert name in str(info.value)


def test_strict_registry_can_be_relaxed_explicitly(clean_env):
    clean_env.setenv("AI_GATEWAY_GENERATION_STRICT_REGISTRY", "0")
    clean_env.setenv("AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY", "false")
    s = Settings()
    assert s.generation_strict_registry is False and s.embeddings_strict_registry is False


def test_removed_settings_are_gone(clean_env):
    s = Settings()
    for attr in ("v2h_enabled", "v2h_shadow_enabled", "structured_extract_enabled",
                 "vision_semantic_enabled", "vision_semantic_timeout_seconds"):
        assert not hasattr(s, attr), attr
