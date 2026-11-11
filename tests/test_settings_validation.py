"""Lot P14 — configuration invalide = échec AU DÉMARRAGE, jamais plus tard.

Avant : `PORT=abc` était avalé (défaut silencieux), `TIMEOUT_SECONDS=0` passait
et explosait en `ValueError` dans `requests` à la première requête,
`MAX_BATCH=-3` rendait 100 % de 400. Après : `Settings()` lève
`ConfigurationError` avec le nom de la variable ; le module `app.settings`
étant instancié à l'import, uvicorn ne démarre pas.
"""

import pytest

from app.settings import ConfigurationError, Settings

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast

ALL_NUMERIC = [
    "AI_GATEWAY_PORT", "AI_GATEWAY_TIMEOUT_SECONDS", "AI_GATEWAY_EMBEDDINGS_DIMENSION",
    "AI_GATEWAY_EMBEDDINGS_MAX_BATCH", "AI_GATEWAY_READINESS_CACHE_SECONDS",
    "AI_GATEWAY_READINESS_TIMEOUT_SECONDS", "AI_GATEWAY_REGISTRY_VERIFY_INTERVAL_SECONDS",
    "AI_GATEWAY_RERANK_TIMEOUT_SECONDS", "AI_GATEWAY_RERANK_MAX_CONCURRENCY",
    "AI_GATEWAY_VISION_TIMEOUT_SECONDS", "AI_GATEWAY_VISION_MAX_CONCURRENCY",
    "AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS",
]
ALL_BOOL = [
    "AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY", "AI_GATEWAY_GENERATION_STRICT_REGISTRY",
    "AI_GATEWAY_RERANK_ENABLED", "AI_GATEWAY_VISION_ENABLED",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ALL_NUMERIC + ALL_BOOL + ["AI_GATEWAY_ENVIRONMENT"]:
        monkeypatch.delenv(name, raising=False)


def _expect_failure(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError) as info:
        Settings()
    assert name in str(info.value), str(info.value)
    return str(info.value)


# --------------------------------------------------------------------------- #
# 1. Cas imposés par le lot
# --------------------------------------------------------------------------- #
# Valeurs HORS DOMAINE par variable — les domaines qualifiés sont inchangés :
#   · AI_GATEWAY_VISION_MAX_CONCURRENCY : 1..4 ; hors domaine = échec, plus un retour
#     silencieux à 1 ;
#   · AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS : allowlist 0/10/30 ; `-1` reste INATTEIGNABLE
#     par configuration — désormais parce que le service refuse de démarrer, pas par
#     remplacement silencieux.
OUT_OF_DOMAIN = {
    "AI_GATEWAY_PORT": ["abc", "0", "65536", "-1", "1.5"],
    "AI_GATEWAY_TIMEOUT_SECONDS": ["0", "-1", "abc"],
    "AI_GATEWAY_EMBEDDINGS_MAX_BATCH": ["0", "-3", "x"],
    "AI_GATEWAY_RERANK_MAX_CONCURRENCY": ["0", "-2"],
    "AI_GATEWAY_VISION_MAX_CONCURRENCY": ["0", "-1", "5"],
    "AI_GATEWAY_VISION_TIMEOUT_SECONDS": ["4", "601", "abc"],
    "AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS": ["5", "-1", "3600", "abc"],
}
_OUT_OF_DOMAIN_CASES = [(name, value) for name, values in OUT_OF_DOMAIN.items() for value in values]


@pytest.mark.parametrize(
    ("name", "value"), _OUT_OF_DOMAIN_CASES, ids=[f"{n}={v}" for n, v in _OUT_OF_DOMAIN_CASES]
)
def test_out_of_domain_value_fails_at_startup(monkeypatch, name, value):
    _expect_failure(monkeypatch, name, value)


@pytest.mark.parametrize("name", [
    "AI_GATEWAY_EMBEDDINGS_DIMENSION", "AI_GATEWAY_READINESS_TIMEOUT_SECONDS",
    "AI_GATEWAY_RERANK_TIMEOUT_SECONDS",
])
def test_other_strictly_positive_settings(monkeypatch, name):
    _expect_failure(monkeypatch, name, "0")
    _expect_failure(monkeypatch, name, "-7")


@pytest.mark.parametrize("name", [
    "AI_GATEWAY_READINESS_CACHE_SECONDS", "AI_GATEWAY_REGISTRY_VERIFY_INTERVAL_SECONDS",
])
def test_non_negative_settings_accept_zero_but_not_negative(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    Settings()
    _expect_failure(monkeypatch, name, "-1")


@pytest.mark.parametrize("name", ALL_NUMERIC)
def test_every_numeric_setting_rejects_garbage(monkeypatch, name):
    for value in ("abc", "1.5", "1e3", " 12x"):
        msg = _expect_failure(monkeypatch, name, value)
        assert "entier" in msg


@pytest.mark.parametrize("name", ALL_BOOL)
def test_every_bool_setting_rejects_garbage(monkeypatch, name):
    _expect_failure(monkeypatch, name, "peut-etre")
    for value, expected in (("1", True), ("true", True), ("YES", True), ("on", True),
                            ("0", False), ("false", False), ("no", False), ("off", False)):
        monkeypatch.setenv(name, value)
        assert getattr(Settings(), _attr(name)) is expected, (name, value)


def _attr(name):
    return {
        "AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY": "embeddings_strict_registry",
        "AI_GATEWAY_GENERATION_STRICT_REGISTRY": "generation_strict_registry",
        "AI_GATEWAY_RERANK_ENABLED": "rerank_enabled",
        "AI_GATEWAY_VISION_ENABLED": "vision_enabled",
    }[name]


# --------------------------------------------------------------------------- #
# 2. Valeurs valides, défauts, atomicité
# --------------------------------------------------------------------------- #
def test_valid_values_load_normally(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_PORT", "8200")
    monkeypatch.setenv("AI_GATEWAY_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("AI_GATEWAY_EMBEDDINGS_MAX_BATCH", "64")
    monkeypatch.setenv("AI_GATEWAY_RERANK_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("AI_GATEWAY_VISION_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS", "10")
    monkeypatch.setenv("AI_GATEWAY_ENVIRONMENT", "staging")
    s = Settings()
    assert (s.port, s.timeout_seconds, s.embeddings_max_batch) == (8200, 180, 64)
    assert (s.rerank_max_concurrency, s.vision_max_concurrency, s.vision_keep_alive_seconds) == (1, 1, 10)


def test_defaults_unchanged():
    s = Settings()
    assert (s.port, s.timeout_seconds, s.embeddings_max_batch, s.embeddings_dimension) == (8100, 60, 64, 768)
    assert (s.rerank_max_concurrency, s.rerank_timeout_seconds) == (2, 15)
    assert (s.vision_timeout_seconds, s.vision_max_concurrency, s.vision_keep_alive_seconds) == (60, 1, 0)
    assert (s.readiness_cache_seconds, s.readiness_timeout_seconds, s.registry_verify_interval_seconds) == (60, 5, 300)


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_explicit_value_means_absent(monkeypatch, value):
    """Une variable vide (fichier env avec `X=`) vaut absence, pas erreur —
    convention historique conservée."""
    monkeypatch.setenv("AI_GATEWAY_PORT", value)
    assert Settings().port == 8100


def test_loading_is_atomic_first_invalid_aborts(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_PORT", "8200")             # valide
    monkeypatch.setenv("AI_GATEWAY_EMBEDDINGS_MAX_BATCH", "0")  # invalide
    with pytest.raises(ConfigurationError) as info:
        Settings()
    assert "AI_GATEWAY_EMBEDDINGS_MAX_BATCH" in str(info.value)


def test_production_keep_alive_guard_unchanged(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS", "10")
    assert Settings().vision_keep_alive_seconds == 0      # garde production, pas validation


def test_base64_bound_derives_from_decoded_limit():
    s = Settings()
    assert s.vision_max_image_bytes == 4 * 1024 * 1024
    assert s.vision_max_image_base64_chars == 4 * ((s.vision_max_image_bytes + 2) // 3) == 5_592_408


# --------------------------------------------------------------------------- #
# ASR en production : le flag a un sens normal, défaut OFF (fail-safe)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("environment", ["production", "staging"])
def test_asr_flag_absent_is_off_in_every_environment(monkeypatch, environment):
    monkeypatch.setenv("AI_GATEWAY_ENVIRONMENT", environment)
    monkeypatch.delenv("AI_GATEWAY_ASR_ENABLED", raising=False)
    assert Settings().asr_enabled is False


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_asr_flag_true_enables_in_every_environment(monkeypatch, environment):
    """Lot production-full-capability-enablement : plus de forçage OFF en production."""
    monkeypatch.setenv("AI_GATEWAY_ENVIRONMENT", environment)
    monkeypatch.setenv("AI_GATEWAY_ASR_ENABLED", "true")
    assert Settings().asr_enabled is True
