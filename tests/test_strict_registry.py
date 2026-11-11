"""Lot P3 — registre strict de génération (AI_GATEWAY_GENERATION_STRICT_REGISTRY).

strict=false (DÉFAUT) : un modèle hors registre suit le relais historique,
compté `generate_legacy_model_total`.
strict=true : il est refusé AVANT Ollama — 400 `model_not_registered`, compté
`generate_rejected_unregistered_total`, jamais compté comme legacy relayé.

Matrice : A nom logique · B alias Ollama · C nom physique · D installé mais
non enregistré · E inexistant. Tout backend est mocké, aucun réseau.
"""

import logging
import os

import pytest

from app import main
from app.metrics import counters
from app.settings import Settings

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast

UNREG_INSTALLED = "unregistered-model:latest"     # D — installé côté Ollama, absent du registre
UNREG_MISSING = "does-not-exist:zz"               # E — inexistant partout

REGISTERED = {
    "A-logique": "qwen-judge",
    "B-alias": "nomic-embed-text:latest" if False else "qwen3:14b",   # alias Ollama d'un profil
    "C-physique": "qwen3:30b-instruct",
    "gemma-logique": "gemma-large",
    "gemma-physique": "gemma3:27b",
}


@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(main.client, "generate",
                        lambda **kw: (seen.update(kw), {"response": "OK", "done": True})[1])
    return seen


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _strict(monkeypatch, value):
    monkeypatch.setattr(main.settings, "generation_strict_registry", value)


def _gen(clientapp, model):
    return clientapp.post("/v1/generate", json={"prompt": "x", "model": model})


# --------------------------------------------------------------------------- #
# 1. Défaut : ACTIVÉ (profil Open, qualification interne — fail-closed) ; 0 rétablit le relais legacy
# --------------------------------------------------------------------------- #
def test_default_is_on(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_GENERATION_STRICT_REGISTRY", raising=False)
    assert Settings().generation_strict_registry is True
    monkeypatch.setenv("AI_GATEWAY_GENERATION_STRICT_REGISTRY", "0")
    assert Settings().generation_strict_registry is False
    monkeypatch.setenv("AI_GATEWAY_GENERATION_STRICT_REGISTRY", "1")
    assert Settings().generation_strict_registry is True
    # Lot P14 : une valeur hors convention n'est plus avalée, elle empêche le
    # démarrage (message nommant la variable).
    from app.settings import ConfigurationError
    monkeypatch.setenv("AI_GATEWAY_GENERATION_STRICT_REGISTRY", "nonsense")
    with pytest.raises(ConfigurationError, match="AI_GATEWAY_GENERATION_STRICT_REGISTRY"):
        Settings()


def test_runtime_exposes_the_flag(clientapp, monkeypatch):
    _strict(monkeypatch, False)
    assert clientapp.get("/health/runtime").json()["generation_strict_registry"] is False
    _strict(monkeypatch, True)
    assert clientapp.get("/health/runtime").json()["generation_strict_registry"] is True


# --------------------------------------------------------------------------- #
# 2. Matrice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("case,model", sorted(REGISTERED.items()))
def test_registered_models_succeed_regardless_of_strict(clientapp, captured, monkeypatch, strict, case, model):
    _strict(monkeypatch, strict)
    r = _gen(clientapp, model)
    assert r.status_code == 200, case
    assert "options" in captured and captured["options"]["num_predict"] > 0   # gouverné
    assert "generate_legacy_model_total" not in counters.snapshot()
    assert "generate_rejected_unregistered_total" not in counters.snapshot()


@pytest.mark.parametrize("model", [UNREG_INSTALLED, UNREG_MISSING])
def test_strict_off_keeps_legacy_relay(clientapp, captured, monkeypatch, logs, model):
    _strict(monkeypatch, False)
    r = _gen(clientapp, model)
    assert r.status_code == 200                       # relais historique (Ollama mocké)
    assert captured["model"] == model and captured["options"] is None
    assert counters.snapshot()["generate_legacy_model_total"] == 1
    assert "generate_rejected_unregistered_total" not in counters.snapshot()
    assert "event=generate.legacy_model" in "\n".join(r_.getMessage() for r_ in logs.records)


@pytest.mark.parametrize("model", [UNREG_INSTALLED, UNREG_MISSING])
def test_strict_on_rejects_before_ollama(clientapp, captured, monkeypatch, logs, model):
    _strict(monkeypatch, True)
    r = _gen(clientapp, model)
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "model_not_registered" and d["strict_registry"] is True
    assert d["provider"] == "ai_gateway"
    assert "127.0.0.1" not in r.text and "11434" not in r.text
    assert captured == {}                             # Ollama JAMAIS appelé
    snap = counters.snapshot()
    assert snap["generate_rejected_unregistered_total"] == 1
    assert "generate_legacy_model_total" not in snap  # pas un relais legacy
    blob = "\n".join(r_.getMessage() for r_ in logs.records)
    assert "event=generate.model_not_registered" in blob
    assert "event=generate.legacy_model" not in blob


def test_strict_on_applies_to_stream(clientapp, monkeypatch):
    from tests.test_generate_stream import DONE_FIELDS, _FakeStream, _install_stream
    _strict(monkeypatch, True)
    captured = _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x", "model": UNREG_INSTALLED})
    assert r.status_code == 400 and captured == {}
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x", "model": "gemma3:27b"})
    assert r.status_code == 200


def test_strict_does_not_touch_embeddings_policy(clientapp, monkeypatch):
    """Périmètre : le flag de génération n'a aucun effet sur /v1/embeddings,
    qui garde son propre AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY."""
    _strict(monkeypatch, True)
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", False)
    from app.ollama_client import ENDPOINT_EMBED, EmbedResult
    monkeypatch.setattr(main.client, "embed",
                        lambda **kw: EmbedResult([[1.0] + [0.0] * 767], ENDPOINT_EMBED,
                                                 fallback_used=False, ollama_metadata={}))
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "bge-m3"})
    assert r.status_code == 200 and r.json()["governed"] is False
