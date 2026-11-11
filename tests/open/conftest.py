"""Fixtures des tests de conformité Open.

Contrairement à la suite héritée (registre de fixture, registres non stricts),
ces tests travaillent sur le registre Compact RÉEL (app/models_registry.yaml)
et sur les défauts du profil Open. Le registre est substitué dans `app.main`
pour la durée du test ; aucune connexion réseau n'est possible (garde-fou du
conftest racine).
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.ollama_client import ENDPOINT_EMBED, EmbedResult
from app.registry import STATUS_READY, load_registry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPACT_REGISTRY = os.path.join(ROOT, "app", "models_registry.yaml")

# Variables posées par le conftest racine pour la suite héritée : les tests
# Open qui instancient `Settings()` les retirent pour observer les DÉFAUTS.
LEGACY_SUITE_ENV = (
    "AI_GATEWAY_MODELS_REGISTRY", "OLLAMA_GENERATE_MODEL", "OLLAMA_EMBEDDING_MODEL",
    "AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY", "AI_GATEWAY_GENERATION_STRICT_REGISTRY",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("AI_GATEWAY_", "OLLAMA_")):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def compact_registry():
    return load_registry(COMPACT_REGISTRY)


def installed_entries(registry, *, include_llm=("compact-generation-8",)):
    """Inventaire /api/tags simulé : embeddings du registre (digest conforme)
    + les LLM listés comme installés. `compact-generation` (7 B) n'y est PAS :
    il est enregistré mais non installé, comme sur l'environnement."""
    out = []
    for p in registry.models.values():
        out.append({"name": p.ollama_name if ":" in p.ollama_name else f"{p.ollama_name}:latest",
                    "digest": p.digest, "size": 1, "modified_at": "2026-11-09T00:00:00Z", "details": {}})
    for name in include_llm:
        p = registry.llm_models[name]
        out.append({"name": p.ollama_name, "digest": p.digest or "", "size": 1,
                    "modified_at": "2026-11-09T00:00:00Z", "details": {"family": "qwen2"}})
    return out


@pytest.fixture
def open_app(monkeypatch, compact_registry):
    """`app.main` reconfiguré sur le profil Open : registre Compact, registres
    stricts, défauts Compact, reranking activé sur la boucle locale."""
    registry = compact_registry
    monkeypatch.setattr(main, "registry", registry)
    monkeypatch.setattr(main.settings, "generation_strict_registry", True)
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", True)
    monkeypatch.setattr(main.settings, "generate_model", "compact-generation")
    monkeypatch.setattr(main.settings, "embedding_model", "fr-en-retrieval")
    monkeypatch.setattr(main.settings, "embeddings_dimension", 768)
    monkeypatch.setattr(main.settings, "environment", "production")
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main.settings, "rerank_backend_url", "http://127.0.0.1:8300")
    monkeypatch.setattr(main.settings, "vision_enabled", False)
    monkeypatch.setattr(main.settings, "asr_enabled", False)
    inventory = installed_entries(registry)
    monkeypatch.setattr(main.probe_client, "list_models", lambda timeout=5: inventory)
    monkeypatch.setattr(main.client, "list_models", lambda timeout=5: inventory)
    monkeypatch.setattr(main.probe_client, "list_running_models", lambda timeout=5: [])

    def fake_probe_embed(model, inputs, keep_alive=None):
        unit = [1.0] + [0.0] * 767
        return EmbedResult([list(unit) for _ in inputs], ENDPOINT_EMBED, fallback_used=False, ollama_metadata={})
    monkeypatch.setattr(main.probe_client, "embed", fake_probe_embed)
    registry.verify_digests(main.probe_client)      # inventaire observé, digests conformes
    for p in registry.models.values():
        assert p.status == STATUS_READY
        p.verified_at = time.time()
    main._readiness_cache_clear()
    main._verify_state["last"] = time.monotonic()   # pas de revérification pendant le test
    return TestClient(main.app)


def ok_generate_response(text="Bonjour.", prompt_tokens=5, completion_tokens=2):
    return {"response": text, "done": True, "done_reason": "stop",
            "prompt_eval_count": prompt_tokens, "eval_count": completion_tokens,
            "total_duration": 1_000_000_000, "load_duration": 100_000_000,
            "prompt_eval_duration": 200_000_000, "eval_duration": 300_000_000}


class FakeStream:
    """Double de GenerateStream : événements prédéfinis, fermeture tracée."""

    def __init__(self, events=None, error=None):
        self._events = events if events is not None else [
            {"response": "Bon", "done": False}, {"response": "jour", "done": False},
            {"response": "", "done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2},
        ]
        self._error = error
        self.closed = False
        self.consumed = 0

    def events(self):
        for event in self._events:
            if self._error is not None and self.consumed == 1:
                raise self._error
            self.consumed += 1
            yield event

    def close(self):
        self.closed = True
