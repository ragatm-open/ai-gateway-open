"""Support des tests du lot C1 (chat) : registre SYNTHÉTIQUE et doubles.

Aucun modèle réel du registre embarqué ne déclare la capability `chat` en
C1 (opt-in explicite, aucun nouveau modèle activé). Les tests chargent donc
un registre de fixtures depuis un YAML temporaire — mêmes profils de
contexte / génération / rétention que le registre réel, plus des profils LLM
synthétiques — et le branchent sur `app.main.registry`. Rien ici ne touche à
`app/models_registry.yaml`.
"""

import json
import textwrap

from app import main
from app.registry import load_registry

REGISTRY_HEAD = textwrap.dedent(
    """\
    version: 1
    models:
      emb-test:
        provider: ollama
        ollama_name: nomic-embed-text
        digest: "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
        kind: embedding
        dimension: 768
        languages: [en]
        cross_lingual: false
        normalized: true
        query_prefix: "search_query: "
        document_prefix: "search_document: "
        max_batch_size: 64
        max_input_chars: 6000
        on_overflow: reject
        production_allowed: true
        staging_allowed: true
    context_profiles:
      verifier:
        num_ctx: 8192
        max_num_ctx: 16384
      long-context:
        num_ctx: 16384
        max_num_ctx: 32768
    generation_profiles:
      chat-bounded:
        default_num_predict: 3072
        max_num_predict: 8192
      judge-bounded:
        default_num_predict: 3072
        max_num_predict: 4096
    keep_alive_profiles:
      resident:
        keep_alive: 30m
      transient-governed:
        keep_alive: 2m
        client_override: false
      default:
        keep_alive: null
    reasoning_profiles:
      # Aucun mode pilotable (situation de tous les modèles réels).
      none:
        type: none
      # Bascule on/off, défaut off — famille glm-4.7-flash (futur C2).
      toggle:
        type: toggle
        default: "off"
      # Bascule dont la trace de réflexion est EXPOSÉE au client.
      toggle-exposed:
        type: toggle
        default: "on"
        expose_thinking: true
      # Niveaux honorés, non désactivable, défaut low — famille gpt-oss (futur C2).
      tiered:
        type: levels
        allowed: [low, medium, high]
        default: low
        disable_supported: false
      # Niveaux restreints et désactivables.
      tiered-off:
        type: levels
        allowed: [low, high]
        default: high
        disable_supported: true
        expose_thinking: true
    """
)

# Profils LLM synthétiques du lot C1. Tout modèle `chat` porte un
# `reasoning_profile` explicite (invariant du registre).
LLM_BLOCK = textwrap.dedent(
    """\
    llm_models:
      chat-test:
        ollama_name: chat-model:latest
        capabilities: [completion, chat]
        resource_class: medium
        context_profile: long-context
        keep_alive_profile: resident
        generation_profile: chat-bounded
        reasoning_profile: none
        production_allowed: true
        staging_allowed: true
      chat-toggle:
        ollama_name: chat-toggle:latest
        capabilities: [completion, chat, thinking]
        resource_class: medium
        context_profile: long-context
        keep_alive_profile: resident
        generation_profile: chat-bounded
        reasoning_profile: toggle
        production_allowed: true
        staging_allowed: true
      chat-toggle-exposed:
        ollama_name: chat-toggle-exposed:latest
        capabilities: [completion, chat, thinking]
        resource_class: medium
        context_profile: long-context
        keep_alive_profile: resident
        generation_profile: chat-bounded
        reasoning_profile: toggle-exposed
        production_allowed: true
        staging_allowed: true
      chat-tiered:
        ollama_name: chat-tiered:latest
        capabilities: [completion, chat, thinking]
        resource_class: medium
        context_profile: long-context
        keep_alive_profile: resident
        generation_profile: chat-bounded
        reasoning_profile: tiered
        production_allowed: true
        staging_allowed: true
      chat-tiered-off:
        ollama_name: chat-tiered-off:latest
        capabilities: [completion, chat, thinking]
        resource_class: medium
        context_profile: long-context
        keep_alive_profile: resident
        generation_profile: chat-bounded
        reasoning_profile: tiered-off
        production_allowed: true
        staging_allowed: true
      nochat-test:
        ollama_name: nochat-model:7b
        capabilities: [completion, tools]
        resource_class: medium
        context_profile: verifier
        keep_alive_profile: resident
        generation_profile: judge-bounded
        production_allowed: true
        staging_allowed: true
      chat-locked:
        ollama_name: chat-locked:27b
        capabilities: [completion, chat]
        resource_class: large
        context_profile: verifier
        keep_alive_profile: transient-governed
        generation_profile: judge-bounded
        reasoning_profile: none
        production_allowed: true
        staging_allowed: true
      chat-staging-only:
        ollama_name: chat-staging:latest
        capabilities: [completion, chat]
        resource_class: medium
        context_profile: verifier
        keep_alive_profile: default
        generation_profile: judge-bounded
        reasoning_profile: none
        production_allowed: false
        staging_allowed: true
    """
)


def write_registry(tmp_path, llm_block=LLM_BLOCK, extra_sections=""):
    path = tmp_path / "registry.yaml"
    path.write_text(REGISTRY_HEAD + extra_sections + llm_block, encoding="utf-8")
    return str(path)


def install_registry(monkeypatch, tmp_path, llm_block=LLM_BLOCK, extra_sections=""):
    """Charge le registre synthétique et le branche sur l'application."""
    registry = load_registry(write_registry(tmp_path, llm_block, extra_sections))
    monkeypatch.setattr(main, "registry", registry)
    return registry


UPSTREAM_OK = {
    "model": "chat-model:latest",
    "message": {"role": "assistant", "content": "Bonjour."},
    "done": True,
    "done_reason": "stop",
    "total_duration": 1_500_000_000,
    "load_duration": 300_000_000,
    "prompt_eval_count": 12,
    "prompt_eval_duration": 200_000_000,
    "eval_count": 5,
    "eval_duration": 1_000_000_000,
}


_KEEP = object()


def upstream(message=_KEEP, **overrides):
    """Réponse `/api/chat` simulée. `message=None` est une forme VOULUE (le
    test de robustesse l'envoie) : seule l'absence d'argument garde le défaut."""
    data = json.loads(json.dumps(UPSTREAM_OK))
    if message is not _KEEP:
        data["message"] = message
    data.update(overrides)
    return data


def install_chat(monkeypatch, data=None, exc=None):
    """Double de `client.chat` : capture les kwargs, rend `data` ou lève."""
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        if exc is not None:
            raise exc
        return json.loads(json.dumps(data if data is not None else UPSTREAM_OK))

    monkeypatch.setattr(main.client, "chat", fake_chat)
    return captured


class FakeChatStream:
    """Double de `ChatStream` au niveau de la route."""

    def __init__(self, events, raises=None, on_event=None):
        self._events = list(events)
        self._raises = raises
        self._on_event = on_event
        self.closed = False

    def events(self):
        for event in self._events:
            if self._on_event is not None:
                self._on_event(event)
            yield event
        if self._raises is not None:
            raise self._raises

    def close(self):
        self.closed = True


def install_chat_stream(monkeypatch, stream):
    captured = {}

    def fake_open(**kwargs):
        captured.update(kwargs)
        return stream

    monkeypatch.setattr(main.client, "open_chat_stream", fake_open)
    return captured


def chat_event(content="", thinking=None, done=False, **extra):
    """Un événement `/api/chat` tel qu'Ollama l'émet en flux."""
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    event = {"model": "chat-model:latest", "message": message, "done": done}
    event.update(extra)
    return event


STREAM_DONE = dict(
    done=True, done_reason="stop", total_duration=1_500_000_000,
    load_duration=300_000_000, prompt_eval_count=12,
    prompt_eval_duration=200_000_000, eval_count=5, eval_duration=1_000_000_000,
)


def ndjson(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def messages(*turns):
    """(role, content) → liste de messages."""
    return [{"role": role, "content": content} for role, content in turns]


BASIC = messages(("system", "Tu es Aramis."), ("user", "Bonjour, qui es-tu ?"))
