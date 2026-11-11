"""Intégration du 2026-09-27 — gouvernance de qwen3:30b-a3b-thinking-2507
(`qwen-reasoning`), STAGING UNIQUEMENT, rail chat seul.

Registre EMBARQUÉ (pas de fixture synthétique) : ce que ces tests figent est
l'entrée réelle déployée. Aucun appel Ollama : `client.chat`,
`client.open_chat_stream` et `client.generate` sont doublés et leurs
arguments capturés ; sur toute valeur refusée, la capture reste vide.

Contrat mesuré (recon du 2026-09-27) : `think: true` sépare la réflexion ;
`think: false` fait FUIR le raisonnement et `</think>` dans `content`. La
Gateway ne doit donc JAMAIS émettre `think: false` pour ce modèle — c'est le
profil `qwen-thinking` (toggle, défaut on, non désactivable) qui l'impose,
sans aucune branche Python propre au modèle.
"""

import logging

import pytest

from app import main
from app.reasoning import ERR_DISABLE_NOT_SUPPORTED, ERR_MODE_NOT_SUPPORTED
from tests.chat_support import (
    BASIC, STREAM_DONE, FakeChatStream, chat_event, install_chat, install_chat_stream, ndjson, upstream,
)

LOGICAL = "qwen-reasoning"
PHYSICAL = "qwen3:30b-a3b-thinking-2507-q4_K_M"
NAMES = (LOGICAL, PHYSICAL)
PROFILE = "qwen-thinking"
SENTINEL = "ZZQWENTHINKZZ"


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "staging")


@pytest.fixture
def production(monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "production")


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _chat(clientapp, model=LOGICAL, path="/v1/chat", **body):
    return clientapp.post(path, json={"model": model, "messages": BASIC, **body})


def _event(caplog, name):
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith(f"event={name} "):
            return dict(p.split("=", 1) for p in m.split(" ")[1:] if "=" in p)
    return None


# --------------------------------------------------------------------------- #
# 1. Registre : entrée et profils EXACTS
# --------------------------------------------------------------------------- #
def test_qwen_reasoning_entry_is_exactly_as_qualified():
    p = main.registry.llm_models[LOGICAL]
    assert p.ollama_name == PHYSICAL
    assert p.capabilities == ["chat", "structured_output"]   # ni completion, ni tools, ni vision
    assert p.resource_class == "large"
    assert p.context_profile == "reasoning-16k"
    assert p.generation_profile == "reasoning-bounded"
    assert p.keep_alive_profile == "transient-governed"
    assert p.reasoning_profile == PROFILE
    assert p.staging_allowed is True and p.production_allowed is True
    assert p.structured_output is False and p.default_think is None
    assert p.modalities == ["text"] and not p.supports_image


def test_qwen_profiles_values():
    p = main.registry.llm_models[LOGICAL]
    ctx = main.registry.context_profile_of(p)
    assert (ctx.num_ctx, ctx.max_num_ctx) == (16384, 16384)      # 32K volontairement fermé
    gen = main.registry.generation_profile_of(p)
    assert (gen.default_num_predict, gen.max_num_predict) == (4096, 8192)
    ka = main.registry.keep_alive_of(p)
    assert (ka.keep_alive, ka.client_override) == ("2m", False)
    r = main.registry.reasoning_profile_of(p)
    assert r.to_public_dict() == {"type": "toggle", "default": "on", "allowed": ["on"],
                                  "disable_supported": False, "expose_thinking": False}


def test_physical_name_resolves_to_the_same_profile_and_no_loose_alias():
    a = main.registry.resolve_llm(LOGICAL)
    b = main.registry.resolve_llm(PHYSICAL)
    assert a.profile is b.profile and a.via_logical and not b.via_logical
    # Le tag sans quantification, l'hybride et l'instruct ne sont PAS ce modèle.
    assert main.registry.resolve_llm("qwen3:30b-a3b-thinking-2507") is None
    assert main.registry.resolve_llm("qwen3:30b-a3b") is None
    assert main.registry.resolve_llm("qwen3:30b-instruct").profile.logical_name == "qwen-large"


def test_oss_reasoning_is_untouched_by_the_qwen_entry():
    p = main.registry.llm_models["oss-reasoning"]
    assert p.ollama_name == "gpt-oss:20b" and p.capabilities == ["chat", "structured_output"]
    assert p.reasoning_profile == "oss-tiered" and p.context_profile == "oss-long-context"
    assert p.generation_profile == "reasoning-bounded" and p.keep_alive_profile == "transient-governed"
    assert p.production_allowed is True and p.staging_allowed is True
    oss = main.registry.reasoning_profiles["oss-tiered"]
    assert oss.to_public_dict() == {"type": "levels", "default": "low", "allowed": ["low", "medium", "high"],
                                    "disable_supported": False, "expose_thinking": False}
    ctx = main.registry.context_profile_of(p)
    assert (ctx.num_ctx, ctx.max_num_ctx) == (16384, 65536)


# --------------------------------------------------------------------------- #
# 2. Environnement : staging seulement, alias sans contournement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NAMES)
def test_production_serves_both_names_with_the_same_plan(clientapp, production, monkeypatch, name):
    """Lot production-full-capability-enablement : qwen-reasoning autorisé en
    production, même plan qu'en staging (toggle on, jamais think:false, thinking masqué)."""
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, model=name)
    assert r.status_code == 200, r.text
    assert captured["model"] == PHYSICAL and captured.get("think") is not False
    assert r.json()["thinking"] is None and r.json()["thinking_exposed"] is False
    assert main.counters.snapshot()["chat_rejected_environment_total"] == 0


def test_production_still_refuses_chat_model_on_generate(clientapp, production, monkeypatch):
    captured = install_chat(monkeypatch)
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": LOGICAL})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "completion_not_supported"
    assert captured == {}


@pytest.mark.parametrize("name", NAMES)
def test_staging_serves_both_names_with_the_same_plan(clientapp, staging, monkeypatch, name):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, model=name)
    assert r.status_code == 200, r.text
    assert captured["model"] == PHYSICAL
    assert captured["think"] is True                                   # défaut du profil : on
    assert captured["options"] == {"num_ctx": 16384, "num_predict": 4096}
    assert captured["keep_alive"] == "2m"
    assert "format" not in captured
    body = r.json()
    assert body["model"] == PHYSICAL and body["reasoning"] == "on"
    assert body["thinking"] is None and body["thinking_exposed"] is False


# --------------------------------------------------------------------------- #
# 3. Raisonnement : on ou rien — `think: false` n'est JAMAIS émis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body,expected_think", [({}, True), ({"reasoning": "on"}, True)])
def test_reasoning_absent_or_on_relays_think_true(clientapp, staging, monkeypatch, body, expected_think):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, **body)
    assert r.status_code == 200 and captured["think"] is expected_think
    assert r.json()["reasoning"] == "on"


@pytest.mark.parametrize("mode,code", [("off", ERR_DISABLE_NOT_SUPPORTED), ("none", ERR_DISABLE_NOT_SUPPORTED),
                                       ("low", ERR_MODE_NOT_SUPPORTED), ("medium", ERR_MODE_NOT_SUPPORTED),
                                       ("high", ERR_MODE_NOT_SUPPORTED)])
@pytest.mark.parametrize("path", ["/v1/chat", "/v1/chat/stream"])
def test_reasoning_refusals_before_ollama(clientapp, staging, monkeypatch, logs, mode, code, path):
    """off/none : non désactivable ; low/medium/high : pas de niveaux pour
    Qwen3 — aucune traduction vers `on`, aucun repli, aucun appel Ollama."""
    chat = install_chat(monkeypatch)
    stream = install_chat_stream(monkeypatch, FakeChatStream([chat_event(**STREAM_DONE)]))
    r = _chat(clientapp, path=path, reasoning=mode)
    assert r.status_code == 400, r.text
    d = r.json()["detail"]
    assert d["code"] == code and d["model_logical"] == LOGICAL and d["reasoning_profile"] == PROFILE
    assert d["allowed"] == ["on"]
    assert chat == {} and stream == {}
    ev = _event(logs, "chat.reasoning_rejected")
    assert ev["model_logical"] == LOGICAL and ev["reasoning_profile"] == PROFILE and ev["requested"] == mode


def test_think_never_false_for_qwen(clientapp, staging, monkeypatch):
    """Balayage de TOUTES les valeurs publiques : soit `think: true` est relayé,
    soit rien ne part. `think: false` n'existe pas pour ce modèle."""
    seen = []
    for body in ({}, {"reasoning": "on"}, {"reasoning": "off"}, {"reasoning": "none"},
                 {"reasoning": "low"}, {"reasoning": "medium"}, {"reasoning": "high"}, {"reasoning": "potato"}):
        captured = install_chat(monkeypatch)
        _chat(clientapp, **body)
        if captured:
            seen.append(captured["think"])
    assert seen == [True, True]


@pytest.mark.parametrize("value", [True, False, "low"])
def test_raw_think_field_is_refused(clientapp, staging, monkeypatch, value):
    """Le contrat public n'a pas de champ `think` : pas de contournement Qwen."""
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, think=value)
    assert r.status_code == 400 and captured == {}


def test_unknown_reasoning_value_is_400_before_ollama(clientapp, staging, monkeypatch):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, reasoning="potato")
    assert r.status_code == 400 and captured == {}


# --------------------------------------------------------------------------- #
# 4. Thinking : séparé en amont, masqué en public, compté, jamais journalisé
# --------------------------------------------------------------------------- #
def test_thinking_hidden_but_counted(clientapp, staging, monkeypatch, logs):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "Réponse.", "thinking": SENTINEL * 3}))
    r = _chat(clientapp)
    body = r.json()
    assert r.status_code == 200
    assert body["message"]["content"] == "Réponse." and body["content_empty"] is False
    assert body["thinking"] is None and body["thinking_exposed"] is False and body["thinking_chars"] == 3 * len(SENTINEL)
    assert SENTINEL not in r.text
    joined = "\n".join(x.getMessage() for x in logs.records)
    assert SENTINEL not in joined and "Réponse." not in joined and "Bonjour" not in joined
    ev = _event(logs, "chat.ok")
    assert ev["model_logical"] == LOGICAL and ev["reasoning"] == "on" and ev["effective_think"] == "True"
    assert ev["thinking_chars"] == str(3 * len(SENTINEL)) and ev["reasoning_profile"] == PROFILE


def test_budget_consumed_by_reasoning_is_content_empty(clientapp, staging, monkeypatch, logs):
    """La réflexion compte dans `num_predict` : budget épuisé → 200,
    `content_empty: true`, `done_reason: length`, jamais un texte de substitution."""
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "", "thinking": SENTINEL}, done_reason="length"))
    r = _chat(clientapp, options={"num_predict": 4096})
    body = r.json()
    assert r.status_code == 200 and body["content_empty"] is True and body["done_reason"] == "length"
    assert body["message"]["content"] == "" and body["thinking"] is None and body["thinking_chars"] == len(SENTINEL)
    assert _event(logs, "chat.empty_content")["done_reason"] == "length"
    assert main.counters.snapshot()["chat_empty_content_total"] == 1


# --------------------------------------------------------------------------- #
# 5. Contexte 16K strict et budget de sortie
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("requested,expected", [(None, 16384), (8192, 8192), (16384, 16384)])
def test_num_ctx_governed(clientapp, staging, monkeypatch, requested, expected):
    captured = install_chat(monkeypatch)
    body = {} if requested is None else {"options": {"num_ctx": requested}}
    r = _chat(clientapp, **body)
    assert r.status_code == 200 and captured["options"]["num_ctx"] == expected


@pytest.mark.parametrize("value", [16385, 32768, 65536, 131072, 262144, 0, -1, "16384", 4096.5, True])
def test_num_ctx_above_16k_or_invalid_is_400_never_clamped(clientapp, staging, monkeypatch, value):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, options={"num_ctx": value})
    assert r.status_code == 400 and captured == {}


@pytest.mark.parametrize("requested,expected", [(None, 4096), (4096, 4096), (8192, 8192), (64, 64)])
def test_num_predict_governed(clientapp, staging, monkeypatch, requested, expected):
    captured = install_chat(monkeypatch)
    body = {} if requested is None else {"options": {"num_predict": requested}}
    r = _chat(clientapp, **body)
    assert r.status_code == 200 and captured["options"]["num_predict"] == expected


@pytest.mark.parametrize("value", [8193, 16384, 0, -1, "4096", True])
def test_num_predict_out_of_bound_or_invalid_is_400(clientapp, staging, monkeypatch, value):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, options={"num_predict": value})
    assert r.status_code == 400 and captured == {}


def test_keep_alive_client_override_refused(clientapp, staging, monkeypatch):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, keep_alive="30m")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "keep_alive_not_overridable" and captured == {}


# --------------------------------------------------------------------------- #
# 6. Capabilities : chat seul — generate, structured, tools refusés
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_generate_refuses_qwen_reasoning_without_completion(clientapp, staging, monkeypatch, name, path):
    called = {}
    monkeypatch.setattr(main.client, "generate", lambda **kw: called.update(kw))
    monkeypatch.setattr(main.client, "open_generate_stream", lambda **kw: called.update(kw))
    r = clientapp.post(path, json={"model": name, "prompt": "x"})
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["code"] == "completion_not_supported" and r.json()["detail"]["model_logical"] == LOGICAL
    assert called == {}


def test_generate_with_response_format_is_refused_too(clientapp, staging, monkeypatch):
    called = {}
    monkeypatch.setattr(main.client, "generate", lambda **kw: called.update(kw))
    r = clientapp.post("/v1/generate", json={"model": LOGICAL, "prompt": "x",
                                             "response_format": {"type": "json_schema", "schema": {"type": "object"}}})
    assert r.status_code == 400 and called == {}


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["status"],
          "properties": {"status": {"type": "string", "enum": ["OK", "KO"]}}}


def test_structured_output_is_relayed_with_think_true(clientapp, staging, monkeypatch, logs):
    """Capability `structured_output` (activée après 15/15 canaries staging) :
    schéma validé puis relayé dans `format`, `think: true` conservé, réflexion
    séparée et masquée, document parsé dans `structured_output`."""
    captured = install_chat(monkeypatch, upstream(message={"role": "assistant", "content": '{"status": "OK"}', "thinking": SENTINEL}))
    r = _chat(clientapp, response_format={"type": "json_schema", "schema": SCHEMA})
    assert r.status_code == 200, r.text
    assert captured["format"] == SCHEMA and captured["think"] is True and captured["model"] == PHYSICAL
    body = r.json()
    assert body["structured_output"] == {"status": "OK"} and body["message"]["content"] == '{"status": "OK"}'
    assert body["thinking"] is None and body["thinking_chars"] == len(SENTINEL) and body["reasoning"] == "on"
    assert SENTINEL not in r.text and SENTINEL not in "\n".join(x.getMessage() for x in logs.records)
    ev = _event(logs, "chat.ok")
    assert ev["structured_output"] == "True" and ev["structured_valid"] == "True" and ev["effective_think"] == "True"


def test_structured_output_refuses_off_and_levels_before_ollama(clientapp, staging, monkeypatch):
    for mode in ("off", "none", "low", "medium", "high"):
        captured = install_chat(monkeypatch)
        r = _chat(clientapp, reasoning=mode, response_format={"type": "json_schema", "schema": SCHEMA})
        assert r.status_code == 400 and captured == {}, mode


def test_structured_output_reasoning_leak_is_a_502(clientapp, staging, monkeypatch):
    """Si un jour le raisonnement fuyait dans `content` (cas mesuré avec
    `think: false`, jamais émis), la sortie structurée échoue fail-closed."""
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "réflexion... </think>\n{\"status\": \"OK\"}", "thinking": None}))
    r = _chat(clientapp, response_format={"type": "json_schema", "schema": SCHEMA})
    assert r.status_code == 502 and r.json()["detail"]["code"] == "structured_output_invalid"


def test_tools_are_refused(clientapp, staging, monkeypatch):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}])
    assert r.status_code == 400 and captured == {}


# --------------------------------------------------------------------------- #
# 7. Flux : même plan, thinking masqué, done unique
# --------------------------------------------------------------------------- #
def test_stream_plan_and_events(clientapp, staging, monkeypatch, logs):
    captured = install_chat_stream(monkeypatch, FakeChatStream([
        chat_event(thinking=SENTINEL), chat_event(content="Bon"), chat_event(content="jour."), chat_event(**STREAM_DONE),
    ]))
    r = _chat(clientapp, path="/v1/chat/stream")
    assert r.status_code == 200
    assert captured["model"] == PHYSICAL and captured["think"] is True
    assert captured["options"] == {"num_ctx": 16384, "num_predict": 4096} and captured["keep_alive"] == "2m"
    events = ndjson(r)
    assert [e["type"] for e in events] == ["content", "content", "done"]
    assert events[-1]["reasoning"] == "on" and events[-1]["thinking_exposed"] is False and events[-1]["content_empty"] is False
    assert SENTINEL not in r.text
    assert SENTINEL not in "\n".join(x.getMessage() for x in logs.records)


def test_stream_refuses_response_format_before_open(clientapp, staging, monkeypatch):
    captured = install_chat_stream(monkeypatch, FakeChatStream([chat_event(**STREAM_DONE)]))
    r = _chat(clientapp, path="/v1/chat/stream", response_format={"type": "json_schema", "schema": {"type": "object"}})
    assert r.status_code == 400 and captured == {}


# --------------------------------------------------------------------------- #
# 8. Health/runtime
# --------------------------------------------------------------------------- #
def test_health_runtime_exposes_the_entry(clientapp, staging, monkeypatch):
    body = clientapp.get("/health/runtime").json()
    entry = {m["model_logical"]: m for m in body["llm_models"]}[LOGICAL]
    assert entry["ollama_name"] == PHYSICAL and entry["capabilities"] == ["chat", "structured_output"]
    assert entry["chat"] is True and entry["allowed_here"] is True
    assert entry["default_num_ctx"] == 16384 and entry["max_num_ctx"] == 16384
    assert entry["keep_alive"] == "2m" and entry["keep_alive_client_override"] is False
    assert entry["production_allowed"] is True and entry["staging_allowed"] is True
    assert entry["reasoning"] == {"type": "toggle", "default": "on", "allowed": ["on"],
                                  "disable_supported": False, "expose_thinking": False}
    assert body["reasoning_profiles"][PROFILE] == entry["reasoning"]
    assert body["generation_profiles"]["reasoning-bounded"] == {"default_num_predict": 4096, "max_num_predict": 8192}
    assert body["context_profiles"]["reasoning-16k"] == {"num_ctx": 16384, "max_num_ctx": 16384}
    monkeypatch.setattr(main.settings, "environment", "production")
    entry = {m["model_logical"]: m for m in clientapp.get("/health/runtime").json()["llm_models"]}[LOGICAL]
    assert entry["allowed_here"] is True


def test_models_listing_relays_the_installed_artifact(clientapp, staging, monkeypatch):
    """`/v1/models` est l'inventaire Ollama (contrat existant, inchangé) :
    l'artefact physique y apparaît sous son tag exact, sans champ inventé."""
    monkeypatch.setattr(main.client, "list_models", lambda timeout=5: [{"name": PHYSICAL}, {"name": "gpt-oss:20b"}])
    r = clientapp.get("/v1/models")
    assert r.status_code == 200
    assert PHYSICAL in {m.get("name") for m in r.json()["models"]}
