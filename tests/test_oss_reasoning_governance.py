"""Lot C2 — gouvernance de gpt-oss:20b (`oss-reasoning`), STAGING UNIQUEMENT.

Registre EMBARQUÉ (pas de fixture synthétique) : ce que ces tests figent est
l'entrée réelle qui sera déployée. Aucun appel Ollama : `client.chat`,
`client.open_chat_stream` et `client.generate` sont doublés et leurs
arguments capturés ; sur toute valeur refusée, la capture reste vide.
"""

import logging

import pytest

from app import main
from app.reasoning import ERR_DISABLE_NOT_SUPPORTED, ERR_MODE_NOT_SUPPORTED
from tests.chat_support import (
    BASIC, STREAM_DONE, FakeChatStream, chat_event, install_chat, install_chat_stream, ndjson, upstream,
)

LOGICAL = "oss-reasoning"
PHYSICAL = "gpt-oss:20b"
NAMES = (LOGICAL, PHYSICAL)
PRE_C2_MODELS = {"llama-chat", "qwen-judge", "qwen-large", "vision-large", "gemma-large"}
# Modèles chat ajoutés APRÈS C2 (chacun a sa propre suite de gouvernance).
POST_C2_CHAT_MODELS = {"qwen-reasoning"}


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
# 1. Registre : entrée et profils
# --------------------------------------------------------------------------- #
def test_oss_reasoning_entry_is_exactly_as_qualified():
    p = main.registry.llm_models[LOGICAL]
    assert p.ollama_name == PHYSICAL
    assert p.capabilities == ["chat", "structured_output"]   # ni completion, ni tools
    assert p.resource_class == "large"
    assert p.context_profile == "oss-long-context"
    assert p.generation_profile == "reasoning-bounded"
    assert p.keep_alive_profile == "transient-governed"
    assert p.reasoning_profile == "oss-tiered"
    assert p.staging_allowed is True and p.production_allowed is True
    assert p.structured_output is False and p.default_think is None
    assert p.modalities == ["text"] and not p.supports_image


def test_oss_profiles_values():
    ctx = main.registry.context_profile_of(main.registry.llm_models[LOGICAL])
    assert (ctx.num_ctx, ctx.max_num_ctx) == (16384, 65536)
    gen = main.registry.generation_profile_of(main.registry.llm_models[LOGICAL])
    assert (gen.default_num_predict, gen.max_num_predict) == (4096, 8192)
    ka = main.registry.keep_alive_of(main.registry.llm_models[LOGICAL])
    assert (ka.keep_alive, ka.client_override) == ("2m", False)
    r = main.registry.reasoning_profile_of(main.registry.llm_models[LOGICAL])
    assert r.to_public_dict() == {"type": "levels", "default": "low", "allowed": ["low", "medium", "high"],
                                  "disable_supported": False, "expose_thinking": False}


def test_physical_name_resolves_to_the_same_profile():
    a = main.registry.resolve_llm(LOGICAL)
    b = main.registry.resolve_llm(PHYSICAL)
    assert a.profile is b.profile and a.via_logical and not b.via_logical
    assert main.registry.resolve_llm("gpt-oss") is None          # tag explicite : pas d'alias nu
    assert main.registry.resolve_llm("gpt-oss:latest") is None


def test_no_glm_and_no_new_default():
    text = open(main.settings.models_registry_path, encoding="utf-8").read()
    assert "glm" not in text.lower()
    assert main.settings.generate_model == "llama3.1:8b"
    assert main.registry.vision_default_model == "vision-large"
    assert set(n for n, p in main.registry.llm_models.items() if "chat" in p.capabilities) == {LOGICAL} | POST_C2_CHAT_MODELS


def test_pre_c2_models_are_untouched():
    # Seule évolution admise depuis C2 : la capability `structured_output` sur le
    # rail completion (G2 : qwen-large ; activation du 2026-09-26 : qwen-judge,
    # vision-large, gemma-large). Toute autre propriété est figée.
    expected = {
        "llama-chat": ("llama3.1:8b", ["completion", "tools"], "medium", "long-context", "resident", "chat-bounded", None, True, True),
        "qwen-judge": ("qwen3:14b", ["completion", "tools", "thinking", "structured_output"], "medium", "verifier", "resident", "judge-bounded", False, True, True),
        "qwen-large": ("qwen3:30b-instruct", ["completion", "tools", "structured_output"], "large", "large-general", "transient", "planner-bounded", None, True, True),
        "vision-large": ("mistral-small3.2:24b", ["completion", "vision", "tools", "structured_output"], "large", "large-general", "transient", "vision-bounded", None, True, True),
        "gemma-large": ("gemma3:27b", ["completion", "structured_output"], "large", "large-general", "transient-governed", "gemma-bounded", None, True, True),
    }
    assert set(main.registry.llm_models) == PRE_C2_MODELS | {LOGICAL} | POST_C2_CHAT_MODELS
    for name, (oll, caps, rc, ctx, ka, gen, think, prod, stag) in expected.items():
        p = main.registry.llm_models[name]
        assert (p.ollama_name, p.capabilities, p.resource_class, p.context_profile, p.keep_alive_profile,
                p.generation_profile, p.default_think, p.production_allowed, p.staging_allowed) == \
               (oll, caps, rc, ctx, ka, gen, think, prod, stag), name
        assert p.reasoning_profile is None and p.structured_output is True
    # Profils historiques inchangés.
    cp = main.registry.context_profiles
    assert {n: (c.num_ctx, c.max_num_ctx) for n, c in cp.items() if n not in ("oss-long-context", "reasoning-16k")} == {
        "chat-small": (4096, 16384), "verifier": (8192, 16384), "large-general": (8192, 16384), "long-context": (16384, 32768)}
    gp = main.registry.generation_profiles
    assert {n: (g.default_num_predict, g.max_num_predict) for n, g in gp.items() if n != "reasoning-bounded"} == {
        "planner-bounded": (2048, 4096), "chat-bounded": (3072, 8192), "judge-bounded": (3072, 4096),
        "vision-bounded": (2048, 4096), "gemma-bounded": (2048, 4096)}
    assert {n: (k.keep_alive, k.client_override) for n, k in main.registry.keep_alive_profiles.items()} == {
        "resident": ("30m", True), "transient": ("2m", True), "default": (None, True), "transient-governed": ("2m", False)}
    assert set(main.registry.reasoning_profiles) == {"oss-tiered", "qwen-thinking"}


# --------------------------------------------------------------------------- #
# 2. Environnement : staging seulement, alias sans contournement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NAMES)
def test_production_serves_both_names_with_the_same_plan(clientapp, production, monkeypatch, name):
    """Lot production-full-capability-enablement : oss-reasoning est autorisé en
    production avec EXACTEMENT le plan staging (profils, budget, think low, thinking masqué)."""
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, model=name)
    assert r.status_code == 200, r.text
    assert captured["model"] == PHYSICAL and captured["think"] == "low" and captured["keep_alive"] == "2m"
    assert captured["options"] == {"num_ctx": 16384, "num_predict": 4096}
    assert r.json()["reasoning"] == "low"
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
    assert captured["options"] == {"num_ctx": 16384, "num_predict": 4096}
    assert captured["think"] == "low" and captured["keep_alive"] == "2m"
    assert r.json()["reasoning"] == "low" and r.json()["model"] == PHYSICAL


def test_both_names_produce_identical_chat_plans(staging):
    a = main._plan_chat(main.ChatRequest(model=LOGICAL, messages=BASIC))
    b = main._plan_chat(main.ChatRequest(model=PHYSICAL, messages=BASIC))
    assert a == b
    assert a.inflight_key == f"chat:{PHYSICAL}" and a.expose_thinking is False


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_registry_governs_regardless_of_strict_flag(clientapp, staging, monkeypatch, logs, strict, name):
    monkeypatch.setattr(main.settings, "generation_strict_registry", strict)
    captured = install_chat(monkeypatch)
    assert _chat(clientapp, model=name).status_code == 200
    assert captured["options"]["num_predict"] == 4096 and captured["think"] == "low"
    snap = main.counters.snapshot()
    assert snap.get("generate_legacy_model_total", 0) == 0 and snap["chat_rejected_unregistered_total"] == 0
    assert _event(logs, "chat.ok")["model_logical"] == LOGICAL
    assert _event(logs, "generate.legacy_model") is None


# --------------------------------------------------------------------------- #
# 3. /v1/generate : capability completion absente → refus, jamais legacy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env", ["staging", "production"])
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_generate_refuses_gpt_oss_without_completion(clientapp, monkeypatch, logs, env, name, path):
    monkeypatch.setattr(main.settings, "environment", env)
    gen = {}
    monkeypatch.setattr(main.client, "generate", lambda **kw: (gen.update(kw), {"response": "x", "done": True})[1])
    monkeypatch.setattr(main.client, "open_generate_stream", lambda **kw: (gen.update(kw), None)[1])
    r = clientapp.post(path, json={"prompt": "x", "model": name})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "completion_not_supported"
    assert r.json()["detail"]["capabilities"] == ["chat", "structured_output"]
    assert gen == {}
    snap = main.counters.snapshot()
    assert snap["generate_rejected_capability_total"] == 1 and snap.get("generate_legacy_model_total", 0) == 0
    assert _event(logs, "generate.capability_not_supported")["model_logical"] == LOGICAL


def test_completion_gate_does_not_touch_pre_c2_models(clientapp, staging, monkeypatch):
    gen = {}
    monkeypatch.setattr(main.client, "generate", lambda **kw: (gen.update(kw), {"response": "x", "done": True})[1])
    for name in sorted(PRE_C2_MODELS):
        gen.clear()
        assert clientapp.post("/v1/generate", json={"prompt": "x", "model": name}).status_code == 200, name
        assert gen["model"] == main.registry.llm_models[name].ollama_name
    assert main.counters.snapshot()["generate_rejected_capability_total"] == 0


# --------------------------------------------------------------------------- #
# 4. Reasoning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("requested,think", [(None, "low"), ("low", "low"), ("medium", "medium"), ("high", "high")])
def test_reasoning_levels(clientapp, staging, monkeypatch, logs, requested, think):
    captured = install_chat(monkeypatch)
    body = {} if requested is None else {"reasoning": requested}
    r = _chat(clientapp, **body)
    assert r.status_code == 200
    assert captured["think"] == think and r.json()["reasoning"] == think
    ev = _event(logs, "chat.ok")
    assert ev["reasoning"] == think and ev["effective_think"] == think and ev["reasoning_profile"] == "oss-tiered"


@pytest.mark.parametrize("requested,code", [("none", ERR_DISABLE_NOT_SUPPORTED), ("off", ERR_DISABLE_NOT_SUPPORTED),
                                            ("on", ERR_MODE_NOT_SUPPORTED)])
def test_reasoning_refusals_before_ollama(clientapp, staging, monkeypatch, requested, code):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, reasoning=requested)
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == code and d["reasoning_profile"] == "oss-tiered" and d["allowed"] == ["low", "medium", "high"]
    assert captured == {}
    assert main.counters.snapshot()["chat_rejected_reasoning_total"] == 1


@pytest.mark.parametrize("value", ["potato", "LOW", True, False, 1, ""])
def test_reasoning_arbitrary_values_are_400(clientapp, staging, monkeypatch, value):
    captured = install_chat(monkeypatch)
    assert _chat(clientapp, reasoning=value).status_code == 400 and captured == {}


def test_think_never_false_for_gpt_oss(staging):
    """Aucune requête acceptée ne relaie `think: false` ni un `think` absent."""
    for requested in (None, "low", "medium", "high"):
        plan = main._plan_chat(main.ChatRequest(model=LOGICAL, messages=BASIC, reasoning=requested))
        assert plan.think in ("low", "medium", "high")


def test_thinking_hidden_but_counted(clientapp, staging, monkeypatch, logs):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "144.", "thinking": "ZZOSSTHINKZZ" * 10}))
    body = _chat(clientapp, reasoning="high").json()
    assert body["thinking"] is None and body["thinking_exposed"] is False
    assert body["thinking_chars"] == 120 and body["message"]["content"] == "144."
    assert "ZZOSSTHINKZZ" not in "\n".join(r.getMessage() for r in logs.records)


def test_budget_consumed_by_reasoning_is_content_empty(clientapp, staging, monkeypatch, logs):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "", "thinking": "x" * 300},
                                       done_reason="length", eval_count=64))
    r = _chat(clientapp, reasoning="high", options={"num_predict": 64})
    assert r.status_code == 200
    b = r.json()
    assert b["content_empty"] is True and b["done_reason"] == "length" and b["thinking_chars"] == 300
    assert main.counters.snapshot()["chat_empty_content_total"] == 1
    assert _event(logs, "chat.empty_content")["done_reason"] == "length"


# --------------------------------------------------------------------------- #
# 5. Contexte et budget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_ctx,expected", [(None, 16384), (8192, 8192), (16384, 16384), (32768, 32768), (65536, 65536)])
def test_num_ctx_governed(clientapp, staging, monkeypatch, num_ctx, expected):
    captured = install_chat(monkeypatch)
    body = {} if num_ctx is None else {"options": {"num_ctx": num_ctx}}
    assert _chat(clientapp, **body).status_code == 200
    assert captured["options"]["num_ctx"] == expected


@pytest.mark.parametrize("num_ctx", [65537, 131072, 0, -1, 4096.0, True, "4096"])
def test_num_ctx_out_of_bound_or_invalid_is_400(clientapp, staging, monkeypatch, num_ctx):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, options={"num_ctx": num_ctx})
    assert r.status_code == 400 and captured == {}
    assert r.json()["detail"].get("context_profile") == "oss-long-context"


@pytest.mark.parametrize("num_predict,expected", [(None, 4096), (1, 1), (64, 64), (4096, 4096), (8192, 8192)])
def test_num_predict_governed(clientapp, staging, monkeypatch, num_predict, expected):
    captured = install_chat(monkeypatch)
    body = {} if num_predict is None else {"options": {"num_predict": num_predict}}
    assert _chat(clientapp, **body).status_code == 200
    assert captured["options"]["num_predict"] == expected


@pytest.mark.parametrize("num_predict", [8193, 0, -1, 2.5, True, "64"])
def test_num_predict_out_of_bound_or_invalid_is_400(clientapp, staging, monkeypatch, num_predict):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, options={"num_predict": num_predict})
    assert r.status_code == 400 and captured == {}
    assert r.json()["detail"].get("generation_profile") == "reasoning-bounded"


# --------------------------------------------------------------------------- #
# 6. keep_alive, tools/format, streaming
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [-1, "-1", "30m", 0])
def test_keep_alive_client_override_refused(clientapp, staging, monkeypatch, value):
    captured = install_chat(monkeypatch)
    r = _chat(clientapp, keep_alive=value)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "keep_alive_not_overridable"
    assert captured == {}


@pytest.mark.parametrize("extra", [{"tools": []}, {"format": "json"}, {"response_format": {}}, {"json_schema": {}}, {"think": "low"}])
def test_tools_and_structured_output_still_refused(clientapp, staging, monkeypatch, extra):
    captured = install_chat(monkeypatch)
    assert _chat(clientapp, **extra).status_code == 400 and captured == {}


def test_stream_plan_and_events(clientapp, staging, monkeypatch, logs):
    captured = install_chat_stream(monkeypatch, FakeChatStream([chat_event(thinking="ZZOSSTHINKZZ"), chat_event(content="144"),
                                                                chat_event(**STREAM_DONE)]))
    r = _chat(clientapp, path="/v1/chat/stream", reasoning="high")
    assert r.status_code == 200 and captured["think"] == "high" and captured["options"]["num_ctx"] == 16384
    events = ndjson(r)
    assert [e["type"] for e in events] == ["content", "done"]
    assert events[-1]["reasoning"] == "high" and events[-1]["thinking_exposed"] is False
    assert "ZZOSSTHINKZZ" not in r.text
    assert "ZZOSSTHINKZZ" not in "\n".join(x.getMessage() for x in logs.records)


def test_health_runtime_and_models_expose_the_entry(clientapp, staging, monkeypatch):
    body = clientapp.get("/health/runtime").json()
    entry = {m["model_logical"]: m for m in body["llm_models"]}[LOGICAL]
    assert entry["chat"] is True and entry["allowed_here"] is True and entry["max_num_ctx"] == 65536
    assert entry["keep_alive"] == "2m" and entry["keep_alive_client_override"] is False
    assert body["reasoning_profiles"]["oss-tiered"] == {"type": "levels", "default": "low", "allowed": ["low", "medium", "high"],
                                                         "disable_supported": False, "expose_thinking": False}
    monkeypatch.setattr(main.settings, "environment", "production")
    entry = {m["model_logical"]: m for m in clientapp.get("/health/runtime").json()["llm_models"]}[LOGICAL]
    assert entry["allowed_here"] is True


def test_health_and_ready_do_not_depend_on_gpt_oss(clientapp, staging, monkeypatch):
    monkeypatch.setattr(main.client, "ping", lambda timeout=5: (True, ["nomic-embed-text:latest"]))
    assert clientapp.get("/health").status_code == 200
    r = clientapp.get("/health/ready")
    assert r.status_code == 200, r.text
    assert "gpt-oss" not in r.text
