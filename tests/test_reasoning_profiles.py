"""Lot C1 — raisonnement gouverné : profils `none` / `toggle` / `levels`,
mapping pur vers `think`, validation registre, route `/v1/chat`
(400 avant Ollama sur toute valeur non honorable), politique d'exposition
de la réflexion, `/health/runtime`.

Fixtures SYNTHÉTIQUES uniquement : aucun profil gpt-oss / glm n'existe dans
le registre embarqué ; les familles sont représentées par `tiered`
(niveaux non désactivables) et `toggle` (on/off sans niveaux).
"""

import logging

import pytest

from app import main
from app.reasoning import (
    ALL_MODES, ERR_DISABLE_NOT_SUPPORTED, ERR_MODE_NOT_SUPPORTED, ERR_NOT_SUPPORTED,
    IMPLICIT_NONE, ReasoningModeError, ReasoningProfile, build_reasoning_profile,
    map_to_think, resolve_reasoning,
)
from app.registry import RegistryError, load_registry
from tests.chat_support import (
    BASIC, LLM_BLOCK, REGISTRY_HEAD, FakeChatStream, STREAM_DONE, chat_event, install_chat,
    install_chat_stream, install_registry, ndjson, upstream, write_registry,
)

THINK_SENTINEL = "ZZREASONINGTRACEZZ"


@pytest.fixture
def reg(monkeypatch, tmp_path):
    return install_registry(monkeypatch, tmp_path)


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _post(clientapp, model, **body):
    return clientapp.post("/v1/chat", json={"model": model, "messages": BASIC, **body})


def _event(caplog, name):
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith(f"event={name} "):
            return dict(p.split("=", 1) for p in m.split(" ")[1:] if "=" in p)
    return None


# --------------------------------------------------------------------------- #
# 1. Fonction pure : mapping Gateway → Ollama `think`
# --------------------------------------------------------------------------- #
TOGGLE = build_reasoning_profile("toggle", {"type": "toggle", "default": "off"})
TOGGLE_ON_ONLY = build_reasoning_profile("t-on", {"type": "toggle", "default": "on", "disable_supported": False})
TIERED = build_reasoning_profile("tiered", {"type": "levels", "allowed": ["low", "medium", "high"],
                                            "default": "low", "disable_supported": False})
TIERED_OFF = build_reasoning_profile("tiered-off", {"type": "levels", "allowed": ["low", "high"],
                                                    "default": "high", "disable_supported": True})


@pytest.mark.parametrize("profile,requested,mode,think", [
    (IMPLICIT_NONE, None, None, None),
    (IMPLICIT_NONE, "none", None, None),
    (TOGGLE, None, "off", False),
    (TOGGLE, "off", "off", False),
    (TOGGLE, "none", "off", False),          # synonyme
    (TOGGLE, "on", "on", True),
    (TOGGLE_ON_ONLY, None, "on", True),
    (TIERED, None, "low", "low"),
    (TIERED, "low", "low", "low"),
    (TIERED, "medium", "medium", "medium"),
    (TIERED, "high", "high", "high"),
    (TIERED_OFF, None, "high", "high"),
    (TIERED_OFF, "none", "none", False),
    (TIERED_OFF, "off", "none", False),
    (TIERED_OFF, "low", "low", "low"),
])
def test_resolve_reasoning_mapping_table(profile, requested, mode, think):
    decision = resolve_reasoning(profile, requested)
    assert (decision.mode, decision.think) == (mode, think)
    assert decision.requested is (requested is not None)


@pytest.mark.parametrize("profile,requested,code", [
    (IMPLICIT_NONE, "low", ERR_NOT_SUPPORTED),
    (IMPLICIT_NONE, "on", ERR_NOT_SUPPORTED),
    (IMPLICIT_NONE, "off", ERR_NOT_SUPPORTED),
    (TOGGLE, "low", ERR_MODE_NOT_SUPPORTED),
    (TOGGLE, "high", ERR_MODE_NOT_SUPPORTED),
    (TOGGLE_ON_ONLY, "off", ERR_DISABLE_NOT_SUPPORTED),
    (TOGGLE_ON_ONLY, "none", ERR_DISABLE_NOT_SUPPORTED),
    (TIERED, "none", ERR_DISABLE_NOT_SUPPORTED),
    (TIERED, "off", ERR_DISABLE_NOT_SUPPORTED),
    (TIERED, "on", ERR_MODE_NOT_SUPPORTED),
    (TIERED_OFF, "medium", ERR_MODE_NOT_SUPPORTED),     # niveau hors allowed
    (TIERED, "potato", ERR_MODE_NOT_SUPPORTED),
    (TOGGLE, "true", ERR_MODE_NOT_SUPPORTED),
])
def test_resolve_reasoning_refusals(profile, requested, code):
    with pytest.raises(ReasoningModeError) as info:
        resolve_reasoning(profile, requested)
    assert info.value.code == code
    assert info.value.extra["reasoning_profile"] == profile.name


def test_non_disableable_profile_never_yields_think_false():
    """Un modèle non désactivable : AUCUN chemin ne produit `think: false`."""
    for requested in (None,) + ALL_MODES:
        try:
            decision = resolve_reasoning(TIERED, requested)
        except ReasoningModeError:
            continue
        assert decision.think is not False and decision.think is not None


def test_map_to_think_is_total_over_modes():
    assert map_to_think(IMPLICIT_NONE, "high") is None
    assert map_to_think(TOGGLE, "on") is True and map_to_think(TOGGLE, "off") is False
    assert map_to_think(TIERED, "medium") == "medium" and map_to_think(TIERED_OFF, "none") is False
    assert map_to_think(TIERED, None) is None


# --------------------------------------------------------------------------- #
# 2. Validation des profils (registre)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,fragment", [
    ("pas un mapping", "mapping"),
    ({"type": "magic"}, "type inconnu"),
    ({"type": "none", "default": "low"}, "n'a pas de sens"),
    ({"type": "none", "disable_supported": False}, "disable_supported"),
    ({"type": "toggle", "default": "low"}, "off ou on"),
    ({"type": "toggle", "allowed": ["on"]}, "allowed"),
    ({"type": "toggle", "default": "off", "disable_supported": False}, "impose default=on"),
    ({"type": "levels", "default": "low"}, "allowed"),
    ({"type": "levels", "allowed": [], "default": "low"}, "allowed"),
    ({"type": "levels", "allowed": ["ultra"], "default": "ultra"}, "low/medium/high"),
    ({"type": "levels", "allowed": ["low", "low"], "default": "low"}, "doublon"),
    ({"type": "levels", "allowed": ["low"]}, "default est obligatoire"),
    ({"type": "levels", "allowed": ["low"], "default": "high"}, "n'est pas dans allowed"),
    ({"type": "levels", "allowed": ["low"], "default": "low", "expose_thinking": "yes"}, "booléen"),
    ({"type": "levels", "allowed": ["low"], "default": "low", "extra": 1}, "champ inconnu"),
    ({"type": "toggle", "disable_supported": "no"}, "booléen"),
])
def test_build_reasoning_profile_rejects_invalid_declarations(raw, fragment):
    with pytest.raises(ValueError) as info:
        build_reasoning_profile("p", raw)
    assert fragment in str(info.value)


def test_levels_allowed_is_normalised_in_canonical_order():
    p = build_reasoning_profile("p", {"type": "levels", "allowed": ["high", "low"], "default": "low"})
    assert p.allowed == ("low", "high") and p.disable_supported is False
    assert p.to_public_dict() == {"type": "levels", "default": "low", "allowed": ["low", "high"],
                                  "disable_supported": False, "expose_thinking": False}


def test_registry_loads_reasoning_profiles(reg):
    assert set(reg.reasoning_profiles) == {"none", "toggle", "toggle-exposed", "tiered", "tiered-off"}
    assert reg.reasoning_profile_of(reg.llm_models["chat-tiered"]).type == "levels"
    assert reg.reasoning_profile_of(reg.llm_models["nochat-test"]) is IMPLICIT_NONE
    assert reg.llm_models["chat-tiered"].to_public_dict()["reasoning_profile"] == "tiered"


def test_registry_refuses_chat_without_reasoning_profile(tmp_path):
    block = LLM_BLOCK.replace("    reasoning_profile: none\n    production_allowed: true\n    staging_allowed: true\n  chat-toggle:",
                              "    production_allowed: true\n    staging_allowed: true\n  chat-toggle:", 1)
    assert block != LLM_BLOCK
    with pytest.raises(RegistryError) as info:
        load_registry(write_registry(tmp_path, block))
    assert "exige un reasoning_profile" in str(info.value)


def test_registry_refuses_unknown_reasoning_profile_reference(tmp_path):
    block = LLM_BLOCK.replace("reasoning_profile: tiered\n", "reasoning_profile: fantome\n", 1)
    with pytest.raises(RegistryError) as info:
        load_registry(write_registry(tmp_path, block))
    assert "reasoning_profile inconnu" in str(info.value)


def test_registry_refuses_invalid_profile_section(tmp_path):
    bad = REGISTRY_HEAD.replace("  none:\n    type: none\n", "  none:\n    type: none\n    default: on\n")
    assert bad != REGISTRY_HEAD
    path = tmp_path / "r.yaml"; path.write_text(bad + LLM_BLOCK, encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(str(path))


EMBEDDED_REASONING = {
    # modèle chat : (profil de raisonnement, capabilities exactes)
    "oss-reasoning": ("oss-tiered", ["chat", "structured_output"]),   # C2 : levels non désactivable
    "qwen-reasoning": ("qwen-thinking", ["chat", "structured_output"]),   # 2026-09-27 : toggle non désactivable
}


def test_embedded_registry_reasoning_is_limited_to_the_c2_entry():
    """Registre EMBARQUÉ : deux profils de raisonnement (`oss-tiered`,
    `qwen-thinking`) portés par deux modèles chat staging-only. Les modèles
    antérieurs restent sans profil et sans capability chat ; glm absent."""
    assert set(main.registry.reasoning_profiles) == {"oss-tiered", "qwen-thinking"}
    for logical, profile in main.registry.llm_models.items():
        if logical in EMBEDDED_REASONING:
            rp, caps = EMBEDDED_REASONING[logical]
            assert profile.reasoning_profile == rp and profile.capabilities == caps, logical
            continue
        assert profile.reasoning_profile is None, logical
        assert "chat" not in profile.capabilities, logical
        assert not profile.ollama_name.startswith(("gpt-oss", "glm")), logical
    text = open(main.settings.models_registry_path, encoding="utf-8").read()
    assert "glm-4.7" not in text


# --------------------------------------------------------------------------- #
# 3. Route : validation avant Ollama, mapping relayé
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model,requested,think,mode", [
    ("chat-test", None, None, None),
    ("chat-test", "none", None, None),
    ("chat-toggle", None, False, "off"),
    ("chat-toggle", "on", True, "on"),
    ("chat-toggle", "off", False, "off"),
    ("chat-tiered", None, "low", "low"),
    ("chat-tiered", "high", "high", "high"),
    ("chat-tiered-off", None, "high", "high"),
    ("chat-tiered-off", "none", False, "none"),
    ("chat-tiered-off", "low", "low", "low"),
])
def test_route_relays_validated_think_only(clientapp, reg, monkeypatch, logs, model, requested, think, mode):
    captured = install_chat(monkeypatch)
    body = {} if requested is None else {"reasoning": requested}
    r = _post(clientapp, model, **body)
    assert r.status_code == 200, r.text
    assert captured["think"] == think and type(captured["think"]) is type(think)
    assert r.json()["reasoning"] == mode
    ev = _event(logs, "chat.ok")
    assert ev.get("reasoning") == mode if mode else "reasoning" not in ev
    assert ev["reasoning_profile"] == reg.llm_models[model].reasoning_profile


@pytest.mark.parametrize("model,requested,code", [
    ("chat-test", "low", ERR_NOT_SUPPORTED),
    ("chat-test", "on", ERR_NOT_SUPPORTED),
    ("chat-toggle", "low", ERR_MODE_NOT_SUPPORTED),
    ("chat-toggle", "medium", ERR_MODE_NOT_SUPPORTED),
    ("chat-tiered", "none", ERR_DISABLE_NOT_SUPPORTED),
    ("chat-tiered", "off", ERR_DISABLE_NOT_SUPPORTED),
    ("chat-tiered", "on", ERR_MODE_NOT_SUPPORTED),
    ("chat-tiered-off", "medium", ERR_MODE_NOT_SUPPORTED),
])
def test_route_refuses_unhonourable_modes_before_ollama(clientapp, reg, monkeypatch, logs, model, requested, code):
    captured = install_chat(monkeypatch)
    r = _post(clientapp, model, reasoning=requested)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == code and detail["requested"] == requested
    assert detail["model_logical"] == model and "allowed" in detail
    assert captured == {}
    assert main.counters.snapshot()["chat_rejected_reasoning_total"] == 1
    ev = _event(logs, "chat.reasoning_rejected")
    assert ev["code"] == code and ev["requested"] == requested


@pytest.mark.parametrize("value", ["potato", "TRUE", True, 1, "High", ""])
def test_unknown_reasoning_values_are_400_by_schema(clientapp, reg, monkeypatch, value):
    captured = install_chat(monkeypatch)
    r = _post(clientapp, "chat-tiered", reasoning=value)
    assert r.status_code == 400 and captured == {}


def test_raw_think_field_is_still_refused(clientapp, reg, monkeypatch):
    captured = install_chat(monkeypatch)
    for value in (True, False, "high"):
        assert _post(clientapp, "chat-tiered", think=value).status_code == 400
    assert captured == {}


def test_stream_applies_the_same_reasoning_plan(clientapp, reg, monkeypatch):
    captured = install_chat_stream(monkeypatch, FakeChatStream([chat_event(content="x"), chat_event(**STREAM_DONE)]))
    r = clientapp.post("/v1/chat/stream", json={"model": "chat-tiered", "messages": BASIC, "reasoning": "medium"})
    assert r.status_code == 200 and captured["think"] == "medium"
    assert ndjson(r)[-1]["reasoning"] == "medium"
    captured.clear()
    r = clientapp.post("/v1/chat/stream", json={"model": "chat-tiered", "messages": BASIC, "reasoning": "none"})
    assert r.status_code == 400 and captured == {}


def test_generate_default_think_policy_does_not_leak_into_chat(clientapp, reg, monkeypatch, tmp_path):
    """`default_think` reste une politique /v1/generate ; /v1/chat ne connaît
    que le reasoning_profile."""
    block = LLM_BLOCK.replace("    reasoning_profile: none\n    production_allowed: true\n    staging_allowed: true\n  chat-toggle:",
                              "    reasoning_profile: none\n    default_think: false\n    production_allowed: true\n    staging_allowed: true\n  chat-toggle:", 1)
    assert block != LLM_BLOCK
    install_registry(monkeypatch, tmp_path, block)
    captured = install_chat(monkeypatch)
    assert _post(clientapp, "chat-test").status_code == 200
    assert captured["think"] is None
    gen = {}
    monkeypatch.setattr(main.client, "generate", lambda **kw: (gen.update(kw), {"response": "ok", "done": True})[1])
    clientapp.post("/v1/generate", json={"prompt": "x", "model": "chat-test"})
    assert gen["think"] is False


# --------------------------------------------------------------------------- #
# 4. Exposition de la réflexion
# --------------------------------------------------------------------------- #
def test_thinking_hidden_by_default_even_when_reasoning_is_on(clientapp, reg, monkeypatch, logs):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "ok", "thinking": THINK_SENTINEL}))
    body = _post(clientapp, "chat-toggle", reasoning="on").json()
    assert body["thinking"] is None and body["thinking_exposed"] is False
    assert body["thinking_chars"] == len(THINK_SENTINEL) and body["reasoning"] == "on"
    assert THINK_SENTINEL not in "\n".join(r.getMessage() for r in logs.records)


def test_thinking_exposed_when_profile_says_so(clientapp, reg, monkeypatch, logs):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "ok", "thinking": THINK_SENTINEL}))
    body = _post(clientapp, "chat-toggle-exposed").json()
    assert body["thinking"] == THINK_SENTINEL and body["thinking_exposed"] is True
    assert body["message"]["content"] == "ok"                      # jamais fusionné
    assert THINK_SENTINEL not in "\n".join(r.getMessage() for r in logs.records)   # jamais loggé


def test_exposed_profile_without_thinking_yields_null(clientapp, reg, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "ok"}))
    body = _post(clientapp, "chat-toggle-exposed").json()
    assert body["thinking"] is None and body["thinking_exposed"] is True and body["thinking_chars"] == 0


def test_stream_emits_thinking_chunks_only_when_exposed(clientapp, reg, monkeypatch, logs):
    events_in = [chat_event(thinking=THINK_SENTINEL[:5]), chat_event(thinking=THINK_SENTINEL[5:]),
                 chat_event(content="ok"), chat_event(**STREAM_DONE)]
    install_chat_stream(monkeypatch, FakeChatStream(events_in))
    r = clientapp.post("/v1/chat/stream", json={"model": "chat-toggle-exposed", "messages": BASIC})
    events = ndjson(r)
    assert [e["type"] for e in events] == ["thinking", "thinking", "content", "done"]
    assert "".join(e["delta"] for e in events[:2]) == THINK_SENTINEL
    assert events[-1]["thinking_exposed"] is True and events[-1]["reasoning"] == "on"
    install_chat_stream(monkeypatch, FakeChatStream(events_in))
    r = clientapp.post("/v1/chat/stream", json={"model": "chat-toggle", "messages": BASIC, "reasoning": "on"})
    assert [e["type"] for e in ndjson(r)] == ["content", "done"] and THINK_SENTINEL not in r.text
    assert THINK_SENTINEL not in "\n".join(x.getMessage() for x in logs.records)


def test_content_empty_under_budget_with_reasoning_on(clientapp, reg, monkeypatch):
    """Cas B avec raisonnement piloté : 200, content_empty, done_reason, mode."""
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "", "thinking": "x" * 500},
                                       done_reason="length"))
    body = _post(clientapp, "chat-tiered", reasoning="high", options={"num_predict": 64}).json()
    assert body["content_empty"] is True and body["done_reason"] == "length"
    assert body["reasoning"] == "high" and body["thinking_chars"] == 500 and body["thinking"] is None


# --------------------------------------------------------------------------- #
# 5. Observabilité
# --------------------------------------------------------------------------- #
def test_health_runtime_exposes_reasoning_profiles_and_chat_flag(clientapp, reg):
    body = clientapp.get("/health/runtime").json()
    assert body["reasoning_profiles"]["tiered"] == {"type": "levels", "default": "low",
                                                    "allowed": ["low", "medium", "high"],
                                                    "disable_supported": False, "expose_thinking": False}
    by_name = {m["model_logical"]: m for m in body["llm_models"]}
    assert by_name["chat-tiered"]["chat"] is True and by_name["chat-tiered"]["reasoning_profile"] == "tiered"
    assert by_name["nochat-test"]["chat"] is False and by_name["nochat-test"]["reasoning"]["type"] == "none"


def test_health_runtime_on_embedded_registry(clientapp):
    body = clientapp.get("/health/runtime").json()
    assert set(body["reasoning_profiles"]) == {"oss-tiered", "qwen-thinking"}
    for m in body["llm_models"]:
        if m["model_logical"] == "oss-reasoning":
            assert m["chat"] is True and m["reasoning"]["type"] == "levels"
            continue
        if m["model_logical"] == "qwen-reasoning":
            assert m["chat"] is True and m["reasoning"] == {"type": "toggle", "default": "on", "allowed": ["on"],
                                                          "disable_supported": False, "expose_thinking": False}
            continue
        assert m["chat"] is False and m["reasoning"] == IMPLICIT_NONE.to_public_dict()
