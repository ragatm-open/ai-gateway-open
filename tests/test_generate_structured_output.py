"""Lot G2 — sortie structurée GOUVERNÉE sur le rail completion `/v1/generate`.

`response_format` (même contrat que /v1/chat) : capability `structured_output`
opt-in et orthogonale à `chat`, garde de compatibilité raisonnement, schéma
validé AVANT Ollama, relayé dans `format` de /api/generate, sortie analysée
strictement et revalidée APRÈS, `structured_output` + `response_schema_sha256`
présents UNIQUEMENT en mode structuré. Absent : comportement legacy intact
(différentiel C1 en plus). Registre SYNTHÉTIQUE pour les profils de test.
"""

import hashlib
import json
import logging

import pytest

from app import main
from app.registry import load_registry
from app.structured import (
    ERR_INVALID_SCHEMA, ERR_OUTPUT_EMPTY, ERR_OUTPUT_INVALID, ERR_OUTPUT_MISMATCH, ERR_SCHEMA_TOO_LARGE,
    ERR_UNSUPPORTED_KEYWORD, SCHEMA_MAX_DEPTH, canonical_schema_bytes, schema_sha256,
)
from tests.chat_support import REGISTRY_HEAD

LLM_BLOCK = """\
llm_models:
  qwen-like:                       # completion + tools + structured_output, SANS chat (cible G2)
    ollama_name: qwen-like:30b
    capabilities: [completion, tools, structured_output]
    resource_class: large
    context_profile: verifier
    keep_alive_profile: resident
    generation_profile: judge-bounded
    structured_output: true
    production_allowed: true
    staging_allowed: true
  plain-completion:                # legacy : completion seule
    ollama_name: plain:latest
    capabilities: [completion]
    resource_class: medium
    context_profile: verifier
    keep_alive_profile: resident
    generation_profile: judge-bounded
    production_allowed: true
    staging_allowed: true
  thinking-completion:             # capability thinking déclarée, default_think false via nom logique
    ollama_name: thinker:14b
    capabilities: [completion, thinking, structured_output]
    resource_class: medium
    context_profile: verifier
    keep_alive_profile: resident
    generation_profile: judge-bounded
    default_think: false
    production_allowed: true
    staging_allowed: true
  locked-reasoning:                # raisonnement non désactivable (famille gpt-oss), completion déclarée pour le test
    ollama_name: locked:20b
    capabilities: [completion, structured_output]
    resource_class: large
    context_profile: verifier
    keep_alive_profile: resident
    generation_profile: judge-bounded
    reasoning_profile: tiered
    production_allowed: true
    staging_allowed: true
  staging-only-structured:
    ollama_name: stg-structured:latest
    capabilities: [completion, structured_output]
    resource_class: medium
    context_profile: verifier
    keep_alive_profile: resident
    generation_profile: judge-bounded
    production_allowed: false
    staging_allowed: true
"""

SIMPLE = {"type": "object", "additionalProperties": False, "required": ["flag_a", "flag_b"],
          "properties": {"flag_a": {"type": "boolean"}, "flag_b": {"type": "boolean"}}}
IDS = ["C1", "C2", "C3", "C4"]
LIST_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["selected_ids"],
               "properties": {"selected_ids": {"type": "array", "items": {"type": "string", "enum": IDS}}}}
NESTED = {"type": "object", "additionalProperties": False, "required": ["links"],
          "properties": {"links": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                         "required": ["target_id", "selected_ids"],
                         "properties": {"target_id": {"type": "string", "enum": ["T1", "T2", "T3"]},
                                        "selected_ids": {"type": "array", "items": {"type": "string", "enum": IDS}}}}}}}
RF = {"type": "json_schema", "schema": SIMPLE}
GOOD = '{"flag_a": true, "flag_b": false}'


@pytest.fixture
def reg(monkeypatch, tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(REGISTRY_HEAD + LLM_BLOCK, encoding="utf-8")
    registry = load_registry(str(path))
    monkeypatch.setattr(main, "registry", registry)
    monkeypatch.setattr(main.settings, "environment", "staging")
    return registry


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _upstream(response=GOOD, **extra):
    data = {"model": "qwen-like:30b", "response": response, "done": True, "done_reason": "stop",
            "prompt_eval_count": 20, "eval_count": 9, "total_duration": 900_000_000, "load_duration": 100_000_000,
            "prompt_eval_duration": 100_000_000, "eval_duration": 700_000_000}
    data.update(extra)
    return data


def _install(monkeypatch, data=None, exc=None):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        if exc is not None:
            raise exc
        return json.loads(json.dumps(data if data is not None else _upstream()))

    monkeypatch.setattr(main.client, "generate", fake)
    return captured


def _post(clientapp, model="qwen-like", rf=RF, path="/v1/generate", **body):
    payload = {"prompt": "canari", "model": model, **body}
    if rf is not None:
        payload["response_format"] = rf
    return clientapp.post(path, json=payload)


def _event(caplog, name):
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith(f"event={name} "):
            return dict(p.split("=", 1) for p in m.split(" ")[1:] if "=" in p)
    return None


def _all_logs(caplog):
    return "\n".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# 1. Chemin nominal : schéma relayé dans format, sortie validée, réponse enrichie
# --------------------------------------------------------------------------- #
def test_structured_request_relays_validated_schema_and_returns_parsed_output(clientapp, reg, monkeypatch, logs):
    captured = _install(monkeypatch)
    r = _post(clientapp, options={"num_predict": 64}, temperature=0)
    assert r.status_code == 200, r.text
    assert captured["format"] == SIMPLE                      # schéma VALIDÉ, relayé tel quel
    assert captured["model"] == "qwen-like:30b" and captured["options"] == {"num_ctx": 8192, "num_predict": 64}
    assert captured["prompt"] == "canari" and captured["temperature"] == 0
    body = r.json()
    assert body["structured_output"] == {"flag_a": True, "flag_b": False}
    assert body["text"] == GOOD                               # texte brut conservé
    assert body["response_schema_sha256"] == schema_sha256(SIMPLE)
    assert body["done_reason"] == "stop" and body["tokens"] == {"prompt": 20, "completion": 9, "total": 29}
    ev = _event(logs, "generate.ok")
    assert ev["structured_output"] == "True" and ev["structured_valid"] == "True"
    assert ev["schema_bytes"] and ev["schema_depth"] == "2" and ev["schema_nodes"] == "3"
    assert ev["response_schema_sha256"] == schema_sha256(SIMPLE) and ev["model_logical"] == "qwen-like"
    snap = main.counters.snapshot()
    assert snap["generate_structured_requests_total"] == 1 and snap["generate_structured_ok_total"] == 1


@pytest.mark.parametrize("schema,response,expected", [
    (LIST_SCHEMA, '{"selected_ids": []}', {"selected_ids": []}),
    (LIST_SCHEMA, '{"selected_ids": ["C1", "C4"]}', {"selected_ids": ["C1", "C4"]}),
    (NESTED, '{"links": [{"target_id": "T1", "selected_ids": ["C1"]}, {"target_id": "T2", "selected_ids": []}, {"target_id": "T3", "selected_ids": []}]}',
     {"links": [{"target_id": "T1", "selected_ids": ["C1"]}, {"target_id": "T2", "selected_ids": []}, {"target_id": "T3", "selected_ids": []}]}),
])
def test_lists_and_nested_targets_including_empty_lists(clientapp, reg, monkeypatch, schema, response, expected):
    _install(monkeypatch, _upstream(response))
    r = _post(clientapp, rf={"type": "json_schema", "schema": schema})
    assert r.status_code == 200 and r.json()["structured_output"] == expected
    assert r.json()["response_schema_sha256"] == schema_sha256(schema)


def test_physical_name_works_the_same(clientapp, reg, monkeypatch):
    captured = _install(monkeypatch)
    r = _post(clientapp, model="qwen-like:30b")
    assert r.status_code == 200 and captured["format"] == SIMPLE and r.json()["structured_output"]


# --------------------------------------------------------------------------- #
# 2. Schéma client : validation AVANT Ollama (mêmes règles que /v1/chat)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("schema", [
    SIMPLE, LIST_SCHEMA, NESTED,
    {"type": "object", "properties": {"opt": {"type": ["string", "null"]}}, "required": []},
    {"type": "object", "properties": {"k": {"type": "string", "enum": ["a", "b"], "const": "a"}}, "additionalProperties": False},
])
def test_valid_schemas_pass(clientapp, reg, monkeypatch, schema):
    captured = _install(monkeypatch, _upstream("{}") if schema is not SIMPLE and schema is not LIST_SCHEMA and schema is not NESTED else None)
    r = _post(clientapp, rf={"type": "json_schema", "schema": schema})
    assert r.status_code in (200, 502) and captured["format"] == schema


def _nest(n):
    node = {"type": "string"}
    for _ in range(n):
        node = {"type": "object", "properties": {"c": node}}
    return node


@pytest.mark.parametrize("schema,code", [
    ({"type": "object", "properties": {"a": {"$ref": "#/x"}}}, ERR_UNSUPPORTED_KEYWORD),
    ({"type": "object", "properties": {"a": {"type": "string", "pattern": "^x"}}}, ERR_UNSUPPORTED_KEYWORD),
    ({"type": "object", "properties": {"a": {"type": "string", "format": "email"}}}, ERR_UNSUPPORTED_KEYWORD),
    ({"type": "object", "properties": {"a": {"oneOf": [{"type": "string"}]}}}, ERR_UNSUPPORTED_KEYWORD),
    ({"type": "object", "properties": {"a": {"type": "wat"}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]}, ERR_INVALID_SCHEMA),
    ({"type": "array", "items": {"type": "string"}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(300)}}, ERR_SCHEMA_TOO_LARGE),
    (_nest(SCHEMA_MAX_DEPTH), ERR_SCHEMA_TOO_LARGE),
    ({"type": "object", "properties": {f"p{i}": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}} for i in range(200)}}, ERR_SCHEMA_TOO_LARGE),
])
def test_invalid_schema_is_400_before_ollama(clientapp, reg, monkeypatch, logs, schema, code):
    captured = _install(monkeypatch)
    r = _post(clientapp, rf={"type": "json_schema", "schema": schema})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == code and r.json()["detail"]["model_logical"] == "qwen-like"
    assert captured == {}
    assert main.counters.snapshot()["generate_structured_schema_rejected_total"] == 1
    assert _event(logs, "generate.schema_rejected")["code"] == code
    assert "response_schema_sha256" not in r.json()["detail"]


@pytest.mark.parametrize("rf", [{"type": "json_object"}, {"type": "text"}, {"type": "json_schema"}, {"schema": SIMPLE},
                                {"type": "json_schema", "schema": SIMPLE, "strict": True}, "json_schema"])
def test_response_format_shape_is_strict(clientapp, reg, monkeypatch, rf):
    captured = _install(monkeypatch)
    assert _post(clientapp, rf=rf).status_code == 400 and captured == {}


# --------------------------------------------------------------------------- #
# 3. Sortie du modèle : validation stricte APRÈS Ollama
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("response,code,counter", [
    ("", ERR_OUTPUT_EMPTY, "generate_structured_empty_total"),
    ("   ", ERR_OUTPUT_EMPTY, "generate_structured_empty_total"),
    ('{"flag_a": true, "flag_b": fal', ERR_OUTPUT_INVALID, "generate_structured_invalid_json_total"),
    ('{"flag_a": true, "flag_a": false, "flag_b": true}', ERR_OUTPUT_INVALID, "generate_structured_invalid_json_total"),
    ('{"flag_a": NaN, "flag_b": true}', ERR_OUTPUT_INVALID, "generate_structured_invalid_json_total"),
    ('{"flag_a": Infinity, "flag_b": true}', ERR_OUTPUT_INVALID, "generate_structured_invalid_json_total"),
    ("```json\n" + GOOD + "\n```", ERR_OUTPUT_INVALID, "generate_structured_invalid_json_total"),
    ("Voici : " + GOOD, ERR_OUTPUT_INVALID, "generate_structured_invalid_json_total"),
    (GOOD + " merci", ERR_OUTPUT_INVALID, "generate_structured_invalid_json_total"),
    (GOOD + GOOD, ERR_OUTPUT_INVALID, "generate_structured_invalid_json_total"),
    ('{"flag_a": true, "flag_b": "false"}', ERR_OUTPUT_MISMATCH, "generate_structured_schema_mismatch_total"),
    ('{"flag_a": true, "flag_b": false, "x": 1}', ERR_OUTPUT_MISMATCH, "generate_structured_schema_mismatch_total"),
    ('{"flag_a": true}', ERR_OUTPUT_MISMATCH, "generate_structured_schema_mismatch_total"),
    ('[true, false]', ERR_OUTPUT_MISMATCH, "generate_structured_schema_mismatch_total"),
])
def test_invalid_output_is_controlled_502(clientapp, reg, monkeypatch, logs, response, code, counter):
    _install(monkeypatch, _upstream(response))
    r = _post(clientapp)
    assert r.status_code == 502, r.text
    d = r.json()["detail"]
    assert d["code"] == code and d["provider"] == "ollama" and d["done_reason"] == "stop"
    assert "flag_a" not in d["message"] and "structured_output" not in r.json()
    snap = main.counters.snapshot()
    assert snap[counter] == 1 and snap["generate_structured_invalid_upstream_total"] == 1
    assert snap["generate_upstream_errors_total"] == 1 and snap["generate_structured_ok_total"] == 0
    ev = _event(logs, "generate.structured_invalid")
    assert ev["code"] == code and ev["response_schema_sha256"] == schema_sha256(SIMPLE) and "structured_valid" not in ev
    assert _event(logs, "generate.ok") is None and main.gauges.snapshot() == {}


def test_wrong_enum_is_mismatch(clientapp, reg, monkeypatch):
    _install(monkeypatch, _upstream('{"selected_ids": ["C9"]}'))
    r = _post(clientapp, rf={"type": "json_schema", "schema": LIST_SCHEMA})
    assert r.status_code == 502 and r.json()["detail"]["code"] == ERR_OUTPUT_MISMATCH


def test_length_with_complete_valid_document_is_accepted(clientapp, reg, monkeypatch):
    _install(monkeypatch, _upstream(GOOD, done_reason="length"))
    r = _post(clientapp)
    assert r.status_code == 200 and r.json()["done_reason"] == "length" and r.json()["structured_output"]["flag_a"] is True


def test_length_with_truncated_document_is_502(clientapp, reg, monkeypatch):
    _install(monkeypatch, _upstream('{"flag_a": tr', done_reason="length"))
    r = _post(clientapp)
    assert r.status_code == 502 and r.json()["detail"] == {"status": "error", "provider": "ollama",
                                                          "message": "sortie structurée : le contenu n'est pas un document JSON strict",
                                                          "code": ERR_OUTPUT_INVALID, "done_reason": "length"}


def test_upstream_failures_keep_legacy_mapping(clientapp, reg, monkeypatch):
    from app.errors import UpstreamError, UpstreamTimeout
    _install(monkeypatch, exc=UpstreamTimeout("échec /api/generate: HTTPConnectionPool(host='127.0.0.1', port=11434)"))
    r = _post(clientapp)
    assert r.status_code == 502 and r.json()["detail"]["message"] == "échec du service de génération : délai dépassé"
    _install(monkeypatch, exc=UpstreamError("échec /api/generate: HTTPConnectionPool(host='127.0.0.1', port=11434)"))
    r = _post(clientapp)
    assert r.status_code == 502 and r.json()["detail"]["message"] == "échec du service de génération"
    assert "127.0.0.1" not in r.text


# --------------------------------------------------------------------------- #
# 4. Gouvernance : capability orthogonale à chat, garde raisonnement, environnement
# --------------------------------------------------------------------------- #
def test_completion_plus_structured_without_chat_is_accepted_and_chat_route_still_refused(clientapp, reg, monkeypatch):
    profile = reg.llm_models["qwen-like"]
    assert "chat" not in profile.capabilities and "structured_output" in profile.capabilities
    _install(monkeypatch)
    assert _post(clientapp).status_code == 200
    chat_called = {}
    monkeypatch.setattr(main.client, "chat", lambda **kw: chat_called.update(kw))
    r = clientapp.post("/v1/chat", json={"model": "qwen-like", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "chat_not_supported" and chat_called == {}


def test_model_without_capability_is_refused_before_ollama(clientapp, reg, monkeypatch, logs):
    captured = _install(monkeypatch)
    r = _post(clientapp, model="plain-completion")
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "structured_output_not_supported" and d["capabilities"] == ["completion"]
    assert captured == {}
    assert main.counters.snapshot()["generate_structured_capability_rejected_total"] == 1
    assert _event(logs, "generate.structured_output_not_supported")["model_logical"] == "plain-completion"


def test_legacy_boolean_structured_output_is_not_the_authority(clientapp, monkeypatch, tmp_path):
    """Le booléen historique `structured_output: true` n'ouvre PAS le rail :
    seule la liste `capabilities` fait foi (cas exact de qwen-large réel)."""
    block = LLM_BLOCK.replace("    capabilities: [completion]\n    resource_class: medium\n    context_profile: verifier\n    keep_alive_profile: resident\n    generation_profile: judge-bounded\n    production_allowed: true",
                              "    capabilities: [completion]\n    resource_class: medium\n    context_profile: verifier\n    keep_alive_profile: resident\n    generation_profile: judge-bounded\n    structured_output: true\n    production_allowed: true", 1)
    assert block != LLM_BLOCK
    path = tmp_path / "r.yaml"; path.write_text(REGISTRY_HEAD + block, encoding="utf-8")
    monkeypatch.setattr(main, "registry", load_registry(str(path))); monkeypatch.setattr(main.settings, "environment", "staging")
    assert main.registry.llm_models["plain-completion"].structured_output is True
    captured = _install(monkeypatch)
    r = _post(clientapp, model="plain-completion")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "structured_output_not_supported" and captured == {}


@pytest.mark.parametrize("name", ["qwen-large", "qwen3:30b-instruct"])
def test_real_registry_qwen_large_is_structured_without_chat(clientapp, monkeypatch, name):
    """Registre EMBARQUÉ : qwen-large = completion + tools + structured_output,
    sans chat ; le rail completion structuré est servi, le rail chat refusé."""
    monkeypatch.setattr(main.settings, "environment", "staging")
    p = main.registry.llm_models["qwen-large"]
    assert p.capabilities == ["completion", "tools", "structured_output"] and "chat" not in p.capabilities
    assert p.reasoning_profile is None and p.context_profile == "large-general" and p.generation_profile == "planner-bounded"
    captured = _install(monkeypatch, _upstream(GOOD, model="qwen3:30b-instruct"))
    r = _post(clientapp, model=name)
    assert r.status_code == 200, r.text
    assert captured["model"] == "qwen3:30b-instruct" and captured["format"] == SIMPLE and captured["options"] == {"num_ctx": 8192, "num_predict": 2048}
    assert r.json()["structured_output"] == {"flag_a": True, "flag_b": False} and r.json()["model"] == "qwen3:30b-instruct"
    chat_called = {}
    monkeypatch.setattr(main.client, "chat", lambda **kw: chat_called.update(kw))
    r = clientapp.post("/v1/chat", json={"model": name, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "chat_not_supported" and chat_called == {}


# Lot d'activation (2026-09-26) : trois modèles completion déjà démontrés
# capables de respecter un JSON Schema (diagnostic natif 9/9) reçoivent la
# capability `structured_output`. Le tableau fige TOUT ce qui doit rester
# identique au plan legacy : upstream, options par défaut, keep_alive, think.
ENABLED_COMPLETION_MODELS = {
    # logique : (physique, capabilities, options par défaut, keep_alive, think)
    "qwen-judge":   ("qwen3:14b",            ["completion", "tools", "thinking", "structured_output"], {"num_ctx": 8192, "num_predict": 3072}, "30m", False),
    "vision-large": ("mistral-small3.2:24b", ["completion", "vision", "tools", "structured_output"],   {"num_ctx": 8192, "num_predict": 2048}, "2m",  None),
    "gemma-large":  ("gemma3:27b",           ["completion", "structured_output"],                     {"num_ctx": 8192, "num_predict": 2048}, "2m",  None),
}


@pytest.mark.parametrize("name", sorted(ENABLED_COMPLETION_MODELS))
def test_real_registry_enabled_models_serve_structured_completion(clientapp, monkeypatch, name):
    """Registre EMBARQUÉ : les trois modèles activés servent `response_format`
    sur le rail completion avec EXACTEMENT le plan legacy (mêmes options,
    keep_alive, think) ; `chat` reste refusé ; aucune autre propriété n'a bougé."""
    physical, caps, options, keep_alive, think = ENABLED_COMPLETION_MODELS[name]
    monkeypatch.setattr(main.settings, "environment", "staging")
    p = main.registry.llm_models[name]
    assert p.capabilities == caps and "chat" not in p.capabilities and p.ollama_name == physical
    assert p.reasoning_profile is None and p.production_allowed and p.staging_allowed
    captured = _install(monkeypatch, _upstream(GOOD, model=physical))
    r = _post(clientapp, model=name)
    assert r.status_code == 200, r.text
    assert captured["model"] == physical and captured["format"] == SIMPLE
    assert captured["options"] == options and captured["keep_alive"] == keep_alive and captured["think"] is think
    body = r.json()
    assert body["structured_output"] == {"flag_a": True, "flag_b": False} and body["model"] == physical
    assert body["response_schema_sha256"] == hashlib.sha256(
        json.dumps(SIMPLE, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    # legacy : sans response_format, même plan, aucun champ structuré
    captured.clear()
    r = _post(clientapp, model=name, rf=None)
    assert r.status_code == 200 and "format" not in captured
    assert captured["options"] == options and captured["keep_alive"] == keep_alive and captured["think"] is think
    assert "structured_output" not in r.json() and "response_schema_sha256" not in r.json()
    # chat toujours refusé
    chat_called = {}
    monkeypatch.setattr(main.client, "chat", lambda **kw: chat_called.update(kw))
    r = clientapp.post("/v1/chat", json={"model": name, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "chat_not_supported" and chat_called == {}


def test_real_registry_qwen_judge_structured_keeps_think_false(clientapp, monkeypatch, logs):
    """qwen-judge + response_format : `effective_think = false` via le nom logique
    (default_think du profil), donc AUCUN `structured_output_reasoning_incompatible` ;
    le nom physique `qwen3:14b` (canal natif actif) reste refusé ; `think: true`
    explicite sur le nom logique est refusé — le blocker n'a pas changé."""
    monkeypatch.setattr(main.settings, "environment", "staging")
    p = main.registry.llm_models["qwen-judge"]
    assert p.default_think is False and "thinking" in p.capabilities
    captured = _install(monkeypatch, _upstream(GOOD, model="qwen3:14b"))
    r = _post(clientapp, model="qwen-judge")
    assert r.status_code == 200, r.text
    assert captured["think"] is False and captured["format"] == SIMPLE and captured["model"] == "qwen3:14b"
    ev = _event(logs, "generate.ok")
    assert ev["model_logical"] == "qwen-judge" and ev["effective_think"] == "False" and ev["structured_output"] == "True"
    assert main.counters.snapshot().get("generate_structured_reasoning_rejected_total", 0) == 0
    captured.clear()
    r = _post(clientapp, model="qwen3:14b")
    assert r.status_code == 400 and r.json()["detail"]["reason"] == "thinking_channel_active" and captured == {}
    r = _post(clientapp, model="qwen-judge", think=True)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "structured_output_reasoning_incompatible" and captured == {}


def test_real_registry_gemma_keep_alive_stays_locked_with_response_format(clientapp, monkeypatch):
    """gemma-large : profil `transient-governed` — un keep_alive client reste
    refusé (400 keep_alive_not_overridable) AVEC response_format, avant Ollama."""
    monkeypatch.setattr(main.settings, "environment", "staging")
    captured = _install(monkeypatch, _upstream(GOOD, model="gemma3:27b"))
    r = _post(clientapp, model="gemma-large", keep_alive="10m")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "keep_alive_not_overridable" and captured == {}


def test_real_registry_negative_controls_unchanged(clientapp, monkeypatch):
    """llama-chat n'a PAS reçu la capability (400 structured_output_not_supported,
    capabilities inchangées) ; oss-reasoning reste hors rail completion
    (400 completion_not_supported) ; qwen-large inchangé. Aucun appel Ollama."""
    monkeypatch.setattr(main.settings, "environment", "staging")
    caps = main.registry.llm_models
    assert caps["llama-chat"].capabilities == ["completion", "tools"]
    assert caps["oss-reasoning"].capabilities == ["chat", "structured_output"]
    assert caps["qwen-large"].capabilities == ["completion", "tools", "structured_output"]
    captured = _install(monkeypatch)
    r = _post(clientapp, model="llama-chat")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "structured_output_not_supported"
    assert r.json()["detail"]["capabilities"] == ["completion", "tools"]
    r = _post(clientapp, model="oss-reasoning")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "completion_not_supported"
    assert captured == {}


def test_unregistered_model_is_refused_even_with_strict_off(clientapp, reg, monkeypatch, logs):
    monkeypatch.setattr(main.settings, "generation_strict_registry", False)
    captured = _install(monkeypatch)
    r = _post(clientapp, model="inconnu:latest")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "structured_output_not_supported"
    assert r.json()["detail"]["model_logical"] is None and captured == {}
    assert main.counters.snapshot().get("generate_legacy_model_total", 0) == 0


def test_non_disableable_reasoning_profile_is_blocked(clientapp, reg, monkeypatch, logs):
    captured = _install(monkeypatch)
    for name in ("locked-reasoning", "locked:20b"):
        r = _post(clientapp, model=name)
        assert r.status_code == 400
        d = r.json()["detail"]
        assert d["code"] == "structured_output_reasoning_incompatible" and d["reason"] == "reasoning_not_disableable"
    assert captured == {}
    assert main.counters.snapshot()["generate_structured_reasoning_rejected_total"] == 2
    assert _event(logs, "generate.structured_output_reasoning_incompatible")["reason"] == "reasoning_not_disableable"


def test_thinking_model_requires_think_false(clientapp, reg, monkeypatch):
    captured = _install(monkeypatch, _upstream(GOOD, model="thinker:14b"))
    # nom logique : default_think=false → admis, think=False relayé
    r = _post(clientapp, model="thinking-completion")
    assert r.status_code == 200 and captured["think"] is False and captured["format"] == SIMPLE
    # nom physique : pas de default_think → canal actif → refus
    captured.clear()
    r = _post(clientapp, model="thinker:14b")
    assert r.status_code == 400 and r.json()["detail"]["reason"] == "thinking_channel_active" and captured == {}
    # nom physique + think=false explicite → admis
    r = _post(clientapp, model="thinker:14b", think=False)
    assert r.status_code == 200 and captured["think"] is False
    # nom logique + think=true explicite → refus (le client rallume le canal)
    captured.clear()
    r = _post(clientapp, model="thinking-completion", think=True)
    assert r.status_code == 400 and captured == {}


def test_blocker_helper_is_pure_and_generic(reg):
    from app.reasoning import IMPLICIT_NONE, build_reasoning_profile
    plain = reg.llm_models["plain-completion"]; thinker = reg.llm_models["thinking-completion"]
    tiered = build_reasoning_profile("t", {"type": "levels", "allowed": ["low"], "default": "low", "disable_supported": False})
    toggle = build_reasoning_profile("g", {"type": "toggle", "default": "off"})
    assert main.completion_structured_output_blocker(plain, IMPLICIT_NONE, None) is None
    assert main.completion_structured_output_blocker(plain, tiered, False) == "reasoning_not_disableable"
    assert main.completion_structured_output_blocker(thinker, IMPLICIT_NONE, None) == "thinking_channel_active"
    assert main.completion_structured_output_blocker(thinker, IMPLICIT_NONE, True) == "thinking_channel_active"
    assert main.completion_structured_output_blocker(thinker, toggle, False) is None


def test_environment_allowlist_applies(clientapp, reg, monkeypatch):
    captured = _install(monkeypatch)
    monkeypatch.setattr(main.settings, "environment", "production")
    r = _post(clientapp, model="staging-only-structured")
    assert r.status_code == 400 and "environnement production" in r.json()["detail"]["message"] and captured == {}


# --------------------------------------------------------------------------- #
# 5. Conflit format / response_format et legacy intact
# --------------------------------------------------------------------------- #
def test_format_and_response_format_conflict_is_400(clientapp, reg, monkeypatch):
    captured = _install(monkeypatch)
    for fmt in ("json", {"type": "object"}):
        r = _post(clientapp, format=fmt)
        assert r.status_code == 400 and r.json()["detail"]["code"] == "generate_format_conflict"
    assert captured == {}
    assert main.counters.snapshot()["generate_structured_conflict_rejected_total"] == 2
    # le conflit est refusé même sur un modèle sans capability / hors registre
    assert _post(clientapp, model="plain-completion", format="json").status_code == 400
    assert _post(clientapp, model="inconnu:latest", format="json").json()["detail"]["code"] == "generate_format_conflict"


def test_format_only_keeps_raw_relay_without_validation(clientapp, reg, monkeypatch):
    captured = _install(monkeypatch, _upstream("pas du json {"))
    r = _post(clientapp, rf=None, format={"type": "object", "properties": {"a": {"$ref": "#/x"}}})
    assert r.status_code == 200                                  # relais brut : ni pré- ni post-validation
    assert captured["format"] == {"type": "object", "properties": {"a": {"$ref": "#/x"}}}
    body = r.json()
    assert body["text"] == "pas du json {" and "structured_output" not in body and "response_schema_sha256" not in body
    assert main.counters.snapshot()["generate_structured_requests_total"] == 0


def test_legacy_response_has_no_structured_fields(clientapp, reg, monkeypatch):
    _install(monkeypatch)
    body = _post(clientapp, rf=None).json()
    assert list(body) == ["text", "provider", "upstream_provider", "model", "tokens", "usage", "cost", "done", "done_reason"]
    assert "structured_output" not in body and "response_schema_sha256" not in body
    # sérialisation directe du modèle : mêmes clés
    from app.schemas import GenerateResponse, TokenCounts, Usage
    dumped = GenerateResponse(text="x", model="m", tokens=TokenCounts(prompt=1, completion=1, total=2),
                              usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)).model_dump()
    assert "structured_output" not in dumped and "response_schema_sha256" not in dumped


def test_legacy_plan_and_wire_unchanged_without_response_format(clientapp, reg, monkeypatch):
    captured = _install(monkeypatch)
    r = _post(clientapp, rf=None, options={"num_ctx": 4096}, think=False, keep_alive="1m", system="s", temperature=0.7)
    assert r.status_code == 200
    assert captured == {"model": "qwen-like:30b", "prompt": "canari", "system": "s", "temperature": 0.7,
                        "options": {"num_ctx": 4096, "num_predict": 3072}, "think": False, "keep_alive": "1m"}
    plan = main._plan_generation(main.GenerateRequest(prompt="x", model="qwen-like"))
    assert plan.response_schema is None and plan.schema_stats is None and plan.response_schema_sha256 is None


def test_structured_plan_carries_schema_and_hash(reg):
    plan = main._plan_generation(main.GenerateRequest(prompt="x", model="qwen-like", response_format=RF))
    assert plan.response_schema == SIMPLE and plan.response_schema_sha256 == schema_sha256(SIMPLE)
    assert plan.schema_stats.nodes == 3 and plan.options == {"num_ctx": 8192, "num_predict": 3072}


# --------------------------------------------------------------------------- #
# 6. Empreinte canonique
# --------------------------------------------------------------------------- #
def test_schema_fingerprint_is_canonical_sha256():
    expected = hashlib.sha256(json.dumps(SIMPLE, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    assert schema_sha256(SIMPLE) == expected
    reordered = {"required": ["flag_a", "flag_b"], "properties": {"flag_b": {"type": "boolean"}, "flag_a": {"type": "boolean"}},
                 "additionalProperties": False, "type": "object"}
    assert schema_sha256(reordered) == expected                  # indépendant de l'ordre des clés
    assert canonical_schema_bytes({"é": 1}) == '{"é":1}'.encode("utf-8")   # UTF-8 brut, pas d'échappement
    assert schema_sha256({"type": "object", "properties": {"x": {"type": "string"}}}) != expected


def test_response_hash_matches_request_schema_exactly(clientapp, reg, monkeypatch):
    _install(monkeypatch, _upstream('{"selected_ids": ["C2"]}'))
    r = _post(clientapp, rf={"type": "json_schema", "schema": LIST_SCHEMA})
    assert r.json()["response_schema_sha256"] == hashlib.sha256(canonical_schema_bytes(LIST_SCHEMA)).hexdigest()


# --------------------------------------------------------------------------- #
# 7. Flux inchangé, confidentialité, contrat OpenAPI
# --------------------------------------------------------------------------- #
def test_stream_refuses_response_format_before_any_plan(clientapp, reg, monkeypatch):
    opened = {}
    monkeypatch.setattr(main.client, "open_generate_stream", lambda **kw: opened.update(kw))
    r = _post(clientapp, path="/v1/generate/stream")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "structured_output_streaming_not_supported"
    assert opened == {} and main.counters.snapshot()["generate_structured_streaming_rejected_total"] == 1
    r = _post(clientapp, path="/v1/generate/stream", model="inconnu:latest", rf={"type": "json_schema", "schema": {"$ref": "#"}})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "structured_output_streaming_not_supported"


def test_stream_legacy_unchanged(clientapp, reg, monkeypatch):
    from tests.test_generate_stream import _FakeStream, DONE_FIELDS
    captured = {}
    monkeypatch.setattr(main.client, "open_generate_stream",
                        lambda **kw: (captured.update(kw), _FakeStream([{"response": "OK"}, DONE_FIELDS]))[1])
    r = _post(clientapp, path="/v1/generate/stream", rf=None, format="json")
    assert r.status_code == 200
    events = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    assert [e["type"] for e in events] == ["delta", "done"] and captured["format"] == "json"
    assert "structured_output" not in events[-1] and "response_schema_sha256" not in events[-1]


def test_logs_never_contain_prompt_schema_or_output(clientapp, reg, monkeypatch, logs):
    schema = {"type": "object", "properties": {"ZZPROPZZ": {"type": "string", "enum": ["ZZENUMZZ"]}},
              "required": ["ZZPROPZZ"], "additionalProperties": False}
    _install(monkeypatch, _upstream('{"ZZPROPZZ": "ZZENUMZZ"}'))
    r = clientapp.post("/v1/generate", json={"prompt": "ZZPROMPTZZ", "model": "qwen-like", "system": "ZZSYSTEMZZ",
                                              "response_format": {"type": "json_schema", "schema": schema}})
    assert r.status_code == 200
    text = _all_logs(logs)
    for s in ("ZZPROPZZ", "ZZENUMZZ", "ZZPROMPTZZ", "ZZSYSTEMZZ"):
        assert s not in text, s
    ev = _event(logs, "generate.ok")
    assert set(ev) <= {"model", "upstream_ms", "total_ms", "ollama_total_ms", "ollama_load_ms", "ollama_prompt_eval_ms", "ollama_eval_ms",
                       "prompt_tokens", "completion_tokens", "prompt_tokens_per_second", "completion_tokens_per_second", "load_state",
                       "done_reason", "model_logical", "context_profile", "effective_num_ctx", "generation_profile", "effective_num_predict",
                       "keep_alive_profile", "structured_output", "schema_bytes", "schema_depth", "schema_nodes", "response_schema_sha256",
                       "structured_valid", "num_ctx", "requested_num_predict", "requested_think", "effective_think"}


def test_openapi_exposes_response_format_on_generate_only_as_optional(clientapp):
    spec = clientapp.get("/openapi.json").json()
    gen = spec["components"]["schemas"]["GenerateRequest"]
    assert "response_format" in gen["properties"] and "response_format" not in gen.get("required", [])
    assert "format" in gen["properties"]
    resp = spec["components"]["schemas"]["GenerateResponse"]
    assert "structured_output" in resp["properties"] and "response_schema_sha256" in resp["properties"]
