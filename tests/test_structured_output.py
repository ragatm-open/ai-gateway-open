"""Lot C3 — sorties structurées gouvernées sur `/v1/chat`.

Trois couches : (1) `app/structured.py` pur — sous-ensemble JSON Schema,
bornes, analyse stricte, validation de l'instance ; (2) la route sur le
registre EMBARQUÉ (oss-reasoning en staging, modèles antérieurs sans la
capability) avec upstream doublé ; (3) confidentialité, non-régression C2,
flux et /v1/generate. Aucun appel Ollama sur un refus.
"""

import json
import logging

import pytest

from app import main
from app.structured import (
    ERR_INVALID_SCHEMA, ERR_OUTPUT_EMPTY, ERR_OUTPUT_INVALID, ERR_OUTPUT_MISMATCH, ERR_SCHEMA_TOO_LARGE,
    ERR_UNSUPPORTED_KEYWORD, SCHEMA_MAX_BYTES, SCHEMA_MAX_DEPTH, SCHEMA_MAX_ENUM_VALUES, SCHEMA_MAX_NODES,
    SCHEMA_MAX_PROPERTIES, SchemaError, StructuredOutputError, parse_strict_json, validate_client_schema,
    validate_structured_output,
)
from tests.chat_support import BASIC, STREAM_DONE, FakeChatStream, chat_event, install_chat, install_chat_stream, ndjson, upstream

LOGICAL = "oss-reasoning"
PHYSICAL = "gpt-oss:20b"
SIMPLE = {"type": "object", "properties": {"language": {"type": "string"}, "score": {"type": "integer"},
                                            "keywords": {"type": "array", "items": {"type": "string"}}},
          "required": ["language", "score", "keywords"], "additionalProperties": False}
COMPLEX = {"type": "object", "additionalProperties": False, "required": ["id", "client", "lines", "status"],
           "properties": {"id": {"type": "string", "minLength": 3, "maxLength": 32},
                          "client": {"type": "object", "required": ["name"], "additionalProperties": False,
                                     "properties": {"name": {"type": "string"}, "vip": {"type": "boolean"}}},
                          "lines": {"type": "array", "minItems": 1, "maxItems": 20,
                                    "items": {"type": "object", "required": ["sku", "qty"], "additionalProperties": False,
                                              "properties": {"sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 1, "maximum": 1000}}}},
                          "status": {"type": "string", "enum": ["pending", "paid", "shipped"]},
                          "note": {"type": ["string", "null"]},
                          "total": {"type": "number", "exclusiveMinimum": 0}}}
RF = {"type": "json_schema", "schema": SIMPLE}
GOOD = '{"language":"fr","score":95,"keywords":["IA"]}'


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.setattr(main.settings, "environment", "staging")


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _post(clientapp, model=LOGICAL, path="/v1/chat", rf=RF, **body):
    payload = {"model": model, "messages": BASIC, **body}
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
# 1. Sous-ensemble et bornes (pur)
# --------------------------------------------------------------------------- #
def test_valid_schemas_are_accepted_with_stats():
    stats = validate_client_schema(SIMPLE)
    assert (stats.depth, stats.nodes, stats.properties, stats.enum_values) == (3, 5, 3, 0)
    stats = validate_client_schema(COMPLEX)
    assert stats.depth == 4 and stats.enum_values == 3 and stats.properties == 10
    validate_client_schema({"type": "object", "properties": {"k": {"type": "string", "enum": ["a", "b"]}}})
    validate_client_schema({"type": "object", "properties": {"opt": {"type": ["integer", "null"]}}, "required": []})
    validate_client_schema({"type": "object", "properties": {"m": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}})
    validate_client_schema({"type": "object", "properties": {"c": {"type": "string", "const": "x", "title": "t", "description": "d"}}})


@pytest.mark.parametrize("schema,code", [
    ("pas un objet", ERR_INVALID_SCHEMA),
    ({}, ERR_INVALID_SCHEMA),                                            # type obligatoire
    ({"type": "string"}, ERR_INVALID_SCHEMA),                            # racine non object
    ({"type": "object", "properties": {"a": {"type": "magic"}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": "a"}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "a"]}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "array", "items": "string"}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "array", "items": [{"type": "string"}]}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string", "enum": []}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string", "enum": [{"x": 1}]}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "integer", "minimum": "1"}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "integer", "minimum": 5, "maximum": 1}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string", "minLength": -1}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string", "minLength": True}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string", "items": {"type": "string"}}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string", "properties": {}}}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "additionalProperties": {"type": "string"}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"": {"type": "string"}}}, ERR_INVALID_SCHEMA),
    ({"type": ["object", "object"]}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {"a": {"type": "string", "const": [1]}}}, ERR_INVALID_SCHEMA),
])
def test_invalid_schemas_are_refused(schema, code):
    with pytest.raises(SchemaError) as info:
        validate_client_schema(schema)
    assert info.value.code == code


@pytest.mark.parametrize("keyword,value,shown", [
    ("$ref", "#/$defs/x", "$ref"), ("$dynamicRef", "#x", "$dynamicRef"), ("$recursiveRef", "#", "$recursiveRef"),
    ("$id", "https://example.org/s.json", "$id"), ("$schema", "https://json-schema.org/draft/2020-12/schema", "$schema"),
    ("$defs", {}, "$defs"), ("definitions", {}, "definitions"),
    ("pattern", "^a+$", "pattern"), ("patternProperties", {}, "patternProperties"), ("format", "email", "format"),
    ("oneOf", [], "oneOf"), ("anyOf", [], "anyOf"), ("allOf", [], "allOf"), ("not", {}, "not"), ("if", {}, "if"),
    ("uniqueItems", True, "uniqueItems"), ("prefixItems", [], "prefixItems"), ("multipleOf", 2, "multipleOf"),
    ("default", 1, "default"), ("x-vendor", 1, "unknown"), ("mystery", 1, "unknown"),
])
def test_unsupported_keywords_are_refused_and_never_resolved(keyword, value, shown):
    schema = {"type": "object", "properties": {"a": {"type": "string", keyword: value}}}
    with pytest.raises(SchemaError) as info:
        validate_client_schema(schema)
    assert info.value.code == ERR_UNSUPPORTED_KEYWORD
    assert info.value.extra["keyword"] == shown
    assert "example.org" not in info.value.message and "mystery" not in info.value.message


def test_ref_at_root_is_refused_too():
    with pytest.raises(SchemaError) as info:
        validate_client_schema({"type": "object", "$ref": "https://evil.example/schema.json"})
    assert info.value.code == ERR_UNSUPPORTED_KEYWORD and info.value.extra["keyword"] == "$ref"


def test_bounds_bytes():
    big = {"type": "object", "properties": {"a": {"type": "string", "description": "x" * 400}}}
    validate_client_schema(big)
    huge = {"type": "object", "properties": {f"k{i}": {"type": "string", "description": "y" * 500} for i in range(70)}}
    with pytest.raises(SchemaError) as info:
        validate_client_schema(huge)
    assert info.value.code == ERR_SCHEMA_TOO_LARGE and info.value.extra["limit"] == "bytes"
    assert info.value.extra["max"] == SCHEMA_MAX_BYTES


def test_bounds_depth():
    def nest(n):
        node = {"type": "string"}
        for _ in range(n):
            node = {"type": "object", "properties": {"c": node}}
        return node
    validate_client_schema(nest(SCHEMA_MAX_DEPTH - 1))
    with pytest.raises(SchemaError) as info:
        validate_client_schema(nest(SCHEMA_MAX_DEPTH))
    assert info.value.code == ERR_SCHEMA_TOO_LARGE and info.value.extra["limit"] == "depth"


def test_bounds_properties_and_nodes():
    ok = {"type": "object", "properties": {f"p{i}": {"type": "boolean"} for i in range(SCHEMA_MAX_PROPERTIES)}}
    validate_client_schema(ok)
    too_many = {"type": "object", "properties": {f"p{i}": {"type": "boolean"} for i in range(SCHEMA_MAX_PROPERTIES + 1)}}
    with pytest.raises(SchemaError) as info:
        validate_client_schema(too_many)
    assert info.value.extra["limit"] == "properties"
    nodes = {"type": "object", "properties": {f"p{i}": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}} for i in range(200)}}
    with pytest.raises(SchemaError) as info:
        validate_client_schema(nodes)
    assert info.value.extra["limit"] in ("nodes", "bytes")
    assert SCHEMA_MAX_NODES == 512


def test_bounds_enum():
    ok = {"type": "object", "properties": {"e": {"type": "integer", "enum": list(range(SCHEMA_MAX_ENUM_VALUES))}}}
    validate_client_schema(ok)
    with pytest.raises(SchemaError) as info:
        validate_client_schema({"type": "object", "properties": {"e": {"type": "integer", "enum": list(range(SCHEMA_MAX_ENUM_VALUES + 1))}}})
    assert info.value.extra["limit"] == "enum_values"
    with pytest.raises(SchemaError) as info:
        validate_client_schema({"type": "object", "properties": {"e": {"type": "string", "enum": ["x" * 257]}}})
    assert info.value.extra["limit"] == "enum_string_chars"
    with pytest.raises(SchemaError):
        validate_client_schema({"type": "object", "properties": {"k" * 129: {"type": "string"}}})


def test_schema_error_messages_never_echo_the_schema():
    sentinel = "ZZSCHEMASENTINELZZ"
    with pytest.raises(SchemaError) as info:
        validate_client_schema({"type": "object", "properties": {sentinel: {"type": "string", "pattern": sentinel}}})
    # Le message est fixe ; seul `extra.path` (rendu au CLIENT, propriétaire
    # du schéma, jamais journalisé) peut citer le nom de propriété fautif.
    assert sentinel not in info.value.message
    assert sentinel not in json.dumps({k: v for k, v in info.value.extra.items() if k != "path"})


# --------------------------------------------------------------------------- #
# 2. Analyse stricte et validation de l'instance (pur)
# --------------------------------------------------------------------------- #
def test_strict_parser_accepts_exact_document_with_surrounding_whitespace():
    assert parse_strict_json('  \n{"a": 1}\n ') == {"a": 1}
    assert validate_structured_output(GOOD, SIMPLE) == {"language": "fr", "score": 95, "keywords": ["IA"]}


@pytest.mark.parametrize("text", [
    '```json\n{"language":"fr","score":1,"keywords":[]}\n```',
    'Voici : {"language":"fr","score":1,"keywords":[]}',
    '{"language":"fr","score":1,"keywords":[]} merci',
    '{"language":"fr","score":1,"keywords":[]}{"language":"en","score":2,"keywords":[]}',
    '{"language":"fr","score":1,"keywords":[',
    '{"language":"fr","score":NaN,"keywords":[]}',
    '{"language":"fr","score":Infinity,"keywords":[]}',
    '{"language":"fr","score":-Infinity,"keywords":[]}',
    "{'language':'fr','score':1,'keywords':[]}",
    '{"language":"fr","score":1,"keywords":[],}',
    '{"language":"fr","score":1,"keywords":[]}\x00',
    '{"language":"fr\n","score":1,"keywords":[]}',
    '{"language":"fr","score":1,"score":2,"keywords":[]}',
])
def test_strict_parser_refuses_repairable_or_ambiguous_documents(text):
    with pytest.raises(StructuredOutputError) as info:
        validate_structured_output(text, SIMPLE)
    assert info.value.code == ERR_OUTPUT_INVALID


def test_duplicate_keys_are_refused_even_nested():
    with pytest.raises(ValueError):
        parse_strict_json('{"a": {"b": 1, "b": 2}}')
    with pytest.raises(ValueError):
        parse_strict_json('[{"k": 1, "k": 1}]')


@pytest.mark.parametrize("text", [
    '{"language":"fr","score":"95","keywords":["IA"]}',          # mauvais type
    '{"language":"fr","score":95}',                              # required absent
    '{"language":"fr","score":95,"keywords":["IA"],"x":1}',      # additionalProperties
    '{"language":"fr","score":95,"keywords":[1]}',               # items
    '{"language":"fr","score":9.5,"keywords":[]}',               # integer
    '[]', '"texte"', 'null', '42',
])
def test_schema_mismatch_is_refused(text):
    with pytest.raises(StructuredOutputError) as info:
        validate_structured_output(text, SIMPLE)
    assert info.value.code == ERR_OUTPUT_MISMATCH


def test_enum_and_nested_constraints_are_enforced():
    base = {"id": "CMD-1", "client": {"name": "S"}, "lines": [{"sku": "A", "qty": 1}], "status": "paid", "note": None, "total": 1.5}
    assert validate_structured_output(json.dumps(base), COMPLEX) == base
    for mutate in (lambda d: d.update(status="lost"), lambda d: d["lines"][0].update(qty=0), lambda d: d.update(total=0),
                   lambda d: d.update(id="ab"), lambda d: d["client"].update(age=3), lambda d: d.update(lines=[])):
        doc = json.loads(json.dumps(base)); mutate(doc)
        with pytest.raises(StructuredOutputError) as info:
            validate_structured_output(json.dumps(doc), COMPLEX)
        assert info.value.code == ERR_OUTPUT_MISMATCH


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_content_is_a_distinct_error(text):
    with pytest.raises(StructuredOutputError) as info:
        validate_structured_output(text, SIMPLE)
    assert info.value.code == ERR_OUTPUT_EMPTY


def test_output_error_messages_never_echo_content():
    sentinel = "ZZOUTPUTSENTINELZZ"
    with pytest.raises(StructuredOutputError) as info:
        validate_structured_output(f'{{"language":"{sentinel}","score":"x","keywords":[]}}', SIMPLE)
    assert sentinel not in info.value.message
    with pytest.raises(StructuredOutputError) as info:
        validate_structured_output(f"```{sentinel}```", SIMPLE)
    assert sentinel not in info.value.message


# --------------------------------------------------------------------------- #
# 3. Route : capability, environnement, validation avant Ollama, relais format
# --------------------------------------------------------------------------- #
STRUCTURED_COMPLETION_MODELS = {
    # Rail completion + response_format (G2, puis lot d'activation du 2026-09-26) :
    # chaque entrée a été qualifiée 15/15 canaries synthétiques en staging.
    "qwen-large": ["completion", "tools", "structured_output"],
    "qwen-judge": ["completion", "tools", "thinking", "structured_output"],
    "vision-large": ["completion", "vision", "tools", "structured_output"],
    "gemma-large": ["completion", "structured_output"],
}


def test_registry_capability_is_limited_to_qualified_models():
    """`structured_output` : oss-reasoning (rail chat, C3) et les quatre modèles
    completion qualifiés (G2 + activation) ; aucun autre — llama-chat (8/9 au
    diagnostic d'éligibilité) ne la porte PAS. Aucun modèle completion n'a `chat`."""
    caps = {n: p.capabilities for n, p in main.registry.llm_models.items()}
    assert caps[LOGICAL] == ["chat", "structured_output"]
    assert caps["qwen-reasoning"] == ["chat", "structured_output"]   # rail chat, qualifié 15/15 en staging le 2026-09-27
    for name, expected in STRUCTURED_COMPLETION_MODELS.items():
        assert caps[name] == expected, name
        assert "chat" not in caps[name], name
    assert caps["llama-chat"] == ["completion", "tools"]
    for name, c in caps.items():
        if name not in STRUCTURED_COMPLETION_MODELS and name not in (LOGICAL, "qwen-reasoning"):
            assert "structured_output" not in c, name


@pytest.mark.parametrize("name", [LOGICAL, PHYSICAL])
def test_structured_request_is_relayed_as_format_after_validation(clientapp, staging, monkeypatch, logs, name):
    captured = install_chat(monkeypatch, upstream(message={"role": "assistant", "content": GOOD, "thinking": "réflexion"}))
    r = _post(clientapp, model=name)
    assert r.status_code == 200, r.text
    assert captured["format"] == SIMPLE and captured["think"] == "low"
    body = r.json()
    assert body["structured_output"] == {"language": "fr", "score": 95, "keywords": ["IA"]}
    assert body["message"]["content"] == GOOD
    assert body["thinking"] is None and body["thinking_exposed"] is False and body["thinking_chars"] == 9
    assert body["reasoning"] == "low" and body["content_empty"] is False and body["done_reason"] == "stop"
    ev = _event(logs, "chat.ok")
    assert ev["structured_output"] == "True" and ev["structured_valid"] == "True"
    assert ev["schema_bytes"] and ev["schema_depth"] == "3" and ev["schema_nodes"] == "5"
    snap = main.counters.snapshot()
    assert snap["chat_structured_requests_total"] == 1 and snap["chat_structured_ok_total"] == 1


@pytest.mark.parametrize("name", ["llama-chat", "llama3.1:8b", "qwen-judge", "qwen3:14b", "qwen-large", "gemma-large", "vision-large"])
def test_models_without_capability_are_refused_before_ollama(clientapp, staging, monkeypatch, logs, name):
    captured = install_chat(monkeypatch)
    r = _post(clientapp, model=name)
    assert r.status_code == 400
    code = r.json()["detail"]["code"]
    # Modèles sans `chat` : refus chat d'abord ; le seul modèle chat sans
    # structured_output serait refusé `structured_output_not_supported`.
    assert code in ("chat_not_supported", "structured_output_not_supported")
    assert captured == {}


@pytest.mark.parametrize("name", ["chat-test", "chat-model:latest", "chat-model"])
def test_chat_model_without_structured_capability_gets_dedicated_code(clientapp, monkeypatch, tmp_path, logs, name):
    """Modèle `chat` SANS `structured_output`, demandé par son nom logique OU
    son nom physique : même refus dédié — aucun alias ne contourne la
    capability."""
    from tests.chat_support import install_registry
    install_registry(monkeypatch, tmp_path)          # chat-test : chat sans structured_output
    captured = install_chat(monkeypatch)
    r = _post(clientapp, model=name)
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "structured_output_not_supported" and d["capabilities"] == ["completion", "chat"]
    assert captured == {}
    assert main.counters.snapshot()["chat_structured_capability_rejected_total"] == 1
    assert _event(logs, "chat.structured_output_not_supported")["model_logical"] == "chat-test"


@pytest.mark.parametrize("name", [LOGICAL, PHYSICAL])
def test_production_serves_structured_gpt_oss(clientapp, monkeypatch, name):
    """Lot production-full-capability-enablement : la sortie structurée du chat
    est servie en production avec le même contrat qu'en staging."""
    monkeypatch.setattr(main.settings, "environment", "production")
    captured = install_chat(monkeypatch, upstream(message={"role": "assistant", "content": GOOD, "thinking": "réflexion"}))
    r = _post(clientapp, model=name)
    assert r.status_code == 200, r.text
    assert captured["format"] == SIMPLE and captured["think"] == "low"
    body = r.json()
    assert body["structured_output"] == {"language": "fr", "score": 95, "keywords": ["IA"]}
    assert body["thinking"] is None and body["thinking_exposed"] is False
    assert main.counters.snapshot()["chat_rejected_environment_total"] == 0


@pytest.mark.parametrize("strict", [False, True])
def test_strict_registry_flag_does_not_change_governance(clientapp, staging, monkeypatch, strict):
    monkeypatch.setattr(main.settings, "generation_strict_registry", strict)
    captured = install_chat(monkeypatch, upstream(message={"role": "assistant", "content": GOOD}))
    assert _post(clientapp, model=PHYSICAL).status_code == 200
    assert captured["format"] == SIMPLE and captured["options"]["num_predict"] == 4096


@pytest.mark.parametrize("schema,code", [
    ({"type": "object", "properties": {"a": {"$ref": "#/x"}}}, ERR_UNSUPPORTED_KEYWORD),
    ({"type": "object", "properties": {"a": {"type": "string", "pattern": "^x"}}}, ERR_UNSUPPORTED_KEYWORD),
    ({"type": "object", "properties": {"a": {"type": "wat"}}}, ERR_INVALID_SCHEMA),
    ({"type": "array", "items": {"type": "string"}}, ERR_INVALID_SCHEMA),
    ({"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(300)}}, ERR_SCHEMA_TOO_LARGE),
])
def test_invalid_client_schema_is_400_before_ollama(clientapp, staging, monkeypatch, logs, schema, code):
    captured = install_chat(monkeypatch)
    r = _post(clientapp, rf={"type": "json_schema", "schema": schema})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == code and r.json()["detail"]["model_logical"] == LOGICAL
    assert captured == {}
    assert main.counters.snapshot()["chat_structured_schema_rejected_total"] == 1
    assert _event(logs, "chat.schema_rejected")["code"] == code


@pytest.mark.parametrize("rf", [
    {"type": "json_object"}, {"type": "text"}, {"type": "json_schema"}, {"schema": SIMPLE},
    {"type": "json_schema", "schema": SIMPLE, "strict": True}, {"type": "json_schema", "schema": "x"}, "json_schema",
])
def test_response_format_shape_is_strict(clientapp, staging, monkeypatch, rf):
    captured = install_chat(monkeypatch)
    assert _post(clientapp, rf=rf).status_code == 400 and captured == {}


@pytest.mark.parametrize("extra", [{"format": SIMPLE}, {"format": "json"}, {"json_schema": SIMPLE}, {"tools": []}])
def test_raw_ollama_fields_stay_forbidden(clientapp, staging, monkeypatch, extra):
    captured = install_chat(monkeypatch)
    assert _post(clientapp, rf=None, **extra).status_code == 400 and captured == {}


# --------------------------------------------------------------------------- #
# 4. Réponse upstream : cas A à K
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("content,code,counter", [
    ('{"language":"fr","score":1,"keywords":[', ERR_OUTPUT_INVALID, "chat_structured_invalid_json_total"),           # B tronqué
    ('```json\n' + GOOD + '\n```', ERR_OUTPUT_INVALID, "chat_structured_invalid_json_total"),                       # C fence
    ("Voici le résultat : " + GOOD, ERR_OUTPUT_INVALID, "chat_structured_invalid_json_total"),                     # D texte + JSON
    (GOOD + "\nJ'espère que cela aide.", ERR_OUTPUT_INVALID, "chat_structured_invalid_json_total"),                 # E JSON + texte
    ('{"language":"fr","score":"95","keywords":["IA"]}', ERR_OUTPUT_MISMATCH, "chat_structured_schema_mismatch_total"),  # F type
    ('{"language":"fr","keywords":["IA"]}', ERR_OUTPUT_MISMATCH, "chat_structured_schema_mismatch_total"),          # G required
    ('{"language":"fr","score":95,"keywords":["IA"],"extra":1}', ERR_OUTPUT_MISMATCH, "chat_structured_schema_mismatch_total"),  # H additional
    ('{"language":"fr","score":NaN,"keywords":[]}', ERR_OUTPUT_INVALID, "chat_structured_invalid_json_total"),      # J NaN
    ('{"language":"fr","score":Infinity,"keywords":[]}', ERR_OUTPUT_INVALID, "chat_structured_invalid_json_total"),
    ('{"language":"fr","score":1,"score":2,"keywords":[]}', ERR_OUTPUT_INVALID, "chat_structured_invalid_json_total"),  # K doublon
    ("", ERR_OUTPUT_EMPTY, "chat_structured_empty_total"),
])
def test_upstream_violations_are_controlled_502(clientapp, staging, monkeypatch, logs, content, code, counter):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": content, "thinking": "ZZTHINKZZ"}))
    r = _post(clientapp)
    assert r.status_code == 502, r.text
    d = r.json()["detail"]
    assert d["code"] == code and d["provider"] == "ollama"
    assert content.strip()[:12] not in r.text if content.strip() else True
    assert "ZZTHINKZZ" not in r.text and "language" not in d["message"]
    snap = main.counters.snapshot()
    assert snap[counter] == 1 and snap["chat_structured_invalid_upstream_total"] == 1 and snap["chat_structured_ok_total"] == 0
    ev = _event(logs, "chat.structured_invalid")
    assert ev["code"] == code and "structured_valid" not in ev
    assert _event(logs, "chat.ok") is None
    assert main.gauges.snapshot() == {}


def test_enum_violation_is_mismatch(clientapp, staging, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": '{"id":"CMD-1","client":{"name":"S"},"lines":[{"sku":"A","qty":1}],"status":"lost"}'}))
    r = _post(clientapp, rf={"type": "json_schema", "schema": COMPLEX})
    assert r.status_code == 502 and r.json()["detail"]["code"] == ERR_OUTPUT_MISMATCH


def test_complex_valid_document_is_returned_parsed(clientapp, staging, monkeypatch):
    doc = {"id": "CMD-1", "client": {"name": "S", "vip": True}, "lines": [{"sku": "A", "qty": 2}], "status": "paid", "note": None, "total": 12.5}
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": json.dumps(doc)}))
    r = _post(clientapp, rf={"type": "json_schema", "schema": COMPLEX})
    assert r.status_code == 200 and r.json()["structured_output"] == doc


def test_length_with_complete_valid_document_is_accepted(clientapp, staging, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": GOOD}, done_reason="length"))
    r = _post(clientapp)
    assert r.status_code == 200
    assert r.json()["done_reason"] == "length" and r.json()["structured_output"]["score"] == 95


def test_length_with_truncated_document_is_502_with_done_reason(clientapp, staging, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": '{"language":"fr","sco', "thinking": "x" * 50}, done_reason="length"))
    r = _post(clientapp)
    assert r.status_code == 502
    assert r.json()["detail"] == {"status": "error", "provider": "ollama",
                                  "message": "sortie structurée : le contenu n'est pas un document JSON strict",
                                  "code": ERR_OUTPUT_INVALID, "done_reason": "length"}


def test_content_empty_differs_between_normal_and_structured_chat(clientapp, staging, monkeypatch):
    data = upstream(message={"role": "assistant", "content": "", "thinking": "x" * 300}, done_reason="length")
    install_chat(monkeypatch, data)
    normal = _post(clientapp, rf=None)
    assert normal.status_code == 200 and normal.json()["content_empty"] is True and normal.json()["done_reason"] == "length"
    assert "structured_output" not in normal.json()
    install_chat(monkeypatch, data)
    structured = _post(clientapp)
    assert structured.status_code == 502 and structured.json()["detail"]["code"] == ERR_OUTPUT_EMPTY
    assert structured.json()["detail"]["done_reason"] == "length"
    snap = main.counters.snapshot()
    assert snap["chat_empty_content_total"] == 1 and snap["chat_structured_empty_total"] == 1


def test_unexpected_tool_calls_still_refused_in_structured_mode(clientapp, staging, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": GOOD, "tool_calls": [{"function": {"name": "f"}}]}))
    r = _post(clientapp)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "unsupported_upstream_response"


# --------------------------------------------------------------------------- #
# 5. Reasoning + structured ; thinking séparé
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("requested,think", [(None, "low"), ("low", "low"), ("medium", "medium"), ("high", "high")])
def test_reasoning_levels_with_structured_output(clientapp, staging, monkeypatch, requested, think):
    captured = install_chat(monkeypatch, upstream(message={"role": "assistant", "content": GOOD, "thinking": "t"}))
    body = {} if requested is None else {"reasoning": requested}
    r = _post(clientapp, **body)
    assert r.status_code == 200 and captured["think"] == think and captured["format"] == SIMPLE
    assert r.json()["reasoning"] == think


@pytest.mark.parametrize("requested", ["none", "off", "on", "potato"])
def test_reasoning_refusals_with_structured_output(clientapp, staging, monkeypatch, requested):
    captured = install_chat(monkeypatch)
    assert _post(clientapp, reasoning=requested).status_code == 400 and captured == {}


def test_thinking_is_never_part_of_the_structured_document(clientapp, staging, monkeypatch, logs):
    thinking = 'ZZTHINKSENTINELZZ {"language":"xx","score":0,"keywords":[]}'
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": GOOD, "thinking": thinking}))
    body = _post(clientapp).json()
    assert body["structured_output"]["language"] == "fr" and body["thinking"] is None
    assert body["thinking_chars"] == len(thinking)
    assert "ZZTHINKSENTINELZZ" not in json.dumps(body) and "ZZTHINKSENTINELZZ" not in _all_logs(logs)


def test_think_is_never_forced_false_for_structured(clientapp, staging, monkeypatch):
    captured = install_chat(monkeypatch, upstream(message={"role": "assistant", "content": GOOD}))
    _post(clientapp)
    assert captured["think"] == "low" and captured["think"] is not False


# --------------------------------------------------------------------------- #
# 6. Confidentialité, non-régression C2, flux, generate
# --------------------------------------------------------------------------- #
def test_logs_contain_only_bounded_facts(clientapp, staging, monkeypatch, logs):
    schema = {"type": "object", "properties": {"ZZPROPSENTINELZZ": {"type": "string", "enum": ["ZZENUMSENTINELZZ"]}},
              "required": ["ZZPROPSENTINELZZ"], "additionalProperties": False}
    content = '{"ZZPROPSENTINELZZ":"ZZENUMSENTINELZZ"}'
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": content, "thinking": "ZZTHINKSENTINELZZ"}))
    r = clientapp.post("/v1/chat", json={"model": LOGICAL, "messages": [{"role": "user", "content": "ZZPROMPTSENTINELZZ"}],
                                          "response_format": {"type": "json_schema", "schema": schema}})
    assert r.status_code == 200
    text = _all_logs(logs)
    for s in ("ZZPROPSENTINELZZ", "ZZENUMSENTINELZZ", "ZZTHINKSENTINELZZ", "ZZPROMPTSENTINELZZ"):
        assert s not in text, s
    ev = _event(logs, "chat.ok")
    assert set(ev) <= {"completion_tokens", "completion_tokens_per_second", "content_chars", "content_empty", "context_profile",
                       "done_reason", "effective_num_ctx", "effective_num_predict", "effective_think", "generation_profile",
                       "keep_alive_profile", "load_state", "messages", "model", "model_logical", "ollama_eval_ms", "ollama_load_ms",
                       "ollama_prompt_eval_ms", "ollama_total_ms", "prompt_tokens", "prompt_tokens_per_second", "reasoning",
                       "reasoning_profile", "thinking_chars", "total_ms", "upstream_ms", "structured_output", "structured_valid",
                       "schema_bytes", "schema_depth", "schema_nodes", "num_ctx", "requested_num_predict"}


def test_normal_chat_response_is_byte_identical_to_c2_contract(clientapp, staging, monkeypatch):
    install_chat(monkeypatch, upstream(message={"role": "assistant", "content": "Bonjour.", "thinking": "t"}))
    body = _post(clientapp, rf=None).json()
    assert "structured_output" not in body
    assert list(body) == ["provider", "upstream_provider", "model", "message", "thinking", "thinking_exposed",
                          "thinking_chars", "reasoning", "done", "done_reason", "content_empty", "tokens", "usage", "cost"]
    assert body["message"] == {"role": "assistant", "content": "Bonjour."} and body["reasoning"] == "low"


def test_normal_chat_never_sends_format(clientapp, staging, monkeypatch):
    captured = install_chat(monkeypatch)
    _post(clientapp, rf=None)
    assert "format" not in captured
    _post(clientapp, rf=None, reasoning="high")
    assert "format" not in captured


@pytest.mark.parametrize("name", [LOGICAL, PHYSICAL])
def test_stream_refuses_structured_before_any_plan(clientapp, staging, monkeypatch, name):
    captured = install_chat_stream(monkeypatch, FakeChatStream([chat_event(content="x"), chat_event(**STREAM_DONE)]))
    r = _post(clientapp, model=name, path="/v1/chat/stream")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "structured_output_streaming_not_supported"
    assert captured == {}
    assert main.counters.snapshot()["chat_structured_streaming_rejected_total"] == 1
    # sans response_format : flux C2 inchangé
    r = _post(clientapp, model=name, path="/v1/chat/stream", rf=None)
    assert r.status_code == 200 and [e["type"] for e in ndjson(r)] == ["content", "done"]
    assert "format" not in captured


def test_stream_refusal_precedes_unknown_model_and_invalid_schema(clientapp, staging, monkeypatch):
    captured = install_chat_stream(monkeypatch, FakeChatStream([]))
    r = _post(clientapp, model="inconnu:latest", path="/v1/chat/stream", rf={"type": "json_schema", "schema": {"$ref": "#"}})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "structured_output_streaming_not_supported"
    assert captured == {}


@pytest.mark.parametrize("extra", [{"response_format": RF}, {"json_schema": SIMPLE}])
def test_generate_never_accepts_structured_fields(clientapp, staging, monkeypatch, extra):
    gen = {}
    monkeypatch.setattr(main.client, "generate", lambda **kw: (gen.update(kw), {"response": "x", "done": True})[1])
    for model in ("llama-chat", LOGICAL):
        r = clientapp.post("/v1/generate", json={"prompt": "x", "model": model, **extra})
        assert r.status_code == 400 and gen == {}


def test_open_chat_stream_transport_has_no_format_parameter():
    import inspect
    from app.ollama_client import OllamaClient
    assert "format" not in inspect.signature(OllamaClient.open_chat_stream).parameters
    assert "format" in inspect.signature(OllamaClient.chat).parameters


def test_transport_relays_only_dict_format(monkeypatch):
    from app.ollama_client import OllamaClient
    payload = OllamaClient._build_chat_payload(model="m", messages=[{"role": "user", "content": "u"}], stream=False, format=SIMPLE)
    assert payload["format"] == SIMPLE
    payload = OllamaClient._build_chat_payload(model="m", messages=[{"role": "user", "content": "u"}], stream=False)
    assert "format" not in payload
    with pytest.raises(ValueError):
        OllamaClient._build_chat_payload(model="m", messages=[{"role": "user", "content": "u"}], stream=False, format="json")
