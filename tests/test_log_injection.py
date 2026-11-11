"""Lot P13 — pas de forgeage de lignes de log.

Démontré : `model = "llama3.1:8b\\nevent=generate.ok model=FORGED"` produisait
une seconde ligne `event=…` parfaitement crédible pour `grep`. Deux verrous :

1. le champ `model` de toute requête est validé (alphabet fermé, 400 avant
   tout backend) ;
2. `log_event()` tient sur UNE ligne quelle que soit la valeur — les tokens
   simples gardent leur rendu historique, le reste est cité via `repr`.
"""

import logging

import pytest
from pydantic import ValidationError

from app import main
from app.metrics import log_event
from app.schemas import MODEL_NAME_RE, validate_model_name

# Tous les noms réellement en usage (registre, alias Ollama, docs, tests).
VALID_NAMES = [
    "qwen3:30b-instruct", "qwen3:14b", "llama3.1:8b", "mistral-small3.2:24b",
    "gemma3:27b", "nomic-embed-text", "nomic-embed-text:latest",
    "granite-embedding:278m", "bge-m3", "BAAI/bge-reranker-v2-m3",
    "foo/bar:latest", "legacy-768", "fr-en-retrieval", "qwen-judge",
    "qwen-large", "llama-chat", "vision-large", "rerank-default", "x:latest",
]
CONTROL_NAMES = ["model\nFORGED", "model\rFORGED", "model\tFORGED",
                 "model\x00FORGED", "model\x1bFORGED", "llama3.1:8b\nevent=generate.ok model=FORGED"]


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _all_logs(caplog):
    return "\n".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# 1. Validator
# --------------------------------------------------------------------------- #
def test_every_registry_name_matches_the_alphabet():
    reg = main.registry
    names = [p.ollama_name for p in reg.models.values()]
    names += [p.ollama_name for p in reg.llm_models.values()]
    names += list(reg.models) + list(reg.llm_models) + list(reg.reranker_models)
    names += [p.artifact for p in reg.reranker_models.values()]
    for name in names:
        assert MODEL_NAME_RE.match(name), name
        assert validate_model_name(name) == name


@pytest.mark.parametrize("name", VALID_NAMES)
def test_valid_names_accepted(name):
    assert validate_model_name(name) == name


@pytest.mark.parametrize("name", CONTROL_NAMES + ["with space", " lead", "trail ", "a=b c"])
def test_control_chars_and_spaces_rejected(name):
    with pytest.raises(ValueError) as info:
        validate_model_name(name)
    # La chaîne fautive n'est pas reproduite dans l'erreur.
    assert "FORGED" not in str(info.value) and "space" not in str(info.value)


def test_absent_and_empty_keep_historical_default():
    assert validate_model_name(None) is None
    assert validate_model_name("") == ""


def test_too_long_rejected():
    with pytest.raises(ValueError):
        validate_model_name("a" * 257)
    assert validate_model_name("a" * 256)


# --------------------------------------------------------------------------- #
# 2. Routes — 400 avant tout backend, aucune ligne forgée
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", CONTROL_NAMES)
def test_generate_rejects_control_chars_before_backend(clientapp, monkeypatch, logs, name):
    calls = []
    monkeypatch.setattr(main.client, "generate", lambda **kw: calls.append(kw))
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": name})
    assert r.status_code == 400
    assert calls == []
    blob = _all_logs(logs)
    assert "FORGED" not in blob
    assert blob.count("event=generate.ok") == 0


@pytest.mark.parametrize("name", CONTROL_NAMES)
def test_embeddings_rejects_control_chars_before_backend(clientapp, monkeypatch, logs, name):
    calls = []
    monkeypatch.setattr(main.client, "embed", lambda **kw: calls.append(kw))
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": name})
    assert r.status_code == 400 and calls == []
    assert "FORGED" not in _all_logs(logs)


def test_rerank_rejects_control_chars_before_backend(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    calls = []
    monkeypatch.setattr(main.rerank_client, "rerank", lambda q, t: calls.append(q))
    r = clientapp.post("/v1/rerank", json={"query": "q", "model": "rerank\nFORGED",
                                            "documents": [{"id": "d", "text": "t"}]})
    assert r.status_code == 400 and calls == []
    assert "FORGED" not in _all_logs(logs)


def test_vision_request_schema_rejects_control_chars():
    from app.vision_schemas import VisionAnalyzeRequest
    with pytest.raises(ValidationError):
        VisionAnalyzeRequest(schema_version="1", task="describe",
                             image={"data": "aGVsbG8=", "mime_type": "image/png"},
                             model="vision\nFORGED")


def test_valid_model_still_reaches_backend(clientapp, monkeypatch):
    seen = {}
    monkeypatch.setattr(main.client, "generate",
                        lambda **kw: (seen.update(kw), {"response": "ok", "done": True})[1])
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen3:30b-instruct"})
    assert r.status_code == 200 and seen["model"] == "qwen3:30b-instruct"


def test_space_in_model_is_rejected_by_contract(clientapp, monkeypatch):
    """Aucun nom actuel ne contient d'espace : le contrat retenu les refuse."""
    calls = []
    monkeypatch.setattr(main.client, "generate", lambda **kw: calls.append(kw))
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen 3"})
    assert r.status_code == 400 and calls == []


# --------------------------------------------------------------------------- #
# 3. log_event — une invocation, une ligne
# --------------------------------------------------------------------------- #
def test_log_event_simple_values_keep_historical_rendering(logs):
    logger = logging.getLogger("ai_gateway")
    log_event(logger, logging.INFO, "generate.ok", total_ms=12, model="llama3.1:8b",
              done_reason="stop", ratio=1.5, flag=True)
    assert logs.records[-1].getMessage() == \
        "event=generate.ok done_reason=stop flag=True model=llama3.1:8b ratio=1.5 total_ms=12"


@pytest.mark.parametrize("value", [
    "a\nevent=generate.ok model=FORGED", "a\rb", "a\tb", "a b", "", "x\x00y", "é\nz",
])
def test_log_event_exotic_value_is_one_line(logs, value):
    logger = logging.getLogger("ai_gateway")
    log_event(logger, logging.INFO, "probe", model=value, n=1)
    message = logs.records[-1].getMessage()
    # UNE ligne : aucun saut de ligne réel ; la valeur exotique est citée et
    # ses caractères de contrôle échappés (`\\n` littéral), donc une fausse
    # séquence `event=…` reste VISIBLEMENT à l'intérieur des quotes.
    assert "\n" not in message and "\r" not in message and "\x00" not in message
    assert message.startswith("event=probe model=")
    assert message == f"event=probe model={value!r} n=1"


def test_log_event_accented_token_untouched(logs):
    logger = logging.getLogger("ai_gateway")
    log_event(logger, logging.INFO, "probe", reason="absent_du_registre", nom="élément")
    assert logs.records[-1].getMessage() == "event=probe nom=élément reason=absent_du_registre"


def test_existing_grep_recipes_still_match(logs, clientapp, monkeypatch):
    monkeypatch.setattr(main.client, "generate",
                        lambda **kw: {"response": "ok", "done": True, "done_reason": "stop",
                                      "prompt_eval_count": 1, "eval_count": 1})
    clientapp.post("/v1/generate", json={"prompt": "x", "model": "qwen-large"})
    line = [r.getMessage() for r in logs.records if r.getMessage().startswith("event=generate.ok ")][0]
    fields = dict(p.split("=", 1) for p in line.split(" ")[1:] if "=" in p)
    assert fields["done_reason"] == "stop" and fields["model"] == "qwen3:30b-instruct"
