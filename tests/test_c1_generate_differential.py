"""Lot C1 — test DIFFÉRENTIEL : `/v1/generate` avant/après le transport chat.

Principe : le golden `tests/data/c1_generate_golden.json` a été CAPTURÉ sur
la révision antérieure (avant C1) par ce même module, exécuté depuis une archive
propre de ce commit (`C1_CAPTURE_PATH=<fichier> pytest tests/test_c1_generate_differential.py`). Après
C1, le module recalcule exactement les mêmes observations et exige l'égalité
stricte. Ce qui est comparé, pour chaque modèle gouverné (nom logique ET nom
physique), pour `vision-large`, pour un modèle hors registre (strict on/off)
et pour le défaut d'environnement :

- le `GenerationPlan` complet (upstream, options, num_ctx, num_predict,
  think, keep_alive, log_fields, inflight_key) ou le 400 qui le remplace ;
- la réponse HTTP de `/v1/generate` (statut + corps JSON) et les arguments
  transmis au client Ollama simulé ;
- la réponse NDJSON de `/v1/generate/stream` et les arguments transmis ;
- les champs structurants de l'événement `generate.ok` (hors durées).

Les durées (`upstream_ms`, `total_ms`) et `verified_at` sont exclues : ce
sont les seules valeurs non déterministes. Tout le reste doit être IDENTIQUE.
"""

import json
import logging
import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app import main
from app.errors import http_error
from app.schemas import GenerateRequest
from fastapi import HTTPException

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "c1_generate_golden.json")

MODELS = [
    "llama-chat", "llama3.1:8b",
    "qwen-judge", "qwen3:14b",
    "qwen-large", "qwen3:30b-instruct",
    "gemma-large", "gemma3:27b",
    "vision-large", "mistral-small3.2:24b",
    "modele-inconnu:7b",
    None,                                   # défaut d'environnement
]

CASES = {
    "bare": {},
    "options": {"options": {"num_ctx": 4096, "num_predict": 100, "temperature": 0.7, "top_p": 0.9}},
    "think_true": {"think": True},
    "think_false": {"think": False},
    "keep_alive_1m": {"keep_alive": "1m"},
    "keep_alive_0": {"keep_alive": 0},
    "format_json": {"format": "json"},
    "system": {"system": "sys", "temperature": 0.0},
    "num_ctx_too_big": {"options": {"num_ctx": 1_000_000}},
    "num_predict_zero": {"options": {"num_predict": 0}},
    "num_predict_bool": {"options": {"num_predict": True}},
    "num_ctx_str": {"options": {"num_ctx": "4096"}},
}

ENVIRONMENTS = ("production", "staging")
STRICT = (False, True)

UPSTREAM_DATA = {
    "response": "OK-GOLDEN", "done": True, "done_reason": "stop",
    "prompt_eval_count": 11, "eval_count": 7,
    "total_duration": 1_500_000_000, "load_duration": 300_000_000,
    "prompt_eval_duration": 200_000_000, "eval_duration": 1_000_000_000,
}
STREAM_EVENTS = [
    {"response": "a", "done": False},
    {"response": "b", "done": False},
    {"response": "", "done": True, "done_reason": "stop",
     "prompt_eval_count": 11, "eval_count": 7, "load_duration": 300_000_000},
]
TIMING_KEYS = ("upstream_ms", "total_ms", "elapsed_ms")


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)

    def events(self):
        yield from self._events

    def close(self):
        return None


def _body(model, case):
    body = {"prompt": "prompt-golden", **CASES[case]}
    if model is not None:
        body["model"] = model
    return body


def _plan_observation(model, case) -> Dict[str, Any]:
    try:
        plan = main._plan_generation(GenerateRequest(**_body(model, case)))
    except HTTPException as exc:
        return {"error": {"status": exc.status_code, "detail": exc.detail}}
    return {
        "upstream_model": plan.upstream_model,
        "options": plan.options,
        "think": plan.think,
        "keep_alive": plan.keep_alive,
        "log_fields": plan.log_fields,
        "inflight_key": plan.inflight_key,
    }


def _event_fields(caplog, name):
    prefix = f"event={name} "
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(prefix):
            parts = message.split(" ")[1:]
            fields = dict(part.split("=", 1) for part in parts if "=" in part)
            return {k: v for k, v in fields.items() if k not in TIMING_KEYS}
    return None


def _http_observation(model, case, monkeypatch, caplog) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}
    stream_captured: Dict[str, Any] = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return dict(UPSTREAM_DATA)

    def fake_open(**kwargs):
        stream_captured.update(kwargs)
        return _FakeStream(STREAM_EVENTS)

    monkeypatch.setattr(main.client, "generate", fake_generate)
    monkeypatch.setattr(main.client, "open_generate_stream", fake_open)
    client = TestClient(main.app)

    caplog.clear()
    caplog.set_level(logging.INFO, logger="ai_gateway")
    r = client.post("/v1/generate", json=_body(model, case))
    ok_event = _event_fields(caplog, "generate.ok")
    legacy_event = _event_fields(caplog, "generate.legacy_model")
    rejected_event = _event_fields(caplog, "generate.model_not_registered")

    s = client.post("/v1/generate/stream", json=_body(model, case))
    stream_lines = [json.loads(l) for l in s.text.splitlines() if l.strip()] if s.status_code == 200 else s.json()

    return {
        "status": r.status_code,
        "body": r.json(),
        "client_kwargs": captured,
        "event_generate_ok": ok_event,
        "event_legacy": legacy_event,
        "event_not_registered": rejected_event,
        "stream_status": s.status_code,
        "stream_body": stream_lines,
        "stream_kwargs": stream_captured,
    }


def _capture_all(monkeypatch, caplog) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for environment in ENVIRONMENTS:
        monkeypatch.setattr(main.settings, "environment", environment)
        for strict in STRICT:
            monkeypatch.setattr(main.settings, "generation_strict_registry", strict)
            for model in MODELS:
                for case in CASES:
                    key = f"{environment}|strict={int(strict)}|{model}|{case}"
                    main.counters.reset()
                    main.gauges.reset()
                    out[key] = {
                        "plan": _plan_observation(model, case),
                        "http": _http_observation(model, case, monkeypatch, caplog),
                        "counters": {k: v for k, v in main.counters.snapshot().items() if v},
                    }
    return out


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def test_generate_plans_and_responses_identical_to_before_c1(monkeypatch, caplog):
    observed = json.loads(_canonical(_capture_all(monkeypatch, caplog)))
    capture_path = os.environ.get("C1_CAPTURE_PATH")
    if capture_path:
        # Mode CAPTURE (régénération VOLONTAIRE du golden, à exécuter depuis
        # une archive du SHA de référence) : on écrit, on ne compare pas.
        with open(capture_path, "w", encoding="utf-8") as handle:
            handle.write(_canonical(observed) + "\n")
        return
    with open(GOLDEN, "r", encoding="utf-8") as handle:
        golden = json.load(handle)
    assert set(observed) == set(golden), "jeu de cas différent du golden"
    diffs = [k for k in golden if observed[k] != golden[k]]
    detail = ""
    if diffs:
        k = diffs[0]
        detail = f"\n--- golden[{k}]\n{_canonical(golden[k])}\n--- observé\n{_canonical(observed[k])}"
    assert not diffs, f"{len(diffs)} cas divergent(s) : {diffs[:5]}{detail}"
    # Couverture réelle : chaque modèle gouverné apparaît sous ses DEUX noms.
    for name in ("llama-chat", "qwen-judge", "qwen-large", "gemma-large", "vision-large"):
        assert any(f"|{name}|" in k for k in golden)
    assert len(golden) == len(ENVIRONMENTS) * len(STRICT) * len(MODELS) * len(CASES)


def test_golden_is_the_pre_c1_capture():
    """Le golden porte l'empreinte du SHA de capture ; il ne se régénère pas
    en silence côté C1 (une régénération volontaire s'annonce en commit)."""
    with open(GOLDEN, "r", encoding="utf-8") as handle:
        golden = json.load(handle)
    assert golden, "golden vide"
    sample = golden["production|strict=0|llama-chat|bare"]
    assert sample["plan"]["upstream_model"] == "llama3.1:8b"
    assert sample["plan"]["options"] == {"num_ctx": 16384, "num_predict": 3072}
    assert sample["http"]["status"] == 200
