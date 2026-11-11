"""Lot P3 — gemma3:27b gouverné par le registre.

Jusqu'ici ce modèle arrivait par le chemin legacy : `resolved is None`, options
relayées brutes — un client contrôlait num_predict (dont -1 = illimité),
num_ctx (sans borne) et keep_alive (dont -1 = épinglage VRAM). Il est pourtant
le seul modèle installé qui ait été RÉELLEMENT appelé hors registre : 6 799
appels de production (2026-09-05 → 09-08).

Les profils sont dimensionnés sur cet usage observé, et les mécanismes sont
EXACTEMENT ceux des autres modèles gouvernés — aucune sémantique propre à
gemma, sauf le verrou `client_override: false` sur la rétention, générique et
opt-in (tous les profils existants gardent `true`).
"""

import logging

import pytest

from app import main
from app.metrics import counters

OK = {"response": "OK", "done": True, "done_reason": "stop",
      "prompt_eval_count": 1, "eval_count": 1}


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return dict(OK)

    monkeypatch.setattr(main.client, "generate", fake)
    return seen


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _fields(caplog, name="generate.ok"):
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith(f"event={name} "):
            return dict(p.split("=", 1) for p in m.split(" ")[1:] if "=" in p)
    raise AssertionError(f"événement {name!r} absent")


def _gen(clientapp, **body):
    return clientapp.post("/v1/generate", json={"prompt": "x", **body})


# --------------------------------------------------------------------------- #
# 1. Résolution et profils
# --------------------------------------------------------------------------- #
def test_gemma_resolves_to_governed_profile():
    for name in ("gemma3:27b", "gemma-large"):
        resolved = main.registry.resolve_llm(name)
        assert resolved is not None, name
        p = resolved.profile
        assert p.logical_name == "gemma-large" and p.ollama_name == "gemma3:27b"
        assert p.context_profile == "large-general"
        assert p.generation_profile == "gemma-bounded"
        assert p.keep_alive_profile == "transient-governed"
        assert p.resource_class == "large"
        assert not p.supports_image and p.vision_tasks == []
        assert p.allowed_in("production") and p.allowed_in("staging")


def test_gemma_profile_values_are_data_backed():
    reg = main.registry
    gen = reg.generation_profiles["gemma-bounded"]
    assert (gen.default_num_predict, gen.max_num_predict) == (2048, 4096)   # p95 259, max 1 200
    ctx = reg.context_profiles["large-general"]
    assert (ctx.num_ctx, ctx.max_num_ctx) == (8192, 16384)                  # demandé 8192 ×6 750
    ka = reg.keep_alive_profiles["transient-governed"]
    assert ka.keep_alive == "2m" and ka.client_override is False
    # Verrou opt-in : tous les autres profils gardent le comportement historique.
    assert all(p.client_override for n, p in reg.keep_alive_profiles.items()
               if n != "transient-governed")


# --------------------------------------------------------------------------- #
# 2. Options réellement transmises au mock Ollama
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["gemma3:27b", "gemma-large"])
def test_gemma_defaults_are_governed(clientapp, captured, logs, name):
    r = _gen(clientapp, model=name)
    assert r.status_code == 200
    assert captured["model"] == "gemma3:27b"
    assert captured["options"] == {"num_ctx": 8192, "num_predict": 2048}
    assert captured["keep_alive"] == "2m"
    assert captured["think"] is None
    f = _fields(logs)
    assert f["model_logical"] == "gemma-large"
    assert f["context_profile"] == "large-general" and f["effective_num_ctx"] == "8192"
    assert f["generation_profile"] == "gemma-bounded" and f["effective_num_predict"] == "2048"
    assert f["keep_alive_profile"] == "transient-governed"


def test_gemma_valid_overrides_pass_through(clientapp, captured):
    r = _gen(clientapp, model="gemma3:27b", options={"num_ctx": 4096, "num_predict": 512, "seed": 3})
    assert r.status_code == 200
    assert captured["options"] == {"num_ctx": 4096, "num_predict": 512, "seed": 3}


# --------------------------------------------------------------------------- #
# 3. TEST CRITIQUE — les paramètres dangereux du chemin legacy
# --------------------------------------------------------------------------- #
def test_gemma_legacy_dangerous_request_is_now_refused(clientapp, captured):
    """num_predict=-1, num_ctx excessif, keep_alive=-1 : avant P3, relayés
    bruts à Ollama. Désormais refusés, chacun par le mécanisme existant."""
    r = _gen(clientapp, model="gemma3:27b", keep_alive=-1,
             options={"num_predict": -1, "num_ctx": 1_000_000_000})
    assert r.status_code == 400
    assert captured == {}                              # Ollama jamais appelé


def test_gemma_num_predict_unbounded_refused(clientapp, captured):
    r = _gen(clientapp, model="gemma3:27b", options={"num_predict": -1})
    assert r.status_code == 400 and captured == {}
    r = _gen(clientapp, model="gemma3:27b", options={"num_predict": 4097})
    assert r.status_code == 400 and captured == {}
    assert r.json()["detail"]["max_num_predict"] == 4096


def test_gemma_num_ctx_excessive_refused(clientapp, captured):
    r = _gen(clientapp, model="gemma3:27b", options={"num_ctx": 1_000_000_000})
    assert r.status_code == 400 and captured == {}
    assert r.json()["detail"]["max_num_ctx"] == 16384
    r = _gen(clientapp, model="gemma3:27b", options={"num_ctx": 32768})   # ancien défaut runner
    assert r.status_code == 400 and captured == {}


@pytest.mark.parametrize("value", [-1, 0, "10m", 3600])
def test_gemma_keep_alive_client_override_refused(clientapp, captured, value):
    """Verrou de rétention : toute valeur client est refusée — jamais
    remplacée en silence par le profil."""
    r = _gen(clientapp, model="gemma3:27b", keep_alive=value)
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "keep_alive_not_overridable"
    assert d["keep_alive_profile"] == "transient-governed"
    assert captured == {}
    assert counters.snapshot()["generate_rejected_keep_alive_total"] == 1


def test_gemma_stream_shares_the_same_plan(clientapp, monkeypatch):
    from tests.test_generate_stream import DONE_FIELDS, _FakeStream, _install_stream
    captured = _install_stream(monkeypatch, _FakeStream([{"response": "a"}, DONE_FIELDS]))
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x", "model": "gemma3:27b"})
    assert r.status_code == 200
    assert captured["options"] == {"num_ctx": 8192, "num_predict": 2048}
    assert captured["keep_alive"] == "2m"
    r = clientapp.post("/v1/generate/stream", json={"prompt": "x", "model": "gemma3:27b", "keep_alive": -1})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# 4. Non-régression : les autres modèles gouvernés gardent le MÊME plan
# --------------------------------------------------------------------------- #
EXPECTED_PLANS = {
    # model demandé : (upstream, options, keep_alive, think, context, generation)
    "llama-chat":         ("llama3.1:8b",        {"num_ctx": 16384, "num_predict": 3072}, "30m", None,  "long-context",  "chat-bounded"),
    "llama3.1:8b":        ("llama3.1:8b",        {"num_ctx": 16384, "num_predict": 3072}, "30m", None,  "long-context",  "chat-bounded"),
    "qwen-judge":         ("qwen3:14b",          {"num_ctx": 8192,  "num_predict": 3072}, "30m", False, "verifier",      "judge-bounded"),
    "qwen3:14b":          ("qwen3:14b",          {"num_ctx": 8192,  "num_predict": 3072}, "30m", None,  "verifier",      "judge-bounded"),
    "qwen-large":         ("qwen3:30b-instruct", {"num_ctx": 8192,  "num_predict": 2048}, "2m",  None,  "large-general", "planner-bounded"),
    "qwen3:30b-instruct": ("qwen3:30b-instruct", {"num_ctx": 8192,  "num_predict": 2048}, "2m",  None,  "large-general", "planner-bounded"),
}


@pytest.mark.parametrize("name", sorted(EXPECTED_PLANS))
def test_other_governed_plans_unchanged(clientapp, captured, logs, name):
    upstream, options, keep_alive, think, context, generation = EXPECTED_PLANS[name]
    assert _gen(clientapp, model=name).status_code == 200
    assert captured["model"] == upstream
    assert captured["options"] == options
    assert captured["keep_alive"] == keep_alive
    assert captured["think"] == think
    f = _fields(logs)
    assert f["context_profile"] == context and f["generation_profile"] == generation


def test_other_governed_models_still_accept_client_keep_alive(clientapp, captured):
    """Le verrou est opt-in : qwen-large (transient, client_override=true)
    relaie toujours un keep_alive explicite, -1 compris — politique inchangée."""
    r = _gen(clientapp, model="qwen-large", keep_alive=-1)
    assert r.status_code == 200 and captured["keep_alive"] == -1
    r = _gen(clientapp, model="llama-chat", keep_alive=0)
    assert r.status_code == 200 and captured["keep_alive"] == 0


def test_registry_rejects_invalid_client_override(tmp_path):
    from app.registry import RegistryError
    text = open(main.registry.source_path, encoding="utf-8").read()
    broken = text.replace("    client_override: false\n", "    client_override: peut-etre\n")
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(RegistryError, match="client_override"):
        main.load_registry(str(path))
