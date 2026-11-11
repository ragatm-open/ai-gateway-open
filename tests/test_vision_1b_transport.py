"""Lot AI-GATEWAY-VISION-1B — transport multimodal Ollama réel (mocké).

Aucune inférence réelle : `requests.post` est intercepté DANS le backend, si
bien que la construction du payload, la sérialisation de l'image et le schéma
model-facing sont réellement exercés. Un garde échoue la suite si un appel
réseau non intercepté était tenté.
"""

import base64
import json

import pytest

from app import main, vision_backend as vb
from app.vision_backend import (
    QUALIFIED_FIRST_ATTEMPT,
    QUALIFIED_RETRY_ATTEMPT,
    OllamaVisionBackend,
)
from app.vision_prompts import (
    DESCRIBE_PROMPT,
    DESCRIBE_PROMPT_SHA256,
    QUALIFIED_PROMPTS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    sha256_of,
)
from tests.test_vision_media import _b64, _png, _post, backend, vision_ready  # noqa: F401

PHOTO_PAYLOAD = {
    "description": "Deux formes posées sur un fond clair.",
    "setting": None,
    "objects": [{"label": "carré bleu", "count": 2, "attributes": [],
                 "certainty": "exact"}],
    "observable_relations": [],
    "visible_text": [{"text": "ZONE B", "certainty": "exact"}],
}


def _ollama_body(payload=None, kind="photo", done_reason="stop",
                 prompt_tokens=780, completion_tokens=138, raw=None):
    body = raw if raw is not None else json.dumps(
        {"visual_kind": kind, "summary": "s", "warnings": [], "payload": payload})
    return {"response": body, "done": True, "done_reason": done_reason,
            "prompt_eval_count": prompt_tokens, "eval_count": completion_tokens,
            "total_duration": 3_000_000_000}


class FakeHTTPResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._body


@pytest.fixture
def ollama(monkeypatch, vision_ready):
    """Mock HTTP du endpoint /api/generate : enregistre chaque requête."""
    calls = []
    queue = [_ollama_body(PHOTO_PAYLOAD)]

    def fake_post(url, json=None, timeout=None, **kw):
        calls.append({"url": url, "payload": json, "timeout": timeout})
        item = queue.pop(0) if queue else _ollama_body(PHOTO_PAYLOAD)
        if isinstance(item, Exception):
            raise item
        return FakeHTTPResponse(item)

    be = OllamaVisionBackend("http://127.0.0.1:11434")
    be._post = fake_post          # transport d'instance : aucun effet global
    monkeypatch.setattr(main, "vision_backend", be)
    # Le profil de prompt qualifié réel doit servir (la fixture générique en
    # installe un factice pour les autres tasks).
    monkeypatch.setattr(main, "QUALIFIED_PROMPTS", dict(QUALIFIED_PROMPTS))

    class Ctl:
        def __init__(self): self.calls = calls
        def responses(self, *items): queue.clear(); queue.extend(items)
    return Ctl()


# ------------------------------------------------- forme exacte de la requête
def test_request_shape_is_exact(clientapp, ollama):
    r = _post(clientapp, task="describe", request_id="rid-1")
    assert r.status_code == 200
    assert len(ollama.calls) == 1
    call = ollama.calls[0]
    assert call["url"].endswith("/api/generate")
    p = call["payload"]
    assert p["model"] == "mistral-small3.2:24b"
    assert p["stream"] is False
    assert len(p["images"]) == 1                      # exactement une image
    assert p["keep_alive"] == 0          # défaut du dépôt (cf. test_vision_residency)
    assert p["system"] == SYSTEM_PROMPT
    assert p["prompt"].startswith(DESCRIBE_PROMPT)
    assert p["options"] == {"temperature": 0.0, "seed": 42,
                            "num_ctx": 8192, "num_predict": 2048}
    # schéma model-facing : exactement celui du générateur commité
    assert p["format"] == main.output_schema_for_task("describe")
    # l'image transmise est bien le PNG normalisé par l'admission
    assert base64.b64decode(p["images"][0])[:4] == b"\x89PNG"


def test_no_caller_controlled_inference_options(clientapp, ollama):
    """Aucune option d'inférence ne peut venir du client (schéma strict)."""
    r = clientapp.post("/v1/vision/analyze", json={
        "schema_version": "1", "task": "describe",
        "image": {"data": _b64(_png()), "mime_type": "image/png"},
        "options": {"temperature": 0.9}})
    assert r.status_code == 400
    assert ollama.calls == []


def test_context_is_appended_as_non_evidence_not_in_system(clientapp, ollama):
    _post(clientapp, task="describe",
          context={"caption": "CAPSENTINEL", "alt_text": None,
                   "nearby_text": None})
    p = ollama.calls[0]["payload"]
    assert "CAPSENTINEL" in p["prompt"]
    assert "CAPSENTINEL" not in p["system"]           # jamais dans les règles
    assert "jamais une preuve" in p["prompt"].lower() or \
           "JAMAIS une preuve" in p["prompt"]


# ------------------------------------------------------------------- retry
def test_valid_first_response_means_single_call(clientapp, ollama):
    assert _post(clientapp, task="describe").status_code == 200
    assert len(ollama.calls) == 1


def test_structural_retry_varies_temperature_not_only_seed(clientapp, ollama):
    ollama.responses(_ollama_body(raw="{pas du json"),
                     _ollama_body(PHOTO_PAYLOAD))
    r = _post(clientapp, task="describe")
    assert r.status_code == 200
    assert len(ollama.calls) == 2
    first, second = (c["payload"] for c in ollama.calls)
    assert first["options"]["temperature"] == 0.0 and first["options"]["seed"] == 42
    assert second["options"]["temperature"] == 0.35 and second["options"]["seed"] == 43
    assert second["options"]["temperature"] != first["options"]["temperature"]
    # tout le reste est identique : même modèle, même image, même schéma
    for key in ("model", "images", "format", "system", "prompt", "keep_alive"):
        assert first[key] == second[key], key


def test_semantically_odd_but_valid_is_not_retried(clientapp, ollama):
    odd = dict(PHOTO_PAYLOAD, description="Description inhabituelle mais valide.")
    ollama.responses(_ollama_body(odd))
    assert _post(clientapp, task="describe").status_code == 200
    assert len(ollama.calls) == 1


@pytest.mark.parametrize("bad,code", [
    ("{malformed", "vision_invalid_output"),
    (json.dumps({"visual_kind": "photo", "warnings": [],
                 "payload": {"description": "d", "objects": [
                     {"label": "o", "count": 1, "attributes": [],
                      "certainty": "high"}],          # enum hors V1
                     "observable_relations": [], "visible_text": [],
                     "setting": None}}), "vision_schema_validation_failed"),
])
def test_two_invalid_outputs_give_502(clientapp, ollama, bad, code):
    ollama.responses(_ollama_body(raw=bad), _ollama_body(raw=bad))
    r = _post(clientapp, task="describe")
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["code"] == code and d["retryable"] is False
    assert len(ollama.calls) == 2                     # jamais trois


def test_truncated_twice_is_502(clientapp, ollama):
    t = _ollama_body(PHOTO_PAYLOAD, done_reason="length")
    ollama.responses(t, t)
    r = _post(clientapp, task="describe")
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "vision_truncated_output"
    assert len(ollama.calls) == 2


def test_transport_error_is_not_retried(clientapp, ollama):
    import requests
    ollama.responses(requests.ConnectionError("refus"))
    r = _post(clientapp, task="describe")
    d = r.json()["detail"]
    assert r.status_code == 502 and d["code"] == "vision_upstream_error"
    assert d["retryable"] is True
    assert len(ollama.calls) == 1                     # aucun retry transport


def test_upstream_error_never_leaks_body(clientapp, ollama):
    ollama.responses(FakeHTTPResponse({"error": "SECRETUPSTREAMBODY"}, status=500))
    r = _post(clientapp, task="describe")
    assert r.status_code == 502
    assert "SECRETUPSTREAMBODY" not in r.text


# ------------------------------------------------------- budget et capacité
def test_total_budget_is_not_reset_by_retry(clientapp, ollama, monkeypatch):
    """Le budget de 60 s est TOTAL : le timeout transmis au retry est le
    reliquat, jamais un budget neuf."""
    monkeypatch.setattr(main.settings, "vision_timeout_seconds", 30)
    ollama.responses(_ollama_body(raw="{"), _ollama_body(PHOTO_PAYLOAD))
    assert _post(clientapp, task="describe").status_code == 200
    t1, t2 = (c["timeout"] for c in ollama.calls)
    assert t1 <= 30 and t2 <= t1                       # décroissant


def test_deadline_exhausted_gives_504(clientapp, ollama, monkeypatch):
    """Horloge qui avance à chaque lecture : le budget s'épuise APRÈS la prise
    d'origine, comme le ferait une attente réelle."""
    import time as _t
    monkeypatch.setattr(main.settings, "vision_timeout_seconds", 1)
    clock = {"t": _t.time()}

    def ticking():
        clock["t"] += 10
        return clock["t"]

    monkeypatch.setattr(main.time, "time", ticking)
    r = _post(clientapp, task="describe")
    d = r.json()["detail"]
    assert r.status_code == 504 and d["code"] == "vision_timeout"
    assert d["retryable"] is True
    assert ollama.calls == []


# ---------------------------------------------------------- gouvernance
@pytest.mark.parametrize("task", ["auto", "table", "diagram", "chart",
                                  "technical_drawing"])
def test_unqualified_tasks_never_reach_ollama(clientapp, ollama, task):
    r = _post(clientapp, task=task)
    d = r.json()["detail"]
    assert r.status_code == 400
    assert d["code"] == "vision_task_not_qualified" and d["retryable"] is False
    assert ollama.calls == []                          # zéro appel backend


def test_describe_passes_governance(clientapp, ollama):
    assert _post(clientapp, task="describe").status_code == 200


def test_auto_does_not_fall_back_to_describe(clientapp, ollama):
    """`auto` échoue fermé : aucune réinterprétation en `describe`."""
    r = _post(clientapp, task="auto")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "vision_task_not_qualified"
    assert ollama.calls == []


def test_concrete_model_name_is_rejected(clientapp, ollama):
    r = _post(clientapp, task="describe", model="mistral-small3.2:24b")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "vision_model_not_allowed"
    assert ollama.calls == []


def test_gemma_is_not_governed():
    """gemma3:27b n'a qualifié aucune task : absent du registre gouverné."""
    names = {p.ollama_name for p in main.registry.vision_profiles()}
    assert names == {"mistral-small3.2:24b"}
    assert "gemma3:27b" not in names


@pytest.mark.parametrize("tags,expected", [
    ([], "vision_model_unavailable"),                                  # absent
    ([{"name": "mistral-small3.2:24b", "digest": "f" * 64}],
     "vision_model_unavailable"),                                      # digest ≠
])
def test_model_availability_blocks_inference(clientapp, ollama, monkeypatch,
                                             tags, expected):
    monkeypatch.setattr(main.probe_client, "list_models", lambda timeout=5: tags)
    r = _post(clientapp, task="describe")
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == expected
    assert ollama.calls == []


def test_feature_off_blocks_everything(clientapp, ollama, monkeypatch):
    monkeypatch.setattr(main.settings, "vision_enabled", False)
    r = _post(clientapp, task="describe")
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "vision_not_enabled"
    assert ollama.calls == []


# ------------------------------------------------- artefacts qualifiés
def test_qualified_prompt_hashes_are_frozen():
    """Verrou de qualification : éditer un prompt sans nouvelle campagne
    MODEL-1 doit casser la suite."""
    assert sha256_of(SYSTEM_PROMPT) == SYSTEM_PROMPT_SHA256
    assert sha256_of(DESCRIBE_PROMPT) == DESCRIBE_PROMPT_SHA256
    assert SYSTEM_PROMPT_SHA256 == \
        "048ef7108332ecc3d52981dc430978d2ff9c892125576497eebec74c14115a87"
    assert DESCRIBE_PROMPT_SHA256 == \
        "3ff92ee7123743f30a5eab621a0c071029aefc52704e0e07ba836766f2b54bb0"
    assert set(QUALIFIED_PROMPTS) == {"describe"}
    assert QUALIFIED_PROMPTS["describe"][0] == "vision_describe_v1"


def test_describe_schema_hash_is_frozen():
    import hashlib
    h = hashlib.sha256(json.dumps(main.output_schema_for_task("describe"),
                                  sort_keys=True).encode()).hexdigest()
    assert h == "bcea8521454921d9a3dbd273420c19c3eee4b5c6833b63a38cee7ef4eb1954cd"


def test_qualified_inference_profile_values():
    assert QUALIFIED_FIRST_ATTEMPT.temperature == 0.0
    assert QUALIFIED_FIRST_ATTEMPT.seed == 42
    assert QUALIFIED_RETRY_ATTEMPT.temperature == 0.35
    assert QUALIFIED_RETRY_ATTEMPT.seed == 43
    for o in (QUALIFIED_FIRST_ATTEMPT, QUALIFIED_RETRY_ATTEMPT):
        # le PROFIL reste à 0 : la résidence est appliquée par copie (residency-02)
        assert o.num_ctx == 8192 and o.num_predict == 2048 and o.keep_alive == 0


def test_qualified_limits():
    s = main.settings
    assert s.vision_max_image_bytes == 4 * 1024 * 1024
    assert s.vision_max_image_pixels == 2_000_000
    assert s.vision_max_context_field_chars == 1000
    assert s.vision_max_context_total_chars == 2000
    assert s.vision_timeout_seconds == 60
    assert s.vision_max_concurrency == 1
    assert s.vision_enabled is False              # défaut dépôt : OFF


# --------------------------------------------- identité serveur / usage
def test_server_identity_cannot_be_spoofed(clientapp, ollama):
    """Un modèle qui renverrait des champs d'enveloppe ne doit pas les imposer."""
    spoof = json.dumps({"visual_kind": "photo", "summary": "s", "warnings": [],
                        "payload": PHOTO_PAYLOAD,
                        "provider": "EVIL", "model": {"logical": "EVIL"},
                        "pipeline": {"pipeline_id": "EVIL"},
                        "request_id": "EVIL", "latency_ms": 999999})
    ollama.responses(_ollama_body(raw=spoof), _ollama_body(PHOTO_PAYLOAD))
    body = _post(clientapp, task="describe", request_id="rid-server").json()
    assert body["provider"] == "ai_gateway"
    assert body["model"]["logical"] == "vision-large"
    assert body["model"]["concrete"] == "mistral-small3.2:24b"
    assert body["model"]["digest"].startswith("5a408ab5")
    assert body["pipeline"]["pipeline_id"] == \
        "vision-large@5a408ab5/vision_describe_v1@v1"
    assert body["request_id"] == "rid-server"
    assert "EVIL" not in json.dumps(body)


def test_usage_mapping_and_null_semantics(clientapp, ollama):
    body = _post(clientapp, task="describe").json()
    assert body["usage"] == {"prompt_tokens": 780, "completion_tokens": 138,
                             "total_tokens": 918}
    ollama.responses(_ollama_body(PHOTO_PAYLOAD, prompt_tokens=None,
                                  completion_tokens=None))
    body = _post(clientapp, task="describe").json()
    assert body["usage"] == {"prompt_tokens": None, "completion_tokens": None,
                             "total_tokens": None}    # jamais 0 inventé


def test_image_diagnostics_come_from_admission(clientapp, ollama):
    body = _post(clientapp, task="describe", image_bytes=_png(120, 80)).json()
    assert body["image"] == {"width": 120, "height": 80, "pixels": 9600,
                             "bytes": len(_png(120, 80)), "format": "png"}


# ------------------------------------------------------------- privacy
def test_no_sensitive_content_in_logs(clientapp, ollama, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="ai_gateway")
    ollama.responses(_ollama_body(raw="{ZZRAWSECRETZZ"),
                     _ollama_body(dict(PHOTO_PAYLOAD,
                                       description="ZZPAYLOADSECRETZZ")))
    r = _post(clientapp, task="describe",
              context={"caption": "ZZCTXSECRETZZ", "alt_text": None,
                       "nearby_text": None})
    assert r.status_code == 200
    logs = "\n".join(rec.getMessage() for rec in caplog.records)
    for sentinel in ("ZZRAWSECRETZZ", "ZZPAYLOADSECRETZZ", "ZZCTXSECRETZZ",
                     "iVBORw0", DESCRIBE_PROMPT[:40], SYSTEM_PROMPT[:40]):
        assert sentinel not in logs, sentinel


# ------------------------------------------------------------ capacité
def test_overload_is_503_without_backend_call(clientapp, ollama, monkeypatch):
    from app.vision_admission import VisionAdmission
    held = VisionAdmission(1); assert held.try_acquire()     # slot occupé (synthétique)
    monkeypatch.setattr(main, "vision_admission", held)
    monkeypatch.setattr(main.settings, "vision_timeout_seconds", 5)
    r = _post(clientapp, task="describe", request_id="rid-ov")
    d = r.json()["detail"]
    assert r.status_code == 503 and d["code"] == "vision_overloaded"
    assert d["retryable"] is True and d["request_id"] == "rid-ov"
    assert ollama.calls == []                      # jamais 429, jamais d'appel


# ------------------------------------------- limites QUALIFIÉES (1B)
@pytest.mark.parametrize("size,status", [
    (4 * 1024 * 1024, 200),          # exactement la borne : accepté
    (4 * 1024 * 1024 + 1, 413),      # borne + 1 : refusé
])
def test_byte_limit_boundaries(clientapp, ollama, monkeypatch, size, status):
    from app import vision_image
    calls = {"n": 0}
    real = vision_image.admit_image

    def sized(raw, mime):
        calls["n"] += 1
        return real(_png(), mime)     # décodage réel sur une image valide

    if status == 200:
        monkeypatch.setattr(main, "admit_image", sized)
    payload = b"\x00" * size
    r = clientapp.post("/v1/vision/analyze", json={
        "schema_version": "1", "task": "describe",
        "image": {"data": _b64(payload), "mime_type": "image/png"}})
    assert r.status_code == status
    if status == 413:
        d = r.json()["detail"]
        assert d["code"] == "image_too_large" and d["retryable"] is False
        assert ollama.calls == []


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_pixel_limit_boundary(clientapp, ollama):
    # 2 000 001 pixels > borne qualifiée de 2 MP
    r = _post(clientapp, task="describe", image_bytes=_png(2000, 1001))
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "image_too_large"
    assert ollama.calls == []
    # et 2 MP pile passe
    assert _post(clientapp, task="describe",
                 image_bytes=_png(2000, 1000)).status_code == 200


@pytest.mark.parametrize("ctx,status", [
    ({"caption": "x" * 1000, "alt_text": None, "nearby_text": None}, 200),
    ({"caption": "x" * 1001, "alt_text": None, "nearby_text": None}, 400),
    ({"caption": "x" * 1000, "alt_text": "y" * 1000, "nearby_text": None}, 200),
    ({"caption": "x" * 1000, "alt_text": "y" * 1000, "nearby_text": "z"}, 400),
])
def test_context_limit_boundaries(clientapp, ollama, ctx, status):
    r = _post(clientapp, task="describe", context=ctx)
    assert r.status_code == status
    if status == 400:
        assert "x" * 50 not in r.text            # aucun écho du contexte
        assert ollama.calls == []


# ------------------------------------------------------------- santé
def test_health_vision_reports_qualified_task_and_loads_nothing(clientapp, ollama,
                                                                monkeypatch):
    # registre RÉEL (la fixture générique simule les 6 tasks pour les tests de
    # contrat ; ici on vérifie ce que le déploiement expose vraiment)
    profile = main.registry.resolve_vision("vision-large")
    monkeypatch.setattr(profile, "vision_tasks", ["describe"])
    monkeypatch.setattr(main.probe_client, "list_running_models",
                        lambda timeout=5: [])
    body = clientapp.get("/health/vision").json()
    entry = body["models"][0]
    assert entry["tasks"] == ["describe"]         # jamais les non qualifiées
    assert entry["digest_match"] is True
    assert ollama.calls == []                     # aucune inférence


def test_models_endpoint_reports_qualified_metadata(clientapp, ollama, monkeypatch):
    profile = main.registry.resolve_vision("vision-large")
    monkeypatch.setattr(profile, "vision_tasks", ["describe"])
    monkeypatch.setattr(main.client, "list_models", lambda timeout=5: [
        {"name": "mistral-small3.2:24b", "size": 1, "modified_at": "x"}])
    vis = clientapp.get("/v1/models").json()["models"][0]
    assert vis["vision"]["tasks"] == ["describe"]
    assert vis["vision"]["max_image_pixels"] == 2_000_000
    assert ollama.calls == []


def test_health_ready_unaffected_by_vision(clientapp, ollama):
    assert clientapp.get("/health/ready").status_code == 200
    assert ollama.calls == []


# ------------------------------------------- garde anti-inférence réelle
def test_no_real_ollama_call_escapes_the_mock(clientapp, monkeypatch, vision_ready):
    """Garde §70 : si le transport n'était pas intercepté, la suite échoue."""
    from app.vision_backend import OllamaVisionBackend, VisionInferencePlan

    def forbidden(*a, **kw):
        raise AssertionError("appel réseau réel tenté vers Ollama")

    be = OllamaVisionBackend("http://127.0.0.1:11434")
    be._post = forbidden
    plan = VisionInferencePlan(
        image_bytes=_png(), image_mime="image/png", task="describe",
        prompt_profile="vision_describe_v1", prompt_profile_version="1",
        output_json_schema=main.output_schema_for_task("describe"),
        locale="fr", caption=None, alt_text=None, nearby_text=None,
        model_logical="vision-large", model_concrete="mistral-small3.2:24b",
        remaining_timeout_seconds=5, system_prompt=SYSTEM_PROMPT,
        task_prompt=DESCRIBE_PROMPT)
    with pytest.raises(AssertionError, match="appel réseau réel"):
        be.analyze(plan)
    # et le backend par défaut du module vise bien Ollama en loopback
    assert vb.vision_backend.base_url.startswith("http://127.0.0.1")
