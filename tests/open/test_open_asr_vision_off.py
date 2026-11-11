"""Conformité Open — ASR et Vision désactivés par défaut, composants exclus."""

import base64
import io

from PIL import Image

from app import main


def _png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_vision_disabled_is_typed(open_app, monkeypatch):
    monkeypatch.setattr(main.vision_backend, "analyze", lambda plan: (_ for _ in ()).throw(AssertionError("modèle vision appelé")))
    r = open_app.post("/v1/vision/analyze", json={"task": "describe", "schema_version": "1",
                                                   "image": {"mime_type": "image/png", "data": _png_b64()}})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "vision_not_enabled"
    cap = open_app.get("/v1/vision/capacity").json()
    assert cap["vision"]["status"] == "disabled" and cap["vision"]["enabled"] is False
    assert cap["model"]["logical"] == "unconfigured"
    health = open_app.get("/health/vision").json()
    assert health["status"] == "disabled" and health["semantic_available"] is False
    assert health["models"] == []                   # aucun modèle vision au registre Compact


def test_semantic_extract_is_unavailable_even_if_vision_enabled(open_app, monkeypatch):
    monkeypatch.setattr(main.settings, "vision_enabled", True)
    r = open_app.post("/v1/vision/analyze", json={"task": "semantic_extract", "schema_version": "3",
                                                   "image": {"mime_type": "image/png", "data": _png_b64()}})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "vision_task_unavailable"


def test_structured_extract_route_absent(open_app):
    r = open_app.post("/v1/vision/structured-extract", json={})
    assert r.status_code == 404


def test_asr_disabled_is_typed(open_app, monkeypatch):
    r = open_app.post("/v1/audio/transcriptions", content=b"--x--", headers={"content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "asr_unavailable"
    assert open_app.get("/health/asr").json()["status"] == "disabled"
    runtime = open_app.get("/health/runtime").json()
    assert runtime["asr"]["enabled"] is False and runtime["asr"]["models"] == []


def test_no_asr_or_vision_worker_started():
    import threading
    names = {t.name for t in threading.enumerate()}
    assert not any(("asr" in n.lower() or "vision" in n.lower() or "v2h" in n.lower()) for n in names), names


def test_excluded_modules_not_present():
    import importlib
    import pytest
    for module in ("app.structx_route", "app.vision_semantic_route", "app.v2h_shadow_hook",
                   "app.vision_semantic_backend"):
        with pytest.raises(ImportError):
            importlib.import_module(module)
