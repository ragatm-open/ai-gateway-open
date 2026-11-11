"""Lot vision-1a — validation de requête et admission d'image (owner tests).

Aucun modèle réel : le backend est un mock injecté ; toute la suite tourne
hors ligne et sans GPU.
"""

import base64
import io

import pytest
from PIL import Image

from app import main
from app.vision_backend import VisionBackend, VisionBackendResult

FULL_TASKS = ("auto", "describe", "table", "diagram", "chart", "technical_drawing")


def _png(width=64, height=48, mode="RGB", color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new(mode, (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _save(fmt, width=64, height=48, mode="RGB", **kwargs):
    buf = io.BytesIO()
    Image.new(mode, (width, height), 128 if mode in ("L",) else (10, 20, 30)).save(
        buf, format=fmt, **kwargs)
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


class RecordingBackend(VisionBackend):
    """Mock : rejoue une file de résultats et enregistre chaque plan."""

    def __init__(self, results):
        self.results = list(results)
        self.plans = []

    def analyze(self, plan):
        self.plans.append(plan)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def ok_photo_result(**overrides):
    import json
    raw = {
        "visual_kind": "photo",
        "summary": "Une scène neutre.",
        "warnings": [],
        "payload": {
            "description": "Deux objets posés sur une table.",
            "setting": None,
            "objects": [{"label": "objet", "count": 2, "attributes": [],
                         "certainty": "exact"}],
            "observable_relations": [],
            "visible_text": [],
        },
    }
    raw.update(overrides)
    return VisionBackendResult(raw_json=json.dumps(raw), prompt_tokens=10,
                               completion_tokens=5, total_tokens=15)


@pytest.fixture
def vision_ready(monkeypatch):
    """Active la capability et rend le modèle installé/qualifié — via mocks
    uniquement (jamais l'environnement déployé, §92)."""
    profile = main.registry.resolve_vision("vision-large")
    monkeypatch.setattr(main.settings, "vision_enabled", True)
    monkeypatch.setattr(profile, "vision_tasks", list(FULL_TASKS))
    # Le contrat V1 couvre les six tasks ; le déploiement n'en QUALIFIE qu'une
    # (lot vision-1b). Les tests de contrat simulent un déploiement où la task
    # testée serait qualifiée, en fournissant un profil de prompt pour chacune.
    from app.vision_schemas import PROMPT_PROFILES
    from app import vision_prompts
    monkeypatch.setattr(main, "QUALIFIED_PROMPTS", {
        task: (PROMPT_PROFILES[task][0], f"[prompt de test pour {task}]")
        for task in FULL_TASKS
    })
    # ÉTEND le stub /api/tags du conftest (ne l'écrase pas : la readiness
    # embeddings vérifie les digests via la même liste).
    previous = main.probe_client.list_models
    monkeypatch.setattr(
        main.probe_client, "list_models",
        lambda timeout=5: list(previous(timeout=timeout))
        + [{"name": "mistral-small3.2:24b", "digest": profile.digest}],
    )
    import threading
    from app.vision_admission import VisionAdmission
    monkeypatch.setattr(main, "vision_admission", VisionAdmission(1))
    return profile


@pytest.fixture
def backend(monkeypatch, vision_ready):
    be = RecordingBackend([ok_photo_result()])
    monkeypatch.setattr(main, "vision_backend", be)
    return be


def _post(clientapp, image_bytes=None, mime="image/png", **overrides):
    body = {
        "schema_version": "1",
        "task": "describe",
        "image": {"data": _b64(image_bytes if image_bytes is not None else _png()),
                  "mime_type": mime},
    }
    body.update(overrides)
    return clientapp.post("/v1/vision/analyze", json=body)


# ------------------------------------------------------- validation requête
@pytest.mark.parametrize("mutation", [
    {"schema_version": "3"},   # "2" est admis depuis CV-1 ; "3" n'existe pas
    {"schema_version": 1},
    {"task": "mixed"},
    {"task": "decorative"},
    {"task": "ocr"},
    {"locale": "de"},
    {"request_id": "x" * 129},
    {"champ_inconnu": 1},
    {"url": "https://example.org/image.png"},
    {"path": "/tmp/image.png"},
    {"images": []},
    {"options": {"temperature": 0}},
])
def test_request_validation_rejections(clientapp, backend, mutation):
    r = _post(clientapp, **mutation)
    assert r.status_code == 400
    assert backend.plans == []  # jamais d'appel backend


def test_missing_image_and_bad_base64(clientapp, backend):
    r = clientapp.post("/v1/vision/analyze",
                       json={"schema_version": "1", "task": "auto"})
    assert r.status_code == 400
    r = _post(clientapp)  # sanity : la forme de base passe
    assert r.status_code == 200
    backend.results = [ok_photo_result()]
    r = clientapp.post("/v1/vision/analyze", json={
        "schema_version": "1", "task": "auto",
        "image": {"data": "%%%pas-du-base64%%%", "mime_type": "image/png"}})
    assert r.status_code == 400
    assert len(backend.plans) == 1  # seul l'appel valide a atteint le backend


def test_validation_errors_never_echo_input(clientapp, backend):
    sentinel = "ZZVISIONSECRETBASE64ZZ"
    r = clientapp.post("/v1/vision/analyze", json={
        "schema_version": "1", "task": "auto",
        "image": {"data": sentinel + "%%%", "mime_type": "image/png"},
        "context": {"caption": "ZZCAPTIONSECRETZZ", "inconnu": 1}})
    assert r.status_code == 400
    assert sentinel not in r.text
    assert "ZZCAPTIONSECRETZZ" not in r.text


def test_second_image_field_rejected(clientapp, backend):
    r = _post(clientapp, image2={"data": _b64(_png()), "mime_type": "image/png"})
    assert r.status_code == 400


def test_context_bounds_are_enforced(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.settings, "vision_max_context_field_chars", 10)
    r = _post(clientapp, context={"caption": "x" * 11, "alt_text": None,
                                  "nearby_text": None})
    assert r.status_code == 400
    assert backend.plans == []


# ------------------------------------------------------------------ médias
def test_png_jpeg_webp_accepted(clientapp, backend):
    backend.results = [ok_photo_result(), ok_photo_result(), ok_photo_result()]
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"),
                      ("WEBP", "image/webp")):
        r = _post(clientapp, image_bytes=_save(fmt), mime=mime)
        assert r.status_code == 200, fmt
        assert r.json()["image"]["format"] == mime.split("/")[1]


def test_mime_spoof_rejected(clientapp, backend):
    r = _post(clientapp, image_bytes=_save("JPEG"), mime="image/png")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unsupported_image_type"
    assert r.json()["detail"]["retryable"] is False
    assert backend.plans == []


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_unsupported_formats_rejected(clientapp, backend, fmt):
    r = _post(clientapp, image_bytes=_save(fmt), mime="image/png")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unsupported_image_type"


def test_animated_webp_rejected_not_first_frame(clientapp, backend):
    frames = [Image.new("RGB", (16, 16), c) for c in ((255, 0, 0), (0, 255, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=100)
    r = _post(clientapp, image_bytes=buf.getvalue(), mime="image/webp")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unsupported_image_type"
    assert backend.plans == []  # jamais d'analyse silencieuse de la frame 1


def test_malformed_bytes_rejected(clientapp, backend):
    r = _post(clientapp, image_bytes=b"pas une image du tout")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "image_decode_failed"


def test_byte_limit_413(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.settings, "vision_max_image_bytes", 100)
    r = _post(clientapp)
    assert r.status_code == 413
    d = r.json()["detail"]
    assert d["code"] == "image_too_large" and d["retryable"] is False
    assert backend.plans == []


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_pixel_limit_413_post_decode(clientapp, backend, monkeypatch):
    monkeypatch.setattr(main.settings, "vision_max_image_pixels", 5000)
    r = _post(clientapp, image_bytes=_png(100, 100))  # 10 000 px, 2x la borne exacte
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "image_too_large"


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_decompression_bomb_header_guard(clientapp, backend, monkeypatch):
    # 100x100 = 10 000 px > 2 x 3 000 : Pillow lève DecompressionBombError dès
    # l'en-tête, AVANT allocation — mappé 413 sans gonfler la mémoire.
    monkeypatch.setattr(main.settings, "vision_max_image_pixels", 3000)
    r = _post(clientapp, image_bytes=_png(100, 100))
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "image_too_large"


def test_exif_orientation_applied_and_metadata_stripped(clientapp, backend):
    img = Image.new("RGB", (40, 20), (200, 10, 10))
    exif = Image.Exif()
    exif[274] = 6                       # orientation : rotation 270°
    exif[271] = "SENTINELCAMERA123"     # Make — ne doit JAMAIS atteindre le backend
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    backend.results = [ok_photo_result()]
    r = _post(clientapp, image_bytes=buf.getvalue(), mime="image/jpeg")
    assert r.status_code == 200
    body = r.json()["image"]
    assert (body["width"], body["height"]) == (20, 40)  # pixels réorientés
    forwarded = backend.plans[-1].image_bytes
    reopened = Image.open(io.BytesIO(forwarded))
    assert dict(reopened.getexif()) == {}               # métadonnées mortes
    assert reopened.size == (20, 40)
    assert "SENTINELCAMERA123" not in forwarded.decode("latin-1")


def test_grayscale_and_alpha_normalized_deterministically(clientapp, backend):
    backend.results = [ok_photo_result(), ok_photo_result()]
    r = _post(clientapp, image_bytes=_save("PNG", mode="L"))
    assert r.status_code == 200
    # RGBA entièrement transparent -> aplati sur fond BLANC fixe.
    rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    buf = io.BytesIO(); rgba.save(buf, format="PNG")
    r = _post(clientapp, image_bytes=buf.getvalue())
    assert r.status_code == 200
    forwarded = Image.open(io.BytesIO(backend.plans[-1].image_bytes))
    assert forwarded.mode == "RGB"
    assert forwarded.getpixel((4, 4)) == (255, 255, 255)
