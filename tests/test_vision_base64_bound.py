"""Lot P7 — une image Base64 trop grande est refusée AVANT tout décodage, et
une image valide n'est décodée qu'UNE fois.

Avant : le validateur Pydantic décodait la chaîne entière pour la valider
(résultat jeté), la route la décodait à nouveau, et la borne de 4 Mio n'était
contrôlée qu'après. Après : `max_length` Pydantic sur la chaîne (borne dérivée
de la taille décodée maximale), validation syntaxique par expression
régulière sans décodage, un seul `b64decode` dans la route.
"""

import base64

import pytest

from app import main
from tests.test_vision_media import (  # noqa: F401 — fixtures réutilisées
    _b64,
    _png,
    _post,
    backend,
    ok_photo_result,
    vision_ready,
)

MAX_BYTES = 4 * 1024 * 1024
MAX_CHARS = 4 * ((MAX_BYTES + 2) // 3)          # 5 592 408


@pytest.fixture
def decode_spy(monkeypatch):
    calls = []
    real = base64.b64decode

    def spy(*args, **kwargs):
        calls.append(kwargs.get("validate"))
        return real(*args, **kwargs)

    monkeypatch.setattr(base64, "b64decode", spy)
    return calls


def _raw_post(clientapp, data, mime="image/png"):
    return clientapp.post("/v1/vision/analyze", json={
        "schema_version": "1", "task": "describe",
        "image": {"data": data, "mime_type": mime}})


def test_bound_is_derived_from_decoded_limit():
    assert main.settings.vision_max_image_base64_chars == MAX_CHARS
    # Toute chaîne plus longue ne PEUT PAS décoder en <= 4 Mio ; toute chaîne
    # <= borne peut décoder en <= 4 Mio + 2 octets (contrôlés ensuite en 413).
    assert len(base64.b64encode(bytes(MAX_BYTES))) == MAX_CHARS
    assert len(base64.b64encode(bytes(MAX_BYTES + 1))) == MAX_CHARS
    assert len(base64.b64encode(bytes(MAX_BYTES + 3))) == MAX_CHARS + 4


def test_small_valid_image_decoded_exactly_once(clientapp, backend, decode_spy):
    r = _post(clientapp)
    assert r.status_code == 200
    assert decode_spy == [True]                      # un seul décodage, strict


def test_oversized_base64_rejected_before_any_decode(clientapp, backend, decode_spy):
    data = "A" * (MAX_CHARS + 4)                     # syntaxiquement valide, trop long
    r = _raw_post(clientapp, data)
    assert r.status_code == 400
    errors = r.json()["detail"]["errors"]
    assert any(e["type"] == "string_too_long" and e["loc"][-1] == "data" for e in errors)
    assert "A" * 100 not in r.text                   # jamais d'écho de l'input
    assert decode_spy == []                          # AUCUN décodage
    assert backend.plans == []


def test_exactly_at_bound_passes_length_gate(clientapp, backend, decode_spy):
    """Longueur = borne : la garde de taille laisse passer, le décodage a lieu
    UNE fois, puis l'admission tranche (ici : octets non décodables en image)."""
    data = base64.b64encode(bytes(MAX_BYTES)).decode()
    assert len(data) == MAX_CHARS
    r = _raw_post(clientapp, data)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "image_decode_failed"
    assert decode_spy == [True]


def test_just_above_decoded_limit_is_413_after_single_decode(clientapp, backend, decode_spy):
    """4 Mio + 1 octet tient dans la même longueur Base64 que 4 Mio : la garde
    de longueur laisse passer, le contrôle d'octets décodés rend 413 — contrat
    `image_too_large` inchangé, un seul décodage."""
    data = base64.b64encode(bytes(MAX_BYTES + 1)).decode()
    assert len(data) == MAX_CHARS
    r = _raw_post(clientapp, data)
    assert r.status_code == 413 and r.json()["detail"]["code"] == "image_too_large"
    assert decode_spy == [True]


@pytest.mark.parametrize("data", ["%%%pas-du-base64%%%", "abc", "abcd=", "ab==cd", "QUJD\n", "QUJDRA=="[:-1]])
def test_invalid_base64_rejected_syntactically_without_decode(clientapp, backend, decode_spy, data):
    r = _raw_post(clientapp, data)
    assert r.status_code == 400
    assert decode_spy == []                          # syntaxe jugée sans décodage
    assert backend.plans == []


def test_existing_image_controls_unchanged(clientapp, backend, monkeypatch):
    r = _post(clientapp, mime="image/jpeg")          # spoof MIME : décodage réel fait foi
    assert r.status_code == 400 and r.json()["detail"]["code"] == "unsupported_image_type"
    monkeypatch.setattr(main.settings, "vision_max_image_bytes", 100)
    r = _post(clientapp)
    assert r.status_code == 413 and r.json()["detail"]["code"] == "image_too_large"
