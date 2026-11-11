"""Lot AI-GATEWAY-VISION-1A — durcissement fail-closed des requêtes publiques.

Le contrat gelé (CONTRACT-0 §5/§D) impose qu'un champ inconnu — notamment
`images` — envoyé à `/v1/generate` ne produise JAMAIS un 200 texte-seul
silencieux : 400, enveloppe de validation existante, aucun appel upstream.
"""

import pytest

from app import main


def _spy_generate(monkeypatch):
    calls = {"n": 0}

    def fake_generate(model, prompt, system=None, temperature=None, options=None,
                      think=None, keep_alive=None):
        calls["n"] += 1
        return {"response": "OK", "done": True}

    monkeypatch.setattr(main.client, "generate", fake_generate)
    return calls


@pytest.mark.parametrize("extra_field", [
    {"images": ["aGVsbG8="]},
    {"image": {"data": "aGVsbG8=", "mime_type": "image/png"}},
    {"champ_inconnu": 1},
])
def test_generate_rejects_unknown_multimodal_fields(clientapp, monkeypatch, extra_field):
    calls = _spy_generate(monkeypatch)
    r = clientapp.post("/v1/generate", json={"prompt": "dis OK", **extra_field})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["message"] == "payload invalide"
    assert any(e.get("type") == "extra_forbidden" for e in detail["errors"])
    assert calls["n"] == 0  # jamais d'appel upstream


def test_generate_stream_rejects_images_before_stream(clientapp, monkeypatch):
    opened = {"n": 0}
    monkeypatch.setattr(main.client, "open_generate_stream",
                        lambda **kw: opened.__setitem__("n", opened["n"] + 1))
    r = clientapp.post("/v1/generate/stream",
                       json={"prompt": "x", "images": ["aGVsbG8="]})
    assert r.status_code == 400
    assert opened["n"] == 0
