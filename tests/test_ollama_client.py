"""Tests du payload construit pour l'upstream Ollama.

Les tests de routes vérifient que la Gateway transmet bien les champs au
client ; ceux-ci vérifient ce qui part réellement sur le fil.
"""

import pytest

from app.ollama_client import OllamaClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def captured_post(monkeypatch):
    """Intercepte requests.post et expose le corps JSON envoyé."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        if "/api/embed" in url:
            return _FakeResponse({"embeddings": [[0.0] * 768]})
        return _FakeResponse(
            {"response": "OK", "prompt_eval_count": 1, "eval_count": 1, "done": True}
        )

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)
    return calls


def _client():
    return OllamaClient("http://127.0.0.1:11434", timeout=30)


# ------------------------------------------------------------------ generate
def test_generate_payload_omits_keep_alive_by_default(captured_post):
    _client().generate(model="m", prompt="p")
    assert "keep_alive" not in captured_post[0]["json"]


def test_generate_payload_carries_keep_alive_zero(captured_post):
    # Le cœur du correctif : 0 est falsy mais signifie « décharge la VRAM
    # maintenant ». Un `if keep_alive:` dans le client le supprimerait, et la
    # campagne de comparaison garderait son modèle résident 5 minutes de plus.
    _client().generate(model="m", prompt="p", keep_alive=0)
    assert captured_post[0]["json"]["keep_alive"] == 0


def test_generate_payload_carries_keep_alive_duration(captured_post):
    _client().generate(model="m", prompt="p", keep_alive="10m")
    assert captured_post[0]["json"]["keep_alive"] == "10m"


def test_generate_payload_carries_keep_alive_forever(captured_post):
    # -1 = rétention indéfinie. Attention : cela n'empêche PAS l'éviction sous
    # pression VRAM (mesuré sur RTX 3090) ; cela ne règle que l'inactivité.
    _client().generate(model="m", prompt="p", keep_alive=-1)
    assert captured_post[0]["json"]["keep_alive"] == -1


# ---------------------------------------------------------------- embeddings
def test_embed_payload_omits_keep_alive_by_default(captured_post):
    _client().embed(model="m", inputs=["a"])
    assert "keep_alive" not in captured_post[0]["json"]


def test_embed_payload_carries_keep_alive_zero(captured_post):
    _client().embed(model="m", inputs=["a"], keep_alive=0)
    assert captured_post[0]["json"]["keep_alive"] == 0
