from app import main
from app.errors import UpstreamError


def test_generate_ok(clientapp, monkeypatch):
    def fake_generate(model, prompt, system=None, temperature=None, options=None,
                      think=None, keep_alive=None):
        return {
            "response": "OK",
            "prompt_eval_count": 10,
            "eval_count": 2,
            "done": True,
        }

    monkeypatch.setattr(main.client, "generate", fake_generate)
    r = clientapp.post("/v1/generate", json={"prompt": "dis OK", "model": "llama3.1:8b"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "OK"
    assert body["provider"] == "ai_gateway"
    assert body["upstream_provider"] == "ollama"
    assert body["tokens"]["total"] == 12
    assert body["usage"]["total_tokens"] == 12
    assert body["cost"] == 0.0
    assert body["done"] is True


def test_generate_empty_prompt(clientapp):
    r = clientapp.post("/v1/generate", json={"prompt": ""})
    assert r.status_code == 400
    assert r.json()["detail"]["status"] == "error"


def test_generate_think_forwarded(clientapp, monkeypatch):
    captured = {}

    def fake_generate(model, prompt, system=None, temperature=None, options=None,
                      think=None, keep_alive=None):
        captured["think"] = think
        return {"response": "OK", "prompt_eval_count": 1, "eval_count": 1, "done": True}

    monkeypatch.setattr(main.client, "generate", fake_generate)
    r = clientapp.post("/v1/generate", json={"prompt": "x", "think": False})
    assert r.status_code == 200
    assert captured["think"] is False


def test_generate_keep_alive_forwarded(clientapp, monkeypatch):
    captured = {}

    def fake_generate(model, prompt, system=None, temperature=None, options=None,
                      think=None, keep_alive=None):
        captured["keep_alive"] = keep_alive
        return {"response": "OK", "prompt_eval_count": 1, "eval_count": 1, "done": True}

    monkeypatch.setattr(main.client, "generate", fake_generate)
    r = clientapp.post("/v1/generate", json={"prompt": "x", "keep_alive": "10m"})
    assert r.status_code == 200
    assert captured["keep_alive"] == "10m"


def test_generate_keep_alive_zero_is_forwarded(clientapp, monkeypatch):
    # Régression : 0 signifie « décharge la VRAM immédiatement » — c'est la
    # valeur utile en fin de campagne. Étant falsy, un test `if keep_alive:`
    # la supprimerait silencieusement et le modèle resterait résident 5 min.
    captured = {}

    def fake_generate(model, prompt, system=None, temperature=None, options=None,
                      think=None, keep_alive=None):
        captured["keep_alive"] = keep_alive
        return {"response": "OK", "prompt_eval_count": 1, "eval_count": 1, "done": True}

    monkeypatch.setattr(main.client, "generate", fake_generate)
    r = clientapp.post("/v1/generate", json={"prompt": "x", "keep_alive": 0})
    assert r.status_code == 200
    assert captured["keep_alive"] == 0


def test_generate_keep_alive_absent_stays_none_for_ungoverned_model(clientapp, monkeypatch):
    # Depuis le lot runtime-hardening, un modèle GOUVERNÉ reçoit le keep_alive
    # de son profil ; pour un modèle HORS registre (nom fictif — gemma3:27b est
    # gouverné depuis une version antérieure), l'intention
    # historique reste vraie : sans le champ, rien n'est transmis et Ollama
    # applique son défaut.
    captured = {}

    def fake_generate(model, prompt, system=None, temperature=None, options=None,
                      think=None, keep_alive=None):
        captured["keep_alive"] = keep_alive
        return {"response": "OK", "prompt_eval_count": 1, "eval_count": 1, "done": True}

    monkeypatch.setattr(main.client, "generate", fake_generate)
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "unregistered-model:latest"})
    assert r.status_code == 200
    assert captured["keep_alive"] is None


def test_generate_blank_prompt_is_400(clientapp):
    # Prompt uniquement blanc : le validateur custom levait un ValueError
    # non sérialisable -> 500. jsonable_encoder garantit désormais un 400.
    r = clientapp.post("/v1/generate", json={"prompt": "   "})
    assert r.status_code == 400
    assert r.json()["detail"]["status"] == "error"


def test_generate_upstream_error(clientapp, monkeypatch):
    def boom(**kwargs):
        raise UpstreamError("ollama down")

    monkeypatch.setattr(main.client, "generate", boom)
    r = clientapp.post("/v1/generate", json={"prompt": "x"})
    assert r.status_code == 502
    assert r.json()["detail"]["provider"] == "ollama"
