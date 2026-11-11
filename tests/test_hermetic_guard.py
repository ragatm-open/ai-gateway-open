"""Lot P9 — la suite unitaire est hermétique : ni Ollama, ni réseau, ni GPU.

Deux invariants :

1. les routes health passent avec les SEULS doubles de `conftest.py` — c'est
   le niveau d'isolation commun, pas un patch test par test ;
2. un appel réseau réel qui échapperait aux mocks échoue immédiatement, nommé,
   au lieu de contacter un Ollama local (vert par accident sur un poste
   équipé, rouge partout ailleurs).
"""

import pytest
import requests

from tests.conftest import ForbiddenNetworkAccess

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast


def test_health_ready_is_green_with_shared_fixtures_only(clientapp):
    # Aucun mock local : si `probe_client.embed` n'était pas doublé au niveau
    # commun, la sonde encoderait PROBE_TEXT sur le réseau.
    r = clientapp.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["checks"]["embedding_call"] is True


def test_health_vision_is_green_with_shared_fixtures_only(clientapp, monkeypatch):
    # `list_running_models` (/api/ps) est la seconde sonde réelle.
    from app import main

    monkeypatch.setattr(main.settings, "vision_enabled", True)
    r = clientapp.get("/health/vision")
    assert r.status_code == 200
    assert "keep_alive_seconds" in r.json()


def test_unexpected_network_call_fails_fast():
    with pytest.raises(ForbiddenNetworkAccess, match="mock manquant"):
        requests.get("http://127.0.0.1:1/", timeout=1)
