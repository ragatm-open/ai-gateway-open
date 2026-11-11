"""Fixtures de test : client FastAPI avec upstream Ollama simulé.

Le lot 1 a ajouté un état de processus (registre vérifié, cache de readiness,
compteurs). La fixture `_isolate_state` le remet à zéro entre deux tests et
neutralise la vérification de digest, qui sinon taperait sur l'Ollama local et
rendrait la suite dépendante de la machine.
"""

import os
import socket
import time

import pytest
from fastapi.testclient import TestClient

# Environnement de la suite HÉRITÉE, posé AVANT l'import de
# `app.settings` (instancié à l'import) :
#   · registre = fixture de gouvernance « forme historique »
#     (tests/data/registry_fixture.yaml), PAS le registre Compact servi par
#     AI Gateway Open — les tests hérités prouvent des invariants de
#     gouvernance sur des profils synthétiques (raisonnement par niveaux /
#     bascule, vision, ASR) absents du profil Compact ;
#   · modèles par défaut historiques et registres NON stricts : la suite
#     héritée couvre explicitement le chemin « legacy » (relais d'un modèle
#     hors registre), que le profil Open désactive par défaut.
# Les défauts Open (registres stricts, registre Compact, modèles Compact) sont
# vérifiés par tests/open/, qui instancient `Settings()` et chargent le
# registre réel sans ces variables.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_LEGACY_SUITE_ENV = {
    "AI_GATEWAY_MODELS_REGISTRY": os.path.join(_TESTS_DIR, "data", "registry_fixture.yaml"),
    "OLLAMA_GENERATE_MODEL": "llama3.1:8b",
    "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
    "AI_GATEWAY_EMBEDDINGS_STRICT_REGISTRY": "0",
    "AI_GATEWAY_GENERATION_STRICT_REGISTRY": "0",
}
for _name, _value in _LEGACY_SUITE_ENV.items():
    os.environ.setdefault(_name, _value)

from app import asr_routes, main  # noqa: E402
from app.metrics import counters
from app.ollama_client import ENDPOINT_EMBED, EmbedResult
from app.registry import STATUS_READY


class ForbiddenNetworkAccess(RuntimeError):
    """Un test unitaire a tenté d'ouvrir une connexion réseau réelle.

    Volontairement PAS une OSError : urllib3 convertirait celle-ci en
    `requests.ConnectionError`, la Gateway en 502/503, et le test verrait un
    « upstream indisponible » plausible au lieu d'une faute de mock. Ici la
    faute remonte telle quelle, nommée, au premier appel.
    """


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Garde-fou : aucune connexion sortante pendant la suite unitaire.

    La suite doit rester verte sous `--network none`. Sur un poste où un
    Ollama local répond, un mock oublié passerait inaperçu (vert par accident)
    et échouerait ailleurs — exactement l'histoire des trois tests health du
    un lot antérieur. `TestClient` parle à l'app en mémoire (ASGI) et n'ouvre aucun
    socket : seul un appel `requests` non intercepté arrive ici.
    """
    def guard(self, address, *args, **kwargs):
        raise ForbiddenNetworkAccess(
            f"appel réseau interdit en test unitaire : connect({address!r}) — "
            "mock manquant"
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch):
    counters.reset()
    main._readiness_cache_clear()
    asr_routes.reset_terminal_accounting()
    main._verify_state["last"] = 0.0

    # Digest conforme par défaut : `list_models` de la sonde renvoie ce que le
    # registre attend, donc aucun accès réseau et un statut `ready` stable.
    installed = [
        {
            "name": f"{profile.ollama_name}:latest"
            if ":" not in profile.ollama_name
            else profile.ollama_name,
            "digest": profile.digest,
            "size": 274302450,
            "modified_at": "2026-07-06T20:07:12Z",
            "details": {
                "family": "nomic-bert",
                "parameter_size": "137M",
                "quantization_level": "F16",
            },
        }
        for profile in main.registry.models.values()
    ]
    monkeypatch.setattr(main.probe_client, "list_models", lambda timeout=5: installed)

    # Les DEUX autres sondes réelles, neutralisées au même niveau :
    # `/health/ready` encode PROBE_TEXT via `probe_client.embed` et
    # `/health/vision` lit la résidence via `probe_client.list_running_models`.
    # Sans ces doubles, ces routes touchaient l'Ollama local — verts sur un
    # poste équipé, rouges partout ailleurs. Un test qui veut un comportement
    # particulier (panne, lenteur, comptage) surcharge ces attributs après
    # coup, comme test_health_ready.py le fait déjà.
    by_alias = {
        alias: profile
        for profile in main.registry.models.values()
        for alias in profile._name_aliases()
    }

    def fake_probe_embed(model, inputs, keep_alive=None):
        profile = by_alias.get(model)
        dimension = profile.dimension if profile is not None else 768
        unit = [1.0] + [0.0] * (dimension - 1)   # normé : passe le contrôle de norme
        return EmbedResult(
            [list(unit) for _ in inputs],
            ENDPOINT_EMBED,
            fallback_used=False,
            ollama_metadata={},
        )

    monkeypatch.setattr(main.probe_client, "embed", fake_probe_embed)
    monkeypatch.setattr(main.probe_client, "list_running_models", lambda timeout=5: [])

    for profile in main.registry.models.values():
        profile.status = STATUS_READY
        profile.observed_digest = profile.digest
        profile.status_detail = None
        profile.verified_at = time.time()      # preuve conforme courante

    yield

    counters.reset()
    main._readiness_cache_clear()
    asr_routes.reset_terminal_accounting()
    main._verify_state["last"] = 0.0


@pytest.fixture
def clientapp(monkeypatch):
    return TestClient(main.app)


@pytest.fixture
def clientapp_tolerant(monkeypatch):
    """Client qui RETOURNE la réponse 500 au lieu de propager l'exception.

    Par défaut `TestClient` relaie les exceptions serveur au test, ce qui est
    pratique pour déboguer mais empêche d'observer ce qu'un vrai client verrait.
    Sous uvicorn, le handler `Exception` de `app/main.py` renvoie bien un 500
    dans l'enveloppe d'erreur habituelle.
    """
    return TestClient(main.app, raise_server_exceptions=False)


# =============================================================================
# Profils de tests
# =============================================================================
# FAST     : `pytest -m fast` — tests/open/ (contrats, garde-fous Open ; marqués ici
#            par chemin) + registres, configuration, santé, provenance, chaîne
#            d'approvisionnement (marqués `pytestmark` dans leur module).
# STANDARD : `pytest tests` — suite complète simulée (aucun service, aucun modèle).
# FULL     : identique à STANDARD ; aucun test de ce dépôt ne dépend d'un service
#            réel, il n'existe donc pas de troisième périmètre à documenter.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "fast: profil FAST — contrats Open, registres, configuration, santé, provenance",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "/tests/open/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(pytest.mark.fast)
