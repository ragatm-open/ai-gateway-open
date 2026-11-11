"""Lot AI-GATEWAY-VISION-RESIDENCY-02 — résidence courte du modèle vision.

Un SEUL facteur varie : `keep_alive`. Ces tests verrouillent trois choses que
la mesure seule ne protège pas dans le temps :

* le **défaut reste 0** — une installation qui ne configure rien conserve
  exactement le comportement qualifié par MODEL-1/1B ;
* la **garde production** est inopérable : en production la valeur est forcée
  à 0 même si l'environnement demande autre chose ;
* la valeur configurée arrive **telle quelle** dans le payload Ollama, sans
  qu'aucune autre option d'inférence ne dérive.

L'allowlist (0/10/30) est volontairement plus stricte qu'un intervalle : une
faute de frappe ne peut pas produire une résidence permanente.
"""

import pytest

from app import main
from app.settings import Settings
from app.vision_backend import QUALIFIED_FIRST_ATTEMPT, QUALIFIED_RETRY_ATTEMPT
from tests.test_vision_1b_transport import ollama  # noqa: F401
from tests.test_vision_media import _post, backend, vision_ready  # noqa: F401

ENV = "AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS"


def _settings(monkeypatch, value=None, environment="staging"):
    monkeypatch.setenv("AI_GATEWAY_ENVIRONMENT", environment)
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    return Settings()


# --------------------------------------------------------------- configuration
def test_default_is_zero(monkeypatch):
    """Sans configuration : comportement actuel, à l'identique."""
    assert _settings(monkeypatch, None).vision_keep_alive_seconds == 0


def test_staging_can_select_qualified_values(monkeypatch):
    assert _settings(monkeypatch, "10").vision_keep_alive_seconds == 10
    assert _settings(monkeypatch, "30").vision_keep_alive_seconds == 30
    assert _settings(monkeypatch, "0").vision_keep_alive_seconds == 0


@pytest.mark.parametrize(
    "bad", ["5", "60", "120", "300", "-1", "3600", "10.5", "abc", "010s"]
)
def test_non_qualified_values_refuse_to_start(monkeypatch, bad):
    """Allowlist : tout ce qui n'est pas 0/10/30 empêche le DÉMARRAGE (un lot antérieur,
    fail-fast) — avant, la valeur retombait silencieusement sur 0.

    `-1` (résidence indéfinie) et les TTL longs restent INATTEIGNABLES par
    configuration — ils exigeraient une modification de code, donc un lot.
    """
    from app.settings import ConfigurationError
    with pytest.raises(ConfigurationError, match="AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS"):
        _settings(monkeypatch, bad)


@pytest.mark.parametrize("empty", ["", "  "])
def test_empty_value_means_absent(monkeypatch, empty):
    """Variable vide (`X=` dans un fichier env) = absente = défaut 0."""
    assert _settings(monkeypatch, empty).vision_keep_alive_seconds == 0


@pytest.mark.parametrize("value", ["10", "30"])
def test_production_guard_forces_zero(monkeypatch, value):
    """Garde production (§7) : une valeur QUALIFIÉE non nulle est forcée à 0 en
    production — politique, pas validation. Les valeurs hors allowlist, elles,
    refusent de démarrer quel que soit l'environnement."""
    assert _settings(monkeypatch, value, environment="production").vision_keep_alive_seconds == 0


@pytest.mark.parametrize("value", ["-1", "3600"])
def test_production_rejects_non_qualified_values_too(monkeypatch, value):
    from app.settings import ConfigurationError
    with pytest.raises(ConfigurationError, match="AI_GATEWAY_VISION_KEEP_ALIVE_SECONDS"):
        _settings(monkeypatch, value, environment="production")


def test_qualified_profile_itself_is_unchanged():
    """Les constantes du profil qualifié ne bougent pas : la résidence est
    appliquée par copie au moment du plan, jamais en mutant le profil."""
    for options in (QUALIFIED_FIRST_ATTEMPT, QUALIFIED_RETRY_ATTEMPT):
        assert options.keep_alive == 0


# ------------------------------------------------------------------- transport
def test_payload_carries_configured_keep_alive(clientapp, ollama, monkeypatch):
    """Le payload RÉELLEMENT envoyé à Ollama porte la valeur configurée."""
    monkeypatch.setattr(main.settings, "vision_keep_alive_seconds", 10)
    assert _post(clientapp, task="describe").status_code == 200
    assert ollama.calls[0]["payload"]["keep_alive"] == 10


def test_default_payload_still_unloads_immediately(clientapp, ollama):
    """Sans configuration, le payload conserve `keep_alive: 0`."""
    assert main.settings.vision_keep_alive_seconds == 0  # défaut du dépôt
    assert _post(clientapp, task="describe").status_code == 200
    assert ollama.calls[0]["payload"]["keep_alive"] == 0


def test_no_other_inference_option_changes(clientapp, ollama, monkeypatch):
    """Un seul facteur varie : tout le reste du payload est identique."""
    assert _post(clientapp, task="describe").status_code == 200
    reference = dict(ollama.calls[0]["payload"])

    monkeypatch.setattr(main.settings, "vision_keep_alive_seconds", 30)
    assert _post(clientapp, task="describe").status_code == 200
    candidate = dict(ollama.calls[1]["payload"])

    assert candidate.pop("keep_alive") == 30
    assert reference.pop("keep_alive") == 0
    assert candidate == reference          # modèle, prompts, format, images, options
    assert candidate["options"] == {"temperature": 0.0, "seed": 42,
                                    "num_ctx": 8192, "num_predict": 2048}


def test_retry_attempt_also_carries_keep_alive(clientapp, ollama, monkeypatch):
    """Le retry structurel hérite de la résidence sans changer son propre
    profil (température variée), sinon la 2e tentative rechargerait le modèle."""
    monkeypatch.setattr(main.settings, "vision_keep_alive_seconds", 10)
    ollama.responses(
        {"response": "pas du json", "done": True, "done_reason": "stop"},
        {"response": '{"visual_kind": "photo", "summary": "s", "warnings": [], '
                     '"payload": {"description": "d", "setting": null, "objects": [], '
                     '"observable_relations": [], "visible_text": []}}',
         "done": True, "done_reason": "stop"},
    )
    assert _post(clientapp, task="describe").status_code == 200
    assert len(ollama.calls) == 2
    assert [c["payload"]["keep_alive"] for c in ollama.calls] == [10, 10]
    assert ollama.calls[1]["payload"]["options"]["temperature"] == 0.35


# ------------------------------------------------------------------ surface HTTP
def test_health_vision_exposes_effective_value(clientapp, monkeypatch, vision_ready):
    """`/health/vision` publie la valeur EFFECTIVE (après garde), ce qui rend
    l'état vérifiable sans lire l'environnement du conteneur."""
    monkeypatch.setattr(main.settings, "vision_keep_alive_seconds", 10)
    body = clientapp.get("/health/vision").json()
    assert body["keep_alive_seconds"] == 10
