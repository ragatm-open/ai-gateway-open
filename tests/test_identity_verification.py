"""Lot P2 — gouvernance ≠ identité vérifiée.

`governed` disait « autorisé par la politique ET aucune divergence prouvée » ;
un client pouvait donc recevoir `governed: true` et un `model_digest` égal au
digest ATTENDU alors que rien n'avait été observé (statut `unavailable`).

Après P2 :
  governed          autorisé par la politique du registre (inchangé, fail-open)
  identity_verified preuve APPLICABLE à l'état courant : digest observé au
                    dernier contrôle ET conforme — fonction pure de l'état,
                    aucun TTL parallèle (`ready` n'est posé que par une
                    observation conforme, toute autre issue le retire)
  expected_digest   déclaratif (registre)
  model_digest      observé, ou null — JAMAIS rempli avec l'attendu
  verified_at       dernière preuve connue (observation conforme), conservée
                    à travers unavailable/degraded ; ne vaut pas preuve courante
"""

import logging
import time

import pytest

from app import main
from app.ollama_client import ENDPOINT_EMBED, EmbedResult
from app.registry import (
    STATUS_DEGRADED,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    STATUS_UNKNOWN,
    load_registry,
)

A = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"   # attendu (legacy-768)
B = "b" * 64                                                                # divergent
ISO_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$"


class FakeTags:
    """Client `list_models` pilotable : liste d'entrées, ou exception."""

    def __init__(self, entries=None, raise_exc=None):
        self.entries, self.raise_exc = entries, raise_exc

    def list_models(self, timeout=5):
        if self.raise_exc:
            raise self.raise_exc
        return list(self.entries or [])


def _entry(name, digest):
    return {"name": name, "digest": digest}


def _fresh_registry():
    return load_registry(main.settings.models_registry_path)


def _row(p):
    return (p.status, p.governed, p.identity_verified, p.digest, p.observed_digest,
            p.verified_at is not None)


# --------------------------------------------------------------------------- #
# 1. Machine d'état — table
# --------------------------------------------------------------------------- #
def test_state_table():
    reg = _fresh_registry()
    p = reg.models["legacy-768"]
    # 5. jamais vérifié
    assert _row(p) == (STATUS_UNKNOWN, True, False, A, None, False)
    # 1. observé conforme
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", A)]))
    assert _row(p) == (STATUS_READY, True, True, A, A, True)
    first_proof = p.verified_at
    # 2. observé divergent
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", B)]))
    assert _row(p) == (STATUS_DEGRADED, False, False, A, B, True)
    assert p.verified_at == first_proof                      # dernière preuve conservée
    # 4. /api/tags inaccessible (Ollama indisponible)
    reg.verify_digests(FakeTags(raise_exc=RuntimeError("tags down")))
    assert _row(p) == (STATUS_UNAVAILABLE, True, False, A, None, True)
    # 3. modèle absent de l'upstream
    reg.verify_digests(FakeTags([]))
    assert _row(p) == (STATUS_UNAVAILABLE, True, False, A, None, True)
    # digest non rapporté
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", "")]))
    assert _row(p) == (STATUS_UNAVAILABLE, True, False, A, None, True)


# --------------------------------------------------------------------------- #
# 2. Transitions
# --------------------------------------------------------------------------- #
def test_transition_ready_to_unavailable_keeps_last_proof_but_not_verified():
    reg = _fresh_registry(); p = reg.models["legacy-768"]
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", A)]))
    assert p.identity_verified and p.verified_at is not None
    proof = p.verified_at
    reg.verify_digests(FakeTags(raise_exc=RuntimeError("tags down")))
    assert p.status == STATUS_UNAVAILABLE
    assert p.identity_verified is False
    assert p.observed_digest is None
    assert p.verified_at == proof                            # dernière preuve connue


def test_transition_unavailable_to_ready_updates_proof():
    reg = _fresh_registry(); p = reg.models["legacy-768"]
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", A)]))
    t1 = p.verified_at
    reg.verify_digests(FakeTags(raise_exc=RuntimeError("down")))
    time.sleep(0.002)
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", A)]))
    assert p.status == STATUS_READY and p.identity_verified is True
    assert p.observed_digest == A
    assert p.verified_at is not None and p.verified_at > t1


def test_transition_ready_to_degraded_exposes_divergent_digest():
    reg = _fresh_registry(); p = reg.models["legacy-768"]
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", A)]))
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", B)]))
    assert p.status == STATUS_DEGRADED and p.governed is False
    assert p.identity_verified is False
    assert p.observed_digest == B and p.digest == A          # rien ne prétend B == A


def test_transition_degraded_to_ready_after_fix():
    reg = _fresh_registry(); p = reg.models["legacy-768"]
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", B)]))
    assert p.status == STATUS_DEGRADED and p.verified_at is None
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", A)]))
    assert p.status == STATUS_READY and p.identity_verified is True
    assert p.verified_at is not None and p.governed is True


def test_verified_at_only_set_by_a_conforming_observation():
    reg = _fresh_registry(); p = reg.models["legacy-768"]
    reg.verify_digests(FakeTags(raise_exc=RuntimeError("down")))
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", B)]))
    assert p.verified_at is None and p.verified_at_iso is None
    reg.verify_digests(FakeTags([_entry("nomic-embed-text:latest", A)]))
    import re
    assert re.match(ISO_RE, p.verified_at_iso)


# --------------------------------------------------------------------------- #
# 3. Surface /v1/embeddings
# --------------------------------------------------------------------------- #
@pytest.fixture
def embed_ok(monkeypatch):
    seen = {}

    def embed(model, inputs, keep_alive=None):
        seen.update({"model": model, "inputs": list(inputs)})
        return EmbedResult([[1.0] + [0.0] * 767 for _ in inputs], ENDPOINT_EMBED,
                           fallback_used=False, ollama_metadata={})

    monkeypatch.setattr(main.client, "embed", embed)
    return seen


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _force_verify(monkeypatch, tags):
    """Fait passer le prochain `_ensure_registry_verified()` par la sonde."""
    monkeypatch.setattr(main.probe_client, "list_models", tags.list_models)
    main._verify_state["last"] = 0.0


def test_ready_response(clientapp, embed_ok, monkeypatch):
    _force_verify(monkeypatch, FakeTags([_entry("nomic-embed-text:latest", A)]))
    body = clientapp.post("/v1/embeddings", json={"texts": ["a"]}).json()
    assert body["governed"] is True and body["identity_verified"] is True
    assert body["expected_digest"] == A and body["model_digest"] == A
    assert body["verified_at"] is not None
    assert body["pipeline_id"] == main.registry.models["legacy-768"].pipeline_id("document")


def test_critical_scenario_tags_down_embed_up(clientapp, embed_ok, monkeypatch, logs):
    """§17 : registre connaît A ; /api/tags échoue ; /api/embed fonctionne."""
    _force_verify(monkeypatch, FakeTags(raise_exc=RuntimeError("tags down")))
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "task": "query"})
    assert r.status_code == 200                              # fail-open CONSERVÉ
    body = r.json()
    assert body["governed"] is True
    assert body["identity_verified"] is False
    assert body["expected_digest"] == A
    assert body["model_digest"] is None                      # A n'a PAS été observé
    assert body["pipeline_id"] == main.registry.models["legacy-768"].pipeline_id("query")
    assert embed_ok["inputs"] == ["search_query: a"]         # transformation intacte
    blob = "\n".join(rec.getMessage() for rec in logs.records)
    assert "event=embeddings.identity_unverified" in blob
    assert "registry_status=unavailable" in blob
    assert "a" not in [rec.getMessage() for rec in logs.records if "identity_unverified" in rec.getMessage()][0].split("model_logical")[0].split()  # pas de texte
    from app.metrics import counters
    assert counters.snapshot()["embeddings_identity_unverified_total"] == 1


def test_unknown_response_before_first_check(clientapp, embed_ok, monkeypatch):
    p = main.registry.models["legacy-768"]
    p.status, p.observed_digest, p.verified_at = STATUS_UNKNOWN, None, None
    main._verify_state["last"] = time.monotonic()            # pas de revérification
    body = clientapp.post("/v1/embeddings", json={"texts": ["a"]}).json()
    assert body["governed"] is True and body["identity_verified"] is False
    assert body["expected_digest"] == A and body["model_digest"] is None
    assert body["verified_at"] is None


def test_degraded_response_does_not_mask_divergence(clientapp, embed_ok, monkeypatch):
    _force_verify(monkeypatch, FakeTags([_entry("nomic-embed-text:latest", B)]))
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", False)
    body = clientapp.post("/v1/embeddings", json={"texts": ["a"]}).json()
    assert body["governed"] is False                         # doctrine inchangée
    assert body["identity_verified"] is False
    assert body["expected_digest"] == A
    assert body["model_digest"] == B                         # divergent, visible
    assert body["pipeline_id"] is None                       # doctrine inchangée


def test_out_of_registry_response_invents_no_provenance(clientapp, embed_ok, monkeypatch):
    monkeypatch.setattr(main.settings, "embeddings_strict_registry", False)
    body = clientapp.post("/v1/embeddings", json={"texts": ["a"], "model": "bge-m3"}).json()
    assert body["governed"] is False and body["identity_verified"] is False
    assert body["expected_digest"] is None and body["model_digest"] is None
    assert body["verified_at"] is None and body["pipeline_id"] is None


def test_ready_response_has_no_unverified_event(clientapp, embed_ok, logs):
    clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert "identity_unverified" not in "\n".join(r.getMessage() for r in logs.records)


# --------------------------------------------------------------------------- #
# 4. Healthchecks
# --------------------------------------------------------------------------- #
def test_health_surfaces_expose_verification(clientapp, monkeypatch):
    # `/health` sonde Ollama via `client.ping` (client métier) : doublé.
    monkeypatch.setattr(main.client, "ping", lambda timeout=5: (1, ["nomic-embed-text:latest"]))
    ready = clientapp.get("/health/ready").json()
    assert ready["identity_verified"] is True and ready["verified_at"] is not None
    assert ready["expected_digest"] == ready["observed_digest"] == A
    entry = [m for m in clientapp.get("/health/embeddings").json()["models"]
             if m["model_logical"] == "legacy-768"][0]
    assert entry["identity_verified"] is True and entry["verified_at"] is not None
    assert clientapp.get("/health").json()["registry"]["default_embedding_identity_verified"] is True


def test_health_ready_unavailable_is_unverified_and_keeps_current_availability(clientapp, monkeypatch):
    monkeypatch.setattr(main.probe_client, "list_models",
                        FakeTags(raise_exc=RuntimeError("down")).list_models)
    main._readiness_cache_clear()
    r = clientapp.get("/health/ready")
    body = r.json().get("detail", r.json())
    assert body["registry_status"] == STATUS_UNAVAILABLE
    assert body["identity_verified"] is False and body["observed_digest"] is None
    assert body["expected_digest"] == A
    assert r.status_code == 503                              # disponibilité readiness INCHANGÉE


# --------------------------------------------------------------------------- #
# 5. P1 et transformation intactes
# --------------------------------------------------------------------------- #
def test_pipeline_id_does_not_depend_on_runtime_state():
    p = main.registry.models["legacy-768"]
    baseline = (p.pipeline_id("query"), p.pipeline_id("document"))
    for status, observed in ((STATUS_UNAVAILABLE, None), (STATUS_DEGRADED, B), (STATUS_UNKNOWN, None)):
        p.status, p.observed_digest = status, observed
        assert (p.pipeline_id("query"), p.pipeline_id("document")) == baseline
    assert baseline == ("emb-v2:3a49e14ec5df7088cebb7630fd1d27eb", "emb-v2:9e883f0fe1d326ef17f86027a1d4a5d8")
