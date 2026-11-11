"""Lot P17 — un score rerank non fini est une réponse backend FAUTIVE.

Découvert pendant une version antérieure : un score NaN atteignait la sérialisation JSON
(`ValueError: Out of range float values are not JSON compliant`) et sortait en
500 non maîtrisé, alors que la doc annonce un 502 contrôlé. Le client
transport validait déjà la finitude, mais la route est le point de passage
obligé quel que soit le backend branché : l'invariant est tenu là.

Aucun clamp, aucun 0 de remplacement, aucun filtrage du seul élément fautif :
un score invalide rend toute la réponse invalide.
"""

import logging
import math

import pytest

from app import main
from app.metrics import counters, gauges
from tests.test_rerank import _post

SECRET_DOC = "TEXTE_UTILISATEUR_SECRET_9f8e7d"


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.setattr(main.settings, "rerank_enabled", True)
    monkeypatch.setattr(main.settings, "environment", "staging")
    gauges.reset()


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _scores(values):
    return lambda q, t: [{"index": i, "score": v} for i, v in enumerate(values)]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "+inf", "-inf"])
def test_non_finite_single_score_is_controlled_502(clientapp, staging, monkeypatch, logs, bad):
    monkeypatch.setattr(main.rerank_client, "rerank", _scores([bad, 0.2, 0.1]))
    # raise_server_exceptions=True : un ValueError de sérialisation ou une
    # ExceptionGroup ferait ÉCHOUER ce test ici même.
    r = _post(clientapp)
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["provider"] == "local_cross_encoder"
    assert d["message"] == "score de reranking invalide (valeur non finie)"
    assert "nan" not in r.text.lower() or "non finie" in r.text   # jamais sérialisé
    assert counters.snapshot()["rerank_upstream_errors_total"] == 1
    assert gauges.total() == 0                                       # permit libéré

    # Le rerank suivant est accepté immédiatement, scores finis inchangés.
    monkeypatch.setattr(main.rerank_client, "rerank", _scores([0.5, 0.9, 0.1]))
    r = _post(clientapp)
    assert r.status_code == 200
    assert [x["relevance_score"] for x in r.json()["results"]] == [0.9, 0.5, 0.1]
    assert gauges.total() == 0


@pytest.mark.parametrize("values", [
    [0.8, math.nan, 0.3],
    [0.8, 0.3, math.inf],
    [-math.inf, 0.8, 0.3],
], ids=["nan-middle", "inf-last", "-inf-first"])
def test_one_bad_score_rejects_the_whole_response(clientapp, staging, monkeypatch, values):
    """Pas de filtrage : la réponse entière est refusée, aucun classement
    partiel des scores restants."""
    monkeypatch.setattr(main.rerank_client, "rerank", _scores(values))
    r = _post(clientapp)
    assert r.status_code == 502
    assert "results" not in r.json()


def test_finite_scores_unchanged(clientapp, staging, monkeypatch):
    monkeypatch.setattr(main.rerank_client, "rerank", _scores([0.5, 0.75, -1.25]))
    r = _post(clientapp)
    assert r.status_code == 200
    assert [x["relevance_score"] for x in r.json()["results"]] == [0.75, 0.5, -1.25]


def test_non_numeric_score_is_502_too(clientapp, staging, monkeypatch):
    monkeypatch.setattr(main.rerank_client, "rerank", _scores(["0.9", 0.2, 0.1]))
    assert _post(clientapp).status_code == 502
    monkeypatch.setattr(main.rerank_client, "rerank", _scores([True, 0.2, 0.1]))
    assert _post(clientapp).status_code == 502


def test_invalid_score_logs_index_not_content(clientapp, staging, monkeypatch, logs):
    monkeypatch.setattr(main.rerank_client, "rerank", _scores([0.2, math.nan, 0.1]))
    r = _post(clientapp, documents=[{"id": f"d{i}", "text": SECRET_DOC + str(i)} for i in range(3)])
    assert r.status_code == 502
    blob = "\n".join(rec.getMessage() for rec in logs.records)
    assert "event=rerank.invalid_score" in blob and "index=1" in blob
    assert SECRET_DOC not in blob and SECRET_DOC not in r.text
