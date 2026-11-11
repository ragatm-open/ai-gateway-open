"""Lot RUNTIME-INFLIGHT : `inflight` = requêtes MÉTIER en cours (filtre positif).

La jauge interne `host_admission_reaper_pending` (file de récupération du client
d'admission hôte) et toute jauge interne future sont exclues par construction ;
le reaper reste observable dans `host_admission.reaper_pending`."""

import pytest

from app import metrics
from app.metrics import (BUSINESS_INFLIGHT_PREFIXES, HOST_ADMISSION_REAPER_PENDING_GAUGE, business_inflight, gauges,
                         is_business_inflight_gauge)


@pytest.fixture(autouse=True)
def _clean_gauges(monkeypatch):
    monkeypatch.setattr(metrics.gauges, "_values", {})
    yield


def _set(name, n):
    for _ in range(n):
        gauges.inc(name)


def _runtime(clientapp):
    return clientapp.get("/health/runtime").json()


def test_zero_state(clientapp):
    body = _runtime(clientapp)
    assert body["inflight"] == {"total": 0, "by_model": {}}
    assert body["host_admission"] == {"reaper_pending": 0}


def test_single_business_request(clientapp):
    _set("vision:vision-large", 1)
    assert _runtime(clientapp)["inflight"] == {"total": 1, "by_model": {"vision:vision-large": 1}}


def test_reaper_only_is_not_inflight(clientapp):
    _set(HOST_ADMISSION_REAPER_PENDING_GAUGE, 3)
    body = _runtime(clientapp)
    # contrat RagATM ensure_quiet et préflight de promotion : aucune requête métier => 0
    assert body["inflight"] == {"total": 0, "by_model": {}}
    assert body["host_admission"] == {"reaper_pending": 3}


def test_mixed_business_and_reaper(clientapp):
    _set("vision:vision-large", 1); _set("chat:model-x", 2); _set(HOST_ADMISSION_REAPER_PENDING_GAUGE, 3)
    body = _runtime(clientapp)
    assert body["inflight"]["total"] == 3
    assert body["inflight"]["by_model"] == {"vision:vision-large": 1, "chat:model-x": 2}
    assert body["host_admission"]["reaper_pending"] == 3


def test_unknown_internal_gauge_excluded_by_construction(clientapp):
    _set("vision:vision-large", 1); _set("foo_internal", 7); _set("host_admission_something_new", 2)
    body = _runtime(clientapp)
    assert body["inflight"] == {"total": 1, "by_model": {"vision:vision-large": 1}}


def test_asr_admission_is_business_inflight(clientapp):
    _set("asr:admission", 1)
    assert _runtime(clientapp)["inflight"] == {"total": 1, "by_model": {"asr:admission": 1}}


@pytest.mark.parametrize("prefix", ["generate:", "generate_stream:", "chat:", "chat_stream:",
                                    "embeddings:", "rerank:", "vision:", "asr:"])
def test_every_registered_prefix_counts(clientapp, prefix):
    assert prefix in BUSINESS_INFLIGHT_PREFIXES
    _set(prefix + "m", 2)
    assert _runtime(clientapp)["inflight"] == {"total": 2, "by_model": {prefix + "m": 2}}


@pytest.mark.parametrize("name", ["host_admission_reaper_pending", "foo_internal", "queued:vision",
                                  "visionary:x", "generate", "chat-x:y", "reaper:vision", ""])
def test_non_registered_names_never_count(name):
    assert is_business_inflight_gauge(name) is False
    assert business_inflight({name: 5}) == {}


def test_queued_semantics_unchanged(clientapp):
    _set("queued:rerank-default", 2); _set("rerank:rerank-default", 1)
    body = _runtime(clientapp)
    assert body["queued"] == {"total": 2, "by_model": {"rerank-default": 2}}
    assert body["inflight"] == {"total": 1, "by_model": {"rerank:rerank-default": 1}}


def test_runtime_schema_only_additive(clientapp):
    body = _runtime(clientapp)
    assert set(body["inflight"]) == {"total", "by_model"} and set(body["queued"]) == {"total", "by_model"}
    assert set(body["host_admission"]) == {"reaper_pending"} and isinstance(body["host_admission"]["reaper_pending"], int)


def test_health_and_ready_unaffected_by_reaper(clientapp, monkeypatch):
    from app import main
    monkeypatch.setattr(main.client, "ping", lambda timeout=5: (True, ["nomic-embed-text:latest"]))
    before = (clientapp.get("/health").status_code, clientapp.get("/health/ready").status_code)
    _set(HOST_ADMISSION_REAPER_PENDING_GAUGE, 5)
    assert (clientapp.get("/health").status_code, clientapp.get("/health/ready").status_code) == before
