"""Lot 3 — observabilité des performances Ollama (génération et embeddings).

Ces tests portent sur trois contrats :

1. les durées natives d'Ollama, quand elles existent, arrivent dans les
   événements structurés `generate.ok` / `embeddings.ok` ;
2. leur absence, leur nullité ou leur corruption n'altèrent **jamais** la
   réponse HTTP — une métrique ratée ne casse pas une génération réussie ;
3. aucun contenu (prompt, système, réponse, texte d'entrée, vecteur) n'entre
   dans les logs, invariant hérité du contrat de sécurité du dépôt.
"""

import logging
import math
import time

import pytest
import requests

from app import main
from app.ollama_client import ENDPOINT_EMBED, ENDPOINT_EMBEDDINGS, EmbedResult, OllamaClient

# Chaînes volontairement improbables : si l'une apparaît dans un log, la fuite
# est certaine et le test la nomme sans ambiguïté.
PROMPT_SENTINEL = "ZZTOPSECRETPROMPTZZ"
SYSTEM_SENTINEL = "ZZTOPSECRETSYSTEMZZ"
RESPONSE_SENTINEL = "ZZTOPSECRETRESPONSEZZ"
TEXT_SENTINEL = "ZZTOPSECRETTEXTEZZ"


def _vec(dim=768):
    return [1.0] + [0.0] * (dim - 1)


@pytest.fixture
def logs(caplog):
    """Capture les événements structurés émis par la Gateway."""
    caplog.set_level(logging.INFO, logger="ai_gateway")
    return caplog


def _event(caplog, name):
    """Retourne la ligne `event=<name> ...`, ou échoue si elle manque."""
    prefix = f"event={name} "
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(prefix):
            return message
    raise AssertionError(
        f"événement {name!r} absent des logs : "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def _fields(line):
    """Découpe `event=x a=1 b=2` en dict {'a': '1', 'b': '2'}."""
    parts = line.split(" ")[1:]
    return dict(part.split("=", 1) for part in parts if "=" in part)


def _all_logs(caplog):
    return "\n".join(record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- #
# Génération
# --------------------------------------------------------------------------- #
def _fake_generate(payload):
    def fake(model, prompt, system=None, temperature=None, options=None,
             think=None, keep_alive=None):
        return payload

    return fake


FULL_GENERATE_PAYLOAD = {
    "response": "OK",
    "done": True,
    "total_duration": 1_500_000_000,   # 1500 ms
    "load_duration": 300_000_000,      #  300 ms
    "prompt_eval_count": 10,
    "prompt_eval_duration": 200_000_000,   # 200 ms -> 50 tok/s
    "eval_count": 50,
    "eval_duration": 1_000_000_000,        # 1000 ms -> 50 tok/s
}


def test_generate_logs_native_ollama_durations(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.client, "generate", _fake_generate(FULL_GENERATE_PAYLOAD))
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "llama3.1:8b"})
    assert r.status_code == 200

    fields = _fields(_event(logs, "generate.ok"))
    assert fields["model"] == "llama3.1:8b"
    assert fields["ollama_total_ms"] == "1500.0"
    assert fields["ollama_load_ms"] == "300.0"
    assert fields["ollama_prompt_eval_ms"] == "200.0"
    assert fields["ollama_eval_ms"] == "1000.0"
    assert fields["prompt_tokens"] == "10"
    assert fields["completion_tokens"] == "50"
    # Mesures murales Gateway : présentes et numériques, quelle que soit leur
    # valeur (l'upstream simulé est instantané).
    assert fields["upstream_ms"].isdigit()
    assert fields["total_ms"].isdigit()


def test_generate_logs_token_rates(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.client, "generate", _fake_generate(FULL_GENERATE_PAYLOAD))
    clientapp.post("/v1/generate", json={"prompt": "x"})

    fields = _fields(_event(logs, "generate.ok"))
    assert fields["prompt_tokens_per_second"] == "50.0"
    assert fields["completion_tokens_per_second"] == "50.0"


def test_generate_missing_native_metrics_still_succeeds(clientapp, monkeypatch, logs):
    # Runtime Ollama ne publiant aucune durée : la requête reste un succès et
    # l'événement est émis, simplement plus pauvre.
    monkeypatch.setattr(
        main.client, "generate", _fake_generate({"response": "OK", "done": True})
    )
    r = clientapp.post("/v1/generate", json={"prompt": "x"})
    assert r.status_code == 200
    assert r.json()["text"] == "OK"

    fields = _fields(_event(logs, "generate.ok"))
    assert "total_ms" in fields
    for absent in (
        "ollama_total_ms",
        "ollama_load_ms",
        "ollama_prompt_eval_ms",
        "ollama_eval_ms",
        "prompt_tokens_per_second",
        "completion_tokens_per_second",
    ):
        assert absent not in fields


def test_generate_zero_duration_does_not_divide_by_zero(clientapp, monkeypatch, logs):
    # Génération si rapide qu'Ollama rapporte des durées nulles : les débits
    # sont omis plutôt que portés à l'infini.
    monkeypatch.setattr(
        main.client,
        "generate",
        _fake_generate(
            {
                "response": "OK",
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 0,
                "eval_count": 50,
                "eval_duration": 0,
            }
        ),
    )
    r = clientapp.post("/v1/generate", json={"prompt": "x"})
    assert r.status_code == 200

    line = _event(logs, "generate.ok")
    fields = _fields(line)
    assert fields["ollama_total_ms"] == "0.0"
    assert "prompt_tokens_per_second" not in fields
    assert "completion_tokens_per_second" not in fields
    for poison in ("inf", "Infinity", "nan", "NaN"):
        assert poison not in line


def test_generate_invalid_native_metrics_still_succeeds(clientapp, monkeypatch, logs):
    # Télémétrie corrompue (types inattendus) : aucun cast aveugle, aucun 500.
    monkeypatch.setattr(
        main.client,
        "generate",
        _fake_generate(
            {
                "response": "OK",
                "done": True,
                "total_duration": "beaucoup",
                "load_duration": None,
                "eval_count": 12,
                "eval_duration": 2_000_000_000,
            }
        ),
    )
    r = clientapp.post("/v1/generate", json={"prompt": "x"})
    assert r.status_code == 200

    fields = _fields(_event(logs, "generate.ok"))
    assert "ollama_total_ms" not in fields
    assert "ollama_load_ms" not in fields
    # La métrique voisine valide, elle, survit.
    assert fields["ollama_eval_ms"] == "2000.0"
    assert fields["completion_tokens_per_second"] == "6.0"


def test_generate_logs_num_ctx_when_client_provides_it(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.client, "generate", _fake_generate(FULL_GENERATE_PAYLOAD))
    clientapp.post("/v1/generate", json={"prompt": "x", "options": {"num_ctx": 8192}})
    assert _fields(_event(logs, "generate.ok"))["num_ctx"] == "8192"


def test_generate_does_not_invent_num_ctx(clientapp, monkeypatch, logs):
    # Sans `num_ctx` explicite, la Gateway n'en fabrique pas : Ollama applique
    # le défaut du modèle, que la Gateway ne connaît pas.
    monkeypatch.setattr(main.client, "generate", _fake_generate(FULL_GENERATE_PAYLOAD))
    clientapp.post("/v1/generate", json={"prompt": "x"})
    assert "num_ctx" not in _fields(_event(logs, "generate.ok"))

    logs.clear()
    clientapp.post("/v1/generate", json={"prompt": "x", "options": {"temperature": 0.1}})
    assert "num_ctx" not in _fields(_event(logs, "generate.ok"))


def test_generate_rejects_non_numeric_num_ctx_and_never_logs_it(clientapp, monkeypatch, logs):
    # Une valeur arbitraire venue du payload client n'entre pas dans un log.
    # Lot P4 : elle n'est plus ignorée en silence (défaut du profil appliqué)
    # mais refusée en 400 — et toujours absente des logs, corps d'erreur inclus.
    calls = []
    monkeypatch.setattr(main.client, "generate", lambda **kw: calls.append(kw))
    r = clientapp.post(
        "/v1/generate",
        json={"prompt": "x", "options": {"num_ctx": TEXT_SENTINEL}},
    )
    assert r.status_code == 400
    assert calls == []                          # Ollama jamais appelé
    assert TEXT_SENTINEL not in r.text          # ni renvoyée au client
    assert "event=generate.ok" not in _all_logs(logs)
    assert TEXT_SENTINEL not in _all_logs(logs)


def test_generate_no_event_when_upstream_fails(clientapp, monkeypatch, logs):
    # Le lot ne touche pas au chemin d'erreur : pas d'événement de succès, et
    # la classification 502 historique est préservée.
    from app.errors import UpstreamError

    def boom(**kwargs):
        raise UpstreamError("ollama down")

    monkeypatch.setattr(main.client, "generate", boom)
    r = clientapp.post("/v1/generate", json={"prompt": "x"})
    assert r.status_code == 502
    assert "event=generate.ok" not in _all_logs(logs)


# ----------------------------------------------------------- non-fuite (LLM)
def test_generate_logs_do_not_contain_prompt(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.client, "generate", _fake_generate(FULL_GENERATE_PAYLOAD))
    r = clientapp.post(
        "/v1/generate",
        json={"prompt": PROMPT_SENTINEL, "system": SYSTEM_SENTINEL},
    )
    assert r.status_code == 200

    everything = _all_logs(logs)
    assert PROMPT_SENTINEL not in everything
    assert SYSTEM_SENTINEL not in everything
    # La LONGUEUR du prompt reste journalisée : c'est le contrat historique.
    assert f"prompt_len={len(PROMPT_SENTINEL)}" in everything


def test_generate_logs_do_not_contain_response(clientapp, monkeypatch, logs):
    payload = dict(FULL_GENERATE_PAYLOAD, response=RESPONSE_SENTINEL)
    monkeypatch.setattr(main.client, "generate", _fake_generate(payload))
    r = clientapp.post("/v1/generate", json={"prompt": "x"})
    assert r.status_code == 200
    # La réponse est bien servie au client…
    assert r.json()["text"] == RESPONSE_SENTINEL
    # …et absente des logs.
    assert RESPONSE_SENTINEL not in _all_logs(logs)


def test_generate_logs_do_not_contain_ollama_context_window(clientapp, monkeypatch, logs):
    # `context` d'Ollama est l'état de tokenisation de la conversation : il
    # encode le prompt et ne doit jamais transiter par un log.
    payload = dict(FULL_GENERATE_PAYLOAD, context=[128009, 271, 5618])
    monkeypatch.setattr(main.client, "generate", _fake_generate(payload))
    clientapp.post("/v1/generate", json={"prompt": "x"})
    assert "128009" not in _all_logs(logs)


# ------------------------------------------------------ contrat inchangé (LLM)
def test_generate_contract_remains_backward_compatible(clientapp, monkeypatch):
    monkeypatch.setattr(main.client, "generate", _fake_generate(FULL_GENERATE_PAYLOAD))
    r = clientapp.post("/v1/generate", json={"prompt": "x", "model": "llama3.1:8b"})
    assert r.status_code == 200
    body = r.json()

    # Ni champ en moins, ni champ en plus : le lot 3 n'expose AUCUNE métrique
    # dans la réponse HTTP (décision documentée — cf. doc/04-api.md).
    # `done_reason` (lot workload-isolation) n'est pas une métrique : c'est la
    # sémantique d'arrêt demandée par RagATM, relayée verbatim d'Ollama.
    assert set(body) == {
        "text", "provider", "upstream_provider", "model",
        "tokens", "usage", "cost", "done", "done_reason",
    }
    assert body["text"] == "OK"
    assert body["provider"] == "ai_gateway"
    assert body["upstream_provider"] == "ollama"
    assert body["model"] == "llama3.1:8b"
    assert body["tokens"] == {"prompt": 10, "completion": 50, "total": 60}
    assert body["usage"] == {
        "prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60,
    }
    assert body["cost"] == 0.0
    assert body["done"] is True


def test_generate_missing_counts_still_report_zero_tokens(clientapp, monkeypatch):
    # Comportement historique : un compte absent vaut 0 dans la RÉPONSE, alors
    # qu'il est omis dans les LOGS. Les deux règles coexistent volontairement.
    monkeypatch.setattr(
        main.client, "generate", _fake_generate({"response": "OK", "done": True})
    )
    body = clientapp.post("/v1/generate", json={"prompt": "x"}).json()
    assert body["tokens"] == {"prompt": 0, "completion": 0, "total": 0}


# --------------------------------------------------------------------------- #
# Embeddings — transport des métriques par EmbedResult
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error", response=self)

    def json(self):
        return self._payload


def test_embed_result_preserves_native_metrics(monkeypatch):
    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None: _Resp(
            {
                "embeddings": [[1.0, 0.0]],
                "total_duration": 45_000_000,
                "load_duration": 12_000_000,
                "prompt_eval_count": 128,
            }
        ),
    )
    result = OllamaClient("http://127.0.0.1:11434", timeout=30).embed(
        model="m", inputs=["a"]
    )
    assert result.upstream_endpoint == ENDPOINT_EMBED
    assert result.ollama_metadata == {
        "total_duration": 45_000_000,
        "load_duration": 12_000_000,
        "prompt_eval_count": 128,
    }
    # Le comportement de liste est intact : c'est tout le contrat d'EmbedResult.
    assert list(result) == [[1.0, 0.0]]


def test_embed_result_metadata_is_a_closed_allowlist(monkeypatch):
    # La réponse upstream n'est jamais recopiée en bloc : un champ inattendu
    # (ici du texte) ne peut pas se retrouver embarqué vers les logs.
    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None: _Resp(
            {
                "embeddings": [[1.0, 0.0]],
                "total_duration": 45_000_000,
                "input": TEXT_SENTINEL,
                "model_dump": TEXT_SENTINEL,
            }
        ),
    )
    result = OllamaClient("http://127.0.0.1:11434", timeout=30).embed(
        model="m", inputs=["a"]
    )
    assert result.ollama_metadata == {"total_duration": 45_000_000}


def test_embed_result_metadata_is_empty_without_metrics(monkeypatch):
    monkeypatch.setattr(
        "app.ollama_client.requests.post",
        lambda url, json=None, timeout=None: _Resp({"embeddings": [[1.0, 0.0]]}),
    )
    result = OllamaClient("http://127.0.0.1:11434", timeout=30).embed(
        model="m", inputs=["a"]
    )
    assert result.ollama_metadata == {}


def test_embed_result_fallback_reports_no_metrics(monkeypatch):
    # `/api/embeddings` ne publie pas de durées : on n'en invente pas, et on ne
    # somme surtout pas les appels unitaires pour fabriquer une durée de lot.
    def fake_post(url, json=None, timeout=None):
        if url.endswith(ENDPOINT_EMBED):
            return _Resp({}, status=404, text="404 page not found")
        return _Resp({"embedding": [3.0, 4.0]})

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)
    result = OllamaClient("http://127.0.0.1:11434", timeout=30).embed(
        model="m", inputs=["a", "b"]
    )
    assert result.upstream_endpoint == ENDPOINT_EMBEDDINGS
    assert result.fallback_used is True
    assert result.ollama_metadata == {}


# --------------------------------------------------------------------------- #
# Embeddings — événement enrichi
# --------------------------------------------------------------------------- #
def _fake_embed(metadata=None, endpoint=ENDPOINT_EMBED, delay=0.0):
    def fake(model, inputs, keep_alive=None):
        if delay:
            time.sleep(delay)
        return EmbedResult(
            [_vec() for _ in inputs], endpoint, ollama_metadata=metadata
        )

    return fake


def test_embeddings_logs_native_ollama_durations(clientapp, monkeypatch, logs):
    monkeypatch.setattr(
        main.client,
        "embed",
        _fake_embed(
            {
                "total_duration": 45_000_000,
                "load_duration": 12_000_000,
                "prompt_eval_count": 128,
            }
        ),
    )
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b"]})
    assert r.status_code == 200

    fields = _fields(_event(logs, "embeddings.ok"))
    assert fields["ollama_total_ms"] == "45.0"
    assert fields["ollama_load_ms"] == "12.0"
    assert fields["ollama_prompt_eval_tokens"] == "128"
    # Les champs historiques de l'événement restent en place.
    assert fields["n_texts"] == "2"
    assert fields["dimension"] == "768"
    assert fields["upstream_endpoint"] == ENDPOINT_EMBED
    assert "upstream_ms" in fields
    assert "total_ms" in fields


def test_embeddings_logs_throughput(clientapp, monkeypatch, logs):
    # Un délai réel garantit une mesure murale non nulle, donc un débit
    # calculable — c'est la seule façon d'observer `texts_per_second`.
    monkeypatch.setattr(main.client, "embed", _fake_embed(delay=0.02))
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b", "c", "d"]})
    assert r.status_code == 200

    fields = _fields(_event(logs, "embeddings.ok"))
    assert float(fields["ms_per_text"]) > 0
    assert float(fields["texts_per_second"]) > 0
    assert math.isfinite(float(fields["texts_per_second"]))


def test_embeddings_missing_native_metrics_still_succeeds(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.client, "embed", _fake_embed(metadata=None))
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200
    assert len(r.json()["embeddings"]) == 1

    fields = _fields(_event(logs, "embeddings.ok"))
    assert "ollama_total_ms" not in fields
    assert "ollama_load_ms" not in fields
    assert "ollama_prompt_eval_tokens" not in fields
    # Les mesures murales Gateway, elles, restent disponibles.
    assert "upstream_ms" in fields
    assert "ms_per_text" in fields


def test_embeddings_plain_list_upstream_still_succeeds(clientapp, monkeypatch, logs):
    # Un double de test (ou un futur chemin) qui retourne une liste nue, sans
    # les attributs d'EmbedResult, ne doit pas faire tomber la route.
    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: [_vec() for _ in inputs],
    )
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200
    assert "ollama_total_ms" not in _fields(_event(logs, "embeddings.ok"))


def test_embeddings_fallback_without_metrics_still_succeeds(clientapp, monkeypatch, logs):
    monkeypatch.setattr(
        main.client, "embed", _fake_embed(metadata={}, endpoint=ENDPOINT_EMBEDDINGS)
    )
    r = clientapp.post("/v1/embeddings", json={"texts": ["a", "b"]})
    assert r.status_code == 200

    fields = _fields(_event(logs, "embeddings.ok"))
    # L'endpoint reste la clé qui distingue les deux régimes de mesure.
    assert fields["upstream_endpoint"] == ENDPOINT_EMBEDDINGS
    assert "ollama_total_ms" not in fields
    assert "ms_per_text" in fields


def test_embeddings_invalid_native_metrics_still_succeeds(clientapp, monkeypatch, logs):
    monkeypatch.setattr(
        main.client,
        "embed",
        _fake_embed({"total_duration": "beaucoup", "load_duration": 12_000_000}),
    )
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200

    fields = _fields(_event(logs, "embeddings.ok"))
    assert "ollama_total_ms" not in fields
    assert fields["ollama_load_ms"] == "12.0"


# ---------------------------------------------------- non-fuite (embeddings)
def test_embeddings_logs_do_not_contain_input_text(clientapp, monkeypatch, logs):
    monkeypatch.setattr(main.client, "embed", _fake_embed({"total_duration": 45_000_000}))
    r = clientapp.post("/v1/embeddings", json={"texts": [TEXT_SENTINEL]})
    assert r.status_code == 200

    everything = _all_logs(logs)
    assert TEXT_SENTINEL not in everything
    # Le préfixe posé par la Gateway ne doit pas non plus entraîner de fuite.
    assert "search_document: " not in everything


def test_embeddings_logs_do_not_contain_vectors(clientapp, monkeypatch, logs):
    marker = 0.123456789
    vector = [marker] + [0.0] * 767

    def fake(model, inputs, keep_alive=None):
        # Vecteur non normalisé volontairement écarté : on veut un 200.
        norm = sum(x * x for x in vector) ** 0.5
        return EmbedResult(
            [[x / norm for x in vector] for _ in inputs],
            ENDPOINT_EMBED,
            ollama_metadata={"total_duration": 45_000_000},
        )

    monkeypatch.setattr(main.client, "embed", fake)
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"]})
    assert r.status_code == 200
    assert str(marker) not in _all_logs(logs)


# ---------------------------------------------- contrat inchangé (embeddings)
def test_embeddings_contract_remains_backward_compatible(clientapp, monkeypatch):
    monkeypatch.setattr(
        main.client,
        "embed",
        _fake_embed(
            {
                "total_duration": 45_000_000,
                "load_duration": 12_000_000,
                "prompt_eval_count": 128,
            }
        ),
    )
    r = clientapp.post(
        "/v1/embeddings",
        json={"texts": ["a", "b"], "model": "nomic-embed-text", "task": "query"},
    )
    assert r.status_code == 200
    body = r.json()

    # Aucune métrique n'entre dans la réponse : le jeu de champs est exactement
    # celui d'avant le lot 3, plus les trois champs ADDITIFS d'une version antérieure
    # (`expected_digest`, `identity_verified`, `verified_at`) — aucun champ
    # historique retiré ni renommé.
    assert set(body) == {
        "provider", "upstream_provider", "model", "dimension", "embeddings",
        "usage", "cost", "governed", "model_logical", "model_digest",
        "pipeline_id", "task", "normalized", "truncated", "upstream_endpoint",
        "expected_digest", "identity_verified", "verified_at",
    }
    assert body["model"] == "nomic-embed-text"
    assert body["dimension"] == 768
    assert len(body["embeddings"]) == 2
    assert body["usage"] == {"total_texts": 2}
    assert body["cost"] == 0.0
    assert body["governed"] is True
    assert body["model_logical"] == "legacy-768"
    assert body["pipeline_id"] == main.registry.models["legacy-768"].pipeline_id("query")
    assert body["task"] == "query"
    assert body["normalized"] is True
    assert body["truncated"] == [False, False]
    assert body["upstream_endpoint"] == ENDPOINT_EMBED


def test_embeddings_vector_order_and_values_untouched(clientapp, monkeypatch):
    # Le lot n'observe que le temps : les vecteurs sortent tels qu'ils entrent.
    vectors = [
        [1.0] + [0.0] * 767,
        [0.0, 1.0] + [0.0] * 766,
        [0.0, 0.0, 1.0] + [0.0] * 765,
    ]
    monkeypatch.setattr(
        main.client,
        "embed",
        lambda model, inputs, keep_alive=None: EmbedResult(
            vectors, ENDPOINT_EMBED, ollama_metadata={"total_duration": 1}
        ),
    )
    body = clientapp.post("/v1/embeddings", json={"texts": ["a", "b", "c"]}).json()
    assert body["embeddings"] == vectors
