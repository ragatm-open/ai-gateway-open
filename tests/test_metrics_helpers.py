"""Lot 3 — helpers de conversion et de normalisation des métriques Ollama.

Invariant testé ici : **aucun helper ne lève**. Une métrique d'observabilité
mal formée doit disparaître de la ligne de log, jamais transformer une réponse
LLM valide en 500 (cf. app/metrics.py). Les tests portent donc sur le contrat
— valeur exploitable / absence — et non sur la mécanique interne.
"""

import math

import pytest

from app.metrics import (
    duration_ms,
    embedding_throughput,
    normalize_embedding_metrics,
    normalize_generation_metrics,
    rate_per_second,
    safe_int,
)

# Valeurs qu'un runtime Ollama ne devrait jamais produire, mais qui ne doivent
# en aucun cas casser la requête : types inattendus, non-finis, négatifs.
INVALID_VALUES = [
    None,
    "1500",
    "",
    True,
    False,
    [1],
    {},
    object(),
    float("nan"),
    float("inf"),
    float("-inf"),
    -1,
    -0.5,
]


# --------------------------------------------------------------- duration_ms
def test_duration_ms_converts_nanoseconds():
    assert duration_ms(1_500_000_000) == 1500.0
    assert duration_ms(200_000_000) == 200.0


def test_duration_ms_accepts_zero():
    # 0 est une mesure légitime : un modèle déjà résident a `load_duration` nul.
    # C'est précisément l'information recherchée pour décider d'un warm-up.
    assert duration_ms(0) == 0.0


def test_duration_ms_keeps_sub_millisecond_precision():
    assert duration_ms(1_234_567) == 1.235


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_duration_ms_rejects_invalid_values(value):
    assert duration_ms(value) is None


# ------------------------------------------------------------ rate_per_second
def test_rate_per_second_nominal():
    # 50 tokens en 1 s (1e9 ns).
    assert rate_per_second(50, 1_000_000_000) == 50.0
    # 10 tokens en 200 ms.
    assert rate_per_second(10, 200_000_000) == 50.0


def test_rate_per_second_zero_duration_is_omitted():
    # Le cœur de la garde : jamais de division par zéro, jamais d'Infinity.
    assert rate_per_second(50, 0) is None


def test_rate_per_second_zero_count_is_a_real_rate():
    # 0 token en 1 s vaut bien 0 tok/s — ce n'est pas une donnée manquante.
    assert rate_per_second(0, 1_000_000_000) == 0.0


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_rate_per_second_rejects_invalid_count(value):
    assert rate_per_second(value, 1_000_000_000) is None


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_rate_per_second_rejects_invalid_duration(value):
    assert rate_per_second(50, value) is None


def test_rate_per_second_never_produces_nan_or_inf():
    for count in INVALID_VALUES + [0, 50]:
        for duration in INVALID_VALUES + [0, 1_000_000_000]:
            result = rate_per_second(count, duration)
            assert result is None or math.isfinite(result)


# ------------------------------------------------------------------ safe_int
def test_safe_int_nominal():
    assert safe_int(42) == 42
    assert safe_int(0) == 0
    assert safe_int(8192.0) == 8192


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_safe_int_rejects_invalid_values(value):
    assert safe_int(value) is None


# ------------------------------------------------ normalize_generation_metrics
def test_normalize_generation_metrics_full_payload():
    metrics = normalize_generation_metrics(
        {
            "response": "peu importe",
            "total_duration": 1_500_000_000,
            "load_duration": 300_000_000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200_000_000,
            "eval_count": 50,
            "eval_duration": 1_000_000_000,
        }
    )
    assert metrics == {
        "ollama_total_ms": 1500.0,
        "ollama_load_ms": 300.0,
        "ollama_prompt_eval_ms": 200.0,
        "ollama_eval_ms": 1000.0,
        "prompt_tokens": 10,
        "completion_tokens": 50,
        "prompt_tokens_per_second": 50.0,
        "completion_tokens_per_second": 50.0,
    }


def test_normalize_generation_metrics_never_copies_content():
    # Garde-fou de sécurité : le normaliseur lit une liste FERMÉE de champs
    # numériques. Même si l'upstream plaçait du texte dans sa réponse, il ne
    # peut pas ressortir par ici et atterrir dans un log.
    metrics = normalize_generation_metrics(
        {
            "response": "SECRET_REPONSE",
            "prompt": "SECRET_PROMPT",
            "context": [1, 2, 3],
            "total_duration": 1_000_000,
        }
    )
    assert metrics == {"ollama_total_ms": 1.0}


def test_normalize_generation_metrics_partial_payload_omits_absent_keys():
    # Runtime Ollama plus ancien : seuls les comptes sont là, pas les durées.
    metrics = normalize_generation_metrics({"prompt_eval_count": 7, "eval_count": 3})
    assert metrics == {"prompt_tokens": 7, "completion_tokens": 3}


def test_normalize_generation_metrics_drops_invalid_fields_only():
    # Une durée corrompue ne doit pas emporter les métriques voisines valides.
    metrics = normalize_generation_metrics(
        {"total_duration": "beaucoup", "eval_count": 12, "eval_duration": 2_000_000_000}
    )
    assert metrics == {
        "completion_tokens": 12,
        "ollama_eval_ms": 2000.0,
        "completion_tokens_per_second": 6.0,
    }


@pytest.mark.parametrize("raw", [None, "texte", 42, [], object()])
def test_normalize_generation_metrics_tolerates_non_dict(raw):
    assert normalize_generation_metrics(raw) == {}


# ------------------------------------------------- normalize_embedding_metrics
def test_normalize_embedding_metrics_full_payload():
    metrics = normalize_embedding_metrics(
        {
            "total_duration": 45_000_000,
            "load_duration": 12_000_000,
            "prompt_eval_count": 128,
        }
    )
    assert metrics == {
        "ollama_total_ms": 45.0,
        "ollama_load_ms": 12.0,
        "ollama_prompt_eval_tokens": 128,
    }


def test_normalize_embedding_metrics_empty_metadata():
    # Chemin de repli /api/embeddings : aucune métadonnée, et c'est normal.
    assert normalize_embedding_metrics({}) == {}


@pytest.mark.parametrize("raw", [None, "texte", 42, [], object()])
def test_normalize_embedding_metrics_tolerates_non_dict(raw):
    assert normalize_embedding_metrics(raw) == {}


# ------------------------------------------------------- embedding_throughput
def test_embedding_throughput_nominal():
    assert embedding_throughput(8, 200) == {"ms_per_text": 25.0, "texts_per_second": 40.0}


def test_embedding_throughput_zero_elapsed_keeps_only_ms_per_text():
    # Mesure murale nulle (lot minuscule, upstream simulé) : `texts_per_second`
    # serait infini, il est donc omis ; `ms_per_text` reste exact.
    assert embedding_throughput(4, 0) == {"ms_per_text": 0.0}


def test_embedding_throughput_zero_texts_yields_nothing():
    assert embedding_throughput(0, 100) == {}


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_embedding_throughput_rejects_invalid_elapsed(value):
    assert embedding_throughput(8, value) == {}


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_embedding_throughput_rejects_invalid_count(value):
    assert embedding_throughput(value, 200) == {}
