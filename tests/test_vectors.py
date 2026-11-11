"""Tests unitaires des contrôles vectoriels."""


import pytest

from app.vectors import (
    NORM_MAX,
    NORM_MIN,
    VectorError,
    is_finite,
    l2_norm,
    l2_normalize,
    norm_in_tolerance,
    validate_vectors,
)


def _unit(dim=4):
    return [1.0] + [0.0] * (dim - 1)


def test_l2_norm():
    assert l2_norm([3.0, 4.0]) == pytest.approx(5.0)


def test_l2_normalize():
    out = l2_normalize([3.0, 4.0])
    assert out == pytest.approx([0.6, 0.8])
    assert l2_norm(out) == pytest.approx(1.0)


def test_l2_normalize_rejects_zero_vector():
    with pytest.raises(VectorError, match="norme"):
        l2_normalize([0.0, 0.0])


def test_l2_normalize_rejects_non_finite():
    with pytest.raises(VectorError):
        l2_normalize([float("nan"), 1.0])


def test_is_finite():
    assert is_finite([1.0, -2.5]) is True
    assert is_finite([1.0, float("inf")]) is False
    assert is_finite([float("nan")]) is False


def test_norm_tolerance_bounds():
    assert norm_in_tolerance(1.0) is True
    assert norm_in_tolerance(NORM_MIN) is True
    assert norm_in_tolerance(NORM_MAX) is True
    assert norm_in_tolerance(NORM_MIN - 1e-6) is False
    assert norm_in_tolerance(19.66) is False
    assert norm_in_tolerance(float("nan")) is False


def test_validate_ok_returns_dimension():
    assert validate_vectors([_unit(768)], expected_count=1, expected_dimension=768) == 768


def test_validate_cardinality():
    with pytest.raises(VectorError, match="nombre de vecteurs"):
        validate_vectors([_unit()], expected_count=2)


def test_validate_dimension():
    with pytest.raises(VectorError, match="dimension inattendue"):
        validate_vectors([_unit(4)], expected_count=1, expected_dimension=768)


def test_validate_heterogeneous():
    with pytest.raises(VectorError, match="vecteur 1 de dimension"):
        validate_vectors([_unit(4), _unit(3)], expected_count=2)


def test_validate_empty_vector():
    with pytest.raises(VectorError, match="vide"):
        validate_vectors([[]], expected_count=1)


def test_validate_nan():
    with pytest.raises(VectorError, match="non finie"):
        validate_vectors([[float("nan"), 1.0]], expected_count=1)


def test_validate_infinity():
    with pytest.raises(VectorError, match="non finie"):
        validate_vectors([[1.0, float("-inf")]], expected_count=1)


def test_validate_zero_vector():
    with pytest.raises(VectorError, match="norme nulle"):
        validate_vectors([[0.0, 0.0]], expected_count=1)


def test_validate_norm_tolerance_when_required():
    with pytest.raises(VectorError, match="hors tolérance"):
        validate_vectors([[19.0, 0.0]], expected_count=1, require_normalized=True)


def test_validate_norm_tolerance_not_checked_by_default():
    assert validate_vectors([[19.0, 0.0]], expected_count=1) == 2


def test_error_messages_never_carry_content():
    """Les messages ne citent que des index et des tailles."""
    try:
        validate_vectors([[0.0]], expected_count=1)
    except VectorError as exc:
        assert "0.0" not in str(exc) or "norme nulle" in str(exc)
