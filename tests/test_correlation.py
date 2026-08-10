"""Correlated random-variable generation and correlation-matrix validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.core.exceptions import VariationError
from siliconstat.variation.correlation import (
    CorrelationSpec,
    apply_correlation,
    build_correlation_matrix,
    cholesky_factor,
    nearest_correlation_matrix,
)

ORDER = ["a", "b", "c"]


# ---------------------------------------------------------------------------
# building the matrix
# ---------------------------------------------------------------------------

def test_equicorrelation_matrix():
    matrix = build_correlation_matrix(CorrelationSpec("g", rho=0.4), ORDER)
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 1.0)
    off = matrix[~np.eye(3, dtype=bool)]
    assert np.allclose(off, 0.4)


def test_zero_rho_gives_the_identity():
    matrix = build_correlation_matrix(CorrelationSpec("g"), ORDER)
    assert np.allclose(matrix, np.eye(3))


def test_explicit_matrix():
    given = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 1.0]]
    matrix = build_correlation_matrix(CorrelationSpec("g", matrix=given), ORDER)
    assert np.allclose(matrix, given)


def test_explicit_matrix_is_reordered_to_match_the_slot_order():
    """'members' names the rows of the supplied matrix; the sampler's own slot
    ordering wins, so the matrix must be permuted to match."""
    given = [[1.0, 0.7, 0.0],
             [0.7, 1.0, 0.0],
             [0.0, 0.0, 1.0]]
    spec = CorrelationSpec("g", matrix=given, members=["c", "a", "b"])
    matrix = build_correlation_matrix(spec, ORDER)      # order a, b, c
    # In the supplied ordering c<->a were correlated 0.7; after permuting to
    # (a, b, c) that correlation must sit at (a, c).
    assert matrix[0, 2] == pytest.approx(0.7)
    assert matrix[0, 1] == pytest.approx(0.0)


def test_member_list_that_disagrees_is_rejected():
    given = [[1.0, 0.5], [0.5, 1.0]]
    spec = CorrelationSpec("g", matrix=given, members=["a", "z"])
    with pytest.raises(VariationError) as excinfo:
        build_correlation_matrix(spec, ["a", "b"])
    assert "members" in str(excinfo.value)


# ---------------------------------------------------------------------------
# validation -- each failure must explain itself
# ---------------------------------------------------------------------------

def test_wrong_shape_is_rejected():
    spec = CorrelationSpec("g", matrix=[[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(VariationError) as excinfo:
        build_correlation_matrix(spec, ORDER)
    assert "3 members" in str(excinfo.value)


def test_asymmetric_matrix_is_rejected():
    spec = CorrelationSpec("g", matrix=[[1.0, 0.5], [0.1, 1.0]])
    with pytest.raises(VariationError) as excinfo:
        build_correlation_matrix(spec, ["a", "b"])
    assert "symmetric" in str(excinfo.value)


def test_non_unit_diagonal_is_rejected_and_names_the_offender():
    spec = CorrelationSpec("g", matrix=[[1.0, 0.0], [0.0, 0.8]])
    with pytest.raises(VariationError) as excinfo:
        build_correlation_matrix(spec, ["a", "b"])
    message = str(excinfo.value)
    assert "diagonal" in message and "b" in message


def test_out_of_range_entries_are_rejected():
    spec = CorrelationSpec("g", matrix=[[1.0, 1.5], [1.5, 1.0]])
    with pytest.raises(VariationError):
        build_correlation_matrix(spec, ["a", "b"])


def test_non_positive_semidefinite_matrix_is_rejected_with_an_explanation():
    """These three pairwise correlations cannot all hold simultaneously."""
    impossible = [[1.0, 0.9, -0.9],
                  [0.9, 1.0, 0.9],
                  [-0.9, 0.9, 1.0]]
    spec = CorrelationSpec("g", matrix=impossible)
    with pytest.raises(VariationError) as excinfo:
        build_correlation_matrix(spec, ORDER)
    message = str(excinfo.value)
    assert "positive semi-definite" in message
    assert "mutually inconsistent" in message


def test_impossible_equicorrelation_is_rejected():
    """For n variables, rho must exceed -1/(n-1)."""
    with pytest.raises(VariationError) as excinfo:
        build_correlation_matrix(CorrelationSpec("g", rho=-0.9), ORDER)
    assert "not achievable" in str(excinfo.value)


@pytest.mark.parametrize("rho", [-1.0, 1.0, 1.5, -2.0])
def test_rho_outside_the_open_interval_is_rejected(rho):
    with pytest.raises(VariationError):
        build_correlation_matrix(CorrelationSpec("g", rho=rho), ORDER)


def test_empty_group_is_rejected():
    with pytest.raises(VariationError):
        build_correlation_matrix(CorrelationSpec("g", rho=0.5), [])


# ---------------------------------------------------------------------------
# factorisation
# ---------------------------------------------------------------------------

def test_cholesky_reproduces_the_matrix():
    matrix = build_correlation_matrix(CorrelationSpec("g", rho=0.6), ORDER)
    chol = cholesky_factor(matrix)
    assert np.allclose(chol @ chol.T, matrix)
    assert np.allclose(np.triu(chol, 1), 0.0)      # lower triangular


def test_cholesky_survives_a_perfectly_correlated_pair():
    """rho = 1 makes the matrix singular; the jitter fallback must cope."""
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    chol = cholesky_factor(singular, name="perfect")
    assert np.allclose(chol @ chol.T, singular, atol=1e-6)


def test_hopeless_matrix_raises():
    with pytest.raises(VariationError):
        cholesky_factor(np.array([[1.0, 5.0], [5.0, 1.0]]), name="bad")


def test_nearest_correlation_matrix_repairs_an_invalid_one():
    broken = np.array([[1.0, 0.9, -0.9],
                       [0.9, 1.0, 0.9],
                       [-0.9, 0.9, 1.0]])
    repaired = nearest_correlation_matrix(broken)
    assert np.allclose(np.diag(repaired), 1.0, atol=1e-8)
    assert np.linalg.eigvalsh(repaired).min() > -1e-8
    # The repair is a projection, so it should stay close to the original.
    assert np.max(np.abs(repaired - broken)) < 1.0


# ---------------------------------------------------------------------------
# empirical recovery -- the property that actually matters
# ---------------------------------------------------------------------------

def test_generated_variables_recover_the_requested_correlation():
    target = np.array([[1.0, 0.7, 0.2],
                       [0.7, 1.0, -0.3],
                       [0.2, -0.3, 1.0]])
    chol = cholesky_factor(target)
    n = 200_000
    z = np.random.default_rng(4242).standard_normal((n, 3))
    correlated = apply_correlation(z, chol)
    empirical = np.corrcoef(correlated, rowvar=False)
    # SE of a sample correlation is about (1-r^2)/sqrt(n); 5 SE is ~0.011.
    assert np.max(np.abs(empirical - target)) < 0.012


def test_generated_variables_keep_unit_variance():
    target = build_correlation_matrix(CorrelationSpec("g", rho=0.5), ORDER)
    z = np.random.default_rng(7).standard_normal((100_000, 3))
    correlated = apply_correlation(z, cholesky_factor(target))
    assert np.allclose(correlated.std(axis=0, ddof=1), 1.0, atol=0.02)


def test_independent_variables_stay_independent():
    identity = np.eye(3)
    z = np.random.default_rng(11).standard_normal((100_000, 3))
    correlated = apply_correlation(z, cholesky_factor(identity))
    empirical = np.corrcoef(correlated, rowvar=False)
    off = empirical[~np.eye(3, dtype=bool)]
    assert np.max(np.abs(off)) < 0.02


def test_correlation_spec_serialises():
    spec = CorrelationSpec("g", rho=0.5, members=ORDER)
    data = spec.to_dict()
    assert data["name"] == "g" and data["rho"] == pytest.approx(0.5)
