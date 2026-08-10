"""Correlated random variable generation.

Parameters that share a ``correlation_group`` are drawn from a multivariate
normal with a prescribed correlation matrix.  The matrix is factorised once
(Cholesky) and reused for every sample, so generating ``N`` correlated samples
costs one matrix multiply.

Invalid correlation matrices are a real user-facing failure mode (a hand-typed
matrix is very often not positive semi-definite), so this module validates
symmetry, unit diagonal and eigenvalues explicitly and reports precisely what
is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.exceptions import VariationError

__all__ = ["CorrelationSpec", "build_correlation_matrix", "cholesky_factor",
           "nearest_correlation_matrix", "apply_correlation"]


@dataclass
class CorrelationSpec:
    """Correlation structure for one group of parameters.

    Either an equicorrelation coefficient *rho* (every pair correlated the
    same amount) or an explicit *matrix* over the group's members.
    """

    name: str
    rho: float | None = None
    matrix: list[list[float]] | None = None
    members: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "rho": self.rho,
                "matrix": self.matrix, "members": self.members}


def build_correlation_matrix(spec: CorrelationSpec, order: Sequence[str]) -> np.ndarray:
    """Materialise the correlation matrix for group members in *order*."""
    n = len(order)
    if n == 0:
        raise VariationError(f"correlation group {spec.name!r} has no members")
    if spec.matrix is not None:
        mat = np.asarray(spec.matrix, dtype=float)
        if mat.shape != (n, n):
            raise VariationError(
                f"correlation group {spec.name!r}: matrix is {mat.shape} but the "
                f"group has {n} members ({', '.join(order)})")
        if spec.members is not None:
            if len(spec.members) != n or set(spec.members) != set(order):
                raise VariationError(
                    f"correlation group {spec.name!r}: 'members' "
                    f"{spec.members} does not match the parameters actually in "
                    f"the group ({list(order)})")
            index = [spec.members.index(name) for name in order]
            mat = mat[np.ix_(index, index)]
    else:
        rho = 0.0 if spec.rho is None else float(spec.rho)
        if not -1.0 < rho < 1.0 and n > 1:
            raise VariationError(
                f"correlation group {spec.name!r}: rho must lie strictly between "
                f"-1 and 1 (got {rho})")
        if n > 1 and rho <= -1.0 / (n - 1):
            raise VariationError(
                f"correlation group {spec.name!r}: equicorrelation rho={rho} is not "
                f"achievable for {n} variables (needs rho > {-1.0 / (n - 1):.4f})")
        mat = np.full((n, n), rho, dtype=float)
        np.fill_diagonal(mat, 1.0)
    validate_correlation_matrix(mat, spec.name, order)
    return mat


def validate_correlation_matrix(mat: np.ndarray, name: str,
                                order: Sequence[str]) -> None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise VariationError(f"correlation group {name!r}: matrix must be square")
    if not np.allclose(mat, mat.T, atol=1e-10):
        raise VariationError(f"correlation group {name!r}: matrix is not symmetric")
    diag = np.diag(mat)
    if not np.allclose(diag, 1.0, atol=1e-8):
        bad = [order[i] for i in np.where(np.abs(diag - 1.0) > 1e-8)[0]]
        raise VariationError(
            f"correlation group {name!r}: diagonal entries must be 1.0 "
            f"(offenders: {', '.join(bad)})")
    if np.any(np.abs(mat) > 1.0 + 1e-10):
        raise VariationError(
            f"correlation group {name!r}: off-diagonal entries must lie in [-1, 1]")
    eigenvalues = np.linalg.eigvalsh(mat)
    if eigenvalues.min() < -1e-8:
        raise VariationError(
            f"correlation group {name!r}: matrix is not positive semi-definite "
            f"(smallest eigenvalue {eigenvalues.min():.3e}). The requested set of "
            "pairwise correlations is mutually inconsistent -- no set of random "
            "variables can have all of them simultaneously.")


def cholesky_factor(mat: np.ndarray, *, name: str = "correlation") -> np.ndarray:
    """Lower-triangular Cholesky factor, with a jitter fallback.

    A correlation matrix that is positive *semi*-definite (a perfectly
    correlated pair, say) has a zero eigenvalue and fails plain Cholesky; a
    tiny diagonal jitter makes it factorisable without materially changing the
    statistics.
    """
    try:
        return np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        pass
    for jitter in (1e-12, 1e-10, 1e-8, 1e-6):
        try:
            return np.linalg.cholesky(mat + jitter * np.eye(mat.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise VariationError(
        f"{name}: correlation matrix could not be factorised even with jitter; "
        "it is too far from positive semi-definite to be usable")


def nearest_correlation_matrix(mat: np.ndarray, iterations: int = 100) -> np.ndarray:
    """Higham's alternating-projection nearest correlation matrix.

    Offered as an explicit repair step -- never applied silently.
    """
    x = np.array(mat, dtype=float, copy=True)
    y = x.copy()
    ds = np.zeros_like(x)
    for _ in range(iterations):
        r = y - ds
        eigenvalues, eigenvectors = np.linalg.eigh(r)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        x = (eigenvectors * eigenvalues) @ eigenvectors.T
        ds = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)
    return y


def apply_correlation(z: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """Transform independent standard normals into correlated ones.

    ``z`` has shape ``(n_samples, k)``; the result has the same shape and
    covariance ``chol @ chol.T``.
    """
    return z @ chol.T
