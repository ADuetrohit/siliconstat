"""Statistical distributions for parameter variation.

Every distribution exposes the same two operations:

``sample(rng, n)``
    draw *n* independent deviations;

``from_standard_normal(z)``
    map standard-normal variates to deviations.

The second operation is what makes correlation work uniformly across
distribution families: correlated standard normals are generated once (via a
Cholesky factor of the correlation matrix) and then pushed through each
parameter's own marginal transform -- a Gaussian copula.  Rank correlation is
preserved exactly; linear correlation is preserved exactly for Gaussian
marginals and approximately for the others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import erf, erfinv

from ..core.exceptions import VariationError

__all__ = [
    "Distribution", "Gaussian", "Uniform", "LogNormal",
    "make_distribution", "DISTRIBUTIONS",
]

_SQRT2 = math.sqrt(2.0)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / _SQRT2))


class Distribution:
    """Base class: a zero-centred *deviation* generator."""

    name = "base"

    @property
    def std(self) -> float:
        """Standard deviation of the generated deviation."""
        raise NotImplementedError  # pragma: no cover

    def from_standard_normal(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.from_standard_normal(rng.standard_normal(n))

    def truncated(self, z: np.ndarray, n_sigma: float | None) -> np.ndarray:
        if n_sigma is None or n_sigma <= 0:
            return z
        return np.clip(z, -n_sigma, n_sigma)

    def to_dict(self) -> dict[str, Any]:
        return {"distribution": self.name, "std": self.std}

    def describe(self) -> str:
        return f"{self.name}(std={self.std:.4g})"


@dataclass(frozen=True)
class Gaussian(Distribution):
    """Normal deviation with standard deviation *sigma*, mean zero."""

    sigma: float
    name: str = "gaussian"

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise VariationError(f"gaussian sigma must be >= 0 (got {self.sigma})")

    @property
    def std(self) -> float:
        return self.sigma

    def from_standard_normal(self, z: np.ndarray) -> np.ndarray:
        return self.sigma * np.asarray(z, dtype=float)


@dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform deviation on ``[-halfwidth, +halfwidth]``.

    Constructed either from the half-width directly or from a target standard
    deviation, using ``halfwidth = sigma * sqrt(3)``.
    """

    halfwidth: float
    name: str = "uniform"

    def __post_init__(self) -> None:
        if self.halfwidth < 0:
            raise VariationError(f"uniform halfwidth must be >= 0 (got {self.halfwidth})")

    @classmethod
    def from_sigma(cls, sigma: float) -> "Uniform":
        return cls(halfwidth=sigma * math.sqrt(3.0))

    @property
    def std(self) -> float:
        return self.halfwidth / math.sqrt(3.0)

    def from_standard_normal(self, z: np.ndarray) -> np.ndarray:
        # Gaussian copula: normal CDF -> U(0,1) -> uniform deviation.
        u = _normal_cdf(np.asarray(z, dtype=float))
        return self.halfwidth * (2.0 * u - 1.0)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(-self.halfwidth, self.halfwidth, size=n)

    def to_dict(self) -> dict[str, Any]:
        return {"distribution": self.name, "halfwidth": self.halfwidth, "std": self.std}


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Multiplicative log-normal deviation, returned as ``exp(sigma_log*z) - 1``.

    Only meaningful for *relative* parameters (a resistor value, a current
    factor) because it is strictly bounded below by ``-1``; applying it to an
    additive parameter such as a threshold shift is rejected by the sampler.

    ``sigma_log`` is the standard deviation of the underlying normal.

    :meth:`from_relative_sigma` inverts the variance exactly.  For
    ``X = exp(N(0, s_log))`` the variance is ``(exp(s_log^2) - 1) * exp(s_log^2)``,
    so requiring ``std(X) = s`` gives, with ``u = exp(s_log^2)``::

        u^2 - u - s^2 = 0   =>   u = (1 + sqrt(1 + 4 s^2)) / 2

    Note this is *not* the more commonly quoted ``s_log = sqrt(ln(1 + s^2))``,
    which fixes the coefficient of variation ``std/mean`` rather than the
    standard deviation itself; the two differ by a factor ``sqrt(1 + s^2)``.
    The standard deviation is the right thing to pin down here because the
    variation model specifies sigmas, not CVs.
    """

    sigma_log: float
    name: str = "lognormal"

    def __post_init__(self) -> None:
        if self.sigma_log < 0:
            raise VariationError(f"lognormal sigma_log must be >= 0 (got {self.sigma_log})")

    @classmethod
    def from_relative_sigma(cls, sigma_rel: float) -> "LogNormal":
        """Build a log-normal whose deviation has standard deviation *sigma_rel*."""
        if sigma_rel < 0:
            raise VariationError("lognormal relative sigma must be >= 0")
        if sigma_rel == 0:
            return cls(sigma_log=0.0)
        u = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * sigma_rel * sigma_rel))
        return cls(sigma_log=math.sqrt(math.log(u)))

    @property
    def std(self) -> float:
        # std of exp(N(0, s)) is exp(s^2/2)*sqrt(exp(s^2)-1); the deviation is
        # that multiplier minus one, so the std is unchanged by the shift.
        s2 = self.sigma_log ** 2
        return math.exp(s2 / 2.0) * math.sqrt(max(math.exp(s2) - 1.0, 0.0))

    @property
    def mean(self) -> float:
        """Mean of the deviation -- log-normals are *not* zero-mean."""
        return math.exp(self.sigma_log ** 2 / 2.0) - 1.0

    def from_standard_normal(self, z: np.ndarray) -> np.ndarray:
        return np.exp(self.sigma_log * np.asarray(z, dtype=float)) - 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"distribution": self.name, "sigma_log": self.sigma_log,
                "std": self.std, "mean": self.mean}


DISTRIBUTIONS = {"gaussian": Gaussian, "normal": Gaussian,
                 "uniform": Uniform, "lognormal": LogNormal}


def make_distribution(kind: str, sigma: float, *, halfwidth: float | None = None
                      ) -> Distribution:
    """Build a distribution from a name and a target standard deviation."""
    key = (kind or "gaussian").strip().lower()
    if key in ("gaussian", "normal"):
        return Gaussian(sigma=sigma)
    if key == "uniform":
        if halfwidth is not None:
            return Uniform(halfwidth=halfwidth)
        return Uniform.from_sigma(sigma)
    if key == "lognormal":
        return LogNormal.from_relative_sigma(sigma)
    raise VariationError(
        f"unknown distribution {kind!r}; supported: "
        f"{', '.join(sorted(set(DISTRIBUTIONS)))}")


def inverse_normal_cdf(p: np.ndarray) -> np.ndarray:
    """Standard-normal quantile function (used by Latin hypercube sampling)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    return _SQRT2 * erfinv(2.0 * p - 1.0)
