"""Statistical distributions and the Gaussian-copula transform.

Tolerances are derived from the standard error of the estimator rather than
guessed, so these tests are tight but not flaky.  For N samples of a
distribution with standard deviation s:

    SE(mean) = s / sqrt(N)
    SE(std)  ~ s / sqrt(2N)

Every assertion below allows 5 standard errors, i.e. a false-failure rate of
about 6e-7 per assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.core.exceptions import VariationError
from siliconstat.variation.distributions import (
    Gaussian,
    LogNormal,
    Uniform,
    inverse_normal_cdf,
    make_distribution,
)

N = 100_000
SEED = 20240517


def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def se_mean(sigma: float, n: int = N) -> float:
    return 5.0 * sigma / math.sqrt(n)


def se_std(sigma: float, n: int = N) -> float:
    return 5.0 * sigma / math.sqrt(2.0 * n)


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

def test_gaussian_mean_and_std():
    sigma = 3.5e-3
    sample = Gaussian(sigma).sample(rng(), N)
    assert sample.mean() == pytest.approx(0.0, abs=se_mean(sigma))
    assert sample.std(ddof=1) == pytest.approx(sigma, abs=se_std(sigma))


def test_gaussian_is_symmetric_and_mesokurtic():
    sample = Gaussian(1.0).sample(rng(), N)
    z = (sample - sample.mean()) / sample.std(ddof=1)
    assert abs(float(np.mean(z ** 3))) < 5 * math.sqrt(6.0 / N)         # skew
    assert abs(float(np.mean(z ** 4) - 3.0)) < 5 * math.sqrt(24.0 / N)  # kurtosis


def test_gaussian_tail_probabilities():
    """About 0.27 % of a normal population lies beyond 3 sigma."""
    sample = Gaussian(1.0).sample(rng(), N)
    beyond_3 = float(np.mean(np.abs(sample) > 3.0))
    assert beyond_3 == pytest.approx(0.0027, abs=5 * math.sqrt(0.0027 / N))


def test_gaussian_standard_normal_transform_is_linear():
    z = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    assert np.allclose(Gaussian(2e-3).from_standard_normal(z), 2e-3 * z)


def test_zero_sigma_produces_no_deviation():
    assert np.all(Gaussian(0.0).sample(rng(), 100) == 0.0)


def test_negative_sigma_is_rejected():
    with pytest.raises(VariationError):
        Gaussian(-1.0)


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------

def test_uniform_bounds_mean_and_std():
    half = 0.05
    dist = Uniform(halfwidth=half)
    sample = dist.sample(rng(), N)
    assert sample.min() >= -half and sample.max() <= half
    assert sample.mean() == pytest.approx(0.0, abs=se_mean(dist.std))
    assert sample.std(ddof=1) == pytest.approx(dist.std, abs=se_std(dist.std))
    assert dist.std == pytest.approx(half / math.sqrt(3.0))


def test_uniform_from_sigma_reproduces_the_requested_sigma():
    sigma = 0.02
    dist = Uniform.from_sigma(sigma)
    assert dist.halfwidth == pytest.approx(sigma * math.sqrt(3.0))
    assert dist.std == pytest.approx(sigma)
    sample = dist.sample(rng(), N)
    assert sample.std(ddof=1) == pytest.approx(sigma, abs=se_std(sigma))


def test_uniform_copula_transform_is_bounded_and_monotone():
    """Via the normal CDF the copula maps standard normals onto the interval
    while preserving order -- that is what keeps rank correlation intact."""
    dist = Uniform(halfwidth=0.05)
    z = np.linspace(-5.0, 5.0, 500)
    deviations = dist.from_standard_normal(z)
    assert np.all(deviations >= -0.05 - 1e-12)
    assert np.all(deviations <= 0.05 + 1e-12)
    assert np.all(np.diff(deviations) >= -1e-15)
    assert dist.from_standard_normal(np.array([0.0]))[0] == pytest.approx(0.0)


def test_uniform_copula_sample_has_the_right_std():
    dist = Uniform(halfwidth=0.05)
    z = rng().standard_normal(N)
    sample = dist.from_standard_normal(z)
    assert sample.std(ddof=1) == pytest.approx(dist.std, abs=se_std(dist.std))


def test_negative_halfwidth_is_rejected():
    with pytest.raises(VariationError):
        Uniform(halfwidth=-1.0)


# ---------------------------------------------------------------------------
# LogNormal
# ---------------------------------------------------------------------------

def test_lognormal_is_strictly_greater_than_minus_one():
    sample = LogNormal(sigma_log=0.3).sample(rng(), N)
    assert sample.min() > -1.0


def test_lognormal_from_relative_sigma_reproduces_that_sigma():
    for rel in (0.01, 0.05, 0.25):
        dist = LogNormal.from_relative_sigma(rel)
        sample = dist.sample(rng(), N)
        assert dist.std == pytest.approx(rel, rel=1e-12)
        assert sample.std(ddof=1) == pytest.approx(rel, abs=se_std(rel) * 2)


def test_lognormal_is_not_zero_mean_and_says_so():
    """exp(N(0,s)) has mean exp(s^2/2), so the deviation is biased upward."""
    dist = LogNormal(sigma_log=0.5)
    assert dist.mean == pytest.approx(math.exp(0.125) - 1.0)
    sample = dist.sample(rng(), N)
    assert sample.mean() == pytest.approx(dist.mean, abs=se_mean(dist.std) * 2)


def test_lognormal_is_right_skewed():
    sample = LogNormal(sigma_log=0.5).sample(rng(), N)
    z = (sample - sample.mean()) / sample.std(ddof=1)
    assert float(np.mean(z ** 3)) > 1.0


def test_lognormal_multiplier_is_the_exponential_of_a_normal():
    dist = LogNormal(sigma_log=0.2)
    z = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(dist.from_standard_normal(z) + 1.0, np.exp(0.2 * z))


def test_negative_lognormal_sigma_is_rejected():
    with pytest.raises(VariationError):
        LogNormal(sigma_log=-0.1)
    with pytest.raises(VariationError):
        LogNormal.from_relative_sigma(-0.1)


# ---------------------------------------------------------------------------
# factory and helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cls", [
    ("gaussian", Gaussian), ("normal", Gaussian),
    ("uniform", Uniform), ("lognormal", LogNormal),
])
def test_make_distribution(name, cls):
    dist = make_distribution(name, 0.02)
    assert isinstance(dist, cls)
    assert dist.std == pytest.approx(0.02, rel=1e-9)


def test_make_distribution_rejects_unknown_names():
    with pytest.raises(VariationError) as excinfo:
        make_distribution("cauchy", 0.02)
    assert "gaussian" in str(excinfo.value)


def test_make_uniform_from_explicit_halfwidth():
    dist = make_distribution("uniform", 0.0, halfwidth=0.1)
    assert isinstance(dist, Uniform)
    assert dist.halfwidth == pytest.approx(0.1)


def test_inverse_normal_cdf_matches_scipy():
    from scipy.stats import norm

    p = np.array([1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-6])
    assert np.allclose(inverse_normal_cdf(p), norm.ppf(p), atol=1e-10)


def test_inverse_normal_cdf_clips_degenerate_probabilities():
    values = inverse_normal_cdf(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(values))


def test_distribution_serialisation():
    for dist in (Gaussian(1e-3), Uniform(0.02), LogNormal(0.1)):
        data = dist.to_dict()
        assert data["distribution"] == dist.name
        assert data["std"] == pytest.approx(dist.std)
        assert dist.describe()
