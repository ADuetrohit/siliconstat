"""Descriptive statistics, including the numerical-stability claim."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.core.exceptions import AnalysisError
from siliconstat.analysis.statistics import (
    cdf_data,
    describe,
    describe_many,
    histogram_data,
    sigma_quantile_data,
    welford,
)


def naive_variance(values) -> float:
    """The textbook one-pass formula, kept here only to be shown failing."""
    n = len(values)
    total = sum(values)
    total_sq = sum(v * v for v in values)
    return (total_sq - total * total / n) / (n - 1)


# ---------------------------------------------------------------------------
# numerical stability -- the reason Welford is used
# ---------------------------------------------------------------------------

def test_welford_matches_numpy_on_well_conditioned_data():
    data = np.random.default_rng(0).normal(0.0, 1.0, 5000)
    n, mean, variance = welford(data.tolist())
    assert n == 5000
    assert mean == pytest.approx(float(np.mean(data)), rel=1e-12)
    assert variance == pytest.approx(float(np.var(data, ddof=1)), rel=1e-10)


@pytest.mark.parametrize("mean,sigma,n,naive_floor", [
    # (mean/sigma ratio, how badly the naive formula is expected to do)
    (1.2, 3e-4, 20_000, 1e-10),      # ordinary analog: naive is merely worse
    (1.2, 3e-6, 20_000, 1e-6),       # tight spread: naive loses 5 digits
])
def test_welford_beats_the_naive_formula_by_orders_of_magnitude(
        mean, sigma, n, naive_floor):
    data = (mean + sigma * np.random.default_rng(7).standard_normal(n)).tolist()
    reference = float(np.var(np.array(data), ddof=1))

    _n, _mean, stable = welford(data)
    unstable = naive_variance(data)
    stable_error = abs(stable - reference) / reference
    naive_error = abs(unstable - reference) / reference

    assert stable_error < 1e-9
    assert naive_error > naive_floor
    assert naive_error > stable_error * 100.0


def test_naive_variance_collapses_completely_at_a_large_offset():
    """At mean/sigma ~ 1e9 the two summands of the one-pass formula agree to
    within a few floating-point ulps, so their difference is noise: the result
    is wrong by orders of magnitude (and is often exactly zero).  Welford is
    unaffected."""
    data = (1e6 + 1e-3 * np.random.default_rng(3).standard_normal(20_000)).tolist()
    reference = float(np.var(np.array(data), ddof=1))

    naive_error = abs(naive_variance(data) - reference) / reference
    assert naive_error > 1.0          # wrong by more than 100 %

    _n, _mean, stable = welford(data)
    assert stable == pytest.approx(reference, rel=1e-6)


def test_welford_edge_cases():
    n, mean, variance = welford([])
    assert n == 0 and math.isnan(mean) and math.isnan(variance)
    n, mean, variance = welford([2.5])
    assert (n, mean) == (1, 2.5) and math.isnan(variance)
    n, _mean, _var = welford([1.0, float("nan"), 2.0])
    assert n == 2                     # NaN skipped, not treated as zero


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

def test_describe_matches_numpy():
    data = np.random.default_rng(1).normal(5.0, 2.0, 4000)
    stats = describe(data, name="x", unit="V")
    assert stats.count == 4000
    assert stats.mean == pytest.approx(float(np.mean(data)), rel=1e-12)
    assert stats.median == pytest.approx(float(np.median(data)), rel=1e-12)
    assert stats.std == pytest.approx(float(np.std(data, ddof=1)), rel=1e-10)
    assert stats.variance == pytest.approx(float(np.var(data, ddof=1)), rel=1e-10)
    assert stats.minimum == pytest.approx(float(np.min(data)))
    assert stats.maximum == pytest.approx(float(np.max(data)))
    assert stats.range == pytest.approx(stats.maximum - stats.minimum)


def test_percentiles_match_numpy_linear_interpolation():
    data = np.random.default_rng(2).normal(0.0, 1.0, 3000)
    stats = describe(data)
    for attr, q in (("p1", 1), ("p5", 5), ("p25", 25), ("p75", 75),
                    ("p95", 95), ("p99", 99)):
        assert getattr(stats, attr) == pytest.approx(
            float(np.percentile(data, q)), rel=1e-12)
    assert stats.iqr == pytest.approx(stats.p75 - stats.p25)


def test_three_sigma_limits_and_cv():
    data = np.random.default_rng(3).normal(10.0, 0.5, 2000)
    stats = describe(data)
    assert stats.sigma3_low == pytest.approx(stats.mean - 3 * stats.std)
    assert stats.sigma3_high == pytest.approx(stats.mean + 3 * stats.std)
    assert stats.cv == pytest.approx(stats.std / abs(stats.mean))


def test_cv_is_infinite_at_zero_mean():
    stats = describe(np.array([-1.0, 1.0, -1.0, 1.0]))
    assert math.isinf(stats.cv)


def test_standard_error_and_mean_confidence_interval():
    data = np.random.default_rng(4).normal(0.0, 1.0, 1000)
    stats = describe(data)
    assert stats.sem == pytest.approx(stats.std / math.sqrt(1000), rel=1e-12)
    assert stats.mean_ci95_low == pytest.approx(stats.mean - 1.959964 * stats.sem,
                                                rel=1e-9)
    assert stats.mean_ci95_high - stats.mean_ci95_low == pytest.approx(
        2 * 1.959964 * stats.sem, rel=1e-9)


def test_sigma_confidence_interval_brackets_the_estimate():
    data = np.random.default_rng(5).normal(0.0, 2.0, 500)
    stats = describe(data)
    assert stats.std_ci95_low < stats.std < stats.std_ci95_high
    assert stats.std_ci95_low > 0


def test_skewness_and_kurtosis_of_a_normal_sample():
    data = np.random.default_rng(6).normal(0.0, 1.0, 50_000)
    stats = describe(data)
    assert abs(stats.skewness) < 0.05
    assert abs(stats.kurtosis_excess) < 0.10


def test_skewness_of_a_deliberately_skewed_sample():
    data = np.random.default_rng(7).lognormal(0.0, 0.8, 20_000)
    assert describe(data).skewness > 1.0


def test_normality_test_accepts_normal_and_rejects_uniform():
    normal = describe(np.random.default_rng(8).normal(0, 1, 800))
    assert normal.normality_test == "shapiro-wilk"
    assert normal.normality_p > 0.01

    uniform = describe(np.random.default_rng(8).uniform(-1, 1, 800))
    assert uniform.normality_p < 0.01


def test_large_samples_switch_to_dagostino_pearson():
    stats = describe(np.random.default_rng(9).normal(0, 1, 6000))
    assert stats.normality_test == "dagostino-pearson"


def test_nan_values_are_excluded_and_counted_not_zeroed():
    data = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
    stats = describe(data)
    assert stats.count == 3
    assert stats.n_input == 5
    assert stats.n_invalid == 2
    assert stats.mean == pytest.approx(2.0)


def test_describe_on_degenerate_inputs_does_not_crash():
    empty = describe(np.array([]))
    assert empty.count == 0 and math.isnan(empty.mean)

    single = describe(np.array([1.5]))
    assert single.count == 1 and single.mean == pytest.approx(1.5)
    assert math.isnan(single.std)

    constant = describe(np.full(50, 2.0))
    assert constant.std == pytest.approx(0.0)
    assert constant.range == pytest.approx(0.0)


def test_describe_many_and_serialisation():
    stats = describe_many({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]},
                          units={"a": "V"})
    assert set(stats) == {"a", "b"}
    assert stats["a"].unit == "V"
    data = stats["a"].to_dict()
    assert data["mean"] == pytest.approx(2.0)
    assert "summary" not in data
    assert stats["a"].summary_line()


# ---------------------------------------------------------------------------
# chart payloads
# ---------------------------------------------------------------------------

def test_histogram_shape_and_conservation():
    data = np.random.default_rng(10).normal(0, 1, 2000)
    hist = histogram_data(data)
    assert len(hist["edges"]) == len(hist["counts"]) + 1
    assert len(hist["centres"]) == len(hist["counts"])
    assert sum(hist["counts"]) == 2000
    assert hist["n"] == 2000


def test_histogram_of_constant_data_is_a_single_bin():
    hist = histogram_data(np.full(20, 1.25))
    assert hist["counts"] == [20]
    assert hist["edges"][0] < 1.25 < hist["edges"][1]


def test_histogram_needs_finite_values():
    with pytest.raises(AnalysisError):
        histogram_data(np.array([np.nan, np.inf]))


def test_cdf_uses_the_hazen_plotting_position():
    """(i - 0.5)/n keeps the tails unbiased, which matters when reading a
    small failure probability off the curve."""
    data = np.array([3.0, 1.0, 2.0, 4.0])
    cdf = cdf_data(data)
    assert cdf["x"] == [1.0, 2.0, 3.0, 4.0]
    assert cdf["p"] == pytest.approx([0.125, 0.375, 0.625, 0.875])
    assert cdf["p"][0] > 0.0 and cdf["p"][-1] < 1.0


def test_cdf_is_monotone():
    data = np.random.default_rng(11).normal(0, 1, 500)
    cdf = cdf_data(data)
    assert all(b >= a for a, b in zip(cdf["x"], cdf["x"][1:]))
    assert all(b > a for a, b in zip(cdf["p"], cdf["p"][1:]))


def test_sigma_plot_is_straight_for_gaussian_data():
    data = np.random.default_rng(12).normal(4.0, 0.5, 3000)
    plot = sigma_quantile_data(data)
    x = np.array(plot["x"])
    z = np.array(plot["sigma"])
    finite = np.isfinite(z)
    correlation = np.corrcoef(x[finite], z[finite])[0, 1]
    assert correlation > 0.999
