"""Sensitivity attribution and correlation analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.analysis import (
    correlation_analysis,
    measurement_correlation,
    sensitivity_analysis,
)
from siliconstat.analysis.correlation import (
    pearson_matrix,
    rank_transform,
    spearman_matrix,
)
from siliconstat.core.exceptions import AnalysisError
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.mc.results import MonteCarloRun, RunCounters, SampleResult
from siliconstat.variation import default_mismatch_model

from conftest import load


def synthetic_run(coefficients, n=800, noise=0.0, seed=0,
                  correlate=False) -> MonteCarloRun:
    """A run whose response is a known linear function of its parameters."""
    rng = np.random.default_rng(seed)
    k = len(coefficients)
    x = rng.standard_normal((n, k))
    if correlate:
        x[:, 1] = 0.9 * x[:, 0] + math.sqrt(1 - 0.81) * x[:, 1]
    y = x @ np.asarray(coefficients, dtype=float)
    if noise:
        y = y + noise * rng.standard_normal(n)

    names = [f"p{i}" for i in range(k)]
    run = MonteCarloRun(run_id="synthetic", circuit_name="synthetic", config={})
    run.measurement_meta = [{"name": "y", "kind": "expr", "unit": ""}]
    run.slot_meta = [{"slot": name, "parameter": "vth", "scope": "local",
                      "distribution": "gaussian", "sigma": 1.0, "unit": "V",
                      "devices": name, "label": ""} for name in names]
    for i in range(n):
        sample = SampleResult(index=i, seed=i)
        sample.slot_values = {name: float(x[i, j]) for j, name in enumerate(names)}
        sample.measurements = {"y": float(y[i])}
        sample.measurement_ok = {"y": True}
        run.samples.append(sample)
        run.counters.add(sample)
    return run


# ---------------------------------------------------------------------------
# variance decomposition on a known response
# ---------------------------------------------------------------------------

def test_variance_shares_recover_known_coefficients():
    """With unit-variance uncorrelated inputs, beta_i^2 / sum(beta^2) is the
    fraction of output variance each input contributes."""
    run = synthetic_run([3.0, 1.0, 0.0])
    report = sensitivity_analysis(run, "y")
    shares = {e.parameter: e.variance_contribution for e in report.entries}
    total = 9.0 + 1.0
    assert shares["p0"] == pytest.approx(9.0 / total, rel=0.05)
    assert shares["p1"] == pytest.approx(1.0 / total, rel=0.10)
    assert abs(shares["p2"]) < 0.01


def test_r_squared_equals_sum_of_beta_times_correlation():
    """The exact identity for a standardised regression: R^2 = sum(beta_i * r_iy).
    It holds for any design, orthogonal or not."""
    run = synthetic_run([2.0, 1.0], noise=0.5)
    report = sensitivity_analysis(run, "y")
    identity = sum(e.beta_standardised * e.pearson for e in report.entries)
    assert identity == pytest.approx(report.r_squared, rel=1e-9)


def test_variance_shares_approximately_sum_to_r_squared():
    """sum(beta^2) equals R^2 only when the inputs are *exactly* orthogonal.
    At n = 800 the empirical correlation between two independent inputs is
    still about 1/sqrt(n) = 0.035, which moves the sum by a couple of percent.
    That gap is a property of finite sampling, not of the implementation."""
    run = synthetic_run([2.0, 1.0], noise=0.5)
    report = sensitivity_analysis(run, "y")
    total = sum(e.variance_contribution for e in report.entries
                if e.variance_contribution == e.variance_contribution)
    assert total == pytest.approx(report.r_squared, rel=0.05)


def test_r_squared_is_one_for_a_noiseless_linear_response():
    report = sensitivity_analysis(synthetic_run([1.0, -2.0]), "y")
    assert report.r_squared == pytest.approx(1.0, abs=1e-9)
    assert report.unexplained_pct == pytest.approx(0.0, abs=1e-6)


def test_unexplained_variance_is_reported_for_a_noisy_response():
    # Signal variance is 1^2 + 1^2 = 2 and noise variance is 1, so the
    # population R^2 is 2/3.
    run = synthetic_run([1.0, 1.0], noise=1.0)
    report = sensitivity_analysis(run, "y")
    assert report.r_squared == pytest.approx(2.0 / 3.0, rel=0.08)
    assert report.unexplained_pct == pytest.approx(100 * (1 - report.r_squared),
                                                   rel=1e-9)
    assert any("nonlinearity" in note for note in report.notes)


def test_ranking_is_ordered_by_influence():
    report = sensitivity_analysis(synthetic_run([0.5, 3.0, 1.5]), "y")
    assert [e.parameter for e in report.entries][:3] == ["p1", "p2", "p0"]
    assert [e.rank for e in report.entries] == [1, 2, 3]


def test_sign_of_beta_follows_the_coefficient():
    report = sensitivity_analysis(synthetic_run([2.0, -2.0]), "y")
    betas = {e.parameter: e.beta_standardised for e in report.entries}
    assert betas["p0"] > 0 and betas["p1"] < 0


def test_output_change_per_input_sigma():
    run = synthetic_run([3.0, 1.0])
    report = sensitivity_analysis(run, "y")
    entry = next(e for e in report.entries if e.parameter == "p0")
    # d(output)/d(1 sigma of input) = beta * sigma(output).
    assert entry.d_output_d_sigma == pytest.approx(
        entry.beta_standardised * report.output_sigma, rel=1e-9)


def test_correlated_inputs_are_flagged():
    run = synthetic_run([1.0, 1.0], correlate=True)
    report = sensitivity_analysis(run, "y")
    assert report.max_input_correlation > 0.8
    assert any("not mutually independent" in note for note in report.notes)


def test_underdetermined_fit_falls_back_instead_of_reporting_a_fake_r_squared():
    """More parameters than samples would give a minimum-norm fit and a
    meaningless R^2 of exactly 1."""
    run = synthetic_run([1.0] * 8, n=6)
    report = sensitivity_analysis(run, "y")
    assert report.method == "correlation"
    assert any("underdetermined" in note for note in report.notes)


def test_correlation_method_is_labelled_as_a_ranking():
    report = sensitivity_analysis(synthetic_run([3.0, 1.0]), "y",
                                  method="correlation")
    assert report.method == "correlation"
    assert math.isnan(report.r_squared)
    assert [e.parameter for e in report.entries][:1] == ["p0"]


def test_unknown_method_and_measurement_are_rejected():
    run = synthetic_run([1.0])
    with pytest.raises(AnalysisError):
        sensitivity_analysis(run, "y", method="magic")
    with pytest.raises(AnalysisError):
        sensitivity_analysis(run, "nope")


def test_a_constant_measurement_has_no_drivers():
    run = synthetic_run([0.0, 0.0])
    report = sensitivity_analysis(run, "y")
    assert any("identical in every sample" in note for note in report.notes)


# ---------------------------------------------------------------------------
# on the real current mirror
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mirror_error_is_dominated_by_the_two_local_thresholds():
    """The textbook result: a current mirror's copy error is set by the local
    threshold mismatch of its two devices, and almost nothing else."""
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=600, seed=12345,
        workers=1))
    report = sensitivity_analysis(run, "ierr")

    top_two = {e.parameter for e in report.entries[:2]}
    assert top_two == {"M1.vth_local", "M2.vth_local"}
    share = sum(e.variance_contribution_pct for e in report.entries[:2])
    assert share > 80.0
    assert report.r_squared > 0.99

    # The two act in opposition: raising M1's threshold raises the mirrored
    # current, raising M2's lowers it.
    betas = {e.parameter: e.beta_standardised for e in report.entries}
    assert betas["M1.vth_local"] * betas["M2.vth_local"] < 0


@pytest.mark.slow
def test_global_variation_barely_affects_a_matched_mirror():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=600, seed=12345,
        workers=1))
    report = sensitivity_analysis(run, "ierr")
    globals_ = [e for e in report.entries if e.scope == "global"]
    assert globals_
    assert max(e.variance_contribution_pct for e in globals_) < 5.0


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

def test_rank_transform_handles_ties():
    x = np.array([[10.0], [20.0], [20.0], [40.0]])
    assert rank_transform(x).ravel() == pytest.approx([1.0, 2.5, 2.5, 4.0])


def test_pearson_and_spearman_agree_on_a_linear_relationship():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((500, 1))
    b = 2.0 * a + 0.01 * rng.standard_normal((500, 1))
    assert pearson_matrix(a, b)[0, 0] == pytest.approx(1.0, abs=0.01)
    assert spearman_matrix(a, b)[0, 0] == pytest.approx(1.0, abs=0.01)


def test_spearman_detects_a_monotone_nonlinearity_that_pearson_understates():
    x = np.linspace(0.1, 3.0, 400).reshape(-1, 1)
    y = np.exp(3.0 * x)
    pearson = pearson_matrix(x, y)[0, 0]
    spearman = spearman_matrix(x, y)[0, 0]
    assert spearman == pytest.approx(1.0, abs=1e-9)
    assert pearson < 0.85


def test_correlation_analysis_on_a_real_run():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=250, seed=99,
        workers=1))
    matrix = correlation_analysis(run)
    assert matrix.parameters == run.slot_names
    assert matrix.measurements == run.measurement_names
    assert len(matrix.pearson) == len(matrix.parameters)
    assert len(matrix.pearson[0]) == len(matrix.measurements)
    top = matrix.top_pairs("ierr", limit=2)
    assert {row["parameter"] for row in top} == {"M1.vth_local", "M2.vth_local"}


def test_correlation_reports_degenerate_columns_instead_of_hiding_them():
    run = synthetic_run([1.0, 0.0])
    for sample in run.samples:
        sample.slot_values["p1"] = 0.0        # no spread at all
    matrix = correlation_analysis(run)
    assert "p1" in matrix.degenerate_parameters
    assert any("zero spread" in note for note in matrix.notes)


def test_correlation_needs_enough_samples():
    run = synthetic_run([1.0], n=2)
    matrix = correlation_analysis(run)
    assert matrix.parameters == []
    assert any("at least 3" in note for note in matrix.notes)


def test_measurement_cross_correlation():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=200, seed=5,
        workers=1))
    result = measurement_correlation(run)
    names = result["measurements"]
    assert "iout" in names and "ierr" in names
    i_out = names.index("iout")
    i_err = names.index("ierr")
    # iout and ierr are two views of the same quantity, so they must be almost
    # perfectly correlated.
    assert result["pearson"][i_out][i_err] == pytest.approx(1.0, abs=0.01)
