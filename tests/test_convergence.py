"""Statistical convergence of the Monte Carlo estimates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.analysis import convergence_analysis, required_samples_for_margin
from siliconstat.core.exceptions import AnalysisError
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.variation import default_mismatch_model

from conftest import load


@pytest.fixture(scope="module")
def mirror_run():
    circuit = load("current_mirror")
    return run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=600, seed=2024,
        workers=1))


# ---------------------------------------------------------------------------
# the trace
# ---------------------------------------------------------------------------

def test_trace_checkpoints_are_increasing_and_end_at_the_total(mirror_run):
    trace = convergence_analysis(mirror_run, "ierr", points=20)
    assert trace.n == sorted(trace.n)
    assert trace.n[-1] == mirror_run.counters.successful
    assert len(trace.mean) == len(trace.n) == len(trace.std)


def test_running_mean_ends_at_the_full_sample_mean(mirror_run):
    trace = convergence_analysis(mirror_run, "ierr")
    assert trace.final_mean == pytest.approx(
        float(np.mean(mirror_run.values("ierr"))), rel=1e-9)
    assert trace.final_std == pytest.approx(
        float(np.std(mirror_run.values("ierr"), ddof=1)), rel=1e-9)


def test_confidence_band_narrows_as_one_over_root_n(mirror_run):
    trace = convergence_analysis(mirror_run, "ierr", points=40)
    widths = np.array(trace.mean_ci_high) - np.array(trace.mean_ci_low)
    n = np.array(trace.n, dtype=float)
    # width * sqrt(n) should be roughly constant once n is not tiny.
    product = widths * np.sqrt(n)
    tail = product[len(product) // 3:]
    assert tail.std() / tail.mean() < 0.25
    assert widths[-1] < widths[0]


def test_the_band_contains_the_final_mean_most_of_the_time(mirror_run):
    trace = convergence_analysis(mirror_run, "ierr", points=40)
    inside = sum(1 for lo, hi in zip(trace.mean_ci_low, trace.mean_ci_high)
                 if lo <= trace.final_mean <= hi)
    assert inside >= 0.7 * len(trace.n)


def test_yield_trace_is_present_when_specs_exist(mirror_run):
    trace = convergence_analysis(mirror_run, "ierr")
    assert trace.yield_pct
    assert all(0.0 <= y <= 100.0 for y in trace.yield_pct)
    assert trace.final_yield_pct == pytest.approx(trace.yield_pct[-1])
    for lo, y, hi in zip(trace.yield_ci_low, trace.yield_pct, trace.yield_ci_high):
        assert lo <= y + 1e-9 and y <= hi + 1e-9


def test_settling_detection(mirror_run):
    trace = convergence_analysis(mirror_run, "ierr", settle_tolerance_pct=5.0)
    for settled in (trace.std_settled_at, trace.yield_settled_at):
        if settled is not None:
            assert settled in trace.n


def test_a_tighter_tolerance_settles_no_earlier(mirror_run):
    loose = convergence_analysis(mirror_run, "ierr", settle_tolerance_pct=10.0)
    tight = convergence_analysis(mirror_run, "ierr", settle_tolerance_pct=0.5)
    if loose.std_settled_at and tight.std_settled_at:
        assert tight.std_settled_at >= loose.std_settled_at


def test_a_measurement_that_has_not_settled_says_so():
    """A deliberately tiny run cannot have settled."""
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=12, seed=1,
        workers=1))
    trace = convergence_analysis(run, "ierr", settle_tolerance_pct=0.1)
    assert trace.mean_settled_at is None or trace.mean_settled_at == trace.n[-1]


def test_trace_defaults_to_the_first_measurement(mirror_run):
    assert convergence_analysis(mirror_run).measurement == \
        mirror_run.measurement_names[0]


def test_unknown_measurement_is_rejected(mirror_run):
    with pytest.raises(AnalysisError):
        convergence_analysis(mirror_run, "nope")


def test_too_few_samples_is_reported_not_crashed():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=2, seed=1,
        workers=1))
    trace = convergence_analysis(run, "ierr")
    assert trace.n == []
    assert any("at least 3" in note for note in trace.notes)


def test_trace_serialises(mirror_run):
    data = convergence_analysis(mirror_run, "ierr").to_dict()
    for key in ("n", "mean", "mean_ci_low", "mean_ci_high", "std", "yield_pct",
                "final_mean", "final_std", "mean_settled_at"):
        assert key in data


# ---------------------------------------------------------------------------
# sample-count planning
# ---------------------------------------------------------------------------

def test_required_samples_follows_the_binomial_formula():
    """n = z^2 p(1-p) / m^2."""
    n = required_samples_for_margin(90.0, 1.0)
    expected = (1.959964 ** 2) * 0.9 * 0.1 / (0.01 ** 2)
    assert n == pytest.approx(math.ceil(expected), rel=0.01)


def test_a_tighter_margin_needs_a_hundred_times_more_samples():
    coarse = required_samples_for_margin(95.0, 1.0)
    fine = required_samples_for_margin(95.0, 0.1)
    assert fine / coarse == pytest.approx(100.0, rel=0.05)


def test_a_more_extreme_yield_needs_fewer_samples():
    assert required_samples_for_margin(99.0, 0.5) < \
        required_samples_for_margin(50.0, 0.5)


def test_higher_confidence_needs_more_samples():
    assert required_samples_for_margin(90.0, 1.0, confidence=0.99) > \
        required_samples_for_margin(90.0, 1.0, confidence=0.95)


@pytest.mark.parametrize("yield_pct,margin", [(-1.0, 1.0), (101.0, 1.0),
                                              (90.0, 0.0), (90.0, -1.0)])
def test_invalid_planning_inputs_are_rejected(yield_pct, margin):
    with pytest.raises(AnalysisError):
        required_samples_for_margin(yield_pct, margin)
