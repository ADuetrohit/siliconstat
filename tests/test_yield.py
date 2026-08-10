"""Yield accounting: denominators, confidence intervals and capability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.analysis import analyse_run, analyse_yield
from siliconstat.analysis.yield_analysis import process_capability, wilson_interval
from siliconstat.core.circuit import SpecLimit
from siliconstat.core.solver import SolverOptions
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.variation import default_mismatch_model

from conftest import load


def mirror_run(samples: int = 400, seed: int = 12345, **kwargs):
    circuit = load("current_mirror")
    config = MonteCarloConfig(variation=default_mismatch_model(circuit),
                              samples=samples, seed=seed, workers=1, **kwargs)
    return circuit, run_monte_carlo(circuit, config)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(80, 100)
    assert low < 80.0 < high


def test_wilson_interval_narrows_as_the_sample_grows():
    narrow = wilson_interval(900, 1000)
    wide = wilson_interval(90, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
    assert (wide[1] - wide[0]) / (narrow[1] - narrow[0]) == pytest.approx(
        math.sqrt(10.0), rel=0.2)


def test_wilson_interval_stays_inside_zero_to_one_hundred():
    """This is why Wilson is used instead of the normal approximation, which
    would report an upper bound above 100 % here."""
    low, high = wilson_interval(50, 50)
    assert 0.0 <= low <= 100.0 and high == pytest.approx(100.0, abs=1e-9)
    assert low > 90.0          # 50/50 is not proof of 100 % yield
    low0, high0 = wilson_interval(0, 50)
    assert low0 == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < high0 < 10.0


def test_wilson_interval_on_an_empty_denominator():
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_wilson_interval_matches_a_hand_computed_value():
    # p = 0.9, n = 100, z = 1.959964, by direct substitution into the
    # Wilson formula: centre 0.885195, margin 0.059570.
    low, high = wilson_interval(90, 100)
    assert low == pytest.approx(82.5634, abs=0.001)
    assert high == pytest.approx(94.4771, abs=0.001)


# ---------------------------------------------------------------------------
# process capability
# ---------------------------------------------------------------------------

def test_cpk_of_a_centred_process_equals_cp():
    values = np.random.default_rng(0).normal(0.0, 1.0, 20_000)
    capability = process_capability(values, lower=-3.0, upper=3.0)
    assert capability["cp"] == pytest.approx(1.0, rel=0.03)
    assert capability["cpk"] == pytest.approx(1.0, rel=0.05)


def test_cpk_penalises_an_off_centre_process():
    values = np.random.default_rng(1).normal(1.0, 1.0, 20_000)
    capability = process_capability(values, lower=-3.0, upper=3.0)
    assert capability["cpk"] < capability["cp"]
    assert capability["cpu"] == pytest.approx(2.0 / 3.0, rel=0.05)


def test_one_sided_capability():
    values = np.random.default_rng(2).normal(0.0, 1.0, 10_000)
    capability = process_capability(values, lower=None, upper=3.0)
    assert math.isinf(capability["cpl"])
    assert capability["cpk"] == pytest.approx(capability["cpu"], rel=1e-9)
    assert math.isnan(capability["cp"])


def test_capability_of_a_constant_process_is_infinite():
    capability = process_capability(np.full(100, 1.0), lower=0.0, upper=2.0)
    assert math.isinf(capability["cpk"])


def test_capability_needs_at_least_two_points():
    capability = process_capability(np.array([1.0]), lower=0.0, upper=2.0)
    assert math.isnan(capability["cpk"])


# ---------------------------------------------------------------------------
# the yield report
# ---------------------------------------------------------------------------

def test_per_spec_and_combined_yield_are_consistent():
    circuit, run = mirror_run()
    report = analyse_yield(run)
    assert report.attempted == 400
    assert report.successful == run.counters.successful
    assert len(report.per_spec) == len(circuit.specs)

    for entry in report.per_spec:
        assert entry.passing + entry.failing == entry.denominator
        assert entry.denominator == report.successful
        assert entry.yield_pct == pytest.approx(
            100.0 * entry.passing / entry.denominator)
        # A 100 %-passing spec puts the point estimate on the interval's
        # upper edge, where the two differ only by float rounding.
        assert entry.ci95_low <= entry.yield_pct <= entry.ci95_high + 1e-9

    # The combined pass count can never exceed the weakest individual one.
    assert report.combined_passing <= min(e.passing for e in report.per_spec)


def test_combined_yield_is_computed_sample_by_sample_not_multiplied():
    """Specs on the same measurement fail together, so the true combined yield
    differs from the product of the individual yields."""
    _circuit, run = mirror_run()
    report = analyse_yield(run)
    assert report.independent_product_pct != pytest.approx(
        report.combined_yield_over_successful, abs=0.5)
    assert any("product of the individual yields" in note
               for note in report.notes)


def test_combined_yield_matches_a_direct_count():
    _circuit, run = mirror_run(samples=200)
    report = analyse_yield(run)
    direct = sum(1 for s in run.successful_samples() if s.passed)
    assert report.combined_passing == direct


def test_both_denominators_are_reported():
    _circuit, run = mirror_run(samples=200)
    report = analyse_yield(run)
    assert report.denominator_basis == "successful simulations"
    assert report.combined_yield_over_successful == pytest.approx(
        100.0 * report.combined_passing / report.successful)
    assert report.combined_yield_over_attempted == pytest.approx(
        100.0 * report.combined_passing / report.attempted)


def test_the_limiting_specification_is_identified():
    _circuit, run = mirror_run()
    report = analyse_yield(run)
    worst = min(report.per_spec, key=lambda e: e.yield_pct)
    assert report.limiting_spec == worst.description


def test_margin_in_sigmas_is_reported():
    _circuit, run = mirror_run()
    report = analyse_yield(run)
    entry = next(e for e in report.per_spec if e.measure == "ierr" and e.op == "<=")
    values = run.values("ierr")
    expected = (entry.limit - values.mean()) / values.std(ddof=1)
    assert entry.margin_sigma == pytest.approx(expected, rel=1e-6)


def test_worst_sample_is_reported_per_spec():
    _circuit, run = mirror_run()
    report = analyse_yield(run)
    upper = next(e for e in report.per_spec if e.op == "<=" and e.measure == "ierr")
    assert upper.worst_value == pytest.approx(run.values("ierr").max())
    lower = next(e for e in report.per_spec if e.op == ">=" and e.measure == "ierr")
    assert lower.worst_value == pytest.approx(run.values("ierr").min())


def test_a_tighter_specification_yields_less():
    circuit = load("current_mirror")
    _c, run = mirror_run(samples=300)
    loose = analyse_yield(run, [SpecLimit(measure="ierr", op="<=", value=6.0)])
    tight = analyse_yield(run, [SpecLimit(measure="ierr", op="<=", value=1.0)])
    assert tight.combined_yield_over_successful < \
        loose.combined_yield_over_successful


def test_a_run_without_specifications_says_so_instead_of_inventing_a_number():
    from siliconstat.core.netlist import parse_netlist
    from siliconstat.variation import ParameterVariation, VariationModel

    circuit = parse_netlist("V1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n.measure v V(b)\n")
    model = VariationModel(variations=[
        ParameterVariation(parameter="r", scope="local", sigma_pct=1.0,
                           targets="type:resistor")])
    run = run_monte_carlo(circuit, MonteCarloConfig(variation=model, samples=20,
                                                    seed=1, workers=1))
    report = analyse_yield(run)
    assert report.per_spec == []
    assert math.isnan(report.combined_yield_over_successful)
    assert any("No specifications" in note for note in report.notes)


def test_a_run_with_no_usable_samples_reports_undefined_yield():
    circuit = load("two_stage_opamp")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=4, seed=1, workers=1,
        solver=SolverOptions(max_iter=2, gmin_steps=2, source_steps=2)))
    report = analyse_yield(run)
    assert report.successful == 0
    assert any("undefined" in note for note in report.notes)


def test_failed_simulations_are_flagged_in_the_notes():
    from siliconstat.variation import ParameterVariation, VariationModel

    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="beta", scope="local", sigma_pct=60.0,
                           targets="type:mosfet")])
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=model, samples=150, seed=7, workers=1))
    report = analyse_yield(run)
    if run.counters.failed:
        assert any("did not produce a usable result" in note
                   for note in report.notes)


def test_yield_report_serialises():
    _circuit, run = mirror_run(samples=100)
    data = analyse_yield(run).to_dict()
    assert "per_spec" in data and data["per_spec"]
    assert set(data["per_spec"][0]) >= {
        "key", "measure", "op", "limit", "passing", "failing", "denominator",
        "yield_pct", "ci95_low", "ci95_high", "cpk"}


def test_summary_lines_are_human_readable():
    _circuit, run = mirror_run(samples=100)
    lines = analyse_yield(run).summary_lines()
    assert any("COMBINED" in line for line in lines)
    assert all("%" in line for line in lines)


def test_analysis_bundles_the_yield_report():
    _circuit, run = mirror_run(samples=100)
    analysis = analyse_run(run)
    assert analysis.yield_report.per_spec
    assert analysis.to_dict()["yield"]["combined_yield_over_successful"] == \
        pytest.approx(analysis.yield_report.combined_yield_over_successful)
