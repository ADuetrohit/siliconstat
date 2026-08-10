"""The Monte Carlo engine: accounting, failure handling and sampling modes.

The rule this module enforces hardest is that **nothing is discarded**.  A run
that attempts N samples reports N records, each with a status and, when it
failed, a reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from siliconstat.core.exceptions import VariationError
from siliconstat.core.solver import SolverOptions
from siliconstat.mc import (
    MonteCarloConfig,
    SampleStatus,
    latin_hypercube_normals,
    run_monte_carlo,
    simulate_nominal,
)
from siliconstat.variation import (
    ParameterVariation,
    VariationModel,
    default_mismatch_model,
)

from conftest import load


def config_for(circuit, **kwargs) -> MonteCarloConfig:
    defaults = dict(variation=default_mismatch_model(circuit), samples=100,
                    seed=4242, workers=1)
    defaults.update(kwargs)
    return MonteCarloConfig(**defaults)


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------

def test_counters_always_add_up():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=80))
    counters = run.counters
    assert counters.total == 80 == len(run.samples)
    assert counters.successful + counters.failed == counters.total
    assert counters.convergence_failures + counters.numerical_errors + \
        counters.invalid_measurements + counters.other_errors == counters.failed


def test_every_attempted_sample_is_recorded_with_an_index():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=50))
    assert [s.index for s in run.samples] == list(range(50))


def test_success_rate_is_reported():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=40))
    assert run.counters.to_dict()["success_rate"] == pytest.approx(
        run.counters.successful / 40)


def test_nominal_simulation_is_stored_alongside_the_samples():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=20))
    assert run.nominal_status == "ok"
    assert run.nominal["ierr"] == pytest.approx(0.082, abs=0.01)
    # The nominal must be the *unperturbed* circuit, not sample 0.
    assert run.nominal["ierr"] != run.samples[0].measurements["ierr"]


def test_simulate_nominal_directly():
    circuit = load("rc_divider")
    values, status = simulate_nominal(circuit, config_for(circuit, samples=1))
    assert status == "ok"
    assert values["vmid"] == pytest.approx(3.75, rel=1e-9)


# ---------------------------------------------------------------------------
# failures are recorded, never dropped
# ---------------------------------------------------------------------------

def test_convergence_failures_are_recorded_with_a_reason():
    """Starve the solver of iterations so every sample fails to converge."""
    circuit = load("two_stage_opamp")
    config = config_for(circuit, samples=6,
                        solver=SolverOptions(max_iter=2, gmin_steps=2,
                                             source_steps=2))
    run = run_monte_carlo(circuit, config)
    assert run.counters.total == 6
    assert run.counters.successful == 0
    assert run.counters.convergence_failures == 6
    for sample in run.samples:
        assert sample.status == SampleStatus.CONVERGENCE_FAILURE
        assert sample.failure_reason
        assert sample.passed is None      # a failed sample has no verdict


def test_variation_errors_are_recorded_per_sample_and_the_run_continues():
    """A beta sigma large enough to invert the device fails only some samples;
    the run must complete and report exactly which ones."""
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="beta", scope="local", sigma_pct=60.0,
                           targets="type:mosfet")])
    run = run_monte_carlo(circuit, config_for(circuit, samples=200,
                                              variation=model, seed=7))
    failures = [s for s in run.samples
                if s.status == SampleStatus.VARIATION_ERROR]
    assert failures, "a 60 % beta sigma should produce some non-physical draws"
    assert run.counters.total == 200
    assert run.counters.successful > 0
    for sample in failures:
        assert "physical" in sample.failure_reason or "<= 0" in sample.failure_reason


def test_failure_breakdown_groups_and_ranks_reasons():
    circuit = load("two_stage_opamp")
    run = run_monte_carlo(circuit, config_for(
        circuit, samples=4,
        solver=SolverOptions(max_iter=2, gmin_steps=2, source_steps=2)))
    breakdown = run.failure_breakdown()
    assert breakdown
    assert sum(row["count"] for row in breakdown) == run.counters.failed
    assert breakdown[0]["percent"] == pytest.approx(
        100.0 * breakdown[0]["count"] / run.counters.total)
    assert breakdown[0]["status"] == SampleStatus.CONVERGENCE_FAILURE


def test_failed_samples_are_excluded_from_value_arrays_but_kept_in_records():
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="beta", scope="local", sigma_pct=60.0,
                           targets="type:mosfet")])
    run = run_monte_carlo(circuit, config_for(circuit, samples=150,
                                              variation=model, seed=7))
    usable = run.values("ierr")
    assert len(usable) == run.counters.successful
    assert len(run.samples) == 150
    assert np.all(np.isfinite(usable))
    assert len(run.failed_samples()) == run.counters.failed


# ---------------------------------------------------------------------------
# specifications
# ---------------------------------------------------------------------------

def test_spec_verdicts_are_recorded_per_sample():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=60))
    for sample in run.successful_samples():
        assert set(sample.spec_pass) == {s.describe() for s in circuit.specs}
        assert sample.passed == all(sample.spec_pass.values())


def test_spec_verdicts_agree_with_the_measured_values():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=60))
    spec = next(s for s in circuit.specs if s.measure == "ierr" and s.op == "<=")
    for sample in run.successful_samples():
        expected = sample.measurements["ierr"] <= spec.value
        assert sample.spec_pass[spec.describe()] == expected


def test_specs_can_be_overridden_at_run_time():
    from siliconstat.core.circuit import SpecLimit

    circuit = load("current_mirror")
    tight = [SpecLimit(measure="ierr", op="<=", value=0.0)]
    run = run_monte_carlo(circuit, config_for(circuit, samples=40, specs=tight))
    assert set(run.samples[0].spec_pass) == {"ierr <= 0"}


def test_a_circuit_without_specs_produces_no_verdict():
    from siliconstat.core.netlist import parse_netlist

    circuit = parse_netlist("V1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n.measure v V(b)\n")
    model = VariationModel(variations=[
        ParameterVariation(parameter="r", scope="local", sigma_pct=1.0,
                           targets="type:resistor")])
    run = run_monte_carlo(circuit, config_for(circuit, samples=10,
                                              variation=model))
    assert all(s.passed is None for s in run.samples)


# ---------------------------------------------------------------------------
# sampling modes
# ---------------------------------------------------------------------------

def test_latin_hypercube_design_is_stratified():
    """Each dimension must contain exactly one point per equal-probability bin."""
    from scipy.stats import norm

    n, k = 200, 4
    z = latin_hypercube_normals(n, k, np.random.default_rng(0))
    assert z.shape == (n, k)
    for j in range(k):
        bins = np.floor(norm.cdf(z[:, j]) * n).astype(int)
        assert len(np.unique(bins)) == n      # one sample per stratum


def test_latin_hypercube_marginals_are_better_behaved_than_plain_sampling():
    """LHS trades independence for a much better-conditioned marginal at a
    fixed budget; the sample mean should sit closer to zero."""
    n, k = 120, 3
    rng = np.random.default_rng(5)
    lhs = latin_hypercube_normals(n, k, rng)
    plain = np.random.default_rng(5).standard_normal((n, k))
    assert np.max(np.abs(lhs.mean(axis=0))) < np.max(np.abs(plain.mean(axis=0)))


def test_latin_hypercube_runs_end_to_end():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=100,
                                              sampling="latin_hypercube"))
    assert run.counters.total == 100
    assert run.counters.successful == 100


def test_zero_dimensional_latin_hypercube():
    assert latin_hypercube_normals(10, 0, np.random.default_rng(0)).shape == (10, 0)


# ---------------------------------------------------------------------------
# configuration validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(samples=0),
    dict(samples=-5),
    dict(samples=10_000_000),
    dict(sampling="sobol"),
    dict(workers=0),
    dict(workers=100_000),
])
def test_invalid_configurations_are_rejected(kwargs):
    circuit = load("current_mirror")
    with pytest.raises(VariationError):
        config_for(circuit, **kwargs)


def test_effective_workers_never_exceeds_the_sample_count():
    circuit = load("current_mirror")
    assert config_for(circuit, samples=3, workers=8).effective_workers == 3


def test_config_serialises_for_the_reproduction_record():
    circuit = load("current_mirror")
    data = config_for(circuit, samples=25).to_dict()
    assert data["samples"] == 25
    assert data["variation"]["mode"] == "process+mismatch"
    assert "reltol" in data["solver"]


# ---------------------------------------------------------------------------
# progress reporting and dataframes
# ---------------------------------------------------------------------------

def test_progress_callback_is_invoked_and_finishes_at_the_total():
    circuit = load("current_mirror")
    seen: list[dict] = []
    run_monte_carlo(circuit, config_for(circuit, samples=50, progress_every=10),
                    progress=seen.append)
    assert seen
    assert seen[-1]["completed"] == 50
    assert seen[-1]["successful"] + seen[-1]["failed"] == 50


def test_cancellation_stops_the_run_early():
    circuit = load("current_mirror")
    state = {"n": 0}

    def should_cancel() -> bool:
        state["n"] += 1
        return state["n"] > 12

    run = run_monte_carlo(circuit, config_for(circuit, samples=500),
                          should_cancel=should_cancel)
    assert run.counters.total < 500


def test_dataframe_has_the_documented_columns():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=15))
    frame = run.to_dataframe()
    assert len(frame) == 15
    for column in ("sample_id", "seed", "simulation_status", "failure_reason",
                   "pass_fail", "dc_iterations", "runtime_s"):
        assert column in frame.columns
    assert "meas.ierr" in frame.columns
    assert "var.M1.vth_local" in frame.columns
    assert "dev.M1.dvth" in frame.columns
    assert any(c.startswith("spec.") for c in frame.columns)


def test_slot_matrix_shape_matches_the_usable_samples():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=30))
    matrix, names = run.slot_matrix()
    assert matrix.shape == (run.counters.successful, len(names))
    assert names == run.slot_names


def test_run_serialises_with_and_without_samples():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, config_for(circuit, samples=8))
    full = run.to_dict()
    assert len(full["samples"]) == 8
    assert "samples" not in run.to_dict(include_samples=False)
