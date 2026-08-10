"""DC, AC and transient solver behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.core.exceptions import (
    ConvergenceError,
    NumericalError,
    SimulationError,
)
from siliconstat.core.netlist import parse_netlist
from siliconstat.core.solver import (
    SolverOptions,
    ac_analysis,
    solve_dc,
    transient_analysis,
)

from conftest import EXAMPLE_NAMES, load

RC_LOWPASS = ("V1 in 0 DC 0 AC 1\nR1 in out 1k\nC1 out 0 1n\n"
              ".ac dec 30 100 10MEG\n.measure v V(out)\n")


# ---------------------------------------------------------------------------
# DC convergence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_every_example_converges(name):
    circuit = load(name)
    op = solve_dc(circuit)
    assert op.strategy in ("direct", "damped", "gmin", "source")
    assert op.iterations >= 2
    assert op.iterations <= 60, "convergence should not need this many iterations"
    assert math.isfinite(op.residual)
    assert np.all(np.isfinite(op.x))


def test_residual_is_far_below_the_requested_tolerance():
    """The converged answer should beat the tolerance, not just meet it."""
    circuit = load("current_mirror")
    op = solve_dc(circuit)
    reference = abs(op.i("M1"))
    assert op.residual < 1e-6 * reference


def test_operating_point_reports_regions_and_small_signal_parameters():
    op = solve_dc(load("current_mirror"))
    m1 = op.device_ops["M1"]
    assert m1["region"] == "saturation"
    assert m1["gm"] > 0 and m1["ro"] > 0
    assert m1["gm_over_id"] == pytest.approx(m1["gm"] / m1["id"], rel=1e-9)


def test_warm_start_from_a_previous_solution_converges_faster():
    circuit = load("current_mirror")
    cold = solve_dc(circuit)
    warm = solve_dc(circuit, x0=cold.x)
    assert warm.iterations <= cold.iterations
    assert warm.v("nref") == pytest.approx(cold.v("nref"), rel=1e-9)


# ---------------------------------------------------------------------------
# failure modes -- never a silent wrong answer
# ---------------------------------------------------------------------------

def test_floating_node_raises_a_specific_numerical_error():
    """A node with no DC path to ground makes the matrix singular.  That is a
    topology fault, so it must be reported as such rather than as a generic
    convergence failure."""
    circuit = parse_netlist("V1 a 0 1\nR1 a 0 1k\nR2 c d 1k\n.measure v V(a)\n")
    with pytest.raises(NumericalError) as excinfo:
        solve_dc(circuit)
    message = str(excinfo.value).lower()
    assert "singular" in message
    assert "floating" in message


def test_shorted_voltage_source_loop_is_reported():
    circuit = parse_netlist("V1 a 0 1\nV2 a 0 2\nR1 a 0 1k\n.measure v V(a)\n")
    with pytest.raises(NumericalError):
        solve_dc(circuit)


def test_iteration_limit_produces_a_convergence_error_with_diagnostics():
    """Force the limit down so every strategy runs out of iterations."""
    circuit = load("two_stage_opamp")
    options = SolverOptions(max_iter=2, gmin_steps=2, source_steps=2)
    with pytest.raises(ConvergenceError) as excinfo:
        solve_dc(circuit, opts=options)
    error = excinfo.value
    assert error.strategies == ["direct", "damped", "gmin", "source"]
    assert "diagnostics" in str(error)


def test_convergence_error_carries_structured_fields():
    circuit = load("current_mirror")
    with pytest.raises(ConvergenceError) as excinfo:
        solve_dc(circuit, opts=SolverOptions(max_iter=1, gmin_steps=1,
                                             source_steps=1))
    assert excinfo.value.iterations >= 1


# ---------------------------------------------------------------------------
# AC analysis
# ---------------------------------------------------------------------------

def test_ac_of_an_rc_lowpass_matches_the_analytic_transfer_function():
    circuit = parse_netlist(RC_LOWPASS)
    freqs = np.logspace(2, 7, 40)
    result = ac_analysis(circuit, freqs)
    gain = result.gain("out", "in")
    expected = 1.0 / (1.0 + 1j * 2 * np.pi * freqs * 1e3 * 1e-9)
    assert np.allclose(gain, expected, rtol=1e-9, atol=1e-12)


def test_ac_three_db_point_of_an_rc_lowpass():
    circuit = parse_netlist(RC_LOWPASS)
    f3db_expected = 1.0 / (2 * math.pi * 1e3 * 1e-9)      # 159.155 kHz
    result = ac_analysis(circuit, [f3db_expected])
    magnitude = abs(result.gain("out", "in")[0])
    assert magnitude == pytest.approx(1 / math.sqrt(2), rel=1e-6)


def test_ac_phase_of_an_rc_lowpass():
    circuit = parse_netlist(RC_LOWPASS)
    f3db = 1.0 / (2 * math.pi * 1e3 * 1e-9)
    phase = np.angle(ac_analysis(circuit, [f3db]).gain("out", "in")[0], deg=True)
    assert phase == pytest.approx(-45.0, abs=1e-4)


def test_ac_of_an_rl_highpass():
    circuit = parse_netlist("V1 in 0 DC 0 AC 1\nR1 in out 1k\nL1 out 0 1m\n"
                            ".measure v V(out)\n")
    freqs = np.array([1e3, 1e5, 1e6])
    gain = ac_analysis(circuit, freqs).gain("out", "in")
    z_l = 1j * 2 * np.pi * freqs * 1e-3
    expected = z_l / (1e3 + z_l)
    assert np.allclose(gain, expected, rtol=1e-9)


def test_ac_without_an_ac_source_is_rejected():
    circuit = parse_netlist("V1 in 0 1\nR1 in out 1k\nC1 out 0 1n\n"
                            ".measure v V(out)\n")
    with pytest.raises(SimulationError) as excinfo:
        ac_analysis(circuit, [1e3])
    assert "AC magnitude" in str(excinfo.value)


@pytest.mark.parametrize("freqs", [[], [0.0], [-1.0]])
def test_invalid_ac_frequencies_are_rejected(freqs):
    circuit = parse_netlist(RC_LOWPASS)
    with pytest.raises(SimulationError):
        ac_analysis(circuit, freqs)


# ---------------------------------------------------------------------------
# transient analysis
# ---------------------------------------------------------------------------

def test_transient_rc_step_matches_the_analytic_exponential():
    """v(t) = V*(1 - exp(-t/RC)) for a step into an RC."""
    circuit = parse_netlist(
        "V1 in 0 DC 0 PULSE(0 1 0 1p 1p 100u 200u)\n"
        "R1 in out 1k\nC1 out 0 1n\n.tran 5n 5u\n.measure v V(out)\n")
    result = transient_analysis(circuit, tstop=5e-6, tstep=5e-9)
    tau = 1e3 * 1e-9
    v = result.v("out")
    analytic = 1.0 - np.exp(-result.time / tau)
    # Trapezoidal integration at dt/tau = 0.005 is accurate to well under 0.1 %.
    assert np.max(np.abs(v - analytic)) < 1e-4
    # After 5 time constants the step has reached 1 - exp(-5) = 99.326 %.
    assert v[-1] == pytest.approx(1.0 - math.exp(-5.0), abs=1e-4)


def test_transient_rc_discharge():
    circuit = parse_netlist(
        "V1 in 0 DC 1 PULSE(1 0 0 1p 1p 100u 200u)\n"
        "R1 in out 1k\nC1 out 0 1n\n.tran 5n 5u\n.measure v V(out)\n")
    result = transient_analysis(circuit, tstop=5e-6, tstep=5e-9)
    analytic = np.exp(-result.time / 1e-6)
    assert np.max(np.abs(result.v("out") - analytic)) < 1e-4


def test_transient_starts_from_the_dc_operating_point():
    circuit = parse_netlist(
        "V1 in 0 DC 0.4 PULSE(0.4 1 1u 1p 1p 100u 200u)\n"
        "R1 in out 1k\nC1 out 0 1n\n.tran 10n 2u\n.measure v V(out)\n")
    result = transient_analysis(circuit, tstop=2e-6, tstep=1e-8)
    assert result.v("out")[0] == pytest.approx(0.4, abs=1e-9)


def test_transient_rl_current_ramp():
    """i(t) = (V/R)*(1 - exp(-tR/L)) into a series RL."""
    circuit = parse_netlist(
        "V1 in 0 DC 0 PULSE(0 1 0 1p 1p 100u 200u)\n"
        "R1 in out 100\nL1 out 0 1m\n.tran 1u 50u\n.measure v V(out)\n")
    result = transient_analysis(circuit, tstop=5e-5, tstep=1e-7)
    inductor = circuit.device("L1")
    ctx = result.op.ctx
    ctx.n_nodes = circuit.n_nodes
    currents = np.array([inductor.current(row, ctx) for row in result.x])
    analytic = (1.0 / 100.0) * (1.0 - np.exp(-result.time * 100.0 / 1e-3))
    assert np.max(np.abs(currents - analytic)) < 2e-5


@pytest.mark.parametrize("tstop,tstep", [(0.0, 1e-9), (1e-6, 0.0), (1e-6, -1e-9)])
def test_invalid_transient_windows_are_rejected(tstop, tstep):
    circuit = parse_netlist(RC_LOWPASS)
    with pytest.raises(SimulationError):
        transient_analysis(circuit, tstop=tstop, tstep=tstep)


def test_absurd_transient_step_count_is_refused():
    circuit = parse_netlist(RC_LOWPASS)
    with pytest.raises(SimulationError) as excinfo:
        transient_analysis(circuit, tstop=1.0, tstep=1e-12)
    assert "increase tstep" in str(excinfo.value)


@pytest.mark.slow
def test_inverter_transient_converges_through_both_edges():
    """The switching edges cross the region boundary, which is where the
    frozen-Meyer-capacitance treatment matters."""
    circuit = load("inverter_transient")
    result = transient_analysis(circuit, tstop=8e-9, tstep=2e-11)
    v = result.v("out")
    assert np.all(np.isfinite(v))
    assert v.max() > 1.7          # pulled up to the rail
    assert v.min() < 0.1          # pulled down to the rail


# ---------------------------------------------------------------------------
# solver options
# ---------------------------------------------------------------------------

def test_options_are_read_from_the_netlist():
    circuit = parse_netlist("V1 a 0 1\nR1 a 0 1k\n"
                            ".option reltol=1e-8 vntol=1e-10 max_iter=42\n"
                            ".measure v V(a)\n")
    options = SolverOptions.from_options(circuit.options)
    assert options.reltol == pytest.approx(1e-8)
    assert options.vntol == pytest.approx(1e-10)
    assert options.max_iter == 42


def test_default_tolerances_are_tight_enough_for_mismatch_work():
    """Solver noise must sit well below the effects being characterised."""
    options = SolverOptions()
    assert options.reltol <= 1e-6
    assert options.abstol <= 1e-15


def test_gmin_stepping_can_be_selected_explicitly():
    circuit = load("current_mirror")
    op = solve_dc(circuit, opts=SolverOptions(strategies=("gmin",)))
    assert op.strategy == "gmin"
    assert op.v("nref") == pytest.approx(solve_dc(circuit).v("nref"), rel=1e-6)


def test_source_stepping_can_be_selected_explicitly():
    circuit = load("current_mirror")
    op = solve_dc(circuit, opts=SolverOptions(strategies=("source",)))
    assert op.strategy == "source"
    assert op.v("nref") == pytest.approx(solve_dc(circuit).v("nref"), rel=1e-6)


def test_unknown_strategy_is_reported():
    circuit = load("current_mirror")
    with pytest.raises((SimulationError, ConvergenceError)):
        solve_dc(circuit, opts=SolverOptions(strategies=("teleport",)))
