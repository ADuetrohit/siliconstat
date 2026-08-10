"""The measurement framework.

The behaviour that matters most here is the failure path: a measurement that
cannot be computed must yield NaN *and a reason*, never a plausible-looking
number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.core.netlist import parse_netlist
from siliconstat.core.solver import solve_dc
from siliconstat.measure import MeasurementContext, evaluate_measurements

from conftest import load

DIVIDER = ("V1 in 0 DC 1 AC 1\nR1 in out 1k\nR2 out 0 1k\n"
           ".measure vo V(out)\n")

RC = ("V1 in 0 DC 0 AC 1\nR1 in out 1k\nC1 out 0 1n\n"
      ".ac dec 30 100 100MEG\n")


# ---------------------------------------------------------------------------
# DC measurement kinds
# ---------------------------------------------------------------------------

def test_node_voltage_and_difference():
    circuit = parse_netlist(DIVIDER + ".measure vd V(in,out)\n")
    values = evaluate_measurements(circuit).values
    assert values["vo"] == pytest.approx(0.5, rel=1e-12)
    assert values["vd"] == pytest.approx(0.5, rel=1e-12)


def test_branch_current():
    circuit = parse_netlist(DIVIDER + ".measure i1 I(R1)\n")
    assert evaluate_measurements(circuit).values["i1"] == pytest.approx(0.5e-3,
                                                                       rel=1e-12)


def test_device_power_and_total_supply_power():
    circuit = parse_netlist(DIVIDER + ".measure pr I(R1)\n"
                            ".measure p1 P(R1)\n.measure pt P(total)\n")
    values = evaluate_measurements(circuit).values
    assert values["p1"] == pytest.approx(0.5 ** 2 / 1e3, rel=1e-12)
    assert values["pt"] == pytest.approx(1.0 * 0.5e-3, rel=1e-12)


def test_expression_measurements_see_earlier_measurements():
    circuit = parse_netlist(
        "V1 in 0 1\nR1 in out 1k\nR2 out 0 3k\n"
        ".measure vi V(in)\n.measure vo V(out)\n"
        ".measure ratio EXPR vo/vi\n.measure pct EXPR 100*ratio\n")
    values = evaluate_measurements(circuit).values
    assert values["ratio"] == pytest.approx(0.75, rel=1e-12)
    assert values["pct"] == pytest.approx(75.0, rel=1e-12)


def test_expression_supports_engineering_notation():
    circuit = parse_netlist(
        "V1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n"
        ".measure i1 I(R1)\n.measure err EXPR 100*(i1-500u)/500u\n")
    assert evaluate_measurements(circuit).values["err"] == pytest.approx(0.0,
                                                                        abs=1e-9)


def test_only_one_dc_solve_is_performed_for_many_dc_measurements():
    circuit = parse_netlist(DIVIDER + ".measure i1 I(R1)\n.measure pt P(total)\n"
                            ".measure r EXPR vo/1\n")
    result = evaluate_measurements(circuit)
    assert result.analyses_run.count("op") == 1


def test_a_supplied_operating_point_is_reused():
    circuit = parse_netlist(DIVIDER)
    op = solve_dc(circuit)
    result = evaluate_measurements(circuit, op=op)
    assert result.analyses_run == ["op (supplied)"]
    assert result.values["vo"] == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# AC measurement kinds
# ---------------------------------------------------------------------------

def test_gain_in_db_and_linear():
    circuit = parse_netlist(RC + ".measure g GAIN in=in out=out\n"
                            ".measure gl GAIN in=in out=out units=lin\n")
    values = evaluate_measurements(circuit).values
    assert values["gl"] == pytest.approx(1.0, rel=1e-6)   # DC gain of an RC
    assert values["g"] == pytest.approx(0.0, abs=1e-5)


def test_gain_at_a_specified_frequency():
    f3db = 1.0 / (2 * math.pi * 1e3 * 1e-9)
    circuit = parse_netlist(RC + f".measure g GAIN in=in out=out freq={f3db}\n")
    value = evaluate_measurements(circuit).values["g"]
    assert value == pytest.approx(-3.0103, abs=1e-3)


def test_bandwidth_of_an_rc_lowpass():
    circuit = parse_netlist(RC + ".measure bw BW in=in out=out\n")
    expected = 1.0 / (2 * math.pi * 1e3 * 1e-9)
    assert evaluate_measurements(circuit).values["bw"] == pytest.approx(
        expected, rel=2e-3)


def test_unity_gain_frequency_and_phase_margin_of_the_opamp():
    circuit = load("two_stage_opamp")
    values = evaluate_measurements(circuit).values
    assert values["ugf"] > 1e6
    assert 0.0 < values["pm"] < 180.0
    assert values["av"] > 60.0


def test_gain_margin_is_undefined_for_a_single_pole_response():
    """A one-pole roll-off never reaches -180 deg, so gain margin does not
    exist -- and the tool must say so rather than invent a number."""
    circuit = parse_netlist(RC + ".measure gmarg GM in=in out=out\n")
    outcome = evaluate_measurements(circuit).outcomes["gmarg"]
    assert not outcome.ok
    assert math.isnan(outcome.value)
    assert "-180" in outcome.reason


# ---------------------------------------------------------------------------
# offset -- two real DC solves, no curve fitting
# ---------------------------------------------------------------------------

def test_offset_is_zero_for_a_matched_pair():
    values = evaluate_measurements(load("diff_pair")).values
    assert abs(values["vos"]) < 1e-9


def test_offset_tracks_an_injected_threshold_mismatch():
    """Vos should come out close to the threshold difference itself, since a
    differential pair refers dVth directly to the input."""
    circuit = load("diff_pair")
    delta = 5e-3
    perturbed = circuit.with_overrides({"M1": {"dvth": delta}})
    values = evaluate_measurements(perturbed).values
    assert abs(values["vos"]) == pytest.approx(delta, rel=0.05)


def test_offset_reverses_sign_with_the_mismatch():
    circuit = load("diff_pair")
    plus = evaluate_measurements(
        circuit.with_overrides({"M1": {"dvth": 5e-3}})).values["vos"]
    minus = evaluate_measurements(
        circuit.with_overrides({"M1": {"dvth": -5e-3}})).values["vos"]
    assert plus == pytest.approx(-minus, rel=0.02)


def test_offset_is_symmetric_between_the_two_devices():
    circuit = load("diff_pair")
    on_m1 = evaluate_measurements(
        circuit.with_overrides({"M1": {"dvth": 3e-3}})).values["vos"]
    on_m2 = evaluate_measurements(
        circuit.with_overrides({"M2": {"dvth": 3e-3}})).values["vos"]
    assert on_m1 == pytest.approx(-on_m2, rel=0.02)


def test_offset_fails_cleanly_when_the_pair_is_not_biased():
    """Starve the tail so the pair is in cutoff: the differential gain
    collapses and the offset becomes undefined."""
    circuit = load("diff_pair")
    dead = circuit.with_overrides({"ITAIL": {"dc": 1e-15}})
    outcome = evaluate_measurements(dead).outcomes["vos"]
    assert not outcome.ok
    assert math.isnan(outcome.value)
    assert "saturation" in outcome.reason or "differential gain" in outcome.reason


# ---------------------------------------------------------------------------
# transient measurement kinds
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_transient_edge_measurements():
    values = evaluate_measurements(load("inverter_transient")).values
    assert 0 < values["tfall"] < 5e-9
    assert 0 < values["trise"] < 5e-9
    assert values["vhigh"] > 1.75
    assert values["vlow"] < 0.05


@pytest.mark.slow
def test_transient_statistics_measurements():
    circuit = parse_netlist(
        "V1 in 0 DC 0 PULSE(0 1 0 1p 1p 100u 200u)\n"
        "R1 in out 1k\nC1 out 0 1n\n.tran 5n 5u\n"
        ".measure vmx VMAX node=out\n.measure vmn VMIN node=out\n"
        ".measure vpp VPP node=out\n.measure vav VAVG node=out\n")
    values = evaluate_measurements(circuit).values
    assert values["vmn"] == pytest.approx(0.0, abs=1e-6)
    assert values["vmx"] == pytest.approx(1 - math.exp(-5), abs=1e-3)
    assert values["vpp"] == pytest.approx(values["vmx"] - values["vmn"], rel=1e-9)
    # The mean of 1-exp(-t/tau) over 5 tau is 1 - (1-exp(-5))/5.
    assert values["vav"] == pytest.approx(1 - (1 - math.exp(-5)) / 5, abs=2e-3)


def test_transient_measurement_without_a_tran_card_is_reported():
    circuit = parse_netlist(DIVIDER + ".measure vmx VMAX node=out\n")
    outcome = evaluate_measurements(circuit).outcomes["vmx"]
    assert not outcome.ok
    assert ".tran" in outcome.reason


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------

def test_unity_gain_frequency_that_does_not_exist_is_reported_not_invented():
    circuit = parse_netlist(
        "V1 in 0 DC 1 AC 1\nR1 in out 1k\nR2 out 0 1k\n"
        ".ac dec 10 1 1G\n.measure u UGF in=in out=out\n")
    outcome = evaluate_measurements(circuit).outcomes["u"]
    assert not outcome.ok
    assert math.isnan(outcome.value)
    assert "unity-gain crossing" in outcome.reason


def test_bandwidth_of_a_flat_response_is_reported_not_invented():
    circuit = parse_netlist(
        "V1 in 0 DC 1 AC 1\nR1 in out 1k\nR2 out 0 1k\n"
        ".ac dec 10 1 1G\n.measure bw BW in=in out=out\n")
    outcome = evaluate_measurements(circuit).outcomes["bw"]
    assert not outcome.ok
    assert "3 dB" in outcome.reason


def test_expression_referring_to_a_failed_measurement_fails_too():
    circuit = parse_netlist(
        "V1 in 0 DC 1 AC 1\nR1 in out 1k\nR2 out 0 1k\n"
        ".ac dec 10 1 1G\n.measure u UGF in=in out=out\n"
        ".measure d EXPR u/2\n")
    result = evaluate_measurements(circuit)
    assert not result.outcomes["u"].ok
    assert not result.outcomes["d"].ok
    assert math.isnan(result.values["d"])


def test_all_valid_and_invalid_reasons():
    circuit = parse_netlist(
        "V1 in 0 DC 1 AC 1\nR1 in out 1k\nR2 out 0 1k\n"
        ".ac dec 10 1 1G\n.measure vo V(out)\n.measure u UGF in=in out=out\n")
    result = evaluate_measurements(circuit)
    assert not result.all_valid
    reasons = result.invalid_reasons()
    assert set(reasons) == {"u"}
    assert result.outcomes["vo"].ok


def test_measurement_result_serialises():
    result = evaluate_measurements(parse_netlist(DIVIDER))
    data = result.to_dict()
    assert data["values"]["vo"] == pytest.approx(0.5)
    assert data["outcomes"]["vo"]["ok"] is True
    assert "op" in data["analyses_run"]


# ---------------------------------------------------------------------------
# the measurement context
# ---------------------------------------------------------------------------

def test_context_caches_the_ac_sweep():
    circuit = parse_netlist(RC + ".measure bw BW in=in out=out\n"
                            ".measure u UGF in=in out=out\n")
    context = MeasurementContext(circuit)
    first = context.ac_sweep()
    assert context.ac_sweep() is first
    assert context.analyses_run.count("ac") == 1


def test_context_resimulate_applies_source_overrides():
    circuit = load("diff_pair")
    context = MeasurementContext(circuit)
    base = context.op()
    perturbed = context.resimulate({"VINP": 0.905, "VINN": 0.895})
    assert perturbed.v("outp") < base.v("outp")   # more current in M1
    assert perturbed.v("outn") > base.v("outn")
