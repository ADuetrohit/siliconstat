"""Closed-form validation of the simulator.

Every assertion here compares simulator output against an answer derived
independently -- from Ohm's law, the Shockley equation, the square law, or
small-signal amplifier theory -- rather than against a previously recorded
simulator output.  These are the tests that would catch a physically wrong
simulator that is nonetheless self-consistent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.core.models import thermal_voltage
from siliconstat.core.solver import solve_dc, transient_analysis
from siliconstat.measure import evaluate_measurements

from conftest import load


# ---------------------------------------------------------------------------
# 1. Ohm's law -- the MNA assembly itself
# ---------------------------------------------------------------------------

def test_resistive_divider_is_exact():
    """5 V across 1k + 3k: V(mid) = 3.75 V, I = 1.25 mA, P = 6.25 mW."""
    circuit = load("rc_divider")
    op = solve_dc(circuit)
    assert op.v("mid") == pytest.approx(5.0 * 3e3 / 4e3, rel=1e-12)
    assert op.i("R1") == pytest.approx(5.0 / 4e3, rel=1e-12)
    assert op.total_supply_power() == pytest.approx(5.0 * 5.0 / 4e3, rel=1e-12)
    assert op.p("R2") == pytest.approx((3.75 ** 2) / 3e3, rel=1e-12)


def test_divider_measurements_agree_with_the_operating_point():
    circuit = load("rc_divider")
    result = evaluate_measurements(circuit)
    assert result.values["vmid"] == pytest.approx(3.75, rel=1e-12)
    assert result.values["i1"] == pytest.approx(1.25e-3, rel=1e-12)
    assert result.values["ptot"] == pytest.approx(6.25e-3, rel=1e-12)
    assert result.values["ratio"] == pytest.approx(0.75, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Shockley diode -- the exponential device
# ---------------------------------------------------------------------------

def test_diode_bias_matches_the_transcendental_solution():
    """Solve  (VDD - Va)/R = IS*(exp((Va - I*RS)/(N*Vt)) - 1)  independently."""
    from scipy.optimize import brentq

    circuit = load("diode_bias")
    op = solve_dc(circuit)
    vt = thermal_voltage(27.0)
    is_, rs, vdd, r = 1e-14, 5.0, 1.8, 1e4

    def residual(current: float) -> float:
        v_node = vdd - current * r          # KVL through R1
        v_junction = v_node - current * rs  # drop across the series resistance
        return is_ * (math.exp(v_junction / vt) - 1.0) - current

    current = brentq(residual, 1e-12, 1e-2, xtol=1e-18, rtol=1e-15)
    assert op.i("D1") == pytest.approx(current, rel=1e-6)
    assert op.v("a") == pytest.approx(vdd - current * r, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. MOSFET square law inside a real circuit
# ---------------------------------------------------------------------------

def test_diode_connected_device_satisfies_the_square_law():
    circuit = load("current_mirror")
    op = solve_dc(circuit)
    m1 = op.device_ops["M1"]
    model = circuit.mos_models["NCH"]
    beta = model.kp * circuit.device("M1").w / circuit.device("M1").l
    vov = m1["vgs"] - m1["vth"]
    expected = 0.5 * beta * vov ** 2 * (1.0 + model.lambda_ * m1["vds"])
    assert m1["id"] == pytest.approx(expected, rel=1e-9)


def test_saturation_gm_relation_holds_in_circuit():
    op = solve_dc(load("current_mirror"))
    m1 = op.device_ops["M1"]
    assert m1["gm"] == pytest.approx(2.0 * m1["id"] / m1["vov"], rel=1e-9)


def test_current_mirror_copy_ratio_is_set_by_channel_length_modulation():
    """With matched devices the only nominal error is CLM:
        Iout/Iref = (1 + lambda*Vds2) / (1 + lambda*Vds1)."""
    circuit = load("current_mirror")
    op = solve_dc(circuit)
    lam = circuit.mos_models["NCH"].lambda_
    vds1 = op.device_ops["M1"]["vds"]
    vds2 = op.device_ops["M2"]["vds"]
    expected_ratio = (1 + lam * vds2) / (1 + lam * vds1)
    assert op.i("M2") / op.i("M1") == pytest.approx(expected_ratio, rel=1e-9)


def test_current_mirror_reference_equals_the_forced_current():
    circuit = load("current_mirror")
    op = solve_dc(circuit)
    assert op.i("M1") == pytest.approx(10e-6, rel=1e-6)


def test_current_mirror_output_node_satisfies_kvl():
    circuit = load("current_mirror")
    op = solve_dc(circuit)
    # KVL holds to the solver's reltol: the residual current at "out" is
    # ~5e-13 A, which across the 125 kohm load is ~7e-8 V.
    assert op.v("out") == pytest.approx(1.8 - 125e3 * op.i("M2"), rel=1e-6)


def test_current_mirror_supply_power():
    circuit = load("current_mirror")
    op = solve_dc(circuit)
    total_from_vdd = 10e-6 + op.i("M2")     # IREF branch plus the R1 branch
    assert op.total_supply_power() == pytest.approx(1.8 * total_from_vdd, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Differential pair -- small-signal gain
# ---------------------------------------------------------------------------

def test_differential_pair_is_balanced_when_matched():
    op = solve_dc(load("diff_pair"))
    assert op.v("outp") == pytest.approx(op.v("outn"), abs=1e-9)
    assert op.i("M1") == pytest.approx(op.i("M2"), rel=1e-9)
    assert op.i("M1") == pytest.approx(10e-6, rel=1e-6)     # half the tail


def test_differential_pair_tail_node_is_self_consistent():
    """V(tail) = Vcm - Vgs, with Vgs including the body effect at that Vsb."""
    circuit = load("diff_pair")
    op = solve_dc(circuit)
    m1 = op.device_ops["M1"]
    model = circuit.mos_models["NCH"]
    vsb = op.v("tail")
    expected_vth = model.vto + model.gamma * (
        math.sqrt(model.phi + vsb) - math.sqrt(model.phi))
    assert m1["vth"] == pytest.approx(expected_vth, rel=1e-9)
    assert op.v("tail") == pytest.approx(0.9 - m1["vgs"], rel=1e-9)


def test_differential_gain_equals_gm_times_the_load():
    """Ad = gm * (RL || ro), measured against the AC analysis."""
    circuit = load("diff_pair")
    op = solve_dc(circuit)
    result = evaluate_measurements(circuit, op=op)
    gm = op.device_ops["M1"]["gm"]
    ro = op.device_ops["M1"]["ro"]
    load_r = 50e3
    expected_linear = gm * (load_r * ro) / (load_r + ro)
    measured_linear = 10 ** (result.values["ad"] / 20.0)
    assert measured_linear == pytest.approx(expected_linear, rel=0.01)


def test_matched_differential_pair_has_zero_offset():
    result = evaluate_measurements(load("diff_pair"))
    assert abs(result.values["vos"]) < 1e-9
    assert abs(result.values["vod"]) < 1e-9


def test_differential_pair_supply_power():
    circuit = load("diff_pair")
    op = solve_dc(circuit)
    assert op.total_supply_power() == pytest.approx(1.8 * 20e-6, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Two-stage op-amp -- gain-bandwidth theory
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_opamp_gain_bandwidth_product_equals_gm1_over_cc():
    """For a Miller-compensated two-stage OTA, GBW = gm1 / (2*pi*Cc)."""
    circuit = load("two_stage_opamp")
    op = solve_dc(circuit)
    result = evaluate_measurements(circuit, op=op)
    gm1 = op.device_ops["M1"]["gm"]
    cc = circuit.device("CC").c
    predicted = gm1 / (2 * math.pi * cc)
    assert result.values["ugf"] == pytest.approx(predicted, rel=0.10)


@pytest.mark.slow
def test_opamp_single_pole_rolloff_ties_gain_bandwidth_and_ugf():
    """A dominant-pole response satisfies A0 * f3dB = funity."""
    result = evaluate_measurements(load("two_stage_opamp"))
    product = result.values["avlin"] * result.values["bw"]
    assert product == pytest.approx(result.values["ugf"], rel=0.05)


@pytest.mark.slow
def test_opamp_two_stage_gain_matches_the_stage_by_stage_estimate():
    """Av = gm1*(ro2||ro4) * gm6*(ro6||ro7)."""
    circuit = load("two_stage_opamp")
    op = solve_dc(circuit)
    result = evaluate_measurements(circuit, op=op)

    def parallel(a: float, b: float) -> float:
        return a * b / (a + b)

    d = op.device_ops
    stage1 = d["M1"]["gm"] * parallel(d["M2"]["ro"], d["M4"]["ro"])
    stage2 = d["M6"]["gm"] * parallel(d["M6"]["ro"], d["M7"]["ro"])
    predicted_db = 20 * math.log10(stage1 * stage2)
    assert result.values["av"] == pytest.approx(predicted_db, abs=1.0)


@pytest.mark.slow
def test_opamp_unity_feedback_parks_the_output_at_the_input_common_mode():
    result = evaluate_measurements(load("two_stage_opamp"))
    assert result.values["vout"] == pytest.approx(0.9, abs=5e-3)
    assert abs(result.values["vos"]) < 1e-3       # small systematic offset only


@pytest.mark.slow
def test_opamp_all_devices_are_in_saturation():
    op = solve_dc(load("two_stage_opamp"))
    for name in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"):
        assert op.device_ops[name]["region"] == "saturation", name


@pytest.mark.slow
def test_opamp_bias_mirror_ratios():
    """M5, M7 mirror M8 with ratios set by their widths."""
    circuit = load("two_stage_opamp")
    op = solve_dc(circuit)
    i8 = op.i("M8")
    w8 = circuit.device("M8").w
    for name in ("M5", "M7"):
        ratio = circuit.device(name).w / w8
        # Channel-length modulation makes the copy inexact by a few percent.
        assert op.i(name) == pytest.approx(i8 * ratio, rel=0.06), name


@pytest.mark.slow
def test_opamp_supply_power_equals_the_sum_of_the_branch_currents():
    circuit = load("two_stage_opamp")
    op = solve_dc(circuit)
    from_vdd = 20e-6 + abs(op.i("M3")) + abs(op.i("M4")) + abs(op.i("M6"))
    assert op.total_supply_power() == pytest.approx(1.8 * from_vdd, rel=1e-4)


# ---------------------------------------------------------------------------
# 6. Inverter -- transient edge rates
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_inverter_edge_ratio_matches_the_drive_strength_ratio():
    """trise/tfall should track beta_n/beta_p, since each edge is driven by
    one device charging the same load."""
    circuit = load("inverter_transient")
    result = evaluate_measurements(circuit)
    nch, pch = circuit.mos_models["NCH"], circuit.mos_models["PCH"]
    m_n, m_p = circuit.device("M2"), circuit.device("M1")
    beta_n = nch.kp * m_n.w / m_n.l
    beta_p = pch.kp * m_p.w / m_p.l
    edge_ratio = result.values["trise"] / result.values["tfall"]
    assert edge_ratio == pytest.approx(beta_n / beta_p, rel=0.06)


@pytest.mark.slow
def test_inverter_rail_to_rail_swing():
    result = evaluate_measurements(load("inverter_transient"))
    assert result.values["vhigh"] > 1.75
    assert result.values["vlow"] < 0.05


@pytest.mark.slow
def test_inverter_edge_time_is_the_right_order_from_slew_reasoning():
    """t ~ CL*dV/I_drive; check the fall edge against the NMOS saturation
    current with a generous factor, since the device leaves saturation part
    way through the transition."""
    circuit = load("inverter_transient")
    result = evaluate_measurements(circuit)
    nch = circuit.mos_models["NCH"]
    m_n = circuit.device("M2")
    beta = nch.kp * m_n.w / m_n.l
    vov = 1.8 - nch.vto
    i_sat = 0.5 * beta * vov ** 2
    swing = 0.8 * 1.8                       # 10 % to 90 %
    optimistic = circuit.device("CL").c * swing / i_sat
    assert optimistic < result.values["tfall"] < 6.0 * optimistic


# ---------------------------------------------------------------------------
# 7. Temperature behaviour
# ---------------------------------------------------------------------------

def test_higher_temperature_lowers_threshold_and_current_factor():
    circuit = load("current_mirror")
    cold = solve_dc(circuit, circuit.build_context(temp_c=-40.0))
    hot = solve_dc(circuit, circuit.build_context(temp_c=125.0))
    # Both carry the same forced 10 uA, but the hot device needs more overdrive
    # because its mobility (and therefore beta) has dropped.
    assert hot.device_ops["M1"]["vov"] > cold.device_ops["M1"]["vov"]
    assert hot.device_ops["M1"]["vth"] < cold.device_ops["M1"]["vth"]
    assert hot.device_ops["M1"]["gm"] < cold.device_ops["M1"]["gm"]


def test_supply_scaling_shifts_the_mirror_output_node():
    circuit = load("current_mirror")
    ctx = circuit.build_context(source_overrides={"VDD": 1.62})
    op = solve_dc(circuit, ctx)
    assert op.v("vdd") == pytest.approx(1.62, rel=1e-12)
    assert op.v("out") == pytest.approx(1.62 - 125e3 * op.i("M2"), rel=1e-6)
