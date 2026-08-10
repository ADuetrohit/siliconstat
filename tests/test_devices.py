"""Device MNA stamps and the KCL sign convention."""

from __future__ import annotations

import numpy as np
import pytest

from siliconstat.core.context import SimContext
from siliconstat.core.devices import (
    Capacitor,
    CurrentSource,
    Inductor,
    Mosfet,
    Resistor,
    VoltageSource,
    node_voltage,
)
from siliconstat.core.exceptions import CircuitError
from siliconstat.core.mna import MnaSystem
from siliconstat.core.netlist import parse_netlist
from siliconstat.core.solver import solve_dc


# ---------------------------------------------------------------------------
# linear elements
# ---------------------------------------------------------------------------

def test_resistor_stamp():
    sys = MnaSystem(n_nodes=3, n_branches=0)
    Resistor(name="R1", nodes=(1, 2), r=1e3).stamp_dc(sys, np.zeros(2), SimContext())
    g = 1e-3
    assert np.allclose(sys.G, np.array([[g, -g], [-g, g]]))
    assert np.allclose(sys.b, 0.0)


def test_resistor_to_ground_only_touches_one_row():
    sys = MnaSystem(n_nodes=2, n_branches=0)
    Resistor(name="R1", nodes=(1, 0), r=1e3).stamp_dc(sys, np.zeros(1), SimContext())
    assert sys.G[0, 0] == pytest.approx(1e-3)


def test_resistor_current_and_power():
    ctx = SimContext()
    r = Resistor(name="R1", nodes=(1, 2), r=1e3)
    x = np.array([3.0, 1.0])
    assert r.current(x, ctx) == pytest.approx(2e-3)
    assert r.power(x, ctx) == pytest.approx(4e-3)


def test_voltage_source_branch_stamp():
    sys = MnaSystem(n_nodes=2, n_branches=1)
    VoltageSource(name="V1", nodes=(1, 0), branch=0, dc=1.8).stamp_dc(
        sys, np.zeros(2), SimContext())
    assert sys.G[0, 1] == pytest.approx(1.0)   # node row, branch column
    assert sys.G[1, 0] == pytest.approx(1.0)   # branch row, node column
    assert sys.b[1] == pytest.approx(1.8)


def test_voltage_source_sign_and_power_convention():
    """Branch current flows + -> - inside the source; delivered power is -v*i."""
    circuit = parse_netlist("V1 a 0 2\nR1 a 0 1k\n.measure v V(a)\n")
    op = solve_dc(circuit)
    src = circuit.device("V1")
    assert src.current(op.x, op.ctx) == pytest.approx(-2e-3)
    assert src.power(op.x, op.ctx) == pytest.approx(4e-3)
    assert op.total_supply_power() == pytest.approx(4e-3)


def test_current_source_injects_into_the_second_node():
    sys = MnaSystem(n_nodes=3, n_branches=0)
    CurrentSource(name="I1", nodes=(1, 2), dc=1e-5).stamp_dc(
        sys, np.zeros(2), SimContext())
    assert sys.b[0] == pytest.approx(-1e-5)
    assert sys.b[1] == pytest.approx(+1e-5)


def test_current_source_drives_a_resistor_the_right_way():
    circuit = parse_netlist("V1 vdd 0 5\nI1 vdd a 1m\nR1 a 0 1k\n.measure v V(a)\n")
    assert solve_dc(circuit).v("a") == pytest.approx(1.0, rel=1e-9)


def test_capacitor_is_open_at_dc_and_admits_current_in_ac():
    sys = MnaSystem(n_nodes=3, n_branches=0)
    cap = Capacitor(name="C1", nodes=(1, 2), c=1e-9)
    cap.stamp_dc(sys, np.zeros(2), SimContext())
    assert np.allclose(sys.G, 0.0)

    ac = MnaSystem(n_nodes=3, n_branches=0, dtype=complex)
    cap.stamp_ac(ac, np.zeros(2), omega=2 * np.pi * 1e6, ctx=SimContext())
    assert ac.G[0, 0] == pytest.approx(1j * 2 * np.pi * 1e6 * 1e-9)


def test_inductor_is_a_short_at_dc():
    sys = MnaSystem(n_nodes=2, n_branches=1)
    Inductor(name="L1", nodes=(1, 0), branch=0, l=1e-9).stamp_dc(
        sys, np.zeros(2), SimContext())
    assert sys.G[1, 1] == pytest.approx(0.0)      # v = 0 constraint
    assert sys.G[0, 1] == pytest.approx(1.0)
    assert sys.b[1] == pytest.approx(0.0)


def test_inductor_shorts_a_divider_at_dc():
    circuit = parse_netlist("V1 a 0 1\nR1 a b 1k\nL1 b c 1u\nR2 c 0 1k\n"
                            ".measure v V(b)\n")
    op = solve_dc(circuit)
    assert op.v("b") == pytest.approx(0.5, rel=1e-9)
    assert op.v("c") == pytest.approx(0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# the nonlinear companion model
# ---------------------------------------------------------------------------

MIRROR = (
    ".model NCH NMOS VTO=0.45 KP=246u LAMBDA=0.08 GAMMA=0.4 PHI=0.8\n"
    "VDD vdd 0 1.8\nIREF vdd g 10u\n"
    "M1 g g 0 0 NCH W=10u L=1u\nM2 out g 0 0 NCH W=10u L=1u\n"
    "R1 vdd out 125k\n.measure i I(M2)\n"
)


def test_mosfet_companion_residual_vanishes_at_the_solution():
    """``G(x)@x - b(x)`` is the exact KCL residual -- the basis of the
    solver's convergence test -- so it must vanish at the converged point."""
    circuit = parse_netlist(MIRROR)
    op = solve_dc(circuit)
    ctx = op.ctx
    ctx.limiting, ctx.x_prev_iter = False, None
    sys = MnaSystem(circuit.n_nodes, circuit.n_branches)
    for dev in circuit.devices:
        dev.stamp_dc(sys, op.x, ctx)
    assert np.max(np.abs(sys.G @ op.x - sys.b)) < 1e-10


def test_kcl_holds_at_every_node():
    circuit = parse_netlist(MIRROR)
    op = solve_dc(circuit)
    ctx = op.ctx
    ctx.limiting, ctx.x_prev_iter = False, None
    assert circuit.device("R1").current(op.x, ctx) == pytest.approx(
        circuit.device("M2").current(op.x, ctx), rel=1e-9)
    assert circuit.device("IREF").current(op.x, ctx) == pytest.approx(
        circuit.device("M1").current(op.x, ctx), rel=1e-9)


def test_pmos_drain_current_is_negative():
    """A PMOS sources current out of its drain, so I(drain) < 0."""
    circuit = parse_netlist(
        ".model PCH PMOS VTO=0.45 KP=86u LAMBDA=0.1\n"
        "VDD vdd 0 1.8\nVG g 0 0.9\n"
        "M1 out g vdd vdd PCH W=20u L=1u\nR1 out 0 10k\n.measure i I(M1)\n")
    op = solve_dc(circuit)
    assert op.i("M1") < 0.0
    assert circuit.device("R1").current(op.x, op.ctx) == pytest.approx(
        -op.i("M1"), rel=1e-9)


def test_gmin_keeps_a_cut_off_device_from_making_the_matrix_singular():
    circuit = parse_netlist(
        ".model NCH NMOS VTO=0.45 KP=246u\n"
        "VDD vdd 0 1.8\nVG g 0 0.0\n"
        "M1 out g 0 0 NCH W=10u L=1u\nR1 vdd out 1meg\n.measure v V(out)\n")
    op = solve_dc(circuit)
    assert op.device_ops["M1"]["region"] == "cutoff"
    assert op.v("out") == pytest.approx(1.8, rel=1e-3)


def test_diode_current_matches_the_shockley_equation():
    import math

    from siliconstat.core.models import thermal_voltage

    circuit = parse_netlist(
        ".model DX D IS=1e-14 N=1.0\nVDD vdd 0 1.8\nR1 vdd a 10k\nD1 a 0 DX\n"
        ".measure v V(a)\n.measure i I(D1)\n")
    op = solve_dc(circuit)
    vd, current = op.v("a"), op.i("D1")
    expected = 1e-14 * (math.exp(vd / thermal_voltage(27.0)) - 1.0)
    assert current == pytest.approx(expected, rel=1e-6)
    # ... and it must also satisfy the loop equation, to within the solver's
    # own reltol (1e-6) rather than to machine precision.
    assert current == pytest.approx((1.8 - vd) / 1e4, rel=1e-6)


# ---------------------------------------------------------------------------
# overrides used by the Monte Carlo layer
# ---------------------------------------------------------------------------

def test_device_overrides_produce_new_objects():
    r = Resistor(name="R1", nodes=(1, 2), r=1e3)
    perturbed = r.with_overrides({"r": 1.1e3})
    assert perturbed is not r
    assert r.r == pytest.approx(1e3)          # the original is untouched
    assert perturbed.r == pytest.approx(1.1e3)


def test_empty_overrides_return_the_same_object():
    r = Resistor(name="R1", nodes=(1, 2), r=1e3)
    assert r.with_overrides({}) is r


def test_mosfet_overrides_are_modifiers_not_replacements():
    m = Mosfet(name="M1", nodes=(1, 2, 0, 0), model="NCH", w=1e-5, l=1e-6)
    perturbed = m.with_overrides({"dvth": 0.01, "beta_scale": 1.05})
    assert perturbed.dvth == pytest.approx(0.01)
    assert perturbed.beta_scale == pytest.approx(1.05)
    assert perturbed.w == pytest.approx(m.w)


@pytest.mark.parametrize("overrides", [
    {"w": 0.0}, {"w": -1e-6}, {"l": 0.0}, {"beta_scale": 0.0}, {"beta_scale": -0.5},
])
def test_non_physical_mosfet_overrides_are_rejected(overrides):
    m = Mosfet(name="M1", nodes=(1, 2, 0, 0), model="NCH", w=1e-5, l=1e-6)
    with pytest.raises(CircuitError):
        m.with_overrides(overrides)


def test_non_physical_passive_overrides_are_rejected():
    with pytest.raises(CircuitError):
        Resistor(name="R1", nodes=(1, 2), r=1e3).with_overrides({"r": -1.0})
    with pytest.raises(CircuitError):
        Capacitor(name="C1", nodes=(1, 2), c=1e-12).with_overrides({"c": 0.0})


def test_unsupported_overrides_are_reported():
    src = VoltageSource(name="V1", nodes=(1, 0), dc=1.8)
    assert src.with_overrides({"dc": 1.62}).dc == pytest.approx(1.62)
    with pytest.raises(CircuitError):
        Inductor(name="L1", nodes=(1, 0), l=1e-9).with_overrides({"nonsense": 1})


def test_variable_params_are_all_targetable_by_the_variation_layer():
    """Anything a device advertises must be something the sampler understands."""
    from siliconstat.variation.spec import PARAMETER_MODES

    for cls in (Resistor, Capacitor, VoltageSource, CurrentSource, Mosfet, Inductor):
        for name in cls.VARIABLE_PARAMS:
            assert name in PARAMETER_MODES, f"{cls.__name__}.{name}"


def test_node_voltage_helper():
    x = np.array([1.5, 2.5])
    assert node_voltage(x, 0) == 0.0
    assert node_voltage(x, 1) == pytest.approx(1.5)
    assert node_voltage(x, 2) == pytest.approx(2.5)


def test_circuit_with_overrides_leaves_the_original_alone():
    circuit = parse_netlist(MIRROR)
    perturbed = circuit.with_overrides({"M1": {"dvth": 0.01}})
    assert circuit.device("M1").dvth == pytest.approx(0.0)
    assert perturbed.device("M1").dvth == pytest.approx(0.01)
    assert perturbed.device("M2").dvth == pytest.approx(0.0)


def test_overriding_an_unknown_device_is_rejected():
    circuit = parse_netlist(MIRROR)
    with pytest.raises(CircuitError) as excinfo:
        circuit.with_overrides({"M99": {"dvth": 0.01}})
    assert "M99" in str(excinfo.value)
