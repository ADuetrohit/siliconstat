"""Process / voltage / temperature corners, and PVT crossed with mismatch."""

from __future__ import annotations

import numpy as np
import pytest

from siliconstat.core.exceptions import VariationError
from siliconstat.core.solver import solve_dc
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.variation import default_mismatch_model
from siliconstat.variation.pvt import (
    CORNERS,
    PVTCondition,
    corner_model_mods,
    default_pvt_grid,
    find_supply_source,
    pvt_grid,
)

from conftest import load


# ---------------------------------------------------------------------------
# corner definitions
# ---------------------------------------------------------------------------

def test_typical_corner_applies_no_shift():
    circuit = load("current_mirror")
    assert corner_model_mods(circuit, "TT") == {}


def test_fast_corner_lowers_threshold_and_raises_beta():
    circuit = load("current_mirror")
    mods = corner_model_mods(circuit, "FF")
    assert mods["NCH"]["dvth"] < 0.0
    assert mods["NCH"]["beta_scale"] > 1.0


def test_slow_corner_is_the_mirror_image_of_fast():
    circuit = load("current_mirror")
    fast = corner_model_mods(circuit, "FF")["NCH"]
    slow = corner_model_mods(circuit, "SS")["NCH"]
    assert slow["dvth"] == pytest.approx(-fast["dvth"])
    assert slow["beta_scale"] - 1.0 == pytest.approx(1.0 - fast["beta_scale"])


def test_skew_corners_split_by_device_type():
    circuit = load("two_stage_opamp")     # has both NCH and PCH
    fs = corner_model_mods(circuit, "FS")
    assert fs["NCH"]["dvth"] < 0.0        # fast NMOS
    assert fs["PCH"]["dvth"] > 0.0        # slow PMOS
    sf = corner_model_mods(circuit, "SF")
    assert sf["NCH"]["dvth"] > 0.0
    assert sf["PCH"]["dvth"] < 0.0


def test_corner_magnitude_is_k_sigma_of_the_global_distribution():
    """Corners are defined against the same global sigma the Monte Carlo
    process model uses, so the two views stay consistent."""
    circuit = load("current_mirror")
    mods = corner_model_mods(circuit, "SS", sigma_vth_global=0.03, k_sigma=3.0)
    assert mods["NCH"]["dvth"] == pytest.approx(0.09)


def test_unknown_corner_is_rejected_and_lists_the_valid_ones():
    circuit = load("current_mirror")
    with pytest.raises(VariationError) as excinfo:
        corner_model_mods(circuit, "XX")
    message = str(excinfo.value)
    assert "XX" in message and "TT" in message and "FF" in message


# ---------------------------------------------------------------------------
# applying a condition
# ---------------------------------------------------------------------------

def test_condition_label():
    condition = PVTCondition(corner="ss", supply=1.62, temp_c=125.0)
    assert "SS" in condition.label and "1.62" in condition.label and "125" in condition.label


def test_condition_applies_the_supply_override():
    circuit = load("current_mirror")
    op = solve_dc(circuit, PVTCondition(supply=1.62).build_context(circuit))
    assert op.v("vdd") == pytest.approx(1.62, rel=1e-12)


def test_condition_applies_the_temperature():
    circuit = load("current_mirror")
    hot = PVTCondition(temp_c=125.0).build_context(circuit)
    assert hot.temp_c == pytest.approx(125.0)
    assert hot.mos_models["NCH"].kp < circuit.mos_models["NCH"].kp


def test_condition_applies_the_corner_before_the_temperature():
    """A corner selects a point in process space; the die then heats up.  The
    two shifts must compose, not overwrite each other."""
    circuit = load("current_mirror")
    nominal_vto = circuit.mos_models["NCH"].vto
    ctx = PVTCondition(corner="SS", temp_c=125.0).build_context(circuit)
    corner_shift = 3.0 * 0.025
    temperature_shift = circuit.mos_models["NCH"].tcv * (125.0 - 27.0)
    assert ctx.mos_models["NCH"].vto == pytest.approx(
        nominal_vto + corner_shift + temperature_shift, rel=1e-9)


def test_slow_corner_needs_more_overdrive_at_a_fixed_current():
    circuit = load("current_mirror")
    typical = solve_dc(circuit, PVTCondition(corner="TT").build_context(circuit))
    slow = solve_dc(circuit, PVTCondition(corner="SS").build_context(circuit))
    assert slow.device_ops["M1"]["vth"] > typical.device_ops["M1"]["vth"]
    assert slow.device_ops["M1"]["vov"] > typical.device_ops["M1"]["vov"]
    assert slow.v("nref") > typical.v("nref")


def test_supply_source_discovery():
    circuit = load("current_mirror")
    assert find_supply_source(circuit).name == "VDD"
    assert find_supply_source(circuit, "VDD").name == "VDD"


def test_supply_source_must_be_a_voltage_source():
    circuit = load("current_mirror")
    with pytest.raises(VariationError):
        find_supply_source(circuit, "IREF")


def test_condition_round_trips_through_dict():
    condition = PVTCondition(corner="FF", supply=1.98, temp_c=-40.0)
    restored = PVTCondition.from_dict(condition.to_dict())
    assert restored.label == condition.label


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------

def test_grid_is_the_full_cross_product():
    grid = pvt_grid(["TT", "SS"], [1.62, 1.8], [-40.0, 125.0])
    assert len(grid) == 8
    assert len({c.label for c in grid}) == 8


def test_default_grid_covers_five_corners_three_supplies_three_temperatures():
    circuit = load("current_mirror")
    grid = default_pvt_grid(circuit)
    assert len(grid) == len(CORNERS) * 3 * 3
    supplies = sorted({c.supply for c in grid})
    assert supplies == pytest.approx([1.62, 1.8, 1.98])


# ---------------------------------------------------------------------------
# PVT crossed with Monte Carlo mismatch
# ---------------------------------------------------------------------------

def mirror_run(condition: PVTCondition, samples: int = 120, seed: int = 31):
    circuit = load("current_mirror")
    return run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=samples, seed=seed,
        workers=1, pvt=condition))


@pytest.mark.slow
def test_mismatch_spread_shrinks_at_high_temperature():
    """sigma(ierr) ~ (gm/Id) * sigma(dVth).  Raising the temperature lowers
    beta, so at a fixed 10 uA the overdrive rises, gm/Id falls, and the copy
    error tightens.  This is a prediction of the model, not a recorded value."""
    cold = mirror_run(PVTCondition(temp_c=-40.0))
    hot = mirror_run(PVTCondition(temp_c=125.0))
    assert np.std(hot.values("ierr"), ddof=1) < np.std(cold.values("ierr"), ddof=1)


@pytest.mark.slow
def test_fast_corner_widens_the_spread_relative_to_slow():
    """FF raises beta, which lowers Vov at a fixed current and therefore
    raises gm/Id -- more sensitivity to the same threshold mismatch."""
    fast = mirror_run(PVTCondition(corner="FF"))
    slow = mirror_run(PVTCondition(corner="SS"))
    assert np.std(fast.values("ierr"), ddof=1) > np.std(slow.values("ierr"), ddof=1)


@pytest.mark.slow
def test_supply_shifts_the_mean_copy_error_through_channel_length_modulation():
    """A higher rail raises V(out), so M2 sees more Vds than the diode-connected
    M1 and copies proportionally more current."""
    low = mirror_run(PVTCondition(supply=1.62))
    high = mirror_run(PVTCondition(supply=1.98))
    assert np.mean(high.values("ierr")) > np.mean(low.values("ierr"))


@pytest.mark.slow
def test_pvt_condition_is_recorded_in_the_run_configuration():
    run = mirror_run(PVTCondition(corner="SS", supply=1.62, temp_c=125.0),
                     samples=10)
    assert run.config["pvt"]["corner"] == "SS"
    assert run.config["pvt"]["supply"] == pytest.approx(1.62)
    assert run.summary()["pvt"]["label"] == "SS/1.62V/125C"


@pytest.mark.slow
def test_every_corner_still_converges_for_the_demo_circuits():
    for name in ("current_mirror", "diff_pair"):
        circuit = load(name)
        for corner in CORNERS:
            for supply, temp in ((1.62, 125.0), (1.98, -40.0)):
                condition = PVTCondition(corner=corner, supply=supply,
                                         temp_c=temp)
                op = solve_dc(circuit, condition.build_context(circuit))
                assert np.all(np.isfinite(op.x)), f"{name} {condition.label}"
