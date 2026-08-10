"""Pelgrom area scaling of local mismatch.

The convention under test: ``AVT`` is quoted for a matched **pair**, so the
per-device sigma injected by the sampler is ``AVT / sqrt(2*W*L)`` and two
independently perturbed devices then reproduce ``sigma(dVth) = AVT/sqrt(W*L)``.
That factor of sqrt(2) is the single easiest thing to get wrong in a mismatch
tool, so it is checked both algebraically and empirically.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.core.exceptions import VariationError
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.variation import (
    ParameterVariation,
    VariationModel,
    VariationSampler,
    default_mismatch_model,
)
from siliconstat.variation.pelgrom import (
    area_for_target_sigma,
    area_um2,
    sigma_beta_device,
    sigma_device,
    sigma_pair,
    sigma_vth_device,
)

from conftest import load

AVT = 3.5e-3      # V*um
ABETA = 0.010     # dimensionless*um


# ---------------------------------------------------------------------------
# the algebra
# ---------------------------------------------------------------------------

def test_area_in_square_microns():
    assert area_um2(10e-6, 1e-6) == pytest.approx(10.0)
    assert area_um2(20e-6, 2e-6) == pytest.approx(40.0)


@pytest.mark.parametrize("w,l", [(1e-6, 1e-6), (10e-6, 1e-6), (20e-6, 2e-6)])
def test_pair_sigma_is_the_quoted_coefficient_over_sqrt_area(w, l):
    assert sigma_pair(AVT, w, l) == pytest.approx(AVT / math.sqrt(area_um2(w, l)))


def test_device_sigma_is_smaller_by_root_two():
    w, l = 10e-6, 1e-6
    assert sigma_vth_device(AVT, w, l) == pytest.approx(
        sigma_pair(AVT, w, l) / math.sqrt(2.0))
    assert sigma_vth_device(AVT, w, l) == pytest.approx(
        AVT / math.sqrt(2 * area_um2(w, l)))


def test_single_device_convention_can_be_selected():
    w, l = 10e-6, 1e-6
    assert sigma_device(AVT, w, l, pair_convention=False) == pytest.approx(
        sigma_pair(AVT, w, l))


def test_beta_sigma_uses_the_same_scaling():
    w, l = 20e-6, 1e-6
    assert sigma_beta_device(ABETA, w, l) == pytest.approx(
        ABETA / math.sqrt(2 * area_um2(w, l)))


def test_quadrupling_area_halves_the_sigma():
    small = sigma_vth_device(AVT, 10e-6, 1e-6)
    large = sigma_vth_device(AVT, 20e-6, 2e-6)      # 4x the area
    assert large == pytest.approx(small / 2.0, rel=1e-12)


def test_inverse_pelgrom_gives_the_area_needed_for_a_target():
    target = 1e-3
    area = area_for_target_sigma(AVT, target)
    side = math.sqrt(area)
    assert sigma_vth_device(AVT, side * 1e-6, side * 1e-6) == pytest.approx(
        target, rel=1e-12)


@pytest.mark.parametrize("w,l", [(0.0, 1e-6), (1e-6, 0.0), (-1e-6, 1e-6)])
def test_non_physical_geometry_is_rejected(w, l):
    with pytest.raises(VariationError):
        area_um2(w, l)


def test_non_positive_target_sigma_is_rejected():
    with pytest.raises(VariationError):
        area_for_target_sigma(AVT, 0.0)


# ---------------------------------------------------------------------------
# what the sampler actually injects
# ---------------------------------------------------------------------------

def test_sampler_uses_the_pair_convention_for_local_vth():
    circuit = load("current_mirror")
    sampler = VariationSampler(circuit, default_mismatch_model(circuit))
    slot = next(s for s in sampler.slots if s.name == "M1.vth_local")
    m1 = circuit.device("M1")
    assert slot.sigma == pytest.approx(
        AVT / math.sqrt(2 * area_um2(m1.w, m1.l)), rel=1e-12)


def test_sampler_uses_the_pair_convention_for_local_beta():
    circuit = load("current_mirror")
    sampler = VariationSampler(circuit, default_mismatch_model(circuit))
    slot = next(s for s in sampler.slots if s.name == "M1.beta_local")
    m1 = circuit.device("M1")
    assert slot.sigma == pytest.approx(
        ABETA / math.sqrt(2 * area_um2(m1.w, m1.l)), rel=1e-12)


def test_drawn_pair_difference_reproduces_the_quoted_pelgrom_sigma():
    """The property the convention exists for: draw both devices and look at
    the *difference*, which must have sigma = AVT/sqrt(W*L)."""
    circuit = load("current_mirror")
    sampler = VariationSampler(circuit, default_mismatch_model(
        circuit, enable_process=False))
    i1 = sampler.slot_index["M1.vth_local"]
    i2 = sampler.slot_index["M2.vth_local"]

    n = 40_000
    rng = np.random.default_rng(31415)
    diffs = np.array([sampler.draw(rng)[i1] - sampler.draw(rng)[i2]
                      for _ in range(n // 2)])
    m1 = circuit.device("M1")
    expected = AVT / math.sqrt(area_um2(m1.w, m1.l))
    # SE(std) = sigma/sqrt(2N); 5 SE with N = 20000 is about 2.5 %.
    assert diffs.std(ddof=1) == pytest.approx(expected, rel=0.03)


def test_pelgrom_cannot_be_applied_to_a_global_variation():
    with pytest.raises(VariationError) as excinfo:
        ParameterVariation(parameter="vth", scope="global", pelgrom=True)
    assert "local" in str(excinfo.value)


def test_pelgrom_is_only_defined_for_vth_and_beta():
    with pytest.raises(VariationError) as excinfo:
        ParameterVariation(parameter="r", scope="local", pelgrom=True)
    assert "vth" in str(excinfo.value)


def test_pelgrom_on_a_non_mosfet_is_rejected():
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="vth", scope="local", pelgrom=True,
                           targets="type:resistor")])
    with pytest.raises(VariationError):
        VariationSampler(circuit, model)


# ---------------------------------------------------------------------------
# end to end: does the circuit-level mismatch really scale?
# ---------------------------------------------------------------------------

def mirror_at_geometry(w_um: float, l_um: float):
    """Rebuild the current mirror with a different device geometry."""
    from siliconstat.core.netlist import parse_netlist

    text = (load("current_mirror").source_text or "")
    text = text.replace(".param WM=10u", f".param WM={w_um}u")
    text = text.replace(".param LM=1u", f".param LM={l_um}u")
    return parse_netlist(text, allow_include=False)


def sigma_of_copy_error(circuit, samples: int = 700, seed: int = 909) -> float:
    """Local-mismatch-only sigma of the mirror copy error, in percent."""
    model = default_mismatch_model(circuit, enable_process=False,
                                   include_passives=False)
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=model, samples=samples, seed=seed, workers=1))
    assert run.counters.failed == 0
    return float(np.std(run.values("ierr"), ddof=1))


@pytest.mark.slow
def test_scaling_both_dimensions_halves_the_copy_error_sigma():
    """Quadrupling the area at *constant W/L* holds the overdrive fixed, so
    Pelgrom's law shows through directly as a factor-of-two improvement."""
    small = sigma_of_copy_error(mirror_at_geometry(10, 1))
    large = sigma_of_copy_error(mirror_at_geometry(20, 2))
    ratio = small / large
    # SE(std) = sigma/sqrt(2N) is 2.7 % per run at N = 700, so the ratio of two
    # independent runs carries about 3.8 %; 4 SE gives a 15 % window.
    assert ratio == pytest.approx(2.0, rel=0.15)


@pytest.mark.slow
def test_widening_alone_does_not_help_at_a_fixed_bias_current():
    """The trap the law invites.  At fixed Id, widening lowers Vov, so gm/Id
    rises as sqrt(W) and exactly cancels the sqrt(W) reduction in sigma(dVth).
    The copy error should therefore be roughly unchanged."""
    narrow = sigma_of_copy_error(mirror_at_geometry(10, 1))
    wide = sigma_of_copy_error(mirror_at_geometry(40, 1))     # 4x area, W only
    assert wide == pytest.approx(narrow, rel=0.2)


@pytest.mark.slow
def test_lengthening_alone_does_improve_matching_at_a_fixed_current():
    """Increasing L raises Vov (beta falls), which lowers gm/Id *and* the area
    grows, so both terms push the same way."""
    short = sigma_of_copy_error(mirror_at_geometry(10, 1))
    long = sigma_of_copy_error(mirror_at_geometry(10, 4))
    assert long < short * 0.6
