"""MOSFET square-law physics and its analytic derivatives."""

from __future__ import annotations

import math

import pytest

from siliconstat.core.context import SimContext
from siliconstat.core.models import MosfetModel, thermal_voltage
from siliconstat.core.mosfet import eval_mos_core, eval_mosfet, fet_limit

KP = 200e-6
BETA = KP * 10.0          # W/L = 10
CORE = dict(vth0=0.45, beta=BETA, lambda_=0.1, gamma=0.0, phi=0.8)
CORE_BODY = dict(CORE, gamma=0.4)


def ids(vgs, vds, vbs=0.0, **overrides):
    params = {**CORE, **overrides}
    return eval_mos_core(vgs, vds, vbs, **params)[0]


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------

def test_cutoff_conducts_nothing():
    current, gm, gds, gmb, vth, region = eval_mos_core(0.4, 1.0, 0.0, **CORE)
    assert region == "cutoff"
    assert (current, gm, gds, gmb) == (0.0, 0.0, 0.0, 0.0)
    assert vth == pytest.approx(0.45)


def test_exactly_at_threshold_is_cutoff():
    assert eval_mos_core(0.45, 1.0, 0.0, **CORE)[5] == "cutoff"


def test_triode_region():
    vgs, vds = 1.0, 0.1          # vov = 0.55 > vds -> triode
    current, gm, gds, _gmb, _vth, region = eval_mos_core(vgs, vds, 0.0, **CORE)
    assert region == "triode"
    expected = BETA * (0.55 * vds - 0.5 * vds ** 2) * (1 + 0.1 * vds)
    assert current == pytest.approx(expected, rel=1e-12)


def test_saturation_region():
    vgs, vds = 1.0, 1.2          # vov = 0.55 < vds -> saturation
    current, gm, gds, _gmb, _vth, region = eval_mos_core(vgs, vds, 0.0, **CORE)
    assert region == "saturation"
    expected = 0.5 * BETA * 0.55 ** 2 * (1 + 0.1 * vds)
    assert current == pytest.approx(expected, rel=1e-12)


def test_saturation_gm_equals_two_id_over_vov():
    """The relation every analog designer uses: gm = 2*Id/Vov in saturation."""
    vgs, vds = 1.0, 1.2
    current, gm, _gds, _gmb, vth, _r = eval_mos_core(vgs, vds, 0.0, **CORE)
    vov = vgs - vth
    assert gm == pytest.approx(2.0 * current / vov, rel=1e-12)


def test_channel_length_modulation_applies_in_both_regions_so_ids_is_continuous():
    """At vds == vov the triode and saturation branches must agree exactly.

    This is why (1 + LAMBDA*vds) multiplies both branches: applying it only in
    saturation would leave a step at the boundary.
    """
    vgs = 1.0
    vov = vgs - 0.45
    below = ids(vgs, vov - 1e-9)
    above = ids(vgs, vov + 1e-9)
    assert below == pytest.approx(above, rel=1e-7)


def test_derivatives_are_continuous_across_the_region_boundary():
    vgs = 1.0
    vov = vgs - 0.45
    _i, gm_lo, gds_lo, _b, _v, _r = eval_mos_core(vgs, vov - 1e-9, 0.0, **CORE)
    _i, gm_hi, gds_hi, _b, _v, _r = eval_mos_core(vgs, vov + 1e-9, 0.0, **CORE)
    assert gm_lo == pytest.approx(gm_hi, rel=1e-6)
    assert gds_lo == pytest.approx(gds_hi, abs=1e-9)


# ---------------------------------------------------------------------------
# derivatives against finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vgs,vds,vbs", [
    (1.0, 1.2, 0.0),     # saturation
    (1.0, 0.1, 0.0),     # triode
    (0.9, 0.8, -0.5),    # saturation with reverse body bias
    (1.4, 0.2, -0.3),    # triode with body bias
])
def test_gm_gds_gmb_match_finite_differences(vgs, vds, vbs):
    """The analytic Jacobian is what Newton-Raphson relies on; verify it."""
    h = 1e-7
    base = eval_mos_core(vgs, vds, vbs, **CORE_BODY)
    _i0, gm, gds, gmb, _vth, _region = base

    d_gm = (ids(vgs + h, vds, vbs, **CORE_BODY)
            - ids(vgs - h, vds, vbs, **CORE_BODY)) / (2 * h)
    d_gds = (ids(vgs, vds + h, vbs, **CORE_BODY)
             - ids(vgs, vds - h, vbs, **CORE_BODY)) / (2 * h)
    d_gmb = (ids(vgs, vds, vbs + h, **CORE_BODY)
             - ids(vgs, vds, vbs - h, **CORE_BODY)) / (2 * h)

    assert gm == pytest.approx(d_gm, rel=2e-5)
    assert gds == pytest.approx(d_gds, rel=2e-5)
    assert gmb == pytest.approx(d_gmb, rel=2e-5)


def test_gmb_closed_form():
    """gmb = gm * GAMMA / (2*sqrt(PHI - vbs))."""
    vgs, vds, vbs = 1.0, 1.2, -0.4
    _i, gm, _gds, gmb, _vth, _r = eval_mos_core(vgs, vds, vbs, **CORE_BODY)
    expected = gm * CORE_BODY["gamma"] / (2.0 * math.sqrt(CORE_BODY["phi"] - vbs))
    assert gmb == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# body effect
# ---------------------------------------------------------------------------

def test_reverse_body_bias_raises_the_threshold():
    _i, _gm, _gds, _gmb, vth_0, _r = eval_mos_core(1.0, 1.2, 0.0, **CORE_BODY)
    _i, _gm, _gds, _gmb, vth_1, _r = eval_mos_core(1.0, 1.2, -0.5, **CORE_BODY)
    assert vth_1 > vth_0
    expected = 0.45 + 0.4 * (math.sqrt(0.8 + 0.5) - math.sqrt(0.8))
    assert vth_1 == pytest.approx(expected, rel=1e-12)


def test_zero_gamma_means_no_body_effect():
    _i, _gm, _gds, gmb, vth, _r = eval_mos_core(1.0, 1.2, -0.5, **CORE)
    assert vth == pytest.approx(0.45)
    assert gmb == 0.0


# ---------------------------------------------------------------------------
# instance-level evaluation: polarity, reverse mode, overrides
# ---------------------------------------------------------------------------

def nmos_model(**kwargs) -> MosfetModel:
    return MosfetModel(name="NCH", mtype="nmos", vto=0.45, kp=KP,
                       lambda_=0.1, gamma=0.0, phi=0.8, **kwargs)


def pmos_model(**kwargs) -> MosfetModel:
    return MosfetModel(name="PCH", mtype="pmos", vto=0.45, kp=KP,
                       lambda_=0.1, gamma=0.0, phi=0.8, **kwargs)


def test_pmos_mirrors_nmos_under_a_sign_flip():
    """A PMOS with mirrored terminal voltages must carry the same |Ids|."""
    n = eval_mosfet(nmos_model(), 10e-6, 1e-6, v_d=1.2, v_g=1.0, v_s=0.0, v_b=0.0)
    p = eval_mosfet(pmos_model(), 10e-6, 1e-6, v_d=-1.2, v_g=-1.0, v_s=0.0, v_b=0.0)
    assert p.ids == pytest.approx(n.ids, rel=1e-12)
    assert p.gm == pytest.approx(n.gm, rel=1e-12)
    assert p.region == n.region


def test_pmos_vto_sign_is_normalised_to_a_magnitude():
    """Both the SPICE (negative) and magnitude conventions must work."""
    negative = MosfetModel(name="P", mtype="pmos", vto=-0.45, kp=KP)
    positive = MosfetModel(name="P", mtype="pmos", vto=0.45, kp=KP)
    assert negative.vto == pytest.approx(positive.vto)


def test_reverse_mode_swaps_drain_and_source():
    """With vds < 0 the model must re-reference to the true source."""
    forward = eval_mosfet(nmos_model(), 10e-6, 1e-6, v_d=1.2, v_g=1.0,
                          v_s=0.0, v_b=0.0)
    reverse = eval_mosfet(nmos_model(), 10e-6, 1e-6, v_d=0.0, v_g=1.0,
                          v_s=1.2, v_b=0.0)
    assert reverse.swapped is True
    assert forward.swapped is False
    assert reverse.ids == pytest.approx(forward.ids, rel=1e-12)
    assert reverse.vds >= 0.0


def test_geometry_scales_the_current_factor():
    narrow = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0)
    wide = eval_mosfet(nmos_model(), 20e-6, 1e-6, 1.2, 1.0, 0.0, 0.0)
    assert wide.ids == pytest.approx(2.0 * narrow.ids, rel=1e-12)
    longer = eval_mosfet(nmos_model(), 10e-6, 2e-6, 1.2, 1.0, 0.0, 0.0)
    assert longer.ids == pytest.approx(0.5 * narrow.ids, rel=1e-12)


def test_multiplicity_multiplies_the_current():
    single = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0)
    doubled = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0,
                          multiplicity=2.0)
    assert doubled.ids == pytest.approx(2.0 * single.ids, rel=1e-12)


def test_statistical_overrides_shift_threshold_and_scale_beta():
    base = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0)
    shifted = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0,
                          vth0=0.50)
    assert shifted.vth == pytest.approx(0.50)
    assert shifted.ids < base.ids
    scaled = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0,
                         kp=2 * KP)
    assert scaled.ids == pytest.approx(2 * base.ids, rel=1e-12)


def test_lateral_diffusion_shortens_the_effective_channel():
    plain = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0)
    with_ld = eval_mosfet(nmos_model(ld=0.1e-6), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0)
    # Leff = 1u - 2*0.1u = 0.8u, so beta rises by 1/0.8.
    assert with_ld.ids == pytest.approx(plain.ids / 0.8, rel=1e-12)


def test_ro_and_gm_over_id_helpers():
    op = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0)
    assert op.ro == pytest.approx(1.0 / op.gds, rel=1e-12)
    assert op.gm_over_id == pytest.approx(op.gm / op.ids, rel=1e-12)
    cutoff = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 0.1, 0.0, 0.0)
    assert math.isinf(cutoff.ro)
    assert math.isinf(cutoff.gm_over_id)


# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------

def test_temperature_lowers_threshold_and_mobility():
    model = MosfetModel(name="N", mtype="nmos", vto=0.45, kp=KP, tnom=27.0,
                        tcv=-1e-3, bex=-1.5)
    hot = model.at_temperature(127.0)
    assert hot.vto == pytest.approx(0.45 - 1e-3 * 100.0)
    ratio = (127.0 + 273.15) / (27.0 + 273.15)
    assert hot.kp == pytest.approx(KP * ratio ** -1.5, rel=1e-12)
    cold = model.at_temperature(-40.0)
    assert cold.vto > model.vto      # colder -> higher threshold
    assert cold.kp > model.kp        # colder -> higher mobility


def test_at_nominal_temperature_returns_the_same_card():
    model = MosfetModel(name="N", mtype="nmos", tnom=27.0)
    assert model.at_temperature(27.0) is model


def test_thermal_voltage():
    assert thermal_voltage(27.0) == pytest.approx(0.025864, abs=1e-6)


# ---------------------------------------------------------------------------
# capacitances
# ---------------------------------------------------------------------------

def test_meyer_capacitances_by_region():
    model = nmos_model()
    cox_total = model.cox_area * 10e-6 * 1e-6

    sat = eval_mosfet(model, 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0, with_caps=True)
    assert sat.region == "saturation"
    assert sat.cgs == pytest.approx(2 / 3 * cox_total + model.cgso * 10e-6, rel=1e-9)
    assert sat.cgd == pytest.approx(model.cgdo * 10e-6, rel=1e-9)

    tri = eval_mosfet(model, 10e-6, 1e-6, 0.1, 1.0, 0.0, 0.0, with_caps=True)
    assert tri.region == "triode"
    assert tri.cgs == pytest.approx(0.5 * cox_total + model.cgso * 10e-6, rel=1e-9)
    assert tri.cgd == pytest.approx(0.5 * cox_total + model.cgdo * 10e-6, rel=1e-9)

    cut = eval_mosfet(model, 10e-6, 1e-6, 1.2, 0.1, 0.0, 0.0, with_caps=True)
    assert cut.region == "cutoff"
    assert cut.cgb == pytest.approx(cox_total, rel=1e-9)


def test_caps_are_zero_when_not_requested():
    op = eval_mosfet(nmos_model(), 10e-6, 1e-6, 1.2, 1.0, 0.0, 0.0)
    assert (op.cgs, op.cgd, op.cgb) == (0.0, 0.0, 0.0)


def test_cox_from_tox():
    model = MosfetModel(name="N", mtype="nmos", tox=4e-9)
    assert model.cox_area == pytest.approx(3.9 * 8.8541878128e-12 / 4e-9)
    explicit = MosfetModel(name="N", mtype="nmos", cox=1e-3)
    assert explicit.cox_area == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# model card validation and limiting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(mtype="bogus"),
    dict(kp=0.0),
    dict(kp=-1.0),
    dict(phi=0.0),
    dict(gamma=-0.1),
])
def test_invalid_model_cards_are_rejected(kwargs):
    from siliconstat.core.exceptions import NetlistSyntaxError

    base = dict(name="X", mtype="nmos")
    base.update(kwargs)
    with pytest.raises(NetlistSyntaxError):
        MosfetModel(**base)


def test_fet_limit_bounds_the_step_but_is_a_no_op_near_convergence():
    # Large step near threshold -> clamped to the tight bound.
    assert fet_limit(10.0, 0.45, 0.45) == pytest.approx(0.95)
    # Large step far from threshold -> clamped to the loose bound.
    assert fet_limit(10.0, 3.0, 0.45) == pytest.approx(5.0)
    # Small step -> untouched, so Newton keeps its quadratic rate.
    assert fet_limit(0.4500001, 0.45, 0.45) == pytest.approx(0.4500001)
    assert fet_limit(-10.0, 0.45, 0.45) == pytest.approx(-0.05)


def test_pelgrom_helpers_on_the_model_card():
    model = MosfetModel(name="N", mtype="nmos", avt=3.5e-3, abeta=0.01)
    # sigma_vth() is the raw AVT/sqrt(area) (pair) convention.
    assert model.sigma_vth(10e-6, 1e-6) == pytest.approx(3.5e-3 / math.sqrt(10.0))
    assert model.sigma_beta_rel(10e-6, 1e-6) == pytest.approx(0.01 / math.sqrt(10.0))
