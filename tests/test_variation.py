"""The variation model: scopes, matched groups, selectors and slot construction.

The central property is the global/local split.  Two transistors sharing a
model card must receive the *same* global draw and *independent* local draws::

    Vth(M1) = Vth_nom + dVth_global(NCH) + dVth_local(M1)
    Vth(M2) = Vth_nom + dVth_global(NCH) + dVth_local(M2)

Everything a current mirror or differential pair does statistically follows
from that, so it is asserted directly on the drawn sample records.
"""

from __future__ import annotations

import numpy as np
import pytest

from siliconstat.core.exceptions import VariationError
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.variation import (
    CorrelationSpec,
    ParameterVariation,
    VariationModel,
    VariationSampler,
    default_mismatch_model,
    select_devices,
)

from conftest import load


def sampler_for(circuit, model=None) -> VariationSampler:
    return VariationSampler(circuit, model or default_mismatch_model(circuit))


# ---------------------------------------------------------------------------
# slot construction
# ---------------------------------------------------------------------------

def test_default_model_builds_the_expected_slots():
    circuit = load("current_mirror")
    names = set(sampler_for(circuit).slot_names)
    assert {"NCH.vth_global", "NCH.beta_global",
            "M1.vth_local", "M2.vth_local",
            "M1.beta_local", "M2.beta_local",
            "ALL.r_global", "R1.r_local"} <= names


def test_slot_order_is_deterministic():
    circuit = load("current_mirror")
    first = sampler_for(circuit).slot_names
    second = sampler_for(load("current_mirror")).slot_names
    assert first == second


def test_global_slots_list_every_device_they_cover():
    circuit = load("current_mirror")
    sampler = sampler_for(circuit)
    slot = next(s for s in sampler.slots if s.name == "NCH.vth_global")
    assert set(slot.devices) == {"M1", "M2"}
    assert slot.scope == "global"


def test_local_slots_cover_exactly_one_device():
    sampler = sampler_for(load("current_mirror"))
    for slot in sampler.slots:
        if slot.scope == "local":
            assert len(slot.devices) == 1


def test_relative_slots_are_labelled_as_fractions_not_ohms():
    """A relative deviation is dimensionless; labelling it 'ohm' would be a
    unit lie in the report."""
    sampler = sampler_for(load("current_mirror"))
    r_slot = next(s for s in sampler.slots if s.name == "R1.r_local")
    assert r_slot.unit == "frac"
    vth_slot = next(s for s in sampler.slots if s.name == "M1.vth_local")
    assert vth_slot.unit == "V"


def test_disabling_process_or_mismatch_removes_the_matching_slots():
    circuit = load("current_mirror")
    mismatch_only = sampler_for(circuit, default_mismatch_model(
        circuit, enable_process=False))
    assert all(s.scope == "local" for s in mismatch_only.slots)

    process_only = sampler_for(circuit, default_mismatch_model(
        circuit, enable_mismatch=False))
    assert all(s.scope == "global" for s in process_only.slots)

    nominal = default_mismatch_model(circuit, enable_process=False,
                                     enable_mismatch=False)
    assert VariationSampler(circuit, nominal).n_slots == 0
    assert nominal.mode_label == "nominal"


def test_mode_labels():
    circuit = load("current_mirror")
    assert default_mismatch_model(circuit).mode_label == "process+mismatch"
    assert default_mismatch_model(
        circuit, enable_mismatch=False).mode_label == "process"
    assert default_mismatch_model(
        circuit, enable_process=False).mode_label == "mismatch"


# ---------------------------------------------------------------------------
# the global / local split
# ---------------------------------------------------------------------------

def test_devices_sharing_a_model_share_the_global_draw_exactly():
    circuit = load("current_mirror")
    model = default_mismatch_model(circuit)
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=model, samples=40, seed=5, workers=1))
    for sample in run.samples:
        # The global slot has one value that both devices receive.
        assert "NCH.vth_global" in sample.slot_values
        # Both device thresholds move by that shared amount plus their own local.
        shared = sample.slot_values["NCH.vth_global"]
        d1 = sample.device_values["M1.dvth"]
        d2 = sample.device_values["M2.dvth"]
        assert d1 - sample.slot_values["M1.vth_local"] == pytest.approx(shared)
        assert d2 - sample.slot_values["M2.vth_local"] == pytest.approx(shared)


def test_local_draws_are_independent_between_devices():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit, enable_process=False),
        samples=2000, seed=17, workers=1))
    a = np.array([s.slot_values["M1.vth_local"] for s in run.samples])
    b = np.array([s.slot_values["M2.vth_local"] for s in run.samples])
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.06   # ~4 SE at N = 2000
    assert not np.allclose(a, b)


def test_process_only_leaves_the_two_devices_perfectly_matched():
    """With no local mismatch the mirror error must stay at its nominal value:
    a common threshold shift cancels in the copy ratio."""
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit, enable_mismatch=False,
                                         include_passives=False),
        samples=60, seed=3, workers=1))
    for sample in run.samples:
        assert sample.device_values["M1.dvth"] == pytest.approx(
            sample.device_values["M2.dvth"])
    # A common threshold shift cancels to first order, but it does move the
    # bias point: nref and out shift, so the channel-length-modulation ratio
    # (1+lambda*Vds2)/(1+lambda*Vds1) changes slightly.  That residual is real
    # physics, and it is an order of magnitude below the mismatch-driven spread.
    spread = np.std(run.values("ierr"), ddof=1)
    assert spread < 0.4           # percent


def test_mismatch_only_produces_a_real_spread():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit, enable_process=False),
        samples=300, seed=3, workers=1))
    assert np.std(run.values("ierr"), ddof=1) > 1.0     # percent


# ---------------------------------------------------------------------------
# matched groups and sharing
# ---------------------------------------------------------------------------

def test_matched_group_sharing_creates_one_slot_per_group():
    circuit = load("two_stage_opamp")
    model = VariationModel(variations=[
        ParameterVariation(parameter="vth", scope="global", sigma=0.02,
                           targets="type:mosfet", share_by="matched_group")])
    sampler = VariationSampler(circuit, model)
    names = set(sampler.slot_names)
    assert "INPAIR.vth_global" in names
    assert "LOADMIR.vth_global" in names
    assert "BIASMIR.vth_global" in names
    inpair = next(s for s in sampler.slots if s.name == "INPAIR.vth_global")
    assert set(inpair.devices) == {"M1", "M2"}


def test_matched_group_members_receive_the_same_shift():
    circuit = load("two_stage_opamp")
    model = VariationModel(variations=[
        ParameterVariation(parameter="vth", scope="global", sigma=0.02,
                           targets="group:INPAIR", share_by="matched_group")])
    sampler = VariationSampler(circuit, model)
    deviations = sampler.draw(np.random.default_rng(1))
    overrides, _ = sampler.apply(deviations)
    assert overrides["M1"]["dvth"] == pytest.approx(overrides["M2"]["dvth"])


def test_share_by_all_puts_every_device_on_one_slot():
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="r", scope="global", sigma_pct=2.0,
                           targets="type:resistor", share_by="all")])
    sampler = VariationSampler(circuit, model)
    assert sampler.slot_names == ["ALL.r_global"]


def test_devices_without_a_group_fall_back_to_their_model():
    circuit = load("two_stage_opamp")
    model = VariationModel(variations=[
        ParameterVariation(parameter="vth", scope="global", sigma=0.02,
                           targets="device:M6", share_by="matched_group")])
    sampler = VariationSampler(circuit, model)
    assert sampler.slot_names == ["PCH.vth_global"]      # M6 has no MATCH=


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------

def test_selector_forms():
    circuit = load("current_mirror")
    assert len(select_devices(circuit, "*")) == len(circuit.devices)
    assert {d.name for d in select_devices(circuit, "model:NCH")} == {"M1", "M2"}
    assert {d.name for d in select_devices(circuit, "group:MIRROR")} == {"M1", "M2"}
    assert {d.name for d in select_devices(circuit, "type:resistor")} == {"R1"}
    assert {d.name for d in select_devices(circuit, "device:M1")} == {"M1"}
    assert {d.name for d in select_devices(circuit, "M1")} == {"M1"}
    assert {d.name for d in select_devices(circuit, "M1,R1")} == {"M1", "R1"}


def test_selector_is_case_insensitive():
    circuit = load("current_mirror")
    assert {d.name for d in select_devices(circuit, "model:nch")} == {"M1", "M2"}
    assert {d.name for d in select_devices(circuit, "group:mirror")} == {"M1", "M2"}


@pytest.mark.parametrize("selector,fragment", [
    ("model:PCH", "models in this circuit"),
    ("group:NOPE", "matched groups in this circuit"),
    ("type:inductor", "matched no devices"),
    ("bogus:thing", "unknown selector prefix"),
])
def test_selectors_that_match_nothing_explain_what_is_available(selector, fragment):
    circuit = load("current_mirror")
    with pytest.raises(VariationError) as excinfo:
        select_devices(circuit, selector)
    assert fragment in str(excinfo.value)


def test_selector_matching_devices_without_the_parameter_is_rejected():
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="vth", scope="local", sigma=1e-3,
                           targets="type:resistor")])
    with pytest.raises(VariationError):
        VariationSampler(circuit, model)


# ---------------------------------------------------------------------------
# validation of the rules themselves
# ---------------------------------------------------------------------------

def test_a_rule_needs_a_sigma():
    with pytest.raises(VariationError) as excinfo:
        ParameterVariation(parameter="vth", scope="local")
    assert "sigma" in str(excinfo.value)


@pytest.mark.parametrize("kwargs", [
    dict(parameter="nonsense", sigma=1e-3),
    dict(parameter="vth", scope="sideways", sigma=1e-3),
    dict(parameter="vth", share_by="vibes", sigma=1e-3),
    dict(parameter="vth", sigma=-1e-3),
    dict(parameter="vth", sigma_pct=-1.0),
])
def test_invalid_rules_are_rejected(kwargs):
    with pytest.raises(VariationError):
        ParameterVariation(**kwargs)


def test_lognormal_on_an_additive_parameter_is_rejected():
    """A log-normal deviation is bounded below by -1, which is meaningless as
    an additive threshold shift."""
    with pytest.raises(VariationError) as excinfo:
        ParameterVariation(parameter="vth", scope="local", sigma=1e-3,
                           distribution="lognormal")
    assert "multiplicative" in str(excinfo.value)


def test_lognormal_is_fine_on_a_relative_parameter():
    rule = ParameterVariation(parameter="beta", scope="local", sigma=0.02,
                              distribution="lognormal")
    assert rule.effective_mode == "relative"


def test_rule_round_trips_through_dict():
    rule = ParameterVariation(parameter="vth", scope="global", sigma=0.025,
                              targets="model:NCH", label="global vth")
    restored = ParameterVariation.from_dict(rule.to_dict())
    assert restored.to_dict() == rule.to_dict()


def test_unknown_rule_fields_are_rejected():
    with pytest.raises(VariationError):
        ParameterVariation.from_dict({"parameter": "vth", "sigma": 1e-3,
                                      "colour": "blue"})


def test_model_round_trips_through_dict():
    circuit = load("current_mirror")
    model = default_mismatch_model(circuit)
    restored = VariationModel.from_dict(model.to_dict())
    assert len(restored.variations) == len(model.variations)
    assert VariationSampler(circuit, restored).slot_names == \
        VariationSampler(circuit, model).slot_names


# ---------------------------------------------------------------------------
# non-physical draws
# ---------------------------------------------------------------------------

def test_a_draw_that_would_invert_beta_is_rejected_with_an_explanation():
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="beta", scope="local", sigma_pct=200.0,
                           targets="type:mosfet")])
    sampler = VariationSampler(circuit, model)
    with pytest.raises(VariationError) as excinfo:
        sampler.apply(np.array([-1.5, -1.5]))
    message = str(excinfo.value)
    assert "non-physical" in message or "<= 0" in message


def test_a_draw_that_would_make_width_negative_is_rejected():
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="w", scope="local", sigma_pct=50.0,
                           targets="type:mosfet")])
    sampler = VariationSampler(circuit, model)
    with pytest.raises(VariationError) as excinfo:
        sampler.apply(np.array([-2.0, 0.0]))
    assert "not physical" in str(excinfo.value)


def test_truncation_clips_the_standard_normal():
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="vth", scope="local", sigma=0.01,
                           targets="type:mosfet", truncate_sigma=2.0)])
    sampler = VariationSampler(circuit, model)
    rng = np.random.default_rng(0)
    draws = np.array([sampler.draw(rng) for _ in range(4000)])
    assert np.max(np.abs(draws)) <= 2.0 * 0.01 + 1e-12


# ---------------------------------------------------------------------------
# correlation groups
# ---------------------------------------------------------------------------

def test_correlated_slots_come_out_correlated():
    circuit = load("current_mirror")
    model = VariationModel(
        variations=[
            ParameterVariation(parameter="vth", scope="local", sigma=5e-3,
                               targets="type:mosfet", correlation_group="pair")],
        correlations=[CorrelationSpec("pair", rho=0.8)])
    sampler = VariationSampler(circuit, model)
    rng = np.random.default_rng(99)
    draws = np.array([sampler.draw(rng) for _ in range(20_000)])
    empirical = float(np.corrcoef(draws[:, 0], draws[:, 1])[0, 1])
    assert empirical == pytest.approx(0.8, abs=0.02)


def test_an_undefined_correlation_group_is_reported():
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="vth", scope="local", sigma=1e-3,
                           targets="type:mosfet", correlation_group="ghost")])
    with pytest.raises(VariationError) as excinfo:
        VariationSampler(circuit, model)
    assert "ghost" in str(excinfo.value)


def test_an_unused_correlation_group_is_reported():
    circuit = load("current_mirror")
    model = VariationModel(
        variations=[ParameterVariation(parameter="vth", scope="local",
                                       sigma=1e-3, targets="type:mosfet")],
        correlations=[CorrelationSpec("unused", rho=0.5)])
    with pytest.raises(VariationError) as excinfo:
        VariationSampler(circuit, model)
    assert "unused" in str(excinfo.value)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_slot_table_is_serialisable_and_complete():
    sampler = sampler_for(load("current_mirror"))
    rows = sampler.slot_sigma_table()
    assert len(rows) == sampler.n_slots
    for row in rows:
        assert set(row) >= {"slot", "parameter", "scope", "distribution",
                            "sigma", "unit", "devices", "label"}
        assert row["sigma"] >= 0.0


def test_circuit_reports_its_variable_parameters():
    circuit = load("current_mirror")
    params = circuit.variable_parameters()
    entries = {(p["device"], p["parameter"]) for p in params}
    assert ("M1", "vth") in entries and ("R1", "r") in entries
