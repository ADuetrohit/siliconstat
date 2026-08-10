"""The ML surrogate.

These tests do not assert a particular speedup or accuracy -- those are
properties of the circuit, not of the code.  What they assert are the *honesty*
properties the module promises: trained on real simulations, scored on
held-out real simulations, and reporting a break-even point that accounts for
the cost of generating the training set.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("sklearn")

from siliconstat.core.exceptions import AnalysisError  # noqa: E402
from siliconstat.ml import MODEL_KINDS, fit_surrogate, surrogate_experiment  # noqa: E402
from siliconstat.variation import default_mismatch_model  # noqa: E402

from conftest import load  # noqa: E402


@pytest.fixture(scope="module")
def experiment():
    circuit = load("current_mirror")
    return surrogate_experiment(
        circuit, default_mismatch_model(circuit), target="ierr",
        train_samples=250, test_samples=120, seed=4242, model_kind="quadratic")


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def test_fit_recovers_a_known_linear_response():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    y = 2.0 * x[:, 0] - 1.0 * x[:, 1]
    surrogate, _notes = fit_surrogate(x, y, ["a", "b", "c"], target="y",
                                      kind="linear")
    predicted = surrogate.predict(x)
    assert np.max(np.abs(predicted - y)) < 1e-6
    assert surrogate.cv_r2_mean > 0.999


def test_auto_selection_reports_which_model_it_chose():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 2))
    y = x[:, 0] ** 2 + x[:, 1]
    _surrogate, notes = fit_surrogate(x, y, ["a", "b"], target="y", kind="auto")
    assert any("model selection" in note for note in notes)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_every_model_kind_fits(kind):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((150, 2))
    y = x[:, 0] + 0.5 * x[:, 1]
    surrogate, _notes = fit_surrogate(x, y, ["a", "b"], target="y", kind=kind)
    assert surrogate.kind == kind
    assert surrogate.predict(x[:5]).shape == (5,)


def test_too_few_training_samples_is_refused():
    rng = np.random.default_rng(3)
    with pytest.raises(AnalysisError) as excinfo:
        fit_surrogate(rng.standard_normal((10, 2)), rng.standard_normal(10),
                      ["a", "b"], target="y")
    assert "at least 20" in str(excinfo.value)


def test_unknown_model_kind_is_rejected():
    rng = np.random.default_rng(4)
    with pytest.raises(AnalysisError):
        fit_surrogate(rng.standard_normal((50, 2)), rng.standard_normal(50),
                      ["a", "b"], target="y", kind="crystal_ball")


def test_predict_from_named_slots():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((200, 2))
    y = 3.0 * x[:, 0]
    surrogate, _ = fit_surrogate(x, y, ["p", "q"], target="y", kind="linear")
    assert surrogate.predict_from_slots({"p": 1.0, "q": 0.0}) == pytest.approx(
        3.0, abs=1e-6)
    assert surrogate.to_dict()["features"] == ["p", "q"]


def test_an_unfitted_surrogate_refuses_to_predict():
    from siliconstat.ml import Surrogate

    with pytest.raises(AnalysisError):
        Surrogate(target="y", feature_names=["a"], kind="linear").predict(
            np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# the experiment: honesty properties
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_training_and_test_sets_are_both_real_simulations(experiment):
    assert experiment["train_samples"] > 0
    assert experiment["test_samples"] > 0
    # Separate runs, so separate run ids -- the score is genuinely held out.
    assert experiment["train_run_id"] != experiment["test_run_id"]


@pytest.mark.slow
def test_accuracy_is_reported_against_the_output_spread(experiment):
    assert math.isfinite(experiment["r2"])
    assert experiment["rmse"] >= 0.0
    assert experiment["mae"] >= 0.0
    assert experiment["max_abs_error"] >= experiment["mae"]
    assert experiment["output_sigma"] > 0
    assert experiment["rmse_pct_of_sigma"] == pytest.approx(
        100.0 * experiment["rmse"] / experiment["output_sigma"], rel=1e-9)


@pytest.mark.slow
def test_cross_validated_score_is_reported_separately(experiment):
    assert math.isfinite(experiment["cv_r2_mean"])
    assert experiment["cv_r2_std"] >= 0.0


@pytest.mark.slow
def test_timing_is_measured_not_asserted(experiment):
    assert experiment["sim_time_per_sample_ms"] > 0
    assert experiment["surrogate_time_per_sample_ms"] > 0
    assert experiment["train_sim_time_s"] > 0
    assert experiment["fit_time_s"] >= 0
    assert experiment["train_time_s"] >= experiment["train_sim_time_s"]


@pytest.mark.slow
def test_raw_speedup_is_the_ratio_of_the_two_measured_times(experiment):
    assert experiment["raw_speedup"] == pytest.approx(
        experiment["sim_time_per_sample_ms"]
        / experiment["surrogate_time_per_sample_ms"], rel=1e-9)


@pytest.mark.slow
def test_break_even_accounts_for_the_cost_of_the_training_set(experiment):
    """The number that makes the claim honest: below it, plain simulation is
    cheaper than training a surrogate."""
    break_even = experiment["break_even_samples"]
    assert break_even > 0
    if math.isfinite(break_even):
        saving = (experiment["sim_time_per_sample_ms"]
                  - experiment["surrogate_time_per_sample_ms"]) / 1e3
        assert break_even == pytest.approx(
            experiment["train_time_s"] / saving, rel=1e-6)


@pytest.mark.slow
def test_net_speedup_at_a_budget_includes_training(experiment):
    """Net speedup must be smaller than the raw per-prediction ratio."""
    assert experiment["net_speedup_10k"] < experiment["raw_speedup"]
    pure = 10_000 * experiment["sim_time_per_sample_ms"] / 1e3
    with_surrogate = (experiment["train_time_s"]
                      + 10_000 * experiment["surrogate_time_per_sample_ms"] / 1e3)
    assert experiment["net_speedup_10k"] == pytest.approx(pure / with_surrogate,
                                                          rel=1e-6)


@pytest.mark.slow
def test_yield_comparison_uses_simulated_ground_truth(experiment):
    assert 0.0 <= experiment["sim_yield_pct"] <= 100.0
    assert 0.0 <= experiment["surrogate_yield_pct"] <= 100.0
    assert experiment["yield_error_pp"] == pytest.approx(
        experiment["surrogate_yield_pct"] - experiment["sim_yield_pct"], rel=1e-9)


@pytest.mark.slow
def test_a_poor_surrogate_says_so_instead_of_being_recommended():
    """Force an underpowered model on a nonlinear target and check the warning
    machinery, rather than assuming every fit is good."""
    circuit = load("current_mirror")
    result = surrogate_experiment(
        circuit, default_mismatch_model(circuit), target="ierr",
        train_samples=60, test_samples=60, seed=11, model_kind="linear")
    if result["r2"] < 0.9:
        assert any("not accurate enough" in note for note in result["notes"])


@pytest.mark.slow
def test_unknown_target_is_rejected():
    circuit = load("current_mirror")
    with pytest.raises(AnalysisError) as excinfo:
        surrogate_experiment(circuit, default_mismatch_model(circuit),
                             target="not_a_measurement", train_samples=40,
                             test_samples=30)
    assert "not_a_measurement" in str(excinfo.value)


@pytest.mark.slow
def test_a_circuit_with_no_variation_has_nothing_to_learn():
    from siliconstat.variation import VariationModel

    circuit = load("current_mirror")
    with pytest.raises(AnalysisError) as excinfo:
        surrogate_experiment(circuit, VariationModel(variations=[]),
                             target="ierr", train_samples=40, test_samples=30)
    assert "nothing for a surrogate to learn" in str(excinfo.value)


@pytest.mark.slow
def test_default_target_prefers_a_specified_measurement():
    circuit = load("current_mirror")
    result = surrogate_experiment(circuit, default_mismatch_model(circuit),
                                  train_samples=40, test_samples=30,
                                  model_kind="linear")
    assert result["target"] in {s.measure for s in circuit.specs}
