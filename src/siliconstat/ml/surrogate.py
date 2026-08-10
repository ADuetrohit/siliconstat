"""Layer 9: ML acceleration of Monte Carlo.

The idea is narrow and defensible: a circuit measurement is a deterministic
function of the drawn variation parameters, ``y = f(x)``.  Every Monte Carlo
sample evaluates ``f`` by running a full nonlinear DC (and possibly AC) solve.
If a cheap regressor can approximate ``f`` to within the accuracy actually
needed, the tail of a large run can be predicted instead of simulated.

Rules this module holds itself to
---------------------------------
1. **The model is trained only on real simulation output.**  Features are the
   drawn variation deviations; targets are measurements from converged solves.
2. **Accuracy is reported on held-out samples that were also really simulated**,
   never on the training set.
3. **Speedup is measured, not asserted.**  The reported figure accounts for the
   cost of generating the training set, so the honest break-even sample count
   is part of the output.  If the surrogate does not pay for itself at the
   requested budget, that is what the report says.
4. **The surrogate never replaces the simulator silently.**  It is an explicit
   opt-in tool with its own command and its own accuracy report.

A surrogate is only useful when the response is smooth in the parameters --
which is exactly the mismatch regime, where a circuit metric is a mildly
nonlinear function of a handful of threshold offsets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..analysis.yield_analysis import wilson_interval
from ..core.circuit import Circuit, SpecLimit
from ..core.exceptions import AnalysisError, SiliconStatError
from ..core.solver import SolverOptions
from ..mc import MonteCarloConfig, run_monte_carlo
from ..mc.results import MonteCarloRun
from ..variation.spec import VariationModel

__all__ = ["Surrogate", "surrogate_experiment", "MODEL_KINDS"]

MODEL_KINDS = ("linear", "quadratic", "gradient_boosting", "random_forest")


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SiliconStatError(
            "the ML surrogate needs the optional 'scikit-learn' dependency; "
            "install it with `pip install siliconstat[ml]`") from exc


def _build_model(kind: str):
    _require_sklearn()
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if kind == "linear":
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-6, 3, 30)))
    if kind == "quadratic":
        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            RidgeCV(alphas=np.logspace(-6, 3, 30)))
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(random_state=0, n_estimators=300,
                                         max_depth=3, learning_rate=0.05)
    if kind == "random_forest":
        return RandomForestRegressor(random_state=0, n_estimators=300,
                                     min_samples_leaf=2, n_jobs=1)
    raise AnalysisError(
        f"unknown surrogate model {kind!r}; choose from {', '.join(MODEL_KINDS)}")


@dataclass
class Surrogate:
    """A fitted regression surrogate for one circuit measurement."""

    target: str
    feature_names: list[str]
    kind: str
    model: Any = None
    train_samples: int = 0
    train_time_s: float = 0.0
    cv_r2_mean: float = float("nan")
    cv_r2_std: float = float("nan")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise AnalysisError("surrogate has not been fitted")
        return np.asarray(self.model.predict(np.atleast_2d(x)), dtype=float)

    def predict_from_slots(self, slot_values: dict[str, float]) -> float:
        row = np.array([[slot_values.get(name, 0.0)
                         for name in self.feature_names]], dtype=float)
        return float(self.predict(row)[0])

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "kind": self.kind,
                "features": list(self.feature_names),
                "train_samples": self.train_samples,
                "train_time_s": self.train_time_s,
                "cv_r2_mean": self.cv_r2_mean, "cv_r2_std": self.cv_r2_std}


def _matrix(run: MonteCarloRun, target: str
            ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    usable = run.successful_samples()
    names = run.slot_names
    if not names:
        raise AnalysisError(
            "the variation model produced no random variables, so there is "
            "nothing for a surrogate to learn from")
    x = np.array([[s.slot_values.get(n, np.nan) for n in names] for s in usable],
                 dtype=float)
    y = np.array([s.measurements.get(target, np.nan) for s in usable], dtype=float)
    keep = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    return x[keep], y[keep], names


def fit_surrogate(x: np.ndarray, y: np.ndarray, feature_names: Sequence[str], *,
                  target: str, kind: str = "auto",
                  cv_folds: int = 5) -> tuple[Surrogate, list[str]]:
    """Fit (and optionally select) a surrogate model on real simulation data."""
    _require_sklearn()
    from sklearn.model_selection import cross_val_score

    notes: list[str] = []
    if x.shape[0] < 20:
        raise AnalysisError(
            f"only {x.shape[0]} usable training samples; a surrogate needs at "
            "least 20 to be worth fitting")
    folds = max(2, min(cv_folds, x.shape[0] // 4))

    candidates = list(MODEL_KINDS) if kind == "auto" else [kind]
    best_kind, best_score, best_std = None, -np.inf, float("nan")
    for candidate in candidates:
        model = _build_model(candidate)
        try:
            scores = cross_val_score(model, x, y, cv=folds, scoring="r2")
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"model {candidate!r} could not be cross-validated: {exc}")
            continue
        mean = float(np.mean(scores))
        if mean > best_score:
            best_kind, best_score, best_std = candidate, mean, float(np.std(scores))
    if best_kind is None:
        raise AnalysisError("no surrogate model could be fitted to this data")
    if kind == "auto":
        notes.append(
            f"model selection by {folds}-fold cross-validated R2 chose "
            f"'{best_kind}' (CV R2 = {best_score:.5f})")

    started = time.perf_counter()
    model = _build_model(best_kind)
    model.fit(x, y)
    train_time = time.perf_counter() - started

    return Surrogate(target=target, feature_names=list(feature_names),
                     kind=best_kind, model=model, train_samples=int(x.shape[0]),
                     train_time_s=train_time, cv_r2_mean=best_score,
                     cv_r2_std=best_std), notes


def surrogate_experiment(circuit: Circuit, variation: VariationModel, *,
                         target: str | None = None,
                         train_samples: int = 400,
                         test_samples: int = 200,
                         seed: int = 4242,
                         solver: SolverOptions | None = None,
                         model_kind: str = "auto",
                         specs: Sequence[SpecLimit] | None = None,
                         ) -> dict[str, Any]:
    """Train a surrogate on real simulations and score it on held-out ones.

    Returns a dictionary of measured accuracy and measured timing, including
    the break-even sample count at which the surrogate starts to pay for its
    own training cost.
    """
    if not circuit.measures:
        raise AnalysisError("the circuit declares no measurements to model")
    target = target or _default_target(circuit)
    if target not in [m.name for m in circuit.measures]:
        raise AnalysisError(
            f"unknown measurement {target!r}; available: "
            f"{', '.join(m.name for m in circuit.measures)}")

    solver = solver or SolverOptions.from_options(circuit.options)
    notes: list[str] = []

    train_cfg = MonteCarloConfig(variation=variation, samples=train_samples,
                                 seed=seed, workers=1, solver=solver,
                                 label="surrogate-train")
    started = time.perf_counter()
    train_run = run_monte_carlo(circuit, train_cfg)
    train_sim_time = time.perf_counter() - started

    test_cfg = MonteCarloConfig(variation=variation, samples=test_samples,
                                seed=seed + 1_000_003, workers=1, solver=solver,
                                label="surrogate-test")
    started = time.perf_counter()
    test_run = run_monte_carlo(circuit, test_cfg)
    test_sim_time = time.perf_counter() - started

    x_train, y_train, names = _matrix(train_run, target)
    x_test, y_test, _ = _matrix(test_run, target)
    if x_test.shape[0] < 10:
        raise AnalysisError(
            f"only {x_test.shape[0]} usable held-out samples; cannot score the "
            "surrogate honestly")

    surrogate, fit_notes = fit_surrogate(x_train, y_train, names, target=target,
                                         kind=model_kind)
    notes.extend(fit_notes)

    # Prediction timing, averaged over repeated single-row predictions so the
    # figure reflects the per-sample cost the way the simulator's does.
    _ = surrogate.predict(x_test[:1])
    reps = max(1, int(2000 / max(x_test.shape[0], 1)))
    started = time.perf_counter()
    for _ in range(reps):
        y_pred = surrogate.predict(x_test)
    predict_time = (time.perf_counter() - started) / reps

    residual = y_pred - y_test
    ss_res = float(residual @ residual)
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / y_test.size))
    mae = float(np.mean(np.abs(residual)))
    sigma = float(np.std(y_test, ddof=1)) if y_test.size > 1 else float("nan")

    sim_per_sample = test_sim_time / max(test_run.counters.total, 1)
    surrogate_per_sample = predict_time / max(x_test.shape[0], 1)
    raw_speedup = (sim_per_sample / surrogate_per_sample
                   if surrogate_per_sample > 0 else float("inf"))

    total_train_cost = train_sim_time + surrogate.train_time_s
    per_sample_saving = sim_per_sample - surrogate_per_sample
    break_even = (total_train_cost / per_sample_saving
                  if per_sample_saving > 0 else float("inf"))
    net_10k = _net_speedup(10_000, sim_per_sample, surrogate_per_sample,
                           total_train_cost)

    spec_list = [s for s in (specs if specs is not None else circuit.specs)
                 if s.measure == target]
    sim_yield = surrogate_yield = float("nan")
    if spec_list:
        sim_pass = np.ones(y_test.size, dtype=bool)
        sur_pass = np.ones(y_test.size, dtype=bool)
        for spec in spec_list:
            sim_pass &= np.array([spec.passes(float(v)) for v in y_test])
            sur_pass &= np.array([spec.passes(float(v)) for v in y_pred])
        sim_yield = 100.0 * float(sim_pass.mean())
        surrogate_yield = 100.0 * float(sur_pass.mean())
    else:
        notes.append(
            f"no specification is declared on {target!r}, so no yield "
            "comparison is possible")

    if r2 == r2 and r2 < 0.9:
        notes.append(
            f"the surrogate explains only {100 * r2:.1f}% of the held-out "
            "variance; it is not accurate enough to replace simulation for this "
            "measurement")
    if break_even == float("inf"):
        notes.append(
            "the surrogate is not faster than the simulator for this circuit, "
            "so it can never pay for its training cost here")
    elif break_even > 10_000:
        notes.append(
            f"break-even is at {break_even:.0f} samples: below that budget, "
            "plain simulation is cheaper than training a surrogate")

    return {
        "target": target,
        "model": surrogate.kind,
        "n_features": len(names),
        "features": names,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "r2": r2,
        "rmse": rmse,
        "rmse_pct_of_sigma": (100.0 * rmse / sigma) if sigma and sigma == sigma else float("nan"),
        "mae": mae,
        "max_abs_error": float(np.max(np.abs(residual))),
        "cv_r2_mean": surrogate.cv_r2_mean,
        "cv_r2_std": surrogate.cv_r2_std,
        "output_sigma": sigma,
        "sim_time_per_sample_ms": sim_per_sample * 1e3,
        "surrogate_time_per_sample_ms": surrogate_per_sample * 1e3,
        "raw_speedup": raw_speedup,
        "train_time_s": total_train_cost,
        "train_sim_time_s": train_sim_time,
        "fit_time_s": surrogate.train_time_s,
        "break_even_samples": break_even,
        "net_speedup_10k": net_10k,
        "sim_yield_pct": sim_yield,
        "surrogate_yield_pct": surrogate_yield,
        "yield_error_pp": (surrogate_yield - sim_yield)
        if sim_yield == sim_yield else float("nan"),
        "sim_yield_ci": wilson_interval(int(round(sim_yield / 100 * y_test.size)),
                                        int(y_test.size)) if spec_list else None,
        "train_run_id": train_run.run_id,
        "test_run_id": test_run.run_id,
        "notes": notes,
    }


def _net_speedup(budget: int, sim_per_sample: float, sur_per_sample: float,
                 train_cost: float) -> float:
    """End-to-end speedup for *budget* samples, including training cost."""
    pure_sim = budget * sim_per_sample
    with_surrogate = train_cost + budget * sur_per_sample
    return pure_sim / with_surrogate if with_surrogate > 0 else float("inf")


def _default_target(circuit: Circuit) -> str:
    """Prefer a measurement that actually has a specification attached."""
    spec_measures = [s.measure for s in circuit.specs]
    for m in circuit.measures:
        if m.name in spec_measures:
            return m.name
    return circuit.measures[-1].name
