"""Statistical convergence of a Monte Carlo estimate.

Monte Carlo answers are estimates, and their uncertainty shrinks only as
``1/sqrt(N)``.  A yield quoted from 100 samples has a confidence interval
roughly three times wider than the same yield from 1000 samples -- which is
exactly why a convergence plot belongs next to every reported number.

This module walks the run in order and reports, at each prefix length, the
running mean, standard deviation and yield together with their confidence
intervals.  The samples are already in a fixed, seed-determined order, so the
trace is reproducible rather than an artefact of scheduling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.circuit import SpecLimit
from ..core.exceptions import AnalysisError
from ..mc.results import MonteCarloRun
from .yield_analysis import _specs_from_run, wilson_interval

__all__ = ["ConvergenceTrace", "convergence_analysis", "required_samples_for_margin"]

_Z95 = 1.959963984540054


@dataclass
class ConvergenceTrace:
    measurement: str = ""
    unit: str = ""
    n: list[int] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    mean_ci_low: list[float] = field(default_factory=list)
    mean_ci_high: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    yield_pct: list[float] = field(default_factory=list)
    yield_ci_low: list[float] = field(default_factory=list)
    yield_ci_high: list[float] = field(default_factory=list)
    final_mean: float = float("nan")
    final_std: float = float("nan")
    final_yield_pct: float = float("nan")
    mean_settled_at: int | None = None
    std_settled_at: int | None = None
    yield_settled_at: int | None = None
    settle_tolerance_pct: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement": self.measurement, "unit": self.unit, "n": self.n,
            "mean": self.mean, "mean_ci_low": self.mean_ci_low,
            "mean_ci_high": self.mean_ci_high, "std": self.std,
            "yield_pct": self.yield_pct, "yield_ci_low": self.yield_ci_low,
            "yield_ci_high": self.yield_ci_high,
            "final_mean": self.final_mean, "final_std": self.final_std,
            "final_yield_pct": self.final_yield_pct,
            "mean_settled_at": self.mean_settled_at,
            "std_settled_at": self.std_settled_at,
            "yield_settled_at": self.yield_settled_at,
            "settle_tolerance_pct": self.settle_tolerance_pct,
            "notes": self.notes,
        }


def _settle_point(values: Sequence[float], ns: Sequence[int], final: float,
                  tol_frac: float) -> int | None:
    """First prefix length after which the estimate stays within *tol_frac*."""
    if not values or final != final:
        return None
    scale = abs(final) if abs(final) > 0 else 1.0
    for i in range(len(values)):
        if all(abs(v - final) <= tol_frac * scale
               for v in values[i:] if v == v):
            return ns[i]
    return None


def convergence_analysis(run: MonteCarloRun, measurement: str | None = None, *,
                         points: int = 40, specs: Sequence[SpecLimit] | None = None,
                         settle_tolerance_pct: float = 1.0) -> ConvergenceTrace:
    """Running mean / sigma / yield as a function of sample count."""
    usable = run.successful_samples()
    trace = ConvergenceTrace(settle_tolerance_pct=settle_tolerance_pct)
    if len(usable) < 3:
        trace.notes.append(
            f"only {len(usable)} usable sample(s); a convergence trace needs at "
            "least 3")
        return trace

    name = measurement or (run.measurement_names[0] if run.measurement_names else None)
    if name is None:
        trace.notes.append("this run has no measurements")
        return trace
    if name not in run.measurement_names:
        raise AnalysisError(
            f"unknown measurement {name!r}; available: "
            f"{', '.join(run.measurement_names)}")
    trace.measurement = name
    for meta in run.measurement_meta:
        if meta["name"] == name:
            trace.unit = meta.get("unit", "")
            break

    values = np.array([s.measurements.get(name, np.nan) for s in usable], dtype=float)
    spec_list = list(specs) if specs is not None else _specs_from_run(run)
    if spec_list:
        passes = np.array([bool(s.passed) for s in usable], dtype=bool)
    else:
        passes = None
        trace.notes.append("no specifications declared, so no yield trace")

    total = len(usable)
    step = max(1, total // max(points, 1))
    checkpoints = sorted(set(list(range(step, total + 1, step)) + [total]))
    checkpoints = [n for n in checkpoints if n >= 2]

    for n in checkpoints:
        window = values[:n]
        finite = window[np.isfinite(window)]
        trace.n.append(int(n))
        if finite.size < 2:
            trace.mean.append(float("nan"))
            trace.std.append(float("nan"))
            trace.mean_ci_low.append(float("nan"))
            trace.mean_ci_high.append(float("nan"))
        else:
            m = float(np.mean(finite))
            s = float(np.std(finite, ddof=1))
            sem = s / math.sqrt(finite.size)
            trace.mean.append(m)
            trace.std.append(s)
            trace.mean_ci_low.append(m - _Z95 * sem)
            trace.mean_ci_high.append(m + _Z95 * sem)
        if passes is not None:
            k = int(passes[:n].sum())
            lo, hi = wilson_interval(k, n)
            trace.yield_pct.append(100.0 * k / n)
            trace.yield_ci_low.append(lo)
            trace.yield_ci_high.append(hi)

    trace.final_mean = trace.mean[-1] if trace.mean else float("nan")
    trace.final_std = trace.std[-1] if trace.std else float("nan")
    trace.final_yield_pct = trace.yield_pct[-1] if trace.yield_pct else float("nan")

    tol = settle_tolerance_pct / 100.0
    trace.mean_settled_at = _settle_point(trace.mean, trace.n, trace.final_mean, tol)
    trace.std_settled_at = _settle_point(trace.std, trace.n, trace.final_std, tol)
    if trace.yield_pct:
        trace.yield_settled_at = _settle_point(
            trace.yield_pct, trace.n, trace.final_yield_pct, tol)

    if trace.mean_settled_at is None:
        trace.notes.append(
            f"the running mean of {name!r} has not settled to within "
            f"{settle_tolerance_pct:g}% of its final value; more samples are needed "
            "before quoting it")
    return trace


def required_samples_for_margin(observed_yield_pct: float, margin_pct: float,
                                confidence: float = 0.95) -> int:
    """Samples needed to pin a yield down to +/- *margin_pct* (normal approx.).

    Answers the practical question "how many runs do I need to claim 99 % yield
    to within half a point?".
    """
    if not 0.0 <= observed_yield_pct <= 100.0:
        raise AnalysisError("observed yield must be a percentage in [0, 100]")
    if margin_pct <= 0:
        raise AnalysisError("margin must be > 0")
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = observed_yield_pct / 100.0
    m = margin_pct / 100.0
    p = min(max(p, 1e-6), 1 - 1e-6)
    return int(math.ceil(z * z * p * (1 - p) / (m * m)))
