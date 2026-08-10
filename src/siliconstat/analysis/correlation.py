"""Correlation between drawn parameters and circuit outputs.

Both Pearson (linear) and Spearman (monotonic rank) coefficients are computed.
The pair matters: a large Spearman with a small Pearson is the signature of a
strong but *nonlinear* dependence, which is precisely the case where a
correlation-based sensitivity ranking would mislead.

Columns with zero variance -- a parameter that was not actually varied, or a
measurement that came out identical for every sample -- yield ``NaN`` and are
listed explicitly rather than being quietly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.exceptions import AnalysisError
from ..mc.results import MonteCarloRun

__all__ = ["CorrelationMatrix", "correlation_analysis", "pearson_matrix",
           "spearman_matrix", "rank_transform"]


def rank_transform(x: np.ndarray) -> np.ndarray:
    """Average ranks along axis 0 (ties share the mean rank)."""
    out = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(col.size, dtype=float)
        ranks[order] = np.arange(1, col.size + 1, dtype=float)
        # Average ties.
        sorted_col = col[order]
        i = 0
        while i < col.size:
            j2 = i
            while j2 + 1 < col.size and sorted_col[j2 + 1] == sorted_col[i]:
                j2 += 1
            if j2 > i:
                ranks[order[i:j2 + 1]] = np.mean(ranks[order[i:j2 + 1]])
            i = j2 + 1
        out[:, j] = ranks
    return out


def _corrcoef(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross-correlation of the columns of *a* against those of *b*."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    a_s = np.sqrt((a_c ** 2).sum(axis=0))
    b_s = np.sqrt((b_c ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        num = a_c.T @ b_c
        denom = np.outer(a_s, b_s)
        result = np.where(denom > 0, num / denom, np.nan)
    return np.clip(result, -1.0, 1.0)


def pearson_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _corrcoef(a, b)


def spearman_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _corrcoef(rank_transform(a), rank_transform(b))


@dataclass
class CorrelationMatrix:
    """Parameter-vs-measurement correlation, both flavours."""

    parameters: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    pearson: list[list[float]] = field(default_factory=list)
    spearman: list[list[float]] = field(default_factory=list)
    n_samples: int = 0
    degenerate_parameters: list[str] = field(default_factory=list)
    degenerate_measurements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def top_pairs(self, measurement: str, limit: int = 10) -> list[dict[str, Any]]:
        if measurement not in self.measurements:
            raise AnalysisError(
                f"unknown measurement {measurement!r}; available: "
                f"{', '.join(self.measurements)}")
        col = self.measurements.index(measurement)
        rows = []
        for i, param in enumerate(self.parameters):
            p = self.pearson[i][col]
            s = self.spearman[i][col]
            rows.append({"parameter": param, "pearson": p, "spearman": s,
                         "abs_pearson": abs(p) if p == p else 0.0})
        rows.sort(key=lambda r: -r["abs_pearson"])
        return rows[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters, "measurements": self.measurements,
            "pearson": self.pearson, "spearman": self.spearman,
            "n_samples": self.n_samples,
            "degenerate_parameters": self.degenerate_parameters,
            "degenerate_measurements": self.degenerate_measurements,
            "notes": self.notes,
        }


def correlation_analysis(run: MonteCarloRun,
                         measurements: Sequence[str] | None = None,
                         parameters: Sequence[str] | None = None,
                         ) -> CorrelationMatrix:
    """Correlate every varied parameter against every measurement."""
    usable = run.successful_samples()
    result = CorrelationMatrix(n_samples=len(usable))
    if len(usable) < 3:
        result.notes.append(
            f"only {len(usable)} usable sample(s); at least 3 are required for a "
            "meaningful correlation coefficient")
        return result

    param_names = list(parameters) if parameters is not None else run.slot_names
    meas_names = list(measurements) if measurements is not None else run.measurement_names
    if not param_names or not meas_names:
        result.notes.append("no varied parameters or no measurements to correlate")
        return result

    x = np.array([[s.slot_values.get(p, np.nan) for p in param_names]
                  for s in usable], dtype=float)
    y = np.array([[s.measurements.get(m, np.nan) for m in meas_names]
                  for s in usable], dtype=float)

    keep_rows = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    dropped = int((~keep_rows).sum())
    if dropped:
        result.notes.append(
            f"{dropped} sample(s) contained a non-finite value and were excluded "
            "from the correlation calculation")
    x, y = x[keep_rows], y[keep_rows]
    if x.shape[0] < 3:
        result.notes.append("too few complete samples remain after filtering")
        return result

    x_var = x.std(axis=0)
    y_var = y.std(axis=0)
    result.degenerate_parameters = [param_names[i] for i in np.where(x_var == 0)[0]]
    result.degenerate_measurements = [meas_names[j] for j in np.where(y_var == 0)[0]]
    if result.degenerate_parameters:
        result.notes.append(
            "parameter(s) with zero spread across samples (correlation "
            f"undefined): {', '.join(result.degenerate_parameters)}")
    if result.degenerate_measurements:
        result.notes.append(
            "measurement(s) identical in every sample (correlation undefined): "
            f"{', '.join(result.degenerate_measurements)}")

    result.parameters = param_names
    result.measurements = meas_names
    result.n_samples = int(x.shape[0])
    result.pearson = pearson_matrix(x, y).tolist()
    result.spearman = spearman_matrix(x, y).tolist()
    return result


def measurement_correlation(run: MonteCarloRun,
                            measurements: Sequence[str] | None = None
                            ) -> dict[str, Any]:
    """Correlation *between* measurements -- shows which specs fail together."""
    usable = run.successful_samples()
    names = list(measurements) if measurements is not None else run.measurement_names
    if len(usable) < 3 or len(names) < 2:
        return {"measurements": names, "pearson": [], "n_samples": len(usable)}
    y = np.array([[s.measurements.get(m, np.nan) for m in names]
                  for s in usable], dtype=float)
    keep = np.all(np.isfinite(y), axis=1)
    y = y[keep]
    if y.shape[0] < 3:
        return {"measurements": names, "pearson": [], "n_samples": int(y.shape[0])}
    return {"measurements": names, "pearson": pearson_matrix(y, y).tolist(),
            "spearman": spearman_matrix(y, y).tolist(),
            "n_samples": int(y.shape[0])}
