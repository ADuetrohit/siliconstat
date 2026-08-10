"""Which parameters actually move the circuit?

Two methods are provided, and the distinction between them is not cosmetic:

**Correlation-based ranking** (``method="correlation"``)
    Orders parameters by ``|Pearson r|``.  Cheap, always available, and
    perfectly adequate for "what should I look at first".  It is *not* a
    variance decomposition: correlations do not add up to anything, and with
    correlated inputs a parameter can rank high purely by association.

**Variance-based decomposition** (``method="regression"``, the default)
    Fits a standardised linear model ``y* = sum(beta_i * x_i*)`` by least
    squares.  When the inputs are mutually uncorrelated -- which is how this
    project's variation model constructs them unless the user declares a
    correlation group -- ``beta_i^2`` *is* the fraction of output variance
    contributed by parameter *i*, and their sum is exactly ``R^2``.
    ``1 - R^2`` is the share of variance the linear model cannot explain, i.e.
    genuine nonlinearity plus simulation noise.

Both report ``r_squared`` so the reader can judge how much to trust the
decomposition, and the input-correlation health check is reported explicitly
rather than assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.exceptions import AnalysisError
from ..mc.results import MonteCarloRun
from .correlation import pearson_matrix, spearman_matrix

__all__ = ["SensitivityEntry", "SensitivityReport", "sensitivity_analysis"]


@dataclass
class SensitivityEntry:
    parameter: str
    rank: int = 0
    pearson: float = float("nan")
    spearman: float = float("nan")
    beta_standardised: float = float("nan")
    variance_contribution: float = float("nan")   # beta^2, fraction of Var(y)
    variance_contribution_pct: float = float("nan")
    share_of_explained_pct: float = float("nan")
    sigma: float = float("nan")                   # parameter sigma as drawn
    unit: str = ""
    scope: str = ""
    d_output_d_sigma: float = float("nan")        # output change per 1-sigma input

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SensitivityReport:
    measurement: str
    method: str = "regression"
    n_samples: int = 0
    r_squared: float = float("nan")
    unexplained_pct: float = float("nan")
    output_sigma: float = float("nan")
    output_mean: float = float("nan")
    entries: list[SensitivityEntry] = field(default_factory=list)
    max_input_correlation: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entries"] = [e.to_dict() for e in self.entries]
        return data

    def ranking(self, limit: int | None = None) -> list[str]:
        names = [e.parameter for e in self.entries]
        return names[:limit] if limit else names


def sensitivity_analysis(run: MonteCarloRun, measurement: str, *,
                         method: str = "regression",
                         parameters: Sequence[str] | None = None,
                         ) -> SensitivityReport:
    """Rank the variation slots by their influence on *measurement*."""
    if method not in ("regression", "correlation"):
        raise AnalysisError(
            f"unknown sensitivity method {method!r}; use 'regression' or "
            "'correlation'")
    if measurement not in run.measurement_names:
        raise AnalysisError(
            f"unknown measurement {measurement!r}; available: "
            f"{', '.join(run.measurement_names)}")

    report = SensitivityReport(measurement=measurement, method=method)
    usable = run.successful_samples()
    param_names = list(parameters) if parameters is not None else run.slot_names
    if not param_names:
        report.notes.append("no varied parameters in this run")
        return report
    if len(usable) < max(5, len(param_names) + 2):
        report.notes.append(
            f"{len(usable)} usable sample(s) for {len(param_names)} parameters: "
            "too few to attribute variance reliably (need at least "
            f"{max(5, len(param_names) + 2)})")
        if len(usable) < 3:
            return report

    x = np.array([[s.slot_values.get(p, np.nan) for p in param_names]
                  for s in usable], dtype=float)
    y = np.array([s.measurements.get(measurement, np.nan) for s in usable],
                 dtype=float)
    keep = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    dropped = int((~keep).sum())
    if dropped:
        report.notes.append(
            f"{dropped} sample(s) excluded for non-finite values")
    x, y = x[keep], y[keep]
    report.n_samples = int(x.shape[0])
    if report.n_samples < 3:
        report.notes.append("too few complete samples for sensitivity analysis")
        return report

    slot_meta = {s["slot"]: s for s in run.slot_meta}
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=1)
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=1))
    report.output_mean = y_mean
    report.output_sigma = y_std

    active = x_std > 0
    if not np.any(active):
        report.notes.append("every parameter had zero spread; nothing to rank")
        return report
    if y_std <= 0:
        report.notes.append(
            f"measurement {measurement!r} is identical in every sample, so no "
            "parameter influences it")
        return report

    pear = pearson_matrix(x, y.reshape(-1, 1)).ravel()
    spear = spearman_matrix(x, y.reshape(-1, 1)).ravel()

    n_active = int(np.count_nonzero(active))
    if method == "regression" and report.n_samples <= n_active + 1:
        # An underdetermined least-squares fit returns the minimum-norm solution
        # and a spurious R^2 of exactly 1.  Reporting that as a variance
        # decomposition would be a fabricated result, so fall back to the
        # ranking method that is still meaningful at this sample count.
        report.notes.append(
            f"only {report.n_samples} usable samples for {n_active} varying "
            "parameters: a variance decomposition would be underdetermined and "
            "would report a meaningless R^2 of 1.0. Falling back to "
            "correlation-based ranking; increase the sample count for a real "
            "variance attribution.")
        method = "correlation"
        report.method = "correlation"

    betas = np.full(len(param_names), np.nan)
    r_squared = float("nan")
    if method == "regression":
        z = np.zeros_like(x)
        z[:, active] = (x[:, active] - x_mean[active]) / x_std[active]
        w = (y - y_mean) / y_std
        design = z[:, active]
        # Report how orthogonal the inputs really are: beta^2 is only a variance
        # share when they are (near-)uncorrelated.
        if design.shape[1] > 1:
            cc = pearson_matrix(design, design)
            off = cc[~np.eye(cc.shape[0], dtype=bool)]
            off = off[np.isfinite(off)]
            report.max_input_correlation = float(np.max(np.abs(off))) if off.size else 0.0
        try:
            coef, *_ = np.linalg.lstsq(design, w, rcond=None)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            raise AnalysisError(f"sensitivity regression failed: {exc}") from exc
        betas[active] = coef
        residual = w - design @ coef
        ss_res = float(residual @ residual)
        ss_tot = float(w @ w)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        report.r_squared = r_squared
        report.unexplained_pct = 100.0 * (1.0 - r_squared) if r_squared == r_squared \
            else float("nan")
        if report.max_input_correlation > 0.3:
            report.notes.append(
                f"the drawn parameters are not mutually independent (largest "
                f"|r| between inputs = {report.max_input_correlation:.2f}); the "
                "per-parameter variance shares below are standardised "
                "regression coefficients and do not partition the variance "
                "cleanly in that case")
        if r_squared == r_squared and r_squared < 0.8:
            report.notes.append(
                f"the linear model explains only {100 * r_squared:.1f}% of the "
                f"variance of {measurement!r}; the remainder is genuine "
                "nonlinearity. Treat the ranking as indicative, not exact.")

    entries: list[SensitivityEntry] = []
    total_beta2 = float(np.nansum(betas ** 2)) if method == "regression" else 0.0
    for i, name in enumerate(param_names):
        meta = slot_meta.get(name, {})
        beta = float(betas[i]) if betas[i] == betas[i] else float("nan")
        contribution = beta * beta if beta == beta else float("nan")
        entry = SensitivityEntry(
            parameter=name,
            pearson=float(pear[i]), spearman=float(spear[i]),
            beta_standardised=beta,
            variance_contribution=contribution,
            variance_contribution_pct=(100.0 * contribution
                                       if contribution == contribution else float("nan")),
            share_of_explained_pct=(100.0 * contribution / total_beta2
                                    if total_beta2 > 0 and contribution == contribution
                                    else float("nan")),
            sigma=float(meta.get("sigma", float("nan"))),
            unit=str(meta.get("unit", "")),
            scope=str(meta.get("scope", "")),
            d_output_d_sigma=(beta * y_std if beta == beta else float("nan")),
        )
        entries.append(entry)

    key = (lambda e: -(abs(e.beta_standardised) if e.beta_standardised == e.beta_standardised else -1)) \
        if method == "regression" else \
        (lambda e: -(abs(e.pearson) if e.pearson == e.pearson else -1))
    entries.sort(key=key)
    for rank, entry in enumerate(entries, start=1):
        entry.rank = rank
    report.entries = entries
    return report
