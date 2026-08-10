"""One-call analysis of a completed Monte Carlo run.

Bundles statistics, yield, correlation, sensitivity, convergence and plot-ready
chart data into a single serialisable object, which is what the CLI, the HTML
report and the REST API all consume.  Everything here is derived from the
stored per-sample records -- no value in this object is invented, defaulted or
carried over from a previous run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.circuit import SpecLimit
from ..mc.results import MonteCarloRun
from .convergence import ConvergenceTrace, convergence_analysis
from .correlation import CorrelationMatrix, correlation_analysis, measurement_correlation
from .sensitivity import SensitivityReport, sensitivity_analysis
from .statistics import Statistics, cdf_data, describe, histogram_data, sigma_quantile_data
from .yield_analysis import YieldReport, analyse_yield

__all__ = ["RunAnalysis", "analyse_run"]


@dataclass
class RunAnalysis:
    run_id: str = ""
    circuit_name: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Statistics] = field(default_factory=dict)
    yield_report: YieldReport = field(default_factory=YieldReport)
    correlation: CorrelationMatrix = field(default_factory=CorrelationMatrix)
    measurement_correlation: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, SensitivityReport] = field(default_factory=dict)
    convergence: dict[str, ConvergenceTrace] = field(default_factory=dict)
    charts: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_breakdown: list[dict[str, Any]] = field(default_factory=list)
    nominal: dict[str, float] = field(default_factory=dict)
    spec_meta: list[dict[str, Any]] = field(default_factory=list)
    measurement_meta: list[dict[str, Any]] = field(default_factory=list)
    slot_meta: list[dict[str, Any]] = field(default_factory=list)
    reproduction: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "circuit_name": self.circuit_name,
            "summary": self.summary,
            "statistics": {k: v.to_dict() for k, v in self.statistics.items()},
            "yield": self.yield_report.to_dict(),
            "correlation": self.correlation.to_dict(),
            "measurement_correlation": self.measurement_correlation,
            "sensitivity": {k: v.to_dict() for k, v in self.sensitivity.items()},
            "convergence": {k: v.to_dict() for k, v in self.convergence.items()},
            "charts": self.charts,
            "failure_breakdown": self.failure_breakdown,
            "nominal": self.nominal,
            "spec_meta": self.spec_meta,
            "measurement_meta": self.measurement_meta,
            "slot_meta": self.slot_meta,
            "reproduction": self.reproduction,
            "notes": self.notes,
        }


def analyse_run(run: MonteCarloRun, *,
                specs: Sequence[SpecLimit] | None = None,
                sensitivity_method: str = "regression",
                convergence_points: int = 40,
                charts: bool = True) -> RunAnalysis:
    """Run the full analysis stack over a completed Monte Carlo run."""
    analysis = RunAnalysis(
        run_id=run.run_id,
        circuit_name=run.circuit_name,
        summary=run.summary(),
        nominal=dict(run.nominal),
        spec_meta=list(run.spec_meta),
        measurement_meta=list(run.measurement_meta),
        slot_meta=list(run.slot_meta),
        reproduction=run.reproduction_record(),
        failure_breakdown=run.failure_breakdown(),
    )
    if run.notes:
        analysis.notes.append(run.notes)

    units = {m["name"]: m.get("unit", "") for m in run.measurement_meta}
    usable = run.successful_samples()
    if not usable:
        analysis.notes.append(
            f"No sample produced a usable result out of {run.counters.total} "
            "attempted; statistics cannot be computed. See the failure "
            "breakdown for the causes.")
        analysis.yield_report = analyse_yield(run, specs)
        return analysis

    for name in run.measurement_names:
        values = run.values(name)
        analysis.statistics[name] = describe(values, name=name,
                                             unit=units.get(name, ""))
        if charts:
            finite = values[np.isfinite(values)]
            if finite.size:
                analysis.charts[name] = {
                    "histogram": histogram_data(finite),
                    "cdf": cdf_data(finite),
                    "sigma_plot": sigma_quantile_data(finite),
                    "unit": units.get(name, ""),
                    "nominal": run.nominal.get(name),
                    "specs": [s for s in run.spec_meta if s["measure"] == name],
                }

    analysis.yield_report = analyse_yield(run, specs)
    analysis.correlation = correlation_analysis(run)
    analysis.measurement_correlation = measurement_correlation(run)

    for name in run.measurement_names:
        stats = analysis.statistics.get(name)
        if stats is None or stats.count < 3:
            continue
        try:
            analysis.sensitivity[name] = sensitivity_analysis(
                run, name, method=sensitivity_method)
        except Exception as exc:  # keep the rest of the analysis usable
            analysis.notes.append(f"sensitivity analysis for {name!r} failed: {exc}")

    primary = _primary_measurements(run)
    for name in primary:
        try:
            analysis.convergence[name] = convergence_analysis(
                run, name, points=convergence_points, specs=specs)
        except Exception as exc:
            analysis.notes.append(f"convergence analysis for {name!r} failed: {exc}")
    return analysis


def _primary_measurements(run: MonteCarloRun, limit: int = 6) -> list[str]:
    """Measurements worth a convergence trace: spec'd ones first."""
    spec_measures = [s["measure"] for s in run.spec_meta]
    ordered: list[str] = []
    for name in spec_measures + run.measurement_names:
        if name not in ordered and name in run.measurement_names:
            ordered.append(name)
    return ordered[:limit]
