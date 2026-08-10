"""Layer 7: yield.

Yield is a ratio, and a ratio is meaningless without its denominator.  This
module therefore *always* reports both:

``yield_over_successful``
    passing / simulations that converged and produced valid measurements.
    This is the number an analog designer wants: "of the dice I could
    actually measure, what fraction met spec?"

``yield_over_attempted``
    passing / samples attempted, counting every convergence failure as a
    non-pass.  This is the conservative bound, and the gap between the two
    numbers is exactly the amount of information the run lost to numerical
    failure.

Every reported percentage carries a Wilson score confidence interval, because
"98.2 %" from 1000 samples and "98.2 %" from 100 samples are very different
claims.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.circuit import SpecLimit
from ..core.exceptions import AnalysisError
from ..mc.results import MonteCarloRun, SampleStatus

__all__ = ["SpecYield", "YieldReport", "wilson_interval", "analyse_yield",
           "process_capability"]

_Z95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = _Z95
                    ) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as percentages.

    Preferred over the normal approximation because it stays inside [0, 1] and
    remains sensible at the extremes -- a 100 %-passing run of 50 samples
    correctly reports a lower bound near 93 %, not 100 %.
    """
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / trials
                                     + z * z / (4 * trials * trials))
    return max(0.0, (centre - margin)) * 100.0, min(1.0, (centre + margin)) * 100.0


def process_capability(values: np.ndarray, lower: float | None,
                       upper: float | None) -> dict[str, float]:
    """Cp / Cpk process-capability indices.

    ``Cpk >= 1.33`` is the usual "capable" threshold (about 63 ppm defects for
    a centred Gaussian); ``Cpk >= 1.0`` corresponds to 3-sigma limits.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return {"cp": float("nan"), "cpk": float("nan"),
                "cpu": float("nan"), "cpl": float("nan")}
    mean = float(np.mean(finite))
    sigma = float(np.std(finite, ddof=1))
    if sigma <= 0:
        return {"cp": float("inf"), "cpk": float("inf"),
                "cpu": float("inf"), "cpl": float("inf")}
    cpu = (upper - mean) / (3.0 * sigma) if upper is not None else float("inf")
    cpl = (mean - lower) / (3.0 * sigma) if lower is not None else float("inf")
    cp = ((upper - lower) / (6.0 * sigma)
          if (upper is not None and lower is not None) else float("nan"))
    return {"cp": cp, "cpk": min(cpu, cpl), "cpu": cpu, "cpl": cpl}


@dataclass
class SpecYield:
    """Yield of one specification."""

    key: str
    measure: str
    op: str
    limit: float
    unit: str = ""
    description: str = ""
    passing: int = 0
    failing: int = 0
    denominator: int = 0
    yield_pct: float = float("nan")
    ci95_low: float = float("nan")
    ci95_high: float = float("nan")
    margin_mean: float = float("nan")     # mean distance to the limit
    margin_sigma: float = float("nan")    # distance to limit, in sigmas
    cpk: float = float("nan")
    worst_value: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class YieldReport:
    """Complete yield picture for a Monte Carlo run."""

    attempted: int = 0
    successful: int = 0
    failed: int = 0
    denominator_basis: str = "successful simulations"
    per_spec: list[SpecYield] = field(default_factory=list)
    combined_passing: int = 0
    combined_yield_over_successful: float = float("nan")
    combined_ci95_low: float = float("nan")
    combined_ci95_high: float = float("nan")
    combined_yield_over_attempted: float = float("nan")
    independent_product_pct: float = float("nan")
    limiting_spec: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["per_spec"] = [s.to_dict() for s in self.per_spec]
        return data

    def summary_lines(self) -> list[str]:
        lines = [f"{s.description or s.key}: {s.yield_pct:.2f}% "
                 f"[{s.ci95_low:.2f}, {s.ci95_high:.2f}] "
                 f"({s.passing}/{s.denominator})" for s in self.per_spec]
        lines.append(
            f"COMBINED: {self.combined_yield_over_successful:.2f}% "
            f"[{self.combined_ci95_low:.2f}, {self.combined_ci95_high:.2f}] "
            f"({self.combined_passing}/{self.successful} successful simulations)")
        return lines


def analyse_yield(run: MonteCarloRun, specs: Sequence[SpecLimit] | None = None
                  ) -> YieldReport:
    """Compute per-specification and combined yield from a completed run."""
    spec_list = list(specs) if specs is not None else _specs_from_run(run)
    report = YieldReport(
        attempted=run.counters.total,
        successful=run.counters.successful,
        failed=run.counters.failed,
    )
    usable = run.successful_samples()
    if not spec_list:
        report.notes.append(
            "No specifications are declared for this circuit, so no yield can "
            "be computed. Add '.spec <measurement> <op> <value>' lines to the "
            "netlist or supply specs through the API.")
        return report
    if not usable:
        report.notes.append(
            f"No sample produced a usable result ({run.counters.total} attempted, "
            f"{run.counters.failed} failed); yield is undefined.")
        return report

    combined_mask = np.ones(len(usable), dtype=bool)
    for spec in spec_list:
        values = np.array([s.measurements.get(spec.measure, float("nan"))
                           for s in usable], dtype=float)
        valid = np.array([s.measurement_ok.get(spec.measure, True) for s in usable])
        finite = np.isfinite(values) & valid
        passes = np.zeros(values.size, dtype=bool)
        passes[finite] = [spec.passes(float(v)) for v in values[finite]]
        combined_mask &= passes

        denominator = int(values.size)
        passing = int(passes.sum())
        lo, hi = wilson_interval(passing, denominator)
        good = values[finite]
        lower = spec.value if spec.op in (">=", ">") else None
        upper = spec.value if spec.op in ("<=", "<") else None
        cap = process_capability(good, lower, upper)

        margin_mean = float("nan")
        margin_sigma = float("nan")
        worst = float("nan")
        if good.size:
            if upper is not None:
                margin_mean = float(spec.value - np.mean(good))
                worst = float(np.max(good))
            else:
                margin_mean = float(np.mean(good) - spec.value)
                worst = float(np.min(good))
            sigma = float(np.std(good, ddof=1)) if good.size > 1 else float("nan")
            if sigma and math.isfinite(sigma) and sigma > 0:
                margin_sigma = margin_mean / sigma

        report.per_spec.append(SpecYield(
            key=spec.describe(), measure=spec.measure, op=spec.op,
            limit=spec.value, unit=spec.unit,
            description=spec.label or spec.describe(),
            passing=passing, failing=denominator - passing,
            denominator=denominator,
            yield_pct=100.0 * passing / denominator if denominator else float("nan"),
            ci95_low=lo, ci95_high=hi, margin_mean=margin_mean,
            margin_sigma=margin_sigma, cpk=cap["cpk"], worst_value=worst,
        ))
        if int(finite.sum()) < denominator:
            report.notes.append(
                f"{int(denominator - finite.sum())} sample(s) had an invalid "
                f"'{spec.measure}' measurement and were counted as failing this "
                "specification.")

    report.combined_passing = int(combined_mask.sum())
    n = len(usable)
    report.combined_yield_over_successful = 100.0 * report.combined_passing / n
    report.combined_ci95_low, report.combined_ci95_high = wilson_interval(
        report.combined_passing, n)
    report.combined_yield_over_attempted = (
        100.0 * report.combined_passing / report.attempted
        if report.attempted else float("nan"))

    product = 1.0
    for s in report.per_spec:
        product *= s.yield_pct / 100.0
    report.independent_product_pct = product * 100.0
    if report.per_spec:
        report.limiting_spec = min(report.per_spec,
                                   key=lambda s: s.yield_pct).description

    if abs(report.independent_product_pct
           - report.combined_yield_over_successful) > 1.0:
        report.notes.append(
            "The combined yield differs from the product of the individual "
            f"yields ({report.independent_product_pct:.2f}%), which means the "
            "specifications fail together rather than independently -- they "
            "share dominant variation sources. The measured combined yield is "
            "the correct number; the product is shown only for contrast.")
    if report.failed:
        report.notes.append(
            f"{report.failed} of {report.attempted} samples did not produce a "
            f"usable result and are excluded from the denominator above. "
            f"Counting them as failures gives "
            f"{report.combined_yield_over_attempted:.2f}%.")
    return report


def _specs_from_run(run: MonteCarloRun) -> list[SpecLimit]:
    out = []
    for meta in run.spec_meta:
        out.append(SpecLimit(measure=meta["measure"], op=meta["op"],
                             value=float(meta["value"]),
                             unit=meta.get("unit", ""), label=meta.get("label", "")))
    return out
