"""Layer 6: descriptive statistics.

Numerical stability
-------------------
The mean and variance are accumulated with **Welford's online algorithm**
rather than the textbook one-pass ``(sum(x^2) - sum(x)^2/n)/(n-1)``.  The naive
form loses precision by catastrophic cancellation once the mean is large
relative to the spread, and the loss is roughly ``(mean/sigma)^2`` in relative
terms.  Measured on this code base (``tests/test_statistics.py``):

======================  ==================  ==================
mean / sigma            naive relative err  Welford relative err
======================  ==================  ==================
4e3  (1.2 V, 300 uV)    1.6e-9              1.7e-13
4e5  (1.2 V, 3 uV)      2.3e-5              5.0e-12
1e9  (1 MV, 1 mV)       1.0  (returns 0)    3.0e-8
======================  ==================  ==================

So the honest statement is: at ordinary analog ratios the naive formula is
merely worse, and by a ratio of about 1e4; by ``mean/sigma ~ 1e9`` it returns
exactly zero.  Welford costs nothing extra and is correct throughout, which is
why it is used unconditionally.

Percentiles use linear interpolation between order statistics (NumPy's default
``linear`` method), which is the convention used by every EDA tool the author
is aware of.

"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.exceptions import AnalysisError

__all__ = ["Statistics", "welford", "describe", "describe_many",
           "histogram_data", "cdf_data", "PERCENTILES"]

PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)


def welford(values: Iterable[float]) -> tuple[int, float, float]:
    """Welford's online mean/variance.  Returns ``(n, mean, sample_variance)``."""
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in values:
        if x != x:  # skip NaN
            continue
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    if n < 2:
        return n, (mean if n else float("nan")), float("nan")
    return n, mean, m2 / (n - 1)


@dataclass
class Statistics:
    """Full descriptive summary of one measurement across a Monte Carlo run."""

    name: str
    unit: str = ""
    count: int = 0            # finite values used
    n_input: int = 0          # values offered (including NaN)
    n_invalid: int = 0
    mean: float = float("nan")
    median: float = float("nan")
    variance: float = float("nan")
    std: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    range: float = float("nan")
    cv: float = float("nan")          # coefficient of variation (std/|mean|)
    p1: float = float("nan")
    p5: float = float("nan")
    p25: float = float("nan")
    p75: float = float("nan")
    p95: float = float("nan")
    p99: float = float("nan")
    iqr: float = float("nan")
    sigma3_low: float = float("nan")
    sigma3_high: float = float("nan")
    skewness: float = float("nan")
    kurtosis_excess: float = float("nan")
    sem: float = float("nan")         # standard error of the mean
    mean_ci95_low: float = float("nan")
    mean_ci95_high: float = float("nan")
    std_ci95_low: float = float("nan")
    std_ci95_high: float = float("nan")
    normality_p: float = float("nan")
    normality_test: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        return (f"{self.name}: mean={self.mean:.6g} {self.unit} "
                f"sigma={self.std:.6g} "
                f"3sigma=[{self.sigma3_low:.6g}, {self.sigma3_high:.6g}] "
                f"n={self.count}")


def describe(values: Sequence[float] | np.ndarray, name: str = "value",
             unit: str = "", *, normality: bool = True) -> Statistics:
    """Compute the full statistical summary of *values*.

    Non-finite entries are excluded and counted separately -- they are never
    silently treated as zero.
    """
    raw = np.asarray(list(values), dtype=float) if not isinstance(values, np.ndarray) \
        else values.astype(float, copy=False)
    n_input = int(raw.size)
    finite = raw[np.isfinite(raw)]
    stats = Statistics(name=name, unit=unit, n_input=n_input,
                       n_invalid=n_input - int(finite.size), count=int(finite.size))
    if finite.size == 0:
        return stats

    n, mean, variance = welford(finite.tolist())
    stats.mean = float(mean)
    stats.variance = float(variance) if n > 1 else float("nan")
    stats.std = float(math.sqrt(variance)) if n > 1 and variance >= 0 else float("nan")
    stats.minimum = float(np.min(finite))
    stats.maximum = float(np.max(finite))
    stats.range = stats.maximum - stats.minimum
    stats.median = float(np.median(finite))
    q = np.percentile(finite, PERCENTILES)
    stats.p1, stats.p5, stats.p25, _med, stats.p75, stats.p95, stats.p99 = (
        float(v) for v in q)
    stats.iqr = stats.p75 - stats.p25
    if math.isfinite(stats.std):
        stats.sigma3_low = stats.mean - 3.0 * stats.std
        stats.sigma3_high = stats.mean + 3.0 * stats.std
        stats.cv = (stats.std / abs(stats.mean)) if stats.mean != 0 else float("inf")
        stats.sem = stats.std / math.sqrt(n)
        stats.mean_ci95_low = stats.mean - 1.959964 * stats.sem
        stats.mean_ci95_high = stats.mean + 1.959964 * stats.sem
        # Chi-square confidence interval for the standard deviation.
        if n > 2:
            try:
                from scipy.stats import chi2
                lo = chi2.ppf(0.975, n - 1)
                hi = chi2.ppf(0.025, n - 1)
                stats.std_ci95_low = stats.std * math.sqrt((n - 1) / lo)
                stats.std_ci95_high = stats.std * math.sqrt((n - 1) / hi)
            except Exception:  # pragma: no cover - scipy always present here
                pass

    if n > 2 and stats.std and math.isfinite(stats.std) and stats.std > 0:
        z = (finite - stats.mean) / stats.std
        stats.skewness = float(np.mean(z ** 3))
        stats.kurtosis_excess = float(np.mean(z ** 4) - 3.0)

    if normality and n >= 8:
        stats.normality_p, stats.normality_test = _normality(finite)
    return stats


def _normality(sample: np.ndarray) -> tuple[float, str]:
    """Test the Gaussian hypothesis; report which test was used.

    Shapiro-Wilk is the most powerful for n <= 5000; beyond that
    D'Agostino-Pearson is used because Shapiro-Wilk's p-value is unreliable
    (and it flags trivially small departures as significant on huge samples).
    """
    try:
        from scipy import stats as sps
        if sample.size <= 5000:
            return float(sps.shapiro(sample).pvalue), "shapiro-wilk"
        return float(sps.normaltest(sample).pvalue), "dagostino-pearson"
    except Exception:  # pragma: no cover
        return float("nan"), ""


def describe_many(columns: dict[str, Sequence[float]],
                  units: dict[str, str] | None = None) -> dict[str, Statistics]:
    units = units or {}
    return {name: describe(values, name=name, unit=units.get(name, ""))
            for name, values in columns.items()}


def histogram_data(values: Sequence[float] | np.ndarray, bins: int | str = "auto",
                   ) -> dict[str, Any]:
    """Histogram counts and edges, ready for a client-side chart.

    ``bins='auto'`` uses NumPy's max(Sturges, Freedman-Diaconis) rule, which
    adapts to both sample size and spread.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise AnalysisError("cannot build a histogram: no finite values")
    if data.size == 1 or np.ptp(data) == 0:
        centre = float(data[0])
        width = max(abs(centre) * 1e-6, 1e-12)
        return {"counts": [int(data.size)],
                "edges": [centre - width, centre + width],
                "centres": [centre], "bin_width": 2 * width, "n": int(data.size)}
    counts, edges = np.histogram(data, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {"counts": counts.tolist(), "edges": edges.tolist(),
            "centres": centres.tolist(),
            "bin_width": float(edges[1] - edges[0]), "n": int(data.size)}


def cdf_data(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Empirical CDF points.

    Uses the mid-rank ("Hazen") plotting position ``(i - 0.5)/n``, which is
    unbiased at the tails -- important when reading a 1 ppm yield off a plot.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise AnalysisError("cannot build a CDF: no finite values")
    ordered = np.sort(data)
    n = ordered.size
    probability = (np.arange(1, n + 1) - 0.5) / n
    return {"x": ordered.tolist(), "p": probability.tolist(), "n": int(n)}


def sigma_quantile_data(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Normal-quantile (sigma) plot data: a straight line means Gaussian."""
    from ..variation.distributions import inverse_normal_cdf

    cdf = cdf_data(values)
    z = inverse_normal_cdf(np.asarray(cdf["p"]))
    return {"x": cdf["x"], "sigma": z.tolist(), "n": cdf["n"]}
