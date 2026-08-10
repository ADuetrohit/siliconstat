"""Layers 6-7: statistics, yield, correlation, sensitivity, convergence."""

from __future__ import annotations

from .convergence import (
    ConvergenceTrace,
    convergence_analysis,
    required_samples_for_margin,
)
from .correlation import (
    CorrelationMatrix,
    correlation_analysis,
    measurement_correlation,
    pearson_matrix,
    spearman_matrix,
)
from .sensitivity import SensitivityEntry, SensitivityReport, sensitivity_analysis
from .statistics import (
    Statistics,
    cdf_data,
    describe,
    describe_many,
    histogram_data,
    sigma_quantile_data,
    welford,
)
from .summary import RunAnalysis, analyse_run
from .yield_analysis import (
    SpecYield,
    YieldReport,
    analyse_yield,
    process_capability,
    wilson_interval,
)

__all__ = [
    "Statistics", "describe", "describe_many", "welford", "histogram_data",
    "cdf_data", "sigma_quantile_data",
    "SpecYield", "YieldReport", "analyse_yield", "wilson_interval",
    "process_capability",
    "CorrelationMatrix", "correlation_analysis", "measurement_correlation",
    "pearson_matrix", "spearman_matrix",
    "SensitivityReport", "SensitivityEntry", "sensitivity_analysis",
    "ConvergenceTrace", "convergence_analysis", "required_samples_for_margin",
    "RunAnalysis", "analyse_run",
]
