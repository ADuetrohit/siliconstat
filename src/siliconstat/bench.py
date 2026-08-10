"""Throughput benchmarking.

Reports wall time, throughput and per-sample cost for a real Monte Carlo run
at each requested sample count and worker count.  Numbers come from actual
runs on the machine executing the benchmark; nothing is extrapolated.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Sequence

from .core.circuit import Circuit
from .core.solver import SolverOptions
from .mc import MonteCarloConfig, run_monte_carlo
from .variation.spec import VariationModel

__all__ = ["run_benchmark", "benchmark_environment"]


def benchmark_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def run_benchmark(circuit: Circuit, variation: VariationModel, *,
                  sample_counts: Sequence[int] = (100, 1000),
                  worker_counts: Sequence[int] = (1,),
                  seed: int = 12345,
                  solver: SolverOptions | None = None) -> list[dict[str, Any]]:
    """Run the benchmark grid and return one record per configuration."""
    solver = solver or SolverOptions.from_options(circuit.options)
    results: list[dict[str, Any]] = []
    for samples in sample_counts:
        for workers in worker_counts:
            config = MonteCarloConfig(
                variation=variation, samples=int(samples), seed=seed,
                workers=int(workers), solver=solver,
                label=f"bench-{samples}-{workers}")
            started = time.perf_counter()
            run = run_monte_carlo(circuit, config)
            wall = time.perf_counter() - started
            sample_times = [s.runtime_s for s in run.samples if s.runtime_s > 0]
            results.append({
                "circuit": circuit.name,
                "samples": int(samples),
                "workers": int(workers),
                "wall_s": wall,
                "samples_per_s": samples / wall if wall > 0 else float("inf"),
                "successful": run.counters.successful,
                "failed": run.counters.failed,
                "avg_sample_ms": (sum(sample_times) / len(sample_times) * 1e3)
                if sample_times else float("nan"),
                "fell_back_to_sequential": bool(run.notes),
                "notes": run.notes,
                "environment": benchmark_environment(),
            })
    return results
