"""Monte Carlo result containers.

Design rule enforced here: **nothing is discarded**.  Every sample that was
attempted appears in :attr:`MonteCarloRun.samples` with its seed, its drawn
parameter values, its status and -- when it failed -- the reason.  Aggregate
statistics are computed downstream from these records, never in place of them.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np

from .. import __version__

__all__ = ["SampleStatus", "SampleResult", "RunCounters", "MonteCarloRun"]


class SampleStatus:
    OK = "ok"
    CONVERGENCE_FAILURE = "convergence_failure"
    NUMERICAL_ERROR = "numerical_error"
    INVALID_MEASUREMENT = "invalid_measurement"
    VARIATION_ERROR = "variation_error"
    CIRCUIT_ERROR = "circuit_error"
    UNEXPECTED_ERROR = "unexpected_error"

    ALL = (OK, CONVERGENCE_FAILURE, NUMERICAL_ERROR, INVALID_MEASUREMENT,
           VARIATION_ERROR, CIRCUIT_ERROR, UNEXPECTED_ERROR)


@dataclass
class SampleResult:
    """One Monte Carlo sample: what was drawn, what happened, what came out."""

    index: int
    seed: int
    status: str = SampleStatus.OK
    failure_reason: str = ""
    measurements: dict[str, float] = field(default_factory=dict)
    measurement_ok: dict[str, bool] = field(default_factory=dict)
    measurement_reasons: dict[str, str] = field(default_factory=dict)
    slot_values: dict[str, float] = field(default_factory=dict)
    device_values: dict[str, float] = field(default_factory=dict)
    spec_pass: dict[str, bool] = field(default_factory=dict)
    passed: bool | None = None
    dc_iterations: int = 0
    dc_strategy: str = ""
    runtime_s: float = 0.0

    @property
    def simulated(self) -> bool:
        """True if the circuit actually solved (measurements may still be invalid)."""
        return self.status in (SampleStatus.OK, SampleStatus.INVALID_MEASUREMENT)

    @property
    def usable(self) -> bool:
        return self.status == SampleStatus.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "seed": self.seed, "status": self.status,
            "failure_reason": self.failure_reason,
            "measurements": dict(self.measurements),
            "measurement_ok": dict(self.measurement_ok),
            "measurement_reasons": dict(self.measurement_reasons),
            "slot_values": dict(self.slot_values),
            "device_values": dict(self.device_values),
            "spec_pass": dict(self.spec_pass), "passed": self.passed,
            "dc_iterations": self.dc_iterations, "dc_strategy": self.dc_strategy,
            "runtime_s": self.runtime_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleResult":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


@dataclass
class RunCounters:
    total: int = 0
    successful: int = 0
    failed: int = 0
    convergence_failures: int = 0
    numerical_errors: int = 0
    invalid_measurements: int = 0
    other_errors: int = 0

    def add(self, result: SampleResult) -> None:
        self.total += 1
        if result.status == SampleStatus.OK:
            self.successful += 1
            return
        self.failed += 1
        if result.status == SampleStatus.CONVERGENCE_FAILURE:
            self.convergence_failures += 1
        elif result.status == SampleStatus.NUMERICAL_ERROR:
            self.numerical_errors += 1
        elif result.status == SampleStatus.INVALID_MEASUREMENT:
            self.invalid_measurements += 1
        else:
            self.other_errors += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total, "successful": self.successful,
            "failed": self.failed,
            "convergence_failures": self.convergence_failures,
            "numerical_errors": self.numerical_errors,
            "invalid_measurements": self.invalid_measurements,
            "other_errors": self.other_errors,
            "success_rate": (self.successful / self.total) if self.total else 0.0,
        }


@dataclass
class MonteCarloRun:
    """A complete Monte Carlo experiment and its provenance."""

    run_id: str
    circuit_name: str
    config: dict[str, Any]
    samples: list[SampleResult] = field(default_factory=list)
    counters: RunCounters = field(default_factory=RunCounters)
    measurement_meta: list[dict[str, Any]] = field(default_factory=list)
    slot_meta: list[dict[str, Any]] = field(default_factory=list)
    spec_meta: list[dict[str, Any]] = field(default_factory=list)
    nominal: dict[str, float] = field(default_factory=dict)
    nominal_status: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    software_version: str = __version__
    platform_info: str = field(default_factory=lambda: (
        f"{platform.python_implementation()} {platform.python_version()} on "
        f"{platform.system()} {platform.machine()}"))
    circuit_source: str = ""
    circuit_sha256: str = ""
    notes: str = ""

    # -- accessors ---------------------------------------------------------
    @property
    def measurement_names(self) -> list[str]:
        return [m["name"] for m in self.measurement_meta]

    @property
    def slot_names(self) -> list[str]:
        return [s["slot"] for s in self.slot_meta]

    def successful_samples(self) -> list[SampleResult]:
        return [s for s in self.samples if s.status == SampleStatus.OK]

    def failed_samples(self) -> list[SampleResult]:
        return [s for s in self.samples if s.status != SampleStatus.OK]

    def values(self, measurement: str, *, only_successful: bool = True) -> np.ndarray:
        """Array of one measurement across samples (successful samples only)."""
        pool = self.successful_samples() if only_successful else self.samples
        return np.array([s.measurements.get(measurement, float("nan"))
                         for s in pool], dtype=float)

    def slot_matrix(self, *, only_successful: bool = True
                    ) -> tuple[np.ndarray, list[str]]:
        """Matrix of drawn parameter deviations, shape ``(n_samples, n_slots)``."""
        pool = self.successful_samples() if only_successful else self.samples
        names = self.slot_names
        if not names:
            return np.zeros((len(pool), 0)), []
        mat = np.array([[s.slot_values.get(n, float("nan")) for n in names]
                        for s in pool], dtype=float)
        return mat, names

    def failure_breakdown(self) -> list[dict[str, Any]]:
        """Failure reasons ranked by frequency, with percentages."""
        buckets: dict[tuple[str, str], int] = {}
        for s in self.failed_samples():
            key = (s.status, _summarise_reason(s.failure_reason))
            buckets[key] = buckets.get(key, 0) + 1
        total = self.counters.total or 1
        rows = [{"status": status, "reason": reason, "count": count,
                 "percent": 100.0 * count / total}
                for (status, reason), count in buckets.items()]
        rows.sort(key=lambda r: -r["count"])
        return rows

    def to_dataframe(self):
        """Flat per-sample table (pandas) for export and inspection."""
        import pandas as pd

        rows = []
        for s in self.samples:
            row: dict[str, Any] = {
                "sample_id": s.index, "seed": s.seed,
                "simulation_status": s.status,
                "failure_reason": s.failure_reason,
                "pass_fail": ("pass" if s.passed else "fail") if s.passed is not None else "",
                "dc_iterations": s.dc_iterations, "dc_strategy": s.dc_strategy,
                "runtime_s": s.runtime_s,
            }
            for name in self.measurement_names:
                row[f"meas.{name}"] = s.measurements.get(name, float("nan"))
            for name in self.slot_names:
                row[f"var.{name}"] = s.slot_values.get(name, float("nan"))
            for key, value in sorted(s.device_values.items()):
                row[f"dev.{key}"] = value
            for key, value in sorted(s.spec_pass.items()):
                row[f"spec.{key}"] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "circuit": self.circuit_name,
            "counters": self.counters.to_dict(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "samples_per_second": (self.counters.total / self.duration_s)
            if self.duration_s > 0 else 0.0,
            "software_version": self.software_version,
            "platform": self.platform_info,
            "seed": self.config.get("seed"),
            "measurements": self.measurement_names,
            "variation_mode": self.config.get("variation", {}).get("mode"),
            "pvt": self.config.get("pvt"),
            "circuit_sha256": self.circuit_sha256,
        }

    def reproduction_record(self) -> dict[str, Any]:
        """Everything required to reproduce this run exactly."""
        return {
            "run_id": self.run_id,
            "seed": self.config.get("seed"),
            "samples": self.config.get("samples"),
            "sampling": self.config.get("sampling"),
            "variation": self.config.get("variation"),
            "pvt": self.config.get("pvt"),
            "solver": self.config.get("solver"),
            "software_version": self.software_version,
            "platform": self.platform_info,
            "circuit_sha256": self.circuit_sha256,
            "timestamp": self.started_at,
        }

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run_id": self.run_id, "circuit_name": self.circuit_name,
            "config": self.config, "counters": self.counters.to_dict(),
            "measurement_meta": self.measurement_meta,
            "slot_meta": self.slot_meta, "spec_meta": self.spec_meta,
            "nominal": self.nominal, "nominal_status": self.nominal_status,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "software_version": self.software_version,
            "platform_info": self.platform_info,
            "circuit_sha256": self.circuit_sha256,
            "notes": self.notes,
        }
        if include_samples:
            data["samples"] = [s.to_dict() for s in self.samples]
        return data


def _summarise_reason(reason: str, max_len: int = 110) -> str:
    """Collapse a detailed exception message into a groupable bucket label."""
    if not reason:
        return "(none)"
    text = " ".join(reason.split())
    for marker in (" (iterations=", " (matrix condition", " at iteration "):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text[:max_len]
