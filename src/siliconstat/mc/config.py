"""Monte Carlo run configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ..core.circuit import SpecLimit
from ..core.exceptions import VariationError
from ..core.solver import SolverOptions
from ..variation.pvt import PVTCondition
from ..variation.spec import VariationModel

__all__ = ["MonteCarloConfig", "SAMPLING_METHODS"]

SAMPLING_METHODS = ("standard", "latin_hypercube")
MAX_SAMPLES = 1_000_000


@dataclass
class MonteCarloConfig:
    """Everything needed to reproduce a Monte Carlo run.

    ``seed`` + ``samples`` + ``variation`` + ``pvt`` + the circuit netlist form
    the complete reproducibility record: re-running with the same values
    reproduces every sample bit-for-bit, sequentially or in parallel.
    """

    variation: VariationModel
    samples: int = 1000
    seed: int = 12345
    workers: int = 1
    sampling: str = "standard"
    pvt: PVTCondition | None = None
    solver: SolverOptions = field(default_factory=SolverOptions)
    specs: list[SpecLimit] | None = None       # None -> use the circuit's own
    progress_every: int = 25
    label: str = ""

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise VariationError(f"sample count must be >= 1 (got {self.samples})")
        if self.samples > MAX_SAMPLES:
            raise VariationError(
                f"sample count {self.samples} exceeds the safety limit of "
                f"{MAX_SAMPLES}")
        if self.sampling not in SAMPLING_METHODS:
            raise VariationError(
                f"unknown sampling method {self.sampling!r}; supported: "
                f"{', '.join(SAMPLING_METHODS)}")
        if self.workers < 1:
            raise VariationError(f"workers must be >= 1 (got {self.workers})")
        max_workers = max((os.cpu_count() or 1), 1)
        if self.workers > max_workers * 4:
            raise VariationError(
                f"workers={self.workers} is far beyond the {max_workers} CPUs "
                "available; this would only add scheduling overhead")
        self.seed = int(self.seed)

    @property
    def effective_workers(self) -> int:
        return max(1, min(self.workers, self.samples))

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "seed": self.seed,
            "workers": self.workers,
            "sampling": self.sampling,
            "variation": self.variation.to_dict(),
            "pvt": self.pvt.to_dict() if self.pvt else None,
            "solver": {
                "reltol": self.solver.reltol, "vntol": self.solver.vntol,
                "abstol": self.solver.abstol, "max_iter": self.solver.max_iter,
                "gmin": self.solver.gmin, "max_dv": self.solver.max_dv,
            },
            "specs": [s.to_dict() for s in self.specs] if self.specs else None,
            "label": self.label,
        }
