"""Simulation context passed down to every device stamp.

Bundles everything a device needs that is *not* part of its own definition:
temperature-adjusted model cards, the continuation parameters used by the
convergence-aid strategies (``gmin``, ``source_scale``), the integration state
for transient analysis and any per-run source overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import DiodeModel, MosfetModel


@dataclass
class SimContext:
    temp_c: float = 27.0
    gmin: float = 1e-12
    source_scale: float = 1.0          # continuation factor for source stepping
    limiting: bool = True              # device-level junction limiting (diodes)
    limit_mos: bool = False            # extra MOSFET gate/drain voltage limiting
    mode: str = "dc"                   # 'dc' | 'tran'
    time: float = 0.0
    dt: float = 0.0
    integration: str = "trap"          # 'trap' | 'be'
    mos_models: dict[str, MosfetModel] = field(default_factory=dict)
    diode_models: dict[str, DiodeModel] = field(default_factory=dict)
    source_overrides: dict[str, float] = field(default_factory=dict)
    x_prev_iter: np.ndarray | None = None   # previous Newton iterate
    state: dict[str, Any] = field(default_factory=dict)  # transient history
    n_nodes: int = 0                   # total circuit nodes incl. ground

    def mos_model(self, name: str) -> MosfetModel:
        return self.mos_models[name]

    def diode_model(self, name: str) -> DiodeModel:
        return self.diode_models[name]
