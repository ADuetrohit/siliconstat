"""Process / Voltage / Temperature corner analysis.

A PVT condition is an orthogonal axis to Monte Carlo mismatch: corners move
the *centre* of the distribution, mismatch describes the *spread* around it.
Running mismatch samples *at* a corner is therefore the interesting
combination, and is what :func:`pvt_grid` enumerates.

Corner definition
-----------------
Corners are defined as +/- ``k`` global sigma on threshold and current factor,
with ``k = 3`` by default, using the *same* global sigmas the Monte Carlo
process model uses.  That keeps the two views consistent: an ``SS`` corner sits
at the 3-sigma tail of the process distribution rather than at an unrelated
made-up number.

===== ============== ==============
Corner NMOS           PMOS
===== ============== ==============
TT     nominal        nominal
FF     fast (-dVth)   fast (-dVth)
SS     slow (+dVth)   slow (+dVth)
FS     fast           slow
SF     slow           fast
===== ============== ==============

"Fast" means lower threshold *and* higher current factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..core.circuit import Circuit
from ..core.devices import VoltageSource
from ..core.exceptions import VariationError

__all__ = ["CORNERS", "PVTCondition", "corner_model_mods", "pvt_grid",
           "default_pvt_grid", "find_supply_source"]

#: (nmos_direction, pmos_direction) where -1 = fast, +1 = slow, 0 = typical.
CORNERS: dict[str, tuple[int, int]] = {
    "TT": (0, 0),
    "FF": (-1, -1),
    "SS": (+1, +1),
    "FS": (-1, +1),   # fast NMOS, slow PMOS
    "SF": (+1, -1),   # slow NMOS, fast PMOS
}

DEFAULT_TEMPERATURES = (-40.0, 27.0, 125.0)
DEFAULT_SUPPLY_TOLERANCE = 0.10   # +/-10 %


def find_supply_source(circuit: Circuit, name: str | None = None) -> VoltageSource:
    """Locate the supply source to scale for the voltage axis."""
    sources = circuit.voltage_sources()
    if not sources:
        raise VariationError(
            "PVT voltage analysis needs at least one independent voltage source")
    if name:
        dev = circuit.device(name)
        if not isinstance(dev, VoltageSource):
            raise VariationError(f"{name!r} is not a voltage source")
        return dev
    for src in sources:
        if src.name.upper() in ("VDD", "VCC", "VSUP", "VSUPPLY"):
            return src
    return max(sources, key=lambda s: abs(s.dc))


def corner_model_mods(circuit: Circuit, corner: str, *,
                      sigma_vth_global: float = 0.025,
                      sigma_beta_global_pct: float = 3.0,
                      k_sigma: float = 3.0) -> dict[str, dict[str, float]]:
    """Model-card modifiers implementing *corner* for this circuit's models."""
    key = corner.strip().upper()
    if key not in CORNERS:
        raise VariationError(
            f"unknown process corner {corner!r}; supported: "
            f"{', '.join(CORNERS)}")
    n_dir, p_dir = CORNERS[key]
    mods: dict[str, dict[str, float]] = {}
    for name, card in circuit.mos_models.items():
        direction = n_dir if card.mtype == "nmos" else p_dir
        if direction == 0:
            continue
        # slow => higher Vth and lower beta; fast => the opposite.
        mods[name] = {
            "dvth": direction * k_sigma * sigma_vth_global,
            "beta_scale": 1.0 - direction * k_sigma * sigma_beta_global_pct / 100.0,
        }
    return mods


@dataclass(frozen=True)
class PVTCondition:
    """One point on the process / voltage / temperature grid."""

    corner: str = "TT"
    supply: float | None = None      # absolute supply voltage [V]
    temp_c: float = 27.0
    supply_source: str | None = None
    k_sigma: float = 3.0
    sigma_vth_global: float = 0.025
    sigma_beta_global_pct: float = 3.0

    @property
    def label(self) -> str:
        v = f"{self.supply:.3g}V" if self.supply is not None else "nomV"
        return f"{self.corner.upper()}/{v}/{self.temp_c:g}C"

    def model_mods(self, circuit: Circuit) -> dict[str, dict[str, float]]:
        return corner_model_mods(
            circuit, self.corner, sigma_vth_global=self.sigma_vth_global,
            sigma_beta_global_pct=self.sigma_beta_global_pct, k_sigma=self.k_sigma)

    def source_overrides(self, circuit: Circuit) -> dict[str, float]:
        if self.supply is None:
            return {}
        src = find_supply_source(circuit, self.supply_source)
        return {src.name: float(self.supply)}

    def build_context(self, circuit: Circuit):
        """Create the :class:`SimContext` for this PVT point."""
        return circuit.build_context(
            temp_c=self.temp_c,
            model_mods=self.model_mods(circuit),
            source_overrides=self.source_overrides(circuit),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"corner": self.corner.upper(), "supply": self.supply,
                "temp_c": self.temp_c, "label": self.label,
                "k_sigma": self.k_sigma}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PVTCondition":
        return cls(
            corner=str(data.get("corner", "TT")),
            supply=None if data.get("supply") is None else float(data["supply"]),
            temp_c=float(data.get("temp_c", 27.0)),
            supply_source=data.get("supply_source"),
            k_sigma=float(data.get("k_sigma", 3.0)),
            sigma_vth_global=float(data.get("sigma_vth_global", 0.025)),
            sigma_beta_global_pct=float(data.get("sigma_beta_global_pct", 3.0)),
        )


def pvt_grid(corners: Sequence[str], supplies: Sequence[float],
             temperatures: Sequence[float], **kwargs: Any) -> list[PVTCondition]:
    """Full cross product of the three axes."""
    out: list[PVTCondition] = []
    for corner in corners:
        for supply in supplies:
            for temp in temperatures:
                out.append(PVTCondition(corner=corner, supply=float(supply),
                                        temp_c=float(temp), **kwargs))
    return out


def default_pvt_grid(circuit: Circuit, *,
                     tolerance: float = DEFAULT_SUPPLY_TOLERANCE,
                     corners: Sequence[str] = tuple(CORNERS),
                     temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
                     **kwargs: Any) -> list[PVTCondition]:
    """Standard 5 corners x 3 supplies x 3 temperatures grid for *circuit*."""
    nominal = find_supply_source(circuit).dc
    supplies = (nominal * (1.0 - tolerance), nominal, nominal * (1.0 + tolerance))
    return pvt_grid(corners, supplies, temperatures, **kwargs)
