"""Internal circuit representation.

A :class:`Circuit` owns the node table, the device list, the model cards and
the analysis/measurement/specification declarations parsed from a netlist.
It is *immutable in practice*: the Monte Carlo layer produces perturbed copies
via :meth:`Circuit.with_overrides` so that samples never share mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from .context import SimContext
from .devices import Device, Mosfet, VoltageSource
from .exceptions import CircuitError
from .models import DiodeModel, MosfetModel

GROUND_ALIASES = frozenset({"0", "gnd", "gnd!", "vss", "ground"})

__all__ = ["Circuit", "MeasureSpec", "SpecLimit", "AnalysisSpec", "GROUND_ALIASES"]


@dataclass(frozen=True)
class MeasureSpec:
    """Declarative description of one circuit measurement.

    ``kind`` selects the evaluator in :mod:`siliconstat.measure`; ``args``
    carries kind-specific parameters already resolved to node/device indices
    where possible.
    """

    name: str
    kind: str
    args: dict[str, Any] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "args": dict(self.args),
                "unit": self.unit, "description": self.description}


@dataclass(frozen=True)
class SpecLimit:
    """A pass/fail specification applied to one measurement."""

    measure: str
    op: str            # '<=', '>=', '<', '>'
    value: float
    unit: str = ""
    label: str = ""

    VALID_OPS = ("<=", ">=", "<", ">")

    def __post_init__(self) -> None:
        if self.op not in self.VALID_OPS:
            raise CircuitError(
                f"specification on {self.measure!r}: operator {self.op!r} is not one of "
                f"{self.VALID_OPS}")

    def passes(self, value: float) -> bool:
        if value != value:  # NaN never passes
            return False
        if self.op == "<=":
            return value <= self.value
        if self.op == ">=":
            return value >= self.value
        if self.op == "<":
            return value < self.value
        return value > self.value

    def describe(self) -> str:
        from .units import format_eng
        if self.unit in ("", "%", "-", "dB", "deg"):
            text = f"{self.value:g}"
            return f"{self.measure} {self.op} {text}{(' ' + self.unit) if self.unit else ''}"
        return f"{self.measure} {self.op} {format_eng(self.value, self.unit)}"

    def to_dict(self) -> dict[str, Any]:
        return {"measure": self.measure, "op": self.op, "value": self.value,
                "unit": self.unit, "label": self.label,
                "description": self.describe()}


@dataclass(frozen=True)
class AnalysisSpec:
    """A requested analysis (``.op``, ``.ac``, ``.tran``)."""

    kind: str                      # 'op' | 'ac' | 'tran'
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "args": dict(self.args)}


class Circuit:
    """Elaborated netlist ready for simulation."""

    def __init__(self, name: str = "circuit") -> None:
        self.name = name
        self.node_names: list[str] = ["0"]
        self.node_index: dict[str, int] = {"0": 0}
        self.devices: list[Device] = []
        self.mos_models: dict[str, MosfetModel] = {}
        self.diode_models: dict[str, DiodeModel] = {}
        self.measures: list[MeasureSpec] = []
        self.specs: list[SpecLimit] = []
        self.analyses: list[AnalysisSpec] = []
        self.options: dict[str, Any] = {}
        self.temp_c: float = 27.0
        self.source_text: str | None = None
        self.source_path: str | None = None
        self._finalized = False

    # -- construction ------------------------------------------------------
    def node(self, name: str) -> int:
        """Return (creating if needed) the index of node *name*."""
        key = name.strip().lower()
        if key in GROUND_ALIASES:
            return 0
        idx = self.node_index.get(key)
        if idx is None:
            idx = len(self.node_names)
            self.node_index[key] = idx
            self.node_names.append(key)
        return idx

    def node_name(self, index: int) -> str:
        return self.node_names[index]

    def has_node(self, name: str) -> bool:
        key = name.strip().lower()
        return key in GROUND_ALIASES or key in self.node_index

    def add_device(self, device: Device) -> None:
        if any(d.name.lower() == device.name.lower() for d in self.devices):
            raise CircuitError(f"duplicate device name {device.name!r}")
        self.devices.append(device)
        self._finalized = False

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    @property
    def n_branches(self) -> int:
        return sum(d.n_branches for d in self.devices)

    @property
    def size(self) -> int:
        return (self.n_nodes - 1) + self.n_branches

    def device(self, name: str) -> Device:
        key = name.lower()
        for d in self.devices:
            if d.name.lower() == key:
                return d
        raise CircuitError(
            f"unknown device {name!r}; known devices: "
            f"{', '.join(sorted(d.name for d in self.devices))}")

    def mosfets(self) -> list[Mosfet]:
        return [d for d in self.devices if isinstance(d, Mosfet)]

    def voltage_sources(self) -> list[VoltageSource]:
        return [d for d in self.devices if isinstance(d, VoltageSource)]

    # -- validation / finalisation ----------------------------------------
    def finalize(self) -> "Circuit":
        """Assign branch unknowns and validate structural integrity."""
        branch = 0
        new_devices: list[Device] = []
        for dev in self.devices:
            if dev.n_branches:
                dev = dev.with_branch(branch)
                branch += dev.n_branches
            new_devices.append(dev)
        self.devices = new_devices

        if not self.devices:
            raise CircuitError(f"circuit {self.name!r} contains no devices")
        if self.n_nodes < 2:
            raise CircuitError(
                f"circuit {self.name!r} has no nodes other than ground")

        max_node = self.n_nodes - 1
        for dev in self.devices:
            for n in dev.nodes:
                if n < 0 or n > max_node:
                    raise CircuitError(
                        f"device {dev.name}: internal node index {n} is out of range")
            if isinstance(dev, Mosfet) and dev.model not in self.mos_models:
                raise CircuitError(
                    f"device {dev.name} references undefined MOSFET model "
                    f"{dev.model!r}; defined models: "
                    f"{', '.join(sorted(self.mos_models)) or '(none)'}")
            model_attr = getattr(dev, "model", None)
            if dev.__class__.__name__ == "Diode" and model_attr not in self.diode_models:
                raise CircuitError(
                    f"device {dev.name} references undefined diode model {model_attr!r}")

        known = {m.name for m in self.measures}
        for spec in self.specs:
            if spec.measure not in known:
                raise CircuitError(
                    f"specification refers to unknown measurement {spec.measure!r}; "
                    f"declared measurements: {', '.join(sorted(known)) or '(none)'}")
        if len(known) != len(self.measures):
            raise CircuitError("duplicate measurement names in circuit")

        self._finalized = True
        return self

    @property
    def finalized(self) -> bool:
        return self._finalized

    # -- Monte Carlo support ----------------------------------------------
    def with_overrides(self, overrides: dict[str, dict[str, float]]) -> "Circuit":
        """Return a copy whose devices carry per-instance parameter overrides."""
        if not overrides:
            return self
        clone = self._shallow_clone()
        lookup = {k.lower(): v for k, v in overrides.items()}
        known = {d.name.lower() for d in self.devices}
        unknown = [k for k in overrides if k.lower() not in known]
        if unknown:
            raise CircuitError(
                f"variation targets unknown device(s): {', '.join(sorted(unknown))}; "
                f"this circuit has: {', '.join(sorted(d.name for d in self.devices))}")
        clone.devices = [
            d.with_overrides(lookup.get(d.name.lower(), {})) for d in self.devices
        ]
        clone._finalized = self._finalized
        return clone

    def _shallow_clone(self) -> "Circuit":
        clone = Circuit(self.name)
        clone.node_names = self.node_names
        clone.node_index = self.node_index
        clone.devices = list(self.devices)
        clone.mos_models = self.mos_models
        clone.diode_models = self.diode_models
        clone.measures = self.measures
        clone.specs = self.specs
        clone.analyses = self.analyses
        clone.options = self.options
        clone.temp_c = self.temp_c
        clone.source_text = self.source_text
        clone.source_path = self.source_path
        clone._finalized = self._finalized
        return clone

    # -- simulation context ------------------------------------------------
    def build_context(self, *, temp_c: float | None = None,
                      model_mods: dict[str, dict[str, float]] | None = None,
                      gmin: float | None = None,
                      source_overrides: dict[str, float] | None = None,
                      supply_scale: float = 1.0,
                      mode: str = "dc") -> SimContext:
        """Create a :class:`SimContext` for this circuit.

        ``model_mods`` applies *corner* shifts to model cards before the
        temperature scaling, matching the physical ordering: a corner selects a
        different point in process space, then the die heats up.

        Recognised model modifiers: ``dvth`` (additive, volts) and
        ``beta_scale`` (multiplicative).
        """
        temp = self.temp_c if temp_c is None else temp_c
        mods = model_mods or {}
        mos: dict[str, MosfetModel] = {}
        for name, card in self.mos_models.items():
            mod = mods.get(name) or mods.get(name.lower()) or {}
            if mod:
                card = replace(
                    card,
                    vto=max(card.vto + mod.get("dvth", 0.0), 1e-3),
                    kp=card.kp * mod.get("beta_scale", 1.0),
                )
            mos[name] = card.at_temperature(temp)
        diodes = {n: c.at_temperature(temp) for n, c in self.diode_models.items()}

        overrides = dict(source_overrides or {})
        if supply_scale != 1.0:
            for dev in self.voltage_sources():
                if dev.name not in overrides:
                    overrides[dev.name] = dev.dc * supply_scale
                else:
                    overrides[dev.name] *= supply_scale

        return SimContext(
            temp_c=temp,
            gmin=self.options.get("gmin", 1e-12) if gmin is None else gmin,
            mode=mode,
            mos_models=mos,
            diode_models=diodes,
            source_overrides=overrides,
            n_nodes=self.n_nodes,
        )

    # -- introspection -----------------------------------------------------
    def variable_parameters(self) -> list[dict[str, Any]]:
        """Enumerate every (device, parameter) pair the variation layer can target."""
        out: list[dict[str, Any]] = []
        for dev in self.devices:
            for p in dev.VARIABLE_PARAMS:
                out.append({
                    "device": dev.name,
                    "parameter": p,
                    "device_type": dev.__class__.__name__.lower(),
                    "model": getattr(dev, "model", None),
                    "matched_group": getattr(dev, "matched_group", None),
                })
        return out

    def describe(self) -> dict[str, Any]:
        """Serialisable summary used by the API / frontend schematic view."""
        devices = []
        for dev in self.devices:
            entry: dict[str, Any] = {
                "name": dev.name,
                "type": dev.__class__.__name__.lower(),
                "nodes": [self.node_names[n] for n in dev.nodes],
            }
            for attr in ("r", "c", "l", "dc", "ac_mag", "w", "m",
                         "model", "matched_group", "area"):
                if hasattr(dev, attr):
                    entry[attr] = getattr(dev, attr)
            if hasattr(dev, "l") and isinstance(dev, Mosfet):
                entry["l"] = dev.l
            devices.append(entry)
        return {
            "name": self.name,
            "nodes": list(self.node_names),
            "n_nodes": self.n_nodes,
            "n_branches": self.n_branches,
            "devices": devices,
            "mos_models": {k: v.to_dict() for k, v in self.mos_models.items()},
            "diode_models": {k: v.to_dict() for k, v in self.diode_models.items()},
            "measures": [m.to_dict() for m in self.measures],
            "specs": [s.to_dict() for s in self.specs],
            "analyses": [a.to_dict() for a in self.analyses],
            "options": dict(self.options),
            "temp_c": self.temp_c,
            "matched_groups": sorted({
                m.matched_group for m in self.mosfets() if m.matched_group
            }),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<Circuit {self.name!r}: {len(self.devices)} devices, "
                f"{self.n_nodes} nodes, {self.n_branches} branches>")


def resolve_nodes(circuit: Circuit, names: Sequence[str]) -> tuple[int, ...]:
    return tuple(circuit.node(n) for n in names)


def iter_devices(circuit: Circuit) -> Iterable[Device]:
    return iter(circuit.devices)
