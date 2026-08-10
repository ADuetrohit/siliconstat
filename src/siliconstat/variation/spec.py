"""Declarative description of a variation model.

A :class:`VariationModel` is a list of :class:`ParameterVariation` rules.  Each
rule says *which* devices vary, in *which* parameter, by *how much*, drawn from
*which* distribution, and whether the draw is shared (global / process) or
independent per device (local / mismatch).

The distinction between the two scopes is the whole point of the exercise:

* **global** -- one draw per model card (or per matched group), applied
  identically to every device it covers.  Models die-to-die process spread.
* **local**  -- one independent draw per device instance.  Models within-die
  random mismatch, and is the only thing a current mirror or differential pair
  actually cares about.

So for two transistors in a mirror::

    Vth(M1) = Vth_nom + dVth_global(NCH) + dVth_local(M1)
    Vth(M2) = Vth_nom + dVth_global(NCH) + dVth_local(M2)

They share the global term exactly and differ only in the local term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..core.circuit import Circuit
from ..core.devices import Device, Mosfet
from ..core.exceptions import VariationError
from .correlation import CorrelationSpec

__all__ = [
    "ParameterVariation", "VariationModel", "select_devices",
    "default_mismatch_model", "PARAMETER_MODES", "PARAMETER_UNITS",
]

#: Whether a parameter's deviation is added to the nominal or scales it.
PARAMETER_MODES: dict[str, str] = {
    "vth": "additive",
    "beta": "relative",
    "w": "relative",
    "l": "relative",
    "r": "relative",
    "c": "relative",
    "dc": "relative",
    "is": "relative",
    "n": "relative",
}

PARAMETER_UNITS: dict[str, str] = {
    "vth": "V", "beta": "-", "w": "m", "l": "m",
    "r": "ohm", "c": "F", "dc": "V/A", "is": "A", "n": "-",
}

#: Which device classes expose which variable parameters.  ``l`` means channel
#: length and is MOSFET-only; inductance is deliberately not variable, because
#: sharing the name would make ``parameter="l"`` ambiguous.
_PARAM_DEVICE_TYPES: dict[str, tuple[str, ...]] = {
    "vth": ("mosfet",),
    "beta": ("mosfet",),
    "w": ("mosfet",),
    "l": ("mosfet",),
    "r": ("resistor",),
    "c": ("capacitor",),
    "dc": ("voltagesource", "currentsource"),
    "is": ("diode",),
    "n": ("diode",),
}


@dataclass
class ParameterVariation:
    """One variation rule."""

    parameter: str
    scope: str = "local"                 # 'global' | 'local'
    distribution: str = "gaussian"       # 'gaussian' | 'uniform' | 'lognormal'
    sigma: float | None = None           # absolute sigma, in parameter units
    sigma_pct: float | None = None       # relative sigma, percent of nominal
    pelgrom: bool = False                # derive sigma from AVT/ABETA and area
    targets: str = "*"                   # device selector, see select_devices()
    share_by: str = "model"              # global scope: 'model'|'matched_group'|'all'
    correlation_group: str | None = None
    truncate_sigma: float | None = None  # clip the standard normal at +/- N sigma
    mode: str = "auto"                   # 'auto' | 'additive' | 'relative'
    enabled: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        self.parameter = self.parameter.strip().lower()
        self.scope = self.scope.strip().lower()
        self.distribution = self.distribution.strip().lower()
        self.share_by = self.share_by.strip().lower()
        if self.parameter not in PARAMETER_MODES:
            raise VariationError(
                f"unknown variation parameter {self.parameter!r}; supported: "
                f"{', '.join(sorted(PARAMETER_MODES))}")
        if self.scope not in ("global", "local"):
            raise VariationError(
                f"variation scope must be 'global' or 'local' (got {self.scope!r})")
        if self.share_by not in ("model", "matched_group", "all"):
            raise VariationError(
                f"share_by must be 'model', 'matched_group' or 'all' "
                f"(got {self.share_by!r})")
        if self.pelgrom and self.scope != "local":
            raise VariationError(
                "Pelgrom area scaling describes *local* mismatch between nearby "
                "devices; it cannot be applied to a global (process) variation")
        if self.pelgrom and self.parameter not in ("vth", "beta"):
            raise VariationError(
                f"Pelgrom scaling is defined for 'vth' and 'beta', not "
                f"{self.parameter!r}")
        if not self.pelgrom and self.sigma is None and self.sigma_pct is None:
            raise VariationError(
                f"variation on {self.parameter!r} needs 'sigma', 'sigma_pct' or "
                "'pelgrom=True'")
        if self.sigma is not None and self.sigma < 0:
            raise VariationError("sigma must be >= 0")
        if self.sigma_pct is not None and self.sigma_pct < 0:
            raise VariationError("sigma_pct must be >= 0")
        if self.effective_mode == "additive" and self.distribution == "lognormal":
            raise VariationError(
                f"a log-normal deviation is strictly greater than -1 and is only "
                f"meaningful for a multiplicative parameter; {self.parameter!r} is "
                "applied additively")

    @property
    def effective_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        return PARAMETER_MODES[self.parameter]

    @property
    def unit(self) -> str:
        return PARAMETER_UNITS.get(self.parameter, "")

    def applies_to(self, device: Device) -> bool:
        allowed = _PARAM_DEVICE_TYPES.get(self.parameter, ())
        return device.__class__.__name__.lower() in allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter, "scope": self.scope,
            "distribution": self.distribution, "sigma": self.sigma,
            "sigma_pct": self.sigma_pct, "pelgrom": self.pelgrom,
            "targets": self.targets, "share_by": self.share_by,
            "correlation_group": self.correlation_group,
            "truncate_sigma": self.truncate_sigma, "mode": self.effective_mode,
            "enabled": self.enabled, "label": self.label, "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterVariation":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known - {"unit"}
        if unknown:
            raise VariationError(
                f"unknown variation field(s): {', '.join(sorted(unknown))}")
        return cls(**{k: v for k, v in data.items() if k in known})


def select_devices(circuit: Circuit, selector: str) -> list[Device]:
    """Resolve a device selector to a concrete device list.

    Accepted forms (comma-separated, combined as a union):

    ``*``               every device
    ``model:NCH``       every device referencing model ``NCH``
    ``group:MIRROR``    every MOSFET in matched group ``MIRROR``
    ``type:resistor``   every device of that class
    ``M1``              a single device by name (``device:M1`` also works)
    """
    out: list[Device] = []
    seen: set[str] = set()
    for raw in str(selector).split(","):
        token = raw.strip()
        if not token:
            continue
        matched: list[Device]
        if token == "*":
            matched = list(circuit.devices)
        elif ":" in token:
            kind, _, value = token.partition(":")
            kind = kind.strip().lower()
            value = value.strip()
            if kind == "model":
                matched = [d for d in circuit.devices
                           if str(getattr(d, "model", "")).lower() == value.lower()]
                if not matched:
                    raise VariationError(
                        f"variation selector {token!r} matched no devices; models in "
                        f"this circuit: "
                        f"{', '.join(sorted(circuit.mos_models) + sorted(circuit.diode_models)) or 'none'}")
            elif kind in ("group", "match", "matched_group"):
                matched = [d for d in circuit.mosfets()
                           if (d.matched_group or "").lower() == value.lower()]
                if not matched:
                    groups = sorted({m.matched_group for m in circuit.mosfets()
                                     if m.matched_group})
                    raise VariationError(
                        f"variation selector {token!r} matched no devices; matched "
                        f"groups in this circuit: {', '.join(groups) or 'none'}")
            elif kind == "type":
                matched = [d for d in circuit.devices
                           if d.__class__.__name__.lower() == value.lower()]
                if not matched:
                    raise VariationError(
                        f"variation selector {token!r} matched no devices")
            elif kind == "device":
                matched = [circuit.device(value)]
            else:
                raise VariationError(
                    f"unknown selector prefix {kind!r} in {token!r}; use model:, "
                    "group:, type: or device:")
        else:
            matched = [circuit.device(token)]
        for dev in matched:
            if dev.name not in seen:
                seen.add(dev.name)
                out.append(dev)
    if not out:
        raise VariationError(f"variation selector {selector!r} matched no devices")
    return out


@dataclass
class VariationModel:
    """A complete variation configuration."""

    variations: list[ParameterVariation] = field(default_factory=list)
    correlations: list[CorrelationSpec] = field(default_factory=list)
    enable_process: bool = True
    enable_mismatch: bool = True
    pelgrom_pair_convention: bool = True
    name: str = "default"

    def active(self) -> list[ParameterVariation]:
        out = []
        for v in self.variations:
            if not v.enabled:
                continue
            if v.scope == "global" and not self.enable_process:
                continue
            if v.scope == "local" and not self.enable_mismatch:
                continue
            out.append(v)
        return out

    @property
    def mode_label(self) -> str:
        if self.enable_process and self.enable_mismatch:
            return "process+mismatch"
        if self.enable_process:
            return "process"
        if self.enable_mismatch:
            return "mismatch"
        return "nominal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variations": [v.to_dict() for v in self.variations],
            "correlations": [c.to_dict() for c in self.correlations],
            "enable_process": self.enable_process,
            "enable_mismatch": self.enable_mismatch,
            "pelgrom_pair_convention": self.pelgrom_pair_convention,
            "mode": self.mode_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariationModel":
        variations = [ParameterVariation.from_dict(v)
                      for v in data.get("variations", [])]
        correlations = [CorrelationSpec(**c) for c in data.get("correlations", [])]
        return cls(
            variations=variations, correlations=correlations,
            enable_process=bool(data.get("enable_process", True)),
            enable_mismatch=bool(data.get("enable_mismatch", True)),
            pelgrom_pair_convention=bool(data.get("pelgrom_pair_convention", True)),
            name=str(data.get("name", "default")),
        )


def default_mismatch_model(circuit: Circuit, *,
                           sigma_vth_global: float = 0.025,
                           sigma_beta_global_pct: float = 3.0,
                           sigma_r_global_pct: float = 2.0,
                           sigma_r_local_pct: float = 0.5,
                           enable_process: bool = True,
                           enable_mismatch: bool = True,
                           include_passives: bool = True) -> VariationModel:
    """Build a physically sensible default variation model for *circuit*.

    * global threshold and current-factor spread, one draw per model card;
    * local Pelgrom-scaled threshold and current-factor mismatch, one draw per
      MOSFET, with sigma computed from that device's own W x L;
    * optional global + local resistor spread.

    Defaults (25 mV global Vth sigma, 3 % global beta sigma) are representative
    of a mature CMOS node; they are configuration, not physics, and every
    number is overridable.
    """
    variations: list[ParameterVariation] = []
    for model_name in sorted(circuit.mos_models):
        if not any(m.model == model_name for m in circuit.mosfets()):
            continue
        variations.append(ParameterVariation(
            parameter="vth", scope="global", distribution="gaussian",
            sigma=sigma_vth_global, targets=f"model:{model_name}",
            label=f"{model_name} global threshold spread"))
        variations.append(ParameterVariation(
            parameter="beta", scope="global", distribution="gaussian",
            sigma_pct=sigma_beta_global_pct, targets=f"model:{model_name}",
            label=f"{model_name} global current-factor spread"))

    if circuit.mosfets():
        variations.append(ParameterVariation(
            parameter="vth", scope="local", distribution="gaussian",
            pelgrom=True, targets="type:mosfet",
            label="local threshold mismatch (Pelgrom AVT / sqrt(2WL))"))
        variations.append(ParameterVariation(
            parameter="beta", scope="local", distribution="gaussian",
            pelgrom=True, targets="type:mosfet",
            label="local current-factor mismatch (Pelgrom ABETA / sqrt(2WL))"))

    if include_passives and any(d.__class__.__name__ == "Resistor"
                                for d in circuit.devices):
        variations.append(ParameterVariation(
            parameter="r", scope="global", distribution="gaussian",
            sigma_pct=sigma_r_global_pct, targets="type:resistor",
            share_by="all", label="global resistor sheet-rho spread"))
        variations.append(ParameterVariation(
            parameter="r", scope="local", distribution="gaussian",
            sigma_pct=sigma_r_local_pct, targets="type:resistor",
            label="local resistor mismatch"))

    return VariationModel(variations=variations, enable_process=enable_process,
                          enable_mismatch=enable_mismatch, name="default")
