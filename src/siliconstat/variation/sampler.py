"""Turning a :class:`VariationModel` into concrete perturbed circuits.

The sampler is the bridge between the statistical description and the
simulator.  It works in three steps:

1. **Enumerate slots.**  Every independent random variable in the model gets a
   named slot, in a deterministic (sorted) order -- ``NCH.vth_global``,
   ``M1.vth_local``, ``M2.beta_local`` ...  These names are the parameter axis
   used later by correlation and sensitivity analysis.

2. **Draw.**  For sample *i*, a private ``numpy.random.Generator`` seeded from
   ``SeedSequence(master_seed).spawn(n)[i]`` produces one standard normal per
   slot.  Correlated groups are then mixed through their Cholesky factor and
   each slot applies its own marginal transform.

   Seeding per *sample* rather than per *run* is what makes parallel execution
   bit-for-bit identical to sequential execution: sample 731 gets the same
   stream regardless of which worker runs it or in what order.

3. **Apply.**  Slot deviations accumulate per (device, parameter) and become
   device overrides -- ``dvth`` in volts, ``beta_scale`` as a multiplier,
   ``w``/``l``/``r``/``c`` as absolute perturbed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.circuit import Circuit
from ..core.devices import Device, Mosfet
from ..core.exceptions import VariationError
from . import pelgrom
from .correlation import CorrelationSpec, build_correlation_matrix, cholesky_factor
from .distributions import Distribution, make_distribution
from .spec import PARAMETER_UNITS, ParameterVariation, VariationModel, select_devices

__all__ = ["VariationSlot", "VariationSampler", "SampleDraw"]


@dataclass(frozen=True)
class VariationSlot:
    """One independent random variable in the variation model."""

    name: str
    parameter: str
    scope: str
    key: str                      # model card, matched group, or device name
    devices: tuple[str, ...]      # devices this slot perturbs
    distribution: Distribution
    mode: str                     # 'additive' | 'relative'
    sigma: float                  # canonical sigma (volts, or fraction)
    correlation_group: str | None
    truncate_sigma: float | None
    unit: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "parameter": self.parameter, "scope": self.scope,
            "key": self.key, "devices": list(self.devices),
            "distribution": self.distribution.name, "mode": self.mode,
            "sigma": self.sigma, "unit": self.unit, "label": self.label,
            "correlation_group": self.correlation_group,
            "truncate_sigma": self.truncate_sigma,
        }


@dataclass
class SampleDraw:
    """The realised random state of one Monte Carlo sample."""

    index: int
    seed: int
    deviations: np.ndarray                       # one value per slot
    overrides: dict[str, dict[str, float]]       # device -> param -> value
    slot_values: dict[str, float] = field(default_factory=dict)
    device_values: dict[str, float] = field(default_factory=dict)


class VariationSampler:
    """Builds perturbed device parameters from a variation model."""

    def __init__(self, circuit: Circuit, model: VariationModel) -> None:
        self.circuit = circuit
        self.model = model
        self.slots: list[VariationSlot] = self._build_slots()
        self.slot_index = {s.name: i for i, s in enumerate(self.slots)}
        self._groups = self._build_correlation_groups()
        self._nominals = self._collect_nominals()

    # -- construction ------------------------------------------------------
    def _build_slots(self) -> list[VariationSlot]:
        slots: list[VariationSlot] = []
        used: dict[str, int] = {}

        for variation in self.model.active():
            devices = [d for d in select_devices(self.circuit, variation.targets)
                       if variation.applies_to(d)]
            if not devices:
                raise VariationError(
                    f"variation on {variation.parameter!r} (targets="
                    f"{variation.targets!r}) matched devices, but none of them "
                    f"expose that parameter")

            if variation.scope == "global":
                buckets: dict[str, list[Device]] = {}
                for dev in devices:
                    buckets.setdefault(self._share_key(dev, variation), []).append(dev)
                for key in sorted(buckets):
                    members = buckets[key]
                    sigma = self._resolve_sigma(variation, members[0], shared=True)
                    slots.append(self._make_slot(
                        variation, f"{key}.{variation.parameter}_global", key,
                        tuple(d.name for d in members), sigma, used))
            else:
                for dev in sorted(devices, key=lambda d: d.name):
                    sigma = self._resolve_sigma(variation, dev, shared=False)
                    slots.append(self._make_slot(
                        variation, f"{dev.name}.{variation.parameter}_local",
                        dev.name, (dev.name,), sigma, used))
        return slots

    def _make_slot(self, variation: ParameterVariation, name: str, key: str,
                   devices: tuple[str, ...], sigma: float,
                   used: dict[str, int]) -> VariationSlot:
        if name in used:
            used[name] += 1
            name = f"{name}#{used[name]}"
        else:
            used[name] = 1
        distribution = make_distribution(variation.distribution, sigma)
        mode = variation.effective_mode
        # A relative deviation is a dimensionless fraction of the nominal, not
        # a quantity in the parameter's own unit -- label it accordingly.
        unit = PARAMETER_UNITS.get(variation.parameter, "") if mode == "additive" else "frac"
        return VariationSlot(
            name=name, parameter=variation.parameter, scope=variation.scope,
            key=key, devices=devices, distribution=distribution,
            mode=mode, sigma=sigma,
            correlation_group=variation.correlation_group,
            truncate_sigma=variation.truncate_sigma, unit=unit,
            label=variation.label or f"{variation.parameter} {variation.scope}",
        )

    @staticmethod
    def _share_key(device: Device, variation: ParameterVariation) -> str:
        if variation.share_by == "all":
            return "ALL"
        if variation.share_by == "matched_group":
            group = getattr(device, "matched_group", None)
            if group:
                return str(group)
        model = getattr(device, "model", None)
        if model:
            return str(model)
        return device.__class__.__name__.upper()

    def _resolve_sigma(self, variation: ParameterVariation, device: Device,
                       *, shared: bool) -> float:
        """Sigma in canonical units: volts for ``vth``, fraction otherwise."""
        param = variation.parameter
        if variation.pelgrom:
            if not isinstance(device, Mosfet):
                raise VariationError(
                    f"Pelgrom scaling requires a MOSFET (device {device.name} is "
                    f"{device.__class__.__name__})")
            card = self.circuit.mos_models[device.model]
            pair = self.model.pelgrom_pair_convention
            if param == "vth":
                return pelgrom.sigma_vth_device(card.avt, device.w, device.l,
                                                pair_convention=pair)
            return pelgrom.sigma_beta_device(card.abeta, device.w, device.l,
                                             pair_convention=pair)
        if variation.sigma is not None:
            return float(variation.sigma)
        pct = float(variation.sigma_pct) / 100.0
        if param == "vth":
            nominal = self._nominal_vth(device)
            return pct * nominal
        return pct

    def _nominal_vth(self, device: Device) -> float:
        if isinstance(device, Mosfet):
            return self.circuit.mos_models[device.model].vto
        raise VariationError(
            f"cannot resolve a nominal threshold for device {device.name}")

    def _build_correlation_groups(self) -> list[tuple[np.ndarray, np.ndarray]]:
        specs = {c.name: c for c in self.model.correlations}
        by_group: dict[str, list[int]] = {}
        for i, slot in enumerate(self.slots):
            if slot.correlation_group:
                by_group.setdefault(slot.correlation_group, []).append(i)
        unknown = set(by_group) - set(specs)
        if unknown:
            raise VariationError(
                f"variation(s) reference correlation group(s) "
                f"{', '.join(sorted(unknown))} which are not defined in the "
                f"model's 'correlations' list")
        unused = set(specs) - set(by_group)
        if unused:
            raise VariationError(
                f"correlation group(s) {', '.join(sorted(unused))} are defined but "
                "no variation references them")
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for name in sorted(by_group):
            idx = np.array(by_group[name], dtype=int)
            order = [self.slots[i].name for i in idx]
            matrix = build_correlation_matrix(specs[name], order)
            out.append((idx, cholesky_factor(matrix, name=name)))
        return out

    def _collect_nominals(self) -> dict[tuple[str, str], float]:
        nominals: dict[tuple[str, str], float] = {}
        for dev in self.circuit.devices:
            for attr, param in (("w", "w"), ("l", "l"), ("r", "r"),
                                ("c", "c"), ("dc", "dc")):
                if hasattr(dev, attr):
                    nominals[(dev.name, param)] = float(getattr(dev, attr))
        return nominals

    # -- introspection -----------------------------------------------------
    @property
    def n_slots(self) -> int:
        return len(self.slots)

    @property
    def slot_names(self) -> list[str]:
        return [s.name for s in self.slots]

    def describe(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.slots]

    # -- sampling ----------------------------------------------------------
    @staticmethod
    def child_seeds(master_seed: int, n_samples: int) -> list[np.random.SeedSequence]:
        """Independent, reproducible seed sequences -- one per sample."""
        return np.random.SeedSequence(master_seed).spawn(n_samples)

    def draw_standard(self, rng: np.random.Generator) -> np.ndarray:
        """One correlated standard-normal vector, one entry per slot."""
        z = rng.standard_normal(self.n_slots)
        for idx, chol in self._groups:
            z[idx] = chol @ z[idx]
        for i, slot in enumerate(self.slots):
            if slot.truncate_sigma:
                z[i] = float(np.clip(z[i], -slot.truncate_sigma, slot.truncate_sigma))
        return z

    def deviations_from_standard(self, z: np.ndarray) -> np.ndarray:
        out = np.empty(self.n_slots, dtype=float)
        for i, slot in enumerate(self.slots):
            out[i] = float(slot.distribution.from_standard_normal(np.array([z[i]]))[0])
        return out

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        return self.deviations_from_standard(self.draw_standard(rng))

    def sample(self, index: int, seed_sequence: np.random.SeedSequence) -> SampleDraw:
        """Draw sample *index* and build its device overrides."""
        rng = np.random.default_rng(seed_sequence)
        deviations = self.draw(rng)
        overrides, device_values = self.apply(deviations)
        return SampleDraw(
            index=index,
            seed=int(seed_sequence.entropy) if isinstance(seed_sequence.entropy, int)
            else int(seed_sequence.generate_state(1, dtype=np.uint32)[0]),
            deviations=deviations,
            overrides=overrides,
            slot_values={s.name: float(deviations[i]) for i, s in enumerate(self.slots)},
            device_values=device_values,
        )

    def apply(self, deviations: np.ndarray
              ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """Accumulate slot deviations into per-device parameter overrides."""
        additive: dict[tuple[str, str], float] = {}
        relative: dict[tuple[str, str], float] = {}
        for i, slot in enumerate(self.slots):
            value = float(deviations[i])
            bucket = additive if slot.mode == "additive" else relative
            for dev_name in slot.devices:
                key = (dev_name, slot.parameter)
                bucket[key] = bucket.get(key, 0.0) + value

        overrides: dict[str, dict[str, float]] = {}
        device_values: dict[str, float] = {}
        for (dev_name, param) in set(additive) | set(relative):
            add = additive.get((dev_name, param), 0.0)
            rel = relative.get((dev_name, param), 0.0)
            target = overrides.setdefault(dev_name, {})
            if param == "vth":
                target["dvth"] = add + rel
                device_values[f"{dev_name}.dvth"] = add + rel
            elif param == "beta":
                scale = (1.0 + rel) * (1.0 + add)
                if scale <= 0.0:
                    raise VariationError(
                        f"device {dev_name}: sampled current-factor scale is "
                        f"{scale:.4g} (<= 0). The configured beta sigma is large "
                        "enough to produce non-physical devices; reduce it or "
                        "truncate the distribution with truncate_sigma.")
                target["beta_scale"] = scale
                device_values[f"{dev_name}.beta_scale"] = scale
            elif param in ("is", "n"):
                scale = (1.0 + rel) * (1.0 + add)
                if scale <= 0.0:
                    raise VariationError(
                        f"device {dev_name}: sampled {param.upper()} scale is "
                        f"{scale:.4g} (<= 0), which is not physical")
                target[f"{param}_scale"] = scale
                device_values[f"{dev_name}.{param}_scale"] = scale
            else:
                nominal = self._nominals.get((dev_name, param))
                if nominal is None:
                    raise VariationError(
                        f"device {dev_name} has no nominal value for {param!r}")
                value = (nominal + add) * (1.0 + rel)
                if param in ("w", "l", "r", "c") and value <= 0.0:
                    raise VariationError(
                        f"device {dev_name}: sampled {param.upper()} is "
                        f"{value:.4g} (<= 0), which is not physical. Reduce the "
                        "configured sigma or truncate the distribution.")
                target[param] = value
                device_values[f"{dev_name}.{param}"] = value
        return overrides, device_values

    def perturbed_circuit(self, draw: SampleDraw) -> Circuit:
        return self.circuit.with_overrides(draw.overrides)

    # -- analytical helpers -------------------------------------------------
    def slot_sigma_table(self) -> list[dict[str, Any]]:
        """Per-slot sigma summary -- used by the UI and the HTML report."""
        rows = []
        for slot in self.slots:
            rows.append({
                "slot": slot.name, "parameter": slot.parameter,
                "scope": slot.scope, "distribution": slot.distribution.name,
                "sigma": slot.sigma, "unit": slot.unit,
                "devices": ", ".join(slot.devices), "label": slot.label,
            })
        return rows
