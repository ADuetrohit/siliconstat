"""Circuit elements and their MNA stamps.

Every device knows how to contribute to three analyses:

``stamp_dc``    real-valued DC / Newton-Raphson companion stamps
``stamp_tran``  same, plus companion models for reactive elements
``stamp_ac``    complex small-signal stamps linearised about a DC operating point

Devices are immutable value objects; the Monte Carlo layer creates perturbed
copies with :meth:`Device.with_overrides` rather than mutating shared state,
which is what makes parallel sampling safe.

Statistical override convention
-------------------------------
Overrides are expressed as *modifiers*, never as absolute replacements of
process quantities, so that they compose correctly with temperature and corner
shifts applied later in the pipeline:

===============  =========================================================
``dvth``         additive threshold shift [V]      ``Vth = Vth(T,corner) + dvth``
``beta_scale``   multiplicative current factor     ``KP  = KP(T,corner) * beta_scale``
``w``, ``l``     absolute geometry [m]
``r``, ``c``     absolute component values
===============  =========================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Iterable

import numpy as np

from .context import SimContext
from .exceptions import CircuitError
from .mna import MnaSystem
from .models import thermal_voltage
from .mosfet import MosOp, eval_mosfet, fet_limit, pn_limit
from .waveforms import Waveform

__all__ = [
    "Device", "Resistor", "Capacitor", "Inductor", "VoltageSource",
    "CurrentSource", "Diode", "Mosfet", "node_voltage", "iter_nonlinear",
]


def node_voltage(x: np.ndarray, node: int) -> float:
    """Voltage of circuit node *node* from solution vector *x* (ground = 0 V)."""
    return 0.0 if node == 0 else float(x[node - 1])


@dataclass(frozen=True)
class Device:
    """Base class for all circuit elements."""

    name: str
    nodes: tuple[int, ...]
    branch: int = field(default=-1, compare=False)

    #: Human-facing names of parameters the variation layer may perturb.
    #: ClassVar keeps this out of the dataclass field list -- without it,
    #: dataclasses turns it into a per-instance field whose default ()
    #: shadows every subclass's value on instances.
    VARIABLE_PARAMS: ClassVar[tuple[str, ...]] = ()

    # -- structure ---------------------------------------------------------
    @property
    def n_branches(self) -> int:
        return 0

    @property
    def is_nonlinear(self) -> bool:
        return False

    def with_branch(self, branch: int) -> "Device":
        return replace(self, branch=branch)

    def with_overrides(self, overrides: dict[str, float]) -> "Device":
        """Return a copy with perturbed parameters (used by Monte Carlo)."""
        if not overrides:
            return self
        raise CircuitError(
            f"device {self.name} ({self.__class__.__name__}) does not support "
            f"parameter overrides {sorted(overrides)}"
        )

    # -- stamping ----------------------------------------------------------
    def stamp_dc(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        raise NotImplementedError  # pragma: no cover

    def stamp_tran(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        self.stamp_dc(sys, x, ctx)

    def stamp_ac(self, sys: MnaSystem, x_op: np.ndarray, omega: float,
                 ctx: SimContext) -> None:
        raise NotImplementedError  # pragma: no cover

    # -- reporting ---------------------------------------------------------
    def current(self, x: np.ndarray, ctx: SimContext) -> float:
        """Current through the device, positive from ``nodes[0]`` to ``nodes[1]``."""
        raise NotImplementedError  # pragma: no cover

    def power(self, x: np.ndarray, ctx: SimContext) -> float:
        """Power dissipated in the device (or delivered, for sources)."""
        raise NotImplementedError  # pragma: no cover

    def op_info(self, x: np.ndarray, ctx: SimContext) -> dict[str, Any]:
        return {"type": self.__class__.__name__.lower(), "name": self.name}

    def init_state(self, x: np.ndarray, ctx: SimContext) -> None:
        """Seed transient history from the DC operating point."""
        return None

    def update_state(self, x: np.ndarray, ctx: SimContext) -> None:
        """Persist transient history after an accepted timestep."""
        return None


# ---------------------------------------------------------------------------
# Linear two-terminal elements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Resistor(Device):
    r: float = 1e3

    VARIABLE_PARAMS: ClassVar[tuple[str, ...]] = ("r",)

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise CircuitError(f"resistor {self.name}: resistance must be > 0 (got {self.r})")

    def with_overrides(self, overrides: dict[str, float]) -> "Resistor":
        if not overrides:
            return self
        r = overrides.get("r", self.r)
        if r <= 0:
            raise CircuitError(f"resistor {self.name}: perturbed resistance {r} is not positive")
        return replace(self, r=r)

    def stamp_dc(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        sys.add_conductance(self.nodes[0], self.nodes[1], 1.0 / self.r)

    def stamp_ac(self, sys: MnaSystem, x_op: np.ndarray, omega: float,
                 ctx: SimContext) -> None:
        sys.add_conductance(self.nodes[0], self.nodes[1], 1.0 / self.r)

    def current(self, x: np.ndarray, ctx: SimContext) -> float:
        return (node_voltage(x, self.nodes[0]) - node_voltage(x, self.nodes[1])) / self.r

    def power(self, x: np.ndarray, ctx: SimContext) -> float:
        dv = node_voltage(x, self.nodes[0]) - node_voltage(x, self.nodes[1])
        return dv * dv / self.r

    def op_info(self, x: np.ndarray, ctx: SimContext) -> dict[str, Any]:
        return {"type": "resistor", "name": self.name, "r": self.r,
                "i": self.current(x, ctx), "p": self.power(x, ctx)}


@dataclass(frozen=True)
class Capacitor(Device):
    c: float = 1e-12
    ic: float | None = None

    VARIABLE_PARAMS: ClassVar[tuple[str, ...]] = ("c",)

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise CircuitError(f"capacitor {self.name}: capacitance must be > 0 (got {self.c})")

    def with_overrides(self, overrides: dict[str, float]) -> "Capacitor":
        if not overrides:
            return self
        c = overrides.get("c", self.c)
        if c <= 0:
            raise CircuitError(f"capacitor {self.name}: perturbed capacitance {c} is not positive")
        return replace(self, c=c)

    def stamp_dc(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        return None  # open circuit at DC

    def stamp_tran(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        dt = ctx.dt
        if dt <= 0:
            return None
        v_prev, i_prev = ctx.state.get(self.name, (0.0, 0.0))
        if ctx.integration == "trap":
            geq = 2.0 * self.c / dt
            ieq = geq * v_prev + i_prev
        else:  # backward Euler
            geq = self.c / dt
            ieq = geq * v_prev
        sys.add_conductance(self.nodes[0], self.nodes[1], geq)
        sys.add_current(self.nodes[1], self.nodes[0], ieq)
        return None

    def stamp_ac(self, sys: MnaSystem, x_op: np.ndarray, omega: float,
                 ctx: SimContext) -> None:
        sys.add_conductance(self.nodes[0], self.nodes[1], 1j * omega * self.c)

    def current(self, x: np.ndarray, ctx: SimContext) -> float:
        if ctx.mode != "tran" or ctx.dt <= 0:
            return 0.0
        v = node_voltage(x, self.nodes[0]) - node_voltage(x, self.nodes[1])
        v_prev, i_prev = ctx.state.get(self.name, (0.0, 0.0))
        if ctx.integration == "trap":
            return (2.0 * self.c / ctx.dt) * (v - v_prev) - i_prev
        return (self.c / ctx.dt) * (v - v_prev)

    def power(self, x: np.ndarray, ctx: SimContext) -> float:
        return 0.0

    def init_state(self, x: np.ndarray, ctx: SimContext) -> None:
        v = node_voltage(x, self.nodes[0]) - node_voltage(x, self.nodes[1])
        ctx.state[self.name] = (self.ic if self.ic is not None else v, 0.0)

    def update_state(self, x: np.ndarray, ctx: SimContext) -> None:
        v = node_voltage(x, self.nodes[0]) - node_voltage(x, self.nodes[1])
        ctx.state[self.name] = (v, self.current(x, ctx))

    def op_info(self, x: np.ndarray, ctx: SimContext) -> dict[str, Any]:
        return {"type": "capacitor", "name": self.name, "c": self.c}


@dataclass(frozen=True)
class Inductor(Device):
    l: float = 1e-9

    #: Inductance is intentionally not exposed to the variation layer: the name
    #: ``l`` already means MOSFET channel length there, and a shared name would
    #: make a variation rule ambiguous.
    VARIABLE_PARAMS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        if self.l <= 0:
            raise CircuitError(f"inductor {self.name}: inductance must be > 0 (got {self.l})")

    @property
    def n_branches(self) -> int:
        return 1

    # No with_overrides(): inductance is not a variable parameter, so the base
    # class's "this device does not support overrides" error is the right
    # answer rather than silently ignoring the request.

    def _stamp_branch(self, sys: MnaSystem, coeff: complex | float,
                      rhs: complex | float) -> None:
        np_, nn = self.nodes
        br = sys.branch_row(self.branch)
        sys.add(np_ - 1, br, 1.0)
        sys.add(nn - 1, br, -1.0)
        sys.add(br, np_ - 1, 1.0)
        sys.add(br, nn - 1, -1.0)
        sys.add(br, br, coeff)
        sys.add_rhs(br, rhs)

    def stamp_dc(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        self._stamp_branch(sys, 0.0, 0.0)  # ideal short at DC

    def stamp_tran(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        if ctx.dt <= 0:
            return self.stamp_dc(sys, x, ctx)
        i_prev, v_prev = ctx.state.get(self.name, (0.0, 0.0))
        if ctx.integration == "trap":
            req = 2.0 * self.l / ctx.dt
            self._stamp_branch(sys, -req, -req * i_prev - v_prev)
        else:
            req = self.l / ctx.dt
            self._stamp_branch(sys, -req, -req * i_prev)
        return None

    def stamp_ac(self, sys: MnaSystem, x_op: np.ndarray, omega: float,
                 ctx: SimContext) -> None:
        self._stamp_branch(sys, -1j * omega * self.l, 0.0)

    def current(self, x: np.ndarray, ctx: SimContext) -> float:
        return float(np.real(x[(ctx.n_nodes - 1) + self.branch]))

    def power(self, x: np.ndarray, ctx: SimContext) -> float:
        return 0.0

    def init_state(self, x: np.ndarray, ctx: SimContext) -> None:
        ctx.state[self.name] = (self.current(x, ctx), 0.0)

    def update_state(self, x: np.ndarray, ctx: SimContext) -> None:
        i = self.current(x, ctx)
        v = node_voltage(x, self.nodes[0]) - node_voltage(x, self.nodes[1])
        ctx.state[self.name] = (i, v)

    def op_info(self, x: np.ndarray, ctx: SimContext) -> dict[str, Any]:
        return {"type": "inductor", "name": self.name, "l": self.l,
                "i": self.current(x, ctx)}


# ---------------------------------------------------------------------------
# Independent sources
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoltageSource(Device):
    dc: float = 0.0
    ac_mag: float = 0.0
    ac_phase: float = 0.0
    wave: Waveform | None = None

    VARIABLE_PARAMS: ClassVar[tuple[str, ...]] = ("dc",)

    @property
    def n_branches(self) -> int:
        return 1

    def with_overrides(self, overrides: dict[str, float]) -> "VoltageSource":
        if not overrides:
            return self
        return replace(self, dc=overrides.get("dc", self.dc))

    def value(self, ctx: SimContext) -> float:
        """DC/transient value including run-time overrides and source stepping."""
        if self.name in ctx.source_overrides:
            base = ctx.source_overrides[self.name]
        elif ctx.mode == "tran" and self.wave is not None:
            base = self.wave.value(ctx.time)
        else:
            base = self.dc
        return base * ctx.source_scale

    def _stamp_source(self, sys: MnaSystem, value: complex | float) -> None:
        np_, nn = self.nodes
        br = sys.branch_row(self.branch)
        sys.add(np_ - 1, br, 1.0)
        sys.add(nn - 1, br, -1.0)
        sys.add(br, np_ - 1, 1.0)
        sys.add(br, nn - 1, -1.0)
        sys.add_rhs(br, value)

    def stamp_dc(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        self._stamp_source(sys, self.value(ctx))

    def stamp_ac(self, sys: MnaSystem, x_op: np.ndarray, omega: float,
                 ctx: SimContext) -> None:
        self._stamp_source(sys, self.ac_mag * np.exp(1j * math.radians(self.ac_phase)))

    def current(self, x: np.ndarray, ctx: SimContext) -> float:
        """Branch current, positive flowing from ``+`` through the source to ``-``."""
        return float(np.real(x[(ctx.n_nodes - 1) + self.branch]))

    def power(self, x: np.ndarray, ctx: SimContext) -> float:
        """Power *delivered* by the source into the rest of the circuit."""
        v = node_voltage(x, self.nodes[0]) - node_voltage(x, self.nodes[1])
        return -v * self.current(x, ctx)

    def op_info(self, x: np.ndarray, ctx: SimContext) -> dict[str, Any]:
        return {"type": "vsource", "name": self.name, "dc": self.dc,
                "v": self.value(ctx), "i": self.current(x, ctx),
                "p_delivered": self.power(x, ctx)}


@dataclass(frozen=True)
class CurrentSource(Device):
    dc: float = 0.0
    ac_mag: float = 0.0
    ac_phase: float = 0.0
    wave: Waveform | None = None

    VARIABLE_PARAMS: ClassVar[tuple[str, ...]] = ("dc",)

    def with_overrides(self, overrides: dict[str, float]) -> "CurrentSource":
        if not overrides:
            return self
        return replace(self, dc=overrides.get("dc", self.dc))

    def value(self, ctx: SimContext) -> float:
        if self.name in ctx.source_overrides:
            base = ctx.source_overrides[self.name]
        elif ctx.mode == "tran" and self.wave is not None:
            base = self.wave.value(ctx.time)
        else:
            base = self.dc
        return base * ctx.source_scale

    def stamp_dc(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        sys.add_current(self.nodes[0], self.nodes[1], self.value(ctx))

    def stamp_ac(self, sys: MnaSystem, x_op: np.ndarray, omega: float,
                 ctx: SimContext) -> None:
        sys.add_current(self.nodes[0], self.nodes[1],
                        self.ac_mag * np.exp(1j * math.radians(self.ac_phase)))

    def current(self, x: np.ndarray, ctx: SimContext) -> float:
        return self.value(ctx)

    def power(self, x: np.ndarray, ctx: SimContext) -> float:
        """Power delivered by the source (negative when it absorbs)."""
        v = node_voltage(x, self.nodes[0]) - node_voltage(x, self.nodes[1])
        return -v * self.value(ctx)

    def op_info(self, x: np.ndarray, ctx: SimContext) -> dict[str, Any]:
        return {"type": "isource", "name": self.name, "dc": self.dc,
                "i": self.value(ctx), "p_delivered": self.power(x, ctx)}


# ---------------------------------------------------------------------------
# Nonlinear devices
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Diode(Device):
    model: str = "DEFAULT"
    area: float = 1.0
    is_scale: float = 1.0
    n_scale: float = 1.0

    VARIABLE_PARAMS: ClassVar[tuple[str, ...]] = ("is", "n")

    @property
    def is_nonlinear(self) -> bool:
        return True

    def with_overrides(self, overrides: dict[str, float]) -> "Diode":
        if not overrides:
            return self
        return replace(
            self,
            is_scale=overrides.get("is_scale", self.is_scale),
            n_scale=overrides.get("n_scale", self.n_scale),
            area=overrides.get("area", self.area),
        )

    def _params(self, ctx: SimContext) -> tuple[float, float]:
        model = ctx.diode_model(self.model)
        is_ = model.is_ * self.is_scale * self.area
        n = model.n * self.n_scale
        return max(is_, 1e-30), thermal_voltage(ctx.temp_c) * max(n, 1e-3)

    def _evaluate(self, x: np.ndarray, ctx: SimContext) -> tuple[float, float, float]:
        na, nc = self.nodes
        is_, vt = self._params(ctx)
        vd = node_voltage(x, na) - node_voltage(x, nc)
        if ctx.limiting and ctx.x_prev_iter is not None:
            vd_old = (node_voltage(ctx.x_prev_iter, na)
                      - node_voltage(ctx.x_prev_iter, nc))
            vcrit = vt * math.log(vt / (math.sqrt(2.0) * is_))
            vd = pn_limit(vd, vd_old, vt, vcrit)
        arg = vd / vt
        if arg > 80.0:  # linearise beyond the exponential overflow horizon
            evd = math.exp(80.0)
            id_ = is_ * (evd * (1.0 + (arg - 80.0)) - 1.0)
            gd = is_ * evd / vt
        else:
            evd = math.exp(arg)
            id_ = is_ * (evd - 1.0)
            gd = is_ * evd / vt
        return vd, id_ + ctx.gmin * vd, gd + ctx.gmin

    def stamp_dc(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        na, nc = self.nodes
        vd, id_, gd = self._evaluate(x, ctx)
        sys.add_conductance(na, nc, gd)
        sys.add_current(na, nc, id_ - gd * vd)

    def stamp_ac(self, sys: MnaSystem, x_op: np.ndarray, omega: float,
                 ctx: SimContext) -> None:
        model = ctx.diode_model(self.model)
        _vd, _id, gd = self._evaluate(x_op, ctx)
        sys.add_conductance(self.nodes[0], self.nodes[1],
                            gd + 1j * omega * model.cj0 * self.area)

    def current(self, x: np.ndarray, ctx: SimContext) -> float:
        return self._evaluate(x, ctx)[1]

    def power(self, x: np.ndarray, ctx: SimContext) -> float:
        vd, id_, _ = self._evaluate(x, ctx)
        return vd * id_

    def op_info(self, x: np.ndarray, ctx: SimContext) -> dict[str, Any]:
        vd, id_, gd = self._evaluate(x, ctx)
        return {"type": "diode", "name": self.name, "model": self.model,
                "vd": vd, "id": id_, "gd": gd}


@dataclass(frozen=True)
class Mosfet(Device):
    """MOSFET instance: ``M<name> D G S B <model> W=.. L=.. [M=..] [MATCH=..]``."""

    model: str = "NMOS"
    w: float = 1e-6
    l: float = 1e-6
    m: float = 1.0
    matched_group: str | None = None
    dvth: float = 0.0          # additive threshold shift from variation [V]
    beta_scale: float = 1.0    # multiplicative current-factor deviation

    VARIABLE_PARAMS: ClassVar[tuple[str, ...]] = ("vth", "beta", "w", "l")

    def __post_init__(self) -> None:
        if self.w <= 0 or self.l <= 0:
            raise CircuitError(
                f"mosfet {self.name}: W and L must be > 0 (got W={self.w}, L={self.l})")
        if self.m <= 0:
            raise CircuitError(f"mosfet {self.name}: multiplicity M must be > 0")
        if self.beta_scale <= 0:
            raise CircuitError(
                f"mosfet {self.name}: beta_scale must be > 0 (got {self.beta_scale}); "
                "a non-physical current factor was requested by the variation model")

    @property
    def is_nonlinear(self) -> bool:
        return True

    @property
    def area_um2(self) -> float:
        return (self.w * 1e6) * (self.l * 1e6)

    def with_overrides(self, overrides: dict[str, float]) -> "Mosfet":
        if not overrides:
            return self
        w = overrides.get("w", self.w)
        l = overrides.get("l", self.l)
        if w <= 0 or l <= 0:
            raise CircuitError(
                f"mosfet {self.name}: perturbed geometry is non-physical (W={w}, L={l})")
        return replace(
            self, w=w, l=l,
            dvth=overrides.get("dvth", self.dvth),
            beta_scale=overrides.get("beta_scale", self.beta_scale),
        )

    # -- physics -----------------------------------------------------------
    def effective_vth0(self, ctx: SimContext) -> float:
        """Zero-bias threshold magnitude including corner/temperature + statistics."""
        return ctx.mos_model(self.model).vto + self.dvth

    def effective_kp(self, ctx: SimContext) -> float:
        return ctx.mos_model(self.model).kp * self.beta_scale

    def evaluate(self, x: np.ndarray, ctx: SimContext, *,
                 with_caps: bool = False) -> MosOp:
        model = ctx.mos_model(self.model)
        d, g, s, b = self.nodes
        vd, vg, vs, vb = (node_voltage(x, d), node_voltage(x, g),
                          node_voltage(x, s), node_voltage(x, b))
        vth0 = self.effective_vth0(ctx)

        if ctx.limiting and ctx.limit_mos and ctx.x_prev_iter is not None:
            xo = ctx.x_prev_iter
            sign = model.sign
            vgs_new = sign * (vg - vs)
            vds_new = sign * (vd - vs)
            vgs_old = sign * (node_voltage(xo, g) - node_voltage(xo, s))
            vds_old = sign * (node_voltage(xo, d) - node_voltage(xo, s))
            vgs_lim = fet_limit(vgs_new, vgs_old, vth0)
            vds_lim = fet_limit(vds_new, vds_old, 0.0, near_step=2.0, far_step=2.0)
            # Re-reference the limited voltages to the unchanged source node.
            vg = vs + sign * vgs_lim
            vd = vs + sign * vds_lim

        return eval_mosfet(model, self.w, self.l, vd, vg, vs, vb,
                           vth0=vth0, kp=self.effective_kp(ctx),
                           multiplicity=self.m, with_caps=with_caps)

    # -- stamps ------------------------------------------------------------
    def _stamp_conductances(self, sys: MnaSystem, op: MosOp, sign: float,
                            gmin: float) -> None:
        d, g, s, b = self.nodes
        if op.swapped:
            d, s = s, d
        sys.add_vccs(d, s, g, s, op.gm)
        sys.add_vccs(d, s, d, s, op.gds)
        if op.gmb:
            sys.add_vccs(d, s, b, s, op.gmb)
        ieq = sign * (op.ids - op.gm * op.vgs - op.gds * op.vds - op.gmb * op.vbs)
        sys.add_current(d, s, ieq)
        if gmin > 0:
            sys.add_conductance(d, s, gmin)

    def stamp_dc(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        model = ctx.mos_model(self.model)
        self._stamp_conductances(sys, self.evaluate(x, ctx), model.sign, ctx.gmin)

    # -- transient capacitance handling ------------------------------------
    def _compute_cap_list(self, x: np.ndarray, ctx: SimContext
                          ) -> list[tuple[int, int, float, str]]:
        op = self.evaluate(x, ctx, with_caps=True)
        d, g, s, b = self.nodes
        if op.swapped:
            d, s = s, d
        return [(g, s, op.cgs, "cgs"), (g, d, op.cgd, "cgd"),
                (g, b, op.cgb, "cgb"), (d, b, op.cdb, "cdb"),
                (s, b, op.csb, "csb")]

    def _cap_list(self, ctx: SimContext) -> list[tuple[int, int, float, str]]:
        """Capacitances frozen for the duration of one timestep.

        Meyer capacitances are piecewise functions of the operating region, so
        recomputing them inside the Newton loop makes the companion model
        discontinuous and the iteration oscillates at fast edges (exactly where
        the region boundary is crossed).  Holding them at the value from the
        last converged timestep restores a smooth Jacobian.  This is the usual
        quasi-static treatment of Meyer models; see ``docs/limitations.md``.
        """
        return ctx.state.get(f"{self.name}:caplist", [])

    def stamp_tran(self, sys: MnaSystem, x: np.ndarray, ctx: SimContext) -> None:
        model = ctx.mos_model(self.model)
        self._stamp_conductances(sys, self.evaluate(x, ctx), model.sign, ctx.gmin)
        if ctx.dt <= 0:
            return
        for na, nb, cap, key in self._cap_list(ctx):
            if cap <= 0:
                continue
            v_prev, i_prev = ctx.state.get(f"{self.name}:{key}", (0.0, 0.0))
            if ctx.integration == "trap":
                geq = 2.0 * cap / ctx.dt
                ieq = geq * v_prev + i_prev
            else:
                geq = cap / ctx.dt
                ieq = geq * v_prev
            sys.add_conductance(na, nb, geq)
            sys.add_current(nb, na, ieq)

    def stamp_ac(self, sys: MnaSystem, x_op: np.ndarray, omega: float,
                 ctx: SimContext) -> None:
        d, g, s, b = self.nodes
        op = self.evaluate(x_op, ctx, with_caps=True)
        if op.swapped:
            d, s = s, d
        sys.add_vccs(d, s, g, s, op.gm)
        sys.add_vccs(d, s, d, s, op.gds)
        if op.gmb:
            sys.add_vccs(d, s, b, s, op.gmb)
        if ctx.gmin > 0:
            sys.add_conductance(d, s, ctx.gmin)
        for na, nb, cap in ((g, s, op.cgs), (g, d, op.cgd), (g, b, op.cgb),
                            (d, b, op.cdb), (s, b, op.csb)):
            if cap > 0:
                sys.add_conductance(na, nb, 1j * omega * cap)

    def init_state(self, x: np.ndarray, ctx: SimContext) -> None:
        cap_list = self._compute_cap_list(x, ctx)
        ctx.state[f"{self.name}:caplist"] = cap_list
        for na, nb, cap, key in cap_list:
            v = node_voltage(x, na) - node_voltage(x, nb)
            ctx.state[f"{self.name}:{key}"] = (v, 0.0)

    def update_state(self, x: np.ndarray, ctx: SimContext) -> None:
        if ctx.dt <= 0:
            return
        # Close out the step with the capacitances that were actually used,
        # so the trapezoidal history (v, i) is self-consistent...
        for na, nb, cap, key in self._cap_list(ctx):
            if cap <= 0:
                continue
            v = node_voltage(x, na) - node_voltage(x, nb)
            v_prev, i_prev = ctx.state.get(f"{self.name}:{key}", (0.0, 0.0))
            if ctx.integration == "trap":
                i = (2.0 * cap / ctx.dt) * (v - v_prev) - i_prev
            else:
                i = (cap / ctx.dt) * (v - v_prev)
            ctx.state[f"{self.name}:{key}"] = (v, i)
        # ... then re-freeze the capacitances for the next step at the newly
        # converged operating point.
        new_list = self._compute_cap_list(x, ctx)
        old_pairs = {(na, nb, key) for na, nb, _c, key in self._cap_list(ctx)}
        new_pairs = {(na, nb, key) for na, nb, _c, key in new_list}
        if old_pairs and old_pairs != new_pairs:
            # The device reversed (drain and source exchanged), so the stored
            # histories refer to the wrong node pairs.  Re-seed them rather
            # than integrate garbage forward.
            for na, nb, _cap, key in new_list:
                v = node_voltage(x, na) - node_voltage(x, nb)
                ctx.state[f"{self.name}:{key}"] = (v, 0.0)
        ctx.state[f"{self.name}:caplist"] = new_list

    # -- reporting ---------------------------------------------------------
    def current(self, x: np.ndarray, ctx: SimContext) -> float:
        """Drain current, positive flowing *into* the physical drain terminal."""
        model = ctx.mos_model(self.model)
        op = self.evaluate(x, ctx)
        i = model.sign * op.ids
        return -i if op.swapped else i

    def power(self, x: np.ndarray, ctx: SimContext) -> float:
        d, _g, s, _b = self.nodes
        vds = node_voltage(x, d) - node_voltage(x, s)
        return abs(vds * self.current(x, ctx))

    def op_info(self, x: np.ndarray, ctx: SimContext) -> dict[str, Any]:
        model = ctx.mos_model(self.model)
        op = self.evaluate(x, ctx, with_caps=True)
        return {
            "type": "mosfet", "name": self.name, "model": self.model,
            "mtype": model.mtype, "w": self.w, "l": self.l, "m": self.m,
            "matched_group": self.matched_group,
            "id": self.current(x, ctx), "vgs": op.vgs, "vds": op.vds,
            "vbs": op.vbs, "vth": op.vth, "vov": op.vov, "region": op.region,
            "gm": op.gm, "gds": op.gds, "gmb": op.gmb, "ro": op.ro,
            "gm_over_id": op.gm_over_id, "beta": op.beta,
            "cgs": op.cgs, "cgd": op.cgd,
            "vth0_used": self.effective_vth0(ctx), "kp_used": self.effective_kp(ctx),
            "dvth": self.dvth, "beta_scale": self.beta_scale,
        }


def iter_nonlinear(devices: Iterable[Device]) -> Iterable[Device]:
    return (d for d in devices if d.is_nonlinear)
