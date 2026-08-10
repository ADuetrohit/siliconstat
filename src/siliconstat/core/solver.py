"""Nonlinear DC, small-signal AC and transient solvers.

DC operating point
------------------
Newton-Raphson on the MNA companion system.  Because each device stamps an
*exact* companion model at the current iterate, the residual of the nonlinear
system is available for free as ``G(x) @ x - b(x)`` -- no second assembly is
needed to test convergence in the current domain.

Convergence is declared only when **both** criteria hold (as in SPICE):

* voltage:  ``|dv| <= reltol * max(|v_new|, |v_old|) + vntol``
* current:  ``|KCL residual| <= reltol * max|branch current| + abstol``

If plain Newton fails, three continuation strategies are tried in order --
damped Newton, gmin stepping, source stepping.  If all fail, a
:class:`~siliconstat.core.exceptions.ConvergenceError` is raised carrying the
iteration count, residual and worst node.  The solver never returns a result
it could not verify.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .circuit import Circuit
from .context import SimContext
from .devices import Capacitor, Device, Mosfet, VoltageSource, node_voltage
from .exceptions import ConvergenceError, NumericalError, SimulationError
from .mna import MnaSystem

__all__ = [
    "SolverOptions", "OperatingPoint", "ACResult", "TransientResult",
    "solve_dc", "ac_analysis", "transient_analysis",
]


@dataclass
class SolverOptions:
    """Numerical tolerances and iteration limits."""

    # SPICE ships reltol=1e-3, which is fine when you only want three digits of
    # a node voltage.  It is *not* fine here: this tool routinely measures
    # current-mirror copy errors of order 0.08 %, so a 0.1 % solver tolerance
    # would sit on top of the very quantity being characterised.  The defaults
    # below put solver noise three orders of magnitude below the smallest
    # mismatch effect the demo circuits exhibit, at a cost of roughly one extra
    # Newton iteration (measured -- see docs/circuit_solver.md).
    reltol: float = 1e-6
    vntol: float = 1e-9        # absolute voltage tolerance [V]
    abstol: float = 1e-15      # absolute current tolerance [A]
    max_iter: int = 100
    gmin: float = 1e-12
    gmin_start: float = 1e-3   # first gmin of the stepping continuation
    gmin_steps: int = 10
    source_steps: int = 12
    max_dv: float = 1.0        # node-voltage update clamp [V]
    max_dv_damped: float = 0.1  # tighter clamp used by the 'damped' retry
    limiting: bool = True      # device-level junction limiting (pnjlim)
    limit_mos: bool = False    # extra MOSFET voltage limiting (usually redundant)
    integration: str = "trap"  # 'trap' | 'be'
    strategies: tuple[str, ...] = ("direct", "damped", "gmin", "source")

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "SolverOptions":
        kwargs = {}
        for f in ("reltol", "vntol", "abstol", "max_iter", "gmin", "gmin_start",
                  "gmin_steps", "source_steps", "max_dv"):
            if f in options:
                kwargs[f] = options[f]
        if "integration" in options:
            kwargs["integration"] = str(options["integration"]).lower()
        if "limiting" in options:
            kwargs["limiting"] = bool(options["limiting"])
        return cls(**kwargs)


@dataclass
class OperatingPoint:
    """Converged DC solution plus per-device operating information."""

    x: np.ndarray
    circuit: Circuit
    ctx: SimContext
    iterations: int
    strategy: str
    residual: float
    node_voltages: dict[str, float] = field(default_factory=dict)
    device_ops: dict[str, dict[str, Any]] = field(default_factory=dict)

    def v(self, node: str) -> float:
        idx = self.circuit.node(node) if isinstance(node, str) else int(node)
        return node_voltage(self.x, idx)

    def i(self, device: str) -> float:
        return self.circuit.device(device).current(self.x, self.ctx)

    def p(self, device: str) -> float:
        return self.circuit.device(device).power(self.x, self.ctx)

    def total_supply_power(self) -> float:
        """Total power delivered by all independent voltage sources [W]."""
        return sum(d.power(self.x, self.ctx) for d in self.circuit.voltage_sources())

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_voltages": dict(self.node_voltages),
            "devices": {k: dict(v) for k, v in self.device_ops.items()},
            "iterations": self.iterations,
            "strategy": self.strategy,
            "residual": self.residual,
            "total_supply_power": self.total_supply_power(),
        }


@dataclass
class ACResult:
    freqs: np.ndarray
    x: np.ndarray            # shape (n_freq, size), complex
    circuit: Circuit
    op: OperatingPoint

    def v(self, node: str) -> np.ndarray:
        idx = self.circuit.node(node) if isinstance(node, str) else int(node)
        if idx == 0:
            return np.zeros(len(self.freqs), dtype=complex)
        return self.x[:, idx - 1]

    def gain(self, out_node: str, in_node: str) -> np.ndarray:
        vin = self.v(in_node)
        vout = self.v(out_node)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(vin) > 0, vout / vin, np.nan)


@dataclass
class TransientResult:
    time: np.ndarray
    x: np.ndarray            # shape (n_t, size), real
    circuit: Circuit
    op: OperatingPoint

    def v(self, node: str) -> np.ndarray:
        idx = self.circuit.node(node) if isinstance(node, str) else int(node)
        if idx == 0:
            return np.zeros(len(self.time))
        return self.x[:, idx - 1]


# ---------------------------------------------------------------------------
# DC operating point
# ---------------------------------------------------------------------------

def _assemble(circuit: Circuit, x: np.ndarray, ctx: SimContext,
              transient: bool = False) -> MnaSystem:
    sys = MnaSystem(circuit.n_nodes, circuit.n_branches)
    for dev in circuit.devices:
        if transient:
            dev.stamp_tran(sys, x, ctx)
        else:
            dev.stamp_dc(sys, x, ctx)
    return sys


def _initial_guess(circuit: Circuit, ctx: SimContext) -> np.ndarray:
    """Seed the iteration with the DC value of every grounded voltage source.

    This costs nothing and removes most of the "everything starts at 0 V"
    convergence pathologies in supply-referenced analog circuits.
    """
    x = np.zeros(circuit.size)
    for dev in circuit.voltage_sources():
        np_, nn = dev.nodes
        val = dev.value(ctx)
        if nn == 0 and np_ != 0:
            x[np_ - 1] = val
        elif np_ == 0 and nn != 0:
            x[nn - 1] = -val
    return x


def _converged(x_new: np.ndarray, x_old: np.ndarray, residual: np.ndarray,
               n_node_rows: int, opts: SolverOptions, ref_current: float,
               ref_voltage: float) -> tuple[bool, float, float, int]:
    """SPICE-style dual convergence test.

    Returns ``(ok, max_update, max_residual, worst_row)``.  Node rows are
    tested in volts (update) and amperes (residual); branch rows are tested in
    amperes (update) and volts (residual), because MNA branch equations are
    voltage constraints.
    """
    delta = np.abs(x_new - x_old)
    tol = np.empty_like(delta)
    scale = np.maximum(np.abs(x_new), np.abs(x_old))
    tol[:n_node_rows] = opts.reltol * scale[:n_node_rows] + opts.vntol
    tol[n_node_rows:] = opts.reltol * scale[n_node_rows:] + opts.abstol
    update_ok = bool(np.all(delta <= tol))
    worst_row = int(np.argmax(delta - tol)) if delta.size else -1

    res_nodes = np.abs(residual[:n_node_rows])
    i_ok = bool(np.all(res_nodes <= opts.reltol * ref_current + opts.abstol)) \
        if res_nodes.size else True

    res_branch = np.abs(residual[n_node_rows:])
    b_ok = bool(np.all(res_branch <= opts.reltol * ref_voltage + opts.vntol)) \
        if res_branch.size else True

    max_res = float(np.max(np.abs(residual))) if residual.size else 0.0
    max_delta = float(np.max(delta)) if delta.size else 0.0
    return (update_ok and i_ok and b_ok), max_delta, max_res, worst_row


def _newton(circuit: Circuit, ctx: SimContext, opts: SolverOptions,
            x0: np.ndarray, *, max_dv: float, transient: bool = False,
            max_iter: int | None = None) -> tuple[np.ndarray, int, float]:
    """Newton-Raphson with per-node update clamping.

    ``max_dv`` bounds how far any single node voltage may move in one
    iteration.  This is the mechanism that tames the square-law model's
    cutoff region, where the companion conductance collapses to ``gmin`` and
    an unclamped solve would fling a node to ``I/gmin`` volts.  Clamping
    cannot move the fixed point (the update is zero at the solution) and stops
    engaging near convergence, so Newton's quadratic rate is preserved.

    Devices additionally see the *previous* iterate through
    ``ctx.x_prev_iter`` so that exponential junctions can apply ``pnjlim``.

    Raises :class:`ConvergenceError` if the iteration limit is reached.
    """
    x = np.asarray(x0, dtype=float).copy()
    x_prev = x.copy()
    n_node_rows = circuit.n_nodes - 1
    limit = opts.max_iter if max_iter is None else max_iter
    last_dv = float("inf")
    last_res = float("inf")
    worst_row = -1

    for iteration in range(1, limit + 1):
        ctx.x_prev_iter = x_prev if ctx.limiting else None
        sys = _assemble(circuit, x, ctx, transient=transient)
        residual = sys.G @ x - sys.b
        ref_current = float(np.max(np.abs(sys.b[:n_node_rows]))) if n_node_rows else 0.0
        ref_voltage = float(np.max(np.abs(x[:n_node_rows]))) if n_node_rows else 0.0

        x_lin = sys.solve(check_condition=False)
        if not np.all(np.isfinite(x_lin)):
            raise NumericalError(
                f"Newton iterate diverged to non-finite values at iteration {iteration}")
        dx = x_lin - x
        if max_dv > 0 and n_node_rows:
            np.clip(dx[:n_node_rows], -max_dv, max_dv, out=dx[:n_node_rows])
        x_new = x + dx

        ok, last_dv, last_res, worst_row = _converged(
            x_new, x, residual, n_node_rows, opts, ref_current, ref_voltage)
        x_prev, x = x, x_new
        if ok and iteration > 1:
            return x, iteration, last_res

    node = None
    if 0 <= worst_row < n_node_rows:
        node = circuit.node_names[worst_row + 1]
    raise ConvergenceError(
        "Newton-Raphson did not converge", iterations=limit,
        residual=last_res, max_dv=last_dv, node=node)


def _verify_residual(circuit: Circuit, ctx: SimContext, x: np.ndarray,
                     opts: SolverOptions, transient: bool = False) -> float:
    """Re-assemble with limiting disabled and return the true KCL residual."""
    saved_limiting, saved_prev = ctx.limiting, ctx.x_prev_iter
    ctx.limiting = False
    ctx.x_prev_iter = None
    try:
        sys = _assemble(circuit, x, ctx, transient=transient)
        residual = sys.G @ x - sys.b
    finally:
        ctx.limiting = saved_limiting
        ctx.x_prev_iter = saved_prev
    n_node_rows = circuit.n_nodes - 1
    return float(np.max(np.abs(residual[:n_node_rows]))) if n_node_rows else 0.0


def solve_dc(circuit: Circuit, ctx: SimContext | None = None,
             opts: SolverOptions | None = None,
             x0: np.ndarray | None = None) -> OperatingPoint:
    """Compute the DC operating point.

    Tries the configured continuation strategies in order and reports which one
    succeeded.  Raises :class:`ConvergenceError` (never a silent bad answer)
    when none of them converge.
    """
    if not circuit.finalized:
        circuit.finalize()
    opts = opts or SolverOptions.from_options(circuit.options)
    ctx = ctx or circuit.build_context()
    ctx.mode = "dc"
    ctx.limiting = opts.limiting
    ctx.limit_mos = opts.limit_mos
    ctx.n_nodes = circuit.n_nodes

    base_gmin = opts.gmin
    tried: list[str] = []
    errors: list[str] = []
    guess = _initial_guess(circuit, ctx) if x0 is None else np.asarray(x0, float).copy()

    for strategy in opts.strategies:
        tried.append(strategy)
        ctx.gmin = base_gmin
        ctx.source_scale = 1.0
        try:
            if strategy == "direct":
                x, iters, _res = _newton(circuit, ctx, opts, guess, max_dv=opts.max_dv)
            elif strategy == "damped":
                x, iters, _res = _newton(circuit, ctx, opts, guess,
                                         max_dv=opts.max_dv_damped,
                                         max_iter=opts.max_iter * 5)
            elif strategy == "gmin":
                x, iters = _gmin_stepping(circuit, ctx, opts, guess)
            elif strategy == "source":
                x, iters = _source_stepping(circuit, ctx, opts, guess)
            else:  # pragma: no cover - guarded by SolverOptions
                raise SimulationError(f"unknown continuation strategy {strategy!r}")
        except NumericalError as exc:
            # A singular matrix is a *structural* fault (floating node, shorted
            # source loop), not a convergence difficulty.  No amount of
            # continuation will fix the topology, so report it straight away
            # with its specific diagnostic instead of burying it.
            if "singular" in str(exc).lower():
                raise
            errors.append(f"{strategy}: {exc}")
            continue
        except ConvergenceError as exc:
            errors.append(f"{strategy}: {exc}")
            continue

        ctx.gmin = base_gmin
        ctx.source_scale = 1.0
        residual = _verify_residual(circuit, ctx, x, opts)
        # Final acceptance test on the *unlimited*, gmin-free system.
        ref = max(abs(d.current(x, ctx)) for d in circuit.devices
                  if not isinstance(d, Capacitor)) if circuit.devices else 0.0
        if residual > opts.reltol * max(ref, 1e-12) + 1e3 * opts.abstol:
            errors.append(
                f"{strategy}: converged iterate failed final KCL check "
                f"(residual={residual:.3e} A)")
            continue
        return _build_op(circuit, ctx, x, iters, strategy, residual)

    detail = ("; ".join(errors)[:800]) if errors else "no diagnostic available"
    raise ConvergenceError(
        f"DC operating point could not be found with any continuation strategy. "
        f"Per-strategy diagnostics: {detail}",
        iterations=opts.max_iter, residual=float("nan"), max_dv=float("nan"),
        strategies=tried,
    ) from (SimulationError("; ".join(errors)) if errors else None)


def _gmin_stepping(circuit: Circuit, ctx: SimContext, opts: SolverOptions,
                   guess: np.ndarray) -> tuple[np.ndarray, int]:
    """Ramp an artificial node-to-node conductance down to its final value."""
    x = guess.copy()
    total_iters = 0
    gmins = np.geomspace(opts.gmin_start, max(opts.gmin, 1e-14), opts.gmin_steps)
    for g in gmins:
        ctx.gmin = float(g)
        x, iters, _ = _newton(circuit, ctx, opts, x, max_dv=opts.max_dv,
                              max_iter=opts.max_iter)
        total_iters += iters
    ctx.gmin = opts.gmin
    x, iters, _ = _newton(circuit, ctx, opts, x, max_dv=opts.max_dv)
    return x, total_iters + iters


def _source_stepping(circuit: Circuit, ctx: SimContext, opts: SolverOptions,
                     guess: np.ndarray) -> tuple[np.ndarray, int]:
    """Ramp all independent sources from 0 to full value."""
    x = np.zeros_like(guess)
    total_iters = 0
    for scale in np.linspace(1.0 / opts.source_steps, 1.0, opts.source_steps):
        ctx.source_scale = float(scale)
        x, iters, _ = _newton(circuit, ctx, opts, x, max_dv=opts.max_dv,
                              max_iter=opts.max_iter)
        total_iters += iters
    ctx.source_scale = 1.0
    return x, total_iters


def _build_op(circuit: Circuit, ctx: SimContext, x: np.ndarray, iters: int,
              strategy: str, residual: float) -> OperatingPoint:
    saved_limiting, saved_prev = ctx.limiting, ctx.x_prev_iter
    ctx.limiting = False
    ctx.x_prev_iter = None
    try:
        node_voltages = {name: node_voltage(x, i)
                         for i, name in enumerate(circuit.node_names)}
        device_ops = {d.name: d.op_info(x, ctx) for d in circuit.devices}
    finally:
        ctx.limiting = saved_limiting
        ctx.x_prev_iter = saved_prev
    return OperatingPoint(x=x, circuit=circuit, ctx=ctx, iterations=iters,
                          strategy=strategy, residual=residual,
                          node_voltages=node_voltages, device_ops=device_ops)


# ---------------------------------------------------------------------------
# AC small-signal analysis
# ---------------------------------------------------------------------------

def ac_analysis(circuit: Circuit, freqs: Sequence[float],
                op: OperatingPoint | None = None,
                ctx: SimContext | None = None,
                opts: SolverOptions | None = None) -> ACResult:
    """Linearise about the DC operating point and solve at each frequency."""
    if not circuit.finalized:
        circuit.finalize()
    opts = opts or SolverOptions.from_options(circuit.options)
    if op is None:
        ctx = ctx or circuit.build_context()
        op = solve_dc(circuit, ctx, opts)
    ctx = op.ctx
    freqs = np.asarray(list(freqs), dtype=float)
    if freqs.size == 0:
        raise SimulationError("AC analysis requires at least one frequency point")
    if np.any(freqs <= 0):
        raise SimulationError("AC analysis frequencies must be strictly positive")

    if not any(getattr(d, "ac_mag", 0.0) for d in circuit.devices):
        raise SimulationError(
            "AC analysis requested but no source declares an AC magnitude; "
            "add 'AC 1' to the stimulus source")

    saved_limiting, saved_prev = ctx.limiting, ctx.x_prev_iter
    ctx.limiting = False
    ctx.x_prev_iter = None
    out = np.zeros((freqs.size, circuit.size), dtype=complex)
    try:
        for k, f in enumerate(freqs):
            omega = 2.0 * math.pi * float(f)
            sys = MnaSystem(circuit.n_nodes, circuit.n_branches, dtype=complex)
            for dev in circuit.devices:
                dev.stamp_ac(sys, op.x, omega, ctx)
            out[k, :] = sys.solve(check_condition=False)
    finally:
        ctx.limiting = saved_limiting
        ctx.x_prev_iter = saved_prev
    return ACResult(freqs=freqs, x=out, circuit=circuit, op=op)


# ---------------------------------------------------------------------------
# Transient analysis
# ---------------------------------------------------------------------------

def transient_analysis(circuit: Circuit, tstop: float, tstep: float,
                       tstart: float = 0.0,
                       op: OperatingPoint | None = None,
                       ctx: SimContext | None = None,
                       opts: SolverOptions | None = None) -> TransientResult:
    """Fixed-step transient analysis (trapezoidal, backward-Euler first step).

    The initial state is the DC operating point unless one is supplied.  The
    step size is fixed: there is no local-truncation-error control, so *tstep*
    must be chosen small enough for the dynamics of interest.  This limitation
    is documented in ``docs/limitations.md``.
    """
    if not circuit.finalized:
        circuit.finalize()
    if tstep <= 0 or tstop <= tstart:
        raise SimulationError(
            f"invalid transient window: tstart={tstart}, tstop={tstop}, tstep={tstep}")
    opts = opts or SolverOptions.from_options(circuit.options)
    if op is None:
        ctx = ctx or circuit.build_context()
        op = solve_dc(circuit, ctx, opts)
    ctx = op.ctx
    ctx.mode = "tran"
    ctx.integration = "be"        # first step: backward Euler
    ctx.n_nodes = circuit.n_nodes

    n_steps = int(math.ceil((tstop - tstart) / tstep))
    if n_steps > 2_000_000:
        raise SimulationError(
            f"transient analysis would need {n_steps} steps; increase tstep")

    times = np.empty(n_steps + 1)
    xs = np.empty((n_steps + 1, circuit.size))
    x = op.x.copy()
    times[0] = tstart
    xs[0, :] = x

    # Seed reactive-element history from the DC solution.
    ctx.dt = tstep
    ctx.time = tstart
    ctx.limiting = False
    ctx.x_prev_iter = None
    for dev in circuit.devices:
        dev.init_state(x, ctx)
    ctx.limiting = opts.limiting

    for step in range(1, n_steps + 1):
        t = tstart + step * tstep
        ctx.time = min(t, tstop)
        ctx.dt = tstep
        try:
            x, _iters, _res = _newton(circuit, ctx, opts, x,
                                      max_dv=opts.max_dv, transient=True,
                                      max_iter=opts.max_iter * 2)
        except (ConvergenceError, NumericalError) as exc:
            raise ConvergenceError(
                f"transient analysis failed at t={ctx.time:.6g} s: {exc}",
                iterations=opts.max_iter * 2, residual=float("nan"),
                max_dv=float("nan"),
            ) from exc
        for dev in circuit.devices:
            dev.update_state(x, ctx)
        times[step] = ctx.time
        xs[step, :] = x
        ctx.integration = opts.integration  # switch to trapezoidal after step 1

    ctx.mode = "dc"
    ctx.dt = 0.0
    return TransientResult(time=times, x=xs, circuit=circuit, op=op)
