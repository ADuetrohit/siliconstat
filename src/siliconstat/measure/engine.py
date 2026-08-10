"""Measurement extraction.

A measurement turns a *simulation* into a *number an engineer specifies on*.
The engine lazily runs whichever analysis each declared measurement needs
(operating point, AC sweep, transient) and caches it, so a circuit that
declares five DC measurements and one gain measurement performs exactly one DC
solve and one AC solve.

Every measurement either produces a finite value or reports *why* it could not
be computed.  Nothing is silently replaced by a plausible-looking number: an
op-amp whose gain never crosses unity yields ``NaN`` plus the reason
``"no unity-gain crossing in 1 Hz .. 10 GHz"``, and the Monte Carlo layer
counts that sample as an invalid measurement rather than a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..core.circuit import AnalysisSpec, Circuit, MeasureSpec
from ..core.context import SimContext
from ..core.devices import VoltageSource
from ..core.exceptions import (
    ConvergenceError,
    MeasurementError,
    NumericalError,
    SimulationError,
)
from ..core.expr import ExpressionError, safe_eval
from ..core.solver import (
    ACResult,
    OperatingPoint,
    SolverOptions,
    TransientResult,
    ac_analysis,
    solve_dc,
    transient_analysis,
)

__all__ = [
    "MeasurementContext", "MeasurementResult", "MeasurementOutcome",
    "evaluate_measurements",
]

DEFAULT_AC_SWEEP = {"sweep": "dec", "points": 10, "fstart": 1.0, "fstop": 1e10}
DEFAULT_OFFSET_DELTA = 1e-4   # differential probe used for input-referred offset [V]

#: The "-3 dB" bandwidth is the *half-power* point, where the magnitude falls
#: to 1/sqrt(2) of its low-frequency value.  That is 10*log10(2) = 3.0103 dB,
#: not 3.000 dB -- the difference is 0.24 % of the bandwidth of a single-pole
#: response, which is large enough to matter when bandwidth carries a
#: specification.
HALF_POWER_DB = 10.0 * math.log10(2.0)


@dataclass
class MeasurementOutcome:
    """Result of one measurement."""

    name: str
    value: float
    unit: str = ""
    kind: str = ""
    ok: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit,
                "kind": self.kind, "ok": self.ok, "reason": self.reason}


@dataclass
class MeasurementResult:
    """All measurements for one simulated circuit instance."""

    values: dict[str, float] = field(default_factory=dict)
    outcomes: dict[str, MeasurementOutcome] = field(default_factory=dict)
    analyses_run: list[str] = field(default_factory=list)
    dc_iterations: int = 0
    dc_strategy: str = ""

    @property
    def all_valid(self) -> bool:
        return all(o.ok for o in self.outcomes.values())

    def invalid_reasons(self) -> dict[str, str]:
        return {n: o.reason for n, o in self.outcomes.items() if not o.ok}

    def to_dict(self) -> dict[str, Any]:
        return {"values": dict(self.values),
                "outcomes": {k: v.to_dict() for k, v in self.outcomes.items()},
                "analyses_run": list(self.analyses_run),
                "dc_iterations": self.dc_iterations,
                "dc_strategy": self.dc_strategy}


def _log_sweep(spec: dict[str, Any]) -> np.ndarray:
    fstart, fstop = float(spec["fstart"]), float(spec["fstop"])
    points = int(spec["points"])
    if spec.get("sweep", "dec") == "lin":
        n = max(points, 2)
        return np.linspace(fstart, fstop, n)
    if spec.get("sweep") == "oct":
        n_oct = max(math.log2(fstop / fstart), 1e-9)
        return np.logspace(math.log10(fstart), math.log10(fstop),
                           int(points * n_oct) + 1)
    n_dec = max(math.log10(fstop / fstart), 1e-9)
    return np.logspace(math.log10(fstart), math.log10(fstop),
                       int(points * n_dec) + 1)


class MeasurementContext:
    """Lazily runs and caches the analyses that measurements depend on."""

    def __init__(self, circuit: Circuit, ctx: SimContext | None = None,
                 opts: SolverOptions | None = None,
                 op: OperatingPoint | None = None) -> None:
        self.circuit = circuit
        self.opts = opts or SolverOptions.from_options(circuit.options)
        self.ctx = ctx if ctx is not None else (op.ctx if op else circuit.build_context())
        self._op: OperatingPoint | None = op
        self._ac_sweep: ACResult | None = None
        self._ac_points: dict[float, ACResult] = {}
        self._tran: TransientResult | None = None
        self.analyses_run: list[str] = ["op (supplied)"] if op is not None else []

    # -- analyses ----------------------------------------------------------
    def op(self) -> OperatingPoint:
        if self._op is None:
            self._op = solve_dc(self.circuit, self.ctx, self.opts)
            self.analyses_run.append("op")
        return self._op

    def resimulate(self, source_overrides: dict[str, float]) -> OperatingPoint:
        """Solve a perturbed DC point (used by the offset measurement).

        The perturbed solve is warm-started from the nominal operating point,
        which keeps the cost of an offset measurement at roughly one extra
        Newton iteration rather than a full solve.
        """
        base = self.op()
        ctx = self.circuit.build_context(
            temp_c=self.ctx.temp_c,
            source_overrides={**self.ctx.source_overrides, **source_overrides},
        )
        ctx.mos_models = self.ctx.mos_models
        ctx.diode_models = self.ctx.diode_models
        return solve_dc(self.circuit, ctx, self.opts, x0=base.x)

    def _ac_spec(self) -> dict[str, Any]:
        for a in self.circuit.analyses:
            if a.kind == "ac":
                return a.args
        return DEFAULT_AC_SWEEP

    def ac_sweep(self) -> ACResult:
        if self._ac_sweep is None:
            freqs = _log_sweep(self._ac_spec())
            self._ac_sweep = ac_analysis(self.circuit, freqs, op=self.op(),
                                         opts=self.opts)
            self.analyses_run.append("ac")
        return self._ac_sweep

    def ac_at(self, freq: float) -> ACResult:
        key = float(freq)
        if key not in self._ac_points:
            self._ac_points[key] = ac_analysis(self.circuit, [key], op=self.op(),
                                               opts=self.opts)
            self.analyses_run.append(f"ac@{key:g}")
        return self._ac_points[key]

    def tran(self) -> TransientResult:
        if self._tran is None:
            spec: AnalysisSpec | None = None
            for a in self.circuit.analyses:
                if a.kind == "tran":
                    spec = a
                    break
            if spec is None:
                raise MeasurementError(
                    "a transient measurement was requested but the netlist has no "
                    "'.tran <tstep> <tstop>' analysis card")
            self._tran = transient_analysis(
                self.circuit, tstop=spec.args["tstop"], tstep=spec.args["tstep"],
                tstart=spec.args.get("tstart", 0.0), op=self.op(), opts=self.opts)
            self.analyses_run.append("tran")
        return self._tran


# ---------------------------------------------------------------------------
# Individual measurement kinds
# ---------------------------------------------------------------------------

def _m_voltage(mc: MeasurementContext, spec: MeasureSpec) -> float:
    op = mc.op()
    value = op.v(spec.args["node"])
    if "node_neg" in spec.args:
        value -= op.v(spec.args["node_neg"])
    return value


def _m_current(mc: MeasurementContext, spec: MeasureSpec) -> float:
    return mc.op().i(spec.args["device"])


def _m_power(mc: MeasurementContext, spec: MeasureSpec) -> float:
    op = mc.op()
    if spec.args.get("total"):
        return op.total_supply_power()
    return op.p(spec.args["device"])


def _transfer(mc: MeasurementContext, spec: MeasureSpec,
              result: ACResult) -> np.ndarray:
    out_node = spec.args["out"]
    in_node = spec.args["in"]
    vin = result.v(in_node)
    vout = result.v(out_node)
    if np.all(np.abs(vin) == 0):
        raise MeasurementError(
            f"measurement {spec.name!r}: the AC input node {in_node!r} has zero "
            "amplitude -- check that the stimulus source declares 'AC 1'")
    return vout / vin


def _m_gain(mc: MeasurementContext, spec: MeasureSpec) -> float:
    freq = float(spec.args.get("freq", 1.0))
    result = mc.ac_at(freq)
    a = _transfer(mc, spec, result)[0]
    mag = abs(a)
    if mag == 0.0:
        raise MeasurementError(
            f"measurement {spec.name!r}: transfer function is exactly zero at {freq:g} Hz")
    if str(spec.args.get("units", "db")).lower() in ("lin", "linear", "v/v"):
        return float(mag)
    return float(20.0 * math.log10(mag))


def _bracket_crossing(x: np.ndarray, y: np.ndarray, target: float
                      ) -> tuple[float, float] | None:
    """First interval of *x* in which ``y`` crosses *target*."""
    for i in range(1, len(y)):
        y0, y1 = y[i - 1], y[i]
        if (y0 - target) == 0.0:
            return float(x[i - 1]), float(x[i - 1])
        if (y0 - target) * (y1 - target) < 0.0:
            return float(x[i - 1]), float(x[i])
    return None


def _interp_crossing(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """First crossing of ``y == target``, log-interpolated in ``x``."""
    bracket = _bracket_crossing(x, y, target)
    if bracket is None:
        return None
    f_lo, f_hi = bracket
    if f_lo == f_hi:
        return f_lo
    i = int(np.searchsorted(x, f_hi))
    y0, y1 = y[i - 1], y[i]
    frac = (target - y0) / (y1 - y0)
    lx0, lx1 = math.log10(f_lo), math.log10(f_hi)
    return float(10.0 ** (lx0 + frac * (lx1 - lx0)))


def _refine_crossing(mc: "MeasurementContext", spec: MeasureSpec,
                     f_lo: float, f_hi: float, target: float, *,
                     tol_rel: float = 1e-7, max_iter: int = 60) -> float:
    """Locate a magnitude crossing precisely, using fresh AC solves.

    Interpolating between swept grid points is only as good as the grid: at
    30 points per decade, adjacent samples are 8 % apart and linear
    interpolation of a curved response lands roughly 0.2 % off.  Bandwidth and
    unity-gain frequency are specification quantities that feed yield, so the
    bracket found on the grid is refined by bisection on the *actual* transfer
    function.  Each step costs one single-frequency AC solve.
    """
    if f_lo >= f_hi:
        return f_lo

    def excess(freq: float) -> float:
        a = _transfer(mc, spec, mc.ac_at(freq))[0]
        return 20.0 * math.log10(max(abs(a), 1e-300)) - target

    lo, hi = f_lo, f_hi
    e_lo = excess(lo)
    e_hi = excess(hi)
    if e_lo == 0.0:
        return lo
    if e_lo * e_hi > 0.0:      # the grid bracket did not survive re-evaluation
        return _log_midpoint(f_lo, f_hi)
    for _ in range(max_iter):
        mid = _log_midpoint(lo, hi)
        e_mid = excess(mid)
        if e_mid == 0.0 or (hi - lo) <= tol_rel * mid:
            return mid
        if e_lo * e_mid < 0.0:
            hi, e_hi = mid, e_mid
        else:
            lo, e_lo = mid, e_mid
    return _log_midpoint(lo, hi)


def _log_midpoint(a: float, b: float) -> float:
    return float(10.0 ** (0.5 * (math.log10(a) + math.log10(b))))


def _m_bandwidth(mc: MeasurementContext, spec: MeasureSpec) -> float:
    result = mc.ac_sweep()
    a = _transfer(mc, spec, result)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(a), 1e-300))
    f = result.freqs
    dc_gain = mag_db[0]
    target = dc_gain - HALF_POWER_DB
    bracket = _bracket_crossing(f, mag_db, target)
    if bracket is None:
        raise MeasurementError(
            f"measurement {spec.name!r}: the response never falls 3 dB below its "
            f"low-frequency value ({dc_gain:.2f} dB) within "
            f"{f[0]:g} Hz .. {f[-1]:g} Hz")
    return _refine_crossing(mc, spec, bracket[0], bracket[1], target)


def _unity_gain_frequency(mc: MeasurementContext, spec: MeasureSpec) -> float:
    result = mc.ac_sweep()
    a = _transfer(mc, spec, result)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(a), 1e-300))
    bracket = _bracket_crossing(result.freqs, mag_db, 0.0)
    if bracket is None:
        raise MeasurementError(
            f"measurement {spec.name!r}: no unity-gain crossing within "
            f"{result.freqs[0]:g} Hz .. {result.freqs[-1]:g} Hz "
            f"(gain range {mag_db.min():.1f} .. {mag_db.max():.1f} dB)")
    return _refine_crossing(mc, spec, bracket[0], bracket[1], 0.0)


def _m_ugf(mc: MeasurementContext, spec: MeasureSpec) -> float:
    return _unity_gain_frequency(mc, spec)


def _phase_deg(a: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.angle(a)))


def _m_phase_margin(mc: MeasurementContext, spec: MeasureSpec) -> float:
    result = mc.ac_sweep()
    a = _transfer(mc, spec, result)
    fu = _unity_gain_frequency(mc, spec)
    # The phase is read from the *swept* response because unwrapping needs the
    # whole curve; only the crossing frequency itself is refined.
    phase = _phase_deg(a)
    lf = np.log10(result.freqs)
    ph_u = float(np.interp(math.log10(fu), lf, phase))
    # Inverting amplifiers start at -180 deg; normalise to the loop convention.
    if phase[0] < -90.0:
        ph_u += 180.0
    return 180.0 + ph_u


def _m_gain_margin(mc: MeasurementContext, spec: MeasureSpec) -> float:
    result = mc.ac_sweep()
    a = _transfer(mc, spec, result)
    phase = _phase_deg(a)
    if phase[0] < -90.0:
        phase = phase + 180.0
    mag_db = 20.0 * np.log10(np.maximum(np.abs(a), 1e-300))
    f180 = _interp_crossing(result.freqs, phase, -180.0)
    if f180 is None:
        raise MeasurementError(
            f"measurement {spec.name!r}: phase never reaches -180 deg, so gain "
            "margin is undefined (the amplifier is unconditionally stable in "
            "the swept range)")
    lf = np.log10(result.freqs)
    return float(-np.interp(math.log10(f180), lf, mag_db))


def _m_offset(mc: MeasurementContext, spec: MeasureSpec) -> float:
    """Input-referred offset voltage.

    Two real DC solves, no curve fitting:

    1. nominal solve gives the output imbalance ``Vod0``;
    2. a small differential probe ``delta`` on the two input sources gives the
       differential gain ``Ad = dVod/dVid``;
    3. ``Vos = -Vod0 / Ad`` -- the input voltage that would null the output.
    """
    circuit = mc.circuit
    srcp = circuit.device(str(spec.args["srcp"]))
    srcn = circuit.device(str(spec.args["srcn"]))
    if not isinstance(srcp, VoltageSource) or not isinstance(srcn, VoltageSource):
        raise MeasurementError(
            f"measurement {spec.name!r}: srcp/srcn must name voltage sources")
    outp = str(spec.args["outp"])
    outn = str(spec.args.get("outn", "0"))
    delta = float(spec.args.get("delta", DEFAULT_OFFSET_DELTA))
    if delta <= 0:
        raise MeasurementError(
            f"measurement {spec.name!r}: probe delta must be > 0 (got {delta})")

    op0 = mc.op()
    vod0 = op0.v(outp) - (0.0 if outn == "0" else op0.v(outn))

    base_p = mc.ctx.source_overrides.get(srcp.name, srcp.dc)
    base_n = mc.ctx.source_overrides.get(srcn.name, srcn.dc)
    op1 = mc.resimulate({srcp.name: base_p + delta / 2.0,
                         srcn.name: base_n - delta / 2.0})
    vod1 = op1.v(outp) - (0.0 if outn == "0" else op1.v(outn))

    ad = (vod1 - vod0) / delta
    if not math.isfinite(ad) or abs(ad) < 1e-9:
        raise MeasurementError(
            f"measurement {spec.name!r}: differential gain is ~0 ({ad:.3e} V/V); "
            "the input pair is probably not biased in saturation, so the "
            "input-referred offset is undefined")
    return float(-vod0 / ad)


# -- transient measurements -------------------------------------------------

def _tran_window(mc: MeasurementContext, spec: MeasureSpec
                 ) -> tuple[np.ndarray, np.ndarray]:
    result = mc.tran()
    t = result.time
    v = result.v(str(spec.args["node"]))
    t0 = float(spec.args.get("tstart", t[0]))
    t1 = float(spec.args.get("tstop", t[-1]))
    mask = (t >= t0) & (t <= t1)
    if mask.sum() < 3:
        raise MeasurementError(
            f"measurement {spec.name!r}: transient window [{t0:g}, {t1:g}] s "
            f"contains only {int(mask.sum())} points")
    return t[mask], v[mask]


def _edge_time(t: np.ndarray, v: np.ndarray, level: float, rising: bool) -> float | None:
    for i in range(1, len(v)):
        v0, v1 = v[i - 1], v[i]
        if rising and v0 < level <= v1:
            frac = (level - v0) / (v1 - v0) if v1 != v0 else 0.0
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
        if not rising and v0 > level >= v1:
            frac = (v0 - level) / (v0 - v1) if v1 != v0 else 0.0
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return None


def _m_risetime(mc: MeasurementContext, spec: MeasureSpec) -> float:
    t, v = _tran_window(mc, spec)
    lo_pct = float(spec.args.get("lo", 10.0))
    hi_pct = float(spec.args.get("hi", 90.0))
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin <= 0:
        raise MeasurementError(f"measurement {spec.name!r}: the signal does not move")
    t_lo = _edge_time(t, v, vmin + lo_pct / 100.0 * (vmax - vmin), rising=True)
    t_hi = _edge_time(t, v, vmin + hi_pct / 100.0 * (vmax - vmin), rising=True)
    if t_lo is None or t_hi is None or t_hi <= t_lo:
        raise MeasurementError(
            f"measurement {spec.name!r}: no monotonic rising edge crossing "
            f"{lo_pct:g}% and {hi_pct:g}% within the transient window")
    return t_hi - t_lo


def _m_falltime(mc: MeasurementContext, spec: MeasureSpec) -> float:
    t, v = _tran_window(mc, spec)
    lo_pct = float(spec.args.get("lo", 10.0))
    hi_pct = float(spec.args.get("hi", 90.0))
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin <= 0:
        raise MeasurementError(f"measurement {spec.name!r}: the signal does not move")
    t_hi = _edge_time(t, v, vmin + hi_pct / 100.0 * (vmax - vmin), rising=False)
    t_lo = _edge_time(t, v, vmin + lo_pct / 100.0 * (vmax - vmin), rising=False)
    if t_lo is None or t_hi is None or t_lo <= t_hi:
        raise MeasurementError(
            f"measurement {spec.name!r}: no falling edge crossing "
            f"{hi_pct:g}% and {lo_pct:g}% within the transient window")
    return t_lo - t_hi


def _m_overshoot(mc: MeasurementContext, spec: MeasureSpec) -> float:
    t, v = _tran_window(mc, spec)
    v_init, v_final = float(v[0]), float(v[-1])
    swing = v_final - v_init
    if abs(swing) < 1e-12:
        raise MeasurementError(
            f"measurement {spec.name!r}: initial and final levels are equal, so "
            "overshoot is undefined")
    peak = float(v.max()) if swing > 0 else float(v.min())
    return float((peak - v_final) / swing * 100.0)


def _m_settling(mc: MeasurementContext, spec: MeasureSpec) -> float:
    t, v = _tran_window(mc, spec)
    tol_pct = float(spec.args.get("tolerance", 1.0))
    v_init, v_final = float(v[0]), float(v[-1])
    band = abs(v_final - v_init) * tol_pct / 100.0
    if band <= 0:
        band = abs(v_final) * tol_pct / 100.0
    if band <= 0:
        raise MeasurementError(
            f"measurement {spec.name!r}: settling band is zero, so settling time "
            "is undefined")
    outside = np.where(np.abs(v - v_final) > band)[0]
    if outside.size == 0:
        return 0.0
    last = int(outside[-1])
    if last + 1 >= len(t):
        raise MeasurementError(
            f"measurement {spec.name!r}: the signal never settles to within "
            f"{tol_pct:g}% of its final value before the end of the transient window")
    return float(t[last + 1] - t[0])


def _m_vpp(mc: MeasurementContext, spec: MeasureSpec) -> float:
    _t, v = _tran_window(mc, spec)
    return float(v.max() - v.min())


def _m_vmax(mc: MeasurementContext, spec: MeasureSpec) -> float:
    return float(_tran_window(mc, spec)[1].max())


def _m_vmin(mc: MeasurementContext, spec: MeasureSpec) -> float:
    return float(_tran_window(mc, spec)[1].min())


def _m_vavg(mc: MeasurementContext, spec: MeasureSpec) -> float:
    t, v = _tran_window(mc, spec)
    return float(np.trapezoid(v, t) / (t[-1] - t[0]))


EVALUATORS: dict[str, Callable[[MeasurementContext, MeasureSpec], float]] = {
    "v": _m_voltage, "i": _m_current, "p": _m_power,
    "gain": _m_gain, "bw": _m_bandwidth, "ugf": _m_ugf,
    "pm": _m_phase_margin, "gm": _m_gain_margin, "vos": _m_offset,
    "risetime": _m_risetime, "falltime": _m_falltime,
    "overshoot": _m_overshoot, "settling": _m_settling,
    "vpp": _m_vpp, "vmax": _m_vmax, "vmin": _m_vmin, "vavg": _m_vavg,
}


def evaluate_measurements(circuit: Circuit, ctx: SimContext | None = None,
                          opts: SolverOptions | None = None,
                          measures: Sequence[MeasureSpec] | None = None,
                          op: OperatingPoint | None = None,
                          ) -> MeasurementResult:
    """Evaluate every declared measurement for one circuit instance.

    ``EXPR`` measurements are evaluated last, in declaration order, and may
    reference any measurement declared before them.  Pass a pre-computed *op*
    to reuse an operating point the caller already solved.
    """
    mc = MeasurementContext(circuit, ctx, opts, op=op)
    specs = list(measures if measures is not None else circuit.measures)
    result = MeasurementResult()

    direct = [s for s in specs if s.kind != "expr"]
    derived = [s for s in specs if s.kind == "expr"]

    for spec in direct:
        evaluator = EVALUATORS.get(spec.kind)
        if evaluator is None:
            result.outcomes[spec.name] = MeasurementOutcome(
                spec.name, float("nan"), spec.unit, spec.kind, ok=False,
                reason=f"no evaluator registered for measurement kind {spec.kind!r}")
            result.values[spec.name] = float("nan")
            continue
        try:
            value = float(evaluator(mc, spec))
            ok, reason = math.isfinite(value), ""
            if not ok:
                reason = "measurement evaluated to a non-finite value"
        except (MeasurementError, SimulationError, ConvergenceError,
                NumericalError, ValueError, ZeroDivisionError) as exc:
            value, ok, reason = float("nan"), False, str(exc)
        result.values[spec.name] = value
        result.outcomes[spec.name] = MeasurementOutcome(
            spec.name, value, spec.unit, spec.kind, ok=ok, reason=reason)

    for spec in derived:
        try:
            value = float(safe_eval(spec.args["expression"], result.values))
            ok = math.isfinite(value)
            reason = "" if ok else "expression evaluated to a non-finite value"
        except (ExpressionError, ValueError, ZeroDivisionError, KeyError) as exc:
            value, ok, reason = float("nan"), False, str(exc)
        result.values[spec.name] = value
        result.outcomes[spec.name] = MeasurementOutcome(
            spec.name, value, spec.unit, spec.kind, ok=ok, reason=reason)

    result.analyses_run = list(mc.analyses_run)
    if mc._op is not None:
        result.dc_iterations = mc._op.iterations
        result.dc_strategy = mc._op.strategy
    return result
