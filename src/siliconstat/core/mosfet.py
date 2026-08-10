"""Layer 0 -- MOSFET circuit physics.

Square-law (SPICE level-1 style) large-signal model with:

* three operating regions (cutoff / triode / saturation),
* channel-length modulation,
* body effect (bulk-referenced threshold shift),
* analytic small-signal derivatives ``gm``, ``gds``, ``gmb`` required by
  Newton-Raphson,
* Meyer-style region-dependent gate capacitances for AC / transient.

Everything in this module works in the **n-type frame**: voltages are
pre-multiplied by the device sign (+1 NMOS, -1 PMOS) and drain/source are
swapped so that ``vds >= 0``.  The caller undoes the transformation when
stamping, which is why the same conductance stamps work for both polarities
(the two sign flips cancel: ``d(sign*I)/dv = sign*gm*sign = gm``).

Model equations
---------------
::

    vth = VTO + GAMMA * (sqrt(PHI - vbs) - sqrt(PHI))
    vov = vgs - vth

    cutoff      (vov <= 0)   Ids = 0
    triode      (vds < vov)  Ids = beta * (vov*vds - vds^2/2) * (1 + LAMBDA*vds)
    saturation  (vds >= vov) Ids = (beta/2) * vov^2 * (1 + LAMBDA*vds)

    beta = KP * Weff / Leff

Applying ``(1 + LAMBDA*vds)`` in *both* regions (as Berkeley SPICE level 1
does) keeps ``Ids`` and its derivatives continuous across ``vds = vov``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import MosfetModel

__all__ = ["MosOp", "eval_mos_core", "eval_mosfet", "fet_limit"]


@dataclass(slots=True)
class MosOp:
    """Large- and small-signal operating point of one MOSFET instance."""

    ids: float          # drain current in the n-type frame [A] (>= 0)
    gm: float           # d Ids / d vgs [S]
    gds: float          # d Ids / d vds [S]
    gmb: float          # d Ids / d vbs [S]
    vth: float          # effective threshold including body effect [V]
    vgs: float          # n-type-frame gate-source voltage [V]
    vds: float          # n-type-frame drain-source voltage [V] (>= 0)
    vbs: float          # n-type-frame bulk-source voltage [V]
    vov: float          # overdrive vgs - vth [V]
    region: str         # 'cutoff' | 'triode' | 'saturation'
    swapped: bool       # True if drain/source were exchanged (reverse mode)
    beta: float         # KP * Weff/Leff [A/V^2]
    cgs: float = 0.0    # [F]
    cgd: float = 0.0    # [F]
    cgb: float = 0.0    # [F]
    cdb: float = 0.0    # [F]
    csb: float = 0.0    # [F]

    @property
    def gm_over_id(self) -> float:
        return self.gm / self.ids if self.ids > 0 else float("inf")

    @property
    def ro(self) -> float:
        return 1.0 / self.gds if self.gds > 0 else float("inf")


def eval_mos_core(vgs: float, vds: float, vbs: float, *, vth0: float, beta: float,
                  lambda_: float, gamma: float, phi: float) -> tuple[float, float, float, float, float, str]:
    """Core square-law evaluation in the n-type frame with ``vds >= 0``.

    Returns ``(ids, gm, gds, gmb, vth, region)``.
    """
    # Body effect.  vbs <= 0 for a reverse-biased bulk junction, so the
    # argument phi - vbs >= phi > 0.  Clamp for forward-biased bulk.
    arg = phi - vbs
    if arg < 1e-6:
        arg = 1e-6
    sqrt_arg = math.sqrt(arg)
    vth = vth0 + gamma * (sqrt_arg - math.sqrt(phi))

    vov = vgs - vth
    if vov <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, vth, "cutoff"

    clm = 1.0 + lambda_ * vds
    if clm < 1e-3:  # guard against pathological lambda*vds
        clm = 1e-3

    if vds < vov:  # linear / triode
        charge = vov * vds - 0.5 * vds * vds
        ids = beta * charge * clm
        gm = beta * vds * clm
        gds = beta * ((vov - vds) * clm + lambda_ * charge)
        region = "triode"
    else:  # saturation
        ids = 0.5 * beta * vov * vov * clm
        gm = beta * vov * clm
        gds = 0.5 * beta * vov * vov * lambda_
        region = "saturation"

    # gmb = dIds/dvbs = (dIds/dvov) * (-dvth/dvbs) = gm * gamma / (2*sqrt(phi-vbs))
    gmb = gm * gamma / (2.0 * sqrt_arg) if gamma > 0.0 else 0.0
    return ids, gm, gds, gmb, vth, region


def _meyer_caps(region: str, w_eff: float, l_eff: float, model: MosfetModel
                ) -> tuple[float, float, float]:
    """Region-dependent Meyer gate capacitances ``(cgs, cgd, cgb)`` in farads."""
    cox_total = model.cox_area * w_eff * l_eff
    ov_s = model.cgso * w_eff
    ov_d = model.cgdo * w_eff
    ov_b = model.cgbo * l_eff
    if region == "cutoff":
        return ov_s, ov_d, cox_total + ov_b
    if region == "triode":
        return 0.5 * cox_total + ov_s, 0.5 * cox_total + ov_d, ov_b
    # saturation
    return (2.0 / 3.0) * cox_total + ov_s, ov_d, ov_b


def eval_mosfet(model: MosfetModel, w: float, length: float,
                v_d: float, v_g: float, v_s: float, v_b: float, *,
                vth0: float | None = None, kp: float | None = None,
                multiplicity: float = 1.0, with_caps: bool = False) -> MosOp:
    """Evaluate one MOSFET instance from its *terminal* voltages.

    Parameters
    ----------
    model:
        Process card supplying KP, LAMBDA, GAMMA, PHI, ...
    w, length:
        Drawn device geometry in metres.
    v_d, v_g, v_s, v_b:
        Real (not sign-adjusted) terminal voltages in volts.
    vth0, kp:
        Per-instance overrides of the threshold magnitude and transconductance
        parameter.  This is the hook the mismatch layer uses: each Monte Carlo
        sample supplies its own perturbed values.
    multiplicity:
        Number of parallel fingers (``m=`` on the instance line).

    Returns
    -------
    MosOp
        Operating point in the n-type frame.  ``swapped`` reports whether the
        physical drain and source were exchanged; the caller must apply the
        same exchange when stamping.
    """
    sign = model.sign
    vth0_eff = model.vto if vth0 is None else vth0
    kp_eff = model.kp if kp is None else kp

    w_eff = max(w - 2.0 * model.wd, 1e-12)
    l_eff = max(length - 2.0 * model.ld, 1e-12)
    beta = kp_eff * (w_eff / l_eff) * multiplicity

    vgs = sign * (v_g - v_s)
    vds = sign * (v_d - v_s)
    vbs = sign * (v_b - v_s)

    swapped = vds < 0.0
    if swapped:
        # Reverse mode: physical source acts as drain.  Re-reference to the
        # true (lower-potential) source terminal.
        vgs = vgs - vds   # = vgd
        vbs = vbs - vds   # = vbd
        vds = -vds

    ids, gm, gds, gmb, vth, region = eval_mos_core(
        vgs, vds, vbs, vth0=vth0_eff, beta=beta,
        lambda_=model.lambda_, gamma=model.gamma, phi=model.phi,
    )

    op = MosOp(ids=ids, gm=gm, gds=gds, gmb=gmb, vth=vth, vgs=vgs, vds=vds,
               vbs=vbs, vov=vgs - vth, region=region, swapped=swapped, beta=beta)

    if with_caps:
        cgs, cgd, cgb = _meyer_caps(region, w_eff, l_eff, model)
        op.cgs = cgs * multiplicity
        op.cgd = cgd * multiplicity
        op.cgb = cgb * multiplicity
        op.cdb = model.cjd * w_eff * multiplicity
        op.csb = model.cjs * w_eff * multiplicity
    return op


def fet_limit(v_new: float, v_old: float, vth: float, *,
              near_step: float = 0.5, far_step: float = 2.0) -> float:
    """Bounded-step voltage limiting for MOSFET gate-source voltages.

    The square-law model has a curvature kink at ``vgs = vth``; an unlimited
    Newton step can jump across it and oscillate.  This limiter bounds the
    per-iteration change, tightening the bound near threshold.

    It is *monotone and step-bounded* -- deliberately simpler than Berkeley
    SPICE's ``FETLIM`` heuristic.  Because the bound only engages while the
    step is large, the quadratic convergence of Newton's method near the
    solution is preserved (see ``tests/test_solver.py``).
    """
    delta = v_new - v_old
    step = near_step if abs(v_old - vth) < near_step else far_step
    if delta > step:
        return v_old + step
    if delta < -step:
        return v_old - step
    return v_new


def pn_limit(v_new: float, v_old: float, vt: float, vcrit: float) -> float:
    """SPICE ``PNJLIM`` limiting for exponential PN-junction voltages."""
    if v_new > vcrit and abs(v_new - v_old) > 2.0 * vt:
        if v_old > 0.0:
            arg = 1.0 + (v_new - v_old) / vt
            if arg > 0.0:
                return v_old + vt * math.log(arg)
            return vcrit
        if v_new > 0:
            return vt * math.log(max(v_new / vt, 1.0 + 1e-12))
        return vcrit
    return v_new
