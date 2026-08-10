"""Empirical test of Pelgrom area scaling on the demo current mirror.

Claims under test
-----------------
1. The *injected* per-device sigma is ``A / sqrt(2*W*L)``, so that the drawn
   difference across a matched pair has ``sigma = A / sqrt(W*L)``.
   (Checked directly on the drawn slot values -- no circuit involved.)

2. Quadrupling the transistor area halves the mismatch, **at constant W/L**.
   Scaling W and L together holds the overdrive Vov fixed, so the only thing
   that changes is the area in Pelgrom's law.

3. Quadrupling the area by widening alone does **not** halve the copy error of
   a current mirror biased at a fixed current.  This is run as a control
   because it is the mistake the law invites: at fixed Id, widening drops Vov,
   gm/Id rises as sqrt(W), and the improvement in sigma(dVth) is cancelled
   exactly by the increased sensitivity to it.

Method
------
``examples/current_mirror.net`` is read as text and its ``.param WM`` / ``.param
LM`` lines are rewritten for each geometry; nothing else changes.  Each variant
is re-parsed and run through the same Monte Carlo engine with:

* ``mode = mismatch`` -- local (per-device) variation only.  Global process
  spread is common to both transistors and largely cancels in the copy ratio;
  leaving it on would only add noise to the quantity being measured.
* ``include_passives = False`` -- the 125 k load resistor's own mismatch is
  area-independent and would put a floor under the curve, masking the scaling
  law we are trying to observe.
* the same seed at every geometry, so all points share the same underlying
  standard-normal draws and the differences are pure area scaling.

Run::

    .venv/Scripts/python.exe docs/scripts/pelgrom_area_sweep.py

Every number printed comes from a completed Monte Carlo run, except the
columns explicitly labelled "predicted".
"""

from __future__ import annotations

import math
import re
import sys
import time

import numpy as np

from siliconstat.analysis.statistics import describe
from siliconstat.core.netlist import parse_netlist
from siliconstat.core.solver import SolverOptions, solve_dc
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.variation import default_mismatch_model
from siliconstat.variation.pelgrom import sigma_device, sigma_pair

NETLIST = "examples/current_mirror.net"
SAMPLES = 2000
SEED = 12345
MEASUREMENT = "ierr"                       # copy error, percent

#: (W [um], L [um]) at constant W/L = 10.  Area x4 between alternate rows.
CONSTANT_SHAPE = [(5.0, 0.5), (7.07107, 0.707107), (10.0, 1.0),
                  (14.14214, 1.414214), (20.0, 2.0)]
#: Width-only sweep at fixed L = 1 um -- the control.
WIDTH_ONLY = [(2.5, 1.0), (5.0, 1.0), (10.0, 1.0), (20.0, 1.0), (40.0, 1.0)]


def netlist_with_geometry(text: str, w_um: float, l_um: float) -> str:
    """Rewrite the ``.param WM`` / ``.param LM`` lines; leave the rest alone."""
    out, n_w = re.subn(r"(?mi)^\.param\s+WM\s*=\s*\S+\s*$",
                       f".param WM={w_um:.6g}u", text)
    out, n_l = re.subn(r"(?mi)^\.param\s+LM\s*=\s*\S+\s*$",
                       f".param LM={l_um:.6g}u", out)
    if n_w != 1 or n_l != 1:
        raise SystemExit(f"expected one WM and one LM line, got {n_w} and {n_l}")
    return out


def run_point(base_text: str, w_um: float, l_um: float) -> dict:
    circuit = parse_netlist(netlist_with_geometry(base_text, w_um, l_um),
                            source=f"current_mirror_W{w_um:g}u_L{l_um:g}u")
    mos = circuit.device("M1")
    card = circuit.mos_models[mos.model]
    area = (mos.w * 1e6) * (mos.l * 1e6)                 # um^2

    # Nominal operating point: gm/Id is what converts a threshold offset into a
    # current error, so it must be reported alongside the geometry.
    op = solve_dc(circuit, circuit.build_context(),
                  SolverOptions.from_options(circuit.options))
    dev = op.device_ops["M2"]

    variation = default_mismatch_model(circuit, enable_process=False,
                                       enable_mismatch=True,
                                       include_passives=False)
    config = MonteCarloConfig(variation=variation, samples=SAMPLES, seed=SEED,
                              solver=SolverOptions.from_options(circuit.options))
    started = time.perf_counter()
    run = run_monte_carlo(circuit, config)
    elapsed = time.perf_counter() - started

    stats = describe(run.values(MEASUREMENT), name=MEASUREMENT, unit="%")

    # Claim 1: check the sqrt(2) injection convention on the raw draws.
    ok = run.successful_samples()
    d1 = np.array([s.slot_values["M1.vth_local"] for s in ok])
    d2 = np.array([s.slot_values["M2.vth_local"] for s in ok])
    b1 = np.array([s.slot_values["M1.beta_local"] for s in ok])
    b2 = np.array([s.slot_values["M2.beta_local"] for s in ok])

    sigma_dvth_pair = sigma_pair(card.avt, mos.w, mos.l)
    sigma_dbeta_pair = sigma_pair(card.abeta, mos.w, mos.l)
    gm_id = float(dev["gm_over_id"])

    return {
        "w_um": w_um, "l_um": l_um, "area_um2": area,
        "vov": float(dev["vov"]), "gm_id": gm_id, "id": float(dev["id"]),
        "sigma_device_theory": sigma_device(card.avt, mos.w, mos.l),
        "sigma_device_drawn": float(np.std(d1, ddof=1)),
        "sigma_pair_theory": sigma_dvth_pair,
        "sigma_pair_drawn": float(np.std(d1 - d2, ddof=1)),
        "sigma_dbeta_pair_drawn": float(np.std(b1 - b2, ddof=1)),
        # Small-signal prediction of the copy error, in percent:
        #   dI/I = (gm/Id) * dVth + dbeta/beta
        "predicted_ierr_pct": 100.0 * math.sqrt(
            (gm_id * sigma_dvth_pair) ** 2 + sigma_dbeta_pair ** 2),
        "sigma_ierr_pct": stats.std,
        "ci_low": stats.std_ci95_low, "ci_high": stats.std_ci95_high,
        "mean_ierr_pct": stats.mean,
        "n_ok": run.counters.successful, "n_fail": run.counters.failed,
        "wall_s": elapsed,
    }


def sweep(base_text: str, geometries, title: str) -> list[dict]:
    print(f"\n### {title}", file=sys.stderr)
    rows = []
    for w_um, l_um in geometries:
        r = run_point(base_text, w_um, l_um)
        rows.append(r)
        print(f"  W={w_um:9.4f} L={l_um:8.4f}  area={r['area_um2']:7.2f} um^2  "
              f"Vov={1e3 * r['vov']:6.1f} mV  gm/Id={r['gm_id']:6.2f}  "
              f"sigma(ierr)={r['sigma_ierr_pct']:.4f} %  ({r['wall_s']:.1f} s)",
              file=sys.stderr)
    return rows


def print_convention_table(rows: list[dict]) -> None:
    print("\n"
          "Injection convention: sigma_device = AVT/sqrt(2*W*L), "
          "so sigma(dVth) across the pair = AVT/sqrt(W*L)")
    head = (f"{'area [um2]':>11} {'per-device theory [mV]':>23} "
            f"{'per-device drawn [mV]':>22} {'pair theory [mV]':>17} "
            f"{'pair drawn [mV]':>16} {'ratio drawn/theory':>19}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['area_um2']:11.2f} {1e3 * r['sigma_device_theory']:23.4f} "
              f"{1e3 * r['sigma_device_drawn']:22.4f} "
              f"{1e3 * r['sigma_pair_theory']:17.4f} "
              f"{1e3 * r['sigma_pair_drawn']:16.4f} "
              f"{r['sigma_pair_drawn'] / r['sigma_pair_theory']:19.4f}")


def print_sweep_table(rows: list[dict], title: str) -> None:
    ref = rows[-1]
    for r in rows:
        r["ideal_pct"] = ref["sigma_ierr_pct"] * math.sqrt(
            ref["area_um2"] / r["area_um2"])
        r["dev_pct"] = 100.0 * (r["sigma_ierr_pct"] / r["ideal_pct"] - 1.0)

    print(f"\n{title}")
    head = (f"{'W [um]':>9} {'L [um]':>8} {'area [um2]':>11} {'Vov [mV]':>9} "
            f"{'gm/Id [1/V]':>12} {'measured sigma(ierr) [%]':>25} "
            f"{'95% CI on sigma':>20} {'analytic [%]':>13} "
            f"{'ideal 1/sqrt(A) [%]':>20} {'dev [%]':>8}")
    print(head)
    print("-" * len(head))
    for r in rows:
        ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(f"{r['w_um']:9.4f} {r['l_um']:8.4f} {r['area_um2']:11.2f} "
              f"{1e3 * r['vov']:9.1f} {r['gm_id']:12.2f} "
              f"{r['sigma_ierr_pct']:25.4f} {ci:>20} "
              f"{r['predicted_ierr_pct']:13.4f} {r['ideal_pct']:20.4f} "
              f"{r['dev_pct']:+8.2f}")

    print("\n  area-quadrupling check (row vs. the geometry with 1/4 the area):")
    for a, b in zip(rows, rows[2:]):
        print(f"    area {a['area_um2']:6.2f} -> {b['area_um2']:6.2f} um^2 "
              f"(x{b['area_um2'] / a['area_um2']:.0f})   "
              f"sigma ratio = {a['sigma_ierr_pct'] / b['sigma_ierr_pct']:.4f}"
              f"   ideal 2.0000")

    log_area = np.log(np.array([r["area_um2"] for r in rows]))
    log_sigma = np.log(np.array([r["sigma_ierr_pct"] for r in rows]))
    slope, intercept = np.polyfit(log_area, log_sigma, 1)
    residual = log_sigma - (slope * log_area + intercept)
    ss_tot = float(((log_sigma - log_sigma.mean()) ** 2).sum())
    r2 = 1.0 - float(residual @ residual) / ss_tot if ss_tot > 0 else float("nan")
    print(f"\n  log-log fit: sigma(ierr) ~ area^({slope:+.4f})   "
          f"[Pelgrom predicts -0.5000]   R^2 = {r2:.6f}")


def main() -> int:
    with open(NETLIST, "r", encoding="utf-8") as fh:
        base_text = fh.read()

    shape_rows = sweep(base_text, CONSTANT_SHAPE,
                       "sweep A -- W and L scaled together (W/L = 10 fixed)")
    width_rows = sweep(base_text, WIDTH_ONLY,
                       "sweep B -- width only, L = 1 um fixed (control)")

    print_convention_table(shape_rows)
    print_sweep_table(shape_rows,
                      "SWEEP A -- constant W/L = 10, area scaled by geometry")
    print_sweep_table(width_rows,
                      "SWEEP B (CONTROL) -- width only at fixed Id and L")

    total = SAMPLES * (len(CONSTANT_SHAPE) + len(WIDTH_ONLY))
    print(f"\nsamples per point = {SAMPLES}, seed = {SEED}, "
          f"total simulations = {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
