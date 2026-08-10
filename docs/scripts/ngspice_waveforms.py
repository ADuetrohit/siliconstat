"""Run the *same* transient and AC analyses in ngspice and diff the traces.

``ngspice_crosscheck.py`` compares operating points -- single numbers.  This
script compares whole waveforms: it converts the example netlists with the same
translator, asks ngspice for the identical analysis, dumps its vectors with
``wrdata``, and lines them up against SiliconStat's own traces.

Both tools are solving the same equations with the same LEVEL=1 model, so the
curves should lie on top of each other to within the integration error.  Where
they do not, the difference is reported rather than smoothed over.

    python docs/scripts/ngspice_waveforms.py [--plot]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ngspice_crosscheck import convert, find_ngspice  # noqa: E402

from siliconstat.core.solver import (  # noqa: E402
    SolverOptions, ac_analysis, solve_dc, transient_analysis)
from siliconstat.core import parse_netlist_file  # noqa: E402
from siliconstat.measure.engine import _log_sweep  # noqa: E402


def card(circuit, kind: str) -> dict:
    """The circuit's own analysis card -- the same one the UI reads."""
    for analysis in circuit.analyses:
        if analysis.kind == kind:
            return dict(analysis.args)
    raise SystemExit(f"netlist declares no .{kind} card")


def run(binary: str, deck: str, workdir: Path) -> str:
    """Write *deck* into *workdir*, run ngspice in batch, return its stdout."""
    path = workdir / "deck.cir"
    path.write_text(deck, encoding="utf-8")
    proc = subprocess.run([binary, "-b", str(path)], capture_output=True,
                          text=True, timeout=300, cwd=workdir)
    return proc.stdout + proc.stderr


def read_wrdata(path: Path) -> np.ndarray:
    """``wrdata`` writes an x column before *every* y column; keep one x."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    columns = [raw[:, 0]] + [raw[:, i] for i in range(1, raw.shape[1], 2)][1:]
    # column 0 is x, columns 1,3,5,... are the repeated x -- keep 1,2 pairs:
    ys = [raw[:, i] for i in range(1, raw.shape[1]) if i % 2 == 1]
    return np.column_stack([raw[:, 0], *ys])


def compare(name: str, ours_x, ours_y, theirs_x, theirs_y, unit: str) -> dict:
    """Resample ngspice onto our grid and report the worst disagreement."""
    lo = max(ours_x.min(), theirs_x.min())
    hi = min(ours_x.max(), theirs_x.max())
    mask = (ours_x >= lo) & (ours_x <= hi)
    x = ours_x[mask]
    a = ours_y[mask]
    b = np.interp(x, theirs_x, theirs_y)
    err = np.abs(a - b)
    i = int(np.argmax(err))
    span = float(np.ptp(a)) or 1.0
    return {"node": name, "unit": unit, "n": int(x.size),
            "max_abs": float(err.max()), "at_x": float(x[i]),
            "max_pct_of_span": 100.0 * float(err.max()) / span,
            "rms": float(np.sqrt(np.mean(err ** 2)))}


# ---------------------------------------------------------------------------
# transient
# ---------------------------------------------------------------------------

def do_transient(binary: str, workdir: Path) -> tuple[list[dict], dict]:
    path = ROOT / "examples" / "inverter_transient.net"
    source = path.read_text("utf-8")
    circuit = parse_netlist_file(path)
    spec = card(circuit, "tran")
    tstep, tstop = spec["tstep"], spec["tstop"]

    opts = SolverOptions()
    op = solve_dc(circuit, opts=opts)
    ours = transient_analysis(circuit, tstep=tstep, tstop=tstop, op=op, opts=opts)

    control = (f"tran {tstep:g} {tstop:g}\n"
               "wrdata tran.txt v(in) v(out) v(vdd)\n")
    deck = convert(source, control)
    log = run(binary, deck, workdir)
    data = read_wrdata(workdir / "tran.txt")

    rows = []
    for index, node in enumerate(("in", "out", "vdd"), start=1):
        rows.append(compare(node, np.asarray(ours.time), np.asarray(ours.v(node)),
                            data[:, 0], data[:, index], "V"))
    extra = {
        "ours_points": len(ours.time), "theirs_points": int(data.shape[0]),
        "ours_vmax": float(np.max(ours.v("out"))),
        "theirs_vmax": float(np.max(data[:, 2])),
        "ours_vmin": float(np.min(ours.v("out"))),
        "theirs_vmin": float(np.min(data[:, 2])),
        "log": log,
        "traces": {"t": data[:, 0], "in": data[:, 1], "out": data[:, 2],
                   "ours_t": np.asarray(ours.time),
                   "ours_in": np.asarray(ours.v("in")),
                   "ours_out": np.asarray(ours.v("out"))},
    }
    return rows, extra


# ---------------------------------------------------------------------------
# ac
# ---------------------------------------------------------------------------

def do_ac(binary: str, workdir: Path) -> tuple[list[dict], dict]:
    path = ROOT / "examples" / "two_stage_opamp.net"
    source = path.read_text("utf-8")
    circuit = parse_netlist_file(path)
    spec = card(circuit, "ac")
    fstart, fstop, per = spec["fstart"], spec["fstop"], spec["points"]

    opts = SolverOptions()
    op = solve_dc(circuit, opts=opts)
    freqs = _log_sweep(spec)          # the identical grid the dashboard plots
    ours = ac_analysis(circuit, freqs, op=op, opts=opts)

    control = (f"ac dec {per:g} {fstart:g} {fstop:g}\n"
               "wrdata ac.txt vdb(out) vp(out) vdb(n2) vdb(nz)\n")
    deck = convert(source, control)
    log = run(binary, deck, workdir)
    data = read_wrdata(workdir / "ac.txt")

    ref = ours.v("inp")
    ours_db = 20.0 * np.log10(np.abs(ours.v("out") / ref))
    ours_ph = np.degrees(np.unwrap(np.angle(ours.v("out") / ref)))
    # ngspice's vp() emits RADIANS wrapped to (-pi, pi], not degrees -- an
    # earlier version here called np.radians() on it a second time and reported
    # a bogus 227 deg disagreement.  Unwrap first, then convert once.
    theirs_db = data[:, 1]
    theirs_ph = np.degrees(np.unwrap(data[:, 2]))

    rows = [compare("out |H|", freqs, ours_db, data[:, 0], theirs_db, "dB"),
            compare("out phase", freqs, ours_ph, data[:, 0], theirs_ph, "deg"),
            compare("n2 |H|", freqs, 20 * np.log10(np.abs(ours.v("n2") / ref)),
                    data[:, 0], data[:, 3], "dB"),
            compare("nz |H|", freqs, 20 * np.log10(np.abs(ours.v("nz") / ref)),
                    data[:, 0], data[:, 4], "dB")]

    def ugf(f, db):
        for i in range(1, len(db)):
            if db[i - 1] > 0 >= db[i]:
                t = db[i - 1] / (db[i - 1] - db[i])
                return float(np.exp(np.log(f[i - 1]) + t * (np.log(f[i]) - np.log(f[i - 1]))))
        return float("nan")

    extra = {
        "ours_dc": float(ours_db[0]), "theirs_dc": float(theirs_db[0]),
        "ours_ugf": ugf(freqs, ours_db), "theirs_ugf": ugf(data[:, 0], theirs_db),
        "ours_points": len(freqs), "theirs_points": int(data.shape[0]),
        "log": log,
        "traces": {"f": data[:, 0], "db": theirs_db, "ph": theirs_ph,
                   "ours_f": freqs, "ours_db": ours_db, "ours_ph": ours_ph},
    }
    return rows, extra


def table(rows: list[dict]) -> None:
    print(f"  {'trace':10s} {'pts':>5s} {'max |diff|':>12s} {'at':>12s} "
          f"{'% of span':>10s} {'rms':>12s}")
    for r in rows:
        print(f"  {r['node']:10s} {r['n']:5d} "
              f"{r['max_abs']:9.3e} {r['unit']:<2s} {r['at_x']:12.4g} "
              f"{r['max_pct_of_span']:9.4f}% {r['rms']:9.3e} {r['unit']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="write overlay PNGs to out/ngspice/")
    parser.add_argument("--show-log", action="store_true")
    args = parser.parse_args()

    binary = find_ngspice()
    if binary is None:
        print("ngspice not found", file=sys.stderr)
        return 1
    version = subprocess.run([binary, "-v"], capture_output=True, text=True)
    print((version.stdout or version.stderr).strip().splitlines()[0])
    print(f"binary: {binary}\n")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        print("== inverter_transient.net -- tran 20p 8n ==")
        tran_rows, tran_extra = do_transient(binary, workdir)
        table(tran_rows)
        print(f"  points     SiliconStat {tran_extra['ours_points']}  "
              f"ngspice {tran_extra['theirs_points']}")
        print(f"  V(out) max SiliconStat {tran_extra['ours_vmax']:.6f} V  "
              f"ngspice {tran_extra['theirs_vmax']:.6f} V")
        print(f"  V(out) min SiliconStat {tran_extra['ours_vmin']:.6f} V  "
              f"ngspice {tran_extra['theirs_vmin']:.6f} V")
        if args.show_log:
            print(tran_extra["log"])

        print("\n== two_stage_opamp.net -- ac dec 12 1 1G ==")
        ac_rows, ac_extra = do_ac(binary, workdir)
        table(ac_rows)
        print(f"  DC gain    SiliconStat {ac_extra['ours_dc']:.4f} dB  "
              f"ngspice {ac_extra['theirs_dc']:.4f} dB  "
              f"(diff {abs(ac_extra['ours_dc'] - ac_extra['theirs_dc']):.4f} dB)")
        print(f"  unity gain SiliconStat {ac_extra['ours_ugf'] / 1e6:.4f} MHz  "
              f"ngspice {ac_extra['theirs_ugf'] / 1e6:.4f} MHz  "
              f"(diff {abs(ac_extra['ours_ugf'] - ac_extra['theirs_ugf']) / 1e3:.2f} kHz)")
        if args.show_log:
            print(ac_extra["log"])

        if args.plot:
            plot(tran_extra, ac_extra)
    return 0


def plot(tran: dict, ac: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = ROOT / "out" / "ngspice"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    t = tran["traces"]
    fig, (ax, axe) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1],
                                  sharex=True)
    ax.plot(t["ours_t"] * 1e9, t["ours_out"], color="#22d3ee", lw=2.4,
            label="SiliconStat V(out)")
    ax.plot(t["t"] * 1e9, t["out"], color="#f97316", lw=1.1, ls="--",
            label="ngspice V(out)")
    ax.plot(t["ours_t"] * 1e9, t["ours_in"], color="#64748b", lw=1.0,
            label="V(in)")
    ax.set_ylabel("voltage [V]")
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("CMOS inverter, tran 20p 8n -- SiliconStat vs ngspice "
                 f"{len(t['ours_t'])} pts")
    ax.grid(alpha=0.2)

    resampled = np.interp(t["ours_t"], t["t"], t["out"])
    axe.plot(t["ours_t"] * 1e9, (t["ours_out"] - resampled) * 1e3,
             color="#a78bfa", lw=1.2)
    axe.set_ylabel("Δ [mV]")
    axe.set_xlabel("time [ns]")
    axe.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "transient_overlay.png", dpi=140)
    plt.close(fig)

    a = ac["traces"]
    fig, (axm, axp, axe) = plt.subplots(3, 1, figsize=(11, 8),
                                        height_ratios=[3, 2, 1], sharex=True)
    axm.semilogx(a["ours_f"], a["ours_db"], color="#22d3ee", lw=2.4,
                 label="SiliconStat")
    axm.semilogx(a["f"], a["db"], color="#f97316", lw=1.1, ls="--",
                 label="ngspice")
    axm.axhline(0, color="#475569", lw=0.8, ls=":")
    axm.set_ylabel("|H| [dB]")
    axm.legend(fontsize=9)
    axm.set_title("Two-stage Miller OTA, ac dec 12 1 1G -- V(out)/V(inp)")
    axm.grid(alpha=0.2)

    axp.semilogx(a["ours_f"], a["ours_ph"], color="#22d3ee", lw=2.4)
    axp.semilogx(a["f"], a["ph"], color="#f97316", lw=1.1, ls="--")
    axp.set_ylabel("∠H [deg]")
    axp.grid(alpha=0.2)

    axe.semilogx(a["ours_f"], a["ours_db"] - np.interp(a["ours_f"], a["f"], a["db"]),
                 color="#a78bfa", lw=1.2)
    axe.set_ylabel("Δ [dB]")
    axe.set_xlabel("frequency [Hz]")
    axe.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "ac_overlay.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {outdir / 'transient_overlay.png'}")
    print(f"wrote {outdir / 'ac_overlay.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
