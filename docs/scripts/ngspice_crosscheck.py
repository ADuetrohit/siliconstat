"""Cross-validate SiliconStat against ngspice.

The meaningful comparison is against ngspice's own ``LEVEL=1`` MOSFET model:
both tools then implement the same equations, so the two answers should agree
to solver tolerance and any real discrepancy points at a bug.  Comparing
against BSIM would mostly measure the (large, known) difference between
level-1 and a modern short-channel model, which is documented in
docs/limitations.md.

Rather than maintaining a second set of hand-written decks -- which could
silently drift from ``examples/`` and turn a real disagreement into "the two
netlists were different" -- this script **converts** each SiliconStat netlist
into ngspice syntax.  Both simulators therefore see the same circuit by
construction.  The conversion is deliberately small:

* ``.measure`` / ``.spec`` are SiliconStat-only and are dropped;
* ``MATCH=`` is a SiliconStat-only instance attribute and is dropped;
* ``.model … NMOS/PMOS`` gains ``LEVEL=1`` and loses the mismatch coefficients
  (``AVT``/``ABETA``), which ngspice does not know;
* a ``.control`` block is appended to print the quantities being compared.

Usage
-----
    .venv/Scripts/python.exe docs/scripts/ngspice_crosscheck.py [--keep]

Exits 2 with an explanation if ngspice cannot be found; it never fabricates
ngspice output.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WORK = ROOT / "out" / "ngspice"

# Parameters ngspice's level-1 model card does not accept.
DROP_MODEL_PARAMS = {"avt", "abeta", "tcv", "bex", "cjd", "cjs"}

# (probe label, ngspice vector, SiliconStat accessor, compare mode)
#
# compare mode "magnitude" exists for PMOS drain currents only: ngspice reports
# a PMOS drain current in the p-type frame (positive when conducting), while
# SiliconStat reports the current flowing *into the physical drain terminal*,
# which is negative for a PMOS.  Both are defensible; they are not a numerical
# disagreement, so the magnitudes are compared and the convention is stated.
Probe = tuple[str, str, str, str]


def find_ngspice() -> str | None:
    """Prefer the console build; look on PATH, then in tools/."""
    for name in ("ngspice_con", "ngspice"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (ROOT / "tools" / "Spice64" / "bin" / "ngspice_con.exe",
                      ROOT / "tools" / "Spice64" / "bin" / "ngspice.exe"):
        if candidate.is_file():
            return str(candidate)
    return None


def _fix_vto(token: str, is_pmos: bool) -> str:
    """SPICE wants a negative VTO for a PMOS; SiliconStat stores a magnitude."""
    if not is_pmos or not token.upper().startswith("VTO="):
        return token
    try:
        value = float(token.split("=", 1)[1])
    except ValueError:
        return token
    return f"VTO={-abs(value)}"


def convert(netlist: str, control: str) -> str:
    """Rewrite a SiliconStat netlist as an ngspice deck."""
    out: list[str] = ["* auto-converted from a SiliconStat netlist"]
    in_pmos_card = [False]
    for raw in netlist.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        lowered = stripped.lower()

        if lowered.startswith((".measure", ".meas", ".spec", ".end")):
            continue
        if lowered.startswith(".title"):
            continue                       # keep ngspice's own title line
        if lowered.startswith(".include"):
            continue                       # examples are self-contained
        if lowered.startswith(".op") or lowered.startswith(".ac") \
                or lowered.startswith(".tran"):
            continue                       # the control block drives analyses

        # Model cards: add LEVEL=1, drop parameters ngspice does not know, and
        # force the SPICE sign convention for a PMOS threshold.  SiliconStat
        # stores VTO as a magnitude; ngspice takes the sign literally, and a
        # PMOS with VTO=+0.45 is a *depletion* device that conducts at Vgs=0.
        if lowered.startswith(".model"):
            tokens = stripped.split()
            if len(tokens) >= 3 and tokens[2].lower() in ("nmos", "pmos"):
                in_pmos_card[0] = tokens[2].lower() == "pmos"
                kept = [_fix_vto(t, in_pmos_card[0]) for t in tokens[3:]
                        if t.split("=")[0].lower() not in DROP_MODEL_PARAMS]
                out.append(" ".join(tokens[:3] + ["LEVEL=1"] + kept))
                continue
            in_pmos_card[0] = False
        if stripped.startswith("+"):
            kept = [_fix_vto(t, in_pmos_card[0]) for t in stripped[1:].split()
                    if t.split("=")[0].lower() not in DROP_MODEL_PARAMS]
            out.append("+ " + " ".join(kept) if kept else "* (dropped)")
            continue

        # Instance lines: MATCH= is SiliconStat-only.
        line = re.sub(r"\s+MATCH=\S+", "", line, flags=re.IGNORECASE)
        out.append(line)

    out.append("")
    out.append(".control")
    # ngspice prints 7 significant digits by default, which would
    # floor the measured agreement at ~1e-7 regardless of the solvers.
    out.append("set numdgt=12")
    out.append("op")
    out.append(control)
    out.append(".endc")
    out.append(".end")
    return "\n".join(out) + "\n"


def run_ngspice(binary: str, deck_path: Path) -> str:
    completed = subprocess.run([binary, "-b", str(deck_path)],
                               capture_output=True, text=True, timeout=300,
                               cwd=str(deck_path.parent), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ngspice exited {completed.returncode}\n"
                           f"{completed.stdout[-1500:]}{completed.stderr[-1500:]}")
    if "Error:" in completed.stdout:
        for line in completed.stdout.splitlines():
            if line.startswith("Error:"):
                raise RuntimeError(f"ngspice reported: {line}")
    return completed.stdout


def parse_printed(output: str, vector: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(vector)}\s*=\s*([-+0-9.eE]+)\s*$",
                         re.MULTILINE | re.IGNORECASE)
    match = pattern.search(output)
    return float(match.group(1)) if match else None


def siliconstat_values(name: str, probes: list[Probe]) -> dict[str, float]:
    from siliconstat.core import parse_netlist_file, solve_dc

    circuit = parse_netlist_file(str(ROOT / "examples" / f"{name}.net"))
    op = solve_dc(circuit)
    values: dict[str, float] = {}
    for label, _vector, accessor, _mode in probes:
        kind, _, target = accessor.partition(":")
        if kind == "node":
            values[label] = op.v(target)
        elif kind == "current":
            values[label] = op.i(target)
        elif kind == "power":
            values[label] = op.total_supply_power()
        else:                                        # device operating point
            device, _, field = target.partition(".")
            values[label] = float(op.device_ops[device][field])
    return values


CASES: dict[str, tuple[str, list[Probe]]] = {
    "rc_divider": (
        "print v(mid) i(v1)",
        [("V(mid)", "v(mid)", "node:mid", "signed"),
         ("I(V1)", "i(v1)", "current:V1", "signed")],
    ),
    "diode_bias": (
        "print v(a) i(vdd)",
        [("V(a)", "v(a)", "node:a", "signed"),
         ("I(VDD)", "i(vdd)", "current:VDD", "signed")],
    ),
    "current_mirror": (
        "print v(nref) v(out) @m1[id] @m2[id] @m1[gm] @m1[gds] @m2[id]",
        [("V(nref)", "v(nref)", "node:nref", "signed"),
         ("V(out)", "v(out)", "node:out", "signed"),
         ("Id(M1)", "@m1[id]", "current:M1", "signed"),
         ("Id(M2)", "@m2[id]", "current:M2", "signed"),
         ("gm(M1)", "@m1[gm]", "device:M1.gm", "signed"),
         ("gds(M1)", "@m1[gds]", "device:M1.gds", "signed")],
    ),
    "diff_pair": (
        "print v(outp) v(outn) v(tail) @m1[id] @m2[id] @m1[gm]",
        [("V(outp)", "v(outp)", "node:outp", "signed"),
         ("V(outn)", "v(outn)", "node:outn", "signed"),
         ("V(tail)", "v(tail)", "node:tail", "signed"),
         ("Id(M1)", "@m1[id]", "current:M1", "signed"),
         ("Id(M2)", "@m2[id]", "current:M2", "signed"),
         ("gm(M1)", "@m1[gm]", "device:M1.gm", "signed")],
    ),
    "two_stage_opamp": (
        "print v(out) v(n1) v(n2) v(tail) @m1[id] @m6[id] @m7[id]",
        [("V(out)", "v(out)", "node:out", "signed"),
         ("V(n1)", "v(n1)", "node:n1", "signed"),
         ("V(n2)", "v(n2)", "node:n2", "signed"),
         ("V(tail)", "v(tail)", "node:tail", "signed"),
         ("Id(M1)", "@m1[id]", "current:M1", "signed"),
         ("Id(M6)", "@m6[id]", "current:M6", "magnitude"),
         ("Id(M7)", "@m7[id]", "current:M7", "signed")],
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="keep the generated ngspice decks for inspection")
    args = parser.parse_args()

    binary = find_ngspice()
    if binary is None:
        print("ngspice is not installed or not on PATH.\n"
              "No cross-validation was performed, and none is claimed.\n"
              "Install ngspice (or extract the Windows build into tools/) and "
              "re-run this script.", file=sys.stderr)
        return 2

    version = subprocess.run([binary, "--version"], capture_output=True,
                             text=True, check=False)
    banner = (version.stdout or version.stderr).strip().splitlines()
    print(f"ngspice binary : {binary}")
    print(f"ngspice version: {banner[0] if banner else '(not reported)'}")

    WORK.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    worst_label = ""
    failures: list[str] = []
    compared = 0

    for name, (control, probes) in CASES.items():
        source = (ROOT / "examples" / f"{name}.net").read_text(encoding="utf-8")
        deck_path = WORK / f"{name}_ngspice.cir"
        deck_path.write_text(convert(source, control), encoding="utf-8")

        print(f"\n=== {name} ===")
        try:
            output = run_ngspice(binary, deck_path)
        except Exception as exc:                      # noqa: BLE001
            print(f"  ngspice failed: {exc}")
            failures.append(f"{name}: ngspice failed")
            continue

        mine = siliconstat_values(name, probes)
        print(f"  {'quantity':<10} {'ngspice 46':>17} {'siliconstat':>17} "
              f"{'rel. diff':>12}")
        for label, vector, _accessor, mode in probes:
            reference = parse_printed(output, vector)
            if reference is None:
                print(f"  {label:<10} {'(not printed)':>17}")
                continue
            value = mine[label]
            if mode == "magnitude":
                reference, value = abs(reference), abs(value)
            scale = max(abs(reference), abs(value), 1e-30)
            diff = abs(value - reference) / scale
            compared += 1
            if diff > worst:
                worst, worst_label = diff, f"{name}/{label}"
            flag = "  <-- LARGE" if diff > 1e-3 else ""
            if mode == "magnitude":
                flag += "   (|.|, PMOS sign convention)"
            print(f"  {label:<10} {reference:>17.9g} {value:>17.9g} "
                  f"{diff:>12.3e}{flag}")
            if diff > 1e-3:
                failures.append(f"{name}/{label}: {diff:.3e}")

    print(f"\n{compared} quantities compared across {len(CASES)} circuits")
    print(f"worst relative difference: {worst:.3e}"
          + (f"  ({worst_label})" if worst_label else ""))
    if failures:
        print("\nDISAGREEMENTS (> 1e-3 relative):")
        for failure in failures:
            print(f"   - {failure}")
    else:
        print("Both tools implement the same level-1 equations; the agreement "
              "above is what that should look like.")

    if not args.keep:
        for deck in WORK.glob("*_ngspice.cir"):
            deck.unlink()
    else:
        print(f"\ngenerated decks kept in {WORK}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
