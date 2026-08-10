"""Cross-validation against ngspice, plus the portability guarantees it found.

The tests in the first section run everywhere and lock in the netlist
conventions that make the shipped examples mean the same thing in any SPICE.
They exist because a cross-check found the opposite: `VTO=0.45` on a PMOS card
is an *enhancement* device to SiliconStat (which normalises with `abs()`) and
a *depletion* device to ngspice (which takes the sign literally), and the two
tools silently simulated different circuits.

The second section runs the real comparison, and is skipped when ngspice is
not installed.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from siliconstat.core import parse_netlist_file, solve_dc
from siliconstat.core.netlist import parse_netlist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS = PROJECT_ROOT / "docs" / "scripts" / "ngspice_crosscheck.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("ngspice_crosscheck", HARNESS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ngspice_crosscheck"] = module
    spec.loader.exec_module(module)
    return module


def model_cards(text: str) -> list[tuple[str, list[str]]]:
    """Return [(type, [parameter tokens])] for every .model card in *text*."""
    cards: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("*") or not line:
            continue
        if line.lower().startswith(".model"):
            tokens = line.split()
            current = (tokens[2].lower() if len(tokens) > 2 else "", list(tokens[3:]))
            cards.append(current)
        elif line.startswith("+") and current is not None:
            current[1].extend(line[1:].split())
        elif not line.startswith("+"):
            current = None
    return cards


EXAMPLES = sorted((PROJECT_ROOT / "examples").rglob("*.net")) + \
    sorted((PROJECT_ROOT / "examples").rglob("*.lib"))


# ---------------------------------------------------------------------------
# portability conventions -- these run everywhere
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_shipped_pmos_cards_use_the_spice_sign_convention(path: Path):
    """A PMOS VTO must be written negative so the card is portable.

    SiliconStat accepts either sign, so this cannot be caught by its own
    results -- only by a second simulator, or by this test.
    """
    for mtype, params in model_cards(path.read_text(encoding="utf-8")):
        if mtype != "pmos":
            continue
        for token in params:
            if token.upper().startswith("VTO="):
                value = float(token.split("=", 1)[1])
                assert value < 0, (
                    f"{path.name}: PMOS card has {token}. SPICE reads a positive "
                    "PMOS VTO as a depletion device; write it negative.")


def test_both_vto_signs_give_identical_results():
    """SiliconStat's abs() normalisation is what makes the fix free."""
    text = (PROJECT_ROOT / "examples" / "two_stage_opamp.net").read_text(
        encoding="utf-8")
    flipped = re.sub(r"VTO=-0\.45", "VTO=0.45", text)
    assert flipped != text, "expected a negative PMOS VTO to flip"

    a = solve_dc(parse_netlist(text, allow_include=False))
    b = solve_dc(parse_netlist(flipped, allow_include=False))
    for node in a.node_voltages:
        assert a.node_voltages[node] == pytest.approx(b.node_voltages[node],
                                                      rel=1e-12)


# ---------------------------------------------------------------------------
# the converter
# ---------------------------------------------------------------------------

def test_converter_drops_siliconstat_only_constructs():
    harness = load_harness()
    source = (PROJECT_ROOT / "examples" / "current_mirror.net").read_text(
        encoding="utf-8")
    deck = harness.convert(source, "print v(out)")
    lowered = deck.lower()
    for construct in (".measure", ".spec", "match=", "avt=", "abeta="):
        assert construct not in lowered, construct
    assert "level=1" in lowered
    assert ".control" in lowered and "op" in lowered


def test_converter_forces_a_negative_pmos_vto_even_if_the_card_is_positive():
    harness = load_harness()
    deck = harness.convert(
        ".model PCH PMOS VTO=0.45 KP=86u\n"
        "VDD vdd 0 1.8\nM1 out g vdd vdd PCH W=10u L=1u\nR1 out 0 10k\n",
        "print v(out)")
    pmos_line = next(line for line in deck.splitlines()
                     if line.lower().startswith(".model pch"))
    assert "VTO=-0.45" in pmos_line


def test_converter_leaves_nmos_vto_alone():
    harness = load_harness()
    deck = harness.convert(
        ".model NCH NMOS VTO=0.45 KP=246u\n"
        "VDD vdd 0 1.8\nM1 out g 0 0 NCH W=10u L=1u\nR1 vdd out 10k\n",
        "print v(out)")
    nmos_line = next(line for line in deck.splitlines()
                     if line.lower().startswith(".model nch"))
    assert "VTO=0.45" in nmos_line and "VTO=-" not in nmos_line


def test_converted_decks_still_parse_as_siliconstat_netlists():
    """A sanity check on the converter: dropping the SiliconStat-only lines
    must not corrupt the circuit itself."""
    harness = load_harness()
    for name in ("rc_divider", "current_mirror", "diff_pair"):
        source = (PROJECT_ROOT / "examples" / f"{name}.net").read_text(
            encoding="utf-8")
        deck = harness.convert(source, "print v(0)")
        body = "\n".join(line for line in deck.splitlines()
                         if not line.strip().lower().startswith(
                             (".control", ".endc", "op", "print", "set ")))
        circuit = parse_netlist(body + "\n.measure probe V(0)\n",
                                allow_include=False)
        original = parse_netlist_file(
            str(PROJECT_ROOT / "examples" / f"{name}.net"))
        assert len(circuit.devices) == len(original.devices), name


# ---------------------------------------------------------------------------
# the real comparison -- skipped when ngspice is absent
# ---------------------------------------------------------------------------

def ngspice_binary() -> str | None:
    return load_harness().find_ngspice()


requires_ngspice = pytest.mark.skipif(
    ngspice_binary() is None,
    reason="ngspice is not installed; see docs/validation.md section 13")


@requires_ngspice
@pytest.mark.slow
@pytest.mark.ngspice
@pytest.mark.parametrize("case", ["rc_divider", "diode_bias", "current_mirror",
                                  "diff_pair", "two_stage_opamp"])
def test_dc_operating_point_matches_ngspice(case, tmp_path):
    """Both tools implement the same level-1 equations, so the DC operating
    points must agree to solver tolerance."""
    harness = load_harness()
    binary = harness.find_ngspice()
    control, probes = harness.CASES[case]

    source = (PROJECT_ROOT / "examples" / f"{case}.net").read_text(encoding="utf-8")
    deck = tmp_path / f"{case}.cir"
    deck.write_text(harness.convert(source, control), encoding="utf-8")

    output = harness.run_ngspice(binary, deck)
    mine = harness.siliconstat_values(case, probes)

    compared = 0
    for label, vector, _accessor, mode in probes:
        reference = harness.parse_printed(output, vector)
        assert reference is not None, f"{case}: ngspice did not print {vector}"
        value = mine[label]
        if mode == "magnitude":
            reference, value = abs(reference), abs(value)
        scale = max(abs(reference), abs(value), 1e-30)
        # 1e-4 is generous against the measured 6.5e-7 worst case; it is set by
        # ngspice's own default reltol of 1e-3, not by either solver.
        assert abs(value - reference) / scale < 1e-4, (
            f"{case}/{label}: ngspice {reference:.9g} vs siliconstat "
            f"{value:.9g}")
        compared += 1
    assert compared == len(probes)


@requires_ngspice
@pytest.mark.slow
@pytest.mark.ngspice
def test_the_full_crosscheck_script_exits_clean():
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-W", "ignore", str(HARNESS)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stdout[-3000:]
    assert "quantities compared across" in completed.stdout
    assert "DISAGREEMENTS" not in completed.stdout


def test_the_script_reports_absence_honestly_when_ngspice_is_missing(monkeypatch):
    """It must never fabricate ngspice output."""
    import subprocess

    harness = load_harness()
    monkeypatch.setattr(harness, "find_ngspice", lambda: None)
    assert harness.find_ngspice() is None

    if ngspice_binary() is not None:
        pytest.skip("ngspice is installed; the absent-path is covered by review")
    completed = subprocess.run(
        [sys.executable, str(HARNESS)], cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 2
    assert "none is claimed" in completed.stderr


# ---------------------------------------------------------------------------
# whole-waveform comparison, not just the operating point
# ---------------------------------------------------------------------------

def load_waveform_harness():
    """docs/scripts/ngspice_waveforms.py, imported as a module."""
    path = PROJECT_ROOT / "docs" / "scripts" / "ngspice_waveforms.py"
    spec = importlib.util.spec_from_file_location("ngspice_waveforms", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ngspice_waveforms"] = module
    spec.loader.exec_module(module)
    return module


@requires_ngspice
@pytest.mark.slow
@pytest.mark.ngspice
def test_ac_sweep_matches_ngspice(tmp_path):
    """The OTA's whole Bode response, not just its DC gain.

    Both tools linearise the same level-1 model about the same operating point
    and solve the same complex MNA system, so there is no integration error to
    hide behind here -- agreement should be at the conditioning limit.
    """
    harness = load_waveform_harness()
    rows, extra = harness.do_ac(harness.find_ngspice(), tmp_path)

    by_name = {row["node"]: row for row in rows}
    assert by_name["out |H|"]["max_abs"] < 1e-3, by_name["out |H|"]
    assert by_name["out phase"]["max_abs"] < 1e-2, by_name["out phase"]
    assert by_name["n2 |H|"]["max_abs"] < 1e-3
    assert by_name["nz |H|"]["max_abs"] < 1e-3
    assert by_name["out |H|"]["n"] == extra["ours_points"] == 109

    # The two figures a designer actually signs off on.
    assert abs(extra["ours_dc"] - extra["theirs_dc"]) < 1e-3
    assert abs(extra["ours_ugf"] - extra["theirs_ugf"]) / extra["theirs_ugf"] < 1e-5


@requires_ngspice
@pytest.mark.slow
@pytest.mark.ngspice
def test_transient_matches_ngspice(tmp_path):
    """The inverter edge, where the two tools legitimately differ.

    SiliconStat integrates on a fixed 20 ps grid; ngspice picks its own
    timestep from a local truncation error estimate and returned 421 points to
    our 402.  So this asserts a real but bounded difference rather than
    equality -- and pins where it lives, which is the switching instant.
    """
    harness = load_waveform_harness()
    rows, extra = harness.do_transient(harness.find_ngspice(), tmp_path)
    by_name = {row["node"]: row for row in rows}

    # The stimulus and the rail are algebraic, not integrated: exact.
    assert by_name["in"]["max_abs"] < 1e-9
    assert by_name["vdd"]["max_abs"] == 0.0

    out = by_name["out"]
    assert out["max_pct_of_span"] < 1.0, out      # measured 0.64 %
    assert out["rms"] < 5e-3, out                 # measured 2.5 mV
    # The worst disagreement is at the falling edge, not spread over the trace.
    assert 1.0e-9 <= out["at_x"] <= 1.2e-9, out

    # Both tools see the gate-drain coupling overshoot above the 1.8 V rail.
    assert extra["ours_vmax"] > 1.8 and extra["theirs_vmax"] > 1.8
    assert abs(extra["ours_vmax"] - extra["theirs_vmax"]) < 0.02
    assert extra["ours_vmin"] < 0.0 and extra["theirs_vmin"] < 0.0


@requires_ngspice
@pytest.mark.slow
@pytest.mark.ngspice
def test_ngspice_vp_is_radians_not_degrees(tmp_path):
    """Guards the units trap that produced a bogus 227 deg disagreement.

    ngspice's ``vp()`` emits radians wrapped to (-pi, pi]; calling np.radians()
    on it a second time silently rescales the whole phase response.
    """
    import numpy as np

    harness = load_waveform_harness()
    rows, extra = harness.do_ac(harness.find_ngspice(), tmp_path)
    phase = extra["traces"]["ph"]
    # Unwrapped and converted once, the response spans roughly 0 to -225 deg.
    assert phase[0] == pytest.approx(0.0, abs=0.1)
    assert -230.0 < phase[-1] < -220.0
    # If vp() were degrees, unwrapping raw values would leave a >1000 deg span.
    assert np.ptp(phase) < 260.0
