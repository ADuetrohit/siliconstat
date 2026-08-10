"""Netlist parsing: every element, every directive, and every error path."""

from __future__ import annotations

import math

import pytest

from siliconstat.core.circuit import SpecLimit
from siliconstat.core.devices import (
    Capacitor,
    CurrentSource,
    Diode,
    Inductor,
    Mosfet,
    Resistor,
    VoltageSource,
)
from siliconstat.core.exceptions import NetlistSyntaxError
from siliconstat.core.netlist import parse_netlist, parse_netlist_file
from siliconstat.core.waveforms import PulseWave, PWLWave, SinWave

from conftest import EXAMPLE_NAMES, load, write_netlist

MOS_MODEL = (".model NCH NMOS VTO=0.45 KP=200u LAMBDA=0.1 GAMMA=0.4 PHI=0.8 "
             "AVT=3.5m ABETA=0.01\n")


def parse(text: str):
    return parse_netlist(text)


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------

def test_resistor():
    circuit = parse("V1 a 0 1\nR1 a 0 10k\n.measure v V(a)\n")
    r = circuit.device("R1")
    assert isinstance(r, Resistor)
    assert r.r == pytest.approx(1e4)
    assert [circuit.node_names[n] for n in r.nodes] == ["a", "0"]


def test_capacitor_and_inductor():
    circuit = parse("V1 a 0 1\nR1 a 0 1k\nC1 a 0 1p IC=0.5\nL1 a 0 2n\n"
                    ".measure v V(a)\n")
    c = circuit.device("C1")
    assert isinstance(c, Capacitor)
    assert c.c == pytest.approx(1e-12)
    assert c.ic == pytest.approx(0.5)
    assert isinstance(circuit.device("L1"), Inductor)
    assert circuit.n_branches == 2  # V1 and L1 each add a branch unknown


def test_voltage_source_forms():
    circuit = parse(
        "V1 a 0 1.8\n"
        "V2 b 0 DC 0.9 AC 1 30\n"
        "V3 c 0 DC 0 PULSE(0 1.8 1n 50p 50p 3n 8n)\n"
        "V4 d 0 SIN(0.9 0.01 1MEG)\n"
        "V5 e 0 PWL(0 0 1n 1.8 2n 0)\n"
        "R1 a 0 1k\nR2 b 0 1k\nR3 c 0 1k\nR4 d 0 1k\nR5 e 0 1k\n"
        ".measure v V(a)\n")
    assert circuit.device("V1").dc == pytest.approx(1.8)
    v2 = circuit.device("V2")
    assert (v2.dc, v2.ac_mag, v2.ac_phase) == pytest.approx((0.9, 1.0, 30.0))
    assert isinstance(circuit.device("V3").wave, PulseWave)
    assert isinstance(circuit.device("V4").wave, SinWave)
    assert isinstance(circuit.device("V5").wave, PWLWave)


def test_pulse_waveform_shape():
    circuit = parse("V1 a 0 PULSE(0 1.8 1n 50p 50p 3n 8n)\nR1 a 0 1k\n"
                    ".measure v V(a)\n")
    wave = circuit.device("V1").wave
    assert wave.value(0.0) == pytest.approx(0.0)
    assert wave.value(0.5e-9) == pytest.approx(0.0)
    assert wave.value(1.025e-9) == pytest.approx(0.9, rel=1e-6)   # mid-rise
    assert wave.value(2e-9) == pytest.approx(1.8)
    assert wave.value(4.075e-9) == pytest.approx(0.9, rel=1e-6)   # mid-fall
    assert wave.value(5e-9) == pytest.approx(0.0)


def test_current_source():
    circuit = parse("V1 a 0 1.8\nI1 a b 10u\nR1 b 0 1k\n.measure v V(b)\n")
    i = circuit.device("I1")
    assert isinstance(i, CurrentSource)
    assert i.dc == pytest.approx(1e-5)


def test_mosfet_with_all_attributes():
    circuit = parse(MOS_MODEL +
                    "V1 d 0 1.8\nM1 d g 0 0 NCH W=10u L=1u M=2 MATCH=MIRROR\n"
                    "R1 g 0 1k\n.measure i I(M1)\n")
    m = circuit.device("M1")
    assert isinstance(m, Mosfet)
    assert (m.w, m.l, m.m) == pytest.approx((1e-5, 1e-6, 2.0))
    assert m.matched_group == "MIRROR"
    assert m.model == "NCH"


def test_diode_with_series_resistance_creates_an_internal_node():
    circuit = parse(".model DX D IS=1e-14 RS=5\nV1 a 0 1.8\nR1 a b 10k\n"
                    "D1 b 0 DX\n.measure v V(b)\n")
    diode = circuit.device("D1")
    assert isinstance(diode, Diode)
    # The anode was rewired to an internal node with an explicit series resistor.
    assert circuit.node_names[diode.nodes[0]].endswith("#int")
    assert circuit.device("RD1#rs").r == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# directives
# ---------------------------------------------------------------------------

def test_title_and_temp_and_options():
    circuit = parse("V1 a 0 1\nR1 a 0 1k\n.title My Circuit\n.temp 85\n"
                    ".option gmin=1e-13 reltol=1e-5 max_iter=250\n"
                    ".measure v V(a)\n")
    assert circuit.name == "My Circuit"
    assert circuit.temp_c == pytest.approx(85.0)
    assert circuit.options["gmin"] == pytest.approx(1e-13)
    assert circuit.options["max_iter"] == 250


def test_param_with_expressions():
    circuit = parse("V1 a 0 1\n.param W0=2u\n.param WM={4*W0}\n"
                    "R1 a 0 {10k/2}\n.measure v V(a)\n")
    assert circuit.device("R1").r == pytest.approx(5e3)


def test_param_substitution_into_device_geometry():
    circuit = parse(MOS_MODEL + ".param WM=10u\n.param LM=1u\nV1 d 0 1.8\n"
                    "M1 d d 0 0 NCH W={WM} L={LM}\n.measure i I(M1)\n")
    assert circuit.device("M1").w == pytest.approx(1e-5)


def test_continuation_lines_and_comments():
    circuit = parse(
        "* leading comment\n"
        ".model NCH NMOS VTO=0.45 KP=200u\n"
        "+ LAMBDA=0.1 GAMMA=0.4\n"
        "+ PHI=0.8 AVT=3.5m\n"
        "V1 d 0 1.8   ; inline comment\n"
        "M1 d d 0 0 NCH W=10u L=1u\n"
        ".measure i I(M1)\n")
    model = circuit.mos_models["NCH"]
    assert model.lambda_ == pytest.approx(0.1)
    assert model.phi == pytest.approx(0.8)
    assert model.avt == pytest.approx(3.5e-3)


def test_analysis_cards():
    circuit = parse("V1 a 0 DC 1 AC 1\nR1 a b 1k\nC1 b 0 1n\n"
                    ".op\n.ac dec 20 1 1G\n.tran 1n 1u 0\n.measure v V(b)\n")
    kinds = [a.kind for a in circuit.analyses]
    assert kinds == ["op", "ac", "tran"]
    ac = next(a for a in circuit.analyses if a.kind == "ac")
    assert ac.args == {"sweep": "dec", "points": 20, "fstart": 1.0, "fstop": 1e9}
    tran = next(a for a in circuit.analyses if a.kind == "tran")
    assert tran.args["tstep"] == pytest.approx(1e-9)


def test_measure_shorthands():
    circuit = parse(MOS_MODEL + "V1 d 0 1.8\nM1 d d 0 0 NCH W=10u L=1u\n"
                    "R1 d out 1k\nR2 out 0 1k\n"
                    ".measure vo V(out)\n.measure vdiff V(out,d)\n"
                    ".measure im I(M1)\n.measure pr P(R1)\n.measure pt P(total)\n")
    kinds = {m.name: m.kind for m in circuit.measures}
    assert kinds == {"vo": "v", "vdiff": "v", "im": "i", "pr": "p", "pt": "p"}
    vdiff = next(m for m in circuit.measures if m.name == "vdiff")
    assert vdiff.args == {"node": "out", "node_neg": "d"}
    assert next(m for m in circuit.measures if m.name == "pt").args == {"total": True}


def test_measure_units_are_inferred():
    circuit = parse("V1 a 0 DC 1 AC 1\nR1 a b 1k\nC1 b 0 1n\n"
                    ".measure v V(b)\n.measure i I(R1)\n.measure p P(total)\n"
                    ".measure g GAIN in=a out=b\n"
                    ".measure glin GAIN in=a out=b units=lin\n"
                    ".measure f BW in=a out=b\n")
    units = {m.name: m.unit for m in circuit.measures}
    assert units == {"v": "V", "i": "A", "p": "W", "g": "dB",
                     "glin": "V/V", "f": "Hz"}


def test_expr_measure_keeps_the_raw_expression():
    circuit = parse("V1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n"
                    ".measure va V(a)\n.measure vb V(b)\n"
                    ".measure ratio EXPR 100*vb/va\n")
    expr = next(m for m in circuit.measures if m.name == "ratio")
    assert expr.args["expression"] == "100*vb/va"


def test_spec_parsing_including_percent():
    circuit = parse("V1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n"
                    ".measure vb V(b)\n.measure err EXPR 100*(vb-0.5)/0.5\n"
                    ".spec vb <= 0.6\n.spec err >= -2%\n"
                    '.spec vb >= 0.4 "lower rail"\n')
    specs = circuit.specs
    assert len(specs) == 3
    assert specs[0].measure == "vb" and specs[0].op == "<=" and specs[0].value == pytest.approx(0.6)
    assert specs[1].value == pytest.approx(-2.0) and specs[1].unit == "%"
    assert specs[2].label == "lower rail"


def test_spec_pass_logic():
    spec = SpecLimit(measure="x", op="<=", value=2.0)
    assert spec.passes(1.9) and spec.passes(2.0)
    assert not spec.passes(2.1)
    assert not spec.passes(float("nan"))


# ---------------------------------------------------------------------------
# .include
# ---------------------------------------------------------------------------

def test_include_works_from_a_file(tmp_path):
    (tmp_path / "models.lib").write_text(MOS_MODEL, encoding="utf-8")
    path = write_netlist(tmp_path,
                         ".include models.lib\nV1 d 0 1.8\n"
                         "M1 d d 0 0 NCH W=10u L=1u\n.measure i I(M1)\n")
    circuit = parse_netlist_file(path)
    assert "NCH" in circuit.mos_models


def test_include_is_refused_for_inline_text():
    """Uploaded netlists must not be able to read the server's filesystem."""
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse_netlist(".include /etc/passwd\nV1 a 0 1\nR1 a 0 1k\n")
    assert "disabled" in str(excinfo.value)


def test_include_of_a_missing_file_reports_the_path(tmp_path):
    path = write_netlist(tmp_path, ".include nope.lib\nV1 a 0 1\nR1 a 0 1k\n")
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse_netlist_file(path)
    assert "not found" in str(excinfo.value)


# ---------------------------------------------------------------------------
# error paths -- each must be specific and helpful
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,fragment", [
    ("R1 a b\n", "resistance value"),
    ("R1 a\n", "2 node names"),
    ("Q1 a b c\n", "unknown element type"),
    ("C1 a b\n", "capacitance value"),
    ("L1 a b\n", "inductance value"),
    ("M1 a b c d NCH W=1u L=1u\n", "undefined MOSFET model"),
    (MOS_MODEL + "M1 a b c d NCH W=1u\n", "L="),
    (MOS_MODEL + "M1 a b c d NCH\n", "W="),
    ("D1 a b DX\n", "undefined diode model"),
    (".model NCH BJT\n", "unsupported model type"),
    (".model NCH NMOS BOGUS=1\n", "unknown MOSFET model parameter"),
    (".bogus 1\n", "unknown directive"),
    (".param\n", "NAME=VALUE"),
    (".temp\n", "temperature"),
    (".ac dec 20\n", "fstart"),
    (".ac bogus 20 1 1G\n", "unknown AC sweep type"),
    (".tran 1n\n", "tstep"),
    ("+ continued\n", "nothing to continue"),
])
def test_malformed_netlists_produce_specific_errors(text, fragment):
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse(text)
    assert fragment in str(excinfo.value)


def test_error_reports_line_number_and_text():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 1\nR1 a 0 1k\nR2 b c\n.measure v V(a)\n")
    message = str(excinfo.value)
    assert "line 3" in message
    assert "R2 b c" in message


def test_duplicate_device_name_is_rejected():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 1\nR1 a 0 1k\nR1 a 0 2k\n.measure v V(a)\n")
    assert "duplicate" in str(excinfo.value)


def test_measurement_on_an_unknown_node_is_rejected():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 1\nR1 a 0 1k\n.measure v V(nowhere)\n")
    assert "nowhere" in str(excinfo.value)


def test_measurement_on_an_unknown_device_is_rejected():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 1\nR1 a 0 1k\n.measure i I(R9)\n")
    assert "R9" in str(excinfo.value)


def test_spec_on_an_unknown_measurement_is_rejected():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 1\nR1 a 0 1k\n.measure v V(a)\n.spec nope <= 1\n")
    assert "nope" in str(excinfo.value)


def test_spec_with_a_bad_operator_is_rejected():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 1\nR1 a 0 1k\n.measure v V(a)\n.spec v != 1\n")
    assert "operator" in str(excinfo.value)


def test_gain_measurement_missing_arguments_is_rejected():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 DC 1 AC 1\nR1 a b 1k\nC1 b 0 1n\n"
              ".measure g GAIN out=b\n")
    assert "in=" in str(excinfo.value)


def test_offset_measurement_requires_voltage_sources():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n"
              ".measure vos VOS srcp=R1 srcn=R2 outp=b\n")
    assert "voltage source" in str(excinfo.value)


def test_empty_netlist_is_rejected():
    with pytest.raises(NetlistSyntaxError):
        parse("* only a comment\n")


def test_negative_resistance_is_rejected():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse("V1 a 0 1\nR1 a 0 -1k\n.measure v V(a)\n")
    assert "must be > 0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# the shipped examples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_every_example_parses_and_finalises(name):
    circuit = load(name)
    assert circuit.finalized
    assert circuit.devices
    assert circuit.measures
    description = circuit.describe()
    assert description["name"]
    assert len(description["devices"]) == len(circuit.devices)


def test_examples_are_self_contained():
    """Examples must parse as inline text too, i.e. without .include."""
    from conftest import example_path

    for name in EXAMPLE_NAMES:
        text = example_path(name).read_text(encoding="utf-8")
        circuit = parse_netlist(text, source=name, allow_include=False)
        assert circuit.devices, name


def test_ground_aliases_all_map_to_node_zero():
    circuit = parse("V1 a 0 1\nR1 a gnd 1k\nR2 a GND 1k\nR3 a vss 1k\n"
                    ".measure v V(a)\n")
    for name in ("R1", "R2", "R3"):
        assert circuit.device(name).nodes[1] == 0
