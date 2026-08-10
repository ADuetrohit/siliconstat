"""SPICE-like netlist parser.

Supported syntax
----------------
::

    * a comment                       ; also a comment
    .title Two-stage OTA
    .param IB=10u  WN={4*1u}
    .model NMOS NMOS VTO=0.45 KP=200u LAMBDA=0.1 GAMMA=0.4 PHI=0.8 AVT=3.5m
    .option gmin=1e-12 reltol=1e-4
    .temp 27

    VDD  vdd 0 1.8
    VIN  in  0 DC 0.9 AC 1 SIN(0.9 0.01 1MEG)
    IREF vdd nref 10u
    R1   out 0 10k
    C1   out 0 1p
    M1   out in vdd vdd PMOS W=10u L=180n MATCH=MIRROR1 M=2
    D1   a   0 DMOD AREA=2

    .op
    .ac dec 20 1 1G
    .tran 10n 2u

    .measure iout  I(M2)
    .measure vout  V(out)
    .measure ptot  P(total)
    .measure ierr  EXPR 100*(iout-iref)/iref
    .measure av    GAIN in=in out=out
    .measure vos   VOS srcp=VINP srcn=VINN outp=outp outn=outn

    .spec ierr <= 2%
    .spec av   >= 60
    .end

Errors carry the offending line number and text.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Sequence

from .circuit import AnalysisSpec, Circuit, MeasureSpec, SpecLimit
from .devices import (
    Capacitor,
    CurrentSource,
    Diode,
    Inductor,
    Mosfet,
    Resistor,
    VoltageSource,
)
from .exceptions import CircuitError, NetlistSyntaxError
from .expr import ExpressionError, safe_eval
from .models import DIODE_PARAM_ALIASES, MOS_PARAM_ALIASES, DiodeModel, MosfetModel
from .units import expand_spice_numbers, parse_value
from .waveforms import DCWave, PulseWave, PWLWave, SinWave, Waveform

__all__ = ["parse_netlist", "parse_netlist_file"]

_SHORTHAND_RE = re.compile(r"^([vip])\s*\(\s*([^)]+?)\s*\)$", re.IGNORECASE)
_BRACE_RE = re.compile(r"\{([^{}]*)\}")

_MEASURE_UNITS = {
    "v": "V", "i": "A", "p": "W", "gain": "dB", "bw": "Hz", "ugf": "Hz",
    "pm": "deg", "gm": "dB", "vos": "V", "risetime": "s", "falltime": "s",
    "settling": "s", "overshoot": "%", "expr": "", "delay": "s",
    "vpp": "V", "vmax": "V", "vmin": "V", "vavg": "V",
}

_MEASURE_KINDS = set(_MEASURE_UNITS)


# ---------------------------------------------------------------------------
# Lexing helpers
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    out = []
    depth = 0
    for ch in line:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        if ch == ";" and depth == 0:
            break
        out.append(ch)
    return "".join(out)


def tokenize(line: str) -> list[str]:
    """Split on whitespace/commas while keeping bracketed groups intact."""
    tokens: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in line:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(depth - 1, 0)
            buf.append(ch)
        elif depth == 0 and (ch.isspace() or ch == ","):
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


class _Line:
    __slots__ = ("no", "text")

    def __init__(self, no: int, text: str) -> None:
        self.no = no
        self.text = text


def _logical_lines(text: str) -> list[_Line]:
    """Join ``+`` continuations and drop comments/blank lines."""
    lines: list[_Line] = []
    for raw_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("*"):
            continue
        body = _strip_comment(raw).rstrip()
        if not body.strip():
            continue
        if body.lstrip().startswith("+"):
            if not lines:
                raise NetlistSyntaxError(
                    "continuation line '+' has nothing to continue", raw_no, raw)
            lines[-1].text += " " + body.lstrip()[1:].strip()
        else:
            lines.append(_Line(raw_no, body.strip()))
    return lines


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, text: str, source: str | None, *, allow_include: bool,
                 base_dir: str | None) -> None:
        self.source = source
        self.allow_include = allow_include
        self.base_dir = base_dir
        self.circuit = Circuit(name="circuit")
        self.circuit.source_text = text
        self.circuit.source_path = source
        self.params: dict[str, float] = {}
        self.raw_measures: list[tuple[_Line, list[str], str]] = []
        self.raw_specs: list[tuple[_Line, list[str]]] = []
        self.lines = _logical_lines(text)

    # -- utilities ---------------------------------------------------------
    def err(self, line: _Line, message: str) -> NetlistSyntaxError:
        return NetlistSyntaxError(message, line.no, line.text, self.source)

    def value(self, token: str, line: _Line) -> float:
        """Parse a numeric token, expanding ``{...}`` parameter expressions."""
        token = token.strip()
        if token.startswith("{") and token.endswith("}"):
            try:
                return safe_eval(expand_spice_numbers(token[1:-1]), self.params)
            except ExpressionError as exc:
                raise self.err(line, str(exc)) from exc
        if "{" in token:
            def _sub(m: re.Match[str]) -> str:
                try:
                    return repr(safe_eval(expand_spice_numbers(m.group(1)),
                                          self.params))
                except ExpressionError as exc:
                    raise self.err(line, str(exc)) from exc
            token = _BRACE_RE.sub(_sub, token)
        if token.lower() in self.params:
            return float(self.params[token.lower()])
        try:
            return parse_value(token, line_no=line.no, source=self.source)
        except NetlistSyntaxError as exc:
            raise NetlistSyntaxError(str(exc.message), line.no, line.text, self.source) from exc

    @staticmethod
    def _kv(tokens: Iterable[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for tok in tokens:
            if "=" in tok:
                key, _, val = tok.partition("=")
                out[key.strip().lower()] = val.strip()
        return out

    @staticmethod
    def _positional(tokens: Sequence[str]) -> list[str]:
        return [t for t in tokens if "=" not in t]

    # -- entry point -------------------------------------------------------
    def parse(self) -> Circuit:
        for line in self.lines:
            head = line.text.split(None, 1)[0]
            if head.startswith("."):
                self.directive(line)
            else:
                self.element(line)

        for line, tokens, raw in self.raw_measures:
            self.measure(line, tokens, raw)
        for line, tokens in self.raw_specs:
            self.spec(line, tokens)

        self._expand_diode_series_resistance()
        try:
            self.circuit.finalize()
        except CircuitError as exc:
            raise NetlistSyntaxError(str(exc), source=self.source) from exc
        return self.circuit

    # -- directives --------------------------------------------------------
    def directive(self, line: _Line) -> None:
        tokens = tokenize(line.text)
        name = tokens[0].lower()

        if name in (".end", ".ends"):
            return
        if name == ".title":
            self.circuit.name = line.text[len(tokens[0]):].strip() or self.circuit.name
            return
        if name == ".param":
            self._param(line, tokens[1:])
            return
        if name == ".model":
            self._model(line, tokens[1:])
            return
        if name in (".option", ".options"):
            self._option(line, tokens[1:])
            return
        if name == ".temp":
            if len(tokens) < 2:
                raise self.err(line, ".temp requires a temperature in degrees Celsius")
            self.circuit.temp_c = self.value(tokens[1], line)
            return
        if name == ".include":
            self._include(line, tokens[1:])
            return
        if name == ".op":
            self.circuit.analyses.append(AnalysisSpec("op", {}))
            return
        if name == ".ac":
            self._ac(line, tokens[1:])
            return
        if name == ".tran":
            self._tran(line, tokens[1:])
            return
        if name in (".measure", ".meas"):
            self.raw_measures.append((line, tokens[1:], line.text))
            return
        if name == ".spec":
            self.raw_specs.append((line, tokens[1:]))
            return
        if name == ".global":
            return  # nodes are global by construction (no subcircuits yet)
        raise self.err(line, f"unknown directive {tokens[0]!r}")

    def _param(self, line: _Line, tokens: Sequence[str]) -> None:
        kv = self._kv(tokens)
        if not kv:
            raise self.err(line, ".param requires NAME=VALUE assignments")
        for key, raw in kv.items():
            self.params[key] = self.value(raw, line)

    def _option(self, line: _Line, tokens: Sequence[str]) -> None:
        kv = self._kv(tokens)
        if not kv:
            raise self.err(line, ".option requires KEY=VALUE assignments")
        for key, raw in kv.items():
            if key in ("integration",):
                self.circuit.options[key] = raw.lower()
            elif key in ("limiting",):
                self.circuit.options[key] = raw.lower() not in ("0", "false", "no")
            elif key in ("max_iter", "maxiter", "gmin_steps", "source_steps"):
                self.circuit.options[key.replace("maxiter", "max_iter")] = int(
                    self.value(raw, line))
            else:
                self.circuit.options[key] = self.value(raw, line)

    def _include(self, line: _Line, tokens: Sequence[str]) -> None:
        if not self.allow_include:
            raise self.err(line, ".include is disabled when parsing netlist text "
                                 "(file access is not permitted in this context)")
        if not tokens:
            raise self.err(line, ".include requires a file name")
        target = tokens[0].strip('"\'')
        path = target if os.path.isabs(target) else os.path.join(self.base_dir or ".", target)
        if not os.path.isfile(path):
            raise self.err(line, f"included file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            included = _logical_lines(fh.read())
        # Splice the included lines in place; nested includes are honoured.
        for inc_line in included:
            head = inc_line.text.split(None, 1)[0]
            if head.startswith("."):
                self.directive(inc_line)
            else:
                self.element(inc_line)

    def _model(self, line: _Line, tokens: Sequence[str]) -> None:
        if len(tokens) < 2:
            raise self.err(line, ".model requires a name and a type "
                                 "(e.g. '.model NCH NMOS VTO=0.45')")
        name = tokens[0]
        mtype = tokens[1].lower()
        kv = self._kv(tokens[2:])
        if mtype in ("nmos", "pmos"):
            kwargs: dict[str, Any] = {}
            for key, raw in kv.items():
                field = MOS_PARAM_ALIASES.get(key)
                if field is None:
                    raise self.err(
                        line, f"unknown MOSFET model parameter {key.upper()!r}; known: "
                              f"{', '.join(sorted(k.upper() for k in MOS_PARAM_ALIASES))}")
                kwargs[field] = self.value(raw, line)
            try:
                model = MosfetModel(name=name, mtype=mtype, **kwargs)
            except NetlistSyntaxError as exc:
                raise self.err(line, str(exc)) from exc
            self.circuit.mos_models[name] = model
        elif mtype in ("d", "diode"):
            kwargs = {}
            for key, raw in kv.items():
                field = DIODE_PARAM_ALIASES.get(key)
                if field is None:
                    raise self.err(line, f"unknown diode model parameter {key.upper()!r}")
                kwargs[field] = self.value(raw, line)
            try:
                self.circuit.diode_models[name] = DiodeModel(name=name, **kwargs)
            except NetlistSyntaxError as exc:
                raise self.err(line, str(exc)) from exc
        else:
            raise self.err(line, f"unsupported model type {tokens[1]!r} "
                                 "(expected NMOS, PMOS or D)")

    def _ac(self, line: _Line, tokens: Sequence[str]) -> None:
        if len(tokens) < 4:
            raise self.err(line, ".ac requires: <dec|oct|lin> <points> <fstart> <fstop>")
        sweep = tokens[0].lower()
        if sweep not in ("dec", "oct", "lin"):
            raise self.err(line, f"unknown AC sweep type {tokens[0]!r} (dec, oct or lin)")
        points = int(self.value(tokens[1], line))
        fstart = self.value(tokens[2], line)
        fstop = self.value(tokens[3], line)
        if points < 1:
            raise self.err(line, ".ac point count must be >= 1")
        if fstart <= 0 or fstop <= 0 or fstop < fstart:
            raise self.err(line, ".ac frequency range must satisfy 0 < fstart <= fstop")
        self.circuit.analyses.append(AnalysisSpec(
            "ac", {"sweep": sweep, "points": points, "fstart": fstart, "fstop": fstop}))

    def _tran(self, line: _Line, tokens: Sequence[str]) -> None:
        if len(tokens) < 2:
            raise self.err(line, ".tran requires: <tstep> <tstop> [tstart]")
        tstep = self.value(tokens[0], line)
        tstop = self.value(tokens[1], line)
        tstart = self.value(tokens[2], line) if len(tokens) > 2 else 0.0
        if tstep <= 0 or tstop <= tstart:
            raise self.err(line, ".tran requires tstep > 0 and tstop > tstart")
        self.circuit.analyses.append(AnalysisSpec(
            "tran", {"tstep": tstep, "tstop": tstop, "tstart": tstart}))

    # -- elements ----------------------------------------------------------
    def element(self, line: _Line) -> None:
        tokens = tokenize(line.text)
        name = tokens[0]
        kind = name[0].upper()
        handler = {
            "R": self._resistor, "C": self._capacitor, "L": self._inductor,
            "V": self._vsource, "I": self._isource, "M": self._mosfet,
            "D": self._diode,
        }.get(kind)
        if handler is None:
            raise self.err(
                line, f"unknown element type {name[0]!r} in {name!r}; supported first "
                      "letters are R, C, L, V, I, M, D")
        try:
            device = handler(line, tokens)
        except CircuitError as exc:
            raise self.err(line, str(exc)) from exc
        try:
            self.circuit.add_device(device)
        except CircuitError as exc:
            raise self.err(line, str(exc)) from exc

    def _nodes(self, line: _Line, tokens: Sequence[str], count: int,
               what: str) -> tuple[int, ...]:
        if len(tokens) < 1 + count:
            raise self.err(line, f"{what} requires {count} node names")
        names = tokens[1:1 + count]
        for n in names:
            if "=" in n or n.startswith("("):
                raise self.err(line, f"{n!r} is not a valid node name")
        return tuple(self.circuit.node(n) for n in names)

    def _resistor(self, line: _Line, tokens: Sequence[str]) -> Resistor:
        nodes = self._nodes(line, tokens, 2, "resistor")
        rest = self._positional(tokens[3:])
        kv = self._kv(tokens[3:])
        if rest:
            r = self.value(rest[0], line)
        elif "r" in kv:
            r = self.value(kv["r"], line)
        else:
            raise self.err(line, "resistor requires a resistance value")
        return Resistor(name=tokens[0], nodes=nodes, r=r)

    def _capacitor(self, line: _Line, tokens: Sequence[str]) -> Capacitor:
        nodes = self._nodes(line, tokens, 2, "capacitor")
        rest = self._positional(tokens[3:])
        kv = self._kv(tokens[3:])
        if rest:
            c = self.value(rest[0], line)
        elif "c" in kv:
            c = self.value(kv["c"], line)
        else:
            raise self.err(line, "capacitor requires a capacitance value")
        ic = self.value(kv["ic"], line) if "ic" in kv else None
        return Capacitor(name=tokens[0], nodes=nodes, c=c, ic=ic)

    def _inductor(self, line: _Line, tokens: Sequence[str]) -> Inductor:
        nodes = self._nodes(line, tokens, 2, "inductor")
        rest = self._positional(tokens[3:])
        if not rest:
            raise self.err(line, "inductor requires an inductance value")
        return Inductor(name=tokens[0], nodes=nodes, l=self.value(rest[0], line))

    def _source_common(self, line: _Line, tokens: Sequence[str]
                       ) -> tuple[float, float, float, Waveform | None]:
        """Parse ``[DC] value [AC mag [phase]] [PULSE(...)|SIN(...)|PWL(...)]``."""
        rest = list(tokens[3:])
        kv = self._kv(rest)
        positional = [t for t in rest if "=" not in t]
        dc = 0.0
        ac_mag = 0.0
        ac_phase = 0.0
        wave: Waveform | None = None
        i = 0
        seen_dc = False
        while i < len(positional):
            tok = positional[i]
            upper = tok.upper()
            if upper == "DC":
                if i + 1 >= len(positional):
                    raise self.err(line, "DC keyword must be followed by a value")
                dc = self.value(positional[i + 1], line)
                seen_dc = True
                i += 2
                continue
            if upper == "AC":
                if i + 1 >= len(positional):
                    raise self.err(line, "AC keyword must be followed by a magnitude")
                ac_mag = self.value(positional[i + 1], line)
                i += 2
                if i < len(positional) and not _is_waveform(positional[i]):
                    try:
                        ac_phase = self.value(positional[i], line)
                        i += 1
                    except NetlistSyntaxError:
                        pass
                continue
            if _is_waveform(tok):
                wave = self._waveform(line, tok)
                i += 1
                continue
            if not seen_dc:
                dc = self.value(tok, line)
                seen_dc = True
                i += 1
                continue
            raise self.err(line, f"unexpected token {tok!r} in source definition")
        if "dc" in kv:
            dc = self.value(kv["dc"], line)
        if "ac" in kv:
            ac_mag = self.value(kv["ac"], line)
        if wave is None:
            wave = DCWave(dc)
        return dc, ac_mag, ac_phase, wave

    def _waveform(self, line: _Line, token: str) -> Waveform:
        head, _, body = token.partition("(")
        body = body.rstrip(")")
        args = [a for a in re.split(r"[\s,]+", body.strip()) if a]
        vals = [self.value(a, line) for a in args]
        kind = head.upper()
        if kind == "PULSE":
            if len(vals) < 2:
                raise self.err(line, "PULSE requires at least (v1 v2)")
            defaults = [0.0, 1.0, 0.0, 1e-9, 1e-9, 1e-6, 2e-6]
            merged = vals + defaults[len(vals):]
            return PulseWave(*merged[:7])
        if kind == "SIN":
            if len(vals) < 3:
                raise self.err(line, "SIN requires at least (vo va freq)")
            defaults = [0.0, 1.0, 1e6, 0.0, 0.0]
            merged = vals + defaults[len(vals):]
            return SinWave(*merged[:5])
        if kind == "PWL":
            if len(vals) < 2 or len(vals) % 2:
                raise self.err(line, "PWL requires an even number of (time value) entries")
            return PWLWave(points=tuple(zip(vals[0::2], vals[1::2])))
        raise self.err(line, f"unknown source waveform {head!r}")

    def _vsource(self, line: _Line, tokens: Sequence[str]) -> VoltageSource:
        nodes = self._nodes(line, tokens, 2, "voltage source")
        dc, ac_mag, ac_phase, wave = self._source_common(line, tokens)
        return VoltageSource(name=tokens[0], nodes=nodes, dc=dc, ac_mag=ac_mag,
                             ac_phase=ac_phase, wave=wave)

    def _isource(self, line: _Line, tokens: Sequence[str]) -> CurrentSource:
        nodes = self._nodes(line, tokens, 2, "current source")
        dc, ac_mag, ac_phase, wave = self._source_common(line, tokens)
        return CurrentSource(name=tokens[0], nodes=nodes, dc=dc, ac_mag=ac_mag,
                             ac_phase=ac_phase, wave=wave)

    def _mosfet(self, line: _Line, tokens: Sequence[str]) -> Mosfet:
        nodes = self._nodes(line, tokens, 4, "mosfet")
        rest = tokens[5:]
        positional = self._positional(rest)
        if not positional:
            raise self.err(line, "mosfet requires a model name after its four nodes")
        model = positional[0]
        if model not in self.circuit.mos_models:
            raise self.err(
                line, f"undefined MOSFET model {model!r}; declare it with a .model card "
                      f"(defined: {', '.join(sorted(self.circuit.mos_models)) or 'none'})")
        kv = self._kv(rest)
        if "w" not in kv or "l" not in kv:
            raise self.err(line, "mosfet requires W= and L= (e.g. 'W=10u L=180n')")
        match = kv.get("match") or kv.get("matched_group") or kv.get("group")
        return Mosfet(
            name=tokens[0], nodes=nodes, model=model,
            w=self.value(kv["w"], line), l=self.value(kv["l"], line),
            m=self.value(kv["m"], line) if "m" in kv else 1.0,
            matched_group=match.upper() if match else None,
        )

    def _diode(self, line: _Line, tokens: Sequence[str]) -> Diode:
        nodes = self._nodes(line, tokens, 2, "diode")
        positional = self._positional(tokens[3:])
        if not positional:
            raise self.err(line, "diode requires a model name after its two nodes")
        model = positional[0]
        if model not in self.circuit.diode_models:
            raise self.err(line, f"undefined diode model {model!r}")
        kv = self._kv(tokens[3:])
        area = self.value(kv["area"], line) if "area" in kv else 1.0
        return Diode(name=tokens[0], nodes=nodes, model=model, area=area)

    def _expand_diode_series_resistance(self) -> None:
        """Realise ``RS`` as an explicit resistor on an internal node."""
        additions: list[tuple[Diode, Resistor]] = []
        for dev in list(self.circuit.devices):
            if not isinstance(dev, Diode):
                continue
            rs = self.circuit.diode_models[dev.model].rs
            if rs <= 0:
                continue
            internal = self.circuit.node(f"{dev.name}#int")
            new_diode = Diode(name=dev.name, nodes=(internal, dev.nodes[1]),
                              model=dev.model, area=dev.area)
            res = Resistor(name=f"R{dev.name}#rs", nodes=(dev.nodes[0], internal), r=rs)
            additions.append((new_diode, res))
            idx = self.circuit.devices.index(dev)
            self.circuit.devices[idx] = new_diode
        for _diode, res in additions:
            self.circuit.add_device(res)

    # -- measurements and specifications -----------------------------------
    def measure(self, line: _Line, tokens: Sequence[str], raw: str) -> None:
        if len(tokens) < 2:
            raise self.err(line, ".measure requires a name and a definition")
        name = tokens[0].lower()
        spec_token = tokens[1]
        kv = self._kv(tokens[2:])
        unit_override = kv.pop("unit", None)

        shorthand = _SHORTHAND_RE.match(spec_token)
        if shorthand:
            kind = shorthand.group(1).lower()
            target = shorthand.group(2).strip()
            args = self._measure_target_args(line, kind, target)
        else:
            kind = spec_token.lower()
            if kind not in _MEASURE_KINDS:
                raise self.err(
                    line, f"unknown measurement kind {spec_token!r}; use V(node), "
                          f"I(device), P(device|total) or one of "
                          f"{', '.join(sorted(k.upper() for k in _MEASURE_KINDS))}")
            if kind == "expr":
                idx = raw.lower().find(" expr ")
                if idx < 0:
                    raise self.err(line, "EXPR measurement requires an expression")
                expression = raw[idx + len(" expr "):].strip()
                if not expression:
                    raise self.err(line, "EXPR measurement requires an expression")
                # Expand engineering notation once, at parse time, so that
                # '100*(iout-10u)/10u' works and the per-sample evaluation
                # stays a pure arithmetic walk.
                args = {"expression": expand_spice_numbers(expression),
                        "source_expression": expression}
            else:
                args = {k: v for k, v in kv.items()}
                self._validate_measure_args(line, kind, args)

        if unit_override is not None:
            unit = unit_override
        elif kind == "gain" and str(args.get("units", "")).lower() in ("lin", "linear", "v/v"):
            unit = "V/V"
        else:
            unit = _MEASURE_UNITS.get(kind, "")
        self.circuit.measures.append(
            MeasureSpec(name=name, kind=kind, args=args, unit=unit))

    def _measure_target_args(self, line: _Line, kind: str, target: str) -> dict[str, Any]:
        if kind == "v":
            if "," in target:
                a, _, b = target.partition(",")
                if not self.circuit.has_node(a.strip()) or not self.circuit.has_node(b.strip()):
                    raise self.err(line, f"V({target}) references an unknown node")
                return {"node": a.strip().lower(), "node_neg": b.strip().lower()}
            if not self.circuit.has_node(target):
                raise self.err(
                    line, f"V({target}) references unknown node {target!r}; known nodes: "
                          f"{', '.join(sorted(self.circuit.node_index))}")
            return {"node": target.lower()}
        if kind == "i":
            try:
                self.circuit.device(target)
            except CircuitError as exc:
                raise self.err(line, str(exc)) from exc
            return {"device": target}
        # power
        if target.lower() in ("total", "all", "supply"):
            return {"total": True}
        try:
            self.circuit.device(target)
        except CircuitError as exc:
            raise self.err(line, str(exc)) from exc
        return {"device": target}

    def _validate_measure_args(self, line: _Line, kind: str, args: dict[str, Any]) -> None:
        required = {
            "gain": ("in", "out"), "bw": ("in", "out"), "ugf": ("in", "out"),
            "pm": ("in", "out"), "gm": ("in", "out"),
            "vos": ("srcp", "srcn", "outp"),
            "risetime": ("node",), "falltime": ("node",), "settling": ("node",),
            "overshoot": ("node",), "delay": ("node",),
            "vpp": ("node",), "vmax": ("node",), "vmin": ("node",), "vavg": ("node",),
        }.get(kind, ())
        missing = [r for r in required if r not in args]
        if missing:
            raise self.err(
                line, f"{kind.upper()} measurement is missing required argument(s): "
                      f"{', '.join(m + '=' for m in missing)}")
        for key in ("in", "out", "outp", "outn", "node"):
            if key in args and not self.circuit.has_node(str(args[key])):
                raise self.err(line, f"{key}={args[key]!r} is not a node in this circuit")
            if key in args:
                args[key] = str(args[key]).lower()
        for key in ("srcp", "srcn"):
            if key in args:
                try:
                    dev = self.circuit.device(str(args[key]))
                except CircuitError as exc:
                    raise self.err(line, str(exc)) from exc
                if not isinstance(dev, VoltageSource):
                    raise self.err(
                        line, f"{key}={args[key]!r} must name a voltage source "
                              "(the offset measurement perturbs it)")
        for key in ("delta", "lo", "hi", "tolerance", "target", "tstart", "tstop"):
            if key in args:
                args[key] = self.value(str(args[key]).rstrip("%"), line)

    def spec(self, line: _Line, tokens: Sequence[str]) -> None:
        if len(tokens) < 3:
            raise self.err(line, ".spec requires: <measurement> <op> <value>")
        measure = tokens[0].lower()
        op = tokens[1]
        if op not in SpecLimit.VALID_OPS:
            raise self.err(
                line, f"unknown comparison operator {op!r}; use one of "
                      f"{', '.join(SpecLimit.VALID_OPS)}")
        raw_value = tokens[2]
        unit = ""
        if raw_value.endswith("%"):
            value = self.value(raw_value[:-1], line)
            unit = "%"
        else:
            value = self.value(raw_value, line)
            known = {m.name: m.unit for m in self.circuit.measures}
            unit = known.get(measure, "")
        label = " ".join(tokens[3:]).strip('"\'') if len(tokens) > 3 else ""
        if measure not in {m.name for m in self.circuit.measures}:
            raise self.err(
                line, f"specification refers to unknown measurement {measure!r}; "
                      f"declared: {', '.join(sorted(m.name for m in self.circuit.measures))}")
        self.circuit.specs.append(
            SpecLimit(measure=measure, op=op, value=value, unit=unit, label=label))


def _is_waveform(token: str) -> bool:
    head = token.split("(", 1)[0].upper()
    return "(" in token and head in ("PULSE", "SIN", "SINE", "PWL", "EXP")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_netlist(text: str, source: str | None = None, *,
                  allow_include: bool = False, base_dir: str | None = None) -> Circuit:
    """Parse netlist *text* into a finalized :class:`Circuit`.

    ``allow_include`` defaults to *False*: netlist text arriving over the REST
    API must not be able to read arbitrary files from the server.
    """
    return _Parser(text, source, allow_include=allow_include, base_dir=base_dir).parse()


def parse_netlist_file(path: str) -> Circuit:
    """Parse a netlist file, enabling ``.include`` relative to its directory."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    circuit = _Parser(text, os.path.basename(path), allow_include=True,
                      base_dir=os.path.dirname(os.path.abspath(path))).parse()
    circuit.source_path = os.path.abspath(path)
    return circuit
