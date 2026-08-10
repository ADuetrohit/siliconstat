"""SPICE engineering-notation value parsing and formatting.

SPICE suffix rules implemented here (case-insensitive, the classic Berkeley
set).  Note the well-known trap: ``M`` means *milli*, ``MEG`` means *mega*.

    T   1e12      G   1e9       MEG 1e6       X   1e6
    K   1e3       M   1e-3      MIL 25.4e-6   U   1e-6
    N   1e-9      P   1e-12     F   1e-15     A   1e-18

Any trailing alphabetic characters after a recognised suffix are treated as a
unit annotation and ignored, so ``10kOhm``, ``1.8V`` and ``180nm`` all parse.
"""

from __future__ import annotations

import math
import re

from .exceptions import NetlistSyntaxError

# Longest-first so that MEG/MIL are matched before M.
_SUFFIXES: list[tuple[str, float]] = [
    ("meg", 1e6),
    ("mil", 25.4e-6),
    ("t", 1e12),
    ("g", 1e9),
    ("x", 1e6),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("µ", 1e-6),  # micro sign
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
    ("a", 1e-18),
]

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")

_SI_PREFIXES = [
    (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
    (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"),
    (1e-12, "p"), (1e-15, "f"),
]


def parse_value(text: str, *, line_no: int | None = None, source: str | None = None) -> float:
    """Parse a SPICE numeric literal such as ``10k``, ``1.8``, ``180n``, ``2.5e-3``.

    Raises
    ------
    NetlistSyntaxError
        If *text* is not a valid engineering-notation number.
    """
    raw = text.strip()
    if not raw:
        raise NetlistSyntaxError("empty numeric value", line_no, text, source)

    match = _NUMBER_RE.match(raw)
    if not match:
        raise NetlistSyntaxError(f"cannot parse numeric value {raw!r}", line_no, text, source)

    mantissa = float(match.group(0))
    rest = raw[match.end():].strip().lower()
    if not rest:
        return mantissa

    if rest.startswith("%"):
        # Percentages are handled by parse_value_or_percent; here treat as literal.
        raise NetlistSyntaxError(
            f"percentage {raw!r} is not allowed in this position", line_no, text, source
        )

    for suffix, scale in _SUFFIXES:
        if rest.startswith(suffix):
            trailing = rest[len(suffix):]
            if trailing and not trailing.isalpha():
                raise NetlistSyntaxError(
                    f"unexpected characters {trailing!r} in value {raw!r}", line_no, text, source
                )
            return mantissa * scale

    # No recognised suffix: allow a pure unit annotation such as "1.8V".
    if rest.isalpha():
        return mantissa
    raise NetlistSyntaxError(f"unknown unit suffix in value {raw!r}", line_no, text, source)


_EMBEDDED_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"                    # not part of an identifier
    r"(\d+\.?\d*|\.\d+)"                     # mantissa
    r"([eE][+-]?\d+)?"                       # optional exponent
    r"([A-Za-zµ]+)?"                         # optional SPICE suffix
)


def expand_spice_numbers(text: str) -> str:
    """Rewrite SPICE-suffixed literals inside an expression as plain floats.

    ``safe_eval`` parses expressions with Python's own grammar, which has no
    idea that ``10k`` means ten thousand.  Expanding first lets netlist
    expressions use engineering notation naturally::

        {10k/2}                 ->  {5000.0}
        100*(iout-10u)/10u      ->  100*(iout-1e-05)/1e-05

    Identifiers are left alone: the lookbehind refuses to match a digit that
    follows a letter, underscore or dot, so ``W0``, ``pi`` and ``iref`` survive
    untouched.  A trailing alphabetic run that is not a recognised suffix is
    also left alone, so ``2*x`` and ``1e3`` are unaffected.
    """

    def _replace(match: re.Match[str]) -> str:
        mantissa, exponent, suffix = match.group(1), match.group(2) or "", match.group(3)
        literal = mantissa + exponent
        if not suffix:
            return literal
        lowered = suffix.lower()
        for name, scale in _SUFFIXES:
            if lowered.startswith(name):
                trailing = suffix[len(name):]
                if trailing and not trailing.isalpha():
                    return match.group(0)
                return repr(float(literal) * scale)
        return match.group(0)

    return _EMBEDDED_NUMBER_RE.sub(_replace, text)


def parse_value_or_percent(text: str, *, line_no: int | None = None,
                           source: str | None = None) -> tuple[float, bool]:
    """Parse a value that may be written as a percentage.

    Returns ``(value, is_percent)``.  For ``"2%"`` returns ``(0.02, True)``;
    the caller decides whether a relative interpretation is meaningful.
    """
    raw = text.strip()
    if raw.endswith("%"):
        body = raw[:-1].strip()
        match = _NUMBER_RE.match(body)
        if not match or match.end() != len(body):
            raise NetlistSyntaxError(f"cannot parse percentage {raw!r}", line_no, text, source)
        return float(body) / 100.0, True
    return parse_value(raw, line_no=line_no, source=source), False


def format_eng(value: float, unit: str = "", digits: int = 4, *,
               spice: bool = False) -> str:
    """Format *value* in engineering notation, e.g. ``1.234 mA``.

    Two conventions collide on the letter ``M``: SI reads it as *mega*, SPICE
    reads it as *milli*.  This project keeps both, explicitly:

    * ``spice=False`` (default) produces **human-readable** output using the SI
      convention -- ``format_eng(24.5e6, "Hz") == "24.5 MHz"``.  This is what
      the CLI, the HTML report and the UI display.
    * ``spice=True`` produces text that :func:`parse_value` reads back to the
      same number -- ``"24.5MEG"`` -- for anything that will be written into a
      netlist.

    Feeding default (display) output back into :func:`parse_value` would
    silently turn megas into millis, so use ``spice=True`` whenever the result
    is machine-read.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan" + (f" {unit}" if unit else "")
    if value == 0:
        return f"0{unit}" if spice else f"0 {unit}".strip()
    if math.isinf(value):
        return ("inf" if value > 0 else "-inf") + (f" {unit}" if unit else "")
    magnitude = abs(value)
    for scale, prefix in _SI_PREFIXES:
        if magnitude >= scale * 0.999:
            scaled = value / scale
            if spice:
                return f"{scaled:.{digits}g}{'MEG' if prefix == 'M' else prefix}{unit}"
            return f"{scaled:.{digits}g} {prefix}{unit}".strip()
    scaled = value / 1e-15
    if spice:
        return f"{scaled:.{digits}g}f{unit}"
    return f"{scaled:.{digits}g} f{unit}".strip()
