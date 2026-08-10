"""Terminal formatting helpers for the CLI."""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Iterable, Sequence

__all__ = ["colour", "supports_colour", "table", "rule", "kv", "bar", "C"]


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"


def supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


_ENABLED = supports_colour()


def colour(text: str, code: str) -> str:
    return f"{code}{text}{C.RESET}" if _ENABLED else text


def rule(title: str = "", width: int | None = None) -> str:
    width = width or min(shutil.get_terminal_size((100, 24)).columns, 100)
    if not title:
        return colour("-" * width, C.GREY)
    head = f"-- {title} "
    return colour(head + "-" * max(width - len(head), 0), C.CYAN)


def _visible_len(text: str) -> int:
    out = 0
    skip = False
    for ch in text:
        if ch == "\033":
            skip = True
            continue
        if skip:
            if ch == "m":
                skip = False
            continue
        out += 1
    return out


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *,
          align: str = "", indent: int = 2) -> str:
    """Render an aligned text table.

    ``align`` is a per-column string of ``l``/``r`` characters.
    """
    data = [[str(c) for c in row] for row in rows]
    if not data:
        return " " * indent + colour("(no rows)", C.GREY)
    widths = [_visible_len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], _visible_len(cell))
    align = (align + "l" * len(headers))[:len(headers)]

    def fmt(cells: Sequence[str], style: str | None = None) -> str:
        parts = []
        for i, cell in enumerate(cells):
            pad = widths[i] - _visible_len(cell)
            text = (" " * pad + cell) if align[i] == "r" else (cell + " " * pad)
            parts.append(text)
        line = " " * indent + "  ".join(parts).rstrip()
        return colour(line, style) if style else line

    lines = [fmt(headers, C.BOLD),
             " " * indent + colour("  ".join("-" * w for w in widths), C.GREY)]
    lines.extend(fmt(row) for row in data)
    return "\n".join(lines)


def kv(pairs: Sequence[tuple[str, Any]], indent: int = 2) -> str:
    width = max((len(k) for k, _ in pairs), default=0)
    return "\n".join(
        " " * indent + colour(f"{k:<{width}}", C.GREY) + "  " + str(v)
        for k, v in pairs)


def bar(fraction: float, width: int = 28, *, good: bool = True) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    body = "#" * filled + "." * (width - filled)
    return colour(body, C.GREEN if good else C.YELLOW)
