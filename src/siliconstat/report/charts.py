"""Inline SVG chart generation for the HTML report.

Deliberately dependency-free: the report must open correctly on a machine with
no network access and no JavaScript, months after the run, so charts are
plain SVG elements written into the document rather than a CDN plotting
library.  The interactive Plotly versions live in the React frontend, which
consumes the same numeric payloads from the API.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["Theme", "DARK", "svg_histogram", "svg_cdf", "svg_heatmap",
           "svg_barh", "svg_convergence", "svg_sigma_plot", "svg_box"]


@dataclass(frozen=True)
class Theme:
    bg: str = "#0d1522"
    panel: str = "#111c2e"
    grid: str = "#1e2b41"
    axis: str = "#3a4d69"
    text: str = "#c9d8ea"
    muted: str = "#7d92ac"
    accent: str = "#22d3ee"
    accent2: str = "#3b82f6"
    good: str = "#22c55e"
    bad: str = "#ef4444"
    warn: str = "#f59e0b"
    violet: str = "#a78bfa"


DARK = Theme()

_FONT = ("font-family=\"ui-monospace,SFMono-Regular,Menlo,Consolas,"
         "'DejaVu Sans Mono',monospace\"")


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or value != value:
        return "n/a"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e5:
        return f"{value:.{digits}g}"
    return f"{value:.{digits - 1}e}"


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """'Nice' tick positions covering [lo, hi]."""
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo] if math.isfinite(lo) else [0.0]
    raw = (hi - lo) / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * magnitude
        if raw <= step:
            break
    start = math.ceil(lo / step) * step
    out = []
    value = start
    while value <= hi + step * 1e-9 and len(out) < count * 3:
        out.append(round(value, 12))
        value += step
    return out or [lo, hi]


class _Canvas:
    """Minimal SVG canvas with linear data-to-pixel mapping."""

    def __init__(self, width: int, height: int, theme: Theme = DARK, *,
                 pad_left: int = 78, pad_right: int = 22,
                 pad_top: int = 40, pad_bottom: int = 54) -> None:
        self.w, self.h = width, height
        self.t = theme
        self.pl, self.pr, self.pt, self.pb = pad_left, pad_right, pad_top, pad_bottom
        self.parts: list[str] = []
        self.x0 = self.y0 = 0.0
        self.x1 = self.y1 = 1.0

    @property
    def plot_w(self) -> float:
        return self.w - self.pl - self.pr

    @property
    def plot_h(self) -> float:
        return self.h - self.pt - self.pb

    def set_domain(self, x0: float, x1: float, y0: float, y1: float) -> None:
        if x1 <= x0:
            x0, x1 = x0 - 0.5, x0 + 0.5
        if y1 <= y0:
            y0, y1 = y0 - 0.5, y0 + 0.5
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def px(self, x: float) -> float:
        return self.pl + (x - self.x0) / (self.x1 - self.x0) * self.plot_w

    def py(self, y: float) -> float:
        return self.pt + self.plot_h - (y - self.y0) / (self.y1 - self.y0) * self.plot_h

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def frame(self, title: str, x_label: str, y_label: str, *,
              log_x: bool = False) -> None:
        t = self.t
        self.add(f'<rect x="0" y="0" width="{self.w}" height="{self.h}" '
                 f'fill="{t.panel}" rx="8"/>')
        self.add(f'<text x="{self.pl}" y="24" fill="{t.text}" font-size="13" '
                 f'font-weight="600" {_FONT}>{_esc(title)}</text>')
        for value in _ticks(self.y0, self.y1, 5):
            y = self.py(value)
            self.add(f'<line x1="{self.pl}" y1="{y:.1f}" x2="{self.w - self.pr}" '
                     f'y2="{y:.1f}" stroke="{t.grid}" stroke-width="1"/>')
            self.add(f'<text x="{self.pl - 8}" y="{y + 4:.1f}" fill="{t.muted}" '
                     f'font-size="10" text-anchor="end" {_FONT}>{_fmt(value, 3)}</text>')
        xt = _ticks(self.x0, self.x1, 6)
        for value in xt:
            x = self.px(value)
            self.add(f'<line x1="{x:.1f}" y1="{self.pt}" x2="{x:.1f}" '
                     f'y2="{self.pt + self.plot_h}" stroke="{t.grid}" '
                     f'stroke-width="1"/>')
            label = f"1e{value:.0f}" if log_x else _fmt(value, 3)
            self.add(f'<text x="{x:.1f}" y="{self.pt + self.plot_h + 16}" '
                     f'fill="{t.muted}" font-size="10" text-anchor="middle" '
                     f'{_FONT}>{_esc(label)}</text>')
        self.add(f'<rect x="{self.pl}" y="{self.pt}" width="{self.plot_w}" '
                 f'height="{self.plot_h}" fill="none" stroke="{t.axis}" '
                 f'stroke-width="1"/>')
        self.add(f'<text x="{self.pl + self.plot_w / 2}" '
                 f'y="{self.h - 12}" fill="{t.muted}" font-size="11" '
                 f'text-anchor="middle" {_FONT}>{_esc(x_label)}</text>')
        self.add(f'<text x="14" y="{self.pt + self.plot_h / 2}" fill="{t.muted}" '
                 f'font-size="11" text-anchor="middle" {_FONT} '
                 f'transform="rotate(-90 14 {self.pt + self.plot_h / 2})">'
                 f'{_esc(y_label)}</text>')

    def vline(self, x: float, colour: str, label: str = "", *,
              dash: str = "", width: float = 1.5, label_dy: int = 0) -> None:
        if not (self.x0 <= x <= self.x1):
            return
        px = self.px(x)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{px:.1f}" y1="{self.pt}" x2="{px:.1f}" '
                 f'y2="{self.pt + self.plot_h}" stroke="{colour}" '
                 f'stroke-width="{width}"{dash_attr}/>')
        if label:
            self.add(f'<text x="{px + 4:.1f}" y="{self.pt + 12 + label_dy}" '
                     f'fill="{colour}" font-size="10" {_FONT}>{_esc(label)}</text>')

    def render(self) -> str:
        return (f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" '
                f'preserveAspectRatio="xMidYMid meet" role="img" '
                f'xmlns="http://www.w3.org/2000/svg">'
                + "".join(self.parts) + "</svg>")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def svg_histogram(hist: dict[str, Any], stats: dict[str, Any],
                  specs: Sequence[dict[str, Any]] = (), *, unit: str = "",
                  title: str = "", nominal: float | None = None,
                  theme: Theme = DARK, width: int = 720, height: int = 340) -> str:
    """Histogram with mean, +/-1/2/3 sigma markers and specification limits."""
    counts = hist["counts"]
    edges = hist["edges"]
    if not counts:
        return ""
    c = _Canvas(width, height, theme)
    lo, hi = float(edges[0]), float(edges[-1])
    mean = float(stats.get("mean", float("nan")))
    sigma = float(stats.get("std", float("nan")))
    marks = []
    if math.isfinite(mean) and math.isfinite(sigma) and sigma > 0:
        marks = [mean + k * sigma for k in (-3, -2, -1, 1, 2, 3)]
        lo = min(lo, mean - 3.4 * sigma)
        hi = max(hi, mean + 3.4 * sigma)
    for spec in specs:
        limit = float(spec.get("value", float("nan")))
        if math.isfinite(limit):
            lo = min(lo, limit - 0.02 * (hi - lo or 1))
            hi = max(hi, limit + 0.02 * (hi - lo or 1))
    c.set_domain(lo, hi, 0.0, max(counts) * 1.15)
    c.frame(title or "Distribution", f"value [{unit}]" if unit else "value", "count")

    for i, count in enumerate(counts):
        x_lo, x_hi = c.px(edges[i]), c.px(edges[i + 1])
        y = c.py(count)
        h = c.py(0) - y
        if h <= 0:
            continue
        c.add(f'<rect x="{x_lo:.2f}" y="{y:.2f}" width="{max(x_hi - x_lo - 1, 0.5):.2f}" '
              f'height="{h:.2f}" fill="{theme.accent2}" fill-opacity="0.55" '
              f'stroke="{theme.accent}" stroke-width="0.6"/>')

    for k, x in zip((-3, -2, -1, 1, 2, 3), marks):
        c.vline(x, theme.muted, f"{k:+d}s", dash="3 3", width=1.0,
                label_dy=(0 if abs(k) == 3 else (12 if abs(k) == 2 else 24)))
    if math.isfinite(mean):
        c.vline(mean, theme.accent, f"mean {_fmt(mean)}", width=2.0)
    median = float(stats.get("median", float("nan")))
    if math.isfinite(median):
        c.vline(median, theme.violet, "median", dash="6 3", width=1.2, label_dy=36)
    if nominal is not None and math.isfinite(float(nominal)):
        c.vline(float(nominal), theme.warn, "nominal", dash="2 4", width=1.4,
                label_dy=48)
    for spec in specs:
        limit = float(spec.get("value", float("nan")))
        if math.isfinite(limit):
            c.vline(limit, theme.bad,
                    f"{spec.get('op', '')} {_fmt(limit)}", width=2.0, label_dy=60)
    return c.render()


def svg_cdf(cdf: dict[str, Any], specs: Sequence[dict[str, Any]] = (), *,
            unit: str = "", title: str = "", theme: Theme = DARK,
            width: int = 720, height: int = 320) -> str:
    """Empirical cumulative distribution with specification limits."""
    xs = cdf.get("x") or []
    ps = cdf.get("p") or []
    if len(xs) < 2:
        return ""
    c = _Canvas(width, height, theme)
    lo, hi = float(min(xs)), float(max(xs))
    for spec in specs:
        limit = float(spec.get("value", float("nan")))
        if math.isfinite(limit):
            lo, hi = min(lo, limit), max(hi, limit)
    span = (hi - lo) or 1.0
    c.set_domain(lo - 0.03 * span, hi + 0.03 * span, 0.0, 1.0)
    c.frame(title or "Cumulative distribution",
            f"value [{unit}]" if unit else "value", "P(X <= x)")
    points = " ".join(f"{c.px(x):.2f},{c.py(p):.2f}" for x, p in zip(xs, ps))
    c.add(f'<polyline points="{points}" fill="none" stroke="{theme.accent}" '
          f'stroke-width="2"/>')
    for level, colour in ((0.5, theme.muted), (0.9973, theme.muted)):
        y = c.py(level)
        c.add(f'<line x1="{c.pl}" y1="{y:.1f}" x2="{c.w - c.pr}" y2="{y:.1f}" '
              f'stroke="{colour}" stroke-width="1" stroke-dasharray="3 4"/>')
    for spec in specs:
        limit = float(spec.get("value", float("nan")))
        if math.isfinite(limit):
            c.vline(limit, theme.bad, f"{spec.get('op', '')} {_fmt(limit)}", width=2.0)
    return c.render()


def svg_sigma_plot(data: dict[str, Any], *, unit: str = "", title: str = "",
                   theme: Theme = DARK, width: int = 720, height: int = 300) -> str:
    """Normal-quantile plot -- a straight line means the data are Gaussian."""
    xs = data.get("x") or []
    zs = data.get("sigma") or []
    if len(xs) < 3:
        return ""
    c = _Canvas(width, height, theme)
    finite = [(x, z) for x, z in zip(xs, zs) if math.isfinite(z)]
    if len(finite) < 3:
        return ""
    xv = [p[0] for p in finite]
    zv = [p[1] for p in finite]
    c.set_domain(min(xv), max(xv), max(-4.0, min(zv)), min(4.0, max(zv)))
    c.frame(title or "Normal quantile plot",
            f"value [{unit}]" if unit else "value", "sigma")
    pts = " ".join(f"{c.px(x):.2f},{c.py(z):.2f}" for x, z in finite
                   if c.y0 <= z <= c.y1)
    c.add(f'<polyline points="{pts}" fill="none" stroke="{theme.accent}" '
          f'stroke-width="1.6"/>')
    n = len(xv)
    mean = sum(xv) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xv) / max(n - 1, 1))
    if sd > 0:
        x_at = [mean + c.y0 * sd, mean + c.y1 * sd]
        c.add(f'<line x1="{c.px(x_at[0]):.1f}" y1="{c.py(c.y0):.1f}" '
              f'x2="{c.px(x_at[1]):.1f}" y2="{c.py(c.y1):.1f}" '
              f'stroke="{theme.warn}" stroke-width="1.2" stroke-dasharray="5 4"/>')
    return c.render()


def svg_heatmap(matrix: Sequence[Sequence[float]], row_labels: Sequence[str],
                col_labels: Sequence[str], *, title: str = "Correlation",
                theme: Theme = DARK, cell: int = 46) -> str:
    """Diverging red/blue correlation heatmap with the values printed in-cell."""
    if not matrix or not matrix[0]:
        return ""
    rows, cols = len(matrix), len(matrix[0])
    left, top = 190, 92
    width = left + cols * cell + 24
    height = top + rows * cell + 28
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" '
             f'fill="{theme.panel}" rx="8"/>',
             f'<text x="20" y="26" fill="{theme.text}" font-size="13" '
             f'font-weight="600" {_FONT}>{_esc(title)}</text>']
    for j, label in enumerate(col_labels):
        cx = left + j * cell + cell / 2
        parts.append(f'<text x="{cx}" y="{top - 8}" fill="{theme.muted}" '
                     f'font-size="10" text-anchor="start" {_FONT} '
                     f'transform="rotate(-55 {cx} {top - 8})">{_esc(label)}</text>')
    for i, label in enumerate(row_labels):
        cy = top + i * cell + cell / 2 + 4
        parts.append(f'<text x="{left - 10}" y="{cy}" fill="{theme.muted}" '
                     f'font-size="10" text-anchor="end" {_FONT}>{_esc(label)}</text>')
    for i in range(rows):
        for j in range(cols):
            value = matrix[i][j]
            x = left + j * cell
            y = top + i * cell
            if value is None or value != value:
                fill, text_colour, label = theme.grid, theme.muted, "n/a"
            else:
                v = max(-1.0, min(1.0, float(value)))
                intensity = abs(v)
                base = theme.bad if v < 0 else theme.accent2
                fill = base
                text_colour = "#ffffff" if intensity > 0.45 else theme.text
                label = f"{v:+.2f}"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" '
                             f'height="{cell - 2}" fill="{theme.grid}" rx="3"/>')
                parts.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" '
                             f'height="{cell - 2}" fill="{fill}" '
                             f'fill-opacity="{0.12 + 0.8 * intensity:.3f}" rx="3"/>')
                parts.append(f'<text x="{x + (cell - 2) / 2}" y="{y + cell / 2 + 4}" '
                             f'fill="{text_colour}" font-size="10" '
                             f'text-anchor="middle" {_FONT}>{_esc(label)}</text>')
                continue
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" '
                         f'height="{cell - 2}" fill="{fill}" rx="3"/>')
            parts.append(f'<text x="{x + (cell - 2) / 2}" y="{y + cell / 2 + 4}" '
                         f'fill="{text_colour}" font-size="9" '
                         f'text-anchor="middle" {_FONT}>{_esc(label)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>")


def svg_barh(labels: Sequence[str], values: Sequence[float], *,
             title: str = "", unit: str = "%", theme: Theme = DARK,
             width: int = 720, row_height: int = 26,
             colour: str | None = None) -> str:
    """Horizontal bar chart used for sensitivity rankings."""
    if not labels:
        return ""
    n = len(labels)
    left = 190
    top = 44
    height = top + n * row_height + 22
    peak = max((abs(v) for v in values if v == v), default=1.0) or 1.0
    plot_w = width - left - 90
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" '
             f'fill="{theme.panel}" rx="8"/>',
             f'<text x="20" y="26" fill="{theme.text}" font-size="13" '
             f'font-weight="600" {_FONT}>{_esc(title)}</text>']
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_height
        parts.append(f'<text x="{left - 10}" y="{y + row_height / 2 + 4}" '
                     f'fill="{theme.muted}" font-size="10" text-anchor="end" '
                     f'{_FONT}>{_esc(label)}</text>')
        if value != value:
            continue
        w = abs(value) / peak * plot_w
        bar_colour = colour or (theme.accent if value >= 0 else theme.bad)
        parts.append(f'<rect x="{left}" y="{y + 4}" width="{max(w, 1):.1f}" '
                     f'height="{row_height - 10}" fill="{bar_colour}" '
                     f'fill-opacity="0.75" rx="3"/>')
        parts.append(f'<text x="{left + w + 8:.1f}" y="{y + row_height / 2 + 4}" '
                     f'fill="{theme.text}" font-size="10" {_FONT}>'
                     f'{_fmt(value, 3)}{_esc(unit)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>")


def svg_convergence(trace: dict[str, Any], *, theme: Theme = DARK,
                    width: int = 720, height: int = 320) -> str:
    """Running mean / sigma / yield versus sample count."""
    ns = trace.get("n") or []
    if len(ns) < 2:
        return ""
    means = trace.get("mean") or []
    lo = trace.get("mean_ci_low") or []
    hi = trace.get("mean_ci_high") or []
    c = _Canvas(width, height, theme)
    values = [v for v in (list(means) + list(lo) + list(hi)) if v == v]
    if not values:
        return ""
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or (abs(vmax) or 1.0) * 0.1
    c.set_domain(ns[0], ns[-1], vmin - 0.1 * span, vmax + 0.1 * span)
    unit = trace.get("unit") or ""
    c.frame(f"Convergence of mean({trace.get('measurement', '')})",
            "samples", f"mean [{unit}]" if unit else "mean")
    if lo and hi:
        top = " ".join(f"{c.px(n):.1f},{c.py(v):.1f}"
                       for n, v in zip(ns, hi) if v == v)
        bottom = " ".join(f"{c.px(n):.1f},{c.py(v):.1f}"
                          for n, v in reversed(list(zip(ns, lo))) if v == v)
        if top and bottom:
            c.add(f'<polygon points="{top} {bottom}" fill="{theme.accent2}" '
                  f'fill-opacity="0.18"/>')
    pts = " ".join(f"{c.px(n):.1f},{c.py(v):.1f}" for n, v in zip(ns, means)
                   if v == v)
    c.add(f'<polyline points="{pts}" fill="none" stroke="{theme.accent}" '
          f'stroke-width="2"/>')
    final = trace.get("final_mean")
    if final is not None and final == final:
        y = c.py(final)
        c.add(f'<line x1="{c.pl}" y1="{y:.1f}" x2="{c.w - c.pr}" y2="{y:.1f}" '
              f'stroke="{theme.warn}" stroke-width="1.2" stroke-dasharray="6 4"/>')
    settled = trace.get("mean_settled_at")
    if settled:
        x = c.px(settled)
        c.add(f'<line x1="{x:.1f}" y1="{c.pt}" x2="{x:.1f}" '
              f'y2="{c.pt + c.plot_h}" stroke="{theme.good}" stroke-width="1.4" '
              f'stroke-dasharray="4 3"/>')
        c.add(f'<text x="{x + 5:.1f}" y="{c.pt + 14}" fill="{theme.good}" '
              f'font-size="10" {_FONT}>settled at n={settled}</text>')
    return c.render()


def svg_yield_convergence(trace: dict[str, Any], *, theme: Theme = DARK,
                          width: int = 720, height: int = 300) -> str:
    ns = trace.get("n") or []
    ys = trace.get("yield_pct") or []
    if len(ns) < 2 or not ys:
        return ""
    lo = trace.get("yield_ci_low") or []
    hi = trace.get("yield_ci_high") or []
    c = _Canvas(width, height, theme)
    values = [v for v in (list(ys) + list(lo) + list(hi)) if v == v]
    c.set_domain(ns[0], ns[-1], max(0.0, min(values) - 3), min(100.0, max(values) + 3))
    c.frame("Yield convergence", "samples", "combined yield [%]")
    if lo and hi:
        top = " ".join(f"{c.px(n):.1f},{c.py(v):.1f}" for n, v in zip(ns, hi))
        bottom = " ".join(f"{c.px(n):.1f},{c.py(v):.1f}"
                          for n, v in reversed(list(zip(ns, lo))))
        c.add(f'<polygon points="{top} {bottom}" fill="{theme.good}" '
              f'fill-opacity="0.15"/>')
    pts = " ".join(f"{c.px(n):.1f},{c.py(v):.1f}" for n, v in zip(ns, ys))
    c.add(f'<polyline points="{pts}" fill="none" stroke="{theme.good}" '
          f'stroke-width="2"/>')
    return c.render()


def svg_box(groups: Sequence[tuple[str, dict[str, Any]]], *, title: str = "",
            unit: str = "", theme: Theme = DARK, width: int = 720,
            height: int = 340) -> str:
    """Box plot comparing several groups (runs, corners, devices).

    Each group supplies a statistics dict with ``p25``/``median``/``p75``/
    ``minimum``/``maximum``; whiskers are drawn at the observed extremes.
    """
    groups = [g for g in groups if g[1]]
    if not groups:
        return ""
    c = _Canvas(width, height, theme, pad_left=86, pad_bottom=64)
    lows, highs = [], []
    for _name, st in groups:
        for key in ("minimum", "p1", "p25", "median", "p75", "p99", "maximum"):
            v = st.get(key)
            if v is not None and v == v:
                lows.append(float(v))
                highs.append(float(v))
    if not lows:
        return ""
    lo, hi = min(lows), max(highs)
    span = (hi - lo) or (abs(hi) or 1.0) * 0.1
    c.set_domain(0.0, float(len(groups)), lo - 0.1 * span, hi + 0.1 * span)
    c.frame(title or "Distribution comparison", "", f"value [{unit}]" if unit else "value")
    box_w = c.plot_w / len(groups) * 0.45
    for i, (name, st) in enumerate(groups):
        cx = c.px(i + 0.5)
        q1 = float(st.get("p25", float("nan")))
        q3 = float(st.get("p75", float("nan")))
        med = float(st.get("median", float("nan")))
        vmin = float(st.get("minimum", float("nan")))
        vmax = float(st.get("maximum", float("nan")))
        if any(v != v for v in (q1, q3, med)):
            continue
        c.add(f'<line x1="{cx:.1f}" y1="{c.py(vmin):.1f}" x2="{cx:.1f}" '
              f'y2="{c.py(vmax):.1f}" stroke="{theme.muted}" stroke-width="1.2"/>')
        for v in (vmin, vmax):
            c.add(f'<line x1="{cx - box_w / 3:.1f}" y1="{c.py(v):.1f}" '
                  f'x2="{cx + box_w / 3:.1f}" y2="{c.py(v):.1f}" '
                  f'stroke="{theme.muted}" stroke-width="1.2"/>')
        y_top, y_bot = c.py(q3), c.py(q1)
        c.add(f'<rect x="{cx - box_w / 2:.1f}" y="{y_top:.1f}" '
              f'width="{box_w:.1f}" height="{max(y_bot - y_top, 1):.1f}" '
              f'fill="{theme.accent2}" fill-opacity="0.35" '
              f'stroke="{theme.accent}" stroke-width="1.2" rx="2"/>')
        c.add(f'<line x1="{cx - box_w / 2:.1f}" y1="{c.py(med):.1f}" '
              f'x2="{cx + box_w / 2:.1f}" y2="{c.py(med):.1f}" '
              f'stroke="{theme.accent}" stroke-width="2.2"/>')
        c.add(f'<text x="{cx:.1f}" y="{c.pt + c.plot_h + 18}" fill="{theme.muted}" '
              f'font-size="10" text-anchor="middle" {_FONT}>{_esc(name)}</text>')
    return c.render()
