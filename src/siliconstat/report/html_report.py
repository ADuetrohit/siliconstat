"""Standalone HTML engineering report.

The output is a single file with no external references -- no CDN, no
JavaScript, no companion assets.  Charts are inline SVG (see
:mod:`siliconstat.report.charts`).  That matters for an engineering artefact:
the report has to still render correctly when it is attached to a review
package, opened from a network share, or archived for two years.

Every number printed here is read from the stored Monte Carlo records.  The
reproducibility block at the top contains everything needed to regenerate the
run byte-for-byte.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .. import __version__
from ..analysis.summary import RunAnalysis
from ..core.units import format_eng
from ..mc.results import MonteCarloRun
from .charts import (
    DARK,
    svg_barh,
    svg_cdf,
    svg_convergence,
    svg_heatmap,
    svg_histogram,
    svg_sigma_plot,
    svg_yield_convergence,
)

__all__ = ["render_html_report", "write_html_report"]


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _num(value: Any, digits: int = 6, unit: str = "") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _e(value)
    if v != v:
        return '<span class="na">n/a</span>'
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if unit and unit not in ("%", "-", "dB", "deg", "frac", ""):
        return _e(format_eng(v, unit, digits=digits))
    formatted = f"{v:.{digits}g}"
    return _e(f"{formatted} {unit}".strip())


def _pct(value: Any, digits: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _e(value)
    return '<span class="na">n/a</span>' if v != v else f"{v:.{digits}f}%"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], *,
           cls: str = "") -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                   for row in rows)
    return (f'<table class="{cls}"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def _kv_grid(pairs: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="kv"><div class="k">{_e(k)}</div><div class="v">{v}</div></div>'
        for k, v in pairs)
    return f'<div class="kv-grid">{cells}</div>'


def _badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge {kind}">{_e(text)}</span>'


CSS = """
:root{--bg:#070d16;--panel:#111c2e;--panel2:#0d1522;--line:#1e2b41;
--text:#dbe6f3;--muted:#8ba0ba;--accent:#22d3ee;--accent2:#3b82f6;
--good:#22c55e;--bad:#ef4444;--warn:#f59e0b;--violet:#a78bfa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
 font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'DejaVu Sans Mono',monospace;
 font-size:13px;line-height:1.55;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 80px;}
h1{font-size:24px;margin:0 0 4px;letter-spacing:.5px}
h2{font-size:16px;margin:38px 0 12px;padding-bottom:8px;
 border-bottom:1px solid var(--line);color:var(--accent);letter-spacing:.6px;
 text-transform:uppercase}
h3{font-size:13px;margin:22px 0 8px;color:var(--text)}
.sub{color:var(--muted);margin:0 0 18px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
 padding:16px 18px;margin:12px 0}
.kv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));
 gap:10px}
.kv{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
 padding:9px 11px}
.kv .k{color:var(--muted);font-size:10px;text-transform:uppercase;
 letter-spacing:.7px}
.kv .v{font-size:14px;margin-top:2px;word-break:break-word}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:12px}
th{background:var(--panel2);color:var(--muted);text-align:left;padding:7px 9px;
 border-bottom:1px solid var(--line);font-weight:600;font-size:10.5px;
 text-transform:uppercase;letter-spacing:.6px;white-space:nowrap}
td{padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:hover{background:rgba(59,130,246,.07)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.na{color:var(--muted);font-style:italic}
.badge{display:inline-block;padding:1px 8px;border-radius:11px;font-size:10.5px;
 font-weight:600;letter-spacing:.4px}
.badge.pass{background:rgba(34,197,94,.16);color:var(--good);
 border:1px solid rgba(34,197,94,.4)}
.badge.fail{background:rgba(239,68,68,.16);color:var(--bad);
 border:1px solid rgba(239,68,68,.4)}
.badge.warn{background:rgba(245,158,11,.16);color:var(--warn);
 border:1px solid rgba(245,158,11,.4)}
.badge.info{background:rgba(34,211,238,.14);color:var(--accent);
 border:1px solid rgba(34,211,238,.36)}
.chart{margin:14px 0}
.note{background:rgba(245,158,11,.08);border-left:3px solid var(--warn);
 padding:9px 13px;margin:9px 0;color:#f3ddb5;border-radius:0 6px 6px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
pre{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
 padding:13px;overflow-x:auto;font-size:11.5px;color:#b8cbe0;max-height:560px}
footer{margin-top:52px;padding-top:16px;border-top:1px solid var(--line);
 color:var(--muted);font-size:11px}
.legend{color:var(--muted);font-size:11px;margin:4px 0 12px}
"""


def render_html_report(analysis: RunAnalysis, run: MonteCarloRun, *,
                       netlist: str = "", title: str = "",
                       project: str = "default") -> str:
    """Render the complete HTML report for one Monte Carlo run."""
    data = analysis.to_dict()
    counters = run.counters
    cfg = run.config or {}
    variation = cfg.get("variation") or {}
    pvt = cfg.get("pvt") or {}
    yr = analysis.yield_report

    parts: list[str] = []
    heading = title or f"Monte Carlo Mismatch Report -- {run.circuit_name}"
    parts.append(f"<h1>{_e(heading)}</h1>")
    parts.append(
        f'<p class="sub">SiliconStat {_e(run.software_version)} &middot; '
        f'run <b>{_e(run.run_id)}</b> &middot; project {_e(project)} &middot; '
        f'generated {_e(datetime.now(timezone.utc).isoformat(timespec="seconds"))}'
        f"</p>")

    # ---- 1. reproducibility ------------------------------------------
    parts.append("<h2>1. Reproducibility</h2>")
    parts.append(_kv_grid([
        ("Run id", _e(run.run_id)),
        ("Seed", _e(cfg.get("seed"))),
        ("Samples requested", _e(cfg.get("samples"))),
        ("Sampling", _e(cfg.get("sampling", "standard"))),
        ("Variation mode", _e(variation.get("mode", "n/a"))),
        ("PVT condition", _e(pvt.get("label", "nominal"))),
        ("Circuit SHA-256", f'<span style="font-size:10px">{_e(run.circuit_sha256[:32])}…</span>'),
        ("Software version", _e(run.software_version)),
        ("Platform", _e(run.platform_info)),
        ("Started (UTC)", _e(run.started_at)),
        ("Finished (UTC)", _e(run.finished_at)),
        ("Wall time", f"{run.duration_s:.2f} s"),
    ]))
    parts.append(
        '<p class="legend">Re-running SiliconStat with the same netlist, seed, '
        "sample count and variation model reproduces every sample exactly: each "
        "sample draws from <code>SeedSequence(seed, spawn_key=(i,))</code>, so "
        "results do not depend on worker count or scheduling order.</p>")
    if run.notes:
        parts.append(f'<div class="note">{_e(run.notes)}</div>')

    # ---- 2. simulation status ----------------------------------------
    parts.append("<h2>2. Simulation status</h2>")
    rate = counters.successful / counters.total * 100.0 if counters.total else 0.0
    parts.append(_kv_grid([
        ("Total attempted", f'<span class="num">{counters.total}</span>'),
        ("Successful", f'<span class="num" style="color:var(--good)">{counters.successful}</span>'),
        ("Failed", f'<span class="num" style="color:var(--bad)">{counters.failed}</span>'),
        ("Success rate", _pct(rate)),
        ("Convergence failures", str(counters.convergence_failures)),
        ("Numerical errors", str(counters.numerical_errors)),
        ("Invalid measurements", str(counters.invalid_measurements)),
        ("Throughput", f"{counters.total / run.duration_s:.1f} samples/s"
         if run.duration_s > 0 else "n/a"),
    ]))
    if analysis.failure_breakdown:
        parts.append("<h3>Failure analysis</h3>")
        parts.append(_table(
            ["Status", "Reason", "Count", "% of attempted"],
            [[_badge(row["status"], "fail"), _e(row["reason"]),
              f'<span class="num">{row["count"]}</span>', _pct(row["percent"])]
             for row in analysis.failure_breakdown]))
    else:
        parts.append('<p class="legend">No sample failed: every attempted '
                     "simulation converged and produced valid measurements.</p>")

    # ---- 3. variation model ------------------------------------------
    parts.append("<h2>3. Variation model</h2>")
    parts.append(_kv_grid([
        ("Model name", _e(variation.get("name", "default"))),
        ("Process (global) variation", _badge(
            "enabled" if variation.get("enable_process") else "disabled",
            "pass" if variation.get("enable_process") else "warn")),
        ("Local mismatch", _badge(
            "enabled" if variation.get("enable_mismatch") else "disabled",
            "pass" if variation.get("enable_mismatch") else "warn")),
        ("Pelgrom convention", "pair (AVT/sqrt(2WL) per device)"
         if variation.get("pelgrom_pair_convention", True)
         else "single device (AVT/sqrt(WL))"),
        ("Random variables", str(len(analysis.slot_meta))),
    ]))
    parts.append(_table(
        ["Random variable", "Parameter", "Scope", "Distribution", "Sigma",
         "Unit", "Devices", "Description"],
        [[_e(s["slot"]), _e(s["parameter"]),
          _badge(s["scope"], "info" if s["scope"] == "global" else "pass"),
          _e(s["distribution"]),
          f'<span class="num">{float(s["sigma"]):.6g}</span>',
          _e(s.get("unit", "")), _e(s.get("devices", "")), _e(s.get("label", ""))]
         for s in analysis.slot_meta]))

    # ---- 4. nominal + statistics -------------------------------------
    parts.append("<h2>4. Statistics</h2>")
    if analysis.nominal:
        parts.append("<h3>Nominal (unperturbed) circuit</h3>")
        units = {m["name"]: m.get("unit", "") for m in analysis.measurement_meta}
        parts.append(_table(
            ["Measurement", "Nominal value"],
            [[_e(k), f'<span class="num">{_num(v, unit=units.get(k, ""))}</span>']
             for k, v in analysis.nominal.items()]))
        if run.nominal_status and run.nominal_status != "ok":
            parts.append(f'<div class="note">Nominal simulation status: '
                         f"{_e(run.nominal_status)}</div>")

    stats_rows = []
    for name, st in analysis.statistics.items():
        d = st.to_dict()
        stats_rows.append([
            _e(name), _e(d["unit"]), f'<span class="num">{d["count"]}</span>',
            f'<span class="num">{_num(d["mean"])}</span>',
            f'<span class="num">{_num(d["median"])}</span>',
            f'<span class="num">{_num(d["std"])}</span>',
            f'<span class="num">{_num(d["cv"], 4)}</span>',
            f'<span class="num">{_num(d["minimum"])}</span>',
            f'<span class="num">{_num(d["maximum"])}</span>',
            f'<span class="num">{_num(d["sigma3_low"])}</span>',
            f'<span class="num">{_num(d["sigma3_high"])}</span>',
        ])
    parts.append(_table(
        ["Measurement", "Unit", "N", "Mean", "Median", "Sigma", "CV",
         "Min", "Max", "-3 sigma", "+3 sigma"], stats_rows))

    parts.append("<h3>Percentiles</h3>")
    parts.append(_table(
        ["Measurement", "P1", "P5", "P25", "P75", "P95", "P99", "IQR",
         "Skew", "Excess kurtosis", "Normality p", "Test"],
        [[_e(name)] + [f'<span class="num">{_num(getattr(st, k))}</span>'
                       for k in ("p1", "p5", "p25", "p75", "p95", "p99", "iqr",
                                 "skewness", "kurtosis_excess", "normality_p")]
         + [_e(st.normality_test or "-")]
         for name, st in analysis.statistics.items()]))
    parts.append(
        '<p class="legend">Mean and variance are accumulated with Welford\'s '
        "algorithm. A normality p-value below 0.05 means the Gaussian "
        "assumption behind the &plusmn;3-sigma limits is measurably violated, so "
        "read the tail percentiles instead.</p>")

    # ---- 5. distributions --------------------------------------------
    parts.append("<h2>5. Distributions</h2>")
    for name, chart in analysis.charts.items():
        st = analysis.statistics.get(name)
        if st is None:
            continue
        parts.append(f"<h3>{_e(name)} [{_e(chart.get('unit', ''))}]</h3>")
        parts.append(f'<div class="chart">' + svg_histogram(
            chart["histogram"], st.to_dict(), chart.get("specs", []),
            unit=chart.get("unit", ""), title=f"{name} distribution",
            nominal=chart.get("nominal")) + "</div>")
        parts.append('<div class="grid2">')
        parts.append(f'<div class="chart">' + svg_cdf(
            chart["cdf"], chart.get("specs", []), unit=chart.get("unit", ""),
            title=f"{name} CDF") + "</div>")
        parts.append(f'<div class="chart">' + svg_sigma_plot(
            chart["sigma_plot"], unit=chart.get("unit", ""),
            title=f"{name} normal-quantile plot") + "</div>")
        parts.append("</div>")

    # ---- 6. yield ------------------------------------------------------
    parts.append("<h2>6. Yield</h2>")
    if not yr.per_spec:
        parts.append('<div class="note">'
                     + _e(" ".join(yr.notes) or "No specifications declared.")
                     + "</div>")
    else:
        parts.append(_kv_grid([
            ("Combined yield", f'<b style="color:var(--accent);font-size:19px">'
                               f"{yr.combined_yield_over_successful:.2f}%</b>"),
            ("95% confidence interval",
             f"[{yr.combined_ci95_low:.2f}%, {yr.combined_ci95_high:.2f}%]"),
            ("Denominator", f"{yr.combined_passing} / {yr.successful} "
                            "successful simulations"),
            ("Over all attempted samples",
             f"{yr.combined_yield_over_attempted:.2f}% "
             f"({yr.combined_passing} / {yr.attempted})"),
            ("Limiting specification", _e(yr.limiting_spec)),
            ("Product of individual yields", _pct(yr.independent_product_pct)),
        ]))
        parts.append(_table(
            ["Specification", "Passing", "Failing", "Denominator", "Yield",
             "95% CI", "Margin to limit", "Margin [sigma]", "Cpk", "Worst sample"],
            [[_e(s.description), f'<span class="num">{s.passing}</span>',
              f'<span class="num">{s.failing}</span>',
              f'<span class="num">{s.denominator}</span>',
              _badge(f"{s.yield_pct:.2f}%",
                     "pass" if s.yield_pct >= 99 else
                     ("warn" if s.yield_pct >= 90 else "fail")),
              f"[{s.ci95_low:.2f}, {s.ci95_high:.2f}]",
              f'<span class="num">{_num(s.margin_mean, unit=s.unit)}</span>',
              f'<span class="num">{_num(s.margin_sigma, 3)}</span>',
              f'<span class="num">{_num(s.cpk, 3)}</span>',
              f'<span class="num">{_num(s.worst_value, unit=s.unit)}</span>']
             for s in yr.per_spec]))
        for note in yr.notes:
            parts.append(f'<div class="note">{_e(note)}</div>')
        parts.append(
            '<p class="legend">Confidence intervals are Wilson score intervals. '
            "Cpk &ge; 1.33 is the conventional capability threshold; Cpk = 1.0 "
            "corresponds to the limit sitting exactly 3 sigma from the mean.</p>")

    # ---- 7. correlation ------------------------------------------------
    parts.append("<h2>7. Correlation</h2>")
    corr = analysis.correlation
    if corr.parameters and corr.measurements:
        parts.append(f'<div class="chart">' + svg_heatmap(
            corr.pearson, corr.parameters, corr.measurements,
            title=f"Pearson correlation: parameters vs measurements "
                  f"(n = {corr.n_samples})") + "</div>")
        parts.append(f'<div class="chart">' + svg_heatmap(
            corr.spearman, corr.parameters, corr.measurements,
            title="Spearman rank correlation") + "</div>")
        parts.append(
            '<p class="legend">A large Spearman coefficient paired with a small '
            "Pearson coefficient indicates a strong but nonlinear dependence.</p>")
    for note in corr.notes:
        parts.append(f'<div class="note">{_e(note)}</div>')

    mcorr = analysis.measurement_correlation
    if mcorr.get("pearson"):
        parts.append("<h3>Correlation between measurements</h3>")
        parts.append(f'<div class="chart">' + svg_heatmap(
            mcorr["pearson"], mcorr["measurements"], mcorr["measurements"],
            title="Measurement cross-correlation") + "</div>")

    # ---- 8. sensitivity -------------------------------------------------
    parts.append("<h2>8. Sensitivity</h2>")
    for name, report in analysis.sensitivity.items():
        if not report.entries:
            continue
        parts.append(f"<h3>{_e(name)}</h3>")
        parts.append(_kv_grid([
            ("Method", "variance-based (standardised regression)"
             if report.method == "regression" else "correlation-based ranking"),
            ("R&sup2; of linear model", _num(report.r_squared, 4)),
            ("Unexplained variance", _pct(report.unexplained_pct)),
            ("Output sigma", _num(report.output_sigma)),
            ("Max |r| between inputs", _num(report.max_input_correlation, 3)),
            ("Samples used", str(report.n_samples)),
        ]))
        top = report.entries[:12]
        parts.append(f'<div class="chart">' + svg_barh(
            [e.parameter for e in top],
            [e.variance_contribution_pct for e in top],
            title=f"Variance contribution to {name}", unit="%") + "</div>")
        parts.append(_table(
            ["#", "Parameter", "Scope", "Standardised beta",
             "Variance share", "% of explained", "Pearson r", "Spearman rho",
             "d(output)/d(1 sigma)"],
            [[str(e.rank), _e(e.parameter), _e(e.scope),
              f'<span class="num">{_num(e.beta_standardised, 4)}</span>',
              f'<span class="num">{_pct(e.variance_contribution_pct)}</span>',
              f'<span class="num">{_pct(e.share_of_explained_pct)}</span>',
              f'<span class="num">{_num(e.pearson, 4)}</span>',
              f'<span class="num">{_num(e.spearman, 4)}</span>',
              f'<span class="num">{_num(e.d_output_d_sigma)}</span>']
             for e in report.entries]))
        for note in report.notes:
            parts.append(f'<div class="note">{_e(note)}</div>')

    # ---- 9. convergence -------------------------------------------------
    parts.append("<h2>9. Statistical convergence</h2>")
    parts.append(
        '<p class="legend">Monte Carlo estimates converge as 1/sqrt(N). These '
        "traces show how the reported numbers stabilise as samples accumulate; "
        "the shaded band is the 95% confidence interval.</p>")
    for name, trace in analysis.convergence.items():
        d = trace.to_dict()
        parts.append(f"<h3>{_e(name)}</h3>")
        parts.append(f'<div class="chart">{svg_convergence(d)}</div>')
        if d.get("yield_pct"):
            parts.append(f'<div class="chart">{svg_yield_convergence(d)}</div>')
        parts.append(_kv_grid([
            ("Final mean", _num(d.get("final_mean"), unit=trace.unit)),
            ("Final sigma", _num(d.get("final_std"), unit=trace.unit)),
            ("Final yield", _pct(d.get("final_yield_pct"))),
            ("Mean settled at",
             _e(d.get("mean_settled_at") or "not settled")),
            ("Sigma settled at",
             _e(d.get("std_settled_at") or "not settled")),
            ("Yield settled at",
             _e(d.get("yield_settled_at") or "not settled")),
        ]))
        for note in d.get("notes", []):
            parts.append(f'<div class="note">{_e(note)}</div>')

    # ---- 10. failed samples --------------------------------------------
    failed = run.failed_samples()
    parts.append("<h2>10. Failed samples</h2>")
    if not failed:
        parts.append('<p class="legend">None.</p>')
    else:
        shown = failed[:60]
        parts.append(_table(
            ["Sample", "Seed", "Status", "Reason"],
            [[str(s.index), f'<span style="font-size:10px">{s.seed}</span>',
              _badge(s.status, "fail"), _e(s.failure_reason[:300])]
             for s in shown]))
        if len(failed) > len(shown):
            parts.append(f'<p class="legend">Showing {len(shown)} of '
                         f"{len(failed)} failed samples; the complete list is in "
                         "the CSV/JSON export.</p>")

    # ---- 11. circuit ----------------------------------------------------
    if netlist:
        parts.append("<h2>11. Circuit netlist</h2>")
        parts.append(f"<pre>{_e(netlist)}</pre>")

    for note in analysis.notes:
        parts.append(f'<div class="note">{_e(note)}</div>')

    parts.append(
        "<footer>Generated by SiliconStat "
        f"{_e(__version__)} &mdash; Monte Carlo Mismatch Analysis and "
        "Statistical Verification Platform for Analog ICs.<br>"
        "The device models used are simplified square-law (SPICE level-1 class) "
        "models intended for engineering study and statistical methodology. "
        "They are not BSIM and are not silicon-calibrated; see "
        "<code>docs/limitations.md</code>.</footer>")

    body = "".join(parts)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(heading)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">{body}</div></body></html>"""


def write_html_report(path: str, analysis: RunAnalysis, run: MonteCarloRun, *,
                      netlist: str = "", title: str = "",
                      project: str = "default") -> str:
    """Render and write the report; returns the path written."""
    markup = render_html_report(analysis, run, netlist=netlist, title=title,
                               project=project)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(markup)
    return path
