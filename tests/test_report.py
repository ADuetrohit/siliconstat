"""Exports and the standalone HTML report."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

from siliconstat.analysis import analyse_run
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.report import (
    export_bundle,
    export_csv,
    export_json,
    export_summary_csv,
    render_html_report,
    svg_barh,
    svg_box,
    svg_cdf,
    svg_convergence,
    svg_heatmap,
    svg_histogram,
    svg_sigma_plot,
)
from siliconstat.report.export import ExportError, export_parquet
from siliconstat.variation import default_mismatch_model

from conftest import load


@pytest.fixture(scope="module")
def mirror():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=120, seed=606,
        workers=1))
    return circuit, run, analyse_run(run)


# ---------------------------------------------------------------------------
# raw data export
# ---------------------------------------------------------------------------

def test_csv_export_has_one_row_per_attempted_sample(mirror, tmp_path):
    import pandas as pd

    _circuit, run, _analysis = mirror
    path = export_csv(run, str(tmp_path / "samples.csv"))
    frame = pd.read_csv(path)
    assert len(frame) == run.counters.total


def test_csv_export_carries_the_documented_columns(mirror, tmp_path):
    import pandas as pd

    _circuit, run, _analysis = mirror
    frame = pd.read_csv(export_csv(run, str(tmp_path / "samples.csv")))
    for column in ("sample_id", "seed", "simulation_status", "failure_reason",
                   "pass_fail", "dc_iterations", "dc_strategy", "runtime_s"):
        assert column in frame.columns
    assert any(c.startswith("meas.") for c in frame.columns)
    assert any(c.startswith("var.") for c in frame.columns)
    assert any(c.startswith("dev.") for c in frame.columns)
    assert any(c.startswith("spec.") for c in frame.columns)


def test_csv_values_match_the_run(mirror, tmp_path):
    import pandas as pd

    _circuit, run, _analysis = mirror
    frame = pd.read_csv(export_csv(run, str(tmp_path / "samples.csv")))
    assert frame["meas.ierr"].to_numpy() == pytest.approx(
        np.array([s.measurements["ierr"] for s in run.samples]))
    assert list(frame["sample_id"]) == [s.index for s in run.samples]


def test_json_export_round_trips(mirror, tmp_path):
    _circuit, run, analysis = mirror
    path = export_json(run, str(tmp_path / "run.json"), analysis)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["run"]["run_id"] == run.run_id
    assert len(payload["run"]["samples"]) == run.counters.total
    assert payload["analysis"]["yield"]["combined_passing"] == \
        analysis.yield_report.combined_passing


def test_json_export_can_omit_samples(mirror, tmp_path):
    _circuit, run, _analysis = mirror
    path = export_json(run, str(tmp_path / "summary.json"), include_samples=False)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "samples" not in payload["run"]


def test_parquet_export_matches_the_csv(mirror, tmp_path):
    import pandas as pd

    _circuit, run, _analysis = mirror
    try:
        path = export_parquet(run, str(tmp_path / "samples.parquet"))
    except ExportError:
        pytest.skip("pyarrow is not installed")
    frame = pd.read_parquet(path)
    assert len(frame) == run.counters.total
    assert "meas.ierr" in frame.columns


def test_statistics_csv(mirror, tmp_path):
    import pandas as pd

    _circuit, _run, analysis = mirror
    frame = pd.read_csv(export_summary_csv(analysis, str(tmp_path / "stats.csv")))
    assert set(frame["measurement"]) == set(analysis.statistics)
    assert "mean" in frame.columns and "sigma3_low" in frame.columns


def test_bundle_writes_every_artefact(mirror, tmp_path):
    circuit, run, analysis = mirror
    written = export_bundle(run, analysis, str(tmp_path / "bundle"),
                            netlist=circuit.source_text or "")
    for key in ("csv", "summary_csv", "json", "html", "netlist"):
        assert key in written
        assert os.path.getsize(written[key]) > 0


# ---------------------------------------------------------------------------
# the HTML report
# ---------------------------------------------------------------------------

def test_report_is_self_contained(mirror):
    """No CDN, no external stylesheet, no remote image: the report has to
    render correctly from an archive years later."""
    circuit, run, analysis = mirror
    html = render_html_report(analysis, run, netlist=circuit.source_text or "")

    assert "<script" not in html.lower()
    assert not re.search(r'src\s*=\s*["\']https?://', html, re.IGNORECASE)
    assert not re.search(r'href\s*=\s*["\']https?://', html, re.IGNORECASE)
    assert "@import" not in html
    assert "<svg" in html


def test_report_contains_the_reproducibility_record(mirror):
    _circuit, run, analysis = mirror
    html = render_html_report(analysis, run)
    assert run.run_id in html
    assert str(run.config["seed"]) in html
    assert run.circuit_sha256[:16] in html
    assert "Reproducibility" in html


def test_report_states_the_sample_accounting(mirror):
    _circuit, run, analysis = mirror
    html = render_html_report(analysis, run)
    assert "Total attempted" in html
    assert str(run.counters.total) in html
    assert "Successful" in html


def test_report_shows_yield_with_its_denominator(mirror):
    _circuit, run, analysis = mirror
    html = render_html_report(analysis, run)
    assert "successful simulations" in html
    assert f"{analysis.yield_report.combined_yield_over_successful:.2f}%" in html


def test_report_includes_charts_for_every_measurement(mirror):
    _circuit, run, analysis = mirror
    html = render_html_report(analysis, run)
    assert html.count("<svg") >= len(analysis.statistics)
    for name in analysis.statistics:
        assert name in html


def test_report_declares_the_model_limitation(mirror):
    _circuit, run, analysis = mirror
    html = render_html_report(analysis, run)
    assert "not BSIM" in html or "square-law" in html
    assert "limitations" in html.lower()


def test_report_embeds_the_netlist_when_supplied(mirror):
    circuit, run, analysis = mirror
    html = render_html_report(analysis, run, netlist=circuit.source_text or "")
    assert "Circuit netlist" in html
    assert "M1 nref nref 0 0 NCH" in html


def test_report_escapes_hostile_content(mirror):
    """A netlist is untrusted input and ends up inside the report."""
    _circuit, run, analysis = mirror
    html = render_html_report(analysis, run,
                              netlist="* <script>alert(1)</script>\n")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_report_handles_a_run_with_no_usable_samples(tmp_path):
    from siliconstat.core.solver import SolverOptions

    circuit = load("two_stage_opamp")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=3, seed=1, workers=1,
        solver=SolverOptions(max_iter=2, gmin_steps=2, source_steps=2)))
    html = render_html_report(analyse_run(run), run)
    assert "Failed samples" in html
    assert "convergence_failure" in html


# ---------------------------------------------------------------------------
# SVG chart helpers on awkward input
# ---------------------------------------------------------------------------

def test_charts_render_for_normal_data(mirror):
    _circuit, run, analysis = mirror
    chart = analysis.charts["ierr"]
    stats = analysis.statistics["ierr"].to_dict()
    assert svg_histogram(chart["histogram"], stats, chart["specs"]).startswith("<svg")
    assert svg_cdf(chart["cdf"], chart["specs"]).startswith("<svg")
    assert svg_sigma_plot(chart["sigma_plot"]).startswith("<svg")
    assert svg_convergence(analysis.convergence["ierr"].to_dict()).startswith("<svg")
    assert svg_heatmap(analysis.correlation.pearson,
                       analysis.correlation.parameters,
                       analysis.correlation.measurements).startswith("<svg")
    assert svg_barh(["a", "b"], [1.0, 2.0]).startswith("<svg")
    assert svg_box([("run A", stats), ("run B", stats)]).startswith("<svg")


def test_charts_do_not_crash_on_degenerate_input():
    assert svg_histogram({"counts": [], "edges": []}, {}) == ""
    assert svg_cdf({"x": [1.0], "p": [0.5]}) == ""
    assert svg_sigma_plot({"x": [1.0], "sigma": [0.0]}) == ""
    assert svg_convergence({"n": [1], "mean": [0.0]}) == ""
    assert svg_heatmap([], [], []) == ""
    assert svg_barh([], []) == ""
    assert svg_box([]) == ""


def test_heatmap_renders_nan_cells_as_not_available():
    markup = svg_heatmap([[float("nan")]], ["p"], ["m"])
    assert "n/a" in markup


def test_barh_marks_negative_values_differently():
    markup = svg_barh(["a", "b"], [1.0, -1.0])
    assert markup.count("<rect") >= 3      # panel plus one bar each
