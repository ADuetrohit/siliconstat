"""End-to-end acceptance test.

Walks the complete workflow the project promises, from a clean database to a
rendered report, and asserts each step.  Every number printed comes from the
step that just ran.

    .venv/Scripts/python.exe scripts/acceptance.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "acceptance"
DB = OUT / "acceptance.sqlite"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable

STEP = 0
FAILURES: list[str] = []


def step(title: str) -> None:
    global STEP
    STEP += 1
    print(f"\n{'=' * 78}\nSTEP {STEP:2d}. {title}\n{'=' * 78}")


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"   [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)
    return condition


def main() -> int:  # noqa: C901 - a linear script is the clearest form here
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    # -- 1-2 environment ---------------------------------------------------
    step("Clean environment and installed dependencies")
    import numpy, pandas, scipy  # noqa: F401

    import siliconstat
    print(f"   python      {sys.version.split()[0]}")
    print(f"   siliconstat {siliconstat.__version__}")
    print(f"   numpy {numpy.__version__} · scipy {scipy.__version__} · "
          f"pandas {pandas.__version__}")
    check("fresh database directory", not DB.exists())

    from siliconstat.analysis import analyse_run, sensitivity_analysis
    from siliconstat.core import parse_netlist_file, solve_dc
    from siliconstat.db import RunStore
    from siliconstat.mc import MonteCarloConfig, run_monte_carlo
    from siliconstat.measure import evaluate_measurements
    from siliconstat.report import export_bundle
    from siliconstat.variation import default_mismatch_model

    # -- 3-4 servers -------------------------------------------------------
    step("Start the backend and confirm the built frontend is served")
    env = dict(os.environ, SILICONSTAT_DB=str(DB))
    server = subprocess.Popen(
        [PY, "-m", "uvicorn", "siliconstat.api.main:app",
         "--host", "127.0.0.1", "--port", "8021", "--log-level", "warning"],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = "http://127.0.0.1:8021"
    try:
        health = None
        for _ in range(80):
            try:
                with urllib.request.urlopen(f"{base}/api/health", timeout=2) as fh:
                    health = json.load(fh)
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        check("API answers /api/health", health is not None,
              f"version {health['version']}" if health else "no response")

        dist = ROOT / "frontend" / "dist" / "index.html"
        check("frontend build exists", dist.exists(),
              f"{dist.stat().st_size:,} bytes" if dist.exists() else "run npm run build")
        if dist.exists():
            with urllib.request.urlopen(f"{base}/", timeout=5) as fh:
                body = fh.read().decode("utf-8", "replace")
            check("API serves the dashboard at /", "<div id=\"root\">" in body)

        for path in ("/api/version", "/api/examples", "/api/runs", "/api/db/stats"):
            with urllib.request.urlopen(f"{base}{path}", timeout=5) as fh:
                check(f"GET {path}", fh.status == 200)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            server.kill()
    print("   backend stopped")

    # -- 5-6 nominal -------------------------------------------------------
    step("Load the current mirror and run a nominal simulation")
    circuit = parse_netlist_file(str(ROOT / "examples" / "current_mirror.net"))
    op = solve_dc(circuit)
    nominal = evaluate_measurements(circuit, op=op)
    print(f"   {op.iterations} Newton iterations, strategy '{op.strategy}', "
          f"residual {op.residual:.3e} A")
    for name, outcome in nominal.outcomes.items():
        print(f"   {name:6s} = {outcome.value:14.8g} {outcome.unit}")
    check("Iref is the forced 10 uA",
          abs(nominal.values["iref"] - 10e-6) < 1e-11,
          f"{nominal.values['iref']:.9g} A")
    check("nominal copy error is small and non-zero",
          0.0 < nominal.values["ierr"] < 0.2, f"{nominal.values['ierr']:.6f} %")

    store = RunStore(str(DB))

    # -- 7-10 the three variation modes ------------------------------------
    step("Run 1000 samples: mismatch only, process only, and both")
    runs = {}
    for mode, process, mismatch in (("mismatch", False, True),
                                    ("process", True, False),
                                    ("both", True, True)):
        model = default_mismatch_model(circuit, enable_process=process,
                                       enable_mismatch=mismatch)
        started = time.perf_counter()
        run = run_monte_carlo(circuit, MonteCarloConfig(
            variation=model, samples=1000, seed=12345, workers=1, label=mode))
        elapsed = time.perf_counter() - started
        analysis = analyse_run(run)
        store.save_run(run, circuit=circuit, project="acceptance", label=mode)
        store.save_analysis(run.run_id, analysis.to_dict())
        runs[mode] = (run, analysis)
        stats = analysis.statistics["ierr"]
        print(f"   {mode:9s} {run.counters.total} attempted, "
              f"{run.counters.successful} ok, {run.counters.failed} failed · "
              f"sigma(ierr) = {stats.std:.4f} % · "
              f"{elapsed:.1f} s ({run.counters.total / elapsed:.0f}/s)")
        check(f"{mode}: every sample accounted for",
              run.counters.total == 1000
              and run.counters.successful + run.counters.failed == 1000)

    sigma_mismatch = runs["mismatch"][1].statistics["ierr"].std
    sigma_process = runs["process"][1].statistics["ierr"].std
    sigma_both = runs["both"][1].statistics["ierr"].std
    check("local mismatch dominates the spread",
          sigma_mismatch > 8 * sigma_process,
          f"mismatch {sigma_mismatch:.4f} % vs process {sigma_process:.4f} %")
    check("process+mismatch is at least the mismatch-only spread",
          sigma_both >= sigma_mismatch * 0.95,
          f"both {sigma_both:.4f} %")

    run, analysis = runs["both"]

    # -- 11 statistics -----------------------------------------------------
    step("Statistics")
    stats = analysis.statistics["ierr"]
    for key in ("count", "mean", "median", "std", "variance", "minimum", "maximum",
                "cv", "p1", "p5", "p25", "p75", "p95", "p99", "sigma3_low",
                "sigma3_high", "skewness", "kurtosis_excess", "normality_p"):
        print(f"   {key:16s} {getattr(stats, key)}")
    check("every measurement has statistics",
          set(analysis.statistics) == set(run.measurement_names))
    check("3-sigma limits are consistent",
          abs(stats.sigma3_high - (stats.mean + 3 * stats.std)) < 1e-12)

    # -- 12 yield ----------------------------------------------------------
    step("Yield")
    report = analysis.yield_report
    for line in report.summary_lines():
        print(f"   {line}")
    print(f"   over all attempted: {report.combined_yield_over_attempted:.2f} %")
    for note in report.notes:
        print(f"   note: {note}")
    check("yield states its denominator",
          report.successful == run.counters.successful and report.per_spec != [])
    check("combined yield never exceeds the weakest spec",
          report.combined_passing <= min(s.passing for s in report.per_spec))

    # -- 13-14 charts ------------------------------------------------------
    step("Histogram and CDF payloads")
    chart = analysis.charts["ierr"]
    print(f"   histogram: {len(chart['histogram']['counts'])} bins, "
          f"{sum(chart['histogram']['counts'])} samples")
    print(f"   cdf:       {chart['cdf']['n']} points, "
          f"p in [{chart['cdf']['p'][0]:.5f}, {chart['cdf']['p'][-1]:.5f}]")
    check("histogram conserves the sample count",
          sum(chart["histogram"]["counts"]) == stats.count)
    check("CDF uses the Hazen plotting position",
          abs(chart["cdf"]["p"][0] - 0.5 / stats.count) < 1e-9)

    # -- 15 correlation ----------------------------------------------------
    step("Correlation matrix")
    correlation = analysis.correlation
    print(f"   {len(correlation.parameters)} parameters x "
          f"{len(correlation.measurements)} measurements, "
          f"n = {correlation.n_samples}")
    top = correlation.top_pairs("ierr", limit=3)
    for row in top:
        print(f"   {row['parameter']:22s} pearson {row['pearson']:+.4f} "
              f"spearman {row['spearman']:+.4f}")
    check("correlation covers every varied parameter",
          correlation.parameters == run.slot_names)

    # -- 16 sensitivity ----------------------------------------------------
    step("Sensitivity ranking")
    sensitivity = sensitivity_analysis(run, "ierr")
    print(f"   method {sensitivity.method}, R2 = {sensitivity.r_squared:.5f}, "
          f"unexplained {sensitivity.unexplained_pct:.2f} %")
    for entry in sensitivity.entries[:5]:
        print(f"   {entry.rank}. {entry.parameter:22s} beta* "
              f"{entry.beta_standardised:+.4f}  variance "
              f"{entry.variance_contribution_pct:6.2f} %")
    top_two = {e.parameter for e in sensitivity.entries[:2]}
    check("the two local thresholds dominate",
          top_two == {"M1.vth_local", "M2.vth_local"}, ", ".join(sorted(top_two)))
    check("they explain most of the variance",
          sum(e.variance_contribution_pct for e in sensitivity.entries[:2]) > 80.0)

    # -- 17-18 exports -----------------------------------------------------
    step("Export raw CSV and generate the HTML report")
    written = export_bundle(run, analysis, str(OUT),
                            netlist=circuit.source_text or "")
    for name, path in written.items():
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"   {name:14s} {os.path.basename(path):46s} {size:>10,} bytes")
    import pandas as pd

    frame = pd.read_csv(written["csv"])
    check("CSV has one row per attempted sample", len(frame) == 1000)
    for column in ("sample_id", "seed", "simulation_status", "failure_reason",
                   "pass_fail", "meas.ierr", "var.M1.vth_local", "dev.M1.dvth"):
        check(f"CSV column {column}", column in frame.columns)

    html = Path(written["html"]).read_text(encoding="utf-8")
    check("report is self-contained (no scripts, no remote refs)",
          "<script" not in html.lower() and "https://" not in html)
    check("report contains charts", html.count("<svg") >= 5,
          f"{html.count('<svg')} SVG elements")
    check("report states the seed and the denominator",
          "12345" in html and "successful simulations" in html)

    # -- 19-20 reproducibility ---------------------------------------------
    step("Re-run with the same seed and verify reproducibility")
    model = default_mismatch_model(circuit)
    repeat = run_monte_carlo(circuit, MonteCarloConfig(
        variation=model, samples=1000, seed=12345, workers=1))
    identical = 0
    compared = 0
    for a, b in zip(run.samples, repeat.samples):
        if a.seed != b.seed or a.status != b.status:
            continue
        for name, value in a.measurements.items():
            compared += 1
            other = b.measurements.get(name)
            if value == other or (value != value and other != other):
                identical += 1
    print(f"   {compared} measurement values compared, {identical} identical")
    check("the same seed reproduces every value", identical == compared)

    different = run_monte_carlo(circuit, MonteCarloConfig(
        variation=model, samples=200, seed=999, workers=1))
    check("a different seed produces different samples",
          different.samples[0].measurements["ierr"]
          != run.samples[0].measurements["ierr"])

    reloaded = store.load_run(run.run_id)
    check("a stored run reloads with identical values",
          all(a.measurements == b.measurements
              for a, b in zip(run.samples, reloaded.samples)))

    # -- 21 test suite -----------------------------------------------------
    step("Run the complete automated test suite")
    started = time.perf_counter()
    completed = subprocess.run([PY, "-W", "ignore", "-m", "pytest", "tests/", "-q",
                                "--no-header", "-p", "no:cacheprovider"],
                               cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()][-1:]
    print(f"   {tail[0] if tail else '(no output)'}")
    print(f"   wall time {elapsed:.1f} s")
    check("test suite passes", completed.returncode == 0)

    # -- summary -----------------------------------------------------------
    print(f"\n{'=' * 78}")
    if FAILURES:
        print(f"ACCEPTANCE FAILED — {len(FAILURES)} check(s) did not pass:")
        for failure in FAILURES:
            print(f"   - {failure}")
        return 1
    print(f"ACCEPTANCE PASSED — all {STEP} steps completed, every check green.")
    print(f"artefacts in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
