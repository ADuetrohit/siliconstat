"""Regenerate every measured number quoted in the documentation.

Run from the project root:

    .venv/Scripts/python.exe docs/scripts/collect_doc_numbers.py

Nothing in docs/ should contain a number that this script (or the benchmark
script, or the test suite) cannot reproduce.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from siliconstat.analysis import analyse_run, sensitivity_analysis
from siliconstat.analysis.convergence import required_samples_for_margin
from siliconstat.analysis.statistics import welford
from siliconstat.core import parse_netlist_file, solve_dc
from siliconstat.core.netlist import parse_netlist
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.measure import evaluate_measurements
from siliconstat.variation import default_mismatch_model
from siliconstat.variation.pelgrom import area_um2


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def naive_variance(values) -> float:
    n = len(values)
    total, total_sq = sum(values), sum(v * v for v in values)
    return (total_sq - total * total / n) / (n - 1)


def nominal_table() -> None:
    rule("1. NOMINAL OPERATING POINTS  (docs/validation.md, README)")
    for name in ("rc_divider", "diode_bias", "current_mirror", "diff_pair",
                 "two_stage_opamp", "inverter_transient"):
        circuit = parse_netlist_file(f"examples/{name}.net")
        start = time.perf_counter()
        result = evaluate_measurements(circuit)
        elapsed = (time.perf_counter() - start) * 1e3
        op = solve_dc(circuit)
        print(f"\n{name}:  {op.iterations} Newton iterations, strategy "
              f"'{op.strategy}', residual {op.residual:.3e} A, {elapsed:.1f} ms")
        for key, outcome in result.outcomes.items():
            flag = "" if outcome.ok else f"   INVALID: {outcome.reason[:60]}"
            print(f"    {key:8s} = {outcome.value:16.8g} {outcome.unit}{flag}")


def solver_accuracy() -> None:
    rule("2. SOLVER ACCURACY  (docs/circuit_solver.md)")
    circuit = parse_netlist_file("examples/rc_divider.net")
    op = solve_dc(circuit)
    print(f"resistive divider   V(mid) = {op.v('mid'):.15f} V "
          f"(exact 3.75), error {abs(op.v('mid') - 3.75):.3e}")
    print(f"                    I(R1)  = {op.i('R1'):.15e} A (exact 1.25e-3)")

    circuit = parse_netlist_file("examples/current_mirror.net")
    op = solve_dc(circuit)
    kcl = circuit.device("R1").current(op.x, op.ctx) - op.i("M2")
    print(f"current mirror      KCL residual at 'out' = {kcl:.3e} A")
    lam = circuit.mos_models["NCH"].lambda_
    vds1 = op.device_ops["M1"]["vds"]
    vds2 = op.device_ops["M2"]["vds"]
    predicted = (1 + lam * vds2) / (1 + lam * vds1)
    print(f"                    Iout/Iref  measured {op.i('M2') / op.i('M1'):.12f}")
    print(f"                    (1+L*Vds2)/(1+L*Vds1) {predicted:.12f}")

    rc = parse_netlist(
        "V1 in 0 DC 0 AC 1\nR1 in out 1k\nC1 out 0 1n\n"
        ".ac dec 30 100 100MEG\n.measure bw BW in=in out=out\n")
    measured = evaluate_measurements(rc).values["bw"]
    analytic = 1.0 / (2 * math.pi * 1e3 * 1e-9)
    print(f"RC low-pass         f3dB measured {measured:.6f} Hz, analytic "
          f"{analytic:.6f} Hz, rel err {abs(measured - analytic) / analytic:.2e}")


def welford_table() -> None:
    rule("3. WELFORD vs NAIVE VARIANCE  (docs/statistics.md)")
    print(f"{'mean':>10} {'sigma':>10} {'n':>7} {'naive rel err':>15} "
          f"{'welford rel err':>16}")
    for mean, sigma, n in ((1.2, 3e-4, 20_000), (1.2, 3e-6, 20_000),
                           (1e6, 1e-3, 20_000), (1e8, 1e-3, 5_000)):
        data = (mean + sigma * np.random.default_rng(7).standard_normal(n)).tolist()
        reference = float(np.var(np.array(data), ddof=1))
        _n, _m, stable = welford(data)
        naive = naive_variance(data)
        print(f"{mean:>10g} {sigma:>10g} {n:>7} "
              f"{abs(naive - reference) / reference:>15.2e} "
              f"{abs(stable - reference) / reference:>16.2e}")


def mismatch_theory() -> None:
    rule("4. MISMATCH vs ANALYTICAL THEORY  (docs/mismatch.md, docs/validation.md)")

    circuit = parse_netlist_file("examples/current_mirror.net")
    model = default_mismatch_model(circuit)
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=model, samples=2000, seed=12345, workers=1))
    analysis = analyse_run(run)
    stats = analysis.statistics["ierr"]
    op = solve_dc(circuit)

    m1 = circuit.device("M1")
    card = circuit.mos_models["NCH"]
    area = area_um2(m1.w, m1.l)
    sigma_dvth = card.avt / math.sqrt(area)          # pair difference
    sigma_dbeta = card.abeta / math.sqrt(area)
    gm_over_id = op.device_ops["M1"]["gm"] / op.device_ops["M1"]["id"]
    lam, r_load = card.lambda_, circuit.device("R1").r
    feedback = 1.0 + lam * r_load * op.i("M2") / (1 + lam * op.v("out"))
    sigma_r = math.sqrt(2) * 0.005                   # two independent 0.5 % ...
    intrinsic = math.hypot(gm_over_id * sigma_dvth * 100.0, sigma_dbeta * 100.0)
    predicted = intrinsic / feedback

    print(f"current mirror, {run.counters.total} samples, seed 12345")
    print(f"    measured   sigma(ierr) = {stats.std:.4f} %")
    print(f"    predicted  sigma(ierr) = {predicted:.4f} %")
    print(f"        gm/Id            = {gm_over_id:.3f} 1/V")
    print(f"        sigma(dVth) pair = {sigma_dvth * 1e3:.4f} mV")
    print(f"        sigma(dbeta/beta)= {sigma_dbeta * 100:.4f} %")
    print(f"        CLM feedback     = {feedback:.4f}")
    print(f"    mean = {stats.mean:.4f} %   nominal = {run.nominal['ierr']:.4f} %")
    print(f"    combined yield = "
          f"{analysis.yield_report.combined_yield_over_successful:.2f} % "
          f"[{analysis.yield_report.combined_ci95_low:.2f}, "
          f"{analysis.yield_report.combined_ci95_high:.2f}]")
    for entry in analysis.yield_report.per_spec:
        print(f"        {entry.description:22s} {entry.yield_pct:6.2f} %  "
              f"Cpk {entry.cpk:.2f}")
    print(f"    product of individual = "
          f"{analysis.yield_report.independent_product_pct:.2f} %")

    report = sensitivity_analysis(run, "ierr")
    print(f"    sensitivity  R^2 = {report.r_squared:.5f}, "
          f"max |r| between inputs {report.max_input_correlation:.3f}")
    for entry in report.entries:
        print(f"        {entry.rank}. {entry.parameter:20s} beta* "
              f"{entry.beta_standardised:+.4f}   variance "
              f"{entry.variance_contribution_pct:6.2f} %")

    # ---- differential pair offset -------------------------------------
    circuit = parse_netlist_file("examples/diff_pair.net")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=2000, seed=12345,
        workers=1))
    measured = float(np.std(run.values("vos"), ddof=1))
    op = solve_dc(circuit)
    m1 = circuit.device("M1")
    card = circuit.mos_models["NCH"]
    area = area_um2(m1.w, m1.l)
    vov = op.device_ops["M1"]["vov"]
    term_vth = card.avt / math.sqrt(area)
    term_beta = (vov / 2.0) * (card.abeta / math.sqrt(area))
    term_r = (vov / 2.0) * math.sqrt(2) * 0.005
    predicted = math.sqrt(term_vth ** 2 + term_beta ** 2 + term_r ** 2)
    print(f"\ndifferential pair, {run.counters.total} samples, seed 12345")
    print(f"    measured  sigma(Vos) = {measured * 1e6:.2f} uV")
    print(f"    predicted sigma(Vos) = {predicted * 1e6:.2f} uV")
    print(f"        dVth term  {term_vth * 1e6:8.2f} uV   (Vov = {vov:.4f} V)")
    print(f"        dbeta term {term_beta * 1e6:8.2f} uV")
    print(f"        dR term    {term_r * 1e6:8.2f} uV")


def pelgrom_sweep() -> None:
    rule("5. PELGROM AREA SCALING  (docs/pelgrom.md)")
    text = open("examples/current_mirror.net", encoding="utf-8").read()
    print(f"{'W [um]':>8} {'L [um]':>8} {'area [um^2]':>12} "
          f"{'sigma(ierr) [%]':>16} {'1/sqrt(area) prediction':>24}")
    reference = None
    for w_um, l_um in ((10, 1), (20, 2), (40, 4), (10, 4), (40, 1)):
        variant = text.replace(".param WM=10u", f".param WM={w_um}u")
        variant = variant.replace(".param LM=1u", f".param LM={l_um}u")
        circuit = parse_netlist(variant, allow_include=False)
        run = run_monte_carlo(circuit, MonteCarloConfig(
            variation=default_mismatch_model(circuit, enable_process=False,
                                             include_passives=False),
            samples=1500, seed=909, workers=1))
        sigma = float(np.std(run.values("ierr"), ddof=1))
        area = w_um * l_um
        if reference is None:
            reference, ref_area = sigma, area
        prediction = reference * math.sqrt(ref_area / area)
        note = "" if w_um / l_um == 10 else "   (W/L changed: Vov moves too)"
        print(f"{w_um:>8} {l_um:>8} {area:>12} {sigma:>16.4f} "
              f"{prediction:>24.4f}{note}")


def convergence_planning() -> None:
    rule("6. SAMPLE-COUNT PLANNING  (docs/yield.md)")
    print(f"{'observed yield':>16} {'+/-1 %':>10} {'+/-0.5 %':>10} "
          f"{'+/-0.1 %':>12}")
    for observed in (50.0, 90.0, 95.0, 99.0, 99.9):
        row = [required_samples_for_margin(observed, m) for m in (1.0, 0.5, 0.1)]
        print(f"{observed:>15.1f}% {row[0]:>10,} {row[1]:>10,} {row[2]:>12,}")


def main() -> int:
    nominal_table()
    solver_accuracy()
    welford_table()
    mismatch_theory()
    pelgrom_sweep()
    convergence_planning()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
