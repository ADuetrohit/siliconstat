"""``siliconstat`` command-line interface.

::

    siliconstat circuit validate examples/current_mirror.net
    siliconstat simulate        examples/diff_pair.net
    siliconstat monte-carlo --circuit examples/current_mirror.net \\
                            --samples 1000 --seed 12345 --mode mismatch
    siliconstat pvt         --circuit examples/current_mirror.net --samples 200
    siliconstat report      --run <run-id> --out report.html
    siliconstat runs list
    siliconstat ml          --circuit examples/current_mirror.net
    siliconstat bench       --circuit examples/current_mirror.net
    siliconstat serve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Sequence

from .. import __version__
from ..analysis import analyse_run
from ..core.exceptions import SiliconStatError
from ..core.netlist import parse_netlist_file
from ..core.solver import SolverOptions, solve_dc
from ..core.units import format_eng
from ..db import RunStore
from ..mc import MonteCarloConfig, run_monte_carlo
from ..measure import evaluate_measurements
from ..variation import PVTCondition, VariationSampler, default_mismatch_model
from ..variation.pvt import CORNERS, DEFAULT_TEMPERATURES, find_supply_source
from .format import C, bar, colour, kv, rule, table

DEFAULT_DB = os.environ.get("SILICONSTAT_DB", "siliconstat.sqlite")

MODES = {
    "nominal": (False, False),
    "process": (True, False),
    "mismatch": (False, True),
    "both": (True, True),
    "process+mismatch": (True, True),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_circuit(path: str):
    if not os.path.isfile(path):
        raise SiliconStatError(f"netlist not found: {path}")
    return parse_netlist_file(path)


def _build_variation(circuit, args: argparse.Namespace):
    process, mismatch = MODES[args.mode]
    model = default_mismatch_model(
        circuit,
        sigma_vth_global=args.sigma_vth_global,
        sigma_beta_global_pct=args.sigma_beta_global,
        enable_process=process,
        enable_mismatch=mismatch,
        include_passives=not args.no_passives,
    )
    if args.no_pelgrom:
        for v in model.variations:
            if v.pelgrom:
                v.pelgrom = False
                v.sigma = args.sigma_vth_local if v.parameter == "vth" else None
                v.sigma_pct = None if v.parameter == "vth" else args.sigma_beta_local
        model.variations = [v for v in model.variations
                            if v.sigma is not None or v.sigma_pct is not None]
    return model


def _pvt_from_args(args: argparse.Namespace) -> PVTCondition | None:
    if args.corner == "TT" and args.supply is None and args.temp is None:
        return None
    return PVTCondition(
        corner=args.corner,
        supply=args.supply,
        temp_c=27.0 if args.temp is None else args.temp,
        sigma_vth_global=args.sigma_vth_global,
        sigma_beta_global_pct=args.sigma_beta_global,
    )


def _progress(state: dict[str, Any]) -> None:
    done, total = state["completed"], state["total"]
    frac = done / total if total else 1.0
    elapsed = state["elapsed_s"]
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    line = (f"\r  [{bar(frac)}] {done:>6}/{total}  "
            f"ok={colour(str(state['successful']), C.GREEN)} "
            f"fail={colour(str(state['failed']), C.RED if state['failed'] else C.GREY)} "
            f"{rate:6.1f}/s  eta {eta:5.1f}s")
    sys.stderr.write(line)
    sys.stderr.flush()
    if done >= total:
        sys.stderr.write("\n")


def _unit_of(circuit, name: str) -> str:
    for m in circuit.measures:
        if m.name == name:
            return m.unit
    return ""


def _fmt_value(value: float, unit: str) -> str:
    if value != value:
        return "n/a"
    if unit in ("", "%", "-", "dB", "deg"):
        return f"{value:.6g}{(' ' + unit) if unit else ''}"
    return format_eng(value, unit)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_circuit_validate(args: argparse.Namespace) -> int:
    circuit = _load_circuit(args.netlist)
    print(rule(f"circuit {circuit.name}"))
    print(kv([
        ("file", args.netlist),
        ("devices", len(circuit.devices)),
        ("nodes", f"{circuit.n_nodes} (incl. ground)"),
        ("MNA unknowns", circuit.size),
        ("models", ", ".join(sorted(circuit.mos_models) +
                             sorted(circuit.diode_models)) or "(none)"),
        ("matched groups", ", ".join(circuit.describe()["matched_groups"]) or "(none)"),
        ("measurements", len(circuit.measures)),
        ("specifications", len(circuit.specs)),
        ("analyses", ", ".join(a.kind for a in circuit.analyses) or "(op only)"),
        ("temperature", f"{circuit.temp_c} C"),
    ]))
    print()
    print(table(["device", "type", "nodes", "parameters"],
                [[d["name"], d["type"], " ".join(d["nodes"]),
                  " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in d.items()
                           if k in ("r", "c", "l", "dc", "w", "m", "model",
                                    "matched_group") and v is not None)]
                 for d in circuit.describe()["devices"]]))
    print()
    print(table(["measurement", "kind", "unit", "definition"],
                [[m.name, m.kind, m.unit,
                  m.args.get("expression") or
                  " ".join(f"{k}={v}" for k, v in m.args.items())]
                 for m in circuit.measures]))
    if circuit.specs:
        print()
        print(table(["specification"], [[s.describe()] for s in circuit.specs]))

    if not args.no_solve:
        print()
        try:
            op = solve_dc(circuit)
            print(colour(f"  DC operating point converged "
                         f"({op.iterations} Newton iterations, strategy "
                         f"'{op.strategy}', KCL residual "
                         f"{op.residual:.2e} A)", C.GREEN))
        except SiliconStatError as exc:
            print(colour(f"  DC operating point FAILED: {exc}", C.RED))
            return 2
    print(colour("\n  netlist is valid", C.GREEN))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    circuit = _load_circuit(args.netlist)
    pvt = _pvt_from_args(args)
    ctx = pvt.build_context(circuit) if pvt else circuit.build_context()
    opts = SolverOptions.from_options(circuit.options)

    started = time.perf_counter()
    op = solve_dc(circuit, ctx, opts)
    elapsed = (time.perf_counter() - started) * 1e3

    print(rule(f"operating point -- {circuit.name}"))
    if pvt:
        print(kv([("PVT condition", pvt.label)]))
    print(kv([
        ("solver", f"{op.iterations} iterations, strategy '{op.strategy}'"),
        ("KCL residual", f"{op.residual:.3e} A"),
        ("wall time", f"{elapsed:.2f} ms"),
        ("supply power", format_eng(op.total_supply_power(), "W")),
    ]))
    print()
    print(table(["node", "voltage"],
                [[name, format_eng(v, "V")]
                 for name, v in op.node_voltages.items() if name != "0"],
                align="lr"))
    mosfets = [(n, d) for n, d in op.device_ops.items() if d.get("type") == "mosfet"]
    if mosfets:
        print()
        print(table(["device", "model", "region", "Id", "Vgs", "Vds", "Vth",
                     "Vov", "gm", "ro", "gm/Id"],
                    [[n, d["model"], d["region"], format_eng(d["id"], "A"),
                      f"{d['vgs']:.4f}", f"{d['vds']:.4f}", f"{d['vth']:.4f}",
                      f"{d['vov']:.4f}", format_eng(d["gm"], "S"),
                      format_eng(d["ro"], "ohm"), f"{d['gm_over_id']:.2f}"]
                     for n, d in mosfets], align="lllrrrrrrrr"))

    result = evaluate_measurements(circuit, ctx, opts, op=op)
    if result.outcomes:
        print()
        print(table(["measurement", "value", "unit", "status"],
                    [[n, f"{o.value:.6g}", o.unit,
                      colour("ok", C.GREEN) if o.ok
                      else colour(f"INVALID: {o.reason[:70]}", C.RED)]
                     for n, o in result.outcomes.items()], align="lrll"))
        print(colour(f"\n  analyses run: {', '.join(result.analyses_run)}", C.GREY))
    if circuit.specs:
        print()
        rows = []
        for spec in circuit.specs:
            value = result.values.get(spec.measure, float("nan"))
            ok = result.outcomes.get(spec.measure)
            passed = bool(ok and ok.ok and spec.passes(value))
            rows.append([spec.describe(), f"{value:.6g}",
                         colour("PASS", C.GREEN) if passed
                         else colour("FAIL", C.RED)])
        print(table(["specification", "nominal value", "verdict"], rows,
                    align="lrl"))
    if args.json:
        payload = {"operating_point": op.to_dict(),
                   "measurements": result.to_dict()}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(colour(f"\n  wrote {args.json}", C.CYAN))
    return 0


def cmd_monte_carlo(args: argparse.Namespace) -> int:
    circuit = _load_circuit(args.circuit)
    variation = _build_variation(circuit, args)
    pvt = _pvt_from_args(args)
    config = MonteCarloConfig(
        variation=variation, samples=args.samples, seed=args.seed,
        workers=args.workers, sampling=args.sampling, pvt=pvt,
        solver=SolverOptions.from_options(circuit.options), label=args.label)

    sampler = VariationSampler(circuit, variation)
    print(rule(f"monte carlo -- {circuit.name}"))
    print(kv([
        ("mode", args.mode),
        ("samples", args.samples),
        ("seed", args.seed),
        ("sampling", args.sampling),
        ("workers", args.workers),
        ("PVT", pvt.label if pvt else "nominal (TT / nominal V / 27 C)"),
        ("random variables", sampler.n_slots),
    ]))
    print()
    print(table(["random variable", "parameter", "scope", "distribution",
                 "sigma", "unit", "devices"],
                [[r["slot"], r["parameter"], r["scope"], r["distribution"],
                  f"{r['sigma']:.6g}", r["unit"], r["devices"]]
                 for r in sampler.slot_sigma_table()], align="llllrll"))
    print()

    run = run_monte_carlo(circuit, config,
                          progress=None if args.quiet else _progress)
    analysis = analyse_run(run)
    _print_run_summary(circuit, run, analysis)

    store = RunStore(args.db)
    store.save_run(run, circuit=circuit, project=args.project, label=args.label)
    store.save_analysis(run.run_id, analysis.to_dict())
    print(colour(f"\n  run {run.run_id} stored in {args.db}", C.CYAN))

    if args.out:
        from ..report import export_bundle
        written = export_bundle(run, analysis, args.out,
                                netlist=circuit.source_text or "")
        print()
        for name, path in written.items():
            print(colour(f"  {name:16s} {path}", C.CYAN))
    return 0


def _print_run_summary(circuit, run, analysis) -> None:
    counters = run.counters
    print()
    print(rule("simulation status"))
    print(kv([
        ("total", counters.total),
        ("successful", colour(str(counters.successful), C.GREEN)),
        ("failed", colour(str(counters.failed),
                          C.RED if counters.failed else C.GREY)),
        ("convergence failures", counters.convergence_failures),
        ("numerical errors", counters.numerical_errors),
        ("invalid measurements", counters.invalid_measurements),
        ("wall time", f"{run.duration_s:.2f} s "
                      f"({counters.total / run.duration_s:.1f} samples/s)"
         if run.duration_s > 0 else "n/a"),
    ]))
    if run.notes:
        print(colour(f"  note: {run.notes}", C.YELLOW))

    if analysis.failure_breakdown:
        print()
        print(rule("failure analysis"))
        print(table(["status", "reason", "count", "%"],
                    [[colour(r["status"], C.RED), r["reason"][:80],
                      r["count"], f"{r['percent']:.2f}"]
                     for r in analysis.failure_breakdown], align="llrr"))

    print()
    print(rule("statistics"))
    rows = []
    for name, st in analysis.statistics.items():
        unit = st.unit
        rows.append([
            name, unit, st.count,
            _fmt_value(st.mean, unit), _fmt_value(st.std, unit),
            _fmt_value(st.sigma3_low, unit), _fmt_value(st.sigma3_high, unit),
            _fmt_value(st.minimum, unit), _fmt_value(st.maximum, unit),
            f"{analysis.nominal.get(name, float('nan')):.6g}",
        ])
    print(table(["measurement", "unit", "N", "mean", "sigma", "-3sigma",
                 "+3sigma", "min", "max", "nominal"], rows,
                align="llrrrrrrrr"))

    yr = analysis.yield_report
    print()
    print(rule("yield"))
    if yr.per_spec:
        print(table(["specification", "pass", "fail", "yield", "95% CI",
                     "margin [sigma]", "Cpk"],
                    [[s.description, s.passing, s.failing,
                      colour(f"{s.yield_pct:6.2f}%",
                             C.GREEN if s.yield_pct >= 99 else
                             (C.YELLOW if s.yield_pct >= 90 else C.RED)),
                      f"[{s.ci95_low:.2f}, {s.ci95_high:.2f}]",
                      f"{s.margin_sigma:.2f}" if s.margin_sigma == s.margin_sigma else "n/a",
                      f"{s.cpk:.2f}" if s.cpk == s.cpk else "n/a"]
                     for s in yr.per_spec], align="lrrrrrr"))
        print()
        print(kv([
            ("COMBINED YIELD",
             colour(f"{yr.combined_yield_over_successful:.2f}%",
                    C.GREEN if yr.combined_yield_over_successful >= 99 else
                    (C.YELLOW if yr.combined_yield_over_successful >= 90 else C.RED))
             + f"  [{yr.combined_ci95_low:.2f}, {yr.combined_ci95_high:.2f}]"),
            ("denominator", f"{yr.combined_passing} / {yr.successful} "
                            "successful simulations"),
            ("over all attempted", f"{yr.combined_yield_over_attempted:.2f}% "
                                   f"({yr.combined_passing} / {yr.attempted})"),
            ("limiting spec", yr.limiting_spec),
        ]))
        for note in yr.notes:
            print(colour(f"  note: {note}", C.YELLOW))
    else:
        for note in yr.notes:
            print(colour(f"  {note}", C.YELLOW))

    for name, sens in analysis.sensitivity.items():
        if not sens.entries or all(e.variance_contribution_pct != e.variance_contribution_pct
                                   for e in sens.entries):
            continue
        top = [e for e in sens.entries[:6]]
        if not top or (top[0].variance_contribution_pct or 0) < 0.5:
            continue
        print()
        print(rule(f"sensitivity -- {name} "
                   f"(R2 = {sens.r_squared:.4f}, "
                   f"{sens.unexplained_pct:.1f}% unexplained)"))
        print(table(["#", "parameter", "scope", "beta*", "variance share",
                     "pearson r"],
                    [[e.rank, e.parameter, e.scope, f"{e.beta_standardised:+.4f}",
                      f"{e.variance_contribution_pct:6.2f}%",
                      f"{e.pearson:+.4f}"] for e in top], align="rlrrrr"))

    for name, trace in analysis.convergence.items():
        print()
        print(rule(f"convergence -- {name}"))
        print(kv([
            ("final mean", f"{trace.final_mean:.6g}"),
            ("final sigma", f"{trace.final_std:.6g}"),
            ("final yield", f"{trace.final_yield_pct:.2f}%"
             if trace.final_yield_pct == trace.final_yield_pct else "n/a"),
            ("mean settled at", trace.mean_settled_at or "not settled"),
            ("sigma settled at", trace.std_settled_at or "not settled"),
            ("yield settled at", trace.yield_settled_at or "not settled"),
        ]))
        break  # one convergence block is enough for the terminal


def cmd_pvt(args: argparse.Namespace) -> int:
    circuit = _load_circuit(args.circuit)
    variation = _build_variation(circuit, args)
    supply_nominal = find_supply_source(circuit).dc
    corners = args.corners or list(CORNERS)
    supplies = args.supplies or [supply_nominal * 0.9, supply_nominal,
                                 supply_nominal * 1.1]
    temps = args.temps or list(DEFAULT_TEMPERATURES)

    print(rule(f"PVT x Monte Carlo -- {circuit.name}"))
    print(kv([("corners", ", ".join(corners)),
              ("supplies", ", ".join(f"{v:.3g} V" for v in supplies)),
              ("temperatures", ", ".join(f"{t:g} C" for t in temps)),
              ("samples per condition", args.samples),
              ("total simulations", args.samples * len(corners) * len(supplies) * len(temps)),
              ("mode", args.mode)]))
    print()

    store = RunStore(args.db)
    rows = []
    numeric: list[tuple[float, str, float, float]] = []
    total = len(corners) * len(supplies) * len(temps)
    done = 0
    for corner in corners:
        for supply in supplies:
            for temp in temps:
                pvt = PVTCondition(
                    corner=corner, supply=float(supply), temp_c=float(temp),
                    sigma_vth_global=args.sigma_vth_global,
                    sigma_beta_global_pct=args.sigma_beta_global)
                config = MonteCarloConfig(
                    variation=variation, samples=args.samples, seed=args.seed,
                    workers=args.workers, pvt=pvt,
                    solver=SolverOptions.from_options(circuit.options),
                    label=f"PVT {pvt.label}")
                run = run_monte_carlo(circuit, config)
                analysis = analyse_run(run, charts=False)
                store.save_run(run, circuit=circuit, project=args.project,
                               label=f"PVT {pvt.label}")
                store.save_analysis(run.run_id, analysis.to_dict())
                yr = analysis.yield_report
                done += 1
                if not args.quiet:
                    sys.stderr.write(f"\r  {done}/{total} conditions simulated")
                    sys.stderr.flush()
                primary = args.metric or (run.measurement_names[0]
                                          if run.measurement_names else "")
                st = analysis.statistics.get(primary)
                rows.append([
                    corner, f"{supply:.3g}", f"{temp:g}",
                    run.counters.successful, run.counters.failed,
                    f"{st.mean:.5g}" if st else "n/a",
                    f"{st.std:.5g}" if st else "n/a",
                    (colour(f"{yr.combined_yield_over_successful:6.2f}%",
                            C.GREEN if yr.combined_yield_over_successful >= 99
                            else (C.YELLOW if yr.combined_yield_over_successful >= 90
                                  else C.RED))
                     if yr.per_spec else "n/a"),
                    run.run_id,
                ])
                if yr.per_spec:
                    numeric.append((yr.combined_yield_over_successful, corner,
                                    float(supply), float(temp)))
    if not args.quiet:
        sys.stderr.write("\n")
    print()
    metric = args.metric or "(first measurement)"
    print(table(["corner", "VDD [V]", "T [C]", "ok", "fail",
                 f"mean({metric})", f"sigma({metric})", "yield", "run id"],
                rows, align="lrrrrrrrl"))
    if numeric:
        worst = min(numeric)
        print()
        print(colour(f"  worst-case condition: {worst[1]} / {worst[2]:g} V / "
                     f"{worst[3]:g} C -> combined yield {worst[0]:.2f}%", C.YELLOW))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = RunStore(args.db)
    run = store.load_run(args.run)
    stored = store.load_analysis(args.run)
    analysis = analyse_run(run)
    netlist = store.load_netlist(args.run) or ""
    from ..report import write_html_report

    out = args.out or f"{args.run}_report.html"
    write_html_report(out, analysis, run, netlist=netlist, project=args.project)
    print(colour(f"  wrote {out} ({os.path.getsize(out):,} bytes)", C.GREEN))
    if args.export:
        from ..report import export_bundle
        written = export_bundle(run, analysis, args.export, netlist=netlist,
                                html=False)
        for name, path in written.items():
            print(colour(f"  {name:16s} {path}", C.CYAN))
    if stored is None:
        store.save_analysis(args.run, analysis.to_dict())
    return 0


def cmd_runs_list(args: argparse.Namespace) -> int:
    store = RunStore(args.db)
    runs = store.list_runs(project=args.project, limit=args.limit)
    if not runs:
        print(colour("  no stored runs", C.GREY))
        return 0
    print(table(["run id", "circuit", "mode", "PVT", "samples", "ok", "fail",
                 "seed", "duration", "started"],
                [[r.run_id, r.circuit_name, r.variation_mode,
                  r.pvt_label or "nominal", r.samples,
                  r.counters.get("successful", 0), r.counters.get("failed", 0),
                  r.seed, f"{r.duration_s:.1f}s", r.started_at]
                 for r in runs], align="lllrrrrrrl"))
    return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    store = RunStore(args.db)
    run = store.load_run(args.run)
    analysis = analyse_run(run)
    print(rule(f"run {run.run_id}"))
    print(kv(list(run.reproduction_record().items())[:8]))
    _print_run_summary(None, run, analysis)
    return 0


def cmd_runs_compare(args: argparse.Namespace) -> int:
    store = RunStore(args.db)
    runs = [store.load_run(rid) for rid in args.runs]
    analyses = [analyse_run(r, charts=False) for r in runs]
    names = sorted({n for r in runs for n in r.measurement_names})
    print(rule("run comparison"))
    print(table(["run id", "circuit", "mode", "PVT", "samples", "ok", "fail",
                 "combined yield"],
                [[r.run_id, r.circuit_name,
                  r.config.get("variation", {}).get("mode", ""),
                  (r.config.get("pvt") or {}).get("label", "nominal"),
                  r.counters.total, r.counters.successful, r.counters.failed,
                  f"{a.yield_report.combined_yield_over_successful:.2f}%"
                  if a.yield_report.per_spec else "n/a"]
                 for r, a in zip(runs, analyses)], align="llllrrrr"))
    for name in names:
        rows = []
        for r, a in zip(runs, analyses):
            st = a.statistics.get(name)
            if st is None:
                continue
            rows.append([r.run_id, st.count, f"{st.mean:.6g}", f"{st.std:.6g}",
                         f"{st.minimum:.6g}", f"{st.maximum:.6g}"])
        if rows:
            print()
            print(rule(f"measurement {name}"))
            print(table(["run id", "N", "mean", "sigma", "min", "max"], rows,
                        align="lrrrrr"))
    return 0


def cmd_ml(args: argparse.Namespace) -> int:
    from ..ml import surrogate_experiment

    circuit = _load_circuit(args.circuit)
    variation = _build_variation(circuit, args)
    result = surrogate_experiment(
        circuit, variation, target=args.metric, train_samples=args.train,
        test_samples=args.test, seed=args.seed,
        solver=SolverOptions.from_options(circuit.options),
        model_kind=args.model)
    print(rule(f"ML surrogate -- {circuit.name}"))
    print(kv([
        ("target measurement", result["target"]),
        ("model", result["model"]),
        ("training samples (real simulations)", result["train_samples"]),
        ("held-out test samples (real simulations)", result["test_samples"]),
        ("features", result["n_features"]),
    ]))
    print()
    print(rule("accuracy on held-out simulations"))
    print(kv([
        ("R2", f"{result['r2']:.6f}"),
        ("RMSE", f"{result['rmse']:.6g} ({result['rmse_pct_of_sigma']:.2f}% of sigma)"),
        ("MAE", f"{result['mae']:.6g}"),
        ("max abs error", f"{result['max_abs_error']:.6g}"),
        ("cross-validated R2", f"{result['cv_r2_mean']:.6f} "
                               f"+/- {result['cv_r2_std']:.6f}"),
    ]))
    print()
    print(rule("measured speed"))
    print(kv([
        ("simulation time / sample", f"{result['sim_time_per_sample_ms']:.3f} ms"),
        ("surrogate time / sample", f"{result['surrogate_time_per_sample_ms']:.6f} ms"),
        ("raw speedup per prediction", f"{result['raw_speedup']:.1f}x"),
        ("training cost", f"{result['train_time_s']:.2f} s "
                          f"({result['train_sim_time_s']:.2f} s of simulation)"),
        ("break-even sample count", f"{result['break_even_samples']:.0f}"),
        ("net speedup at 10,000 samples", f"{result['net_speedup_10k']:.2f}x"),
    ]))
    print()
    print(rule("yield estimate agreement"))
    print(table(["source", "yield", "note"],
                [["simulated (held-out)", f"{result['sim_yield_pct']:.2f}%",
                  "ground truth"],
                 ["surrogate", f"{result['surrogate_yield_pct']:.2f}%",
                  f"error {result['yield_error_pp']:+.2f} percentage points"]],
                align="lrl"))
    if result["notes"]:
        print()
        for note in result["notes"]:
            print(colour(f"  note: {note}", C.YELLOW))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(colour(f"\n  wrote {args.json}", C.CYAN))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from ..bench import run_benchmark

    circuit = _load_circuit(args.circuit)
    variation = _build_variation(circuit, args)
    results = run_benchmark(circuit, variation, sample_counts=args.samples_list,
                            worker_counts=args.workers_list, seed=args.seed)
    print(rule(f"benchmark -- {circuit.name}"))
    print(table(["samples", "workers", "wall time [s]", "samples/s",
                 "ok", "fail", "avg sim [ms]"],
                [[r["samples"], r["workers"], f"{r['wall_s']:.3f}",
                  f"{r['samples_per_s']:.1f}", r["successful"], r["failed"],
                  f"{r['avg_sample_ms']:.3f}"] for r in results],
                align="rrrrrrr"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(colour(f"\n  wrote {args.json}", C.CYAN))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(colour("  uvicorn is not installed; run "
                     "`pip install siliconstat[api]`", C.RED))
        return 1
    os.environ.setdefault("SILICONSTAT_DB", args.db)
    print(colour(f"  SiliconStat API on http://{args.host}:{args.port}  "
                 f"(docs at /api/docs, database {args.db})", C.CYAN))
    uvicorn.run("siliconstat.api.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level=args.log_level)
    return 0


def cmd_db_stats(args: argparse.Namespace) -> int:
    store = RunStore(args.db)
    print(kv(list(store.stats().items())))
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _add_variation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=sorted(MODES), default="both",
                        help="which variation sources to enable (default: both)")
    parser.add_argument("--sigma-vth-global", type=float, default=0.025,
                        metavar="V",
                        help="global threshold sigma in volts (default: 0.025)")
    parser.add_argument("--sigma-beta-global", type=float, default=3.0,
                        metavar="PCT",
                        help="global current-factor sigma in %% (default: 3)")
    parser.add_argument("--sigma-vth-local", type=float, default=0.003,
                        metavar="V",
                        help="local threshold sigma used when --no-pelgrom is set")
    parser.add_argument("--sigma-beta-local", type=float, default=1.0,
                        metavar="PCT",
                        help="local beta sigma in %% used when --no-pelgrom is set")
    parser.add_argument("--no-pelgrom", action="store_true",
                        help="use fixed local sigmas instead of area scaling")
    parser.add_argument("--no-passives", action="store_true",
                        help="do not vary resistors and capacitors")


def _add_pvt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corner", choices=sorted(CORNERS), default="TT")
    parser.add_argument("--supply", type=float, default=None, metavar="V")
    parser.add_argument("--temp", type=float, default=None, metavar="C")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siliconstat",
        description="SiliconStat -- Monte Carlo mismatch analysis and "
                    "statistical verification for analog ICs")
    parser.add_argument("--version", action="version",
                        version=f"siliconstat {__version__}")
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"SQLite database path (default: {DEFAULT_DB})")
    parser.add_argument("--project", default="default")
    sub = parser.add_subparsers(dest="command", required=True)

    circuit = sub.add_parser("circuit", help="circuit utilities")
    circuit_sub = circuit.add_subparsers(dest="subcommand", required=True)
    validate = circuit_sub.add_parser("validate", help="parse and check a netlist")
    validate.add_argument("netlist")
    validate.add_argument("--no-solve", action="store_true",
                          help="skip the DC operating-point check")
    validate.set_defaults(func=cmd_circuit_validate)

    simulate = sub.add_parser("simulate",
                              help="nominal simulation and measurements")
    simulate.add_argument("netlist")
    simulate.add_argument("--json", help="also write results to this JSON file")
    _add_pvt_args(simulate)
    simulate.add_argument("--sigma-vth-global", type=float, default=0.025,
                          help=argparse.SUPPRESS)
    simulate.add_argument("--sigma-beta-global", type=float, default=3.0,
                          help=argparse.SUPPRESS)
    simulate.set_defaults(func=cmd_simulate)

    mc = sub.add_parser("monte-carlo", help="run a Monte Carlo experiment")
    mc.add_argument("--circuit", required=True)
    mc.add_argument("--samples", type=int, default=1000)
    mc.add_argument("--seed", type=int, default=12345)
    mc.add_argument("--workers", type=int, default=1)
    mc.add_argument("--sampling", choices=("standard", "latin_hypercube"),
                    default="standard")
    mc.add_argument("--label", default="")
    mc.add_argument("--out", help="directory for CSV/JSON/Parquet/HTML exports")
    mc.add_argument("--quiet", action="store_true")
    _add_variation_args(mc)
    _add_pvt_args(mc)
    mc.set_defaults(func=cmd_monte_carlo)

    pvt = sub.add_parser("pvt", help="PVT corner sweep with Monte Carlo mismatch")
    pvt.add_argument("--circuit", required=True)
    pvt.add_argument("--samples", type=int, default=200)
    pvt.add_argument("--seed", type=int, default=12345)
    pvt.add_argument("--workers", type=int, default=1)
    pvt.add_argument("--metric", default=None,
                     help="measurement to tabulate (default: the first one)")
    pvt.add_argument("--corners", nargs="*", choices=sorted(CORNERS))
    pvt.add_argument("--supplies", nargs="*", type=float)
    pvt.add_argument("--temps", nargs="*", type=float)
    pvt.add_argument("--quiet", action="store_true")
    _add_variation_args(pvt)
    pvt.set_defaults(func=cmd_pvt)

    report = sub.add_parser("report", help="render the HTML report for a run")
    report.add_argument("--run", required=True)
    report.add_argument("--out")
    report.add_argument("--export", help="also write raw-data exports here")
    report.set_defaults(func=cmd_report)

    runs = sub.add_parser("runs", help="stored run management")
    runs_sub = runs.add_subparsers(dest="subcommand", required=True)
    rlist = runs_sub.add_parser("list")
    rlist.add_argument("--limit", type=int, default=30)
    rlist.set_defaults(func=cmd_runs_list)
    rshow = runs_sub.add_parser("show")
    rshow.add_argument("run")
    rshow.set_defaults(func=lambda a: cmd_runs_show(
        argparse.Namespace(**{**vars(a), "run": a.run})))
    rcompare = runs_sub.add_parser("compare")
    rcompare.add_argument("runs", nargs="+")
    rcompare.set_defaults(func=cmd_runs_compare)

    ml = sub.add_parser("ml", help="train and score the ML surrogate model")
    ml.add_argument("--circuit", required=True)
    ml.add_argument("--metric", default=None)
    ml.add_argument("--train", type=int, default=400)
    ml.add_argument("--test", type=int, default=200)
    ml.add_argument("--seed", type=int, default=4242)
    ml.add_argument("--model", choices=("auto", "linear", "quadratic",
                                        "gradient_boosting", "random_forest"),
                    default="auto")
    ml.add_argument("--json")
    _add_variation_args(ml)
    ml.set_defaults(func=cmd_ml)

    bench = sub.add_parser("bench", help="benchmark simulation throughput")
    bench.add_argument("--circuit", required=True)
    bench.add_argument("--samples-list", nargs="*", type=int,
                       default=[100, 1000])
    bench.add_argument("--workers-list", nargs="*", type=int, default=[1])
    bench.add_argument("--seed", type=int, default=12345)
    bench.add_argument("--json")
    _add_variation_args(bench)
    bench.set_defaults(func=cmd_bench)

    serve = sub.add_parser("serve", help="start the REST API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    dbcmd = sub.add_parser("db", help="database utilities")
    db_sub = dbcmd.add_subparsers(dest="subcommand", required=True)
    dbstats = db_sub.add_parser("stats")
    dbstats.set_defaults(func=cmd_db_stats)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except SiliconStatError as exc:
        print(colour(f"\nerror: {exc}", C.RED), file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print(colour("\ninterrupted", C.YELLOW), file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(colour(f"\nerror: {exc}", C.RED), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
