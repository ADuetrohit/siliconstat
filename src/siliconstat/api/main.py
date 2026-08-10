"""FastAPI application exposing SiliconStat over REST.

Security posture (see ``docs/architecture.md``):

* netlists arriving over HTTP are parsed with ``.include`` **disabled**, so an
  uploaded file can never read from the server's filesystem;
* expressions in netlists are evaluated by a whitelisting AST walker, never
  by :func:`eval`;
* no endpoint spawns a shell or a subprocess;
* request sizes, sample counts and PVT grid sizes are bounded at the schema
  layer, so a single request cannot queue an unbounded amount of compute;
* example netlists are addressed by basename only, blocking path traversal.
"""

from __future__ import annotations

import io
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from .. import __version__
from ..analysis import analyse_run
from ..core.exceptions import SiliconStatError
from ..core.solver import SolverOptions
from ..db import DatabaseError
from ..mc import MonteCarloConfig
from ..measure import evaluate_measurements
from ..variation import VariationSampler
from ..variation.pvt import CORNERS, find_supply_source
from .jobs import Job, JobStatus
from .schemas import (
    CompareIn,
    HealthOut,
    JobOut,
    JobStatusOut,
    MonteCarloIn,
    NetlistIn,
    PVTSweepIn,
    SimulateIn,
    SurrogateIn,
    WaveformIn,
    VariationIn,
)
from .service import (
    build_pvt,
    build_variation,
    environment_info,
    execute_run,
    get_jobs,
    get_store,
    list_examples,
    resolve_circuit,
)

app = FastAPI(
    title="SiliconStat API",
    version=__version__,
    description="Monte Carlo Mismatch Analysis and Statistical Verification "
                "Platform for Analog ICs",
    # The dashboard's client router owns /docs (its Documentation page), so the
    # interactive API docs live under /api like everything else server-side.
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "SILICONSTAT_CORS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
    ).split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SiliconStatError)
async def _domain_error(_request: Any, exc: SiliconStatError) -> JSONResponse:
    return JSONResponse(status_code=400,
                        content={"detail": str(exc),
                                 "error_type": type(exc).__name__})


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    store = get_store()
    return HealthOut(status="ok", version=__version__, database=store.path,
                     runs_stored=store.run_count(),
                     active_jobs=get_jobs().active_count(),
                     python=environment_info()["python"])


@app.get("/api/version", tags=["meta"])
def version() -> dict[str, Any]:
    return {"version": __version__, "environment": environment_info(),
            "corners": list(CORNERS)}


@app.get("/api/db/stats", tags=["meta"])
def db_stats() -> dict[str, Any]:
    return get_store().stats()


# ---------------------------------------------------------------------------
# circuits
# ---------------------------------------------------------------------------

@app.get("/api/examples", tags=["circuits"])
def examples() -> list[dict[str, Any]]:
    return list_examples()


@app.post("/api/circuits/validate", tags=["circuits"])
def validate_circuit(body: NetlistIn) -> dict[str, Any]:
    """Parse a netlist and return its elaborated description, or the error."""
    from ..core.netlist import parse_netlist
    from ..core.solver import solve_dc

    try:
        circuit = parse_netlist(body.netlist, source=body.name or "uploaded",
                                allow_include=False)
    except SiliconStatError as exc:
        return {"valid": False, "error": str(exc),
                "error_type": type(exc).__name__}
    payload: dict[str, Any] = {"valid": True, "circuit": circuit.describe()}
    try:
        op = solve_dc(circuit)
        payload["operating_point"] = {
            "converged": True, "iterations": op.iterations,
            "strategy": op.strategy, "residual": op.residual,
            "node_voltages": op.node_voltages,
            "devices": op.device_ops,
            "total_supply_power": op.total_supply_power(),
        }
    except SiliconStatError as exc:
        payload["operating_point"] = {"converged": False, "error": str(exc)}
    try:
        variation = build_variation(circuit, VariationIn())
        payload["default_variation"] = {
            "model": variation.to_dict(),
            "slots": VariationSampler(circuit, variation).slot_sigma_table(),
        }
    except SiliconStatError as exc:
        payload["default_variation"] = {"error": str(exc)}
    return payload


@app.get("/api/circuits", tags=["circuits"])
def list_circuits(project: str | None = None) -> list[dict[str, Any]]:
    return get_store().list_circuits(project)


@app.post("/api/circuits", tags=["circuits"])
def store_circuit(body: NetlistIn, project: str = "default") -> dict[str, Any]:
    from ..core.netlist import parse_netlist

    circuit = parse_netlist(body.netlist, source=body.name or "uploaded",
                            allow_include=False)
    if body.name:
        circuit.name = body.name
    circuit_id = get_store().save_circuit(circuit, project=project,
                                          netlist_text=body.netlist)
    return {"circuit_id": circuit_id, "name": circuit.name,
            "summary": circuit.describe()}


@app.get("/api/circuits/{circuit_id}", tags=["circuits"])
def get_circuit(circuit_id: int) -> dict[str, Any]:
    try:
        return get_store().get_circuit(circuit_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

@app.post("/api/simulate", tags=["simulation"])
def simulate(body: SimulateIn) -> dict[str, Any]:
    """Nominal operating point plus all declared measurements."""
    from ..core.solver import solve_dc

    circuit, _text = resolve_circuit(body)
    pvt = build_pvt(body.pvt)
    ctx = pvt.build_context(circuit) if pvt else circuit.build_context()
    opts = SolverOptions.from_options(circuit.options)
    op = solve_dc(circuit, ctx, opts)
    measured = evaluate_measurements(circuit, ctx, opts, op=op)
    specs = []
    for spec in circuit.specs:
        value = measured.values.get(spec.measure, float("nan"))
        outcome = measured.outcomes.get(spec.measure)
        specs.append({
            "description": spec.describe(), "measure": spec.measure,
            "value": value,
            "passed": bool(outcome and outcome.ok and spec.passes(value)),
        })
    return {
        "circuit": circuit.describe(),
        "pvt": pvt.to_dict() if pvt else None,
        "operating_point": op.to_dict(),
        "measurements": measured.to_dict(),
        "specs": specs,
    }


@app.post("/api/waveforms", tags=["simulation"])
def waveforms(body: WaveformIn) -> dict[str, Any]:
    """Raw simulation traces: the AC response and the transient waveform.

    Everything else the API returns is a *derived* number -- a measurement, a
    statistic, a yield.  This is the underlying signal: what the nodes actually
    do versus frequency and versus time, which is what an engineer looks at
    first when a measurement is surprising.

    The sweep windows default to the circuit's own ``.ac`` / ``.tran`` cards.
    Long traces are decimated to ``max_points`` for transport; the decimation
    factor is reported so the client can say so.
    """
    import math

    import numpy as np

    from ..core.solver import ac_analysis, solve_dc, transient_analysis
    from ..measure.engine import _log_sweep

    circuit, _text = resolve_circuit(body)
    pvt = build_pvt(body.pvt)
    ctx = pvt.build_context(circuit) if pvt else circuit.build_context()
    opts = SolverOptions.from_options(circuit.options)
    op = solve_dc(circuit, ctx, opts)

    names = [n for n in circuit.node_names if n != "0"]
    if body.nodes:
        unknown = [n for n in body.nodes if n.lower() not in
                   {x.lower() for x in circuit.node_names}]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown node(s): {', '.join(unknown)}; this circuit has "
                       f"{', '.join(names)}")
        names = [n.lower() for n in body.nodes]

    def decimate(count: int) -> int:
        return max(1, math.ceil(count / body.max_points))

    payload: dict[str, Any] = {
        "circuit": circuit.describe(),
        "pvt": pvt.to_dict() if pvt else None,
        "operating_point": op.to_dict(),
        "ac": None,
        "tran": None,
        "notes": [],
    }

    # ---- AC -------------------------------------------------------------
    ac_spec = body.ac.model_dump() if body.ac else None
    if ac_spec is None:
        for analysis in circuit.analyses:
            if analysis.kind == "ac":
                ac_spec = dict(analysis.args)
                break
    stimulus = next((d for d in circuit.voltage_sources()
                     if getattr(d, "ac_mag", 0.0)), None)
    if ac_spec and stimulus is not None:
        freqs = _log_sweep(ac_spec)
        result = ac_analysis(circuit, freqs, op=op, opts=opts)
        ref_node = circuit.node_names[stimulus.nodes[0]]
        reference = result.v(ref_node)
        step = decimate(len(freqs))
        traces: dict[str, Any] = {}
        for name in names:
            with np.errstate(divide="ignore", invalid="ignore"):
                transfer = np.where(np.abs(reference) > 0,
                                    result.v(name) / reference, np.nan)
            magnitude = 20.0 * np.log10(np.maximum(np.abs(transfer), 1e-300))
            phase = np.degrees(np.unwrap(np.angle(transfer)))
            traces[name] = {
                "mag_db": [None if not math.isfinite(v) else float(v)
                           for v in magnitude[::step]],
                "phase_deg": [None if not math.isfinite(v) else float(v)
                              for v in phase[::step]],
            }
        payload["ac"] = {
            "freqs": [float(f) for f in freqs[::step]],
            "reference": ref_node,
            "stimulus": stimulus.name,
            "nodes": traces,
            "points": int(len(freqs[::step])),
            "decimation": step,
        }
    elif ac_spec:
        payload["notes"].append(
            "AC analysis needs a source with an AC magnitude; add 'AC 1' to the "
            "stimulus source.")

    # ---- transient -------------------------------------------------------
    tran_spec = body.tran.model_dump() if body.tran else None
    if tran_spec is None:
        for analysis in circuit.analyses:
            if analysis.kind == "tran":
                tran_spec = dict(analysis.args)
                break
    if tran_spec:
        result = transient_analysis(
            circuit, tstop=tran_spec["tstop"], tstep=tran_spec["tstep"],
            tstart=tran_spec.get("tstart", 0.0), op=op, opts=opts)
        step = decimate(len(result.time))
        payload["tran"] = {
            "time": [float(t) for t in result.time[::step]],
            "nodes": {name: [float(v) for v in result.v(name)[::step]]
                      for name in names},
            "points": int(len(result.time[::step])),
            "decimation": step,
            "integration": opts.integration,
        }
        if step > 1:
            payload["notes"].append(
                f"the transient was decimated by {step}x for transport "
                f"({len(result.time)} timesteps simulated).")

    if payload["ac"] is None and payload["tran"] is None:
        payload["notes"].append(
            "This circuit declares neither a '.ac' nor a '.tran' card, so it has "
            "no waveform to show -- only the DC operating point above.")
    return payload


@app.post("/api/monte-carlo", response_model=JobOut, tags=["monte-carlo"])
def start_monte_carlo(body: MonteCarloIn) -> JobOut:
    """Start a Monte Carlo run in the background and return a job handle."""
    circuit, text = resolve_circuit(body)
    variation = build_variation(circuit, body.variation)
    config = MonteCarloConfig(
        variation=variation, samples=body.samples, seed=body.seed,
        workers=body.workers, sampling=body.sampling,
        pvt=build_pvt(body.pvt, body.variation),
        solver=SolverOptions.from_options(circuit.options), label=body.label)

    def target(job: Job) -> dict[str, Any]:
        return execute_run(circuit, text, config, job=job, project=body.project,
                           store=body.store, label=body.label)

    job = get_jobs().submit("monte-carlo", target, total=body.samples,
                            message=f"{circuit.name}: {body.samples} samples")
    return JobOut(job_id=job.job_id, status=job.status,
                  message=f"queued {body.samples} samples")


@app.post("/api/monte-carlo/sync", tags=["monte-carlo"])
def run_monte_carlo_sync(body: MonteCarloIn) -> dict[str, Any]:
    """Blocking variant, capped at 2000 samples -- convenient for scripts."""
    if body.samples > 2000:
        raise HTTPException(
            status_code=400,
            detail="the synchronous endpoint is capped at 2000 samples; use "
                   "POST /api/monte-carlo for larger runs")
    circuit, text = resolve_circuit(body)
    variation = build_variation(circuit, body.variation)
    config = MonteCarloConfig(
        variation=variation, samples=body.samples, seed=body.seed,
        workers=body.workers, sampling=body.sampling,
        pvt=build_pvt(body.pvt, body.variation),
        solver=SolverOptions.from_options(circuit.options), label=body.label)
    return execute_run(circuit, text, config, project=body.project,
                       store=body.store, label=body.label)


@app.post("/api/pvt", response_model=JobOut, tags=["monte-carlo"])
def start_pvt_sweep(body: PVTSweepIn) -> JobOut:
    """PVT grid, each condition running a full Monte Carlo mismatch sweep."""
    from ..variation import PVTCondition

    circuit, text = resolve_circuit(body)
    variation = build_variation(circuit, body.variation)
    nominal_supply = find_supply_source(circuit).dc
    supplies = body.supplies or [
        nominal_supply * (1.0 - body.supply_tolerance),
        nominal_supply,
        nominal_supply * (1.0 + body.supply_tolerance),
    ]
    conditions = [
        PVTCondition(corner=c, supply=float(v), temp_c=float(t),
                     sigma_vth_global=body.variation.sigma_vth_global,
                     sigma_beta_global_pct=body.variation.sigma_beta_global_pct)
        for c in body.corners for v in supplies for t in body.temperatures
    ]

    def target(job: Job) -> dict[str, Any]:
        store = get_store()
        rows: list[dict[str, Any]] = []
        job.total = len(conditions) * body.samples
        done = 0
        for condition in conditions:
            if job.cancelled:
                break
            config = MonteCarloConfig(
                variation=variation, samples=body.samples, seed=body.seed,
                workers=1, pvt=condition,
                solver=SolverOptions.from_options(circuit.options),
                label=f"PVT {condition.label}")
            from ..mc import run_monte_carlo as _run
            run = _run(circuit, config)
            analysis = analyse_run(run, charts=False)
            store.save_run(run, circuit=circuit, project=body.project,
                           label=f"PVT {condition.label}")
            store.save_analysis(run.run_id, analysis.to_dict())
            done += body.samples
            job.completed = done
            job.successful += run.counters.successful
            job.failed += run.counters.failed
            metric = body.metric or (run.measurement_names[0]
                                     if run.measurement_names else "")
            stats = analysis.statistics.get(metric)
            yr = analysis.yield_report
            rows.append({
                "corner": condition.corner, "supply": condition.supply,
                "temp_c": condition.temp_c, "label": condition.label,
                "run_id": run.run_id,
                "successful": run.counters.successful,
                "failed": run.counters.failed,
                "metric": metric,
                "mean": stats.mean if stats else None,
                "std": stats.std if stats else None,
                "yield_pct": (yr.combined_yield_over_successful
                              if yr.per_spec else None),
                "yield_ci": [yr.combined_ci95_low, yr.combined_ci95_high]
                if yr.per_spec else None,
            })
        worst = min((r for r in rows if r["yield_pct"] is not None),
                    key=lambda r: r["yield_pct"], default=None)
        return {"grid": rows, "worst_case": worst,
                "conditions": len(conditions), "samples_each": body.samples,
                "metric": body.metric}

    job = get_jobs().submit("pvt", target, total=len(conditions) * body.samples,
                            message=f"{len(conditions)} PVT conditions")
    return JobOut(job_id=job.job_id, status=job.status,
                  message=f"queued {len(conditions)} PVT conditions")


@app.post("/api/ml/surrogate", response_model=JobOut, tags=["ml"])
def start_surrogate(body: SurrogateIn) -> JobOut:
    """Train an ML surrogate on real simulations and score it on held-out ones."""
    from ..ml import surrogate_experiment

    circuit, _text = resolve_circuit(body)
    variation = build_variation(circuit, body.variation)

    def target(job: Job) -> dict[str, Any]:
        job.total = body.train_samples + body.test_samples
        return surrogate_experiment(
            circuit, variation, target=body.metric,
            train_samples=body.train_samples, test_samples=body.test_samples,
            seed=body.seed, model_kind=body.model,
            solver=SolverOptions.from_options(circuit.options))

    job = get_jobs().submit("surrogate", target,
                            total=body.train_samples + body.test_samples,
                            message=f"{circuit.name}: surrogate")
    return JobOut(job_id=job.job_id, status=job.status,
                  message="queued surrogate experiment")


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

@app.get("/api/jobs", tags=["jobs"])
def list_jobs(limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
    return [j.to_dict() for j in get_jobs().list(limit)]


@app.get("/api/jobs/{job_id}", response_model=JobStatusOut, tags=["jobs"])
def job_status(job_id: str, include_result: bool = True) -> JobStatusOut:
    job = get_jobs().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    data = job.to_dict()
    if not include_result:
        data["result"] = None
    return JobStatusOut(**data)


@app.post("/api/jobs/{job_id}/cancel", tags=["jobs"])
def cancel_job(job_id: str) -> dict[str, Any]:
    ok = get_jobs().cancel(job_id)
    if not ok:
        raise HTTPException(status_code=409,
                            detail="job is not running or does not exist")
    return {"job_id": job_id, "status": "cancellation requested"}


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

@app.get("/api/runs", tags=["runs"])
def list_runs(project: str | None = None, circuit: str | None = None,
              limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    return [r.to_dict() for r in get_store().list_runs(
        project=project, circuit_name=circuit, limit=limit)]


@app.get("/api/runs/{run_id}", tags=["runs"])
def get_run(run_id: str) -> dict[str, Any]:
    store = get_store()
    try:
        run = store.load_run(run_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"summary": run.summary(), "config": run.config,
            "counters": run.counters.to_dict(),
            "nominal": run.nominal, "nominal_status": run.nominal_status,
            "measurement_meta": run.measurement_meta,
            "slot_meta": run.slot_meta, "spec_meta": run.spec_meta,
            "reproduction": run.reproduction_record(),
            "failure_breakdown": run.failure_breakdown(),
            "notes": run.notes}


@app.get("/api/runs/{run_id}/analysis", tags=["runs"])
def get_analysis(run_id: str, refresh: bool = False) -> dict[str, Any]:
    store = get_store()
    if not refresh:
        cached = store.load_analysis(run_id)
        if cached is not None:
            return cached
    try:
        run = store.load_run(run_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    analysis = analyse_run(run).to_dict()
    store.save_analysis(run_id, analysis)
    return analysis


@app.get("/api/runs/{run_id}/samples", tags=["runs"])
def get_samples(run_id: str, offset: int = Query(0, ge=0),
                limit: int = Query(200, ge=1, le=5000),
                status: str | None = None) -> dict[str, Any]:
    try:
        run = get_store().load_run(run_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    samples = run.samples
    if status:
        samples = [s for s in samples if s.status == status]
    window = samples[offset:offset + limit]
    return {"total": len(samples), "offset": offset, "limit": limit,
            "samples": [s.to_dict() for s in window]}


@app.get("/api/runs/{run_id}/export.csv", tags=["runs"])
def export_run_csv(run_id: str) -> Response:
    try:
        run = get_store().load_run(run_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    buffer = io.StringIO()
    run.to_dataframe().to_csv(buffer, index=False)
    return Response(
        content=buffer.getvalue(), media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="{run_id}_samples.csv"'})


@app.get("/api/runs/{run_id}/report.html", response_class=HTMLResponse,
         tags=["runs"])
def export_run_report(run_id: str) -> HTMLResponse:
    from ..report import render_html_report

    store = get_store()
    try:
        run = store.load_run(run_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    analysis = analyse_run(run)
    netlist = store.load_netlist(run_id) or ""
    return HTMLResponse(render_html_report(analysis, run, netlist=netlist))


@app.get("/api/runs/{run_id}/netlist", response_class=PlainTextResponse,
         tags=["runs"])
def get_run_netlist(run_id: str) -> PlainTextResponse:
    netlist = get_store().load_netlist(run_id)
    if netlist is None:
        raise HTTPException(status_code=404,
                            detail="no netlist stored for this run")
    return PlainTextResponse(netlist)


@app.post("/api/runs/{run_id}/reproduce", response_model=JobOut, tags=["runs"])
def reproduce_run(run_id: str) -> JobOut:
    """Re-execute a stored run from its recorded configuration.

    Same netlist, same seed, same variation model -- the new run must produce
    identical per-sample results, which is exactly what makes it a check.
    """
    from ..core.netlist import parse_netlist
    from ..variation.pvt import PVTCondition
    from ..variation.spec import VariationModel

    store = get_store()
    try:
        original = store.load_run(run_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    netlist = store.load_netlist(run_id)
    if not netlist:
        raise HTTPException(
            status_code=409,
            detail="this run has no stored netlist, so it cannot be reproduced")
    circuit = parse_netlist(netlist, source=original.circuit_name,
                            allow_include=False)
    cfg = original.config
    variation = VariationModel.from_dict(cfg["variation"])
    pvt = PVTCondition.from_dict(cfg["pvt"]) if cfg.get("pvt") else None
    config = MonteCarloConfig(
        variation=variation, samples=int(cfg["samples"]), seed=int(cfg["seed"]),
        workers=1, sampling=cfg.get("sampling", "standard"), pvt=pvt,
        solver=SolverOptions(**{k: v for k, v in (cfg.get("solver") or {}).items()}),
        label=f"reproduction of {run_id}")

    def target(job: Job) -> dict[str, Any]:
        result = execute_run(circuit, netlist, config, job=job,
                             label=f"reproduction of {run_id}")
        new_run = store.load_run(result["run_id"])
        identical = _compare_runs(original, new_run)
        result["reproduction_check"] = identical
        return result

    job = get_jobs().submit("reproduce", target, total=int(cfg["samples"]),
                            message=f"reproducing {run_id}")
    return JobOut(job_id=job.job_id, status=job.status,
                  message=f"reproducing run {run_id}")


def _compare_runs(a: Any, b: Any) -> dict[str, Any]:
    """Exact per-sample comparison used by the reproduce endpoint."""
    names = sorted(set(a.measurement_names) & set(b.measurement_names))
    mismatches: list[dict[str, Any]] = []
    compared = 0
    for sa, sb in zip(a.samples, b.samples):
        if sa.status != sb.status:
            mismatches.append({"sample": sa.index, "field": "status",
                               "a": sa.status, "b": sb.status})
            continue
        for name in names:
            va = sa.measurements.get(name)
            vb = sb.measurements.get(name)
            compared += 1
            if va is None or vb is None:
                continue
            if va != vb and not (va != va and vb != vb):
                mismatches.append({"sample": sa.index, "field": name,
                                   "a": va, "b": vb})
    return {
        "identical": not mismatches and len(a.samples) == len(b.samples),
        "samples_compared": min(len(a.samples), len(b.samples)),
        "values_compared": compared,
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
        "original_run": a.run_id, "new_run": b.run_id,
    }


@app.post("/api/runs/compare", tags=["runs"])
def compare_runs(body: CompareIn) -> dict[str, Any]:
    """Side-by-side statistics for several runs (e.g. 100 vs 1000 samples)."""
    store = get_store()
    payload: list[dict[str, Any]] = []
    for run_id in body.run_ids:
        try:
            run = store.load_run(run_id)
        except DatabaseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        analysis = analyse_run(run, charts=True)
        payload.append({
            "run_id": run_id,
            "summary": run.summary(),
            "config": run.config,
            "counters": run.counters.to_dict(),
            "statistics": {k: v.to_dict() for k, v in analysis.statistics.items()},
            "yield": analysis.yield_report.to_dict(),
            "charts": analysis.charts,
            "nominal": analysis.nominal,
        })
    shared = sorted(set.intersection(
        *[set(p["statistics"]) for p in payload]) if payload else set())
    return {"runs": payload, "shared_measurements": shared}


@app.delete("/api/runs/{run_id}", tags=["runs"])
def delete_run(run_id: str) -> dict[str, Any]:
    if not get_store().delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return {"deleted": run_id}


# ---------------------------------------------------------------------------
# static frontend (served when a production build exists)
# ---------------------------------------------------------------------------

def _mount_frontend() -> None:
    root = os.environ.get("SILICONSTAT_FRONTEND") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), "frontend", "dist")
    if not os.path.isdir(root):
        return
    from fastapi.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException as StarletteHTTPException

    class _SPA(StaticFiles):
        """Serve the built dashboard, falling back to index.html for deep links.

        The client router owns paths like ``/circuits`` and ``/results``, which
        do not exist on disk.  Note that StaticFiles *raises* on a miss rather
        than returning a 404 response, so the fallback has to catch -- an
        earlier version checked ``response.status_code`` and never fired, which
        made every deep link 404 while ``/`` worked.
        """

        async def get_response(self, path: str, scope: Any) -> Response:
            try:
                return await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code == 404 and not path.startswith("api"):
                    return await super().get_response("index.html", scope)
                raise

    app.mount("/", _SPA(directory=root, html=True), name="frontend")


_mount_frontend()


@app.get("/", include_in_schema=False)
def _root() -> dict[str, Any]:  # pragma: no cover - replaced when UI is built
    return {"name": "SiliconStat API", "version": __version__,
            "docs": "/api/docs",
            "note": "Build the React frontend (npm run build in ./frontend) to "
                    "serve the dashboard from this origin."}
