"""Shared service helpers used by the API routes.

Keeps the route handlers thin: circuit resolution, variation-model
construction and run execution all live here so the CLI-facing and HTTP-facing
paths behave identically.
"""

from __future__ import annotations

import os
import platform
import threading
from typing import Any

from ..analysis import analyse_run
from ..core.circuit import Circuit
from ..core.exceptions import SiliconStatError
from ..core.netlist import parse_netlist, parse_netlist_file
from ..core.solver import SolverOptions
from ..db import RunStore
from ..mc import MonteCarloConfig, run_monte_carlo
from ..variation import PVTCondition, default_mismatch_model
from ..variation.spec import VariationModel
from .jobs import Job, JobManager

__all__ = ["get_store", "get_jobs", "resolve_circuit", "build_variation",
           "build_pvt", "execute_run", "list_examples", "EXAMPLES_DIR"]

_STORE_LOCK = threading.Lock()
_STORE: RunStore | None = None
_JOBS: JobManager | None = None

EXAMPLES_DIR = os.environ.get(
    "SILICONSTAT_EXAMPLES",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "examples"))


def get_store() -> RunStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = RunStore(os.environ.get("SILICONSTAT_DB",
                                             "siliconstat.sqlite"))
        return _STORE


def get_jobs() -> JobManager:
    global _JOBS
    with _STORE_LOCK:
        if _JOBS is None:
            _JOBS = JobManager(max_workers=int(
                os.environ.get("SILICONSTAT_JOB_WORKERS", "2")))
        return _JOBS


def list_examples() -> list[dict[str, Any]]:
    """Built-in demo circuits shipped with the project."""
    out: list[dict[str, Any]] = []
    if not os.path.isdir(EXAMPLES_DIR):
        return out
    for entry in sorted(os.listdir(EXAMPLES_DIR)):
        if not entry.endswith(".net"):
            continue
        path = os.path.join(EXAMPLES_DIR, entry)
        try:
            circuit = parse_netlist_file(path)
        except SiliconStatError as exc:
            out.append({"id": entry, "name": entry, "error": str(exc)})
            continue
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        out.append({
            "id": entry, "name": circuit.name, "file": entry,
            "devices": len(circuit.devices), "nodes": circuit.n_nodes,
            "measurements": [m.name for m in circuit.measures],
            "specs": [s.describe() for s in circuit.specs],
            "matched_groups": circuit.describe()["matched_groups"],
            "netlist": text,
        })
    return out


def _example_path(name: str) -> str:
    safe = os.path.basename(name)
    if safe != name or not safe.endswith(".net"):
        raise SiliconStatError(f"invalid example name {name!r}")
    path = os.path.join(EXAMPLES_DIR, safe)
    if not os.path.isfile(path):
        raise SiliconStatError(f"unknown example {name!r}")
    return path


def resolve_circuit(ref: Any) -> tuple[Circuit, str]:
    """Turn a :class:`CircuitRef`-shaped object into ``(circuit, netlist_text)``.

    Inline netlists are parsed with ``.include`` disabled: text arriving over
    HTTP must not be able to read files from the server's filesystem.
    """
    if getattr(ref, "example", None):
        path = _example_path(ref.example)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        return parse_netlist_file(path), text
    if getattr(ref, "circuit_id", None) is not None:
        record = get_store().get_circuit(int(ref.circuit_id))
        text = record["netlist"]
        return parse_netlist(text, source=record["name"], allow_include=False), text
    text = ref.netlist
    circuit = parse_netlist(text, source="uploaded", allow_include=False)
    name = getattr(ref, "name", None)
    if name:
        circuit.name = name
    return circuit, text


def build_variation(circuit: Circuit, spec: Any) -> VariationModel:
    """Construct a :class:`VariationModel` from the API's variation block."""
    modes = {"nominal": (False, False), "process": (True, False),
             "mismatch": (False, True), "both": (True, True)}
    process, mismatch = modes[spec.mode]
    model = default_mismatch_model(
        circuit,
        sigma_vth_global=spec.sigma_vth_global,
        sigma_beta_global_pct=spec.sigma_beta_global_pct,
        enable_process=process, enable_mismatch=mismatch,
        include_passives=spec.include_passives,
    )
    # Rebuild each rule through the dataclass constructor so the edited fields
    # go through ParameterVariation.__post_init__ validation rather than being
    # mutated past it.
    from ..variation.spec import ParameterVariation

    rebuilt: list[ParameterVariation] = []
    for variation in model.variations:
        data = variation.to_dict()
        data.pop("unit", None)
        data["distribution"] = spec.distribution
        if spec.truncate_sigma:
            data["truncate_sigma"] = spec.truncate_sigma
        if not spec.pelgrom and data["pelgrom"]:
            data["pelgrom"] = False
            if data["parameter"] == "vth":
                data["sigma"] = spec.sigma_vth_local
                data["sigma_pct"] = None
            else:
                data["sigma"] = None
                data["sigma_pct"] = spec.sigma_beta_local_pct
        if data["distribution"] == "lognormal" and data["mode"] == "additive":
            # A log-normal deviation cannot be added to a threshold; keep that
            # rule Gaussian and say so in its label.
            data["distribution"] = "gaussian"
            data["label"] = (data.get("label", "") +
                             " (kept gaussian: additive parameter)").strip()
        rebuilt.append(ParameterVariation.from_dict(data))
    model.variations = rebuilt
    return model


def build_pvt(spec: Any, variation_spec: Any = None) -> PVTCondition | None:
    if spec is None:
        return None
    kwargs: dict[str, Any] = {}
    if variation_spec is not None:
        kwargs = {"sigma_vth_global": variation_spec.sigma_vth_global,
                  "sigma_beta_global_pct": variation_spec.sigma_beta_global_pct}
    return PVTCondition(corner=spec.corner, supply=spec.supply,
                        temp_c=spec.temp_c, **kwargs)


def execute_run(circuit: Circuit, netlist: str, config: MonteCarloConfig, *,
                job: Job | None = None, project: str = "default",
                store: bool = True, label: str = "") -> dict[str, Any]:
    """Run Monte Carlo, analyse, optionally persist -- shared by API and jobs."""

    def progress(state: dict[str, Any]) -> None:
        if job is None:
            return
        job.completed = state["completed"]
        job.total = state["total"]
        job.successful = state["successful"]
        job.failed = state["failed"]
        job.elapsed_s = state["elapsed_s"]
        job.run_id = state["run_id"]

    run = run_monte_carlo(
        circuit, config, progress=progress,
        should_cancel=(lambda: job.cancelled) if job else None)
    analysis = analyse_run(run)
    if job is not None:
        job.run_id = run.run_id
    if store:
        db = get_store()
        db.save_run(run, circuit=circuit, project=project, label=label)
        db.save_analysis(run.run_id, analysis.to_dict())
    return {"run_id": run.run_id, "summary": run.summary(),
            "analysis": analysis.to_dict()}


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
    }
