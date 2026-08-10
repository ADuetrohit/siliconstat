"""The Monte Carlo engine.

Pipeline per sample::

    seed -> random draw -> device overrides -> perturbed circuit
         -> DC solve -> measurements -> spec pass/fail -> record

Every step can fail, and every failure is *recorded with its cause* rather
than dropped.  A run that attempts 1000 samples always reports 1000 records;
the yield analysis downstream then states explicitly which denominator it used.

Reproducibility
---------------
Sample *i* draws from ``SeedSequence(entropy=seed, spawn_key=(i,))``.  This
is a pure function of the master seed and the sample index, so:

* the same seed reproduces the same run exactly;
* worker count and scheduling order have no effect on the results;
* a single sample can be re-simulated in isolation for debugging.

Latin hypercube sampling necessarily couples samples, so that mode builds the
whole standard-normal matrix up front from one seeded generator instead.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import numpy as np

from ..core.circuit import Circuit, SpecLimit
from ..core.exceptions import (
    CircuitError,
    ConvergenceError,
    MeasurementError,
    NumericalError,
    SimulationError,
    VariationError,
)
from ..core.solver import OperatingPoint, solve_dc
from ..measure.engine import evaluate_measurements
from ..variation.distributions import inverse_normal_cdf
from ..variation.sampler import VariationSampler
from .config import MonteCarloConfig
from .results import MonteCarloRun, RunCounters, SampleResult, SampleStatus

__all__ = ["run_monte_carlo", "simulate_sample", "simulate_nominal",
           "latin_hypercube_normals", "ProgressCallback"]

ProgressCallback = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_seed_sequence(master_seed: int, index: int) -> np.random.SeedSequence:
    """Deterministic per-sample seed sequence (parallel-safe)."""
    return np.random.SeedSequence(entropy=int(master_seed), spawn_key=(int(index),))


def sample_seed(master_seed: int, index: int) -> int:
    """A reportable 64-bit seed for sample *index*."""
    state = sample_seed_sequence(master_seed, index).generate_state(2, dtype=np.uint32)
    return int((int(state[0]) << 32) | int(state[1]))


def latin_hypercube_normals(n_samples: int, n_dims: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube design mapped to standard normals.

    Each dimension is stratified into ``n_samples`` equal-probability bins with
    exactly one sample per bin, then the bins are randomly permuted across
    dimensions.  For a fixed sample budget this reduces the variance of
    estimated means and quantiles compared with independent sampling -- at the
    cost of samples no longer being mutually independent, which is why the
    engine keeps it opt-in.
    """
    if n_dims == 0:
        return np.zeros((n_samples, 0))
    grid = np.empty((n_samples, n_dims))
    edges = np.arange(n_samples)
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        grid[:, j] = (edges[perm] + rng.random(n_samples)) / n_samples
    return inverse_normal_cdf(grid)


# ---------------------------------------------------------------------------
# Single-sample simulation
# ---------------------------------------------------------------------------

def _spec_key(spec: SpecLimit, used: dict[str, int]) -> str:
    key = spec.describe()
    if key in used:
        used[key] += 1
        return f"{key} #{used[key]}"
    used[key] = 1
    return key


def _evaluate_specs(specs: Sequence[SpecLimit], values: dict[str, float],
                    ok_flags: dict[str, bool]) -> tuple[dict[str, bool], bool | None]:
    if not specs:
        return {}, None
    used: dict[str, int] = {}
    out: dict[str, bool] = {}
    for spec in specs:
        key = _spec_key(spec, used)
        value = values.get(spec.measure, float("nan"))
        valid = ok_flags.get(spec.measure, math.isfinite(value))
        out[key] = bool(valid and spec.passes(value))
    return out, all(out.values())


def simulate_sample(circuit: Circuit, sampler: VariationSampler,
                    config: MonteCarloConfig, index: int,
                    z_row: np.ndarray | None = None,
                    specs: Sequence[SpecLimit] | None = None) -> SampleResult:
    """Draw, perturb, simulate and measure one Monte Carlo sample."""
    started = time.perf_counter()
    seed = sample_seed(config.seed, index)
    result = SampleResult(index=index, seed=seed)

    try:
        if z_row is None:
            rng = np.random.default_rng(sample_seed_sequence(config.seed, index))
            z = sampler.draw_standard(rng)
        else:
            z = np.asarray(z_row, dtype=float)
            for i, slot in enumerate(sampler.slots):
                if slot.truncate_sigma:
                    z[i] = float(np.clip(z[i], -slot.truncate_sigma,
                                         slot.truncate_sigma))
        deviations = sampler.deviations_from_standard(z)
        overrides, device_values = sampler.apply(deviations)
        result.slot_values = {s.name: float(deviations[i])
                              for i, s in enumerate(sampler.slots)}
        result.device_values = device_values
    except VariationError as exc:
        result.status = SampleStatus.VARIATION_ERROR
        result.failure_reason = str(exc)
        result.runtime_s = time.perf_counter() - started
        return result

    try:
        perturbed = circuit.with_overrides(overrides)
    except (CircuitError, ValueError) as exc:
        result.status = SampleStatus.CIRCUIT_ERROR
        result.failure_reason = str(exc)
        result.runtime_s = time.perf_counter() - started
        return result

    ctx = (config.pvt.build_context(perturbed) if config.pvt
           else perturbed.build_context())

    op: OperatingPoint | None = None
    try:
        op = solve_dc(perturbed, ctx, config.solver)
        result.dc_iterations = op.iterations
        result.dc_strategy = op.strategy
    except ConvergenceError as exc:
        result.status = SampleStatus.CONVERGENCE_FAILURE
        result.failure_reason = str(exc)
        result.runtime_s = time.perf_counter() - started
        return result
    except NumericalError as exc:
        result.status = SampleStatus.NUMERICAL_ERROR
        result.failure_reason = str(exc)
        result.runtime_s = time.perf_counter() - started
        return result
    except SimulationError as exc:
        result.status = SampleStatus.UNEXPECTED_ERROR
        result.failure_reason = str(exc)
        result.runtime_s = time.perf_counter() - started
        return result

    try:
        measured = evaluate_measurements(perturbed, ctx, config.solver, op=op)
    except (MeasurementError, SimulationError) as exc:
        result.status = SampleStatus.INVALID_MEASUREMENT
        result.failure_reason = str(exc)
        result.runtime_s = time.perf_counter() - started
        return result

    result.measurements = dict(measured.values)
    result.measurement_ok = {n: o.ok for n, o in measured.outcomes.items()}
    result.measurement_reasons = {n: o.reason for n, o in measured.outcomes.items()
                                  if not o.ok}
    if not measured.all_valid:
        result.status = SampleStatus.INVALID_MEASUREMENT
        first = next(iter(measured.invalid_reasons().items()))
        result.failure_reason = f"{first[0]}: {first[1]}"

    spec_list = list(specs if specs is not None else [])
    result.spec_pass, result.passed = _evaluate_specs(
        spec_list, result.measurements, result.measurement_ok)
    if result.status != SampleStatus.OK:
        result.passed = None
    result.runtime_s = time.perf_counter() - started
    return result


def simulate_nominal(circuit: Circuit, config: MonteCarloConfig
                     ) -> tuple[dict[str, float], str]:
    """Simulate the unperturbed circuit at the run's PVT condition."""
    ctx = (config.pvt.build_context(circuit) if config.pvt
           else circuit.build_context())
    try:
        op = solve_dc(circuit, ctx, config.solver)
        measured = evaluate_measurements(circuit, ctx, config.solver, op=op)
        status = "ok" if measured.all_valid else "invalid_measurement"
        if not measured.all_valid:
            first = next(iter(measured.invalid_reasons().items()))
            status = f"invalid_measurement: {first[0]}: {first[1]}"
        return dict(measured.values), status
    except (SimulationError, MeasurementError) as exc:
        return {}, f"failed: {exc}"


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

_WORKER_STATE: dict[str, Any] = {}


def _worker_init(circuit: Circuit, config: MonteCarloConfig,
                 specs: list[SpecLimit]) -> None:
    circuit.finalize()
    _WORKER_STATE["circuit"] = circuit
    _WORKER_STATE["config"] = config
    _WORKER_STATE["specs"] = specs
    _WORKER_STATE["sampler"] = VariationSampler(circuit, config.variation)


def _worker_chunk(payload: tuple[list[int], np.ndarray | None]) -> list[SampleResult]:
    indices, z_rows = payload
    circuit = _WORKER_STATE["circuit"]
    config = _WORKER_STATE["config"]
    sampler = _WORKER_STATE["sampler"]
    specs = _WORKER_STATE["specs"]
    out = []
    for k, index in enumerate(indices):
        row = None if z_rows is None else z_rows[k]
        out.append(simulate_sample(circuit, sampler, config, index, row, specs))
    return out


def _chunk(indices: list[int], n_chunks: int) -> list[list[int]]:
    if n_chunks <= 1:
        return [indices]
    size = max(1, math.ceil(len(indices) / n_chunks))
    return [indices[i:i + size] for i in range(0, len(indices), size)]


@contextlib.contextmanager
def _single_threaded_blas():
    """Pin BLAS/OpenMP to one thread per process while a worker pool exists."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = "1"
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_monte_carlo(circuit: Circuit, config: MonteCarloConfig, *,
                    progress: ProgressCallback | None = None,
                    run_id: str | None = None,
                    should_cancel: Callable[[], bool] | None = None,
                    ) -> MonteCarloRun:
    """Execute a Monte Carlo experiment and return the complete record."""
    if not circuit.finalized:
        circuit.finalize()

    sampler = VariationSampler(circuit, config.variation)
    specs = list(config.specs if config.specs is not None else circuit.specs)

    run = MonteCarloRun(
        run_id=run_id or uuid.uuid4().hex[:16],
        circuit_name=circuit.name,
        config=config.to_dict(),
        measurement_meta=[m.to_dict() for m in circuit.measures],
        slot_meta=sampler.slot_sigma_table(),
        spec_meta=[s.to_dict() for s in specs],
        circuit_source=circuit.source_text or "",
        circuit_sha256=hashlib.sha256(
            (circuit.source_text or repr(circuit.describe())).encode("utf-8")
        ).hexdigest(),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    run.nominal, run.nominal_status = simulate_nominal(circuit, config)

    z_matrix: np.ndarray | None = None
    if config.sampling == "latin_hypercube":
        rng = np.random.default_rng(config.seed)
        z_matrix = latin_hypercube_normals(config.samples, sampler.n_slots, rng)

    indices = list(range(config.samples))
    started = time.perf_counter()
    counters = RunCounters()
    collected: list[SampleResult] = []

    def _emit(force: bool = False) -> None:
        if progress is None:
            return
        done = len(collected)
        if not force and config.progress_every > 0 and done % config.progress_every:
            return
        progress({
            "run_id": run.run_id, "completed": done, "total": config.samples,
            "successful": counters.successful, "failed": counters.failed,
            "elapsed_s": time.perf_counter() - started,
        })

    def _run_sequential(todo: list[int]) -> None:
        for index in todo:
            if should_cancel and should_cancel():
                break
            row = None if z_matrix is None else z_matrix[index]
            res = simulate_sample(circuit, sampler, config, index, row, specs)
            counters.add(res)
            collected.append(res)
            _emit()

    workers = config.effective_workers
    if workers == 1:
        _run_sequential(indices)
    else:
        chunks = _chunk(indices, workers * 4)
        payloads = [(chunk, None if z_matrix is None else z_matrix[chunk])
                    for chunk in chunks]
        try:
            # Each worker solves tiny dense systems; BLAS threading inside a
            # worker only adds memory and contention.  Children inherit this.
            with _single_threaded_blas(), \
                    ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                        initargs=(circuit, config, specs)) as pool:
                futures = {pool.submit(_worker_chunk, p): p for p in payloads}
                for future in as_completed(futures):
                    for res in future.result():
                        counters.add(res)
                        collected.append(res)
                    _emit()
                    if should_cancel and should_cancel():
                        for f in futures:
                            f.cancel()
                        break
        except (BrokenProcessPool, OSError, ImportError) as exc:
            # Worker processes could not be started or died (on Windows this is
            # usually the page-file limit being hit by N independent NumPy
            # imports).  Parallelism is an optimisation, never a correctness
            # requirement: fall back to sequential and say so in the record.
            run.notes = (
                f"parallel execution with {workers} workers failed "
                f"({type(exc).__name__}: {exc}); the run completed sequentially. "
                "Results are unaffected -- per-sample seeding makes sequential "
                "and parallel execution identical."
            )
            counters = RunCounters()
            collected = []
            _run_sequential(indices)

    collected.sort(key=lambda r: r.index)
    run.samples = collected
    run.counters = counters
    run.duration_s = time.perf_counter() - started
    run.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _emit(force=True)
    return run
