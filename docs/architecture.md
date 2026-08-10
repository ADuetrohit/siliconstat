# Architecture

SiliconStat is organised as a strict layer stack.  Each layer depends only on
the layers beneath it, and each is independently testable.

```
                                                       ┌──────────────────────┐
  9  ML acceleration        siliconstat/ml             │ surrogate.py         │
                                                       └──────────┬───────────┘
                                                                  │
  8  visualization          siliconstat/report         ┌──────────▼───────────┐
                            siliconstat/api            │ charts, html_report, │
                            frontend/                  │ export, REST, React  │
                                                       └──────────┬───────────┘
                                                                  │
  7  yield                  siliconstat/analysis       ┌──────────▼───────────┐
                                                       │ yield_analysis.py    │
                                                       └──────────┬───────────┘
                                                                  │
  6  statistics             siliconstat/analysis       ┌──────────▼───────────┐
                                                       │ statistics, corr.,   │
                                                       │ sensitivity, conv.   │
                                                       └──────────┬───────────┘
                                                                  │
  5  Monte Carlo            siliconstat/mc             ┌──────────▼───────────┐
                                                       │ engine, config,      │
                                                       │ results              │
                                                       └──────────┬───────────┘
                                                                  │
  4  mismatch model         siliconstat/variation      ┌──────────▼───────────┐
                                                       │ spec, sampler,       │
                                                       │ distributions,       │
                                                       │ correlation, pelgrom │
                                                       │ pvt                  │
                                                       └──────────┬───────────┘
                                                                  │
  3  measurements           siliconstat/measure        ┌──────────▼───────────┐
                                                       │ engine.py            │
                                                       └──────────┬───────────┘
                                                                  │
  2  simulator              siliconstat/core           ┌──────────▼───────────┐
                                                       │ mna, solver,         │
                                                       │ circuit, netlist     │
                                                       └──────────┬───────────┘
                                                                  │
  1  circuit physics        siliconstat/core           ┌──────────▼───────────┐
                                                       │ mosfet, models,      │
                                                       │ devices, waveforms   │
                                                       └──────────────────────┘
```

The dependency rule is enforced by construction: `core` imports nothing from
`variation`, `variation` imports nothing from `mc`, `mc` imports nothing from
`analysis`, and so on.  The only upward reference anywhere is the `mc` layer's
use of `measure`, which sits below it.

## Why this order

The stack is not arbitrary.  Each layer exists because the one above it needs
a guarantee that only the one below can provide:

| Layer | Provides upward | Depends downward on |
| --- | --- | --- |
| circuit physics | `Ids(v)` and its exact derivatives | nothing |
| simulator | a converged operating point, or an explicit failure | analytic Jacobians |
| measurements | a number an engineer can specify on | a trustworthy operating point |
| mismatch model | a reproducible perturbed device set | device parameters it may modify |
| Monte Carlo | N recorded outcomes with provenance | perturbation + measurement |
| statistics | estimates with stated uncertainty | a complete sample record |
| yield | a pass fraction with a stated denominator | statistics + specifications |
| visualization | the same numbers, legibly | everything below |
| ML acceleration | a cheap approximation of the whole stack | real simulation data to learn from |

Notice that ML sits at the *top*.  It is an optimisation over a pipeline that
already works, trained on that pipeline's own output — not a substitute for
any layer beneath it.  See [ml surrogate](#ml-acceleration) below.

## Data flow for one Monte Carlo sample

```
  seed, index
      │
      ▼
  SeedSequence(entropy=seed, spawn_key=(index,))        variation/sampler.py
      │
      ▼
  standard normals ──► correlation (Cholesky) ──► marginal transform
      │
      ▼
  slot deviations        e.g. {"NCH.vth_global": -0.019, "M1.vth_local": +0.0007}
      │
      ▼
  device overrides       {"M1": {"dvth": -0.0183, "beta_scale": 1.004}, …}
      │
      ▼
  Circuit.with_overrides(...)                           core/circuit.py
      │
      ▼
  SimContext  ◄── corner shift ──► temperature scaling  variation/pvt.py
      │
      ▼
  solve_dc  ──► Newton-Raphson + continuation           core/solver.py
      │                 │
      │                 └── ConvergenceError ──► sample recorded as failed
      ▼
  evaluate_measurements ──► op / AC sweep / transient   measure/engine.py
      │                 │
      │                 └── MeasurementError ──► NaN + reason, sample flagged
      ▼
  spec verdicts          {"ierr <= 2 %": True, …}
      │
      ▼
  SampleResult           index, seed, status, reason, values, draws, verdicts
```

Every arrow that can fail produces a *recorded* failure, never a dropped
sample.  A run of 1000 samples always yields 1000 `SampleResult` records.

## Process and thread model

There are two independent pools, and they solve different problems.

**Monte Carlo workers** (`mc/engine.py`) are OS processes.  Sample evaluation
is Python-bound — device stamping dominates, not BLAS — so threads would
serialise on the GIL.  Work is split into `4 × workers` chunks so that a slow
chunk cannot leave a worker idle at the end.  BLAS is pinned to one thread per
worker while the pool exists, because each solve is a tiny dense system and
per-worker thread pools only add memory pressure.

If the pool cannot start — on Windows, `N` independent NumPy imports readily
exhaust the page file — `BrokenProcessPool` is caught, the run restarts
sequentially, and the fallback is recorded in `run.notes`.  Parallelism is an
optimisation; correctness never depends on it.

**API jobs** (`api/jobs.py`) are threads.  A job thread only supervises: it
calls into the Monte Carlo engine (which may itself fan out to processes) and
updates a progress record that `GET /api/jobs/{id}` reads.  Threads are right
here because the work is elsewhere and the thread spends its life blocked.

Because sample *i* is seeded from `(master_seed, i)` alone, none of this
affects results.  Sequential and parallel runs are bit-identical, which is
asserted in `tests/test_reproducibility.py`.

## Persistence

`db/schema.py` defines eight tables.  The design rule is that **per-sample
records are stored, not just aggregates** — a run you cannot re-open sample by
sample is a run you cannot debug.

| Table | Holds | Why it exists |
| --- | --- | --- |
| `projects` | name, description | grouping |
| `circuits` | netlist text, SHA-256, elaborated summary | lets a run be reproduced from storage alone |
| `runs` | config, counters, metas, nominal, provenance | one row per experiment |
| `samples` | index, seed, status, failure reason, verdict, timing | the audit trail |
| `sample_measurements` | one row per (sample, measurement) | raw data, NaN stored as NULL |
| `sample_variations` | one row per (sample, random variable) | what was actually drawn |
| `sample_devices` | one row per (sample, derived device parameter) | the resulting `dvth`, `beta_scale`, W, L |
| `sample_specs` | one row per (sample, specification) | per-sample pass/fail |
| `run_statistics`, `run_yield`, `run_analysis` | cached analysis JSON | derived, disposable |

The last three are *caches*.  Deleting them loses nothing: `analyse_run()`
recomputes everything from the sample rows.  The first eight are the source of
truth.

SQLite is the default because it needs no configuration and handles millions
of sample rows comfortably.  The DDL is deliberately plain — surrogate integer
keys, `TEXT` for JSON, explicit foreign keys, no SQLite-only types — and is
generated through `schema_statements(dialect)`, so a PostgreSQL backend needs
only a connection factory and the `SERIAL` keyword.

## Security

Netlists are untrusted input: they arrive over HTTP from a browser.  The
threat model is "a hostile netlist should not be able to read, write or
execute anything on the server."

1. **No `eval`.**  `.param` expressions and `EXPR` measurements are evaluated
   by `core/expr.py`, which parses to an AST and walks it with a whitelist.
   Attribute access, subscripting, lambdas, comprehensions, string literals,
   comparisons and calls to anything outside a fixed function list are all
   rejected *structurally* — there is no code path that could execute them.
   `tests/test_expr.py` asserts this against thirty specific attacks.
2. **`.include` is disabled for uploaded text.**  `parse_netlist()` defaults to
   `allow_include=False`; only `parse_netlist_file()` (the CLI path) enables
   it, and then only relative to the netlist's own directory.  A posted
   `.include /etc/passwd` is refused with an explicit message.
3. **Example netlists are addressed by basename.**  `_example_path()` rejects
   any name that is not `os.path.basename` of itself, which blocks traversal.
4. **Bounded work per request.**  `api/schemas.py` caps netlist size (512 kB),
   sample count, worker count, PVT grid size (200 conditions) and total
   simulations (2 M).  The synchronous Monte Carlo endpoint is capped at 2000
   samples; larger runs must go through the job queue.
5. **No subprocess anywhere.**  The package never shells out.  The optional
   ngspice cross-check in `docs/scripts/` is a developer tool that is not
   importable from the API.

There is deliberately **no authentication**.  SiliconStat is a single-user
engineering tool; exposing it on a shared network would need an auth layer in
front of it.  This is stated in [limitations.md](limitations.md).

## Extension points

**A new device.**  Subclass `core/devices.py:Device`, implement `stamp_dc`,
`stamp_ac`, `current`, `power` and `op_info`; add a letter to the parser's
handler table in `core/netlist.py:element`.  Declare `VARIABLE_PARAMS` as a
`ClassVar` if the variation layer should be able to perturb it, and add the
parameter to `variation/spec.py:PARAMETER_MODES` and `_PARAM_DEVICE_TYPES`.

**A new measurement kind.**  Write a function `(MeasurementContext,
MeasureSpec) -> float` in `measure/engine.py`, register it in `EVALUATORS`,
and add the name to `_MEASURE_UNITS` and the required-argument table in
`core/netlist.py`.  Raise `MeasurementError` with a *reason* when the quantity
is undefined — never return a plausible substitute.

**A new distribution.**  Subclass `variation/distributions.py:Distribution`
with `std` and `from_standard_normal`; register it in `DISTRIBUTIONS` and
`make_distribution`.  Implementing `from_standard_normal` is what lets the
correlation machinery work with it unchanged (Gaussian copula).

**A new corner.**  Add an entry to `variation/pvt.py:CORNERS` mapping the name
to `(nmos_direction, pmos_direction)`.

## Testing

738 tests, 36 s for the full suite and 25 s excluding the ones marked `slow`.
The suite is organised by layer, and `tests/test_analytical.py` is the one
that matters most: it checks the simulator against closed-form answers rather
than against its own previous output.  See [validation.md](validation.md).
