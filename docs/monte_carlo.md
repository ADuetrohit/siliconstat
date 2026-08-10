# The Monte Carlo engine

Layer 5 of the [architecture](architecture.md).

## Pipeline

```
seed, index → random draw → device overrides → perturbed circuit
            → DC solve → measurements → spec verdicts → SampleResult
```

Every stage can fail, and every failure is recorded with its cause.  A run that
attempts N samples produces N records — see [failure taxonomy](#failures)
below.

The engine perturbs *device parameters* and re-solves the circuit.  It never
perturbs an output.  If a mismatch produces a 2.4 % copy error, that number
came out of a nonlinear solve of a circuit whose transistors had different
thresholds, not from adding noise to a nominal answer.

## Reproducibility

Sample `i` draws from

```python
numpy.random.SeedSequence(entropy=master_seed, spawn_key=(i,))
```

which is a pure function of the master seed and the sample index.  Three
properties follow, and all three are asserted in
`tests/test_reproducibility.py`:

1. **The same seed reproduces a run exactly** — every measurement, every draw,
   every verdict.
2. **Worker count and completion order cannot change anything.** A run with
   `workers=4` is bit-identical to `workers=1`.
3. **A single sample can be re-simulated in isolation.**  Pull sample 37 out of
   a 1000-sample run, re-run just that index, and get the same numbers — which
   is how you debug a convergence failure.

A fourth, occasionally useful, property: a 1000-sample run's first 100 samples
are identical to a 100-sample run with the same seed, because sample `i` does
not depend on the sample count.

The `reproduction_record()` carries everything needed to re-run: seed, sample
count, sampling method, the full variation model, the PVT condition, solver
tolerances, software version, platform string and the circuit's SHA-256.  The
REST API exposes `POST /api/runs/{id}/reproduce`, which re-executes and then
compares the two runs sample by sample, reporting `identical: true/false` and
the number of values compared.

## Sampling methods

**`standard`** — independent samples, seeded per index as above.  Streaming,
memory-free, parallel-safe.

**`latin_hypercube`** — each dimension is stratified into `N` equal-probability
bins with exactly one sample per bin, and the bins are permuted independently
across dimensions.  For a fixed budget this reduces the variance of estimated
means and quantiles.

LHS necessarily *couples* samples: the design has to be built as a whole.  So
that mode constructs the full standard-normal matrix up front from one seeded
generator instead of seeding per sample.  It is reproducible, but it loses the
"re-simulate sample 37 alone" property, which is why it is opt-in.

## Failures

Nothing is discarded.  `SampleStatus` distinguishes:

| status | cause |
| --- | --- |
| `ok` | converged, all measurements valid |
| `convergence_failure` | Newton failed under every continuation strategy |
| `numerical_error` | singular or non-finite matrix |
| `invalid_measurement` | solved, but a measurement could not be computed |
| `variation_error` | the draw was non-physical (negative W, β ≤ 0) |
| `circuit_error` | the perturbed circuit was structurally invalid |
| `unexpected_error` | anything else, with its message |

Each failed sample keeps its index, its seed and a human-readable reason, so
you can reproduce exactly the failing case.  The counters always satisfy

```
total == successful + failed
failed == convergence + numerical + invalid_measurement + other
```

and `failure_breakdown()` groups reasons by frequency for the report and the
UI.

The distinction between "the circuit did not solve" and "the circuit solved but
the measurement is undefined" matters.  An op-amp whose gain never crosses
unity has a perfectly good operating point; what it does not have is a
unity-gain frequency.  That sample is `invalid_measurement`, its `ugf` is NaN,
and its reason says *"no unity-gain crossing within 1 Hz .. 10 GHz (gain range
−40.2 .. 12.6 dB)"*.

## Parallel execution

Work is chunked into `4 × workers` pieces and distributed over a
`ProcessPoolExecutor`.  Processes, not threads: sample evaluation is
Python-bound, so threads would serialise on the GIL.

**Measured on this machine** (`examples/current_mirror.net`, 2000 samples):

| workers | wall time | throughput |
| --- | --- | --- |
| 1 | 10.97 s | 182 samples/s |
| 2 | 10.30 s | 194 samples/s |
| 4 | 11.02 s | 182 samples/s |

Parallelism is worth roughly 6 % here and nothing at 4 workers.  Two reasons:
each Windows worker pays a full NumPy/SciPy import at spawn (about a second),
and a single sample of this circuit takes under 5 ms, so the per-chunk IPC
overhead is a significant fraction of the work.  Parallelism starts paying for
circuits where a sample costs tens of milliseconds — the two-stage op-amp at
19 samples/s, or any transient analysis.

Attempting 8 workers on this machine exhausted the Windows page file and killed
the pool.  The engine catches `BrokenProcessPool`, restarts sequentially, and
records the fallback in `run.notes`:

> *parallel execution with 8 workers failed (BrokenProcessPool: ...); the run
> completed sequentially. Results are unaffected — per-sample seeding makes
> sequential and parallel execution identical.*

Correctness never depends on parallelism succeeding.

## Throughput

Measured with `scripts/run_benchmarks.py` (Python 3.12, Windows):

| circuit | samples | wall time | throughput | per sample |
| --- | ---: | ---: | ---: | ---: |
| resistive divider | 1 000 | 1.07 s | 937 /s | 1.0 ms |
| current mirror | 100 | 0.55 s | 182 /s | 5.4 ms |
| current mirror | 1 000 | 3.99 s | 251 /s | 3.9 ms |
| current mirror | 10 000 | 34.68 s | 288 /s | 3.4 ms |
| differential pair | 1 000 | 7.17 s | 140 /s | 7.2 ms |
| two-stage OTA | 200 | 10.32 s | 19 /s | 51.6 ms |
| inverter (transient) | 25 | 4.26 s | 5.9 /s | 170 ms |

Throughput *rises* with sample count because the fixed cost — parsing,
building the variation model, the nominal simulation — is amortised.  The
op-amp is slow because every sample runs a 109-point AC sweep plus bisection
refinement on top of the DC solve; the inverter is slow because every sample
integrates 400 transient timesteps.

## Using it

```bash
siliconstat monte-carlo \
    --circuit examples/current_mirror.net \
    --samples 1000 --seed 12345 --mode mismatch
```

`--mode` selects which variation sources are enabled: `nominal`, `process`,
`mismatch`, or `both` (default).  Running the same circuit three ways and
comparing is the standard way to attribute spread — `siliconstat runs compare`
puts the results side by side.

Programmatically:

```python
from siliconstat.core import parse_netlist_file
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.analysis import analyse_run
from siliconstat.variation import default_mismatch_model

circuit = parse_netlist_file("examples/current_mirror.net")
config = MonteCarloConfig(
    variation=default_mismatch_model(circuit),
    samples=1000, seed=12345, workers=1)

run = run_monte_carlo(circuit, config)
analysis = analyse_run(run)

print(run.counters.to_dict())
print(analysis.yield_report.summary_lines())
```

A progress callback receives `{completed, total, successful, failed,
elapsed_s}`; `should_cancel` is polled between samples so a UI can stop a long
run.

## Choosing a sample count

Monte Carlo error falls as `1/√N`, so precision is expensive.  For a yield
estimate to `±m` percentage points at 95 % confidence:

| observed yield | ±1 % | ±0.5 % | ±0.1 % |
| ---: | ---: | ---: | ---: |
| 50 % | 9 604 | 38 415 | 960 365 |
| 90 % | 3 458 | 13 830 | 345 732 |
| 99 % | 381 | 1 522 | 38 031 |
| 99.9 % | 39 | 154 | 3 838 |

`siliconstat.analysis.required_samples_for_margin()` computes these.  The
convergence trace in every report shows how the running mean, sigma and yield
settle, so you can see whether the number you are about to quote has actually
stabilised — see [statistics.md](statistics.md) and [yield.md](yield.md).
