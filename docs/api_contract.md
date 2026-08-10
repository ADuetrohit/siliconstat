# REST API contract

Base URL `http://127.0.0.1:8000`. Interactive schema at `/docs` (OpenAPI).
All bodies are JSON. Domain errors return HTTP 400 with
`{"detail": "...", "error_type": "NetlistSyntaxError" | ...}`.

## Circuit references

Several endpoints accept a **CircuitRef**: exactly one of

```jsonc
{ "example": "current_mirror.net" }   // built-in demo circuit
{ "circuit_id": 3 }                   // previously stored circuit
{ "netlist": "VDD vdd 0 1.8\n..." }   // inline text (max 512 kB)
```

Inline netlists are parsed with `.include` **disabled**.

## Variation block

```jsonc
{
  "mode": "both",                 // "nominal" | "process" | "mismatch" | "both"
  "sigma_vth_global": 0.025,      // volts
  "sigma_beta_global_pct": 3.0,   // percent
  "include_passives": true,
  "pelgrom": true,                // area-scaled local mismatch
  "sigma_vth_local": 0.003,       // used only when pelgrom = false
  "sigma_beta_local_pct": 1.0,    // used only when pelgrom = false
  "distribution": "gaussian",     // "gaussian" | "uniform" | "lognormal"
  "truncate_sigma": null          // e.g. 3 clips draws at +/-3 sigma
}
```

## PVT block

```jsonc
{ "corner": "TT", "supply": 1.8, "temp_c": 27.0 }
// corner in TT | FF | SS | FS | SF ; supply in (0,20] V ; temp in (-273,400] C
```

## Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/health` | – | `{status, version, database, runs_stored, active_jobs, python}` |
| GET | `/api/version` | – | `{version, environment, corners}` |
| GET | `/api/db/stats` | – | row counts and file size |
| GET | `/api/examples` | – | `[{id, name, file, devices, nodes, measurements[], specs[], matched_groups[], netlist}]` |
| POST | `/api/circuits/validate` | `{netlist, name?}` | `{valid, circuit?, operating_point?, default_variation?, error?}` |
| GET | `/api/circuits` | – | stored circuits |
| POST | `/api/circuits` | `{netlist, name?}` | `{circuit_id, name, summary}` |
| GET | `/api/circuits/{id}` | – | stored circuit record |
| POST | `/api/simulate` | CircuitRef + `{pvt?}` | `{circuit, pvt, operating_point, measurements, specs[]}` |
| POST | `/api/waveforms` | CircuitRef + `{pvt?, ac?, tran?, nodes?, max_points?}` | `{circuit, pvt, operating_point, ac?, tran?, notes[]}` |
| POST | `/api/monte-carlo` | CircuitRef + `{samples, seed, workers, sampling, variation, pvt?, label, project, store}` | `{job_id, status, message}` |
| POST | `/api/monte-carlo/sync` | same (≤ 2000 samples) | `{run_id, summary, analysis}` |
| POST | `/api/pvt` | CircuitRef + `{samples, seed, corners[], supplies[]?, supply_tolerance, temperatures[], variation, metric?}` | `{job_id, ...}` |
| POST | `/api/ml/surrogate` | CircuitRef + `{metric?, train_samples, test_samples, seed, model, variation}` | `{job_id, ...}` |
| GET | `/api/jobs` | – | recent jobs |
| GET | `/api/jobs/{job_id}` | `?include_result=` | `{job_id, kind, status, run_id, completed, total, successful, failed, elapsed_s, eta_s, error, result}` |
| POST | `/api/jobs/{job_id}/cancel` | – | `{job_id, status}` |
| GET | `/api/runs` | `?project=&circuit=&limit=` | run listing |
| GET | `/api/runs/{run_id}` | – | summary, config, counters, nominal, metas, reproduction, failure_breakdown |
| GET | `/api/runs/{run_id}/analysis` | `?refresh=` | full RunAnalysis (see below) |
| GET | `/api/runs/{run_id}/samples` | `?offset=&limit=&status=` | `{total, offset, limit, samples[]}` |
| GET | `/api/runs/{run_id}/export.csv` | – | CSV download |
| GET | `/api/runs/{run_id}/report.html` | – | standalone HTML report |
| GET | `/api/runs/{run_id}/netlist` | – | text/plain |
| POST | `/api/runs/{run_id}/reproduce` | – | `{job_id}`; result carries `reproduction_check.identical` |
| POST | `/api/runs/compare` | `{run_ids: [..]}` | `{runs[], shared_measurements[]}` |
| DELETE | `/api/runs/{run_id}` | – | `{deleted}` |

`job.status` ∈ `queued | running | done | failed | cancelled`. Poll
`/api/jobs/{id}` roughly once a second while `running`.

Interactive OpenAPI documentation is served at `/api/docs` (ReDoc at
`/api/redoc`, schema at `/api/openapi.json`). It is *not* at the usual `/docs`
because the dashboard's client-side router owns that path for its own
Documentation page.

Any non-`/api` path that does not match a file in `frontend/dist` falls back to
`index.html` so client-side routes survive a hard refresh. Unmatched `/api`
paths still return 404 rather than HTML.

## Waveforms

`/api/waveforms` returns the raw traces behind a circuit — what a waveform
viewer plots — as opposed to `/api/simulate`, which returns the scalar
measurements extracted from them.

The analysis windows default to the `.ac` and `.tran` cards in the netlist; the
`ac` and `tran` request blocks override them:

| Field | Meaning | Default |
| --- | --- | --- |
| `ac` | `{fstart, fstop, points, sweep}`, `sweep` ∈ `dec \| oct \| lin` | the netlist's `.ac` card |
| `tran` | `{tstep, tstop, tstart?}` | the netlist's `.tran` card |
| `nodes` | restrict the returned traces | every node |
| `max_points` | decimate longer traces to at most this many samples | 2000 |

```jsonc
{
  "ac": {
    "points": 109, "decimation": 1,
    "reference": "inp", "stimulus": "VINP",   // AC magnitudes are referred to this source
    "freqs": [1.0, ...],
    "nodes": {"out": {"mag_db": [86.94, ...], "phase_deg": [0.0, ...]}}
  },
  "tran": {
    "points": 402, "decimation": 1, "method": "trapezoidal",
    "time": [0.0, ...],
    "nodes": {"out": [1.8285, ...]}           // volts
  }
}
```

Both blocks are `null` when the netlist declares no corresponding card and the
request supplies no override — a circuit with only a DC operating point has no
waveform, and the endpoint says so in `notes` instead of returning an invented
trace. `notes` also records decimation and any analysis that failed to
converge. Frequencies at which the stimulus itself has zero amplitude — so the
transfer function is undefined — yield `null` in `mag_db`/`phase_deg` rather
than `NaN`, which is not valid JSON. A node that is merely silent floors at
−6000 dB instead of `-Infinity`.

## RunAnalysis payload

```jsonc
{
  "run_id": "…", "circuit_name": "…",
  "summary": { "counters": {total, successful, failed, convergence_failures,
                            numerical_errors, invalid_measurements, success_rate},
               "duration_s": 1.9, "samples_per_second": 262.3, "seed": 12345,
               "measurements": ["iref", …], "variation_mode": "process+mismatch",
               "pvt": null, "software_version": "0.1.0" },
  "statistics": { "<measurement>": { count, mean, median, std, variance, minimum,
                    maximum, range, cv, p1, p5, p25, p75, p95, p99, iqr,
                    sigma3_low, sigma3_high, skewness, kurtosis_excess, sem,
                    mean_ci95_low, mean_ci95_high, normality_p, normality_test,
                    unit } },
  "yield": { attempted, successful, failed, combined_passing,
             combined_yield_over_successful, combined_ci95_low, combined_ci95_high,
             combined_yield_over_attempted, independent_product_pct,
             limiting_spec, notes[],
             per_spec: [{ key, measure, op, limit, unit, description, passing,
                          failing, denominator, yield_pct, ci95_low, ci95_high,
                          margin_mean, margin_sigma, cpk, worst_value }] },
  "correlation": { parameters[], measurements[], pearson[[]], spearman[[]],
                   n_samples, degenerate_parameters[], notes[] },
  "measurement_correlation": { measurements[], pearson[[]], spearman[[]] },
  "sensitivity": { "<measurement>": { method, n_samples, r_squared,
      unexplained_pct, output_sigma, max_input_correlation, notes[],
      entries: [{ rank, parameter, scope, pearson, spearman, beta_standardised,
                  variance_contribution_pct, share_of_explained_pct, sigma,
                  unit, d_output_d_sigma }] } },
  "convergence": { "<measurement>": { measurement, unit, n[], mean[],
      mean_ci_low[], mean_ci_high[], std[], yield_pct[], yield_ci_low[],
      yield_ci_high[], final_mean, final_std, final_yield_pct,
      mean_settled_at, std_settled_at, yield_settled_at, notes[] } },
  "charts": { "<measurement>": {
      "histogram": { counts[], edges[], centres[], bin_width, n },
      "cdf": { x[], p[], n },
      "sigma_plot": { x[], sigma[], n },
      "unit": "A", "nominal": 1e-5,
      "specs": [{ measure, op, value, unit, description }] } },
  "failure_breakdown": [{ status, reason, count, percent }],
  "nominal": { "<measurement>": value },
  "spec_meta": [...], "measurement_meta": [{name, kind, unit, args, description}],
  "slot_meta": [{ slot, parameter, scope, distribution, sigma, unit, devices, label }],
  "reproduction": { run_id, seed, samples, sampling, variation, pvt, solver,
                    software_version, platform, circuit_sha256, timestamp },
  "notes": []
}
```

## Circuit description payload

`circuit.describe()` (returned by `/api/simulate` and `/api/circuits/validate`):

```jsonc
{ "name": "NMOS Current Mirror",
  "nodes": ["0","vdd","nref","out"], "n_nodes": 4, "n_branches": 1,
  "devices": [{ name, type, nodes[], r?, c?, l?, dc?, w?, m?, model?,
                matched_group? }],
  "mos_models": { "NCH": {name, type, vto, kp, lambda, gamma, phi, avt, abeta, …} },
  "diode_models": {...},
  "measures": [{name, kind, args, unit, description}],
  "specs": [{measure, op, value, unit, label, description}],
  "analyses": [{kind, args}],
  "options": {...}, "temp_c": 27.0,
  "matched_groups": ["MIRROR"] }
```

`operating_point`:

```jsonc
{ "node_voltages": {"vdd": 1.8, …},
  "devices": { "M1": { type: "mosfet", model, mtype, w, l, m, matched_group,
                       id, vgs, vds, vbs, vth, vov, region, gm, gds, gmb, ro,
                       gm_over_id, beta, cgs, cgd, vth0_used, kp_used,
                       dvth, beta_scale } },
  "iterations": 7, "strategy": "direct", "residual": 8.5e-14,
  "total_supply_power": 3.6e-5 }
```
