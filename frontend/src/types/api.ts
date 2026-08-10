/**
 * Typed payloads for the SiliconStat REST API.
 * The authority is docs/api_contract.md; keep the two in step.
 */

// ---------------------------------------------------------------------------
// meta
// ---------------------------------------------------------------------------

export interface Health {
  status: string
  version: string
  database: string
  runs_stored: number
  active_jobs: number
  python: string
}

export interface VersionInfo {
  version: string
  environment: { python: string; system: string; cpu_count: number }
  corners: string[]
}

export interface DbStats {
  path: string
  size_bytes: number
  projects: number
  circuits: number
  runs: number
  samples: number
  measurements: number
  variations: number
  schema_version: number
}

// ---------------------------------------------------------------------------
// circuits
// ---------------------------------------------------------------------------

export type DeviceType =
  | 'mosfet' | 'resistor' | 'capacitor' | 'inductor'
  | 'voltagesource' | 'currentsource' | 'diode'

export interface CircuitDevice {
  name: string
  type: DeviceType | string
  nodes: string[]
  r?: number
  c?: number
  l?: number
  dc?: number
  ac_mag?: number
  w?: number
  m?: number
  model?: string
  matched_group?: string | null
  area?: number
}

export interface MosModel {
  name: string
  type: 'nmos' | 'pmos'
  vto: number
  kp: number
  lambda: number
  gamma: number
  phi: number
  avt: number
  abeta: number
  tnom: number
  tcv: number
  bex: number
  tox: number
  cgso: number
  cgdo: number
}

export interface MeasureMeta {
  name: string
  kind: string
  args: Record<string, unknown>
  unit: string
  description: string
}

export interface SpecMeta {
  measure: string
  op: string
  value: number
  unit: string
  label: string
  description: string
}

export interface AnalysisSpec {
  kind: string
  args: Record<string, unknown>
}

export interface CircuitDescription {
  name: string
  nodes: string[]
  n_nodes: number
  n_branches: number
  devices: CircuitDevice[]
  mos_models: Record<string, MosModel>
  diode_models: Record<string, Record<string, unknown>>
  measures: MeasureMeta[]
  specs: SpecMeta[]
  analyses: AnalysisSpec[]
  options: Record<string, unknown>
  temp_c: number
  matched_groups: string[]
}

export interface MosfetOp {
  type: string
  name: string
  model: string
  mtype: 'nmos' | 'pmos'
  w: number
  l: number
  m: number
  matched_group: string | null
  id: number
  vgs: number
  vds: number
  vbs: number
  vth: number
  vov: number
  region: 'cutoff' | 'triode' | 'saturation' | string
  gm: number
  gds: number
  gmb: number
  ro: number
  gm_over_id: number
  beta: number
  cgs: number
  cgd: number
  vth0_used: number
  kp_used: number
  dvth: number
  beta_scale: number
}

export type DeviceOp = MosfetOp | Record<string, unknown>

export interface OperatingPoint {
  node_voltages: Record<string, number>
  devices: Record<string, DeviceOp>
  iterations: number
  strategy: string
  residual: number
  total_supply_power: number
}

export interface MeasurementOutcome {
  name: string
  value: number
  unit: string
  kind: string
  ok: boolean
  reason: string
}

export interface MeasurementResult {
  values: Record<string, number>
  outcomes: Record<string, MeasurementOutcome>
  analyses_run: string[]
  dc_iterations: number
  dc_strategy: string
}

export interface Example {
  id: string
  name: string
  file?: string
  devices?: number
  nodes?: number
  measurements?: string[]
  specs?: string[]
  matched_groups?: string[]
  netlist: string
  error?: string
}

export interface StoredCircuit {
  id: number
  name: string
  sha256: string
  created_at: string
  project: string | null
  netlist_bytes: number
}

export interface ValidateResponse {
  valid: boolean
  circuit?: CircuitDescription
  operating_point?: {
    converged: boolean
    iterations?: number
    strategy?: string
    residual?: number
    node_voltages?: Record<string, number>
    devices?: Record<string, DeviceOp>
    total_supply_power?: number
    error?: string
  }
  default_variation?: { model?: VariationModelInfo; slots?: SlotMeta[]; error?: string }
  error?: string
  error_type?: string
}

export interface SimulateResponse {
  circuit: CircuitDescription
  pvt: PVTInfo | null
  operating_point: OperatingPoint
  measurements: MeasurementResult
  specs: { description: string; measure: string; value: number; passed: boolean }[]
}

// ---------------------------------------------------------------------------
// variation, PVT
// ---------------------------------------------------------------------------

export type VariationMode = 'nominal' | 'process' | 'mismatch' | 'both'
export type DistributionName = 'gaussian' | 'uniform' | 'lognormal'

export interface VariationBlock {
  mode: VariationMode
  sigma_vth_global: number
  sigma_beta_global_pct: number
  include_passives: boolean
  pelgrom: boolean
  sigma_vth_local: number
  sigma_beta_local_pct: number
  distribution: DistributionName
  truncate_sigma: number | null
}

export interface VariationModelInfo {
  name: string
  mode: string
  enable_process: boolean
  enable_mismatch: boolean
  pelgrom_pair_convention: boolean
  variations: Record<string, unknown>[]
  correlations: Record<string, unknown>[]
}

export interface PVTBlock {
  corner: string
  supply: number | null
  temp_c: number
}

export interface PVTInfo {
  corner: string
  supply: number | null
  temp_c: number
  label: string
  k_sigma: number
}

export interface SlotMeta {
  slot: string
  parameter: string
  scope: 'global' | 'local' | string
  distribution: string
  sigma: number
  unit: string
  devices: string
  label: string
}

// ---------------------------------------------------------------------------
// requests
// ---------------------------------------------------------------------------

export interface CircuitRef {
  netlist?: string
  circuit_id?: number
  example?: string
}

export interface MonteCarloRequest extends CircuitRef {
  samples: number
  seed: number
  workers?: number
  sampling?: 'standard' | 'latin_hypercube'
  variation?: Partial<VariationBlock>
  pvt?: PVTBlock | null
  label?: string
  project?: string
  store?: boolean
}

export interface PVTSweepRequest extends CircuitRef {
  samples: number
  seed: number
  corners: string[]
  supplies?: number[] | null
  supply_tolerance?: number
  temperatures: number[]
  variation?: Partial<VariationBlock>
  metric?: string | null
}

export interface SurrogateRequest extends CircuitRef {
  metric?: string | null
  train_samples: number
  test_samples: number
  seed: number
  model: 'auto' | 'linear' | 'quadratic' | 'gradient_boosting' | 'random_forest'
  variation?: Partial<VariationBlock>
}

// ---------------------------------------------------------------------------
// jobs
// ---------------------------------------------------------------------------

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'

export interface Job {
  job_id: string
  kind: string
  status: JobStatus
  run_id: string | null
  completed: number
  total: number
  successful: number
  failed: number
  elapsed_s: number
  eta_s: number | null
  message: string
  error: string | null
  result: Record<string, any> | null
  created_at: string
  finished_at: string | null
}

// ---------------------------------------------------------------------------
// runs and analysis
// ---------------------------------------------------------------------------

export interface RunCounters {
  total: number
  successful: number
  failed: number
  convergence_failures: number
  numerical_errors: number
  invalid_measurements: number
  other_errors: number
  success_rate: number
}

export interface RunListing {
  run_id: string
  circuit_name: string
  label: string
  samples: number
  seed: number
  variation_mode: string
  pvt_label: string
  started_at: string
  finished_at: string
  duration_s: number
  counters: RunCounters
  software_version: string
  notes: string
}

export interface RunSummary {
  run_id: string
  circuit: string
  counters: RunCounters
  started_at: string
  finished_at: string
  duration_s: number
  samples_per_second: number
  software_version: string
  platform: string
  seed: number
  measurements: string[]
  variation_mode: string
  pvt: PVTInfo | null
  circuit_sha256: string
}

export interface ReproductionRecord {
  run_id: string
  seed: number
  samples: number
  sampling: string
  variation: VariationModelInfo
  pvt: PVTInfo | null
  solver: Record<string, number>
  software_version: string
  platform: string
  circuit_sha256: string
  timestamp: string
}

export interface FailureRow {
  status: string
  reason: string
  count: number
  percent: number
}

export interface RunDetail {
  summary: RunSummary
  config: Record<string, any>
  counters: RunCounters
  nominal: Record<string, number>
  nominal_status: string
  measurement_meta: MeasureMeta[]
  slot_meta: SlotMeta[]
  spec_meta: SpecMeta[]
  reproduction: ReproductionRecord
  failure_breakdown: FailureRow[]
  notes: string
}

export interface Statistics {
  name: string
  unit: string
  count: number
  n_input: number
  n_invalid: number
  mean: number
  median: number
  variance: number
  std: number
  minimum: number
  maximum: number
  range: number
  cv: number
  p1: number
  p5: number
  p25: number
  p75: number
  p95: number
  p99: number
  iqr: number
  sigma3_low: number
  sigma3_high: number
  skewness: number
  kurtosis_excess: number
  sem: number
  mean_ci95_low: number
  mean_ci95_high: number
  std_ci95_low: number
  std_ci95_high: number
  normality_p: number
  normality_test: string
}

export interface SpecYield {
  key: string
  measure: string
  op: string
  limit: number
  unit: string
  description: string
  passing: number
  failing: number
  denominator: number
  yield_pct: number
  ci95_low: number
  ci95_high: number
  margin_mean: number
  margin_sigma: number
  cpk: number
  worst_value: number
}

export interface YieldReport {
  attempted: number
  successful: number
  failed: number
  denominator_basis: string
  per_spec: SpecYield[]
  combined_passing: number
  combined_yield_over_successful: number
  combined_ci95_low: number
  combined_ci95_high: number
  combined_yield_over_attempted: number
  independent_product_pct: number
  limiting_spec: string
  notes: string[]
}

export interface CorrelationMatrix {
  parameters: string[]
  measurements: string[]
  pearson: number[][]
  spearman: number[][]
  n_samples: number
  degenerate_parameters: string[]
  degenerate_measurements: string[]
  notes: string[]
}

export interface SensitivityEntry {
  parameter: string
  rank: number
  pearson: number
  spearman: number
  beta_standardised: number
  variance_contribution: number
  variance_contribution_pct: number
  share_of_explained_pct: number
  sigma: number
  unit: string
  scope: string
  d_output_d_sigma: number
}

export interface SensitivityReport {
  measurement: string
  method: string
  n_samples: number
  r_squared: number
  unexplained_pct: number
  output_sigma: number
  output_mean: number
  entries: SensitivityEntry[]
  max_input_correlation: number
  notes: string[]
}

export interface ConvergenceTrace {
  measurement: string
  unit: string
  n: number[]
  mean: number[]
  mean_ci_low: number[]
  mean_ci_high: number[]
  std: number[]
  yield_pct: number[]
  yield_ci_low: number[]
  yield_ci_high: number[]
  final_mean: number
  final_std: number
  final_yield_pct: number
  mean_settled_at: number | null
  std_settled_at: number | null
  yield_settled_at: number | null
  settle_tolerance_pct: number
  notes: string[]
}

export interface ChartBundle {
  histogram: { counts: number[]; edges: number[]; centres: number[]; bin_width: number; n: number }
  cdf: { x: number[]; p: number[]; n: number }
  sigma_plot: { x: number[]; sigma: number[]; n: number }
  unit: string
  nominal: number | null
  specs: SpecMeta[]
}

export interface RunAnalysis {
  run_id: string
  circuit_name: string
  summary: RunSummary
  statistics: Record<string, Statistics>
  yield: YieldReport
  correlation: CorrelationMatrix
  measurement_correlation: {
    measurements: string[]
    pearson?: number[][]
    spearman?: number[][]
    n_samples: number
  }
  sensitivity: Record<string, SensitivityReport>
  convergence: Record<string, ConvergenceTrace>
  charts: Record<string, ChartBundle>
  failure_breakdown: FailureRow[]
  nominal: Record<string, number>
  spec_meta: SpecMeta[]
  measurement_meta: MeasureMeta[]
  slot_meta: SlotMeta[]
  reproduction: ReproductionRecord
  notes: string[]
}

export interface SampleRecord {
  index: number
  seed: number
  status: string
  failure_reason: string
  measurements: Record<string, number>
  measurement_ok: Record<string, boolean>
  measurement_reasons: Record<string, string>
  slot_values: Record<string, number>
  device_values: Record<string, number>
  spec_pass: Record<string, boolean>
  passed: boolean | null
  dc_iterations: number
  dc_strategy: string
  runtime_s: number
}

export interface SamplePage {
  total: number
  offset: number
  limit: number
  samples: SampleRecord[]
}

export interface CompareEntry {
  run_id: string
  summary: RunSummary
  config: Record<string, any>
  counters: RunCounters
  statistics: Record<string, Statistics>
  yield: YieldReport
  charts: Record<string, ChartBundle>
  nominal: Record<string, number>
}

export interface CompareResponse {
  runs: CompareEntry[]
  shared_measurements: string[]
}

export interface PVTGridRow {
  corner: string
  supply: number
  temp_c: number
  label: string
  run_id: string
  successful: number
  failed: number
  metric: string
  mean: number | null
  std: number | null
  yield_pct: number | null
  yield_ci: [number, number] | null
}

export interface PVTSweepResult {
  grid: PVTGridRow[]
  worst_case: PVTGridRow | null
  conditions: number
  samples_each: number
  metric: string | null
}

export interface SurrogateResult {
  target: string
  model: string
  n_features: number
  features: string[]
  train_samples: number
  test_samples: number
  r2: number
  rmse: number
  rmse_pct_of_sigma: number
  mae: number
  max_abs_error: number
  cv_r2_mean: number
  cv_r2_std: number
  output_sigma: number
  sim_time_per_sample_ms: number
  surrogate_time_per_sample_ms: number
  raw_speedup: number
  train_time_s: number
  train_sim_time_s: number
  fit_time_s: number
  break_even_samples: number
  net_speedup_10k: number
  sim_yield_pct: number
  surrogate_yield_pct: number
  yield_error_pp: number
  notes: string[]
}

// ---------------------------------------------------------------------------
// raw simulation traces
// ---------------------------------------------------------------------------

export interface ACTrace {
  mag_db: (number | null)[]
  phase_deg: (number | null)[]
}

export interface ACResponse {
  freqs: number[]
  reference: string
  stimulus: string
  nodes: Record<string, ACTrace>
  points: number
  decimation: number
}

export interface TransientResponse {
  time: number[]
  nodes: Record<string, number[]>
  points: number
  decimation: number
  integration: string
}

export interface WaveformResponse {
  circuit: CircuitDescription
  pvt: PVTInfo | null
  operating_point: OperatingPoint
  ac: ACResponse | null
  tran: TransientResponse | null
  notes: string[]
}
