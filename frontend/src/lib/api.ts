/** Typed client for the SiliconStat REST API. */

import type {
  CircuitDescription, CircuitRef, CompareResponse, DbStats, Example, Health,
  Job, MonteCarloRequest, PVTSweepRequest, RunAnalysis, RunDetail, RunListing,
  SamplePage, SimulateResponse, StoredCircuit, SurrogateRequest,
  ValidateResponse, VersionInfo, WaveformResponse,
} from '@/types/api'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  readonly status: number
  readonly errorType: string

  constructor(message: string, status: number, errorType = '') {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.errorType = errorType
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
      ...init,
    })
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the SiliconStat API at ${BASE || window.location.origin}. ` +
      'Is the backend running (siliconstat serve)?', 0, 'NetworkError',
    )
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    let errorType = ''
    try {
      const payload = await response.json()
      if (typeof payload?.detail === 'string') detail = payload.detail
      else if (Array.isArray(payload?.detail)) {
        detail = payload.detail
          .map((d: any) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg}`)
          .join('; ')
      }
      errorType = payload?.error_type ?? ''
    } catch {
      /* keep the status line */
    }
    throw new ApiError(detail, response.status, errorType)
  }
  return response.json() as Promise<T>
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body) })
}

// ---------------------------------------------------------------------------
// meta
// ---------------------------------------------------------------------------

export const getHealth = () => request<Health>('/api/health')
export const getVersion = () => request<VersionInfo>('/api/version')
export const getDbStats = () => request<DbStats>('/api/db/stats')

// ---------------------------------------------------------------------------
// circuits
// ---------------------------------------------------------------------------

export const getExamples = () => request<Example[]>('/api/examples')
export const listCircuits = () => request<StoredCircuit[]>('/api/circuits')
export const getCircuit = (id: number) =>
  request<StoredCircuit & { netlist: string; summary: CircuitDescription }>(
    `/api/circuits/${id}`)

export const validateNetlist = (netlist: string, name?: string) =>
  post<ValidateResponse>('/api/circuits/validate', { netlist, name })

export const storeCircuit = (netlist: string, name?: string) =>
  post<{ circuit_id: number; name: string; summary: CircuitDescription }>(
    '/api/circuits', { netlist, name })

// ---------------------------------------------------------------------------
// simulation
// ---------------------------------------------------------------------------

export const simulate = (body: CircuitRef & { pvt?: unknown }) =>
  post<SimulateResponse>('/api/simulate', body)

export const startMonteCarlo = (body: MonteCarloRequest) =>
  post<{ job_id: string; status: string; message: string }>(
    '/api/monte-carlo', body)

export const runMonteCarloSync = (body: MonteCarloRequest) =>
  post<{ run_id: string; summary: unknown; analysis: RunAnalysis }>(
    '/api/monte-carlo/sync', body)

export const startPvtSweep = (body: PVTSweepRequest) =>
  post<{ job_id: string; status: string; message: string }>('/api/pvt', body)

export const startSurrogate = (body: SurrogateRequest) =>
  post<{ job_id: string; status: string; message: string }>(
    '/api/ml/surrogate', body)

// ---------------------------------------------------------------------------
// jobs
// ---------------------------------------------------------------------------

export const listJobs = () => request<Job[]>('/api/jobs')
export const getJob = (id: string) => request<Job>(`/api/jobs/${id}`)
export const cancelJob = (id: string) =>
  post<{ job_id: string; status: string }>(`/api/jobs/${id}/cancel`, {})

const TERMINAL: ReadonlySet<string> = new Set(['done', 'failed', 'cancelled'])

/**
 * Poll a background job about once a second until it reaches a terminal state.
 * `onProgress` is called with every observation, including the last.
 */
export async function pollJob(
  jobId: string,
  onProgress?: (job: Job) => void,
  options: { intervalMs?: number; signal?: AbortSignal } = {},
): Promise<Job> {
  const interval = options.intervalMs ?? 900
  for (;;) {
    if (options.signal?.aborted) throw new ApiError('polling aborted', 0, 'Aborted')
    const job = await getJob(jobId)
    onProgress?.(job)
    if (TERMINAL.has(job.status)) return job
    await new Promise((resolve) => setTimeout(resolve, interval))
  }
}

// ---------------------------------------------------------------------------
// runs
// ---------------------------------------------------------------------------

export const listRuns = (limit = 50) =>
  request<RunListing[]>(`/api/runs?limit=${limit}`)

export const getRun = (id: string) => request<RunDetail>(`/api/runs/${id}`)

export const getRunAnalysis = (id: string, refresh = false) =>
  request<RunAnalysis>(`/api/runs/${id}/analysis${refresh ? '?refresh=true' : ''}`)

export const getRunSamples = (id: string, offset = 0, limit = 100, status?: string) =>
  request<SamplePage>(
    `/api/runs/${id}/samples?offset=${offset}&limit=${limit}` +
    (status ? `&status=${encodeURIComponent(status)}` : ''))

export const getRunNetlist = async (id: string): Promise<string> => {
  const response = await fetch(`${BASE}/api/runs/${id}/netlist`)
  if (!response.ok) throw new ApiError('no netlist stored for this run', response.status)
  return response.text()
}

export const reproduceRun = (id: string) =>
  post<{ job_id: string; status: string; message: string }>(
    `/api/runs/${id}/reproduce`, {})

export const compareRuns = (runIds: string[]) =>
  post<CompareResponse>('/api/runs/compare', { run_ids: runIds })

export const deleteRun = (id: string) =>
  request<{ deleted: string }>(`/api/runs/${id}`, { method: 'DELETE' })

export const runCsvUrl = (id: string) => `${BASE}/api/runs/${id}/export.csv`
export const runReportUrl = (id: string) => `${BASE}/api/runs/${id}/report.html`

export const getWaveforms = (body: CircuitRef & { pvt?: unknown; nodes?: string[] }) =>
  post<WaveformResponse>('/api/waveforms', body)
