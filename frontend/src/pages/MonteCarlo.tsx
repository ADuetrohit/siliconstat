import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import {
  cancelJob, getExamples, pollJob, reproduceRun, runCsvUrl, runReportUrl,
  startMonteCarlo,
} from '@/lib/api'
import {
  formatDuration, formatEng, formatPercent, formatWithInterval, yieldTone,
} from '@/lib/format'
import { useAsync } from '@/lib/hooks'
import {
  ConvergencePlot, CorrelationHeatmap, CdfPlot, Histogram, SensitivityBars,
} from '@/components/plots'
import {
  Badge, Button, DataTable, EmptyState, ErrorBanner, Field, KeyValueGrid,
  NoteBanner, NumberInput, Panel, ProgressBar, SectionTitle, Select, Spinner,
  StatTile, Toggle, type Column,
} from '@/components/ui'
import type {
  DistributionName, FailureRow, Job, RunAnalysis, SensitivityEntry, SpecYield,
} from '@/types/api'

const CORNERS = ['TT', 'FF', 'SS', 'FS', 'SF']
const PRESETS = [10, 100, 1000, 10000]

export default function MonteCarlo() {
  const [params] = useSearchParams()
  const examples = useAsync(getExamples, [])

  const [example, setExample] = useState('')
  const [samples, setSamples] = useState(1000)
  const [seed, setSeed] = useState(12345)
  const [workers, setWorkers] = useState(1)
  const [sampling, setSampling] = useState<'standard' | 'latin_hypercube'>('standard')

  const [process, setProcess] = useState(true)
  const [mismatch, setMismatch] = useState(true)
  const [pelgrom, setPelgrom] = useState(true)
  const [passives, setPassives] = useState(true)
  const [sigmaVth, setSigmaVth] = useState(0.025)
  const [sigmaBeta, setSigmaBeta] = useState(3)
  const [distribution, setDistribution] = useState<DistributionName>('gaussian')
  const [truncate, setTruncate] = useState(0)

  const [usePvt, setUsePvt] = useState(false)
  const [corner, setCorner] = useState('TT')
  const [supply, setSupply] = useState(1.8)
  const [temperature, setTemperature] = useState(27)

  const [job, setJob] = useState<Job | null>(null)
  const [analysis, setAnalysis] = useState<RunAnalysis | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [metric, setMetric] = useState<string>('')
  const [reproduction, setReproduction] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!examples.data?.length) return
    const requested = params.get('example')
    const chosen = examples.data.find((e) => e.id === requested) ?? examples.data[0]
    setExample((current) => current || chosen.id)
  }, [examples.data, params])

  const mode = process && mismatch ? 'both' : process ? 'process'
    : mismatch ? 'mismatch' : 'nominal'

  const measurementNames = useMemo(
    () => (analysis ? Object.keys(analysis.statistics) : []), [analysis])

  useEffect(() => {
    if (!analysis) return
    const specced = analysis.spec_meta[0]?.measure
    setMetric(specced && analysis.statistics[specced]
      ? specced : Object.keys(analysis.statistics)[0] ?? '')
  }, [analysis])

  const launch = async () => {
    setError(null)
    setAnalysis(null)
    setReproduction(null)
    setRunId(null)
    abort.current?.abort()
    abort.current = new AbortController()
    try {
      const started = await startMonteCarlo({
        example,
        samples, seed, workers, sampling,
        variation: {
          mode, sigma_vth_global: sigmaVth, sigma_beta_global_pct: sigmaBeta,
          include_passives: passives, pelgrom,
          distribution, truncate_sigma: truncate > 0 ? truncate : null,
        },
        pvt: usePvt ? { corner, supply, temp_c: temperature } : null,
      })
      const finished = await pollJob(started.job_id, setJob,
                                     { signal: abort.current.signal })
      setJob(finished)
      if (finished.status === 'failed') {
        setError(finished.error ?? 'the run failed')
      } else if (finished.result?.analysis) {
        setAnalysis(finished.result.analysis as RunAnalysis)
        setRunId(finished.result.run_id as string)
      }
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const running = job?.status === 'queued' || job?.status === 'running'
  const chart = analysis && metric ? analysis.charts[metric] : undefined
  const stats = analysis && metric ? analysis.statistics[metric] : undefined
  const sensitivity = analysis && metric ? analysis.sensitivity[metric] : undefined
  const trace = analysis && metric ? analysis.convergence[metric] : undefined
  const [corrKind, setCorrKind] = useState<'pearson' | 'spearman'>('pearson')

  const specColumns: Column<SpecYield>[] = [
    { key: 'spec', header: 'Specification', render: (s) => s.description },
    { key: 'pass', header: 'Pass', align: 'right', render: (s) => s.passing },
    { key: 'fail', header: 'Fail', align: 'right', render: (s) => s.failing },
    { key: 'yield', header: 'Yield', align: 'right', render: (s) => (
      <Badge tone={yieldTone(s.yield_pct)}>{formatPercent(s.yield_pct)}</Badge>) },
    { key: 'ci', header: '95% CI', align: 'right',
      render: (s) => `[${s.ci95_low.toFixed(2)}, ${s.ci95_high.toFixed(2)}]` },
    { key: 'margin', header: 'Margin [σ]', align: 'right',
      render: (s) => (Number.isFinite(s.margin_sigma) ? s.margin_sigma.toFixed(2) : 'n/a') },
    { key: 'cpk', header: 'Cpk', align: 'right',
      render: (s) => (Number.isFinite(s.cpk) ? s.cpk.toFixed(2) : 'n/a') },
  ]

  const sensitivityColumns: Column<SensitivityEntry>[] = [
    { key: 'rank', header: '#', align: 'right', render: (e) => e.rank },
    { key: 'param', header: 'Parameter', render: (e) => e.parameter },
    { key: 'scope', header: 'Scope', render: (e) => (
      <Badge tone={e.scope === 'global' ? 'info' : 'pass'}>{e.scope}</Badge>) },
    { key: 'beta', header: 'β*', align: 'right',
      render: (e) => e.beta_standardised.toFixed(4) },
    { key: 'share', header: 'Variance', align: 'right',
      render: (e) => formatPercent(e.variance_contribution_pct) },
    { key: 'pearson', header: 'Pearson r', align: 'right',
      render: (e) => e.pearson.toFixed(4) },
    { key: 'spearman', header: 'Spearman ρ', align: 'right',
      render: (e) => e.spearman.toFixed(4) },
  ]

  const failureColumns: Column<FailureRow>[] = [
    { key: 'status', header: 'Status', render: (r) => <Badge tone="fail">{r.status}</Badge> },
    { key: 'reason', header: 'Reason', render: (r) => (
      <span className="text-muted">{r.reason}</span>) },
    { key: 'count', header: 'Count', align: 'right', render: (r) => r.count },
    { key: 'pct', header: '%', align: 'right', render: (r) => formatPercent(r.percent) },
  ]

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-[18px] font-semibold text-ink">Monte Carlo</h1>
        <p className="text-[11.5px] text-muted">
          Randomise device parameters, re-simulate, and analyse the resulting distributions.
        </p>
      </header>

      <Panel title="Monte Carlo configuration">
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="space-y-3">
            <Field label="Circuit">
              <Select
                value={example}
                onChange={setExample}
                options={(examples.data ?? []).map((e) => ({ value: e.id, label: e.name }))}
              />
            </Field>
            <Field label="Samples">
              <div className="flex flex-wrap gap-1.5">
                {PRESETS.map((preset) => (
                  <Button key={preset}
                          variant={samples === preset ? 'primary' : 'default'}
                          onClick={() => setSamples(preset)}>
                    {preset.toLocaleString()}
                  </Button>
                ))}
              </div>
              <div className="mt-2">
                <NumberInput value={samples} onChange={setSamples} min={1} max={1000000} />
              </div>
            </Field>
            <Field
              label="Seed"
              hint="The seed is the whole reproducibility record: same seed, same run."
            >
              <div className="flex gap-2">
                <NumberInput value={seed} onChange={setSeed} min={0} />
                <Button onClick={() => setSeed(Math.floor(Math.random() * 2 ** 31))}>
                  Randomise
                </Button>
              </div>
            </Field>
          </div>

          <div className="space-y-3">
            <Field label="Variation sources">
              <div className="space-y-1.5">
                <Toggle checked={process} onChange={setProcess}
                        label="Process (global, shared by model card)" />
                <Toggle checked={mismatch} onChange={setMismatch}
                        label="Local mismatch (independent per device)" />
                <Toggle checked={pelgrom} onChange={setPelgrom}
                        label="Pelgrom area scaling" />
                <Toggle checked={passives} onChange={setPassives}
                        label="Include passives" />
              </div>
              <div className="mt-1.5">
                <Badge tone="muted">mode: {mode}</Badge>
              </div>
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="σ Vth global [V]">
                <NumberInput value={sigmaVth} onChange={setSigmaVth} step={0.005} min={0} />
              </Field>
              <Field label="σ β global [%]">
                <NumberInput value={sigmaBeta} onChange={setSigmaBeta} step={0.5} min={0} />
              </Field>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Distribution">
                <Select
                  value={distribution}
                  onChange={setDistribution}
                  options={[
                    { value: 'gaussian', label: 'Gaussian' },
                    { value: 'uniform', label: 'Uniform' },
                    { value: 'lognormal', label: 'Log-normal' },
                  ]}
                />
              </Field>
              <Field label="Truncate [σ], 0 = off">
                <NumberInput value={truncate} onChange={setTruncate} min={0} max={10}
                             step={0.5} />
              </Field>
            </div>
          </div>

          <div className="space-y-3">
            <Field label="PVT condition">
              <Toggle checked={usePvt} onChange={setUsePvt}
                      label="Apply a corner / supply / temperature" />
            </Field>
            {usePvt && (
              <div className="grid grid-cols-3 gap-2">
                <Field label="Corner">
                  <Select value={corner} onChange={setCorner}
                          options={CORNERS.map((c) => ({ value: c, label: c }))} />
                </Field>
                <Field label="Supply [V]">
                  <NumberInput value={supply} onChange={setSupply} step={0.06} min={0.1} />
                </Field>
                <Field label="Temp [°C]">
                  <NumberInput value={temperature} onChange={setTemperature} step={5} />
                </Field>
              </div>
            )}
            <div className="grid grid-cols-2 gap-2">
              <Field label="Sampling">
                <Select
                  value={sampling}
                  onChange={setSampling}
                  options={[
                    { value: 'standard', label: 'Standard' },
                    { value: 'latin_hypercube', label: 'Latin hypercube' },
                  ]}
                />
              </Field>
              <Field label="Workers"
                     hint="Per-sample seeding makes parallel identical to sequential.">
                <NumberInput value={workers} onChange={setWorkers} min={1} max={16} />
              </Field>
            </div>
            <Button variant="primary" onClick={() => void launch()}
                    disabled={running || !example}>
              {running ? 'Running…' : 'Run Monte Carlo'}
            </Button>
          </div>
        </div>
      </Panel>

      {error && <ErrorBanner message={error} />}

      {job && (
        <Panel
          title="Simulation status"
          actions={running
            ? <Button variant="danger" onClick={() => void cancelJob(job.job_id)}>Cancel</Button>
            : <Badge tone={job.status === 'done' ? 'pass' : 'fail'}>{job.status}</Badge>}
        >
          <ProgressBar value={job.completed} total={job.total || 1} />
          <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-[12px]">
            <span>{job.completed.toLocaleString()} / {job.total.toLocaleString()}</span>
            <span className="text-good">successful {job.successful.toLocaleString()}</span>
            <span className={job.failed ? 'text-bad' : 'text-muted'}>
              failed {job.failed.toLocaleString()}
            </span>
            <span className="text-muted">elapsed {formatDuration(job.elapsed_s)}</span>
            {job.elapsed_s > 0 && job.completed > 0 && (
              <span className="text-muted">
                {(job.completed / job.elapsed_s).toFixed(1)} samples/s
              </span>
            )}
            {job.eta_s !== null && running && (
              <span className="text-accent">eta {formatDuration(job.eta_s)}</span>
            )}
          </div>
          {job.error && <div className="mt-2"><ErrorBanner message={job.error} /></div>}
        </Panel>
      )}

      {running && !analysis && <Spinner label="Simulating…" />}

      {analysis && (
        <>
          <div className="flex items-center gap-3">
            <SectionTitle>Results</SectionTitle>
            <Select
              value={metric}
              onChange={setMetric}
              options={measurementNames.map((n) => ({ value: n, label: n }))}
            />
          </div>

          {stats && (
            <div className="grid gap-3 md:grid-cols-5">
              <StatTile label={`mean(${metric})`}
                        value={formatEng(stats.mean, stats.unit)} />
              <StatTile label="sigma" value={formatEng(stats.std, stats.unit)} />
              <StatTile label="±3σ window"
                        value={<span className="text-[13px]">
                          {formatEng(stats.sigma3_low, stats.unit)} …{' '}
                          {formatEng(stats.sigma3_high, stats.unit)}
                        </span>} />
              <StatTile label="nominal"
                        value={formatEng(analysis.nominal[metric], stats.unit)} />
              <StatTile
                label="Combined yield"
                tone={yieldTone(analysis.yield.combined_yield_over_successful)}
                value={formatPercent(analysis.yield.combined_yield_over_successful)}
                sub={`${analysis.yield.combined_passing} / ${analysis.yield.successful} successful`}
              />
            </div>
          )}

          {chart && stats && (
            <div className="grid gap-4 xl:grid-cols-2">
              <Panel title={`Distribution — ${metric}`}>
                <Histogram chart={chart} stats={stats} />
              </Panel>
              <Panel title={`Cumulative distribution — ${metric}`}>
                <CdfPlot chart={chart} />
              </Panel>
            </div>
          )}

          <Panel title="Yield">
            <KeyValueGrid columns={3} items={[
              ['Combined yield', formatWithInterval(
                analysis.yield.combined_yield_over_successful,
                analysis.yield.combined_ci95_low, analysis.yield.combined_ci95_high)],
              ['Denominator',
                `${analysis.yield.combined_passing} of ${analysis.yield.successful} successful simulations`],
              ['Over all attempted',
                `${formatPercent(analysis.yield.combined_yield_over_attempted)} (${analysis.yield.combined_passing} / ${analysis.yield.attempted})`],
              ['Limiting specification', analysis.yield.limiting_spec || '—'],
            ]} />
            <div className="mt-3">
              <DataTable columns={specColumns} rows={analysis.yield.per_spec}
                         rowKey={(s) => s.key} dense
                         emptyMessage="No specifications declared, so no yield can be computed." />
            </div>
            {analysis.yield.notes.map((note) => (
              <div key={note} className="mt-2"><NoteBanner>{note}</NoteBanner></div>
            ))}
          </Panel>

          <Panel
            title="Correlation"
            subtitle="Drawn parameters against measured outputs"
            actions={
              <Select
                value={corrKind}
                onChange={setCorrKind}
                options={[
                  { value: 'pearson', label: 'Pearson (linear)' },
                  { value: 'spearman', label: 'Spearman (rank)' },
                ]}
              />
            }
          >
            <CorrelationHeatmap
              matrix={corrKind === 'pearson'
                ? analysis.correlation.pearson : analysis.correlation.spearman}
              rows={analysis.correlation.parameters}
              columns={analysis.correlation.measurements}
            />
            {analysis.correlation.notes.map((note) => (
              <div key={note} className="mt-2"><NoteBanner>{note}</NoteBanner></div>
            ))}
          </Panel>

          {sensitivity && (
            <Panel
              title={`Sensitivity — ${metric}`}
              subtitle={sensitivity.method === 'regression'
                ? 'variance-based (standardised regression)'
                : 'correlation-based ranking, not a variance decomposition'}
            >
              <KeyValueGrid columns={4} items={[
                ['R²', Number.isFinite(sensitivity.r_squared)
                  ? sensitivity.r_squared.toFixed(5) : 'n/a'],
                ['Unexplained variance', formatPercent(sensitivity.unexplained_pct)],
                ['Max |r| between inputs', sensitivity.max_input_correlation.toFixed(3)],
                ['Samples used', String(sensitivity.n_samples)],
              ]} />
              <div className="mt-3">
                <SensitivityBars entries={sensitivity.entries} />
              </div>
              <div className="mt-3">
                <DataTable columns={sensitivityColumns} rows={sensitivity.entries}
                           rowKey={(e) => e.parameter} dense />
              </div>
              {sensitivity.notes.map((note) => (
                <div key={note} className="mt-2"><NoteBanner>{note}</NoteBanner></div>
              ))}
            </Panel>
          )}

          {trace && (
            <div className="grid gap-4 xl:grid-cols-2">
              <Panel title={`Convergence of the mean — ${metric}`}>
                <ConvergencePlot trace={trace} mode="mean" />
              </Panel>
              {trace.yield_pct.length > 0 && (
                <Panel title="Convergence of the yield">
                  <ConvergencePlot trace={trace} mode="yield" />
                </Panel>
              )}
            </div>
          )}

          <Panel title="Failure analysis">
            {analysis.failure_breakdown.length ? (
              <DataTable columns={failureColumns} rows={analysis.failure_breakdown}
                         rowKey={(r, i) => `${r.status}-${i}`} dense />
            ) : (
              <p className="text-[12px] text-good">
                No sample failed: every attempted simulation converged and produced
                valid measurements.
              </p>
            )}
          </Panel>

          <Panel
            title="Reproducibility"
            actions={runId && (
              <>
                <a href={runCsvUrl(runId)} download>
                  <Button>Download CSV</Button>
                </a>
                <a href={runReportUrl(runId)} target="_blank" rel="noreferrer">
                  <Button>Open report</Button>
                </a>
                <Button
                  variant="primary"
                  onClick={async () => {
                    setReproduction('running…')
                    try {
                      const started = await reproduceRun(runId)
                      const done = await pollJob(started.job_id)
                      const check = done.result?.reproduction_check
                      setReproduction(check
                        ? (check.identical
                          ? `Identical: ${check.values_compared} values compared across ${check.samples_compared} samples, 0 mismatches.`
                          : `MISMATCH: ${check.mismatch_count} differing values.`)
                        : (done.error ?? 'reproduction failed'))
                    } catch (err) {
                      setReproduction((err as Error).message)
                    }
                  }}
                >
                  Reproduce this run
                </Button>
              </>
            )}
          >
            <KeyValueGrid columns={3} items={[
              ['Run id', analysis.reproduction.run_id],
              ['Seed', String(analysis.reproduction.seed)],
              ['Samples', String(analysis.reproduction.samples)],
              ['Sampling', analysis.reproduction.sampling],
              ['Variation mode', analysis.reproduction.variation?.mode ?? mode],
              ['PVT', analysis.reproduction.pvt?.label ?? 'nominal'],
              ['Software version', analysis.reproduction.software_version],
              ['Platform', analysis.reproduction.platform],
              ['Circuit SHA-256', analysis.reproduction.circuit_sha256.slice(0, 24) + '…'],
              ['Timestamp', analysis.reproduction.timestamp],
            ]} />
            {reproduction && (
              <div className="mt-3">
                {reproduction.startsWith('Identical')
                  ? <div className="rounded-md border-l-[3px] border-good bg-good/10 px-3 py-2 text-[12px] text-good">
                      {reproduction}
                    </div>
                  : <NoteBanner>{reproduction}</NoteBanner>}
              </div>
            )}
          </Panel>
        </>
      )}

      {!job && !analysis && !error && (
        <EmptyState
          title="No run yet"
          hint="Pick a circuit, choose a sample count and a seed, and press Run Monte Carlo."
        />
      )}
    </div>
  )
}
