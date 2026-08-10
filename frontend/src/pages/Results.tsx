import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import {
  compareRuns, deleteRun, getRunAnalysis, getRunSamples, listRuns, runCsvUrl,
  runReportUrl,
} from '@/lib/api'
import {
  formatDuration, formatEng, formatPercent, formatTimestamp, yieldTone,
} from '@/lib/format'
import { useAsync } from '@/lib/hooks'
import { defaultMetric } from '@/lib/metrics'
import { BoxPlot, CdfPlot, Histogram, OverlaidHistograms } from '@/components/plots'
import {
  Badge, Button, DataTable, EmptyState, ErrorBanner, Panel,
  SectionTitle, Select, Spinner, StatTile, Toggle, type Column,
} from '@/components/ui'
import type {
  CompareResponse, RunAnalysis, RunListing, SampleRecord,
} from '@/types/api'

export default function Results() {
  const [params, setParams] = useSearchParams()
  const runs = useAsync(() => listRuns(80), [])
  const [selected, setSelected] = useState<string | null>(params.get('run'))
  const [compareMode, setCompareMode] = useState(false)
  const [compareIds, setCompareIds] = useState<string[]>([])
  const [comparison, setComparison] = useState<CompareResponse | null>(null)
  const [compareError, setCompareError] = useState<string | null>(null)
  const [metric, setMetric] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    if (!selected && runs.data?.length) setSelected(runs.data[0].run_id)
  }, [runs.data, selected])

  const analysis = useAsync<RunAnalysis | null>(
    () => (selected ? getRunAnalysis(selected) : Promise.resolve(null)), [selected])

  const samples = useAsync(
    () => (selected
      ? getRunSamples(selected, offset, 50, statusFilter || undefined)
      : Promise.resolve(null)),
    [selected, offset, statusFilter])

  useEffect(() => {
    if (!analysis.data) return
    const names = Object.keys(analysis.data.statistics)
    setMetric((current) => (current && names.includes(current)
      ? current : defaultMetric(analysis.data!, names)))
  }, [analysis.data])

  const runColumns: Column<RunListing>[] = [
    ...(compareMode ? [{
      key: 'pick', header: '', width: '34px',
      render: (r: RunListing) => (
        <input
          type="checkbox"
          checked={compareIds.includes(r.run_id)}
          onChange={(e) => setCompareIds((current) =>
            e.target.checked
              ? [...current, r.run_id].slice(0, 4)
              : current.filter((id) => id !== r.run_id))}
        />
      ),
    } as Column<RunListing>] : []),
    { key: 'run', header: 'Run', render: (r) => (
      <span className="text-accent">{r.run_id.slice(0, 12)}</span>) },
    { key: 'circuit', header: 'Circuit', render: (r) => r.circuit_name,
      sortValue: (r) => r.circuit_name },
    { key: 'mode', header: 'Mode', render: (r) => (
      <Badge tone="muted">{r.variation_mode || 'nominal'}</Badge>) },
    { key: 'pvt', header: 'PVT', render: (r) => r.pvt_label || 'nominal' },
    { key: 'n', header: 'Samples', align: 'right', render: (r) => r.samples,
      sortValue: (r) => r.samples },
    { key: 'ok', header: 'OK', align: 'right',
      render: (r) => <span className="text-good">{r.counters.successful}</span> },
    { key: 'fail', header: 'Fail', align: 'right', render: (r) => (
      <span className={r.counters.failed ? 'text-bad' : 'text-muted'}>
        {r.counters.failed}
      </span>) },
    { key: 'seed', header: 'Seed', align: 'right', render: (r) => r.seed },
    { key: 'dur', header: 'Duration', align: 'right',
      render: (r) => formatDuration(r.duration_s), sortValue: (r) => r.duration_s },
    { key: 'started', header: 'Started', render: (r) => formatTimestamp(r.started_at),
      sortValue: (r) => r.started_at },
  ]

  const sampleColumns: Column<SampleRecord>[] = useMemo(() => {
    const names = analysis.data ? Object.keys(analysis.data.statistics).slice(0, 5) : []
    return [
      { key: 'i', header: '#', align: 'right', render: (s) => s.index },
      { key: 'status', header: 'Status', render: (s) => (
        <Badge tone={s.status === 'ok' ? 'pass' : 'fail'}>{s.status}</Badge>) },
      { key: 'verdict', header: 'Verdict', render: (s) =>
        s.passed === null ? <span className="text-muted">—</span>
          : <Badge tone={s.passed ? 'pass' : 'fail'}>{s.passed ? 'pass' : 'fail'}</Badge> },
      ...names.map((name) => ({
        key: `m-${name}`, header: name, align: 'right' as const,
        render: (s: SampleRecord) => {
          const value = s.measurements[name]
          return value === undefined || Number.isNaN(value)
            ? <span className="text-muted">n/a</span>
            : value.toPrecision(5)
        },
      })),
      { key: 'iters', header: 'Newton', align: 'right', render: (s) => s.dc_iterations },
      { key: 'seed', header: 'Seed', align: 'right', render: (s) => (
        <span className="text-[10px] text-muted">{s.seed}</span>) },
      { key: 'reason', header: 'Failure reason', render: (s) => (
        <span className="text-[10.5px] text-muted">{s.failure_reason.slice(0, 90)}</span>) },
    ]
  }, [analysis.data])

  const runComparison = async () => {
    setCompareError(null)
    try {
      setComparison(await compareRuns(compareIds))
    } catch (error) {
      setCompareError((error as Error).message)
      setComparison(null)
    }
  }

  const sharedMetric = comparison?.shared_measurements.includes(metric)
    ? metric : comparison?.shared_measurements[0] ?? ''

  return (
    <div className="space-y-5">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-[18px] font-semibold text-ink">Results</h1>
          <p className="text-[11.5px] text-muted">
            Browse stored runs, inspect individual samples, and compare experiments.
          </p>
        </div>
        <Toggle checked={compareMode} onChange={(value) => {
          setCompareMode(value)
          if (!value) { setComparison(null); setCompareIds([]) }
        }} label="Comparison mode" />
      </header>

      <Panel
        title="Stored runs"
        subtitle={compareMode ? 'Select 2 to 4 runs, then compare' : undefined}
        actions={compareMode && (
          <Button variant="primary" disabled={compareIds.length < 2}
                  onClick={() => void runComparison()}>
            Compare {compareIds.length} runs
          </Button>
        )}
      >
        {runs.loading && <Spinner />}
        {runs.error && <ErrorBanner message={runs.error} onRetry={runs.reload} />}
        {runs.data && (
          <DataTable
            columns={runColumns}
            rows={runs.data}
            rowKey={(r) => r.run_id}
            selectedKey={selected ?? undefined}
            onRowClick={compareMode ? undefined : (r) => {
              setSelected(r.run_id)
              setOffset(0)
              setParams({ run: r.run_id })
            }}
            dense
            emptyMessage="No runs stored yet."
          />
        )}
      </Panel>

      {compareError && <ErrorBanner message={compareError} />}

      {comparison && (
        <>
          <Panel title="Run comparison">
            <DataTable
              columns={[
                { key: 'run', header: 'Run', render: (e) => e.run_id.slice(0, 12) },
                { key: 'circuit', header: 'Circuit', render: (e) => e.summary.circuit },
                { key: 'mode', header: 'Mode', render: (e) => e.summary.variation_mode },
                { key: 'pvt', header: 'PVT',
                  render: (e) => e.summary.pvt?.label ?? 'nominal' },
                { key: 'n', header: 'Samples', align: 'right',
                  render: (e) => e.counters.total },
                { key: 'ok', header: 'OK', align: 'right',
                  render: (e) => e.counters.successful },
                { key: 'yield', header: 'Combined yield', align: 'right', render: (e) =>
                  e.yield.per_spec.length
                    ? <Badge tone={yieldTone(e.yield.combined_yield_over_successful)}>
                        {formatPercent(e.yield.combined_yield_over_successful)}
                      </Badge>
                    : <span className="text-muted">n/a</span> },
              ]}
              rows={comparison.runs}
              rowKey={(e) => e.run_id}
              dense
            />
          </Panel>

          {sharedMetric && (
            <>
              <div className="flex items-center gap-3">
                <SectionTitle>Shared measurement</SectionTitle>
                <Select
                  value={sharedMetric}
                  onChange={setMetric}
                  options={comparison.shared_measurements.map((n) => ({ value: n, label: n }))}
                />
              </div>
              <div className="grid gap-4 xl:grid-cols-2">
                <Panel title={`Overlaid distributions — ${sharedMetric}`}>
                  <OverlaidHistograms
                    unit={comparison.runs[0]?.statistics[sharedMetric]?.unit ?? ''}
                    series={comparison.runs
                      .filter((e) => e.charts[sharedMetric])
                      .map((e) => ({
                        label: `${e.run_id.slice(0, 8)} (${e.counters.total})`,
                        chart: e.charts[sharedMetric],
                      }))}
                  />
                </Panel>
                <Panel title="Spread comparison">
                  <BoxPlot
                    unit={comparison.runs[0]?.statistics[sharedMetric]?.unit ?? ''}
                    groups={comparison.runs
                      .filter((e) => e.statistics[sharedMetric])
                      .map((e) => ({
                        label: e.run_id.slice(0, 8),
                        stats: e.statistics[sharedMetric],
                      }))}
                  />
                </Panel>
              </div>
              <Panel title="Delta table">
                <DataTable
                  columns={[
                    { key: 'run', header: 'Run', render: (e) => e.run_id.slice(0, 12) },
                    { key: 'mode', header: 'Mode', render: (e) => e.summary.variation_mode },
                    { key: 'n', header: 'N', align: 'right',
                      render: (e) => e.statistics[sharedMetric]?.count ?? 0 },
                    { key: 'mean', header: 'mean', align: 'right', render: (e) =>
                      formatEng(e.statistics[sharedMetric]?.mean,
                                e.statistics[sharedMetric]?.unit) },
                    { key: 'sigma', header: 'sigma', align: 'right', render: (e) =>
                      formatEng(e.statistics[sharedMetric]?.std,
                                e.statistics[sharedMetric]?.unit) },
                    { key: 'dsigma', header: 'Δsigma vs first', align: 'right',
                      render: (e) => {
                        const base = comparison.runs[0]?.statistics[sharedMetric]?.std
                        const value = e.statistics[sharedMetric]?.std
                        if (!base || !value) return '—'
                        return `${(((value - base) / base) * 100).toFixed(2)} %`
                      } },
                    { key: 'yield', header: 'yield', align: 'right', render: (e) =>
                      e.yield.per_spec.length
                        ? formatPercent(e.yield.combined_yield_over_successful) : 'n/a' },
                  ]}
                  rows={comparison.runs}
                  rowKey={(e) => e.run_id}
                  dense
                />
              </Panel>
            </>
          )}
        </>
      )}

      {!compareMode && selected && (
        <>
          {analysis.loading && <Spinner label="Loading analysis…" />}
          {analysis.error && <ErrorBanner message={analysis.error} onRetry={analysis.reload} />}
          {analysis.data && (
            <>
              <Panel
                title={`Run ${analysis.data.run_id}`}
                subtitle={analysis.data.circuit_name}
                actions={
                  <>
                    <a href={runCsvUrl(selected)} download><Button>CSV</Button></a>
                    <a href={runReportUrl(selected)} target="_blank" rel="noreferrer">
                      <Button>Report</Button>
                    </a>
                    <Button variant="danger" onClick={async () => {
                      await deleteRun(selected)
                      setSelected(null)
                      runs.reload()
                    }}>Delete</Button>
                  </>
                }
              >
                <div className="grid gap-3 md:grid-cols-5">
                  <StatTile label="Attempted" value={analysis.data.summary.counters.total} />
                  <StatTile label="Successful" tone="pass"
                            value={analysis.data.summary.counters.successful} />
                  <StatTile label="Failed"
                            tone={analysis.data.summary.counters.failed ? 'fail' : 'default'}
                            value={analysis.data.summary.counters.failed} />
                  <StatTile label="Duration"
                            value={formatDuration(analysis.data.summary.duration_s)}
                            sub={`${analysis.data.summary.samples_per_second.toFixed(1)} samples/s`} />
                  <StatTile
                    label="Combined yield"
                    tone={yieldTone(analysis.data.yield.combined_yield_over_successful)}
                    value={analysis.data.yield.per_spec.length
                      ? formatPercent(analysis.data.yield.combined_yield_over_successful)
                      : 'n/a'}
                  />
                </div>
              </Panel>

              <Panel
                title="Statistics"
                actions={
                  <Select
                    value={metric}
                    onChange={setMetric}
                    options={Object.keys(analysis.data.statistics)
                      .map((n) => ({ value: n, label: n }))}
                  />
                }
              >
                <DataTable
                  columns={[
                    { key: 'name', header: 'Measurement', render: ([name]) => name },
                    { key: 'unit', header: 'Unit', render: ([, s]) => s.unit || '—' },
                    { key: 'n', header: 'N', align: 'right', render: ([, s]) => s.count },
                    { key: 'mean', header: 'mean', align: 'right',
                      render: ([, s]) => formatEng(s.mean, s.unit) },
                    { key: 'median', header: 'median', align: 'right',
                      render: ([, s]) => formatEng(s.median, s.unit) },
                    { key: 'std', header: 'sigma', align: 'right',
                      render: ([, s]) => formatEng(s.std, s.unit) },
                    { key: 'p1', header: 'P1', align: 'right',
                      render: ([, s]) => formatEng(s.p1, s.unit) },
                    { key: 'p99', header: 'P99', align: 'right',
                      render: ([, s]) => formatEng(s.p99, s.unit) },
                    { key: 's3lo', header: '−3σ', align: 'right',
                      render: ([, s]) => formatEng(s.sigma3_low, s.unit) },
                    { key: 's3hi', header: '+3σ', align: 'right',
                      render: ([, s]) => formatEng(s.sigma3_high, s.unit) },
                    { key: 'skew', header: 'skew', align: 'right',
                      render: ([, s]) => s.skewness.toFixed(3) },
                    { key: 'norm', header: 'normality p', align: 'right', render: ([, s]) =>
                      Number.isFinite(s.normality_p)
                        ? <span className={s.normality_p < 0.05 ? 'text-warn' : ''}>
                            {s.normality_p.toFixed(4)}
                          </span>
                        : '—' },
                  ]}
                  rows={Object.entries(analysis.data.statistics)}
                  rowKey={([name]) => name}
                  dense
                />
              </Panel>

              {metric && analysis.data.charts[metric] && (
                <div className="grid gap-4 xl:grid-cols-2">
                  <Panel title={`Distribution — ${metric}`}>
                    <Histogram chart={analysis.data.charts[metric]}
                               stats={analysis.data.statistics[metric]} />
                  </Panel>
                  <Panel title={`Cumulative distribution — ${metric}`}>
                    <CdfPlot chart={analysis.data.charts[metric]} />
                  </Panel>
                </div>
              )}

              <Panel
                title="Samples"
                subtitle="Every attempted sample, including the failures"
                actions={
                  <>
                    <Select
                      value={statusFilter}
                      onChange={(value) => { setStatusFilter(value); setOffset(0) }}
                      options={[
                        { value: '', label: 'All statuses' },
                        { value: 'ok', label: 'ok' },
                        { value: 'convergence_failure', label: 'convergence_failure' },
                        { value: 'invalid_measurement', label: 'invalid_measurement' },
                        { value: 'variation_error', label: 'variation_error' },
                        { value: 'numerical_error', label: 'numerical_error' },
                      ]}
                    />
                    <Button disabled={offset === 0}
                            onClick={() => setOffset(Math.max(0, offset - 50))}>Prev</Button>
                    <Button
                      disabled={!samples.data || offset + 50 >= samples.data.total}
                      onClick={() => setOffset(offset + 50)}
                    >Next</Button>
                  </>
                }
              >
                {samples.loading && <Spinner />}
                {samples.data && (
                  <>
                    <p className="mb-2 text-[11px] text-muted">
                      showing {samples.data.offset + 1}–
                      {Math.min(samples.data.offset + samples.data.limit, samples.data.total)}
                      {' '}of {samples.data.total}
                    </p>
                    <DataTable columns={sampleColumns} rows={samples.data.samples}
                               rowKey={(s) => String(s.index)} dense />
                  </>
                )}
              </Panel>
            </>
          )}
        </>
      )}

      {!runs.loading && !(runs.data ?? []).length && (
        <EmptyState title="No stored runs"
                    hint="Run a Monte Carlo experiment first — the results are saved automatically." />
      )}
    </div>
  )
}
