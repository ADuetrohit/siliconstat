import { useEffect, useState } from 'react'

import { getRunAnalysis, listRuns } from '@/lib/api'
import { formatEng, formatPercent } from '@/lib/format'
import { useAsync } from '@/lib/hooks'
import { defaultMetric } from '@/lib/metrics'
import { CorrelationHeatmap, SensitivityBars } from '@/components/plots'
import {
  Badge, DataTable, EmptyState, ErrorBanner, KeyValueGrid, NoteBanner, Panel,
  Select, Spinner, type Column,
} from '@/components/ui'
import type { RunAnalysis, SensitivityEntry } from '@/types/api'

export default function Sensitivity() {
  const runs = useAsync(() => listRuns(80), [])
  const [selected, setSelected] = useState('')
  const [metric, setMetric] = useState('')
  const [kind, setKind] = useState<'pearson' | 'spearman'>('pearson')

  useEffect(() => {
    if (!selected && runs.data?.length) setSelected(runs.data[0].run_id)
  }, [runs.data, selected])

  const analysis = useAsync<RunAnalysis | null>(
    () => (selected ? getRunAnalysis(selected) : Promise.resolve(null)), [selected])

  useEffect(() => {
    if (!analysis.data) return
    const names = Object.keys(analysis.data.sensitivity)
    setMetric((current) => (current && names.includes(current)
      ? current : defaultMetric(analysis.data!, names)))
  }, [analysis.data])

  const report = analysis.data && metric ? analysis.data.sensitivity[metric] : undefined
  const correlation = analysis.data?.correlation
  const measurementCorrelation = analysis.data?.measurement_correlation

  const columns: Column<SensitivityEntry>[] = [
    { key: 'rank', header: '#', align: 'right', render: (e) => e.rank },
    { key: 'param', header: 'Parameter', render: (e) => e.parameter },
    { key: 'scope', header: 'Scope', render: (e) => (
      <Badge tone={e.scope === 'global' ? 'info' : 'pass'}>{e.scope}</Badge>) },
    { key: 'beta', header: 'β* (standardised)', align: 'right',
      render: (e) => (Number.isFinite(e.beta_standardised)
        ? e.beta_standardised.toFixed(4) : 'n/a'),
      sortValue: (e) => Math.abs(e.beta_standardised) },
    { key: 'var', header: 'Variance share', align: 'right',
      render: (e) => formatPercent(e.variance_contribution_pct),
      sortValue: (e) => e.variance_contribution_pct },
    { key: 'expl', header: '% of explained', align: 'right',
      render: (e) => formatPercent(e.share_of_explained_pct) },
    { key: 'pearson', header: 'Pearson r', align: 'right',
      render: (e) => e.pearson.toFixed(4), sortValue: (e) => Math.abs(e.pearson) },
    { key: 'spearman', header: 'Spearman ρ', align: 'right',
      render: (e) => e.spearman.toFixed(4) },
    { key: 'sigma', header: 'Input σ', align: 'right',
      render: (e) => `${e.sigma.toPrecision(4)} ${e.unit}` },
    { key: 'slope', header: 'Δoutput / 1σ', align: 'right',
      render: (e) => formatEng(e.d_output_d_sigma) },
  ]

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-[18px] font-semibold text-ink">Sensitivity</h1>
        <p className="text-[11.5px] text-muted">
          Which drawn parameters actually move each measurement, and how much of the
          spread the model can explain.
        </p>
      </header>

      <Panel title="Selection">
        <div className="grid gap-3 md:grid-cols-2">
          {runs.data && (
            <Select
              value={selected}
              onChange={setSelected}
              options={runs.data.map((r) => ({
                value: r.run_id,
                label: `${r.circuit_name} · ${r.variation_mode || 'nominal'} · ` +
                       `${r.samples} samples · ${r.run_id.slice(0, 8)}`,
              }))}
            />
          )}
          {analysis.data && (
            <Select
              value={metric}
              onChange={setMetric}
              options={Object.keys(analysis.data.sensitivity)
                .map((n) => ({ value: n, label: n }))}
            />
          )}
        </div>
        {runs.data && !runs.data.length && (
          <EmptyState title="No stored runs" hint="Run a Monte Carlo experiment first." />
        )}
      </Panel>

      {analysis.loading && <Spinner label="Loading analysis…" />}
      {analysis.error && <ErrorBanner message={analysis.error} onRetry={analysis.reload} />}

      {report && (
        <>
          <Panel
            title={`Method — ${metric}`}
            subtitle={report.method === 'regression'
              ? 'Variance-based decomposition (standardised least-squares regression)'
              : 'Correlation-based ranking, not a variance decomposition'}
          >
            <KeyValueGrid columns={4} items={[
              ['Method', report.method === 'regression'
                ? 'variance-based (β²)' : 'correlation ranking'],
              ['R²', Number.isFinite(report.r_squared)
                ? report.r_squared.toFixed(5) : 'n/a'],
              ['Unexplained variance', formatPercent(report.unexplained_pct)],
              ['Max |r| between inputs', report.max_input_correlation.toFixed(3)],
              ['Output sigma', formatEng(report.output_sigma)],
              ['Output mean', formatEng(report.output_mean)],
              ['Samples used', String(report.n_samples)],
              ['Parameters', String(report.entries.length)],
            ]} />
            {report.method === 'correlation' && (
              <div className="mt-3">
                <NoteBanner>
                  This ranking is correlation-based. Correlations do not sum to
                  anything, so the ordering is indicative — it is not a partition of
                  the output variance.
                </NoteBanner>
              </div>
            )}
            {report.notes.map((note) => (
              <div key={note} className="mt-2"><NoteBanner>{note}</NoteBanner></div>
            ))}
          </Panel>

          <Panel title="Ranked contribution">
            <SensitivityBars entries={report.entries} />
          </Panel>

          <Panel title="Full ranking">
            <DataTable columns={columns} rows={report.entries}
                       rowKey={(e) => e.parameter} dense />
          </Panel>
        </>
      )}

      {correlation && correlation.parameters.length > 0 && (
        <Panel
          title="Parameter × measurement correlation"
          subtitle={`${correlation.n_samples} samples`}
          actions={
            <Select
              value={kind}
              onChange={setKind}
              options={[
                { value: 'pearson', label: 'Pearson (linear)' },
                { value: 'spearman', label: 'Spearman (rank)' },
              ]}
            />
          }
        >
          <CorrelationHeatmap
            matrix={kind === 'pearson' ? correlation.pearson : correlation.spearman}
            rows={correlation.parameters}
            columns={correlation.measurements}
          />
          <p className="mt-2 text-[11px] text-muted">
            A large Spearman coefficient paired with a small Pearson coefficient is the
            signature of a strong but nonlinear dependence — exactly where a
            correlation-based ranking would mislead.
          </p>
          {correlation.notes.map((note) => (
            <div key={note} className="mt-2"><NoteBanner>{note}</NoteBanner></div>
          ))}
        </Panel>
      )}

      {measurementCorrelation?.pearson?.length ? (
        <Panel
          title="Measurement cross-correlation"
          subtitle="Which specifications fail together"
        >
          <CorrelationHeatmap
            matrix={measurementCorrelation.pearson}
            rows={measurementCorrelation.measurements}
            columns={measurementCorrelation.measurements}
          />
        </Panel>
      ) : null}
    </div>
  )
}
