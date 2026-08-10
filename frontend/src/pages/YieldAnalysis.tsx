import { useEffect, useMemo, useState } from 'react'

import { getRunAnalysis, listRuns } from '@/lib/api'
import { formatEng, formatPercent, formatWithInterval, yieldTone } from '@/lib/format'
import { useAsync } from '@/lib/hooks'
import { YieldBars } from '@/components/plots'
import {
  Badge, DataTable, EmptyState, ErrorBanner, KeyValueGrid, NoteBanner, Panel,
  SectionTitle, Select, Spinner, StatTile, type Column,
} from '@/components/ui'
import type { RunAnalysis, RunListing, SpecYield } from '@/types/api'

const Z95 = 1.959963984540054

/** n = z^2 p(1-p) / m^2 -- the same normal approximation the backend uses. */
function requiredSamples(yieldPct: number, marginPct: number): number {
  const p = Math.min(Math.max(yieldPct / 100, 1e-6), 1 - 1e-6)
  const m = marginPct / 100
  return Math.ceil((Z95 * Z95 * p * (1 - p)) / (m * m))
}

interface PvtCell {
  runId: string
  corner: string
  supply: number | null
  temp: number
  yieldPct: number | null
  label: string
}

export default function YieldAnalysis() {
  const runs = useAsync(() => listRuns(80), [])
  const [selected, setSelected] = useState<string>('')
  const [pvtRuns, setPvtRuns] = useState<PvtCell[]>([])
  const [pvtLoading, setPvtLoading] = useState(false)

  useEffect(() => {
    if (!selected && runs.data?.length) setSelected(runs.data[0].run_id)
  }, [runs.data, selected])

  const analysis = useAsync<RunAnalysis | null>(
    () => (selected ? getRunAnalysis(selected) : Promise.resolve(null)), [selected])

  const circuitName = analysis.data?.circuit_name

  // Build the PVT matrix from every stored run of the same circuit that
  // carries a PVT label.
  useEffect(() => {
    if (!runs.data || !circuitName) { setPvtRuns([]); return }
    const candidates = runs.data.filter(
      (r) => r.circuit_name === circuitName && r.pvt_label && r.pvt_label !== 'nominal')
    if (!candidates.length) { setPvtRuns([]); return }
    let alive = true
    setPvtLoading(true)
    Promise.all(candidates.slice(0, 60).map(async (run: RunListing): Promise<PvtCell> => {
      try {
        const detail = await getRunAnalysis(run.run_id)
        const pvt = detail.reproduction.pvt
        return {
          runId: run.run_id,
          corner: pvt?.corner ?? '?',
          supply: pvt?.supply ?? null,
          temp: pvt?.temp_c ?? 0,
          label: run.pvt_label,
          yieldPct: detail.yield.per_spec.length
            ? detail.yield.combined_yield_over_successful : null,
        }
      } catch {
        return { runId: run.run_id, corner: '?', supply: null, temp: 0,
                 label: run.pvt_label, yieldPct: null }
      }
    })).then((cells) => { if (alive) { setPvtRuns(cells); setPvtLoading(false) } })
    return () => { alive = false }
  }, [runs.data, circuitName])

  const pvtMatrix = useMemo(() => {
    if (!pvtRuns.length) return null
    const corners = [...new Set(pvtRuns.map((c) => c.corner))].sort()
    const conditions = [...new Set(pvtRuns.map(
      (c) => `${c.supply?.toFixed(2) ?? '—'} V / ${c.temp} °C`))].sort()
    const lookup = new Map<string, PvtCell>()
    for (const cell of pvtRuns) {
      lookup.set(`${cell.corner}|${cell.supply?.toFixed(2) ?? '—'} V / ${cell.temp} °C`, cell)
    }
    const worst = pvtRuns
      .filter((c) => c.yieldPct !== null)
      .sort((a, b) => (a.yieldPct ?? 0) - (b.yieldPct ?? 0))[0] ?? null
    return { corners, conditions, lookup, worst }
  }, [pvtRuns])

  const yieldReport = analysis.data?.yield
  const specColumns: Column<SpecYield>[] = [
    { key: 'spec', header: 'Specification', render: (s) => s.description },
    { key: 'pass', header: 'Pass', align: 'right', render: (s) => s.passing },
    { key: 'fail', header: 'Fail', align: 'right', render: (s) => s.failing },
    { key: 'denom', header: 'Denominator', align: 'right', render: (s) => s.denominator },
    { key: 'yield', header: 'Yield', align: 'right', render: (s) => (
      <Badge tone={yieldTone(s.yield_pct)}>{formatPercent(s.yield_pct)}</Badge>) },
    { key: 'ci', header: '95% CI', align: 'right',
      render: (s) => `[${s.ci95_low.toFixed(2)}, ${s.ci95_high.toFixed(2)}]` },
    { key: 'margin', header: 'Margin', align: 'right',
      render: (s) => formatEng(s.margin_mean, s.unit) },
    { key: 'msigma', header: 'Margin [σ]', align: 'right',
      render: (s) => (Number.isFinite(s.margin_sigma) ? s.margin_sigma.toFixed(2) : 'n/a') },
    { key: 'cpk', header: 'Cpk', align: 'right', render: (s) => (
      Number.isFinite(s.cpk)
        ? <span className={s.cpk >= 1.33 ? 'text-good' : s.cpk >= 1 ? 'text-warn' : 'text-bad'}>
            {s.cpk.toFixed(2)}
          </span>
        : 'n/a') },
    { key: 'worst', header: 'Worst sample', align: 'right',
      render: (s) => formatEng(s.worst_value, s.unit) },
  ]

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-[18px] font-semibold text-ink">Yield analysis</h1>
        <p className="text-[11.5px] text-muted">
          Pass fractions with their denominators stated, Wilson confidence intervals,
          capability indices and the PVT matrix.
        </p>
      </header>

      <Panel title="Run">
        {runs.loading && <Spinner />}
        {runs.error && <ErrorBanner message={runs.error} onRetry={runs.reload} />}
        {runs.data && (
          <Select
            value={selected}
            onChange={setSelected}
            options={runs.data.map((r) => ({
              value: r.run_id,
              label: `${r.circuit_name} · ${r.variation_mode || 'nominal'} · ` +
                     `${r.pvt_label || 'nominal'} · ${r.samples} samples · ${r.run_id.slice(0, 8)}`,
            }))}
          />
        )}
        {runs.data && !runs.data.length && (
          <EmptyState title="No stored runs" hint="Run a Monte Carlo experiment first." />
        )}
      </Panel>

      {analysis.loading && <Spinner label="Loading analysis…" />}
      {analysis.error && <ErrorBanner message={analysis.error} onRetry={analysis.reload} />}

      {yieldReport && (
        yieldReport.per_spec.length ? (
          <>
            <div className="grid gap-3 md:grid-cols-4">
              <StatTile
                label="Combined yield"
                tone={yieldTone(yieldReport.combined_yield_over_successful)}
                value={formatPercent(yieldReport.combined_yield_over_successful)}
                sub={formatWithInterval(yieldReport.combined_yield_over_successful,
                                        yieldReport.combined_ci95_low,
                                        yieldReport.combined_ci95_high)}
              />
              <StatTile
                label="Denominator"
                value={`${yieldReport.combined_passing} / ${yieldReport.successful}`}
                sub="successful simulations"
              />
              <StatTile
                label="Over all attempted"
                value={formatPercent(yieldReport.combined_yield_over_attempted)}
                sub={`${yieldReport.combined_passing} / ${yieldReport.attempted} attempted`}
              />
              <StatTile
                label="Limiting specification"
                value={<span className="text-[13px]">{yieldReport.limiting_spec || '—'}</span>}
                sub={`product of individual: ${formatPercent(yieldReport.independent_product_pct)}`}
              />
            </div>

            <Panel title="Per-specification yield">
              <DataTable columns={specColumns} rows={yieldReport.per_spec}
                         rowKey={(s) => s.key} dense />
              <div className="mt-3">
                <YieldBars entries={yieldReport.per_spec} />
              </div>
              <p className="mt-2 text-[11px] text-muted">
                Error bars are Wilson score intervals. Cpk ≥ 1.33 is the conventional
                capability threshold; Cpk = 1.0 puts the limit exactly 3σ from the mean.
              </p>
            </Panel>

            {yieldReport.notes.length > 0 && (
              <Panel title="Caveats">
                <div className="space-y-2">
                  {yieldReport.notes.map((note) => (
                    <NoteBanner key={note}>{note}</NoteBanner>
                  ))}
                </div>
              </Panel>
            )}

            <Panel
              title="How many samples would I need?"
              subtitle="Normal approximation: n = z² p(1−p) / m², z = 1.96 at 95 % confidence"
            >
              <KeyValueGrid columns={4} items={[
                ['Observed yield',
                  formatPercent(yieldReport.combined_yield_over_successful)],
                ['for ±1 %', requiredSamples(
                  yieldReport.combined_yield_over_successful, 1).toLocaleString()],
                ['for ±0.5 %', requiredSamples(
                  yieldReport.combined_yield_over_successful, 0.5).toLocaleString()],
                ['for ±0.1 %', requiredSamples(
                  yieldReport.combined_yield_over_successful, 0.1).toLocaleString()],
              ]} />
              <p className="mt-2 text-[11px] text-muted">
                This run used {yieldReport.attempted.toLocaleString()} samples. Plain Monte
                Carlo cannot resolve ppm-level yields — that needs importance sampling,
                which is not implemented here.
              </p>
            </Panel>
          </>
        ) : (
          <Panel title="Yield">
            <EmptyState
              title="No specifications declared"
              hint={yieldReport.notes[0] ??
                'Add .spec lines to the netlist so a pass/fail verdict can be computed.'}
            />
          </Panel>
        )
      )}

      <Panel
        title="PVT matrix"
        subtitle={circuitName
          ? `Stored PVT runs for ${circuitName}`
          : 'Stored PVT runs for the selected circuit'}
      >
        {pvtLoading && <Spinner label="Collecting PVT runs…" />}
        {!pvtLoading && !pvtMatrix && (
          <EmptyState
            title="No PVT runs stored for this circuit"
            hint="Run a PVT sweep to populate this matrix:
                  siliconstat pvt --circuit examples/current_mirror.net --samples 150"
          />
        )}
        {pvtMatrix && (
          <>
            <div className="overflow-auto rounded-md border border-line">
              <table className="w-full border-collapse text-[11.5px]">
                <thead>
                  <tr>
                    <th className="border-b border-line bg-panel2 px-2 py-1.5 text-left text-[10px] uppercase tracking-wider text-muted">
                      Corner
                    </th>
                    {pvtMatrix.conditions.map((condition) => (
                      <th key={condition}
                          className="border-b border-line bg-panel2 px-2 py-1.5 text-right text-[10px] uppercase tracking-wider text-muted">
                        {condition}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pvtMatrix.corners.map((corner) => (
                    <tr key={corner}>
                      <td className="border-b border-line/60 px-2 py-1.5 text-ink">{corner}</td>
                      {pvtMatrix.conditions.map((condition) => {
                        const cell = pvtMatrix.lookup.get(`${corner}|${condition}`)
                        const isWorst = cell && pvtMatrix.worst
                          && cell.runId === pvtMatrix.worst.runId
                        return (
                          <td key={condition}
                              className={`num border-b border-line/60 px-2 py-1.5 ${
                                isWorst ? 'ring-1 ring-bad' : ''}`}>
                            {cell?.yieldPct !== null && cell?.yieldPct !== undefined
                              ? <Badge tone={yieldTone(cell.yieldPct)}>
                                  {formatPercent(cell.yieldPct)}
                                </Badge>
                              : <span className="text-muted">—</span>}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {pvtMatrix.worst && (
              <div className="mt-3">
                <SectionTitle>Worst case</SectionTitle>
                <NoteBanner>
                  {pvtMatrix.worst.label} → combined yield{' '}
                  {formatPercent(pvtMatrix.worst.yieldPct ?? NaN)} (run{' '}
                  {pvtMatrix.worst.runId.slice(0, 12)})
                </NoteBanner>
              </div>
            )}
          </>
        )}
      </Panel>
    </div>
  )
}
