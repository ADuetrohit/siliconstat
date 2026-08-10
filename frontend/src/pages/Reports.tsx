import { useEffect, useState } from 'react'

import { getRunAnalysis, getRunNetlist, listRuns, runCsvUrl, runReportUrl } from '@/lib/api'
import { formatDuration, formatTimestamp } from '@/lib/format'
import { useAsync } from '@/lib/hooks'
import {
  Button, CopyButton, EmptyState, ErrorBanner, KeyValueGrid, Panel, Select,
  Spinner,
} from '@/components/ui'
import type { RunAnalysis } from '@/types/api'

export default function Reports() {
  const runs = useAsync(() => listRuns(80), [])
  const [selected, setSelected] = useState('')
  const [showNetlist, setShowNetlist] = useState(false)

  useEffect(() => {
    if (!selected && runs.data?.length) setSelected(runs.data[0].run_id)
  }, [runs.data, selected])

  const analysis = useAsync<RunAnalysis | null>(
    () => (selected ? getRunAnalysis(selected) : Promise.resolve(null)), [selected])

  const netlist = useAsync<string | null>(
    () => (selected && showNetlist ? getRunNetlist(selected) : Promise.resolve(null)),
    [selected, showNetlist])

  const record = analysis.data?.reproduction

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-[18px] font-semibold text-ink">Reports</h1>
        <p className="text-[11.5px] text-muted">
          The standalone engineering report — no external references, no JavaScript,
          openable years later from an archive.
        </p>
      </header>

      <Panel
        title="Run"
        actions={selected && (
          <>
            <a href={runReportUrl(selected)} download={`${selected}_report.html`}>
              <Button>Download report</Button>
            </a>
            <a href={runCsvUrl(selected)} download={`${selected}_samples.csv`}>
              <Button>Download CSV</Button>
            </a>
            <a href={runReportUrl(selected)} target="_blank" rel="noreferrer">
              <Button variant="primary">Open in a tab</Button>
            </a>
          </>
        )}
      >
        {runs.loading && <Spinner />}
        {runs.error && <ErrorBanner message={runs.error} onRetry={runs.reload} />}
        {runs.data && runs.data.length > 0 && (
          <Select
            value={selected}
            onChange={setSelected}
            options={runs.data.map((r) => ({
              value: r.run_id,
              label: `${r.circuit_name} · ${r.variation_mode || 'nominal'} · ` +
                     `${r.pvt_label || 'nominal'} · ${r.samples} samples · ` +
                     `${formatTimestamp(r.started_at)}`,
            }))}
          />
        )}
        {runs.data && !runs.data.length && (
          <EmptyState title="No stored runs"
                      hint="Run a Monte Carlo experiment; every run is stored automatically." />
        )}
      </Panel>

      {record && (
        <Panel
          title="Reproducibility record"
          subtitle="Everything needed to regenerate this run byte for byte"
          actions={<CopyButton text={JSON.stringify(record, null, 2)} label="Copy JSON" />}
        >
          <KeyValueGrid columns={3} items={[
            ['Run id', record.run_id],
            ['Seed', String(record.seed)],
            ['Samples', String(record.samples)],
            ['Sampling', record.sampling],
            ['Variation mode', record.variation?.mode ?? '—'],
            ['PVT condition', record.pvt?.label ?? 'nominal'],
            ['Software version', record.software_version],
            ['Platform', record.platform],
            ['Circuit SHA-256',
              <span className="font-mono text-[10.5px]" key="sha">
                {record.circuit_sha256}
              </span>],
            ['Timestamp', formatTimestamp(record.timestamp)],
            ['Duration', formatDuration(analysis.data?.summary.duration_s)],
            ['Throughput',
              `${(analysis.data?.summary.samples_per_second ?? 0).toFixed(1)} samples/s`],
          ]} />
          <div className="mt-3">
            <Panel title="Solver tolerances">
              <KeyValueGrid columns={4} items={Object.entries(record.solver ?? {})
                .map(([key, value]) => [key, String(value)])} />
            </Panel>
          </div>
        </Panel>
      )}

      {selected && (
        <Panel title="Report preview">
          <iframe
            key={selected}
            src={runReportUrl(selected)}
            title="SiliconStat engineering report"
            sandbox="allow-same-origin"
            className="h-[900px] w-full rounded-md border border-line bg-[#070d16]"
          />
        </Panel>
      )}

      <Panel
        title="Circuit netlist"
        actions={
          <>
            <Button onClick={() => setShowNetlist(!showNetlist)}>
              {showNetlist ? 'Hide' : 'Show'}
            </Button>
            {netlist.data && <CopyButton text={netlist.data} />}
          </>
        }
      >
        {!showNetlist && (
          <p className="text-[11.5px] text-muted">
            The exact netlist stored with this run.
          </p>
        )}
        {showNetlist && netlist.loading && <Spinner />}
        {showNetlist && netlist.error && <ErrorBanner message={netlist.error} />}
        {showNetlist && netlist.data && (
          <pre className="max-h-[420px] overflow-auto rounded-md border border-line bg-panel2 p-3 font-mono text-[11px] leading-[1.5] text-ink/90">
            {netlist.data}
          </pre>
        )}
      </Panel>
    </div>
  )
}
