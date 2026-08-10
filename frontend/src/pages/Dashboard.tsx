import { useNavigate } from 'react-router-dom'

import { getDbStats, getExamples, getHealth, getVersion, listRuns } from '@/lib/api'
import { formatBytes, formatDuration, formatTimestamp } from '@/lib/format'
import { useAsync } from '@/lib/hooks'
import {
  Badge, Button, DataTable, EmptyState, ErrorBanner, Panel, Spinner, StatTile,
  type Column,
} from '@/components/ui'
import type { RunListing } from '@/types/api'

const LAYERS = [
  'circuit physics', 'simulator', 'mismatch model', 'Monte Carlo',
  'statistics', 'yield', 'visualization', 'ML acceleration',
]

export default function Dashboard() {
  const navigate = useNavigate()
  const health = useAsync(getHealth, [])
  const version = useAsync(getVersion, [])
  const stats = useAsync(getDbStats, [])
  const examples = useAsync(getExamples, [])
  const runs = useAsync(() => listRuns(8), [])

  const runColumns: Column<RunListing>[] = [
    { key: 'run', header: 'Run', render: (r) => (
      <span className="text-accent">{r.run_id.slice(0, 12)}</span>) },
    { key: 'circuit', header: 'Circuit', render: (r) => r.circuit_name },
    { key: 'mode', header: 'Mode', render: (r) => (
      <Badge tone="muted">{r.variation_mode || 'nominal'}</Badge>) },
    { key: 'pvt', header: 'PVT', render: (r) => r.pvt_label || 'nominal' },
    { key: 'n', header: 'Samples', align: 'right', render: (r) => r.samples },
    { key: 'ok', header: 'OK', align: 'right', render: (r) => (
      <span className="text-good">{r.counters.successful}</span>) },
    { key: 'fail', header: 'Fail', align: 'right', render: (r) => (
      <span className={r.counters.failed ? 'text-bad' : 'text-muted'}>
        {r.counters.failed}
      </span>) },
    { key: 'dur', header: 'Duration', align: 'right',
      render: (r) => formatDuration(r.duration_s) },
    { key: 'started', header: 'Started', render: (r) => formatTimestamp(r.started_at) },
  ]

  const totalSamples = stats.data?.samples ?? 0

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-[22px] font-semibold tracking-[0.04em] text-ink">SiliconStat</h1>
        <p className="mt-0.5 text-[12px] text-muted">
          Monte Carlo Mismatch Analysis and Statistical Verification Platform for Analog ICs
        </p>
        {health.data && version.data && (
          <p className="mt-1 text-[11px] text-muted/80">
            v{health.data.version} · Python {version.data.environment.python} ·{' '}
            {version.data.environment.system} · {version.data.environment.cpu_count} CPUs ·
            database <span className="text-ink/80">{health.data.database}</span>
          </p>
        )}
      </header>

      {health.error && <ErrorBanner message={health.error} onRetry={health.reload} />}

      <div className="grid gap-3 md:grid-cols-4">
        <StatTile label="Circuits stored" value={stats.data?.circuits ?? '—'} />
        <StatTile label="Runs stored" value={stats.data?.runs ?? '—'} tone="accent" />
        <StatTile
          label="Samples simulated"
          value={totalSamples.toLocaleString()}
          sub={stats.data ? `${formatBytes(stats.data.size_bytes)} on disk` : undefined}
        />
        <StatTile
          label="Active jobs"
          value={health.data?.active_jobs ?? '—'}
          tone={health.data?.active_jobs ? 'warn' : 'default'}
        />
      </div>

      <Panel title="How it works" subtitle="Each layer depends only on the layers beneath it">
        <div className="flex flex-wrap items-center gap-1.5">
          {LAYERS.map((layer, index) => (
            <span key={layer} className="flex items-center gap-1.5">
              <span className="rounded-full border border-line bg-panel2 px-2.5 py-1 text-[11px] text-ink">
                {layer}
              </span>
              {index < LAYERS.length - 1 && <span className="text-accent/60">→</span>}
            </span>
          ))}
        </div>
      </Panel>

      <Panel
        title="Demo circuits"
        subtitle="Shipped netlists, each with declared measurements and specifications"
      >
        {examples.loading && <Spinner />}
        {examples.error && <ErrorBanner message={examples.error} onRetry={examples.reload} />}
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {(examples.data ?? []).map((example) => (
            <div
              key={example.id}
              className="flex flex-col rounded-lg border border-line bg-panel2 p-3"
            >
              <div className="text-[13px] text-ink">{example.name}</div>
              <div className="mt-0.5 font-mono text-[10.5px] text-muted">{example.id}</div>
              {example.error ? (
                <div className="mt-2 text-[11px] text-bad">{example.error}</div>
              ) : (
                <>
                  <div className="mt-2 text-[11px] text-muted">
                    {example.devices} devices · {example.nodes} nodes
                    {example.matched_groups?.length
                      ? ` · groups ${example.matched_groups.join(', ')}`
                      : ''}
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {(example.measurements ?? []).slice(0, 8).map((name) => (
                      <Badge key={name} tone="muted">{name}</Badge>
                    ))}
                  </div>
                  {!!example.specs?.length && (
                    <ul className="mt-2 space-y-0.5 text-[10.5px] text-muted">
                      {example.specs.slice(0, 4).map((spec) => (
                        <li key={spec}>· {spec}</li>
                      ))}
                    </ul>
                  )}
                </>
              )}
              <div className="mt-auto flex gap-2 pt-3">
                <Button onClick={() => navigate(`/circuits?example=${example.id}`)}>
                  Inspect
                </Button>
                <Button
                  variant="primary"
                  onClick={() => navigate(`/monte-carlo?example=${example.id}`)}
                >
                  Run Monte Carlo
                </Button>
              </div>
            </div>
          ))}
        </div>
        {!examples.loading && !examples.error && !(examples.data ?? []).length && (
          <EmptyState title="No example circuits found"
                      hint="Set SILICONSTAT_EXAMPLES to the examples/ directory." />
        )}
      </Panel>

      <Panel
        title="Recent runs"
        actions={<Button onClick={() => navigate('/results')}>All runs</Button>}
      >
        {runs.loading && <Spinner />}
        {runs.error && <ErrorBanner message={runs.error} onRetry={runs.reload} />}
        {runs.data && (
          runs.data.length ? (
            <DataTable
              columns={runColumns}
              rows={runs.data}
              rowKey={(row) => row.run_id}
              onRowClick={(row) => navigate(`/results?run=${row.run_id}`)}
              dense
            />
          ) : (
            <EmptyState
              title="No runs yet"
              hint="Start one from the Monte Carlo page, or with the CLI:
                    siliconstat monte-carlo --circuit examples/current_mirror.net --samples 1000"
              action={<Button variant="primary" onClick={() => navigate('/monte-carlo')}>
                Go to Monte Carlo
              </Button>}
            />
          )
        )}
      </Panel>
    </div>
  )
}
