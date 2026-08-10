import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import { getExamples, getWaveforms, listCircuits, storeCircuit, validateNetlist } from '@/lib/api'
import { formatEng, regionTone } from '@/lib/format'
import { useAsync } from '@/lib/hooks'
import Schematic from '@/components/Schematic'
import { BodePlot, TransientPlot } from '@/components/plots'
import {
  Badge, Button, DataTable, ErrorBanner, KeyValueGrid, Panel, SectionTitle,
  Spinner, type Column,
} from '@/components/ui'
import type {
  CircuitDevice, MosfetOp, SlotMeta, SpecMeta, ValidateResponse, WaveformResponse,
} from '@/types/api'

export default function Circuits() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const examples = useAsync(getExamples, [])
  const stored = useAsync(listCircuits, [])

  const [netlist, setNetlist] = useState('')
  const [name, setName] = useState('')
  const [result, setResult] = useState<ValidateResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [storeMessage, setStoreMessage] = useState<string | null>(null)
  const [waves, setWaves] = useState<WaveformResponse | null>(null)
  const [wavesError, setWavesError] = useState<string | null>(null)
  const [wavesBusy, setWavesBusy] = useState(false)
  const [traceNodes, setTraceNodes] = useState<string[]>([])

  const requested = params.get('example')

  // Load the requested example (or the first one) once the list arrives.
  useEffect(() => {
    if (!examples.data?.length || netlist) return
    const chosen = examples.data.find((e) => e.id === requested) ?? examples.data[0]
    setNetlist(chosen.netlist)
    setName(chosen.name)
  }, [examples.data, requested, netlist])

  const validate = async (text: string, label?: string) => {
    setBusy(true)
    setStoreMessage(null)
    try {
      setResult(await validateNetlist(text, label || undefined))
    } catch (error) {
      setResult({ valid: false, error: (error as Error).message })
    } finally {
      setBusy(false)
    }
  }

  // Validate automatically the first time a netlist is loaded.
  useEffect(() => {
    if (netlist && !result && !busy) void validate(netlist, name)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [netlist])

  // Fetch the raw traces whenever a circuit validates.
  useEffect(() => {
    if (!result?.valid || !netlist) { setWaves(null); return }
    let alive = true
    setWavesBusy(true)
    setWavesError(null)
    getWaveforms({ netlist })
      .then((data) => {
        if (!alive) return
        setWaves(data)
        // Default to the nodes that actually carry signal.  A supply rail is
        // a flat line and a bias node barely moves; showing those first buries
        // the trace the user came to look at.
        const rails = new Set(data.circuit.devices
          .filter((d) => d.type === 'voltagesource' && Math.abs(d.dc ?? 0) > 0)
          .map((d) => d.nodes[0]))
        const candidates = data.circuit.nodes.filter((n) => n !== '0' && !rails.has(n))

        const activity = (node: string): number => {
          if (data.tran?.nodes[node]?.length) {
            const trace = data.tran.nodes[node]
            return Math.max(...trace) - Math.min(...trace)   // peak-to-peak swing
          }
          if (data.ac?.nodes[node]) {
            const mag = data.ac.nodes[node].mag_db
              .filter((v): v is number => v !== null && Number.isFinite(v))
            return mag.length ? Math.max(...mag) : -Infinity  // strongest response
          }
          return 0
        }
        const ranked = [...candidates].sort((a, b) => activity(b) - activity(a))
        // Always keep the stimulus node so the response has a reference.
        const stimulus = data.ac?.reference
        const chosen = ranked.slice(0, 3)
        if (stimulus && candidates.includes(stimulus) && !chosen.includes(stimulus)) {
          chosen.push(stimulus)
        }
        setTraceNodes(chosen.length ? chosen : candidates.slice(0, 3))
      })
      .catch((err: Error) => { if (alive) { setWavesError(err.message); setWaves(null) } })
      .finally(() => { if (alive) setWavesBusy(false) })
    return () => { alive = false }
  }, [result, netlist])

  const circuit = result?.valid ? result.circuit : undefined
  const op = result?.operating_point
  const devices = circuit?.devices ?? []
  const opDevices = op?.devices ?? {}

  const deviceColumns: Column<CircuitDevice>[] = useMemo(() => {
    const base: Column<CircuitDevice>[] = [
      { key: 'name', header: 'Device', render: (d) => (
        <span className={d.name === selectedDevice ? 'text-accent' : ''}>{d.name}</span>) },
      { key: 'type', header: 'Type', render: (d) => d.type },
      { key: 'nodes', header: 'Terminals', render: (d) => (
        <span className="text-muted">{d.nodes.join(' ')}</span>) },
      { key: 'value', header: 'Value / geometry', render: (d) => {
        if (d.type === 'mosfet') {
          return `W=${formatEng(d.w ?? 0, 'm')} L=${formatEng(d.l ?? 0, 'm')}` +
                 (d.m && d.m !== 1 ? ` m=${d.m}` : '')
        }
        if (d.r !== undefined) return formatEng(d.r, 'Ω')
        if (d.c !== undefined) return formatEng(d.c, 'F')
        if (d.l !== undefined) return formatEng(d.l, 'H')
        if (d.dc !== undefined) {
          return formatEng(d.dc, d.type === 'currentsource' ? 'A' : 'V')
        }
        return '—'
      } },
      { key: 'model', header: 'Model', render: (d) => d.model ?? '—' },
      { key: 'group', header: 'Matched', render: (d) => (
        d.matched_group ? <Badge tone="info">{d.matched_group}</Badge> : '—') },
    ]
    if (!op?.converged) return base
    return [...base,
      { key: 'region', header: 'Region', render: (d) => {
        const entry = opDevices[d.name] as MosfetOp | undefined
        return entry?.region
          ? <Badge tone={regionTone(entry.region)}>{entry.region}</Badge> : '—'
      } },
      { key: 'id', header: 'Id', align: 'right', render: (d) => {
        const entry = opDevices[d.name] as MosfetOp | undefined
        return entry?.id !== undefined ? formatEng(entry.id, 'A') : '—'
      } },
      { key: 'vgs', header: 'Vgs', align: 'right', render: (d) => {
        const entry = opDevices[d.name] as MosfetOp | undefined
        return entry?.vgs !== undefined ? entry.vgs.toFixed(4) : '—'
      } },
      { key: 'vov', header: 'Vov', align: 'right', render: (d) => {
        const entry = opDevices[d.name] as MosfetOp | undefined
        return entry?.vov !== undefined ? entry.vov.toFixed(4) : '—'
      } },
      { key: 'gm', header: 'gm', align: 'right', render: (d) => {
        const entry = opDevices[d.name] as MosfetOp | undefined
        return entry?.gm !== undefined ? formatEng(entry.gm, 'S') : '—'
      } },
      { key: 'gmid', header: 'gm/Id', align: 'right', render: (d) => {
        const entry = opDevices[d.name] as MosfetOp | undefined
        return entry?.gm_over_id !== undefined && Number.isFinite(entry.gm_over_id)
          ? entry.gm_over_id.toFixed(2) : '—'
      } },
    ]
  }, [op, opDevices, selectedDevice])

  const slotColumns: Column<SlotMeta>[] = [
    { key: 'slot', header: 'Random variable', render: (s) => s.slot },
    { key: 'param', header: 'Parameter', render: (s) => s.parameter },
    { key: 'scope', header: 'Scope', render: (s) => (
      <Badge tone={s.scope === 'global' ? 'info' : 'pass'}>{s.scope}</Badge>) },
    { key: 'dist', header: 'Distribution', render: (s) => s.distribution },
    { key: 'sigma', header: 'Sigma', align: 'right',
      render: (s) => `${s.sigma.toPrecision(5)} ${s.unit}` },
    { key: 'devices', header: 'Devices', render: (s) => s.devices },
  ]

  const specColumns: Column<SpecMeta>[] = [
    { key: 'desc', header: 'Specification', render: (s) => s.description },
    { key: 'measure', header: 'Measurement', render: (s) => s.measure },
    { key: 'label', header: 'Label', render: (s) => s.label || '—' },
  ]

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-[18px] font-semibold text-ink">Circuits</h1>
        <p className="text-[11.5px] text-muted">
          Edit a netlist, validate it, and inspect the elaborated topology and operating point.
        </p>
      </header>

      <div className="grid gap-4 xl:grid-cols-[300px_1fr]">
        <div className="space-y-4">
          <Panel title="Example circuits">
            {examples.loading && <Spinner />}
            {examples.error && <ErrorBanner message={examples.error} />}
            <div className="space-y-1">
              {(examples.data ?? []).map((example) => (
                <button
                  key={example.id}
                  onClick={() => {
                    setNetlist(example.netlist)
                    setName(example.name)
                    setResult(null)
                    setSelectedDevice(null)
                    setParams({ example: example.id })
                    void validate(example.netlist, example.name)
                  }}
                  className="block w-full rounded-md border border-line bg-panel2 px-3 py-2 text-left transition-colors hover:border-accent/50"
                >
                  <div className="text-[12px] text-ink">{example.name}</div>
                  <div className="font-mono text-[10px] text-muted">{example.id}</div>
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="Stored circuits">
            {stored.loading && <Spinner />}
            {!stored.loading && !(stored.data ?? []).length && (
              <p className="text-[11.5px] text-muted">Nothing stored yet.</p>
            )}
            <div className="space-y-1">
              {(stored.data ?? []).map((entry) => (
                <div key={entry.id}
                     className="rounded-md border border-line bg-panel2 px-3 py-1.5">
                  <div className="text-[12px] text-ink">{entry.name}</div>
                  <div className="font-mono text-[10px] text-muted">
                    #{entry.id} · {entry.sha256.slice(0, 12)}…
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel
            title="Netlist"
            actions={
              <>
                <Button onClick={() => void validate(netlist, name)} disabled={busy}>
                  {busy ? 'Validating…' : 'Validate'}
                </Button>
                <Button
                  disabled={!result?.valid || busy}
                  onClick={async () => {
                    try {
                      const saved = await storeCircuit(netlist, name || undefined)
                      setStoreMessage(`Stored as circuit #${saved.circuit_id}`)
                      stored.reload()
                    } catch (error) {
                      setStoreMessage((error as Error).message)
                    }
                  }}
                >
                  Store
                </Button>
                <Button
                  variant="primary"
                  disabled={!result?.valid}
                  onClick={() => navigate(`/monte-carlo?example=${requested ?? ''}`)}
                >
                  Run Monte Carlo
                </Button>
              </>
            }
          >
            <textarea
              value={netlist}
              spellCheck={false}
              onChange={(e) => setNetlist(e.target.value)}
              rows={16}
              className="w-full resize-y font-mono text-[11.5px] leading-[1.5]"
            />
            {storeMessage && (
              <p className="mt-2 text-[11.5px] text-accent">{storeMessage}</p>
            )}
          </Panel>

          {result && !result.valid && (
            <ErrorBanner
              title={result.error_type || 'Netlist error'}
              message={result.error ?? 'unknown parse failure'}
            />
          )}

          {circuit && (
            <>
              <Panel
                title="Topology"
                subtitle={`${circuit.devices.length} devices · ${circuit.n_nodes} nodes · ${circuit.n_branches} branch unknowns`}
              >
                <Schematic
                  circuit={circuit}
                  operatingPoint={op?.converged ? {
                    node_voltages: op.node_voltages ?? {},
                    devices: op.devices ?? {},
                    iterations: op.iterations ?? 0,
                    strategy: op.strategy ?? '',
                    residual: op.residual ?? 0,
                    total_supply_power: op.total_supply_power ?? 0,
                  } : null}
                  selected={selectedDevice}
                  onSelect={setSelectedDevice}
                />
              </Panel>

              <Panel
                title="Simulation output"
                subtitle={
                  waves?.tran && waves?.ac ? 'transient waveform and AC response'
                  : waves?.tran ? 'transient waveform'
                  : waves?.ac ? 'AC response (Bode)'
                  : 'raw traces from the declared .ac / .tran analyses'
                }
                actions={
                  waves && (waves.ac || waves.tran) ? (
                    <div className="flex flex-wrap items-center gap-1.5">
                      {circuit.nodes.filter((n) => n !== '0').map((node) => (
                        <button
                          key={node}
                          onClick={() => setTraceNodes((current) =>
                            current.includes(node)
                              ? current.filter((n) => n !== node)
                              : [...current, node])}
                          className={`rounded-md border px-2 py-0.5 text-[11px] transition-colors ${
                            traceNodes.includes(node)
                              ? 'border-accent/60 bg-accent/15 text-accent'
                              : 'border-line bg-panel2 text-muted hover:text-ink'}`}
                        >
                          {node}
                        </button>
                      ))}
                    </div>
                  ) : undefined
                }
              >
                {wavesBusy && <Spinner label="Simulating…" />}
                {wavesError && <ErrorBanner message={wavesError} />}
                {waves?.tran && (
                  <div className="mb-3">
                    <SectionTitle
                      hint={`${waves.tran.points} points · ${waves.tran.integration === 'trap'
                        ? 'trapezoidal' : 'backward Euler'}`}
                    >
                      Transient
                    </SectionTitle>
                    <TransientPlot tran={waves.tran} nodes={traceNodes} />
                  </div>
                )}
                {waves?.ac && (
                  <div>
                    <SectionTitle
                      hint={`${waves.ac.points} points · referred to ${waves.ac.stimulus} at ${waves.ac.reference}`}
                    >
                      AC response
                    </SectionTitle>
                    <BodePlot ac={waves.ac} nodes={traceNodes} />
                  </div>
                )}
                {waves && !waves.ac && !waves.tran && !wavesBusy && (
                  <p className="text-[11.5px] text-muted">
                    {waves.notes[0] ?? 'No waveform declared for this circuit.'}
                    {' '}Add a <code className="text-ink">.tran</code> or{' '}
                    <code className="text-ink">.ac</code> card to see one.
                  </p>
                )}
                {/* The empty state above already shows notes[0]; listing every
                    note unconditionally printed that first one twice. */}
                {waves?.notes.slice(waves.ac || waves.tran ? 0 : 1).map((note) => (
                  <p key={note} className="mt-2 text-[11px] text-muted">{note}</p>
                ))}
              </Panel>

              <Panel title="Operating point">
                {op?.converged ? (
                  <>
                    <KeyValueGrid columns={4} items={[
                      ['Newton iterations', String(op.iterations)],
                      ['Strategy', op.strategy ?? '—'],
                      ['KCL residual', `${(op.residual ?? 0).toExponential(3)} A`],
                      ['Supply power', formatEng(op.total_supply_power ?? 0, 'W')],
                    ]} />
                    <div className="mt-3">
                      <SectionTitle>Node voltages</SectionTitle>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(op.node_voltages ?? {})
                          .filter(([node]) => node !== '0')
                          .map(([node, voltage]) => (
                            <span key={node}
                                  className="rounded-md border border-line bg-panel2 px-2 py-1 text-[11px]">
                              <span className="text-muted">{node}</span>{' '}
                              <span className="text-ink">{formatEng(voltage, 'V')}</span>
                            </span>
                          ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <ErrorBanner
                    title="Operating point did not converge"
                    message={op?.error ?? 'unknown solver failure'}
                  />
                )}
              </Panel>

              <Panel title="Devices">
                <DataTable
                  columns={deviceColumns}
                  rows={devices}
                  rowKey={(d) => d.name}
                  selectedKey={selectedDevice ?? undefined}
                  onRowClick={(d) =>
                    setSelectedDevice(selectedDevice === d.name ? null : d.name)}
                  dense
                />
              </Panel>

              <div className="grid gap-4 xl:grid-cols-2">
                <Panel title="Measurements">
                  <DataTable
                    columns={[
                      { key: 'name', header: 'Name', render: (m) => m.name },
                      { key: 'kind', header: 'Kind', render: (m) => m.kind },
                      { key: 'unit', header: 'Unit', render: (m) => m.unit || '—' },
                      { key: 'def', header: 'Definition', render: (m) =>
                        (m.args as Record<string, unknown>).expression as string ??
                        Object.entries(m.args).map(([k, v]) => `${k}=${v}`).join(' ') },
                    ]}
                    rows={circuit.measures}
                    rowKey={(m) => m.name}
                    dense
                  />
                </Panel>

                <Panel title="Specifications">
                  <DataTable columns={specColumns} rows={circuit.specs}
                             rowKey={(s, i) => `${s.measure}-${i}`} dense
                             emptyMessage="No specifications declared; no yield can be computed." />
                </Panel>
              </div>

              <Panel
                title="Default variation model"
                subtitle="What a Monte Carlo run would perturb, with each random variable's sigma"
              >
                {result?.default_variation?.error ? (
                  <ErrorBanner message={result.default_variation.error} />
                ) : (
                  <DataTable
                    columns={slotColumns}
                    rows={result?.default_variation?.slots ?? []}
                    rowKey={(s) => s.slot}
                    dense
                  />
                )}
              </Panel>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
