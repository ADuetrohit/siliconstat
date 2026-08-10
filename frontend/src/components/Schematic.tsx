/**
 * Circuit visualisation.
 *
 * Drawn from the elaborated circuit description, so it always shows the real
 * topology.  The layout is a deterministic node-rail ladder: every node is a
 * horizontal rail (supplies at the top, ground at the bottom, internal nodes
 * ordered by their operating-point voltage when one is available), and every
 * device occupies a column whose terminals drop onto the rails they connect
 * to.  It is not a hand-drawn schematic, but it is readable, labelled and
 * faithful.
 */

import { useMemo, useState } from 'react'

import type { CircuitDescription, CircuitDevice, MosfetOp, OperatingPoint } from '@/types/api'
import { formatEng } from '@/lib/format'

const COLOURS = {
  rail: '#1e2b41',
  railSupply: '#f59e0b',
  railGround: '#8ba0ba',
  wire: '#3a4d69',
  ink: '#dbe6f3',
  muted: '#8ba0ba',
  glyph: '#22d3ee',
  glyphP: '#a78bfa',
  passive: '#3b82f6',
  source: '#22c55e',
  selected: '#22d3ee',
}

const ROW_H = 46
const COL_W = 92
const PAD_LEFT = 92
const PAD_TOP = 30
const PAD_BOTTOM = 34

interface Terminal { name: string; label: string; dx: number }

function terminalsFor(device: CircuitDevice): Terminal[] {
  const t = device.type
  if (t === 'mosfet') {
    return [
      { name: device.nodes[0], label: 'D', dx: 0 },
      { name: device.nodes[1], label: 'G', dx: -26 },
      { name: device.nodes[2], label: 'S', dx: 0 },
      { name: device.nodes[3], label: 'B', dx: 26 },
    ]
  }
  return device.nodes.slice(0, 2).map((node, index) => ({
    name: node, label: index === 0 ? '+' : '-', dx: 0,
  }))
}

function deviceLabel(device: CircuitDevice): string {
  switch (device.type) {
    case 'resistor': return formatEng(device.r ?? 0, 'Ω')
    case 'capacitor': return formatEng(device.c ?? 0, 'F')
    case 'inductor': return formatEng(device.l ?? 0, 'H')
    case 'voltagesource': return formatEng(device.dc ?? 0, 'V')
    case 'currentsource': return formatEng(device.dc ?? 0, 'A')
    case 'mosfet': return `${formatEng(device.w ?? 0, 'm')}/${formatEng(device.l ?? 0, 'm')}`
    case 'diode': return device.model ?? ''
    default: return ''
  }
}

function Glyph({ device, x, y, selected, mtype }: {
  device: CircuitDevice
  x: number
  y: number
  selected: boolean
  mtype?: string
}) {
  const stroke = selected ? COLOURS.selected
    : device.type === 'mosfet' ? (mtype === 'pmos' ? COLOURS.glyphP : COLOURS.glyph)
    : device.type.endsWith('source') ? COLOURS.source
    : COLOURS.passive
  const width = selected ? 2.1 : 1.5

  switch (device.type) {
    case 'resistor':
      return (
        <polyline
          points={`${x},${y - 14} ${x - 7},${y - 10} ${x + 7},${y - 4} ${x - 7},${y + 2} ${x + 7},${y + 8} ${x},${y + 14}`}
          fill="none" stroke={stroke} strokeWidth={width} strokeLinejoin="round"
        />
      )
    case 'capacitor':
      return (
        <g stroke={stroke} strokeWidth={width} fill="none">
          <line x1={x} y1={y - 14} x2={x} y2={y - 4} />
          <line x1={x - 10} y1={y - 4} x2={x + 10} y2={y - 4} />
          <line x1={x - 10} y1={y + 3} x2={x + 10} y2={y + 3} />
          <line x1={x} y1={y + 3} x2={x} y2={y + 14} />
        </g>
      )
    case 'inductor':
      return (
        <g stroke={stroke} strokeWidth={width} fill="none">
          <line x1={x} y1={y - 14} x2={x} y2={y - 9} />
          {[-6, 0, 6].map((dy) => (
            <circle key={dy} cx={x + 4} cy={y + dy} r={4.5} />
          ))}
          <line x1={x} y1={y + 10} x2={x} y2={y + 14} />
        </g>
      )
    case 'voltagesource':
      return (
        <g stroke={stroke} strokeWidth={width} fill="none">
          <circle cx={x} cy={y} r={11} />
          <text x={x} y={y - 2} fontSize="9" fill={stroke} textAnchor="middle"
                stroke="none">+</text>
          <text x={x} y={y + 8} fontSize="9" fill={stroke} textAnchor="middle"
                stroke="none">−</text>
        </g>
      )
    case 'currentsource':
      return (
        <g stroke={stroke} strokeWidth={width} fill="none">
          <circle cx={x} cy={y} r={11} />
          <line x1={x} y1={y + 6} x2={x} y2={y - 6} />
          <polyline points={`${x - 3.5},${y - 2} ${x},${y - 7} ${x + 3.5},${y - 2}`} />
        </g>
      )
    case 'diode':
      return (
        <g stroke={stroke} strokeWidth={width} fill="none">
          <line x1={x} y1={y - 14} x2={x} y2={y - 7} />
          <polygon points={`${x - 8},${y - 7} ${x + 8},${y - 7} ${x},${y + 4}`}
                   fill={stroke} fillOpacity={0.28} />
          <line x1={x - 8} y1={y + 4} x2={x + 8} y2={y + 4} />
          <line x1={x} y1={y + 4} x2={x} y2={y + 14} />
        </g>
      )
    case 'mosfet': {
      const p = mtype === 'pmos'
      return (
        <g stroke={stroke} strokeWidth={width} fill="none">
          {/* channel */}
          <line x1={x - 7} y1={y - 13} x2={x - 7} y2={y + 13} />
          {/* gate plate */}
          <line x1={x - 13} y1={y - 9} x2={x - 13} y2={y + 9} />
          {/* drain / source stubs */}
          <line x1={x - 7} y1={y - 11} x2={x} y2={y - 11} />
          <line x1={x} y1={y - 11} x2={x} y2={y - 15} />
          <line x1={x - 7} y1={y + 11} x2={x} y2={y + 11} />
          <line x1={x} y1={y + 11} x2={x} y2={y + 15} />
          {/* bulk */}
          <line x1={x - 7} y1={y} x2={x + 14} y2={y} />
          {/* polarity arrow on the bulk connection */}
          {p
            ? <polyline points={`${x + 2},${y - 3.5} ${x - 3},${y} ${x + 2},${y + 3.5}`} />
            : <polyline points={`${x - 3},${y - 3.5} ${x + 2},${y} ${x - 3},${y + 3.5}`} />}
        </g>
      )
    }
    default:
      return <circle cx={x} cy={y} r={9} stroke={stroke} strokeWidth={width} fill="none" />
  }
}

export function Schematic({ circuit, operatingPoint, onSelect, selected }: {
  circuit: CircuitDescription
  operatingPoint?: OperatingPoint | null
  onSelect?: (name: string | null) => void
  selected?: string | null
}) {
  const [hover, setHover] = useState<string | null>(null)

  const { rails, width, height } = useMemo(() => {
    const voltages = operatingPoint?.node_voltages ?? {}
    const supplies = new Set(
      circuit.devices
        .filter((d) => d.type === 'voltagesource' && Math.abs(d.dc ?? 0) > 0)
        .map((d) => d.nodes[0]),
    )
    const ordered = [...circuit.nodes]
      .filter((n) => n !== '0')
      .sort((a, b) => {
        const sa = supplies.has(a) ? 1 : 0
        const sb = supplies.has(b) ? 1 : 0
        if (sa !== sb) return sb - sa                       // supplies on top
        const va = voltages[a]
        const vb = voltages[b]
        if (Number.isFinite(va) && Number.isFinite(vb) && va !== vb) return vb - va
        return a.localeCompare(b)
      })
    ordered.push('0')

    const map = new Map<string, { y: number; supply: boolean; ground: boolean }>()
    ordered.forEach((name, index) => {
      map.set(name, {
        y: PAD_TOP + index * ROW_H,
        supply: supplies.has(name),
        ground: name === '0',
      })
    })
    return {
      rails: map,
      width: PAD_LEFT + circuit.devices.length * COL_W + 40,
      height: PAD_TOP + ordered.length * ROW_H + PAD_BOTTOM,
    }
  }, [circuit, operatingPoint])

  const active = hover ?? selected ?? null
  const activeDevice = circuit.devices.find((d) => d.name === active) ?? null
  const activeOp = active ? (operatingPoint?.devices?.[active] as MosfetOp | undefined) : undefined

  return (
    <div className="space-y-2">
      <div className="overflow-auto rounded-md border border-line bg-panel2">
        <svg width={width} height={height} className="block" role="img"
             aria-label={`Schematic of ${circuit.name}`}>
          {/* node rails */}
          {[...rails.entries()].map(([name, rail]) => (
            <g key={name}>
              <line
                x1={PAD_LEFT - 34} y1={rail.y} x2={width - 16} y2={rail.y}
                stroke={rail.supply ? COLOURS.railSupply
                  : rail.ground ? COLOURS.railGround : COLOURS.rail}
                strokeWidth={rail.supply || rail.ground ? 1.8 : 1.2}
                strokeDasharray={rail.supply || rail.ground ? undefined : '3 4'}
              />
              <text
                x={PAD_LEFT - 40} y={rail.y + 3.5} textAnchor="end" fontSize="10.5"
                fill={rail.supply ? COLOURS.railSupply
                  : rail.ground ? COLOURS.railGround : COLOURS.muted}
              >
                {rail.ground ? 'gnd' : name}
              </text>
              {operatingPoint && Number.isFinite(operatingPoint.node_voltages[name]) && (
                <text x={width - 18} y={rail.y - 5} textAnchor="end" fontSize="9.5"
                      fill={COLOURS.muted}>
                  {formatEng(operatingPoint.node_voltages[name], 'V')}
                </text>
              )}
            </g>
          ))}

          {/* devices */}
          {circuit.devices.map((device, index) => {
            const x = PAD_LEFT + index * COL_W + COL_W / 2
            const terminals = terminalsFor(device)
            const ys = terminals
              .map((t) => rails.get(t.name)?.y)
              .filter((y): y is number => y !== undefined)
            if (!ys.length) return null
            const mid = (Math.min(...ys) + Math.max(...ys)) / 2
            const isActive = active === device.name
            const mtype = (operatingPoint?.devices?.[device.name] as MosfetOp | undefined)?.mtype
              ?? (device.model && circuit.mos_models[device.model]?.type)

            return (
              <g
                key={device.name}
                onMouseEnter={() => setHover(device.name)}
                onMouseLeave={() => setHover(null)}
                onClick={() => onSelect?.(selected === device.name ? null : device.name)}
                style={{ cursor: onSelect ? 'pointer' : 'default' }}
              >
                {isActive && (
                  <rect
                    x={x - COL_W / 2 + 6} y={Math.min(...ys) - 16}
                    width={COL_W - 12} height={Math.max(...ys) - Math.min(...ys) + 32}
                    rx={6} fill={COLOURS.selected} fillOpacity={0.07}
                    stroke={COLOURS.selected} strokeOpacity={0.35}
                  />
                )}
                {terminals.map((terminal, ti) => {
                  const rail = rails.get(terminal.name)
                  if (!rail) return null
                  const tx = x + terminal.dx
                  return (
                    <g key={`${terminal.label}-${ti}`}>
                      <polyline
                        points={`${tx},${rail.y} ${tx},${mid} ${x},${mid}`}
                        fill="none" stroke={COLOURS.wire} strokeWidth={1.2}
                      />
                      <circle cx={tx} cy={rail.y} r={2.6} fill={COLOURS.wire} />
                      {device.type === 'mosfet' && (
                        <text x={tx + (terminal.dx < 0 ? -5 : 5)} y={rail.y - 5}
                              fontSize="8.5" fill={COLOURS.muted}
                              textAnchor={terminal.dx < 0 ? 'end' : 'start'}>
                          {terminal.label}
                        </text>
                      )}
                    </g>
                  )
                })}
                <Glyph device={device} x={x} y={mid} selected={isActive} mtype={mtype} />
                <text x={x} y={mid - 22} textAnchor="middle" fontSize="10.5"
                      fill={isActive ? COLOURS.selected : COLOURS.ink}>
                  {device.name}
                </text>
                <text x={x} y={mid + 30} textAnchor="middle" fontSize="9"
                      fill={COLOURS.muted}>
                  {deviceLabel(device)}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      {activeDevice && (
        <div className="rounded-md border border-line bg-panel2 px-3 py-2 text-[11.5px]">
          <span className="text-accent">{activeDevice.name}</span>
          <span className="text-muted"> · {activeDevice.type}</span>
          <span className="text-muted"> · {activeDevice.nodes.join(' ')}</span>
          {activeDevice.matched_group && (
            <span className="text-violet"> · group {activeDevice.matched_group}</span>
          )}
          {activeOp?.region && (
            <>
              <span className="text-muted"> · </span>
              <span className={activeOp.region === 'saturation' ? 'text-good' : 'text-warn'}>
                {activeOp.region}
              </span>
              <span className="text-muted">
                {' '}· Id {formatEng(activeOp.id, 'A')} · Vgs {activeOp.vgs.toFixed(4)} V
                {' '}· Vov {activeOp.vov.toFixed(4)} V · gm {formatEng(activeOp.gm, 'S')}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default Schematic
