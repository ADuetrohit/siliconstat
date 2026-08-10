/**
 * Plotly wrappers.
 *
 * Every chart consumes the exact payload shapes the API returns (see
 * docs/api_contract.md) and applies one shared dark layout, so the browser
 * charts and the SVG charts in the HTML report read as the same system.
 */

import Plotly from 'plotly.js-cartesian-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import { useMemo } from 'react'

import type {
  ChartBundle, ConvergenceTrace, SensitivityEntry, SpecMeta, Statistics,
} from '@/types/api'

const Plot = createPlotlyComponent(Plotly)

export const THEME = {
  bg: '#111c2e',
  panel2: '#0d1522',
  grid: '#1e2b41',
  axis: '#3a4d69',
  ink: '#dbe6f3',
  muted: '#8ba0ba',
  accent: '#22d3ee',
  accent2: '#3b82f6',
  good: '#22c55e',
  bad: '#ef4444',
  warn: '#f59e0b',
  violet: '#a78bfa',
}

const FONT = {
  family: "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace",
  size: 11,
  color: THEME.muted,
}

function baseLayout(overrides: Record<string, any> = {}): Record<string, any> {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: FONT,
    margin: { l: 62, r: 18, t: 26, b: 46 },
    showlegend: false,
    hovermode: 'closest',
    hoverlabel: {
      bgcolor: THEME.panel2,
      bordercolor: THEME.axis,
      font: { ...FONT, color: THEME.ink },
    },
    ...overrides,
    xaxis: {
      gridcolor: THEME.grid, zerolinecolor: THEME.axis, linecolor: THEME.axis,
      tickfont: FONT, automargin: true, ...(overrides.xaxis ?? {}),
    },
    yaxis: {
      gridcolor: THEME.grid, zerolinecolor: THEME.axis, linecolor: THEME.axis,
      tickfont: FONT, automargin: true, ...(overrides.yaxis ?? {}),
    },
  }
}

const CONFIG = { displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'] }

function Chart({ data, layout, height = 320 }: {
  data: any[]
  layout: Record<string, any>
  height?: number
}) {
  return (
    <Plot
      data={data}
      layout={{ ...layout, height }}
      config={CONFIG}
      style={{ width: '100%', height }}
      useResizeHandler
    />
  )
}

function specShapes(specs: SpecMeta[], orientation: 'v' | 'h' = 'v') {
  return specs
    .filter((spec) => Number.isFinite(spec.value))
    .map((spec) => ({
      type: 'line',
      ...(orientation === 'v'
        ? { x0: spec.value, x1: spec.value, yref: 'paper', y0: 0, y1: 1 }
        : { y0: spec.value, y1: spec.value, xref: 'paper', x0: 0, x1: 1 }),
      line: { color: THEME.bad, width: 2 },
    }))
}

function specAnnotations(specs: SpecMeta[]) {
  return specs
    .filter((spec) => Number.isFinite(spec.value))
    .map((spec, index) => ({
      x: spec.value, yref: 'paper', y: 1 - index * 0.09,
      text: `${spec.op} ${spec.value}`,
      showarrow: false, xanchor: 'left', xshift: 5,
      font: { ...FONT, color: THEME.bad, size: 10 },
    }))
}

// ---------------------------------------------------------------------------
// histogram
// ---------------------------------------------------------------------------

export function Histogram({ chart, stats, height = 340 }: {
  chart: ChartBundle
  stats: Statistics
  height?: number
}) {
  const { counts, centres, bin_width: width } = chart.histogram
  const markers = useMemo(() => {
    const out: { x: number; label: string; colour: string; dash?: string }[] = []
    if (Number.isFinite(stats.mean)) {
      out.push({ x: stats.mean, label: 'mean', colour: THEME.accent })
    }
    if (Number.isFinite(stats.median)) {
      out.push({ x: stats.median, label: 'median', colour: THEME.violet, dash: 'dash' })
    }
    if (Number.isFinite(stats.std) && stats.std > 0) {
      for (const k of [-3, -2, -1, 1, 2, 3]) {
        out.push({
          x: stats.mean + k * stats.std,
          label: `${k > 0 ? '+' : ''}${k}σ`,
          colour: THEME.muted, dash: 'dot',
        })
      }
    }
    if (chart.nominal !== null && Number.isFinite(chart.nominal)) {
      out.push({ x: chart.nominal!, label: 'nominal', colour: THEME.warn, dash: 'dashdot' })
    }
    return out
  }, [stats, chart.nominal])

  const layout = baseLayout({
    xaxis: { title: { text: chart.unit ? `value [${chart.unit}]` : 'value', font: FONT } },
    yaxis: { title: { text: 'count', font: FONT } },
    bargap: 0.02,
    shapes: [
      ...markers.map((m) => ({
        type: 'line', x0: m.x, x1: m.x, yref: 'paper', y0: 0, y1: 1,
        line: { color: m.colour, width: m.dash ? 1.2 : 2, dash: m.dash },
      })),
      ...specShapes(chart.specs),
    ],
    annotations: [
      ...markers
        .filter((m) => !m.dash || m.label.includes('σ') === false)
        .map((m, index) => ({
          x: m.x, yref: 'paper', y: 1 - index * 0.085, text: m.label,
          showarrow: false, xanchor: 'left', xshift: 4,
          font: { ...FONT, color: m.colour, size: 10 },
        })),
      ...specAnnotations(chart.specs),
    ],
  })

  return (
    <Chart
      height={height}
      data={[{
        type: 'bar', x: centres, y: counts, width,
        marker: { color: THEME.accent2, opacity: 0.62,
          line: { color: THEME.accent, width: 0.6 } },
        hovertemplate: `%{x:.6g} ${chart.unit}<br>%{y} samples<extra></extra>`,
      }]}
      layout={layout}
    />
  )
}

// ---------------------------------------------------------------------------
// CDF
// ---------------------------------------------------------------------------

export function CdfPlot({ chart, height = 300 }: { chart: ChartBundle; height?: number }) {
  const layout = baseLayout({
    xaxis: { title: { text: chart.unit ? `value [${chart.unit}]` : 'value', font: FONT } },
    yaxis: { title: { text: 'P(X ≤ x)', font: FONT }, range: [0, 1] },
    shapes: [
      ...specShapes(chart.specs),
      { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5,
        line: { color: THEME.muted, width: 1, dash: 'dot' } },
    ],
    annotations: specAnnotations(chart.specs),
  })
  return (
    <Chart
      height={height}
      data={[{
        type: 'scatter', mode: 'lines', x: chart.cdf.x, y: chart.cdf.p,
        line: { color: THEME.accent, width: 2 },
        hovertemplate: `%{x:.6g} ${chart.unit}<br>P = %{y:.4f}<extra></extra>`,
      }]}
      layout={layout}
    />
  )
}

export function SigmaPlot({ chart, height = 300 }: { chart: ChartBundle; height?: number }) {
  const layout = baseLayout({
    xaxis: { title: { text: chart.unit ? `value [${chart.unit}]` : 'value', font: FONT } },
    yaxis: { title: { text: 'sigma', font: FONT } },
  })
  return (
    <Chart
      height={height}
      data={[{
        type: 'scatter', mode: 'lines', x: chart.sigma_plot.x, y: chart.sigma_plot.sigma,
        line: { color: THEME.accent, width: 1.6 },
        hovertemplate: '%{x:.6g}<br>%{y:.3f} σ<extra></extra>',
      }]}
      layout={layout}
    />
  )
}

// ---------------------------------------------------------------------------
// heatmap
// ---------------------------------------------------------------------------

export function CorrelationHeatmap({ matrix, rows, columns, title, height }: {
  matrix: number[][]
  rows: string[]
  columns: string[]
  title?: string
  height?: number
}) {
  if (!matrix.length || !columns.length) {
    return <div className="py-6 text-center text-[12px] text-muted">No correlation data.</div>
  }
  const computed = height ?? Math.max(240, 40 + rows.length * 26)
  const text = matrix.map((row) =>
    row.map((v) => (Number.isFinite(v) ? v.toFixed(2) : 'n/a')))
  const layout = baseLayout({
    margin: { l: 150, r: 20, t: title ? 34 : 16, b: 110 },
    title: title ? { text: title, font: { ...FONT, color: THEME.ink, size: 12 } } : undefined,
    xaxis: { tickangle: -45, gridcolor: 'rgba(0,0,0,0)' },
    yaxis: { autorange: 'reversed', gridcolor: 'rgba(0,0,0,0)' },
  })
  return (
    <Chart
      height={computed}
      data={[{
        type: 'heatmap', z: matrix, x: columns, y: rows,
        zmin: -1, zmax: 1,
        colorscale: [
          [0, THEME.bad], [0.5, THEME.panel2], [1, THEME.accent2],
        ],
        text, texttemplate: '%{text}',
        textfont: { ...FONT, size: 9.5 },
        hovertemplate: '%{y} × %{x}<br>r = %{z:.4f}<extra></extra>',
        colorbar: { tickfont: FONT, outlinecolor: THEME.axis, thickness: 10 },
      }]}
      layout={layout}
    />
  )
}

// ---------------------------------------------------------------------------
// sensitivity
// ---------------------------------------------------------------------------

export function SensitivityBars({ entries, height }: {
  entries: SensitivityEntry[]
  height?: number
}) {
  const top = entries.slice(0, 14).filter((e) => Number.isFinite(e.variance_contribution_pct))
  if (!top.length) {
    return <div className="py-6 text-center text-[12px] text-muted">No ranking available.</div>
  }
  const ordered = [...top].reverse()
  const layout = baseLayout({
    margin: { l: 165, r: 60, t: 16, b: 40 },
    xaxis: { title: { text: 'variance contribution [%]', font: FONT } },
    yaxis: { gridcolor: 'rgba(0,0,0,0)' },
  })
  return (
    <Chart
      height={height ?? Math.max(220, 40 + ordered.length * 24)}
      data={[{
        type: 'bar', orientation: 'h',
        x: ordered.map((e) => e.variance_contribution_pct),
        y: ordered.map((e) => e.parameter),
        marker: {
          color: ordered.map((e) => (e.beta_standardised >= 0 ? THEME.accent : THEME.bad)),
          opacity: 0.78,
        },
        text: ordered.map((e) => `${e.variance_contribution_pct.toFixed(2)}%`),
        textposition: 'outside',
        textfont: { ...FONT, color: THEME.ink },
        hovertemplate: '%{y}<br>%{x:.3f}% of variance<extra></extra>',
      }]}
      layout={layout}
    />
  )
}

// ---------------------------------------------------------------------------
// convergence
// ---------------------------------------------------------------------------

export function ConvergencePlot({ trace, mode = 'mean', height = 300 }: {
  trace: ConvergenceTrace
  mode?: 'mean' | 'yield'
  height?: number
}) {
  const isYield = mode === 'yield'
  const centre = isYield ? trace.yield_pct : trace.mean
  const low = isYield ? trace.yield_ci_low : trace.mean_ci_low
  const high = isYield ? trace.yield_ci_high : trace.mean_ci_high
  if (!centre?.length) {
    return <div className="py-6 text-center text-[12px] text-muted">No trace available.</div>
  }
  const colour = isYield ? THEME.good : THEME.accent
  const settled = isYield ? trace.yield_settled_at : trace.mean_settled_at
  const finalValue = isYield ? trace.final_yield_pct : trace.final_mean

  const layout = baseLayout({
    xaxis: { title: { text: 'samples', font: FONT } },
    yaxis: {
      title: {
        text: isYield ? 'combined yield [%]'
          : `mean${trace.unit ? ` [${trace.unit}]` : ''}`,
        font: FONT,
      },
    },
    shapes: [
      ...(Number.isFinite(finalValue)
        ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: finalValue, y1: finalValue,
             line: { color: THEME.warn, width: 1.2, dash: 'dash' } }]
        : []),
      ...(settled
        ? [{ type: 'line', x0: settled, x1: settled, yref: 'paper', y0: 0, y1: 1,
             line: { color: THEME.good, width: 1.4, dash: 'dot' } }]
        : []),
    ],
    annotations: settled
      ? [{ x: settled, yref: 'paper', y: 0.06, text: `settled at n=${settled}`,
           showarrow: false, xanchor: 'left', xshift: 5,
           font: { ...FONT, color: THEME.good, size: 10 } }]
      : [],
  })

  const band = low?.length && high?.length
    ? [{
        type: 'scatter', mode: 'lines',
        x: [...trace.n, ...[...trace.n].reverse()],
        y: [...high, ...[...low].reverse()],
        fill: 'toself', fillcolor: `${colour}22`,
        line: { width: 0 }, hoverinfo: 'skip',
      }]
    : []

  return (
    <Chart
      height={height}
      data={[
        ...band,
        { type: 'scatter', mode: 'lines', x: trace.n, y: centre,
          line: { color: colour, width: 2 },
          hovertemplate: 'n = %{x}<br>%{y:.5g}<extra></extra>' },
      ]}
      layout={layout}
    />
  )
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------

export function OverlaidHistograms({ series, unit, height = 340 }: {
  series: { label: string; chart: ChartBundle }[]
  unit: string
  height?: number
}) {
  const palette = [THEME.accent, THEME.warn, THEME.violet, THEME.good]
  const layout = baseLayout({
    showlegend: true,
    legend: { font: FONT, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: 1.12 },
    barmode: 'overlay',
    xaxis: { title: { text: unit ? `value [${unit}]` : 'value', font: FONT } },
    yaxis: { title: { text: 'density', font: FONT } },
  })
  return (
    <Chart
      height={height}
      data={series.map((entry, index) => {
        const total = entry.chart.histogram.counts.reduce((a, b) => a + b, 0) || 1
        return {
          type: 'bar', name: entry.label,
          x: entry.chart.histogram.centres,
          y: entry.chart.histogram.counts.map((c) => c / total),
          width: entry.chart.histogram.bin_width,
          marker: { color: palette[index % palette.length], opacity: 0.45 },
          hovertemplate: `${entry.label}<br>%{x:.5g}<br>%{y:.4f}<extra></extra>`,
        }
      })}
      layout={layout}
    />
  )
}

export function BoxPlot({ groups, unit, height = 320 }: {
  groups: { label: string; stats: Statistics }[]
  unit: string
  height?: number
}) {
  const usable = groups.filter((g) => Number.isFinite(g.stats.median))
  if (!usable.length) {
    return <div className="py-6 text-center text-[12px] text-muted">No data to compare.</div>
  }
  const layout = baseLayout({
    yaxis: { title: { text: unit ? `value [${unit}]` : 'value', font: FONT } },
  })
  return (
    <Chart
      height={height}
      data={usable.map((group, index) => ({
        type: 'box',
        name: group.label,
        q1: [group.stats.p25], median: [group.stats.median], q3: [group.stats.p75],
        lowerfence: [group.stats.minimum], upperfence: [group.stats.maximum],
        mean: [group.stats.mean],
        boxmean: true,
        marker: { color: [THEME.accent, THEME.warn, THEME.violet, THEME.good][index % 4] },
        line: { width: 1.4 },
        hoverinfo: 'y',
      }))}
      layout={{ ...layout, showlegend: false }}
    />
  )
}

/** Per-spec yield with Wilson intervals as error bars. */
export function YieldBars({ entries, height = 300 }: {
  entries: { description: string; yield_pct: number; ci95_low: number; ci95_high: number }[]
  height?: number
}) {
  if (!entries.length) {
    return <div className="py-6 text-center text-[12px] text-muted">No specifications.</div>
  }
  const layout = baseLayout({
    margin: { l: 210, r: 40, t: 16, b: 44 },
    xaxis: { title: { text: 'yield [%]', font: FONT }, range: [0, 105] },
    yaxis: { gridcolor: 'rgba(0,0,0,0)' },
  })
  const ordered = [...entries].reverse()
  return (
    <Chart
      height={height}
      data={[{
        type: 'bar', orientation: 'h',
        x: ordered.map((e) => e.yield_pct),
        y: ordered.map((e) => e.description),
        marker: {
          color: ordered.map((e) =>
            e.yield_pct >= 99 ? THEME.good : e.yield_pct >= 90 ? THEME.warn : THEME.bad),
          opacity: 0.75,
        },
        error_x: {
          type: 'data', symmetric: false,
          array: ordered.map((e) => Math.max(0, e.ci95_high - e.yield_pct)),
          arrayminus: ordered.map((e) => Math.max(0, e.yield_pct - e.ci95_low)),
          color: THEME.muted, thickness: 1.2, width: 4,
        },
        hovertemplate: '%{y}<br>%{x:.2f}%<extra></extra>',
      }]}
      layout={layout}
    />
  )
}

// ---------------------------------------------------------------------------
// raw simulation traces
// ---------------------------------------------------------------------------

const TRACE_COLOURS = [
  THEME.accent, THEME.warn, THEME.violet, THEME.good, THEME.accent2, THEME.bad,
]

/** Transient waveform: node voltages versus time. */
export function TransientPlot({ tran, nodes, height = 380 }: {
  tran: { time: number[]; nodes: Record<string, number[]> }
  nodes: string[]
  height?: number
}) {
  const shown = nodes.filter((n) => tran.nodes[n]?.length)
  if (!shown.length) {
    return <div className="py-6 text-center text-[12px] text-muted">
      No transient trace.
    </div>
  }
  const layout = baseLayout({
    showlegend: true,
    legend: { font: FONT, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: 1.14 },
    margin: { l: 66, r: 20, t: 34, b: 48 },
    xaxis: { title: { text: 'time [s]', font: FONT }, exponentformat: 'power' },
    yaxis: { title: { text: 'voltage [V]', font: FONT } },
  })
  return (
    <Chart
      height={height}
      data={shown.map((name, i) => ({
        type: 'scatter', mode: 'lines', name,
        x: tran.time, y: tran.nodes[name],
        line: { color: TRACE_COLOURS[i % TRACE_COLOURS.length], width: 1.9 },
        hovertemplate: `${name}<br>t = %{x:.4e} s<br>%{y:.5g} V<extra></extra>`,
      }))}
      layout={layout}
    />
  )
}

/** Bode plot: magnitude and phase of each node against the AC stimulus. */
export function BodePlot({ ac, nodes, height = 430 }: {
  ac: { freqs: number[]; reference: string; nodes: Record<string, { mag_db: (number|null)[]; phase_deg: (number|null)[] }> }
  nodes: string[]
  height?: number
}) {
  const shown = nodes.filter((n) => ac.nodes[n] && n !== ac.reference)
  if (!shown.length || ac.freqs.length < 2) {
    return <div className="py-6 text-center text-[12px] text-muted">
      No AC response.
    </div>
  }
  const layout = baseLayout({
    showlegend: true,
    legend: { font: FONT, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: 1.1 },
    margin: { l: 66, r: 66, t: 30, b: 46 },
    grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
    xaxis: { type: 'log', gridcolor: THEME.grid, linecolor: THEME.axis,
             tickfont: FONT, showticklabels: false, domain: [0, 1] },
    yaxis: { title: { text: 'magnitude [dB]', font: FONT }, gridcolor: THEME.grid,
             linecolor: THEME.axis, tickfont: FONT, domain: [0.56, 1] },
    xaxis2: { type: 'log', title: { text: 'frequency [Hz]', font: FONT },
              gridcolor: THEME.grid, linecolor: THEME.axis, tickfont: FONT,
              anchor: 'y2', domain: [0, 1] },
    yaxis2: { title: { text: 'phase [deg]', font: FONT }, gridcolor: THEME.grid,
              linecolor: THEME.axis, tickfont: FONT, anchor: 'x2',
              domain: [0, 0.42] },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0,
               yref: 'y', line: { color: THEME.muted, width: 1, dash: 'dot' } }],
  })
  const data: any[] = []
  shown.forEach((name, i) => {
    const colour = TRACE_COLOURS[i % TRACE_COLOURS.length]
    data.push({
      type: 'scatter', mode: 'lines', name, legendgroup: name,
      x: ac.freqs, y: ac.nodes[name].mag_db,
      line: { color: colour, width: 2 }, xaxis: 'x', yaxis: 'y',
      hovertemplate: `${name}<br>%{x:.4g} Hz<br>%{y:.3f} dB<extra></extra>`,
    })
    data.push({
      type: 'scatter', mode: 'lines', name, legendgroup: name, showlegend: false,
      x: ac.freqs, y: ac.nodes[name].phase_deg,
      line: { color: colour, width: 1.6, dash: 'dot' },
      xaxis: 'x2', yaxis: 'y2',
      hovertemplate: `${name}<br>%{x:.4g} Hz<br>%{y:.2f} deg<extra></extra>`,
    })
  })
  return <Chart height={height} data={data} layout={layout} />
}
