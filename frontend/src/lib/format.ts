/**
 * Engineering-notation formatting.
 *
 * Mirrors src/siliconstat/core/units.py:format_eng, including its convention:
 * display uses the SI reading of `M` (mega).  These strings are for humans,
 * not for feeding back into a netlist parser.
 */

const PREFIXES: [number, string][] = [
  [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k'],
  [1, ''], [1e-3, 'm'], [1e-6, 'u'], [1e-9, 'n'],
  [1e-12, 'p'], [1e-15, 'f'],
]

/** Units that should never take an SI prefix. */
const BARE_UNITS = new Set(['', '%', '-', 'dB', 'deg', 'frac', 'V/V'])

export function formatEng(value: number | null | undefined, unit = '', digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a'
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '-∞'
  if (BARE_UNITS.has(unit)) return `${trim(value, digits)}${unit ? ` ${unit}` : ''}`
  if (value === 0) return `0 ${unit}`.trim()

  const magnitude = Math.abs(value)
  for (const [scale, prefix] of PREFIXES) {
    if (magnitude >= scale * 0.999) {
      return `${trim(value / scale, digits)} ${prefix}${unit}`.trim()
    }
  }
  return `${trim(value / 1e-15, digits)} f${unit}`.trim()
}

function trim(value: number, digits: number): string {
  const text = value.toPrecision(digits)
  return text.includes('e') ? Number(text).toExponential(digits - 1)
    : String(Number(text))
}

export function formatPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a'
  return `${value.toFixed(digits)}%`
}

export function formatSigned(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a'
  return (value >= 0 ? '+' : '') + value.toPrecision(digits)
}

/** "60.20% [58.04, 62.32]" */
export function formatWithInterval(value: number, low: number, high: number,
                                   digits = 2): string {
  if (!Number.isFinite(value)) return 'n/a'
  if (!Number.isFinite(low) || !Number.isFinite(high)) return formatPercent(value, digits)
  return `${value.toFixed(digits)}% [${low.toFixed(digits)}, ${high.toFixed(digits)}]`
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return 'n/a'
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`
  if (seconds < 90) return `${seconds.toFixed(1)} s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`
}

export function formatBytes(bytes: number): string {
  const units = ['B', 'kB', 'MB', 'GB']
  let value = bytes
  let index = 0
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return 'n/a'
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString()
}

/** Colour token for a yield percentage. */
export function yieldTone(pct: number): 'pass' | 'warn' | 'fail' {
  if (!Number.isFinite(pct)) return 'warn'
  if (pct >= 99) return 'pass'
  if (pct >= 90) return 'warn'
  return 'fail'
}

export function regionTone(region: string): 'pass' | 'warn' | 'fail' | 'info' {
  if (region === 'saturation') return 'pass'
  if (region === 'triode') return 'warn'
  if (region === 'cutoff') return 'fail'
  return 'info'
}
