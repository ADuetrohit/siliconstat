/** Shared presentational primitives for the dark EDA shell. */

import type { ReactNode } from 'react'
import { useMemo, useState } from 'react'

// ---------------------------------------------------------------------------
// containers
// ---------------------------------------------------------------------------

export function Panel({ title, subtitle, actions, children, className = '' }: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={`rounded-lg border border-line bg-panel/85 backdrop-blur-sm ${className}`}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-4 py-2.5">
          <div>
            {title && (
              <h2 className="text-[11px] font-semibold uppercase tracking-[0.09em] text-accent">
                {title}
              </h2>
            )}
            {subtitle && <p className="mt-0.5 text-[11px] text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="mb-2 flex items-baseline justify-between gap-3">
      <h3 className="text-[12px] font-semibold text-ink">{children}</h3>
      {hint && <span className="text-[11px] text-muted">{hint}</span>}
    </div>
  )
}

export function StatTile({ label, value, sub, tone = 'default' }: {
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: 'default' | 'pass' | 'fail' | 'warn' | 'accent'
}) {
  const colour = {
    default: 'text-ink',
    accent: 'text-accent',
    pass: 'text-good',
    warn: 'text-warn',
    fail: 'text-bad',
  }[tone]
  return (
    <div className="rounded-lg border border-line bg-panel2 px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-[0.08em] text-muted">{label}</div>
      <div className={`mt-1 text-[19px] leading-tight ${colour}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-muted">{sub}</div>}
    </div>
  )
}

export function KeyValueGrid({ items, columns = 3 }: {
  items: [string, ReactNode][]
  columns?: number
}) {
  return (
    <dl
      className="grid gap-2"
      style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${Math.floor(760 / columns)}px, 1fr))` }}
    >
      {items.map(([key, value]) => (
        <div key={key} className="rounded-md border border-line bg-panel2 px-3 py-2">
          <dt className="text-[10px] uppercase tracking-[0.07em] text-muted">{key}</dt>
          <dd className="mt-0.5 break-words text-[13px] text-ink">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

// ---------------------------------------------------------------------------
// badges, banners, progress
// ---------------------------------------------------------------------------

export type Tone = 'pass' | 'fail' | 'warn' | 'info' | 'muted'

export function Badge({ children, tone = 'info' }: { children: ReactNode; tone?: Tone }) {
  const styles: Record<Tone, string> = {
    pass: 'border-good/45 bg-good/15 text-good',
    fail: 'border-bad/45 bg-bad/15 text-bad',
    warn: 'border-warn/45 bg-warn/15 text-warn',
    info: 'border-accent/40 bg-accent/12 text-accent',
    muted: 'border-line bg-panel2 text-muted',
  }
  return (
    <span
      className={`inline-block rounded-full border px-2 py-[1px] text-[10.5px] font-semibold tracking-[0.03em] ${styles[tone]}`}
    >
      {children}
    </span>
  )
}

export function ErrorBanner({ title = 'Error', message, onRetry }: {
  title?: string
  message: string
  onRetry?: () => void
}) {
  return (
    <div className="rounded-md border-l-[3px] border-bad bg-bad/10 px-4 py-3">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[12px] font-semibold text-bad">{title}</div>
          <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-[11.5px] text-ink/90">
            {message}
          </pre>
        </div>
        {onRetry && (
          <Button onClick={onRetry} variant="ghost">
            Retry
          </Button>
        )}
      </div>
    </div>
  )
}

export function NoteBanner({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-r-md border-l-[3px] border-warn bg-warn/8 px-3.5 py-2 text-[11.5px] text-warn/90">
      {children}
    </div>
  )
}

export function EmptyState({ title, hint, action }: {
  title: string
  hint?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line bg-panel2/60 px-6 py-10 text-center">
      <div className="text-[13px] text-ink">{title}</div>
      {hint && <div className="max-w-lg text-[11.5px] text-muted">{hint}</div>}
      {action}
    </div>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-[12px] text-muted">
      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-line border-t-accent" />
      {label ?? 'Loading…'}
    </div>
  )
}

export function ProgressBar({ value, total, tone = 'accent' }: {
  value: number
  total: number
  tone?: 'accent' | 'good'
}) {
  const pct = total > 0 ? Math.min(100, (value / total) * 100) : 0
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-panel2 ring-1 ring-line">
      <div
        className={`h-full rounded-full transition-[width] duration-300 ${tone === 'good' ? 'bg-good' : 'bg-accent'}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------

export function Button({ children, onClick, disabled, variant = 'default', type = 'button',
                        title }: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: 'default' | 'primary' | 'ghost' | 'danger'
  type?: 'button' | 'submit'
  title?: string
}) {
  const styles = {
    default: 'border-line bg-panel2 text-ink hover:border-axis hover:bg-panel',
    primary: 'border-accent/60 bg-accent/15 text-accent hover:bg-accent/25',
    ghost: 'border-transparent bg-transparent text-muted hover:text-ink',
    danger: 'border-bad/50 bg-bad/12 text-bad hover:bg-bad/22',
  }[variant]
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md border px-3 py-1.5 text-[12px] transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  )
}

export function Field({ label, hint, children }: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <label className="block">
      <span className="block text-[10px] uppercase tracking-[0.07em] text-muted">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-1 block text-[10.5px] text-muted/80">{hint}</span>}
    </label>
  )
}

export function Toggle({ checked, onChange, label }: {
  checked: boolean
  onChange: (value: boolean) => void
  label: ReactNode
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-[12px] text-ink">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}

export function Select<T extends string>({ value, onChange, options }: {
  value: T
  onChange: (value: T) => void
  options: { value: T; label: string }[]
}) {
  return (
    <select
      className="w-full"
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
}

export function NumberInput({ value, onChange, min, max, step = 1, className = '' }: {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  className?: string
}) {
  return (
    <input
      type="number"
      className={`w-full ${className}`}
      value={Number.isFinite(value) ? value : ''}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        const parsed = Number(e.target.value)
        if (!Number.isNaN(parsed)) onChange(parsed)
      }}
    />
  )
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      variant="ghost"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1400)
        })
      }}
    >
      {copied ? 'Copied' : label}
    </Button>
  )
}

// ---------------------------------------------------------------------------
// table
// ---------------------------------------------------------------------------

export interface Column<T> {
  key: string
  header: ReactNode
  align?: 'left' | 'right'
  render: (row: T) => ReactNode
  sortValue?: (row: T) => number | string
  width?: string
}

export function DataTable<T>({ columns, rows, rowKey, onRowClick, selectedKey, dense,
                              emptyMessage = 'No rows.' }: {
  columns: Column<T>[]
  rows: T[]
  rowKey: (row: T, index: number) => string
  onRowClick?: (row: T) => void
  selectedKey?: string
  dense?: boolean
  emptyMessage?: string
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null)

  const sorted = useMemo(() => {
    if (!sort) return rows
    const column = columns.find((c) => c.key === sort.key)
    if (!column?.sortValue) return rows
    return [...rows].sort((a, b) => {
      const va = column.sortValue!(a)
      const vb = column.sortValue!(b)
      if (typeof va === 'number' && typeof vb === 'number') {
        return (Number.isNaN(va) ? -Infinity : va) > (Number.isNaN(vb) ? -Infinity : vb)
          ? sort.dir : -sort.dir
      }
      return String(va).localeCompare(String(vb)) * sort.dir
    })
  }, [rows, sort, columns])

  if (rows.length === 0) {
    return <div className="px-1 py-3 text-[12px] text-muted">{emptyMessage}</div>
  }

  const pad = dense ? 'px-2 py-1' : 'px-2.5 py-1.5'

  return (
    <div className="max-h-[560px] overflow-auto rounded-md border border-line">
      <table className="w-full border-collapse text-[12px]">
        <thead className="sticky top-0 z-10">
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                style={column.width ? { width: column.width } : undefined}
                onClick={() => {
                  if (!column.sortValue) return
                  setSort((current) =>
                    current?.key === column.key
                      ? { key: column.key, dir: current.dir === 1 ? -1 : 1 }
                      : { key: column.key, dir: 1 })
                }}
                className={`whitespace-nowrap border-b border-line bg-panel2 ${pad} text-[10px] font-semibold uppercase tracking-[0.07em] text-muted ${
                  column.align === 'right' ? 'text-right' : 'text-left'
                } ${column.sortValue ? 'cursor-pointer select-none hover:text-ink' : ''}`}
              >
                {column.header}
                {sort?.key === column.key && (sort.dir === 1 ? ' ▲' : ' ▼')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, index) => {
            const key = rowKey(row, index)
            return (
              <tr
                key={key}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                className={`border-b border-line/60 last:border-0 ${
                  onRowClick ? 'cursor-pointer' : ''
                } ${selectedKey === key ? 'bg-accent2/15' : 'hover:bg-accent2/8'}`}
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className={`${pad} align-top ${column.align === 'right' ? 'num' : ''}`}
                  >
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
