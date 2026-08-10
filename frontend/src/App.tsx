import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'

import { getHealth } from '@/lib/api'
import type { Health } from '@/types/api'

import Dashboard from '@/pages/Dashboard'
import Circuits from '@/pages/Circuits'
import MonteCarlo from '@/pages/MonteCarlo'
import Results from '@/pages/Results'
import YieldAnalysis from '@/pages/YieldAnalysis'
import Sensitivity from '@/pages/Sensitivity'
import Reports from '@/pages/Reports'
import Documentation from '@/pages/Documentation'

const NAV = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/circuits', label: 'Circuits' },
  { to: '/monte-carlo', label: 'Monte Carlo' },
  { to: '/results', label: 'Results' },
  { to: '/yield', label: 'Yield Analysis' },
  { to: '/sensitivity', label: 'Sensitivity' },
  { to: '/reports', label: 'Reports' },
  { to: '/docs', label: 'Documentation' },
]

function HealthPill() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let alive = true
    const tick = () => {
      getHealth()
        .then((value) => { if (alive) { setHealth(value); setError(false) } })
        .catch(() => { if (alive) setError(true) })
    }
    tick()
    const timer = setInterval(tick, 15_000)
    return () => { alive = false; clearInterval(timer) }
  }, [])

  const tone = error ? 'bg-bad' : health ? 'bg-good' : 'bg-warn'
  return (
    <div className="flex items-center gap-2 text-[11px] text-muted">
      <span className={`inline-block h-2 w-2 rounded-full ${tone} ${error ? '' : 'pulse'}`} />
      {error
        ? <span className="text-bad">API unreachable</span>
        : health
          ? (
            <span>
              v{health.version} · {health.runs_stored} runs
              {health.active_jobs > 0 && (
                <span className="text-accent"> · {health.active_jobs} active</span>
              )}
            </span>
          )
          : <span>connecting…</span>}
    </div>
  )
}

export default function App() {
  return (
    <div className="flex min-h-full">
      <aside className="sticky top-0 flex h-screen w-[212px] shrink-0 flex-col border-r border-line bg-panel/80 backdrop-blur">
        <div className="border-b border-line px-4 py-4">
          <div className="text-[15px] font-semibold tracking-[0.06em] text-accent">
            SiliconStat
          </div>
          <div className="mt-0.5 text-[10px] leading-snug text-muted">
            Monte Carlo mismatch analysis<br />for analog ICs
          </div>
        </div>

        <nav className="flex-1 space-y-0.5 px-2 py-3">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `block rounded-md px-3 py-1.5 text-[12px] transition-colors ${
                  isActive
                    ? 'bg-accent/12 text-accent ring-1 ring-accent/30'
                    : 'text-muted hover:bg-panel2 hover:text-ink'
                }`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="space-y-1 border-t border-line px-4 py-3">
          <HealthPill />
          <div className="text-[9.5px] leading-snug text-muted/70">
            Level-1 device models. Not for sign-off.
          </div>
        </div>
      </aside>

      <main className="min-w-0 flex-1 px-6 py-5">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/circuits" element={<Circuits />} />
          <Route path="/monte-carlo" element={<MonteCarlo />} />
          <Route path="/results" element={<Results />} />
          <Route path="/yield" element={<YieldAnalysis />} />
          <Route path="/sensitivity" element={<Sensitivity />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/docs" element={<Documentation />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  )
}
