/** Choosing which measurement to show when the user has not picked one. */

import type { RunAnalysis } from '@/types/api'

/**
 * The measurement a page should open on.
 *
 * Not simply the first one declared. A netlist usually declares its stimulus
 * before its result, so the first entry tends to be something like the current
 * mirror's 10 uA reference — a quantity the run holds fixed, whose entire
 * spread is solver noise. Opening there is misleading rather than merely dull:
 * the sensitivity page will happily decompose a sigma of 2.6e-14 A and report
 * that one parameter explains 99.98 % of it.
 *
 * So: prefer a measurement the netlist actually constrains with a `.spec`,
 * since that is what the run was for. Failing that, take whichever varies most
 * relative to its own mean.
 *
 * `names` is passed separately because callers key off different parts of the
 * analysis — `statistics` on the results page, `sensitivity` on the
 * sensitivity page — and those key sets are not always identical.
 */
export function defaultMetric(analysis: RunAnalysis, names: string[]): string {
  if (names.length === 0) return ''

  const constrained = analysis.spec_meta
    .map((spec) => spec.measure)
    .find((measure) => names.includes(measure))
  if (constrained) return constrained

  let best = names[0]
  let bestSpread = -Infinity
  for (const name of names) {
    const stat = analysis.statistics[name]
    if (!stat) continue
    // cv is undefined for a zero mean (offsets, error terms); std still ranks.
    const spread = Number.isFinite(stat.cv) ? Math.abs(stat.cv) : stat.std
    if (Number.isFinite(spread) && spread > bestSpread) {
      [best, bestSpread] = [name, spread]
    }
  }
  return best
}
