# Yield

Layer 7 of the [architecture](architecture.md).

## Yield is a ratio, and a ratio needs a denominator

The single rule this module is built around: **never report a yield without
saying what it was divided by.**

A Monte Carlo run of 1000 samples does not necessarily produce 1000 usable
results.  Some samples fail to converge; some converge but produce a
measurement that is undefined.  There are two defensible denominators, and
SiliconStat reports both:

**`yield_over_successful`** — passing / simulations that converged and produced
valid measurements.  This is the number an analog designer wants: *"of the dice
I could actually measure, what fraction met spec?"*

**`yield_over_attempted`** — passing / samples attempted, counting every
failure as a non-pass.  The conservative bound.

The gap between them is exactly the information the run lost to numerical
failure.  If they differ materially, the honest response is to fix the
convergence problem, not to pick the friendlier number.  Every report states
the denominator in words: *"614 of 1000 successful simulations"*.

Samples whose measurement was invalid are counted as **failing** the specs on
that measurement, and a note says how many.  A NaN is not a pass.

## Per-specification and combined

Each `.spec` line produces its own yield with a confidence interval, a margin
and a capability index.  The combined yield is computed **sample by sample** —
a sample passes only if it passes every specification — not by multiplying the
individual yields.

That distinction is measurable.  On the demo current mirror, 2000 samples:

| specification | yield | Cpk |
| --- | ---: | ---: |
| `ierr <= 2 %` | 79.40 % | 0.27 |
| `ierr >= -2 %` | 80.80 % | 0.30 |
| `ptot <= 100 uW` | 100.00 % | 50.42 |
| **combined (measured)** | **60.20 %** | |
| product of individual yields | 64.16 % | |

The product overstates the yield by 4 points, because the two `ierr` specs are
two sides of the same distribution and fail *together* in a correlated way, not
independently.  The report emits a note whenever the two differ by more than a
point:

> *The combined yield differs from the product of the individual yields
> (64.16 %), which means the specifications fail together rather than
> independently — they share dominant variation sources.  The measured combined
> yield is the correct number; the product is shown only for contrast.*

The **limiting specification** — the one with the lowest individual yield — is
identified, because that is where design effort goes.

## Confidence intervals

Every percentage carries a **Wilson score interval**, not the normal
approximation `p ± 1.96·√(p(1−p)/n)`.

The normal approximation fails exactly where yield analysis lives.  At
`p = 1.0` it gives a zero-width interval — claiming that 50 out of 50 passing
proves 100 % yield — and near the extremes it can produce bounds outside
[0, 1].  The Wilson interval stays inside the unit interval and remains
sensible at the boundaries:

| observed | n | Wilson 95 % interval |
| --- | ---: | --- |
| 50/50 | 50 | [92.9 %, 100 %] |
| 0/50 | 50 | [0 %, 7.1 %] |
| 90/100 | 100 | [82.56 %, 94.48 %] |
| 900/1000 | 1000 | [87.9 %, 91.8 %] |

A 100 % result from 50 samples is *not* evidence of 100 % yield; it is evidence
of at least 92.9 %.  The report shows that.

## Margins and capability

**Margin to the limit** — the mean distance to the specification, in the
measurement's own units.

**Margin in sigmas** — the same distance divided by the standard deviation.
This is the number that tells you how much room the design has: a spec sitting
1σ from the mean will fail 16 % of the time no matter how many samples you run.

**Cp and Cpk** — the process capability indices:

```
Cp  = (USL - LSL) / (6σ)
Cpu = (USL - µ) / (3σ)        Cpl = (µ - LSL) / (3σ)
Cpk = min(Cpu, Cpl)
```

`Cp` is the capability the spread *could* achieve if perfectly centred; `Cpk`
is what it achieves given where the mean actually sits.  The gap between them
is a centring problem, which is usually cheaper to fix than a spread problem.

Conventional thresholds: `Cpk ≥ 1.33` is "capable" (about 63 ppm for a centred
Gaussian); `Cpk = 1.0` puts the limit exactly 3σ from the mean (0.27 %
defective, two-sided).  The current mirror's `Cpk ≈ 0.27` is a plainly
incapable process — consistent with its 60 % yield — while its power spec at
`Cpk = 50` is not a constraint at all.

Cpk assumes normality.  Check the normality p-value in the statistics table
before believing a Cpk-derived defect rate; see
[statistics.md](statistics.md).

## How many samples?

Monte Carlo resolves yield to `±m` percentage points at 95 % confidence with

```
n = z² · p(1−p) / m²
```

`siliconstat.analysis.required_samples_for_margin()` evaluates this:

| observed yield | ±1 % | ±0.5 % | ±0.1 % |
| ---: | ---: | ---: | ---: |
| 50 % | 9 604 | 38 415 | 960 365 |
| 90 % | 3 458 | 13 830 | 345 732 |
| 95 % | 1 825 | 7 299 | 182 470 |
| 99 % | 381 | 1 522 | 38 031 |
| 99.9 % | 39 | 154 | 3 838 |

Two things follow.  First, the common practice of running 100 samples gives a
yield to roughly ±10 points — enough to distinguish "broken" from "fine" and
nothing more.  Second, **plain Monte Carlo cannot reach ppm yields.**
Confirming a 1 ppm defect rate needs on the order of 10⁸ samples.  Real
sign-off flows use importance sampling, worst-case distance or extreme-value
extrapolation for that; none of those are implemented here, and
[limitations.md](limitations.md) says so.

## The convergence trace

Because the numbers above are only bounds on *precision*, every report also
shows how the yield estimate actually behaved as samples accumulated: the
running yield with its Wilson band, and the sample count at which it settled
to within a tolerance of its final value.

On the demo mirror the yield estimate wanders between 72 % and 61 % over the
first hundred samples and settles around 700 — which is a much more direct
argument for "run more samples" than any formula.

## PVT and yield

Yield at a single nominal condition is not yield.  A design has to hold across
the process corners, the supply tolerance and the temperature range.

```bash
siliconstat pvt --circuit examples/current_mirror.net --samples 150 \
    --metric ierr --corners TT FF SS --temps -40 27 125
```

produces a yield per condition and identifies the worst case.  On the demo
mirror the worst corner is **FF / 1.98 V / 125 °C at 37.33 %**, against 56.67 %
at TT / 1.80 V / 27 °C — a 19-point spread that a nominal-only run would have
missed entirely.

The physics behind that ordering is worth stating, because it is a check that
the tool is doing something real: FF raises `beta`, which at a fixed 10 µA
lowers the overdrive and therefore *raises* `gm/Id`, making the mirror more
sensitive to the same threshold mismatch.  A higher supply raises `V(out)` and
so raises the mean copy error through channel-length modulation.  The two
effects compound at FF / high supply.  See [pvt](mismatch.md) and
`tests/test_pvt.py`, which assert these orderings as predictions rather than
recorded values.

## Specifying

```
.spec ierr  <= 2%
.spec ierr  >= -2%
.spec ptot  <= 100u
.spec av    >= 70
.spec pm    >= 45           "phase margin"
```

Operators are `<=`, `>=`, `<`, `>`.  Values accept SPICE engineering suffixes
and a trailing `%`.  An optional quoted label is used in reports in place of
the generated description.  A spec naming a measurement that does not exist is
a parse error listing the ones that do.

Specifications can also be supplied at run time — through the API, or through
`MonteCarloConfig(specs=[...])` — which is how you sweep a limit to find where
yield falls off without editing the netlist.
