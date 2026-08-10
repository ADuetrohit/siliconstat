# The circuit solver

Layer 2 of the [architecture](architecture.md): Modified Nodal Analysis plus a
Newton-Raphson loop, with AC and transient built on top of the same assembly.

## Modified Nodal Analysis

The unknown vector is

```
x = [ v_1 … v_(N-1) | i_b0 … i_b(M-1) ]
```

Node 0 is ground and is eliminated, so circuit node `k` occupies matrix row
`k-1`.  Each branch unknown — one per voltage source, one per inductor —
occupies a row after the node block.

The convention is `G x = b`, where `b` is the current **injected into** each
node.  A device current that flows *out of* a node into the device appears with
a `+` sign on the left-hand side.  `MnaSystem` exposes four stamping
primitives (`core/mna.py`) and every device is written in terms of them.

### Stamps

**Resistor** between nodes *a* and *b*, conductance `g = 1/R`:

```
G[a,a] += g   G[b,b] += g   G[a,b] -= g   G[b,a] -= g
```

**Capacitor.**  Open at DC (no stamp at all).  In AC, a complex conductance
`jωC` stamped exactly like a resistor.  In transient, a companion model — see
below.

**Inductor.**  A branch unknown `i`.  At DC the branch row asserts `v_a - v_b =
0`, i.e. an ideal short.  In AC the row becomes `v_a - v_b - jωL·i = 0`.

**Independent voltage source** with branch `b`:

```
G[a,br] += 1   G[b,br] -= 1        (current leaves node a through the source)
G[br,a] += 1   G[br,b] -= 1        (the constraint row)
b[br]    += value
```

The branch current is positive flowing from `+` through the source to `−`, so
a source *delivering* power has a negative branch current, and delivered power
is `-v·i`.  `tests/test_devices.py` pins this convention down.

**Independent current source** from `n+` through the source to `n-`:

```
b[n+] -= I    b[n-] += I
```

**Diode** and **MOSFET** are nonlinear and stamp a companion model, described
next.

### Companion models and the free residual

At iterate `x`, a nonlinear device is replaced by its first-order expansion
about that point.  For a MOSFET, with `gm`, `gds`, `gmb` from
[the device model](mosfet_model.md):

```
I_d = gm·(v_g - v_s) + gds·(v_d - v_s) + gmb·(v_b - v_s) + Ieq
Ieq = sign · ( Ids - gm·vgs - gds·vds - gmb·vbs )
```

The conductances are stamped as voltage-controlled current sources and `Ieq`
as an independent current source.

This has a property worth stating explicitly, because the convergence test
depends on it: **the companion model is exact at the linearisation point.**
The companion current at `x` equals the true device current at `x`.  Therefore

```
F(x) = G(x)·x - b(x)
```

*is* the KCL residual of the nonlinear system at `x` — not an approximation of
it.  The solver gets the residual for free from the matrix it already
assembled, with no second evaluation pass.  `tests/test_devices.py::
test_mosfet_companion_residual_vanishes_at_the_solution` asserts it directly.

## Newton-Raphson

```
x_{k+1} = x_k + clamp( solve(G(x_k), b(x_k)) - x_k )
```

Note the form: because the companion system is built *about* `x_k`, solving it
yields the next iterate directly rather than an increment, so the Newton step
is `x_lin - x_k`.

### Convergence test

Convergence is declared only when **both** criteria hold, in both domains, as
in SPICE:

| Rows | Update criterion | Residual criterion |
| --- | --- | --- |
| node rows | `\|Δv\| ≤ reltol·max(\|v\|) + vntol` | `\|F\| ≤ reltol·max\|I\| + abstol` (amperes) |
| branch rows | `\|Δi\| ≤ reltol·max(\|i\|) + abstol` | `\|F\| ≤ reltol·max\|v\| + vntol` (volts) |

Branch rows are voltage *constraints*, so their residual is a voltage and
their update is a current — the two domains swap, and testing them with the
node tolerances would be a units error.

**Default tolerances are tighter than SPICE's.**  SPICE ships `reltol=1e-3`,
which is fine when you want three digits of a node voltage.  It is not fine
here: this tool routinely characterises current-mirror copy errors of order
0.08 %, so a 0.1 % solver tolerance would sit directly on top of the quantity
being measured.  SiliconStat defaults to `reltol=1e-6`, `vntol=1e-9`,
`abstol=1e-15`.  Measured cost of that choice:

| circuit | Newton iterations | KCL residual | solve time |
| --- | --- | --- | --- |
| resistive divider | 5 | 0 (exact) | 0.41 ms |
| diode bias | 12 | 1.4e-14 A | 1.07 ms |
| current mirror | 8 | 3.4e-21 A | 0.84 ms |
| differential pair | 9 | 2.7e-19 A | 1.09 ms |
| two-stage OTA | 10 | 2.7e-19 A | 1.84 ms |

About one extra iteration.  The resulting accuracy is far better than the
tolerance because Newton converges quadratically and the final step overshoots
the criterion by orders of magnitude: the current mirror's KCL residual at the
output node is 5.5e-13 A against a 10 µA branch current.

### Update clamping

Each node voltage is limited to `max_dv` (default 1 V) of movement per
iteration.

This is not cosmetic.  The square-law model has *zero* conductance in cutoff,
so the companion matrix there contains only `gmin`; solving it with a forced
10 µA into the node asks for `I/gmin = 10⁷ V`.  Without clamping the iterate
is flung across the entire model's domain and the iteration crawls back
linearly — the current mirror took 46 iterations before clamping was added and
takes 8 after.

Clamping cannot move the fixed point, because the update is zero at the
solution, and it stops engaging once steps are small, so Newton's quadratic
rate near the solution is preserved.

Exponential junctions are handled separately, at the device level, by SPICE's
`pnjlim` (`core/mosfet.py:pn_limit`): a diode voltage is limited
logarithmically so that `exp(v/Vt)` cannot overflow between iterations.
Devices see the previous iterate through `ctx.x_prev_iter`.

### Continuation strategies

If plain Newton fails, three fallbacks run in order.  The strategy that
succeeded is reported on the operating point.

| Strategy | Mechanism |
| --- | --- |
| `direct` | clamp 1 V, up to `max_iter` iterations |
| `damped` | clamp 0.1 V, up to `5 × max_iter` iterations |
| `gmin` | ramp an artificial node conductance from 1e-3 down to `gmin`, warm-starting each step |
| `source` | ramp every independent source from 0 to full value in `source_steps` |

A **singular** matrix short-circuits all of this.  A floating node or a shorted
source loop is a topology fault, not a convergence difficulty, and no amount
of continuation will fix it — so `NumericalError` is re-raised immediately with
its specific diagnostic ("the circuit likely has a floating node, a shorted
voltage-source loop, or a missing DC path to ground") rather than being buried
under a generic convergence failure.

If every strategy genuinely fails, `ConvergenceError` carries the iteration
count, the residual, the worst node, the list of strategies attempted and each
strategy's own diagnostic.  **The solver never returns a result it could not
verify** — after a strategy reports success, the system is re-assembled with
limiting disabled and the true residual re-checked before the answer is
accepted.

## AC analysis

Small-signal AC linearises about the converged DC operating point and solves a
complex MNA system at each frequency.  Device conductances are frozen at their
operating-point values; capacitors contribute `jωC`; inductors contribute
`-jωL` on their branch row; sources contribute their `AC` magnitude and phase.

Validation against the analytic RC low-pass `H(f) = 1/(1 + j2πfRC)` agrees to
`rtol=1e-9` across five decades (`tests/test_solver.py`).

### Reading a crossing accurately

Bandwidth and unity-gain frequency are *specification* quantities, so grid
resolution matters.  Interpolating between swept points at 30 points/decade
leaves adjacent samples 8 % apart, and linear interpolation of a curved
response lands about 0.2 % off.

SiliconStat therefore brackets the crossing on the swept grid and then
**refines it by bisection on the actual transfer function**, one
single-frequency AC solve per step.  Measured on the RC low-pass:

```
f3dB measured  159155.002614 Hz
     analytic  159154.943092 Hz      relative error 3.7e-07
```

One further subtlety: the "−3 dB" bandwidth is the *half-power* point, where
the magnitude falls to `1/√2`.  That is `10·log10(2) = 3.0103 dB`, not
3.000 dB.  Using the round number shifts a single-pole bandwidth by 0.24 % —
which is exactly the error the refinement was added to remove, so
`HALF_POWER_DB` is used throughout.

## Transient analysis

Fixed-step trapezoidal integration, with a backward-Euler first step (the
DC operating point supplies `v` but not `dv/dt`, so a trapezoidal first step
would ring).

Reactive elements contribute companion models.  For a capacitor under
trapezoid:

```
geq = 2C/Δt        ieq = geq·v_prev + i_prev
```

and the pair `(v, i)` is carried forward after each accepted step.

### Frozen Meyer capacitances

This one was a real bug, found and fixed, and it is worth recording.

MOSFET gate capacitances are region-dependent — `Cgs` jumps from `⅔·Cox` in
saturation to `½·Cox` in triode to an overlap-only value in cutoff.
Recomputing them inside the Newton loop makes the companion conductance
`2C/Δt` a *discontinuous* function of the iterate, so the Jacobian is
discontinuous exactly at a region boundary.  The inverter transient failed to
converge at t = 4.06 ns — precisely the input edge, where both devices cross
regions.

The fix is the standard quasi-static treatment: **capacitances are held
constant for the duration of a timestep**, evaluated at the last converged
point and refrozen afterwards.  The Jacobian becomes smooth again, and the
inverter transient went from failing at 1063 ms to succeeding in 147 ms.

If a device reverses (drain and source exchange) between steps, the stored
`(v, i)` histories refer to the wrong node pairs, so they are re-seeded rather
than integrated forward.

Validation against the analytic RC step response `v(t) = 1 - exp(-t/RC)` gives
a maximum deviation below 1e-4 at `Δt/τ = 0.005`.

## Numerical choices

**Dense LU.**  `MnaSystem` stores a dense matrix and calls `numpy.linalg.solve`.
For transistor-level analog blocks of tens of nodes this is faster than sparse
bookkeeping and vastly simpler to keep correct.  It is the wrong choice above a
few hundred nodes; see [limitations.md](limitations.md).

**Explicit failure detection.**  `MnaSystem.solve` checks for NaN/Inf in the
matrix, in the right-hand side and in the solution, and converts
`LinAlgError` into a `NumericalError` that names the likely topological cause.

**gmin.**  A conductance of `1e-12` is stamped across every MOSFET
drain-source pair, which keeps the matrix non-singular when a device is in
cutoff.  At 10 µA operating currents it contributes a relative error of order
1e-8 — six orders below `reltol`.
