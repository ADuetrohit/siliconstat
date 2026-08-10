# Limitations

> **SiliconStat is a tool for education, methodology demonstration and design
> exploration.  It is not a sign-off tool.**  Do not make tape-out decisions
> with it.

The list below is deliberately unflattering.  A statistical tool whose
limitations are not stated is worse than no tool, because its numbers look
authoritative.

## Device models

The MOSFET model is square-law, in the SPICE level-1 tradition.  Absent:

- **velocity saturation** — a real 180 nm device at high overdrive is closer to
  linear in `Vov` than quadratic, so this model overestimates drive current
  and underestimates `gm/Id` degradation at high overdrive;
- **drain-induced barrier lowering** — no `Vth(Vds)` dependence, so output
  conductance is modelled only by the empirical `LAMBDA`;
- **short-channel threshold roll-off** — `Vth` does not depend on `L`, so
  comparing geometries assumes an idealised process;
- **subthreshold conduction** — cutoff current is exactly zero.  Weak
  inversion, subthreshold slope, off-state leakage and any circuit that relies
  on them (subthreshold bias generators, leakage-dominated retention) cannot be
  simulated at all;
- **mobility degradation with vertical field** — `KP` is bias-independent;
- gate leakage, GIDL, junction breakdown, self-heating, narrow-width effects,
  poly depletion, quantum corrections;
- **charge-conserving capacitance** — the Meyer model is not charge-based, so
  transient charge is not exactly conserved.

`LAMBDA` is a fitted constant, not a physical quantity: it does not vary with
`L` or `Vds` as real output conductance does.

**The shipped model cards are not a PDK.**  `examples/*.net` contain plausible
180 nm-class numbers assembled from published typical values.  They are not
extracted from measurement and are not calibrated to any real process.  The
mismatch coefficients (`AVT = 3.5 mV·µm`, `ABETA = 1 %·µm`) are representative
literature figures.  Any absolute number this tool produces should be read as
"what this model implies", not "what silicon will do".  The methodology — the
Pelgrom scaling, the global/local split, the yield accounting, the sensitivity
attribution — is the part meant to transfer.

## Solver

- **Fixed-step transient with no local-truncation-error control.**  `TSTEP`
  must be chosen small relative to the fastest dynamics; the solver will not
  detect that it is too coarse.  No automatic timestep, no LTE estimation, no
  breakpoint handling at source discontinuities.
- **Meyer capacitances are frozen for the duration of a timestep.**  This is a
  standard quasi-static treatment and is what makes the transient converge at
  region boundaries (see [circuit_solver.md](circuit_solver.md)), but it
  introduces a first-order error in `dQ/dt` that a charge-based model would not
  have.
- **Dense LU.**  Fine for the tens of nodes these circuits have; the wrong
  choice above a few hundred.  Cost grows as O(n³) with no sparsity exploited.
- **No subcircuits (`.subckt`).**  Hierarchical netlists must be flattened by
  hand, which makes anything of realistic size impractical.
- **No behavioural sources** (`B`, `E`, `F`, `G`, `H` elements), no
  transmission lines, no switches, no BJTs, no JFETs.
- **No pole-zero, noise, harmonic-balance, distortion or DC-sweep analyses.**
  There is no noise model at all, so this tool cannot say anything about
  SNR, jitter or kT/C.
- **No convergence aids beyond the four implemented** — no pseudo-transient
  continuation, no homotopy on device parameters.

## Statistics

- **Mismatch coefficients are inputs, not measurements.**  Everything the tool
  says about spread is downstream of `AVT` and `ABETA`.  Wrong coefficients
  give confidently wrong yields.
- **Corners are synthetic.**  `FF`/`SS`/`FS`/`SF` are defined as ±3σ of the
  same global distribution the Monte Carlo process model uses, which keeps the
  two views self-consistent but is *not* how a foundry defines corner models
  (those come from characterised corner lots and are not simply ±3σ of the
  typical model).
- **No proximity or gradient term.**  Pelgrom's law has a distance-dependent
  component for systematic across-die gradients.  The netlist carries no layout
  coordinates, so only the area term is modelled.  Real matched-pair
  performance also depends on common-centroid layout, dummies and orientation,
  none of which exist here.
- **The Gaussian copula preserves rank correlation exactly but linear
  correlation only approximately** for non-Gaussian marginals.
- **Latin hypercube sampling couples samples**, so a single LHS sample cannot
  be re-simulated in isolation and the usual independence-based error formulas
  do not strictly apply.
- **Normality is assumed by the ±3σ limits and by Cpk.**  The tool reports a
  normality p-value so you can check, but it does not stop you from reading a
  Cpk off a bimodal distribution.

## Yield

- **Plain Monte Carlo cannot resolve ppm yields.**  Confirming a 1 ppm defect
  rate needs on the order of 10⁸ samples; at the ~290 samples/s this tool
  achieves on a simple mirror that is about four days of compute for one
  circuit at one corner.  Real high-sigma flows use importance sampling,
  worst-case distance, scaled-sigma sampling or extreme-value extrapolation.
  **None of those are implemented.**  The `required_samples_for_margin()`
  helper will tell you honestly how many samples a target precision needs.
- **Yield is reported at the conditions you simulate.**  A nominal-only run
  says nothing about corners; the demo mirror swings from 57 % at TT/1.8 V/27 °C
  to 37 % at FF/1.98 V/125 °C.
- **No aging, no reliability** — NBTI, HCI, TDDB and electromigration are not
  modelled, so "yield" here means "meets spec at time zero".

## ML surrogate

- **It interpolates within the sampled distribution.**  A surrogate trained on
  ±3σ draws has no basis for extrapolating to ±5σ, which is exactly where
  high-sigma yield lives.  It is not a route around the previous point.
- **It learns one measurement at a time** from the variation slots only.  It
  does not learn the circuit, cannot transfer to a modified topology, and must
  be retrained if the variation model changes.
- **Break-even is real.**  Training costs a full Monte Carlo run; below the
  reported break-even sample count, plain simulation is cheaper.  The tool
  reports that number rather than quoting a raw per-prediction speedup.

## Engineering

- **No authentication or authorisation on the REST API.**  It is a single-user
  tool.  Exposing it on a shared network requires an auth layer in front.
- **Single-node SQLite.**  Fine for millions of sample rows on one machine; not
  a multi-writer or distributed store.  The schema is written to be portable to
  PostgreSQL but that backend does not exist yet.
- **In-process job manager.**  Jobs live in the server's memory and are lost on
  restart; there is no durable queue, no retry and no multi-machine
  distribution.
- **Parallel speedup is modest** and can be negative for fast circuits — see
  the measured table in [monte_carlo.md](monte_carlo.md).  On Windows, high
  worker counts can exhaust the page file; the engine detects this and falls
  back to sequential.
- **No incremental or resumable runs.**  A cancelled run keeps the samples it
  completed but cannot be continued from where it stopped.

## Cross-validated, but only against level-1

SiliconStat is cross-validated against **ngspice 46** on all five DC example
circuits, agreeing to a worst relative difference of 6.5e-07 across 23
quantities — see [validation.md](validation.md).

That check is against ngspice's own `LEVEL=1` model, which is the only
comparison that isolates *implementation* correctness. It says nothing about
how far level-1 itself sits from a real short-channel device: that gap is the
subject of the "Device models" section above and is large. A BSIM comparison
would measure the modelling gap rather than the implementation, and has not
been run.

AC, transient and the statistical layers are **not** cross-validated against
ngspice; only the DC operating points are. The AC and transient results are
validated against closed forms instead (see validation.md sections 8 and 6).
