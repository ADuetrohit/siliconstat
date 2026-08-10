# Validation

Evidence that the simulator computes the right thing.  Every check below
compares against an answer derived **independently** — from Ohm's law, the
Shockley equation, the square law, small-signal amplifier theory or a
closed-form statistical result — rather than against the simulator's own
previous output.

All of it is automated: `tests/test_analytical.py`, `tests/test_solver.py`,
`tests/test_pelgrom.py` and `tests/test_correlation.py`.  The numbers below are
reproduced by `docs/scripts/collect_doc_numbers.py`.

```
738 passed in 36.54s
698 passed, 40 deselected in 25.33s      (excluding tests marked 'slow')
```

## 1. Ohm's law — the MNA assembly

`examples/rc_divider.net`: 5 V across 1 kΩ + 3 kΩ.

| quantity | closed form | simulated | error |
| --- | --- | --- | --- |
| V(mid) | 3.75 V | 3.750000000000000 V | 0 |
| I(R1) | 1.25 mA | 1.250000000000000e-3 A | 0 |
| P(total) | 6.25 mW | 6.25 mW | < 1e-12 rel |
| P(R2) | 4.6875 mW | 4.6875 mW | < 1e-12 rel |

Exact to double precision, as a linear system should be.

## 2. The Shockley diode — the exponential device

`examples/diode_bias.net`: 1.8 V through 10 kΩ into a diode with
`IS = 1e-14 A`, `N = 1`, `RS = 5 Ω`.

The reference is obtained by solving the transcendental loop equation with
`scipy.optimize.brentq` to `xtol=1e-18`:

```
I = IS·( exp( (VDD − I·R − I·RS) / Vt ) − 1 )
```

| quantity | brentq reference | simulated | agreement |
| --- | --- | --- | --- |
| I(D1) | 119.914 µA | 119.914 µA | 1e-6 rel |
| V(a) | 0.600859 V | 0.600859 V | 1e-6 rel |

Converges from a cold start in 12 Newton iterations.  `RS` is realised as an
explicit series resistor on an auto-created internal node, so the exponential
element itself stays two-terminal.

## 3. The MOSFET square law

**Device level.**  `gm`, `gds` and `gmb` are checked against central finite
differences to 2e-5 relative, in saturation and triode, with and without body
bias.  `Ids` and both derivatives are verified continuous across the
triode/saturation boundary.  In saturation the identity `gm = 2·Id/Vov` holds
to 1e-12.

**In circuit.**  For the diode-connected M1 of the demo mirror, the reported
drain current matches `½·β·Vov²·(1+λ·Vds)` recomputed from the model card and
the operating point to 1e-9 relative.

**Polarity.**  A PMOS with mirrored terminal voltages carries exactly the same
current as the equivalent NMOS (1e-12 relative), confirming the sign-flip
frame described in [mosfet_model.md](mosfet_model.md).

## 4. The current mirror

With matched devices the only nominal copy error is channel-length modulation:

```
Iout/Iref = (1 + λ·Vds2) / (1 + λ·Vds1)
```

| | value |
| --- | --- |
| measured `Iout/Iref` | 1.000819827401 |
| `(1+λVds2)/(1+λVds1)` | 1.000819827401 |

Agreement to 12 digits.  `I(M1)` reproduces the forced 10 µA to 1e-6, `V(out)`
satisfies `1.8 − 125 kΩ·I(M2)` to the solver's `reltol`, and the supply power
equals `1.8 V × (IREF + I(M2))`.

The KCL residual at the output node is **5.5e-13 A** against a 10 µA branch
current.

## 5. The differential pair

**Balance.**  Matched devices give `V(outp) = V(outn)` to 1e-9 V, equal drain
currents to 1e-9 relative, and `Vos = −1.1e-16 V` — zero to numerical noise.

**Tail node.**  `V(tail) = Vcm − Vgs`, with `Vgs` including the body-effect
threshold shift at that `Vsb`, to 1e-9.

**Differential gain.**  AC analysis against the small-signal prediction
`Ad = gm·(RL ∥ ro)`:

| | value |
| --- | --- |
| measured Ad | 23.9199 dB = 15.71 V/V |
| `gm·(RL ∥ ro)` | 15.70 V/V |

within 1 %.  Note that using `gm·RL` alone would give 16.28 V/V and be 3.5 %
off — the output resistance is not negligible against a 50 kΩ load.

## 6. The two-stage op-amp

| property | theory | measured | agreement |
| --- | --- | --- | --- |
| GBW = `gm1/(2π·Cc)` | 25.6 MHz | UGF 24.50 MHz | 4 % |
| `A0 · f3dB = funity` | — | 24.9 MHz vs 24.50 MHz | 1.6 % |
| `Av = gm1(ro2∥ro4)·gm6(ro6∥ro7)` | 86.4 dB | 86.94 dB | 0.5 dB |
| all devices saturated | — | M1–M8 all in saturation | ✓ |
| bias mirror ratios | `W/W8` | within 6 % (CLM) | ✓ |
| supply power | `1.8 V × ΣI` | 145.10 µW | 1e-4 |

The residual GBW discrepancy is the nulling resistor `RZ` and the second pole,
neither of which the single-pole formula accounts for.

## 7. The inverter — transient

The rise and fall edges are driven by one device each charging the same 1 pF
load, so their ratio should track the drive-strength ratio:

```
β_n / β_p  =  (246µ · 8)/(86µ · 16)  =  1.430
trise/tfall = 1.5455 ns / 1.0879 ns  =  1.421      agreement 0.6 %
```

The output overshoots to 1.829 V and undershoots to −10.1 mV, which is
gate-drain capacitive coupling of the input edge — real, and reproduced
because the Meyer capacitances are in the transient companion model.

## 8. AC and transient against closed forms

**RC low-pass, AC.**  `H(f) = 1/(1 + j2πfRC)` over five decades, `rtol=1e-9`.
Phase at `f3dB` is −45.0000° to 1e-4.

**RC low-pass, bandwidth.**

```
f3dB measured  159155.002614 Hz
     analytic  159154.943092 Hz        relative error 3.7e-07
```

**RL high-pass, AC.**  `H = jωL/(R + jωL)`, `rtol=1e-9`.

**RC step, transient.**  `v(t) = 1 − exp(−t/RC)`; maximum deviation over 5τ is
below 1e-4 at `Δt/τ = 0.005`.

**RL current ramp, transient.**  `i(t) = (V/R)(1 − exp(−tR/L))`; maximum
deviation below 2e-5.

## 9. The random-number layer

**Gaussian.**  100 000 samples: mean and standard deviation within 5 standard
errors of the requested values; skewness and excess kurtosis within 5 SE of 0;
the fraction beyond 3σ within 5 SE of 0.0027.

**Uniform.**  Bounded, `σ = a/√3`, and `Uniform.from_sigma` reproduces a
requested sigma.

**Log-normal.**  `from_relative_sigma` reproduces the requested standard
deviation exactly (analytically) and to sampling error empirically, for σ from
1 % to 50 %.  The distribution's non-zero mean is reported rather than hidden.

**Correlation.**  A requested 3×3 correlation matrix with entries 0.7, 0.2 and
−0.3 is recovered from 200 000 generated samples to within 0.012 — about 5
standard errors of a sample correlation.  Independent variables stay
independent to within 0.02.

## 10. Pelgrom scaling

Injected sigma is `AVT/√(2WL)`, and drawing both devices of a pair 20 000 times
confirms that the *difference* has `σ = AVT/√(WL)` to within 3 %.

At circuit level, 1500 mismatch-only samples per geometry:

| W×L | area | σ(ierr) | 1/√area prediction |
| --- | ---: | ---: | ---: |
| 10×1 | 10 µm² | 2.3318 % | reference |
| 20×2 | 40 µm² | 1.1662 % | 1.1659 % |
| 40×4 | 160 µm² | 0.5831 % | 0.5830 % |

Better than 0.05 % agreement across a 16× area range.  The two control cases
that deliberately change `W/L` behave as the physics requires and not as the
naive law suggests — see [pelgrom.md](pelgrom.md).

## 11. Mismatch against first-order theory

| circuit | predicted | measured | agreement |
| --- | ---: | ---: | ---: |
| current mirror, σ(ΔI/I) | 2.3061 % | 2.3502 % | 1.9 % |
| differential pair, σ(Vos) | 815.07 µV | 824.29 µV | 1.1 % |

Both predictions are assembled from the operating point and the model card's
Pelgrom coefficients only — no fitted constants.  Derivations in
[mismatch.md](mismatch.md).

## 12. Reproducibility

- The same seed reproduces every sample's draws, measurements and verdicts
  exactly.
- A different seed produces different samples with a statistically
  indistinguishable distribution.
- Sample 37 re-simulated in isolation matches sample 37 inside a 120-sample
  run, bit for bit.
- A 150-sample run's first 60 samples equal a 60-sample run with the same seed.
- `workers=2` matches `workers=1` exactly.
- Through the REST API: reproducing an 80-sample run compared 560 values with
  **0 mismatches**.

## 13. ngspice cross-validation — performed

Cross-validated against **ngspice 46** (released 2026-03-29), Windows x64,
extracted portably into `tools/Spice64` — no system install.

`docs/scripts/ngspice_crosscheck.py` does not maintain a second set of decks.
It **converts** each netlist in `examples/` into ngspice syntax, so both
simulators see the same circuit by construction and a disagreement cannot be
explained away as "the two netlists differed". The conversion drops
SiliconStat-only constructs (`.measure`, `.spec`, `MATCH=`, the Pelgrom
coefficients) and adds `LEVEL=1`.

The comparison is against ngspice's own **level-1** MOSFET model. Both tools
then implement the same equations, so agreement should be to solver tolerance
and any real difference indicates a bug in one of them.

```
23 quantities compared across 5 circuits
worst relative difference: 6.549e-07  (two_stage_opamp/Id(M6))
```

| circuit | quantities | worst relative difference |
| --- | ---: | ---: |
| resistive divider | 2 | 1.7e-16 |
| diode bias | 2 | 3.2e-07 |
| current mirror | 6 | 5.5e-08 |
| differential pair | 6 | 2.6e-07 |
| two-stage op-amp | 7 | 6.5e-07 |

Node voltages agree to between 1e-10 and 1e-8 relative; currents to about
1e-7. The current figure is the floor set by ngspice's own default
`reltol = 1e-3`, not by either solver's arithmetic — SiliconStat runs at
`reltol = 1e-6` (see [circuit_solver.md](circuit_solver.md)).

Small-signal parameters were compared too, not just node voltages:
`gm(M1)` and `gds(M1)` of the current mirror agree to 5e-10 and 1e-9.

Reproduce with:

```bash
.venv/Scripts/python.exe docs/scripts/ngspice_crosscheck.py --keep
```

The script exits 2 with an explanation if ngspice is absent, and never
fabricates ngspice output.

### What the cross-check found

It caught a real portability defect, which is the point of doing it.

The first run agreed to 1e-7 on four circuits and disagreed by up to **194 %**
on the two-stage op-amp. The four that matched were NMOS-only; the op-amp was
the only circuit containing PMOS devices — exactly the signature of a
polarity problem.

The cause: SPICE convention gives a PMOS a **negative** `VTO`. The shipped
model cards said `VTO=0.45`. SiliconStat normalises `VTO` with `abs()` and read
that as an enhancement PMOS with |Vth| = 0.45; ngspice took the sign literally
and built a *depletion* PMOS that conducts at Vgs = 0. The two tools were
simulating different devices, and ngspice's answer showed it plainly — M3 and
M6 carrying 20 µA and 44 µA at Vsg of 26 mV and 2 nV.

Neither tool was arithmetically wrong. The netlist was ambiguous, and only a
second implementation could reveal it. The fix was to write the shipped cards
in the portable SPICE convention (`VTO=-0.45` for PMOS). SiliconStat's own
results are byte-identical either way — `abs()` sees to that, and
`tests/test_mosfet.py` asserts it — but the cards now mean the same thing in
any SPICE. `tests/test_ngspice.py` locks the convention in so it cannot
regress.

### One convention difference that is not a bug

ngspice reports a PMOS drain current in the p-type frame (positive when
conducting). SiliconStat reports the current flowing *into the physical drain
terminal*, which is negative for a PMOS. Both are defensible; the harness
compares magnitudes for that one quantity and says so in its output.

## 14. ngspice cross-validation — whole waveforms

Section 13 compared operating points: single numbers. This section compares
entire traces, because a DC match does not exercise the reactive stamps or the
integration at all. `docs/scripts/ngspice_waveforms.py` converts the same
example netlists with the same translator, asks ngspice for the identical
analysis card, dumps its vectors with `wrdata`, and resamples them onto
SiliconStat's grid. `tests/test_ngspice.py` asserts the results.

### AC — `two_stage_opamp.net`, `.ac dec 12 1 1G`

Both tools linearise the same LEVEL=1 model about the same operating point and
solve the same complex MNA system, so there is no integration error to hide
behind. Measured over all 109 frequency points:

| trace | max &#124;difference&#124; | rms |
| --- | --- | --- |
| V(out)/V(inp) magnitude | 3.334e-05 dB | 1.021e-05 dB |
| V(out)/V(inp) phase | 1.255e-04 deg | 3.860e-05 deg |
| V(n2) magnitude | 2.749e-05 dB | 8.286e-06 dB |
| V(nz) magnitude | 9.035e-05 dB | 3.196e-05 dB |

The two numbers a designer signs off on agree to every digit printed:

| | SiliconStat | ngspice |
| --- | --- | --- |
| DC gain | 86.9434 dB | 86.9434 dB |
| Unity-gain frequency | 24.4915 MHz | 24.4915 MHz |

### Transient — `inverter_transient.net`, `.tran 20p 8n`

This one is expected to differ, and the test asserts a bounded difference
rather than equality. SiliconStat integrates on a fixed 20 ps grid (see
`limitations.md`); ngspice chooses its own timestep from a local truncation
error estimate and returned 421 points against our 402.

| trace | max &#124;difference&#124; | % of span | rms |
| --- | --- | --- | --- |
| V(in) | 9.044e-12 V | 0.0000 % | 6.100e-13 V |
| V(out) | 1.182e-02 V | 0.6429 % | 2.472e-03 V |
| V(vdd) | 0 V | 0.0000 % | 0 V |

The stimulus and the supply are algebraic, not integrated, so they match to
round-off. The 11.8 mV worst case on `V(out)` occurs at t = 1.04 ns — the
falling edge, where the waveform is steepest and a fixed grid pays the most for
its rigidity. Everywhere away from the two edges the traces sit within 2 mV.
Both tools show the gate-drain coupling overshoot above the rail (1.8285 V vs
1.8188 V) and both dip below ground on the falling edge.

### ngspice's own plots

`docs/scripts/ngspice_plots.py` runs ngspice's `hardcopy` command so the
figures in `out/ngspice/ng_*.png` are drawn by ngspice's renderer, not by
re-plotting exported data. The shipped Windows build has no PNG device but does
have SVG, which is rasterised afterwards.

One difference in those figures is presentation only: ngspice plots phase
**wrapped** to (−180°, 180°], so `vp(out)` jumps from −180° to +180° near
3e8 Hz. The dashboard unwraps, continuing to −224.6°. The numeric comparison
above unwraps both before differencing.

### A units trap worth recording

ngspice's `vp()` returns **radians** wrapped to (−π, π], not degrees, even
though the plot axis produced with `set units=degrees` is labelled "Degree".
Converting it again with `np.radians()` rescales the entire phase response and
produced an apparent 227° disagreement that was purely an artefact of the
harness. `test_ngspice_vp_is_radians_not_degrees` pins the convention.
