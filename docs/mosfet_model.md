# The MOSFET model

> **This is a simplified square-law model, in the SPICE level-1 tradition.  It
> is not BSIM, it is not calibrated to silicon, and it must not be used for
> tape-out decisions.**  It exists so that the statistical machinery above it
> operates on physically sensible device behaviour.  What is and is not
> modelled is listed at the end and in [limitations.md](limitations.md).

## Equations

All expressions are written in the **n-type frame**: terminal voltages are
pre-multiplied by the device sign (+1 NMOS, −1 PMOS), and drain and source are
exchanged if needed so that `vds ≥ 0`.

### Threshold with body effect

```
vth = VTO + GAMMA · ( sqrt(PHI - vbs) - sqrt(PHI) )
```

`PHI` is the surface potential `2·φF`.  For a reverse-biased bulk, `vbs ≤ 0`,
so the argument exceeds `PHI` and the threshold rises.  The argument is
clamped at 1e-6 to stay defined if the bulk junction is forward-biased.

### The three regions

With `vov = vgs - vth` and `clm = 1 + LAMBDA·vds`:

```
cutoff       vov ≤ 0        Ids = 0
triode       vds < vov      Ids = beta · (vov·vds - vds²/2) · clm
saturation   vds ≥ vov      Ids = (beta/2) · vov² · clm

beta = KP · Weff / Leff · M
```

### Why channel-length modulation multiplies *both* branches

Berkeley SPICE level 1 applies `(1 + LAMBDA·vds)` in triode as well as in
saturation, and so does SiliconStat.  The reason is continuity.  At the
boundary `vds = vov`, the triode expression's charge term is

```
vov·vds - vds²/2  →  vov² - vov²/2  =  vov²/2
```

so the triode current becomes `beta·(vov²/2)·clm`, which is exactly the
saturation expression.  Applying `clm` only in saturation would leave a step of
`beta·vov²/2 · LAMBDA·vov` at the boundary — a discontinuity the Newton
iteration would sit on and oscillate across.  `tests/test_mosfet.py` checks
that `Ids`, `gm` and `gds` are all continuous there.

### Small-signal derivatives

The solver needs an exact Jacobian, so these are analytic, not differenced:

```
triode        gm  = beta·vds·clm
              gds = beta·[ (vov - vds)·clm + LAMBDA·(vov·vds - vds²/2) ]

saturation    gm  = beta·vov·clm
              gds = (beta/2)·vov²·LAMBDA
```

and in saturation this reduces to the relation every analog designer uses:

```
gm = 2·Ids / vov
```

### The body transconductance

`gmb` follows from the chain rule through the threshold:

```
gmb = dIds/dvbs
    = (dIds/dvov) · (dvov/dvbs)
    = gm · ( -dvth/dvbs )
```

and since `vth = VTO + GAMMA·(sqrt(PHI - vbs) - sqrt(PHI))`,

```
dvth/dvbs = -GAMMA / ( 2·sqrt(PHI - vbs) )
```

so

```
gmb = gm · GAMMA / ( 2·sqrt(PHI - vbs) )
```

which is positive: pulling the source above the bulk raises the threshold and
therefore reduces the current, and `vbs` becoming *less* negative does the
opposite.  All three derivatives are verified against central finite
differences to 2e-5 relative in four bias conditions.

## PMOS: one sign, no duplicated code

A PMOS is evaluated by mapping into the n-type frame:

```
vgs' = sign·(vg - vs)     vds' = sign·(vd - vs)     vbs' = sign·(vb - vs)
Ids' = f(vgs', vds', vbs')                          I_d  = sign · Ids'
```

with `sign = -1`.  The important consequence is that **the conductance stamps
are identical for both polarities**, because the two sign flips cancel:

```
dI_d/dv_g = sign · gm · dvgs'/dv_g = sign · gm · sign = gm
```

So `Device.stamp_dc` has no polarity branch at all — only the current source
term `Ieq` carries the sign.  `tests/test_mosfet.py::
test_pmos_mirrors_nmos_under_a_sign_flip` checks that a PMOS with mirrored
terminal voltages carries exactly the same current as the equivalent NMOS.

### The PMOS `VTO` sign — a portability trap

`VTO` is stored as a **magnitude**.  Both the SPICE convention (`VTO=-0.45`
for a PMOS) and the magnitude convention (`VTO=0.45`) are accepted; the model
card normalises with `abs()`.  This precludes depletion devices, which are not
supported.

That forgiveness has a cost, and cross-validation against ngspice found it.  A
real SPICE takes the sign **literally**: a PMOS card with `VTO=+0.45` is a
*depletion* device that conducts at `Vgs = 0`.  So a card written the
magnitude way means one thing here and something entirely different anywhere
else, and nothing in SiliconStat's own output can reveal it — its results are
identical under either sign.

The op-amp disagreed with ngspice by up to 194 % until the shipped cards were
rewritten in the portable form.  See
[validation.md § 13](validation.md#what-the-cross-check-found).

**Write PMOS thresholds negative.**  Every card in `examples/` now does, and
`tests/test_ngspice.py` fails the build if one does not.

## Reverse mode

If `vds' < 0` the physical source is at the higher potential, so the model
re-references to the true source:

```
vgs' ← vgs' - vds'    (= vgd)
vbs' ← vbs' - vds'    (= vbd)
vds' ← -vds'
```

and reports `swapped=True`.  The caller applies the same exchange when
stamping and negates the reported drain current.  This is what lets a
transmission gate or a cross-coupled pair work.

## Temperature

```
VTH(T) = VTO + TCV · (T - TNOM)          TCV ≈ -0.9 mV/K
KP(T)  = KP  · (T/TNOM)^BEX              BEX  = -1.5
```

Both are first-order: the threshold falls roughly linearly with temperature,
and mobility (and therefore the current factor) falls as a power law.  `T` is
absolute in the `KP` expression, Celsius in the `VTH` expression.

The consequence that matters for mismatch: at a *fixed* bias current, a hotter
device has lower `beta`, therefore higher overdrive `vov = sqrt(2·Id/beta)`,
therefore lower `gm/Id = 2/vov` — and therefore *less* sensitivity to threshold
mismatch.  A current mirror's copy-error spread shrinks at high temperature,
which `tests/test_pvt.py` asserts as a prediction rather than a recorded value.

## Capacitances

Region-dependent Meyer capacitances, used by AC and transient only.  With
`Ctot = COX·Weff·Leff`:

| region | Cgs | Cgd | Cgb |
| --- | --- | --- | --- |
| cutoff | `CGSO·W` | `CGDO·W` | `Ctot + CGBO·L` |
| triode | `½·Ctot + CGSO·W` | `½·Ctot + CGDO·W` | `CGBO·L` |
| saturation | `⅔·Ctot + CGSO·W` | `CGDO·W` | `CGBO·L` |

`COX` is derived from `TOX` as `ε_ox/TOX` unless given directly.  These are
piecewise in the operating region, which is why transient analysis freezes
them for the duration of a timestep — see
[circuit_solver.md](circuit_solver.md#frozen-meyer-capacitances).

## Model card parameters

| Parameter | Meaning | Unit | Default |
| --- | --- | --- | --- |
| `LEVEL` | SPICE model level; only 1 is implemented | – | 1 |
| `VTO` | zero-bias threshold (write it **negative** for PMOS) | V | 0.45 |
| `KP` | transconductance parameter `µ·Cox` | A/V² | 200 µ |
| `LAMBDA` | channel-length modulation | 1/V | 0.10 |
| `GAMMA` | body-effect coefficient | V^½ | 0 |
| `PHI` | surface potential `2φF` | V | 0.8 |
| `LD`, `WD` | lateral diffusion / width narrowing per side | m | 0 |
| `TOX` | gate oxide thickness | m | 4 n |
| `COX` | oxide capacitance (overrides `TOX`) | F/m² | – |
| `CGSO`, `CGDO`, `CGBO` | overlap capacitances | F/m | 2e-10 |
| `CJD`, `CJS` | junction capacitance per unit width | F/m | 0 |
| `TNOM` | nominal temperature | °C | 27 |
| `TCV` | `dVth/dT` | V/K | −1 m |
| `BEX` | mobility temperature exponent | – | −1.5 |
| `AVT` | Pelgrom threshold coefficient | **V·µm** | 3.5 m |
| `ABETA` | Pelgrom current-factor coefficient | **1·µm** | 0.010 |

The last two use engineering units rather than SI because that is universally
how they are quoted; see [pelgrom.md](pelgrom.md).

Aliases are accepted: `VT0`/`VTH`/`VTH0` for `VTO`, `LAM` for `LAMBDA`,
`BETA`/`U0COX` for `KP`, `AB` for `ABETA`.

`LEVEL=1` is accepted so that a card written for any SPICE loads unchanged.
`LEVEL=2`, `3`, `49` and friends are **rejected** with a message naming the
limitation rather than a confusing "unknown parameter" — asking for BSIM and
silently getting square-law would be the worse failure.

## What is *not* modelled

Every one of these is present in a modern short-channel device and absent
here:

- **velocity saturation** — the drain current of a 180 nm device at high
  overdrive is closer to linear in `vov` than quadratic;
- **drain-induced barrier lowering (DIBL)** — the threshold falls with `vds`;
- **short-channel threshold roll-off** — `Vth` depends on `L`;
- **subthreshold conduction** — this model's cutoff current is exactly zero,
  so weak inversion, subthreshold slope and off-state leakage do not exist;
- **mobility degradation with vertical field** — `KP` is bias-independent;
- **gate leakage**, **GIDL**, **junction breakdown**, **self-heating**;
- **narrow-width effects**, **poly depletion**, **quantum corrections**;
- **charge-conserving capacitance** — the Meyer model is not charge-based, so
  transient charge is not conserved exactly.

The model cards shipped in `examples/` use plausible 180 nm-class numbers.
They are **not a foundry PDK**, are not extracted from measurement, and their
mismatch coefficients are representative literature values rather than
characterised ones.  Any absolute number this tool produces should be read as
"what this model implies", not "what silicon will do".  The *methodology* —
the Pelgrom scaling, the global/local split, the yield accounting — is the
part meant to transfer.
