# Pelgrom's law and area scaling

## The physics

A transistor's threshold voltage is set by the average of a large number of
microscopic random events — individual dopant atoms in the depletion region,
grain boundaries in the polysilicon, atomic-scale roughness at the oxide
interface.  Averaging `N` independent contributions gives a standard deviation
falling as `1/√N`, and `N` is proportional to the active area.  So

```
σ(ΔVT) = A_VT / sqrt(W · L)
```

Pelgrom's 1989 result is that this holds remarkably well across processes,
that the coefficient `A_VT` is a process constant, and that a second term
proportional to the *distance* between the devices accounts for systematic
gradients.  SiliconStat models the area term; the distance term is not
modelled, because the netlist carries no layout coordinates.

The same form applies to the current factor:

```
σ(Δβ/β) = A_β / sqrt(W · L)
```

## The sqrt(2) convention

This is the easiest thing in a mismatch tool to get wrong, so it is stated
explicitly and tested empirically.

`A_VT` as quoted by foundries and in Pelgrom's paper describes the standard
deviation of the **difference between a matched pair**.  A Monte Carlo engine
perturbs each device independently, so the sigma it must *inject per device* is
smaller by √2:

```
σ_device = A_VT / sqrt(2 · W · L)
```

Then, for two independently perturbed devices,

```
σ(ΔVT) = sqrt( σ_device² + σ_device² ) = √2 · σ_device = A_VT / sqrt(W·L)   ✓
```

reproducing the quoted figure.  `pair_convention=False` switches to the other
reading (`A_VT` as a single-device sigma) if your model card is documented that
way.  Getting this wrong scales every reported sigma by 1.41 — enough to turn
a 90 % yield into a 70 % one.

`tests/test_pelgrom.py` checks both the algebra and the empirical result: draw
both devices 20 000 times and confirm that the *difference* has
`σ = A_VT/√(W·L)` to within 3 %.

## Units

Universally quoted in engineering units rather than SI, and SiliconStat
follows that convention:

| symbol | unit | typical 180 nm value | model card |
| --- | --- | --- | --- |
| `A_VT` | **V·µm** | 3.5 mV·µm (NMOS), 4.2 mV·µm (PMOS) | `AVT=3.5m` |
| `A_β` | **dimensionless·µm** | 1.0 %·µm | `ABETA=0.010` |

`W` and `L` are converted to micrometres before the division.  So a
10 µm × 1 µm NMOS with `AVT=3.5m` gets

```
area      = 10 µm²
σ pair    = 3.5e-3 / √10          = 1.107 mV
σ device  = 3.5e-3 / √20          = 0.783 mV
```

## Worked example: the demo current mirror

`examples/current_mirror.net` biases two 10 µm × 1 µm NMOS devices at 10 µA.

```
gm/Id           = 22.654 1/V          (from the operating point)
σ(ΔVth) pair    = 1.107 mV
σ(Δβ/β) pair    = 0.316 %

σ(ΔI/I) ≈ (gm/Id)·σ(ΔVth) ⊕ σ(Δβ/β)
        = 2.508 % ⊕ 0.316 %  =  2.528 %
```

divided by the 1.096 feedback factor the load resistor introduces through
channel-length modulation, giving **2.306 %** predicted against **2.350 %**
measured over 2000 samples — 1.9 % agreement.

The threshold term dominates by a factor of eight in sigma, i.e. 63× in
variance, which is why the sensitivity analysis attributes 93 % of the
variance to the two `vth_local` slots.

## Does the law actually show up? (measured)

`docs/scripts/collect_doc_numbers.py` runs 1500 mismatch-only samples per
geometry on the demo mirror and reports the copy-error sigma:

| W (µm) | L (µm) | area (µm²) | σ(ierr) measured | 1/√area prediction |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 10 | 2.3318 % | 2.3318 % (reference) |
| 20 | 2 | 40 | **1.1662 %** | 1.1659 % |
| 40 | 4 | 160 | **0.5831 %** | 0.5830 % |
| 10 | 4 | 40 | 0.5946 % | 1.1659 % |
| 40 | 1 | 40 | 2.3214 % | 1.1659 % |

The first three rows scale **W and L together**, holding `W/L` — and therefore
the overdrive — fixed.  They reproduce `1/√area` to better than 0.05 %.
Quadrupling the area halves the mismatch, exactly as advertised.

`docs/scripts/pelgrom_area_sweep.py` runs the same experiment at five
geometries with 2000 samples each and fits the exponent directly:

```
SWEEP A -- W and L scaled together (W/L fixed, so Vov is fixed)
  log-log fit: sigma(ierr) ~ area^(-0.5000)   [Pelgrom predicts -0.5000]
  R^2 = 1.000000
```

An exponent of −0.5000 with `R² = 1.000000` across a 16× area range is about
as clean a confirmation of the law as a simulation can produce.

## The trap the law invites

The last two rows are the interesting ones, and they are the reason the sweep
is in the documentation at all.  Both quadruple the area, and neither behaves
like the naive reading of the law.

**Widening alone (40 × 1) barely helps: 2.32 % against 2.33 %.**  At a fixed
bias current, widening lowers the overdrive as `Vov ∝ 1/√W`, so
`gm/Id = 2/Vov` *rises* as `√W`.  The device is 2× better matched in threshold
and simultaneously 2× more sensitive to threshold mismatch.  The two cancel
exactly:

```
σ(ΔI/I) ≈ (gm/Id) · σ(ΔVth) ∝ √W · (1/√W) = constant
```

The control sweep in `docs/scripts/pelgrom_area_sweep.py` makes this
unmistakable — sixteen-fold more area, at a fixed current, buys 2.6 %:

```
SWEEP B (CONTROL) -- width only at fixed Id and L
   W [um]   area   Vov [mV]   gm/Id [1/V]   sigma(ierr) [%]
    2.5      2.5     176.0        11.36          2.3729
    5.0      5.0     124.7        16.04          2.3405
   10.0     10.0      88.3        22.65          2.3244
   20.0     20.0      62.5        32.01          2.3167
   40.0     40.0      44.2        45.23          2.3130

  log-log fit: sigma(ierr) ~ area^(-0.0089)   [Pelgrom predicts -0.5000]
```

An exponent of −0.009 instead of −0.5.  The `gm/Id` column is the mechanism:
it rises by exactly the factor the threshold matching improved by.

**Lengthening alone (10 × 4) helps more than the area law predicts: 0.59 %
against a 1.17 % prediction.**  Increasing `L` lowers `beta`, which *raises*
`Vov` and lowers `gm/Id`.  Now both terms push the same way — better matching
*and* lower sensitivity — so the improvement is roughly `L` rather than `√L`.

The engineering conclusion, which the numbers above demonstrate rather than
assert: **for current-source matching at a fixed current, spend area on length,
not width.**  What actually matters is `Vov`, and the honest form of the rule is

```
σ(ΔI/I) ≈ (2/Vov) · A_VT / sqrt(2·W·L)  ⊕  A_β / sqrt(2·W·L)
```

in which `W` appears in both the numerator (through `Vov ∝ 1/√W`) and the
denominator.

## Inverse Pelgrom: sizing for a target

`area_for_target_sigma()` answers the question the law is actually used for —
"how big must this pair be to hold offset under 2 mV?":

```python
from siliconstat.variation.pelgrom import area_for_target_sigma
area_for_target_sigma(3.5e-3, 2e-3)     # -> 1.53 µm² of active area per device
```

## Configuration

Pelgrom scaling is the default for local `vth` and `beta` rules.  To use fixed
sigmas instead — for a process whose mismatch you have characterised directly,
or to isolate an effect — pass `--no-pelgrom` on the CLI, set
`"pelgrom": false` in the API's variation block, or construct the rules with
explicit `sigma`/`sigma_pct` values.

Pelgrom scaling is rejected for *global* rules (die-to-die spread is not an
area effect) and for parameters other than `vth` and `beta`, with an error
that says why.
