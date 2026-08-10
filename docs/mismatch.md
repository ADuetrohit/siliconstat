# The mismatch and process-variation model

Layer 4 of the [architecture](architecture.md).  This layer turns a
*declarative* description of variation into concrete perturbed circuits.

## Global versus local

The single most important distinction in the whole tool.

**Global (process) variation** is one draw per model card, applied identically
to every device that references it.  It represents die-to-die and lot-to-lot
spread: the whole wafer came out a bit fast or a bit slow.

**Local (mismatch) variation** is an independent draw per device instance.  It
represents within-die random fluctuation — dopant number, line-edge roughness,
oxide granularity — between transistors that are millimetres or micrometres
apart on the same die.

For two NMOS transistors sharing a model card:

```
Vth(M1) = Vth_nom + ΔVth_global(NCH) + ΔVth_local(M1)
Vth(M2) = Vth_nom + ΔVth_global(NCH) + ΔVth_local(M2)
```

They share the global term **exactly** and differ only in the local terms.
That is the entire reason a current mirror works at all: the global term
cancels in the ratio, and only the local terms produce a copy error.

The measured consequence, on `examples/current_mirror.net`, 2000 samples,
seed 12345:

| mode | passives varied | σ(copy error) |
| --- | --- | ---: |
| process only (global, no local) | yes | 0.2463 % |
| process only (global, no local) | no | **0.1775 %** |
| mismatch only (local, no global) | yes | 2.3236 % |
| mismatch only (local, no global) | no | 2.3244 % |
| process + mismatch (the default) | yes | **2.3502 %** |

Two things to read off this table.  First, global variation contributes
essentially nothing to a *matched* structure: 0.18 % against 2.32 %, and what
remains is not mismatch at all but bias-point re-arrangement — a common
threshold shift moves `nref` and `out`, which changes the
channel-length-modulation ratio slightly.  Second, the load resistor's own 2 %
global tolerance is the larger part of even that small residual, which is why
the "devices only" row is smaller.

`tests/test_variation.py` asserts both halves of the global/local split
directly on the sample records: devices sharing a model card receive
*identical* global draws, and their local draws are uncorrelated.

## Rules and slots

A `VariationModel` is a list of `ParameterVariation` rules.  Each says which
devices vary, in which parameter, by how much, from which distribution, and at
which scope:

```python
ParameterVariation(
    parameter="vth",           # vth | beta | w | l | r | c | dc | is | n
    scope="local",             # global | local
    distribution="gaussian",   # gaussian | uniform | lognormal
    pelgrom=True,              # derive sigma from AVT and device area
    targets="type:mosfet",     # selector, see below
    share_by="model",          # global scope: model | matched_group | all
    correlation_group=None,
    truncate_sigma=None,
)
```

The sampler expands rules into **slots** — the independent random variables of
the experiment — with deterministic names:

```
NCH.vth_global      one draw, shared by every NCH device
NCH.beta_global
M1.vth_local        one draw each
M2.vth_local
M1.beta_local
M2.beta_local
ALL.r_global
R1.r_local
```

These names are the parameter axis of every downstream analysis: the
correlation matrix, the sensitivity ranking and the `var.*` columns of the CSV
export all use them.

### Selectors

| Form | Matches |
| --- | --- |
| `*` | every device |
| `model:NCH` | every device referencing that model card |
| `group:MIRROR` | every MOSFET in that matched group |
| `type:resistor` | every device of that class |
| `M1` or `device:M1` | one device by name |
| `M1,R1` | union of several |

A selector that matches nothing is an error, and the message lists what *is*
available — the models, the matched groups or the device names — because a
silently empty variation is the kind of mistake that produces a beautifully
converged run with no spread in it.

### Matched groups

`share_by="matched_group"` makes a *global* rule share its draw within a
`MATCH=` group rather than across a whole model card:

```
M1 n1 inn tail 0 NCH W=20u L=1u MATCH=INPAIR
M2 n2 inp tail 0 NCH W=20u L=1u MATCH=INPAIR
```

produces `INPAIR.vth_global`, covering exactly M1 and M2.  This is how you
express "these two devices are laid out together, so they share a systematic
gradient that other NCH devices elsewhere on the die do not".  A device with no
`MATCH=` falls back to its model card.

## Distributions and the Gaussian copula

Three families are supported: `gaussian`, `uniform`, `lognormal`.

They are unified by requiring each to implement `from_standard_normal(z)`.
Correlated standard normals are generated once, then each slot pushes them
through its own marginal transform — a **Gaussian copula**.  This is what lets
a correlation group mix a Gaussian threshold with a log-normal resistor and
still mean something.

| distribution | transform | notes |
| --- | --- | --- |
| Gaussian | `σ·z` | mean zero |
| Uniform | `a·(2·Φ(z) - 1)` | bounded, `a = σ·√3` from a target σ |
| LogNormal | `exp(s·z) - 1` | strictly `> -1`; relative parameters only |

Rank correlation is preserved *exactly* by a copula; linear correlation is
preserved exactly for Gaussian marginals and approximately otherwise.

The log-normal σ conversion is worth a note.  For `X = exp(N(0,s))`,
`Var(X) = (e^{s²} - 1)·e^{s²}`, so requiring `std(X) = σ` gives, with
`u = e^{s²}`:

```
u² - u - σ² = 0    ⟹    u = (1 + sqrt(1 + 4σ²)) / 2
```

This is **not** the more commonly quoted `s = sqrt(ln(1 + σ²))`, which pins the
*coefficient of variation* `std/mean` rather than the standard deviation; the
two differ by `sqrt(1 + σ²)`.  Since the variation model specifies sigmas, the
exact inversion is used.

A log-normal deviation is bounded below by −1 and is therefore meaningless as
an additive threshold shift; the rule constructor rejects that combination
outright rather than producing quiet nonsense.

## Correlated variation

Slots sharing a `correlation_group` are drawn from a multivariate normal:

```python
VariationModel(
    variations=[ParameterVariation(parameter="vth", scope="local", sigma=5e-3,
                                   targets="type:mosfet",
                                   correlation_group="pair")],
    correlations=[CorrelationSpec("pair", rho=0.8)],
)
```

The matrix is validated before use — symmetry, unit diagonal, entries in
[−1, 1], and positive semi-definiteness — and each failure names its cause.  A
non-PSD matrix gets the explanation it deserves: *"The requested set of
pairwise correlations is mutually inconsistent — no set of random variables can
have all of them simultaneously."*  For equicorrelation with `n` variables the
achievable range `ρ > −1/(n−1)` is checked explicitly.

Factorisation is Cholesky, with a jitter fallback so a perfectly correlated
pair (a singular but valid correlation matrix) still works.
`nearest_correlation_matrix` implements Higham's alternating projection as an
*explicit* repair step — it is never applied silently.

Empirically, generated variables recover a requested 3×3 correlation matrix to
within 0.012 at 200 000 samples (`tests/test_correlation.py`).

## Override convention: modifiers, not replacements

Slots accumulate into device overrides, and those overrides are always
**modifiers**:

| override | meaning |
| --- | --- |
| `dvth` | additive threshold shift, volts: `Vth = Vth(T, corner) + dvth` |
| `beta_scale` | multiplicative: `KP = KP(T, corner) · beta_scale` |
| `w`, `l` | absolute perturbed geometry, metres |
| `r`, `c` | absolute perturbed component values |

Modifiers rather than absolute replacements, because the statistical shift has
to *compose* with the corner shift and the temperature scaling that are applied
later in the pipeline.  If `vth` were replaced outright, a Monte Carlo run at
125 °C would silently lose its temperature dependence.

The ordering is: nominal card → corner shift → temperature scaling →
statistical modifier.

Additive parameters accumulate by summation; relative parameters accumulate as
`(1 + Σδ)`.  A draw that would produce a non-physical device — negative width,
zero or negative current factor — raises `VariationError` naming the device and
the offending value, and the Monte Carlo engine records that sample as a
`variation_error` rather than pretending it simulated.

## Area scaling

Local `vth` and `beta` rules default to `pelgrom=True`, which derives each
device's sigma from its own `W × L` and the model card's `AVT`/`ABETA`.  The
sqrt(2) convention and the empirical verification are in
[pelgrom.md](pelgrom.md).

## Verification against theory

The point of the whole layer is that circuit-level spread should match
first-order mismatch analysis.  Both demo cells do, at 2000 samples:

**Current mirror.**  σ(ΔI/I) ≈ `(gm/Id)·σ(ΔVth) ⊕ σ(Δβ/β)`, divided by the
channel-length-modulation feedback factor introduced by the load resistor:

```
gm/Id              22.654 1/V
σ(ΔVth) pair       1.1068 mV
σ(Δβ/β)            0.3162 %
CLM feedback       1.0959
                   ------
predicted          2.3061 %
measured           2.3502 %          agreement 1.9 %
```

**Differential pair.**  σ(Vos) ≈ `σ(ΔVth) ⊕ (Vov/2)·σ(Δβ/β) ⊕ (Vov/2)·σ(ΔR/R)`:

```
ΔVth term          782.62 µV        (Vov = 61.4 mV)
Δβ term             68.65 µV
ΔR term            217.09 µV
                   ---------
predicted          815.07 µV
measured           824.29 µV         agreement 1.1 %
```

Reproduce both with `docs/scripts/collect_doc_numbers.py`.
