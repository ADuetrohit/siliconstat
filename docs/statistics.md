# Statistics

Layer 6 of the [architecture](architecture.md).  Everything here is computed
from the stored per-sample records; nothing is carried over from a previous
run or defaulted.

## What is computed

For every measurement, over the samples that produced a finite value:

| quantity | notes |
| --- | --- |
| count, n_input, n_invalid | non-finite values are *excluded and counted*, never zeroed |
| mean, median | Welford for the mean |
| variance, standard deviation | sample (ddof = 1) |
| minimum, maximum, range | |
| coefficient of variation | `σ/\|µ\|`; infinite at zero mean |
| P1, P5, P25, P75, P95, P99, IQR | linear interpolation between order statistics |
| ±3σ limits | `µ ± 3σ` |
| skewness, excess kurtosis | standardised third and fourth moments |
| standard error of the mean | `σ/√n` |
| 95 % CI of the mean | `µ ± 1.96·SEM` |
| 95 % CI of the standard deviation | chi-square interval |
| normality p-value and test name | see below |

## Welford's algorithm

The mean and variance are accumulated with Welford's online recurrence

```
n   ← n + 1
δ   ← x - mean
mean ← mean + δ/n
M2  ← M2 + δ·(x - mean)
```

rather than the textbook one-pass form `(Σx² - (Σx)²/n)/(n-1)`.

The naive form loses precision by catastrophic cancellation when the mean is
large relative to the spread, and the loss goes roughly as `(µ/σ)²`.  Measured
on this code base:

| mean | sigma | µ/σ | n | naive rel. error | Welford rel. error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.2 | 300 µ | 4e3 | 20 000 | 1.6e-09 | 1.7e-13 |
| 1.2 | 3 µ | 4e5 | 20 000 | 9.0e-06 | 1.4e-11 |
| 1e6 | 1 m | 1e9 | 20 000 | **1.0** | 9.9e-09 |
| 1e8 | 1 m | 1e11 | 5 000 | **1.0** | 4.9e-06 |

The honest reading: at ordinary analog ratios the naive formula is merely
*worse*, by about four orders of magnitude; by `µ/σ ~ 1e9` it has lost
everything and returns a number wrong by more than 100 % (often exactly zero,
occasionally negative).  Welford costs nothing extra and is correct throughout,
so it is used unconditionally.  Reproduce the table with
`docs/scripts/collect_doc_numbers.py`.

## Percentiles and the CDF

Percentiles use linear interpolation between order statistics — NumPy's
default `linear` method, and the convention used by the EDA tools this is
meant to sit alongside.

The empirical CDF uses the **Hazen plotting position** `(i − 0.5)/n` rather
than `i/n` or `i/(n+1)`.  The choice matters at the tails, which is exactly
where you read a yield off the curve: `i/n` puts the largest sample at
probability 1.0, implying that nothing can ever exceed it, and `i/(n+1)` is
biased the other way.  Hazen is unbiased for the median and symmetric at both
ends.  A 1000-sample run's most extreme point therefore sits at 0.05 % and
99.95 %, which is an honest statement of what 1000 samples can resolve.

## Confidence intervals

**Mean.**  `µ ± 1.96·σ/√n`.  Normal rather than Student-t, because the sample
counts here (hundreds to tens of thousands) make the difference negligible —
at n = 30 the t-factor is 2.04 against 1.96, and the tool warns about small
samples in other ways.

**Standard deviation.**  The chi-square interval

```
σ · sqrt( (n-1) / χ²_{0.975, n-1} )   …   σ · sqrt( (n-1) / χ²_{0.025, n-1} )
```

which is asymmetric — the upper bound is further from the estimate than the
lower one.  Worth showing, because a sigma from 100 samples is much less
certain than people assume: the 95 % interval is roughly [0.88σ, 1.16σ].

**Proportions** (yield) use the Wilson score interval — see
[yield.md](yield.md).

## Shape and normality

Skewness and excess kurtosis are the standardised third and fourth moments,
with the normal reference at 0 and 0.

The normality test switches on sample size:

- **n ≤ 5000**: Shapiro–Wilk, the most powerful general test at these sizes.
- **n > 5000**: D'Agostino–Pearson.  Shapiro–Wilk's p-value is unreliable on
  very large samples and, more importantly, at n = 50 000 it will flag a
  departure so small that no engineering conclusion depends on it.

This matters because the ±3σ limits, the Cpk figures and any extrapolation
past the sampled range all assume a Gaussian.  A p-value below 0.05 says that
assumption is measurably violated and you should read the tail percentiles
instead.  For the demo current mirror the response is near-linear in the
inputs and the output is Gaussian (Shapiro–Wilk p ≈ 0.8); for a circuit with a
hard nonlinearity it will not be.

The **normal-quantile plot** in every report is the visual version: plot the
sorted data against the standard-normal quantile of its plotting position, and
a Gaussian sample is a straight line.  Curvature at one end is skew; an S-shape
is heavy or light tails.

## Correlation

Both Pearson (linear) and Spearman (monotonic rank) coefficients are computed
between every varied parameter and every measurement, and between measurements.

The pair is what makes it useful.  A large Spearman with a small Pearson is the
signature of a strong but *nonlinear* dependence — precisely the case where a
correlation-based sensitivity ranking would mislead you.  On a deliberately
exponential relationship, Spearman reads 1.000 and Pearson reads 0.83.

Columns with zero variance — a parameter that was not actually varied, a
measurement identical in every sample — give NaN and are **listed explicitly**
in `degenerate_parameters` / `degenerate_measurements` rather than quietly
dropped.  A silently missing row in a correlation matrix is how you fail to
notice that your variation model did nothing.

## Sensitivity

Two methods, and the distinction is not cosmetic.

### Correlation-based ranking

Order parameters by `|Pearson r|`.  Cheap, always available, adequate for
"what should I look at first".  It is **not** a variance decomposition:
correlations do not sum to anything meaningful, and with correlated inputs a
parameter can rank high purely by association.  When this method is used the
report says so.

### Variance-based decomposition (default)

Fit a standardised linear model `y* = Σ βᵢ xᵢ*` by least squares.  When the
inputs are mutually uncorrelated — which is how this project constructs them
unless you declare a correlation group — `βᵢ²` *is* the fraction of output
variance contributed by input `i`, and they sum to `R²`.

Two identities are worth separating, because only one is exact in finite
samples:

```
R² = Σ βᵢ · rᵢy          exact, for any design
R² ≈ Σ βᵢ²               exact only for exactly orthogonal inputs
```

At n = 800 the empirical correlation between two genuinely independent inputs
is still about `1/√n = 0.035`, which moves `Σβ²` by a couple of percent.  The
report shows `max_input_correlation` so you can see how orthogonal the design
actually was, and warns when it exceeds 0.3.

`1 − R²` is the share of variance the linear model cannot explain: genuine
nonlinearity plus solver noise.  The report shows it prominently and warns
below `R² = 0.8` that the ranking is indicative rather than exact.

**An underdetermined fit is refused rather than faked.**  With more parameters
than samples, least squares returns the minimum-norm solution and an `R²` of
exactly 1.0 — a number that looks like a perfect explanation and means nothing.
The analysis detects this, falls back to correlation ranking, and says why.

### On the demo current mirror

2000 samples, `R² = 0.99974`, maximum inter-input correlation 0.056:

| # | parameter | β* | variance share |
| --- | --- | ---: | ---: |
| 1 | `M2.vth_local` | −0.6826 | 46.60 % |
| 2 | `M1.vth_local` | +0.6825 | 46.59 % |
| 3 | `M1.beta_local` | −0.0867 | 0.75 % |
| 4 | `M2.beta_local` | +0.0852 | 0.73 % |
| 5 | `ALL.r_global` | −0.0762 | 0.58 % |
| 6 | `NCH.vth_global` | −0.0760 | 0.58 % |
| 7 | `R1.r_local` | −0.0177 | 0.03 % |
| 8 | `NCH.beta_global` | +0.0036 | 0.00 % |

The two local thresholds account for 93 % of the variance, in exact opposition
— raising M1's threshold raises the mirrored current, raising M2's lowers it.
Global variation contributes 0.6 %, because it is common-mode and cancels in
the ratio.  This is the textbook answer, arrived at from simulation.

## Convergence

Monte Carlo answers are estimates whose uncertainty shrinks only as `1/√N`, so
every report carries a convergence trace: the running mean, sigma and yield
with their confidence bands, evaluated at forty prefix lengths.

The samples are in a fixed, seed-determined order, so the trace is
reproducible rather than an artefact of scheduling.

`mean_settled_at` / `std_settled_at` / `yield_settled_at` report the first
prefix length after which the estimate stays within a tolerance (1 % by
default) of its final value.  If the mean has not settled, the trace says so —
which is the signal that the number you were about to quote needs more samples.
