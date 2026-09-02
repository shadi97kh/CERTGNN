# Which claim does the occurrence experiment actually support?

**Both numbers are right. The claim built on them was wrong.**

Figure 3 and the 8.8e7 figure do not contradict each other, because they are not
measuring the same thing. `n_separate` is computed by identical code in both
experiments, on the same noise model and the same scale; what differs is *which
pair of functions* is being compared. Independently trained accuracy-tied models
are far apart as functions. A fit and its own reparameterized twin are not.

The paper's original sentence --- that the data cannot adjudicate between
accuracy-matched models --- is **false for the population the paper is about**.
It was true only of the twin comparison, and was imported across.

---

## 1. Occurrence: independently trained, accuracy-tied pairs

Field `cells[*].n_separate_median`: the per-seed median over accuracy-tied pairs
of `crit / snr2`, aggregated over 10 seeds as mean with a 95% bootstrap interval.

**Units are held-out measurements.** `crit = (z_{1-a/2} + z_power)^2` is
dimensionless and `snr2 = mean((d_i / sigma_i)^2)` is dimensionless, so the
ratio is a count of points. It is not normalized and not a rate.

| cell | mean | 95% CI |
|---|---|---|
| 16x1 | 0.5530 | [0.4432, 0.6622] |
| 16x2 | 0.3382 | [0.2842, 0.4055] |
| 16x3 | 0.3067 | [0.2629, 0.3514] |
| 32x1 | 0.6376 | [0.5200, 0.7660] |
| 32x2 | 0.3944 | [0.3329, 0.4584] |
| 32x3 | 0.3035 | [0.2699, 0.3398] |
| 64x1 | 1.5071 | [1.0662, 2.0453] |
| 64x2 | 0.5476 | [0.4107, 0.7346] |
| 64x3 | 0.3126 | [0.2779, 0.3478] |
| 128x1 | 2.1064 | [1.6895, 2.5527] |
| 128x2 | 1.1683 | [0.8643, 1.5109] |
| 128x3 | 0.4540 | [0.3853, 0.5337] |

A value below 1 means a single held-out point already carries more than the
required power; the honest reading is "one measurement suffices", since a
fractional measurement is not a thing.

## 2. Separation: one fit against its own reparameterized twin

Field `cells[*].n_separate`, same formula, same units.

| cell | mean | 95% CI |
|---|---|---|
| 16x1 | 0.56 | [0.43, 0.71] |
| 16x2 | 42.18 | [24.59, 59.84] |
| 16x3 | 313.17 | [87.04, 570.83] |
| 32x1 | 0.49 | [0.35, 0.66] |
| 32x2 | 86.76 | [13.84, 192.58] |
| 32x3 | 833.34 | [18.42, 2135.54] |
| 64x1 | 1.87 | [0.99, 3.01] |
| 64x2 | 4176.29 | [1319.62, 7333.14] |
| 64x3 | 7299.69 | [2156.00, 14436.00] |
| 128x1 | 8.69 | [6.57, 10.93] |
| 128x2 | 3158529.05 | [1402633.36, 5004082.78] |
| 128x3 | 87751463.72 | [39140215.02, 138392840.08] |

## 3. Same test, same noise model, same scale

| | occurrence | separation |
|---|---|---|
| statistic | `crit / snr2` | `crit / snr2` |
| `snr2` | `mean(((preds_i - preds_j) / sd)^2)` | `mean(((y_twin - y_ref) / sd)^2)` |
| source | `occurrence.py:300,356` | `separation.py:528,606` |
| `crit` | `(z_{1-a/2} + z_power)^2` = 7.8489 | same |
| alpha / power | 0.05 / 0.8 | 0.05 / 0.8 |
| noise model | `noise_sd_standardized` | `noise_sd_standardized` |

The expressions are identical. Occurrence does not reimplement the noise model:
it **imports `noise_sd_standardized` and `phenotype_slope` from
`separation.py`** (`occurrence.py:94-104`), so it is the same code on the same
library and the same seeded split. Occurrence additionally raises at runtime if
its alpha differs from separation's (`occurrence.py:202-212`), which exists
precisely so the two cannot drift onto different scales.

**Conclusion: the difference is a property of the model pairs, not of the
method.**

## 4. Sanity check: is the order-1 result a units error?

No. Cell 128x1, seed 0, 156 accuracy-tied pairs:

| quantity | value |
|---|---|
| median RMS prediction difference \|d\| | 0.1598 (standardized units) |
| median noise scale sigma | 0.3413 |
| \|d\| / sigma | 0.468 |
| hand estimate `crit / (\|d\|/sigma)^2` | 35.8 |
| recorded median `n_separate` | 3.34 |

Both are order 1 to order 10. A units error would show as a factor of 1e3 or
more, and does not appear. The remaining 10.7x gap is expected and is not an
error: the hand estimate uses `(median|d| / median sigma)^2`, while the code
uses the pointwise `mean((d_i / sigma_i)^2)`. Because sigma is heavy-tailed
across the library --- `mean(1/sigma^2)` is 3.0x its median under the Poisson
log-ratio model --- the pointwise mean is dominated by low-noise points and is
larger than the naive ratio, which makes the required n correspondingly smaller.
The code's weighting is the correct one.

## 5. Why the two answers differ by eight orders of magnitude

Because the pairs differ by eight orders of magnitude in prediction distance.

| comparison | RMS prediction difference |
|---|---|
| twin vs reference, 128x3 | 0.000067 |
| twin vs reference, 128x1 | 0.136449 |
| tied independent pair, 128x1 | 0.1598 |
| tied independent pair, 64x3 | 0.8448 |

Median `|d| / sigma` for tied independent pairs runs 0.47 (128x1) to 2.48
(64x3). Their prediction differences are **comparable to or larger than the
observation noise itself**. Two such models are trivially separable. The twin at
128x3 differs by 6.7e-5, four orders below the noise, which is why it needs 8.8e7
measurements.

## 6. What the occurrence experiment actually supports

**Not** "the data cannot adjudicate between accuracy-matched models". It
supports something sharper:

> Two models tied on aggregate held-out accuracy are **easily distinguished as
> functions** --- one or two held-out measurements suffice --- and they
> nonetheless disagree about which positions matter for 43% to 94% of
> sequences.

The failure is therefore not that the data is too weak to choose. It is that
**the criterion the field actually uses does not choose**. Accuracy-tying is a
statement about aggregate squared error, resolvable only to a held-out R^2 gap
of 0.032 (128x1) to 0.127 (64x3). Two models can sit inside that gap while
making pointwise predictions that differ by more than the noise. Practitioners
then read attributions off whichever one they happened to train.

This is a better result for the venue than the original claim. "No amount of
data could settle it" invites the reply that a bigger assay would fix it. "The
data settles it in one measurement and nobody looks" does not.

The twin comparison keeps its own finding, and it is a real one: a fit and a
reparameterized twin *are* genuinely indistinguishable, up to 8.8e7 measurements
at 128x3. That belongs to the twin experiment and must be labelled as such.

## 7. Where the wrong claim still lives

Section 3 was already corrected. **The abstract has not been**, and still reads:

> "And the data required to adjudicate between two models, computed from the
> assay's own Poisson count noise rather than assumed, exceeds the full library
> by up to three orders of magnitude."

That sentence is the twin result attached to "two models", which in the
abstract's context means the accuracy-tied pairs. It is the last live instance
of the conflation and should be replaced. The tex has deliberately not been
patched; this note is the finding only.

One further consistency check that supports the corrected reading: across cells,
median `n_separate` and top-3 divergence are rank-correlated **-0.94**. The
pairs that are hardest to tell apart as functions are the ones that agree most
about which positions matter. That is the direction the corrected claim
predicts, and the opposite of what "indistinguishable models disagree most"
would require.

---

*Sources: `results/runs/20260830T002223Z_6f98de3_6e25cf7d` (occurrence, git
`6f98de3`, config `6e25cf7d`, 10 seeds) and
`results/runs/20260830T030657Z_972496a_5da8cb38` (separation, git `972496a`,
config `5da8cb38`, 10 seeds). Every number above is read from those run records
or derived from them by the arithmetic shown.*
