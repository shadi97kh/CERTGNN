# Epistasis spectra of three landscapes, side by side

git SHA `2bfb7d4`, config `3353d957`. This is a measurement, not a test: the rank-one discriminator itself is prior art (Husain & Murugan 2020, see `paper/prior_art/spectral_discriminator.md`). The question here is only whether rank structure differs across landscapes at all.

**Read the double-centered rows.** Measurement error on the single mutants enters E as `-(e 1ᵀ + 1 eᵀ)`, which is symmetric of rank 2 and would imitate the low-rank structure under test. Double-centering removes any additive row/column term and maps a rank-one signal to a rank-one signal, so it removes that artifact without removing the effect.

**The null is Wigner, not Marchenko–Pastur.** MP describes a sample covariance matrix; E is not a covariance but a symmetric matrix of direct measurements, so the matching bulk edge is `2·s·√n`.

| landscape | singles | pairs (obs/possible) | λ₁/λ₂ | participation ratio | eigs for 90% mass | rank-1 var. expl. vs size-matched null | outside bulk |
|---|---|---|---|---|---|---|---|
| FAS exon 6 | 189 | 16,728/17,577 (5% missing) | **1.25** | **114.3** | **121** of 189 | **0.090** vs null 0.021 (**4.3x**) | 135 eigs, 94% of mass |
| GB1 | 1045 | 530,737/536,085 (1% missing) | **1.86** | **340.1** | **390** of 1045 | **0.128** vs null 0.004 (**33.5x**) | 596 eigs, 99% of mass |
| eqFP611 (red) | 13 | 78/78 (0% missing) | **1.49** | **7.3** | **9** of 13 | **0.471** vs null 0.269 (**1.7x**) | 0 eigs, 0% of mass |
| eqFP611 (blue) | 13 | 78/78 (0% missing) | **1.16** | **5.8** | **6** of 13 | **0.369** vs null 0.269 (**1.4x**) | 5 eigs, 89% of mass |

| landscape | raw λ₁/λ₂ | raw part. ratio | raw rank-1 | ‖v₁ vs β‖ alignment | GE fit R² | noise SD | Wigner edge |
|---|---|---|---|---|---|---|---|
| FAS exon 6 | 1.13 | 103.7 | 0.131 | **0.422** | 0.613 | 0.1624 | 4.466 |
| GB1 | 1.36 | 282.2 | 0.211 | **0.513** | 0.449 | 0.1847 | 11.941 |
| eqFP611 (red) | 2.04 | 5.4 | 0.668 | **0.188** | 0.827 | 0.0997 | 0.719 |
| eqFP611 (blue) | 1.72 | 5.7 | 0.533 | **0.750** | 0.940 | 0.0560 | 0.404 |

**The null column is what makes the three comparable.** A pure-noise symmetric matrix has a rank-one fraction of roughly 4/n — about 0.31 at n=13, 0.021 at n=189, 0.004 at n=1045 — so the raw fraction largely measures matrix size, and does so in the direction that flatters the smallest landscape. Each null uses that landscape's own noise scale and its own observed-pair mask, and passes through the same double-centering.

`‖v₁ vs β‖` is |⟨v̂₁, β̂⟩| for unit vectors, on the double-centered spectrum. Under the global-epistasis reading v₁ must be proportional to β, so a high rank-one fraction with a LOW alignment would refute that reading. β comes from an additive-latent-plus-monotone-nonlinearity fit implemented here, because neither MAVE-NN nor MoCHI is installed in this environment; it is the same model those packages fit.

- **FAS exon 6** — noise: SD across 3 replicate enrichment scores (median 0.0938), x sqrt(3) for the three terms in E. Enrichment scores are relative to wild type, so y_0 = 0 exactly.
- **GB1** — noise: Poisson counting noise on the log selection ratio: median SD 0.1841 for doubles and 0.0104 for singles, combined as sqrt(sd_d^2 + 2 sd_s^2). Fitness from raw counts relative to wild type; input count >= 10.
- **eqFP611 (red)** — noise: Poisson counting on red counts, median relative SD 0.1715 scaled by the brightness spread, x2 for the four terms in E. 13 single mutants, so E is 13x13: far too small for an asymptotic random-matrix edge, included for the rank comparison only.
- **eqFP611 (blue)** — noise: Poisson counting on blue counts, median relative SD 0.0756 scaled by the brightness spread, x2 for the four terms in E. 13 single mutants, so E is 13x13: far too small for an asymptotic random-matrix edge, included for the rank comparison only.

## Verdict — RETRACTED

**This direction is closed. The measurements below stand; the conclusion drawn
from them does not, and no claim in this table should be cited.**

**1. Closed by prior art.** The rank-one result is already published: Husain &
Murugan, "Physical Constraints on Epistasis", *Mol Biol Evol* 37(10):2865 (2020).
Their Equation 2 states the epistasis matrix as a rank-1 term plus a sparse
contact term, they use deviation from rank-one as their discriminating statistic
(their "epistatic complexity"), and they apply an SVD low-rank decomposition to
the GB1 epistasis matrix of Olson et al. (2014) — the same matrix analysed here.
See `paper/prior_art/spectral_discriminator.md` for the verbatim quotations and
the full check, including why Poelwijk et al. (2019) is *not* the collision and
why the latent-dimensionality reframing is also claimed (LANTERN, PNAS 2021).

**2. A false claim about eqFP611 (red).** The retracted verdict named eqFP611
(red) as showing "substantial rank-one structure whose leading eigenvector is NOT
aligned with β". That is wrong. eqFP611 (red) has **zero eigenvalues outside its
own noise bulk** — its entire spectrum lies inside its noise floor. Its rank-one
fraction of 0.471 and its β-alignment of 0.188 are both consistent with noise, so
there is no structure there to be unexplained. Below the BBP threshold a spike
sticks at the bulk edge *and* its eigenvector overlap goes to zero, which is
exactly the observed signature; the branch that produced this claim did not gate
on the noise floor.

**3. The double-centering justification is wrong.** This table asserted that
double-centering "removes that artifact without removing the effect". It removes
a large part of the effect. Centering maps `β βᵀ` to `(β − β̄)(β − β̄)ᵀ`, whose
norm carries `Var(β)` in place of `E[β²]`; in DMS the mean effect `|β̄|` routinely
exceeds `sd(β)`, so centering **degrades the signal-to-noise ratio of exactly the
quantity under test** while leaving the noise bulk edge unchanged. It also fails
to preserve rank one on the hollow matrix actually formed, since the zero
diagonal contributes a term of rank up to n — an O(1/n) effect, negligible at
n=1045 but 8–15% at n=13, which is where this table's eqFP611 claims live.

Further defects found by the proof-check and not repaired: the noise
decomposition omits the wild-type error term `η₀·11ᵀ`, which is rank one and
non-zero for GB1 (contaminating the raw, though not the double-centred, columns);
the Wigner null is misspecified, predicting one outlier where 135 of 189 and 596
of 1045 were observed; and every number here is a single run, contrary to this
project's five-seed minimum.

The code in `experiments/epistasis_spectrum.py` and the run directory are kept so
the measurements remain reproducible and auditable. They are not evidence for any
claim about mechanism.
