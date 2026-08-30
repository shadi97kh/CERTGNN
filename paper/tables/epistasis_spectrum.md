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

## Verdict

**Rank structure does differ across landscapes.** The leading term explains 0.471 of the double-centered matrix in eqFP611 (red) and only 0.090 in FAS exon 6, a spread of 0.381, with effective rank 7.3 against 114.3. That is a real contrast, not a constant. **Against a size-matched null the ordering is not what the raw numbers suggest.** Pure noise alone would give a rank-one fraction of FAS exon 6 0.021, GB1 0.004, eqFP611 (red) 0.269, eqFP611 (blue) 0.269, purely because that fraction scales as 4/n. Relative to its own null each landscape sits at FAS exon 6 4.3x, GB1 33.5x, eqFP611 (red) 1.7x, eqFP611 (blue) 1.4x. Reading the raw column across landscapes of different size would invert this comparison, which is why it is not the column to read. **The control on the leading eigenvector.** Under the global-epistasis reading v₁ must be parallel to β. Alignment is FAS exon 6 0.422, GB1 0.513, eqFP611 (red) 0.188, eqFP611 (blue) 0.750. **eqFP611 (red) show substantial rank-one structure whose leading eigenvector is NOT aligned with β**, so for those the low-rank structure is not explained by a monotone nonlinearity on an additive trait, whatever else produces it. This is a description of three matrices, not a test of a mechanism, and it does not become one: the rank-one criterion and its application to GB1 are already published (Husain & Murugan, Mol Biol Evol 37:2865, 2020). Nothing here should be written up as a new discriminator.
