# How much data separates a fit from its reparameterized twin?

git SHA `972496a`, config `5da8cb38`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed, 3000 fit / 1000 held out.

**The claim, per cell.** *X*: how well the twin's latent agrees with the monotone reparameterization out of sample. *Y*: Spearman between the two fits' per-locus attribution magnitudes, which is what a reader ranking positions would see. *Z*: held-out measurements needed to reject that the two fits are the same function, at two-sided α = 0.05 with power 0.8.

| width | depth | **X** held-out R² | **Y** per-instance ρ (p10 / median / p90) | frac ρ<0.5 | **Z** n to separate |
|---|---|---|---|---|---|
| 16 | 1 | **0.984087** | +0.519 / **+0.892** / +0.978 | **9%** | **1** [0, 1] |
| 16 | 2 | **0.999241** | +0.873 / **+0.978** / +1.000 | **1%** | **42** [25, 60] |
| 16 | 3 | **0.999583** | +0.910 / **+0.984** / +1.000 | **1%** | **313** [87, 571] |
| 32 | 1 | **0.992155** | +0.469 / **+0.872** / +0.973 | **11%** | **0** [0, 1] |
| 32 | 2 | **0.997328** | +0.668 / **+0.915** / +0.985 | **5%** | **87** [14, 193] |
| 32 | 3 | **0.999166** | +0.700 / **+0.923** / +0.988 | **4%** | **833** [18, 2,136] |
| 64 | 1 | **0.998324** | +0.551 / **+0.899** / +0.985 | **8%** | **2** [1, 3] |
| 64 | 2 | **0.999997** | +0.541 / **+0.855** / +0.967 | **9%** | **4,176** [1,320, 7,333] |
| 64 | 3 | **0.999996** | +0.561 / **+0.847** / +0.958 | **7%** | **7,300** [2,156, 14,436] |
| 128 | 1 | **0.999876** | +0.654 / **+0.923** / +0.985 | **5%** | **9** [7, 11] |
| 128 | 2 | **1.000000** | +0.653 / **+0.905** / +0.977 | **5%** | **3,158,529** [1,402,633, 5,004,083] |
| 128 | 3 | **1.000000** | +0.639 / **+0.890** / +0.975 | **5%** | **87,751,464** [39,140,215, 138,392,840] |

**Y is now per instance.** For each held-out sequence, the two fits' 9 per-position attribution magnitudes are rank-correlated, and the table reports the distribution of those correlations over instances. Averaging attributions across instances first, as an earlier version did, discards exactly the variation the claim is about: the question is whether two fits rank the positions of a GIVEN splice site the same way.

**Is the tail degenerate?** If attribution concentrates in a few positions, a low all-position ρ may only be reporting the order of positions that carry no mass. `top-3 mass` is the median fraction of total |Δ| in a sequence's three strongest positions; `eff. positions` is exp(entropy) of the normalized magnitudes, where 9.0 means all positions contribute equally and 1.0 means one dominates. `Jaccard` is the median overlap of the two fits' top-k position SETS — the quantity a reader actually uses.

| width | depth | top-3 mass | eff. positions | mean Jaccard top2 / top3 | exact top-1 / top-3 | ρ on top-3 union |
|---|---|---|---|---|---|---|
| 16 | 1 | 0.911 | 3.25 | 0.72 / **0.75** | 68% / **56%** | +0.680 (n≈3.3) |
| 16 | 2 | 0.827 | 4.32 | 0.90 / **0.91** | 92% / **82%** | +1.000 (n≈3.0) |
| 16 | 3 | 0.774 | 5.01 | 0.93 / **0.93** | 93% / **86%** | +1.000 (n≈3.0) |
| 32 | 1 | 0.955 | 2.58 | 0.68 / **0.71** | 64% / **48%** | +0.570 (n≈3.4) |
| 32 | 2 | 0.821 | 4.45 | 0.78 / **0.79** | 77% / **61%** | +0.740 (n≈3.2) |
| 32 | 3 | 0.734 | 5.60 | 0.77 / **0.79** | 76% / **61%** | +0.740 (n≈3.2) |
| 64 | 1 | 0.962 | 2.47 | 0.71 / **0.74** | 69% / **53%** | +0.660 (n≈3.4) |
| 64 | 2 | 0.775 | 4.94 | 0.69 / **0.71** | 70% / **48%** | +0.630 (n≈3.6) |
| 64 | 3 | 0.594 | 7.16 | 0.66 / **0.69** | 67% / **43%** | +0.510 (n≈3.8) |
| 128 | 1 | 0.972 | 2.28 | 0.70 / **0.75** | 68% / **54%** | +0.630 (n≈3.4) |
| 128 | 2 | 0.872 | 3.76 | 0.70 / **0.76** | 66% / **54%** | +0.540 (n≈3.4) |
| 128 | 3 | 0.673 | 6.27 | 0.72 / **0.75** | 74% / **53%** | +0.625 (n≈3.0) |

**Conditioned on a top-3 existing.** Set overlap only means something where the profile has a well-defined top; a near-uniform profile has an ill-conditioned top-3 and disagreement there says nothing about which positions matter. These restrict to instances where BOTH fits place at least 80% of their magnitude in their own top 3.

| width | depth | instances qualifying | mean Jaccard top3 (all → cond.) | exact top-3 (all → cond.) |
|---|---|---|---|---|
| 16 | 1 | 67% | 0.75 → **0.83** | 56% → **67%** |
| 16 | 2 | 53% | 0.91 → **0.94** | 82% → **89%** |
| 16 | 3 | 42% | 0.93 → **0.96** | 86% → **93%** |
| 32 | 1 | 76% | 0.71 → **0.77** | 48% → **56%** |
| 32 | 2 | 48% | 0.79 → **0.86** | 61% → **72%** |
| 32 | 3 | 31% | 0.79 → **0.88** | 61% → **76%** |
| 64 | 1 | 82% | 0.74 → **0.79** | 53% → **60%** |
| 64 | 2 | 40% | 0.71 → **0.83** | 48% → **66%** |
| 64 | 3 | 8% | 0.69 → **0.89** | 43% → **78%** |
| 128 | 1 | 84% | 0.75 → **0.79** | 54% → **60%** |
| 128 | 2 | 61% | 0.76 → **0.82** | 54% → **66%** |
| 128 | 3 | 18% | 0.75 → **0.90** | 53% → **79%** |

| width | depth | ρ over position averages (n=9) | ρ over substitutions (n=35) |
|---|---|---|---|
| 16 | 1 | +0.893 | +0.891 |
| 16 | 2 | +0.982 | +0.987 |
| 16 | 3 | +0.987 | +0.993 |
| 32 | 1 | +0.695 | +0.728 |
| 32 | 2 | +0.905 | +0.916 |
| 32 | 3 | +0.968 | +0.949 |
| 64 | 1 | +0.693 | +0.729 |
| 64 | 2 | +0.873 | +0.839 |
| 64 | 3 | +0.928 | +0.925 |
| 128 | 1 | +0.722 | +0.722 |
| 128 | 2 | +0.807 | +0.771 |
| 128 | 3 | +0.920 | +0.886 |

**These two columns are coarse and are retained only for continuity.** A Spearman over 9 items has an approximate standard error of 1/sqrt(9-1) = 0.35 under independence, so a value of +0.37 is about one standard error from zero and even +0.9 is not precise. The substitution view uses 35 items and is correspondingly less granular. The bracketed intervals elsewhere in this table are bootstrap intervals over SEEDS and do not include the rank correlation's own granularity, so neither column should be read as a precise quantity. The per-instance distribution above is the one the verdict uses, because it is summarised by percentiles over hundreds of instances rather than by a single coarse statistic.

Noise is not assumed. The phenotype is affine in log₁₀ of the count ratio ex_ct/tot_ct (slope 0.933 on this library). Those are two independent count pools rather than a proportion — ex_ct exceeds tot_ct in 4.9% of rows — so both are treated as Poisson and the delta method applied to the log ratio, giving `sd = |a|·sqrt(1/ex + 1/tot)/ln10`: median 0.3114 in the standardized units the models see. This counts sequencing noise only; library preparation and biological variation add more. **Z is therefore a lower bound** — at least this many measurements, likely more.

## Verdict

**The identifiability claim, stated quantitatively.** The twin is not identical to the original and it is not unrelated to it, so neither the strong claim nor the word 'collapse' describes the measurement. At the closest cell (128x3) the twin agrees to held-out R² 1.000000, its predictions differ from the original's by RMS 0.0001 in standardized phenotype units, and separating the two as functions takes at least 87,751,464 held-out measurements [39,140,215, 138,392,840] at α = 0.05 with power 0.8. Across the grid the requirement ranges from 0 measurements (32x1) to 87,751,464 (128x3). 10 of 12 cells are separable within this library's 30,483 sequences; the rest would need a larger experiment. **The attribution consequence is what makes this practical rather than philosophical, and it is measured per instance.** For a given splice site, the two fits' rankings of its 9 positions agree at a median Spearman of +0.847 in the worst cell (64x3) and +0.890 at 128x3; in 32x1 11% of held-out sites the two rankings agree at ρ below 0.5. A reader who ranks positions by attribution magnitude for a particular sequence is therefore reading a quantity the data does not pin down, which is the failure mode the certificates in this project are meant to prevent. The per-instance distribution is used here rather than the single ρ over 9 position-averages, which is too coarse to carry a claim: its standard error under independence is about 0.35. **Which reading this supports: it depends on the cell, and both must be stated.** Attribution is concentrated (82% of each sequence's |Δ| mass in three positions, an effective 4.38 contributing positions of 9), so the top-3 set is the part that carries signal. In 16x2, 16x3 the two fits AGREE on which positions those are (median top-3 Jaccard exact in at least 82% of instances), so there the full-rank disagreement is tail ordering and the low per-instance ρ is NOT the headline; the honest claim for those cells is the narrow one. In 128x1, 128x2, 128x3, 16x1, 32x1, 32x2, 32x3, 64x1, 64x2, 64x3 they do NOT agree (top-3 sets exact in only 43% of instances, strongest position matching exactly in only 67% of sequences), so there the models differ about what matters and the claim stands as written. **Top-k set overlap is the statistic to lead with either way**, because it is what gets used downstream and it is the one that separates these two readings. This triple holds regardless of how the containment question in `paper/tables/closure_search.md` resolves. It describes the two fits actually obtained and the data actually needed to tell them apart, not whether some member of the class equals ψ∘φ̂ exactly.
