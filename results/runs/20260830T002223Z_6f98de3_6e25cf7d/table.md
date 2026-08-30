# Do models the data cannot tell apart rank the loci differently?

git SHA `6f98de3`, **aggregated post hoc by `673300f` via --retable**, config `6e25cf7d`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed, 3000 fit / 1000 held out. 20 models per cell, differing ONLY in initialization seed: same architecture, same fit split, same learning-rate selection, same budget. No warp, no refit, no reparameterization.

A pair is **indistinguishable** when a paired two-sided t-test on per-point held-out squared errors does not reject at α = 0.05. Pairs that fail are ones the data can choose between and are excluded, not counted as identifiability instances. No multiple-comparison correction is applied, which makes the filter stricter rather than looser: more pairs are called distinguishable, so fewer survive.

| width | depth | held-out R² | accuracy-tied pairs | filter resolution (min. detectable ΔR²) | **n to separate as functions** | per-instance ρ (p10 / median) | frac ρ<0.5 |
|---|---|---|---|---|---|---|---|
| 16 | 1 | 0.6011 | 162/190 | 0.0699 | **1** | +0.217 / **+0.650** | **30%** |
| 16 | 2 | 0.5591 | 158/190 | 0.1059 | **0** | -0.167 / **+0.417** | **58%** |
| 16 | 3 | 0.5349 | 134/190 | 0.1161 | **0** | -0.300 / **+0.283** | **69%** |
| 32 | 1 | 0.6019 | 167/190 | 0.0651 | **1** | +0.365 / **+0.717** | **19%** |
| 32 | 2 | 0.5537 | 143/190 | 0.1032 | **0** | -0.168 / **+0.408** | **58%** |
| 32 | 3 | 0.4792 | 102/190 | 0.1256 | **0** | -0.317 / **+0.258** | **71%** |
| 64 | 1 | 0.6278 | 160/190 | 0.0402 | **1** | +0.600 / **+0.833** | **4%** |
| 64 | 2 | 0.5381 | 100/190 | 0.0882 | **0** | -0.183 / **+0.417** | **57%** |
| 64 | 3 | 0.3929 | 92/190 | 0.1267 | **0** | -0.383 / **+0.175** | **78%** |
| 128 | 1 | 0.6372 | 154/190 | 0.0318 | **2** | +0.700 / **+0.883** | **2%** |
| 128 | 2 | 0.5836 | 144/190 | 0.0461 | **1** | +0.400 / **+0.750** | **16%** |
| 128 | 3 | 0.4761 | 95/190 | 0.0937 | **0** | -0.267 / **+0.317** | **67%** |

**Accuracy-tied** means a paired two-sided t-test on per-point held-out squared errors does not reject at α = 0.05. That is a failure to reject, NOT evidence of equivalence, and its resolution is finite: the *filter resolution* column gives the smallest held-out R² difference the test could detect at 80% power with 1000 held-out points, so differences below it would not have been seen. The claim made here does not rest on accepting that null. It is the conjunction: these pairs are **not separable by accuracy at this resolution**, they **are separable as functions** with the stated number of measurements, and they **attribute differently**.

**Is the tail degenerate?** If attribution concentrates in a few positions, a low all-position ρ may only be reporting the order of positions that carry no mass. `top-3 mass` is the median fraction of a sequence's total |Δ| in its three strongest positions; `eff. positions` is exp(entropy) of the normalized magnitudes, where 9.0 means all contribute equally and 1.0 means one dominates. `Jaccard` is the median overlap of the two models' top-k position SETS — what a reader actually uses.

| width | depth | top-3 mass | eff. pos. | mean Jaccard top2 / top3 | exact top-1 / top-3 | ρ on top-3 union |
|---|---|---|---|---|---|---|
| 16 | 1 | 0.926 | 3.03 | 0.56 / **0.57** | 55% / **26%** | +0.400 (n≈4.0) |
| 16 | 2 | 0.819 | 4.44 | 0.39 / **0.43** | 40% / **13%** | +0.000 (n≈4.0) |
| 16 | 3 | 0.768 | 5.11 | 0.32 / **0.37** | 32% / **9%** | -0.250 (n≈5.0) |
| 32 | 1 | 0.956 | 2.56 | 0.61 / **0.62** | 63% / **32%** | +0.500 (n≈4.0) |
| 32 | 2 | 0.851 | 4.05 | 0.38 / **0.42** | 38% / **12%** | -0.100 (n≈4.0) |
| 32 | 3 | 0.759 | 5.28 | 0.31 / **0.36** | 31% / **8%** | -0.300 (n≈5.0) |
| 64 | 1 | 0.969 | 2.36 | 0.75 / **0.73** | 76% / **48%** | +0.600 (n≈4.0) |
| 64 | 2 | 0.873 | 3.79 | 0.38 / **0.43** | 39% / **14%** | -0.050 (n≈4.0) |
| 64 | 3 | 0.540 | 7.80 | 0.27 / **0.31** | 26% / **6%** | -0.400 (n≈5.0) |
| 128 | 1 | 0.976 | 2.20 | 0.80 / **0.78** | 82% / **57%** | +0.800 (n≈3.0) |
| 128 | 2 | 0.918 | 3.16 | 0.59 / **0.60** | 64% / **27%** | +0.400 (n≈4.0) |
| 128 | 3 | 0.728 | 5.67 | 0.33 / **0.38** | 32% / **10%** | -0.200 (n≈4.0) |

Jaccard on sets this small takes only four values at k=3 (0, 0.2, 0.5, 1) and two at k=1, so its median carries almost nothing — the median top-1 Jaccard is just the exact-match fraction thresholded at one half. The MEAN is reported instead, together with the exact-set-match fraction, which is what the verdict's branch condition uses.

**Quality-matched.** Depth and fit quality are confounded in this grid, so an agree/disagree split by depth is also a split by how well the models fit. These columns repeat the comparison on pairs where BOTH models are in their cell's better-fitting half, with the median of the pair's lower held-out R² shown so the quality level is visible.

| width | depth | pairs | pair R² (all → top half) | mean Jaccard top3 (all → top half) | exact top-3 (all → top half) |
|---|---|---|---|---|---|
| 16 | 1 | 430 | 0.5971 → **0.6181** | 0.57 → **0.59** | 26% → **28%** |
| 16 | 2 | 431 | 0.5601 → **0.5833** | 0.43 → **0.45** | 13% → **14%** |
| 16 | 3 | 426 | 0.5474 → **0.5827** | 0.37 → **0.39** | 9% → **11%** |
| 32 | 1 | 443 | 0.5936 → **0.6131** | 0.62 → **0.64** | 32% → **35%** |
| 32 | 2 | 428 | 0.5551 → **0.5872** | 0.42 → **0.45** | 12% → **14%** |
| 32 | 3 | 380 | 0.4965 → **0.5760** | 0.36 → **0.37** | 8% → **9%** |
| 64 | 1 | 436 | 0.6275 → **0.6440** | 0.73 → **0.74** | 48% → **50%** |
| 64 | 2 | 399 | 0.5668 → **0.5983** | 0.43 → **0.47** | 14% → **16%** |
| 64 | 3 | 258 | 0.3620 → **0.4393** | 0.31 → **0.33** | 6% → **6%** |
| 128 | 1 | 440 | 0.6409 → **0.6523** | 0.78 → **0.79** | 57% → **60%** |
| 128 | 2 | 437 | 0.6171 → **0.6295** | 0.60 → **0.61** | 27% → **30%** |
| 128 | 3 | 344 | 0.5331 → **0.5835** | 0.38 → **0.41** | 10% → **11%** |

**Conditioned on a top-3 existing.** Set overlap only means something where the attribution profile has a well-defined top. A model spreading its magnitude near-uniformly over 9 positions has an ill-conditioned top-3, and two such models disagree for a reason unrelated to which positions matter. These columns restrict to instances where BOTH models place at least 80% of their magnitude in their own top 3, and report what fraction of instances qualify. Low agreement here cannot be blamed on a flat profile.

| width | depth | eff. positions | instances qualifying | mean Jaccard top3 (all → conditioned) | exact top-3 (all → conditioned) |
|---|---|---|---|---|---|
| 16 | 1 | 3.03 | **68%** | 0.57 → **0.61** | 26% → **29%** |
| 16 | 2 | 4.44 | **38%** | 0.43 → **0.53** | 13% → **20%** |
| 16 | 3 | 5.11 | **25%** | 0.37 → **0.50** | 9% → **18%** |
| 32 | 1 | 2.56 | **79%** | 0.62 → **0.64** | 32% → **33%** |
| 32 | 2 | 4.05 | **42%** | 0.42 → **0.50** | 12% → **17%** |
| 32 | 3 | 5.28 | **21%** | 0.36 → **0.47** | 8% → **14%** |
| 64 | 1 | 2.36 | **84%** | 0.73 → **0.74** | 48% → **49%** |
| 64 | 2 | 3.79 | **50%** | 0.43 → **0.51** | 14% → **18%** |
| 64 | 3 | 7.80 | **0%** | 0.31 → **0.75** | 6% → **50%** |
| 128 | 1 | 2.20 | **86%** | 0.78 → **0.79** | 57% → **59%** |
| 128 | 2 | 3.16 | **70%** | 0.60 → **0.62** | 27% → **29%** |
| 128 | 3 | 5.67 | **11%** | 0.38 → **0.51** | 10% → **20%** |

| width | depth | ρ over position averages (n=9) | ρ over substitutions |
|---|---|---|---|
| 16 | 1 | +0.917 | +0.883 |
| 16 | 2 | +0.833 | +0.784 |
| 16 | 3 | +0.817 | +0.749 |
| 32 | 1 | +0.917 | +0.887 |
| 32 | 2 | +0.817 | +0.765 |
| 32 | 3 | +0.833 | +0.754 |
| 64 | 1 | +0.933 | +0.899 |
| 64 | 2 | +0.850 | +0.794 |
| 64 | 3 | +0.900 | +0.800 |
| 128 | 1 | +0.933 | +0.902 |
| 128 | 2 | +0.883 | +0.853 |
| 128 | 3 | +0.833 | +0.774 |

**The n=9 column is coarse and is retained only for continuity.** A Spearman over 9 items has an approximate standard error of 0.35 under independence, so single values near ±0.4 are barely distinguishable from zero and even ±0.9 is imprecise. Bootstrap intervals elsewhere are over SEEDS and do not capture that granularity. The per-instance distribution in the main table is what the verdict uses: each instance contributes one correlation over its own 9 positions, and hundreds of instances determine the percentiles.

The ρ column pools every surviving pair across all seeds, so it is the distribution the claim is about rather than a mean of per-seed summaries. `n to separate` uses the Poisson log-ratio noise model of `paper/tables/separation.md` applied to the two models' own predictions.

## Verdict

**1610 of 2280 pairs are not separable by predictive accuracy** at the resolution this held-out set provides (paired t-test, α = 0.05, uncorrected and therefore strict). This is a failure to reject, not a demonstration of equivalence, and its resolution is stated rather than implied: with 1000 held-out points the test could detect a held-out R² difference of about 0.0909 at 80% power, so smaller differences in accuracy would not have been seen. Nothing below depends on accepting that null. **The claim is the conjunction, and each part is measured.** These pairs share an architecture, a fit split, a budget and a learning-rate selection, differing only in initialization seed. Their predictive ACCURACY is not separable at the resolution above. Their identity as FUNCTIONS is separable, and cheaply: the median pair needs about 0 held-out measurements to reject that the two are the same function under the Poisson log-ratio noise model. So these are not near-identical models -- they are models that predict differently while scoring the same. **And they attribute differently -- measured per instance.** For a given splice site, do the two models rank ITS positions the same way? Median agreement is +0.175 in the worst cell (64x3), and in 64x3 78% of held-out sites the two rankings agree at ρ below 0.5. Averaging attributions across instances first, as an earlier version did, discards exactly this variation. **And the disagreement is about which positions matter, not about an irrelevant tail.** Attribution is concentrated — a median 86% of |Δ| mass in three positions, an effective 3.92 contributing positions of 9 — yet the models differ on which those are: median top-3 Jaccard falls to 0.06 (64x3) and the single strongest position matches exactly in only 26% of sequences there. Restricted to the union of the two top-3 sets, the positions that carry the signal, median rank agreement is -0.400. **This is the occurrence result, and top-k set overlap is the statistic to lead with**, since a reader asks which positions matter for a site, not how the bottom of the list is ordered. It needs no constructed twin, no reparameterization and no refit, so it is immune to the search and warm-start confounds `paper/tables/closure_search.md` found in the warp-based route. Per cell, the median number of held-out measurements needed to separate an indistinguishable pair as functions is 16x3 0, 32x3 0, 64x3 0, 16x2 0 (four smallest). The latent agreement among surviving pairs has median 128x1 0.9592, 128x2 0.8250, 128x3 0.4836, so where the rankings differ the underlying latents differ too. **Is the depth split just fit quality?** Depth and held-out R² are confounded in this grid, so the question is answered on pairs matched for quality rather than argued. Restricting to pairs where both models are in their cell's better-fitting half lifts the median pair R² at depth 3 from 0.5148 to 0.5793, and top-3 exact agreement goes from 6–10% to 6–11% (mean Jaccard 0.33–0.41). Note what this does and does not match: the restriction equalises quality WITHIN a cell, while the confound is BETWEEN depths. Depth-1 pairs sit at a median R² of 0.6123, and the best depth-3 pairs reach 0.5793, so the two still do NOT overlap. The depth-3 models remain the worse fits even after restriction, so this test bounds the confound rather than eliminating it: it shows how much of the split survives a quality improvement of +0.0645, not what would happen at equal fit. **The split is not a quality artifact.** Well-fitting depth-3 pairs still disagree about which positions matter, so the effect is expressivity rather than fit. This matches `paper/tables/closure_search.md`, where a cold-started refit recovers an in-class target at depth 1 and fails at depths 2 and 3: one mechanism -- what the optimiser can reach in a deeper class -- would produce both results. **Does the disagreement survive where a top-3 exists?** Agreement tracks concentration across this grid — cells with few effective positions agree most — so a flat attribution profile is a live alternative explanation. Effective positions range 2.20 to 7.80 of 9. Restricting to instances where both models put at least 80% of their magnitude in their own top 3, exact top-3 agreement moves from 6–57% to 14–59% (mean Jaccard up to 0.79), with 0–86% of instances qualifying. **These cells are excluded from the claim, not caveated**: 64x3 retain under 10% of instances, meaning almost no sequence has a well-defined top-3 under either model. For them the question of which positions matter is not well posed, so their low agreement is not evidence of disagreement about anything; a near-uniform attribution profile is simply what those models produce. **The claim is therefore made on 128x1, 128x2, 128x3, 16x1, 16x2, 16x3, 32x1, 32x2, 32x3, 64x1, 64x2**, where 11% to 86% of instances have a well-defined top-3 under both models. There, models tied on accuracy still pick different top-3 position sets in 41% to 86% of sequences. **Is this just weak models?** Partly, but not mainly, and the question deserves the number rather than a reassurance. Across cells the held-out predictive R² does correlate with agreement (Spearman +0.650 against the minimum ρ, +0.673 against the median), so better-fitting cells do agree more. The claim therefore rests on the best-fitting cell, not the worst: at 128x1, held-out R² 0.6372, the highest in the grid, 154 of 190 pairs are indistinguishable and among them 37% rank the loci at ρ below 0.9, 15% below 0.8 and 6% below 0.7, reaching +0.350. The disagreement is not confined to the cells that predict badly. This experiment corrects the Part B reading in the original identifiability probe, which recorded multi-restart fits differing as a negative on the grounds that they were different fits rather than different representatives. Two fits that explain the data equally well are the identifiability problem; what that analysis lacked was the performance filter applied here.
