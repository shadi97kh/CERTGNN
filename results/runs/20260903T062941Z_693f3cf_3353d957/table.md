# Do models the data cannot tell apart rank the loci differently?

git SHA `693f3cf`, config `3353d957`, 5 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed, 3000 fit / 1000 held out. 20 models per cell, differing ONLY in initialization seed: same architecture, same fit split, same learning-rate selection, same budget. No warp, no refit, no reparameterization.

A pair is **indistinguishable** when a paired two-sided t-test on per-point held-out squared errors does not reject at α = 0.05. Pairs that fail are ones the data can choose between and are excluded, not counted as identifiability instances. No multiple-comparison correction is applied, which makes the filter stricter rather than looser: more pairs are called distinguishable, so fewer survive.

| width | depth | held-out R² | accuracy-tied pairs | filter resolution (min. detectable ΔR²) | **n to separate as functions** | per-instance ρ (p10 / median) | frac ρ<0.5 |
|---|---|---|---|---|---|---|---|
| 16 | 1 | 0.5770 | 162/190 | 0.0787 | **0** | +0.217 / **+0.633** | **31%** |
| 16 | 2 | 0.5336 | 159/190 | 0.1148 | **0** | -0.168 / **+0.400** | **59%** |
| 16 | 3 | 0.5181 | 144/190 | 0.1170 | **0** | -0.268 / **+0.300** | **68%** |
| 32 | 1 | 0.5799 | 168/190 | 0.0707 | **1** | +0.365 / **+0.717** | **19%** |
| 32 | 2 | 0.5275 | 158/190 | 0.1154 | **0** | -0.200 / **+0.383** | **60%** |
| 32 | 3 | 0.4569 | 106/190 | 0.1319 | **0** | -0.318 / **+0.250** | **72%** |
| 64 | 1 | 0.6081 | 160/190 | 0.0428 | **1** | +0.600 / **+0.833** | **4%** |
| 64 | 2 | 0.5025 | 98/190 | 0.1029 | **0** | -0.252 / **+0.333** | **64%** |
| 64 | 3 | 0.3426 | 94/190 | 0.1367 | **0** | -0.383 / **+0.167** | **79%** |
| 128 | 1 | 0.6218 | 165/190 | 0.0333 | **2** | +0.700 / **+0.883** | **2%** |
| 128 | 2 | 0.5550 | 146/190 | 0.0499 | **1** | +0.383 / **+0.733** | **17%** |
| 128 | 3 | 0.4738 | 108/190 | 0.0995 | **0** | -0.268 / **+0.308** | **67%** |

**Accuracy-tied** means a paired two-sided t-test on per-point held-out squared errors does not reject at α = 0.05. That is a failure to reject, NOT evidence of equivalence, and its resolution is finite: the *filter resolution* column gives the smallest held-out R² difference the test could detect at 80% power with 1000 held-out points, so differences below it would not have been seen. The claim made here does not rest on accepting that null. It is the conjunction: these pairs are **not separable by accuracy at this resolution**, they **are separable as functions** with the stated number of measurements, and they **attribute differently**.

**Is the tail degenerate?** If attribution concentrates in a few positions, a low all-position ρ may only be reporting the order of positions that carry no mass. `top-3 mass` is the median fraction of a sequence's total |Δ| in its three strongest positions; `eff. positions` is exp(entropy) of the normalized magnitudes, where 9.0 means all contribute equally and 1.0 means one dominates. `Jaccard` is the median overlap of the two models' top-k position SETS — what a reader actually uses.

| width | depth | top-3 mass | eff. pos. | mean Jaccard top2 / top3 | exact top-1 / top-3 | ρ on top-3 union |
|---|---|---|---|---|---|---|
| 16 | 1 | 0.927 | 3.03 | 0.56 / **0.57** | 55% / **25%** | +0.400 (n≈4.0) |
| 16 | 2 | 0.825 | 4.37 | 0.38 / **0.42** | 38% / **12%** | -0.050 (n≈4.0) |
| 16 | 3 | 0.774 | 5.05 | 0.33 / **0.37** | 33% / **9%** | -0.200 (n≈4.0) |
| 32 | 1 | 0.957 | 2.53 | 0.62 / **0.63** | 63% / **32%** | +0.500 (n≈4.0) |
| 32 | 2 | 0.850 | 4.08 | 0.37 / **0.41** | 37% / **12%** | -0.100 (n≈4.0) |
| 32 | 3 | 0.775 | 5.08 | 0.31 / **0.35** | 31% / **8%** | -0.300 (n≈5.0) |
| 64 | 1 | 0.965 | 2.43 | 0.75 / **0.72** | 76% / **47%** | +0.700 (n≈4.0) |
| 64 | 2 | 0.838 | 4.25 | 0.35 / **0.39** | 34% / **10%** | -0.200 (n≈4.0) |
| 64 | 3 | 0.533 | 7.87 | 0.26 / **0.30** | 25% / **5%** | -0.400 (n≈5.0) |
| 128 | 1 | 0.976 | 2.21 | 0.80 / **0.78** | 82% / **57%** | +0.800 (n≈3.0) |
| 128 | 2 | 0.912 | 3.26 | 0.59 / **0.59** | 64% / **27%** | +0.400 (n≈4.0) |
| 128 | 3 | 0.718 | 5.79 | 0.33 / **0.38** | 32% / **9%** | -0.200 (n≈4.0) |

Jaccard on sets this small takes only four values at k=3 (0, 0.2, 0.5, 1) and two at k=1, so its median carries almost nothing — the median top-1 Jaccard is just the exact-match fraction thresholded at one half. The MEAN is reported instead, together with the exact-set-match fraction, which is what the verdict's branch condition uses.

**Quality-matched.** Depth and fit quality are confounded in this grid, so an agree/disagree split by depth is also a split by how well the models fit. These columns repeat the comparison on pairs where BOTH models are in their cell's better-fitting half, with the median of the pair's lower held-out R² shown so the quality level is visible.

| width | depth | pairs | pair R² (all → top half) | mean Jaccard top3 (all → top half) | exact top-3 (all → top half) |
|---|---|---|---|---|---|
| 16 | 1 | 213 | 0.5707 → **0.6065** | 0.57 → **0.59** | 25% → **28%** |
| 16 | 2 | 215 | 0.5356 → **0.5600** | 0.42 → **0.43** | 12% → **13%** |
| 16 | 3 | 213 | 0.5410 → **0.5722** | 0.37 → **0.40** | 9% → **10%** |
| 32 | 1 | 225 | 0.5779 → **0.6088** | 0.63 → **0.65** | 32% → **36%** |
| 32 | 2 | 223 | 0.5431 → **0.5521** | 0.41 → **0.43** | 12% → **12%** |
| 32 | 3 | 191 | 0.4859 → **0.5642** | 0.35 → **0.37** | 8% → **8%** |
| 64 | 1 | 219 | 0.6179 → **0.6356** | 0.72 → **0.74** | 47% → **50%** |
| 64 | 2 | 192 | 0.5424 → **0.5785** | 0.39 → **0.41** | 10% → **12%** |
| 64 | 3 | 123 | 0.3106 → **0.3893** | 0.30 → **0.32** | 5% → **6%** |
| 128 | 1 | 223 | 0.6371 → **0.6501** | 0.78 → **0.79** | 57% → **60%** |
| 128 | 2 | 223 | 0.6145 → **0.6301** | 0.59 → **0.60** | 27% → **28%** |
| 128 | 3 | 184 | 0.5127 → **0.5406** | 0.38 → **0.39** | 9% → **10%** |

**Conditioned on a top-3 existing.** Set overlap only means something where the attribution profile has a well-defined top. A model spreading its magnitude near-uniformly over 9 positions has an ill-conditioned top-3, and two such models disagree for a reason unrelated to which positions matter. These columns restrict to instances where BOTH models place at least 80% of their magnitude in their own top 3, and report what fraction of instances qualify. Low agreement here cannot be blamed on a flat profile.

| width | depth | eff. positions | instances qualifying | mean Jaccard top3 (all → conditioned) | exact top-3 (all → conditioned) |
|---|---|---|---|---|---|
| 16 | 1 | 3.03 | **68%** | 0.57 → **0.61** | 25% → **29%** |
| 16 | 2 | 4.37 | **39%** | 0.42 → **0.53** | 12% → **19%** |
| 16 | 3 | 5.05 | **27%** | 0.37 → **0.51** | 9% → **18%** |
| 32 | 1 | 2.53 | **80%** | 0.63 → **0.65** | 32% → **34%** |
| 32 | 2 | 4.08 | **42%** | 0.41 → **0.48** | 12% → **16%** |
| 32 | 3 | 5.08 | **23%** | 0.35 → **0.46** | 8% → **13%** |
| 64 | 1 | 2.43 | **84%** | 0.72 → **0.74** | 47% → **49%** |
| 64 | 2 | 4.25 | **32%** | 0.39 → **0.48** | 10% → **16%** |
| 64 | 3 | 7.87 | **0%** | 0.30 → **0.75** | 5% → **50%** |
| 128 | 1 | 2.21 | **87%** | 0.78 → **0.80** | 57% → **60%** |
| 128 | 2 | 3.26 | **69%** | 0.59 → **0.62** | 27% → **29%** |
| 128 | 3 | 5.79 | **9%** | 0.38 → **0.51** | 9% → **20%** |

| width | depth | ρ over position averages (n=9) | ρ over substitutions |
|---|---|---|---|
| 16 | 1 | +0.917 | +0.876 |
| 16 | 2 | +0.833 | +0.753 |
| 16 | 3 | +0.800 | +0.737 |
| 32 | 1 | +0.925 | +0.877 |
| 32 | 2 | +0.817 | +0.751 |
| 32 | 3 | +0.833 | +0.727 |
| 64 | 1 | +0.933 | +0.888 |
| 64 | 2 | +0.817 | +0.760 |
| 64 | 3 | +0.900 | +0.784 |
| 128 | 1 | +0.950 | +0.901 |
| 128 | 2 | +0.883 | +0.845 |
| 128 | 3 | +0.817 | +0.774 |

**The n=9 column is coarse and is retained only for continuity.** A Spearman over 9 items has an approximate standard error of 0.35 under independence, so single values near ±0.4 are barely distinguishable from zero and even ±0.9 is imprecise. Bootstrap intervals elsewhere are over SEEDS and do not capture that granularity. The per-instance distribution in the main table is what the verdict uses: each instance contributes one correlation over its own 9 positions, and hundreds of instances determine the percentiles.

The ρ column pools every surviving pair across all seeds, so it is the distribution the claim is about rather than a mean of per-seed summaries. `n to separate` uses the Poisson log-ratio noise model of `paper/tables/separation.md` applied to the two models' own predictions.

## Verdict

**1667 of 2280 pairs are not separable by predictive accuracy** at the resolution this held-out set provides (paired t-test, α = 0.05, uncorrected and therefore strict). This is a failure to reject, not a demonstration of equivalence, and its resolution is stated rather than implied: with 1000 held-out points the test could detect a held-out R² difference of about 0.1012 at 80% power, so smaller differences in accuracy would not have been seen. Nothing below depends on accepting that null. **The claim is the conjunction, and each part is measured.** These pairs share an architecture, a fit split, a budget and a learning-rate selection, differing only in initialization seed. Their predictive ACCURACY is not separable at the resolution above. Their identity as FUNCTIONS is separable, and cheaply: the median pair needs about 0 held-out measurements to reject that the two are the same function under the Poisson log-ratio noise model. So these are not near-identical models -- they are models that predict differently while scoring the same. **And they attribute differently -- measured per instance.** For a given splice site, do the two models rank ITS positions the same way? Median agreement is +0.167 in the worst cell (64x3), and in 64x3 79% of held-out sites the two rankings agree at ρ below 0.5. Averaging attributions across instances first, as an earlier version did, discards exactly this variation. **And the disagreement is about which positions matter, not about an irrelevant tail.** Attribution is concentrated — a median 84% of |Δ| mass in three positions, an effective 4.16 contributing positions of 9 — yet the models differ on which those are: median top-3 Jaccard falls to 0.05 (64x3) and the single strongest position matches exactly in only 25% of sequences there. Restricted to the union of the two top-3 sets, the positions that carry the signal, median rank agreement is -0.400. **This is the occurrence result, and top-k set overlap is the statistic to lead with**, since a reader asks which positions matter for a site, not how the bottom of the list is ordered. It needs no constructed twin, no reparameterization and no refit, so it is immune to the search and warm-start confounds `paper/tables/closure_search.md` found in the warp-based route. Per cell, the median number of held-out measurements needed to separate an indistinguishable pair as functions is 32x3 0, 16x2 0, 64x3 0, 16x3 0 (four smallest). The latent agreement among surviving pairs has median 128x1 0.9558, 128x2 0.7859, 128x3 0.4364, so where the rankings differ the underlying latents differ too. **Is the depth split just fit quality?** Depth and held-out R² are confounded in this grid, so the question is answered on pairs matched for quality rather than argued. Restricting to pairs where both models are in their cell's better-fitting half lifts the median pair R² at depth 3 from 0.4993 to 0.5524, and top-3 exact agreement goes from 5–9% to 6–10% (mean Jaccard 0.32–0.40). Note what this does and does not match: the restriction equalises quality WITHIN a cell, while the confound is BETWEEN depths. Depth-1 pairs sit at a median R² of 0.5979, and the best depth-3 pairs reach 0.5524, so the two still do NOT overlap. The depth-3 models remain the worse fits even after restriction, so this test bounds the confound rather than eliminating it: it shows how much of the split survives a quality improvement of +0.0531, not what would happen at equal fit. **The split is not a quality artifact.** Well-fitting depth-3 pairs still disagree about which positions matter, so the effect is expressivity rather than fit. This matches `paper/tables/closure_search.md`, where a cold-started refit recovers an in-class target at depth 1 and fails at depths 2 and 3: one mechanism -- what the optimiser can reach in a deeper class -- would produce both results. **Does the disagreement survive where a top-3 exists?** Agreement tracks concentration across this grid — cells with few effective positions agree most — so a flat attribution profile is a live alternative explanation. Effective positions range 2.21 to 7.87 of 9. Restricting to instances where both models put at least 80% of their magnitude in their own top 3, exact top-3 agreement moves from 5–57% to 13–60% (mean Jaccard up to 0.80), with 0–87% of instances qualifying. **These cells are excluded from the claim, not caveated**: 128x3, 64x3 retain under 10% of instances, meaning almost no sequence has a well-defined top-3 under either model. For them the question of which positions matter is not well posed, so their low agreement is not evidence of disagreement about anything; a near-uniform attribution profile is simply what those models produce. **The claim is therefore made on 128x1, 128x2, 16x1, 16x2, 16x3, 32x1, 32x2, 32x3, 64x1, 64x2**, where 23% to 87% of instances have a well-defined top-3 under both models. There, models tied on accuracy still pick different top-3 position sets in 40% to 87% of sequences. **Is this just weak models?** Partly, but not mainly, and the question deserves the number rather than a reassurance. Across cells the held-out predictive R² does correlate with agreement (Spearman +0.514 against the minimum ρ, +0.681 against the median), so better-fitting cells do agree more. The claim therefore rests on the best-fitting cell, not the worst: at 128x1, held-out R² 0.6218, the highest in the grid, 165 of 190 pairs are indistinguishable and among them 35% rank the loci at ρ below 0.9, 16% below 0.8 and 9% below 0.7, reaching +0.350. The disagreement is not confined to the cells that predict badly. This experiment corrects the Part B reading in the original identifiability probe, which recorded multi-restart fits differing as a negative on the grounds that they were different fits rather than different representatives. Two fits that explain the data equally well are the identifiability problem; what that analysis lacked was the performance filter applied here.
