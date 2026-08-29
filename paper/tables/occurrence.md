# Do models the data cannot tell apart rank the loci differently?

git SHA `c652d0b`, **aggregated post hoc by `c652d0b` via --retable**, config `01c4abfe`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed, 3000 fit / 1000 held out. 20 models per cell, differing ONLY in initialization seed: same architecture, same fit split, same learning-rate selection, same budget. No warp, no refit, no reparameterization.

A pair is **indistinguishable** when a paired two-sided t-test on per-point held-out squared errors does not reject at α = 0.05. Pairs that fail are ones the data can choose between and are excluded, not counted as identifiability instances. No multiple-comparison correction is applied, which makes the filter stricter rather than looser: more pairs are called distinguishable, so fewer survive.

| width | depth | params | held-out R² | indist. pairs | attribution ρ (min–median–max) | median n to separate | latent R² |
|---|---|---|---|---|---|---|---|
| 16 | 1 | 609 | 0.6011 | 162.3/190 | **+0.367 – +0.917 – +1.000** | 1 | 0.7851 |
| 16 | 2 | 881 | 0.5591 | 157.9/190 | **+0.167 – +0.833 – +1.000** | 0 | 0.6698 |
| 16 | 3 | 1,153 | 0.5349 | 134.3/190 | **+0.117 – +0.817 – +1.000** | 0 | 0.5641 |
| 32 | 1 | 1,217 | 0.6019 | 166.9/190 | **+0.383 – +0.917 – +1.000** | 1 | 0.8427 |
| 32 | 2 | 2,273 | 0.5537 | 142.6/190 | **+0.183 – +0.817 – +1.000** | 0 | 0.5998 |
| 32 | 3 | 3,329 | 0.4792 | 101.9/190 | **+0.200 – +0.833 – +1.000** | 0 | 0.4204 |
| 64 | 1 | 2,433 | 0.6278 | 160.1/190 | **+0.367 – +0.933 – +1.000** | 1 | 0.9287 |
| 64 | 2 | 6,593 | 0.5381 | 99.7/190 | **+0.167 – +0.850 – +1.000** | 0 | 0.5729 |
| 64 | 3 | 10,753 | 0.3929 | 91.7/190 | **+0.250 – +0.900 – +1.000** | 0 | 0.3983 |
| 128 | 1 | 4,865 | 0.6372 | 153.8/190 | **+0.350 – +0.933 – +1.000** | 2 | 0.9592 |
| 128 | 2 | 21,377 | 0.5836 | 144.3/190 | **+0.200 – +0.883 – +1.000** | 1 | 0.8250 |
| 128 | 3 | 37,889 | 0.4761 | 94.7/190 | **-0.100 – +0.833 – +1.000** | 0 | 0.4836 |

The ρ column pools every surviving pair across all seeds, so it is the distribution the claim is about rather than a mean of per-seed summaries. `n to separate` uses the Poisson log-ratio noise model of `paper/tables/separation.md` applied to the two models' own predictions.

## Verdict

**1610 of 2280 pairs are indistinguishable on held-out performance** (paired t-test, α = 0.05, uncorrected and therefore strict). These are pairs of models the data cannot choose between: same architecture, same fit split, same budget, differing only in initialization seed. **Models the data cannot tell apart rank the loci differently.** Among the indistinguishable pairs the per-locus attribution Spearman runs down to -0.100 (128x3), with a typical cell median of +0.867. The disagreement is not between a good fit and a bad one: these pairs are exactly the ones the held-out data cannot choose between. **This is the occurrence result.** It needs no constructed twin, no reparameterization and no refit, so it is immune to the search and warm-start confounds that `paper/tables/closure_search.md` found in the warp-based route. Per cell, the median number of held-out measurements needed to separate an indistinguishable pair as functions is 16x3 0, 32x3 0, 64x3 0, 16x2 0 (four smallest). The latent agreement among surviving pairs has median 128x1 0.9592, 128x2 0.8250, 128x3 0.4836, so where the rankings differ the underlying latents differ too. **Is this just weak models?** Partly, but not mainly, and the question deserves the number rather than a reassurance. Across cells the held-out predictive R² does correlate with agreement (Spearman +0.650 against the minimum ρ, +0.673 against the median), so better-fitting cells do agree more. The claim therefore rests on the best-fitting cell, not the worst: at 128x1, held-out R² 0.6372, the highest in the grid, 154 of 190 pairs are indistinguishable and among them 37% rank the loci at ρ below 0.9, 15% below 0.8 and 6% below 0.7, reaching +0.350. The disagreement is not confined to the cells that predict badly. This experiment corrects the Part B reading in the original identifiability probe, which recorded multi-restart fits differing as a negative on the grounds that they were different fits rather than different representatives. Two fits that explain the data equally well are the identifiability problem; what that analysis lacked was the performance filter applied here.
