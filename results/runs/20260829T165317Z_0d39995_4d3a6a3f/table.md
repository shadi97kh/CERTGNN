# Does high capacity re-represent monotone warps only, or anything?

git SHA `0d39995`, **aggregated post hoc by `7bc0e0c` via --retable**, config `4d3a6a3f`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed; warp family sinusoid at strength 0.95; refit 6,000 epochs, warm-started, best iterate, per-cell learning rate from the pilot.

`warp` is a monotone reparameterization of the fitted latent, the only target a closed model class must re-represent. `shuffled` is a random permutation of the same values and `noise` is Gaussian with the same mean and variance; neither is monotone in the latent and neither carries any relation to the input. `null` is the latent itself, the numerical floor. **Selectivity** is `warp` minus the better of `shuffled` and `noise`: it is high for a class that re-represents reparameterizations specifically, and near zero for a map that fits anything.

| width | depth | params | params/point | R² warp | R² shuffled | R² noise | R² null | selectivity |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 609 | 0.15 | 0.987906 | 0.2983 [0.2939, 0.3031] | 0.2970 | 1.000000 | +0.6842 |
| 16 | 2 | 881 | 0.22 | 0.999282 | 0.5746 [0.5357, 0.6142] | 0.4664 | 1.000000 | +0.4205 |
| 16 | 3 | 1,153 | 0.29 | 0.999735 | 0.6691 [0.6267, 0.7105] | 0.5190 | 1.000000 | +0.3306 |
| 32 | 1 | 1,217 | 0.30 | 0.993761 | 0.5598 [0.5538, 0.5664] | 0.5615 | 1.000000 | +0.4291 |
| 32 | 2 | 2,273 | 0.57 | 0.997438 | 0.8868 [0.8732, 0.8985] | 0.8801 | 1.000000 | +0.1064 |
| 32 | 3 | 3,329 | 0.83 | 0.999525 | 0.9711 [0.9677, 0.9744] | 0.9650 | 1.000000 | +0.0256 |
| 64 | 1 | 2,433 | 0.61 | 0.998773 | 0.8195 [0.8068, 0.8361] | 0.8159 | 1.000000 | +0.1783 |
| 64 | 2 | 6,593 | 1.65 | 0.999997 | 1.0000 [0.9999, 1.0000] | 1.0000 | 1.000000 | +0.0000 |
| 64 | 3 | 10,753 | 2.69 | 0.999997 | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.000000 | +0.0000 |
| 128 | 1 | 4,865 | 1.22 | 0.999882 | 0.9757 [0.9745, 0.9769] | 0.9765 | 1.000000 | +0.0228 |
| 128 | 2 | 21,377 | 5.34 | 1.000000 | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.000000 | +0.0000 |
| 128 | 3 | 37,889 | 9.47 | 1.000000 | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.000000 | +0.0000 |

## Which variable governs closure

Two measures over all 12 cells. **Discordance** is the fraction of cell pairs the variable distinguishes in which MORE capacity goes with a LARGER gap; near 0 means the variable governs. Ties are excluded, so a coarse variable is not penalised. Spearman against log₁₀ of the gap is shown alongside, but it penalises ties and so favours the finest-grained variable (depth has 3 distinct values, width 4, params/point 12) and should not be used to rank them.

| variable | discordance | pairs | Spearman |
|---|---|---|---|
| depth | 0.229 | 48 | -0.562 |
| width | 0.185 | 54 | -0.713 |
| params/point | 0.121 | 66 | -0.860 |

Smallest warp gap reachable at each depth: depth 1 → 1.18e-04, depth 2 → 1.25e-08, depth 3 → 2.58e-10. At each width: 16 → 2.65e-04, 32 → 4.75e-04, 64 → 2.70e-06, 128 → 2.58e-10.

## Verdict

**The closed cells interpolate.** At 128x2, 128x3 the shuffled target is re-represented at R² up to 1.0000, and a random permutation of the latent carries no relation to the input whatsoever. A map that fits that is fitting arbitrary values at arbitrary points, so its closure of 1.000000 on the monotone warp is not evidence that the model class is closed under reparameterization. **The strong identifiability claim does NOT follow.** The classes remain separable on real data, the mechanism test is viable, and the identifiability claim is the weaker practical one. **128x1 interpolates too** (shuffled R² 0.9757 at 1.22 parameters per datapoint) while still failing to close the warp target (R² 0.999882). Interpolation capacity and closure therefore come apart, and the ratio does not predict closure. **No single variable governs closure cleanly.** Discordance is depth 0.229, width 0.185, params/point 0.121; the lowest is params/point, but not by a clear margin, so closure is set by depth and width jointly rather than by any one of them. (Spearman is reported in the table but not used to rank: it penalises ties and so favours params/point, which takes a distinct value in every cell, over depth, which takes three.) The smallest gap reachable at each depth is depth 1 → 1.18e-04, depth 2 → 1.25e-08, depth 3 → 2.58e-10, monotone in depth; at each width 16 → 2.65e-04, 32 → 4.75e-04, 64 → 2.70e-06, 128 → 2.58e-10, not monotone in width. **No cell is both closed and selective.** Closure of 1.000000 occurs at 128x2, 128x3, and selectivity of at least 0.05 occurs at 16x1, 16x2, 16x3, 32x1, 32x2, 64x1, and these sets do not intersect. Structural closure requires both at once, so it is not exhibited anywhere in this grid: every cell that re-represents a monotone warp exactly also re-represents a random permutation of the same values. The numerical floor is the null target, re-represented to within 0.00e+00 in the worst cell, so every contrast above is far larger than the pipeline's own error.
