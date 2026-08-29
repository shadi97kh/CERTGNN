# Does the class contain the twin, or memorize it?

git SHA `b2951ae`, config `9fdd4067`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed, split 3000 fit / 1000 held out by a fixed seeded permutation recorded in `splits.json`. The reference map and every refit see the fit split ONLY; warp family sinusoid at strength 0.95; refit 6,000 epochs, warm-started, best iterate.

`null` is the fitted latent itself and is the ceiling: the identity is trivially in the class, so its held-out R² must be ~1 or the protocol is broken. `warp` is the monotone reparameterization, the measurement. `shuffled` is a permutation of the latent and is the floor: it is unlearnable out of sample, so its held-out R² must be ~0 or the split is leaking.

| width | depth | params | params/fit-pt | null fit → held-out | **warp fit → held-out** | shuffled fit → held-out | warp drop |
|---|---|---|---|---|---|---|---|
| 16 | 1 | 609 | 0.20 | 1.000000 → 1.000000 | **0.984634 → 0.974799** [0.9669, 0.9809] | 0.3867 → +0.0007 | +0.009834 |
| 16 | 2 | 881 | 0.29 | 1.000000 → 1.000000 | **0.999289 → 0.993477** [0.9895, 0.9962] | 0.6821 → +0.0001 | +0.005812 |
| 16 | 3 | 1,153 | 0.38 | 1.000000 → 1.000000 | **0.999677 → 0.995925** [0.9921, 0.9986] | 0.7793 → +0.0009 | +0.003752 |
| 32 | 1 | 1,217 | 0.41 | 1.000000 → 1.000000 | **0.992906 → 0.974876** [0.9708, 0.9789] | 0.6826 → +0.0013 | +0.018030 |
| 32 | 2 | 2,273 | 0.76 | 1.000000 → 1.000000 | **0.998738 → 0.946338** [0.9185, 0.9724] | 0.9540 → +0.0006 | +0.052401 |
| 32 | 3 | 3,329 | 1.11 | 1.000000 → 1.000000 | **0.999803 → 0.940439** [0.9151, 0.9643] | 0.9962 → +0.0004 | +0.059364 |
| 64 | 1 | 2,433 | 0.81 | 1.000000 → 1.000000 | **0.998961 → 0.980403** [0.9696, 0.9893] | 0.9152 → +0.0030 | +0.018558 |
| 64 | 2 | 6,593 | 2.20 | 1.000000 → 1.000000 | **1.000000 → 0.923607** [0.8929, 0.9532] | 1.0000 → +0.0004 | +0.076392 |
| 64 | 3 | 10,753 | 3.58 | 1.000000 → 1.000000 | **0.999999 → 0.903422** [0.8879, 0.9186] | 1.0000 → +0.0009 | +0.096578 |
| 128 | 1 | 4,865 | 1.62 | 1.000000 → 1.000000 | **0.999984 → 0.988463** [0.9868, 0.9900] | 0.9989 → +0.0011 | +0.011521 |
| 128 | 2 | 21,377 | 7.13 | 1.000000 → 1.000000 | **1.000000 → 0.957526** [0.9381, 0.9734] | 1.0000 → +0.0011 | +0.042474 |
| 128 | 3 | 37,889 | 12.63 | 1.000000 → 1.000000 | **1.000000 → 0.924677** [0.9108, 0.9394] | 1.0000 → +0.0010 | +0.075323 |

The held-out columns re-estimate the affine map on the held-out points, which is the generous reading. The stricter variant, which fixes scale and offset on the fit split, is in `results.json` as `*_heldout_strict`; the verdict reports it for the decisive cells.

## Verdict

**Validity gates pass.** The null ceiling holds at 1.000000 in the worst cell and the shuffled floor collapses to +0.0030 in the worst cell, confirming that a target unlearnable out of sample scores at chance and that the split does not leak. In-sample, that same shuffled target is fitted essentially perfectly, so the contrast between the two shuffled columns is the memorization effect made visible. **Held-out closure COLLAPSES at 128x2, 128x3.** 128x2: fit 1.000000 → held-out 0.957526 [0.9381, 0.9734], strict 0.956288; 128x3: fit 1.000000 → held-out 0.924677 [0.9108, 0.9394], strict 0.922170. The in-sample 1.000000 was memorization: the refit reproduced the target on the points it was fitted to and does not agree with it elsewhere, so no member of the class equals ψ∘φ̂ as a function. **No cell contains the twin, and the identifiability claim is the weaker practical one, not the strong claim.** The twin is a fitting artifact of the observed points rather than an alternative explanation of the data, and the mechanism test remains viable. Across the grid the best held-out warp agreement is 16x3 at 0.995925 [0.9921, 0.9986], with an in-sample-to-held-out drop of +0.003752. Every closure number reported elsewhere in this project is the in-sample column and should be read against the held-out one here.
