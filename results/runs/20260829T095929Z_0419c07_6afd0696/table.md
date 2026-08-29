# Does closure converge to 1.000000 as G-P map capacity grows?

git SHA `0419c07`, **aggregated post hoc by `82e6095` via --retable** (the seed values below were computed by `0419c07`; the aggregation, asymptote and verdict code is the later one), config `6afd0696`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed; warp family sinusoid; refit 6,000 epochs, warm-started, best iterate.

| width | depth | params | params/point | fit R² | closure R² at s=0.95 | null-control closure | null effect | effect-vs-strength ρ |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 609 | 0.15 | 0.7144 | 0.987906 [0.985809, 0.989885] | 1.000000 | 0.0057 | 0.980 [0.940, 1.000] (n=10) |
| 16 | 2 | 881 | 0.22 | 0.7288 | 0.999282 [0.998894, 0.999579] | 1.000000 | 0.0250 | 0.980 [0.940, 1.000] (n=10) |
| 16 | 3 | 1,153 | 0.29 | 0.7565 | 0.999735 [0.999435, 0.999919] | 1.000000 | 0.0199 | 0.640 [0.380, 0.880] (n=10) |
| 32 | 1 | 1,217 | 0.30 | 0.7209 | 0.993761 [0.992907, 0.994579] | 1.000000 | 0.0111 | 1.000 [1.000, 1.000] (n=10) |
| 32 | 2 | 2,273 | 0.57 | 0.7706 | 0.997438 [0.995413, 0.999064] | 1.000000 | 0.0238 | 0.940 [0.880, 1.000] (n=10) |
| 32 | 3 | 3,329 | 0.83 | 0.7916 | 0.999525 [0.999145, 0.999819] | 1.000000 | 0.0756 | 0.500 [0.100, 0.840] (n=10) |
| 64 | 1 | 2,433 | 0.61 | 0.7131 | 0.998773 [0.998152, 0.999207] | 1.000000 | 0.0133 | 0.980 [0.940, 1.000] (n=10) |
| 64 | 2 | 6,593 | 1.65 | 0.8159 | 0.999997 [0.999996, 0.999999] | 1.000000 | 0.0525 | 0.300 [-0.140, 0.660] (n=10) |
| 64 | 3 | 10,753 | 2.69 | 0.8796 | 0.999997 [0.999995, 0.999999] | 1.000000 | 0.0422 | 0.060 [-0.400, 0.480] (n=10) |
| 128 | 1 | 4,865 | 1.22 | 0.7118 | 0.999882 [0.999866, 0.999895] | 1.000000 | 0.0052 | 0.860 [0.740, 0.960] (n=10) |
| 128 | 2 | 21,377 | 5.34 | 0.7477 | 1.000000 [1.000000, 1.000000] | 1.000000 | 0.0225 | -0.160 [-0.520, 0.220] (n=10) |
| 128 | 3 | 37,889 | 9.47 | 0.8383 | 1.000000 [1.000000, 1.000000] | 1.000000 | 0.0201 | -0.140 [-0.600, 0.340] (n=10) |

## Closure gap by depth and width

The gap `1 - R²` at s=0.95, arranged the way the data is actually ordered. Depth sets what is reachable and width amplifies within a depth; parameters per datapoint cuts across both, which is why it does not order this table.

| | width 16 | width 32 | width 64 | width 128 |
|---|---|---|---|---|
| **depth 1** | 1.21e-02 | 6.24e-03 | 1.23e-03 | 1.18e-04 |
| **depth 2** | 7.18e-04 | 2.56e-03 | 2.70e-06 | 1.25e-08 **←1.000000** |
| **depth 3** | 2.65e-04 | 4.75e-04 | 2.78e-06 | 2.58e-10 **←1.000000** |

## Asymptote

Fitting the closure gap as `1 - R² = g_inf + a·params^(-b)` over all 12 cells gives `g_inf` = 8.368e-04 [6.055e-04, 1.058e-03] (b = 5.000, 400 bootstrap fits over seeds), so closure tends to **0.999163** [0.998942, 0.999395].

Restricted to the 7 cells with fewer parameters than datapoints (16x1, 16x2, 16x3, 32x1, 32x2, 32x3, 64x1), the same fit gives `g_inf` = 1.572e-03 [1.185e-03, 1.996e-03], i.e. closure tends to **0.998428** [0.998004, 0.998815].

**Neither fit is used, and neither number should be quoted.** Both have b pinned at the bound, because the gap is not monotone in parameter count. The split above is by parameters per datapoint, and the gap table shows that ratio does not order the cells either: 128x1 is above it and far from closure while 16x3 is below it and comparable. Both fits are retained only to document that a power law in parameter count fails on this data.

## Verdict

**This experiment establishes the structure. It does not determine the mechanism, and neither branch of the original question is decided by the numbers in this table; the held-out control below decides them.** No cell reaches closure of 1.000000 at depth 1, at any width tested. The best is 128x1 at 0.999882, a gap of 1.2e-04, and it sits at 1.22 parameters per datapoint -- above one, yet nowhere near closure. Closure of 1.000000 is reached at depth 2 and depth 3 at width 128, and approached to within 2.7e-06 at 64x2, 2.8e-06 at 64x3. The gap is ordered by depth, with width amplifying within depth; see the gap table above.

**Two mechanisms predict this pattern and this experiment does not separate them.** *Depth-dependent expressivity*: composing more layers may make the class genuinely closed under monotone reparameterization, because a deeper map can absorb a warp into its own hidden layers, in which case closure at the 128-wide cells is structural. *Interpolation*: those same cells hold more parameters than datapoints and can fit an arbitrary target on the observed points, in which case their closure carries no information about the model class. Both predict exactly what the table shows for the widest cells. The parameters-per-datapoint reading is additionally contradicted by the table: it predicts that every cell above one parameter per datapoint should close, and 64x2, 64x3, 128x1 do not (gap 2.7e-06, 2.8e-06, 1.2e-04).

**The shuffled-target control has run and did not settle this.** (`experiments/closure_shuffled.py`, `paper/tables/closure_shuffled.md`.) It asked whether the closed cells are SELECTIVE: whether they re-represent a monotone warp while failing a random permutation and Gaussian noise. They are not -- at 128x2 and 128x3 the permutation is re-represented at R2 1.0000. But selectivity was the wrong instrument, because closure does not imply it. Closure under monotone reparameterization is a claim about what the function class CONTAINS, and a class flexible enough to contain the reparameterized twin is generally also flexible enough to fit a permutation. Failing the non-monotone targets is therefore not a property a closed class must have, and its absence is what any sufficiently flexible neural class would show. That control does establish something narrower and important: at exactly the cells reading 1.000000, a fit evaluated on the points it was fitted to cannot distinguish a class that contains the twin from one that memorizes n values at n points.

**The discriminating experiment is the held-out control** (`experiments/closure_heldout.py`, `paper/tables/closure_heldout.md`). Every closure number in this table is in-sample: the refit is fitted on all n points and scored on those same n points. The held-out control refits on a subset and scores the warp target on points the refit never saw. Containment implies the twin agrees there, because it is the same FUNCTION; memorization does not.

**That control has now reported, and the closure in this table does not survive it.** Held-out agreement collapses at every cell that reads 1.000000 here, while its own two validity gates hold: the fitted latent is recovered out of sample and an unlearnable permutation target falls to chance. So the 1.000000 entries above are the refit reproducing the target on the points it was fitted to, not a member of the class equal to psi(phi) as a function. **No cell contains the twin, and the identifiability claim is the weaker practical one, not the strong claim.** Read every closure number in this table as the in-sample column of `paper/tables/closure_heldout.md`, where the held-out column is the one that bears on identifiability. The pipeline's numerical floor is the null control's closure, 1.000000 in every cell (largest null gap 0.00e+00), so the shortfalls above that floor are real. (The null EFFECT size reaches 0.0756; effect size is a different quantity and is not used as the closure floor.) The power-law asymptote is reported above but is NOT used: b is pinned at its bound, because the closure gap is not monotone in parameter count -- depth and width do not trade off along a single axis.
