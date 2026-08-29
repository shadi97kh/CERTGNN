# Does closure converge to 1.000000 as G-P map capacity grows?

git SHA `0419c07`, **aggregated post hoc by `ecf38d1` via --retable** (the seed values below were computed by `0419c07`; the aggregation, asymptote and verdict code is the later one), config `6afd0696`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed; warp family sinusoid; refit 6,000 epochs, warm-started, best iterate.

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

## Asymptote

Fitting the closure gap as `1 - R² = g_inf + a·params^(-b)` over all 12 cells gives `g_inf` = 8.368e-04 [6.055e-04, 1.058e-03] (b = 5.000, 400 bootstrap fits over seeds), so closure tends to **0.999163** [0.998942, 0.999395].

That fit is dominated by the widest cells, which are the ones able to interpolate any target on the observed points. Restricted to the 7 cells with FEWER parameters than datapoints (16x1, 16x2, 16x3, 32x1, 32x2, 32x3, 64x1), the same fit gives `g_inf` = 1.572e-03 [1.185e-03, 1.996e-03], i.e. closure tends to **0.998428** [0.998004, 0.998815]. This is the fit that bears on structural closure; the unrestricted one cannot separate closure from interpolation.

## Verdict

**Closure reaches 1.000000 ONLY by interpolation.** No cell with fewer parameters than datapoints reaches it: the best such cell is 16x3 at closure 0.999735 (gap 2.65e-04). Closure appears only once the map is overparameterized (128x2, 128x3; best 128x3 at gap 2.58e-10), where it can fit ANY target on the observed points, so reaching 1 there says nothing about whether the model class is closed under reparameterization. **The strong identifiability claim does NOT follow.** On the evidence that is not interpolation, the classes remain separable on real data, the mechanism test is viable, and the identifiability claim is the weaker practical one. The pipeline's numerical floor is the null control's closure, which is 1.000000 in every cell (largest null gap 0.00e+00), so a shortfall above that floor is real. (The null EFFECT size reaches 0.0756, which is why effect size is not used as the closure floor here.) The power-law asymptote is reported above but is NOT used for this verdict: b is pinned at its bound in the fit, because the closure gap is not monotone in parameter count. The verdict rests on what the cells reached.
