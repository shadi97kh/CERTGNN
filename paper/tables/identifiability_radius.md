# Indistinguishability radius for a nonlinear G-P map

git SHA `99aa3bb`, config `0821419f`, 10 seeds, mean [95% bootstrap CI].

A twin is (psi.phi_hat, g_hat.psi^-1) for a monotone psi. It predicts identically only where the model class can represent psi.phi_hat; the class's approximation error is what makes a twin detectable. The radius is therefore the largest warp strength whose *best in-class approximation* stays statistically indistinguishable from the original fit at a given sample size.

**Test.** Paired on squared residuals: d_i = (y_i - twin_i)^2 - (y_i - orig_i)^2, null E[d] = 0. The paired t statistic at size N is t(N) = (mean d / sd d) * sqrt(N); the standardized effect size is estimated once per grid point, so t is available at any N without regenerating data. Indistinguishable means |t(N)| < 1.96 (two-sided, alpha = 0.05). Noise is the substrate's own observation noise (sigma = 0.01). Sample sizes are real MPSA designs: FAS exon 6 N = 3072, BRCA2 exon 17 N = 32768.

Every grid point is checked for strict monotonicity of psi on a dense grid; points that fail are skipped (0 of 1920 grid points skipped).

## Closure: can the class represent the warped latent? (six significant figures)

| G-P map | warp family | closure R^2 at max monotone strength |
|---|---|---|
| linear | sinusoid | 0.960657 [0.960096, 0.961241] (n=10) |
| linear | spline | 0.962382 [0.948203, 0.975791] (n=10) |
| linear | sigmoid_mixture | 0.953941 [0.948941, 0.958523] (n=10) |
| pairwise | sinusoid | 0.961745 [0.960552, 0.962841] (n=10) |
| pairwise | spline | 0.962684 [0.948114, 0.976351] (n=10) |
| pairwise | sigmoid_mixture | 0.934484 [0.929103, 0.940885] (n=10) |
| neural | sinusoid | 1.000000 [1.000000, 1.000000] (n=10) |
| neural | spline | 1.000000 [1.000000, 1.000000] (n=10) |
| neural | sigmoid_mixture | 1.000000 [1.000000, 1.000000] (n=10) |

A class whose closure R^2 is 1.000000 is closed under the warp: the twin is a legitimate member and no amount of data separates it from the original.

## Radius at N = 3072 (FAS exon 6)

| G-P map | warp family | radius (largest indistinguishable s) | cross-instance magnitude Spearman at the boundary | within-instance cosine |
|---|---|---|---|---|
| linear | sinusoid | 0.075 [0.010, 0.170] (n=10) | 1.000 [1.000, 1.000] (n=7) | 1.000 [1.000, 1.000] (n=10) |
| linear | spline | 0.135 [0.000, 0.326] (n=10) | 1.000 [1.000, 1.000] (n=8) | 1.000 [1.000, 1.000] (n=10) |
| linear | sigmoid_mixture | 0.010 [0.000, 0.030] (n=10) | 1.000 [1.000, 1.000] (n=9) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | sinusoid | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | spline | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | sigmoid_mixture | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| neural | sinusoid | 0.140 [0.030, 0.285] (n=10) | 0.982 [0.956, 0.999] (n=10) | 0.991 [0.977, 1.000] (n=10) |
| neural | spline | 0.175 [0.010, 0.410] (n=10) | 0.991 [0.973, 1.000] (n=10) | 0.995 [0.986, 1.000] (n=10) |
| neural | sigmoid_mixture | 0.130 [0.025, 0.240] (n=10) | 0.996 [0.993, 0.999] (n=10) | 0.999 [0.997, 1.000] (n=10) |

## Radius at N = 32768 (BRCA2 exon 17)

| G-P map | warp family | radius (largest indistinguishable s) | cross-instance magnitude Spearman at the boundary | within-instance cosine |
|---|---|---|---|---|
| linear | sinusoid | 0.010 [0.000, 0.030] (n=10) | 1.000 [1.000, 1.000] (n=9) | 1.000 [1.000, 1.000] (n=10) |
| linear | spline | 0.025 [0.000, 0.075] (n=10) | 1.000 [1.000, 1.000] (n=9) | 1.000 [1.000, 1.000] (n=10) |
| linear | sigmoid_mixture | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | sinusoid | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | spline | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | sigmoid_mixture | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| neural | sinusoid | 0.010 [0.000, 0.030] (n=10) | 1.000 [0.999, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| neural | spline | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| neural | sigmoid_mixture | 0.025 [0.000, 0.075] (n=10) | 0.999 [0.998, 1.000] (n=10) | 1.000 [0.999, 1.000] (n=10) |

## Headline

**The radius is not measurable for the neural class, and the numbers in the tables above must not be read as measurements.** The class is closed under every warp family (closure R^2 = 1.000000 to six figures), so the twin's predictions are identical in exact arithmetic and the true prediction-space effect is zero. What the test picks up is the pipeline's own numerical floor. Three checks establish this rather than asserting it: the effect size shows no increasing trend with warp strength (Spearman of |effect| against strength: sinusoid +0.115, spline -0.112, sigmoid_mixture -0.020); the effect sits within sinusoid 2.8, spline 1.5, sigmoid_mixture 2.2 SD of the strength-0 null control (floor 0.0277 +/- 0.0200); and the classes that are NOT closed (linear, pairwise) show radii as large or larger than the closed class, which is backwards -- a class that cannot represent the twin should be easier to separate, not harder.

The correct statement is that for a G-P map flexible enough to be closed under monotone reparameterization, no MPSA sample size separates the twin, so the radius is bounded by the monotonicity limit and not by the data. The quantity that IS measurable is the attribution divergence, which falls monotonically as the warp strengthens while predictions stay at the floor:

- neural, sinusoid, cross-instance attribution Spearman -- s=0.10: 0.994, s=0.25: 0.977, s=0.40: 0.953, s=0.55: 0.924, s=0.70: 0.895, s=0.85: 0.866, s=0.95: 0.846
- neural, spline, cross-instance attribution Spearman -- s=0.10: 0.996, s=0.25: 0.987, s=0.40: 0.971, s=0.55: 0.949, s=0.70: 0.920, s=0.85: 0.883, s=0.95: 0.856
- neural, sigmoid_mixture, cross-instance attribution Spearman -- s=0.10: 0.996, s=0.25: 0.988, s=0.40: 0.974, s=0.55: 0.954, s=0.70: 0.928, s=0.85: 0.894, s=0.95: 0.866

Within-instance direction degrades far more slowly than the cross-instance ranking, but it is NOT preserved: the analytic twin preserves it exactly (grad(psi.phi) = psi'(phi) grad phi, a per-instance positive scalar), while the in-class representative that realizes the twin agrees in value at closure R^2 = 1.000000 and still departs in gradient. Value-level closure does not imply gradient-level closure, and attributions are gradients. Note this is established for autograd gradients. The field's primitive on MPSA data is in-silico mutagenesis, a finite difference, where the mean value theorem gives a per-mutation multiplier psi'(xi) rather than a per-instance one, so the within-instance invariance does not transfer to ISM without further argument.
