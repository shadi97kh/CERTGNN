# Indistinguishability radius for a nonlinear G-P map

git SHA `9b723de`, config `f662d93c`, 10 seeds, mean [95% bootstrap CI].

A twin is (psi.phi_hat, g_hat.psi^-1) for a monotone psi. It predicts identically only where the model class can represent psi.phi_hat; the class's approximation error is what makes a twin detectable. The radius is therefore the largest warp strength whose *best in-class approximation* stays statistically indistinguishable from the original fit at a given sample size.

**Test.** Paired on squared residuals: d_i = (y_i - twin_i)^2 - (y_i - orig_i)^2, null E[d] = 0. The paired t statistic at size N is t(N) = (mean d / sd d) * sqrt(N); the standardized effect size is estimated once per grid point, so t is available at any N without regenerating data. Indistinguishable means |t(N)| < 1.96 (two-sided, alpha = 0.05). Noise is the substrate's own observation noise (sigma = 0.01). Sample sizes are real MPSA designs: FAS exon 6 N = 3072, BRCA2 exon 17 N = 32768.

Every grid point is checked for strict monotonicity of psi on a dense grid; points that fail are skipped (0 of 1920 grid points skipped).

## Closure: can the class represent the warped latent? (six significant figures)

| G-P map | warp family | closure R^2 at max monotone strength |
|---|---|---|
| linear | sinusoid | 0.960636 [0.960086, 0.961216] (n=10) |
| linear | spline | 0.962456 [0.948376, 0.975739] (n=10) |
| linear | sigmoid_mixture | 0.951488 [0.946082, 0.956334] (n=10) |
| pairwise | sinusoid | 0.961760 [0.960531, 0.962881] (n=10) |
| pairwise | spline | 0.962737 [0.948305, 0.976303] (n=10) |
| pairwise | sigmoid_mixture | 0.930599 [0.924702, 0.937610] (n=10) |
| neural | sinusoid | 1.000000 [1.000000, 1.000000] (n=10) |
| neural | spline | 1.000000 [1.000000, 1.000000] (n=10) |
| neural | sigmoid_mixture | 1.000000 [1.000000, 1.000000] (n=10) |

A class whose closure R^2 is 1.000000 is closed under the warp: the twin is a legitimate member and no amount of data separates it from the original.

## Radius at N = 3072 (FAS exon 6)

| G-P map | warp family | radius (largest indistinguishable s) | cross-instance magnitude Spearman at the boundary | within-instance cosine |
|---|---|---|---|---|
| linear | sinusoid | 0.065 [0.000, 0.160] (n=10) | 1.000 [1.000, 1.000] (n=8) | 1.000 [1.000, 1.000] (n=10) |
| linear | spline | 0.095 [0.000, 0.285] (n=10) | 1.000 [1.000, 1.000] (n=9) | 1.000 [1.000, 1.000] (n=10) |
| linear | sigmoid_mixture | 0.010 [0.000, 0.030] (n=10) | 1.000 [1.000, 1.000] (n=9) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | sinusoid | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | spline | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | sigmoid_mixture | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| neural | sinusoid | 0.080 [0.000, 0.190] (n=10) | 0.993 [0.984, 1.000] (n=10) | 0.997 [0.992, 1.000] (n=10) |
| neural | spline | 0.110 [0.000, 0.281] (n=10) | 0.960 [0.884, 1.000] (n=10) | 0.989 [0.969, 1.000] (n=10) |
| neural | sigmoid_mixture | 0.105 [0.000, 0.240] (n=10) | 0.997 [0.993, 1.000] (n=10) | 0.999 [0.998, 1.000] (n=10) |

## Radius at N = 32768 (BRCA2 exon 17)

| G-P map | warp family | radius (largest indistinguishable s) | cross-instance magnitude Spearman at the boundary | within-instance cosine |
|---|---|---|---|---|
| linear | sinusoid | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| linear | spline | 0.025 [0.000, 0.075] (n=10) | 1.000 [1.000, 1.000] (n=9) | 1.000 [1.000, 1.000] (n=10) |
| linear | sigmoid_mixture | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | sinusoid | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | spline | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| pairwise | sigmoid_mixture | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| neural | sinusoid | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| neural | spline | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |
| neural | sigmoid_mixture | 0.000 [0.000, 0.000] (n=10) | 1.000 [1.000, 1.000] (n=10) | 1.000 [1.000, 1.000] (n=10) |

## Headline

**The radius is not measurable for the neural class, and the numbers in the tables above must not be read as measurements.** The class is closed under every warp family (closure R^2 = 1.000000 to six figures), so the twin's predictions are identical in exact arithmetic and the true prediction-space effect is zero. What the test picks up is the pipeline's own numerical floor. Three checks establish this rather than asserting it: the effect size shows no increasing trend with warp strength (Spearman of |effect| against strength: sinusoid -0.151, spline -0.118, sigmoid_mixture -0.115); the effect sits within sinusoid 2.5, spline 1.8, sigmoid_mixture 2.3 SD of the strength-0 null control (floor 0.0309 +/- 0.0187); and the classes that are NOT closed (linear, pairwise) show radii as large or larger than the closed class, which is backwards -- a class that cannot represent the twin should be easier to separate, not harder.

The correct statement is that for a G-P map flexible enough to be closed under monotone reparameterization, no MPSA sample size separates the twin, so the radius is bounded by the monotonicity limit and not by the data. The quantity that IS measurable is the attribution divergence, which falls monotonically as the warp strengthens while predictions stay at the floor:

- neural, sinusoid, cross-instance attribution Spearman -- s=0.10: 0.994, s=0.25: 0.977, s=0.40: 0.952, s=0.55: 0.923, s=0.70: 0.893, s=0.85: 0.862, s=0.95: 0.842
- neural, spline, cross-instance attribution Spearman -- s=0.10: 0.996, s=0.25: 0.986, s=0.40: 0.969, s=0.55: 0.945, s=0.70: 0.914, s=0.85: 0.877, s=0.95: 0.849
- neural, sigmoid_mixture, cross-instance attribution Spearman -- s=0.10: 0.995, s=0.25: 0.982, s=0.40: 0.962, s=0.55: 0.936, s=0.70: 0.908, s=0.85: 0.877, s=0.95: 0.856

Within-instance attribution direction is preserved throughout; what diverges is the cross-instance comparison. Note this is established for autograd gradients. The field's primitive on MPSA data is in-silico mutagenesis, a finite difference, where the mean value theorem gives a per-mutation multiplier psi'(xi) rather than a per-instance one, so the within-instance invariance does not transfer to ISM without further argument.
