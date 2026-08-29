# Identifiability on the real BRCA2 5' splice site MPSA

git SHA `ac351aa`, config `29e9a37e`, 10 seeds, mean [95% bootstrap CI]. 4000 sequences of length 9 per seed.

The true latent is unobservable on real data, so closure is measured against each class's OWN fitted latent: fit, warp the fit, and ask whether the class re-represents the warped version. A class that does contains the twin, and the twin predicts identically, which needs no access to the truth.

## Model fits

| G-P map | fit R² on log10 PSI | null-control closure R² | null-control effect size |
|---|---|---|---|
| linear | 0.590 [0.577, 0.602] (n=10) | 1.000000 [1.000000, 1.000000] (n=10) | -0.0002 [-0.0125, 0.0112] (n=10) |
| pairwise | 0.728 [0.715, 0.737] (n=10) | 1.000000 [1.000000, 1.000000] (n=10) | -0.0073 [-0.0152, 0.0015] (n=10) |
| neural | 0.776 [0.752, 0.799] (n=10) | 1.000000 [1.000000, 1.000000] (n=10) | 0.0071 [-0.0205, 0.0369] (n=10) |

## Closure at the largest monotone warp, six significant figures

| G-P map | sinusoid | spline | sigmoid_mixture |
|---|---|---|---|
| linear | 0.936460 [0.935405, 0.937574] (n=10) | 0.953728 [0.926762, 0.977741] (n=10) | 0.979161 [0.976826, 0.981494] (n=10) |
| pairwise | 0.956364 [0.955355, 0.957327] (n=10) | 0.964054 [0.938787, 0.984025] (n=10) | 0.953199 [0.948602, 0.958011] (n=10) |
| neural | 0.996644 [0.994501, 0.998376] (n=10) | 0.998827 [0.997761, 0.999512] (n=10) | 0.998691 [0.997807, 0.999385] (n=10) |

## Cross-instance attribution divergence, neural class

| warp family | s=0.1 | s=0.25 | s=0.4 | s=0.55 | s=0.7 | s=0.85 | s=0.95 |
|---|---|---|---|---|---|---|---|
| sinusoid | 0.996 | 0.975 | 0.940 | 0.895 | 0.845 | 0.790 | 0.758 |
| spline | 0.999 | 0.993 | 0.981 | 0.963 | 0.935 | 0.900 | 0.869 |
| sigmoid_mixture | 0.999 | 0.996 | 0.987 | 0.969 | 0.935 | 0.877 | 0.828 |

## ISM within-locus invariance, binned by this library's own effect sizes

Single-mutation latent effects come from mutating every position of every assayed site to each of its three alternative bases. Effect-size quantiles (latent units): 0.05=2.263, 0.25=3.281, 0.5=4.454, 0.75=6.299, 0.95=10.112.

Reported as a RANGE over warp family, shape parameter omega in 1, 2, 4, and strength, never as a point estimate: the ratio error is partly a property of the warp's shape, and quoting one cell would repeat the defect that made the indistinguishability radius unusable.

| effect-size bin | median effect | ranking Spearman (min-max) | ratio error (min-max) | worst ratio cell |
|---|---|---|---|---|
| 0.6-1 | 0.822 | 0.9901-1.0000 | 0.000-0.182 | sinusoid, omega=2, s=0.95 |
| 1-1.75 | 1.550 | 0.9597-1.0000 | 0.000-0.438 | sinusoid, omega=4, s=0.95 |
| 1.75-3 | 2.569 | 0.9579-1.0000 | 0.000-0.520 | sinusoid, omega=4, s=0.95 |
| >3 | 5.492 | 0.9632-0.9999 | 0.002-0.562 | sinusoid, omega=4, s=0.95 |

## Verdict

**The neural class did NOT reach closure on real data** (best 0.998827), so the twin is not demonstrably in the class here and the synthetic finding does not transfer as stated. Composite predictions of original and twin agree at r >= 0.9466, so the two are the same function rather than two different fits; without that control a latent difference could not be attributed to reparameterization. Artifact checks: effect-versus-strength Spearman sinusoid +0.788, spline +0.440, sigmoid_mixture +0.718; null-control effect 0.0381 +/- 0.0315; effect does increase with strength, so the radius is measurable here. **The constructive rule, with its real-data support.** Over every warp family, shape parameter and strength tested, and at every effect size this library produces, within-locus ISM ranking never falls below **0.9579**, while ratio comparisons degrade by up to **0.562 log units, a factor of 1.75** (worst cell: sinusoid, omega=4, s=0.95). **Rank within a locus; never compare magnitudes across loci on a nonlinear latent.** Both numbers are worst cases over the warp grid, not point estimates at one shape.
