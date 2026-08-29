# Within-locus ISM invariance under a monotone warp

git SHA `99aa3bb` (DIRTY), config `0821419f`, 10 seeds, mean [95% bootstrap CI]. 800 instances x 15 positions x 3 alternatives per position.

The autograd arm is a CONTROL, not a result: under a monotone warp the gradient scales by the per-instance factor psi'(phi(x)), so its within-locus ranking is exactly invariant. A value below 1.000 there means the pipeline is wrong, not the theory. The ISM arm is a finite difference, where the mean value theorem gives psi'(xi_j) at a point between phi(x) and phi(x'_j), hence a per-MUTATION multiplier.

The twin is the analytic psi . phi_hat, which isolates the mean-value-theorem effect; the in-class representative adds further departure on top, so these are a LOWER bound.

## Control: autograd within-locus ranking (must be 1.000)

| warp family | strength | within-locus Spearman | ratio-invariance error |
|---|---|---|---|
| sinusoid | 0.1 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sinusoid | 0.25 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sinusoid | 0.4 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sinusoid | 0.55 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sinusoid | 0.7 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sinusoid | 0.85 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sinusoid | 0.95 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| spline | 0.1 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| spline | 0.25 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| spline | 0.4 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| spline | 0.55 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| spline | 0.7 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| spline | 0.85 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| spline | 0.95 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sigmoid_mixture | 0.1 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sigmoid_mixture | 0.25 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sigmoid_mixture | 0.4 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sigmoid_mixture | 0.55 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sigmoid_mixture | 0.7 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sigmoid_mixture | 0.85 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| sigmoid_mixture | 0.95 | 1.000 [1.000, 1.000] (n=10) | 0.000 [0.000, 0.000] (n=10) |

## ISM within-locus ranking by single-mutation effect size

Instances binned by the spread of their single-mutation latent effects, max_j |phi(x'_j) - phi(x)|, in latent (logit) units.

### sinusoid

| effect-size bin | median effect | s=0.1 | s=0.25 | s=0.4 | s=0.55 | s=0.7 | s=0.85 | s=0.95 |
|---|---|---|---|---|---|---|---|---|
| 0.2-0.35 | 0.277 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | 0.991 |
| 0.35-0.6 | 0.498 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.997 | 0.990 |
| 0.6-1 | 0.893 | 1.000 | 0.999 | 0.997 | 0.995 | 0.993 | 0.988 | 0.981 |
| 1-1.75 | 1.459 | 0.999 | 0.998 | 0.996 | 0.993 | 0.989 | 0.984 | 0.977 |
| 1.75-3 | 2.180 | 0.999 | 0.998 | 0.996 | 0.993 | 0.989 | 0.982 | 0.974 |
| >3 | 3.303 | 1.000 | 0.999 | 0.997 | 0.995 | 0.992 | 0.984 | 0.971 |

### spline

| effect-size bin | median effect | s=0.1 | s=0.25 | s=0.4 | s=0.55 | s=0.7 | s=0.85 | s=0.95 |
|---|---|---|---|---|---|---|---|---|
| 0.2-0.35 | 0.277 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.35-0.6 | 0.498 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6-1 | 0.893 | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 | 0.998 | 0.997 |
| 1-1.75 | 1.459 | 1.000 | 0.999 | 0.999 | 0.998 | 0.997 | 0.995 | 0.994 |
| 1.75-3 | 2.180 | 1.000 | 0.999 | 0.999 | 0.998 | 0.996 | 0.994 | 0.993 |
| >3 | 3.303 | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 | 0.997 | 0.996 |

### sigmoid_mixture

| effect-size bin | median effect | s=0.1 | s=0.25 | s=0.4 | s=0.55 | s=0.7 | s=0.85 | s=0.95 |
|---|---|---|---|---|---|---|---|---|
| 0.2-0.35 | 0.277 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 |
| 0.35-0.6 | 0.498 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.998 |
| 0.6-1 | 0.893 | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 | 0.997 | 0.995 |
| 1-1.75 | 1.459 | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 | 0.997 | 0.996 |
| 1.75-3 | 2.180 | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 | 0.997 | 0.996 |
| >3 | 3.303 | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 | 0.997 | 0.995 |

## ISM within-locus RATIO invariance error

Median |log of the ratio-of-ratios| over mutation pairs at one locus, which is exactly |log(psi'(xi_j) / psi'(xi_k))|. Zero iff the two mutations share a multiplier. Ranking asks which mutation matters more; the ratio asks how much more, and they can come apart.

### sinusoid

| effect-size bin | median effect | s=0.1 | s=0.25 | s=0.4 | s=0.55 | s=0.7 | s=0.85 | s=0.95 |
|---|---|---|---|---|---|---|---|---|
| 0.2-0.35 | 0.277 | 0.001 | 0.004 | 0.007 | 0.013 | 0.024 | 0.053 | 0.138 |
| 0.35-0.6 | 0.498 | 0.003 | 0.008 | 0.015 | 0.025 | 0.042 | 0.085 | 0.205 |
| 0.6-1 | 0.893 | 0.012 | 0.030 | 0.050 | 0.073 | 0.104 | 0.158 | 0.257 |
| 1-1.75 | 1.459 | 0.018 | 0.044 | 0.071 | 0.101 | 0.137 | 0.189 | 0.260 |
| 1.75-3 | 2.180 | 0.021 | 0.052 | 0.087 | 0.127 | 0.177 | 0.252 | 0.354 |
| >3 | 3.303 | 0.018 | 0.048 | 0.085 | 0.134 | 0.205 | 0.329 | 0.523 |

### spline

| effect-size bin | median effect | s=0.1 | s=0.25 | s=0.4 | s=0.55 | s=0.7 | s=0.85 | s=0.95 |
|---|---|---|---|---|---|---|---|---|
| 0.2-0.35 | 0.277 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.35-0.6 | 0.498 | 0.001 | 0.002 | 0.003 | 0.004 | 0.005 | 0.005 | 0.006 |
| 0.6-1 | 0.893 | 0.001 | 0.003 | 0.005 | 0.007 | 0.009 | 0.012 | 0.015 |
| 1-1.75 | 1.459 | 0.003 | 0.006 | 0.010 | 0.014 | 0.019 | 0.024 | 0.028 |
| 1.75-3 | 2.180 | 0.004 | 0.009 | 0.015 | 0.021 | 0.027 | 0.035 | 0.042 |
| >3 | 3.303 | 0.003 | 0.007 | 0.011 | 0.016 | 0.021 | 0.027 | 0.032 |

### sigmoid_mixture

| effect-size bin | median effect | s=0.1 | s=0.25 | s=0.4 | s=0.55 | s=0.7 | s=0.85 | s=0.95 |
|---|---|---|---|---|---|---|---|---|
| 0.2-0.35 | 0.277 | 0.002 | 0.006 | 0.010 | 0.017 | 0.026 | 0.039 | 0.052 |
| 0.35-0.6 | 0.498 | 0.003 | 0.007 | 0.013 | 0.021 | 0.032 | 0.048 | 0.065 |
| 0.6-1 | 0.893 | 0.004 | 0.011 | 0.020 | 0.030 | 0.045 | 0.064 | 0.083 |
| 1-1.75 | 1.459 | 0.004 | 0.012 | 0.021 | 0.032 | 0.045 | 0.062 | 0.077 |
| 1.75-3 | 2.180 | 0.005 | 0.014 | 0.025 | 0.037 | 0.053 | 0.074 | 0.092 |
| >3 | 3.303 | 0.006 | 0.017 | 0.030 | 0.046 | 0.068 | 0.099 | 0.130 |

## Headline: where within-locus ISM ranking breaks

Effect size (latent units) at which the binned within-locus Spearman drops below 0.95:

| warp family | s=0.1 | s=0.25 | s=0.4 | s=0.55 | s=0.7 | s=0.85 | s=0.95 |
|---|---|---|---|---|---|---|---|
| sinusoid | never | never | never | never | never | never | never |
| spline | never | never | never | never | never | never | never |
| sigmoid_mixture | never | never | never | never | never | never | never |

**Within-locus ISM ranking does not break anywhere on the grid.** The Spearman stays above 0.95 in every effect-size bin, at every warp strength and family tested. On this evidence the constructive rule survives the move from gradients to in-silico mutagenesis, and the per-mutation multiplier of the mean value theorem is too weak to reorder mutations at a locus.

**But ratio invariance does NOT survive.** The within-locus ratio error reaches 0.523 in log units (sinusoid at s=0.95), a factor of 1.69 distortion in how much more one mutation matters than another at the same locus, against exactly 0.000 for the autograd control. Ranking and ratio come apart: a rule that licenses "position j matters more than k here" survives the move to ISM; a rule that licenses "j has twice the effect of k" does not.
