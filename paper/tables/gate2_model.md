# ABLATIONS 1.9 with a trained model (experiments/gate2_model.py)

git SHA `797a3df` (dirty: False), config hash `5766d0a3`, 10 seeds, mean [95% bootstrap CI].

Scores come from a trained GNN under a soft node mask, and from the generator's latent function, on the *same* instances with the *same* resistance-ball explanation.

## Trained models

| arm | node features | val (latent R²) | test R² | r(model latent, logit p₀) | learned every seed |
|---|---|---|---|---|---|
| baseline hidden | z, degree, distance | 0.121 [0.085, 0.154] (n=10) | 0.096 [0.083, 0.111] (n=10) | -0.005 [-0.024, 0.014] (n=10) | False |
| baseline observable | z, degree, distance, eta0 | 0.965 [0.958, 0.971] (n=10) | 0.964 [0.960, 0.967] (n=10) | 0.900 [0.895, 0.905] (n=10) | True |

## Conditional coverage, max−min across strata

| arm | scorer | probability-space gap | latent-space gap | marginal (prob / latent) |
|---|---|---|---|---|
| baseline hidden | oracle | 0.184 [0.162, 0.206] (n=10) | 0.036 [0.027, 0.045] (n=10) | 0.895 / 0.893 |
| baseline hidden | model | 0.043 [0.037, 0.051] (n=10) | 0.035 [0.028, 0.042] (n=10) | 0.895 / 0.898 |
| baseline observable | oracle | 0.184 [0.162, 0.206] (n=10) | 0.036 [0.027, 0.045] (n=10) | 0.895 / 0.893 |
| baseline observable | model | 0.124 [0.103, 0.140] (n=10) | 0.281 [0.243, 0.316] (n=10) | 0.908 / 0.903 |

## Per-stratum coverage (mean over seeds)

| arm | scorer | space | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|---|
| baseline hidden | oracle | probability | 0.807 | 0.864 | 0.885 | 0.930 | 0.990 |
| baseline hidden | oracle | latent | 0.898 | 0.892 | 0.889 | 0.891 | 0.894 |
| baseline hidden | model | probability | 0.898 | 0.895 | 0.894 | 0.900 | 0.891 |
| baseline hidden | model | latent | 0.898 | 0.900 | 0.896 | 0.903 | 0.893 |
| baseline observable | oracle | probability | 0.807 | 0.864 | 0.885 | 0.930 | 0.990 |
| baseline observable | oracle | latent | 0.898 | 0.892 | 0.889 | 0.891 | 0.894 |
| baseline observable | model | probability | 0.888 | 0.866 | 0.883 | 0.926 | 0.979 |
| baseline observable | model | latent | 0.983 | 0.978 | 0.952 | 0.897 | 0.703 |

## Sanity controls (CLAUDE.md: mandatory)

| arm | control | probability-space gap | latent-space gap |
|---|---|---|---|
| baseline hidden | model randomization | 0.049 [0.039, 0.060] (n=10) | 0.048 [0.035, 0.063] (n=10) |
| baseline hidden | label randomization | 0.040 [0.030, 0.051] (n=10) | 0.040 [0.029, 0.052] (n=10) |
| baseline observable | model randomization | 0.235 [0.180, 0.298] (n=10) | 0.368 [0.287, 0.447] (n=10) |
| baseline observable | label randomization | 0.426 [0.381, 0.464] (n=10) | 0.439 [0.401, 0.475] (n=10) |

## Reading

- **baseline hidden**: informative = False (the trained model does not track the baseline (|r| = 0.005) or did not reach the configured val threshold, so a flat profile carries no information about the link); model reproduces the pathology = False.
- **baseline observable**: informative = True; model reproduces the pathology = False.

**The pathology does NOT reproduce with a trained model on the informative arms; on this evidence Theorem 2's empirical support is limited to the generative process and the paper must say so.**

This run does not re-decide G2. G2's verdict rests on its pre-registered thresholds and is unchanged.
