# ABLATIONS 1.9 on the real splice substrate, scored by a trained model

git SHA `ae28103`, config `f3ea80bc`, 10 seeds, mean [95% bootstrap CI].

Substrate: MFASS via `certgnn.substrates.splice`, splits grouped by exon. The scorer is a trained GNN under a Theorem 4 resistance-ball soft mask threaded through every message-passing layer. There is no oracle anywhere in this experiment, which is the circularity it exists to remove.

## Trained model

| quantity | value |
|---|---|
| validation R² (latent scale) | -0.016 [-0.023, -0.010] (n=10) |
| test R² (latent scale) | -0.015 [-0.021, -0.009] (n=10) |
| r(model latent, logit baseline) | 0.004 [-0.024, 0.037] (n=10) |
| degenerate explanations | 0.000 [0.000, 0.000] (n=10) |
| clamped at the logit boundary (train) | 0.258 [0.254, 0.261] (n=10) |
| informative every seed | False |

## Conditional coverage, max−min across baseline strata

| arm | probability-space gap | latent-space gap | marginal (prob / latent) |
|---|---|---|---|
| trained model | 0.070 [0.055, 0.085] (n=10) | 0.075 [0.057, 0.094] (n=10) | 0.910 / 0.906 |
| model randomization | 0.062 [0.050, 0.074] (n=10) | 0.062 [0.051, 0.074] (n=10) | 0.901 / 0.901 |
| label randomization | 0.087 [0.070, 0.105] (n=10) | 0.080 [0.058, 0.105] (n=10) | 0.889 / 0.893 |

## Per-stratum coverage (mean over seeds)

| arm | space | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| trained model | probability | 0.913 | 0.909 | 0.906 | 0.918 | 0.906 |
| trained model | latent | 0.904 | 0.904 | 0.918 | 0.915 | 0.888 |
| model rand. | probability | 0.911 | 0.890 | 0.890 | 0.902 | 0.915 |
| model rand. | latent | 0.911 | 0.888 | 0.889 | 0.902 | 0.915 |
| label rand. | probability | 0.888 | 0.890 | 0.888 | 0.890 | 0.890 |
| label rand. | latent | 0.888 | 0.888 | 0.896 | 0.900 | 0.892 |

## Reading

**Uninformative.** The trained model did not reach the configured test R² on every seed (mean -0.015), so a flat coverage profile carries no information about the link and this run cannot support or refute Theorem 2(iii).
