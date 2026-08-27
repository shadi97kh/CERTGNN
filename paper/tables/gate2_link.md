# Gate 2 results (ABLATIONS 1.6 to 1.9), synthetic substrate

git dc7b132, config 0028d6eb, 10 seeds, mean [95% bootstrap CI].

## 1.6 link sweep

| link | ratio range across |η₀| strata | R² vs predicted Jacobian | cross-instance reversal rate |
|---|---|---|---|
| identity | 0.000 [0.000, 0.000] (n=10) | n/a | 0.000 [0.000, 0.000] (n=10) |
| logit | 0.211 [0.210, 0.212] (n=10) | 0.998 [0.998, 0.998] (n=10) | 0.204 [0.200, 0.209] (n=10) |
| probit | 0.378 [0.377, 0.379] (n=10) | 0.997 [0.997, 0.997] (n=10) | 0.311 [0.305, 0.317] (n=10) |
| cloglog | 0.334 [0.332, 0.336] (n=10) | 0.997 [0.997, 0.997] (n=10) | 0.309 [0.302, 0.315] (n=10) |

Expected: identity range and reversal rate at 0; non-identity R² > 0.9 (R² undefined for identity: ratio ≡ 1).

## 1.7 curve fit, ratio = C·[p₀(1−p₀)]^γ, predicted γ = 1, C = 1

| δ | γ̂ | Ĉ | R² (log fit) | R² vs predicted form |
|---|---|---|---|---|
| 0.1 | 1.000 [0.999, 1.000] (n=10) | 0.999 [0.998, 1.000] (n=10) | 0.999 [0.999, 0.999] (n=10) | 0.998 [0.998, 0.998] (n=10) |
| 0.3 | 0.997 [0.994, 0.999] (n=10) | 0.990 [0.986, 0.994] (n=10) | 0.989 [0.988, 0.990] (n=10) | 0.983 [0.982, 0.984] (n=10) |
| 0.7 | 0.982 [0.976, 0.988] (n=10) | 0.947 [0.937, 0.957] (n=10) | 0.944 [0.941, 0.946] (n=10) | 0.913 [0.908, 0.917] (n=10) |
| 1.0 | 0.965 [0.957, 0.973] (n=10) | 0.899 [0.884, 0.913] (n=10) | 0.893 [0.888, 0.897] (n=10) | 0.833 [0.824, 0.841] (n=10) |

## 1.8 cross-instance rank reversal (|Δ_a| > |Δ_b|, |obs_a| < |obs_b|)

| baseline pair (b vs a) | reversal rate, probability | reversal rate, latent | example (seed 0) |
|---|---|---|---|
| 0.5 vs 0.9 | 0.523 [0.507, 0.541] (n=10) | 0.000 [0.000, 0.000] (n=10) | a: p₀=0.900, Δ=+3.100, obs=+0.0950; b: p₀=0.500, Δ=+0.386, obs=+0.0954 |
| 0.5 vs 0.95 | 0.728 [0.709, 0.748] (n=10) | 0.000 [0.000, 0.000] (n=10) | a: p₀=0.950, Δ=+3.323, obs=+0.0481; b: p₀=0.500, Δ=-0.197, obs=-0.0490 |
| 0.5 vs 0.99 | 0.932 [0.920, 0.945] (n=10) | 0.000 [0.000, 0.000] (n=10) | a: p₀=0.990, Δ=+2.693, obs=+0.0093; b: p₀=0.500, Δ=-0.056, obs=-0.0139 |

## 1.9 conditional coverage (split conformal, α = 0.1, 5 strata of |logit p₀|)

| score space | max−min coverage gap | marginal coverage | per-stratum coverage (mean) |
|---|---|---|---|
| probability | 0.162 [0.153, 0.172] (n=10) | 0.906 [0.900, 0.911] (n=10) | 0.823, 0.859, 0.916, 0.944, 0.986 |
| latent | 0.020 [0.016, 0.024] (n=10) | 0.899 [0.893, 0.905] (n=10) | 0.898, 0.900, 0.898, 0.900, 0.900 |

Infinite quantile in any seed: False

## Gate G2 verdict

Pre-registered row: `| G2 | Conditional coverage gap, prob-space vs latent-space | prob-space max-min gap > 0.10 AND latent-space gap < 0.05 | Theorem 2 narrows to a stated instance class; paper rests on T1/T3/T4/T5 |`

- probability-space gap mean 0.1622 > 0.1: **True**
- latent-space gap mean 0.0203 < 0.05: **True**
- 95% CI bounds also clear both thresholds: True
- vacuous (infinite quantile): False

**PASS**
