## G2 — Conditional coverage gap, prob-space vs latent-space

- gate: G2
- git SHA: dc7b132 (tree clean)
- config hash: 0028d6eb (configs/base.yaml, no overrides)
- run: results/runs/20260827T062120Z_dc7b132_0028d6eb
- seeds: 10, mean [95% bootstrap CI]
- measured: probability-space max−min coverage gap = 0.1622 [0.1527, 0.1721];
  latent-space max−min coverage gap = 0.0203 [0.0165, 0.0240];
  infinite quantile in any seed: no
- pre-registered threshold: prob-space max-min gap > 0.10 AND latent-space gap < 0.05
- comparison: 0.1622 > 0.10 (met); 0.0203 < 0.05 (met); CI bounds also clear both
- verdict: **PASS**
- timestamp (UTC): 2026-08-27T06:21:20Z (run start); logged 2026-08-27
- substrate: synthetic (ground truth by construction); rows ABLATIONS 1.6–1.9
- artifacts: paper/figures/gate2_link.{png,pdf}, paper/tables/gate2_link.md
