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

### G2 disclosure (appended 2026-08-27 after the statistics audit; the verdict above is unchanged)

- Substrate: the G2 run is on the synthetic substrate at the PI's instruction. ABLATIONS rows 1.7 and 1.9 list splice/connectome; G2 must be re-run on the adopted real substrate before the paper can cite it as a real-data gate. On synthetic the link is applied exactly by the oracle; no trained model is involved.
- Sample size: a 2-seed debugging run at gate2.coverage.n_test=600 (120 instances per stratum), made before the pre-specified config was run, showed a latent-space gap of 0.067, above the 0.05 threshold. The max−min statistic has a null expectation that depends on n_test (about 0.063 at 120 per stratum, about 0.022 at 1000 per stratum, for 5 strata at coverage 0.9). The pre-specified config (n_test=5000, written before any run) was not changed; the measured latent gap 0.0203 is at that noise floor, and the probability-space gap 0.162 is roughly seven times its null expectation. Any future gate with this statistic should report the null-expected gap next to the measured one or use a null-calibrated heterogeneity statistic.
- Provenance: the dc7b132 run directory predates the environment lockfile and per-seed value files. Gate 2 is being regenerated at a clean commit to add them; the seed-0 re-run reproduced every recorded value bit-for-bit.
