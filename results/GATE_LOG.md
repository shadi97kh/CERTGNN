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

---

## G2-M — ABLATIONS 1.9 recomputed with a TRAINED model (follow-up check, not a gate)

- experiment: `experiments/gate2_model.py`
- git SHA: 797a3df (tree clean)
- config hash: 5766d0a3 (configs/base.yaml, no overrides)
- run: results/runs/20260827T161346Z_797a3df_5766d0a3
- seeds: 10, mean [95% bootstrap CI]
- timestamp (UTC): 2026-08-27T16:13:46Z (run start); logged 2026-08-27
- artifacts: paper/figures/gate2_model.{png,pdf}, paper/tables/gate2_model.md

**This entry does not re-decide G2.** G2's verdict rests on its pre-registered
thresholds and is unchanged. `PREREGISTRATION.md` is untouched.

### Why it was run

G2's scores come from `Oracle.latent`, the generator's own latent function. The
data are made by pushing a latent effect through a sigmoid and the score is read
back off that same sigmoid, so the result cannot separate "the link matters for a
model" from "we generated through a link". This run replaces the scorer with a
trained GNN under a soft node mask and changes nothing else — same topology,
link, delta, resistance-ball explanation, conformal machinery, stratification,
and the same instances.

### Two substrate facts that had to be handled, not worked around

- The shipped label `y` is `mu`, a probability, so the inferred task is
  regression onto a probability and the trained head would live in probability
  space — against CLAUDE.md's latent-space rule. A "latent gap" read off such a
  model is a probability gap with a latent label. Models here are trained on
  `eta`.
- `eta0` appears in no node feature, so a model trained on the substrate as
  shipped **cannot represent the baseline**. Measured: r(model latent, logit p0)
  = -0.005 [-0.024, 0.014], val R² = 0.121. Its coverage profile is flat in both
  spaces for that reason alone, so that arm carries no information about the
  link. A second arm appends `eta0` as a node feature (`baseline_feature`, new,
  default off), giving r = 0.900 [0.895, 0.905], val R² = 0.965. Only the second
  arm can carry evidence.

### Result, oracle scorer vs model scorer, same instances

| scorer | arm | probability-space gap | latent-space gap |
|---|---|---|---|
| oracle | — | 0.1840 [0.1617, 0.2063] | 0.0363 [0.0267, 0.0450] |
| model | baseline hidden (uninformative) | 0.0433 [0.0367, 0.0510] | 0.0353 [0.0283, 0.0423] |
| model | baseline observable (informative) | 0.1243 [0.1030, 0.1400] | 0.2807 [0.2427, 0.3157] |

The oracle reproduces G2: probability-space gap five times the latent-space gap.
The trained model does not. On the informative arm the ordering **reverses** —
the latent-space gap is more than twice the probability-space gap, driven by the
top |logit p0| stratum, where model-latent coverage falls to 0.703 (per-stratum
0.983, 0.978, 0.952, 0.897, 0.703). That is a model-approximation effect at
extreme baselines, not a link effect.

### The sanity controls do not pass, and that limits the reading

| arm | control | probability-space gap | latent-space gap |
|---|---|---|---|
| baseline observable | model randomization | 0.2347 [0.1796, 0.2977] | 0.3680 [0.2867, 0.4473] |
| baseline observable | label randomization | 0.4260 [0.3807, 0.4637] | 0.4393 [0.4013, 0.4747] |

Both randomized models produce **larger** gaps than the trained model, not
smaller. The conditional-coverage gap therefore does not degrade toward zero when
the model is destroyed; it grows, because the statistic responds to
score-distribution heterogeneity and a broken model has plenty of it. Applied to
a model, max−min conditional coverage is not a clean read on the link. The
trained model's gaps sitting *below* both controls is the only thing the controls
support, and it says the trained model is not simply noise.

In the baseline-hidden arm the model's gaps (0.043 / 0.035) are within CI of its
own randomization controls (0.049 / 0.040), which is the predicted signature of an
arm with no signal.

No explanation was degenerate in any arm (0.0000 [0.0000, 0.0000]).

### What this supports

- The G2 pathology, as measured, is a property of the generative process. It did
  not reproduce with a trained model in the loop on the one arm where a trained
  model could in principle exhibit it.
- On this evidence **Theorem 2's empirical support is limited to the generative
  process, and the paper must say so** rather than cite G2 as evidence about
  trained models.
- The negative should be read as "the oracle result does not transfer", not as
  "the link provably does not matter for models", because the statistic fails its
  own sanity controls. A clean model-side test needs a null-calibrated
  heterogeneity statistic — the same defect the G2 disclosure already flagged for
  `n_test`.
- Deciding the field-level question needs published benchmarks and real
  explainers, not this substrate. That is `experiments/benchmark_reeval.py`,
  whose datasets and explainers are proposed in ABLATIONS.md under "candidate
  additions" and are awaiting approval.

---

## G2 (regenerated) — Conditional coverage gap, prob-space vs latent-space

- gate: G2
- git SHA: cdcc06a (tree clean)
- config hash: 5766d0a3 (configs/base.yaml, no overrides; the gate2 section is unchanged from 0028d6eb, the hash differs because base.yaml gained group selections, seed, torch_threads and tier0 keys)
- run: results/runs/20260827T194159Z_cdcc06a_5766d0a3 (full provenance: config, environment.lock, seeds, per-seed values, gate_stats.json with bootstrap p-values)
- seeds: 10, mean [95% bootstrap CI]
- measured: probability-space max−min coverage gap = 0.1622 [0.1527, 0.1721]; latent-space max−min coverage gap = 0.0203 [0.0165, 0.0240]; infinite quantile in any seed: no — identical to the dc7b132 run to every printed digit (seed 0 was independently verified bit-for-bit by the reproducibility audit)
- pre-registered threshold: prob-space max-min gap > 0.10 AND latent-space gap < 0.05
- comparison: 0.1622 > 0.10 (met); 0.0203 < 0.05 (met); CI bounds also clear both
- bootstrap p (CI-inversion, (b+1)/(B+1)): 0.0005 for each condition; Holm-adjusted across the gate family (only G2 has run): 0.0005
- verdict: **PASS** (unchanged)
- timestamp (UTC): 2026-08-27T19:41:59Z (run start)
- substrate: synthetic; scorer: generator's latent function (see the G2-M entry for the trained-model follow-up, which did not reproduce the pattern)
- artifacts: paper/figures/gate2_link.{png,pdf}, paper/tables/gate2_link.md now trace to this run
