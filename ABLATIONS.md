# ABLATIONS.md — experiment matrix

Machine-readable ablation matrix for certgnn. Parsed by the `ablation-run`
skill and `scripts/launch_sweep.py --matrix ABLATIONS.md --tier N`.
Claims, gates and thresholds are defined in `PREREGISTRATION.md`; this file
never restates a threshold, it references the gate id.

## Rules

1. **Seeds.** 5 seeds minimum for any cell. 10 seeds for headline numbers,
   which means every row with `gate` set or `blocking` set. Every reported
   number is a mean with a 95% bootstrap CI over seeds. Never a single run.
2. **Sweep discipline.** Every row varies its `axes` one factor at a time from
   the reference config `configs/base.yaml` (`sweep: one_factor`). Targeted
   two-factor sweeps (`sweep: two_factor`) list exactly two axes and are run
   only after the one-factor result exists. **Never the full cross-product.**
   `sweep: fixed` rows have no axes and run the reference config under the
   row's control condition.
3. **Failed ablations get reported, not dropped.** A cell that errors, times
   out, or returns a result against the claim goes into the table with its
   status. Never backfill from a neighbouring config, never omit a row from
   the paper's ablation table because it went the wrong way.
4. **Sanity rows ship with every result.** Tier 0 rows 0.1 to 0.5 are re-run
   on every configuration whose explanation numbers are reported
   (`certgnn/eval/sanity.py`).
5. **Every cell writes to `results/runs/<timestamp>_<gitsha>_<confighash>/`.**
   A number without a git SHA and config hash does not exist.
6. **Soft masks only** in every explainer axis. A hard-mask value is permitted
   only where a row explicitly names it as a negative control.
7. **Row status.** `specified` rows were dictated by the PI and are final.
   `draft` rows (Tiers 2 to 6) were drafted from the codebase and
   pre-registration and require PI approval before launch. `candidate` rows
   live only in the final section and are never launched until promoted.

## Schema

The parser collects every fenced `yaml` block in document order. Each block is
one row: a mapping with exactly the keys below (plus optional `notes`).

| key | type | meaning |
|---|---|---|
| `id` | string `"T.N"` | tier and row number, unique across the file |
| `name` | string | short name |
| `tier` | int 0..6 | must equal the integer part of `id` |
| `status` | `specified` \| `draft` \| `candidate` | see rule 7 |
| `blocking` | `null` \| `WEEK1` | `WEEK1` rows decide whether the paper exists |
| `claim` | string | theorem, claim (C1..C4) or gate the row defends |
| `defends` | string | the claim in one plain sentence |
| `substrate` | list of `splice` \| `connectome` \| `synthetic` | where the row runs |
| `axes` | mapping `config.key -> [values]` | dotted hydra keys relative to `configs/base.yaml`; `{}` for fixed rows |
| `sweep` | `fixed` \| `one_factor` \| `two_factor` | see rule 2 |
| `seeds` | int >= 5 | 10 when `gate` or `blocking` is set |
| `metrics` | list of strings | what is measured |
| `expected` | string | pre-stated expected outcome |
| `falsified_if` | string | the concrete result that falsifies the claim |
| `gate` | `null` \| `G1`..`G5` | gate whose verdict this row feeds |

Tiers: 0 sanity and negative controls, 1 theorem validation, 2 architecture,
3 explanation, 4 conformal, 5 shift, 6 efficiency.

---

## Tier 0 — sanity and negative controls

### 0.1 Cascading model parameter randomization

```yaml
id: "0.1"
name: cascading model parameter randomization
tier: 0
status: specified
blocking: null
claim: explanation validity prerequisite (Adebayo et al. 2018)
defends: >-
  Explanations depend on the learned parameters, not only on the input graph.
substrate: [splice, connectome]
axes:
  sanity.randomize_layers: [none, last, last2, last3, all]
sweep: one_factor
seeds: 5
metrics: [spearman_vs_trained_explanation, topk_jaccard_vs_trained, faithfulness]
expected: >-
  Similarity to the trained-model explanation decays monotonically as more
  layers are randomized (cascading from the output backwards) and reaches the
  random-explainer floor (row 0.4) at `all`.
falsified_if: >-
  Similarity at `all` stays above the random floor by more than its 95% CI.
  The explainer is then edge-detecting the input, and every explanation
  result in the paper is void until the explainer is replaced.
gate: null
```

### 0.2 Last-layer-only parameter randomization

```yaml
id: "0.2"
name: last-layer-only parameter randomization
tier: 0
status: specified
blocking: null
claim: explanation validity prerequisite (Adebayo et al. 2018)
defends: >-
  Explanations are sensitive to the readout head in isolation, not just to
  the whole network.
substrate: [splice, connectome]
axes:
  sanity.randomize_layers: [none, last]
sweep: one_factor
seeds: 5
metrics: [spearman_vs_trained_explanation, topk_jaccard_vs_trained]
expected: >-
  A statistically significant drop in similarity relative to `none`, smaller
  than the drop at `all` in row 0.1.
falsified_if: >-
  Similarity under `last` is within CI of `none`. The explanation then does
  not depend on the head that produces the logit it claims to explain.
gate: null
```

### 0.3 Label randomization then retrain

```yaml
id: "0.3"
name: label randomization then retrain
tier: 0
status: specified
blocking: null
claim: explanation validity prerequisite; G5 interpretability
defends: >-
  Explanation signal comes from the label-to-structure relationship the model
  learned, not from structural regularities of the input alone.
substrate: [splice, connectome]
axes:
  sanity.label_permutation: [false, true]
sweep: one_factor
seeds: 10
metrics: [ground_truth_window_recovery, certified_topk_recovery_vs_random, faithfulness]
expected: >-
  With permuted labels the retrained model is at chance and explanation
  recovery of annotated windows is at the random floor.
falsified_if: >-
  Recovery on permuted labels exceeds 2x random (the G5 threshold). The
  retrospective recovery metric would then be measuring the input and G5
  cannot be interpreted.
gate: G5
```

### 0.4 Random explainer floor

```yaml
id: "0.4"
name: random explainer floor
tier: 0
status: specified
blocking: null
claim: floor for every faithfulness number
defends: >-
  Every reported faithfulness value is compared against a random subset of
  the same size.
substrate: [splice, connectome, synthetic]
axes:
  explainer.method: [random]
  explainer.k: [5, 10, 20]
sweep: two_factor
seeds: 5
metrics: [faithfulness, latent_fidelity_gap]
expected: >-
  Defines the floor. The proposed explainer and every baseline beat it by more
  than the sum of the two CIs at every k.
falsified_if: >-
  The proposed explainer's faithfulness CI overlaps the random floor at any k.
gate: null
```

### 0.5 Degree-ranked explainer floor

```yaml
id: "0.5"
name: degree-ranked explainer floor
tier: 0
status: specified
blocking: null
claim: structural floor for every faithfulness number
defends: >-
  Explanations are not a proxy for node degree.
substrate: [splice, connectome, synthetic]
axes:
  explainer.method: [degree]
  explainer.k: [5, 10, 20]
sweep: two_factor
seeds: 5
metrics: [faithfulness, latent_fidelity_gap, topk_jaccard_vs_degree]
expected: >-
  Degree ranking beats random but is beaten by the proposed explainer at every
  k, and top-k overlap between the proposed explainer and degree ranking is
  below 0.5.
falsified_if: >-
  The proposed explainer does not beat degree ranking beyond CI, or its
  top-k overlap with degree ranking exceeds 0.8. The explanation is then a
  degree heuristic.
gate: null
```

### 0.6 Degree-preserving topology shuffle — WEEK 1 BLOCKING

```yaml
id: "0.6"
name: degree-preserving topology shuffle
tier: 0
status: specified
blocking: WEEK1
claim: C1 / G1 prerequisite; substrate is topologically informative
defends: >-
  The model's predictions and the resistance-sensitivity relation depend on
  the actual wiring, not on the degree sequence or node features alone.
substrate: [splice, connectome]
axes:
  sanity.topology_shuffle: [none, degree_preserving]
sweep: one_factor
seeds: 10
metrics: [test_accuracy, spearman_resistance_vs_neg_sensitivity, faithfulness]
expected: >-
  Under a degree-preserving rewiring with retraining, accuracy falls to the
  edge-free MLP level of row 0.7 and Spearman(resistance, -sensitivity)
  collapses toward zero.
falsified_if: >-
  Shuffled accuracy is within CI of unshuffled accuracy. The substrate is
  topologically inert; together with row 0.7 this triggers the G1 fallback
  (switch substrate) and the topology claims are unsupported on it.
gate: G1
notes: >-
  `certgnn.eval.sanity.shuffle_topology` currently permutes node labels
  (`perm[edge_index]`), which decouples features from positions but is not a
  degree-preserving rewiring. This row requires a true degree-preserving
  rewire (e.g. double-edge swaps) and a check that the degree sequence is
  unchanged and edge overlap with the original is below 0.2.
```

### 0.7 MLP on node features with edges removed — WEEK 1 BLOCKING

```yaml
id: "0.7"
name: MLP on node features with edges removed
tier: 0
status: specified
blocking: WEEK1
claim: C1 / G1 prerequisite; message passing is used
defends: >-
  The GNN's accuracy is not achievable from node features alone; the edges
  carry information the model uses.
substrate: [splice, connectome]
axes:
  model.arch: [mlp]
  substrate.edges: [none]
sweep: two_factor
seeds: 10
metrics: [test_accuracy, test_auroc]
expected: >-
  MLP accuracy is below the reference GNN by more than the sum of the CIs,
  and defines the accuracy floor that row 0.6 is compared against.
falsified_if: >-
  MLP accuracy is within CI of the reference GNN. Topology is inert on this
  substrate; with row 0.6 this triggers the G1 fallback.
gate: G1
```

### 0.8 Hadamard / BQN baseline, connectome only

```yaml
id: "0.8"
name: Hadamard / BQN non-message-passing baseline
tier: 0
status: specified
blocking: null
claim: connectome go/no-go; message-passing critique on brain graphs
defends: >-
  On the connectome substrate, message passing is not beaten by the simple
  non-message-passing baselines from the message-passing critique.
substrate: [connectome]
axes:
  model.arch: [hadamard, bqn]
sweep: one_factor
seeds: 5
metrics: [test_accuracy, test_auroc]
expected: >-
  The reference GNN matches or exceeds both baselines under equal tuning
  budget (`results/tuning_budget.md`).
falsified_if: >-
  Either baseline exceeds the GNN beyond CI. The connectome is then not a
  message-passing substrate for this task and the go/no-go tilts to splice.
gate: null
```

### 0.9 Degenerate-explanation detector

```yaml
id: "0.9"
name: degenerate-explanation detector
tier: 0
status: specified
blocking: null
claim: faithfulness numbers are not produced by degenerate masks (Azzolin et al.)
defends: >-
  Reported faithfulness is computed only on explanations that are neither
  near-empty nor near-full nor near-constant.
substrate: [splice, connectome, synthetic]
axes:
  explainer.mask_temperature: [0.1, 0.5, 1.0, 2.0]
  explainer.sparsity_weight: [0.0, 0.01, 0.1, 1.0]
sweep: two_factor
seeds: 5
metrics: [degenerate_fraction, faithfulness_on_nondegenerate]
expected: >-
  Fewer than 5% of explanations are flagged by
  `degenerate_explanation_check` at the reference config, and the flagged
  fraction is reported next to every faithfulness number.
falsified_if: >-
  More than 10% of explanations are flagged at the reference config. That
  explainer configuration is invalid and its faithfulness numbers are
  withdrawn, not filtered.
gate: null
```

### 0.10 Certified vs uncertified explanation agreement

```yaml
id: "0.10"
name: certified vs uncertified explanation agreement
tier: 0
status: specified
blocking: null
claim: certification is a guarantee on the explanation, not a different explanation
defends: >-
  The certified explanation is a refinement of the explainer's own ranking,
  so "certified explanation" is not a misnomer.
substrate: [splice, connectome]
axes:
  certify.enabled: [false, true]
sweep: one_factor
seeds: 5
metrics: [topk_jaccard_certified_vs_uncertified, rank_correlation, certified_subset_of_top2k]
expected: >-
  Certified nodes are a subset of the uncertified top-2k on more than 95% of
  instances and rank correlation between the two exceeds 0.8.
falsified_if: >-
  Agreement is within CI of the random floor. The certificate is then
  selecting nodes unrelated to the explainer's ranking.
gate: null
```

---

## Tier 1 — theorem validation

### 1.1 Solver vs pseudo-inverse agreement

```yaml
id: "1.1"
name: series-parallel solver vs Laplacian pseudo-inverse
tier: 1
status: specified
blocking: null
claim: Theorem 1 prerequisite; resistance computation is exact
defends: >-
  The O(E) series-parallel reduction returns the same effective resistance as
  the O(n^3) pseudo-inverse on every graph it accepts, and rejects every
  graph with a K4 minor.
substrate: [synthetic, splice]
axes:
  substrate.n_nodes: [10, 100, 1000]
sweep: one_factor
seeds: 5
metrics: [max_abs_resistance_error, k4_detection_false_negative_rate]
expected: >-
  Max absolute error below 1e-8 on all accepted graphs; zero false negatives
  on K4-containing graphs.
falsified_if: >-
  Any disagreement above 1e-8. Per the module docstring this is a bug in the
  reduction, not a tolerance issue; every resistance-based result is void
  until fixed.
gate: null
```

### 1.2 Resistance vs measured Jacobian sensitivity

```yaml
id: "1.2"
name: resistance vs measured Jacobian sensitivity
tier: 1
status: specified
blocking: null
claim: C1 / Theorem 1; G1
defends: >-
  Model sensitivity to a node decreases monotonically in its effective
  resistance to the target.
substrate: [splice, connectome]
axes: {}
sweep: fixed
seeds: 10
metrics: [spearman_resistance_vs_neg_sensitivity, p_value]
expected: >-
  Spearman rho exceeds the G1 threshold with p below the G1 threshold on
  every seed, using the reference architecture and depth.
falsified_if: >-
  The G1 threshold is not met. Pre-registered fallback applies (substrate is
  topologically inert; switch substrate).
gate: G1
```

### 1.3 Pendant vs bubble separation

```yaml
id: "1.3"
name: pendant vs bubble separation
tier: 1
status: specified
blocking: null
claim: C1 / Theorem 1, first clause
defends: >-
  Nodes on pendant paths have higher effective resistance to the target than
  nodes in bubbles at equal path length, and sensitivity follows that order.
substrate: [synthetic]
axes:
  substrate.synthetic.motif: [pendant, bubble]
  substrate.synthetic.path_length: [1, 2, 4, 8]
sweep: two_factor
seeds: 5
metrics: [resistance_pendant_minus_bubble, sensitivity_order_agreement]
expected: >-
  Resistance is strictly higher on the pendant path at every path length and
  the sensitivity ordering agrees on more than 95% of instances.
falsified_if: >-
  Sensitivity ordering is reversed on more than 5% of instances beyond CI at
  any path length.
gate: null
```

### 1.4 Depth sweep

```yaml
id: "1.4"
name: depth sweep
tier: 1
status: specified
blocking: null
claim: C1 / Theorem 1 is not a single-depth artefact
defends: >-
  The resistance-sensitivity relation holds across message-passing depths,
  not only at the reference depth.
substrate: [splice, connectome]
axes:
  model.depth: [1, 2, 3, 4, 6, 8]
sweep: one_factor
seeds: 5
metrics: [spearman_resistance_vs_neg_sensitivity, test_accuracy]
expected: >-
  Rho stays above the G1 threshold for every depth at which the receptive
  field covers the target's resistance ball; it may weaken at depth 8 from
  oversmoothing but does not change sign.
falsified_if: >-
  Rho drops below the G1 threshold at any depth from 2 to 6, or changes
  sign at any depth. Theorem 1 would then hold only at a tuned depth.
gate: null
```

### 1.5 Graph-transformer control

```yaml
id: "1.5"
name: graph-transformer control
tier: 1
status: specified
blocking: null
claim: Theorem 1 mechanism is message passing
defends: >-
  The resistance-sensitivity relation is a property of local message passing,
  not of the data, so a full-attention model without locality does not show
  it to the same degree.
substrate: [splice, connectome]
axes:
  model.arch: [graph_transformer]
sweep: one_factor
seeds: 5
metrics: [spearman_resistance_vs_neg_sensitivity, test_accuracy]
expected: >-
  Rho for the graph transformer is below the reference GNN's rho by more than
  the sum of the CIs.
falsified_if: >-
  The transformer's rho is within CI of the GNN's. The relation is then a
  property of the substrate rather than of message passing, and Theorem 1's
  mechanistic reading is unsupported even if the correlation stands.
gate: null
```

### 1.6 Synthetic link sweep — WEEK 1 BLOCKING

```yaml
id: "1.6"
name: synthetic link sweep across identity / logit / probit / cloglog
tier: 1
status: specified
blocking: WEEK1
claim: C2 / Theorem 2
defends: >-
  The observed output-space effect of a fixed latent perturbation scales with
  the link function's Jacobian at the operating point, and only the identity
  link is baseline-independent.
substrate: [synthetic]
axes:
  substrate.synthetic.link: [identity, logit, probit, cloglog]
sweep: one_factor
seeds: 10
metrics: [observed_effect_vs_baseline_curve, r2_vs_predicted_jacobian]
expected: >-
  Identity shows no baseline dependence. Logit, probit and cloglog each
  follow their predicted Jacobian shape (p(1-p) for logit, the Gaussian
  density at the probit quantile, and the asymmetric cloglog derivative) with
  R^2 above 0.9.
falsified_if: >-
  Any non-identity link has R^2 below 0.9 against its predicted Jacobian, or
  the identity link shows baseline dependence beyond CI. Theorem 2's
  first-order argument is then wrong in its simplest setting.
gate: G2
```

### 1.7 Observed effect vs baseline rate curve fit — WEEK 1 BLOCKING

```yaml
id: "1.7"
name: observed effect vs baseline rate curve fit
tier: 1
status: specified
blocking: WEEK1
claim: C2 / Theorem 2(i)
defends: >-
  On a real substrate the measured probability-space effect of a latent
  perturbation delta follows |delta| * p0 * (1 - p0) as a function of the
  baseline rate p0.
substrate: [splice, connectome]
axes:
  link.delta_latent: [0.1, 0.3, 0.7, 1.0]
sweep: one_factor
seeds: 10
metrics: [r2_curve_fit, fitted_delta_vs_applied_delta, effect_range_across_baseline_bins]
expected: >-
  R^2 above 0.9 against `predicted_observed_effect` at every delta and the
  fitted delta lies within CI of the applied delta.
falsified_if: >-
  R^2 below 0.9, or the curve is flat in p0 (effect range across baseline
  bins within CI of zero). The link correction is then unnecessary on this
  substrate and Theorem 2 is vacuous here; G2 fallback territory.
gate: G2
```

### 1.8 Explicit cross-instance rank reversal — WEEK 1 BLOCKING

```yaml
id: "1.8"
name: explicit cross-instance rank reversal
tier: 1
status: specified
blocking: WEEK1
claim: C2 / Theorem 2, cross-instance comparability
defends: >-
  Probability-space attribution scores reverse the ranking of node effects
  across instances with different baselines, while latent-space scores
  preserve it.
substrate: [synthetic, splice]
axes:
  link.baseline_pair: ["0.5_vs_0.9", "0.5_vs_0.95", "0.5_vs_0.99"]
sweep: one_factor
seeds: 10
metrics: [reversal_rate_probability_space, reversal_rate_latent_space]
expected: >-
  Probability-space reversal rate exceeds 20% of cross-instance pairs and
  grows with baseline separation; latent-space reversal rate is within CI of
  zero.
falsified_if: >-
  Probability-space reversal rate is within CI of the latent-space rate.
  Probability-space scores would then be cross-instance comparable in
  practice and Theorem 2's motivation is moot.
gate: G2
```

### 1.9 Conditional coverage in probability vs latent space — WEEK 1 BLOCKING

```yaml
id: "1.9"
name: conditional coverage in probability vs latent space
tier: 1
status: specified
blocking: WEEK1
claim: C2 / Theorem 2(iii); G2
defends: >-
  Conformal fidelity certificates built on probability-space scores have
  conditional coverage that varies with baseline rate; latent-space scores
  do not.
substrate: [splice, connectome]
axes:
  conformal.score: [probability_fidelity_gap, latent_fidelity_gap]
sweep: one_factor
seeds: 10
metrics: [max_min_conditional_coverage_gap_across_baseline_strata, marginal_coverage, infinite_quantile_fraction]
expected: >-
  Probability-space gap exceeds the G2 threshold and latent-space gap is
  below it, with marginal coverage at or above 1 - alpha for both.
falsified_if: >-
  Either G2 threshold is not met. Pre-registered fallback applies (Theorem 2
  narrows to a stated instance class).
gate: G2
notes: >-
  Strata with fewer than (1 - alpha) / alpha calibration points return an
  infinite quantile and report coverage 1.0. Those strata are excluded from
  the gap and their fraction is reported; a gate passed only through
  infinite quantiles is vacuous.
```

### 1.10 Epsilon sweep vs explanation size

```yaml
id: "1.10"
name: epsilon sweep vs explanation size
tier: 1
status: specified
blocking: null
claim: C3 / Theorem 4
defends: >-
  Minimal epsilon-sufficient explanation size is non-increasing in epsilon
  and never exceeds the resistance-ball bound at any epsilon.
substrate: [splice, connectome, synthetic]
axes:
  explainer.epsilon: [0.01, 0.05, 0.1, 0.2, 0.5]
sweep: one_factor
seeds: 5
metrics: [min_explanation_size, resistance_ball_size, bound_violation_rate]
expected: >-
  Size is monotone non-increasing in epsilon on every instance and the bound
  holds at every epsilon.
falsified_if: >-
  Size increases with epsilon on more than 5% of instances beyond CI (an
  implementation bug in the minimality search), or the bound is violated on
  any instance beyond numerical tolerance (Theorem 4 false).
gate: null
```

### 1.11 Measured vs predicted critical epsilon

```yaml
id: "1.11"
name: measured vs predicted critical epsilon
tier: 1
status: specified
blocking: null
claim: Theorem 3 (certified radius); G3
defends: >-
  The certificate's predicted critical epsilon never exceeds the measured
  epsilon at which the explanation stops being sufficient, and is not
  vacuously small.
substrate: [splice, connectome]
axes: {}
sweep: fixed
seeds: 10
metrics: [soundness_violation_rate, median_ratio_measured_over_predicted, nontrivial_certificate_fraction]
expected: >-
  Zero soundness violations; median ratio below 3; fraction of candidate
  nodes with a non-trivial certificate above the G3 threshold.
falsified_if: >-
  Any instance with predicted epsilon above measured epsilon. The
  certificate is unsound and Theorem 3's proof or implementation is wrong;
  stop. Separately, a non-trivial fraction below the G3 threshold triggers
  the G3 fallback (randomized smoothing).
gate: G3
```

### 1.12 Resistance-ball bound tightness

```yaml
id: "1.12"
name: resistance-ball bound tightness
tier: 1
status: specified
blocking: null
claim: C3 / Theorem 4 is not vacuous
defends: >-
  The resistance-ball bound on explanation size is tight enough to be
  informative, not merely true.
substrate: [splice, connectome, synthetic]
axes:
  topology.ball_radius_quantile: [0.1, 0.25, 0.5, 0.75]
sweep: one_factor
seeds: 5
metrics: [ratio_explanation_size_over_ball_size, bound_violation_rate]
expected: >-
  Ratio at most 1 on every instance and median ratio at least 0.3 at the
  reference radius.
falsified_if: >-
  Ratio above 1 on any instance beyond numerical tolerance (Theorem 4
  false). Median ratio below 0.1 means the bound holds but is vacuous and
  C3 is demoted to a remark.
gate: null
```

### 1.13 Rewiring budget vs explanation size

```yaml
id: "1.13"
name: rewiring budget vs explanation size
tier: 1
status: specified
blocking: null
claim: C4 / Theorem 5; G4
defends: >-
  Increasing the rewiring budget increases minimal epsilon-sufficient
  explanation size.
substrate: [splice, connectome]
axes:
  rewire.budget: [0.0, 0.05, 0.1, 0.2, 0.4]
sweep: one_factor
seeds: 10
metrics: [min_explanation_size, relative_increase_zero_to_max_budget, test_accuracy]
expected: >-
  Size is monotone non-decreasing in budget and the increase from zero to
  max budget meets the G4 threshold.
falsified_if: >-
  Size is non-monotone in budget beyond CI (Theorem 5 false), or the
  increase is below the G4 threshold (Theorem 5 true but uninteresting;
  demote to remark per fallback).
gate: G4
```

### 1.14 Global vs targeted rewiring on the Pareto plane

```yaml
id: "1.14"
name: global vs targeted rewiring on the Pareto plane
tier: 1
status: specified
blocking: null
claim: C4 / Theorem 5; resistance is the operative quantity
defends: >-
  Rewiring that targets resistance to the target node moves along the
  accuracy-vs-explanation-size frontier differently from untargeted global
  rewiring at equal budget.
substrate: [splice, connectome]
axes:
  rewire.strategy: [global_random, targeted_resistance, targeted_degree]
  rewire.budget: [0.05, 0.1, 0.2]
sweep: two_factor
seeds: 5
metrics: [test_accuracy, min_explanation_size, frontier_area]
expected: >-
  Targeted-resistance rewiring produces a larger explanation-size increase
  per unit accuracy loss than global or degree-targeted rewiring at every
  budget.
falsified_if: >-
  The three strategies are indistinguishable on the plane within CI.
  Resistance is then not the operative quantity in Theorem 5.
gate: null
```

### 1.15 Rayleigh monotonicity check

```yaml
id: "1.15"
name: Rayleigh monotonicity check
tier: 1
status: specified
blocking: null
claim: Theorem 5 prerequisite; resistance implementation respects Rayleigh's law
defends: >-
  Adding edges never increases effective resistance between any pair, which
  Theorem 5's proof relies on and which the resistance code must reproduce.
substrate: [synthetic, splice, connectome]
axes:
  rewire.mode: [add_only, swap]
sweep: one_factor
seeds: 5
metrics: [monotonicity_violation_rate, max_violation_magnitude]
expected: >-
  Zero violations above 1e-8 under `add_only`. Under `swap` violations are
  permitted by the law; their frequency is reported for context.
falsified_if: >-
  Any `add_only` violation above 1e-8. Rayleigh monotonicity is a theorem,
  so a violation is a bug in the resistance implementation; all resistance
  results are void until fixed.
gate: null
```

---

## Tier 2 — architecture (draft, requires approval)

### 2.1 Message-passing backbone sweep

```yaml
id: "2.1"
name: message-passing backbone sweep
tier: 2
status: draft
blocking: null
claim: C1 / Theorem 1 is architecture-general within message passing
defends: >-
  The resistance-sensitivity relation holds for the standard message-passing
  families, not only for the reference backbone.
substrate: [splice, connectome]
axes:
  model.arch: [gcn, gin, gat, sage]
sweep: one_factor
seeds: 5
metrics: [spearman_resistance_vs_neg_sensitivity, test_accuracy, faithfulness]
expected: >-
  Rho above the G1 threshold for every backbone.
falsified_if: >-
  Any message-passing backbone has rho below 0.3. Theorem 1 then needs an
  architecture restriction stated in the paper.
gate: null
```

### 2.2 Hidden width

```yaml
id: "2.2"
name: hidden width
tier: 2
status: draft
blocking: null
claim: results are not a capacity artefact
defends: >-
  Faithfulness, certificate non-vacuity and the resistance-sensitivity
  relation are stable across model capacity.
substrate: [splice, connectome]
axes:
  model.hidden: [32, 64, 128, 256]
sweep: one_factor
seeds: 5
metrics: [spearman_resistance_vs_neg_sensitivity, nontrivial_certificate_fraction, test_accuracy]
expected: >-
  All three metrics vary by less than their CI across widths once accuracy
  saturates.
falsified_if: >-
  Non-trivial certificate fraction falls below the G3 threshold at any width
  that reaches reference accuracy.
gate: null
```

### 2.3 Normalization and residual connections

```yaml
id: "2.3"
name: normalization and residual connections
tier: 2
status: draft
blocking: null
claim: Theorem 3 Lipschitz route is robust to standard architectural choices
defends: >-
  Layer normalization and residual connections change the Lipschitz constant
  but not the soundness of the certificate.
substrate: [splice, connectome]
axes:
  model.norm: [none, layer, batch]
  model.residual: [false, true]
sweep: two_factor
seeds: 5
metrics: [soundness_violation_rate, nontrivial_certificate_fraction, test_accuracy]
expected: >-
  Zero soundness violations in every cell; non-trivial fraction may vary.
falsified_if: >-
  Any soundness violation. The Lipschitz bound is then wrong for that
  architecture and it must be excluded from Theorem 3's scope.
gate: null
```

### 2.4 Readout

```yaml
id: "2.4"
name: readout
tier: 2
status: draft
blocking: null
claim: Theorem 1 assumes a target-node readout
defends: >-
  Resistance to the target is the right quantity for target-node readout;
  pooled readouts are outside Theorem 1's scope and should show weaker
  dependence.
substrate: [splice, connectome]
axes:
  model.readout: [target_node, mean_pool, attention_pool]
sweep: one_factor
seeds: 5
metrics: [spearman_resistance_vs_neg_sensitivity, test_accuracy]
expected: >-
  Rho is highest for target-node readout and lower for pooled readouts.
falsified_if: >-
  Pooled readouts show rho within CI of target-node readout. The
  "resistance to target" framing would then be unnecessary.
gate: null
```

### 2.5 Activation function

```yaml
id: "2.5"
name: activation function
tier: 2
status: draft
blocking: null
claim: Theorem 3 Lipschitz constants are correct per activation
defends: >-
  The certificate remains sound under activations with different Lipschitz
  constants and smoothness.
substrate: [splice]
axes:
  model.activation: [relu, gelu, tanh]
sweep: one_factor
seeds: 5
metrics: [soundness_violation_rate, nontrivial_certificate_fraction]
expected: >-
  Zero soundness violations for every activation.
falsified_if: >-
  Any soundness violation. The per-activation Lipschitz constant in the
  certificate is wrong.
gate: null
```

---

## Tier 3 — explanation (draft, requires approval)

### 3.1 Explainer method with equal tuning budget

```yaml
id: "3.1"
name: explainer method with equal tuning budget
tier: 3
status: draft
blocking: null
claim: the proposed explainer is competitive before certification
defends: >-
  Certification is applied to an explainer that is at least as faithful as
  standard baselines under equal hyperparameter search budget.
substrate: [splice, connectome]
axes:
  explainer.method: [gnnexplainer, pgexplainer, gradient, integrated_gradients, ours]
sweep: one_factor
seeds: 5
metrics: [faithfulness, latent_fidelity_gap, degenerate_fraction, wallclock]
expected: >-
  The proposed explainer is within CI of the best baseline on faithfulness
  and beats every baseline on certified coverage.
falsified_if: >-
  The proposed explainer is below the best baseline by more than the sum of
  the CIs. The method claim then rests on certification alone and the paper
  must say so.
gate: null
```

### 3.2 Mask temperature

```yaml
id: "3.2"
name: mask temperature
tier: 3
status: draft
blocking: null
claim: results are stable across the soft-mask regime
defends: >-
  The soft mask required by Theorems 3 to 5 does not need a tuned
  temperature to produce faithful, non-degenerate explanations.
substrate: [splice, connectome]
axes:
  explainer.mask_temperature: [0.1, 0.25, 0.5, 1.0, 2.0]
sweep: one_factor
seeds: 5
metrics: [faithfulness, degenerate_fraction, nontrivial_certificate_fraction]
expected: >-
  Faithfulness and certificate fraction vary by less than CI across the
  middle three temperatures; extremes are reported.
falsified_if: >-
  Only a single temperature gives non-degenerate explanations with
  non-trivial certificates.
gate: null
```

### 3.3 Sparsity regularization

```yaml
id: "3.3"
name: sparsity regularization
tier: 3
status: draft
blocking: null
claim: explanation size is driven by the certificate, not by a sparsity prior
defends: >-
  Minimal epsilon-sufficient size (Theorem 4) is recovered by the search, not
  imposed by the sparsity weight.
substrate: [splice, connectome]
axes:
  explainer.sparsity_weight: [0.0, 0.01, 0.1, 1.0]
sweep: one_factor
seeds: 5
metrics: [min_explanation_size, faithfulness, degenerate_fraction]
expected: >-
  Minimal size at the certified epsilon is stable across sparsity weights
  below 0.1.
falsified_if: >-
  Minimal size tracks the sparsity weight monotonically across the whole
  range. Size claims in Theorems 4 and 5 would then be measuring the prior.
gate: null
```

### 3.4 Explanation size k

```yaml
id: "3.4"
name: explanation size k
tier: 3
status: draft
blocking: null
claim: faithfulness comparisons are not k-specific
defends: >-
  The ordering of explainers by faithfulness is stable across explanation
  sizes.
substrate: [splice, connectome]
axes:
  explainer.k: [5, 10, 20, 50]
sweep: one_factor
seeds: 5
metrics: [faithfulness, random_floor_gap, degree_floor_gap]
expected: >-
  The proposed explainer beats both floors (rows 0.4 and 0.5) at every k.
falsified_if: >-
  The floors are not beaten at any k reported in the paper.
gate: null
```

### 3.5 Restricted-scope certificate on annotated instances

```yaml
id: "3.5"
name: restricted-scope certificate on annotated instances
tier: 3
status: draft
blocking: null
claim: application claim; G5
defends: >-
  On instances with an annotated functional window, the certified top-k
  recovers the window at a rate above random.
substrate: [splice]
axes:
  explainer.scope: [full, ground_truth_window]
sweep: one_factor
seeds: 10
metrics: [annotated_window_recovery_rate, recovery_over_random_ratio]
expected: >-
  Recovery ratio meets the G5 threshold; the restricted scope improves it
  further.
falsified_if: >-
  The G5 threshold is not met. Pre-registered fallback applies (application
  claim dropped; method claim stands).
gate: G5
```

---

## Tier 4 — conformal (draft, requires approval)

### 4.1 Calibration set size

```yaml
id: "4.1"
name: calibration set size
tier: 4
status: draft
blocking: null
claim: Proposition 22 finite-sample validity
defends: >-
  Marginal coverage holds at every calibration size and the quantile
  tightens as n grows; vacuous (infinite) quantiles are reported.
substrate: [splice, connectome]
axes:
  conformal.n_cal: [20, 50, 100, 200, 500, 1000]
sweep: one_factor
seeds: 5
metrics: [marginal_coverage, quantile_value, infinite_quantile_fraction]
expected: >-
  Coverage at or above 1 - alpha within 2 SE at every n; infinite-quantile
  fraction is 1 for n below (1 - alpha) / alpha and 0 above.
falsified_if: >-
  Coverage below 1 - alpha by more than 2 SE at any n with a finite
  quantile.
gate: null
```

### 4.2 Miscoverage level alpha

```yaml
id: "4.2"
name: miscoverage level alpha
tier: 4
status: draft
blocking: null
claim: Proposition 22 validity across alpha
defends: >-
  Coverage tracks 1 - alpha across the reported range.
substrate: [splice, connectome]
axes:
  conformal.alpha: [0.05, 0.1, 0.2]
sweep: one_factor
seeds: 5
metrics: [marginal_coverage, quantile_value]
expected: >-
  Coverage within 2 SE of 1 - alpha at each level.
falsified_if: >-
  Coverage below 1 - alpha by more than 2 SE at any level.
gate: null
```

### 4.3 Calibration method under no shift

```yaml
id: "4.3"
name: calibration method under no shift
tier: 4
status: draft
blocking: null
claim: weighted and Mondrian calibration reduce to split when they should
defends: >-
  Under exchangeable data, split, Mondrian and unit-weight weighted
  calibration all give valid coverage and near-identical quantiles.
substrate: [splice, connectome]
axes:
  conformal.method: [split, mondrian, weighted]
sweep: one_factor
seeds: 5
metrics: [marginal_coverage, quantile_value, infinite_quantile_fraction]
expected: >-
  All three methods have coverage within 2 SE of 1 - alpha; weighted with
  unit weights matches split exactly.
falsified_if: >-
  Any method undercovers by more than 2 SE under no shift.
gate: null
```

### 4.4 Nonconformity score choice

```yaml
id: "4.4"
name: nonconformity score choice
tier: 4
status: draft
blocking: null
claim: Theorem 2 applied to conformal scores
defends: >-
  Latent-space and variance-stabilized scores are valid and conditionally
  calibrated; the probability-space score is valid but not conditionally
  calibrated.
substrate: [splice, connectome]
axes:
  conformal.score: [latent_fidelity_gap, probability_fidelity_gap, arcsin_sqrt_gap]
sweep: one_factor
seeds: 5
metrics: [marginal_coverage, max_min_conditional_coverage_gap_across_baseline_strata]
expected: >-
  Marginal coverage valid for all three; conditional gap small for latent
  and arcsin-sqrt, large for probability.
falsified_if: >-
  Latent-space conditional gap is within CI of the probability-space gap.
gate: null
```

### 4.5 Mondrian stratum count vs vacuity

```yaml
id: "4.5"
name: Mondrian stratum count vs vacuity
tier: 4
status: draft
blocking: null
claim: conditional coverage is not bought with vacuous strata
defends: >-
  Increasing the number of baseline strata restores conditional coverage
  without pushing strata below the size at which the quantile is infinite.
substrate: [splice, connectome]
axes:
  conformal.n_strata: [2, 3, 5, 10, 20]
sweep: one_factor
seeds: 5
metrics: [max_min_conditional_coverage_gap_across_baseline_strata, infinite_quantile_fraction, median_stratum_size]
expected: >-
  Gap shrinks with stratum count while the infinite-quantile fraction stays
  at zero up to the reference count.
falsified_if: >-
  The gap only meets the G2 threshold at a stratum count where any stratum
  has an infinite quantile.
gate: null
```

---

## Tier 5 — shift (draft, requires approval)

### 5.1 Leave-one-site-out, unweighted split

```yaml
id: "5.1"
name: leave-one-site-out, unweighted split
tier: 5
status: draft
blocking: null
claim: shift diagnostic
defends: >-
  Establishes how much unweighted split conformal undercovers on a held-out
  site, which is the deficit Tier 5 then tries to remove.
substrate: [splice, connectome]
axes:
  shift.held_out_site: [all_sites]
sweep: one_factor
seeds: 5
metrics: [per_site_marginal_coverage, coverage_deficit]
expected: >-
  Coverage below 1 - alpha on shifted sites; magnitude reported per site.
falsified_if: >-
  Not a falsifiable claim; this row is a diagnostic. If no site undercovers
  beyond 2 SE, there is no measurable site shift and rows 5.2 to 5.5 are
  moot on this substrate.
gate: null
```

### 5.2 Weighted conformal with oracle likelihood ratio

```yaml
id: "5.2"
name: weighted conformal with oracle likelihood ratio
tier: 5
status: draft
blocking: null
claim: Tibshirani et al. 2019 guarantee, true weights
defends: >-
  With the true likelihood ratio, weighted conformal restores coverage under
  a known covariate shift.
substrate: [synthetic]
axes:
  shift.mu: [0.0, 0.25, 0.5, 1.0, 2.0]
sweep: one_factor
seeds: 5
metrics: [marginal_coverage, effective_sample_size, infinite_quantile_fraction]
expected: >-
  Coverage within 2 SE of 1 - alpha at every mu; effective sample size and
  infinite-quantile fraction reported alongside.
falsified_if: >-
  Coverage below 1 - alpha by more than 2 SE at any mu with effective sample
  size above 50. That is an implementation error, since the guarantee is a
  theorem for true weights.
gate: null
```

### 5.3 Weighted conformal with estimated likelihood ratio

```yaml
id: "5.3"
name: weighted conformal with estimated likelihood ratio
tier: 5
status: draft
blocking: null
claim: Tibshirani et al. 2019 guarantee degrades by at most half the L1 weight error
defends: >-
  With a calibrated density-ratio estimator fit on data disjoint from the
  calibration set, leave-one-site-out coverage is restored to within the
  weight-estimation error; fitting the estimator in-sample breaks the
  guarantee.
substrate: [splice, connectome]
axes:
  shift.weight_estimator: [logistic_calibrated, gbm_calibrated, kmm]
  shift.estimator_split: [held_out, in_sample]
sweep: two_factor
seeds: 10
metrics: [per_site_marginal_coverage, estimated_weight_l1_error, effective_sample_size, infinite_quantile_fraction, clip_fraction]
expected: >-
  Held-out estimators bring coverage to within max(2 SE, half the L1 weight
  error) of 1 - alpha on every site with effective sample size above 50;
  in-sample estimators undercover.
falsified_if: >-
  A held-out calibrated estimator leaves coverage below 1 - alpha minus
  2 SE minus half the L1 error on a site with effective sample size above
  50. Either overlap or the covariate-shift assumption (invariant score
  given covariates) fails, and weighted conformal is not applicable to this
  substrate's site shift; any undercoverage is then not attributable to
  conformal validity.
gate: null
```

### 5.4 Overlap diagnostic and weight clipping

```yaml
id: "5.4"
name: overlap diagnostic and weight clipping
tier: 5
status: draft
blocking: null
claim: positivity assumption of covariate-shift conformal
defends: >-
  Held-out sites overlap the training covariate support well enough that
  likelihood ratios are finite and the effective sample size is usable.
substrate: [splice, connectome]
axes:
  shift.weight_clip: [none, 10, 100]
sweep: one_factor
seeds: 5
metrics: [effective_sample_size_per_site, clip_fraction, max_weight, infinite_quantile_fraction]
expected: >-
  Effective sample size above 50 and clip fraction below 5% on every site
  at clip 100, with unclipped maximum weight finite.
falsified_if: >-
  Any site with effective sample size below 20 or clip fraction above 5%.
  That site is outside training support; its coverage number is reported as
  not certifiable rather than as a coverage failure.
gate: null
```

### 5.5 Shift magnitude sweep, weighted vs unweighted

```yaml
id: "5.5"
name: shift magnitude sweep, weighted vs unweighted
tier: 5
status: draft
blocking: null
claim: the weighted correction is what restores coverage
defends: >-
  Across increasing shift, unweighted coverage degrades monotonically while
  weighted coverage stays in band.
substrate: [synthetic]
axes:
  shift.mu: [0.0, 0.25, 0.5, 1.0, 2.0]
  conformal.method: [split, weighted]
sweep: two_factor
seeds: 5
metrics: [marginal_coverage, effective_sample_size]
expected: >-
  Split coverage decreases with mu; weighted coverage within 2 SE of
  1 - alpha at every mu.
falsified_if: >-
  Split coverage does not degrade with mu (the synthetic shift is not
  affecting scores), or weighted coverage leaves the band at any mu.
gate: null
```

---

## Tier 6 — efficiency (draft, requires approval)

### 6.1 Resistance solver scaling

```yaml
id: "6.1"
name: resistance solver scaling
tier: 6
status: draft
blocking: null
claim: series-parallel reduction is O(E)
defends: >-
  The series-parallel solver scales linearly in edges while the
  pseudo-inverse scales cubically in nodes.
substrate: [synthetic, splice]
axes:
  substrate.n_nodes: [100, 1000, 10000, 100000]
sweep: one_factor
seeds: 5
metrics: [wallclock_series_parallel, wallclock_pinv, fitted_exponent]
expected: >-
  Fitted exponent near 1 for the reduction and near 3 for the pseudo-inverse.
falsified_if: >-
  The reduction's fitted exponent exceeds 1.5, or it is not faster than the
  pseudo-inverse at 1000 nodes.
gate: null
```

### 6.2 Explanation wall-clock per instance

```yaml
id: "6.2"
name: explanation wall-clock per instance
tier: 6
status: draft
blocking: null
claim: the method is practical at substrate scale
defends: >-
  Per-instance explanation time is within an order of magnitude of the
  fastest baseline.
substrate: [splice, connectome]
axes:
  explainer.method: [gradient, gnnexplainer, ours]
  explainer.k: [10, 50]
sweep: two_factor
seeds: 5
metrics: [wallclock_per_instance, peak_memory]
expected: >-
  Proposed explainer within 10x of the gradient baseline.
falsified_if: >-
  Proposed explainer exceeds 10x the gradient baseline at the reference k.
gate: null
```

### 6.3 Certification overhead

```yaml
id: "6.3"
name: certification overhead
tier: 6
status: draft
blocking: null
claim: certification cost is dominated by explanation cost
defends: >-
  Producing the certificate adds bounded overhead on top of the explanation.
substrate: [splice, connectome]
axes:
  certify.enabled: [false, true]
sweep: one_factor
seeds: 5
metrics: [wallclock_per_instance, overhead_ratio]
expected: >-
  Overhead ratio below 2x at the reference config.
falsified_if: >-
  Overhead ratio above 5x.
gate: null
```

### 6.4 Per-test-point weighted quantile cost

```yaml
id: "6.4"
name: per-test-point weighted quantile cost
tier: 6
status: draft
blocking: null
claim: weighted conformal is affordable at leave-one-site-out scale
defends: >-
  The weighted quantile, which must be recomputed per test point under
  shift, is cheap enough for the Tier 5 rows.
substrate: [splice, connectome]
axes:
  conformal.n_cal: [100, 1000, 10000]
sweep: one_factor
seeds: 5
metrics: [wallclock_per_test_point, wallclock_total_per_site]
expected: >-
  Total per-site time under one minute at the reference calibration size.
falsified_if: >-
  Total per-site time above ten minutes; the calibration sort must then be
  hoisted out of the per-test-point loop before Tier 5 runs.
gate: null
```

---

## Candidate additions (not launched until promoted by the PI)

Rows the author believes are missing from the specified list. Each is
`status: candidate` and is ignored by the launcher.

### C.1 Explanation self-consistency across seeds

```yaml
id: "C.1"
name: explanation self-consistency across seeds
tier: 0
status: candidate
blocking: null
claim: noise floor for every explanation comparison
defends: >-
  Top-k overlap between explanations of the same instance from different
  training seeds bounds how much any between-method difference can mean.
substrate: [splice, connectome]
axes: {}
sweep: fixed
seeds: 10
metrics: [topk_jaccard_across_seeds, rank_correlation_across_seeds]
expected: >-
  Overlap well above the random floor; reported as the noise floor next to
  every between-method comparison.
falsified_if: >-
  Cross-seed overlap is within CI of the random floor. Explanations are then
  seed noise and no between-method comparison is meaningful.
gate: null
```

### C.2 Coverage under tied and discretized scores

```yaml
id: "C.2"
name: coverage under tied and discretized scores
tier: 4
status: candidate
blocking: null
claim: Proposition 22 lower bound survives ties
defends: >-
  Rounding fidelity gaps to float32 or coarser creates tied scores; the
  coverage lower bound must still hold (only the upper bound needs distinct
  scores).
substrate: [synthetic]
axes:
  conformal.score_rounding_decimals: [none, 3, 2, 1]
sweep: one_factor
seeds: 5
metrics: [marginal_coverage, tie_fraction]
expected: >-
  Coverage at or above 1 - alpha within 2 SE at every rounding level;
  over-coverage may grow with the tie fraction.
falsified_if: >-
  Coverage below 1 - alpha by more than 2 SE at any rounding level.
gate: null
```

### C.3 Series-parallel applicability per substrate

```yaml
id: "C.3"
name: series-parallel applicability per substrate
tier: 6
status: candidate
blocking: null
claim: solver route selection
defends: >-
  Determines what fraction of each substrate's graphs are series-parallel,
  hence which resistance path (O(E) reduction or O(n^3) pseudo-inverse) the
  substrate actually uses and what Tier 6 scaling claims apply to it.
substrate: [splice, connectome]
axes: {}
sweep: fixed
seeds: 5
metrics: [series_parallel_fraction, k4_minor_fraction]
expected: >-
  Splice graphs are series-parallel at a rate near 1; connectome graphs near
  0.
falsified_if: >-
  Not a falsifiable claim; informs which scaling claim the paper may make
  for each substrate.
gate: null
```

### C.4 Hard-mask negative control for Theorems 3 to 5

```yaml
id: "C.4"
name: hard-mask negative control for Theorems 3 to 5
tier: 3
status: candidate
blocking: null
claim: the soft-mask non-negotiable is load-bearing
defends: >-
  Replacing the soft mask with a hard binary mask breaks the Jacobian
  arguments, so the certificate becomes unsound; this justifies the
  soft-mask rule empirically rather than by assertion.
substrate: [synthetic, splice]
axes:
  explainer.mask_type: [soft, hard]
sweep: one_factor
seeds: 5
metrics: [soundness_violation_rate, nontrivial_certificate_fraction]
expected: >-
  Zero soundness violations under `soft`; a non-zero violation rate under
  `hard`.
falsified_if: >-
  Hard masks show zero violations across all seeds. The soft-mask
  restriction would then be unnecessary and the theorems' assumptions could
  be weakened.
gate: null
```
