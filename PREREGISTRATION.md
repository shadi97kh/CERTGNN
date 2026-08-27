# Pre-registration

Frozen once any gate has run. The hook in `.claude/settings.json` enforces this.

## Claims

- **C1 (Theorem 1).** On the chosen substrate, nodes on pendant paths attain
  higher effective resistance to the target than nodes in bubbles, and model
  sensitivity to a node decreases monotonically in its effective resistance.
- **C2 (Theorem 2).** Faithfulness scores measured in probability space have
  conditional coverage that varies with baseline rate; scores measured in latent
  space do not.
- **C3 (Theorem 4).** Minimal epsilon-sufficient explanation size is bounded by
  the size of a resistance ball around the target.
- **C4 (Theorem 5).** Increasing the rewiring budget increases minimal
  epsilon-sufficient explanation size.

## Gates and thresholds

| Gate | Test | Pre-registered threshold | Fallback if FAIL |
|---|---|---|---|
| G1 | Spearman(resistance, -sensitivity) across nodes | rho > 0.5, p < 0.01, >= 5 seeds | Substrate is topologically inert; switch substrate |
| G2 | Conditional coverage gap, prob-space vs latent-space | prob-space max-min gap > 0.10 AND latent-space gap < 0.05 | Theorem 2 narrows to a stated instance class; paper rests on T1/T3/T4/T5 |
| G3 | Certified radius non-vacuity | > 30% of candidate nodes receive a non-trivial certificate | Swap Lipschitz route for randomized smoothing |
| G4 | Pareto frontier curvature | explanation size increases >= 20% from zero to max rewiring budget | T5 is true but uninteresting; demote to remark |
| G5 | Retrospective recovery | annotated window in certified top-k at rate > 2x random | Application claim is dropped; method claim stands |

## Analysis plan

- 5 seeds minimum, 10 for headline numbers. Mean and 95% bootstrap CI.
- Holm-Bonferroni across the gate family.
- Test set touched once per gate. Model selection on validation only.
- Baselines receive equal hyperparameter search budget (documented in
  `results/tuning_budget.md`).

## What would falsify the paper

If G1 and G2 both fail, the substrate is wrong and the topology and link claims
are both unsupported. In that case the work does not proceed as an ICLR paper.
