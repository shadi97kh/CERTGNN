# Development diagnostics (NOT results)

Raw logs of ad-hoc diagnostic runs used to configure the synthetic
pipeline-validation substrate and choose the reference GNN backbone. They
have no run directory, no config hash and a single seed, so nothing here
may be cited as a result. They are kept because the stats audit
(2026-08-27) correctly noted that the backbone choice in
`configs/base.yaml` and the substrate settings in
`configs/substrate/synthetic.yaml` were otherwise untraceable.

Disclosures:

- `2026-08-27_backbone_diag1_gcn_vs_sage.log`: GCN vs sum-aggregation SAGE
  on the synthetic tier-0 task without a target indicator; both report
  train/val R^2 per epoch. Code state: between commits a9c04bf and 0888545.
- `2026-08-27_backbone_diag2_target_indicator.log`: gcn / sage_sum / gin
  with the `is_target` feature, plus shuffled and edge-free MLP controls,
  600 train graphs, seed 0. **The R^2 values printed are on the test
  split of that synthetic draw.** The reference backbone `sage_sum` was
  chosen from this log (test R^2 0.773 vs gin 0.566 vs gcn 0.132). This
  is a design decision for a validation substrate with ground truth by
  construction, not a tuned result on real data; it must not be repeated
  on splice or connectome, whose test splits are touched once per gate.
- The substrate settings `p0: 0.5`, `regulator: far`,
  `structural_features: false`, `target_indicator: true` were each set
  after a diagnostic showed the previous setting made the task
  unidentifiable (sampled eta0 dominates Var(y); a random regulator has no
  marker; degree/distance features leak topology to the edge-free MLP).
  These make the synthetic tier-0 PASS partly a property of the
  configuration; it validates the controls, not any substrate.
- The GNN's `configs_tried` in tuning_budget.json is therefore 3 (three
  backbones), the MLP's and BQN's 1. Equal-effort is NOT satisfied for the
  BQN row (never run here) and must be arranged when a brain substrate
  exists.
