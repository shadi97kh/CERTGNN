# Tuning budget

Search effort per model and experiment, from tuning_budget.json in every run directory. Baselines must receive equal effort (PREREGISTRATION analysis plan); unequal rows are flagged.

| experiment | substrate | model | configs tried | epochs | gradient steps | search space | selection | runs |
|---|---|---|---|---|---|---|---|---|
| gate2_link | - | none | 0 | 0 | 0 | no trained model: oracle substrate | n/a | 1 |
| gate2_model | - | sage_sum | 1 | 60 | not counted; epochs x ceil(n_train/batch_size) per fit | none: the pre-selected configs/model config was used as is | best validation epoch, validation split only | 2 |
| tier0_controls | synthetic | gnn | 1 | 60 | 600 | backbone chosen among gcn/sage_sum/gin in results/diagnostics/ (test-split R2 was inspected; disclosed there) | epoch with best validation metric; test split touched once | 20 |
| tier0_controls | synthetic | mlp | 1 | 60 | 600 | none: one pre-specified configuration (configs/model/mlp.yaml, tier0.bqn) | epoch with best validation metric; test split touched once | 20 |
| tier0_controls | synthetic | shuffled | 1 | 60 | 600 | backbone chosen among gcn/sage_sum/gin in results/diagnostics/ (test-split R2 was inspected; disclosed there) | epoch with best validation metric; test split touched once | 20 |

All models within each experiment received equal search effort by the recorded counts.

## Known discrepancies (results/tuning_budget_notes.md)

- Sweep `tier0_20260827T153742Z_271475f` (runs at 271475f and a523ecc): every `tuning_budget.json` records `configs_tried: 1` for `gnn` and `shuffled`. That is a code bug (fixed after the sweep): the reference backbone was chosen among three (gcn / sage_sum / gin) by inspecting test-split R² on the synthetic validation substrate, see `results/diagnostics/README.md`. The honest count is 3 for the GNN arms and 1 for the MLP; equal effort is NOT satisfied for tier 0 on the synthetic substrate, and the run artifacts were left as written rather than edited.
- `gate2_model` (797a3df) used the pre-selected `sage_sum` config as is; the same 3-vs-1 disparity therefore applies to it.
