# Tuning budget

Search effort per model and experiment, from tuning_budget.json in every run directory. Baselines must receive equal effort (PREREGISTRATION analysis plan); unequal rows are flagged.

| experiment | substrate | model | configs tried | epochs | gradient steps | search space | selection | runs |
|---|---|---|---|---|---|---|---|---|
| tier0_controls | synthetic | gnn | 1 | 60 | 600 | none: one pre-specified configuration per model (configs/model/*.yaml, tier0.bqn) | epoch with best validation metric; test split touched once | 10 |
| tier0_controls | synthetic | mlp | 1 | 60 | 600 | none: one pre-specified configuration per model (configs/model/*.yaml, tier0.bqn) | epoch with best validation metric; test split touched once | 10 |
| tier0_controls | synthetic | shuffled | 1 | 60 | 600 | none: one pre-specified configuration per model (configs/model/*.yaml, tier0.bqn) | epoch with best validation metric; test split touched once | 10 |

All models within each experiment received equal search effort.
