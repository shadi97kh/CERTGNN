---
name: ablation-run
description: Launch an ablation sweep from the ablation matrix. Use when the user asks to run ablations, sweep a hyperparameter, or fill in a results table.
---

# Ablation run

1. Read `ABLATIONS.md` and identify the requested tier and rows.
2. Generate the config cross-product into `configs/generated/`.
3. Launch with at least 5 seeds per cell. Use `scripts/launch_sweep.py`.
4. Write results to `results/runs/`. Do not aggregate until all cells finish.
5. Aggregate with `scripts/aggregate.py`, which emits mean, 95% CI, and n.
6. If any cell fails, report the failure. Never silently drop a cell from the
   table or backfill it from a neighbouring config.
