---
name: repro-audit
description: Verify a claimed number is reproducible and traceable. Use before submission, or when the user asks to check a number in the paper.
---

# Reproducibility audit

For each number in `paper/`:

1. Locate the results directory that produced it.
2. Confirm the directory contains: git SHA, full config, seed list, environment
   lockfile, and raw per-seed values.
3. Re-run one seed and confirm it matches to tolerance.
4. Flag any number with no traceable source. These must be removed from the
   paper or regenerated, not left in place.
