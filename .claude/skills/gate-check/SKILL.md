---
name: gate-check
description: Run a decision gate and write a verdict. Use when the user says "run gate 1", "check gate 2", or asks whether a go/no-go threshold was met.
---

# Gate check

1. Read `PREREGISTRATION.md` and locate the gate's declared threshold. Do not
   proceed if the gate is not pre-registered.
2. Run the corresponding script in `experiments/`.
3. Compare the measured value to the pre-registered threshold. Do not
   reinterpret the threshold.
4. Append a verdict block to `results/GATE_LOG.md` with: gate id, git SHA,
   config hash, measured value, threshold, PASS or FAIL, timestamp.
5. If FAIL, state the pre-registered fallback and stop. Do not propose a new
   threshold or argue the result is close enough.
