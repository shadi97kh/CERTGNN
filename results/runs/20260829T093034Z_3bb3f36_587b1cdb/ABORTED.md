# ABORTED — results in this directory are INVALID, not merely incomplete

Two seeds completed before this run was killed. Their numbers must not be
used, including the narrow cells.

At the then-fixed learning rate of 0.01 the reference fit in the 128-wide
cells collapsed onto the mean predictor: training loss froze at var(y)
within 200 epochs, `fit_r2` was exactly -0.0000 (128x2) and 0.0000 (128x3),
and `null_effect_size` was exactly 0. The closure those cells report
(0.999989 and 0.999994) is the agreement of two constant functions. It is
not evidence about closure, and because the asymptote fit weights the
high-parameter cells most heavily, this run was on course to report
convergence to 1 on the strength of the two cells that had learned nothing.
Seed 0's best cell (0.999999) and seed 1's (1.000000) both sit in that
regime.

Diagnosis, reproduced on CPU bit-for-bit against this run's own seed_0.json:

| cell  | lr 0.01  | lr 0.003 | lr 0.001 |
|-------|----------|----------|----------|
| 128x2 | -0.0000  | 0.6956   | 0.6832   |
| 128x3 |  0.0000  | 0.9446   | 0.7294   |

An optimizer artifact, not a property of the model class. Fixed in 0419c07
by choosing the rate per cell from a short pilot scored on the reference
fit's training loss alone, plus a degeneracy flag that excludes unfit cells
from the asymptote.

The one result here that survives: the null control read exactly 1.000000
in all 12 cells, so the pipeline's numerical floor is clean. That is
re-measured in the superseding run and does not need to be cited from here.

Superseded by `20260829T095929Z_0419c07_6afd0696`.
