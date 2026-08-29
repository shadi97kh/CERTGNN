# Can the refit find a class member, or only stay where it starts?

git SHA `94dd5a2`, config `2267b74e`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed, 3000 fit / 1000 held out. Held-out R² throughout.

**Arm 1** refits the NULL target φ̂ -- a function the reference realises, so known to be in the class -- from three starting points that are not the answer. This is the ceiling control that `closure_heldout.md` lacked: there the refit began at the target (initial loss 6.7e-19, below the 1e-10 early stop) and returned unchanged, so its 1.000000 measured arithmetic rather than search.

| width | depth | cold | other seed | other cell | copied |
|---|---|---|---|---|---|
| 16 | 1 | **0.998821** [0.9982, 0.9993] | 0.999378 | 0.999050 (from 16x2) | 2 tensors |
| 16 | 2 | **0.947572** [0.9244, 0.9703] | 0.905943 | 0.941726 (from 16x1) | 2 tensors |
| 16 | 3 | **0.776257** [0.6820, 0.8589] | 0.757236 | 0.752281 (from 16x1) | 2 tensors |
| 32 | 1 | **0.999369** [0.9990, 0.9997] | 0.999845 | 0.999500 (from 16x1) | 1 tensors |
| 32 | 2 | **0.825838** [0.7303, 0.9103] | 0.793126 | 0.823077 (from 16x1) | 0 tensors |
| 32 | 3 | **0.562632** [0.4404, 0.6891] | 0.528119 | 0.570145 (from 16x1) | 0 tensors |
| 64 | 1 | **0.997139** [0.9955, 0.9984] | 0.997304 | 0.997094 (from 16x1) | 1 tensors |
| 64 | 2 | **0.712236** [0.5620, 0.8561] | 0.721136 | 0.719367 (from 16x1) | 0 tensors |
| 64 | 3 | **0.450081** [0.3529, 0.5575] | 0.457235 | 0.459023 (from 16x1) | 0 tensors |
| 128 | 1 | **0.996875** [0.9962, 0.9975] | 0.995519 | 0.997067 (from 16x1) | 1 tensors |
| 128 | 2 | **0.899694** [0.8316, 0.9529] | 0.907960 | 0.899377 (from 16x1) | 0 tensors |
| 128 | 3 | **0.532812** [0.4000, 0.6714] | 0.571755 | 0.549081 (from 16x1) | 0 tensors |

**Arm 2** refits the WARP target in the grid-maximal class `128x3` from a cold start, with the null target in the same class as a control. `base` is the same-capacity warm-started refit, recomputed here so the comparison is internal to this run.

| width | depth | base warp | arm2 warp | rise | arm2 null (control) |
|---|---|---|---|---|---|
| 16 | 1 | 0.974799 | **0.893234** | -0.081566 [-0.1126, -0.0614] | 0.914534 |
| 16 | 2 | 0.993477 | **0.832698** | -0.160779 [-0.2046, -0.1205] | 0.849621 |
| 16 | 3 | 0.995925 | **0.738527** | -0.257398 [-0.3618, -0.1707] | 0.752966 |
| 32 | 1 | 0.974876 | **0.914226** | -0.060651 [-0.0790, -0.0435] | 0.929395 |
| 32 | 2 | 0.946338 | **0.739920** | -0.206417 [-0.2958, -0.1259] | 0.782882 |
| 32 | 3 | 0.940439 | **0.528680** | -0.411759 [-0.5102, -0.3120] | 0.594826 |
| 64 | 1 | 0.980403 | **0.950947** | -0.029457 [-0.0365, -0.0227] | 0.957296 |
| 64 | 2 | 0.923607 | **0.621883** | -0.301724 [-0.4467, -0.1685] | 0.684943 |
| 64 | 3 | 0.903422 | **0.321983** | -0.581439 [-0.6749, -0.4633] | 0.402921 |
| 128 | 1 | 0.988463 | **0.969817** | -0.018646 [-0.0224, -0.0145] | 0.970245 |
| 128 | 2 | 0.957526 | **0.835044** | -0.122482 [-0.2112, -0.0541] | 0.864963 |
| 128 | 3 | 0.924677 | — | — | — |

## Verdict

**Arm 1: search does NOT reliably work, and this invalidates the containment conclusion.** Refitting the null target from a cold random start reaches only held-out R² 0.450081 in the worst cell (64x3); 11 of 12 cells fall below 0.999 (128x1, 128x2, 128x3, 16x1, 16x2, 16x3, 32x2, 32x3, 64x1, 64x2, 64x3). The target is a function the reference realises, so it IS in the class; the optimiser simply does not travel to it. **The collapse of the warp target reported in `paper/tables/closure_heldout.md` is therefore confounded with optimisation failure, and does not establish that no member of the class equals ψ∘φ̂.** What both experiments establish is narrower: this refit procedure does not find such a member. Whether one exists is unresolved. The two warm alternatives behave the same way: starting from the same cell's reference trained at a different init seed gives worst-cell 0.457235, and starting from another cell's reference, copied where shapes match, gives 0.459023. Neither begins at the answer, so both are subject to the same search requirement as the cold start. **Arm 2 is uninterpretable.** **The control fails**: the larger class recovers even the plain latent φ̂ at only 0.402921 in the worst cell, so it cannot reliably represent the reference's own function and its performance on the warp target says nothing about containment. Best arm-2 warp agreement is 0.969817, best change -0.018646. **Together: no containment conclusion can be drawn from either experiment.** Arm 1 shows the optimiser does not reliably reach in-class targets, so Arm 2's outcome cannot be attributed to the class either. `paper/tables/closure_heldout.md` should be read as reporting what its procedure found, not what the class contains.
