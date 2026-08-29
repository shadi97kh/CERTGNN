# Identifiability of y = g(phi(x)) for nonlinear phi

git SHA `99aa3bb`, config `0821419f`, 10 seeds, mean [95% bootstrap CI].

## A. Constructive: is the explicit monotone twin inside the model class?

| G-P map | twin in class (R^2) | twin affinely related to phi_hat (R^2) |
|---|---|---|
| linear | 0.948 [0.947, 0.949] (n=10) | 0.947 [0.946, 0.948] (n=10) |
| pairwise | 0.967 [0.966, 0.968] (n=10) | 0.960 [0.959, 0.961] (n=10) |
| neural | 1.000 [0.999, 1.000] (n=10) | 0.965 [0.963, 0.967] (n=10) |

The twin (psi.phi_hat, g_hat.psi^-1) predicts identically by construction. If it is *in* the class, the warp is an unconstrained mode of that class.

## B. Empirical: how are equally-good refits related?

Read with the composite control at the foot of section C. Refits speak to the equivalence class only if they fit the same function; where their composite attributions disagree, they are different fits, not different representatives.

| G-P map | equally-good fits | affine R^2 | Spearman | recovers phi* (Spearman) |
|---|---|---|---|---|
| linear | 12.0 [12.0, 12.0] (n=10) | 0.987 [0.974, 0.998] (n=10) | 0.993 [0.986, 0.999] (n=10) | 0.405 [0.335, 0.482] (n=10) |
| pairwise | 2.2 [1.5, 3.0] (n=10) | 1.000 [1.000, 1.000] (n=7) | 1.000 [1.000, 1.000] (n=7) | 0.999 [0.999, 0.999] (n=10) |
| neural | 2.2 [1.4, 3.1] (n=10) | 0.995 [0.995, 0.996] (n=6) | 0.999 [0.999, 0.999] (n=6) | 0.999 [0.998, 0.999] (n=10) |

## C. Are attributions invariant across the equivalence class?

| G-P map | source | latent: within-instance cosine | latent: cross-instance magnitude Spearman |
|---|---|---|---|
| linear | constructed twin | 1.000 [1.000, 1.000] (n=10) | n/a |
| linear | random restarts | 0.993 [0.987, 0.999] (n=10) | n/a |
| pairwise | constructed twin | 1.000 [1.000, 1.000] (n=10) | 0.602 [0.565, 0.642] (n=10) |
| pairwise | random restarts | 1.000 [1.000, 1.000] (n=7) | 1.000 [1.000, 1.000] (n=7) |
| neural | constructed twin | 1.000 [1.000, 1.000] (n=10) | 0.774 [0.736, 0.805] (n=10) |
| neural | random restarts | 0.764 [0.747, 0.786] (n=6) | 0.598 [0.568, 0.637] (n=6) |

Composite attributions (grad of f = g(phi)), which are invariant algebraically and are shown only as a control:

| G-P map | within-instance cosine | cross-instance magnitude Spearman |
|---|---|---|
| linear | 0.993 [0.987, 0.999] (n=10) | 0.905 [0.806, 0.980] (n=10) |
| pairwise | 1.000 [1.000, 1.000] (n=7) | 0.997 [0.996, 0.999] (n=7) |
| neural | 0.764 [0.747, 0.786] (n=6) | 0.746 [0.728, 0.767] (n=6) |

## Verdict

**Attribution on a nonlinear G-P map is not identifiable.** The neural class is closed under a monotone warp of the latent (twin in class R^2 1.000), so the twin fits the data identically by construction and is a legitimate member of the same model class. Its latent attributions keep their direction within an instance (cosine 1.000) but not their magnitude across instances (Spearman 0.774). The linear class is not closed under the same warp (R^2 0.948), which is exactly why MAVE-NN can fix its modes affinely. Consequence for the field: any interpretation of a nonlinear surrogate's latent that depends on comparing attribution magnitudes across inputs -- which includes cross-locus motif-effect comparisons -- is reporting one arbitrary representative of an infinite equivalence class. Within-instance rankings survive. The multi-restart arm cannot corroborate this: the restarts' *composite* attributions disagree too (cosine 0.764), and the composite is invariant under reparameterization algebraically, so those restarts are different fits rather than different representatives of one function. Part B is therefore reported but carries no weight; the result rests on the construction, which is exact.
