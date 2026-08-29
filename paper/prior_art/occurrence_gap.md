# The occurrence gap

Written before the theorem, to state in advance what it does and does not
claim. The objection this pre-empts is the strongest one available and it is
correct as far as it goes:

> You hand-constructed a twin and showed it exists. Monotone
> reparameterization has always been unidentifiable. Show me this ever
> happens when someone actually fits a model.

Everything below is organised around that sentence. Nothing here is softened,
and where the answer is "we do not know", it says so.

---

## 1. What is established

**(E1) The equivalence class is non-trivial and representable.** For a neural
G-P map, an explicit monotone twin `(psi . phi_hat, g_hat . psi^-1)` is not
merely a mathematical object: the model class *contains* it. Measured as the
R^2 of the best in-class approximation to the warped latent, over 10 seeds,
three warp families and seven strengths (`paper/tables/identifiability_radius.md`):

| G-P map | sinusoid | spline | sigmoid mixture |
|---|---|---|---|
| linear | 0.960648 | 0.962380 | 0.951439 |
| pairwise | 0.961711 | 0.962605 | 0.930564 |
| **neural** | **1.000000** | **1.000000** | **1.000000** |

The neural class is closed to six significant figures under every family
tested, including the positively-weighted sigmoid mixture, which is the family
MAVE-NN fits as its own GE nonlinearity. The linear and pairwise classes are
not closed, which is the mechanism behind MAVE-NN's ability to fix their modes
affinely, and is a prediction of the same argument rather than a separate
finding.

**(E2) The twin is statistically indistinguishable at both real MPSA sample
sizes, and the radius is not finite.** Because closure is exact, the twin's
predictions are identical in exact arithmetic and the true prediction-space
effect is zero. Every tested strength up to the monotonicity limit is
indistinguishable at FAS exon 6 (n = 3072) and BRCA2 exon 17 (n = 32768). The
finite radii printed in the tables are artifacts of the pipeline's own
numerical floor, established by three checks rather than asserted: the effect
size shows no increasing trend with warp strength (Spearman of |effect|
against strength: -0.151, -0.118, -0.115); it sits within 1.8 to 2.5 SD of the
strength-0 null control (floor 0.031 +/- 0.019); and the classes that are *not*
closed show radii as large or larger than the closed class, which is backwards.

So the honest form of "the radius is R at sample size n" is: **R is not
bounded by the data at either n; it is bounded by the monotonicity limit.**

**(E3) Attribution magnitude across instances diverges within the class while
predictions do not.** Cross-instance attribution Spearman for the neural class,
as a function of warp strength:

| s | 0.10 | 0.25 | 0.40 | 0.55 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|
| sinusoid | 0.994 | 0.977 | 0.952 | 0.923 | 0.893 | 0.862 | 0.842 |
| spline | 0.996 | 0.986 | 0.969 | 0.945 | 0.914 | 0.877 | 0.849 |
| sigmoid mixture | 0.995 | 0.982 | 0.962 | 0.936 | 0.908 | 0.877 | 0.856 |

This is the quantity the interpretability claim turns on, and it degrades
monotonically across a range over which the model's predictions are
indistinguishable.

---

## 2. What is NOT established: occurrence

**We have no evidence that ordinary training explores the class.** This is not
a gap we are declining to fill for lack of time; it is a gap where our own
experiment returned a negative and the negative stands.

The experiment that would have answered it was the multi-restart arm of
`experiments/identifiability_probe.py` (Part B): fit the same neural G-P map
from many random initializations, keep the fits with equal loss, and ask
whether their recovered latents are affinely or merely monotonically related.
It failed, for a reason that is itself informative.

**The control that killed it.** The gradient of the composite `f = g . phi` is
invariant under reparameterization *algebraically*: for a twin, `grad f' =
grad f` exactly. So composite attributions are the control that says whether
two fits are the same function. Across neural restarts they agreed at only
**0.761 [0.747, 0.778]** within-instance cosine. Two fits whose composites
disagree are **different functions, not different representatives of one
function**, and their latent differences cannot be attributed to
reparameterization. Part B therefore measures fitting variability, not
equivalence-class exploration, and we report it as carrying no weight.

That correction was made after the first draft claimed the restarts
corroborated the finding. It was wrong and it stays corrected.

**Consequences for what may be claimed.** Absent occurrence evidence:

- We may **not** claim that two labs fitting the same data get different motif
  rankings.
- We may **not** claim that any published number is wrong, or that any
  pipeline has a bug. SQUID, MAVE-NN and the surrounding literature are not
  accused of an error by this work.
- We may **not** claim that re-running a published analysis would disagree
  with itself.

What we may claim is narrower and is about **meaning, not variability**: a
cross-instance comparison of attribution magnitudes on a nonlinear latent is
not a property of the fitted model's predictions. It is a property of an
arbitrary choice of representative from an equivalence class that the data
cannot narrow. Even if every optimizer in the field reliably picks the same
representative -- which would make occurrence empirically absent -- that choice
is made by the estimator and not by the data, so the quantity has no
data-grounded interpretation. **The claim is about what the reported quantity
means.**

---

## 3. Who is actually exposed

Honesty requires narrowing this further, because it substantially reduces the
blast radius and a reviewer will notice if we do not say it.

Our own result (E1) is that the **linear and pairwise classes are not closed**
under monotone warps. SQUID's headline analyses fit additive and
pairwise-interaction surrogates. Those are exactly the classes for which
MAVE-NN's gauge and diffeomorphic fixing applies and for which the equivalence
class is affine, hence for which cross-instance attribution magnitude ranking
*is* invariant (a positive constant scales every instance alike, and Spearman
is invariant under positive scaling).

**So SQUID's principal published analyses are, by our own argument, immune.**
The exposure is to work that fits a *custom or neural* G-P map -- which MAVE-NN
explicitly supports and explicitly does not gauge-fix -- and then interprets
the latent's attributions across sequences. That is the population the claim
addresses, and it must be stated as such rather than as a claim about the
field at large.

---

## 4. What evidence would establish occurrence, and whether it is obtainable

Ordered by strength. Each entry states the protocol, the control that makes it
interpretable, and whether we can actually run it.

### 4.1 Seed-replication of published surrogates (the direct test)

**Protocol.** Take a public genomic DNN oracle. Generate one in-silico MAVE
dataset with `squid`, fixed. Fit K surrogates with a **neural** G-P map to that
single dataset, differing only in seed. For each pair: (a) confirm equal
held-out loss; (b) confirm the *composite* predictions agree to within
measurement noise, by the same paired squared-residual test used for the
radius; (c) only then compare recovered latents and the reported motif effect
sizes across loci.

**Why (b) is not optional.** It is precisely the control our Part B failed.
Without it, any disagreement is confounded with ordinary fitting variability
and the experiment answers nothing.

**Obtainable: yes, in principle.** `squid-nn` is public (we have it cloned at
commit 08e3da0e), MAVE-NN is installable, and several oracles are public.
**Obstacles that must be stated:** SQUID's defaults use additive/pairwise
surrogates, so the neural arm must be constructed rather than taken off the
shelf; the oracle weights and the exact in-silico MAVE draws must be pinned or
the comparison is not like-for-like; and neither `captum` nor GraphXAI is
installed here, though neither is needed for this test.

**What a negative would mean.** If K seeds pass (b) and their cross-locus
rankings agree within noise, occurrence is absent for that pipeline, and the
paper's limitation section says so explicitly. That outcome does not
invalidate (E1)-(E3); it converts the contribution entirely into the
meaning-not-variability claim of section 2.

### 4.2 Attractor test: does training *stay* in a twin?

**Protocol.** Initialize training at an explicit twin `psi . phi_hat` and
continue training to convergence on the same data. If it converges back to
`phi_hat`, the class is not an attractor set and occurrence is unlikely by any
route. If it stays, the class is reachable and stable under the optimizer.

**Obtainable: yes, cheaply, on our own substrate,** and it is the natural next
experiment. **Weaker than 4.1** because we place the model in the twin rather
than observing training arrive there; it establishes reachability, not
spontaneous occurrence. It should be reported as such.

### 4.3 Implicit-bias characterization (the route that could settle it either way)

**Protocol.** Characterize which representative gradient descent selects.
If the optimizer has a systematic implicit bias -- say toward a minimum-norm
or maximally-smooth latent -- then the representative is effectively pinned by
the estimator, occurrence is absent by construction, and the meaning claim
becomes the whole contribution.

**Obtainable: partially.** Empirically measurable on our substrate; a theorem
is out of scope. This is the route by which the occurrence question could be
answered *negatively and permanently*, and it deserves stating because it
would be the most useful outcome for the field even though it is the least
favourable for a variability claim.

### 4.4 Published ensembles

**Protocol.** If any published work reports an ensemble of surrogates fit to
the same data with per-seed attribution spread, that spread is direct evidence.

**Obtainable: only by literature search,** and we have not found such a report.
SQUID reports averages over runs and cross-sequence attribution error, not
per-seed latent variability with the composite control of 4.1(b). Absent that
control, published spread would not be interpretable anyway.

---

## 5. What the limitations section must say

Verbatim intent, to be carried into the paper:

1. The equivalence class is shown to exist and to be representable by the
   model class. It is **not** shown to be visited by ordinary training. Our
   multi-restart experiment returned a negative and the composite-attribution
   control shows why it could not have answered the question.
2. The claim is about the interpretation of a reported quantity, not about an
   error in any existing pipeline, and not about run-to-run disagreement.
3. Classes that are not closed under monotone warps -- additive and pairwise
   G-P maps, which cover SQUID's principal analyses -- are not affected.
4. All evidence is on a synthetic substrate with a 15-node graph input, not on
   sequence data, and at an observation-noise level (sigma = 0.01) that was
   chosen for an earlier experiment rather than calibrated to MPSA
   measurements. The radius conclusion is a function of that sigma.
5. The within-instance invariance is established for autograd gradients. The
   field's primitive is in-silico mutagenesis, a finite difference, whose warp
   multiplier is `psi'(xi)` at an intermediate point by the mean value theorem
   and is therefore per-mutation rather than per-instance. The invariance does
   not transfer to ISM without further argument.
