# Theorems: statements, assumptions, enforcement, falsification

**Source status (2026-08-27): the SpliceCert proposal is not in this
repository.** Every field marked `PENDING PROPOSAL` must be filled verbatim
from the proposal. Nothing in those fields has been guessed. What is filled
now comes from three sources only, each labelled inline:

- `[PREREG]` — `PREREGISTRATION.md` (claims C1 to C4, gates G1 to G5).
- `[CODE]` — docstrings and behaviour in `certgnn/` at the current commit.
- `[TEST]` — theorem clauses named in `tests/`.

Do not cite this file for a theorem statement while any `PENDING PROPOSAL`
marker remains. The enforcement audit and the work queue at the end are
complete as of this commit and do not depend on the proposal text.

Consumers: `.claude/skills/theorem-audit` walks the assumption lists below
and writes findings to `paper/audits/<theorem>.md`. The proposal's own
assumption numbering (the skill refers to "Assumption 5" for soft masks)
must be reconciled with the `A-*` ids here when the proposal is pasted.

---

## Enforcement legend

| status | meaning |
|---|---|
| **ENFORCED** | a code path raises, or makes the violation impossible to express |
| **PARTIAL** | enforced on one path but bypassable, silently corrected, or only documented |
| **UNENFORCED** | nothing in `certgnn/` checks it |

"Enforced by docstring" is UNENFORCED. A test is evidence, not enforcement,
unless the tested function is the only entry point.

---

## Flagged assumptions that code must enforce

These five are the cross-cutting assumptions named by the PI. Each theorem
section below references them by id.

| id | assumption | used by | must be enforced by | status | evidence |
|---|---|---|---|---|---|
| **A-SOFT** | Explanation masks are soft (continuous in [0,1]); never hard binary. Jacobian arguments in Theorems 3 to 5 need a differentiable mask. | T3, T4, T5 | `certgnn/explain/masks.py` (named in `CLAUDE.md`) | **UNENFORCED** | `certgnn/explain/` contains only `__init__.py`; `masks.py` does not exist. The only mask-touching code, `certgnn/eval/sanity.py::degenerate_explanation_check` (line 52), thresholds at 0.5 and accepts a hard mask without complaint. |
| **A-SPEC** | Spectral normalization is active on every linear map whose operator norm enters a Lipschitz constant. | T3 | `certgnn/models/` (spectral-normalized layers) and the certificate in `certgnn/certify/` | **UNENFORCED** | `certgnn/models/` is empty. `grep -ri "spectral\|lipschitz" certgnn/` returns nothing. No certificate code exists. |
| **A-LATENT** | Wherever the link function appears, scores are computed on the latent (logit) output, never on probabilities. The head returns logits; conversion is display-only. | T2, T4 (epsilon-sufficiency), P22 | model head in `certgnn/models/`; `certgnn/certify/link.py::logit`, `::latent_fidelity_gap` | **PARTIAL** | `link.py::latent_fidelity_gap` (line 31) does convert to logit before differencing, and `::stratify_by_baseline` (line 46) stratifies on `\|logit p0\|`. But: (a) no model head exists, so "head returns logits" is not enforced anywhere; (b) `latent_fidelity_gap` *accepts probabilities* and `logit` clamps them to `[1e-6, 1-1e-6]` (line 17), silently capping `\|logit\|` at ~13.8, so saturation at extreme baselines is hidden rather than rejected; (c) `probability_fidelity_gap` (line 36) is exported as a public symbol and nothing restricts it to the negative-control row (ABLATIONS 1.9, 4.4). |
| **A-EXCH** | Calibration scores are exchangeable with the test score: calibration split disjoint from training, drawn from the same distribution as test (or reweighted by the true likelihood ratio under covariate shift), and the test set touched once. | P22, T2(iii), G2 | a split module (does not exist); `certgnn/certify/conformal.py`; `certgnn/substrates/base.py::Substrate.load(split)` | **UNENFORCED** | `conformal.py` takes a score tensor and cannot see where it came from. `Substrate.load(split)` returns graphs per split name but nothing asserts disjointness, group-awareness (same gene / same subject across splits), or single-touch of the test set. `weighted_conformal_quantile` states the likelihood-ratio assumption in its docstring (lines 43 to 49) and validates positivity, but accepts `inf` weights and cannot know whether weights were estimated in-sample. |
| **A-SP** | The exact series-parallel solver is trusted only after the graph is verified K4-minor-free (series-parallel reducible) between the terminals. | T1, T4, T5 (all resistance values) | `certgnn/topology/resistance.py::is_series_parallel` (line 31), `::resistance_series_parallel` (line 51), `::resistance_to_target` (line 109) | **ENFORCED** at the entry point, with two caveats | `resistance_to_target` calls `is_series_parallel(G)` before the reduction (line 114) and `resistance_series_parallel` raises `ValueError` rather than returning a wrong value if the 2-terminal reduction does not reach a single edge (lines 100 to 103). Caveats: (1) `resistance_to_target` *silently* falls back to the pseudo-inverse on `ValueError` (lines 121 to 123), which contradicts the reducer's own docstring ("Do not silently fall back") and hides reduction bugs behind a correct-but-O(n^3) answer; (2) the two paths disagree on weighted graphs: the reducer reads the `resistance` edge attribute (line 59) while `resistance_via_pinv` builds a 0/1 adjacency (lines 22 to 24) and the fallback in `resistance_to_target` drops attributes (lines 127 to 131). Tests only cover unit resistances. |

---

## Theorem 1 — effective resistance and model sensitivity

**Informal claim** `[PREREG C1]`: on the chosen substrate, nodes on pendant
paths attain higher effective resistance to the target than nodes in bubbles,
and model sensitivity to a node decreases monotonically in its effective
resistance. Clauses named in tests `[TEST]`: (ii) k parallel paths combine
harmonically; (iii) a pendant path of length d has R = d.

**Formal statement.** `PENDING PROPOSAL`

**Assumptions.**

| id | assumption | source | enforced by | status |
|---|---|---|---|---|
| A1.1 | Effective resistance is computed exactly (series-parallel reduction validated against the Laplacian pseudo-inverse). | `[CODE]` module docstring | A-SP | ENFORCED (see caveats) |
| A1.2 | Edge resistances: unit unless an edge carries a `resistance` attribute; both solver paths must agree on the convention. | `[CODE]` | `resistance.py` lines 22 to 24 vs 59 | **PARTIAL** — paths disagree on weighted graphs (A-SP caveat 2) |
| A1.3 | Target node is reachable from every candidate node (connected graph, or resistance defined only within the target's component). | `[CODE]` inferred: `pinv` of a disconnected Laplacian returns finite values that are not resistances | nothing | **UNENFORCED** — `is_series_parallel` returns False on any graph with two non-trivial components (reduces to >2 nodes), so disconnected graphs always take the pinv path and get meaningless cross-component values |
| A1.4 | The model is a message-passing GNN (locality); the claim is not made for full-attention models. | `[PREREG]` title; ABLATIONS 1.5 | `certgnn/models/` | **UNENFORCED** — no model exists |
| A1.5 | "Sensitivity" is a stated functional of the target logit's Jacobian with respect to node features, measured in latent space. | `PENDING PROPOSAL` for the exact functional | A-LATENT; sensitivity code does not exist | **UNENFORCED** |
| A1.6 | Message-passing depth covers the target's resistance ball (nodes outside the receptive field have zero sensitivity regardless of resistance). | `[CODE]` inferred; ABLATIONS 1.4 | nothing | **UNENFORCED** |
| A1.7+ | `PENDING PROPOSAL` (remaining assumptions in the formal statement) | | | |

**Proof sketch.** `PENDING PROPOSAL`

**Falsification** (cross-referenced to `ABLATIONS.md`):

- 1.1 solver vs pseudo-inverse disagreement above 1e-8 → A1.1 violated; all resistance results void.
- 1.2 Spearman(resistance, −sensitivity) below the G1 threshold → **G1 fails**; pre-registered fallback (switch substrate).
- 1.3 sensitivity ordering reversed on >5% of pendant-vs-bubble instances → first clause false.
- 1.4 rho below G1 threshold at any depth 2 to 6, or sign change → holds only at a tuned depth.
- 1.5 graph transformer shows rho within CI of the GNN → mechanism claim unsupported.
- 0.6, 0.7 (WEEK 1 BLOCKING): topology inert → G1 fallback before 1.2 is even meaningful.
- 1.15 Rayleigh violation under add-only rewiring → resistance implementation bug.

**Known attack surface.**

- Code-side, now: (a) the monotonicity claim is about a *trained* model, and
  a reviewer will ask whether the proof linearizes the network (random-walk /
  GCN-as-diffusion argument) and whether nonlinearity and training break it;
  (b) receptive-field truncation (A1.6) produces zero-sensitivity nodes at
  large resistance, which *inflates* the correlation for the wrong reason;
  (c) on connectome graphs, which are weighted and dense, the two solver
  paths disagree (A1.2) and `is_series_parallel` will almost always route to
  the pinv, so the "exact O(E) solver" claim applies to splice only
  (ABLATIONS C.3).
- Proof-side: `PENDING PROPOSAL`

---

## Theorem 2 — link-function correction, with Corollary

**Informal claim** `[PREREG C2]`: faithfulness scores measured in probability
space have conditional coverage that varies with baseline rate; scores
measured in latent space do not. Clauses named in code `[CODE]`: (i)
first-order observed effect of a latent shift δ at baseline p0 is
|δ|·p0(1−p0) (`link.py::predicted_observed_effect`); (iii) conditional
coverage fails without Mondrian stratification on |logit p0|
(`link.py::stratify_by_baseline`); (iv) equal latent shifts give equal
latent scores regardless of baseline (`tests/test_link.py`). Clause (ii) and
the Corollary: `PENDING PROPOSAL`.

**Formal statement.** `PENDING PROPOSAL`

**Corollary.** `PENDING PROPOSAL`

**Assumptions.**

| id | assumption | source | enforced by | status |
|---|---|---|---|---|
| A2.1 | Output is latent (logit); the link is applied only at display time. | `CLAUDE.md`; A-LATENT | `link.py::logit`, `::latent_fidelity_gap`; model head | **PARTIAL** (A-LATENT) |
| A2.2 | The link is the logistic sigmoid (Jacobian p(1−p)). Probit / cloglog appear only in the synthetic sweep. | `[CODE]` `sigmoid_jacobian` | `link.py` hardcodes sigmoid | ENFORCED by construction; generality claimed in ABLATIONS 1.6 is **UNENFORCED** in code |
| A2.3 | First-order regime: the latent perturbation δ is small relative to the curvature of the link at p0. | `[CODE]` "first-order" in `predicted_observed_effect` docstring | nothing | **UNENFORCED** — no magnitude check on δ; 1.7 sweeps δ up to 1.0 |
| A2.4 | Baseline p0 is bounded away from 0 and 1, or the statement is asymptotic in the saturation. | `[CODE]` inferred from the `EPS = 1e-6` clamp | `link.py::logit` clamp | **PARTIAL** — clamp silently caps rather than rejecting; `\|logit\| ≤ 13.8` |
| A2.5 | For clause (iii): calibration scores exchangeable within each baseline stratum, and strata are a function of the input (and fixed model) only, not of the masked output. | `[CODE]` `stratify_by_baseline(p0)`; A-EXCH | `Substrate.baseline_rate` (base.py line 36) provides p0 | **UNENFORCED** (A-EXCH) |
| A2.6 | Each stratum has at least (1−α)/α calibration points; otherwise its quantile is +∞ and its coverage is reported as 1.0. | `[CODE]` `conformal.py::split_conformal_quantile` line 15; `::empirical_coverage` line 144 | `mondrian_quantiles` returns `inf` (visible) but `empirical_coverage` hides it | **PARTIAL** — vacuous strata are not flagged; G2 could pass vacuously (ABLATIONS 1.9 note) |
| A2.7+ | `PENDING PROPOSAL` | | | |

**Proof sketch.** `PENDING PROPOSAL`

**Falsification:**

- 1.6 (BLOCKING) synthetic link sweep: non-identity link with R² < 0.9 against its predicted Jacobian, or identity link showing baseline dependence → clause (i) wrong in its simplest setting.
- 1.7 (BLOCKING) curve fit on a real substrate: R² < 0.9 or flat in p0 → correction unnecessary here; Theorem 2 vacuous on this substrate.
- 1.8 (BLOCKING) cross-instance rank reversal: probability-space reversal rate within CI of latent-space rate → cross-instance comparability motivation moot. This is the row for the Corollary once its statement is known.
- 1.9 (BLOCKING) conditional coverage gap: either G2 threshold missed → **G2 fails**; pre-registered fallback (narrow to a stated instance class).
- 4.4, 4.5: latent-space gap within CI of probability-space gap; or gap met only at a stratum count with an infinite quantile.

**Known attack surface.**

- Code-side, now: (a) the first-order approximation (A2.3) — a reviewer will
  take p0 → 1 and δ = 1 and show the predicted effect misses the true
  sigmoid difference by a factor that grows with |logit p0|; the fit in 1.7
  must be reported per baseline bin, not pooled; (b) the `EPS` clamp (A2.4)
  makes two instances with p0 = 1−1e-7 and p0 = 1−1e-9 identical in latent
  space, which is a discontinuity the "baseline-invariant" claim (iv) does
  not survive at the boundary; (c) clause (iii)'s G2 gap can be met through
  vacuous strata (A2.6).
- Proof-side: `PENDING PROPOSAL`

---

## Theorem 3 — certified radius (Lipschitz route)

**Informal claim** `[PREREG G3]`: a candidate node receives a non-trivial
certificate — a radius in mask space within which the explanation remains
ε-sufficient — derived from a Lipschitz constant of the network; the
pre-registered fallback if >70% of certificates are trivial is randomized
smoothing.

**Formal statement.** `PENDING PROPOSAL`

**Assumptions.**

| id | assumption | source | enforced by | status |
|---|---|---|---|---|
| A3.1 | Masks are soft; the certified quantity is a Jacobian with respect to a continuous mask. | `CLAUDE.md` "Theorems 3 through 5"; A-SOFT | `certgnn/explain/masks.py` | **UNENFORCED** (file does not exist) |
| A3.2 | The Lipschitz constant is a product of per-layer operator norms, each controlled by spectral normalization. | A-SPEC | `certgnn/models/` | **UNENFORCED** (no models, no certificate code) |
| A3.3 | Activations are 1-Lipschitz (ReLU / GELU / tanh); any other activation's constant is accounted for. | ABLATIONS 2.5 | nothing | **UNENFORCED** |
| A3.4 | Normalization layers are in eval mode and their affine scale enters the constant; the aggregation operator has norm ≤ 1 (symmetric normalization) or its norm is included. | ABLATIONS 2.3 | nothing | **UNENFORCED** |
| A3.5 | The radius is stated in a named norm on the mask vector, and the same norm is used when measuring the critical ε in ABLATIONS 1.11. | `PENDING PROPOSAL` | nothing | **UNENFORCED** |
| A3.6 | Sufficiency is defined on the latent output. | A-LATENT | `link.py` | **PARTIAL** |
| A3.7+ | `PENDING PROPOSAL` | | | |

**Proof sketch.** `PENDING PROPOSAL`

**Falsification:**

- 1.11: any instance with predicted critical ε above the measured critical ε → certificate **unsound**; stop. Non-trivial fraction below the G3 threshold → **G3 fails**; fallback to randomized smoothing.
- 2.3, 2.5: any soundness violation under a normalization / residual / activation variant → that architecture leaves Theorem 3's scope.
- C.4 (candidate): hard masks must produce violations; if they do not, the soft-mask restriction is not load-bearing.

**Known attack surface.**

- Code-side, now: the product-of-norms bound is the standard first target —
  it is vacuous past a few layers (the reason G3's fallback exists), and
  message-passing aggregation adds a factor per layer that is ≤ 1 only under
  symmetric normalization. With `certgnn/models/` empty, every assumption in
  this theorem is currently unenforced; this theorem has the largest gap
  between statement and code in the repository.
- Proof-side: `PENDING PROPOSAL`

---

## Theorem 4 — resistance-ball bound on explanation size

**Informal claim** `[PREREG C3]`: minimal ε-sufficient explanation size is
bounded by the size of a resistance ball around the target.

**Formal statement.** `PENDING PROPOSAL`

**Assumptions.**

| id | assumption | source | enforced by | status |
|---|---|---|---|---|
| A4.1 | Masks are soft; ε-sufficiency is evaluated on a soft-masked forward pass. | A-SOFT | `certgnn/explain/masks.py` | **UNENFORCED** |
| A4.2 | ε-sufficiency is measured on the latent output. | A-LATENT | `link.py::latent_fidelity_gap` | **PARTIAL** |
| A4.3 | Effective resistance to the target is exact. | A-SP | `resistance.py::resistance_to_target` | ENFORCED (caveats) |
| A4.4 | "Minimal" size is either computed exactly (small graphs) or lower-bounded; an explainer's returned size is an upper bound and cannot test the theorem. | ABLATIONS 1.10, 1.12 | nothing (no explain module) | **UNENFORCED** |
| A4.5 | The ball is defined over `Substrate.candidate_nodes` (base.py line 24), not all nodes. | `[CODE]` | `Substrate` protocol | PARTIAL — protocol only, no caller |
| A4.6+ | `PENDING PROPOSAL` (in particular the relation between ε and the ball radius, and any dependence on the Lipschitz constant of Theorem 3) | | | |

**Proof sketch.** `PENDING PROPOSAL`

**Falsification:**

- 1.10: size increases with ε on >5% of instances → minimality search bug; bound violated on any instance → **Theorem 4 false**.
- 1.12: ratio explanation-size / ball-size above 1 on any instance → false; median ratio below 0.1 → true but vacuous, C3 demoted to a remark.

**Known attack surface.**

- Code-side, now: A4.4 — minimal ε-sufficient sets are combinatorial; if the
  ablation uses the explainer's own output as "minimal size", the bound is
  unfalsifiable (an upper bound compared against an upper bound). Row 1.12
  must use exhaustive search on small synthetic graphs or a certified lower
  bound.
- Proof-side: `PENDING PROPOSAL`

---

## Theorem 5 — rewiring budget and explanation size, with Corollary

**Informal claim** `[PREREG C4]`: increasing the rewiring budget increases
minimal ε-sufficient explanation size. `[PREREG G4]` adds the
non-triviality requirement (≥ 20% increase from zero to max budget).

**Formal statement.** `PENDING PROPOSAL`

**Corollary.** `PENDING PROPOSAL`

**Assumptions.**

| id | assumption | source | enforced by | status |
|---|---|---|---|---|
| A5.1 | Rewiring only *adds* edges (Rayleigh monotonicity: adding edges never increases effective resistance). Edge swaps are outside the theorem. | ABLATIONS 1.15 | rewire module (does not exist) | **UNENFORCED** |
| A5.2 | Resistance is recomputed exactly after rewiring. | A-SP | `resistance.py` | ENFORCED (caveats) |
| A5.3 | Masks are soft. | A-SOFT | `certgnn/explain/masks.py` | **UNENFORCED** |
| A5.4 | Whether the model is held fixed or retrained after rewiring, and which the statement covers. | `PENDING PROPOSAL` | nothing | **UNENFORCED** |
| A5.5 | Budget is measured as a fraction of edges (ABLATIONS `rewire.budget`) and the statement's monotonicity is in that quantity. | `PENDING PROPOSAL` | nothing | **UNENFORCED** |
| A5.6+ | `PENDING PROPOSAL` | | | |

**Proof sketch.** `PENDING PROPOSAL`

**Falsification:**

- 1.13: size non-monotone in budget beyond CI → **Theorem 5 false**; increase below the G4 threshold → **G4 fails**, demote to remark.
- 1.14: global, resistance-targeted and degree-targeted rewiring indistinguishable on the accuracy / size plane → resistance is not the operative quantity.
- 1.15: any Rayleigh violation under add-only rewiring → resistance implementation bug (A5.2), not a theorem failure.

**Known attack surface.**

- Code-side, now: (a) if the rewiring strategy in the experiments includes
  removals (`rewire.mode: swap`), Rayleigh monotonicity does not apply and
  the corollary presumably fails; (b) if the model is retrained after
  rewiring (A5.4), the theorem is about a different model per budget and the
  "explanation size" comparison conflates topology with training; (c) G4's
  20% threshold is an effect-size claim that the theorem (a monotonicity
  statement) does not itself make.
- Proof-side: `PENDING PROPOSAL`

---

## Proposition 22 — conformal fidelity certificate

**Informal claim** `[CODE]`: the latent fidelity gap
|logit p_masked − logit p_full| is a valid nonconformity score requiring no
ground truth (`link.py::latent_fidelity_gap` docstring), so split, Mondrian
and weighted conformal calibration on it give finite-sample marginal
(and, with Mondrian strata, per-stratum) coverage ≥ 1−α.

**Formal statement.** `PENDING PROPOSAL`

**Assumptions.**

| id | assumption | source | enforced by | status |
|---|---|---|---|---|
| A22.1 | Calibration and test scores are exchangeable. | A-EXCH | split module (does not exist) | **UNENFORCED** |
| A22.2 | Grouped structure respected: instances sharing a gene (splice) or a subject / session (connectome) do not straddle the calibration / test boundary. | inferred from `Substrate.load(split)` | nothing | **UNENFORCED** |
| A22.3 | Finite-sample (n+1) correction. | `[CODE]` | `conformal.py::split_conformal_quantile` line 14; `::weighted_conformal_quantile` +∞ atom (lines 117 to 130) | **ENFORCED** |
| A22.4 | Scores are computed in latent space. | A-LATENT | `link.py::latent_fidelity_gap` | **PARTIAL** |
| A22.5 | Under covariate shift: weights are the *true* likelihood ratio dP_test/dP_train, finite (overlap), and not estimated on the calibration data. | `[CODE]` docstring lines 43 to 49; proof-check 2026-08-27 | `weighted_conformal_quantile` validates positivity only | **UNENFORCED** — `inf` weights pass (`inf > 0`), two infinite weights return a finite quantile; in-sample estimation is undetectable; coverage degrades by ½·E\|w − ŵ\| with estimated weights |
| A22.6 | Mondrian strata are a function of the input and fixed model only. | `[CODE]` `stratify_by_baseline(p0)` | nothing verifies p0 is the unmasked output | **UNENFORCED** |
| A22.7 | Test set touched once per gate; model selection on validation only. | `[PREREG]` analysis plan | nothing | **UNENFORCED** |
| A22.8 | An infinite quantile is reported as "not certifiable", not as coverage 1.0. | `[CODE]` `empirical_coverage` line 144 | nothing | **UNENFORCED** — currently reports 1.0 |
| A22.9+ | `PENDING PROPOSAL` | | | |

**Proof sketch.** `PENDING PROPOSAL` (expected: weighted exchangeability
lemma of Tibshirani et al. 2019 with unit weights for the split case).

**Falsification:**

- 4.1, 4.2, 4.3: coverage below 1−α by more than 2 SE at any n, α, or method under no shift → implementation error (the guarantee is a theorem under exchangeability).
- 1.9 (BLOCKING), 4.4, 4.5: conditional coverage claims → see Theorem 2.
- 5.2: oracle-weight coverage out of band with n_eff > 50 → implementation error.
- 5.3: held-out calibrated ŵ leaves coverage below 1−α − 2 SE − ½·L1 error on a site with n_eff > 50 → overlap or covariate-shift assumption fails on this substrate; undercoverage is **not** attributable to conformal validity.
- 5.4: n_eff < 20 or clip fraction > 5% on a site → that site is not certifiable; reported as such, not as a coverage failure.
- C.2 (candidate): ties / discretized scores — lower bound must survive.

**Known attack surface.**

- Code-side, now: A22.2 is the first thing a reviewer familiar with the
  substrates will ask — splice windows from the same gene and connectome
  sessions from the same subject are not exchangeable across a random
  split, and nothing in the repo does a grouped split. Second: A22.5 —
  Tier 5 coverage numbers with an estimated ŵ carry no finite-sample
  guarantee at 1−α, so the leave-one-site-out table must report n_eff, the
  +∞ fraction, and an L1 weight-error estimate next to every coverage
  number or the numbers are uninterpretable.
- Proof-side: `PENDING PROPOSAL`

---

## Work queue — everything currently UNENFORCED or PARTIAL

Ordered by how many theorems depend on it. Each item names the enforcement
point to create or fix. This list is the deliverable; nothing on it is
optional for the theorems to be citable.

| # | assumption | theorems | what to build / fix | where |
|---|---|---|---|---|
| 1 | **A-SOFT** soft masks only | T3, T4, T5 | Create `certgnn/explain/masks.py` with a `SoftMask` type whose constructor rejects values outside (0,1) exclusive of exactly-{0,1} masks (or asserts `mask.dtype.is_floating_point and 0 < mask.std()`), and make every explainer / certificate entry point accept only that type. Add a hard-mask negative-control path that is explicitly named (ABLATIONS C.4). | `certgnn/explain/masks.py` (new) |
| 2 | **A-EXCH** exchangeable calibration split, grouped, single-touch | P22, T2(iii), G2, all of Tier 4 and 5 | Create a split module that (a) performs a grouped split keyed on `Substrate`-provided group ids (gene / subject), (b) asserts train ∩ cal ∩ test = ∅ by instance and by group, (c) records the split hash into `results/runs/<...>/`, (d) exposes a test-set access counter. Extend the `Substrate` protocol with `group_id(data)`. | `certgnn/eval/splits.py` (new); `certgnn/substrates/base.py` |
| 3 | **A-SPEC** spectral normalization where a Lipschitz constant is used | T3 | Build the model package with spectral-normalized linear maps; have the certificate constructor *read* the Lipschitz constant from the layers and raise if any layer on the path is not normalized. | `certgnn/models/` (new); `certgnn/certify/radius.py` (new) |
| 4 | **A-LATENT** head returns logits; scores never see probabilities | T2, T3, T4, P22 | (a) Model head returns logits and has no sigmoid; (b) change `latent_fidelity_gap` to take *logits* (`z_full, z_masked`) so a probability cannot be passed without an explicit conversion, and keep `logit()` for display / stratification only; (c) replace the silent `EPS` clamp with a `ValueError` on `p` outside `(0,1)` and document the saturation bound; (d) move `probability_fidelity_gap` under a `negative_controls` namespace. | `certgnn/certify/link.py`; `certgnn/models/` |
| 5 | **A-SP** caveats | T1, T4, T5 | (a) Make the pinv path honour the `resistance` edge attribute (weighted Laplacian) so both solvers agree on weighted graphs, and add a weighted-graph agreement test; (b) make the fallback in `resistance_to_target` log which path was taken and expose it (`exact_used: bool`) rather than silently switching; (c) reject or special-case disconnected graphs (resistance is +∞ across components; pinv returns finite garbage). | `certgnn/topology/resistance.py` |
| 6 | A22.5 finite, non-estimated-in-sample weights | P22, Tier 5 | Add `np.isfinite` checks in `weighted_conformal_quantile` (raise); make `test_weight` required (no default of 1.0); add a caller-side protocol in the Tier 5 experiment that fits ŵ on a disjoint split and records n_eff, +∞ fraction, clip fraction, and an L1 error estimate. | `certgnn/certify/conformal.py`; `experiments/` (Tier 5) |
| 7 | A22.8 / A2.6 vacuous quantiles invisible | P22, T2(iii), G2 | `empirical_coverage` and `conditional_coverage_gap` return (or the caller records) the fraction of infinite quantiles; gate G2 excludes vacuous strata explicitly. Extend `q`'s type to accept a per-test-point tensor for the weighted method. | `certgnn/certify/conformal.py` |
| 8 | A1.3 target reachability | T1 | Assert connectivity to the target (or restrict to the target's component) before computing resistance. | `certgnn/topology/resistance.py::resistance_to_target` |
| 9 | A1.4, A1.6 message-passing model, depth covers ball | T1 | Exists only once `certgnn/models/` exists; record depth and ball radius in the run config so ABLATIONS 1.4 can check A1.6. | `certgnn/models/` |
| 10 | A4.4 minimal size vs explainer size | T4 | Implement exhaustive minimal-set search for small synthetic graphs and a certified lower bound for real substrates; never test the bound against the explainer's own output. | `certgnn/explain/` (new) |
| 11 | A5.1, A5.4, A5.5 add-only rewiring, fixed model, budget definition | T5 | Rewire module with `mode: add_only` as the default and a Rayleigh assertion after each step; statement of whether the model is retrained lives in the config and is echoed in the results table. | `certgnn/topology/rewire.py` (new) |
| 12 | A2.2, A2.3 sigmoid-only, first-order regime | T2 | Parameterize the link (identity / logit / probit / cloglog) so ABLATIONS 1.6 is not a synthetic-only special case; assert or report the δ magnitude relative to the curvature bound in 1.7. | `certgnn/certify/link.py` |
| 13 | `sanity.shuffle_topology` is a node permutation, not a degree-preserving rewire | ABLATIONS 0.6 (WEEK 1 BLOCKING) | Replace with double-edge swaps and assert the degree sequence is unchanged. Not a theorem assumption, but it gates G1. | `certgnn/eval/sanity.py::shuffle_topology` |
