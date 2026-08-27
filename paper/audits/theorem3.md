# Audit: Theorem 3 — certified radius (Lipschitz route)

Audited 2026-08-27 against the working tree at `3259c7e`. Statuses are
re-derived from the current code. Legend as in `theorem1.md`.

Formal statement and proof sketch: `PENDING PROPOSAL` — **UNVERIFIABLE**.
No certificate code exists: there is no `certgnn/certify/radius.py`, no
`experiments/gate3_vacuity.py` (the `Makefile` `gate3` target points at a
missing module), and `grep -rni "spectral\|lipschitz\|parametrizations"
certgnn/` returns nothing. What *has* changed since `theorems.md`: the
soft-mask machinery exists and the models exist, so A3.1 and A3.6 can now be
audited against code; A3.2 to A3.4 can be audited against the model
definitions even though nothing consumes them.

## Assumption table

| id | assumption | enforcing file:symbol | quoted lines | status | evidence |
|---|---|---|---|---|---|
| A3.1 | Masks are soft; the certified quantity is a Jacobian w.r.t. a continuous mask. | `certgnn/explain/masks.py::optimize_soft_mask`, `::binarize`, `::HardMaskWarning`; `certgnn/eval/faithfulness.py::_check_mask`, `::_candidates`; `certgnn/eval/sanity.py::degenerate_explanation_check` (A-SOFT) | `masks.py:246-247` `if bool((init[cand] <= 0).any()) or bool((init[cand] >= 1).any()): raise ValueError("init values on candidates must lie strictly inside (0, 1)")`; `:263` `m[cand] = torch.sigmoid(th)`; `:302-304` `if mask_type == "hard": warnings.warn(HARD_MASK_MESSAGE, HardMaskWarning, ...); mask = binarize(mask, acknowledge_assumption_5=True)`; `:330-334` `if not acknowledge_assumption_5: raise ValueError("binarize() produces a hard mask, which violates Assumption 5; ...")`; `:40` `warnings.simplefilter("always", HardMaskWarning)`; `faithfulness.py:85-86` `if bool((mask < 0).any()) or bool((mask > 1).any()): raise ValueError("mask entries must lie in [0, 1]")` (inclusive); `:101-108` `if bool(((vals == 0.0) \| (vals == 1.0)).all()): warnings.warn("Evaluating a HARD (binary) explanation mask. " + ...)`; `sanity.py:86` `frac = float((mask > 0.5).float().mean())` | **PARTIAL** (was UNENFORCED) | The optimizer's output is strictly inside (0,1) by construction and a hard mask can only be produced through a named flag that raises without acknowledgement and warns with it — that part is ENFORCED. It is PARTIAL overall because (a) every downstream consumer accepts a plain `torch.Tensor` and a binary one passes `faithfulness` with a *warning*, not an error (verified this session: `comprehensiveness(f, [1,1,0], 0)` returned 2.0 and emitted one `HardMaskWarning`); (b) `sanity.degenerate_explanation_check` accepts a hard mask silently (verified: returns `False`); (c) warnings are filterable by any caller; (d) there is no `SoftMask` type, `MaskResult.mask_type` is a free string; and, decisive for this theorem, (e) **no code path binds a mask to a torch model** — `gnn.py:84-91` `forward(self, x, edge_index, batch, ptr, target_idx)` has no mask or `edge_weight` argument, so the only `latent_fn` that has ever been differentiated w.r.t. a mask is `substrates/synthetic/oracle.py::Oracle.latent` (`oracle.py:138-140`). The mask-Jacobian the certificate needs has never been taken through a GNN. |
| A3.2 | The Lipschitz constant is a product of per-layer operator norms, each controlled by spectral normalization. | `certgnn/models/gnn.py`, `bqn.py`; a certificate constructor (does not exist) (A-SPEC) | `gnn.py:63` `convs = [GCNConv(dims[i], dims[i + 1]) for i in range(depth)]`; `:65` `SAGEConv(dims[i], dims[i + 1], aggr="add")`; `:68-74` `GINConv(nn.Sequential(nn.Linear(...), nn.ReLU(), nn.Linear(...)))`; `:81` `self.head = nn.Linear(hidden, 1)`; `bqn.py:46-47` `self.mlp_r = nn.Linear(n, n); self.mlp_g = nn.Linear(n, n)`; no `torch.nn.utils.parametrizations.spectral_norm` anywhere | **UNENFORCED** | Models exist and none of their linear maps is spectrally normalized; no code reads a Lipschitz constant from them; no certificate exists to raise on an unnormalized layer. Worse than at the time of `theorems.md`: there is now a concrete default model (`configs/base.yaml:8` `model: sage_sum`) whose layers are unconstrained, and any certificate built later must reject it or bound its norms explicitly. |
| A3.3 | Activations are 1-Lipschitz (ReLU / GELU / tanh); any other activation's constant is accounted for. | nothing | `gnn.py:94` `h = torch.relu(conv(h, edge_index))`; `gnn.py:70` `nn.ReLU()`; `bqn.py:36` `_ACTS = {"gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}`; `bqn.py:52` `h * self.mlp_r(corr) + self.mlp_g(h * h)` (quadratic) | **UNENFORCED** | Nothing accounts for activation constants. The GNN uses ReLU (1-Lipschitz) by construction, but the assumption as written is itself wrong about GELU: `max \|GELU'(x)\| = 1.1289` (verified this session by autograd on a fine grid), so a GELU network's constant is *not* the product of layer norms. The BQN's layer is quadratic in `h` and not globally Lipschitz at all; nothing marks it as outside T3's scope. |
| A3.4 | Normalization layers in eval mode with affine scale included; aggregation operator norm ≤ 1 (symmetric normalization) or its norm included. | nothing | `gnn.py:63` `GCNConv` (symmetric normalization, self-loops: norm ≤ 1); `gnn.py:65` `SAGEConv(..., aggr="add")` (sum aggregation: norm up to max degree); `gnn.py:68` `GINConv` (sum aggregation, `(1+ε)I + A`); `configs/base.yaml:8` `model: sage_sum     # diag 2026-08-27: sage_sum R2 0.77, gin 0.57, gcn 0.13` | **UNENFORCED** | No normalization layers exist (BatchNorm/LayerNorm absent), so that clause is vacuous. The aggregation clause is *violated by the default configuration*: the project moved from GCN to `sage_sum` for accuracy reasons, and sum aggregation has operator norm equal to the largest degree, which nothing computes or includes. Any certificate that assumes ≤ 1 aggregation is unsound on the default model. |
| A3.5 | The radius is stated in a named norm on the mask vector; the same norm is used for the measured critical ε in ABLATIONS 1.11. | `PENDING PROPOSAL`; nothing | — | **UNVERIFIABLE** (statement) and **UNENFORCED** (no code) | No norm is named, no certificate computes a radius, no experiment measures a critical ε. `masks.py` uses an L1 sparsity term (`:276` `l1 = m[cand].sum()`) and the infinity norm for convergence (`:288` `(cur_m - prev_m).abs().max()`), neither of which is documented as the certificate's norm. |
| A3.6 | Sufficiency is defined on the latent output. | `masks.py::latent_gap`, `::optimize_soft_mask`; `link.py` (A-LATENT) | `masks.py:103-107` `def latent_gap(latent_fn, mask): ... return float((_as_scalar(latent_fn(full)) - _as_scalar(latent_fn(mask))).abs())`; `:277-279` `gap = (z_full - ...).abs(); viol = torch.relu(gap - eps_inner); loss = sparsity_weight * l1 + mu * viol + 0.5 * gap_weight * viol**2`; `:316` `sufficient=gap_final <= eps + 1e-12`; contract `:13-14` "It must return logits, never probabilities" | **PARTIAL** | Stronger than in `theorems.md`: the explainer's sufficiency test is latent by construction and never touches `link.py`'s clamp. Remains PARTIAL because the latent-ness of `latent_fn` is a docstring contract only (a caller can bind `sigmoid(model(...))` and nothing detects it), and `link.py::latent_fidelity_gap` still accepts probabilities. |
| A3.7+ | `PENDING PROPOSAL` | — | — | **UNVERIFIABLE** | — |

## UNENFORCED

- A3.2 spectral normalization / Lipschitz product (A-SPEC) — no normalized layer, no certificate, no reader of layer norms.
- A3.3 activation constants — none accounted for; the GELU clause of the assumption is false as stated (1.129-Lipschitz).
- A3.4 aggregation norm ≤ 1 — default model `sage_sum` uses sum aggregation; nothing includes its norm.
- A3.5 certificate norm — no certificate code (also UNVERIFIABLE).

## UNVERIFIABLE

- Formal statement, proof sketch, A3.5's norm, A3.7+ — `PENDING PROPOSAL`.

## PARTIAL

- A3.1 (A-SOFT) — hard masks gated behind a flag and warnings; consumers accept tensors; no mask has ever entered a torch model.
- A3.6 (A-LATENT) — latent by construction in the explainer; `latent_fn` contract unverifiable; `link.py` still accepts probabilities.

## Single weakest point

**A-SPEC: there is still no certificate and no spectrally normalized layer, and the project has since chosen a default model (`sage_sum`, sum aggregation) whose per-layer operator norms are unbounded by construction — so the product-of-norms bound Theorem 3 needs is not merely unenforced but contradicted by the shipped configuration.** Close behind: the soft-mask Jacobian the theorem certifies has only ever been taken through the synthetic oracle, because the GNN `forward` accepts no mask at all.
