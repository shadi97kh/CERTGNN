# Audit: Theorem 4 — resistance-ball bound on explanation size

Audited 2026-08-27 against the working tree at `3259c7e`. Statuses are
re-derived from the current code. Legend as in `theorem1.md`.

Formal statement and proof sketch: `PENDING PROPOSAL` — **UNVERIFIABLE**.
In particular the relation between ε and the ball radius, and any
dependence on Theorem 3's Lipschitz constant (A4.6+), are unknown. What can
be audited: the ball construction (`masks.py::resistance_ball_init`), the
exact minimal-set ground truth on the synthetic substrate
(`oracle.py::minimal_sufficient_set*`), and the soft-mask / latent-space
requirements. No experiment for ABLATIONS 1.10 / 1.12 exists yet.

## Assumption table

| id | assumption | enforcing file:symbol | quoted lines | status | evidence |
|---|---|---|---|---|---|
| A4.1 | Masks are soft; ε-sufficiency is evaluated on a soft-masked forward pass. | `certgnn/explain/masks.py::optimize_soft_mask`, `::binarize`, `::resistance_ball_init` (A-SOFT) | `masks.py:148-149` `if not (0.0 < outside < inside < 1.0): raise ValueError("need 0 < outside < inside < 1")`; `:246-247` `raise ValueError("init values on candidates must lie strictly inside (0, 1)")`; `:263` `m[cand] = torch.sigmoid(th)`; `:330-334` `binarize` raises without `acknowledge_assumption_5=True`; `:302-304` hard path warns; `faithfulness.py:101-108` warns (does not raise) on a binary mask; `sanity.py:86` `(mask > 0.5)` accepts hard masks silently | **PARTIAL** (was UNENFORCED) | The warm start is strictly soft (`:141` "the warm start is a soft mask, not a hard pre-selection") and the optimizer cannot emit a binary mask without the flag. PARTIAL because downstream consumers accept any tensor with a warning at most, and because the mask has never entered a torch model (`gnn.py:84-91` `forward` takes no mask) — the only "soft-masked forward pass" in the repository is `oracle.py:92-126` `_conductance`, where the mask enters as edge conductance `m_u * m_v` (`:101`). Whether the theorem's forward pass is that mechanism or a masked GNN is `PENDING PROPOSAL`. **Definitional tension, see weakest point:** the ground-truth *minimal* set is evaluated with hard indicator masks (`oracle.py:82-88` `set_mask`: `m = torch.zeros(...); m[self.index[v]] = 1.0`; `:151-153` `is_sufficient` uses it), and `Oracle.kappa` (`:134-135`) accepts them without warning. |
| A4.2 | ε-sufficiency is measured on the latent output. | `masks.py::latent_gap`, `::optimize_soft_mask`; `oracle.py::Oracle.latent`, `::is_sufficient`; `link.py` (A-LATENT) | `masks.py:104` docstring "`\|z(1) - z(mask)\|` in latent space (Theorem 2: never in probability)"; `:277` `gap = (z_full - _as_scalar(latent_fn(m)).to(torch.float64)).abs()`; `oracle.py:16-17` "An epsilon-sufficient explanation is a node set S ... with `\|eta(1_S) - eta(1)\| <= eps` in latent space"; `:152` `gap = abs(self.latent_full() - float(self.latent(self.set_mask(keep))))`; `link.py:16-18` clamp; `link.py:31-33` `latent_fidelity_gap(p_full, p_masked)` takes probabilities | **PARTIAL** | Latent by construction on both the explainer and the oracle path. Remains PARTIAL because `latent_fn`'s latent-ness is a docstring contract (`masks.py:13-14`) and `link.py` still exposes a probability-accepting, clamping gap. |
| A4.3 | Effective resistance to the target is exact. | `resistance.py::resistance_to_target` (A-SP); `masks.py::resistance_ball_init` | `resistance.py:114` `if exact and is_series_parallel(G):`; `:121-123` silent pinv fallback; `masks.py:134` `exact: bool = True`; `:150` `R = resistance_to_target(G, target, exact=exact)`; `:142-143` docstring "`exact=False` uses the Laplacian pseudo-inverse directly (same values, O(n^3))"; `experiments/gate2_link.py::_fidelity_scores` calls `resistance_ball_init(..., exact=False)` | **ENFORCED at the entry point, with caveats** | Same caveats as `theorem1.md` A1.1/A1.2/A1.3, plus one specific to the ball: `exact=False` is documented as value-identical but is only so on unit-resistance graphs (verified: weighted triangle 2.0 vs 0.667), and disconnected graphs yield finite cross-component radii (verified) which would place unreachable nodes *inside* a ball. Synthetic graphs are unit-weight and connected by construction, so the ball is correct there. |
| A4.4 | "Minimal" size is computed exactly (small graphs) or lower-bounded; the explainer's returned size is an upper bound and cannot test the theorem. | `certgnn/substrates/synthetic/oracle.py::Oracle.minimal_sufficient_set`, `::minimal_sufficient_set_bruteforce`, `::relevant_nodes`; `substrate.py::SyntheticSubstrate.minimal_sufficient_set*` | `oracle.py:178-192` brute force "Enumerates all subsets of the candidates in increasing size, so the first feasible subset is minimal"; `:194-213` pruned search "Still exact"; `:158-176` Menger pruning `nx.node_connectivity(H, v, sink) >= 2`; `tests/test_synthetic.py:143-168` asserts brute force and pruned search agree on every motif under 12 nodes; explainer size is `masks.py:319` `sparsity=float(mask[cand].sum())` (an L1 sum of a soft mask, not a set cardinality) | **PARTIAL** (was UNENFORCED) | An exact minimal-set oracle now exists — but only for the synthetic mechanism (it solves the oracle circuit, not a GNN), and the pruning's correctness rests on `relevant_nodes` (Menger criterion), which is tested against brute force only below 12 nodes. No certified lower bound exists for real substrates, no 1.10/1.12 experiment exists, and nothing prevents a future experiment from comparing `MaskResult.sparsity` (an upper bound, and a real-valued one) against the ball. `theorems.md` Attack surface for this row still applies to any non-synthetic substrate. |
| A4.5 | The ball is defined over `Substrate.candidate_nodes`, not all nodes. | `substrates/base.py::Substrate.candidate_nodes`; `masks.py::resistance_ball_init`, `::explain` | `base.py:24-27` protocol method; `masks.py:156` `others = sorted(R[v] for v in nodes if v != target)` (all non-target nodes); `:160-164` sets `inside`/`outside` over all nodes; `:361` `ball_size = max(1, (n - 1) // 2)` (counts all non-target nodes); `explain()` `:358-366` never receives or forwards `candidates` to the init; `optimize_soft_mask` `:249,:263` fixes non-candidates at 1 regardless of the init | **PARTIAL** | `resistance_ball_init` ignores candidates entirely: the ball radius (via `ball_size`) is computed over every node, and non-candidate nodes that fall outside the ball are then silently reset to 1 by the optimizer. On the synthetic substrate `candidate_nodes` is all non-target nodes (`substrate.py:165-167`), so this is invisible today; on splice (intronic windows only) the ball's size and the theorem's bound would be computed over different node sets. |
| A4.6+ | `PENDING PROPOSAL` (ε–radius relation; dependence on Theorem 3's constant) | — | — | **UNVERIFIABLE** | The test `tests/test_masks.py:107-110` defines "Theorem 4's ball" as the ball whose radius reaches the regulator (`theorem4_radius(o) = resistance_to_target(o.graph, o.target)[o.regulator]`), which uses ground truth the theorem cannot assume; it is a test convenience, not the theorem's radius. |

## UNENFORCED

- None strictly at UNENFORCED after re-derivation; the items below are PARTIAL with the enforced half confined to the synthetic path.

## UNVERIFIABLE

- Formal statement, proof sketch, A4.6+ (ε-to-radius relation, Lipschitz dependence) — `PENDING PROPOSAL`.
- Whether the "soft-masked forward pass" of A4.1 is the oracle circuit or a masked GNN.

## PARTIAL

- A4.1 (A-SOFT) — optimizer soft by construction; consumers warn rather than raise; ground-truth minimal sets use hard indicator masks.
- A4.2 (A-LATENT) — latent by construction; `latent_fn` contract unverifiable.
- A4.3 (A-SP, caveats) — `exact=False` misdocumented; disconnected graphs give finite radii.
- A4.4 — exact minimal search exists for synthetic only; no lower bound for real substrates; no consumer guards against using explainer sparsity.
- A4.5 — ball computed over all nodes, not `candidate_nodes`.

## Single weakest point

**The two sides of the inequality are currently different objects: the "minimal ε-sufficient size" is the cardinality of a hard indicator set `1_S` (`oracle.py::set_mask`, which Assumption 5 excludes from Theorems 3–5), while the explainer reports an L1 sparsity of a strictly-soft mask (`masks.py:319`), and the ball is counted over all nodes rather than candidates (A4.5).** Until the proposal fixes which object the bound is about — and the ε–radius relation (A4.6) — ABLATIONS 1.12 cannot be implemented without either comparing an upper bound to an upper bound or comparing a set size to a real number. The exact synthetic oracle is the right ground truth and is the one substantive improvement since `theorems.md`.
