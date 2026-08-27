# Audit: Theorem 1 — effective resistance and model sensitivity

Audited 2026-08-27 against the working tree at `3259c7e` (plus untracked
sweep outputs). Statuses are re-derived from the current code, not copied
from `paper/theorems.md`. Legend: ENFORCED = a code path raises or makes the
violation impossible to express; PARTIAL = enforced on one path but
bypassable, silently corrected, or documented only; UNENFORCED = nothing in
`certgnn/` checks it; UNVERIFIABLE = the assumption text is `PENDING
PROPOSAL`, so there is nothing to check against.

Formal statement, clauses (i) and (iv)+, proof sketch: `PENDING PROPOSAL`
— **UNVERIFIABLE**. Only the informal claim `[PREREG C1]` and the test-named
clauses (ii), (iii) can be audited.

## Assumption table

| id | assumption | enforcing file:symbol | quoted lines | status | evidence |
|---|---|---|---|---|---|
| A1.1 | Effective resistance is computed exactly; the series-parallel solver is validated against the pseudo-inverse. | `certgnn/topology/resistance.py::resistance_to_target`, `::is_series_parallel`, `::resistance_series_parallel` (A-SP) | `resistance.py:114` `if exact and is_series_parallel(G):`; `:99-103` `if H.number_of_nodes() != 2 or not H.has_edge(source, target): raise ValueError("Graph did not reduce to a single edge; ...")`; `:121-123` `except ValueError: exact = False; break` | **ENFORCED at the entry point, with caveats** | The SP result is never returned without the K4-minor check at line 114, and the reducer raises rather than returning a wrong value. Caveats: (1) `:121-123` falls back to `resistance_via_pinv` *silently*, contradicting the reducer's own docstring `:54-56` ("Do not silently fall back"); (2) `certgnn/explain/masks.py:150` `R = resistance_to_target(G, target, exact=exact)` lets callers skip the exact path with `exact=False`, and `experiments/gate2_link.py:276` does so; this is only equivalent on unit-resistance graphs (see A1.2). Validation against pinv exists only in `tests/test_resistance.py` on unit resistances. |
| A1.2 | Edge resistances: unit unless an edge carries a `resistance` attribute; both solver paths must agree on the convention. | `resistance.py::resistance_via_pinv` vs `::resistance_series_parallel` | `:22-24` `A = torch.zeros(...); A[edge_index[0], edge_index[1]] = 1.0; A = torch.maximum(A, A.T)` (0/1 adjacency); `:60` `H.add_edge(u, v, r=float(data.get("resistance", 1.0)))`; `:129-131` the fallback builds `ei` from `G.edges()` and drops attributes | **PARTIAL** | Verified this session on a triangle with `resistance=3.0` on every edge: `resistance_to_target(Gw, 0, exact=True)` = `{1: 2.0, 2: 2.0}` (correct, 3 ‖ 6), `exact=False` = `{1: 0.667, 2: 0.667}` (unit-weight answer). The `masks.py:142-143` docstring claim that `exact=False` gives "same values" is false on weighted graphs. The synthetic substrate never carries weights (`substrates/synthetic/substrate.py:175-182` `to_networkx` adds bare edges; `oracle.py:38` "Unit-resistance graph"), so the discrepancy is unexercised; it will surface on connectome graphs. |
| A1.3 | Target reachable from every candidate node (connected graph, or resistance defined only within the target's component). | nothing in `resistance.py`; `substrates/synthetic/oracle.py:74-75` guards the regulator only | `resistance.py:112` docstring "Tries exact reduction, falls back to pinv" — no connectivity check; `oracle.py:74-75` `if self._c_full <= 0: raise ValueError("regulator is not connected to target")` | **UNENFORCED** | Verified this session on `G = {(0,1),(2,3)}`: `is_series_parallel(G)` returns **True** (both components reduce to nothing, `H.number_of_nodes() <= 2` at `:48`), `resistance_series_parallel(G, 2, 0)` raises, the silent fallback returns `{1: 1.0, 2: 0.5, 3: 0.5}` — finite, meaningless cross-component values with no error. (`theorems.md` said disconnected graphs "always take the pinv path"; the precise behaviour is: they may pass the SP check and reach pinv through the fallback.) The oracle's guard covers the regulator-target pair only, and only on the synthetic path. Synthetic motifs are connected by construction (`topology.py::_Builder` attaches every node to an existing one). |
| A1.4 | The model is a message-passing GNN (locality); no claim for full-attention or edge-free models. | `certgnn/models/gnn.py::TargetReadoutGCN` | `gnn.py:62-76` `GCNConv` / `SAGEConv(..., aggr="add")` / `GINConv`; `gnn.py:93-94` `for conv in self.convs: h = torch.relu(conv(h, edge_index))`; but the same package exports `gnn.py:98` `class NodeFeatureMLP` ("no message passing") and `bqn.py:66` `class BrainQuadraticNetwork` (Hadamard products on the dense matrix) | **UNENFORCED** | Message-passing models now exist (change in evidence since `theorems.md`), and the default config selects one (`configs/base.yaml:8` `model: sage_sum`). But no code that would *use* A1.4 exists (no sensitivity functional, no `experiments/gate1_resistance.py` — `Makefile` targets it, `experiments/registry.py` registers only `gate2_link`), and the `LatentFn` contract (`masks.py:29`) and the `Substrate` protocol accept any callable/module, so an MLP or BQN can be substituted anywhere without a check. |
| A1.5 | "Sensitivity" is a stated functional of the target logit's Jacobian w.r.t. node features, measured in latent space. | `PENDING PROPOSAL` for the functional; no sensitivity code | `grep -rn "sensitivity\|jacobian\|autograd" certgnn/` matches only `link.py::sigmoid_jacobian` and the synthetic `links.py` Jacobians (link derivatives, not model sensitivities) | **UNVERIFIABLE** (functional) and **UNENFORCED** (no code) | The functional is pending; even its latent-space part cannot be checked because nothing computes a model Jacobian. A-LATENT is satisfied on the head side (`gnn.py:81` `self.head = nn.Linear(hidden, 1)`; `:95` returns `self.head(...)` with no sigmoid; `train.py:89` `BCEWithLogitsLoss`) so a future sensitivity functional applied to `model(...)` would be in latent space by construction. |
| A1.6 | Message-passing depth covers the target's resistance ball. | nothing; config comment only | `configs/model/gcn.yaml:3`, `gin.yaml:3`, `sage_sum.yaml:3` `depth: 5            # must span regulator -> target (theorems audit A1.6)` | **UNENFORCED** | A yaml comment is documentation. Nothing compares `model.depth` to the graph distance to the regulator or to the resistance-ball radius, and nothing records the ball radius in the run directory. On the default synthetic topology (`nested_bubbles`, `depth: 2`) the target-regulator distance is 4 edges, so depth 5 happens to suffice; a `TopologySpec(depth=3)` override (distance 8) would silently violate it. |
| A1.7+ | `PENDING PROPOSAL` | — | — | **UNVERIFIABLE** | No text to audit. |

## UNENFORCED

- A1.3 target reachability — `resistance_to_target` returns finite values across components (verified).
- A1.4 message-passing model — models exist, nothing restricts the consumer to them; no consumer exists.
- A1.5 sensitivity functional — no code computes any model sensitivity; Gate 1 experiment does not exist.
- A1.6 depth covers the ball — config comment only.

## UNVERIFIABLE

- Formal statement, clause (i), clauses beyond (iii), proof sketch — `PENDING PROPOSAL`.
- A1.5 (the functional itself), A1.7+.

## PARTIAL

- A1.2 weighted-graph convention — the two solvers disagree on weighted graphs (verified: 2.0 vs 0.667); `exact=False` bypass documented as "same values".

## Single weakest point

**Nothing in the repository computes "sensitivity" (A1.5), so the second clause of C1 — the one Gate G1 tests — has no implementation to audit, and `experiments/gate1_resistance.py` referenced by the `Makefile` does not exist.** The resistance half of the theorem is in better shape but has one silent bug a reviewer can hit immediately: on a disconnected graph the code passes the series-parallel check, the reducer raises, and the silent pinv fallback returns finite cross-component resistances (A1.3), which would enter a Spearman correlation as if they were real.
