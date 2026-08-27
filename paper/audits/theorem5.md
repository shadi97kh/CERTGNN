# Audit: Theorem 5 — rewiring budget and explanation size, with Corollary

Audited 2026-08-27 against the working tree at `3259c7e`. Statuses are
re-derived from the current code. Legend as in `theorem1.md`.

Formal statement, Corollary, proof sketch: `PENDING PROPOSAL` —
**UNVERIFIABLE**. There is still no rewiring module: `certgnn/topology/`
contains only `resistance.py`, no code reads `configs/rewire/*.yaml`
(`grep -rn "rewire\.mode\|rewire\b" experiments/ scripts/` matches
nothing), and no `experiments/gate4_pareto.py` exists (the `Makefile`
target points at a missing module). The only rewiring code in the
repository is the Tier-0 *control* `certgnn/eval/sanity.py::degree_preserving_rewire`,
which is an edge-swap procedure and therefore outside this theorem.

## Assumption table

| id | assumption | enforcing file:symbol | quoted lines | status | evidence |
|---|---|---|---|---|---|
| A5.1 | Rewiring only *adds* edges (Rayleigh monotonicity); edge swaps are outside the theorem. | rewire module (does not exist); configs only | `configs/rewire/add_only.yaml:1` `mode: add_only      # Rayleigh monotonicity applies (ABLATIONS 1.15)`; `configs/rewire/swap.yaml:1` `mode: swap          # outside Theorem 5's add-only assumption; reported for context`; `configs/base.yaml:10` `rewire: none`; `sanity.py:72` `nx.double_edge_swap(G, nswap=n_swaps, ...)` (removes edges; Tier-0 only); `experiments/tier0_controls.py:130-144` `_rewire` applies it | **UNENFORCED** | The `mode: swap` configuration is selectable and nothing in code distinguishes it from `add_only`; no Rayleigh assertion exists anywhere (`grep -rn Rayleigh certgnn/ tests/` hits only the `oracle.py:13-14` docstring "`kappa` is monotone in the unmasked set (Rayleigh)", which is a claim about masking, not rewiring, and is untested). Rows 1.13–1.15 cannot run. |
| A5.2 | Resistance is recomputed exactly after rewiring. | `resistance.py::resistance_to_target` (A-SP) | `resistance.py:114`, `:121-123` (see `theorem1.md` A1.1–A1.3) | **ENFORCED at the entry point, with caveats** | Nothing rewires, so nothing recomputes; the solver itself carries the A-SP caveats. One caveat is specific to rewiring: adding edges to a series-parallel graph can create a K4 minor, after which `is_series_parallel` returns False and *all* resistances silently switch to the pinv path (`:127-134`) — correct on unit weights, but the "exact O(E) solver" claim then no longer applies to the rewired graphs, and nothing records which path was taken (`theorems.md` work-queue item 5b is not done). Adding edges also cannot disconnect a graph, so A1.3 is not worsened. |
| A5.3 | Masks are soft. | `masks.py` (A-SOFT) | `masks.py:246-247`, `:263`, `:302-304`, `:330-334`; `faithfulness.py:101-108` (warn only); `sanity.py:86` (silent) | **PARTIAL** (was UNENFORCED) | See `theorem3.md` A3.1. The same caveat applies with extra force here: a rewired *GNN* forward pass with a soft mask does not exist (`gnn.py:84-91` takes no mask), so "explanation size after rewiring" can currently only be measured on the synthetic oracle, whose mechanism (`oracle.py:92-126`) is itself an effective-resistance computation — making a rewiring-vs-size experiment on the oracle close to circular. |
| A5.4 | Whether the model is held fixed or retrained after rewiring, and which the statement covers. | `PENDING PROPOSAL`; nothing | `tier0_controls.py:189-194` retrains after the control rewire (`fit(gcn, shuffled["train"], ...)`), which is the Tier-0 protocol, not T5's; no T5 code | **UNVERIFIABLE** (statement) and **UNENFORCED** (no code, not echoed in any config or results table) | The only precedent in the repo retrains, and `theorems.md`'s attack surface (b) — retraining conflates topology with training — is unaddressed. |
| A5.5 | Budget is a fraction of edges; monotonicity is in that quantity. | `PENDING PROPOSAL`; configs only | `configs/rewire/add_only.yaml:2` `budget: 0.1`; `swap.yaml:2` `budget: 0.1`; `none.yaml:2` `budget: 0.0`; no consumer | **UNVERIFIABLE** (statement) and **UNENFORCED** (no code) | A `budget` key exists with no reader, no unit, and no range check. G4's "zero to max budget" has no defined max. |
| A5.6+ | `PENDING PROPOSAL` | — | — | **UNVERIFIABLE** | — |

## UNENFORCED

- A5.1 add-only rewiring — no rewire module; `mode: swap` selectable; no Rayleigh assertion.
- A5.4 fixed vs retrained model — no code, no config echo (also UNVERIFIABLE).
- A5.5 budget definition — config key with no reader (also UNVERIFIABLE).

## UNVERIFIABLE

- Formal statement, Corollary, proof sketch, A5.4, A5.5, A5.6+ — `PENDING PROPOSAL`.

## PARTIAL

- A5.3 (A-SOFT) — as in Theorem 3; additionally, no masked GNN forward exists, so size-after-rewiring is measurable only on the oracle.
- A5.2 (A-SP, caveats) — exact path silently abandoned once rewiring creates a K4 minor; path taken is not recorded.

## Single weakest point

**Nothing implements rewiring, and the only related code (`degree_preserving_rewire`) is exactly the swap operation the theorem excludes; meanwhile `configs/rewire/swap.yaml` is a first-class selectable option that no code rejects, so the first Tier-1.13 run could be configured to violate A5.1 with no error.** The theorem is the least-implemented in the repository after Theorem 3, and unlike Theorem 3 its ablation rows (1.13–1.15) have no prerequisite beyond a rewire module with an add-only default and a Rayleigh assertion (`theorems.md` work-queue item 11).
