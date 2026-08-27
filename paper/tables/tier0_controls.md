# Tier 0 controls (ABLATIONS 0.6, 0.7, 0.8)

git 3259c7e, config sweep:tier0_20260827T152529Z_3259c7e, 10 seed(s), mean [95% bootstrap CI]. Test-split performance; model selection on validation only.

## splice — candidate

**NOT EVALUATED.** splice: certgnn/substrates/splice/ defines no SpliceSubstrate (package is an empty stub) and data/raw holds 0 files. The substrate has not been implemented; nothing was run for it.

## connectome — candidate

**NOT EVALUATED.** connectome: certgnn/substrates/connectome/ defines no ConnectomeSubstrate (package is an empty stub) and data/raw holds 0 files. The substrate has not been implemented; nothing was run for it.

## synthetic — pipeline validation (not a candidate)

task: regression, metric: r2, seeds: 10

| model | row | test metric | paired GNN − model |
|---|---|---|---|
| gnn | reference | 0.763 [0.755, 0.771] (n=10) | — |
| shuffled | 0.6 degree-preserving shuffle | 0.016 [-0.001, 0.035] (n=10) | 0.747 [0.730, 0.765] (n=10) |
| mlp | 0.7 edge-free MLP | -0.031 [-0.034, -0.027] (n=10) | 0.794 [0.784, 0.803] (n=10) |
| bqn | 0.8 Hadamard/BQN | not run (brain-substrate row) | — |

- GNN beats the edge-free MLP by +0.794 [+0.784, +0.803]: the edges carry information the model uses.
- Real topology beats degree-preserving shuffle by +0.747 [+0.730, +0.765]: the specific wiring matters, not just the degree sequence.

**Tier 0: PASS**

## Verdict: SpliceCert vs ConnectomeCert

- splice: NOT EVALUATED — splice: certgnn/substrates/splice/ defines no SpliceSubstrate (package is an empty stub) and data/raw holds 0 files. The substrate has not been implemented; nothing was run for it.
- connectome: NOT EVALUATED — connectome: certgnn/substrates/connectome/ defines no ConnectomeSubstrate (package is an empty stub) and data/raw holds 0 files. The substrate has not been implemented; nothing was run for it.

**UNDECIDED: no candidate substrate could be evaluated. The SpliceCert vs ConnectomeCert question is blocked on implementing the substrates and obtaining their data, not on results.**
