# Tier 0 controls (ABLATIONS 0.6, 0.7, 0.8)

git e3bc94a, config 48b6bc98, 10 seed(s), mean [95% bootstrap CI]. Test-split performance; model selection on validation only.

## splice — candidate

task: regression, metric: r2, seeds: 10

| model | row | test metric | paired GNN − model |
|---|---|---|---|
| gnn | reference | -0.028 [-0.042, -0.017] (n=10) | — |
| shuffled | 0.6 degree-preserving shuffle | -0.008 [-0.011, -0.005] (n=10) | -0.021 [-0.035, -0.009] (n=10) |
| mlp | 0.7 edge-free MLP (target readout) | -0.017 [-0.021, -0.013] (n=10) | -0.011 [-0.026, 0.002] (n=10) |
| mlp_mean | 0.7b edge-free MLP (mean pool) | -0.006 [-0.014, -0.001] (n=10) | -0.022 [-0.039, -0.008] (n=10) |
| bqn | 0.8 Hadamard/BQN | not run (brain-substrate row) | — |

- UNINFORMATIVE: the reference GNN scores -0.028 [-0.042, -0.017], at or below the threshold of 0.02, i.e. no better than predicting the mean. Rows 0.6 and 0.7 compare the GNN against controls, so with no reference performance to compare against they are UNANSWERABLE rather than answered. This is not evidence that the topology is decorative or that the substrate is not a graph problem; it is the absence of evidence either way.

**Tier 0: UNINFORMATIVE**

## Verdict: SpliceCert vs ConnectomeCert

- splice: UNINFORMATIVE — UNINFORMATIVE: the reference GNN scores -0.028 [-0.042, -0.017], at or below the threshold of 0.02, i.e. no better than predicting the mean. Rows 0.6 and 0.7 compare the GNN against controls, so with no reference performance to compare against they are UNANSWERABLE rather than answered. This is not evidence that the topology is decorative or that the substrate is not a graph problem; it is the absence of evidence either way.

**UNDECIDED: every evaluated substrate is UNINFORMATIVE. The reference model does not beat predicting the mean, so rows 0.6 and 0.7 cannot discriminate and no topology verdict follows in either direction. A model that generalises across exons is needed before these rows can be answered.**
