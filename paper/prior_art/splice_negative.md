# The splice-graph negative result

This is a result. It is recorded here in full because it determines whether
the topology framing survives, and because the experiment that produced it
had never been run on anything real before.

Everything below comes from three 10-seed runs on the MFASS exon-recognition
assay (Cheung et al., *Molecular Cell* 73:183-194, 2019), whose provenance,
sizes and SHA-256 sums are in `data/raw/PROVENANCE.md`. Tables:
`paper/tables/tier0_controls.md`, `paper/tables/gate2_splice.md`.

---

## 1. The substrate works and is faithful to Definition 1

Nothing here failed for want of an implementation.

- `certgnn/substrates/splice/` satisfies the `Substrate` protocol in
  `certgnn/substrates/base.py` without modifying it. The protocol fits
  better on real data than it did synthetically: `baseline_rate` returns the
  natural exon's own splicing index, a genuine rate, and
  `ground_truth_window` returns the window carrying the assayed variant,
  resolved for 12,360 of 18,571 training graphs.
- Graphs follow Definition 1: the assayed fragment is tiled into windows;
  exon and intronic windows are nodes; edges are the backbone plus the
  skipping junction from the last upstream intronic window to the first
  downstream one. That junction is what makes the object a graph rather than
  a path, and its endpoints move with the intron and exon lengths:
  **15 distinct chord placements** across the training split, over 2,339
  exons.
- Splits are **grouped by `ensembl_id`**. Variants of one exon share nearly
  all of their sequence, so a random split would leak exon identity.
- The intronic radius is **81 nt effective against 300 nt requested**. MFASS
  assays a 170 nt fragment whose flanking introns have a median length of
  45 nt, so Definition 1's radius exceeds the sequence that was measured.
  Windows are tiled only within the construct; drawing them from hg38 beyond
  the minigene would build a graph over sequence the assay never
  interrogated.

## 2. What the runs returned

### Tier 0, rows 0.6 and 0.7, 10 seeds

| model | row | test R² |
|---|---|---|
| GNN | reference | **-0.028** [-0.042, -0.017] |
| shuffled | 0.6 degree-preserving shuffle | -0.008 [-0.011, -0.005] |
| edge-free MLP, target readout | 0.7 | -0.017 [-0.021, -0.013] |
| edge-free MLP, mean pool | 0.7b | -0.006 [-0.014, -0.001] |

**Every arm is at or below zero on held-out exons.** The reference model is
worse than predicting the mean, and the paired differences run the wrong way
(GNN minus mean-pool MLP = -0.022, i.e. the edge-free model is the less bad
of the two).

### Capacity is not the constraint; generalization is

Same architecture, 3,000 training graphs, monitored on train and test:

| epoch | train R² | test R² |
|---|---|---|
| 5 | +0.230 | -0.060 |
| 30 | +0.338 | -0.090 |
| 60 | **+0.545** | **-0.251** |

The model fits the training exons and transfers nothing. This is
overfitting, not underfitting.

### Why the task is hard, measured rather than asserted

- **78.5% of MFASS variance is within-exon**, i.e. driven by the single
  nucleotide changed, not by which exon it is. Only 21.5% is between-exon.
- Using the natural exon's index as the prediction gives **R² = 0.042**.
- A variant's effect is real and often large: `v2_dpsi` has sd 0.231 and
  6.4% of variants move the index by more than 0.5.

So the task an exon-grouped split poses is: predict the splicing effect of
one nucleotide in an exon the model has never seen. That is the SpliceAI
problem.

### Gate 2 on real data, 10 seeds

`experiments/gate2_splice.py` removes the circularity in Gate 2 -- trained
GNN, soft mask threaded through every message-passing layer, real measured
baseline rate, no oracle. It returns **uninformative**, for the same reason:

| quantity | value |
|---|---|
| test R² (latent scale) | -0.015 [-0.021, -0.009] |
| r(model latent, logit baseline) | 0.004 [-0.024, 0.037] |
| clamped at the logit boundary (train) | 0.258 |

| arm | probability gap | latent gap |
|---|---|---|
| trained model | 0.070 [0.055, 0.085] | 0.075 [0.057, 0.094] |
| model randomization | 0.062 | 0.062 |
| label randomization | 0.087 | 0.080 |

The trained model's gap is indistinguishable from its own randomization
controls, and label randomization produces the largest gap of the three.
A model that cannot represent the baseline (r = 0.004) cannot exhibit
baseline-dependent coverage, so a flat profile is the absence of information
rather than evidence about the link. Separately, **25.8% of training labels
are clamped at the logit boundary**: MFASS indices pile up at 0 and 1, which
is an obstacle to any latent-space treatment of this assay independent of
the model.

---

## 3. What this establishes, and what it does not

**It does NOT establish that splice graphs lack topological structure.**
No result here bears on that question. The comparison rows 0.6 and 0.7 make
is between a reference model and its controls; with the reference at
R² = -0.028 there is nothing to compare against, and "the MLP matches the
GNN" is a statement about two models that both predict nothing.

**It establishes** that on MFASS, with an exon-grouped split and the
reference architecture, no arm predicts well enough for rows 0.6 and 0.7 to
mean anything. They are **unanswerable here, not answered**. The same holds
for Gate 2 on this substrate.

**The grouped split is what exposed this.** A random split would have put
variants of the same exon on both sides, leaked exon identity, and produced
a flattering R² from which rows 0.6 and 0.7 would have returned a confident
and meaningless verdict. The theorems audit flagged grouped splits as
load-bearing (`paper/theorems.md`, assumption A22.2); this is what that was
load-bearing for.

**An earlier version of this analysis got it wrong.** The automated
interpretation in `experiments/tier0_controls.py` originally reported "the
substrate is NOT a graph problem and cannot carry a topology paper" and
"the topology is decorative" from these numbers, because it applied the
pre-stated rules without checking that the reference model had learned
anything. That verdict was an overclaim drawn from noise. The function is
now gated on the reference model beating the mean, and `gate2_splice` had
that gate from the start.

---

## 4. What would be needed to answer rows 0.6 and 0.7

A model that generalizes across exons. That is the SpliceAI problem and is
out of scope for this work. The specific alternatives, with an assessment of
each:

1. **A pretrained splicing model as a feature extractor.** Replace the
   window one-hot features with per-window embeddings from SpliceAI,
   Pangolin, or the ConvSplice weights already in `data/raw/resources/`
   (obtained, checksummed, unused). The GNN would then reason over
   representations that already generalize, and rows 0.6 and 0.7 would test
   whether topology adds anything on top of them. **Worth pursuing.** It is
   the only option that keeps the original question intact, the weights are
   already on disk, and it converts an unanswerable comparison into an
   answerable one. The cost is that the finding becomes conditional on the
   extractor.

2. **A within-exon prediction task.** Predict `v2_dpsi` for held-out
   variants of exons seen in training. This is learnable, since 78.5% of the
   variance is within-exon. **Not worth pursuing for these rows.** Within one
   exon the topology is constant, so it reintroduces exactly the defect that
   ruled out the FAS and BRCA2 libraries: no topological variation across
   instances, hence no test of topology. It would be a fine substrate for
   Theorem 2 and the explanation work, and a null one for Theorem 1.

3. **A substrate with more between-exon signal.** Vex-seq or MaPSy, or
   MFASS restricted to exons whose natural indices span a wide range.
   **Weakly worth pursuing.** It raises the 21.5% between-exon share that a
   grouped split must predict, but it does not make single-nucleotide effects
   learnable, and the ceiling stays low. It is a mitigation, not a fix.

4. **Abandon the topology framing for splice.** Legitimate given the
   evidence, but premature: nothing here shows the topology is uninformative,
   only that this pipeline cannot test it.

**Recommendation: option 1.** If a pretrained extractor still yields a
reference model at R² near zero on an exon-grouped split, then the topology
question is genuinely unanswerable on MFASS and option 4 follows on
evidence rather than on a null.
