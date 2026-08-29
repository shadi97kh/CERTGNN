# How much data separates a fit from its reparameterized twin?

git SHA `3493d3f`, **aggregated post hoc by `52c7521` via --retable**, config `85d003d2`, 10 seeds, mean [95% bootstrap CI]. 4000 real BRCA2 5' splice sites per seed, 3000 fit / 1000 held out.

**The claim, per cell.** *X*: how well the twin's latent agrees with the monotone reparameterization out of sample. *Y*: Spearman between the two fits' per-locus attribution magnitudes, which is what a reader ranking positions would see. *Z*: held-out measurements needed to reject that the two fits are the same function, at two-sided α = 0.05 with power 0.8.

| width | depth | params | **X** held-out R² | **Y** attribution ρ | **Z** n to separate | RMS pred. diff |
|---|---|---|---|---|---|---|
| 16 | 1 | 609 | **0.984087** | **+0.893** [+0.850, +0.932] | **1** [0, 1] | 0.5841 |
| 16 | 2 | 881 | **0.999241** | **+0.982** [+0.967, +0.993] | **42** [25, 60] | 0.0878 |
| 16 | 3 | 1,153 | **0.999583** | **+0.987** [+0.968, +0.998] | **313** [87, 571] | 0.0654 |
| 32 | 1 | 1,217 | **0.992155** | **+0.695** [+0.552, +0.815] | **0** [0, 1] | 0.5892 |
| 32 | 2 | 2,273 | **0.997328** | **+0.905** [+0.860, +0.947] | **87** [14, 193] | 0.1681 |
| 32 | 3 | 3,329 | **0.999166** | **+0.968** [+0.952, +0.985] | **833** [18, 2,136] | 0.1109 |
| 64 | 1 | 2,433 | **0.998324** | **+0.693** [+0.598, +0.793] | **2** [1, 3] | 0.3657 |
| 64 | 2 | 6,593 | **0.999997** | **+0.873** [+0.822, +0.923] | **4,176** [1,320, 7,333] | 0.0083 |
| 64 | 3 | 10,753 | **0.999996** | **+0.928** [+0.850, +0.987] | **7,300** [2,156, 14,436] | 0.0069 |
| 128 | 1 | 4,865 | **0.999876** | **+0.722** [+0.657, +0.775] | **9** [7, 11] | 0.1364 |
| 128 | 2 | 21,377 | **1.000000** | **+0.807** [+0.735, +0.875] | **3,158,529** [1,402,633, 5,004,083] | 0.0010 |
| 128 | 3 | 37,889 | **1.000000** | **+0.920** [+0.887, +0.952] | **87,751,464** [39,140,215, 138,392,840] | 0.0001 |

Noise is not assumed. The phenotype is affine in log₁₀ of the count ratio ex_ct/tot_ct (slope 0.933 on this library). Those are two independent count pools rather than a proportion — ex_ct exceeds tot_ct in 4.9% of rows — so both are treated as Poisson and the delta method applied to the log ratio, giving `sd = |a|·sqrt(1/ex + 1/tot)/ln10`: median 0.3114 in the standardized units the models see. This counts sequencing noise only; library preparation and biological variation add more. **Z is therefore a lower bound** — at least this many measurements, likely more.

## Verdict

**The identifiability claim, stated quantitatively.** The twin is not identical to the original and it is not unrelated to it, so neither the strong claim nor the word 'collapse' describes the measurement. At the closest cell (128x3) the twin agrees to held-out R² 1.000000, its predictions differ from the original's by RMS 0.0001 in standardized phenotype units, and separating the two as functions takes at least 87,751,464 held-out measurements [39,140,215, 138,392,840] at α = 0.05 with power 0.8. Across the grid the requirement ranges from 0 measurements (32x1) to 87,751,464 (128x3). 10 of 12 cells are separable within this library's 30,483 sequences; the rest would need a larger experiment. **The attribution consequence is what makes this practical rather than philosophical.** Two fits agreeing this closely still rank the loci differently: Spearman +0.693 at the worst cell (64x1) and +0.920 at 128x3. A reader who ranks positions by attribution magnitude is reading a quantity that the data does not pin down at the sample sizes above, which is the failure mode the certificates in this project are meant to prevent. This triple holds regardless of how the containment question in `paper/tables/closure_search.md` resolves. It describes the two fits actually obtained and the data actually needed to tell them apart, not whether some member of the class equals ψ∘φ̂ exactly.
