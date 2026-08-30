# Prior art: reliability normalization of variant-effect benchmarks

**Verdict: (b) IMPLICIT BUT NEVER USED TO RENORMALIZE A BENCHMARK.**

ProteinGym does **not** normalize by assay reliability. It acknowledges the
problem in prose, uses replicate noise as an assay *selection* criterion, and
then reports raw correlations and averages them with weights chosen on a
different basis entirely. So the specific claim survives the decisive check.

But the statistical content is old, and that has to be stated up front: the
correction itself is **Spearman's 1904 attenuation formula**, and the
noise-ceiling form of it has been standard in neuroscience for twenty years.
Nothing here would be a new theorem. What appears unclaimed is the *application*
to a variant-effect leaderboard, and the consequence — that rankings are
reliability-weighted by an accident of assay curation.

---

## 1. ProteinGym — the decisive check

Notin et al., *ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction
and Design*, NeurIPS 2023 Datasets & Benchmarks.
[PMC10723403](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/) ·
[repo](https://github.com/OATML-Markslab/ProteinGym)

**Does it estimate or report per-assay reliability?** No. It acknowledges the
issue without quantifying it:

> "noise is a perennial issue in high-throughput assays, and some assays have
> poor experimental replicate correlation"

> "Experiments do not have a perfect dynamic range, often imposing a restrictive
> ceiling and/or floor"

**Does it normalize scores by a ceiling?** No. Spearman, AUC, MCC, NDCG and
top-K recall are reported raw.

**How does it aggregate?** By functional category, with each category weighted
equally — not by reliability:

> "we first compute each of these metrics within groups of assays that measure
> similar functions. The final value of the metric is then the average of these
> averages, giving each functional group equal weight"

**Where replicate noise does enter:** as an assay *selection* criterion. The
selection list is "public availability of data", "experimental throughput",
"level of noise between experiment replicates", "dynamic range", and "assay
type". Noise is used to decide *which assays are in the benchmark*, and then
discarded. That is precisely the gap the claim identifies: a threshold on
reliability, followed by an unweighted average over the survivors, is still a
reliability-weighted average of true model quality — the weights are just
implicit and unexamined.

**Repo documentation:** the README (26 KB) has **zero** occurrences of noise,
replicate, reliability, ceiling, explainable variance, or signal-to-noise. The
only "weight" hits are MSA sequence weights, unrelated.

## 2. Per-variant error models — real, but never propagated to a ceiling

Enrich2 (Rubin et al., *Genome Biol* 2017), DiMSum (Faure et al., *Genome Biol*
2020), Rosace (Rao et al., *Genome Biol* 2023) and Lilace (Freudenberg et al.,
*Genome Biol* 2026) all model count noise and emit **per-variant** standard
errors. Çubuk et al. (*Mol Syst Biol* 2025) review twelve such tools.

None of them propagates that error into an **achievable-performance bound for a
benchmark**. They produce error bars for hypothesis testing and variant
filtering, which is a different object from `1 - mean(sigma^2)/Var(y)`. This is
the distinction the claim rests on and it holds.

MaveDB (Rubin et al., *Genome Biol* 2025) and the MIMAVE minimum-information
standard (Claussnitzer et al., *Genome Biol* 2024) define reporting conventions
for MAVE data; neither defines a per-assay reliability statistic for
benchmarking use.

One near-miss worth recording. Reeb et al. (*BMC Bioinformatics* 2019) observed
that "for the few proteins with multiple independent experimental measurements,
experiments differed substantially, but agreed more with each other than with
predictions." That is the ceiling idea in observational form — assay-assay
agreement bounding model-assay agreement — but it is an aside, not a
normalization, and it covers a handful of proteins.

## 3. The neuroscience method exists and is twenty years old

- Hsu, Borst & Theunissen, "Quantifying variability in neural responses and its
  application for the validation of model predictions", *Network* 15:91 (2004) —
  introduces normalizing raw correlation by `CC_max`, the ceiling implied by
  inter-trial variability, giving `CC_norm`.
- Schoppe, Harper, Willmore, King & Schnupp, "Measuring the performance of neural
  models", *Front Comput Neurosci* 10:10 (2016).
  [PMC4748266](https://ncbi.nlm.nih.gov/pmc/articles/PMC4748266)

In neuroscience this is routine: explainable variance from repeated trials, raw
correlation divided by the ceiling, and models compared in normalized units. The
search found the technique essentially confined to neuroscience and brain
encoding; no transfer to a biological ML benchmark surfaced.

**And the underlying statistics are older still.** Correcting a correlation for
attenuation due to measurement error is Spearman (1904); the psychometric
formula `r_true = r_obs / sqrt(rel_x · rel_y)` is textbook. The proposed
`corr(y,h) <= sqrt(1 - mean(sigma^2)/Var(y))` is the one-sided version of it with
reliability computed from counts. **This is not a novel bound and must not be
presented as one.**

## 4. Ranking bias from differing task noise — claimed in ML, not in biology

This is the closest live competitor and it is recent. The ML-evaluation
literature has begun addressing exactly the aggregation problem:

- "Signal and Noise in LLM Evaluation" — computes per-subtask signal-to-noise
  ratio and curates high-SNR subsets to improve aggregate decision accuracy.
- "AI Cartography: Mapping the Latent Landscape of AI Benchmark Ecosystems"
  ([arXiv 2605.25272](https://arxiv.org/pdf/2605.25272)) — finds that after
  conditioning on noise, "only one model in the top 1% of the original overall
  leaderboard remains in the top 1%", while ~90% stay within the top decile.
  That is the reordering claim, demonstrated, for LLM benchmarks.
- "Rank Intervals for Leaderboards: A Hierarchical Framework for Model
  Evaluation" ([arXiv 2606.08679](https://arxiv.org/pdf/2606.08679)).

So the *phenomenon* — heterogeneous task noise reorders leaderboards, with top
ranks most affected — is established in the LLM-benchmark setting. It has not
been applied to DMS or variant-effect prediction, where the noise model is far
better founded (counts, not human labels) and therefore where the correction is
actually computable rather than estimated.

## 5. What this would and would not be

**Would be:** an audit. Take ProteinGym, compute a per-assay ceiling from
replicate or count data, renormalize, and report whether the ranking changes.
The finding is a fact about a widely used benchmark, and its value is entirely
empirical.

**Would not be:** a theorem. The bound is Spearman 1904, the ceiling method is
Hsu 2004, and the leaderboard-reordering phenomenon is already demonstrated
elsewhere. Presenting any of these as new would repeat this project's existing
failure mode.

**Two things to check before any implementation, both of which can kill it:**

1. **Is sigma computable for enough of ProteinGym?** The paper says replicate
   quality was used "where available", which implies it is not available for
   all assays. If replicate-level or count-level data exists for only a minority
   of the 217-plus assays, no leaderboard-wide renormalization is possible and
   the audit cannot be run as specified. This is a data-availability question
   answerable from the ProteinGym release without writing any modelling code,
   and it should be answered first.
2. **Does the ranking actually change?** If ceiling-normalization leaves the
   ordering intact, there is no result. Note the AI Cartography finding cuts both
   ways: it reports top ranks reordering sharply while the broad ordering is
   "comparatively stable". A null here is publishable only as a short negative.

## Verification notes

- The ProteinGym quotations were retrieved from the PMC full text; the README
  keyword counts were run against the raw file from the repository.
- proteingym.org itself returned no usable content to the fetcher, so the
  website's leaderboard documentation was **not** inspected. If it documents a
  normalization absent from the paper and README, this verdict would need
  revisiting — that is the one gap in this check.
- MaveDB's exact schema for per-variant standard error was not confirmed from
  primary documentation; the fetch returned empty and the conclusion above rests
  on the database and MIMAVE papers' descriptions.
