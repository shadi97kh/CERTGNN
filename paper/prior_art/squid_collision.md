# Prior-art collision: Theorem 2 vs. SQUID and MAVE-NN

Adversarial reading, written on the assumption that the reviewer has read SQUID
and will ask "what is left after Seitz et al.?" Every quotation below is
verbatim. Where our claims are weaker than we have been writing them, that is
stated plainly rather than softened.

## Sources actually read (versions matter)

| source | version read | how |
|---|---|---|
| SQUID | bioRxiv preprint v1, doi 10.1101/2023.11.14.567120, posted 16 November 2023, CC-BY-NC-ND, full PDF text (18 pp. main + Methods) | `curl` of `.../567120v1.full.pdf`, `pdftotext` |
| SQUID (published) | Seitz, McCandlish, Kinney & Koo, *Nat. Mach. Intell.* 6:701–713 (2024); PMC11823438 is **not** open access in Europe PMC (`isOpenAccess: N`, `fullTextXML` empty) | PMC landing page fetched once for a cross-check of the model list and the absence of theorems; **all quotes below are from the v1 preprint** |
| MAVE-NN | Tareen et al., *Genome Biol.* 23:98 (2022), PMC9011994, Europe PMC `fullTextXML`, CC-BY | `curl`, tag-stripped locally |
| Kinney & Atwal 2014 | arXiv:1212.3647v3, "Parametric inference in the large data limit using maximally informative models" (= *Neural Computation* 26(4):637–653) | `curl` PDF, `pdftotext` |
| Atwal & Kinney 2016 | **not read.** The arXiv id in our source list, arXiv:1506.04236, is a different paper entirely: its title is "On the space of connections having non-trivial twisted harmonic spinors". arXiv's API returned no record for the J. Stat. Phys. title. Cite it by journal reference (*J. Stat. Phys.* 162:1203–1243, 2016) only; do not cite an arXiv id we have not verified. |
| SQUID code | `github.com/evanseitz/squid-nn` @ `08e3da0e4fe00de15d48b959bbe16b6199bee0e6` (11 Aug 2025), cloned to the scratchpad | `git clone --depth 1` |

Our side, read first: `PREREGISTRATION.md` (C2, gate G2), `paper/theorems.md`
§"Theorem 2", `certgnn/certify/link.py`, `ABLATIONS.md` rows 1.6–1.9,
`paper/tables/gate2_link.md`.

---

## 1. What SQUID and MAVE-NN already establish, claim by claim

Our Theorem 2, as it currently exists in the repo, is four clauses. Note before
starting: the formal statement, the corollary, and the proof sketch are all
`PENDING PROPOSAL` in `paper/theorems.md`. What we are defending against SQUID
right now is an informal claim plus four ablation rows.

### Claim (i) — scores on a bounded output inherit the Jacobian p0(1−p0), and are baseline-dependent to first order

Our version (`certgnn/certify/link.py::predicted_observed_effect`): the observed
effect of a latent shift δ at baseline p0 is `|δ| * p0 * (1 - p0)`.

**Anticipated in kind, not in form.** SQUID's entire premise is that a monotone
output nonlinearity confounds additive attribution. Abstract:

> "Importantly, SQUID removes the confounding effects that nonlinearities and
> heteroscedastic noise in functional genomics data can have on model
> interpretation."

Introduction:

> "the most widely-used attribution methods in genomics – including Saliency
> Maps, DeepLIFT, in silico mutagenesis (ISM), SmoothGrad, Integrated
> Gradients, and DeepSHAP – assume that nucleotide effects on DNN predictions
> are locally additive. As a result, these attribution methods do not account
> for the genetic interactions (i.e., specific epistasis), global
> nonlinearities (i.e., global epistasis), and heteroscedastic noise that are
> often present in functional genomics data."

Results, "Saliency Maps" subsection (Fig. 2c–e) — this is the saturation
statement, i.e. the qualitative content of our p0(1−p0) factor at the top of the
range:

> "Plotting the effects that mutations in a representative genomic sequence have
> on DNN predictions, we found that virtually all combinations of 2 or more
> mutations to the core 7-nt AP-1 site reduced DNN predictions to
> near-background levels (Fig. 2c). Moreover, the GE nonlinearity learned by
> SQUID as part of the surrogate model accurately recapitulated this saturation
> effect."

And the cleanest statement of the failure mode our clause (i) describes, from
the epistasis analysis (Fig. 6e):

> "The reason is that, in the linear pairwise model, the pairwise-interaction
> parameters are co-opted to describe a global nonlinearity instead of the
> nucleotide-specific interactions they are intended to model."

**What is genuinely not there.** SQUID never writes a Jacobian, never states a
first-order expansion, and never names an operating point. Its nonlinearity is
*learned*, not known: main text, "a genotype-phenotype (G-P) map, a global
epistasis (GE) nonlinearity, and a noise model":

> "The GE nonlinearity, which is modeled using a linear combination of sigmoids,
> maps the latent phenotype to a most-probable DNN prediction."

Methods, "Surrogate models" (note the main text says sigmoids and the Methods
say hyperbolic tangents — the family is not the point for them):

> "The GE nonlinearity, g(φ), maps the latent phenotype φ to a most-probable
> scalar DNN prediction ŷ. By default, g(φ) is defined to be an
> over-parameterized linear combination of hyperbolic tangents. For models
> 'without' a GE nonlinearity, g(φ) is defined to be a linear function of φ."

**Verdict on (i).** The *phenomenon* is prior art and is the stated motivation of
a Nature MI paper. Our added content is a closed-form factor for a link that is
*known* rather than fitted. As mathematics this is the chain rule; it is not a
result. `gate2_link.md` row 1.6 reports R² = 0.998 [0.998, 0.998] of the measured
ratio against p(1−p) for the logit link — that is a unit test of calculus, not
evidence for a theorem, and it should be presented as such. Do not write "we
show that" for clause (i).

### Claim (ii) — not comparable across instances with different baselines; explicit rank reversals

Our version: `ABLATIONS.md` 1.8, "Probability-space attribution scores reverse
the ranking of node effects across instances with different baselines, while
latent-space scores preserve it"; `gate2_link.md` reports probability-space
reversal rates 0.523 / 0.728 / 0.932 for baseline pairs 0.5-vs-0.9 / 0.95 / 0.99
against 0.000 in latent space.

**This is the clause SQUID hurts most, and we have been overclaiming it.**

First, SQUID does exactly the cross-instance comparison we say is invalid — it
compares attribution maps across genomic loci as its headline benchmark
(Abstract: "identifies motifs that are more consistent across genomic loci"),
Results, "SQUID improves motif consistency":

> "Finally, we calculated the Euclidean distance between the vector of
> attribution scores for individual sequences and the mean vector of attribution
> scores. We refer to this distance as the 'attribution error' (Fig. 2a; see
> Methods for details)."

Second — and this is the sentence a SQUID-aware reviewer will quote back at us —
SQUID standardizes each map *by its own scale* before any cross-sequence
comparison. Methods, "Attribution map standardization":

> "For plotting sequence logos and computing attribution errors: Attribution map
> values were standardized using the transformation v_{l:c} → (v_{l:c} − v̄_l)/σ,
> where v̄_l = (1/4)∑_{c′} v_{l:c} and σ² = (1/4L)∑_{c,l}(v_{l:c} − v̄_l)². This
> transformation is essentially a gradient correction at each position, followed
> by a normalization with the square root of the total variation of the
> attribution scores across the sequence."

To first order, our claim is that a probability-space score for instance *a*
equals its latent-space score times p0_a(1−p0_a). That is a **per-instance
positive scalar**. Dividing each map by its own σ cancels a per-instance positive
scalar. So the standard practice in the field already neutralises, empirically
and without any theory, precisely the leading-order effect our clause (ii)
identifies. Our 1.8 ablation compares *unstandardized* probability-space scores
against latent-space scores, and its reversal rates are therefore measured
against a baseline nobody in this literature uses.

Third, SQUID has already reported that attribution quality degrades with the
operating point, in the direction our clause (ii) predicts, and attributed it to
site strength rather than to link geometry. Results, "SQUID resolves weak binding
sites" (Fig. 5b):

> "Figure 5b shows that, as expected, the resulting errors for all attribution
> methods increased as PWM score decreased."

**Verdict on (ii).** The claim "probability-space scores are not comparable
across instances" is *true*, is *not new*, and its standard mitigation is
already in the SQUID Methods. The specific rank-reversal framing is new as a
measurement, but is a straw man in its present form. **Required change to the
experiment, not just the prose:** 1.8 must add a third arm — the probability-
space score standardized per instance in SQUID's way (divide by the per-instance
score scale) — and report its reversal rate. If that arm is also near zero, the
honest conclusion is that the latent transform buys nothing that per-instance
standardization does not already buy for *ranking*, and the contribution
collapses onto clause (iii). We should find this out before a reviewer does.

### Claim (iii) — split-conformal fidelity certificates lose conditional coverage across baseline strata

Our version: `link.py::stratify_by_baseline`, Proposition 22, `ABLATIONS.md` 1.9;
`gate2_link.md` reports a probability-space max−min conditional coverage gap of
0.162 [0.153, 0.172] vs. 0.020 [0.016, 0.024] in latent space, α = 0.1, 5 strata,
10 seeds, no infinite quantiles.

**Not prior art in any of these papers, by text and by grep.** Neither SQUID nor
MAVE-NN contains the words "conformal" or "coverage" in a technical sense. In
the SQUID preprint the single hit for "coverage" is the title of reference 1
("Predicting RNA-seq coverage from DNA sequence"). In the MAVE-NN full text,
"conformal" and "coverage" both occur zero times.

What they do have is *parametric* uncertainty. SQUID's Fig. 1b legend lists
"PI, prediction interval", i.e. an interval from the fitted noise model:

> "The noise model, p(y|ŷ), describes the expected distribution of DNN
> predictions y about the most-probable prediction ŷ. The noise model can be
> defined using a Gaussian distribution, a Student's t-distribution, or the
> skewed t-distributed of Jones and Faddy."

MAVE-NN's uncertainty discussion is about parameter error bars, and it is
explicitly conditioned on gauge fixing (Methods, "Parameter uncertainty"):

> "Another important detail when assessing parameter uncertainty is to ensure
> that both the gauge modes and the diffeomorphic modes of each model are fixed.
> This is necessary so that differences in the parameters that do not affect
> model predictions do not inflate uncertainty estimates."

Neither is a distribution-free finite-sample statement, and neither is
conditional on a covariate stratum. **This clause is where Theorem 2 survives
contact with SQUID.**

### Claim (iv) — latent-space scores are baseline-invariant

Our version: `tests/test_link.py`, "equal latent shifts give equal latent scores
regardless of baseline".

**Established as practice in SQUID itself, and our statement is narrower than we
write it.** SQUID's Methods already logit-transform a probability output before
surrogate fitting, for exactly the reason we give, with no theory attached.
Methods, "Deep learning models", baseline CNN bullet:

> "This model takes as input a DNA sequence of length 200 nt and outputs a single
> probability. In our analysis of the effects of benign overfitting, y was
> computed as the logit of the output probability."

That single sentence is the strongest piece of prior art against our framing of
(iv): the *practice* of measuring in latent space when the head emits a
probability is already in the SQUID Methods. If we write anything resembling
"we propose measuring faithfulness in latent space", this sentence is the
counterexample and it will be found.

Separately, clause (iv) as stated is invariance under an *additive* change of
operating point only. See §3: Kinney & Atwal show a scalar latent representation
retains two diffeomorphic modes, additive **and multiplicative**. |Δη| is
invariant to η → η + a but transforms as |Δη| → |b||Δη| under η → a + bη. So (iv)
holds only for a fixed model with a fixed head scale and a fixed, known link, and
must be scoped that way in the formal statement. It is also broken at the
boundary by our own `EPS = 1e-6` clamp, which `paper/theorems.md` already records
under A2.4.

---

## 2. What SQUID does not do

Each item verified against the v1 preprint text and against the code at
`08e3da0e`. Where our assumed claim is false, it is marked **FALSE**.

**(a) No theorem, no formal statement.** True. `grep -ci` over the preprint text:
`theorem` 0, `guarantee` 0, `confidence` 0. The paper's inferential apparatus is
entirely non-parametric hypothesis tests: "p-values in panels b and e were
computed using a one-sided Mann-Whitney U test", and for Table 1 "P-values report
results from a paired Wilcoxon signed-rank test on the 15 locus-specific
correlation values". There is no proposition, lemma, or proof anywhere in SQUID.

**(b) No coverage guarantee.** True. `conformal` 0 and `coverage` 0 (technical
sense) in both SQUID and MAVE-NN; `conformal` and `coverage` 0 in the squid-nn
codebase. SQUID reports prediction intervals from a fitted parametric noise
model; that is a model-based interval, not a finite-sample coverage claim, and
SQUID never validates its interval's empirical coverage.

**(c) No identifiability statement about the link beyond MAVE-NN gauge and
diffeomorphic modes.** True, and the qualifier is doing all the work. SQUID
inherits identifiability entirely from MAVE-NN and says so; its only
identifiability act is to fix a gauge for display, Methods:

> "For plotting heatmaps of additive and pairwise-interaction model parameters
> (Fig. 6c,d,e and Fig. 7d): Parameters were standardized as in ref.26 using the
> 'empirical gauge'."

The substantive theory is MAVE-NN's and Kinney & Atwal's; see §3.

**(d) No treatment of comprehensiveness / sufficiency / AOPC / deletion /
insertion.** True, emphatically: all five terms are 0 hits in the preprint text
and 0 hits in the codebase. SQUID's faithfulness proxies are different objects:
surrogate–DNN agreement (R² on held-out in-silico MAVE data), cross-locus
attribution error (Euclidean distance to the ensemble-mean map), robustness to
benign overfitting, and zero-shot SNV-effect correlation (Table 1). None of them
is an occlusion-curve metric. Our Theorem 2 is about occlusion-curve metrics
computed under soft masks; SQUID has no analogue, so there is no collision here
at all — but equally, we cannot claim SQUID as *support* for our metric choice.

**(e) "Sequence CNNs only."** **FALSE as written — fix this in our prose.** SQUID
evaluates six models and one of them is a transformer. Methods, "Deep learning
models":

> "This study used six DNNs: ResidualBind-32, Basenji-32, DeepSTARR, Enformer,
> BPNet, and a baseline CNN that predicts ChIP-seq data for the human TF GABPA."

Enformer is attention-based, not a CNN. The correct and still-sufficient
statement is: **every model SQUID analyses takes a one-hot DNA sequence as input;
none is a graph or message-passing model.** Verified: `GNN` 0 hits and
`torch_geometric` 0 hits in both the preprint and the codebase; `message passing`
0 hits in the preprint; the four `graph` hits in the codebase are TensorFlow 1
session plumbing (`tf.Graph()` in `squid/predictor.py:130-134`) plus a boilerplate
disclaimer in `README.md`, and the single `graph` hit in the preprint is inside a
reference title. There is also no notion of a node-level target readout, a
neighbourhood, or a soft node mask anywhere in SQUID: its perturbation primitive
is random mutagenesis of one-hot characters in a window
(`squid/mave.py::InSilicoMAVE.generate`, `squid/mutagenizer.py`).

**Additional finding not in our list, and it cuts against us.** SQUID fits its
surrogate to whatever scalar the predictor returns, with no transform applied by
the library. `squid/mave.py` calls `self.mut_predictor(x_mut, x, self.save_window)`
and stores the result; `SurrogateMAVENN.train` casts `mave_df['y']` to float32 and
passes it straight to `mavenn.Model.set_data`. The only transforms in the package
are BPNet's own profile reductions (`squid/predictor.py`: `'wn'` softmax-weighted
reduction, `profile_pca`, summation). So the *library* is agnostic to output
space; the logit choice quoted in §1(iv) is a per-analysis decision made by the
authors in the paper's Methods. That is worse for us than a library default would
be: it means the authors knew to do it and did it deliberately.

---

## 3. Does MAVE-NN's gauge / diffeomorphic machinery subsume our latent-space correction?

### Definitions, verbatim

Kinney & Atwal 2014 (arXiv:1212.3647v3), §4 "Diffeomorphic modes". The setting is
a signal S, a model ("filter") θ mapping S to an internal representation R, and a
measurement M produced from R by an unknown noise function:

> "In Appendix D we prove that two filters are information equivalent if and only
> if their predicted representations are related by an invertible
> transformation."

> "As an objective function, mutual information is inherently incapable of
> distinguishing between information equivalent filters. In practice this means
> that maximizing mutual information within a set of parametrized filters can
> leave some directions in parameter space unconstrained. Here we term these
> directions 'diffeomorphic modes.'"

§4.1 "An equation for diffeomorphic modes" gives the defining condition. For an
infinitesimal transport θ'^i = θ^i + g^i(θ):

> "If the vector field g^i(θ) represents a diffeomorphic mode of Θ, this
> transformation must be invertible, meaning the values ∑_i g^i(θ)∂_i R^µ cannot
> depend on S except through the value of R."

> "∑_i g^i(θ) ∂_i R^µ = h^µ(R, θ). (16) … This is the equation that any
> diffeomorphic mode g^i(θ) must satisfy."

§4.2 "General linear filters", the case that matters for a scalar latent
phenotype:

> "In particular, if R is a scalar, then h = a + bR. In this case we observe two
> diffeomorphic modes, corresponding to additive and multiplicative
> transformations of R."

And, critically for us, §4 already contains the idea that knowing the
nonlinearity removes these degeneracies:

> "Such up-front knowledge about the nonlinearities of linear-nonlinear filters
> can eliminate diffeomorphic modes of the underlying linear filters in useful
> and non-obvious ways (Kinney, 2008; Kinney et al., 2010)."

MAVE-NN, §"Gauge modes and diffeomorphic modes" (Results/Methods):

> "G-P maps typically have non-identifiable degrees of freedom that must be
> fixed, i.e., pinned down, before the values of individual parameters can be
> meaningfully interpreted or compared between models. These degrees of freedom
> come in two flavors: gauge modes and diffeomorphic modes. Gauge modes are
> changes to θ that do not alter the values of the latent phenotype ϕ.
> Diffeomorphic modes [20, 24] are changes to θ that do alter ϕ, but do so in
> ways that can be undone by transformations of the measurement process p(y|ϕ)."

> "As shown by Kinney and Atwal [20, 24], the diffeomorphic modes of linear G-P
> maps (such as the additive and pairwise G-P maps featured in Figs. 3, 4, and 5)
> will typically correspond to affine transformations of ϕ, although additional
> unconstrained modes can occur in special situations."

> "MAVE-NN automatically fixes the gauge modes and diffeomorphic modes of
> inferred models (except when using custom G-P maps). The diffeomorphic modes of
> G-P maps are fixed by transforming θ via θ_0 → θ_0 − a, and then … p(y|ϕ) →
> p(y|a + bϕ)."

### (a) versus (b)

**(a) Inference of a latent phenotype up to diffeomorphism.** This is a statement
about *what data can determine*. When the map from latent phenotype to
observation is unknown and must be inferred jointly, the likelihood (or the
mutual information) is blind to any invertible reparameterisation of ϕ that can
be absorbed into p(y|ϕ). Consequence: ϕ and its parameters are determined only up
to a group of transformations (affine, for linear G-P maps), so two independently
fitted models' parameters cannot be compared until a convention is imposed. The
fix is a *convention* (gauge fixing), not a correction; nothing is being
undistorted, and no quantity is being made more accurate.

**(b) Our Theorem 2.** The link is **known and fixed** (`link.py` hardcodes the
sigmoid; A2.2). There is no inference problem, no likelihood, no unidentified
nonlinearity. Our object is not a parameter but an *explanation score* — a
functional of a masked and an unmasked model output — and our claims are about
(i) how that score transforms as a function of the operating point p0, (ii) how a
*ranking* of such scores across instances behaves, (iii) what happens to the
finite-sample conditional coverage of a split-conformal certificate calibrated on
those scores, and (iv) invariance of the latent-space score.

**Does (b) follow trivially from (a)? No — but it is closer than our current
framing admits, and the honest answer has three parts.**

1. **(b)(i) and (b)(iv) do not follow from (a); they follow from the chain rule.**
   Diffeomorphic-mode theory addresses an *unknown* nonlinearity; ours is known.
   Given a known link, the p0(1−p0) factor and the invariance of latent
   differences are one line of calculus. So (b)(i)/(iv) are neither implied by
   (a) nor a contribution over it — they are implied by first-year analysis. We
   cannot claim novelty by pointing at the gap between (a) and (b) here.

2. **(a) actively weakens our clause (iv).** Kinney & Atwal's scalar result —
   "h = a + bR … two diffeomorphic modes, corresponding to additive and
   multiplicative transformations" — says the latent scale is fixed only by
   convention. Our "latent scores are baseline-invariant" is invariance under the
   additive mode. Under the multiplicative mode |Δη| → |b||Δη|, which is exactly
   the same *kind* of per-instance positive rescaling that ruins probability-space
   comparability in clause (ii). Latent space is therefore not a canonical space;
   it is a space in which *our particular* nuisance (a shifting operating point at
   fixed head scale) happens to be the additive mode. That must be stated as a
   scope condition, and the reviewer will state it for us if we do not.

3. **(b)(iii) genuinely does not follow from (a), or from anything in either
   paper.** Gauge fixing makes parameters comparable; it says nothing about the
   distribution of a nonconformity score, about exchangeability, about
   finite-sample quantiles, or about coverage conditional on a covariate stratum.
   The step from "the score is a monotone reparameterisation of another score" to
   "split conformal calibrated on it loses conditional coverage across baseline
   strata by 0.162 while the latent version loses 0.020" is not in MAVE-NN, is not
   in Kinney & Atwal, and is not in SQUID. Neither is the Mondrian remedy
   (stratify on |logit p0|).

**Summary of §3.** MAVE-NN's machinery does not subsume our correction, because
it is answering a different question (identifiability of inferred parameters vs.
transformation of an explanation score under a known link). But it does subsume
the *motivation* we have been using, and it supplies a sharper version of the
objection to our clause (iv) than we had written down ourselves.

---

## 4. Verdict

### Is Theorem 2 still a contribution?

**Yes, but only clause (iii), and only if it is restated as a conformal-validity
result.** Clauses (i), (ii) and (iv) are, respectively: the chain rule; a known
problem with a standard mitigation already in the SQUID Methods; and a practice
already used in the SQUID Methods (`y was computed as the logit of the output
probability`). Presented as headline claims they will read to a SQUID author as
rediscovery. Presented as a background lemma plus scope conditions, feeding a
result about certificates, they are fine.

There is one further honesty requirement before we can even keep (iii) at full
strength: `gate2_link.md` reports Gate 2 on the **synthetic** substrate only,
while `ABLATIONS.md` 1.7 and 1.9 name substrates `[splice, connectome]`. Our
current G2 PASS is therefore a PASS on the substrate where the link is imposed by
construction. A reviewer who has read SQUID — a paper whose every claim is
benchmarked on six real models — will notice that instantly.

### One sentence a SQUID author would accept as fair

> We do not claim to discover that an output nonlinearity confounds attribution —
> SQUID and MAVE-NN established that, and SQUID's own Methods already take logits
> of a probability output and standardize attribution maps per sequence — we
> claim only that this confound propagates into split-conformal fidelity
> certificates as a loss of *conditional* coverage across baseline strata, which
> no prior work measures, and that Mondrian stratification on |logit p0| restores
> it.

### Citations we must add

Required, load-bearing:

1. Seitz, E. E., McCandlish, D. M., Kinney, J. B. & Koo, P. K. Interpreting
   *cis*-regulatory mechanisms from genomic deep neural networks using surrogate
   models. *Nat. Mach. Intell.* **6**, 701–713 (2024). Preprint: bioRxiv
   2023.11.14.567120. — Cite for: nonlinearity-as-confound, GE surrogates,
   latent phenotype, per-sequence attribution standardization, cross-locus
   comparison, the logit-of-probability choice in Methods.
2. Tareen, A. et al. MAVE-NN: learning genotype-phenotype maps from multiplex
   assays of variant effect. *Genome Biol.* **23**, 98 (2022). — Cite for: latent
   phenotype models, GE regression, gauge modes, diffeomorphic modes, the
   p(y|ϕ) → p(y|a+bϕ) fixing procedure.
3. Kinney, J. B. & Atwal, G. S. Parametric inference in the large data limit
   using maximally informative models. *Neural Comput.* **26**(4), 637–653
   (2014). arXiv:1212.3647. — Cite for: the definition of diffeomorphic modes
   (Eq. 16), information equivalence ⇔ invertible transformation of the
   representation, and the scalar case h = a + bR (two modes: additive and
   multiplicative). This is the citation that must appear next to our scope
   condition on clause (iv).
4. Atwal, G. S. & Kinney, J. B. Learning quantitative sequence–function
   relationships from massively parallel experiments. *J. Stat. Phys.* **162**,
   1203–1243 (2016). — Cite alongside 3 as MAVE-NN does ("[20, 24]"). **Do not
   cite arXiv:1506.04236 for this; that identifier is a different paper.**
   Verify the correct preprint id before submission.
5. Otwinowski, J., McCandlish, D. M. & Plotkin, J. B. Inferring the shape of
   global epistasis. *PNAS* **115** (2018). — The origin of the GE-nonlinearity
   idea SQUID uses; cheap to cite and expected.

Already needed elsewhere but now load-bearing for Theorem 2's framing:

6. Vovk, V., Lindsay, D., Nouretdinov, I. & Gammerman, A. Mondrian confidence
   machine / Mondrian conformal prediction (2003) — for the stratification our
   clause (iii) uses; it is not our invention.
7. Romano, Y., Barber, R. F., Sabatti, C. & Candès, E. With malice toward none:
   assessing conditional coverage / Barber, Candès, Ramdas & Tibshirani, The
   limits of distribution-free conditional predictive inference (2021) — the
   impossibility background that makes clause (iii) a *stratum-conditional*, not
   *conditional*, statement. Without this citation clause (iii) is overclaimed.
8. Tibshirani, R. J., Barber, R. F., Candès, E. & Ramdas, A. Conformal prediction
   under covariate shift (2019) — already named in Proposition 22.
9. DeYoung, J. et al. ERASER (2020) for comprehensiveness / sufficiency; Samek,
   W. et al. (2017) for AOPC; Petsiuk, V., Das, A. & Saenko, K. RISE (2018) for
   deletion / insertion. SQUID uses none of these, so their provenance is on us.

### Sentences in our framing that must change

1. **`certgnn/certify/link.py` module docstring**, currently:
   > "Attribution and faithfulness scores on bounded outputs inherit the link's
   > Jacobian, which varies with the operating point. Comparing raw scores across
   > instances with different baselines is therefore not meaningful."

   Change to attribute the observation (Seitz et al. 2024; Tareen et al. 2022) and
   to say "raw, unstandardized scores", since per-instance standardization is the
   field's existing mitigation.

2. **`paper/theorems.md`, Theorem 2 informal claim / forthcoming formal
   statement.** Add explicit scope: *known* link, *fixed* model and head scale,
   *first-order* regime (A2.3), p0 bounded away from the clamp (A2.4). Add a
   remark citing Kinney & Atwal 2014 that latent scores are invariant under the
   additive but not the multiplicative diffeomorphic mode, so "baseline-invariant"
   means "invariant to the operating point at fixed head scale" and nothing more.

3. **Anything of the form "we propose measuring faithfulness in latent space".**
   Delete. SQUID's Methods already do this ("y was computed as the logit of the
   output probability"). The proposal is the *certificate*, not the space.

4. **`ABLATIONS.md` 1.8 `defends` field**, currently claims probability-space
   scores "reverse the ranking of node effects across instances". Must add the
   per-instance-standardized probability score as a third arm; the claim as
   stated is against a baseline the prior art does not use. (This is an
   experimental change, not a wording change. `ABLATIONS.md` is not frozen;
   `PREREGISTRATION.md` is, and C2 stays exactly as written.)

5. **Any sentence describing SQUID as "sequence CNNs".** SQUID analyses six DNNs
   including Enformer, which is attention-based. Use "sequence-input models; no
   graph or message-passing model" instead.

6. **Introduction framing of the gap.** The gap is not "nobody noticed the link
   distorts explanations" — Nature MI 2024 is that paper. The gap is "no prior
   work attaches a distribution-free certificate to an explanation's fidelity, so
   no prior work could notice that the link destroys the certificate's
   conditional validity."

---

## Appendix: grep commands and results

In the clone (`squid-nn` @ `08e3da0e`), over `squid/ examples/ README.md docs/`
with `--include=*.py --include=*.md --include=*.rst --include=*.ipynb`:

```
$ grep -rniI '<term>' squid examples README.md docs --include=*.py --include=*.md --include=*.rst --include=*.ipynb | wc -l
conformal            0
coverage             0
comprehensiveness    0
sufficiency          0
AOPC                 0
deletion             0
insertion            0
graph                4   # squid/predictor.py:130,132,134 -> tf.Graph()/tf.Session(graph=...);
                         # README.md:103 -> boilerplate disclaimer
GNN                  0
torch_geometric      0
```

Output-space transforms in the library:

```
$ grep -rniI "np\.log\|log(\|log2\|log10\|transform\|softmax\|sigmoid\|exp(" squid/*.py examples/*.py | grep -v "logo\|logomaker\|logs"
squid/impress.py:38     # plot histogram of transformed deepnet predictions
squid/predictor.py:123  if self.reduce_fun == 'wn': # transformation used in the original BPNet paper
squid/predictor.py:133  wn = tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(pred)) * pred, axis=-2), axis=-1)
squid/predictor.py:235  """Function to transform predictions to scalars using summation."""
squid/predictor.py:252  """Function to transform predictions to scalars using principal component analysis (PCA)."""
squid/surrogate_zoo.py:608  Number of hidden nodes (i.e. sigmoidal contributions) ... nonlinearity component of a GE model.
```

No logit or log transform is applied by the library to the predictor's output;
the surrogate is fit to whatever scalar the `Predictor` returns
(`squid/mave.py:110,117` → `SurrogateMAVENN.train` → `mavenn.Model.set_data`).

On the SQUID v1 preprint text (`pdftotext` of the bioRxiv full PDF),
`grep -ci`:

```
global epistasis   9      latent phenotype  7      logit        1   (Methods, baseline CNN)
conformal          0      coverage          1  (ref. 1 title)  theorem      0
guarantee          0      confidence        0      comprehensiveness 0
sufficiency        0      AOPC              0      deletion     0      insertion 0
graph              1  (ref. title)          GNN    0      message passing 0
transformer        0      attention         0      Enformer     7      Basenji  7
ResidualBind      15      DeepSTARR         8      BPNet       19
```

On the MAVE-NN Europe PMC full text (`PMC9011994`), case-insensitive counts:

```
diffeomorphic 9    gauge 13    latent phenotype 53    global epistasis 7
conformal 0        coverage 0  logit 0                attribution 1     identifiab* 3
```
