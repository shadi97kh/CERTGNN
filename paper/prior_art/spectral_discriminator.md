# Prior art: is the rank-one spectral discriminator already claimed?

**Verdict: (a) ALREADY PUBLISHED. Stop.**

The claim we were about to build — that under global epistasis with an additive
latent trait and a smooth nonlinearity `g`, the pairwise epistasis matrix is
`g''(phi_0) * beta beta^T` to second order and therefore rank one, while dense
microscopic epistasis gives a full-rank matrix, so the spectrum discriminates
the two mechanisms — is stated explicitly, derived, and applied to data in:

> Kabir Husain and Arvind Murugan, **"Physical Constraints on Epistasis"**,
> *Molecular Biology and Evolution* 37(10):2865–2874 (2020).
> Preprint: [arXiv:1910.09491](https://arxiv.org/abs/1910.09491).
> [Journal](https://academic.oup.com/mbe/article/37/10/2865/5839750) ·
> [PubMed](https://pubmed.ncbi.nlm.nih.gov/32421772/)

This is not a near-miss or a buried implication. It is their Equation 2 and the
sentence immediately following it.

## What they state, verbatim

From the arXiv full text (text extracted from the PDF; equation reflowed):

> ```
> ΔΔF_ij ∝ (1/λ₀²) θ_i θ_j + c_ij                                    (2)
> ```
> "i.e., a rank-1 matrix θ_i θ_j (here, θ_i = v̂₀ · ∇δ_i E), and a sparse matrix
> c_ij that is non-zero only for residues i, j in physical contact. Notably,
> rank-1 matrices are highly constrained, with much fewer independent numbers
> than a full-rank matrix, and can therefore be reconstructed by fewer
> measurements."

And, generalising beyond their physical-mode derivation to *any* global
epistatic function:

> "This form, with an overall non-linearity applied to an underlying linear
> trait, is that of global epistasis. Moreover, expanding any global epistatic
> function F(s_i) = g(Σ θ_i s_i) generates strong yet low-rank epistasis of all
> orders: **the 2nd order term is rank-1** (as in Eq. 2) and higher order terms
> are similarly constrained."

Point by point against what we intended to claim:

| Our intended claim | Status in Husain & Murugan 2020 |
|---|---|
| Global epistasis ⇒ pairwise epistasis matrix is rank one | Stated as Eq. 2 and in the sentence above |
| The rank-one factor is the outer product of the additive coefficients | Stated: `θ_i θ_j`, with `θ_i` the per-mutation coefficient |
| Curvature `g''` sets the scale | Present; the MBE text gives the prefactor as `g''(0) λ₀²` |
| Higher orders constrained for smooth `g` | Stated: "higher order terms are similarly constrained" |
| Specific/microscopic epistasis breaks the structure | Stated: additive sparse `c_ij`, non-zero only at physical contacts |
| Deviation from rank-one is the discriminating signal | Stated: "epistatic complexity, defined as the median residual ΔΔF_ij − ΔΔF⁽¹⁾_ij" |
| Applied to real deep mutational scanning data | Yes — SVD / low-rank approximation of the GB1 epistasis matrix of Olson et al. (2014) |

They also report the result the discriminator would have been *for*: deviations
from the low-rank approximation are preferentially enriched at physical
contacts. That is the same inference we would have drawn from a spectral
outlier analysis, obtained by the same decomposition.

## The rest of the requested search

- **Poelwijk, Socolich & Ranganathan (Nat Commun 10:4213, 2019).** Walsh–Hadamard
  analysis of the combinatorially complete 2^13 eqFP611 landscape. They remove a
  global nonlinearity by a power transform (α ≈ 0.44) and report that the WH
  spectrum is **sparse** — a handful of coefficients reproduce the landscape.
  They do **not** compute the rank or eigenvalue spectrum of an epistasis
  matrix, and do **not** state the `β_i β_j` product form. Sparsity in the WH
  basis is a different structural claim from low rank in the matrix sense.
  *Not the collision; Husain & Murugan is.*

- **Carlson, Andrews & Simons (PNAS 2025, e2509444122; bioRxiv
  2025.04.08.647864).** Confirmed from the abstract: their method rests on
  monotonicity implying that "the rank-order of mutant phenotypes should be
  preserved across genetic backgrounds", and is a "simple semiparametric
  method". **"Rank" there is rank-order statistics, not matrix rank.** They do
  not use spectral structure. So our discriminator would have been distinct
  *from them* — but that no longer matters, because Husain & Murugan already
  occupy the spectral claim.

- **Otwinowski, McCandlish & Plotkin (PNAS 2018)**, **Sailer & Harms (Genetics
  205:1079, 2017)**, **Reddy & Desai (eLife 2021)**: not examined in detail.
  Once the claim was found stated outright in Husain & Murugan, further checking
  could only add prior art, not remove it.

- **Random matrix theory nulls (Marchenko–Pastur, Tracy–Widom) for epistasis
  matrices**: no such application surfaced in this search.

## What is *not* in Husain & Murugan

Recorded for completeness, and none of it rescues the direction:

1. No comparison of the empirical spectrum against a **random-matrix null**.
   They measure a residual from a rank-1 (or rank-3) fit, not a spectral test
   with a calibrated null distribution.
2. No explicit **"rank k−1 for g ∈ C^k"** statement. They assert higher-order
   terms are "similarly constrained" without the exact rank formula.
3. Their derivation motivates the rank-one form from a **slow collective
   physical mode**, though the generalising sentence quoted above is stated for
   any global epistatic function and does not depend on that motivation.

These are refinements of a published result, not a new discriminator. A paper
whose contribution was "we add a Marchenko–Pastur null to Husain & Murugan's
rank-one test" is a different and much smaller paper than the one we were
scoping, and it is not what this project needs.

## Verification notes

- Equation 2 and both quoted sentences were read from the extracted text of the
  arXiv PDF (arXiv:1910.09491), not from a summary.
- The GB1 / Olson et al. (2014) SVD application and the `g''(0) λ₀²` prefactor
  come from the published MBE version, retrieved separately; the arXiv preprint
  text cites Olson et al. as reference [53] and discusses DMS comparison, with
  the figure-level empirical analysis in the journal version.
- The exact wording of the "fundamental ambiguity" passage in Carlson et al.
  could **not** be verified: PNAS returned 403 and bioRxiv 429. The
  characterisation of their method above is from the published abstract, which
  is unambiguous about the rank-order basis of the method but does not contain
  that phrase. If that exact wording matters for citation, it still needs
  checking against the PDF.


---

# Addendum, 2026-08-30: operating points, and the latent-dimensionality reframe

## Check A — which landscapes can carry a rank-one prediction at all

For a logistic link, `g'' = sigma(1-sigma)(1-2sigma)`, which **vanishes exactly at
sigma = 0.5** and is maximised at sigma = 0.211 or 0.789 with |g''| = 0.0962. So
a landscape whose wild type sits at the inflection has no rank-one term at
leading order at all: the leading structure there is the rank-TWO third-order
term. Wild-type operating points, read from the data on disk (no fitting), with
sigma taken as the wild type's quantile within each library's own phenotype
distribution:

| landscape | sigma at WT | \|g''\| | status |
|---|---:|---:|---|
| eqFP611 red | 0.236 | **0.0952** | usable, essentially at the optimum |
| GB1 | 0.826 | **0.0938** | usable, near-optimal |
| BRCA2 MPSA (consensus 5'ss) | 0.989 | 0.0105 | weak, saturated |
| eqFP611 blue | 0.998 | 0.0017 | weak, saturated |
| **FAS exon 6 (Julien 2016)** | **0.500** | **0.0000** | **degenerate** |
| **FAS combinatorial (Baeza-Centurion 2019)** | **0.500** | **0.0000** | **degenerate** |

**FAS is the worst available dataset for this statistic, and by construction.**
Julien et al. state they transfected "under conditions that lead to
approximately 50% exon inclusion, matching the levels of exon 6 inclusion in
endogenous transcripts in this cell line" -- chosen so mutations could move
inclusion in both directions. That places the wild type precisely where g''
vanishes. The Baeza-Centurion combinatorial library is the same exon under the
same design and inherits the same defect, so obtaining it would not help; it was
not downloaded.

Caveats on this table. sigma is a quantile proxy, not a fitted link, per the
instruction not to fit anything; the proxy choice matters, and switching
eqFP611 red from a min-max position to a quantile moved it from 0.036 to 0.236
and from "modest" to "near-optimal". FAS is the one row that is proxy-
independent, because its 50% figure is the paper's own statement about the
assay. Note also the cross-check against `paper/tables/epistasis_spectrum.md`:
eqFP611 red has the best operating point in the table and yet has ZERO
eigenvalues outside its noise bulk, so a good operating point does not by itself
buy a measurable signal.

## Check B, part 2 — the latent-dimensionality reframe is also published

The reframed claim -- how many latent dimensions does a genotype-phenotype map
need, and is the one-dimensional assumption of MAVE-NN and SQUID correct -- is
answered in:

> Peter D. Tonner, Abe Pressman, David J. Ross, **"Interpretable modeling of
> genotype-phenotype landscapes with state-of-the-art predictive power"**,
> *PNAS* 118 (2021). [PMC9245639](https://pmc.ncbi.nlm.nih.gov/articles/PMC9245639/).
> Software: `github.com/usnistgov/lantern`.

LANTERN is a hierarchical Bayesian model that learns a low-dimensional latent
space of additive mutational effects and **estimates its dimensionality from the
data**. Verified from the paper: it ranks latent dimensions by variance,
computes "the expected log-likelihood of each observation with an increasing
number of dimensions included in the model", and applies "a one-sided,
two-sample Kolmogorov-Smirnov test to compare the empirical distributions" of
those likelihoods, counting dimensions with p <= 0.05.

Its answers on real data: **"Across these datasets, the latent dimensionality
learned by LANTERN ranged from three to five"** -- three for LacI, five for
SARS-CoV-2, three for avGFP with the first dimension carrying 96.7% of the
mutational-effect variance. And on the one-dimensional assumption specifically:
**"we allow for multiple different biophysical mechanisms to influence
biological function by modeling multiple latent dimensions"**.

So the question is not open, and the answer is not one. LANTERN also estimates
the dimensionality *directly by model comparison* rather than reading it off the
rank of an epistasis matrix, which is the stronger method: it does not inherit
the small-beta expansion, the g''(phi_0) = 0 degeneracy, the rank-2 noise
artifact, or the misspecified spectral null that the proof-check found in the
matrix-rank route.

Related and also prior: Husain & Murugan's abstract already frames their result
as "the dimensionality of mutational effects is reduced"; and Ghosh et al.,
"Genotype-fitness mapping of adaptive mutants reveals shifting low-dimensional
structure across divergent environments", *PLOS Biology* (2026), infers
low-dimensional fitness landscapes and how their dimensionality shifts across
environments.

## Verdict on both checks

**Check A: FAS is degenerate for this statistic and must not be the first
dataset.** GB1 and eqFP611 red are the only usable operating points.

**Check B: already published, in both framings.** The rank-one discriminator is
Husain & Murugan 2020. The latent-dimensionality estimate is LANTERN, PNAS 2021,
with software and published answers of three to five dimensions. Per the
instruction, stop.
