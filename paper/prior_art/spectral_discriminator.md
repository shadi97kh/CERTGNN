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
