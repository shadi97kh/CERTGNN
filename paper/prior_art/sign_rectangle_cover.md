# Prior art: minimal width of a sign-constrained layer as a signed rectangle cover

**Verdict: (b) KNOWN IN COMBINATORICS, NEVER CONNECTED TO NETWORK WIDTH.**

The combinatorial object in the reduction is not new and is not even obscure: a
rank-1 sign pattern supported on a combinatorial rectangle is precisely a
**balanced signed biclique**, and covering a signed bipartite graph by them is a
studied, NP-hard problem. The mathematics is therefore citable rather than
inventable.

What did **not** surface anywhere is the consequence: that the minimal width of a
monotone-activation layer realizing a prescribed Jacobian sign pattern equals
that covering number. The nearest width lower bounds for sign-constrained
networks prove exponential separations by entirely different means and never
mention a covering number, a sign pattern, or a Jacobian.

---

## 1. Sign-pattern algebra

Sign pattern matrices — entries from {+, −, 0} — are a mature field, with
Brualdi & Shader, *Matrices of Sign-Solvable Linear Systems* (Cambridge Tracts
in Mathematics 116, 1995) the standard reference. The governing idea is exactly
ours: "the special circumstances under which an algebraic, analytic or geometric
property of a matrix can be determined from the combinatorial arrangement of the
positive, negative and zero elements".

**What is already named.** For *nonnegative* sign patterns the minimal-inner-
dimension factorization has a name:

> "The **Schein rank** of a nonnegative sign pattern matrix A is the smallest
> positive integer k such that A = HK for some m×k (k×n) nonnegative sign
> pattern matrices H and K."

That is our quantity in the all-positive special case, and it coincides with
Boolean rank. The minimum rank of a sign pattern, `mr(A)`, is likewise a
standard object, and is bounded by Boolean rank (`mr(A) ≤ min{Boolean row rank,
Boolean column rank}`).

**What was not found.** The signed version — minimal k such that a *signed*
target factors as a product of sign patterns with every entry of the product
**unambiguously determined** — did not surface as a named, characterized
quantity. Note this is strictly stronger than `mr`: minimum rank asks only that
*some* real matrix with that sign pattern have low rank, whereas we require the
sign to be forced by the patterns alone, uniformly over the positive diagonal
and over magnitudes.

**Largest residual risk, stated plainly.** Brualdi & Shader is a 1995 Cambridge
monograph and is not readable online; this check could not open it. The
determined-product question is exactly the kind of result that book would
contain, and a library copy should be checked before any claim of novelty is
made for the sign-pattern half. Treat section 1 as *unverified against the
canonical source*.

## 2. Combinatorics — this is where the object already lives

**A rank-1 sign pattern on a rectangle is a balanced signed biclique.** By
Harary's balance theorem a signed graph is balanced iff its vertices admit a
bipartition `V = X ∪ Y` with every positive edge inside a part and every negative
edge crossing. Specialised to a signed bipartite graph with parts R and C, set
`u_i = ±1` according to whether `i ∈ R` lies in X or Y, and `v_j = ±1` likewise
for `j ∈ C`; then an edge is positive exactly when its endpoints are on the same
side, i.e. `s(i,j) = u_i · v_j`. Balance is therefore *identical* to the sign
submatrix being rank one — which is the structure each hidden unit contributes.

**And covering by them is studied and hard.**

- *k-Balanced Biclique Partition on Signed Bipartite Graphs* treats partitioning
  the edges into at most k balanced signed bicliques and proves the decision
  problem **NP-hard by reduction from Nonnegative Matrix Factorization**.
- In the unsigned case the correspondence is textbook: Boolean rank of a 0-1
  matrix equals the **minimum biclique cover number** of its bipartite graph, and
  equals the least q with `M = BC` over the Boolean semiring.
- Minimum biclique cover is NP-complete (Orlin), and strongly inapproximable: it
  is NP-hard to distinguish instances admitting `k ∈ O(n^ε)` from those requiring
  `k ∈ Ω(n^{1−ε})` for every ε > 0, for both cover and partition.

So the covering number is a named quantity with known hardness and known
inapproximability. A "minimal width is a covering number" theorem inherits all of
that for free — which is a strength, not a weakness, provided it is cited rather
than rediscovered.

**One gap.** The NP-hardness result located is for **partition**; the reduction
needs the **cover** version, where rectangles may overlap provided they agree in
sign. Overlap-permitted covers are generally easier than partitions, so hardness
does not transfer automatically. The Springer chapter was paywalled and could not
be read, so whether it also treats cover is unverified.

## 3. Neural width lower bounds

The closest work is **Mikulincer & Reichman, "Size and depth of monotone neural
networks: interpolation and approximation"** (arXiv:2207.05275, v2 2024). They
study threshold networks with all non-negative weights and prove that "there are
monotone real functions that can be computed efficiently by networks with no
restriction on the gates whereas monotone networks approximating these functions
need exponential size in the dimension."

That is a genuine size lower bound for a sign-constrained network. But it is a
different theorem about a different object, and the extracted full text contains
**zero occurrences** of *Boolean rank*, *biclique*, *rectangle cover*, *covering
number*, *sign pattern*, or *Jacobian*. Their target is a monotone **function**;
ours is a Jacobian **sign pattern**, i.e. which input-output pairs have a
determined derivative sign. Their bound is exponential-in-dimension by a
counting/approximation argument, not an exact combinatorial invariant of a
prescribed target.

Adjacent and also not the collision: Runje & Shankaranarayana, *Constrained
Monotonic Neural Networks* (ICML 2023), and the 2025 follow-up on universal
approximation beyond bounded activations, both concern universal approximation
under weight-sign constraints rather than width lower bounds from a target's
combinatorial structure. Known results that non-negative-weight ReLU networks are
severely limited, and that non-positive constraints are more expressive than
non-negative ones, are expressivity statements of the same different kind.

## Verdict, and the one mathematical gap to close first

**(b), the strongest of the three outcomes.** The combinatorics is established
and citable — balanced signed bicliques, Harary balance, biclique cover as
Boolean rank, NP-hardness and inapproximability. The identification of that
covering number with the **minimal width of a monotone-activation layer realizing
a prescribed Jacobian sign pattern** was not found in either literature.

Before building anything, one step of the reduction needs a proof rather than an
assertion, and it is not a prior-art question:

> Entry (i,j) is claimed determined **iff** all nonzero terms `S2[i,k]·S1[k,j]`
> share a sign. The "if" direction is immediate since `σ' > 0`. The "only if"
> direction is not: it requires that whenever two terms carry opposite signs,
> some input `x` makes the sum take either sign — which depends on the range of
> `(σ'(·), …, σ'(·))` achievable across hidden units at a common input, and on
> the freedom in the weight magnitudes. For activations whose derivative range is
> bounded away from a full positive cone this may fail, leaving entries that are
> determined despite mixed-sign terms and making the covering number an upper
> bound on width rather than an equality.

Also note the cover is constrained beyond ordinary biclique cover: zeros of the
target must remain **uncovered**, and overlapping rectangles must agree in sign.
That is a cover of the support graph by balanced signed bicliques *consistent
with the target*, which is the right object but not literally the classical one,
so the imported hardness needs restating for the constrained version.

## Verification notes

- The Consensus and Scholar Gateway search tools disconnected mid-session; this
  check used general web search and direct fetches only, which is weaker coverage
  than the earlier prior-art checks in this directory.
- Brualdi & Shader (1995) was **not** read. This is the main open risk.
- The k-balanced-biclique-partition NP-hardness is taken from a search summary of
  the Springer chapter, not from the chapter itself (paywalled).
- Mikulincer & Reichman was read from extracted PDF text; the keyword counts
  above are from that extraction.
