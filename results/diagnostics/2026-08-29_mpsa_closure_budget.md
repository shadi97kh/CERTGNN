# Is the real-data closure shortfall an optimizer artifact?

On the synthetic substrate the neural class reached closure R^2 = 1.000000
to six figures, but only after three fixes: warm-starting the refit from
the reference map's own weights, keeping the best iterate rather than the
last, and inverting the warp exactly rather than by interpolation. Before
those, it reported 0.999848 and that shortfall drove an entire spurious
indistinguishability radius.

On the real BRCA2 5' splice site MPSA the same pipeline, with all three
fixes in place, reports 0.998827 [0.997761, 0.999512] over 10 seeds. The
obvious question is whether that is the same artifact again.

**It is not.** Refit budget against closure, one seed, 4000 sequences,
warm-started with best-iterate tracking, sinusoid warp at s = 0.95,
omega = 2:

| refit epochs | closure R^2 |
|---|---|
| 1,200 | 0.998979 |
| 4,000 | 0.999381 |
| 12,000 | 0.999522 |
| 30,000 | 0.999591 |

Twenty-five times the budget buys 0.0006 and the curve asymptotes near
0.9996. It is converging, but not to 1.

**Likely mechanism.** On the synthetic substrate the latent was a smooth
function of continuous node features. Here the input is one-hot over 9
sequence positions, a discrete domain, and the G-P map is a network with
32 hidden units. Re-representing a monotone warp of its own output on that
domain at that capacity is not exactly achievable.

**Consequence, which reverses the synthetic conclusion.** Because closure
is not exact, the twin's predictions differ from the original by more than
numerical noise, so the twin IS detectable. The artifact checks agree: the
effect size increases with warp strength on real data (Spearman +0.788,
+0.440, +0.718 across the three families) where on synthetic data it
showed no trend. The indistinguishability radius is therefore measurable
here and unmeasurable there, and the identifiability problem is WEAKER on
real data than the synthetic probe implied.

**Scope.** This is conditional on the architecture. The honest statement is
"not closed at 32 hidden units", not "not closed". Whether a larger network
closes to 1.000000 is untested and would change the conclusion if it does.
