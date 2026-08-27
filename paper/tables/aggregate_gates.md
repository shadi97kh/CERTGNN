# Gate family (Holm-Bonferroni)

| gate | verdict | p (bootstrap vs threshold) | Holm-adjusted p | reject at 0.05 | seeds | run |
|---|---|---|---|---|---|---|
| G2 | PASS | 0.0005 | 0.0005 | True | 10 | 20260827T194159Z_cdcc06a_5766d0a3 |

Verdicts are the pre-registered threshold comparisons; p is the one-sided bootstrap probability that the seed-mean fails its threshold, (b+1)/(B+1) convention; it is a CI inversion, not a null-calibrated test, and saturates near 1/(B+1) whenever the threshold lies outside the seed range.
