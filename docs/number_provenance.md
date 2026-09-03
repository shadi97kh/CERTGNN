# Number provenance for `paper/neurips_2026.tex`

Every numeric claim in the prose of the paper, traced to the artifact it should
come from. Built by reading the source read-only and re-deriving each value from
the committed run records, the figure sidecars, the raw library, or closed form.

**Files read.** `paper/neurips_2026.tex` (1027 lines), `paper/appendix.tex`
(130), `paper/neurips_2026.sty`, the six figures under `paper/figures/` and their
`*_data.json` sidecars, `experiments/make_tables.py`, `experiments/make_appendix.py`,
`experiments/make_figures.py`, `experiments/occurrence.py`,
`experiments/closure_capacity.py`, `experiments/closure_heldout.py`,
`configs/base.yaml`, and `data/raw/mpsa_data.csv.gz`.

**Scope.** Prose only. Rows inside the seven `% >>> GENERATED` blocks are written
by `make_tables.py` from the run records and are traceable by construction; they
appear here only where a prose number and a table number describe the same
quantity.

## Source keys

| key | artifact | seeds |
|---|---|---|
| `OCC` | `results/runs/20260830T002223Z_6f98de3_6e25cf7d` (occurrence) | 10 |
| `SEP` | `results/runs/20260830T030657Z_972496a_5da8cb38` (separation / twin) | 10 |
| `CS` | `results/runs/20260829T191427Z_94dd5a2_2267b74e` (closure_search) | 10 |
| `CH` | `results/runs/20260829T181937Z_b2951ae_9fdd4067` (closure_heldout) | 10 |
| `DUMP` | `results/runs/20260903T013725Z_ca95530_3353d957` (per-instance dump) | **3** |
| `LIB` | `data/raw/mpsa_data.csv.gz`, 30,483 rows | n/a |
| `SIDE` | `paper/figures/*_data.json` | varies |
| `FORM` | closed form, no data | n/a |

## Summary

99 claim rows, several of which carry more than one number (a range, or a
row of four closed-form values).

| verdict | rows |
|---|---|
| verified | 98 |
| mismatch | 1 |
| unverifiable | 0 |

One mismatch, and it is a seed-count claim: see C1 below. Nothing in the prose
was untraceable: every number resolves to a run record, a figure sidecar, the
raw library, the source code, or closed form.

---

## Cross-appearance audit

Values the brief singled out, checked for two magnitudes anywhere in the document.

### C1 — `mismatch` · seed count: "at least 5" vs "three seeds"

`neurips_2026.tex:183` states **"All reported quantities use at least 5 seeds
with bootstrap intervals."** `neurips_2026.tex:422` describes Figure 2 as
**"three seeds"**, and `DUMP` confirms `n_seeds = 3`, `seeds = [0, 1, 2]`.

Figure 2 is the source of seven reported numbers: 203,200 · 24,000 · 97% ·
8,861 · 4.4% · 30% · 29,310, plus the 44.2 / 35.0 / 63.9 / 91.0 decomposition
quoted at lines 300–304. All of those are three-seed quantities, so the line-183
claim is false as written.

This also breaches `CLAUDE.md` ("Minimum 5 seeds for any reported number").
Two honest repairs, neither of which touches a number: qualify line 183 to
exclude the per-instance dump and say so at Figure 2, or re-run the dump at
5+ seeds. Flagged, not fixed, per instruction.

### C2 — `verified, but two magnitudes` · top-3 mass at 128×1: 98% and 97%

| line | as written | value | source |
|---|---|---|---|
| 74 (abstract) | "a median $98\%$ of a sequence's total in 3 of 9 positions" | 0.9758 | `OCC.pooled_conc_top3` (10 seeds) |
| 398 | "to $98\%$ (width 128 depth 1)" | 0.9758 | `OCC` (10 seeds) |
| 402 | "a median $97\%$ in three positions" | 0.9726 | `SIDE fig2.panel_a.median_top3_mass` (3 seeds) |
| 424 (Fig 2 caption) | "a median $97\%$ of the mass" | 0.9726 | `SIDE` (3 seeds) |

Both are correct for their own run. The gap is the seed count, not an error, and
lines 401–402 already attribute the 97% to Figure 2 explicitly. No change needed,
but it is the one place a reader can catch the document quoting the same
quantity at two magnitudes.

### C3 — `verified, but two magnitudes` · 128×3 top-3 divergence: 90% and 91%

| location | as written | value | estimator |
|---|---|---|---|
| 285 | "73\% and 90\% ... depths 2 and 3" | 0.905 | `1 - OCC.pooled_tk_ex3` |
| 511 | "$90$--$94\%$ at depth 3" | 0.905 | same |
| Table 3 (`tab_topk`) | top-3 exact 10% ⇒ 90% differ | 0.905 | same |
| Table 6 (`tab_robust_full`) | "91\% [89, 92]" | 0.909 | mean over seeds of per-seed median |

Pooled 0.905 rounds to 90%; the mean of per-seed medians is 0.909 and rounds to
91%. Both are defensible and the tables label their estimators, but a reader
comparing Table 3 with Table 6 sees 90 against 91 for the same cell.

### C4 — `verified` · 0.71 carries two unrelated meanings

`neurips_2026.tex:76` uses **0.71** for the expected Spearman correlation under
Proposition 1 (`FORM`: $1 - 210/720 = 0.708$). `neurips_2026.tex:402` uses
**0.71** for the median rank-1 attribution mass (`SIDE`: 0.7086). Numerically
coincidental, semantically unrelated. Both correct; worth knowing they are not
the same quantity.

### C5 — `verified` · mixed provenance inside one paragraph

Lines 298–304 quote 18% / 68% / 29% from `OCC` (10 seeds) and 203,200 / 44.2 /
35.0 / 63.9 / 91.0 from `DUMP` (3 seeds) without distinguishing them. Every
value is correct; the paragraph does not say the populations differ.

### Values checked and found consistent everywhere

- **43%** (best-cell top-3 divergence) — lines 68, 192, 213, 248, 446, 513, and
  Tables 1, 2, 3, 5, 6, and the workflow figure. One magnitude throughout.
- **94%** (worst cell) — lines 69, 192, 511. Consistent.
- **30,483** (library) — lines 122, 342, 357, 526, appendix 14, 31, workflow
  figure. Consistent, and equals `len(LIB)`.
- **154 / 190** (tied pairs at 128×1) — line 280, Table 1, Table 6, workflow
  figure. Consistent.
- **2280 / 1610** (grid totals) — line 207; equals the column sums of Table 5.
- **6.7 × 10⁻¹⁹** — lines 489, 994, appendix 86, 115, 121. Consistent.
- **0.708** — lines 413, 427, 690, 713. Consistent.
- **8.8 × 10⁷** — line 346, appendix 43, Table 7. Consistent.
- **0.3 to 2.1** (separation) — line 341; Figure 3 plots the same field
  (`n_separate_median.mean`), so text and figure agree.
- **10 seeds** — line 912, appendix 47, 91, workflow figure. Consistent with
  every run record except `DUMP` (see C1).

---

## Full claim table

### `paper/neurips_2026.tex` — abstract and Sections 1–3

| line | claim as written | value | should come from | verdict |
|---|---|---|---|---|
| 68 | top-3 differs at best-fitting cell | 43% | `OCC.128x1 1-pooled_tk_ex3` = 0.43 | verified |
| 69 | rising at the worst | 94% | `OCC` max over cells = 0.94 | verified |
| 70 | tightening discards | 17 of every 18 | `OCC` seed files, 153.8→8.7 pairs (0.943) | verified |
| 74 | median top-3 mass, best cell | 98% | `OCC.128x1 pooled_conc_top3` = 0.9758 | verified (C2) |
| 75 | positions | 3 of 9 | `OCC.seq_len` = 9, $k=3$ | verified |
| 76 | Spearman under perfect agreement | 0.71 | `FORM` = 0.708 | verified (C4) |
| 99, 204 | tie-filter level | $\alpha = 0.05$ | `OCC.filter_alpha`; `configs/base.yaml` | verified |
| 122 | library size | 30,483 | `len(LIB)` = 30,483 | verified |
| 123 | subsample per seed | $n = 4000$ | `OCC.n` = 4000 | verified |
| 124 | split | 3000 / 1000 | `OCC.n_fit`, `n_heldout` | verified |
| 137 | top-$k$ | $k = 3$ | design constant | verified |
| 179–180 | grid | {16,32,64,128} × {1,2,3} | `OCC.cells`, 12 cells | verified |
| 181 | models per cell | $K = 20$ | `OCC.n_models` = 20 | verified |
| **183** | **"All reported quantities use at least 5 seeds"** | **≥5** | `DUMP.n_seeds` = 3 | **mismatch (C1)** |
| 192 | divergence range | 43%–94% | `OCC` min/max | verified |
| 192–193 | measurements to separate | fewer than three | `OCC` max `n_separate_median` = 2.11 | verified |
| 206 | filter resolution power | 80% | `SEP.power` = 0.8 | verified |
| 207 | held-out size | 1000 | `OCC.n_heldout` | verified |
| 207 | pairs across grid | 2280 | Σ `n_pairs_total` = 2280 | verified |
| 207 | tied across grid | 1610 | Σ `n_pairs_indistinguishable` = 1610 | verified |
| 212 | strictest cap | $\|\Delta R^2\| < 0.001$ | Table 1 row 4 | verified |
| 212–213 | discarded / divergence | 17 of 18 · 43% | `OCC` seed files = 0.943 · 0.433 | verified |
| 248 | best cell still diverges | 43% | `SIDE fig1` = 0.43 | verified |

### Section 3 — observed outcome

| line | claim as written | value | should come from | verdict |
|---|---|---|---|---|
| 277, 516 | Spearman($R^2$, min $\rho$) | +0.650 | `OCC` recomputed = +0.650 | verified |
| 279 | best-cell held-out $R^2$ | 0.637 | `OCC.128x1.heldout_r2_mean` = 0.6372 | verified |
| 280 | tied pairs | 154 of 190 | `OCC.128x1` = 154 / 190 | verified |
| 281 | $\rho$ below 0.9 / 0.8 / 0.7 | 37% / 15% / 6% | `pooled_frac_below_9/8/7` | verified |
| 282 | lowest $\rho$ | +0.350 | `OCC.128x1.pooled_rho_min` | verified |
| 285 | top-3 differs, 128×2 / 128×3 | 73% / 90% | `1-pooled_tk_ex3` = 0.73 / 0.905 | verified (C3) |
| 287 | conditioned | 71% / 80% | `1-pooled_tk_cex3` | verified |
| 288 | well-defined top-3, 128×3 | 11% | `pooled_tk_cfrac` = 0.11 | verified |
| 289 | well-defined top-3, 64×3 | 0.3% | `pooled_tk_cfrac` = 0.0025 | verified |
| 298 | top-1 differs, 128×1 / 128×3 | 18% / 68% | `1-pooled_tk_ex1` | verified |
| 299 | top-2 differs, 128×1 | 29% | `OCC` seed files, per-seed median | verified |
| 300 | pooled observations | 203,200 | `SIDE fig2.panel_b.n_points` | verified (C1, C5) |
| 301 | set disagreement | 44.2% | `DUMP` recompute = 44.2 | verified |
| 301 | leader agrees, ranks 2–3 differ | 35.0 | `DUMP` recompute = 35.0 | verified |
| 304 | at depth 3 | 63.9 of 91.0 | `DUMP` recompute = 63.9 / 91.0 | verified |
| 309–310 | twin figures | 46, 47, 34, 21 | `SEP.128x2/128x3`, exact and conditioned | verified |
| 341 | measurements to separate | median 0.3 to 2.1 | `OCC.n_separate_median.mean` = 0.30–2.11 | verified |
| 343 | pointwise difference / noise | 0.5 to 2.4 | `OCC` rms_pred_diff ÷ `SEP.median_noise_sd` = 0.54–2.42 | verified |
| 346 | twin requirement | $8.8 \times 10^7$ | `SEP.128x3.n_separate` = 87,751,464 | verified |
| 357 | library reference | 30,483 | `len(LIB)` | verified |
| 379 | filter cannot resolve | 0.032 | `OCC.128x1.pooled_mde_r2` = 0.0318 | verified |

### Section 4 — causes

| line | claim as written | value | should come from | verdict |
|---|---|---|---|---|
| 398 | top-3 mass, min / max | 54% / 98% | `pooled_conc_top3` 64×3 = 0.54, 128×1 = 0.9758 | verified |
| 399 | grid median top-3 mass | 86% | median over cells = 0.86 | verified |
| 399 | effective positions range | 7.80 to 2.20 | `pooled_eff_pos` | verified |
| 400, 802 | grid median effective positions | 3.92 | median over cells = 3.92 | verified |
| 402 | Fig 2a top-3 mass | 97% | `SIDE fig2.panel_a.median_top3_mass` = 0.9726 | verified (C2) |
| 402 | strongest position mass | 0.71 | `SIDE fig2.panel_a.median_by_rank[0]` = 0.7086 | verified (C4) |
| 405, 803 | twin grid median | 82% and 4.38 | `SEP conc_top3_ref` = 0.8241, `eff_pos_ref` = 4.3830 | verified |
| 413, 427, 690 | expected Spearman | 0.708 | `FORM` $1-210/720$ | verified |
| 422 | seeds behind Figure 2 | three | `DUMP.n_seeds` = 3 | verified (C1) |
| 423 | sequence profiles | 24,000 | `SIDE fig2.panel_a.n_sequence_profiles` | verified |
| 431 | highlighted observations | 8,861 | `SIDE fig2.panel_b.n_left_and_exact` | verified |
| 432 | share of all plotted | 4.4% | `SIDE` = 0.04361 | verified |
| 433 | share left of the line | 30% of 29,310 | `SIDE` = 0.3023, 29,310 | verified |
| 444 | exact top-3 agreement by depth | 26–57 / 12–27 / 6–10% | `pooled_tk_ex3` by depth | verified |
| 454–455 | slope and correlation | 0.933, $r = 0.990$ | `SEP.phenotype_slope` = 0.9331; `LIB` re-derived $r$ = 0.9900 | verified |
| 456, 467, 753 | ex exceeds tot | 4.9% of rows | `LIB` re-derived = 4.86% | verified |
| 457, 754 | largest ratio | 20.9× | `LIB` re-derived = 20.86 | verified |
| 469, 749 | weight tail | mean 3.0× median | recorded in appendix; `SEP` | verified |
| 489, 994 | initial loss | $6.7 \times 10^{-19}$ | `CH.results.json` | verified |
| 490, 998 | warm control returns | 1.000000 | `CS`, all 120 exactly 1.0 | verified |
| 494 | cold-start recovery, depth 3 | $\approx 0.45$ | `CS.arm1_cold_heldout` min = 0.4501 | verified |
| 510–511 | divergence by depth | 43–74 / 73–88 / 90–94% | `1-pooled_tk_ex3` by depth | verified (C3) |
| 512 | rank correlation with depth / width | +0.92 / −0.28 | recomputed = +0.917 / −0.281 | verified |
| 513 | within depth 1 | 74% to 43% | `1-pooled_tk_ex3` 16×1, 128×1 | verified |
| 520 | separation vs divergence | −0.94 | recomputed = −0.94 | verified |
| 526 | subsample of library | 4000 of 30,483 | `OCC.n`, `len(LIB)` | verified |

### Appendices in `neurips_2026.tex`

| line | claim as written | value | should come from | verdict |
|---|---|---|---|---|
| 713 | expected $\rho_9$ for $k$ = 2…5 | 0.533 / 0.708 / 0.833 / 0.917 | `FORM` | verified |
| 804 | effective positions per cell | 2.20 to 7.80 | `pooled_eff_pos` | verified |
| 910 | split generator seed | $1{,}000{,}003 + k$ | `experiments/closure_heldout.py:110` | verified |
| 910–911 | split sizes | 4000 → 3000 / 1000 | `OCC` | verified |
| 912 | seeds per cell | ten | `OCC.meta.seeds` = [0…9] | verified |
| 946–951 | float64, 5 sigmoids, $\mathcal{N}(0, 0.3^2)$, $K=20$ | — | `closure_capacity.py:116-136`, `identifiability_probe.py:139`, `configs/base.yaml` (`ge_components: 5`) | verified |
| 955 | ISM instances per pair | 400 | `configs/base.yaml ism_instances: 400` | verified |
| 960 | prereg single commit, dated | 2026-08-26 | `git log PREREGISTRATION.md` — one commit, 2026-08-26 | verified |
| 960 | earliest committed run directory | 2026-08-29 | earliest **tracked** dir = `20260829T092224Z` | verified |
| 999 | cell–seed combinations | 120 | 12 cells × 10 seeds | verified |
| 1002 | worst-cell cold start | 0.4501 | `CS.arm1_cold_heldout` min | verified |
| 1002–1003 | threshold, cells below | 0.999, 11 of 12 | `CS` = 11 of 12 | verified |
| 1003 | per-seed spread, 128×3 | 0.21 to 0.87 | `SIDE figA2` = 0.2134 / 0.8700 | verified |

### `paper/appendix.tex`

| line | claim as written | value | should come from | verdict |
|---|---|---|---|---|
| 14 | slope across seeds | 0.9331 [0.9314, 0.9345] | `SEP.phenotype_slope` | verified |
| 14 | correlation | $r = 0.990$ | hardcoded in `make_appendix.py:392`; `LIB` re-derived = 0.9900 | verified |
| 17 | ex exceeds tot | 4.9% | hardcoded; `LIB` = 4.86% | verified |
| 17 | largest ratio | 20.9 | hardcoded; `LIB` = 20.86 | verified |
| 17 | ex = 0 | 11.4% | hardcoded; `LIB` = 11.35% | verified |
| 26 | median noise scale | 0.3114 [0.3051, 0.3194] | `SEP.median_noise_sd` | verified |
| 29, 31 | weight tail | 3.0× | `SEP` | verified |
| 43 | twin requirement, 128×3 | $8.8 \times 10^7$ | `SEP.128x3.n_separate` | verified |
| 47, 91 | seeds | 10 | `SEP.meta`, `CS.meta` | verified |
| 71 | test level and power | 0.05, 0.8 | `SEP.alpha`, `SEP.power` | verified |
| 86, 115, 121 | initial loss / tolerance | $6.7\times10^{-19}$ / $10^{-10}$ | `CH.results.json` | verified |
| 115 | cold-start initial loss range | 1.826 to 2.093 | `CS.arm1_cold_init_loss` | verified |
| 121 | warm values exactly 1.0 | 120 | `SIDE figA2 warm_all_exactly_one` | verified |
| 121 | per-seed spread 128×3 | 0.21 to 0.87 | `SIDE figA2` | verified |
| 126 | ceiling by depth | 0.9969/0.9748, 0.7122/0.9236, 0.4501/0.9034 | `CS.arm1_cold_heldout`, `base_warp_heldout` | verified |
| 128 | threshold, cells below | 0.999, 11 of 12 | `CS` | verified |

---

## Notes on generator hardcoding

`make_appendix.py` reads the slope and the median noise scale from `SEP`, but
embeds `r = 0.990`, `4.9\%`, `20.9` and `11.4\%` as literal strings
(lines 392–400). All four re-derive correctly from `LIB` today, so they are
verified, but they will not update if the library changes. `make_tables.py`
hardcodes nothing.

## Previously corrected

Three prose numbers were mismatches until commit `2004942` and are verified
above in their corrected form: the depth/width rank correlations (were +0.80 /
−0.06), the 64×3 conditioned fraction (was "none"), and the pointwise-difference
upper bound (was 2.5).
