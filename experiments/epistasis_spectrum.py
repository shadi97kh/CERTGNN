"""Is the rank structure of the empirical epistasis matrix visibly different across landscapes?

RETIRED. The verdict this produced is RETRACTED -- see `verdict_text` below and
`paper/prior_art/spectral_discriminator.md`. The direction is closed by Husain &
Murugan (Mol Biol Evol 37:2865, 2020), who state the rank-one result, use
deviation from it as their discriminating statistic, and apply SVD to the same
GB1 epistasis matrix. The measurement code is kept so the numbers remain
reproducible; it should not be used to support a claim about mechanism, and two
of its own choices are known to be wrong (the eqFP611-red reading and the
double-centering justification, both detailed in the retraction).

This is a MEASUREMENT, not a test. Husain & Murugan (Mol Biol Evol 37:2865,
2020) already established the theory and already applied a low-rank
decomposition to GB1 -- see `paper/prior_art/spectral_discriminator.md`, which
closes the discriminator direction. What is asked here is narrower and still
worth knowing: put three landscapes side by side and see whether their rank
structure differs at all, or whether they all look the same.

For each landscape the pairwise epistasis matrix over the single-mutant index
set is

    E_ij = y_ij - y_i - y_j + y_0

and the following are reported: the full eigenvalue spectrum sorted by |lambda|,
an effective rank, the variance explained by the leading rank-one term, the
alignment of the leading eigenvector with the additive coefficients of a fitted
global-epistasis model, and a noise floor.

**Two things about this construction that change how the numbers read.**

*The noise in E is not iid, and its structured part is low rank.* Writing the
measurement errors as e_ij, e_i, e_j, the observed matrix carries

    E_obs = E_true + e_ij - e_i - e_j  =  E_true + e_ij - (e 1^T + 1 e^T)

and `e 1^T + 1 e^T` is symmetric of rank at most 2. Single-mutant measurement
error therefore injects a rank-2 component whose eigenvalues scale with n, and
that component would masquerade as exactly the low-rank structure under test.
Double-centering E removes any additive row/column term, kills this artifact,
and leaves a rank-one signal rank one -- centering `beta beta^T` gives
`(beta - mean)(beta - mean)^T`. Both raw and double-centered spectra are
reported for that reason, and the double-centered one is the one to read.

*The random-matrix null here is Wigner, not Marchenko-Pastur.* MP describes the
eigenvalues of a sample covariance matrix built from many observations of a
random vector. E is not a covariance: it is a symmetric matrix whose entries are
themselves the measurements. The matching null is the Wigner semicircle, whose
bulk edge for iid entries of standard deviation s is `2 s sqrt(n)`. That edge is
what gets marked. Where the landscape's own replicates or counts give s, they
are used; where they do not, the fallback is stated in the output rather than
assumed.

The additive-plus-global-epistasis fit that supplies `beta` is implemented here
rather than delegated: neither MAVE-NN nor MoCHI is installed in this
environment, and the model needed -- a purely additive latent passed through a
learned monotone nonlinearity -- is the additive+GE model those packages fit,
and is already available in this codebase as `MonotoneGE`. The substitution is
recorded in the output.

Usage:
    python -m experiments.epistasis_spectrum --config configs/base.yaml
    python -m experiments.epistasis_spectrum --config configs/base.yaml --only fas
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from experiments._common import (
    configure_torch,
    load_config,
    make_run_dir,
    to_jsonable,
)
from experiments.identifiability_probe import MonotoneGE

FAS_DIR = pathlib.Path("data/raw/fas_supp")
GB1_XLSX = pathlib.Path("data/raw/gb1_olson2014_mmc2.xlsx")
EQFP_XLSX = pathlib.Path("data/raw/poelwijk_supp/41467_2019_12130_MOESM7_ESM.xlsx")


class Landscape:
    """One epistasis matrix plus everything needed to interpret its spectrum."""

    def __init__(
        self,
        name: str,
        E: np.ndarray,
        labels: list[str],
        observed: np.ndarray,
        noise_sd: float,
        noise_source: str,
        design: np.ndarray,
        phenotype: np.ndarray,
        n_missing: int,
        n_possible: int,
        note: str = "",
    ) -> None:
        self.name = name
        self.E = E
        self.labels = labels
        self.observed = observed  # boolean mask of measured (i, j)
        self.noise_sd = noise_sd
        self.noise_source = noise_source
        self.design = design  # [n_genotypes, n_singles] additive design
        self.phenotype = phenotype  # [n_genotypes]
        self.n_missing = n_missing
        self.n_possible = n_possible
        self.note = note


def _pad_indices(groups: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length mutation index lists into a padded array + mask."""
    width = max((len(g) for g in groups), default=1)
    idx = np.zeros((len(groups), width), dtype=np.int64)
    msk = np.zeros((len(groups), width), dtype=float)
    for r, g in enumerate(groups):
        for c, v in enumerate(g):
            idx[r, c] = v
            msk[r, c] = 1.0
    return idx, msk


def load_fas() -> Landscape:
    """FAS/CD95 exon 6, Julien et al. 2016. 189 singles, 16,728 doubles.

    Enrichment scores are already relative to wild type, so y_0 = 0 and the
    supplementary table's `EmpiricalEpistasis` column is exactly
    y_ij - y_i - y_j. Three replicate scores per single mutant give the noise
    scale directly.
    """
    import pandas as pd

    singles = pd.read_excel(FAS_DIR / "ncomms11558-s2.xlsx")
    pairs = pd.read_excel(FAS_DIR / "ncomms11558-s4.xlsx")

    labels = list(singles["ID"].astype(str))
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    E = np.zeros((n, n), dtype=float)
    obs = np.zeros((n, n), dtype=bool)
    used = 0
    for a, b, e in zip(
        pairs["IDA"].astype(str),
        pairs["IDB"].astype(str),
        pairs["EmpiricalEpistasis"].to_numpy(dtype=float),
    ):
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None or not np.isfinite(e):
            continue
        E[i, j] = E[j, i] = e
        obs[i, j] = obs[j, i] = True
        used += 1

    # Same-position pairs cannot exist: 63 positions x C(3,2) alternatives.
    pos = np.array([int(lab.split("-")[0]) for lab in labels])
    same_pos = pos[:, None] == pos[None, :]
    n_possible = int((~same_pos).sum() // 2)
    n_missing = n_possible - used

    # Noise: SD across the three replicate enrichment scores, per single.
    reps = singles[[c for c in singles.columns if c.startswith("Replicate_")]]
    per_meas_sd = float(np.nanmedian(reps.to_numpy(dtype=float).std(axis=1, ddof=1)))
    # E = y_ij - y_i - y_j with y_0 identically zero: three noisy terms.
    noise_sd = per_meas_sd * np.sqrt(3.0)

    # Additive design over singles, for the additive+GE fit.
    rows, ys = [], []
    for lab, y in zip(labels, singles["EnrichmentScore"].to_numpy(dtype=float)):
        rows.append([idx[lab]])
        ys.append(y)
    for a, b, y in zip(
        pairs["IDA"].astype(str),
        pairs["IDB"].astype(str),
        pairs["ABscore"].to_numpy(dtype=float),
    ):
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None or not np.isfinite(y):
            continue
        rows.append([i, j])
        ys.append(y)

    return Landscape(
        "FAS exon 6",
        E,
        labels,
        obs,
        noise_sd,
        f"SD across 3 replicate enrichment scores (median {per_meas_sd:.4f}), "
        f"x sqrt(3) for the three terms in E",
        _pad_indices(rows),
        np.asarray(ys),
        n_missing,
        n_possible,
        "Enrichment scores are relative to wild type, so y_0 = 0 exactly.",
    )


def load_gb1(min_input_count: int) -> Landscape:
    """GB1 domain, Olson et al. 2014. Singles and doubles from raw counts.

    Fitness is the selection/input ratio relative to wild type; the phenotype
    used is its log, so that wild type is 0 and E is a log-scale interaction.
    Counts give a Poisson noise scale.
    """
    import pandas as pd

    df = pd.read_excel(GB1_XLSX, sheet_name="DoubleSub.xls", header=2)
    df.columns = [str(c).strip() for c in df.columns]
    wt_in = float(df["Input Count.2"].dropna().iloc[0])
    wt_sel = float(df["Selection Count.2"].dropna().iloc[0])
    wt_ratio = wt_sel / wt_in

    sing = df[
        ["WT amino acid", "Position", "Mutation", "Input Count.1", "Selection Count.1"]
    ].dropna()
    sing = sing[sing["Input Count.1"] >= min_input_count]
    labels = [f"{int(p)}{m}" for p, m in zip(sing["Position"], sing["Mutation"])]
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    s_in = sing["Input Count.1"].to_numpy(dtype=float)
    s_sel = sing["Selection Count.1"].to_numpy(dtype=float)
    y_single = np.log((s_sel / s_in) / wt_ratio + 1e-12)

    dbl = df.dropna(subset=["Mut1 Position", "Mut2 Position"])
    dbl = dbl[dbl["Input Count"] >= min_input_count]
    d_in = dbl["Input Count"].to_numpy(dtype=float)
    d_sel = dbl["Selection Count"].to_numpy(dtype=float)
    y_double = np.log((d_sel / d_in) / wt_ratio + 1e-12)
    l1 = [f"{int(p)}{m}" for p, m in zip(dbl["Mut1 Position"], dbl["Mut1 Mutation"])]
    l2 = [f"{int(p)}{m}" for p, m in zip(dbl["Mut2 Position"], dbl["Mut2 Mutation"])]

    E = np.zeros((n, n), dtype=float)
    obs = np.zeros((n, n), dtype=bool)
    rows, ys = [], []
    for lab, y in zip(labels, y_single):
        rows.append([idx[lab]])
        ys.append(y)
    used = 0
    for a, b, y in zip(l1, l2, y_double):
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None or not np.isfinite(y):
            continue
        E[i, j] = E[j, i] = y - y_single[i] - y_single[j]
        obs[i, j] = obs[j, i] = True
        rows.append([i, j])
        ys.append(y)
        used += 1

    pos = np.array([int(lab[:-1]) for lab in labels])
    same_pos = pos[:, None] == pos[None, :]
    n_possible = int((~same_pos).sum() // 2)

    # Poisson counting noise on a log ratio: var(log) ~ 1/sel + 1/input.
    sd_single = float(np.median(np.sqrt(1.0 / np.maximum(s_sel, 1) + 1.0 / s_in)))
    sd_double = float(np.median(np.sqrt(1.0 / np.maximum(d_sel, 1) + 1.0 / d_in)))
    noise_sd = float(np.sqrt(sd_double**2 + 2.0 * sd_single**2))

    return Landscape(
        "GB1",
        E,
        labels,
        obs,
        noise_sd,
        f"Poisson counting noise on the log selection ratio: median SD "
        f"{sd_double:.4f} for doubles and {sd_single:.4f} for singles, combined "
        "as sqrt(sd_d^2 + 2 sd_s^2)",
        _pad_indices(rows),
        np.asarray(ys),
        n_possible - used,
        n_possible,
        f"Fitness from raw counts relative to wild type; input count >= "
        f"{min_input_count}.",
    )


def load_eqfp611(color: str) -> Landscape:
    """eqFP611, Poelwijk et al. 2019. Combinatorially complete 2^13.

    The single mutants are the 13 one-bit genotypes, so E is 13 x 13. Small,
    and deliberately included on those terms.
    """
    import pandas as pd

    raw = pd.read_excel(EQFP_XLSX, sheet_name="genodata", header=None, skiprows=2)
    bits = [str(b).strip().strip("'") for b in raw[0]]
    col = {"red": 6, "blue": 7, "combined": 9}[color]
    y = raw[col].to_numpy(dtype=float)
    keep = np.array([len(b) == 13 and set(b) <= {"0", "1"} for b in bits])
    bits = [b for b, k in zip(bits, keep) if k]
    y = y[keep]

    G = np.array([[int(c) for c in b] for b in bits], dtype=float)
    by_bits = {b: yy for b, yy in zip(bits, y)}
    wt = "0" * 13
    if wt not in by_bits:
        raise RuntimeError("wild-type genotype absent from eqFP611 table")
    y0 = by_bits[wt]

    n = 13
    labels = [f"m{i + 1}" for i in range(n)]
    y_single = np.full(n, np.nan)
    for i in range(n):
        b = ["0"] * n
        b[i] = "1"
        y_single[i] = by_bits.get("".join(b), np.nan) - y0

    E = np.zeros((n, n), dtype=float)
    obs = np.zeros((n, n), dtype=bool)
    used = 0
    for i in range(n):
        for j in range(i + 1, n):
            b = ["0"] * n
            b[i] = b[j] = "1"
            yij = by_bits.get("".join(b))
            if (
                yij is None
                or not np.isfinite(y_single[i])
                or not np.isfinite(y_single[j])
            ):
                continue
            E[i, j] = E[j, i] = (yij - y0) - y_single[i] - y_single[j]
            obs[i, j] = obs[j, i] = True
            used += 1
    n_possible = n * (n - 1) // 2

    # Every genotype enters the additive+GE fit, not just singles and doubles.
    design = _pad_indices([list(np.flatnonzero(row)) for row in G.astype(int)])
    phen = y - y0

    # Counts are available; use them for a Poisson scale on the brightness.
    cnt_col = {"red": 3, "blue": 4, "combined": 3}[color]
    counts = raw[cnt_col].to_numpy(dtype=float)[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    per_meas = float(np.nanmedian(rel) * np.nanstd(y))
    noise_sd = per_meas * 2.0

    return Landscape(
        f"eqFP611 ({color})",
        E,
        labels,
        obs,
        noise_sd,
        f"Poisson counting on {color} counts, median relative SD "
        f"{float(np.nanmedian(rel)):.4f} scaled by the brightness spread, "
        "x2 for the four terms in E",
        design,
        phen,
        n_possible - used,
        n_possible,
        "13 single mutants, so E is 13x13: far too small for an asymptotic "
        "random-matrix edge, included for the rank comparison only.",
    )


def double_center(M: np.ndarray) -> np.ndarray:
    """Remove additive row and column effects.

    Kills the rank-2 artifact that single-mutant measurement error injects,
    and maps a rank-one signal to a rank-one signal.
    """
    r = M.mean(axis=1, keepdims=True)
    c = M.mean(axis=0, keepdims=True)
    return M - r - c + M.mean()


def spectrum_stats(M: np.ndarray, noise_sd: float) -> dict[str, Any]:
    w, V = np.linalg.eigh(M)
    order = np.argsort(-np.abs(w))
    w, V = w[order], V[:, order]
    aw = np.abs(w)
    total = float(aw.sum())
    l2 = float((w**2).sum())
    pr = float(total**2 / l2) if l2 > 0 else float("nan")
    csum = np.cumsum(aw) / (total if total > 0 else 1.0)
    n90 = int(np.searchsorted(csum, 0.90) + 1)
    n = M.shape[0]
    rank1 = np.outer(V[:, 0], V[:, 0]) * w[0]
    denom = float((M**2).sum())
    var1 = float(1.0 - ((M - rank1) ** 2).sum() / denom) if denom > 0 else float("nan")
    # Wigner semicircle edge for a symmetric matrix with iid entries.
    edge = 2.0 * noise_sd * np.sqrt(n)
    return {
        "n": n,
        "eigenvalues": w.tolist(),
        "top_vector": V[:, 0].tolist(),
        "participation_ratio": pr,
        "n_for_90pct_mass": n90,
        "rank1_variance_explained": var1,
        "abs_lambda_1": float(aw[0]),
        "abs_lambda_2": float(aw[1]) if n > 1 else float("nan"),
        "lambda1_over_lambda2": float(aw[0] / aw[1])
        if n > 1 and aw[1] > 0
        else float("nan"),
        "wigner_edge": float(edge),
        "n_outside_bulk": int((aw > edge).sum()),
        "frac_mass_outside_bulk": float(aw[aw > edge].sum() / total)
        if total > 0
        else float("nan"),
    }


def null_reference(
    mask: np.ndarray, noise_sd: float, reps: int, seed: int
) -> dict[str, Any]:
    """Size-matched noise baseline for the same statistics.

    This is not optional garnish. The rank-one fraction of a PURE NOISE
    symmetric matrix scales as roughly 4/n: about 0.31 at n=13, 0.021 at n=189
    and 0.004 at n=1045. Comparing the raw fraction across landscapes of
    different size therefore measures matrix size more than it measures
    structure, and does so in the direction that flatters the smallest
    landscape. Each null draw uses this landscape's own noise scale and its own
    observed-pair mask, so the structural zeros and the missing pairs enter the
    null exactly as they enter the data, and goes through the same
    double-centering.
    """
    n = mask.shape[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    keep = mask[iu]
    r1, pr, ratio = [], [], []
    for _ in range(max(reps, 2)):
        A = np.zeros((n, n))
        A[iu] = rng.normal(0.0, noise_sd, size=keep.size) * keep
        A = A + A.T
        st = spectrum_stats(double_center(A), noise_sd)
        r1.append(st["rank1_variance_explained"])
        pr.append(st["participation_ratio"])
        ratio.append(st["lambda1_over_lambda2"])

    def q(v: list[float]) -> tuple[float, float, float]:
        return (
            float(np.mean(v)),
            float(np.percentile(v, 2.5)),
            float(np.percentile(v, 97.5)),
        )

    m1, lo1, hi1 = q(r1)
    mp, lop, hip = q(pr)
    mr, lor, hir = q(ratio)
    return {
        "reps": len(r1),
        "rank1_mean": m1,
        "rank1_lo": lo1,
        "rank1_hi": hi1,
        "pr_mean": mp,
        "pr_lo": lop,
        "pr_hi": hip,
        "l1l2_mean": mr,
        "l1l2_lo": lor,
        "l1l2_hi": hir,
    }


def fit_additive_ge(
    idx: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    n_beta: int,
    epochs: int,
    lr: float,
    k: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Additive latent through a learned monotone nonlinearity.

    This is the additive+GE model of MAVE-NN / MoCHI, neither of which is
    installed here. phi = sum of beta over the mutations a genotype carries,
    y ~ g(phi) with g monotone.

    The latent is formed by INDEXING rather than by a dense design matmul: GB1
    has 531,782 genotypes over 1,045 single mutants, and the dense design would
    be 4.4 GB of float64 to hold at most two non-zeros per row.
    """
    torch.manual_seed(seed)
    ii = torch.as_tensor(idx, dtype=torch.long)
    M = torch.as_tensor(mask, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.float64)
    beta = torch.zeros(n_beta, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        beta += 0.01 * torch.randn(n_beta, dtype=torch.float64)
    g = MonotoneGE(k).double()
    opt = torch.optim.Adam([beta, *g.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        phi = (beta[ii] * M).sum(dim=1)
        loss = ((g(phi) - t) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = g((beta[ii] * M).sum(dim=1))
        ss = float(((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return beta.detach().numpy(), float(1.0 - ss)


def analyse(ls: Landscape, cfg: Any) -> dict[str, Any]:
    es = cfg.epistasis_spectrum
    raw = spectrum_stats(ls.E, ls.noise_sd)
    cen = spectrum_stats(double_center(ls.E), ls.noise_sd)
    n = ls.E.shape[0]
    reps = int(es.null_reps_large if n > 500 else es.null_reps_small)
    null = null_reference(ls.observed, ls.noise_sd, reps, 12345)
    beta, fit_r2 = fit_additive_ge(
        ls.design[0],
        ls.design[1],
        ls.phenotype,
        len(ls.labels),
        int(es.ge_epochs),
        float(es.ge_lr),
        int(es.ge_components),
        int(cfg.seed) if cfg.get("seed") is not None else 0,
    )
    v1 = np.asarray(cen["top_vector"])
    ok = np.isfinite(beta) & np.isfinite(v1)
    pr_ = pearsonr(np.abs(beta[ok]), np.abs(v1[ok]))
    sp_ = spearmanr(np.abs(beta[ok]), np.abs(v1[ok]))
    # Sign-free alignment: |<v1, beta_hat>| for unit vectors.
    bn = beta[ok] / (np.linalg.norm(beta[ok]) + 1e-30)
    vn = v1[ok] / (np.linalg.norm(v1[ok]) + 1e-30)
    align = float(abs(float(bn @ vn)))
    return {
        "name": ls.name,
        "n_singles": len(ls.labels),
        "n_pairs_observed": int(ls.observed.sum() // 2),
        "n_pairs_possible": ls.n_possible,
        "n_pairs_missing": ls.n_missing,
        "frac_missing": ls.n_missing / max(ls.n_possible, 1),
        "noise_sd": ls.noise_sd,
        "noise_source": ls.noise_source,
        "note": ls.note,
        "raw": raw,
        "centered": cen,
        "null": null,
        "rank1_excess": cen["rank1_variance_explained"] - null["rank1_mean"],
        "rank1_ratio_to_null": (
            cen["rank1_variance_explained"] / null["rank1_mean"]
            if null["rank1_mean"] > 0
            else float("nan")
        ),
        "ge_fit_r2": fit_r2,
        "beta_v1_alignment": align,
        "beta_v1_pearson_abs": float(pr_.statistic),
        "beta_v1_spearman_abs": float(sp_.statistic),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--only", default=None, help="fas | gb1 | eqfp")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    configure_torch(cfg)
    run = make_run_dir(cfg, "epistasis_spectrum", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    print(f"run dir: {run}")

    want = args.only
    results = []
    loaders = [
        ("fas", lambda: load_fas()),
        ("gb1", lambda: load_gb1(int(cfg.epistasis_spectrum.gb1_min_input))),
        ("eqfp", lambda: load_eqfp611("red")),
        ("eqfp", lambda: load_eqfp611("blue")),
    ]
    for key, fn in loaders:
        if want and key != want:
            continue
        ls = fn()
        print(
            f"loaded {ls.name}: {len(ls.labels)} singles, "
            f"{int(ls.observed.sum() // 2)} pairs",
            flush=True,
        )
        results.append(analyse(ls, cfg))
        print(f"  done {ls.name}", flush=True)

    L = ["# Epistasis spectra of three landscapes, side by side\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}, "
        f"config `{meta['config_hash']}`. This is a measurement, not a test: the "
        "rank-one discriminator itself is prior art (Husain & Murugan 2020, see "
        "`paper/prior_art/spectral_discriminator.md`). The question here is only "
        "whether rank structure differs across landscapes at all.\n"
    )
    L.append(
        "**Read the double-centered rows.** Measurement error on the single "
        "mutants enters E as `-(e 1ᵀ + 1 eᵀ)`, which is symmetric of rank 2 and "
        "would imitate the low-rank structure under test. Double-centering removes "
        "any additive row/column term and maps a rank-one signal to a rank-one "
        "signal, so it removes that artifact without removing the effect.\n"
    )
    L.append(
        "**The null is Wigner, not Marchenko–Pastur.** MP describes a sample "
        "covariance matrix; E is not a covariance but a symmetric matrix of direct "
        "measurements, so the matching bulk edge is `2·s·√n`.\n"
    )
    L.append(
        "| landscape | singles | pairs (obs/possible) | λ₁/λ₂ | participation ratio | "
        "eigs for 90% mass | rank-1 var. expl. vs size-matched null | outside bulk |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r["centered"]
        L.append(
            f"| {r['name']} | {r['n_singles']} | "
            f"{r['n_pairs_observed']:,}/{r['n_pairs_possible']:,} "
            f"({r['frac_missing']:.0%} missing) | "
            f"**{c['lambda1_over_lambda2']:.2f}** | "
            f"**{c['participation_ratio']:.1f}** | "
            f"**{c['n_for_90pct_mass']}** of {c['n']} | "
            f"**{c['rank1_variance_explained']:.3f}** vs null "
            f"{r['null']['rank1_mean']:.3f} (**{r['rank1_ratio_to_null']:.1f}x**) | "
            f"{c['n_outside_bulk']} eigs, {c['frac_mass_outside_bulk']:.0%} of mass |"
        )
    L.append("")
    L.append(
        "| landscape | raw λ₁/λ₂ | raw part. ratio | raw rank-1 | "
        "‖v₁ vs β‖ alignment | GE fit R² | noise SD | Wigner edge |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        w, c = r["raw"], r["centered"]
        L.append(
            f"| {r['name']} | {w['lambda1_over_lambda2']:.2f} | "
            f"{w['participation_ratio']:.1f} | {w['rank1_variance_explained']:.3f} | "
            f"**{r['beta_v1_alignment']:.3f}** | {r['ge_fit_r2']:.3f} | "
            f"{r['noise_sd']:.4f} | {c['wigner_edge']:.3f} |"
        )
    L.append("")
    L.append(
        "**The null column is what makes the three comparable.** A pure-noise "
        "symmetric matrix has a rank-one fraction of roughly 4/n — about 0.31 at "
        "n=13, 0.021 at n=189, 0.004 at n=1045 — so the raw fraction largely "
        "measures matrix size, and does so in the direction that flatters the "
        "smallest landscape. Each null uses that landscape's own noise scale and "
        "its own observed-pair mask, and passes through the same double-centering.\n"
    )
    L.append(
        "`‖v₁ vs β‖` is |⟨v̂₁, β̂⟩| for unit vectors, on the double-centered "
        "spectrum. Under the global-epistasis reading v₁ must be proportional to "
        "β, so a high rank-one fraction with a LOW alignment would refute that "
        "reading. β comes from an additive-latent-plus-monotone-nonlinearity fit "
        "implemented here, because neither MAVE-NN nor MoCHI is installed in this "
        "environment; it is the same model those packages fit.\n"
    )
    for r in results:
        L.append(f"- **{r['name']}** — noise: {r['noise_source']}. {r['note']}")
    L.append("")
    L.append("## Verdict\n")
    L.append(verdict_text(results, cfg))
    table = "\n".join(L) + "\n"

    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "epistasis_spectrum.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(json.dumps(to_jsonable(results), indent=1))
    print("\n" + table)
    return 0


def verdict_text(results: list[dict[str, Any]], cfg: Any) -> str:
    """The verdict is RETRACTED; emit the retraction, not a fresh conclusion.

    The mechanism claim this experiment was built to support is closed by prior
    art (Husain & Murugan, Mol Biol Evol 37:2865, 2020, Equation 2), and the
    verdict this function used to compute contained two defects: it called
    eqFP611 (red) structured when that channel's whole spectrum sits inside its
    own noise floor, and it justified double-centering as removing the artifact
    "without removing the effect" when centering discards the mean effect, which
    in DMS typically exceeds the standard deviation.

    The measurement code above is deliberately unchanged so the numbers stay
    reproducible. Only the conclusion is withdrawn, and it is withdrawn here
    rather than only in the checked-in table so that re-running this experiment
    regenerates the retraction instead of a fresh, unretracted claim.
    """
    del results, cfg
    return (
        "**RETRACTED.** The measurements above stand; the conclusion drawn from "
        "them does not, and no claim in this table should be cited. The rank-one "
        "result is already published (Husain & Murugan, *Mol Biol Evol* "
        "37(10):2865, 2020, Eq. 2), who also use deviation from rank-one as their "
        "discriminating statistic and apply SVD to the same GB1 epistasis matrix. "
        "The withdrawn verdict additionally (i) called eqFP611 (red) substantially "
        "rank-one with a leading eigenvector unaligned to β, when that channel has "
        "zero eigenvalues outside its own noise bulk so both numbers are noise, "
        "and (ii) justified double-centering as removing the rank-2 artifact "
        "without removing the effect, when centering discards the mean effect and "
        "so degrades the SNR of the quantity under test. See "
        "`paper/prior_art/spectral_discriminator.md`."
    )


def _retired_verdict_text(results: list[dict[str, Any]], cfg: Any) -> str:
    if not results:
        return "No landscape loaded."
    hi = float(cfg.epistasis_spectrum.rank_one_high)
    lo = float(cfg.epistasis_spectrum.rank_one_low)
    r1 = {r["name"]: r["centered"]["rank1_variance_explained"] for r in results}
    rel = {r["name"]: r["rank1_ratio_to_null"] for r in results}
    nul = {r["name"]: r["null"]["rank1_mean"] for r in results}
    pr = {r["name"]: r["centered"]["participation_ratio"] for r in results}
    al = {r["name"]: r["beta_v1_alignment"] for r in results}
    parts: list[str] = []

    allhi = min(r1.values()) >= hi
    alllo = max(r1.values()) <= lo
    spread = max(r1.values()) - min(r1.values())
    if allhi:
        parts.append(
            f"**Every landscape looks rank-one.** The leading term explains "
            f"{min(r1.values()):.3f} to {max(r1.values()):.3f} of the "
            "double-centered epistasis matrix in all of them, and the effective "
            f"rank runs {min(pr.values()):.1f} to {max(pr.values()):.1f}. There is "
            "no contrast to exploit: rank structure does not distinguish these "
            "landscapes."
        )
    elif alllo:
        parts.append(
            f"**No landscape looks rank-one.** The leading term explains at most "
            f"{max(r1.values()):.3f}, and effective ranks run "
            f"{min(pr.values()):.1f} to {max(pr.values()):.1f}. Whatever structure "
            "global epistasis imposes on these matrices, it does not show up as a "
            "dominant single eigenvalue at the noise level these data carry."
        )
    else:
        best = max(r1, key=lambda k: r1[k])
        worst = min(r1, key=lambda k: r1[k])
        parts.append(
            f"**Rank structure does differ across landscapes.** The leading term "
            f"explains {r1[best]:.3f} of the double-centered matrix in {best} and "
            f"only {r1[worst]:.3f} in {worst}, a spread of {spread:.3f}, with "
            f"effective rank {pr[best]:.1f} against {pr[worst]:.1f}. That is a real "
            "contrast, not a constant."
        )

    lowal = sorted(k for k in al if al[k] < 0.5 and r1[k] >= lo)
    parts.append(
        "**Against a size-matched null the ordering is not what the raw numbers "
        "suggest.** Pure noise alone would give a rank-one fraction of "
        + ", ".join(f"{k} {nul[k]:.3f}" for k in nul)
        + ", purely because that fraction scales as 4/n. Relative to its own null "
        "each landscape sits at "
        + ", ".join(f"{k} {rel[k]:.1f}x" for k in rel)
        + ". Reading the raw column across landscapes of different size would "
        "invert this comparison, which is why it is not the column to read."
    )
    parts.append(
        "**The control on the leading eigenvector.** Under the global-epistasis "
        "reading v₁ must be parallel to β. Alignment is "
        + ", ".join(f"{k} {al[k]:.3f}" for k in al)
        + "."
        + (
            " **"
            + ", ".join(lowal)
            + " show substantial rank-one structure whose leading eigenvector is "
            "NOT aligned with β**, so for those the low-rank structure is not "
            "explained by a monotone nonlinearity on an additive trait, whatever "
            "else produces it."
            if lowal
            else " Where rank-one structure is present it is aligned with β, which "
            "is what the global-epistasis reading requires."
        )
    )

    parts.append(
        "This is a description of three matrices, not a test of a mechanism, and it "
        "does not become one: the rank-one criterion and its application to GB1 are "
        "already published (Husain & Murugan, Mol Biol Evol 37:2865, 2020). Nothing "
        "here should be written up as a new discriminator."
    )
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
