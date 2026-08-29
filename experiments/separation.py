"""How much data separates a fit from its reparameterized twin, and what breaks first?

The identifiability question has been asked as a binary -- is the class closed,
does it contain the twin -- and answered with numbers like "held-out R2
0.995925" that a binary cannot carry. The strong claim was that the twin
predicts IDENTICALLY, so that no sample size separates the two fits. Held-out
agreement of 0.9959 refutes that, because 0.9959 is not 1. But calling it a
collapse oversells it in the other direction: 0.9959 is a long way from the
0.000 floor, and the fits are very nearly the same function.

The honest statement is quantitative, and it is a triple, reported per cell:

  X  how well the twin agrees with the original out of sample (held-out R2);
  Y  how differently the two fits rank the loci, by Spearman over per-position
     attribution magnitude -- the thing a biologist would actually read off;
  Z  how many held-out measurements are needed to reject the hypothesis that
     the two fits are the same function.

This triple is true regardless of how the containment question in
`closure_search.md` resolves, because it does not depend on whether some member
of the class equals psi(phi_hat) exactly. It describes the two fits actually
obtained.

**Where the noise comes from.** Z requires an observation-noise scale, and this
library supplies one rather than needing an assumption. Each sequence carries
two count pools, `ex_ct` and `tot_ct`, and the phenotype is affine in log10 of
their ratio: fitting y on log10(ex/tot) over the full 30,483-sequence library
gives slope 0.933 with r = 0.990. These are NOT a proportion -- ex_ct exceeds
tot_ct in 4.9% of rows -- so both are treated as Poisson and the delta method
applied to the log ratio,

    sd(y_i) = |a| * sqrt(1/ex_i + 1/tot_i) / ln(10)

with a half-count continuity correction. This counts sequencing noise ONLY. Real noise also includes library preparation and
biological variation, so this is a LOWER bound on the true noise, and therefore
Z is a lower bound on the measurements required. "At least Z" is the claim.

**The test behind Z.** Two candidate functions differing by d_i on point i,
under Gaussian noise of scale sd_i, are two simple hypotheses. The optimal test
statistic separates them by sqrt(sum_i d_i^2 / sd_i^2) standard deviations, so
rejecting at two-sided significance alpha with power 1 - beta needs

    n >= (z_{1-alpha/2} + z_{1-beta})^2 / mean_i(d_i^2 / sd_i^2)

evaluated on held-out points, where d is the difference between the reference's
prediction and its twin's.

Usage:
    python -m experiments.separation --config configs/base.yaml
    python -m experiments.separation --retable results/runs/<dir>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import norm, spearmanr

from experiments._common import (
    configure_torch,
    git_sha,
    load_config,
    make_run_dir,
    mean_ci,
    resolve_seeds,
    to_jsonable,
    write_tuning_budget,
)
from experiments.closure_capacity import (
    FAMILY,
    CapacityModel,
    load_mpsa_capped,
    n_params,
    pick_device,
    refit_to,
    select_lr,
    train,
)
from experiments.closure_heldout import split_indices
from experiments.identifiability_probe import (
    affine_r2,
    invert_warp,
    is_monotone,
    warp,
)

LN10 = math.log(10.0)


def load_counts(cfg: Any, seed: int) -> dict[str, np.ndarray]:
    """The read counts for exactly the rows `load_mpsa` sampled for this seed.

    Mirrors that function's dropna-then-sample so the counts line up row for row
    with the tensors it returned; there is no join key in the frame, so the
    sampling must be replicated rather than looked up.
    """
    import pandas as pd

    im = cfg.identifiability_mpsa
    n = int(cfg.closure_capacity.n_max)
    df = pd.read_csv(pathlib.Path(im.path)).dropna(subset=["x", "y"])
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return {
        "tot_ct": df.tot_ct.to_numpy().astype(float),
        "ex_ct": df.ex_ct.to_numpy().astype(float),
        "y_raw": df.y.to_numpy().astype(float),
    }


def noise_sd_standardized(counts: dict[str, np.ndarray], slope: float) -> np.ndarray:
    """Per-observation phenotype noise, in the standardized units the models see.

    The phenotype is affine in log10 of ex_ct/tot_ct. Those are two independent
    count pools, NOT a proportion: ex_ct exceeds tot_ct in 4.9% of rows, with
    ratios up to 20.9, so a binomial model is wrong for this library. Treating
    both as Poisson and applying the delta method to the log of their ratio,

        var(log10 R) = (1/ex + 1/tot) / ln(10)^2

    with a half-count continuity correction, which also covers the 11.4% of rows
    with ex_ct = 0. This matters beyond correctness: the binomial form sends the
    noise of a near-saturated sequence to zero, and since the test below weights
    points by 1/sd^2, a handful of such points swamped everything else -- the
    mean of 1/sd^2 ran 1.1e6 times its median and the required sample size
    collapsed to zero. Under the Poisson log-ratio that factor is 3.0.

    Counting noise is only part of the story; library preparation and biological
    variation add more, so this remains a lower bound on the true noise and
    therefore the sample size derived from it is a lower bound too.
    """
    ex = np.maximum(counts["ex_ct"], 0.0) + 0.5
    tot = np.maximum(counts["tot_ct"], 0.0) + 0.5
    sd_log10 = np.sqrt(1.0 / ex + 1.0 / tot) / LN10
    sd_raw = abs(slope) * sd_log10
    # load_mpsa standardizes with torch's unbiased std (ddof=1).
    return sd_raw / float(np.std(counts["y_raw"], ddof=1))


def phenotype_slope(counts: dict[str, np.ndarray]) -> float:
    """Slope of y on log10(PSI), fitted on this seed's own sample."""
    N = np.maximum(counts["tot_ct"], 1.0)
    p = counts["ex_ct"] / N
    ok = (p > 0) & np.isfinite(counts["y_raw"]) & np.isfinite(p)
    if ok.sum() < 50:
        return float("nan")
    v = np.log10(p[ok])
    return float(np.polyfit(v, counts["y_raw"][ok], 1)[0])


def fit_affine(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Least-squares scale and offset mapping `a` onto `b`."""
    am, bm = a.mean(), b.mean()
    va = ((a - am) ** 2).mean()
    slope = float(((a - am) * (b - bm)).mean() / (va + 1e-30))
    return slope, float(bm - slope * am)


def twin_prediction(
    phi_t: torch.Tensor,
    ref: CapacityModel,
    w: torch.Tensor,
    fit_idx: torch.Tensor,
    mu: float,
    sd: float,
    strength: float,
    omega: float,
    seed: int,
) -> torch.Tensor:
    """The twin's phenotype prediction: g(psi^-1(refit latent)).

    The refit matches the warped latent up to an affine transform, so it is put
    back on the warp's scale using coefficients estimated on the FIT SPLIT only,
    then unwarped and pushed through the reference's own readout. If the refit
    equalled psi(phi_hat) exactly this would reproduce the reference's
    predictions exactly; the extent to which it does not is the whole quantity
    of interest.
    """
    slope, inter = fit_affine(phi_t[fit_idx], w[fit_idx])
    z_back = invert_warp(
        (slope * phi_t + inter).detach().cpu(), FAMILY, strength, omega, seed
    ).to(phi_t.device)
    with torch.no_grad():
        return ref.g(mu + sd * z_back).detach()


def ism_by_position(
    predict: Any, x: torch.Tensor, seq_len: int, idx: torch.Tensor
) -> np.ndarray:
    """Mean |change in prediction| per sequence position, over all substitutions.

    The per-locus attribution magnitude a reader would rank positions by. Uses
    the real 4-letter alphabet: for each position, each of the three bases the
    sequence does not carry is substituted in turn.
    """
    xs = x[idx]
    with torch.no_grad():
        base = predict(xs).reshape(-1)
    out = np.zeros(seq_len, dtype=float)
    for j in range(seq_len):
        block = slice(4 * j, 4 * j + 4)
        orig = xs[:, block].clone()
        acc = torch.zeros_like(base)
        for k in range(4):
            mut = xs.clone()
            mut[:, block] = 0.0
            mut[:, 4 * j + k] = 1.0
            with torch.no_grad():
                acc = acc + (predict(mut).reshape(-1) - base).abs()
        # Divide by 3: the substitution matching the original contributes zero.
        out[j] = float(acc.mean() / 3.0)
        xs[:, block] = orig
    return out


def ism_by_position_per_instance(
    predict: Any, x: torch.Tensor, seq_len: int, idx: torch.Tensor
) -> np.ndarray:
    """Per-locus attribution magnitude WITHOUT averaging over instances.

    Returns [n_instances, seq_len]. `ism_by_position` averages over instances
    before anything else looks at the numbers, which discards exactly the
    variation the identifiability claim is about: the question a biologist asks
    is whether two models rank the positions of a GIVEN splice site the same
    way, not whether they agree about the library on average.
    """
    xs = x[idx]
    with torch.no_grad():
        base = predict(xs).reshape(-1)
    out = np.zeros((xs.shape[0], seq_len), dtype=float)
    for j in range(seq_len):
        block = slice(4 * j, 4 * j + 4)
        acc = torch.zeros_like(base)
        for k in range(4):
            mut = xs.clone()
            mut[:, block] = 0.0
            mut[:, 4 * j + k] = 1.0
            with torch.no_grad():
                acc = acc + (predict(mut).reshape(-1) - base).abs()
        # Divide by 3: the substitution matching the original contributes zero.
        out[:, j] = (acc / 3.0).cpu().numpy()
    return out


def ism_by_substitution(
    predict: Any, x: torch.Tensor, seq_len: int, idx: torch.Tensor
) -> np.ndarray:
    """Mean |change in prediction| for each (position, base), length seq_len*4.

    A finer view than the seq_len position averages, giving a rank correlation
    more items to work with. Each entry averages only over the instances for
    which that base is an actual substitution -- the instances already carrying
    it contribute a structural zero and are excluded, since including them would
    measure how often a base appears rather than what changing to it does.

    Note this is 4 bases per position, not 3. A fixed 27-element vector would
    need the reference base to be the same in every sequence, and in a
    randomized library it is not, so the three "alternatives" are not a
    well-defined set across instances.
    """
    xs = x[idx]
    with torch.no_grad():
        base = predict(xs).reshape(-1)
    out = np.zeros(seq_len * 4, dtype=float)
    for j in range(seq_len):
        block = slice(4 * j, 4 * j + 4)
        for k in range(4):
            is_ref = xs[:, 4 * j + k] > 0.5
            mut = xs.clone()
            mut[:, block] = 0.0
            mut[:, 4 * j + k] = 1.0
            with torch.no_grad():
                delta = (predict(mut).reshape(-1) - base).abs()
            keep = ~is_ref
            out[4 * j + k] = (
                float(delta[keep].mean()) if bool(keep.any()) else float("nan")
            )
    return out


def per_instance_rho(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Distribution of per-instance rank agreement between two attribution maps.

    Each instance contributes one Spearman over the seq_len positions. A single
    such value is coarse -- with 9 positions the statistic takes few distinct
    values -- but the DISTRIBUTION over hundreds of instances is well determined,
    which is why the summary below is percentiles and an exceedance fraction
    rather than a mean with a confidence interval.
    """
    rhos = []
    for i in range(a.shape[0]):
        r = spearmanr(a[i], b[i]).statistic
        if np.isfinite(r):
            rhos.append(float(r))
    if not rhos:
        return {k: float("nan") for k in ("median", "p10", "p90", "frac_below_05")}
    v = np.asarray(rhos)
    return {
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "frac_below_05": float(np.mean(v < 0.5)),
        "n_instances": int(v.size),
    }


def attribution_concentration(pi: np.ndarray, k: int = 3) -> dict[str, float]:
    """How concentrated is one model's attribution, per instance?

    The BRCA2 5' splice-site library is NNN/GYNNNN, so a handful of positions
    may carry nearly all the signal with a near-zero tail. If so, a rank
    correlation over all positions is largely comparing the ORDER OF THE TAIL,
    which is noise in both models, and a low value would say nothing about
    whether the models disagree on what matters. This reports the fraction of
    total attribution mass in the top k positions and the effective number of
    contributing positions, exp of the entropy of the normalized magnitudes:
    9.0 means all positions contribute equally, near 1.0 means one dominates.
    """
    m = np.abs(np.asarray(pi, dtype=float))
    tot = m.sum(axis=1, keepdims=True)
    tot = np.where(tot <= 0, np.nan, tot)
    q = m / tot
    srt = np.sort(q, axis=1)[:, ::-1]
    topk = srt[:, :k].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(q > 0, q * np.log(q), 0.0), axis=1)
    return {
        f"top{k}_mass_median": float(np.nanmedian(topk)),
        "effective_positions_median": float(np.nanmedian(np.exp(ent))),
    }


def top_k_agreement(
    a: np.ndarray, b: np.ndarray, ks: tuple[int, ...] = (1, 2, 3)
) -> dict[str, float]:
    """Do two models pick the same top positions, per instance?

    Jaccard overlap of the top-k position SETS, which is what actually gets used
    downstream: a reader asks which positions matter for this splice site, not
    how positions seven through nine are ordered. Reported as the median over
    instances, plus the fraction of instances whose single top position matches
    exactly.
    """
    out: dict[str, float] = {}
    A = np.abs(np.asarray(a, dtype=float))
    B = np.abs(np.asarray(b, dtype=float))
    for k in ks:
        ia = np.argsort(-A, axis=1)[:, :k]
        ib = np.argsort(-B, axis=1)[:, :k]
        jac = np.empty(A.shape[0], dtype=float)
        for r in range(A.shape[0]):
            sa, sb = set(ia[r].tolist()), set(ib[r].tolist())
            jac[r] = len(sa & sb) / len(sa | sb)
        out[f"jaccard_top{k}_median"] = float(np.median(jac))
        if k == 1:
            out["top1_exact_frac"] = float(np.mean(jac >= 1.0))
    return out


def per_instance_rho_topk(a: np.ndarray, b: np.ndarray, k: int = 3) -> dict[str, float]:
    """Per-instance rank agreement restricted to the positions that carry mass.

    The rank correlation is taken over the UNION of the two models' top-k
    positions, so it asks whether they order the important positions the same
    way rather than whether they order the irrelevant ones the same way. The
    union holds between k and 2k items, which is coarser still than the full
    vector, so the median union size is reported alongside and this number is
    read together with the top-k set overlap rather than on its own.
    """
    A = np.abs(np.asarray(a, dtype=float))
    B = np.abs(np.asarray(b, dtype=float))
    rhos, sizes = [], []
    for r in range(A.shape[0]):
        idx = sorted(
            set(np.argsort(-A[r])[:k].tolist()) | set(np.argsort(-B[r])[:k].tolist())
        )
        sizes.append(len(idx))
        if len(idx) < 3:
            continue
        rr = spearmanr(a[r][idx], b[r][idx]).statistic
        if np.isfinite(rr):
            rhos.append(float(rr))
    if not rhos:
        return {"topk_rho_median": float("nan"), "topk_union_median": float("nan")}
    return {
        "topk_rho_median": float(np.median(rhos)),
        "topk_rho_p10": float(np.percentile(rhos, 10)),
        "topk_union_median": float(np.median(sizes)),
    }


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    cc, ip = cfg.closure_capacity, cfg.identifiability
    sp = cfg.separation
    dev = pick_device(cfg)
    x, y, seqs = load_mpsa_capped(cfg, seed)
    x, y = x.to(dev), y.to(dev)
    n, d = int(x.shape[0]), int(x.shape[1])
    seq_len = len(seqs[0])
    fit_idx, ho_idx, _ = split_indices(n, int(sp.n_fit), seed)
    fit_idx, ho_idx = fit_idx.to(dev), ho_idx.to(dev)
    ho_np = ho_idx.cpu().numpy()
    x_fit, y_fit = x[fit_idx], y[fit_idx]

    counts = load_counts(cfg, seed)
    slope = phenotype_slope(counts)
    sd_all = noise_sd_standardized(counts, slope)
    sd_ho = sd_all[ho_np]

    strength, omega = float(sp.strength), float(cc.omega)
    if not is_monotone(FAMILY, strength, omega, seed):
        raise RuntimeError(f"warp not monotone at seed {seed}")

    z_alpha = float(norm.ppf(1.0 - float(sp.alpha) / 2.0))
    z_beta = float(norm.ppf(float(sp.power)))
    crit = (z_alpha + z_beta) ** 2
    n_ism = min(int(sp.ism_instances), int(ho_idx.numel()))
    ism_idx = ho_idx[:n_ism]

    cells: list[dict[str, Any]] = []
    for hidden in [int(h) for h in cc.hidden]:
        for depth in [int(dp) for dp in cc.depth]:
            gen_seed = seed * 7919 + hidden * 31 + depth
            torch_seed = seed * 100 + hidden + depth
            lr = select_lr(
                d,
                hidden,
                depth,
                int(ip.ge_components),
                gen_seed,
                torch_seed,
                x_fit,
                y_fit,
                [float(v) for v in cc.lr_ladder],
                int(cc.pilot_epochs),
            )
            gen = torch.Generator().manual_seed(gen_seed)
            torch.manual_seed(torch_seed)
            ref = CapacityModel(d, hidden, depth, int(ip.ge_components), gen).to(dev)
            train(ref, x_fit, y_fit, int(cc.ref_epochs), lr)
            with torch.no_grad():
                lat = ref.latent(x).detach()
            mu = float(lat[fit_idx].mean())
            sd = float(lat[fit_idx].std()) + 1e-12
            z = (lat - mu) / sd
            w = warp(z.detach().cpu(), FAMILY, strength, omega, seed).to(dev)

            phi_t, _m = refit_to(
                w,
                x,
                ref.phi,
                d,
                hidden,
                depth,
                gen,
                int(cc.refit_epochs),
                lr,
                float(ip.radius.fit_tol),
            )
            # X: how well the twin's latent agrees, out of sample.
            x_stat = affine_r2(phi_t.cpu().numpy()[ho_np], w.cpu().numpy()[ho_np])

            y_twin = twin_prediction(
                phi_t, ref, w, fit_idx, mu, sd, strength, omega, seed
            )
            with torch.no_grad():
                y_ref = ref(x).detach()
            diff = (y_twin - y_ref).cpu().numpy()[ho_np]
            fin = np.isfinite(diff) & np.isfinite(sd_ho) & (sd_ho > 0)
            snr2 = float(np.mean((diff[fin] / sd_ho[fin]) ** 2)) if fin.any() else 0.0
            z_stat = float(crit / snr2) if snr2 > 0 else float("inf")

            # Y: do the two fits rank the loci the same way? The twin's
            # prediction on a MUTATED sequence needs the refit module itself,
            # not the stored latent values, so `_m` is what gets evaluated.
            t_slope, t_inter = fit_affine(phi_t[fit_idx], w[fit_idx])

            def pred_ref(xx: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    return ref(xx).detach()

            def pred_twin(xx: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    lat_t = _m(xx).reshape(-1).detach()
                zb = invert_warp(
                    (t_slope * lat_t + t_inter).cpu(), FAMILY, strength, omega, seed
                ).to(xx.device)
                with torch.no_grad():
                    return ref.g(mu + sd * zb).detach()

            attr_ref = ism_by_position(pred_ref, x, seq_len, ism_idx)
            attr_twin = ism_by_position(pred_twin, x, seq_len, ism_idx)
            rho = spearmanr(attr_ref, attr_twin).statistic
            y_stat = float(rho) if np.isfinite(rho) else float("nan")

            # Per-instance: the biologically meaningful question, and the one
            # with enough samples to summarise. Each instance's rho is over
            # seq_len positions and so is coarse on its own; the distribution
            # over instances is what is reported.
            pi_ref = ism_by_position_per_instance(pred_ref, x, seq_len, ism_idx)
            pi_twin = ism_by_position_per_instance(pred_twin, x, seq_len, ism_idx)
            pi = per_instance_rho(pi_ref, pi_twin)
            # Is the tail near-degenerate? If attribution is concentrated in a
            # few positions, a low all-9 rho may only reflect the ordering of
            # positions that carry no mass.
            conc_r = attribution_concentration(pi_ref)
            conc_t = attribution_concentration(pi_twin)
            tk = top_k_agreement(pi_ref, pi_twin)
            tkr = per_instance_rho_topk(pi_ref, pi_twin)

            # Substitution level: same comparison with seq_len*4 items instead
            # of seq_len, so the rank correlation is less granular.
            sub_ref = ism_by_substitution(pred_ref, x, seq_len, ism_idx)
            sub_twin = ism_by_substitution(pred_twin, x, seq_len, ism_idx)
            ok_sub = np.isfinite(sub_ref) & np.isfinite(sub_twin)
            r_sub = (
                spearmanr(sub_ref[ok_sub], sub_twin[ok_sub]).statistic
                if ok_sub.sum() > 2
                else float("nan")
            )

            cells.append(
                {
                    "hidden": hidden,
                    "depth": depth,
                    "n_params": n_params(ref.phi),
                    "lr": lr,
                    "heldout_r2": x_stat,
                    "attr_spearman": y_stat,
                    "attr_rho_sub": float(r_sub) if np.isfinite(r_sub) else float("nan"),
                    "n_sub_items": int(ok_sub.sum()),
                    "pi_median": pi["median"],
                    "pi_p10": pi["p10"],
                    "pi_p90": pi["p90"],
                    "pi_frac_below_05": pi["frac_below_05"],
                    "conc_top3_ref": conc_r["top3_mass_median"],
                    "conc_top3_twin": conc_t["top3_mass_median"],
                    "eff_pos_ref": conc_r["effective_positions_median"],
                    "eff_pos_twin": conc_t["effective_positions_median"],
                    **{f"tk_{k}": v for k, v in tk.items()},
                    **{f"tk_{k}": v for k, v in tkr.items()},
                    "n_separate": z_stat,
                    "rms_pred_diff": float(np.sqrt(np.mean(diff[fin] ** 2)))
                    if fin.any()
                    else float("nan"),
                    "median_noise_sd": float(np.median(sd_ho[np.isfinite(sd_ho)])),
                    "mean_snr2": snr2,
                    "attr_ref": attr_ref.tolist(),
                    "attr_twin": attr_twin.tolist(),
                }
            )

    return {
        "seed": seed,
        "cells": cells,
        "n": n,
        "n_fit": int(fit_idx.numel()),
        "n_heldout": int(ho_idx.numel()),
        "seq_len": seq_len,
        "phenotype_slope": slope,
        "median_noise_sd": float(np.median(sd_all)),
        "crit": crit,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--retable", metavar="RUNDIR")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    configure_torch(cfg)

    if args.retable:
        run = pathlib.Path(args.retable)
        meta = json.loads((run / "meta.json").read_text())
        files = sorted(run.glob("seed_*.json"), key=lambda f: int(f.stem.split("_")[1]))
        if not files:
            print(f"no seed_*.json in {run}", file=sys.stderr)
            return 2
        per_seed = [json.loads(f.read_text()) for f in files]
        retabled = True
        print(f"retable: {run} ({len(per_seed)} seeds)")
    else:
        retabled = False
        run = make_run_dir(cfg, "separation", allow_dirty=args.allow_dirty)
        meta = json.loads((run / "meta.json").read_text())
        print(f"run dir: {run}")
        per_seed = []
        for seed in resolve_seeds(cfg):
            r = run_seed(cfg, seed)
            per_seed.append(r)
            c = [k for k in r["cells"] if k["hidden"] == 16 and k["depth"] == 3]
            msg = (
                f" 16x3 R2 {c[0]['heldout_r2']:.6f} rho {c[0]['attr_spearman']:+.3f} "
                f"n_sep {c[0]['n_separate']:.0f}"
                if c
                else ""
            )
            print(f"seed {seed}:{msg}", flush=True)
            (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    nb = int(cfg.bootstrap_resamples)
    keys = [(c["hidden"], c["depth"]) for c in per_seed[0]["cells"]]
    agg: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "n": per_seed[0]["n"],
        "n_fit": per_seed[0]["n_fit"],
        "n_heldout": per_seed[0]["n_heldout"],
        "seq_len": per_seed[0]["seq_len"],
        "phenotype_slope": mean_ci([r["phenotype_slope"] for r in per_seed], n_boot=nb),
        "median_noise_sd": mean_ci([r["median_noise_sd"] for r in per_seed], n_boot=nb),
        "alpha": float(cfg.separation.alpha),
        "power": float(cfg.separation.power),
        "cells": {},
    }
    for h, dp in keys:
        sel = [
            c
            for r in per_seed
            for c in r["cells"]
            if c["hidden"] == h and c["depth"] == dp
        ]
        agg["cells"][f"{h}x{dp}"] = {
            "hidden": h,
            "depth": dp,
            "n_params": sel[0]["n_params"],
            "n_sub_items": sel[0]["n_sub_items"],
            "heldout_r2": mean_ci([c["heldout_r2"] for c in sel], n_boot=nb),
            "attr_spearman": mean_ci([c["attr_spearman"] for c in sel], n_boot=nb),
            "attr_rho_sub": mean_ci([c["attr_rho_sub"] for c in sel], n_boot=nb),
            "pi_median": mean_ci([c["pi_median"] for c in sel], n_boot=nb),
            "pi_p10": mean_ci([c["pi_p10"] for c in sel], n_boot=nb),
            "pi_p90": mean_ci([c["pi_p90"] for c in sel], n_boot=nb),
            "pi_frac_below_05": mean_ci(
                [c["pi_frac_below_05"] for c in sel], n_boot=nb
            ),
            **{
                f: mean_ci([c[f] for c in sel], n_boot=nb)
                for f in (
                    "conc_top3_ref", "conc_top3_twin", "eff_pos_ref", "eff_pos_twin",
                    "tk_jaccard_top1_median", "tk_jaccard_top2_median",
                    "tk_jaccard_top3_median", "tk_top1_exact_frac",
                    "tk_topk_rho_median", "tk_topk_union_median",
                )
            },
            "n_separate": mean_ci([c["n_separate"] for c in sel], n_boot=nb),
            "rms_pred_diff": mean_ci([c["rms_pred_diff"] for c in sel], n_boot=nb),
            "mean_snr2": mean_ci([c["mean_snr2"] for c in sel], n_boot=nb),
        }

    agg["verdict"] = verdict_text(agg, cfg)

    L = ["# How much data separates a fit from its reparameterized twin?\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}"
        + (
            f", **aggregated post hoc by `{git_sha()}` via --retable**"
            if retabled
            else ""
        )
        + f", config `{meta['config_hash']}`, {agg['n_seeds']} seeds, mean "
        f"[95% bootstrap CI]. {agg['n']} real BRCA2 5' splice sites per seed, "
        f"{agg['n_fit']} fit / {agg['n_heldout']} held out.\n"
    )
    L.append(
        "**The claim, per cell.** *X*: how well the twin's latent agrees with the "
        "monotone reparameterization out of sample. *Y*: Spearman between the two "
        "fits' per-locus attribution magnitudes, which is what a reader ranking "
        "positions would see. *Z*: held-out measurements needed to reject that the "
        "two fits are the same function, at two-sided α = "
        f"{agg['alpha']} with power {agg['power']}.\n"
    )
    nsub = agg["cells"][f"{keys[0][0]}x{keys[0][1]}"]["n_sub_items"]
    se9 = 1.0 / math.sqrt(max(agg["seq_len"] - 1, 1))
    L.append(
        "| width | depth | **X** held-out R² | **Y** per-instance ρ "
        "(p10 / median / p90) | frac ρ<0.5 | **Z** n to separate |"
    )
    L.append("|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        z = c["n_separate"]
        L.append(
            f"| {h} | {dp} | **{c['heldout_r2']['mean']:.6f}** | "
            f"{c['pi_p10']['mean']:+.3f} / **{c['pi_median']['mean']:+.3f}** / "
            f"{c['pi_p90']['mean']:+.3f} | "
            f"**{c['pi_frac_below_05']['mean'] * 100:.0f}%** | "
            f"**{z['mean']:,.0f}** [{z['lo']:,.0f}, {z['hi']:,.0f}] |"
        )
    L.append("")
    L.append(
        f"**Y is now per instance.** For each held-out sequence, the two fits' "
        f"{agg["seq_len"]} per-position attribution magnitudes are rank-correlated, and the "
        "table reports the distribution of those correlations over instances. "
        "Averaging attributions across instances first, as an earlier version did, "
        "discards exactly the variation the claim is about: the question is whether "
        "two fits rank the positions of a GIVEN splice site the same way.\n"
    )
    L.append(
        "**Is the tail degenerate?** If attribution concentrates in a few "
        "positions, a low all-position ρ may only be reporting the order of "
        "positions that carry no mass. `top-3 mass` is the median fraction of "
        "total |Δ| in a sequence's three strongest positions; `eff. positions` is "
        "exp(entropy) of the normalized magnitudes, where "
        f"{agg['seq_len']:.1f} means all positions contribute equally and 1.0 means "
        "one dominates. `Jaccard` is the median overlap of the two fits' top-k "
        "position SETS — the quantity a reader actually uses.\n"
    )
    L.append(
        "| width | depth | top-3 mass | eff. positions | Jaccard top1 / top2 / top3 "
        "| top-1 exact | ρ on top-3 union |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        L.append(
            f"| {h} | {dp} | {c['conc_top3_ref']['mean']:.3f} | "
            f"{c['eff_pos_ref']['mean']:.2f} | "
            f"{c['tk_jaccard_top1_median']['mean']:.2f} / "
            f"{c['tk_jaccard_top2_median']['mean']:.2f} / "
            f"**{c['tk_jaccard_top3_median']['mean']:.2f}** | "
            f"{c['tk_top1_exact_frac']['mean'] * 100:.0f}% | "
            f"{c['tk_topk_rho_median']['mean']:+.3f} "
            f"(n≈{c['tk_topk_union_median']['mean']:.1f}) |"
        )
    L.append("")
    L.append(
        "| width | depth | ρ over position averages (n="
        + str(agg["seq_len"])
        + ") | ρ over substitutions (n="
        + str(nsub)
        + ") |"
    )
    L.append("|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        L.append(
            f"| {h} | {dp} | {c['attr_spearman']['mean']:+.3f} | "
            f"{c['attr_rho_sub']['mean']:+.3f} |"
        )
    L.append("")
    L.append(
        f"**These two columns are coarse and are retained only for continuity.** A "
        f"Spearman over {agg["seq_len"]} items has an approximate standard error of "
        f"1/sqrt({agg["seq_len"]}-1) = {se9:.2f} under independence, so a value of +0.37 is "
        f"about one standard error from zero and even +0.9 is not precise. The "
        f"substitution view uses {nsub} items and is correspondingly less granular. "
        "The bracketed intervals elsewhere in this table are bootstrap intervals "
        "over SEEDS and do not include the rank correlation's own granularity, so "
        "neither column should be read as a precise quantity. The per-instance "
        "distribution above is the one the verdict uses, because it is summarised "
        "by percentiles over hundreds of instances rather than by a single coarse "
        "statistic.\n"
    )
    L.append(
        f"Noise is not assumed. The phenotype is affine in log₁₀ of the count ratio "
        f"ex_ct/tot_ct (slope {agg['phenotype_slope']['mean']:.3f} on this library). "
        "Those are two independent count pools rather than a proportion — ex_ct "
        "exceeds tot_ct in 4.9% of rows — so both are treated as Poisson and the "
        "delta method applied to the log ratio, giving "
        "`sd = |a|·sqrt(1/ex + 1/tot)/ln10`: median "
        f"{agg['median_noise_sd']['mean']:.4f} in the standardized units the models "
        "see. This counts sequencing noise only; library preparation and biological "
        "variation add more. **Z is therefore a lower bound** — at least this many "
        "measurements, likely more.\n"
    )
    L.append("## Verdict\n")
    L.append(agg["verdict"])
    table = "\n".join(L) + "\n"

    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "separation.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(json.dumps(to_jsonable(agg), indent=1))
    write_tuning_budget(
        run,
        [
            {
                "model": f"neural_{h}x{dp}",
                "configs_tried": len(list(cfg.closure_capacity.lr_ladder)),
                "epochs": int(cfg.closure_capacity.ref_epochs),
                "gradient_steps": int(cfg.closure_capacity.ref_epochs),
                "search_space": "learning rate over "
                + str([float(v) for v in cfg.closure_capacity.lr_ladder]),
                "selection": "pilot on the fit split's training loss",
            }
            for h, dp in keys
        ],
    )
    print("\n" + table)
    return 0


def verdict_text(agg: dict[str, Any], cfg: Any) -> str:
    cells = agg["cells"]
    best = max(cells, key=lambda k: cells[k]["heldout_r2"]["mean"])
    b = cells[best]
    zs = {k: v["n_separate"]["mean"] for k, v in cells.items()}
    z_hi_cell = max(zs, key=lambda k: zs[k])
    z_lo_cell = min(zs, key=lambda k: zs[k])
    lib = int(cfg.separation.library_size)

    reach = [k for k in zs if zs[k] <= lib]
    parts = [
        "**The identifiability claim, stated quantitatively.** The twin is not "
        "identical to the original and it is not unrelated to it, so neither the "
        "strong claim nor the word 'collapse' describes the measurement. At the "
        f"closest cell ({best}) the twin agrees to held-out R² "
        f"{b['heldout_r2']['mean']:.6f}, its predictions differ from the "
        f"original's by RMS {b['rms_pred_diff']['mean']:.4f} in standardized "
        f"phenotype units, and separating the two as functions takes at least "
        f"{b['n_separate']['mean']:,.0f} held-out measurements "
        f"[{b['n_separate']['lo']:,.0f}, {b['n_separate']['hi']:,.0f}] at "
        f"α = {agg['alpha']} with power {agg['power']}."
    ]
    parts.append(
        f"Across the grid the requirement ranges from {zs[z_lo_cell]:,.0f} "
        f"measurements ({z_lo_cell}) to {zs[z_hi_cell]:,.0f} ({z_hi_cell})."
        + (
            f" All {len(reach)} of {len(zs)} cells are separable within this "
            f"library's {lib:,} sequences, so the twin is distinguishable with data "
            "that already exists."
            if len(reach) == len(zs)
            else f" {len(reach)} of {len(zs)} cells are separable within this "
            f"library's {lib:,} sequences; the rest would need a larger experiment."
        )
    )
    pim = {k: v["pi_median"]["mean"] for k, v in cells.items()}
    pif = {k: v["pi_frac_below_05"]["mean"] for k, v in cells.items()}
    worst_pi = min(pim, key=lambda k: pim[k])
    most_disagree = max(pif, key=lambda k: pif[k])
    parts.append(
        "**The attribution consequence is what makes this practical rather than "
        "philosophical, and it is measured per instance.** For a given splice site, "
        "the two fits' rankings of its "
        f"{agg['seq_len']} positions agree at a median Spearman of "
        f"{pim[worst_pi]:+.3f} in the worst cell ({worst_pi}) and "
        f"{pim[best]:+.3f} at {best}; in {most_disagree} "
        f"{pif[most_disagree] * 100:.0f}% of held-out sites the two rankings agree "
        f"at ρ below 0.5. A reader who ranks positions by attribution magnitude for a "
        "particular sequence is therefore reading a quantity the data does not pin "
        "down, which is the failure mode the certificates in this project are meant "
        "to prevent. The per-instance distribution is used here rather than the "
        f"single ρ over {agg['seq_len']} position-averages, which is too coarse to "
        f"carry a claim: its standard error under independence is about "
        f"{1.0 / (max(agg['seq_len'] - 1, 1) ** 0.5):.2f}."
    )

    # Which reading does the per-instance number support? A low all-position rho
    # is ambiguous between "the fits disagree about what matters" and "they agree
    # about what matters and order the irrelevant tail independently". The top-k
    # set overlap separates those, so the verdict is stated conditionally on it.
    j3 = {k: v["tk_jaccard_top3_median"]["mean"] for k, v in cells.items()}
    ex1 = {k: v["tk_top1_exact_frac"]["mean"] for k, v in cells.items()}
    mass = {k: v["conc_top3_ref"]["mean"] for k, v in cells.items()}
    effp = {k: v["eff_pos_ref"]["mean"] for k, v in cells.items()}
    thr = float(cfg.separation.jaccard_agree)
    worst_j = min(j3, key=lambda k: j3[k])
    concentrated = float(np.median(list(mass.values())))
    agree = sorted(k for k in j3 if j3[k] >= thr)
    disagree = sorted(k for k in j3 if k not in agree)
    if agree and disagree:
        parts.append(
            f"**Which reading this supports: it depends on the cell, and both must be "
            f"stated.** Attribution is concentrated ({concentrated:.0%} of each "
            f"sequence's |Δ| mass in three positions, an effective "
            f"{float(np.median(list(effp.values()))):.2f} contributing positions of "
            f"{agg['seq_len']}), so the top-3 set is the part that carries signal. In "
            + ", ".join(agree)
            + f" the two fits AGREE on which positions those are (median top-3 Jaccard "
            f"at least {min(j3[k] for k in agree):.2f}), so there the full-rank "
            "disagreement is tail ordering and the low per-instance ρ is NOT the "
            "headline; the honest claim for those cells is the narrow one. In "
            + ", ".join(disagree)
            + f" they do NOT agree (median top-3 Jaccard down to {j3[worst_j]:.2f}, "
            f"strongest position matching exactly in only {ex1[worst_j] * 100:.0f}% of "
            "sequences), so there the models differ about what matters and the claim "
            "stands as written. **Top-k set overlap is the statistic to lead with "
            "either way**, because it is what gets used downstream and it is the one "
            "that separates these two readings."
        )
    elif not disagree:
        parts.append(
            f"**Which reading this supports: the narrower one.** Attribution is "
            f"concentrated — a median {concentrated:.0%} of each sequence's total |Δ| "
            f"sits in three positions, an effective "
            f"{float(np.median(list(effp.values()))):.2f} contributing positions of "
            f"{agg['seq_len']} — and the two fits agree on WHICH positions those are: "
            f"the median top-3 Jaccard is at least {min(j3.values()):.2f} in every "
            f"cell and the single strongest position matches exactly in "
            f"{min(ex1.values()) * 100:.0f}% to {max(ex1.values()) * 100:.0f}% of "
            "sequences. **The full-rank disagreement is therefore mostly in the tail, "
            "among positions carrying little attribution mass, and the low "
            "per-instance ρ must NOT be read as the headline.** The honest claim is "
            "narrower: the two fits agree on which positions matter and disagree "
            "about the ordering of those that do not."
        )
    else:
        parts.append(
            f"**Which reading this supports: the claim as written.** The disagreement "
            f"is not confined to an irrelevant tail. The two fits differ on WHICH "
            f"positions matter: the median top-3 Jaccard falls to {j3[worst_j]:.2f} "
            f"({worst_j}) and the single strongest position matches exactly in only "
            f"{ex1[worst_j] * 100:.0f}% of sequences there. Restricted to the union "
            "of the two top-3 sets — the positions that actually carry mass — the "
            f"median rank agreement is {cells[worst_j]['tk_topk_rho_median']['mean']:+.3f}. "
            f"Attribution is concentrated ({concentrated:.0%} of mass in three "
            "positions), so this is disagreement about the part that carries the "
            "signal. **Top-k set overlap is the better statistic to lead with**, "
            "because a reader asks which positions matter for a given site, not how "
            "the bottom of the list is ordered."
        )
    parts.append(
        "This triple holds regardless of how the containment question in "
        "`paper/tables/closure_search.md` resolves. It describes the two fits "
        "actually obtained and the data actually needed to tell them apart, not "
        "whether some member of the class equals ψ∘φ̂ exactly."
    )
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
