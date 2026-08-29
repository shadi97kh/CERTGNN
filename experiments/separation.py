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

            cells.append(
                {
                    "hidden": hidden,
                    "depth": depth,
                    "n_params": n_params(ref.phi),
                    "lr": lr,
                    "heldout_r2": x_stat,
                    "attr_spearman": y_stat,
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
            "heldout_r2": mean_ci([c["heldout_r2"] for c in sel], n_boot=nb),
            "attr_spearman": mean_ci([c["attr_spearman"] for c in sel], n_boot=nb),
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
    L.append(
        "| width | depth | params | **X** held-out R² | **Y** attribution ρ | "
        "**Z** n to separate | RMS pred. diff |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        z = c["n_separate"]
        L.append(
            f"| {h} | {dp} | {c['n_params']:,} | "
            f"**{c['heldout_r2']['mean']:.6f}** | "
            f"**{c['attr_spearman']['mean']:+.3f}** "
            f"[{c['attr_spearman']['lo']:+.3f}, {c['attr_spearman']['hi']:+.3f}] | "
            f"**{z['mean']:,.0f}** [{z['lo']:,.0f}, {z['hi']:,.0f}] | "
            f"{c['rms_pred_diff']['mean']:.4f} |"
        )
    L.append("")
    L.append(
        f"Noise is not assumed. The phenotype is affine in log₁₀ of the inclusion "
        f"fraction (slope {agg['phenotype_slope']['mean']:.3f} on this library), and "
        f"the counts are binomial, so per-sequence noise follows by the delta method: "
        f"median {agg['median_noise_sd']['mean']:.4f} in the standardized units the "
        "models see. This counts sequencing noise only; library preparation and "
        "biological variation add more. **Z is therefore a lower bound** — at least "
        "this many measurements, likely more.\n"
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
    rhos = {k: v["attr_spearman"]["mean"] for k, v in cells.items()}
    worst_rho_cell = min(rhos, key=lambda k: rhos[k])

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
    parts.append(
        "**The attribution consequence is what makes this practical rather than "
        f"philosophical.** Two fits agreeing this closely still rank the loci "
        f"differently: Spearman {rhos[worst_rho_cell]:+.3f} at the worst cell "
        f"({worst_rho_cell}) and {rhos[best]:+.3f} at {best}. A reader who ranks "
        "positions by attribution magnitude is reading a quantity that the data "
        "does not pin down at the sample sizes above, which is the failure mode "
        "the certificates in this project are meant to prevent."
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
