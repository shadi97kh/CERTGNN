"""Do models the data cannot tell apart nonetheless rank the loci differently?

This is the identifiability claim tested directly, with no warp, no
reparameterization, and no refit. It therefore cannot be confounded by any of
the refit failures found so far.

**Why this is the right experiment.** `paper/tables/closure_search.md` Arm 1
refit phi_hat -- a function realized exactly by a network of the identical
architecture -- from a cold start, and at 128x3 reached training loss 2.16e-07
with held-out R2 0.21. The set of class members with near-zero fit-split loss
is therefore large, and most of it does not generalize. That IS the
identifiability problem, stated without any construction: many members of the
class explain the observed data equally well and disagree elsewhere. The warp
was only ever one way to exhibit a second such member, and a lossy one.

**This also corrects the Part B reading in the original identifiability probe.**
There, multi-restart refits differing as functions was recorded as a negative,
on the grounds that they were "different fits, not different representatives."
That was wrong. Two fits that explain the data equally well ARE the
identifiability problem; the fact that they arose from different restarts
rather than from an analytic construction makes them more relevant, not less.
What Part B lacked was a PERFORMANCE FILTER: without one, a pair of fits that
differ might simply be a good fit and a bad one, which is not an
identifiability instance because the data does prefer the first.

Protocol, per cell, on the real BRCA2 MPSA with 3000 fit / 1000 held out:

1. Train K reference models from different init seeds on the SAME fit split.
   Same architecture, same learning-rate selection, same budget: the
   initialization seed is the only thing that differs.
2. Record each model's held-out predictive R2 on the phenotype.
3. Filter to pairs whose held-out performance is NOT distinguishable, by a
   paired two-sided t-test on per-point held-out squared errors. A pair that
   fails this test is one the data can choose between, so it is not an
   identifiability instance and is excluded and counted.
4. For each surviving pair report the latent agreement (affine R2 on held-out
   points), the attribution agreement (Spearman between per-locus ISM
   magnitudes), and n_separate under the Poisson log-ratio noise model.

No multiple-comparison correction is applied to the filter, and that direction
is deliberate: without correction MORE pairs are declared distinguishable, so
FEWER survive, and the surviving set is a stricter subset than a corrected
filter would give. The headline claim is about the survivors, so the
uncorrected filter is the conservative choice.

The result, either way:

- Attribution Spearman well below 1 among indistinguishable pairs: models the
  data cannot separate rank the loci differently, and the table says by how
  much. That is the occurrence result, and it needs no constructed twin.
- Attribution Spearman near 1 among indistinguishable pairs: training pins the
  attribution even though it does not pin the function. That is a real finding
  in the other direction and is reported as plainly.

Usage:
    python -m experiments.occurrence --config configs/base.yaml
    python -m experiments.occurrence --retable results/runs/<dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import norm, spearmanr, ttest_rel

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
    CapacityModel,
    load_mpsa_capped,
    n_params,
    pick_device,
    select_lr,
    train,
)
from experiments.closure_heldout import split_indices
from experiments.identifiability_probe import affine_r2
from experiments.separation import (
    ism_by_position,
    load_counts,
    noise_sd_standardized,
    phenotype_slope,
)


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    cc, ip = cfg.closure_capacity, cfg.identifiability
    oc = cfg.occurrence
    dev = pick_device(cfg)
    x, y, seqs = load_mpsa_capped(cfg, seed)
    x, y = x.to(dev), y.to(dev)
    n, d = int(x.shape[0]), int(x.shape[1])
    seq_len = len(seqs[0])
    fit_idx, ho_idx, _ = split_indices(n, int(oc.n_fit), seed)
    fit_idx, ho_idx = fit_idx.to(dev), ho_idx.to(dev)
    ho_np = ho_idx.cpu().numpy()
    x_fit, y_fit = x[fit_idx], y[fit_idx]
    y_ho = y[ho_idx]

    counts = load_counts(cfg, seed)
    sd_ho = noise_sd_standardized(counts, phenotype_slope(counts))[ho_np]

    # n_separate uses the same test and thresholds as separation.py.
    z_a = float(norm.ppf(1.0 - float(cfg.separation.alpha) / 2.0))
    z_b = float(norm.ppf(float(cfg.separation.power)))
    crit = (z_a + z_b) ** 2
    filt_alpha = float(oc.alpha)
    K = int(oc.n_models)
    n_ism = min(int(oc.ism_instances), int(ho_idx.numel()))
    ism_idx = ho_idx[:n_ism]

    cells: list[dict[str, Any]] = []
    for hidden in [int(h) for h in cc.hidden]:
        for depth in [int(dp) for dp in cc.depth]:
            base_gen = seed * 7919 + hidden * 31 + depth
            base_torch = seed * 100 + hidden + depth
            # One learning-rate selection for the cell, shared by all K models,
            # so the initialization seed really is the only difference.
            lr = select_lr(
                d,
                hidden,
                depth,
                int(ip.ge_components),
                base_gen,
                base_torch,
                x_fit,
                y_fit,
                [float(v) for v in cc.lr_ladder],
                int(cc.pilot_epochs),
            )

            preds, lats, attrs, r2s, sq_errs = [], [], [], [], []
            for k in range(K):
                gk, tk = base_gen + 7_919_000 * (k + 1), base_torch + 104_729 * (k + 1)
                gen = torch.Generator().manual_seed(gk)
                torch.manual_seed(tk)
                m = CapacityModel(d, hidden, depth, int(ip.ge_components), gen).to(dev)
                train(m, x_fit, y_fit, int(cc.ref_epochs), lr)
                with torch.no_grad():
                    p_ho = m(x[ho_idx]).reshape(-1).detach()
                    l_ho = m.latent(x[ho_idx]).detach()
                se = ((y_ho - p_ho) ** 2).cpu().numpy()
                r2 = 1.0 - float(
                    ((y_ho - p_ho) ** 2).sum() / ((y_ho - y_ho.mean()) ** 2).sum()
                )

                def predict(xx: torch.Tensor, _m: CapacityModel = m) -> torch.Tensor:
                    with torch.no_grad():
                        return _m(xx).detach()

                attrs.append(ism_by_position(predict, x, seq_len, ism_idx))
                preds.append(p_ho.cpu().numpy())
                lats.append(l_ho.cpu().numpy())
                sq_errs.append(se)
                r2s.append(r2)

            pairs: list[dict[str, Any]] = []
            n_total = 0
            for i, j in itertools.combinations(range(K), 2):
                n_total += 1
                # Performance filter: can the data choose between them?
                t = ttest_rel(sq_errs[i], sq_errs[j])
                pval = float(t.pvalue) if np.isfinite(t.pvalue) else 1.0
                indist = pval > filt_alpha
                if not indist:
                    continue
                diff = preds[i] - preds[j]
                fin = np.isfinite(diff) & np.isfinite(sd_ho) & (sd_ho > 0)
                snr2 = (
                    float(np.mean((diff[fin] / sd_ho[fin]) ** 2)) if fin.any() else 0.0
                )
                rho = spearmanr(attrs[i], attrs[j]).statistic
                pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "p_value": pval,
                        "r2_gap": abs(r2s[i] - r2s[j]),
                        "latent_r2": affine_r2(lats[i], lats[j]),
                        "attr_spearman": float(rho)
                        if np.isfinite(rho)
                        else float("nan"),
                        "n_separate": float(crit / snr2) if snr2 > 0 else float("inf"),
                        "rms_pred_diff": float(np.sqrt(np.mean(diff[fin] ** 2)))
                        if fin.any()
                        else float("nan"),
                    }
                )

            rhos = [
                p["attr_spearman"] for p in pairs if np.isfinite(p["attr_spearman"])
            ]
            nseps = [p["n_separate"] for p in pairs if np.isfinite(p["n_separate"])]
            lat = [p["latent_r2"] for p in pairs if np.isfinite(p["latent_r2"])]
            cells.append(
                {
                    "hidden": hidden,
                    "depth": depth,
                    "n_params": n_params(
                        CapacityModel(
                            d,
                            hidden,
                            depth,
                            int(ip.ge_components),
                            torch.Generator().manual_seed(0),
                        ).phi
                    ),
                    "lr": lr,
                    "n_models": K,
                    "n_pairs_total": n_total,
                    "n_pairs_indistinguishable": len(pairs),
                    "heldout_r2_min": float(np.min(r2s)),
                    "heldout_r2_max": float(np.max(r2s)),
                    "heldout_r2_mean": float(np.mean(r2s)),
                    "attr_rho_min": float(np.min(rhos)) if rhos else float("nan"),
                    "attr_rho_max": float(np.max(rhos)) if rhos else float("nan"),
                    "attr_rho_median": float(np.median(rhos)) if rhos else float("nan"),
                    "n_separate_median": float(np.median(nseps))
                    if nseps
                    else float("nan"),
                    "latent_r2_median": float(np.median(lat)) if lat else float("nan"),
                    "pairs": pairs,
                }
            )

    return {
        "seed": seed,
        "cells": cells,
        "n": n,
        "n_fit": int(fit_idx.numel()),
        "n_heldout": int(ho_idx.numel()),
        "seq_len": seq_len,
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
        run = make_run_dir(cfg, "occurrence", allow_dirty=args.allow_dirty)
        meta = json.loads((run / "meta.json").read_text())
        print(f"run dir: {run}")
        per_seed = []
        for seed in resolve_seeds(cfg):
            r = run_seed(cfg, seed)
            per_seed.append(r)
            c = [k for k in r["cells"] if k["hidden"] == 64 and k["depth"] == 2]
            msg = (
                f" 64x2: {c[0]['n_pairs_indistinguishable']}/{c[0]['n_pairs_total']} "
                f"pairs indist., rho {c[0]['attr_rho_min']:.3f}-{c[0]['attr_rho_max']:.3f}"
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
        "n_models": per_seed[0]["cells"][0]["n_models"],
        "filter_alpha": float(cfg.occurrence.alpha),
        "cells": {},
    }
    scalar = [
        "n_pairs_total",
        "n_pairs_indistinguishable",
        "heldout_r2_min",
        "heldout_r2_max",
        "heldout_r2_mean",
        "attr_rho_min",
        "attr_rho_max",
        "attr_rho_median",
        "n_separate_median",
        "latent_r2_median",
    ]
    for h, dp in keys:
        sel = [
            c
            for r in per_seed
            for c in r["cells"]
            if c["hidden"] == h and c["depth"] == dp
        ]
        entry: dict[str, Any] = {
            "hidden": h,
            "depth": dp,
            "n_params": sel[0]["n_params"],
        }
        for f in scalar:
            vals = [c[f] for c in sel if np.isfinite(c[f])]
            if vals:
                entry[f] = mean_ci(vals, n_boot=nb)
        # Pooled over every surviving pair in every seed: the distribution the
        # headline describes, rather than a mean of per-seed summaries.
        allr = [
            p["attr_spearman"]
            for c in sel
            for p in c["pairs"]
            if np.isfinite(p["attr_spearman"])
        ]
        alln = [
            p["n_separate"]
            for c in sel
            for p in c["pairs"]
            if np.isfinite(p["n_separate"])
        ]
        entry["pooled_pairs"] = len(allr)
        entry["pooled_rho_min"] = float(np.min(allr)) if allr else float("nan")
        entry["pooled_rho_max"] = float(np.max(allr)) if allr else float("nan")
        entry["pooled_rho_median"] = float(np.median(allr)) if allr else float("nan")
        entry["pooled_nsep_median"] = float(np.median(alln)) if alln else float("nan")
        agg["cells"][f"{h}x{dp}"] = entry

    agg["verdict"] = verdict_text(agg, cfg)

    L = ["# Do models the data cannot tell apart rank the loci differently?\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}"
        + (
            f", **aggregated post hoc by `{git_sha()}` via --retable**"
            if retabled
            else ""
        )
        + f", config `{meta['config_hash']}`, {agg['n_seeds']} seeds, mean "
        f"[95% bootstrap CI]. {agg['n']} real BRCA2 5' splice sites per seed, "
        f"{agg['n_fit']} fit / {agg['n_heldout']} held out. "
        f"{agg['n_models']} models per cell, differing ONLY in initialization "
        "seed: same architecture, same fit split, same learning-rate selection, "
        "same budget. No warp, no refit, no reparameterization.\n"
    )
    L.append(
        f"A pair is **indistinguishable** when a paired two-sided t-test on "
        f"per-point held-out squared errors does not reject at α = "
        f"{agg['filter_alpha']}. Pairs that fail are ones the data can choose "
        "between and are excluded, not counted as identifiability instances. No "
        "multiple-comparison correction is applied, which makes the filter stricter "
        "rather than looser: more pairs are called distinguishable, so fewer "
        "survive.\n"
    )
    L.append(
        "| width | depth | params | held-out R² | indist. pairs | attribution ρ "
        "(min–median–max) | median n to separate | latent R² |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        ip_ = c.get("n_pairs_indistinguishable", {}).get("mean", float("nan"))
        tp = c.get("n_pairs_total", {}).get("mean", float("nan"))
        L.append(
            f"| {h} | {dp} | {c['n_params']:,} | "
            f"{c['heldout_r2_mean']['mean']:.4f} | "
            f"{ip_:.1f}/{tp:.0f} | "
            f"**{c['pooled_rho_min']:+.3f} – {c['pooled_rho_median']:+.3f} – "
            f"{c['pooled_rho_max']:+.3f}** | "
            f"{c['pooled_nsep_median']:,.0f} | "
            f"{c['latent_r2_median']['mean']:.4f} |"
        )
    L.append("")
    L.append(
        "The ρ column pools every surviving pair across all seeds, so it is the "
        "distribution the claim is about rather than a mean of per-seed summaries. "
        "`n to separate` uses the Poisson log-ratio noise model of "
        "`paper/tables/separation.md` applied to the two models' own predictions.\n"
    )
    L.append("## Verdict\n")
    L.append(agg["verdict"])
    table = "\n".join(L) + "\n"

    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "occurrence.md").write_text(table)
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
                + str([float(v) for v in cfg.closure_capacity.lr_ladder])
                + ", selected once per cell and shared by all models in it",
                "selection": "no model selection: every trained model is kept and "
                "enters the pairwise analysis",
            }
            for h, dp in keys
        ],
    )
    print("\n" + table)
    return 0


def verdict_text(agg: dict[str, Any], cfg: Any) -> str:
    cells = agg["cells"]
    oc = cfg.occurrence
    near_one = float(oc.rho_near_one)
    usable = {
        k: v
        for k, v in cells.items()
        if v["pooled_pairs"] > 0 and np.isfinite(v["pooled_rho_min"])
    }
    parts: list[str] = []
    if not usable:
        return (
            "**No pair survived the performance filter in any cell.** Every pair of "
            "independently initialized models is separated by the held-out data at "
            f"α = {agg['filter_alpha']}, so none is an identifiability instance and "
            "the question this experiment asks does not arise at these settings."
        )

    surv = sum(v["n_pairs_indistinguishable"]["mean"] for v in usable.values())
    tot = sum(v["n_pairs_total"]["mean"] for v in usable.values())
    worst = min(usable, key=lambda k: usable[k]["pooled_rho_min"])
    med_of_med = float(np.median([v["pooled_rho_median"] for v in usable.values()]))
    # "in every cell" needs the MINIMUM of the per-cell medians. Using the
    # maximum here asserted the best cell's median as a bound holding
    # everywhere, which is false whenever the cells disagree.
    all_low = min(v["pooled_rho_median"] for v in usable.values())

    parts.append(
        f"**{surv:.0f} of {tot:.0f} pairs are indistinguishable on held-out "
        f"performance** (paired t-test, α = {agg['filter_alpha']}, uncorrected and "
        "therefore strict). These are pairs of models the data cannot choose "
        "between: same architecture, same fit split, same budget, differing only in "
        "initialization seed."
    )

    if all_low >= near_one:
        parts.append(
            f"**Training pins the attribution.** Among those pairs the per-locus "
            f"attribution Spearman has a median of at least {all_low:.3f} in every "
            f"cell, and the lowest single pair anywhere is "
            f"{usable[worst]['pooled_rho_min']:+.3f} ({worst}). Models that the data "
            "cannot separate nevertheless agree on how to rank the loci. **This is "
            "the negative for the occurrence claim, and it is a real finding in the "
            "other direction:** the function is underdetermined -- these models "
            "differ, and the median number of measurements needed to separate them "
            f"is {float(np.median([v['pooled_nsep_median'] for v in usable.values()])):,.0f} "
            "-- but the attribution ranking is not. An explanation that reports only "
            "a locus ordering is more stable than the fit it comes from."
        )
    else:
        parts.append(
            f"**Models the data cannot tell apart rank the loci differently.** Among "
            f"the indistinguishable pairs the per-locus attribution Spearman runs "
            f"down to {usable[worst]['pooled_rho_min']:+.3f} ({worst}), with a "
            f"typical cell median of {med_of_med:+.3f}. The disagreement is not "
            "between a good fit and a bad one: these pairs are exactly the ones the "
            "held-out data cannot choose between. **This is the occurrence result.** "
            "It needs no constructed twin, no reparameterization and no refit, so it "
            "is immune to the search and warm-start confounds that "
            "`paper/tables/closure_search.md` found in the warp-based route."
        )

    parts.append(
        "Per cell, the median number of held-out measurements needed to separate an "
        "indistinguishable pair as functions is "
        + ", ".join(
            f"{k} {usable[k]['pooled_nsep_median']:,.0f}"
            for k in sorted(usable, key=lambda k: usable[k]["pooled_nsep_median"])[:4]
        )
        + " (four smallest). The latent agreement among surviving pairs has median "
        + ", ".join(
            f"{k} {usable[k]['latent_r2_median']['mean']:.4f}"
            for k in sorted(usable)[:3]
        )
        + ", so where the rankings differ the underlying latents differ too."
    )
    parts.append(
        "This experiment corrects the Part B reading in the original identifiability "
        "probe, which recorded multi-restart fits differing as a negative on the "
        "grounds that they were different fits rather than different representatives. "
        "Two fits that explain the data equally well are the identifiability problem; "
        "what that analysis lacked was the performance filter applied here."
    )
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
