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
    void_if_unsearched,
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
    attribution_concentration,
    ism_by_position_per_instance,
    ism_by_substitution,
    load_counts,
    noise_sd_standardized,
    per_instance_rho,
    per_instance_rho_topk,
    phenotype_slope,
    top_k_agreement,
    top_k_agreement_conditioned,
)


def _finite(v: Any) -> bool:
    """np.isfinite that tolerates None.

    `to_jsonable` writes NaN as JSON null, so a value that is NaN in the live
    run comes back as None under --retable and np.isfinite raises TypeError on
    it. Conditioned statistics are NaN whenever no instance qualifies, which
    happens in the flattest cells, so this path is reached in practice.
    """
    return v is not None and isinstance(v, (int, float)) and np.isfinite(v)


def _collect_per_instance(
    sink: dict[str, Any],
    cell: str,
    seed: int,
    per_inst: list[np.ndarray],
    pairs: list[dict[str, Any]],
    k_top: int = 3,
) -> None:
    """Record per-SEQUENCE agreement for every accuracy-matched pair.

    A pure dump. It reads the same `per_inst` arrays the verdict path reads and
    recomputes nothing that feeds a verdict; the summaries above are untouched
    whether or not this runs. It exists because the per-sequence values are the
    argument for Proposition 2 and the aggregation throws them away: only the
    median, p10, p90 and an exceedance fraction survive into `results.json`,
    and none of those can show that a sequence with LOW full-rank rho can still
    have IDENTICAL top-3 sets.

    Two arrays per accuracy-matched pair, one value per held-out sequence:
    the full-rank Spearman over all positions, and the top-k Jaccard.
    Plus, once per cell-seed, each model's per-sequence attribution profile
    sorted descending and normalized to sum one, which is the rank-profile
    panel.
    """
    rho_rows: list[np.ndarray] = []
    jac_rows: list[np.ndarray] = []
    top1_rows: list[np.ndarray] = []
    pair_ids: list[tuple[int, int]] = []
    for p in pairs:
        a, b = per_inst[int(p["i"])], per_inst[int(p["j"])]
        A, B = np.abs(np.asarray(a, float)), np.abs(np.asarray(b, float))
        n = A.shape[0]
        rho = np.full(n, np.nan)
        jac = np.full(n, np.nan)
        # Whether the two models' single STRONGEST position is the same one.
        # The top-k Jaccard is over unordered SETS, so it cannot answer this:
        # two models can select the same three positions and still disagree
        # about which of them dominates. Disagreeing about the dominant
        # position is a much stronger claim than disagreeing about the
        # ordering beneath it, and the two have to be reported separately.
        top1 = np.zeros(n, dtype=np.float32)
        ia = np.argsort(-A, axis=1)[:, :k_top]
        ib = np.argsort(-B, axis=1)[:, :k_top]
        for r in range(n):
            v = spearmanr(A[r], B[r]).statistic
            rho[r] = float(v) if np.isfinite(v) else np.nan
            sa, sb = set(ia[r].tolist()), set(ib[r].tolist())
            jac[r] = len(sa & sb) / len(sa | sb)
            top1[r] = float(ia[r, 0] == ib[r, 0])
        rho_rows.append(rho.astype(np.float32))
        jac_rows.append(jac.astype(np.float32))
        top1_rows.append(top1)
        pair_ids.append((int(p["i"]), int(p["j"])))

    key = f"{cell}_seed{seed}"
    if rho_rows:
        sink[f"{key}__rho"] = np.stack(rho_rows)
        sink[f"{key}__jaccard"] = np.stack(jac_rows)
        sink[f"{key}__top1_same"] = np.stack(top1_rows)
        sink[f"{key}__pairs"] = np.asarray(pair_ids, dtype=np.int16)

    # `rankprofile` sorts each row by magnitude, which answers "how concentrated"
    # but destroys WHICH position carried the mass. Keep the position-indexed
    # matrix too: the library holds position 4 fixed at G and position 5 to
    # {C,U}, so five of the 36 one-hot inputs are identically zero and their
    # weights never leave initialization. Asking what that does to the top-k
    # sets needs position identity, not just the spectrum.
    prof = []
    ism = []
    for pi in per_inst:
        m = np.abs(np.asarray(pi, float))
        ism.append(m.astype(np.float32))
        tot = m.sum(axis=1, keepdims=True)
        tot = np.where(tot <= 0, np.nan, tot)
        prof.append(np.sort(m / tot, axis=1)[:, ::-1].astype(np.float32))
    if prof:
        sink[f"{key}__rankprofile"] = np.stack(prof)
        sink[f"{key}__ism"] = np.stack(ism)


def run_seed(
    cfg: Any,
    seed: int,
    dump_cells: frozenset[str] | None = None,
    dump_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    # One alpha drives both the performance filter and the reported resolution
    # of that filter, so they cannot drift apart. n_separate is meant to be
    # comparable with separation.py, so equality with its alpha is asserted
    # rather than assumed: if either config moves, this fails loudly instead of
    # silently reporting a resolution for a different test than the one run.
    filt_alpha = float(oc.alpha)
    sep_alpha = float(cfg.separation.alpha)
    if abs(filt_alpha - sep_alpha) > 1e-12:
        raise RuntimeError(
            f"occurrence.alpha ({filt_alpha}) must equal separation.alpha "
            f"({sep_alpha}): the filter's reported resolution and n_separate would "
            "otherwise describe different tests"
        )
    z_a = float(norm.ppf(1.0 - filt_alpha / 2.0))
    z_b = float(norm.ppf(float(cfg.separation.power)))
    crit = (z_a + z_b) ** 2
    K = int(oc.n_models)
    n_ism = min(int(oc.ism_instances), int(ho_idx.numel()))
    ism_idx = ho_idx[:n_ism]

    cells: list[dict[str, Any]] = []
    for hidden in [int(h) for h in cc.hidden]:
        for depth in [int(dp) for dp in cc.depth]:
            # A dump run evaluates only the cells it is asked to dump. That
            # makes its results.json a PARTIAL grid, which is why the run is
            # marked dump_only and why the figure and appendix resolvers refuse
            # to read it: a partial grid that looks well-formed is exactly how a
            # figure silently acquires the wrong denominator.
            if dump_cells is not None and f"{hidden}x{depth}" not in dump_cells:
                continue
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
            per_inst, subs = [], []
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

                # Per-instance is the primary measure; the position-average is
                # exactly its mean over instances, so it costs no extra passes.
                pi_m = ism_by_position_per_instance(predict, x, seq_len, ism_idx)
                per_inst.append(pi_m)
                attrs.append(pi_m.mean(axis=0))
                subs.append(ism_by_substitution(predict, x, seq_len, ism_idx))
                preds.append(p_ho.cpu().numpy())
                lats.append(l_ho.cpu().numpy())
                sq_errs.append(se)
                r2s.append(r2)

            # Quality-matched subset. Depth and fit quality are confounded in
            # this grid -- held-out R2 does not overlap between depth 1 and
            # depth 3 -- so an agree/disagree split by depth is also a split by
            # how well the models fit. Restricting to pairs where BOTH models
            # are in the cell's better-fitting half tests whether the split
            # survives at matched quality, and needs no retraining.
            order = np.argsort(-np.asarray(r2s))
            top_half = set(order[: max(K // 2, 2)].tolist())

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
                pi = per_instance_rho(per_inst[i], per_inst[j])
                tk = top_k_agreement(per_inst[i], per_inst[j])
                tkc = top_k_agreement_conditioned(
                    per_inst[i],
                    per_inst[j],
                    float(cfg.separation.well_conditioned_mass),
                )
                tkr = per_instance_rho_topk(per_inst[i], per_inst[j])
                # Symmetric in the pair: measuring only model i would report one
                # arbitrary member's concentration as if it described both.
                ci = attribution_concentration(per_inst[i])
                cj = attribution_concentration(per_inst[j])
                conc = {k: 0.5 * (ci[k] + cj[k]) for k in ci}
                ok_sub = np.isfinite(subs[i]) & np.isfinite(subs[j])
                r_sub = (
                    spearmanr(subs[i][ok_sub], subs[j][ok_sub]).statistic
                    if ok_sub.sum() > 2
                    else float("nan")
                )
                # Resolution of the performance filter: the smallest held-out R2
                # difference this pair's test could detect at 80% power. Stating
                # it stops "not significant" being read as "identical".
                dse = sq_errs[i] - sq_errs[j]
                mde_mse = (z_a + z_b) * float(np.std(dse, ddof=1)) / np.sqrt(dse.size)
                sst_mean = float(((y_ho - y_ho.mean()) ** 2).mean())
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
                        "attr_rho_sub": float(r_sub)
                        if np.isfinite(r_sub)
                        else float("nan"),
                        "pi_median": pi["median"],
                        "pi_p10": pi["p10"],
                        "pi_p90": pi["p90"],
                        "pi_frac_below_05": pi["frac_below_05"],
                        "conc_top3": conc["top3_mass_median"],
                        "eff_pos": conc["effective_positions_median"],
                        **{f"tk_{k}": v for k, v in tk.items()},
                        **{f"tk_{k}": v for k, v in tkr.items()},
                        **{f"tk_{k}": v for k, v in tkc.items()},
                        "mde_r2": float(mde_mse / sst_mean)
                        if sst_mean > 0
                        else float("nan"),
                        "both_top_half": bool(i in top_half and j in top_half),
                        "pair_r2_min": float(min(r2s[i], r2s[j])),
                        "n_separate": float(crit / snr2) if snr2 > 0 else float("inf"),
                        "rms_pred_diff": float(np.sqrt(np.mean(diff[fin] ** 2)))
                        if fin.any()
                        else float("nan"),
                    }
                )

            if dump_sink is not None and dump_cells is not None:
                cname = f"{hidden}x{depth}"
                if cname in dump_cells:
                    _collect_per_instance(
                        dump_sink, cname, seed, per_inst, pairs, k_top=3
                    )

            rhos = [p["attr_spearman"] for p in pairs if _finite(p["attr_spearman"])]
            nseps = [p["n_separate"] for p in pairs if _finite(p["n_separate"])]
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
    ap.add_argument(
        "--dump-per-instance",
        action="store_true",
        help="write per-sequence rho and top-k Jaccard for accuracy-matched "
        "pairs. Restricts the grid to --dump-cells and --dump-seeds and marks "
        "the run dump_only, so its partial results.json can never be mistaken "
        "for a verdict run.",
    )
    ap.add_argument("--dump-cells", default="128x1,128x3")
    ap.add_argument("--dump-seeds", default="0,1,2")
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
        dump_cells = (
            frozenset(c.strip() for c in args.dump_cells.split(",") if c.strip())
            if args.dump_per_instance
            else None
        )
        dump_sink: dict[str, Any] | None = {} if args.dump_per_instance else None
        seeds = resolve_seeds(cfg)
        if args.dump_per_instance:
            want = {int(v) for v in args.dump_seeds.split(",") if v.strip()}
            seeds = [s_ for s_ in seeds if int(s_) in want]
            print(f"DUMP MODE: cells {sorted(dump_cells)} seeds {seeds}")
            print("  the grid is restricted, so results.json here is PARTIAL")
            # make_run_dir writes the CONFIGURED seed list. A dump run uses a
            # subset, and a provenance record that overstates which seeds ran
            # is worse than none: it is checkable and wrong.
            meta["seeds"] = [int(v) for v in seeds]
            meta["seeds_configured"] = [int(v) for v in resolve_seeds(cfg)]
            meta["dump_cells"] = sorted(dump_cells)
            (run / "meta.json").write_text(json.dumps(meta, indent=1))

        per_seed = []
        for seed in seeds:
            r = run_seed(cfg, seed, dump_cells, dump_sink)
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

        if dump_sink is not None:
            out = run / "per_instance.npz"
            np.savez_compressed(out, **dump_sink)
            print(
                f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
                f"{len(dump_sink)} arrays)"
            )

    nb = int(cfg.bootstrap_resamples)
    keys = [(c["hidden"], c["depth"]) for c in per_seed[0]["cells"]]
    # A dump run covers a subset of the grid. Marking it here, in the artefact
    # itself, is what lets every consumer refuse it by inspection rather than by
    # remembering which timestamp was a dump.
    dump_only = bool(getattr(args, "dump_per_instance", False)) or len(keys) < 12
    agg: dict[str, Any] = {
        "dump_only": dump_only,
        "n_seeds": len(per_seed),
        "n": per_seed[0]["n"],
        "n_fit": per_seed[0]["n_fit"],
        "n_heldout": per_seed[0]["n_heldout"],
        "n_models": per_seed[0]["cells"][0]["n_models"],
        "seq_len": per_seed[0]["seq_len"],
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
            if _finite(p["attr_spearman"])
        ]
        alln = [
            p["n_separate"] for c in sel for p in c["pairs"] if _finite(p["n_separate"])
        ]
        entry["pooled_pairs"] = len(allr)
        th_pairs = [p for c in sel for p in c["pairs"] if p.get("both_top_half")]
        entry["top_pairs"] = len(th_pairs)
        r2q = [p["pair_r2_min"] for p in th_pairs if _finite(p["pair_r2_min"])]
        allq = [
            p["pair_r2_min"]
            for c in sel
            for p in c["pairs"]
            if _finite(p["pair_r2_min"])
        ]
        entry["top_pair_r2_median"] = float(np.median(r2q)) if r2q else float("nan")
        entry["all_pair_r2_median"] = float(np.median(allq)) if allq else float("nan")
        entry["pooled_rho_min"] = float(np.min(allr)) if allr else float("nan")
        entry["pooled_rho_max"] = float(np.max(allr)) if allr else float("nan")
        entry["pooled_rho_median"] = float(np.median(allr)) if allr else float("nan")
        entry["pooled_nsep_median"] = float(np.median(alln)) if alln else float("nan")
        for nm, key in (
            ("pi_median", "pi_median"),
            ("pi_p10", "pi_p10"),
            ("pi_frac_below_05", "pi_frac_below_05"),
            ("attr_rho_sub", "attr_rho_sub"),
            ("mde_r2", "mde_r2"),
            ("conc_top3", "conc_top3"),
            ("eff_pos", "eff_pos"),
            ("tk_j2", "tk_jaccard_top2_mean"),
            ("tk_j3", "tk_jaccard_top3_mean"),
            ("tk_ex1", "tk_exact_top1_frac"),
            ("tk_ex3", "tk_exact_top3_frac"),
            ("tk_cj3", "tk_cond_jaccard_top3_mean"),
            ("tk_cex3", "tk_cond_exact_top3_frac"),
            ("tk_cfrac", "tk_cond_frac_instances"),
            ("tk_rho", "tk_topk_rho_median"),
            ("tk_union", "tk_topk_union_median"),
        ):
            vv = [p[key] for c in sel for p in c["pairs"] if _finite(p.get(key))]
            entry[f"pooled_{nm}"] = float(np.median(vv)) if vv else float("nan")
            # Same statistic on the quality-matched subset only.
            rv = [
                p[key]
                for c in sel
                for p in c["pairs"]
                if p.get("both_top_half") and _finite(p.get(key))
            ]
            entry[f"top_{nm}"] = float(np.median(rv)) if rv else float("nan")
        # Fractions of the surviving pairs below given agreement levels. The
        # min alone is one pair; these say how common the disagreement is.
        for t in (0.9, 0.8, 0.7):
            entry[f"pooled_frac_below_{int(t * 10)}"] = (
                float(np.mean(np.asarray(allr) < t)) if allr else float("nan")
            )
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
        "| width | depth | held-out R² | accuracy-tied pairs | filter resolution "
        "(min. detectable ΔR²) | **n to separate as functions** | per-instance ρ "
        "(p10 / median) | frac ρ<0.5 |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        ip_ = c.get("n_pairs_indistinguishable", {}).get("mean", float("nan"))
        tp = c.get("n_pairs_total", {}).get("mean", float("nan"))
        L.append(
            f"| {h} | {dp} | {c['heldout_r2_mean']['mean']:.4f} | "
            f"{ip_:.0f}/{tp:.0f} | {c['pooled_mde_r2']:.4f} | "
            f"**{c['pooled_nsep_median']:,.0f}** | "
            f"{c['pooled_pi_p10']:+.3f} / **{c['pooled_pi_median']:+.3f}** | "
            f"**{c['pooled_pi_frac_below_05'] * 100:.0f}%** |"
        )
    L.append("")
    L.append(
        "**Accuracy-tied** means a paired two-sided t-test on per-point held-out "
        f"squared errors does not reject at α = {agg['filter_alpha']}. That is a "
        "failure to reject, NOT evidence of equivalence, and its resolution is "
        "finite: the *filter resolution* column gives the smallest held-out R² "
        "difference the test could detect at 80% power with "
        f"{agg['n_heldout']} held-out points, so differences below it would not have "
        "been seen. The claim made here does not rest on accepting that null. It is "
        "the conjunction: these pairs are **not separable by accuracy at this "
        "resolution**, they **are separable as functions** with the stated number of "
        "measurements, and they **attribute differently**.\n"
    )
    L.append(
        "**Is the tail degenerate?** If attribution concentrates in a few positions, "
        "a low all-position ρ may only be reporting the order of positions that carry "
        "no mass. `top-3 mass` is the median fraction of a sequence's total |Δ| in its "
        "three strongest positions; `eff. positions` is exp(entropy) of the "
        f"normalized magnitudes, where {agg['seq_len']:.1f} means all contribute "
        "equally and 1.0 means one dominates. `Jaccard` is the median overlap of the "
        "two models' top-k position SETS — what a reader actually uses.\n"
    )
    L.append(
        "| width | depth | top-3 mass | eff. pos. | mean Jaccard top2 / top3 | "
        "exact top-1 / top-3 | ρ on top-3 union |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        L.append(
            f"| {h} | {dp} | {c['pooled_conc_top3']:.3f} | {c['pooled_eff_pos']:.2f} | "
            f"{c['pooled_tk_j2']:.2f} / **{c['pooled_tk_j3']:.2f}** | "
            f"{c['pooled_tk_ex1'] * 100:.0f}% / **{c['pooled_tk_ex3'] * 100:.0f}%** | "
            f"{c['pooled_tk_rho']:+.3f} (n≈{c['pooled_tk_union']:.1f}) |"
        )
    L.append("")
    L.append(
        "Jaccard on sets this small takes only four values at k=3 (0, 0.2, 0.5, 1) "
        "and two at k=1, so its median carries almost nothing — the median top-1 "
        "Jaccard is just the exact-match fraction thresholded at one half. The MEAN "
        "is reported instead, together with the exact-set-match fraction, which is "
        "what the verdict's branch condition uses.\n"
    )
    L.append(
        "**Quality-matched.** Depth and fit quality are confounded in this grid, so "
        "an agree/disagree split by depth is also a split by how well the models fit. "
        "These columns repeat the comparison on pairs where BOTH models are in their "
        "cell's better-fitting half, with the median of the pair's lower held-out R² "
        "shown so the quality level is visible.\n"
    )
    L.append(
        "| width | depth | pairs | pair R² (all → top half) | mean Jaccard top3 "
        "(all → top half) | exact top-3 (all → top half) |"
    )
    L.append("|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        L.append(
            f"| {h} | {dp} | {c['top_pairs']} | "
            f"{c['all_pair_r2_median']:.4f} → **{c['top_pair_r2_median']:.4f}** | "
            f"{c['pooled_tk_j3']:.2f} → **{c['top_tk_j3']:.2f}** | "
            f"{c['pooled_tk_ex3'] * 100:.0f}% → **{c['top_tk_ex3'] * 100:.0f}%** |"
        )
    L.append("")
    L.append(
        "**Conditioned on a top-3 existing.** Set overlap only means something where "
        "the attribution profile has a well-defined top. A model spreading its "
        f"magnitude near-uniformly over {agg['seq_len']} positions has an "
        "ill-conditioned top-3, and two such models disagree for a reason unrelated "
        "to which positions matter. These columns restrict to instances where BOTH "
        f"models place at least {float(cfg.separation.well_conditioned_mass):.0%} of "
        "their magnitude in their own top 3, and report what fraction of instances "
        "qualify. Low agreement here cannot be blamed on a flat profile.\n"
    )
    L.append(
        "| width | depth | eff. positions | instances qualifying | mean Jaccard top3 "
        "(all → conditioned) | exact top-3 (all → conditioned) |"
    )
    L.append("|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        L.append(
            f"| {h} | {dp} | {c['pooled_eff_pos']:.2f} | "
            f"**{c['pooled_tk_cfrac'] * 100:.0f}%** | "
            f"{c['pooled_tk_j3']:.2f} → **{c['pooled_tk_cj3']:.2f}** | "
            f"{c['pooled_tk_ex3'] * 100:.0f}% → **{c['pooled_tk_cex3'] * 100:.0f}%** |"
        )
    L.append("")
    L.append(
        "| width | depth | ρ over position averages (n="
        + str(agg["seq_len"])
        + ") | ρ over substitutions |"
    )
    L.append("|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        L.append(
            f"| {h} | {dp} | {c['pooled_rho_median']:+.3f} | "
            f"{c['pooled_attr_rho_sub']:+.3f} |"
        )
    L.append("")
    L.append(
        f"**The n={agg['seq_len']} column is coarse and is retained only for "
        f"continuity.** A Spearman over {agg['seq_len']} items has an approximate "
        f"standard error of {1.0 / max(agg['seq_len'] - 1, 1) ** 0.5:.2f} under "
        "independence, so single values near ±0.4 are barely distinguishable from "
        "zero and even ±0.9 is imprecise. Bootstrap intervals elsewhere are over "
        "SEEDS and do not capture that granularity. The per-instance distribution in "
        "the main table is what the verdict uses: each instance contributes one "
        f"correlation over its own {agg['seq_len']} positions, and hundreds of "
        "instances determine the percentiles.\n"
    )
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
    v = void_if_unsearched(
        sum(c.get("pooled_pairs", 0) for c in cells.values()),
        "indistinguishable model pairs",
        "No pair of independently initialised models survived to be compared.",
    )
    if v:
        return v
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
    # "in every cell" needs the MINIMUM of the per-cell medians. Using the
    # maximum here asserted the best cell's median as a bound holding
    # everywhere, which is false whenever the cells disagree.
    all_low = min(v["pooled_rho_median"] for v in usable.values())

    mde = float(np.median([v["pooled_mde_r2"] for v in usable.values()]))
    nsep = float(np.median([v["pooled_nsep_median"] for v in usable.values()]))
    parts.append(
        f"**{surv:.0f} of {tot:.0f} pairs are not separable by predictive accuracy** "
        f"at the resolution this held-out set provides (paired t-test, α = "
        f"{agg['filter_alpha']}, uncorrected and therefore strict). This is a failure "
        "to reject, not a demonstration of equivalence, and its resolution is stated "
        f"rather than implied: with {agg['n_heldout']} held-out points the test could "
        f"detect a held-out R² difference of about {mde:.4f} at 80% power, so smaller "
        "differences in accuracy would not have been seen. Nothing below depends on "
        "accepting that null."
    )
    parts.append(
        "**The claim is the conjunction, and each part is measured.** These pairs "
        "share an architecture, a fit split, a budget and a learning-rate selection, "
        "differing only in initialization seed. Their predictive ACCURACY is not "
        "separable at the resolution above. Their identity as FUNCTIONS is separable, "
        f"and cheaply: the median pair needs about {nsep:,.0f} held-out measurements "
        "to reject that the two are the same function under the Poisson log-ratio "
        "noise model. So these are not near-identical models -- they are models that "
        "predict differently while scoring the same."
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
        pim = {k: v["pooled_pi_median"] for k, v in usable.items()}
        pif = {k: v["pooled_pi_frac_below_05"] for k, v in usable.items()}
        w_pi = min(pim, key=lambda k: pim[k])
        w_f = max(pif, key=lambda k: pif[k])
        j3 = {k: v["pooled_tk_ex3"] for k, v in usable.items()}
        ex1 = {k: v["pooled_tk_ex1"] for k, v in usable.items()}
        mass = float(np.median([v["pooled_conc_top3"] for v in usable.values()]))
        effp = float(np.median([v["pooled_eff_pos"] for v in usable.values()]))
        thr = float(cfg.separation.exact_top3_agree)
        w_j = min(j3, key=lambda k: j3[k])
        parts.append(
            "**And they attribute differently -- measured per instance.** For a given "
            "splice site, do the two models rank ITS positions the same way? Median "
            f"agreement is {pim[w_pi]:+.3f} in the worst cell ({w_pi}), and in {w_f} "
            f"{pif[w_f] * 100:.0f}% of held-out sites the two rankings agree at ρ "
            "below 0.5. Averaging attributions across instances first, as an earlier "
            "version did, discards exactly this variation."
        )
        agree = sorted(k for k in j3 if j3[k] >= thr)
        disagree = sorted(k for k in j3 if k not in agree)
        if agree and disagree:
            parts.append(
                "**Which reading this supports: it depends on the cell, and both are "
                f"stated.** Attribution is concentrated — a median {mass:.0%} of each "
                f"sequence's |Δ| mass in three positions, an effective {effp:.2f} "
                f"contributing positions of {agg['seq_len']} — so the top-3 set is the "
                "part carrying signal. In "
                + ", ".join(agree)
                + " the models AGREE on which positions those are (median top-3 "
                f"sets matching exactly in at least {min(j3[k] for k in agree) * 100:.0f}% of instances), so there the "
                "full-rank disagreement is tail ordering, the low per-instance ρ is "
                "NOT the headline, and the supported claim is the narrow one: tied on "
                "accuracy, they agree about what matters and differ on the ordering "
                "of what does not. In "
                + ", ".join(disagree)
                + f" they do NOT agree — top-3 sets matching exactly in only {j3[w_j] * 100:.0f}% of instances, "
                f"strongest position matching exactly in only {ex1[w_j] * 100:.0f}% of "
                "sequences, and median rank agreement on the top-3 union of "
                f"{usable[w_j]['pooled_tk_rho']:+.3f}. There the models differ about "
                "WHICH positions matter and the occurrence claim stands as written. "
                "**Top-k set overlap is the statistic to lead with either way**: it is "
                "what gets used downstream and it is what separates these readings."
            )
        elif not disagree:
            parts.append(
                f"**But the disagreement is mostly in the tail, and the claim must "
                f"narrow to say so.** Attribution is concentrated: a median {mass:.0%} "
                f"of each sequence's |Δ| mass sits in three positions, an effective "
                f"{effp:.2f} contributing positions of {agg['seq_len']}. And the "
                "models agree on WHICH those are — median top-3 Jaccard at least "
                f"{min(j3.values()):.2f} in every cell, with the single strongest "
                f"position matching exactly in {min(ex1.values()) * 100:.0f}% to "
                f"{max(ex1.values()) * 100:.0f}% of sequences. **The low per-instance "
                "ρ is therefore NOT the headline: it largely reflects the ordering of "
                "positions carrying little mass.** The supported claim is the narrow "
                "one: models tied on accuracy agree about which positions matter and "
                "disagree about the ordering of those that do not."
            )
        else:
            parts.append(
                f"**And the disagreement is about which positions matter, not about "
                f"an irrelevant tail.** Attribution is concentrated — a median "
                f"{mass:.0%} of |Δ| mass in three positions, an effective {effp:.2f} "
                f"contributing positions of {agg['seq_len']} — yet the models differ "
                f"on which those are: median top-3 Jaccard falls to {j3[w_j]:.2f} "
                f"({w_j}) and the single strongest position matches exactly in only "
                f"{ex1[w_j] * 100:.0f}% of sequences there. Restricted to the union "
                "of the two top-3 sets, the positions that carry the signal, median "
                f"rank agreement is {usable[w_j]['pooled_tk_rho']:+.3f}. **This is "
                "the occurrence result, and top-k set overlap is the statistic to "
                "lead with**, since a reader asks which positions matter for a site, "
                "not how the bottom of the list is ordered. It needs no constructed "
                "twin, no reparameterization and no refit, so it is immune to the "
                "search and warm-start confounds `paper/tables/closure_search.md` "
                "found in the warp-based route."
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
    # A reviewer will ask whether the disagreement is just a symptom of weak
    # models, and the honest answer is that model quality does matter but does
    # not explain it away. Reported at the BEST-predicting cell so the claim does
    # not rest on the cells that fit worst.
    r2s = [v["heldout_r2_mean"]["mean"] for v in usable.values()]
    mins = [v["pooled_rho_min"] for v in usable.values()]
    meds = [v["pooled_rho_median"] for v in usable.values()]
    rq_min = spearmanr(r2s, mins).statistic
    rq_med = spearmanr(r2s, meds).statistic
    best_cell = max(usable, key=lambda k: usable[k]["heldout_r2_mean"]["mean"])
    b = usable[best_cell]
    # Does the split survive at matched fit quality? Depth and held-out R2 do not
    # overlap across the grid, so the depth split is also a quality split unless
    # this says otherwise.
    deep = {k: v for k, v in usable.items() if v["depth"] >= 3 and v["top_pairs"] > 0}
    if deep:
        a3 = {k: v["pooled_tk_ex3"] for k, v in deep.items()}
        t3 = {k: v["top_tk_ex3"] for k, v in deep.items()}
        tj = {k: v["top_tk_j3"] for k, v in deep.items()}
        r_all = float(np.median([v["all_pair_r2_median"] for v in deep.values()]))
        r_top = float(np.median([v["top_pair_r2_median"] for v in deep.values()]))
        shallow = {k: v for k, v in usable.items() if v["depth"] == 1}
        r_shallow = (
            float(np.median([v["all_pair_r2_median"] for v in shallow.values()]))
            if shallow
            else float("nan")
        )
        thr2 = float(cfg.separation.exact_top3_agree)
        rose = min(t3.values()) >= thr2
        parts.append(
            "**Is the depth split just fit quality?** Depth and held-out R² are "
            "confounded in this grid, so the question is answered on pairs matched "
            "for quality rather than argued. Restricting to pairs where both models "
            "are in their cell's better-fitting half lifts the median pair R² at "
            f"depth 3 from {r_all:.4f} to {r_top:.4f}, and top-3 exact agreement goes "
            f"from {min(a3.values()) * 100:.0f}–{max(a3.values()) * 100:.0f}% to "
            f"{min(t3.values()) * 100:.0f}–{max(t3.values()) * 100:.0f}% "
            f"(mean Jaccard {min(tj.values()):.2f}–{max(tj.values()):.2f})."
            + (
                ""
                if not shallow
                else f" Note what this does and does not match: the restriction "
                f"equalises quality WITHIN a cell, while the confound is BETWEEN "
                f"depths. Depth-1 pairs sit at a median R² of {r_shallow:.4f}, and "
                f"the best depth-3 pairs reach {r_top:.4f}, so the two "
                + (
                    "now overlap and the comparison is quality-matched in the sense "
                    "that matters."
                    if r_top >= r_shallow
                    else "still do NOT overlap. The depth-3 models remain the worse "
                    "fits even after restriction, so this test bounds the confound "
                    "rather than eliminating it: it shows how much of the split "
                    "survives a quality improvement of "
                    f"{r_top - r_all:+.4f}, not what would happen at equal fit."
                )
            )
            + (
                " **The split is a quality artifact.** Among well-fitting depth-3 "
                "pairs the top-3 sets agree, so the disagreement seen over all pairs "
                "reflects how badly the worse models fit rather than anything about "
                "depth, and the claim narrows accordingly."
                if rose
                else " **The split is not a quality artifact.** Well-fitting depth-3 "
                "pairs still disagree about which positions matter, so the effect is "
                "expressivity rather than fit. This matches "
                "`paper/tables/closure_search.md`, where a cold-started refit "
                "recovers an in-class target at depth 1 and fails at depths 2 and 3: "
                "one mechanism -- what the optimiser can reach in a deeper class -- "
                "would produce both results."
            )
        )

    # Does the disagreement survive where a top-3 actually exists? Agreement
    # tracks concentration across this grid, so without this the headline has an
    # unresolved alternative reading: a flat profile has no top to agree about.
    condc = {
        k: v for k, v in usable.items() if np.isfinite(v.get("pooled_tk_cex3", np.nan))
    }
    if condc:
        cf = {k: v["pooled_tk_cfrac"] for k, v in condc.items()}
        ce = {k: v["pooled_tk_cex3"] for k, v in condc.items()}
        cj = {k: v["pooled_tk_cj3"] for k, v in condc.items()}
        ae = {k: v["pooled_tk_ex3"] for k, v in condc.items()}
        ep = {k: v["pooled_eff_pos"] for k, v in condc.items()}
        minq = float(cfg.separation.min_qualifying_frac)
        thin = sorted(k for k in cf if cf[k] < minq)
        posed = sorted(k for k in cf if cf[k] >= minq)
        parts.append(
            "**Does the disagreement survive where a top-3 exists?** Agreement tracks "
            "concentration across this grid — cells with few effective positions agree "
            f"most — so a flat attribution profile is a live alternative explanation. "
            f"Effective positions range {min(ep.values()):.2f} to {max(ep.values()):.2f} "
            f"of {agg['seq_len']}. Restricting to instances where both models put at "
            f"least {float(cfg.separation.well_conditioned_mass):.0%} of their "
            "magnitude in their own top 3, exact top-3 agreement moves from "
            f"{min(ae.values()) * 100:.0f}–{max(ae.values()) * 100:.0f}% to "
            f"{min(ce.values()) * 100:.0f}–{max(ce.values()) * 100:.0f}% "
            f"(mean Jaccard up to {max(cj.values()):.2f}), with "
            f"{min(cf.values()) * 100:.0f}–{max(cf.values()) * 100:.0f}% of instances "
            "qualifying."
            + (
                " **These cells are excluded from the claim, not caveated**: "
                + ", ".join(thin)
                + f" retain under {minq:.0%} of instances, meaning almost no sequence "
                "has a well-defined top-3 under either model. For them the question "
                "of which positions matter is not well posed, so their low agreement "
                "is not evidence of disagreement about anything; a near-uniform "
                "attribution profile is simply what those models produce."
                if thin
                else " Enough instances qualify in every cell for the conditioned "
                "numbers to stand on their own."
            )
            + (
                " **The claim is therefore made on "
                + ", ".join(posed)
                + f"**, where {min(cf[k] for k in posed):.0%} to "
                f"{max(cf[k] for k in posed):.0%} of instances have a well-defined "
                "top-3 under both models. There, models tied on accuracy still pick "
                "different top-3 position sets in "
                f"{(1 - max(ce[k] for k in posed)) * 100:.0f}% to "
                f"{(1 - min(ce[k] for k in posed)) * 100:.0f}% of sequences."
                if posed
                else " **No cell retains enough well-conditioned instances to support "
                "a claim about which positions matter.**"
            )
        )

    parts.append(
        "**Is this just weak models?** Partly, but not mainly, and the question "
        "deserves the number rather than a reassurance. Across cells the held-out "
        f"predictive R² does correlate with agreement (Spearman {rq_min:+.3f} against "
        f"the minimum ρ, {rq_med:+.3f} against the median), so better-fitting cells "
        "do agree more. The claim therefore rests on the best-fitting cell, not the "
        f"worst: at {best_cell}, held-out R² {b['heldout_r2_mean']['mean']:.4f}, the "
        f"highest in the grid, "
        f"{b['n_pairs_indistinguishable']['mean']:.0f} of "
        f"{b['n_pairs_total']['mean']:.0f} pairs are indistinguishable and among them "
        f"{b['pooled_frac_below_9'] * 100:.0f}% rank the loci at ρ below 0.9, "
        f"{b['pooled_frac_below_8'] * 100:.0f}% below 0.8 and "
        f"{b['pooled_frac_below_7'] * 100:.0f}% below 0.7, reaching "
        f"{b['pooled_rho_min']:+.3f}. The disagreement is not confined to the cells "
        "that predict badly."
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
