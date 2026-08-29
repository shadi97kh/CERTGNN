"""Does a high-capacity G-P map re-represent monotone warps only, or anything?

`paper/tables/closure_capacity.md` found that closure reaches 1.000000 only at
128x2 and 128x3. That result was first read as interpolation, on the grounds
that those cells carry more parameters than datapoints. **The table contradicts
that reading.** 128x1 sits at 1.22 parameters per datapoint, above the same
boundary, yet its gap is 1.18e-04, four orders of magnitude short of 128x2's
1.25e-08, and no better than 16x3 at 0.29 parameters per datapoint. Sorted by
depth the picture is clean instead:

    gap        w=16      w=32      w=64      w=128
    depth 1  1.21e-02  6.24e-03  1.23e-03  1.18e-04   <- never beats 1.2e-04
    depth 2  7.18e-04  2.56e-03  2.70e-06  1.25e-08
    depth 3  2.65e-04  4.75e-04  2.78e-06  2.58e-10

Depth governs and width amplifies within depth; parameters per datapoint cuts
across both, which is why a power law fitted in parameter count degenerated
with its exponent pinned at the bound. So the interpolation claim cannot be
inferred from a ratio. It has to be measured.

This experiment measures it. For every cell the same reference map is refit to
four targets under the protocol of `closure_capacity.py` -- per-cell learning
rate from the pilot, warm start from the reference weights, best iterate, the
same epoch budget:

- **warp**: a monotone reparameterization of the fitted latent. The existing
  measurement, and the only target a closed model class must re-represent.
- **shuffled**: a random permutation of the fitted latent values across
  instances. Same marginal distribution, non-monotone, and carrying no relation
  whatsoever to the input.
- **noise**: Gaussian with the fitted latent's mean and variance. No relation to
  the input and not even the latent's own empirical distribution.
- **null**: the fitted latent itself, at zero warp. The pipeline's numerical
  floor, present in every cell.

What this measures, and what it does NOT. If a cell fits the shuffled and noise
targets as well as the warp, then on the observed points it can fit essentially
anything, and an in-sample closure number from that cell cannot distinguish a
class that CONTAINS the reparameterized twin from one that memorizes n values
at n points.

**It does not follow that such a cell fails to be closed.** Closure under
monotone reparameterization is a claim about what the function class contains.
A universal approximator contains psi(phi) AND fits permutations; the second is
not evidence against the first. So a cell that fits everything is uninformative
here, not disconfirming, and a grid in which no cell is both closed and
selective is the expected behaviour of a flexible neural class rather than a
finding about closure. An earlier version of this file's verdict read the
disjointness as evidence against closure, which was wrong.

Decisive contrast: at 128x2 and 128x3, is the shuffled R2 near 1 or near 0?

- **Near 1.** In-sample closure at those cells carries no information either
  way, because the same fit succeeds on targets that cannot be learned at all.
  The question moves to held-out generalization.
- **Near 0, while the warp target still reaches 1e-10.** Their in-sample
  closure is at least not explained by an ability to fit anything.

Either way this experiment does not settle identifiability. Containment implies
the twin predicts on points the refit never saw, and memorization does not, so
the discriminating measurement is `experiments/closure_heldout.py`.

128x1 is the counterexample that separates the readings, and is reported
explicitly. It is above the parameters-per-datapoint boundary but far from
closure. If it fails the shuffled target too, then that ratio is not the
governing variable and the boundary reading is dead independently of the rest.

Usage:
    python -m experiments.closure_shuffled --config configs/base.yaml
    python -m experiments.closure_shuffled --retable results/runs/<dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

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
from experiments.identifiability_probe import (
    _standardize,
    affine_r2,
    is_monotone,
    warp,
)

TARGETS = ("warp", "shuffled", "noise", "null")


def build_targets(
    z_ref: torch.Tensor, strength: float, omega: float, seed: int, gen: torch.Generator
) -> dict[str, torch.Tensor]:
    """The four refit targets, all on the reference latent's own scale.

    `shuffled` and `noise` share the latent's first two moments so that a
    failure to re-represent them cannot be blamed on a scale mismatch; what
    differs is that neither is a monotone function of the latent, and neither
    carries any relation to the input.
    """
    dev = z_ref.device
    zc = z_ref.detach().cpu()
    mu, sd = float(zc.mean()), float(zc.std())
    perm = torch.randperm(zc.numel(), generator=gen)
    return {
        "warp": warp(zc, FAMILY, strength, omega, seed).to(dev),
        "shuffled": zc[perm].to(dev),
        "noise": (torch.randn(zc.shape, generator=gen, dtype=zc.dtype) * sd + mu).to(
            dev
        ),
        "null": z_ref.detach().clone(),
    }


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    cc, ip = cfg.closure_capacity, cfg.identifiability
    cs = cfg.closure_shuffled
    dev = pick_device(cfg)
    x, y, _ = load_mpsa_capped(cfg, seed)
    x, y = x.to(dev), y.to(dev)
    d = x.shape[1]
    strength = float(cs.strength)
    omega = float(cc.omega)
    if not is_monotone(FAMILY, strength, omega, seed):
        raise RuntimeError(
            f"warp at strength {strength}, omega {omega}, seed {seed} is not "
            "monotone; the warp target would not be a reparameterization"
        )
    cells: list[dict[str, Any]] = []

    for hidden in [int(h) for h in cc.hidden]:
        for depth in [int(dp) for dp in cc.depth]:
            gen_seed = seed * 7919 + hidden * 31 + depth
            torch_seed = seed * 100 + hidden + depth
            cell_lr = select_lr(
                d,
                hidden,
                depth,
                int(ip.ge_components),
                gen_seed,
                torch_seed,
                x,
                y,
                [float(v) for v in cc.lr_ladder],
                int(cc.pilot_epochs),
            )
            gen = torch.Generator().manual_seed(gen_seed)
            torch.manual_seed(torch_seed)
            ref = CapacityModel(d, hidden, depth, int(ip.ge_components), gen).to(dev)
            train(ref, x, y, int(cc.ref_epochs), cell_lr)
            z_ref = _standardize(ref.latent(x).detach())
            with torch.no_grad():
                pred = ref.g(ref.latent(x)).detach()
            fit_r2 = 1.0 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

            tgen = torch.Generator().manual_seed(gen_seed + 104729)
            targets = build_targets(z_ref, strength, omega, seed, tgen)
            r2: dict[str, float] = {}
            for name in TARGETS:
                t = targets[name]
                phi_t, _m = refit_to(
                    t,
                    x,
                    ref.phi,
                    d,
                    hidden,
                    depth,
                    gen,
                    int(cc.refit_epochs),
                    cell_lr,
                    float(ip.radius.fit_tol),
                )
                r2[name] = affine_r2(phi_t.cpu().numpy(), t.cpu().numpy())

            cells.append(
                {
                    "hidden": hidden,
                    "depth": depth,
                    "n_params": n_params(ref.phi),
                    "fit_r2": fit_r2,
                    "lr": cell_lr,
                    "degenerate": bool(
                        not np.isfinite(fit_r2) or fit_r2 < float(cc.min_fit_r2)
                    ),
                    **{f"r2_{k}": v for k, v in r2.items()},
                    # The dissociation itself: how much better the class does on
                    # a monotone warp than on a target with no relation to input.
                    "selectivity": r2["warp"] - max(r2["shuffled"], r2["noise"]),
                }
            )
    return {"seed": seed, "cells": cells, "n": int(x.shape[0])}


def governing_variable(agg: dict[str, Any]) -> dict[str, Any]:
    """Which capacity variable orders the closure gap: depth, width, or ratio?

    Two measures, because Spearman alone is unfair here. Depth takes 3 distinct
    values and width 4, while parameters per datapoint takes a different value
    in all 12 cells; Spearman penalises ties, so the finest-grained variable
    wins by construction rather than by governing anything.

    The primary measure is therefore the DISCORDANCE RATE: over all cell pairs
    that the variable actually distinguishes (ties excluded, so granularity
    costs nothing), the fraction where more capacity goes with a LARGER gap. A
    variable that governs has a discordance near 0. Spearman is still reported
    alongside, with its bias stated.
    """
    cells = list(agg["cells"].values())
    gap = np.array([max(1.0 - c["r2_warp"]["mean"], 1e-12) for c in cells])
    lg = np.log10(gap)
    variables = {
        "depth": np.array([c["depth"] for c in cells], dtype=float),
        "width": np.array([c["hidden"] for c in cells], dtype=float),
        "params_per_point": np.array([c["params_per_point"] for c in cells]),
    }
    out: dict[str, Any] = {"spearman": {}, "discordance": {}, "n_pairs": {}}
    for name, v in variables.items():
        r = spearmanr(v, lg).statistic
        out["spearman"][name] = float(r) if np.isfinite(r) else float("nan")
        disc = tot = 0
        for i in range(len(v)):
            for j in range(len(v)):
                if v[i] < v[j]:
                    tot += 1
                    if gap[i] < gap[j]:  # more capacity, larger gap
                        disc += 1
        out["discordance"][name] = disc / tot if tot else float("nan")
        out["n_pairs"][name] = tot

    out["min_gap_by_depth"] = {
        str(d): float(min(1.0 - c["r2_warp"]["mean"] for c in cells if c["depth"] == d))
        for d in sorted({c["depth"] for c in cells})
    }
    out["min_gap_by_width"] = {
        str(w): float(
            min(1.0 - c["r2_warp"]["mean"] for c in cells if c["hidden"] == w)
        )
        for w in sorted({c["hidden"] for c in cells})
    }

    def monotone(d: dict[str, float]) -> bool:
        vals = [d[k] for k in sorted(d, key=float)]
        return all(b <= a for a, b in zip(vals, vals[1:]))

    out["min_monotone_in_depth"] = monotone(out["min_gap_by_depth"])
    out["min_monotone_in_width"] = monotone(out["min_gap_by_width"])

    ranked = sorted(variables, key=lambda k: out["discordance"][k])
    out["strongest"] = ranked[0]
    best, second = out["discordance"][ranked[0]], out["discordance"][ranked[1]]
    # "Clean" means few violations AND a clear margin over the runner-up.
    out["clean"] = bool(best <= 0.10 and (second - best) >= 0.05)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument(
        "--retable",
        metavar="RUNDIR",
        help="rebuild the aggregate and table from an existing run's seed JSONs",
    )
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
        run = make_run_dir(cfg, "closure_shuffled", allow_dirty=args.allow_dirty)
        meta = json.loads((run / "meta.json").read_text())
        print(f"run dir: {run}")
        per_seed = []
        for seed in resolve_seeds(cfg):
            r = run_seed(cfg, seed)
            per_seed.append(r)
            w = [c for c in r["cells"] if c["hidden"] == 128 and c["depth"] == 3]
            msg = (
                f" 128x3 warp {w[0]['r2_warp']:.6f} shuffled {w[0]['r2_shuffled']:.4f}"
                if w
                else ""
            )
            print(f"seed {seed}: {len(r['cells'])} cells;{msg}", flush=True)
            (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    nb = int(cfg.bootstrap_resamples)
    keys = [(c["hidden"], c["depth"]) for c in per_seed[0]["cells"]]
    agg: dict[str, Any] = {"n_seeds": len(per_seed), "n": per_seed[0]["n"], "cells": {}}
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
            "params_per_point": sel[0]["n_params"] / float(per_seed[0]["n"]),
            "fit_r2": mean_ci([c["fit_r2"] for c in sel], n_boot=nb),
            "selectivity": mean_ci([c["selectivity"] for c in sel], n_boot=nb),
            "n_degenerate": sum(1 for c in sel if c.get("degenerate", False)),
        }
        for t in TARGETS:
            entry[f"r2_{t}"] = mean_ci([c[f"r2_{t}"] for c in sel], n_boot=nb)
        agg["cells"][f"{h}x{dp}"] = entry

    agg["governing"] = governing_variable(agg)

    # The decisive cells, named in the experiment's own terms.
    def cell(k: str) -> dict[str, Any]:
        return agg["cells"][k]

    hi_thresh, lo_thresh = (
        float(cfg.closure_shuffled.interpolating_r2),
        float(cfg.closure_shuffled.selective_r2),
    )
    closed = [k for k, v in agg["cells"].items() if 1.0 - v["r2_warp"]["mean"] < 5e-7]
    interpolating = [
        k for k, v in agg["cells"].items() if v["r2_shuffled"]["mean"] >= hi_thresh
    ]
    selective = [
        k
        for k, v in agg["cells"].items()
        if v["r2_warp"]["mean"] > 0.99 and v["r2_shuffled"]["mean"] <= lo_thresh
    ]
    agg["closed_cells"] = sorted(closed)
    agg["interpolating_cells"] = sorted(interpolating)
    agg["selective_cells"] = sorted(selective)

    L = ["# Does high capacity re-represent monotone warps only, or anything?\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}"
        + (
            f", **aggregated post hoc by `{git_sha()}` via --retable**"
            if retabled
            else ""
        )
        + f", config `{meta['config_hash']}`, {agg['n_seeds']} seeds, mean "
        f"[95% bootstrap CI]. {agg['n']} real BRCA2 5' splice sites per seed; warp "
        f"family {FAMILY} at strength {float(cfg.closure_shuffled.strength)}; refit "
        f"{int(cfg.closure_capacity.refit_epochs):,} epochs, warm-started, best "
        "iterate, per-cell learning rate from the pilot.\n"
    )
    L.append(
        "`warp` is a monotone reparameterization of the fitted latent, the only "
        "target a closed model class must re-represent. `shuffled` is a random "
        "permutation of the same values and `noise` is Gaussian with the same mean "
        "and variance; neither is monotone in the latent and neither carries any "
        "relation to the input. `null` is the latent itself, the numerical floor. "
        "**Selectivity** is `warp` minus the better of `shuffled` and `noise`: it is "
        "high for a class that re-represents reparameterizations specifically, and "
        "near zero for a map that fits anything.\n"
    )
    L.append(
        "| width | depth | params | params/point | R² warp | R² shuffled | R² noise | "
        "R² null | selectivity |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for h, dp in keys:
        c = cell(f"{h}x{dp}")
        flag = f" ⚠️{c['n_degenerate']}/{agg['n_seeds']}" if c["n_degenerate"] else ""
        L.append(
            f"| {h} | {dp} | {c['n_params']:,} | {c['params_per_point']:.2f} | "
            f"{c['r2_warp']['mean']:.6f}{flag} | "
            f"{c['r2_shuffled']['mean']:.4f} "
            f"[{c['r2_shuffled']['lo']:.4f}, {c['r2_shuffled']['hi']:.4f}] | "
            f"{c['r2_noise']['mean']:.4f} | {c['r2_null']['mean']:.6f} | "
            f"{c['selectivity']['mean']:+.4f} |"
        )
    L.append("")

    g = agg["governing"]
    L.append("## Which variable governs closure\n")
    L.append(
        "Two measures over all "
        + str(len(keys))
        + " cells. **Discordance** is the fraction of cell pairs the variable "
        "distinguishes in which MORE capacity goes with a LARGER gap; near 0 means "
        "the variable governs. Ties are excluded, so a coarse variable is not "
        "penalised. Spearman against log₁₀ of the gap is shown alongside, but it "
        "penalises ties and so favours the finest-grained variable "
        "(depth has 3 distinct values, width 4, params/point 12) and should not be "
        "used to rank them.\n"
    )
    L.append("| variable | discordance | pairs | Spearman |")
    L.append("|---|---|---|---|")
    for nm in ("depth", "width", "params_per_point"):
        L.append(
            f"| {'params/point' if nm == 'params_per_point' else nm} | "
            f"{g['discordance'][nm]:.3f} | "
            f"{g['n_pairs'][nm]} | {g['spearman'][nm]:+.3f} |"
        )
    L.append("")
    L.append(
        "Smallest warp gap reachable at each depth: "
        + ", ".join(f"depth {k} → {v:.2e}" for k, v in g["min_gap_by_depth"].items())
        + ". At each width: "
        + ", ".join(f"{k} → {v:.2e}" for k, v in g["min_gap_by_width"].items())
        + ".\n"
    )

    L.append("## Verdict\n")
    L.append(verdict_text(agg, cfg, hi_thresh, lo_thresh))
    table = "\n".join(L) + "\n"

    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "closure_shuffled.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(json.dumps(to_jsonable(agg), indent=1))
    write_tuning_budget(
        run,
        [
            {
                "model": f"neural_{h}x{dp}",
                "configs_tried": len(list(cfg.closure_capacity.lr_ladder)),
                "epochs": int(cfg.closure_capacity.ref_epochs),
                "gradient_steps": int(cfg.closure_capacity.ref_epochs)
                + len(list(cfg.closure_capacity.lr_ladder))
                * int(cfg.closure_capacity.pilot_epochs),
                "search_space": "learning rate over "
                + str([float(v) for v in cfg.closure_capacity.lr_ladder])
                + ", chosen per cell by a "
                + str(int(cfg.closure_capacity.pilot_epochs))
                + "-epoch pilot on the reference fit's training loss",
                "selection": "final iterate for the reference, best iterate for "
                "refits; identical protocol across all four targets",
            }
            for h, dp in keys
        ],
    )
    print("\n" + table)
    return 0


def verdict_text(agg: dict[str, Any], cfg: Any, hi: float, lo: float) -> str:
    c128x3, c128x2, c128x1 = (
        agg["cells"].get("128x3"),
        agg["cells"].get("128x2"),
        agg["cells"].get("128x1"),
    )
    g = agg["governing"]
    parts: list[str] = []

    wide = [c for c in (c128x2, c128x3) if c is not None]
    wide_shuf = max((c["r2_shuffled"]["mean"] for c in wide), default=float("nan"))
    wide_names = ", ".join(f"{c['hidden']}x{c['depth']}" for c in wide)
    if wide and wide_shuf >= hi:
        parts.append(
            f"**In-sample closure is uninformative at {wide_names}.** There the "
            f"shuffled target is re-represented at R² up to {wide_shuf:.4f}: the "
            "class can fit a random permutation of the latent, a target with no "
            "relation to the input, on the observed points. Any measurement that "
            "asks only whether a target can be fitted on those same points "
            "therefore cannot distinguish a class that CONTAINS the "
            "reparameterized twin from one that memorizes n values at n points, and "
            "the closure of 1.000000 reported for these cells is such a "
            "measurement.\n\n"
            "**This does not refute closure, and must not be read as doing so.** "
            "Closure under monotone reparameterization is a claim about what the "
            "function class contains. A universal approximator contains ψ∘φ *and* "
            "fits permutations; both are consequences of flexibility, and the second "
            "does not deny the first. The dissociation this experiment set out to "
            "find -- a class that re-represents warps while failing arbitrary "
            "targets -- is not a property any sufficiently flexible neural class "
            "would have, so its absence is expected and carries no evidence either "
            "way. **The discriminating measurement is held-out generalization** "
            "(`experiments/closure_heldout.py`, `paper/tables/closure_heldout.md`), "
            "which refits on a subset and evaluates the warp target on points the "
            "refit never saw. Containing the twin implies it predicts on unseen "
            "points; memorizing does not. Until that is measured, neither branch of "
            "the identifiability question is decided."
        )
    elif wide and wide_shuf <= lo:
        parts.append(
            f"**The closed cells do not fit arbitrary targets.** At {wide_names} the "
            f"monotone warp is re-represented essentially exactly while the shuffled "
            f"target reaches only R² {wide_shuf:.4f}, so in-sample closure there is "
            "not explained by an ability to fit anything on the observed points. "
            "That removes the memorization reading as an explanation of these "
            "numbers, but does not by itself establish closure out of sample; "
            "`experiments/closure_heldout.py` is what tests that directly."
        )
    elif wide:
        parts.append(
            f"**Intermediate at the closed cells.** At {wide_names} the shuffled "
            f"target reaches R² {wide_shuf:.4f}, between the thresholds for fitting "
            f"anything ({hi}) and for failing arbitrary targets ({lo}). No reading "
            "is supported from this contrast alone; see "
            "`experiments/closure_heldout.py`."
        )

    if c128x1 is not None:
        pp = c128x1["params_per_point"]
        s = c128x1["r2_shuffled"]["mean"]
        w = c128x1["r2_warp"]["mean"]
        if s < hi:
            parts.append(
                f"**Parameters per datapoint is not the governing variable.** 128x1 "
                f"sits at {pp:.2f} parameters per datapoint, above the boundary that "
                f"the capacity experiment's verdict leaned on, yet it fails the "
                f"shuffled target (R² {s:.4f}) and does not reach closure on the warp "
                f"either (R² {w:.6f}, gap {1.0 - w:.2e}). Being overparameterized by "
                "that ratio neither confers closure nor implies interpolation, so the "
                "ratio cannot carry the argument."
            )
        else:
            parts.append(
                f"**128x1 interpolates too** (shuffled R² {s:.4f} at {pp:.2f} "
                f"parameters per datapoint) while still failing to close the warp "
                f"target (R² {w:.6f}). Interpolation capacity and closure therefore "
                "come apart, and the ratio does not predict closure."
            )

    strongest = g["strongest"]
    LABELS = {"depth": "depth", "width": "width", "params_per_point": "params/point"}
    label = {
        "depth": "**Depth governs closure.**",
        "width": "**Width governs closure.**",
        "params_per_point": "**Parameters per datapoint governs closure.**",
    }[strongest]
    d = g["discordance"]
    parts.append(
        (
            label
            + f" Discordance -- the fraction of distinguished cell pairs in which more "
            f"capacity goes with a larger gap -- is depth {d['depth']:.3f}, width "
            f"{d['width']:.3f}, params/point {d['params_per_point']:.3f}"
            if g["clean"]
            else "**No single variable governs closure cleanly.** Discordance is depth "
            f"{d['depth']:.3f}, width {d['width']:.3f}, params/point "
            f"{d['params_per_point']:.3f}; the lowest is "
            f"{LABELS[strongest]}, but not by a clear margin, so closure is "
            "set by depth and width jointly rather than by any one of them"
        )
        + ". (Spearman is reported in the table but not used to rank: it penalises "
        "ties and so favours params/point, which takes a distinct value in every "
        "cell, over depth, which takes three.)"
        + " The smallest gap reachable at each depth is "
        + ", ".join(f"depth {k} → {v:.2e}" for k, v in g["min_gap_by_depth"].items())
        + (
            ", monotone in depth"
            if g["min_monotone_in_depth"]
            else ", not monotone in depth"
        )
        + "; at each width "
        + ", ".join(f"{k} → {v:.2e}" for k, v in g["min_gap_by_width"].items())
        + (
            ", monotone in width."
            if g["min_monotone_in_width"]
            else ", not monotone in width."
        )
    )

    # Whether any cell is both closed and selective. This is reported as an
    # observation about the grid, NOT as a test of closure. Closure under
    # monotone reparameterization is a claim about what the class contains, and
    # a universal approximator contains the reparameterized twin AND fits
    # permutations; selectivity is therefore not a requirement of closure and
    # its absence is the expected behaviour of a flexible neural class. An
    # earlier version of this verdict treated the disjointness as evidence
    # against closure, which was wrong.
    margin = float(cfg.closure_shuffled.selectivity_margin)
    both = sorted(
        k
        for k, v in agg["cells"].items()
        if 1.0 - v["r2_warp"]["mean"] < 5e-7 and v["selectivity"]["mean"] >= margin
    )
    sel_any = sorted(
        k for k, v in agg["cells"].items() if v["selectivity"]["mean"] >= margin
    )
    closed_any = sorted(
        k for k, v in agg["cells"].items() if 1.0 - v["r2_warp"]["mean"] < 5e-7
    )
    agg["closed_and_selective"] = both
    if both:
        parts.append(
            f"**{', '.join(both)} {'is' if len(both) == 1 else 'are'} both closed and "
            f"selective** (selectivity at least {margin}): the monotone warp is "
            "re-represented essentially exactly while the non-monotone targets are "
            "not. In-sample closure at these cells is therefore not explained by an "
            "ability to fit anything, though out-of-sample behaviour is still what "
            "settles containment."
        )
    elif closed_any and sel_any:
        parts.append(
            f"**No cell is both closed and selective**: closure of 1.000000 occurs at "
            f"{', '.join(closed_any)}, selectivity of at least {margin} at "
            f"{', '.join(sel_any)}, and the sets are disjoint. This is reported as a "
            "description of the grid and is **not** evidence against closure. A class "
            "flexible enough to contain the reparameterized twin is generally also "
            "flexible enough to fit a permutation, so selectivity is not a property "
            "closure implies, and its absence at the closed cells is what any "
            "sufficiently flexible neural class would show. What the disjointness "
            "does establish is narrower and still useful: at exactly the cells where "
            "closure reads 1.000000, an in-sample fit cannot distinguish containment "
            "from memorization, which is why the held-out experiment exists."
        )
    elif closed_any:
        parts.append(
            f"**No cell is selective at all** (none reaches selectivity {margin}), "
            f"while {', '.join(closed_any)} reach closure. Every cell that closes "
            "also fits arbitrary targets."
        )

    floor = max(1.0 - agg["cells"][k]["r2_null"]["mean"] for k in agg["cells"])
    parts.append(
        f"The numerical floor is the null target, re-represented to within "
        f"{floor:.2e} in the worst cell, so every contrast above is far larger than "
        "the pipeline's own error."
    )
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
