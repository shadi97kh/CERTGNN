"""Can the refit FIND a member of the class, or only stay at one it starts on?

`paper/tables/closure_heldout.md` concluded that no cell contains the twin,
because held-out agreement with psi(phi_hat) collapsed. That conclusion rested
on a ceiling control that does not do the work it appears to do.

**The defect.** `closure_heldout.refit_on_split` loads the reference weights,
then computes `best = loss_fn()` and snapshots BEFORE the optimisation loop. For
the null target, which is phi_hat itself, the refit therefore starts AT the
answer: measured initial loss is 6.7e-19, already below the 1e-10 early stop, so
the loop breaks on its first iteration and returns the reference weights
unchanged. A held-out R2 of 1.000000 on that target is arithmetic -- the model
is the reference and the target is the reference's own latent. It tests that
Adam does not wander away from a perfect solution. It does NOT test whether the
refit can find a solution it does not start at, which is exactly the capability
the warp result depends on.

So the honest reading of that experiment is narrower than what it claimed: the
warm-started refit procedure did not FIND a member equal to psi(phi_hat). That
is consistent with no such member existing, and equally consistent with one
existing that the optimiser failed to reach.

Two arms separate those.

**ARM 1 -- a ceiling that requires search.** Refit to the null target phi_hat
WITHOUT warm-starting from the reference, from three initializations: cold
random; the same cell's reference trained from a different init seed; and a
different cell's reference, copied where shapes match and randomly initialized
elsewhere. The target is known to be in the class -- the reference realises it
-- so this asks only whether the optimiser can travel to it.

- Cold-start null reaches ~1.000000 held out: search finds functions it does not
  start at, and the warp collapse is a statement about the class.
- Cold-start null falls short: the refit cannot reliably reach distant members,
  the warp collapse is confounded with optimisation failure, and the
  containment conclusion is NOT reported.

**ARM 2 -- asymmetric capacity.** Refit the warp target in a class strictly
LARGER than the reference's. If phi_hat is depth d, then psi(phi_hat) is
naturally a depth d+1 computation with a width-1 bottleneck, so asking a depth-d
class to represent it asks it to absorb an extra composition. A larger class
should contain it if anything does. A null arm runs in the same larger class as
a control, separating "the larger class cannot represent phi_hat at all" from
"the larger class cannot represent psi(phi_hat)".

- Held-out agreement rises toward 1.000000 in the larger class: the twin IS
  containable and the same-capacity collapse was a capacity mismatch, not
  absence.
- It does not rise: the collapse is structural and the containment conclusion
  is safe.

A same-capacity, warm-started warp refit runs here too, so the comparison Arm 2
needs is internal to this run rather than borrowed from another one.

Usage:
    python -m experiments.closure_search --config configs/base.yaml
    python -m experiments.closure_search --retable results/runs/<dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

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
    _snapshot,
    load_mpsa_capped,
    make_neural,
    n_params,
    pick_device,
    select_lr,
    train,
)
from experiments.closure_heldout import split_indices, strict_heldout_r2
from experiments.identifiability_probe import affine_r2, is_monotone, warp

# Arm 1 initializations, in the order reported.
INITS = ("cold", "other_seed", "other_cell")


def partial_load(m: nn.Module, state: dict[str, torch.Tensor]) -> int:
    """Copy every parameter whose shape matches; leave the rest at random init.

    Used for the cross-cell initialization in Arm 1 and nowhere else. Two cells
    of different width or depth have mostly incompatible shapes, so this is a
    partial transfer by construction; the count of copied tensors is recorded so
    the arm cannot be read as a full warm start when it was not one.
    """
    own = m.state_dict()
    copied = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v.detach().clone()
            copied += 1
    m.load_state_dict(own)
    return copied


def refit_from(
    target: torch.Tensor,
    x: torch.Tensor,
    fit_idx: torch.Tensor,
    d: int,
    hidden: int,
    depth: int,
    gen: torch.Generator,
    epochs: int,
    lr: float,
    tol: float,
    init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, float, float, int]:
    """Fit a fresh class member to `target` on `fit_idx` only.

    Returns the latent on ALL points, the initial loss, the final best loss, and
    the number of tensors copied from `init_state`. The initial loss is returned
    because it is what exposes a vacuous control: a refit whose initial loss is
    already at the early-stop threshold has not searched for anything.
    """
    m = make_neural(d, hidden, depth, gen).to(x.device)
    copied = partial_load(m, init_state) if init_state is not None else 0
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    x_fit = x[fit_idx]
    t_fit = target[fit_idx]
    t = (t_fit - t_fit.mean()) / (t_fit.std() + 1e-9)

    def loss_fn() -> torch.Tensor:
        pred = m(x_fit).reshape(-1)
        return ((pred - pred.mean()) / (pred.std() + 1e-9) - t).pow(2).mean()

    init_loss = float(loss_fn().detach())
    best = init_loss
    best_state = _snapshot(m)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        cur = float(loss.detach())
        if cur < best:
            best, best_state = cur, _snapshot(m)
        if best < tol:
            break
    m.load_state_dict(best_state)
    with torch.no_grad():
        return m(x).reshape(-1).detach(), init_loss, best, copied


def _scores(
    phi: torch.Tensor, target: torch.Tensor, fit_np: np.ndarray, ho_np: np.ndarray
) -> dict[str, float]:
    pn, tn = phi.cpu().numpy(), target.cpu().numpy()
    return {
        "fit": affine_r2(pn[fit_np], tn[fit_np]),
        "heldout": affine_r2(pn[ho_np], tn[ho_np]),
        "heldout_strict": strict_heldout_r2(pn, tn, fit_np, ho_np),
    }


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    cc, ip = cfg.closure_capacity, cfg.identifiability
    cs = cfg.closure_search
    dev = pick_device(cfg)
    x, y, _ = load_mpsa_capped(cfg, seed)
    x, y = x.to(dev), y.to(dev)
    n, d = int(x.shape[0]), int(x.shape[1])
    fit_idx, ho_idx, split_record = split_indices(n, int(cs.n_fit), seed)
    fit_idx, ho_idx = fit_idx.to(dev), ho_idx.to(dev)
    fit_np, ho_np = fit_idx.cpu().numpy(), ho_idx.cpu().numpy()
    x_fit, y_fit = x[fit_idx], y[fit_idx]

    strength, omega = float(cs.strength), float(cc.omega)
    if not is_monotone(FAMILY, strength, omega, seed):
        raise RuntimeError(f"warp not monotone at seed {seed}")

    grid = [(int(h), int(dp)) for h in cc.hidden for dp in cc.depth]
    big = (max(int(h) for h in cc.hidden), max(int(dp) for dp in cc.depth))
    epochs, tol = int(cc.refit_epochs), float(ip.radius.fit_tol)

    # Pass 1: a reference per cell, plus a second reference from a different
    # init seed to serve as Arm 1's "other seed" starting point.
    refs: dict[tuple[int, int], dict[str, Any]] = {}
    for hidden, depth in grid:
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
        gen2 = torch.Generator().manual_seed(gen_seed + 555_557)
        torch.manual_seed(torch_seed + 555_557)
        ref2 = CapacityModel(d, hidden, depth, int(ip.ge_components), gen2).to(dev)
        train(ref2, x_fit, y_fit, int(cc.ref_epochs), lr)
        with torch.no_grad():
            lat = ref.latent(x).detach()
            pred_fit = ref.g(ref.latent(x_fit)).detach()
        fit_r2 = 1.0 - float(
            ((y_fit - pred_fit) ** 2).sum() / ((y_fit - y_fit.mean()) ** 2).sum()
        )
        mu = float(lat[fit_idx].mean())
        sd = float(lat[fit_idx].std()) + 1e-12
        z = (lat - mu) / sd
        refs[(hidden, depth)] = {
            "lr": lr,
            "gen_seed": gen_seed,
            "phi_state": {
                k: v.detach().clone() for k, v in ref.phi.state_dict().items()
            },
            "phi2_state": {
                k: v.detach().clone() for k, v in ref2.phi.state_dict().items()
            },
            "z": z,
            "warp": warp(z.detach().cpu(), FAMILY, strength, omega, seed).to(dev),
            "fit_r2": fit_r2,
            "n_params": n_params(ref.phi),
        }

    cells: list[dict[str, Any]] = []
    for hidden, depth in grid:
        R = refs[(hidden, depth)]
        lr, gen_seed = R["lr"], R["gen_seed"]
        rec: dict[str, Any] = {
            "hidden": hidden,
            "depth": depth,
            "n_params": R["n_params"],
            "fit_r2": R["fit_r2"],
            "lr": lr,
            "degenerate": bool(
                not np.isfinite(R["fit_r2"]) or R["fit_r2"] < float(cc.min_fit_r2)
            ),
        }

        # Baseline: same-capacity warp refit, warm-started -- the closure_heldout
        # measurement, recomputed here so Arm 2's comparison is internal.
        gen = torch.Generator().manual_seed(gen_seed + 11)
        phi, il, bl, _ = refit_from(
            R["warp"],
            x,
            fit_idx,
            d,
            hidden,
            depth,
            gen,
            epochs,
            lr,
            tol,
            init_state=R["phi_state"],
        )
        s = _scores(phi, R["warp"], fit_np, ho_np)
        rec["base_warp_fit"] = s["fit"]
        rec["base_warp_heldout"] = s["heldout"]
        rec["base_warp_heldout_strict"] = s["heldout_strict"]
        rec["base_warp_init_loss"] = il
        rec["base_warp_final_loss"] = bl

        # ARM 1: the null target, reached by search rather than started on.
        other = next(c for c in grid if c != (hidden, depth))
        for i, name in enumerate(INITS):
            init = {
                "cold": None,
                "other_seed": R["phi2_state"],
                "other_cell": refs[other]["phi_state"],
            }[name]
            gen = torch.Generator().manual_seed(gen_seed + 101 + i)
            phi, il, bl, copied = refit_from(
                R["z"],
                x,
                fit_idx,
                d,
                hidden,
                depth,
                gen,
                epochs,
                lr,
                tol,
                init_state=init,
            )
            s = _scores(phi, R["z"], fit_np, ho_np)
            rec[f"arm1_{name}_fit"] = s["fit"]
            rec[f"arm1_{name}_heldout"] = s["heldout"]
            rec[f"arm1_{name}_heldout_strict"] = s["heldout_strict"]
            rec[f"arm1_{name}_init_loss"] = il
            rec[f"arm1_{name}_final_loss"] = bl
            rec[f"arm1_{name}_copied"] = copied
        rec["arm1_other_cell_source"] = f"{other[0]}x{other[1]}"

        # ARM 2: warp and null refit in the grid-maximal class, cold start.
        rec["arm2_class"] = f"{big[0]}x{big[1]}"
        if (hidden, depth) != big:
            big_lr = refs[big]["lr"]
            for tname, tgt in (("warp", R["warp"]), ("null", R["z"])):
                gen = torch.Generator().manual_seed(gen_seed + 202 + len(tname))
                phi, il, bl, _ = refit_from(
                    tgt,
                    x,
                    fit_idx,
                    d,
                    big[0],
                    big[1],
                    gen,
                    epochs,
                    big_lr,
                    tol,
                    init_state=None,
                )
                s = _scores(phi, tgt, fit_np, ho_np)
                rec[f"arm2_{tname}_fit"] = s["fit"]
                rec[f"arm2_{tname}_heldout"] = s["heldout"]
                rec[f"arm2_{tname}_heldout_strict"] = s["heldout_strict"]
                rec[f"arm2_{tname}_init_loss"] = il
                rec[f"arm2_{tname}_final_loss"] = bl
        cells.append(rec)

    return {
        "seed": seed,
        "cells": cells,
        "n": n,
        "n_fit": int(fit_idx.numel()),
        "n_heldout": int(ho_idx.numel()),
        "arm2_class": f"{big[0]}x{big[1]}",
        "split": split_record,
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
        run = make_run_dir(cfg, "closure_search", allow_dirty=args.allow_dirty)
        meta = json.loads((run / "meta.json").read_text())
        print(f"run dir: {run}")
        per_seed, splits = [], {}
        for seed in resolve_seeds(cfg):
            r = run_seed(cfg, seed)
            splits[str(seed)] = r.pop("split")
            per_seed.append(r)
            c = [k for k in r["cells"] if k["hidden"] == 64 and k["depth"] == 2]
            msg = (
                f" 64x2 arm1 cold heldout {c[0]['arm1_cold_heldout']:.4f}; "
                f"arm2 warp heldout {c[0].get('arm2_warp_heldout', float('nan')):.4f}"
                if c
                else ""
            )
            print(f"seed {seed}:{msg}", flush=True)
            (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))
            (run / "splits.json").write_text(json.dumps(splits, indent=1))

    nb = int(cfg.bootstrap_resamples)
    keys = [(c["hidden"], c["depth"]) for c in per_seed[0]["cells"]]
    agg: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "n": per_seed[0]["n"],
        "n_fit": per_seed[0]["n_fit"],
        "n_heldout": per_seed[0]["n_heldout"],
        "arm2_class": per_seed[0]["arm2_class"],
        "cells": {},
    }
    fields = ["base_warp_fit", "base_warp_heldout", "base_warp_heldout_strict"]
    for nm in INITS:
        fields += [
            f"arm1_{nm}_fit",
            f"arm1_{nm}_heldout",
            f"arm1_{nm}_heldout_strict",
            f"arm1_{nm}_init_loss",
            f"arm1_{nm}_final_loss",
        ]
    for t in ("warp", "null"):
        fields += [f"arm2_{t}_fit", f"arm2_{t}_heldout", f"arm2_{t}_heldout_strict"]

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
            "fit_r2": mean_ci([c["fit_r2"] for c in sel], n_boot=nb),
            "n_degenerate": sum(1 for c in sel if c.get("degenerate", False)),
            "arm1_other_cell_source": sel[0]["arm1_other_cell_source"],
            "arm1_other_cell_copied": sel[0]["arm1_other_cell_copied"],
            "has_arm2": "arm2_warp_heldout" in sel[0],
        }
        for f in fields:
            vals = [c[f] for c in sel if f in c]
            if vals:
                entry[f] = mean_ci(vals, n_boot=nb)
        if entry["has_arm2"]:
            entry["arm2_rise"] = mean_ci(
                [c["arm2_warp_heldout"] - c["base_warp_heldout"] for c in sel],
                n_boot=nb,
            )
        agg["cells"][f"{h}x{dp}"] = entry

    agg["verdict"] = verdict_text(agg, cfg)

    L = ["# Can the refit find a class member, or only stay where it starts?\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}"
        + (
            f", **aggregated post hoc by `{git_sha()}` via --retable**"
            if retabled
            else ""
        )
        + f", config `{meta['config_hash']}`, {agg['n_seeds']} seeds, mean "
        f"[95% bootstrap CI]. {agg['n']} real BRCA2 5' splice sites per seed, "
        f"{agg['n_fit']} fit / {agg['n_heldout']} held out. Held-out R² throughout.\n"
    )
    L.append(
        "**Arm 1** refits the NULL target φ̂ -- a function the reference realises, so "
        "known to be in the class -- from three starting points that are not the "
        "answer. This is the ceiling control that `closure_heldout.md` lacked: there "
        "the refit began at the target (initial loss 6.7e-19, below the 1e-10 early "
        "stop) and returned unchanged, so its 1.000000 measured arithmetic rather "
        "than search.\n"
    )
    L.append("| width | depth | cold | other seed | other cell | copied |")
    L.append("|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        L.append(
            f"| {h} | {dp} | **{c['arm1_cold_heldout']['mean']:.6f}** "
            f"[{c['arm1_cold_heldout']['lo']:.4f}, {c['arm1_cold_heldout']['hi']:.4f}] | "
            f"{c['arm1_other_seed_heldout']['mean']:.6f} | "
            f"{c['arm1_other_cell_heldout']['mean']:.6f} "
            f"(from {c['arm1_other_cell_source']}) | "
            f"{c['arm1_other_cell_copied']} tensors |"
        )
    L.append("")
    L.append(
        f"**Arm 2** refits the WARP target in the grid-maximal class "
        f"`{agg['arm2_class']}` from a cold start, with the null target in the same "
        "class as a control. `base` is the same-capacity warm-started refit, "
        "recomputed here so the comparison is internal to this run.\n"
    )
    L.append("| width | depth | base warp | arm2 warp | rise | arm2 null (control) |")
    L.append("|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        if not c["has_arm2"]:
            L.append(
                f"| {h} | {dp} | {c['base_warp_heldout']['mean']:.6f} | — | — | — |"
            )
            continue
        L.append(
            f"| {h} | {dp} | {c['base_warp_heldout']['mean']:.6f} | "
            f"**{c['arm2_warp_heldout']['mean']:.6f}** | "
            f"{c['arm2_rise']['mean']:+.6f} "
            f"[{c['arm2_rise']['lo']:+.4f}, {c['arm2_rise']['hi']:+.4f}] | "
            f"{c['arm2_null_heldout']['mean']:.6f} |"
        )
    L.append("")
    L.append("## Verdict\n")
    L.append(agg["verdict"])
    table = "\n".join(L) + "\n"

    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "closure_search.md").write_text(table)
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
                + ", chosen per cell by a pilot on the FIT SPLIT's training loss",
                "selection": "best iterate for every refit; Arm 2 reuses the "
                "maximal cell's learning rate",
            }
            for h, dp in keys
        ],
    )
    print("\n" + table)
    return 0


def verdict_text(agg: dict[str, Any], cfg: Any) -> str:
    cells = agg["cells"]
    cs = cfg.closure_search
    ok = float(cs.search_ok)
    rise_margin = float(cs.rise_margin)
    parts: list[str] = []

    cold = {k: v["arm1_cold_heldout"]["mean"] for k, v in cells.items()}
    worst_cold = min(cold.values())
    worst_cell = min(cold, key=lambda k: cold[k])
    best_cold = max(cold.values())
    passing = sorted(k for k, v in cold.items() if v >= ok)
    search_works = len(passing) == len(cold)

    if search_works:
        parts.append(
            f"**Arm 1: search works.** Refitting the null target from a cold random "
            f"start -- a target the reference realises, so known to be in the class, "
            f"but not where the optimiser begins -- reaches held-out R² "
            f"{worst_cold:.6f} in the worst cell ({worst_cell}) and {best_cold:.6f} "
            f"in the best, clearing {ok} everywhere. The refit can find a member it "
            "did not start at, so a failure to reach the warp target is a statement "
            "about the class rather than about the optimiser."
        )
    else:
        failing = sorted(k for k in cold if k not in passing)
        parts.append(
            f"**Arm 1: search does NOT reliably work, and this invalidates the "
            f"containment conclusion.** Refitting the null target from a cold random "
            f"start reaches only held-out R² {worst_cold:.6f} in the worst cell "
            f"({worst_cell}); {len(failing)} of {len(cold)} cells fall below {ok} "
            f"({', '.join(failing)}). The target is a function the reference "
            "realises, so it IS in the class; the optimiser simply does not travel "
            "to it. **The collapse of the warp target reported in "
            "`paper/tables/closure_heldout.md` is therefore confounded with "
            "optimisation failure, and does not establish that no member of the "
            "class equals ψ∘φ̂.** What both experiments establish is narrower: this "
            "refit procedure does not find such a member. Whether one exists is "
            "unresolved."
        )

    # Whether the warp comparison is INTERPRETABLE at a given depth. This is a
    # weaker question than the search_ok gate above and deliberately uses no
    # absolute threshold, because none is needed: the null target is guaranteed
    # to be in the class, so the procedure's score on it is a ceiling for what
    # the same procedure can say about any other target. If a depth scores
    # HIGHER on the warp than on that ceiling, the ranking is inverted and the
    # warp number is reporting initialization rather than the class. Depth 1
    # clears this (cold null above warp) while depths 2 and 3 invert it, so the
    # two regimes support different conclusions and must not be pooled.
    depths = sorted({v["depth"] for v in cells.values()})
    by_depth = {d: {k: v for k, v in cells.items() if v["depth"] == d} for d in depths}
    worst_null = {
        d: min(c["arm1_cold_heldout"]["mean"] for c in by_depth[d].values())
        for d in depths
    }
    worst_warp = {
        d: min(c["base_warp_heldout"]["mean"] for c in by_depth[d].values())
        for d in depths
    }
    ok_depths = [d for d in depths if worst_null[d] >= worst_warp[d]]
    bad_depths = [d for d in depths if d not in ok_depths]
    agg["interpretable_depths"] = ok_depths
    if ok_depths and bad_depths:
        parts.append(
            "**Interpretability splits by depth, and so does what can be concluded.** "
            "The null target is in the class by construction, so the procedure's "
            "score on it bounds what the same procedure can say about anything else. "
            + "; ".join(
                f"depth {d}: cold-start null {worst_null[d]:.4f} vs warp "
                f"{worst_warp[d]:.4f}"
                for d in depths
            )
            + ". At depth "
            + ", ".join(str(d) for d in ok_depths)
            + " the ceiling sits above the measurement, so the comparison is valid "
            "there, and the warp target is genuinely not reached as well as an "
            "in-class one: that is evidence, for those classes specifically, that "
            "ψ∘φ̂ is not in them as exactly as φ̂ is. **At depth "
            + ", ".join(str(d) for d in bad_depths)
            + " the ranking is INVERTED** -- the warm-started warp refit scores "
            "higher than a cold-started refit to the reference's own latent, so the "
            "procedure does better on the harder target than on one guaranteed to be "
            "in the class. That is initialization dominating the result, and no "
            "conclusion about the class can be drawn at those depths."
        )

    other_seed = {k: v["arm1_other_seed_heldout"]["mean"] for k, v in cells.items()}
    other_cell = {k: v["arm1_other_cell_heldout"]["mean"] for k, v in cells.items()}
    parts.append(
        f"The two warm alternatives behave the same way: starting from the same "
        f"cell's reference trained at a different init seed gives worst-cell "
        f"{min(other_seed.values()):.6f}, and starting from another cell's "
        f"reference, copied where shapes match, gives {min(other_cell.values()):.6f}. "
        "Neither begins at the answer, so both are subject to the same search "
        "requirement as the cold start."
    )

    with_arm2 = {k: v for k, v in cells.items() if v["has_arm2"]}
    if with_arm2:
        rises = {k: v["arm2_rise"]["mean"] for k, v in with_arm2.items()}
        best_rise_cell = max(rises, key=lambda k: rises[k])
        best_arm2 = max(v["arm2_warp_heldout"]["mean"] for v in with_arm2.values())
        n_rising = sum(
            1 for k, v in with_arm2.items() if v["arm2_rise"]["lo"] > rise_margin
        )
        null_ctrl = min(v["arm2_null_heldout"]["mean"] for v in with_arm2.values())
        ctrl_ok = null_ctrl >= ok
        ctrl_txt = (
            f"The control holds: the same larger class recovers the plain latent φ̂ "
            f"at held-out R² {null_ctrl:.6f} in the worst cell, so it can represent "
            "the reference's own function and any shortfall on the warp target is "
            "not a failure to represent φ̂ itself."
            if ctrl_ok
            else f"**The control fails**: the larger class recovers even the plain "
            f"latent φ̂ at only {null_ctrl:.6f} in the worst cell, so it cannot "
            "reliably represent the reference's own function and its performance on "
            "the warp target says nothing about containment."
        )
        if n_rising and ctrl_ok:
            parts.append(
                f"**Arm 2: held-out agreement RISES in the larger class.** Refitting "
                f"the warp target in `{agg['arm2_class']}` instead of the reference's "
                f"own class improves held-out agreement in {n_rising} of "
                f"{len(with_arm2)} cells, by up to "
                f"{rises[best_rise_cell]:+.6f} at {best_rise_cell}, reaching "
                f"{best_arm2:.6f}. {ctrl_txt} The same-capacity collapse was at least "
                "partly a capacity mismatch rather than absence: ψ∘φ̂ is naturally a "
                "depth d+1 computation, and a depth-d class was being asked to "
                "absorb an extra composition."
            )
        elif ctrl_ok:
            parts.append(
                f"**Arm 2: held-out agreement does NOT rise in the larger class.** "
                f"Refitting the warp target in `{agg['arm2_class']}` improves no cell "
                f"beyond the {rise_margin} margin; the best change is "
                f"{rises[best_rise_cell]:+.6f} at {best_rise_cell} and the best "
                f"absolute is {best_arm2:.6f}. {ctrl_txt} Extra capacity does not "
                "recover the twin, so the shortfall is not a capacity mismatch."
            )
        else:
            parts.append(
                f"**Arm 2 is uninterpretable.** {ctrl_txt} Best arm-2 warp agreement "
                f"is {best_arm2:.6f}, best change {rises[best_rise_cell]:+.6f}."
            )

    if search_works and with_arm2:
        rises = {k: v["arm2_rise"]["mean"] for k, v in with_arm2.items()}
        n_rising = sum(
            1 for k, v in with_arm2.items() if v["arm2_rise"]["lo"] > rise_margin
        )
        ctrl_ok = min(v["arm2_null_heldout"]["mean"] for v in with_arm2.values()) >= ok
        if ctrl_ok and not n_rising:
            parts.append(
                "**Together: the containment conclusion is safe.** Search reaches "
                "in-class targets it does not start at, and extra capacity does not "
                "help, so the failure to reach ψ∘φ̂ out of sample is a property of "
                "the model class. No member of the classes tested equals ψ∘φ̂ as a "
                "function, and the identifiability claim is the weaker practical one."
            )
        elif ctrl_ok and n_rising:
            parts.append(
                "**Together: the containment conclusion does NOT stand as stated.** "
                "Search is capable, but a larger class does reach the twin better, so "
                "the same-capacity collapse reflects the reference class being too "
                "small to absorb the extra composition rather than the twin being "
                "absent. `paper/tables/closure_heldout.md` must be read as a "
                "statement about same-capacity refits, not about containment in "
                "general."
            )
    elif not search_works:
        parts.append(
            "**Together: no containment conclusion can be drawn from either "
            "experiment.** Arm 1 shows the optimiser does not reliably reach in-class "
            "targets, so Arm 2's outcome cannot be attributed to the class either. "
            "`paper/tables/closure_heldout.md` should be read as reporting what its "
            "procedure found, not what the class contains."
        )
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
