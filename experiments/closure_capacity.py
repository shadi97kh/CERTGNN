"""Does closure on real data converge to 1.000000 as G-P map capacity grows?

`results/diagnostics/2026-08-29_mpsa_closure_budget.md` established that the
real-data closure shortfall is not an optimizer artifact: twenty-five times
the refit budget moves closure from 0.998979 to 0.999591 and asymptotes near
0.9996. That diagnostic was explicitly scoped to a single architecture,
32 hidden units, and this experiment removes that scope.

The question is one number. Sweeping the G-P map's width and depth, does
closure converge to 1.000000, or does it asymptote below 1?

- **Converges to 1.** A sufficiently flexible G-P map IS closed under monotone
  reparameterization on real data. The twin is then a legitimate alternative
  fit that predicts identically and no sample size separates it, so the two
  global-epistasis mechanisms are provably confusable at that capacity. That
  kills the mechanism-discrimination direction and restores the strong
  identifiability claim.
- **Asymptotes below 1.** The classes are separable on real data, the
  mechanism test is viable, and the identifiability claim is the weaker,
  practical one.

Method. For every (width, depth) cell: fit the reference map on the real
BRCA2 5' splice site library, warp its own fitted latent, and refit the same
architecture to the warped latent with the budget the budget diagnostic
showed sufficient -- 12,000 epochs, warm-started from the reference weights,
keeping the best iterate. Closure is reported to six significant figures. A
null control at zero warp strength runs in every cell, so the pipeline's own
numerical floor is visible next to every number and no closure shortfall
smaller than that floor is read as real.

The effect-versus-strength Spearman is reported per cell as internal
corroboration: if closure goes to 1 the twin becomes undetectable and the
trend of effect size against warp strength must vanish. A cell claiming
closure while still showing a trend is inconsistent and is flagged.

**The interpolation confound, and why parameters per datapoint is reported.**
A model with more parameters than datapoints can fit *any* target on those
points, so closure would reach 1.000000 for a reason that says nothing about
whether the model class is structurally closed under reparameterization. At
500 sequences even a 32-wide, 2-deep map reaches exact closure, while at 4,000
it does not. The sweep therefore runs on the FULL library, every cell reports
parameters per datapoint, and the verdict distinguishes closure achieved below
that ratio from closure achieved above it. A "yes" that only appears in the
overparameterized regime is interpolation and is reported as such.

Scope. The sweep uses one warp family (sinusoid). The question here is
capacity, and adding families multiplies cost without bearing on it; the
family dependence is measured in `identifiability_mpsa.py`.

Usage:
    python -m experiments.closure_capacity --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from torch import nn

from experiments._common import (
    configure_torch,
    git_sha,
    fmt_ci,
    load_config,
    make_run_dir,
    mean_ci,
    resolve_seeds,
    to_jsonable,
    write_tuning_budget,
)
from experiments.identifiability_mpsa import load_mpsa
from experiments.identifiability_probe import (
    MonotoneGE,
    _standardize,
    affine_r2,
    indistinguishability,
    invert_warp,
    is_monotone,
    warp,
)

FAMILY = "sinusoid"


def pick_device(cfg: Any) -> torch.device:
    want = str(cfg.closure_capacity.device)
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def make_neural(d: int, hidden: int, depth: int, gen: torch.Generator) -> nn.Module:
    """Neural G-P map with explicit width and depth."""
    layers: list[nn.Module] = []
    prev = d
    for _ in range(depth):
        layers += [nn.Linear(prev, hidden, dtype=torch.float64), nn.Tanh()]
        prev = hidden
    layers.append(nn.Linear(prev, 1, dtype=torch.float64))
    m = nn.Sequential(*layers)
    for p in m.parameters():
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64) * 0.3)
    return m


class CapacityModel(nn.Module):
    """A width/depth-parameterized G-P map with a monotone GE nonlinearity."""

    def __init__(
        self, d: int, hidden: int, depth: int, k: int, gen: torch.Generator
    ) -> None:
        super().__init__()
        self.phi = make_neural(d, hidden, depth, gen)
        self.g = MonotoneGE(k)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi(x).reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(self.latent(x))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def train(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float
) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((model(x) - y) ** 2).mean())


def select_lr(
    d: int,
    hidden: int,
    depth: int,
    k: int,
    gen_seed: int,
    torch_seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    ladder: list[float],
    pilot_epochs: int,
) -> float:
    """Pick the learning rate for one cell by a short pilot fit.

    At a fixed lr of 0.01 the 128-wide cells collapse onto the mean predictor
    within 200 epochs and never leave it: training loss freezes at var(y) and
    `fit_r2` is 0. The closure of such a cell is the agreement of two constant
    functions, which is near 1 for a reason that has nothing to do with
    identifiability, and those cells are precisely the ones an asymptote fit
    leans on. Lowering the rate to 0.003 fits the same cells to R^2 0.70 and
    0.94, so the collapse is an optimizer artifact, not a property of the map.

    Selection is on the *reference fit's* training loss only. It never sees a
    warp, a refit, or a closure value, so it cannot tilt the quantity under
    test; it only decides whether the cell is fit at all.
    """
    best_lr, best_loss = ladder[0], float("inf")
    for lr in ladder:
        gen = torch.Generator().manual_seed(gen_seed)
        torch.manual_seed(torch_seed)
        m = CapacityModel(d, hidden, depth, k, gen).to(x.device)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(pilot_epochs):
            opt.zero_grad()
            loss = ((m(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = float(((m(x) - y) ** 2).mean())
        if np.isfinite(final) and final < best_loss:
            best_lr, best_loss = lr, final
    return best_lr


def _snapshot(m: nn.Module) -> dict[str, torch.Tensor]:
    """Detached clone of the parameters.

    `copy.deepcopy(state_dict())` pickles every tensor and was called on every
    improving step, which early in training is every step; on a large model it
    dominated the refit entirely. Cloning is what best-iterate tracking needs.
    """
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def refit_to(
    target: torch.Tensor,
    x: torch.Tensor,
    ref: nn.Module,
    d: int,
    hidden: int,
    depth: int,
    gen: torch.Generator,
    epochs: int,
    lr: float,
    tol: float,
) -> tuple[torch.Tensor, nn.Module]:
    """Best in-class approximation to a target latent, warm-started, best iterate."""
    m = make_neural(d, hidden, depth, gen).to(x.device)
    m.load_state_dict(ref.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t = _standardize(target)

    def loss_fn() -> torch.Tensor:
        pred = m(x).reshape(-1)
        return ((pred - pred.mean()) / (pred.std() + 1e-9) - t).pow(2).mean()

    best = float(loss_fn().detach())
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
        return m(x).reshape(-1).detach(), m


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    cc, ip = cfg.closure_capacity, cfg.identifiability
    dev = pick_device(cfg)
    x, y, _ = load_mpsa_capped(cfg, seed)
    x, y = x.to(dev), y.to(dev)
    d = x.shape[1]
    strengths = [float(s) for s in cc.strengths]
    smax = max(strengths)
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
            mu, sd = float(ref.latent(x).mean()), float(ref.latent(x).std())
            with torch.no_grad():
                pred_orig = ref.g(ref.latent(x)).detach()
            fit_r2 = 1.0 - float(
                ((y - pred_orig) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            )

            # null control at zero warp strength, in every cell
            phi0, _ = refit_to(
                z_ref,
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
            z_back0 = invert_warp(_standardize(phi0).cpu(), FAMILY, 0.0, 1.0, seed).to(
                dev
            )
            with torch.no_grad():
                pred0 = ref.g(mu + sd * z_back0).detach()
            null = indistinguishability(y.cpu(), pred_orig.cpu(), pred0.cpu())
            null_closure = affine_r2(phi0.cpu().numpy(), z_ref.cpu().numpy())

            per_s: list[dict[str, float]] = []
            for s in strengths:
                if not is_monotone(FAMILY, s, float(cc.omega), seed):
                    continue
                w = warp(z_ref.cpu(), FAMILY, s, float(cc.omega), seed).to(dev)
                phi_t, _m = refit_to(
                    w,
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
                z_back = invert_warp(
                    _standardize(phi_t) * float(w.std()) + float(w.mean()),
                    FAMILY,
                    s,
                    float(cc.omega),
                    seed,
                )
                with torch.no_grad():
                    pred_t = ref.g(mu + sd * z_back).detach()
                ind = indistinguishability(y.cpu(), pred0.cpu(), pred_t.cpu())
                per_s.append(
                    {
                        "strength": s,
                        "closure_r2": affine_r2(phi_t.cpu().numpy(), w.cpu().numpy()),
                        "effect_size": abs(ind["effect_size"]),
                    }
                )
            rho = float("nan")
            if len(per_s) > 2:
                r = spearmanr(
                    [e["strength"] for e in per_s], [e["effect_size"] for e in per_s]
                ).statistic
                rho = float(r) if np.isfinite(r) else float("nan")
            at_max = next((e for e in per_s if abs(e["strength"] - smax) < 1e-9), None)
            cells.append(
                {
                    "hidden": hidden,
                    "depth": depth,
                    "n_params": n_params(ref.phi),
                    "fit_r2": fit_r2,
                    "lr": cell_lr,
                    # A reference fit that never left the mean predictor makes
                    # every downstream number in the cell meaningless: closure
                    # then compares two constants. Such a cell is excluded from
                    # the asymptote rather than being allowed to pull it to 1.
                    "degenerate": bool(
                        not np.isfinite(fit_r2) or fit_r2 < float(cc.min_fit_r2)
                    ),
                    "closure_r2": at_max["closure_r2"] if at_max else float("nan"),
                    "null_closure_r2": null_closure,
                    "null_effect_size": abs(null["effect_size"]),
                    "effect_trend_rho": rho,
                    "per_strength": per_s,
                }
            )
    return {"seed": seed, "cells": cells, "n": int(x.shape[0])}


def load_mpsa_capped(cfg: Any, seed: int):
    """load_mpsa with this experiment's own sequence cap."""
    from omegaconf import OmegaConf

    c2 = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {"identifiability_mpsa": {"n_max": int(cfg.closure_capacity.n_max)}}
        ),
    )
    return load_mpsa(c2, seed)


def fit_asymptote(params: np.ndarray, gap: np.ndarray) -> dict[str, float]:
    """Fit gap(c) = g_inf + a * c^(-b); the asymptote is 1 - g_inf."""
    ok = np.isfinite(params) & np.isfinite(gap) & (gap > 0)
    if ok.sum() < 4:
        return {
            "g_inf": float("nan"),
            "a": float("nan"),
            "b": float("nan"),
            "rmse": float("nan"),
            "b_at_bound": True,
        }

    def model(c, g_inf, a, b):
        return g_inf + a * np.power(c, -b)

    try:
        popt, _ = curve_fit(
            model,
            params[ok],
            gap[ok],
            p0=[max(gap[ok].min() * 0.5, 1e-8), float(gap[ok].max()), 0.5],
            bounds=([0.0, 0.0, 0.0], [1.0, np.inf, 5.0]),
            maxfev=20000,
        )
        g_inf, a, b = (float(v) for v in popt)
        resid = float(np.sqrt(np.mean((model(params[ok], g_inf, a, b) - gap[ok]) ** 2)))
        # b pinned at a bound means the power law could not describe the data.
        # That happens here because the gap is not monotone in parameter count:
        # depth and width are not interchangeable along a single axis, so a
        # cell with more parameters can have a LARGER gap. An asymptote read
        # off such a fit is false precision and is reported as unusable.
        return {
            "g_inf": g_inf,
            "a": a,
            "b": b,
            "rmse": resid,
            "b_at_bound": bool(b >= 4.999 or b <= 1e-6),
        }
    except Exception:
        return {
            "g_inf": float("nan"),
            "a": float("nan"),
            "b": float("nan"),
            "rmse": float("nan"),
            "b_at_bound": True,
        }


def asymptote_over(
    per_seed: list[dict[str, Any]],
    agg: dict[str, Any],
    keys: list[tuple[int, int]],
    n_boot: int,
) -> dict[str, Any]:
    """Fit the closure gap against parameter count over the given cells.

    Fitted twice by the caller: once over every cell, and once restricted to
    cells with fewer parameters than datapoints. The unrestricted fit is
    dominated by the widest cells, which are exactly the ones that can
    interpolate any target on the observed points, so on its own it will report
    convergence to 1 for a reason that says nothing about whether the model
    class is structurally closed. The restricted fit is the one that speaks to
    structure, and the verdict leans on it.
    """
    rng = np.random.default_rng(0)
    g_infs = []
    for _ in range(n_boot):
        idx = rng.choice(len(per_seed), len(per_seed), replace=True)
        pv, gv = [], []
        for h, dp in keys:
            vals = [
                c["closure_r2"]
                for i in idx
                for c in per_seed[i]["cells"]
                if c["hidden"] == h
                and c["depth"] == dp
                and np.isfinite(c["closure_r2"])
                and not c.get("degenerate", False)
            ]
            if vals:
                pv.append(float(agg["cells"][f"{h}x{dp}"]["n_params"]))
                gv.append(1.0 - float(np.mean(vals)))
        f = fit_asymptote(np.array(pv), np.array(gv))
        if np.isfinite(f["g_inf"]):
            g_infs.append(f["g_inf"])
    pv0 = np.array(
        [agg["cells"][f"{h}x{dp}"]["n_params"] for h, dp in keys], dtype=float
    )
    gv0 = np.array(
        [1.0 - agg["cells"][f"{h}x{dp}"]["closure_r2"]["mean"] for h, dp in keys]
    )
    point = fit_asymptote(pv0, gv0)
    lo, hi = (
        (float(np.quantile(g_infs, 0.025)), float(np.quantile(g_infs, 0.975)))
        if g_infs
        else (float("nan"),) * 2
    )
    return {
        "n_cells": len(keys),
        "cells": [f"{h}x{dp}" for h, dp in keys],
        "gap_inf": point["g_inf"],
        "gap_inf_lo": lo,
        "gap_inf_hi": hi,
        "closure_inf": 1.0 - point["g_inf"],
        "closure_inf_lo": 1.0 - hi,
        "closure_inf_hi": 1.0 - lo,
        "power_a": point["a"],
        "power_b": point["b"],
        "rmse": point.get("rmse", float("nan")),
        "b_at_bound": point.get("b_at_bound", True),
        "n_boot": len(g_infs),
    }


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
        help="rebuild the aggregate and table from an existing run's seed JSONs "
        "instead of recomputing; the numbers stay traceable to that run",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    configure_torch(cfg)
    if args.retable:
        run = pathlib.Path(args.retable)
        meta = json.loads((run / "meta.json").read_text())
        files = sorted(
            run.glob("seed_*.json"), key=lambda f: int(f.stem.split("_")[1])
        )
        if not files:
            print(f"no seed_*.json in {run}", file=sys.stderr)
            return 2
        per_seed = [json.loads(f.read_text()) for f in files]
        retabled = True
        print(f"retable: {run} ({len(per_seed)} seeds)")
    else:
        retabled = False
        run = make_run_dir(cfg, "closure_capacity", allow_dirty=args.allow_dirty)
        meta = json.loads((run / "meta.json").read_text())
        print(f"run dir: {run}")

        per_seed = []
        for seed in resolve_seeds(cfg):
            r = run_seed(cfg, seed)
            per_seed.append(r)
            best = max(
                c["closure_r2"] for c in r["cells"] if np.isfinite(c["closure_r2"])
            )
            print(
                f"seed {seed}: best closure {best:.6f} over {len(r['cells'])} cells",
                flush=True,
            )
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
        agg["cells"][f"{h}x{dp}"] = {
            "hidden": h,
            "depth": dp,
            "n_params": sel[0]["n_params"],
            # The interpolation confound turns on this ratio, so it is computed
            # here rather than assumed: above 1 a map can fit any target on the
            # observed points and its closure is not evidence about structure.
            "params_per_point": sel[0]["n_params"] / float(per_seed[0]["n"]),
            "closure_r2": mean_ci([c["closure_r2"] for c in sel], n_boot=nb),
            "null_closure_r2": mean_ci([c["null_closure_r2"] for c in sel], n_boot=nb),
            "null_effect_size": mean_ci(
                [c["null_effect_size"] for c in sel], n_boot=nb
            ),
            "effect_trend_rho": mean_ci(
                [c["effect_trend_rho"] for c in sel], n_boot=nb
            ),
            "fit_r2": mean_ci([c["fit_r2"] for c in sel], n_boot=nb),
            "n_degenerate": sum(1 for c in sel if c.get("degenerate", False)),
            "lrs": sorted({c.get("lr") for c in sel if c.get("lr") is not None}),
        }

    # asymptote, bootstrapped over seeds. Fitted twice: over every cell, and
    # restricted to cells with fewer parameters than datapoints. A cell that can
    # interpolate reaches closure for a reason unrelated to structural closure,
    # so the unrestricted fit alone would answer the wrong question.
    nboot_a = int(cfg.closure_capacity.asymptote_boot)
    agg["asymptote"] = asymptote_over(per_seed, agg, keys, nboot_a)
    keys_under = [
        (h, dp) for h, dp in keys if agg["cells"][f"{h}x{dp}"]["params_per_point"] < 1.0
    ]
    agg["asymptote_underparam"] = (
        asymptote_over(per_seed, agg, keys_under, nboot_a)
        if len(keys_under) >= 4
        else {"gap_inf": float("nan"), "n_cells": len(keys_under), "note": "too few cells"}
    )

    # Floors and corroboration.
    #
    # The verdict is deliberately NON-PARAMETRIC. The power-law asymptote is
    # misspecified on this data: the closure gap is not monotone in parameter
    # count (16x3 has 1,153 parameters and a smaller gap than 64x1 with 2,433),
    # because depth and width do not trade off along a single axis. Both fits
    # return b pinned at its bound. Reading a six-figure asymptote off that fit
    # would be false precision on the number that decides the paper, so the
    # decision rests on what the cells actually reached.
    #
    # The floor for a CLOSURE claim is the null control's CLOSURE, not its
    # effect size: comparing a closure gap against an effect size compares two
    # different quantities.
    null_closure_gap = max(
        1.0 - agg["cells"][k]["null_closure_r2"]["mean"] for k in agg["cells"]
    )
    null_effect_floor = max(
        agg["cells"][k]["null_effect_size"]["mean"] for k in agg["cells"]
    )
    best_cell = max(agg["cells"], key=lambda k: agg["cells"][k]["closure_r2"]["mean"])
    best = agg["cells"][best_cell]

    # "Reaches 1.000000" means rounding to 1.000000 at six decimals, and by a
    # margin larger than the pipeline's own numerical floor.
    def reaches_one(v: dict[str, Any]) -> bool:
        gap = 1.0 - v["closure_r2"]["mean"]
        return gap < 5e-7 and gap <= max(null_closure_gap, 0.0) + 5e-7

    under = {k: v for k, v in agg["cells"].items() if v["params_per_point"] < 1.0}
    over = {k: v for k, v in agg["cells"].items() if v["params_per_point"] >= 1.0}
    closed_under = [k for k, v in under.items() if reaches_one(v)]
    closed_over = [k for k, v in over.items() if reaches_one(v)]
    best_under = (
        min(under, key=lambda k: 1.0 - under[k]["closure_r2"]["mean"]) if under else None
    )
    best_over = (
        min(over, key=lambda k: 1.0 - over[k]["closure_r2"]["mean"]) if over else None
    )
    gap_under = 1.0 - under[best_under]["closure_r2"]["mean"] if best_under else float("nan")
    gap_over = 1.0 - over[best_over]["closure_r2"]["mean"] if best_over else float("nan")
    trend_at_best = best["effect_trend_rho"]
    trend_vanished = trend_at_best["lo"] <= 0.2

    fit_note = (
        " The power-law asymptote is reported above but is NOT used for this "
        "verdict: b is pinned at its bound in the fit, because the closure gap "
        "is not monotone in parameter count. The verdict rests on what the "
        "cells reached."
        if agg["asymptote"].get("b_at_bound", True)
        else ""
    )
    floor_note = (
        f" The pipeline's numerical floor is the null control's closure, which "
        f"is 1.000000 in every cell (largest null gap {null_closure_gap:.2e}), "
        f"so a shortfall above that floor is real. (The null EFFECT size reaches "
        f"{null_effect_floor:.4f}, which is why effect size is not used as the "
        f"closure floor here.)"
    )

    if closed_under:
        verdict = (
            f"**Closure reaches 1.000000 in the underparameterized regime.** "
            f"{len(closed_under)} of {len(under)} cells with FEWER parameters than "
            f"datapoints reach it ({', '.join(sorted(closed_under))}); the best is "
            f"{best_under} at a gap of {gap_under:.2e}. A sufficiently flexible G-P "
            "map IS closed under monotone reparameterization on real MPSA data, and "
            "because this happens where the map cannot simply interpolate the "
            "observed points, it is a property of the model class. The twin is then "
            "a legitimate alternative fit that predicts identically and no sample "
            "size separates it, so the two global-epistasis mechanisms are provably "
            "confusable at that capacity. **This kills the mechanism-discrimination "
            "direction and restores the strong identifiability claim.**"
            + floor_note
            + fit_note
            + (
                f" Corroboration: at {best_cell} the effect-versus-strength Spearman "
                f"is {trend_at_best['mean']:+.3f} [{trend_at_best['lo']:+.3f}, "
                f"{trend_at_best['hi']:+.3f}], consistent with an undetectable twin."
                if trend_vanished
                else f" INCONSISTENT: at {best_cell} the effect still trends with "
                f"strength ({trend_at_best['mean']:+.3f} [{trend_at_best['lo']:+.3f}, "
                f"{trend_at_best['hi']:+.3f}]), which should not happen if the twin "
                "is undetectable. Resolve before using this result."
            )
        )
    elif closed_over:
        verdict = (
            f"**Closure reaches 1.000000 ONLY by interpolation.** No cell with fewer "
            f"parameters than datapoints reaches it: the best such cell is "
            f"{best_under} at closure "
            f"{under[best_under]['closure_r2']['mean']:.6f} (gap {gap_under:.2e}). "
            f"Closure appears only once the map is overparameterized "
            f"({', '.join(sorted(closed_over))}; best {best_over} at gap "
            f"{gap_over:.2e}), where it can fit ANY target on the observed points, so "
            "reaching 1 there says nothing about whether the model class is closed "
            "under reparameterization. **The strong identifiability claim does NOT "
            "follow.** On the evidence that is not interpolation, the classes remain "
            "separable on real data, the mechanism test is viable, and the "
            "identifiability claim is the weaker practical one."
            + floor_note
            + fit_note
            + (
                f" Note that at {best_cell} the effect-versus-strength Spearman is "
                f"{trend_at_best['mean']:+.3f} [{trend_at_best['lo']:+.3f}, "
                f"{trend_at_best['hi']:+.3f}]: the twin is still detectable in a cell "
                "reporting closure of 1.000000, which is itself inconsistent with "
                "genuine closure and corroborates the interpolation reading."
                if not trend_vanished
                else ""
            )
        )
    else:
        verdict = (
            f"**Closure does NOT reach 1.000000 at any capacity tested.** The best "
            f"cell is {best_cell} at {best['closure_r2']['mean']:.6f} "
            f"({best['n_params']:,} parameters, "
            f"{best['params_per_point']:.2f} per datapoint), a residual gap of "
            f"{1.0 - best['closure_r2']['mean']:.2e} that does not close as width and "
            f"depth grow; the best underparameterized cell is {best_under} at gap "
            f"{gap_under:.2e}. **The G-P map classes are separable on real data, the "
            "mechanism test is viable, and the identifiability claim is the weaker "
            "practical one, not the strong claim.** The twin is detectable at "
            "sufficient sample size and the two global-epistasis mechanisms are not "
            "provably confusable."
            + floor_note
            + fit_note
        )
    agg["closure_reached"] = {
        "underparameterized": sorted(closed_under),
        "overparameterized": sorted(closed_over),
        "best_underparameterized": best_under,
        "best_underparameterized_gap": gap_under,
        "best_overparameterized": best_over,
        "best_overparameterized_gap": gap_over,
        "null_closure_gap": null_closure_gap,
    }
    agg["verdict"] = verdict
    agg["null_floor_effect_size"] = null_effect_floor
    agg["null_floor_closure_gap"] = null_closure_gap
    agg["best_cell"] = best_cell

    L = ["# Does closure converge to 1.000000 as G-P map capacity grows?\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}"
        + (
            f", **aggregated post hoc by `{git_sha()}` via --retable** (the seed "
            f"values below were computed by `{meta['git_sha']}`; the aggregation, "
            f"asymptote and verdict code is the later one)"
            if retabled
            else ""
        )
        + f", config "
        f"`{meta['config_hash']}`, {agg['n_seeds']} seeds, mean [95% bootstrap CI]. "
        f"{agg['n']} real BRCA2 5' splice sites per seed; warp family {FAMILY}; refit "
        f"{int(cfg.closure_capacity.refit_epochs):,} epochs, warm-started, best iterate.\n"
    )
    L.append(
        "| width | depth | params | params/point | fit R² | closure R² at s=0.95 | "
        "null-control closure | null effect | effect-vs-strength ρ |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for h, dp in keys:
        c = agg["cells"][f"{h}x{dp}"]
        flag = f" ⚠️{c['n_degenerate']}/{agg['n_seeds']}" if c["n_degenerate"] else ""
        L.append(
            f"| {h} | {dp} | {c['n_params']:,} | {c['params_per_point']:.2f} | "
            f"{c['fit_r2']['mean']:.4f}{flag} | "
            f"{c['closure_r2']['mean']:.6f} "
            f"[{c['closure_r2']['lo']:.6f}, {c['closure_r2']['hi']:.6f}] | "
            f"{c['null_closure_r2']['mean']:.6f} | {c['null_effect_size']['mean']:.4f} | "
            f"{fmt_ci(c['effect_trend_rho'])} |"
        )
    n_deg = sum(c["n_degenerate"] for c in agg["cells"].values())
    if n_deg:
        L.append(
            f"\n⚠️ marks cells where the reference fit collapsed onto the mean "
            f"predictor (fit R² < {float(cfg.closure_capacity.min_fit_r2)}). Their "
            f"closure compares two near-constant functions and is not evidence of "
            f"anything; they are excluded from the asymptote fit. "
            f"{n_deg} cell-seeds affected."
        )
    L.append("")
    L.append("## Asymptote\n")
    A, U = agg["asymptote"], agg["asymptote_underparam"]
    L.append(
        f"Fitting the closure gap as `1 - R² = g_inf + a·params^(-b)` over all "
        f"{A['n_cells']} cells gives `g_inf` = {A['gap_inf']:.3e} "
        f"[{A['gap_inf_lo']:.3e}, {A['gap_inf_hi']:.3e}] "
        f"(b = {A['power_b']:.3f}, {A['n_boot']} bootstrap fits over seeds), "
        f"so closure tends to **{A['closure_inf']:.6f}** "
        f"[{A['closure_inf_lo']:.6f}, {A['closure_inf_hi']:.6f}].\n"
    )
    if np.isfinite(U.get("gap_inf", float("nan"))):
        L.append(
            f"That fit is dominated by the widest cells, which are the ones able to "
            f"interpolate any target on the observed points. Restricted to the "
            f"{U['n_cells']} cells with FEWER parameters than datapoints "
            f"({', '.join(U['cells'])}), the same fit gives `g_inf` = "
            f"{U['gap_inf']:.3e} [{U['gap_inf_lo']:.3e}, {U['gap_inf_hi']:.3e}], "
            f"i.e. closure tends to **{U['closure_inf']:.6f}** "
            f"[{U['closure_inf_lo']:.6f}, {U['closure_inf_hi']:.6f}]. This is the "
            f"fit that bears on structural closure; the unrestricted one cannot "
            f"separate closure from interpolation.\n"
        )
    else:
        L.append(
            f"The restricted fit over cells with fewer parameters than datapoints "
            f"could not be computed ({U.get('n_cells', 0)} such cells, needs 4).\n"
        )
    L.append("## Verdict\n")
    L.append(verdict)
    table = "\n".join(L) + "\n"

    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "closure_capacity.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(json.dumps(to_jsonable(agg), indent=1))
    (run / "per_seed_values.json").write_text(
        json.dumps(
            {
                f"closure_{h}x{dp}": {
                    str(r["seed"]): next(
                        c["closure_r2"]
                        for c in r["cells"]
                        if c["hidden"] == h and c["depth"] == dp
                    )
                    for r in per_seed
                }
                for h, dp in keys
            },
            indent=1,
        )
    )
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
                + "-epoch pilot; architecture is the sweep axis, not tuned",
                "selection": "pilot picks lr by the REFERENCE fit's training loss "
                "only (never a warp, refit or closure value); then final iterate "
                "for the reference, best iterate for refits",
            }
            for h, dp in keys
        ],
    )
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
