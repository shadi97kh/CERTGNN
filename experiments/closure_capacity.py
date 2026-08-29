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

**Result: the structure is settled, the mechanism is not.** The gap is ordered
by DEPTH, with width amplifying within a depth. No cell closes at depth 1 at
any width; closure of 1.000000 is reached at depths 2 and 3 at width 128.

**The interpolation confound, and why parameters per datapoint is reported.**
A model with more parameters than datapoints can fit *any* target on those
points, so closure would reach 1.000000 for a reason that says nothing about
whether the model class is structurally closed under reparameterization. At
500 sequences even a 32-wide, 2-deep map reaches exact closure, while at 4,000
it does not. The sweep therefore runs on the FULL library and every cell
reports parameters per datapoint.

That ratio turned out NOT to order this data, so it cannot carry the argument:
128x1 sits at 1.22 parameters per datapoint and is four orders of magnitude
short of closure, while 16x3 at 0.29 is comparable to it. Depth-dependent
expressivity and interpolation both predict the observed pattern at the widest
cells, and this experiment does not separate them. The verdict is therefore
explicitly PENDING. The shuffled-target control in `closure_shuffled.py` has
since run and did not settle it: it tested selectivity, and closure does not
imply selectivity, since a class flexible enough to contain the twin is
generally also flexible enough to fit a permutation. What it did establish is
that at these capacities an in-sample fit cannot tell containment from
memorization. The discriminating experiment is therefore `closure_heldout.py`,
which scores the warp target on points the refit never saw. Neither branch of
the question above is decided here.

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
        files = sorted(run.glob("seed_*.json"), key=lambda f: int(f.stem.split("_")[1]))
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
        else {
            "gap_inf": float("nan"),
            "n_cells": len(keys_under),
            "note": "too few cells",
        }
    )

    # Structure, and an explicitly undecided mechanism.
    #
    # An earlier version of this verdict attributed the pattern to interpolation
    # on the basis of parameters per datapoint. The table above contradicts that
    # reading: 128x1 sits at 1.22 parameters per datapoint, above the boundary,
    # and is four orders of magnitude short of closure. The gap is ordered by
    # DEPTH, with width amplifying within depth, and the ratio cuts across both.
    # The causal question is therefore deferred to the shuffled-target control
    # rather than answered here from a ratio the data does not support.
    cells_v = agg["cells"]

    def gap_of(k: str) -> float:
        return 1.0 - cells_v[k]["closure_r2"]["mean"]

    # The floor for a CLOSURE claim is the null control's CLOSURE, not its
    # effect size: those are different quantities.
    null_closure_gap = max(1.0 - cells_v[k]["null_closure_r2"]["mean"] for k in cells_v)
    null_effect_floor = max(cells_v[k]["null_effect_size"]["mean"] for k in cells_v)

    depths = sorted({v["depth"] for v in cells_v.values()})
    widths = sorted({v["hidden"] for v in cells_v.values()})

    def reaches_one(k: str) -> bool:
        """Rounds to 1.000000 at six decimals, above the pipeline's own floor."""
        return gap_of(k) < 5e-7 and gap_of(k) <= max(null_closure_gap, 0.0) + 5e-7

    closed = sorted(k for k in cells_v if reaches_one(k))
    closed_depths = sorted({cells_v[k]["depth"] for k in closed})
    open_depths = [d for d in depths if d not in closed_depths]

    best_by_depth = {
        d: min((k for k in cells_v if cells_v[k]["depth"] == d), key=gap_of)
        for d in depths
    }
    fit_flag = agg["asymptote"].get("b_at_bound", True)

    # Depth-1 sentence, built from the data rather than asserted.
    d_open = open_depths[0] if open_depths else None
    kb = best_by_depth[d_open] if d_open is not None else None
    open_txt = (
        f"No cell reaches closure of 1.000000 at depth {d_open}, at any width "
        f"tested. The best is {kb} at {cells_v[kb]['closure_r2']['mean']:.6f}, a gap "
        f"of {gap_of(kb):.1e}, and it sits at {cells_v[kb]['params_per_point']:.2f} "
        f"parameters per datapoint -- above one, yet nowhere near closure."
        if kb is not None
        else "Closure of 1.000000 is reached at every depth tested."
    )
    closed_txt = (
        "Closure of 1.000000 is reached at "
        + " and ".join(f"depth {d}" for d in closed_depths)
        + f" at width {max(cells_v[k]['hidden'] for k in closed)}"
        + (
            ", and approached to within "
            + ", ".join(
                f"{gap_of(k):.1e} at {k}"
                for k in sorted(
                    (
                        k
                        for k in cells_v
                        if cells_v[k]["depth"] in closed_depths
                        and cells_v[k]["hidden"] == 64
                    ),
                    key=gap_of,
                )
            )
            + "."
            if any(cells_v[k]["hidden"] == 64 for k in cells_v)
            else "."
        )
        if closed
        else "No cell reaches closure of 1.000000 at any depth or width tested."
    )

    # The competing mechanisms, and what each predicts.
    over_one = [k for k in cells_v if cells_v[k]["params_per_point"] >= 1.0]
    over_one_open = sorted((k for k in over_one if not reaches_one(k)), key=gap_of)
    ratio_txt = (
        "The parameters-per-datapoint reading is additionally contradicted by the "
        "table: it predicts that every cell above one parameter per datapoint "
        f"should close, and {', '.join(over_one_open)} "
        f"{'does' if len(over_one_open) == 1 else 'do'} not "
        f"(gap {', '.join(f'{gap_of(k):.1e}' for k in over_one_open)})."
        if over_one_open
        else ""
    )

    verdict = (
        "**Verdict pending. This experiment establishes the structure but does not "
        "determine the mechanism, and neither branch of the original question is "
        "decided by it.**"
        f" {open_txt} {closed_txt}"
        " The gap is ordered by depth, with width amplifying within depth; see the "
        "gap table above."
        "\n\n"
        "**Two mechanisms predict this pattern and this experiment does not separate "
        "them.** *Depth-dependent expressivity*: composing more layers may make the "
        "class genuinely closed under monotone reparameterization, because a deeper "
        "map can absorb a warp into its own hidden layers, in which case closure at "
        "the 128-wide cells is structural. *Interpolation*: those same cells hold "
        "more parameters than datapoints and can fit an arbitrary target on the "
        "observed points, in which case their closure carries no information about "
        "the model class. Both predict exactly what the table shows for the widest "
        f"cells. {ratio_txt}"
        "\n\n"
        "**The shuffled-target control has run and did not settle this.** "
        "(`experiments/closure_shuffled.py`, `paper/tables/closure_shuffled.md`.) It "
        "asked whether the closed cells are SELECTIVE: whether they re-represent a "
        "monotone warp while failing a random permutation and Gaussian noise. They "
        "are not -- at 128x2 and 128x3 the permutation is re-represented at R2 "
        "1.0000. But selectivity was the wrong instrument, because closure does not "
        "imply it. Closure under monotone reparameterization is a claim about what "
        "the function class CONTAINS, and a class flexible enough to contain the "
        "reparameterized twin is generally also flexible enough to fit a "
        "permutation. Failing the non-monotone targets is therefore not a property "
        "a closed class must have, and its absence is what any sufficiently "
        "flexible neural class would show. That control does establish something "
        "narrower and important: at exactly the cells reading 1.000000, a fit "
        "evaluated on the points it was fitted to cannot distinguish a class that "
        "contains the twin from one that memorizes n values at n points.\n\n"
        "**The discriminating experiment is the held-out control** "
        "(`experiments/closure_heldout.py`, `paper/tables/closure_heldout.md`). "
        "Every closure number in this table is in-sample: the refit is fitted on all "
        "n points and scored on those same n points. The held-out control refits on "
        "a subset and scores the warp target on points the refit never saw. "
        "Containment implies the twin agrees there, because it is the same FUNCTION; "
        "memorization does not. Until that is reported, whether a sufficiently "
        "flexible G-P map is closed on real MPSA data -- and therefore whether the "
        "strong identifiability claim holds or the mechanism test remains viable -- "
        "is **open**."
        f" The pipeline's numerical floor is the null control's closure, 1.000000 in "
        f"every cell (largest null gap {null_closure_gap:.2e}), so the shortfalls "
        f"above that floor are real. (The null EFFECT size reaches "
        f"{null_effect_floor:.4f}; effect size is a different quantity and is not "
        "used as the closure floor.)"
        + (
            " The power-law asymptote is reported above but is NOT used: b is pinned "
            "at its bound, because the closure gap is not monotone in parameter "
            "count -- depth and width do not trade off along a single axis."
            if fit_flag
            else ""
        )
    )
    agg["closure_reached"] = {
        "closed_cells": closed,
        "closed_depths": closed_depths,
        "open_depths": open_depths,
        "best_per_depth": {str(d): best_by_depth[d] for d in depths},
        "gap_per_depth_best": {str(d): gap_of(best_by_depth[d]) for d in depths},
        "overparameterized_not_closed": over_one_open,
        "null_closure_gap": null_closure_gap,
        "mechanism": "undetermined; pending experiments/closure_shuffled.py",
    }
    agg["gap_by_depth_width"] = {
        f"depth{d}": {f"w{w}": gap_of(f"{w}x{d}") for w in widths} for d in depths
    }
    agg["verdict"] = verdict
    agg["null_floor_effect_size"] = null_effect_floor
    agg["null_floor_closure_gap"] = null_closure_gap
    agg["best_cell"] = min(cells_v, key=gap_of)

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
    L.append("## Closure gap by depth and width\n")
    L.append(
        "The gap `1 - R²` at s=0.95, arranged the way the data is actually "
        "ordered. Depth sets what is reachable and width amplifies within a "
        "depth; parameters per datapoint cuts across both, which is why it does "
        "not order this table.\n"
    )
    L.append("| | " + " | ".join(f"width {w}" for w in widths) + " |")
    L.append("|---|" + "---|" * len(widths))
    for dp_ in depths:
        row = [f"**depth {dp_}**"]
        for w in widths:
            g_ = 1.0 - agg["cells"][f"{w}x{dp_}"]["closure_r2"]["mean"]
            row.append(f"{g_:.2e}" + (" **←1.000000**" if g_ < 5e-7 else ""))
        L.append("| " + " | ".join(row) + " |")
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
            f"Restricted to the {U['n_cells']} cells with fewer parameters than "
            f"datapoints ({', '.join(U['cells'])}), the same fit gives `g_inf` = "
            f"{U['gap_inf']:.3e} [{U['gap_inf_lo']:.3e}, {U['gap_inf_hi']:.3e}], "
            f"i.e. closure tends to **{U['closure_inf']:.6f}** "
            f"[{U['closure_inf_lo']:.6f}, {U['closure_inf_hi']:.6f}].\n\n"
            f"**Neither fit is used, and neither number should be quoted.** Both "
            f"have b pinned at the bound, because the gap is not monotone in "
            f"parameter count. The split above is by parameters per datapoint, and "
            f"the gap table shows that ratio does not order the cells either: "
            f"128x1 is above it and far from closure while 16x3 is below it and "
            f"comparable. Both fits are retained only to document that a power law "
            f"in parameter count fails on this data.\n"
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
