"""Does within-locus invariance survive in-silico mutagenesis, and up to what effect size?

Under a monotone warp psi the autograd gradient picks up a per-INSTANCE
scalar:

    grad(psi . phi)(x) = psi'(phi(x)) grad phi(x),

so within-instance attribution ratios are invariant and only cross-instance
magnitudes move. The field's primitive is not the gradient but in-silico
mutagenesis, a finite difference. There the mean value theorem gives

    psi(phi(x'_j)) - psi(phi(x)) = psi'(xi_j) (phi(x'_j) - phi(x)),
    xi_j strictly between phi(x) and phi(x'_j),

so the multiplier is per-MUTATION, and two mutations at the same locus are
scaled by psi'(xi_j) and psi'(xi_k) with xi_j != xi_k. Within-locus
invariance therefore does not transfer.

That is a statement about exactness, not about magnitude. When single-mutation
effects are small, xi_j and xi_k both sit close to phi(x), psi' is nearly the
same at both, and the ratios nearly survive; when effects are large they
diverge. This experiment measures where it breaks, which is what decides
whether the constructive rule is usable in practice.

Design notes:

- The twin here is the ANALYTIC one, psi . phi_hat. That isolates the mean
  value theorem effect. The in-class representative that realizes the twin
  adds further error on top (it matches the warped latent in value at closure
  R^2 = 1.000000 and still departs in gradient), so the numbers below are a
  LOWER bound on the departure a practitioner would see.
- The autograd arm is a control, not a result: the theory says its
  within-locus ranking is exactly invariant. If it is not ~1.000 the pipeline
  is wrong rather than the theory.
- ISM latents under the reference fit are computed once; each warp is then a
  pointwise transform of the stored values, so the whole grid is cheap.

Usage:
    python -m experiments.ism_invariance --config configs/base.yaml
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
from scipy.stats import spearmanr

from experiments._common import (
    configure_torch,
    fmt_ci,
    load_config,
    make_run_dir,
    mean_ci,
    resolve_seeds,
    to_jsonable,
    write_tuning_budget,
)
from experiments.identifiability_probe import (
    LatentModel,
    fit_model,
    is_monotone,
    make_dataset,
    warp,
)

# Effect-size bins in latent (logit) units. Fixed rather than quantile-based so
# the threshold is reported in interpretable units.
BIN_EDGES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.75, 3.0, float("inf"))


def ism_latents(
    model: LatentModel, x: torch.Tensor, n_alt: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference latent per instance, and latent for every single-position mutant.

    Returns (phi_ref [N], phi_mut [N, d, n_alt]). A "position" is a node and a
    "mutation" replaces that node's signal feature with an alternative drawn
    from the feature distribution, the analogue of an alternative base.
    """
    n, d = x.shape
    g = torch.Generator().manual_seed(seed + 4242)
    alts = torch.randn(d, n_alt, generator=g, dtype=torch.float64)
    with torch.no_grad():
        phi_ref = model.latent(x).detach()
        phi_mut = torch.empty(n, d, n_alt, dtype=torch.float64)
        for j in range(d):
            for a in range(n_alt):
                xm = x.clone()
                xm[:, j] = alts[j, a]
                phi_mut[:, j, a] = model.latent(xm).detach()
    return phi_ref, phi_mut


def autograd_attr(model: LatentModel, x: torch.Tensor) -> torch.Tensor:
    xr = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(model.latent(xr).sum(), xr)
    return grad.detach()


def _within_locus_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman between two attribution vectors for ONE instance."""
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _ratio_error(a: np.ndarray, b: np.ndarray, max_pairs: int = 200) -> float:
    """Median |log ratio-of-ratios| over mutation pairs at one locus.

    Exactly |log(psi'(xi_j) / psi'(xi_k))|; zero iff the multiplier is common.
    """
    idx = np.where((a > 1e-12) & (b > 1e-12))[0]
    if idx.size < 2:
        return float("nan")
    pairs = list(itertools.combinations(idx.tolist(), 2))
    if len(pairs) > max_pairs:
        step = len(pairs) // max_pairs
        pairs = pairs[::step][:max_pairs]
    errs = [abs(np.log(b[j] / b[k]) - np.log(a[j] / a[k])) for j, k in pairs]
    return float(np.median(errs))


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    ip = cfg.identifiability
    im = cfg.ism
    x, _phi_true, y = make_dataset(cfg, seed)
    d = x.shape[1]

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed * 7919)
    model = LatentModel("neural", d, int(ip.hidden), int(ip.ge_components), gen)
    fit_model(model, x, y, int(ip.epochs), float(ip.lr))

    phi_ref, phi_mut = ism_latents(model, x, int(im.n_alt), seed)
    grad_ref = autograd_attr(model, x)
    # the warp acts on the standardized latent, as in identifiability_probe
    mu, sd = float(phi_ref.mean()), float(phi_ref.std())
    z_ref = (phi_ref - mu) / sd
    z_mut = (phi_mut - mu) / sd

    delta = (phi_mut - phi_ref[:, None, None]).reshape(len(x), -1)  # [N, d*n_alt]
    spread = delta.abs().max(dim=1).values.numpy()  # max_j |phi(x'_j) - phi(x)|

    rows: list[dict[str, Any]] = []
    for family in list(ip.radius.families):
        for strength in list(ip.radius.strengths):
            omega = float(im.omega)
            if not is_monotone(family, float(strength), omega, seed):
                continue
            w_ref = warp(z_ref, family, float(strength), omega, seed)
            w_mut = warp(
                z_mut.reshape(-1), family, float(strength), omega, seed
            ).reshape(z_mut.shape)
            delta_tw = (w_mut - w_ref[:, None, None]).reshape(len(x), -1)

            # autograd control: the twin's gradient is psi'(phi) * grad phi, a
            # per-instance positive scalar, so the within-locus ranking is exact.
            # psi' is taken by autograd on warp itself so it is correct for every
            # family (warp_derivative in identifiability_probe is sinusoid-only).
            zr = z_ref.clone().requires_grad_(True)
            (psi_p,) = torch.autograd.grad(
                warp(zr, family, float(strength), omega, seed).sum(), zr
            )
            psi_p = psi_p.detach()
            grad_tw = psi_p[:, None] * grad_ref

            a_np, b_np = delta.abs().numpy(), delta_tw.abs().numpy()
            ga, gb = grad_ref.abs().numpy(), grad_tw.abs().numpy()
            for i in range(len(x)):
                rows.append(
                    {
                        "family": family,
                        "strength": float(strength),
                        "instance": i,
                        "spread": float(spread[i]),
                        "ism_spearman": _within_locus_spearman(a_np[i], b_np[i]),
                        "ism_ratio_error": _ratio_error(a_np[i], b_np[i]),
                        "grad_spearman": _within_locus_spearman(ga[i], gb[i]),
                        "grad_ratio_error": _ratio_error(ga[i], gb[i]),
                    }
                )
    return {
        "seed": seed,
        "rows": rows,
        "spread_quantiles": {
            str(q): float(np.quantile(spread, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "n_instances": int(len(x)),
        "n_positions": int(d),
        "n_alternatives": int(im.n_alt),
    }


def bin_index(v: float) -> int:
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= v < BIN_EDGES[i + 1]:
            return i
    return len(BIN_EDGES) - 2


def threshold_effect_size(
    binned: dict[int, dict[str, Any]], key: str, cut: float
) -> float:
    """Effect size at which the binned statistic first drops below `cut`.

    Linear interpolation in the bin's median effect size between the last bin
    above the cut and the first bin below it; nan if it never drops.
    """
    order = sorted(binned)
    prev = None
    for b in order:
        m = binned[b][key]["mean"]
        e = binned[b]["median_effect"]
        if not np.isfinite(m):
            continue
        if m < cut:
            if prev is None:
                return float(e)
            (pe, pm) = prev
            if pm == m:
                return float(e)
            t = (pm - cut) / (pm - m)
            return float(pe + t * (e - pe))
        prev = (e, m)
    return float("nan")


def make_table(agg: dict[str, Any], cfg: Any, meta: dict[str, Any]) -> str:
    im = cfg.ism
    L = ["# Within-locus ISM invariance under a monotone warp\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}, config "
        f"`{meta['config_hash']}`, {agg['n_seeds']} seeds, mean [95% bootstrap CI]. "
        f"{agg['n_instances']} instances x {agg['n_positions']} positions x "
        f"{agg['n_alternatives']} alternatives per position.\n"
    )
    L.append(
        "The autograd arm is a CONTROL, not a result: under a monotone warp the gradient "
        "scales by the per-instance factor psi'(phi(x)), so its within-locus ranking is exactly "
        "invariant. A value below 1.000 there means the pipeline is wrong, not the theory. "
        "The ISM arm is a finite difference, where the mean value theorem gives psi'(xi_j) at a "
        "point between phi(x) and phi(x'_j), hence a per-MUTATION multiplier.\n"
    )
    L.append(
        "The twin is the analytic psi . phi_hat, which isolates the mean-value-theorem effect; "
        "the in-class representative adds further departure on top, so these are a LOWER bound.\n"
    )

    L.append("## Control: autograd within-locus ranking (must be 1.000)\n")
    L.append(
        "| warp family | strength | within-locus Spearman | ratio-invariance error |"
    )
    L.append("|---|---|---|---|")
    for fam in agg["families"]:
        for s in agg["strengths"]:
            c = agg["control"].get((fam, s))
            if c:
                L.append(
                    f"| {fam} | {s:g} | {fmt_ci(c['grad_spearman'])} | {fmt_ci(c['grad_ratio_error'])} |"
                )
    L.append("")

    L.append("## ISM within-locus ranking by single-mutation effect size\n")
    L.append(
        "Instances binned by the spread of their single-mutation latent effects, "
        "max_j |phi(x'_j) - phi(x)|, in latent (logit) units.\n"
    )
    for fam in agg["families"]:
        L.append(f"### {fam}\n")
        L.append(
            "| effect-size bin | median effect | "
            + " | ".join(f"s={s:g}" for s in agg["strengths"])
            + " |"
        )
        L.append("|---" * (len(agg["strengths"]) + 2) + "|")
        for b in sorted(agg["bins"].get(fam, {})):
            lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
            label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
            cells = []
            med = None
            for s in agg["strengths"]:
                e = agg["bins"][fam][b].get(s)
                if e:
                    med = e["median_effect"]
                    cells.append(f"{e['ism_spearman']['mean']:.3f}")
                else:
                    cells.append("n/a")
            L.append(
                f"| {label} | {med:.3f} | " % ()
                if False
                else f"| {label} | {(med if med is not None else float('nan')):.3f} | "
                + " | ".join(cells)
                + " |"
            )
        L.append("")

    L.append("## Headline: where within-locus ISM ranking breaks\n")
    L.append(
        f"Effect size (latent units) at which the binned within-locus Spearman drops below {im.cut:g}:\n"
    )
    L.append(
        "| warp family | " + " | ".join(f"s={s:g}" for s in agg["strengths"]) + " |"
    )
    L.append("|---" * (len(agg["strengths"]) + 1) + "|")
    for fam in agg["families"]:
        cells = []
        for s in agg["strengths"]:
            t = agg["threshold"].get((fam, s))
            cells.append("never" if t is None or not np.isfinite(t) else f"{t:.2f}")
        L.append(f"| {fam} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(agg["verdict"])
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    configure_torch(cfg)
    run = make_run_dir(cfg, "ism_invariance", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    print(f"run dir: {run}")

    per_seed = []
    for seed in resolve_seeds(cfg):
        r = run_seed(cfg, seed)
        per_seed.append(r)
        sm = np.nanmean([w["ism_spearman"] for w in r["rows"]])
        gm = np.nanmean([w["grad_spearman"] for w in r["rows"]])
        print(
            f"seed {seed}: mean ISM rho {sm:.3f}  autograd control {gm:.3f}", flush=True
        )
        (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    nb = int(cfg.bootstrap_resamples)
    fams = list(cfg.identifiability.radius.families)
    strengths = [float(s) for s in cfg.identifiability.radius.strengths]
    agg: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "families": fams,
        "strengths": strengths,
        "n_instances": per_seed[0]["n_instances"],
        "n_positions": per_seed[0]["n_positions"],
        "n_alternatives": per_seed[0]["n_alternatives"],
        "spread_quantiles": per_seed[0]["spread_quantiles"],
    }

    def collect(pred) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {
            k: []
            for k in (
                "ism_spearman",
                "ism_ratio_error",
                "grad_spearman",
                "grad_ratio_error",
            )
        }
        eff: list[float] = []
        for rec in per_seed:
            sel = [w for w in rec["rows"] if pred(w)]
            if not sel:
                continue
            for k in out:
                vals = [w[k] for w in sel if w[k] is not None and np.isfinite(w[k])]
                if vals:
                    out[k].append(float(np.mean(vals)))
            eff += [w["spread"] for w in sel]
        out["_median_effect"] = [float(np.median(eff))] if eff else []
        return out

    control: dict[tuple[str, float], dict[str, Any]] = {}
    for fam in fams:
        for s in strengths:
            c = collect(
                lambda w, f=fam, ss=s: w["family"] == f
                and abs(w["strength"] - ss) < 1e-9
            )
            if c["grad_spearman"]:
                control[(fam, s)] = {
                    k: mean_ci(v, n_boot=nb)
                    for k, v in c.items()
                    if not k.startswith("_")
                }
    agg["control"] = control

    bins: dict[str, dict[int, dict[float, Any]]] = {}
    for fam in fams:
        bins[fam] = {}
        for b in range(len(BIN_EDGES) - 1):
            for s in strengths:
                c = collect(
                    lambda w, f=fam, ss=s, bb=b: w["family"] == f
                    and abs(w["strength"] - ss) < 1e-9
                    and bin_index(w["spread"]) == bb
                )
                if c["ism_spearman"] and c["_median_effect"]:
                    bins[fam].setdefault(b, {})[s] = {
                        **{
                            k: mean_ci(v, n_boot=nb)
                            for k, v in c.items()
                            if not k.startswith("_")
                        },
                        "median_effect": c["_median_effect"][0],
                    }
    agg["bins"] = bins

    cut = float(cfg.ism.cut)
    threshold: dict[tuple[str, float], float] = {}
    for fam in fams:
        for s in strengths:
            per_bin = {b: v[s] for b, v in bins[fam].items() if s in v}
            threshold[(fam, s)] = (
                threshold_effect_size(per_bin, "ism_spearman", cut)
                if per_bin
                else float("nan")
            )
    agg["threshold"] = threshold

    ctrl_ok = all(
        c["grad_spearman"]["lo"] > 0.999
        for c in control.values()
        if np.isfinite(c["grad_spearman"]["lo"])
    )
    finite = [t for t in threshold.values() if np.isfinite(t)]
    q = agg["spread_quantiles"]
    if not ctrl_ok:
        verdict = (
            "**Pipeline failure, not a result.** The autograd control is below 1.000 where the "
            "theory says the within-locus ranking is exactly invariant, so the measurement is "
            "wrong and the ISM numbers must not be read."
        )
    elif not finite:
        verdict = (
            "**Within-locus ISM ranking does not break anywhere on the grid.** The Spearman stays "
            f"above {cut:g} in every effect-size bin, at every warp strength and family tested. On "
            "this evidence the constructive rule survives the move from gradients to in-silico "
            "mutagenesis, and the per-mutation multiplier of the mean value theorem is too weak to "
            "reorder mutations at a locus."
        )
    else:
        worst = min(finite)
        verdict = (
            f"**Within-locus ISM ranking breaks at a single-mutation effect of about "
            f"{worst:.2f} latent units** in the worst family and strength tested "
            f"(Spearman below {cut:g}). The substrate's own single-mutation effects have median "
            f"{q['0.5']:.2f} and 95th percentile {q['0.95']:.2f} latent units, so the break "
            + (
                "falls INSIDE the range this substrate produces, and the constructive rule is not "
                "usable as stated."
                if worst < q["0.95"]
                else "sits ABOVE the range this substrate produces, so the rule survives here."
            )
        )
    agg["verdict"] = verdict
    agg["control_ok"] = bool(ctrl_ok)

    table = make_table(agg, cfg, meta)
    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "ism_invariance.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    k: (
                        {str(kk): vv for kk, vv in v.items()}
                        if isinstance(v, dict)
                        else v
                    )
                    for k, v in agg.items()
                }
            ),
            indent=1,
        )
    )
    (run / "per_seed_values.json").write_text(
        json.dumps(
            {
                "mean_ism_spearman": {
                    str(r["seed"]): float(
                        np.nanmean([w["ism_spearman"] for w in r["rows"]])
                    )
                    for r in per_seed
                },
                "mean_grad_spearman": {
                    str(r["seed"]): float(
                        np.nanmean([w["grad_spearman"] for w in r["rows"]])
                    )
                    for r in per_seed
                },
            },
            indent=1,
        )
    )
    write_tuning_budget(
        run,
        [
            {
                "model": "neural",
                "configs_tried": 1,
                "epochs": int(cfg.identifiability.epochs),
                "gradient_steps": int(cfg.identifiability.epochs),
                "search_space": "the pre-selected identifiability config; no search",
                "selection": "final iterate; identifiability, not generalization",
            }
        ],
    )
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
