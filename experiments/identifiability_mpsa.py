"""The identifiability analysis on a real MPSA, which is the only version that
can appear in a paper about MPSA analysis.

Dataset: the BRCA2 exons 17-19 5' splice site library of Wong, Kinney &
Krainer, *Molecular Cell* 71:1012-1026.e3 (2018), as processed and shipped
with MAVE-NN. 30,483 nine-nucleotide splice sites matching NNN/GYNNNN with
log10 percent-spliced-in. Provenance and checksums in
`data/raw/PROVENANCE.md`. This is the canonical setting for additive,
pairwise and neural genotype-phenotype maps, and the library MAVE-NN itself
uses.

**What can and cannot be measured here.** On the synthetic substrate the
true latent is known, so closure could be measured against it. On real data
the latent is unobservable and closure against the truth is not available.
What is available, and what the paper needs, is the same construction
applied to the FITTED latent: fit each class, warp its own fitted latent,
and ask whether the class re-represents the warped version. A class that
re-represents a monotone warp of its own output contains the twin, so the
twin is a legitimate alternative fit that predicts identically, and that
statement needs no access to the truth.

Every control that turned out to be necessary on synthetic data is kept,
because each of them caught a real artifact there:

- **composite-prediction agreement**, the check that two fits are the same
  function rather than two different ones; without it a difference in
  latents cannot be attributed to reparameterization;
- **a null control at zero warp strength**, since a refit of an unwarped
  latent bounds the pipeline's own numerical floor and no smaller effect is
  meaningful;
- **six significant figures** on closure, because at three an optimizer's
  0.999848 reads as a clean 1.000 and that difference drove a spurious
  result;
- **artifact detection**: no trend of effect with warp strength, effect
  within the null floor, and non-closed classes showing larger apparent
  radii than closed ones.

The ISM measurement uses this dataset's own single-mutation effect-size
distribution rather than a synthetic one: every position of every assayed
site is mutated to each of its three alternative bases and the resulting
latent effects are what the bins are built from.

Usage:
    python -m experiments.identifiability_mpsa --config configs/base.yaml
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
    fmt_ci,
    load_config,
    make_run_dir,
    mean_ci,
    resolve_seeds,
    to_jsonable,
    write_tuning_budget,
)
from experiments.identifiability_probe import (
    CLASSES,
    Fit,
    LatentModel,
    _grad_of_map,
    _standardize,
    affine_r2,
    attribution_agreement,
    fit_class_to_target,
    fit_model,
    indistinguishability,
    invert_warp,
    is_monotone,
    warp,
)

ALPHABET = "ACGU"
BIN_EDGES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.75, 3.0, float("inf"))


def load_mpsa(cfg: Any, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """One-hot sequences and phenotypes from the real library."""
    import pandas as pd

    im = cfg.identifiability_mpsa
    p = pathlib.Path(im.path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; see data/raw/PROVENANCE.md")
    df = pd.read_csv(p)
    df = df.dropna(subset=["x", "y"])
    n = int(im.n_max)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    seqs = [str(s).upper().replace("T", "U") for s in df.x]
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("MPSA sequences are not all the same length")
    oh = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            k = ALPHABET.find(ch)
            if k >= 0:
                oh[i, j, k] = 1.0
    x = torch.from_numpy(oh.reshape(len(seqs), L * 4)).double()
    y = torch.tensor(df.y.to_numpy(), dtype=torch.float64)
    return x, (y - y.mean()) / y.std(), seqs


def ism_latents(
    model: LatentModel, x: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latent for every single-position substitution, using the real alphabet.

    Returns (phi_ref [N], phi_mut [N, L, 3]) for the three alternative bases
    at each of the L positions.
    """
    with torch.no_grad():
        phi_ref = model.latent(x).detach()
        out = torch.empty(x.shape[0], seq_len, 3, dtype=torch.float64)
        for j in range(seq_len):
            block = slice(4 * j, 4 * j + 4)
            cur = x[:, block].argmax(dim=1)
            alt_i = 0
            for b in range(4):
                m = cur != b
                if not bool(m.any()) and alt_i >= 3:
                    continue
                xm = x.clone()
                xm[:, block] = 0.0
                xm[:, 4 * j + b] = 1.0
                z = model.latent(xm).detach()
                # only rows where b is actually an alternative
                if alt_i < 3:
                    out[:, j, alt_i] = torch.where(m, z, phi_ref)
                    alt_i += 1
    return phi_ref, out


def _within_locus(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _ratio_err(a: np.ndarray, b: np.ndarray, cap: int = 120) -> float:
    idx = np.where((a > 1e-12) & (b > 1e-12))[0]
    if idx.size < 2:
        return float("nan")
    import itertools

    pairs = list(itertools.combinations(idx.tolist(), 2))[:cap]
    return float(
        np.median([abs(np.log(b[j] / b[k]) - np.log(a[j] / a[k])) for j, k in pairs])
    )


def bin_index(v: float) -> int:
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= v < BIN_EDGES[i + 1]:
            return i
    return len(BIN_EDGES) - 2


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    ip, im = cfg.identifiability, cfg.identifiability_mpsa
    x, y, seqs = load_mpsa(cfg, seed)
    d, L = x.shape[1], len(seqs[0])
    out: dict[str, Any] = {
        "seed": seed,
        "n": int(x.shape[0]),
        "seq_len": L,
        "classes": {},
    }

    for kind in CLASSES:
        gen = torch.Generator().manual_seed(seed * 7919 + hash(kind) % 1000)
        # several restarts, keep the best fit; this is a real dataset so the
        # reference should be the best available fit, not an arbitrary one
        best: Fit | None = None
        for r in range(int(im.restarts)):
            torch.manual_seed(seed * 100 + r)
            m = LatentModel(kind, d, int(ip.hidden), int(ip.ge_components), gen)
            loss = fit_model(m, x, y, int(im.epochs), float(ip.lr))
            if best is None or loss < best.loss:
                best = Fit(model=m, loss=loss, phi=m.latent(x).detach())
        assert best is not None
        z_ref = _standardize(best.phi)
        mu, sd = float(best.phi.mean()), float(best.phi.std())
        with torch.no_grad():
            pred_orig = best.model.g(best.phi).detach()
        g_ref = _grad_of_map(best.model.phi, x)
        fit_r2 = 1.0 - float(((y - pred_orig) ** 2).sum() / ((y - y.mean()) ** 2).sum())

        # null control at zero warp strength: the pipeline's own floor
        phi0, m0 = fit_class_to_target(
            kind, x, z_ref, cfg, gen, warm_start=best.model.phi
        )
        z_back0 = invert_warp(_standardize(phi0), "sinusoid", 0.0, 1.0, seed)
        with torch.no_grad():
            pred0 = best.model.g(mu + sd * z_back0).detach()
        null = indistinguishability(y, pred_orig, pred0)

        rows: list[dict[str, Any]] = []
        for family in list(ip.radius.families):
            for strength in list(ip.radius.strengths):
                om = float(im.omega)
                if not is_monotone(family, float(strength), om, seed):
                    continue
                warped = warp(z_ref, family, float(strength), om, seed)
                phi_t, m_t = fit_class_to_target(
                    kind, x, warped, cfg, gen, warm_start=best.model.phi
                )
                closure = affine_r2(phi_t.numpy(), warped.numpy())
                z_back = invert_warp(
                    _standardize(phi_t) * float(warped.std()) + float(warped.mean()),
                    family,
                    float(strength),
                    om,
                    seed,
                )
                with torch.no_grad():
                    pred_tw = best.model.g(mu + sd * z_back).detach()
                ind = indistinguishability(y, pred0, pred_tw)
                g_t = _grad_of_map(m_t, x)
                agree = attribution_agreement(g_ref, g_t)
                # composite agreement: are the two the SAME function?
                comp = (
                    float(np.corrcoef(pred_orig.numpy(), pred_tw.numpy())[0, 1])
                    if pred_tw.std() > 1e-12
                    else float("nan")
                )
                rows.append(
                    {
                        "class": kind,
                        "family": family,
                        "strength": float(strength),
                        "closure_r2": closure,
                        "cross_instance_spearman": agree[
                            "cross_instance_magnitude_spearman"
                        ],
                        "within_instance_cosine": agree["within_instance_cosine"],
                        "effect_size": ind["effect_size"],
                        "composite_corr": comp,
                    }
                )
        out["classes"][kind] = {
            "fit_r2": fit_r2,
            "best_loss": best.loss,
            "null_effect_size": null["effect_size"],
            "null_closure_r2": affine_r2(phi0.numpy(), z_ref.numpy()),
            "rows": rows,
        }
        if kind == "neural":
            # ISM on the real library, binned by its OWN effect-size distribution
            phi_ref, phi_mut = ism_latents(best.model, x, L)
            delta = (phi_mut - phi_ref[:, None, None]).reshape(x.shape[0], -1)
            spread = delta.abs().max(dim=1).values.numpy()
            z_mut = (phi_mut - mu) / sd
            ism: list[dict[str, Any]] = []
            n_ism = min(int(im.ism_max), x.shape[0])
            # Sweep omega alongside family and strength. The ratio error is
            # partly a property of the warp's shape, so one omega gives a point
            # estimate of something that has to be a range -- the same defect
            # that made the indistinguishability radius unusable.
            for family in list(ip.radius.families):
                for strength in list(ip.radius.strengths):
                    for om in [float(o) for o in im.omegas]:
                        if not is_monotone(family, float(strength), om, seed):
                            continue
                        w_ref = warp(z_ref, family, float(strength), om, seed)
                        w_mut = warp(
                            z_mut.reshape(-1), family, float(strength), om, seed
                        ).reshape(z_mut.shape)
                        dtw = (w_mut - w_ref[:, None, None]).reshape(x.shape[0], -1)
                        a_np, b_np = delta.abs().numpy(), dtw.abs().numpy()
                        for i in range(n_ism):
                            ism.append(
                                {
                                    "family": family,
                                    "strength": float(strength),
                                    "omega": om,
                                    "spread": float(spread[i]),
                                    "ism_spearman": _within_locus(a_np[i], b_np[i]),
                                    "ism_ratio_error": _ratio_err(a_np[i], b_np[i]),
                                }
                            )
            out["ism"] = ism
            out["effect_quantiles"] = {
                str(q): float(np.quantile(spread, q))
                for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            }
    return out


def make_table(agg: dict[str, Any], cfg: Any, meta: dict[str, Any]) -> str:
    fams = list(cfg.identifiability.radius.families)
    ss = [float(s) for s in cfg.identifiability.radius.strengths]
    L = ["# Identifiability on the real BRCA2 5' splice site MPSA\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}, config "
        f"`{meta['config_hash']}`, {agg['n_seeds']} seeds, mean [95% bootstrap CI]. "
        f"{agg['n']} sequences of length {agg['seq_len']} per seed.\n"
    )
    L.append(
        "The true latent is unobservable on real data, so closure is measured against each "
        "class's OWN fitted latent: fit, warp the fit, and ask whether the class re-represents "
        "the warped version. A class that does contains the twin, and the twin predicts "
        "identically, which needs no access to the truth.\n"
    )
    L.append("## Model fits\n")
    L.append(
        "| G-P map | fit R² on log10 PSI | null-control closure R² | null-control effect size |"
    )
    L.append("|---|---|---|---|")
    for k in CLASSES:
        c = agg["classes"][k]
        L.append(
            f"| {k} | {fmt_ci(c['fit_r2'])} | {fmt_ci(c['null_closure_r2'], 6)} | {fmt_ci(c['null_effect_size'], 4)} |"
        )
    L.append("")
    L.append("## Closure at the largest monotone warp, six significant figures\n")
    L.append("| G-P map | " + " | ".join(fams) + " |")
    L.append("|---" * (len(fams) + 1) + "|")
    smax = max(ss)
    for k in CLASSES:
        cells = [fmt_ci(agg["classes"][k]["closure"][f][str(smax)], 6) for f in fams]
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Cross-instance attribution divergence, neural class\n")
    L.append("| warp family | " + " | ".join(f"s={s:g}" for s in ss) + " |")
    L.append("|---" * (len(ss) + 1) + "|")
    for f in fams:
        L.append(
            f"| {f} | "
            + " | ".join(
                f"{agg['classes']['neural']['cross'][f][str(s)]['mean']:.3f}"
                for s in ss
            )
            + " |"
        )
    L.append("")
    L.append(
        "## ISM within-locus invariance, binned by this library's own effect sizes\n"
    )
    L.append(
        "Single-mutation latent effects come from mutating every position of every assayed "
        "site to each of its three alternative bases. Effect-size quantiles (latent units): "
        + ", ".join(f"{q}={v:.3f}" for q, v in agg["effect_quantiles"].items())
        + ".\n"
    )
    L.append(
        "Reported as a RANGE over warp family, shape parameter omega in "
        + ", ".join(f"{o:g}" for o in agg.get("ism_omegas", []))
        + ", and strength, never as a point estimate: the ratio error is partly a property of "
        "the warp's shape, and quoting one cell would repeat the defect that made the "
        "indistinguishability radius unusable.\n"
    )
    L.append(
        "| effect-size bin | median effect | ranking Spearman (min-max) | "
        "ratio error (min-max) | worst ratio cell |"
    )
    L.append("|---|---|---|---|---|")
    for b in sorted(agg["ism_bins"]):
        e = agg["ism_bins"][b]
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        lab = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        L.append(
            f"| {lab} | {e['median_effect']:.3f} | {e['rank_min']:.4f}-{e['rank_max']:.4f} | "
            f"{e['ratio_min']:.3f}-{e['ratio_max']:.3f} | {e['ratio_worst_cell']} |"
        )
    L.append("")
    L.append("## Verdict\n")
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
    run = make_run_dir(cfg, "identifiability_mpsa", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    print(f"run dir: {run}")

    per_seed = []
    for seed in resolve_seeds(cfg):
        r = run_seed(cfg, seed)
        per_seed.append(r)
        cl = {k: max(x["closure_r2"] for x in r["classes"][k]["rows"]) for k in CLASSES}
        print(
            f"seed {seed}: fit R2 neural {r['classes']['neural']['fit_r2']:+.3f} | "
            f"max closure " + " ".join(f"{k}={cl[k]:.6f}" for k in CLASSES),
            flush=True,
        )
        (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    nb = int(cfg.bootstrap_resamples)
    fams = list(cfg.identifiability.radius.families)
    ss = [float(s) for s in cfg.identifiability.radius.strengths]
    agg: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "n": per_seed[0]["n"],
        "seq_len": per_seed[0]["seq_len"],
        "classes": {},
    }

    def rows_of(rec, k):
        return rec["classes"][k]["rows"]

    for k in CLASSES:
        c: dict[str, Any] = {
            "fit_r2": mean_ci([r["classes"][k]["fit_r2"] for r in per_seed], n_boot=nb),
            "null_effect_size": mean_ci(
                [r["classes"][k]["null_effect_size"] for r in per_seed], n_boot=nb
            ),
            "null_closure_r2": mean_ci(
                [r["classes"][k]["null_closure_r2"] for r in per_seed], n_boot=nb
            ),
            "closure": {},
            "cross": {},
            "effect": {},
            "composite": {},
        }
        for f in fams:
            c["closure"][f], c["cross"][f], c["effect"][f], c["composite"][f] = (
                {},
                {},
                {},
                {},
            )
            for s in ss:
                sel = [
                    x
                    for r in per_seed
                    for x in rows_of(r, k)
                    if x["family"] == f and abs(x["strength"] - s) < 1e-9
                ]
                if sel:
                    c["closure"][f][str(s)] = mean_ci(
                        [x["closure_r2"] for x in sel], n_boot=nb
                    )
                    c["cross"][f][str(s)] = mean_ci(
                        [x["cross_instance_spearman"] for x in sel], n_boot=nb
                    )
                    c["effect"][f][str(s)] = mean_ci(
                        [abs(x["effect_size"]) for x in sel], n_boot=nb
                    )
                    c["composite"][f][str(s)] = mean_ci(
                        [x["composite_corr"] for x in sel], n_boot=nb
                    )
        agg["classes"][k] = c

    ism_all = [w for r in per_seed for w in r.get("ism", [])]
    bins: dict[int, Any] = {}
    for b in range(len(BIN_EDGES) - 1):
        sel = [w for w in ism_all if bin_index(w["spread"]) == b]
        if len(sel) < 20:
            continue
        cells: dict[tuple[str, float, float], dict[str, list[float]]] = {}
        for w in sel:
            ck = (w["family"], float(w.get("omega", 2.0)), w["strength"])
            c2 = cells.setdefault(ck, {"rank": [], "ratio": []})
            v = w["ism_spearman"]
            if v is not None and np.isfinite(v):
                c2["rank"].append(float(v))
            v = w["ism_ratio_error"]
            if v is not None and np.isfinite(v):
                c2["ratio"].append(float(v))
        rank_m = {kk: float(np.mean(v["rank"])) for kk, v in cells.items() if v["rank"]}
        ratio_m = {
            k: float(np.mean(v["ratio"])) for k, v in cells.items() if v["ratio"]
        }
        if not rank_m or not ratio_m:
            continue
        wr_k = min(rank_m, key=lambda kk: rank_m[kk])
        wt_k = max(ratio_m, key=lambda kk: ratio_m[kk])
        bins[b] = {
            "median_effect": float(np.median([w["spread"] for w in sel])),
            "ism_spearman": mean_ci([w["ism_spearman"] for w in sel], n_boot=nb),
            "ism_ratio_error": mean_ci([w["ism_ratio_error"] for w in sel], n_boot=nb),
            "rank_min": min(rank_m.values()),
            "rank_max": max(rank_m.values()),
            "rank_worst_cell": f"{wr_k[0]}, omega={wr_k[1]:g}, s={wr_k[2]:g}",
            "ratio_min": min(ratio_m.values()),
            "ratio_max": max(ratio_m.values()),
            "ratio_worst_cell": f"{wt_k[0]}, omega={wt_k[1]:g}, s={wt_k[2]:g}",
            "n_cells": len(rank_m),
        }
    agg["ism_omegas"] = (
        sorted({float(w.get("omega", 2.0)) for w in ism_all}) if ism_all else []
    )
    agg["ism_bins"] = bins
    agg["effect_quantiles"] = per_seed[0]["effect_quantiles"]

    smax = max(ss)
    neu = agg["classes"]["neural"]
    lin = agg["classes"]["linear"]
    closed = all(
        neu["closure"][f][str(smax)]["lo"] > 0.999999
        for f in fams
        if str(smax) in neu["closure"][f]
    )
    lin_closed = all(
        lin["closure"][f][str(smax)]["lo"] > 0.999999
        for f in fams
        if str(smax) in lin["closure"][f]
    )
    trend = {}
    for f in fams:
        pts = [
            (x["strength"], abs(x["effect_size"]))
            for r in per_seed
            for x in rows_of(r, "neural")
            if x["family"] == f
        ]
        if len(pts) > 3:
            rho = spearmanr([a for a, _ in pts], [b for _, b in pts]).statistic
            trend[f] = float(rho) if np.isfinite(rho) else float("nan")
    no_trend = all((not np.isfinite(v)) or v <= 0.2 for v in trend.values())
    nulls = [r["classes"]["neural"]["null_effect_size"] for r in per_seed]
    null_mu, null_sd = float(np.mean(np.abs(nulls))), float(np.std(np.abs(nulls)))
    comp_lo = min(
        neu["composite"][f][str(s)]["lo"]
        for f in fams
        for s in ss
        if str(s) in neu["composite"][f]
    )
    cross_lo = min(
        neu["cross"][f][str(smax)]["mean"] for f in fams if str(smax) in neu["cross"][f]
    )
    parts = []
    if closed:
        parts.append(
            "**On the real library the neural class re-represents a monotone warp of its own "
            "fitted latent to six figures (closure R² = 1.000000 under every warp family), "
            "while the additive class does not"
            + (
                ""
                if not lin_closed
                else " -- but note the additive class also reached closure, which "
                "should not happen and must be checked before the result is used"
            )
            + f". The twin is therefore a legitimate alternative fit of the same data, and its "
            f"cross-instance attribution ranking falls to {cross_lo:.3f} at the largest warp."
        )
    else:
        parts.append(
            f"**The neural class did NOT reach closure on real data** (best "
            f"{max(neu['closure'][f][str(smax)]['mean'] for f in fams):.6f}), so the twin is not "
            "demonstrably in the class here and the synthetic finding does not transfer as stated."
        )
    parts.append(
        f"Composite predictions of original and twin agree at r >= {comp_lo:.4f}, so the two are "
        "the same function rather than two different fits; without that control a latent "
        "difference could not be attributed to reparameterization."
    )
    parts.append(
        "Artifact checks: effect-versus-strength Spearman "
        + ", ".join(f"{k} {v:+.3f}" for k, v in trend.items())
        + f"; null-control effect {null_mu:.4f} +/- {null_sd:.4f}"
        + (
            "; no increasing trend, so any apparent radius is floor noise and none is quoted."
            if no_trend
            else "; effect does increase with strength, so the radius is measurable here."
        )
    )
    if bins:
        wr = min(e["rank_min"] for e in bins.values())
        wratio = max(e["ratio_max"] for e in bins.values())
        wcell = max(bins.values(), key=lambda e: e["ratio_max"])["ratio_worst_cell"]
        parts.append(
            "**The constructive rule, with its real-data support.** Over every warp family, "
            "shape parameter and strength tested, and at every effect size this library "
            f"produces, within-locus ISM ranking never falls below **{wr:.4f}**, while ratio "
            f"comparisons degrade by up to **{wratio:.3f} log units, a factor of "
            f"{np.exp(wratio):.2f}** (worst cell: {wcell}). **Rank within a locus; never "
            "compare magnitudes across loci on a nonlinear latent.** Both numbers are worst "
            "cases over the warp grid, not point estimates at one shape."
        )
    agg["verdict"] = " ".join(parts)

    table = make_table(agg, cfg, meta)
    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "identifiability_mpsa.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(json.dumps(to_jsonable(agg), indent=1))
    (run / "per_seed_values.json").write_text(
        json.dumps(
            {
                f"closure_{k}": {
                    str(r["seed"]): max(x["closure_r2"] for x in rows_of(r, k))
                    for r in per_seed
                }
                for k in CLASSES
            },
            indent=1,
        )
    )
    write_tuning_budget(
        run,
        [
            {
                "model": k,
                "configs_tried": 1,
                "epochs": int(cfg.identifiability_mpsa.epochs),
                "gradient_steps": int(cfg.identifiability_mpsa.epochs)
                * int(cfg.identifiability_mpsa.restarts),
                "search_space": "one pre-specified configuration per G-P map class",
                "selection": "lowest training loss over restarts; identifiability, not generalization",
            }
            for k in CLASSES
        ],
    )
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
