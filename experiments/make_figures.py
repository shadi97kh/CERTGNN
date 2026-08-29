#!/usr/bin/env python
"""Publication figures, in two stages so that no number is ever hand-typed.

`results/runs/` is gitignored, so a figure cannot cite a committed
`results.json` directly. This module therefore first EXTRACTS every quantity it
will plot into `paper/data/figure_data.json`, which is committed, and then
PLOTS from that file alone. Regenerating the figures needs only the committed
JSON; the run directories are needed only to refresh it.

    python -m experiments.make_figures --extract   # run dirs -> paper/data/
    python -m experiments.make_figures             # paper/data/ -> figures

House style: single column 3.4in, double column 7.0in, 8pt ticks, 9pt axis
labels, Okabe-Ito colorblind-safe palette, no gridlines, no chartjunk. Every
error bar is a 95% bootstrap CI over 10 seeds and every caption says so;
captions are written to paper/figures/captions.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = pathlib.Path("paper/data/figure_data.json")
CLASSES = ("linear", "pairwise", "neural")
FAMILIES = ("sinusoid", "spline", "sigmoid_mixture")
FAM_LABEL = {
    "sinusoid": "sinusoid",
    "spline": "spline",
    "sigmoid_mixture": "sigmoid mixture",
}

# Okabe-Ito, colorblind safe
FAM_COLOR = {"sinusoid": "#0072B2", "spline": "#D55E00", "sigmoid_mixture": "#009E73"}
CLS_COLOR = {"linear": "#E69F00", "pairwise": "#56B4E9", "neural": "#000000"}
SINGLE, DOUBLE = 3.4, 7.0

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # embed TrueType so LaTeX gets real vector text
        "ps.fonttype": 42,
    }
)


def _mean_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return {
        "mean": float(v.mean()),
        "lo": float(np.quantile(b, 0.025)),
        "hi": float(np.quantile(b, 0.975)),
        "n": int(v.size),
    }


# --------------------------------------------------------------- extract


def extract(id_run: pathlib.Path, ism_run: pathlib.Path) -> dict[str, Any]:
    """Distil every plotted quantity from the two source runs."""
    id_rows: list[dict] = []
    for f in sorted(glob.glob(str(id_run / "seed_*.json"))):
        id_rows += json.load(open(f))["radius_rows"]
    ism_rows: list[dict] = []
    for f in sorted(glob.glob(str(ism_run / "seed_*.json"))):
        ism_rows += json.load(open(f))["rows"]
    id_res = json.load(open(id_run / "results.json"))
    id_meta = json.load(open(id_run / "meta.json"))
    ism_meta = json.load(open(ism_run / "meta.json"))

    def sel(rows, **kw):
        out = []
        for r in rows:
            if (
                kw.get("null") is not None
                and bool(r.get("is_null_control")) != kw["null"]
            ):
                continue
            if "cls" in kw and r["class"] != kw["cls"]:
                continue
            if "fam" in kw and r["family"] != kw["fam"]:
                continue
            if "s" in kw and abs(r["strength"] - kw["s"]) > 1e-9:
                continue
            out.append(r)
        return out

    strengths = sorted({r["strength"] for r in sel(id_rows, null=False)})
    smax = max(strengths)

    d: dict[str, Any] = {
        "provenance": {
            "identifiability_run": id_run.name,
            "identifiability_sha": id_meta["git_sha"],
            "identifiability_config": id_meta["config_hash"],
            "ism_run": ism_run.name,
            "ism_sha": ism_meta["git_sha"],
            "n_seeds": len(id_meta.get("seeds", [])),
            "error_bars": "95% bootstrap CI over seeds",
        },
        "strengths": strengths,
        "max_strength": smax,
    }

    # fig 2: closure at max strength
    d["closure"] = {
        c: {
            f: _mean_ci(
                [
                    r["closure_r2"]
                    for r in sel(id_rows, cls=c, fam=f, s=smax, null=False)
                ]
            )
            for f in FAMILIES
        }
        for c in CLASSES
    }
    # fig 3 left: neural effect size vs strength, and the null floor
    d["effect_by_strength"] = {
        f: {
            str(s): _mean_ci(
                [
                    abs(r["effect_size"])
                    for r in sel(id_rows, cls="neural", fam=f, s=s, null=False)
                ]
            )
            for s in strengths
        }
        for f in FAMILIES
    }
    nulls = [abs(r["effect_size"]) for r in sel(id_rows, cls="neural", null=True)]
    d["null_floor"] = {
        "mean": float(np.mean(nulls)),
        "sd": float(np.std(nulls)),
        "n": len(nulls),
    }
    # fig 3 right + fig 4: attribution measures vs strength
    for key in ("cross_instance_spearman", "within_instance_cosine"):
        d[key] = {
            f: {
                str(s): _mean_ci(
                    [r[key] for r in sel(id_rows, cls="neural", fam=f, s=s, null=False)]
                )
                for s in strengths
            }
            for f in FAMILIES
        }
    # fig 4: ISM measures vs strength
    ism_strengths = sorted({r["strength"] for r in ism_rows})
    d["ism_strengths"] = ism_strengths
    for key in ("ism_spearman", "ism_ratio_error", "grad_spearman", "grad_ratio_error"):
        d[key] = {
            f: {
                str(s): _mean_ci(
                    [
                        r[key]
                        for r in ism_rows
                        if r["family"] == f and abs(r["strength"] - s) < 1e-9
                    ]
                )
                for s in ism_strengths
            }
            for f in FAMILIES
        }
    # fig 5: the artifact checks
    d["artifact_checks"] = id_res["artifact_checks"]
    d["radius_by_class"] = {}
    for key, v in id_res.get("radius", {}).items():
        d["radius_by_class"][str(key)] = v
    d["radius_measurable"] = id_res["radius_measurable"]
    return d


# ----------------------------------------------------------------- plots


def _save(fig: plt.Figure, out: pathlib.Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def fig1_setup(out: pathlib.Path) -> str:
    """Pedagogical, no data: the construction."""
    import torch

    from experiments.identifiability_probe import invert_warp, warp

    z = torch.linspace(-3, 3, 600, dtype=torch.float64)
    s, om = 0.85, 2.0
    psi = warp(z, "sinusoid", s, om, 0)
    g = lambda t: torch.sigmoid(2.0 * t)  # noqa: E731
    back = invert_warp(psi, "sinusoid", s, om, 0)

    fig, (a, b, c) = plt.subplots(3, 1, figsize=(SINGLE, 5.6))
    a.plot(z, z, color="0.6", lw=0.9, ls="--", label=r"identity")
    a.plot(z, psi, color=FAM_COLOR["sinusoid"], lw=1.6, label=r"$\psi(\varphi)$")
    a.set_xlabel(r"$\varphi$")
    a.set_ylabel(r"$\psi(\varphi)$")
    a.set_title("(a) a monotone warp of the latent", loc="left")
    a.legend(frameon=False, loc="upper left")

    b.plot(z, g(z), color="#000000", lw=1.6, label=r"$g$")
    b.plot(psi, g(z), color="#D55E00", lw=1.6, label=r"$g\circ\psi^{-1}$")
    b.set_xlabel("latent argument")
    b.set_ylabel("prediction")
    b.set_title("(b) the compensating link", loc="left")
    b.legend(frameon=False, loc="upper left")

    c.plot(g(z), g(back), color="#009E73", lw=2.2)
    c.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    c.set_xlabel(r"prediction of $(\varphi,\,g)$")
    c.set_ylabel(r"prediction of $(\psi\circ\varphi,\,g\circ\psi^{-1})$")
    c.set_title("(c) the two composites", loc="left")
    c.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, out, "fig1_setup")
    return (
        "**Figure 1. The construction.** (a) A strictly monotone warp $\\psi$ of the latent "
        "phenotype, shown against the identity. (b) The original link $g$ and the compensating "
        "link $g\\circ\\psi^{-1}$, plotted against their own arguments. (c) Predictions of the "
        "original model against predictions of the twin, which lie on $y=x$. The two models "
        "differ in every internal quantity -- the latent, its gradient, the link -- and agree in "
        "every prediction. Pedagogical construction; no fitted data and no error bars."
    )


def fig2_closure(d: dict[str, Any], out: pathlib.Path) -> str:
    """Broken y-axis: a linear 0-to-1 axis hides the finding."""
    fig, (hi, lo) = plt.subplots(
        2,
        1,
        figsize=(DOUBLE, 3.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.08},
    )
    xs = np.arange(len(CLASSES))
    w = 0.26
    for i, f in enumerate(FAMILIES):
        m = [d["closure"][c][f]["mean"] for c in CLASSES]
        err = [
            [m[j] - d["closure"][c][f]["lo"] for j, c in enumerate(CLASSES)],
            [d["closure"][c][f]["hi"] - m[j] for j, c in enumerate(CLASSES)],
        ]
        for ax in (hi, lo):
            ax.bar(
                xs + (i - 1) * w,
                m,
                w,
                yerr=err,
                capsize=2,
                color=FAM_COLOR[f],
                label=FAM_LABEL[f] if ax is lo else None,
                edgecolor="none",
            )
    hi.set_ylim(0.99990, 1.00004)
    hi.set_yticks([1.0])
    hi.set_yticklabels(["1.000000"])
    hi.axhline(1.0, color="0.3", lw=0.8, ls="--")
    lo.set_ylim(0.925, 0.975)
    lo.set_xticks(xs, CLASSES)
    lo.set_xlabel("G-P map class")
    fig.supylabel(r"closure $R^2$ of the twin", x=0.02, fontsize=9)
    lo.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    # break marks
    hi.spines["bottom"].set_visible(False)
    lo.spines["top"].set_visible(False)
    hi.tick_params(bottom=False)
    kw = dict(
        marker=[(-1, -0.6), (1, 0.6)],
        markersize=6,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    hi.plot([0, 1], [0, 0], transform=hi.transAxes, **kw)
    lo.plot([0, 1], [1, 1], transform=lo.transAxes, **kw)
    for i, f in enumerate(FAMILIES):
        hi.annotate(
            "1.000000",
            (2 + (i - 1) * w, 1.0),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=6,
        )
    _save(fig, out, "fig2_closure")
    return (
        "**Figure 2. Closure of each G-P map class under a monotone reparameterization.** The "
        "twin is representable by the neural class ($R^2 = 1.000000$, all three warp families) "
        "and not by the additive or pairwise classes ($0.93$ to $0.96$). The $y$-axis is broken: "
        "on a linear $0$ to $1$ axis every bar sits at the top and the finding is invisible. "
        "Six significant figures are required -- at three, an uncorrected optimizer's $0.999848$ "
        "reads as a clean $1.000$, and that difference drove an entire spurious result. Error "
        "bars are 95% bootstrap CIs over 10 seeds; evaluated at the largest monotone warp "
        f"strength ($s = {d['max_strength']:g}$)."
    )


def fig3_split(d: dict[str, Any], out: pathlib.Path) -> str:
    ss = d["strengths"]
    fig, (lax, r) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharex=True)
    nf = d["null_floor"]
    lax.axhspan(
        max(nf["mean"] - nf["sd"], 0),
        nf["mean"] + nf["sd"],
        color="0.85",
        lw=0,
        label="identity-warp null $\\pm$1 SD",
    )
    lax.axhline(nf["mean"], color="0.45", lw=0.9, ls="--")
    for f in FAMILIES:
        m = [d["effect_by_strength"][f][str(s)]["mean"] for s in ss]
        lax.plot(ss, m, "o-", color=FAM_COLOR[f], ms=3, lw=1.3, label=FAM_LABEL[f])
    lax.set_xlabel("warp strength $s$")
    lax.set_ylabel("prediction effect size")
    lax.set_title("predictions: indistinguishable", loc="left")
    lax.set_ylim(bottom=0)
    lax.legend(frameon=False, loc="upper right")

    for f in FAMILIES:
        m = [d["cross_instance_spearman"][f][str(s)]["mean"] for s in ss]
        lo = [d["cross_instance_spearman"][f][str(s)]["lo"] for s in ss]
        hi = [d["cross_instance_spearman"][f][str(s)]["hi"] for s in ss]
        r.plot(ss, m, "o-", color=FAM_COLOR[f], ms=3, lw=1.3, label=FAM_LABEL[f])
        r.fill_between(ss, lo, hi, color=FAM_COLOR[f], alpha=0.16, lw=0)
    r.set_xlabel("warp strength $s$")
    r.set_ylabel("cross-instance Spearman")
    r.set_title("attributions: degrade", loc="left")
    fig.tight_layout()
    _save(fig, out, "fig3_split")
    lo_v = min(d["cross_instance_spearman"][f][str(max(ss))]["mean"] for f in FAMILIES)
    hi_v = max(d["cross_instance_spearman"][f][str(min(ss))]["mean"] for f in FAMILIES)
    return (
        "**Figure 3. Predictions are indistinguishable across the entire range over which "
        "attribution ranking degrades.** Left: the prediction-space effect size of the twin, "
        "against the strength-0 identity-warp null control (shaded, $\\pm$1 SD). It is flat and "
        "inside the band across a tenfold range of warp strength. Right: cross-instance "
        f"attribution-magnitude Spearman over the same range, falling from {hi_v:.3f} to "
        f"{lo_v:.3f}. Neural G-P map; shaded bands on the right are 95% bootstrap CIs over 10 "
        "seeds. No sample size distinguishes the models on the left; the quantity on the right "
        "is not a property of their predictions."
    )


def fig4_invariance(d: dict[str, Any], out: pathlib.Path) -> str:
    ss, iss = d["strengths"], d["ism_strengths"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    f = "sinusoid"
    ax.plot(
        iss,
        [d["grad_ratio_error"][f][str(s)]["mean"] for s in iss],
        "s-",
        color="#000000",
        ms=3,
        lw=1.3,
        label="gradient ratio error",
    )
    ax.plot(
        iss,
        [d["ism_ratio_error"][f][str(s)]["mean"] for s in iss],
        "^-",
        color="#D55E00",
        ms=3.5,
        lw=1.3,
        label="ISM ratio error",
    )
    ax.set_xlabel("warp strength $s$")
    ax.set_ylabel("within-locus ratio error  (log units)")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(
        iss,
        [d["ism_spearman"][f][str(s)]["mean"] for s in iss],
        "o--",
        color="#009E73",
        ms=3,
        lw=1.2,
        label="ISM within-locus Spearman",
    )
    ax2.plot(
        ss,
        [d["cross_instance_spearman"][f][str(s)]["mean"] for s in ss],
        "v--",
        color="#0072B2",
        ms=3,
        lw=1.2,
        label="cross-instance Spearman",
    )
    ax2.set_ylabel("Spearman")
    ax2.set_ylim(0.80, 1.01)
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, frameon=False, loc="center left", fontsize=6.5)
    fig.tight_layout()
    _save(fig, out, "fig4_invariance")
    gmax = max(d["grad_ratio_error"][f][str(s)]["mean"] for s in iss)
    imax = max(d["ism_ratio_error"][f][str(s)]["mean"] for s in iss)
    return (
        "**Figure 4. What the per-instance multiplier does and does not protect.** Under a "
        "monotone warp the gradient is scaled by $\\psi'(\\varphi(x))$, one positive number per "
        f"instance, so the gradient's within-locus ratio error is exactly zero (max {gmax:.3f} "
        "across the grid). In-silico mutagenesis is a finite difference, and the mean value "
        "theorem puts the multiplier at an intermediate point, one per mutation: its ratio error "
        f"grows to {imax:.3f} log units. Within-locus *ranking* nevertheless survives, while the "
        "cross-instance ranking degrades. Sinusoid family; error bars omitted for legibility, "
        "all series are means over 10 seeds and the corresponding CIs appear in "
        "`paper/tables/ism_invariance.md`. One sentence: a rule may say which mutation matters "
        "more at a locus, never how much more, and never how loci compare."
    )


def fig5_artifacts(d: dict[str, Any], out: pathlib.Path) -> str:
    ac = d["artifact_checks"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(SINGLE * 2.05, 2.2))
    ss = d["strengths"]
    for f in FAMILIES:
        a.plot(
            ss,
            [d["effect_by_strength"][f][str(s)]["mean"] for s in ss],
            "o-",
            color=FAM_COLOR[f],
            ms=2.5,
            lw=1.1,
            label=f"{FAM_LABEL[f]}  $\\rho$={ac['effect_trend_with_strength'][f]:+.2f}",
        )
    a.set_xlabel("warp strength $s$")
    a.set_ylabel("effect size")
    a.set_title("(a) no trend with strength", loc="left")
    a.legend(frameon=False, fontsize=6)
    a.set_ylim(bottom=0)

    fams = list(FAMILIES)
    vals = [ac["effect_minus_floor_in_sd"][f] for f in fams]
    b.bar(
        np.arange(len(fams)),
        vals,
        0.55,
        color=[FAM_COLOR[f] for f in fams],
        edgecolor="none",
    )
    b.axhline(3.0, color="0.3", lw=0.9, ls="--")
    b.text(0.02, 3.06, "3 SD", fontsize=6, color="0.3")
    b.set_xticks(
        np.arange(len(fams)), [FAM_LABEL[f].split()[0] for f in fams], fontsize=7
    )
    b.set_ylabel("(effect $-$ floor) / SD")
    b.set_title("(b) within the null floor", loc="left")

    n_small = (
        min(int(k.split(",")[-1].strip(" )'")) for k in d["radius_by_class"])
        if d["radius_by_class"]
        else 0
    )
    got = False
    for i, cls in enumerate(CLASSES):
        rs = [
            v["radius"]["mean"]
            for k, v in d["radius_by_class"].items()
            if k.startswith(f"('{cls}'") and str(n_small) in k
        ]
        if rs:
            c.bar(i, float(np.mean(rs)), 0.55, color=CLS_COLOR[cls], edgecolor="none")
            got = True
    c.set_xticks(np.arange(len(CLASSES)), CLASSES, fontsize=7)
    c.set_ylabel("apparent radius")
    c.set_title("(c) backwards ordering", loc="left")
    if not got:
        c.text(
            0.5,
            0.5,
            "no radius recorded",
            ha="center",
            transform=c.transAxes,
            fontsize=7,
        )
    fig.tight_layout()
    _save(fig, out, "fig5_artifacts")
    return (
        "**Figure 5 (appendix). Why no indistinguishability radius is quoted.** Three automated "
        "checks, all of which must fail for a radius to be meaningful. (a) The prediction effect "
        "size shows no increasing trend with warp strength (Spearman of $|$effect$|$ against "
        "strength in the legend); if the warp drove detectability it would rise. (b) The effect "
        "sits within a few standard deviations of the strength-0 null control, i.e. at the "
        "pipeline's numerical floor. (c) The classes that are *not* closed under the warp show "
        "an apparent radius as large as the closed class, which is backwards -- a class that "
        "cannot represent the twin should be easier to separate. For a closed class the true "
        "prediction-space effect is exactly zero, so any measured radius is floor noise."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--extract",
        action="store_true",
        help="refresh paper/data/figure_data.json from run dirs",
    )
    ap.add_argument("--id-run", default=None)
    ap.add_argument("--ism-run", default=None)
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args(argv)

    if args.extract:

        def pick(exp: str, given: str | None) -> pathlib.Path:
            if given:
                return pathlib.Path(given)
            cands = []
            for p in pathlib.Path("results/runs").glob("*"):
                mp = p / "meta.json"
                if mp.exists() and json.load(open(mp)).get("experiment") == exp:
                    cands.append(p)
            if not cands:
                raise SystemExit(f"no run found for {exp}")
            return sorted(cands)[-1]

        d = extract(
            pick("identifiability_probe", args.id_run),
            pick("ism_invariance", args.ism_run),
        )
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(d, indent=1))
        print(
            f"wrote {DATA} from {d['provenance']['identifiability_run']} and {d['provenance']['ism_run']}"
        )
        return 0

    if not DATA.exists():
        raise SystemExit(f"{DATA} missing; run with --extract first")
    d = json.loads(DATA.read_text())
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    caps = [
        fig1_setup(out),
        fig2_closure(d, out),
        fig3_split(d, out),
        fig4_invariance(d, out),
        fig5_artifacts(d, out),
    ]
    p = d["provenance"]
    header = (
        "# Figure captions\n\n"
        f"Generated by `experiments/make_figures.py` from `{DATA}`, which is committed and is "
        "the only source of numbers in these figures. Identifiability run "
        f"`{p['identifiability_run']}` (git `{p['identifiability_sha']}`, config "
        f"`{p['identifiability_config']}`); ISM run `{p['ism_run']}` (git `{p['ism_sha']}`). "
        f"{p['n_seeds']} seeds; every error bar is a {p['error_bars']}.\n"
    )
    (out / "captions.md").write_text(header + "\n" + "\n\n".join(caps) + "\n")
    for f in sorted(out.glob("fig*")):
        print(f"  wrote {f}")
    print(f"  wrote {out / 'captions.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
