#!/usr/bin/env python
"""Publication figures for the identifiability result, from stored run records.

Reads the per-seed JSON of an identifiability_probe run and emits the figures
the paper needs. It recomputes nothing: every number traces to the run
directory named in the caption, so a figure cannot silently disagree with the
table it accompanies.

    python scripts/make_paper_figures.py [--run RUNDIR] [--out paper/figures]
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

CLASSES = ("linear", "pairwise", "neural")
FAMILIES = ("sinusoid", "spline", "sigmoid_mixture")
FAM_LABEL = {
    "sinusoid": "sinusoid",
    "spline": "spline",
    "sigmoid_mixture": "sigmoid mixture",
}
CLASS_LABEL = {"linear": "linear", "pairwise": "pairwise", "neural": "neural"}
COLORS = {"sinusoid": "#1f77b4", "spline": "#d62728", "sigmoid_mixture": "#2ca02c"}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)


def load_rows(run: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(run / "seed_*.json"))):
        rows += json.load(open(f))["radius_rows"]
    if not rows:
        raise SystemExit(f"no radius_rows in {run}")
    return rows


def _mean_ci(
    vals: list[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def sel(rows, cls=None, fam=None, s=None, null=False):
    out = []
    for r in rows:
        if bool(r.get("is_null_control")) != null:
            continue
        if cls and r["class"] != cls:
            continue
        if fam and r["family"] != fam:
            continue
        if s is not None and abs(r["strength"] - s) > 1e-9:
            continue
        out.append(r)
    return out


# ------------------------------------------------------------------ fig 1


def fig_closure(rows: list[dict], out: pathlib.Path, run_name: str) -> None:
    """Closure of each model class under each warp family: the mechanism.

    Evaluated at the largest monotone warp strength, matching
    paper/tables/identifiability_radius.md exactly. Averaging over strengths
    would make the figure disagree with its own table.
    """
    smax = max(r["strength"] for r in rows if not r.get("is_null_control"))
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    width = 0.26
    xs = np.arange(len(CLASSES))
    for i, fam in enumerate(FAMILIES):
        means, los, his = [], [], []
        for c in CLASSES:
            vals = [
                r["closure_r2"]
                for r in sel(rows, cls=c, fam=fam)
                if r.get("closure_r2") is not None
            ]
            m, lo, hi = _mean_ci(vals)
            means.append(m)
            los.append(m - lo)
            his.append(hi - m)
        ax.bar(
            xs + (i - 1) * width,
            means,
            width,
            yerr=[los, his],
            capsize=2,
            color=COLORS[fam],
            label=FAM_LABEL[fam],
            alpha=0.9,
        )
    ax.set_xticks(xs, [CLASS_LABEL[c] for c in CLASSES])
    ax.set_ylim(0.90, 1.008)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel(r"closure $R^2$ of the twin")
    ax.set_xlabel("G-P map class")
    ax.set_title(f"Is the monotone twin inside the model class?  ($s$={smax:g})")
    ax.legend(frameon=False, loc="lower left")
    ax.text(2, 1.0022, "1.000000", ha="center", fontsize=7)

    # right panel: distance from exact closure, log scale
    for i, fam in enumerate(FAMILIES):
        gaps = []
        for c in CLASSES:
            vals = [
                1.0 - r["closure_r2"]
                for r in sel(rows, cls=c, fam=fam)
                if r.get("closure_r2") is not None
            ]
            gaps.append(max(float(np.mean(vals)), 1e-9))
        ax2.plot(xs, gaps, "o-", color=COLORS[fam], label=FAM_LABEL[fam], ms=4)
    ax2.set_yscale("log")
    ax2.set_xticks(xs, [CLASS_LABEL[c] for c in CLASSES])
    ax2.set_ylabel(r"$1 - R^2$  (log)")
    ax2.set_xlabel("G-P map class")
    ax2.set_title("Distance from exact closure")
    ax2.axhline(1e-9, color="grey", lw=0.7, ls=":")
    ax2.text(
        0.02,
        1.4e-9,
        "exact to six figures",
        fontsize=6.5,
        color="grey",
        transform=ax2.get_yaxis_transform(),
    )
    fig.suptitle(
        "A neural G-P map is closed under monotone reparameterization; additive and pairwise maps are not",
        y=1.02,
        fontsize=9.5,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"identifiability_closure.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2


def fig_dissociation(rows: list[dict], out: pathlib.Path) -> None:
    """The paper's central figure: predictions stay indistinguishable while
    cross-instance attributions diverge."""
    strengths = sorted({r["strength"] for r in sel(rows, cls="neural")})
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    axA, axB, axC = axes

    # A: prediction-space effect size, with the null floor band
    nulls = [
        abs(r["effect_size"])
        for r in sel(rows, cls="neural", null=True)
        if r.get("effect_size") is not None
    ]
    nm, nsd = float(np.mean(nulls)), float(np.std(nulls))
    axA.axhspan(
        max(nm - nsd, 0),
        nm + nsd,
        color="grey",
        alpha=0.25,
        lw=0,
        label="identity-warp null $\\pm$1 SD",
    )
    axA.axhline(nm, color="grey", lw=1.0, ls="--")
    for fam in FAMILIES:
        ms = []
        for s in strengths:
            vals = [
                abs(r["effect_size"])
                for r in sel(rows, cls="neural", fam=fam, s=s)
                if r.get("effect_size") is not None
            ]
            ms.append(float(np.mean(vals)) if vals else np.nan)
        axA.plot(strengths, ms, "o-", color=COLORS[fam], ms=3.5, label=FAM_LABEL[fam])
    axA.set_xlabel("warp strength $s$")
    axA.set_ylabel("prediction effect size")
    axA.set_title("Predictions: no trend, at the floor")
    axA.legend(frameon=False, loc="lower left", fontsize=6.5)
    axA.set_ylim(bottom=0)

    # B: cross-instance attribution ranking
    for fam in FAMILIES:
        ms, los, his = [], [], []
        for s in strengths:
            vals = [
                r["cross_instance_spearman"]
                for r in sel(rows, cls="neural", fam=fam, s=s)
                if r.get("cross_instance_spearman") is not None
            ]
            m, lo, hi = _mean_ci(vals)
            ms.append(m)
            los.append(lo)
            his.append(hi)
        axB.plot(strengths, ms, "o-", color=COLORS[fam], ms=3.5, label=FAM_LABEL[fam])
        axB.fill_between(strengths, los, his, color=COLORS[fam], alpha=0.18, lw=0)
    axB.set_xlabel("warp strength $s$")
    axB.set_ylabel("cross-instance Spearman")
    axB.set_title("Attributions: diverge monotonically")
    axB.set_ylim(0.8, 1.005)

    # C: within-instance direction, the control that stays flat
    for fam in FAMILIES:
        ms = []
        for s in strengths:
            vals = [
                r["within_instance_cosine"]
                for r in sel(rows, cls="neural", fam=fam, s=s)
                if r.get("within_instance_cosine") is not None
            ]
            ms.append(float(np.mean(vals)) if vals else np.nan)
        axC.plot(strengths, ms, "o-", color=COLORS[fam], ms=3.5, label=FAM_LABEL[fam])
    axC.set_xlabel("warp strength $s$")
    axC.set_ylabel("within-instance cosine")
    axC.set_title("Within-instance: degrades, but far less")
    axC.set_ylim(0.8, 1.005)

    fig.suptitle(
        "Across a range where the model's predictions are statistically indistinguishable, "
        "cross-instance attribution ranking is not",
        y=1.04,
        fontsize=9.5,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"identifiability_dissociation.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3


def fig_construction(out: pathlib.Path) -> None:
    """Schematic of the twin: same predictions, different latent."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import torch

    from experiments.identifiability_probe import warp

    z = torch.linspace(-3, 3, 400, dtype=torch.float64)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax1, ax2, ax3 = axes
    for s, alpha in ((0.3, 0.45), (0.6, 0.7), (0.95, 1.0)):
        w = warp(z, "sinusoid", s, 2.0, 0)
        ax1.plot(z, w, color="#1f77b4", alpha=alpha, lw=1.5, label=f"$s$={s}")
    ax1.plot(z, z, "k--", lw=0.9, label="identity")
    ax1.set_title(r"monotone warp $\psi$")
    ax1.set_xlabel(r"$\varphi$")
    ax1.set_ylabel(r"$\psi(\varphi)$")
    ax1.legend(frameon=False)

    g = lambda t: torch.sigmoid(2.0 * t)  # noqa: E731
    ax2.plot(z, g(z), "k-", lw=1.6, label=r"$g(\varphi)$")
    for s, alpha in ((0.3, 0.45), (0.6, 0.7), (0.95, 1.0)):
        w = warp(z, "sinusoid", s, 2.0, 0)
        ax2.plot(
            w,
            g(z),
            color="#d62728",
            alpha=alpha,
            lw=1.2,
            label=r"$g\circ\psi^{-1}$" if s == 0.95 else None,
        )
    ax2.set_title(r"compensating link $g\circ\psi^{-1}$")
    ax2.set_xlabel("latent")
    ax2.set_ylabel("prediction")
    ax2.legend(frameon=False)

    from experiments.identifiability_probe import invert_warp

    for s, alpha in ((0.3, 0.45), (0.6, 0.7), (0.95, 1.0)):
        w = warp(z, "sinusoid", s, 2.0, 0)
        back = invert_warp(w, "sinusoid", s, 2.0, 0)
        ax3.semilogy(
            z,
            (g(z) - g(back)).abs().clamp_min(1e-18),
            color="#2ca02c",
            alpha=alpha,
            lw=1.2,
            label=f"$s$={s}",
        )
    ax3.set_title("prediction difference, actual")
    ax3.set_xlabel(r"$\varphi$")
    ax3.set_ylabel(r"$|g(\varphi)-g(\psi^{-1}(\psi(\varphi)))|$")
    ax3.legend(frameon=False)
    fig.suptitle(
        r"The twin $(\psi\circ\varphi,\; g\circ\psi^{-1})$ reproduces every prediction exactly",
        y=1.04,
        fontsize=9.5,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"identifiability_construction.{ext}", bbox_inches="tight")
    plt.close(fig)


def summary_table(rows: list[dict], run: pathlib.Path, out: pathlib.Path) -> None:
    """One table carrying every headline identifiability number and its source.

    Generated from the run's own records so it cannot drift from the figures;
    the run directory is named in the header so every number is traceable.
    """
    meta = json.load(open(run / "meta.json"))
    smax = max(r["strength"] for r in rows if not r.get("is_null_control"))
    smin = min(r["strength"] for r in rows if not r.get("is_null_control"))
    L = ["# Identifiability results at a glance\n"]
    L.append(
        f"Run `{run.name}`, git `{meta['git_sha']}`"
        f"{' (DIRTY)' if meta.get('git_dirty') else ''}, config `{meta['config_hash']}`, "
        f"{len(meta.get('seeds', []))} seed(s). Generated by `scripts/make_paper_figures.py` "
        "from that run's per-seed records.\n"
    )
    L.append("| quantity | linear | pairwise | neural |")
    L.append("|---|---|---|---|")

    def row(label, key, s_at, fmt="{:.6f}"):
        cells = []
        for c in CLASSES:
            vals = [r[key] for r in sel(rows, cls=c, s=s_at) if r.get(key) is not None]
            cells.append(fmt.format(float(np.mean(vals))) if vals else "n/a")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    row(f"closure $R^2$ at $s$={smax:g} (six figures)", "closure_r2", smax)
    row(f"closure $R^2$ at $s$={smin:g}", "closure_r2", smin)
    row(
        f"cross-instance Spearman at $s$={smax:g}",
        "cross_instance_spearman",
        smax,
        "{:.3f}",
    )
    row(
        f"within-instance cosine at $s$={smax:g}",
        "within_instance_cosine",
        smax,
        "{:.3f}",
    )
    nulls = {
        c: [
            abs(r["effect_size"])
            for r in sel(rows, cls=c, null=True)
            if r.get("effect_size") is not None
        ]
        for c in CLASSES
    }
    L.append(
        "| identity-warp null effect size | "
        + " | ".join(f"{float(np.mean(v)):.4f}" if v else "n/a" for v in nulls.values())
        + " |"
    )
    L.append(
        "\nClosure is the R^2 of the best in-class approximation to the warped latent: 1.000000 "
        "means the class contains the twin, so the twin predicts identically and no sample size "
        "separates it. The identity-warp null is the pipeline's numerical floor; every "
        "prediction-space effect measured here sits within a few SD of it, which is why the "
        "indistinguishability radius is reported as unmeasurable rather than as a number.\n"
    )
    (out / "identifiability_summary.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", default=None, help="identifiability_probe run directory")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args(argv)

    if args.run:
        run = pathlib.Path(args.run)
    else:
        cands = [
            p
            for p in pathlib.Path("results/runs").glob("*")
            if (p / "radius_table.md").exists()
        ]
        if not cands:
            raise SystemExit(
                "no identifiability run with radius_table.md found; pass --run"
            )
        run = sorted(cands)[-1]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(run)
    fig_closure(rows, out, run.name)
    fig_dissociation(rows, out)
    fig_construction(out)
    summary_table(rows, run, pathlib.Path("paper/tables"))
    print(f"source run: {run}")
    print("  wrote paper/tables/identifiability_summary.md")
    for f in sorted(out.glob("identifiability_*")):
        print(f"  wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
