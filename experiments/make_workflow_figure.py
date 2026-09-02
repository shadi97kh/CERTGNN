#!/usr/bin/env python
"""The page-2 workflow figure, with every count read from a committed run.

An abstract box-and-arrow diagram would say nothing a sentence could not. This
one carries the actual numbers -- how many sequences, how many models, how many
pairs survive the accuracy filter, what fraction end up disagreeing -- so a
reader can follow the whole pipeline and see where the population shrinks and
where the disagreement appears, without turning to a table.

    python experiments/make_workflow_figure.py
    python experiments/make_workflow_figure.py --cell 128x1 --out paper/figures

Writes `fig0_workflow.pdf` (and `.png`) plus a `_data.json` sidecar recording
the run, git SHA and every number drawn.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"

# Okabe-Ito. Stage colour encodes what KIND of step it is, so the eye can group
# data handling, model fitting, and measurement without reading every label.
C_DATA = "#0072B2"
C_FIT = "#009E73"
C_MEASURE = "#D55E00"
C_FIND = "#CC79A7"
C_CTRL = "#7a7a7a"

plt.rcParams.update(
    {
        "font.size": 7,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "figure.dpi": 200,
    }
)


def _usable(run: pathlib.Path, experiment: str) -> bool:
    if not (run / "meta.json").exists() or not (run / "results.json").exists():
        return False
    try:
        meta = json.load(open(run / "meta.json"))
        res = json.load(open(run / "results.json"))
    except Exception:  # noqa: BLE001
        return False
    return (
        meta.get("experiment") == experiment
        and not res.get("dump_only")
        and len(res.get("cells") or {}) >= 12
    )


def resolve(experiment: str) -> pathlib.Path:
    cands = []
    for run in sorted(RUNS.glob("*/")):
        if _usable(run, experiment):
            m = json.load(open(run / "meta.json"))
            cands.append((m.get("timestamp_utc", ""), bool(m.get("git_dirty")), run))
    if not cands:
        raise SystemExit(f"no committed {experiment} run with a full grid")
    clean = [c for c in cands if not c[1]]
    return max(clean or cands, key=lambda c: c[0])[2]


def box(ax, x, y, w, h, title, lines, color, lw=1.1):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=lw,
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.045,
        title,
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=color,
        zorder=4,
    )
    for i, ln in enumerate(lines):
        ax.text(
            x + w / 2,
            y + h - 0.125 - i * 0.078,
            ln,
            ha="center",
            va="top",
            fontsize=6.1,
            color="0.15",
            zorder=4,
        )


def arrow(ax, x0, y0, x1, y1, color="0.35", label=None, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            lw=0.9,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )
    if label:
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2 + 0.028,
            label,
            ha="center",
            va="bottom",
            fontsize=6.0,
            color="0.3",
            zorder=6,
        )


def build(occ_run: pathlib.Path, cell: str, out: pathlib.Path) -> dict:
    res = json.load(open(occ_run / "results.json"))
    meta = json.load(open(occ_run / "meta.json"))
    tb = json.load(open(occ_run / "tuning_budget.json"))
    c = res["cells"][cell]
    seed0 = json.load(open(occ_run / "seed_0.json"))
    cfgc = next(
        (e for e in seed0["cells"] if f"{e['hidden']}x{e['depth']}" == cell), None
    )

    n_cells = len(res["cells"])
    K = int(res["n_models"])
    n_pairs = int(round(c["n_pairs_total"]["mean"]))
    n_tied = int(round(c["n_pairs_indistinguishable"]["mean"]))
    seq_len = int(res.get("seq_len", 9))
    ladder = int(tb[0]["configs_tried"])
    epochs = int(tb[0]["epochs"])
    div = 1.0 - float(c["pooled_tk_ex3"])
    rho_med = float(c["pooled_pi_median"])
    mde = float(c["pooled_mde_r2"])
    conc = float(c["pooled_conc_top3"])

    v = {
        "library": 30483,
        "n": int(res["n"]),
        "n_fit": int(res["n_fit"]),
        "n_heldout": int(res["n_heldout"]),
        "n_cells": n_cells,
        "K": K,
        "models_per_seed": n_cells * K,
        "seeds": int(res["n_seeds"]),
        "ladder": ladder,
        "epochs": epochs,
        "n_pairs": n_pairs,
        "n_tied": n_tied,
        "alpha": float(res["filter_alpha"]),
        "mde_r2": mde,
        "seq_len": seq_len,
        "rho_median": rho_med,
        "top3_mass": conc,
        "top3_divergence": div,
        "cell": cell,
    }

    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Five stages across 5.5in leaves about 1in per box. At 6pt that is roughly
    # 16 characters per line, so every line below is kept under that; a longer
    # phrase silently overhangs the border rather than wrapping.
    w, h, y = 0.1855, 0.44, 0.34
    gap = 0.0165
    xs = [0.004 + i * (w + gap) for i in range(5)]

    stages = [
        (
            "1. Assay",
            [
                f"{v['library']:,} BRCA2",
                "5' splice sites",
                f"subsample {v['n']:,}",
                f"{v['seeds']} seeds",
            ],
            C_DATA,
        ),
        (
            "2. Split",
            [
                "seeded permute",
                f"{v['n_fit']:,} fit",
                f"{v['n_heldout']:,} held out",
                "indices saved",
            ],
            C_DATA,
        ),
        (
            "3. Fit",
            [
                f"{v['n_cells']} width$\\times$depth",
                f"1 of {v['ladder']} rates,",
                f"then $K$={v['K']} models",
                "differing in seed",
            ],
            C_FIT,
        ),
        (
            "4. Tie",
            [
                f"{v['n_pairs']} pairs / cell",
                f"$t$-test $\\alpha$={v['alpha']}",
                f"{v['n_tied']}/{v['n_pairs']} tied",
                f"$\\Delta R^2$ res. {v['mde_r2']:.3f}",
            ],
            C_FIT,
        ),
        (
            "5. Attribute",
            [
                "in-silico",
                "mutagenesis over",
                f"{v['seq_len']} positions, for",
                "every model",
            ],
            C_MEASURE,
        ),
    ]
    for i, (title, lines, col) in enumerate(stages):
        box(ax, xs[i], y, w, h, title, lines, col)
    for i in range(4):
        arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    # The finding, spanning the pipeline beneath it.
    fy, fh = 0.030, 0.225
    full_w = (xs[4] + w) - xs[0]
    box(
        ax,
        xs[0],
        fy,
        full_w,
        fh,
        f"6. Compare the two attributions, sequence by sequence   ({cell})",
        [],
        C_FIND,
    )
    ax.text(
        xs[0] + full_w / 2,
        fy + fh - 0.105,
        f"median full-rank $\\rho$ = {v['rho_median']:+.2f}      "
        f"median {v['top3_mass'] * 100:.0f}% of the mass in 3 of "
        f"{v['seq_len']} positions",
        ha="center",
        va="top",
        fontsize=6.3,
        color="0.15",
        zorder=4,
    )
    ax.text(
        xs[0] + full_w / 2,
        fy + 0.030,
        f"yet the top-3 set differs in {v['top3_divergence'] * 100:.0f}% "
        "of held-out sequences",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=C_FIND,
        zorder=4,
    )
    arrow(ax, xs[4] + w / 2, y, xs[4] + w / 2, fy + fh, color=C_FIND)

    # The two controls that nearly hid it.
    ax.text(
        xs[2] + w / 2,
        y + h + 0.075,
        "two controls reported success for the wrong reason (Sec. 4.4):\n"
        "a ceiling that began at its target, and a count model with "
        "zero-variance points",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=C_CTRL,
        style="italic",
    )
    arrow(
        ax,
        xs[2] + w / 2,
        y + h + 0.068,
        xs[2] + w / 2,
        y + h + 0.010,
        color=C_CTRL,
        ls=(0, (2, 1.6)),
    )

    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=220)
    plt.close(fig)

    payload = {
        "figure": out.name,
        "source": {
            "run": occ_run.name,
            "git_sha": meta.get("git_sha"),
            "git_dirty": meta.get("git_dirty"),
            "config_hash": meta.get("config_hash"),
        },
        "cell": cell,
        "lr_chosen_seed0": (cfgc or {}).get("lr"),
        "values": v,
    }
    side = out.with_name(out.stem + "_data.json")
    json.dump(payload, open(side, "w"), indent=2)
    payload["sidecar"] = side.name
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="128x1", help="cell whose counts are shown")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run = resolve("occurrence")
    p = build(run, args.cell, out_dir / "fig0_workflow.pdf")
    print(f"  run       : {p['source']['run']}  git {p['source']['git_sha']}")
    print(f"  cell shown: {p['cell']}")
    for k, val in p["values"].items():
        print(f"    {k:20} {val}")
    print(f"\nwrote {out_dir / 'fig0_workflow.pdf'} and .png, sidecar {p['sidecar']}")


if __name__ == "__main__":
    main()
