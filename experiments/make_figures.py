#!/usr/bin/env python
"""Paper figures 1 and 3, built from committed run records.

Recomputes nothing. Every plotted number is read from a `results.json` under
`results/runs/`, and every figure is written next to a `<stem>_data.json`
recording the run directory, git SHA, config hash and the exact values drawn,
so a figure cannot silently disagree with the table it accompanies.

    python experiments/make_figures.py
    python experiments/make_figures.py --occurrence-run RUNDIR --separation-run RUNDIR

Figure 1  paper/figures/fig1_not_weak_models.pdf
    x  per-cell held-out predictive R^2                 (occurrence: heldout_r2_mean)
    y  per-cell top-3 attributed-set divergence         (occurrence: 1 - pooled_tk_ex3)

Figure 3  paper/figures/fig3_separation.pdf
    y  held-out measurements needed to separate a fit
       from its reparameterized twin, log scale         (separation: n_separate)
    twin axis  top-3 divergence                         (separation: 1 - tk_exact_top3_frac)
    reference  the full BRCA2 5'ss library size, read from the config, not typed here
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"

# Okabe-Ito, colourblind-safe. Depth is carried by BOTH hue and marker shape so
# the encoding survives greyscale printing and every common colour deficiency.
DEPTH_COLOR = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
DEPTH_MARKER = {1: "o", 2: "s", 3: "^"}
SECOND_SERIES_COLOR = "#CC79A7"

# Keys each figure needs. A run that lacks one of these is rejected loudly
# rather than silently plotted with a substituted quantity.
OCC_CELL_KEYS = ("hidden", "depth", "heldout_r2_mean", "pooled_tk_ex3")
SEP_CELL_KEYS = ("hidden", "depth", "n_separate", "tk_exact_top3_frac")

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
        "pdf.fonttype": 42,  # embed TrueType, not Type 3; some venues reject Type 3
        "savefig.bbox": "tight",
    }
)


# --------------------------------------------------------------------------- io


def _missing(cell: dict, keys: tuple[str, ...]) -> list[str]:
    return [k for k in keys if k not in cell]


def _describe(run: pathlib.Path) -> dict:
    meta = json.load(open(run / "meta.json"))
    return {
        "run": run.name,
        "experiment": meta.get("experiment"),
        "git_sha": meta.get("git_sha"),
        "git_dirty": meta.get("git_dirty"),
        "config_hash": meta.get("config_hash"),
        "seeds": meta.get("seeds"),
        "timestamp_utc": meta.get("timestamp_utc"),
    }


def _usable(run: pathlib.Path, experiment: str, keys: tuple[str, ...]) -> str | None:
    """None if the run can back the figure, else why it cannot."""
    if not (run / "meta.json").exists() or not (run / "results.json").exists():
        return "missing meta.json or results.json"
    try:
        meta = json.load(open(run / "meta.json"))
        res = json.load(open(run / "results.json"))
    except Exception as exc:  # noqa: BLE001
        return f"unreadable: {exc}"
    if meta.get("experiment") != experiment:
        return f"meta.json names experiment {meta.get('experiment')!r}"
    cells = res.get("cells") or {}
    if not cells:
        return "no cells"
    for name, cell in cells.items():
        miss = _missing(cell, keys)
        if miss:
            return f"cell {name} lacks {', '.join(miss)}"
    return None


def resolve_run(
    experiment: str, keys: tuple[str, ...], explicit: str | None
) -> pathlib.Path:
    """Pick a run directory that can actually back the figure.

    An explicit path is honoured but still validated: silently falling back
    after the caller named a directory would hide exactly the mismatch this
    check exists to catch. Otherwise the newest clean run carrying every
    required key wins, and the rejected candidates are printed so the choice is
    auditable rather than mysterious.
    """
    if explicit:
        run = pathlib.Path(explicit)
        if not run.is_absolute():
            run = ROOT / run
        why = _usable(run, experiment, keys)
        if why:
            raise SystemExit(
                f"--{experiment}-run {run} cannot back this figure: {why}\n"
                f"Re-run without the flag to auto-select, or pick another run."
            )
        return run

    candidates, rejected = [], []
    for run in sorted(RUNS.glob("*/")):
        why = _usable(run, experiment, keys)
        if why is None:
            meta = json.load(open(run / "meta.json"))
            candidates.append(
                (meta.get("timestamp_utc", ""), bool(meta.get("git_dirty")), run)
            )
        elif not why.startswith("meta.json names experiment"):
            rejected.append((run.name, why))

    if not candidates:
        print(f"\nno usable {experiment} run. Rejected:", file=sys.stderr)
        for name, why in rejected:
            print(f"  {name}: {why}", file=sys.stderr)
        raise SystemExit(f"no committed {experiment} run carries {', '.join(keys)}")

    # Prefer a clean tree, then the most recent.
    clean = [c for c in candidates if not c[1]]
    pool = clean or candidates
    pick = max(pool, key=lambda c: c[0])[2]
    if rejected:
        print(
            f"  [{experiment}] rejected {len(rejected)} run(s) lacking required keys:"
        )
        for name, why in rejected[-4:]:
            print(f"      {name}: {why}")
    if not clean:
        print(
            f"  [{experiment}] WARNING: no clean-tree run available; using a dirty one"
        )
    return pick


def load_cells(run: pathlib.Path) -> dict:
    return json.load(open(run / "results.json"))["cells"]


def read_library_size() -> tuple[float, str]:
    """The 'full library' reference, read from the config rather than typed."""
    import yaml

    path = ROOT / "configs" / "base.yaml"
    cfg = yaml.safe_load(open(path))
    for section in ("separation", "occurrence"):
        block = cfg.get(section) or {}
        if "library_size" in block:
            return float(block["library_size"]), f"{path.name}:{section}.library_size"
    raise SystemExit(f"library_size not found in {path}")


def _mean_lo_hi(entry: dict) -> tuple[float, float, float]:
    return float(entry["mean"]), float(entry["lo"]), float(entry["hi"])


def _cell_order(cells: dict) -> list[str]:
    """Cells as width x depth, ascending, so the depth trend reads left to right."""
    return sorted(cells, key=lambda k: (cells[k]["hidden"], cells[k]["depth"]))


def _write_sidecar(out: pathlib.Path, payload: dict) -> pathlib.Path:
    side = out.with_name(out.stem + "_data.json")
    json.dump(payload, open(side, "w"), indent=2)
    return side


# ---------------------------------------------------------------------- fig 1


def figure1(run: pathlib.Path, out: pathlib.Path) -> dict:
    cells = load_cells(run)
    names = _cell_order(cells)

    rows = []
    for name in names:
        c = cells[name]
        r2, r2lo, r2hi = _mean_lo_hi(c["heldout_r2_mean"])
        div = 1.0 - float(c["pooled_tk_ex3"])
        rows.append(
            {
                "cell": name,
                "hidden": int(c["hidden"]),
                "depth": int(c["depth"]),
                "heldout_r2_mean": r2,
                "heldout_r2_lo": r2lo,
                "heldout_r2_hi": r2hi,
                "pooled_tk_ex3": float(c["pooled_tk_ex3"]),
                "top3_divergence": div,
            }
        )

    best = max(rows, key=lambda r: r["heldout_r2_mean"])

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for d in sorted({r["depth"] for r in rows}):
        sub = [r for r in rows if r["depth"] == d]
        ax.scatter(
            [r["heldout_r2_mean"] for r in sub],
            [r["top3_divergence"] for r in sub],
            s=46,
            c=DEPTH_COLOR[d],
            marker=DEPTH_MARKER[d],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=f"depth {d}",
        )

    # y = 0 is "the two models never disagree", the only reference that matters.
    ax.axhline(0.0, color="0.35", lw=0.8, ls="-", zorder=1)

    ax.annotate(
        f"{best['cell']}\n(best fit)",
        xy=(best["heldout_r2_mean"], best["top3_divergence"]),
        xytext=(-8, -20),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=7.5,
        color="0.15",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="0.45", shrinkA=0, shrinkB=3),
    )

    ax.set_xlabel("held-out predictive $R^2$")
    ax.set_ylabel("top-3 attributed set differs\n(fraction of held-out sequences)")
    # y starts at 0 so the reader sees absolute magnitude rather than an
    # autoscaled band, with a hair of room below so the y=0 rule is not hidden
    # under the spine. x is given the data range plus padding: forcing x to 0
    # would spend half the panel on empty space and flatten the depth trend,
    # which is the one thing this figure exists to show.
    xs = [r["heldout_r2_mean"] for r in rows]
    pad = (max(xs) - min(xs)) * 0.12
    ax.set_ylim(-0.03, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.3, borderpad=0.2)

    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), format="png")
    plt.close(fig)

    payload = {
        "figure": out.name,
        "source": _describe(run),
        "x": "cells[*].heldout_r2_mean.mean",
        "y": "1 - cells[*].pooled_tk_ex3",
        "y_note": (
            "pooled_tk_ex3 is the MEDIAN over accuracy-matched model pairs of that "
            "pair's fraction of held-out sequences whose top-3 attributed sets match "
            "exactly. So y is a median over pairs of a per-pair fraction, not a single "
            "pooled fraction over sequences."
        ),
        "annotated_cell": best["cell"],
        "rows": rows,
    }
    payload["sidecar"] = _write_sidecar(out, payload).name
    return payload


# ---------------------------------------------------------------------- fig 3


def figure3(run: pathlib.Path, out: pathlib.Path) -> dict:
    cells = load_cells(run)
    library, library_src = read_library_size()

    rows = []
    for name, c in cells.items():
        mean, lo, hi = _mean_lo_hi(c["n_separate"])
        tk = c["tk_exact_top3_frac"]
        rows.append(
            {
                "cell": name,
                "hidden": int(c["hidden"]),
                "depth": int(c["depth"]),
                "n_separate_mean": mean,
                "n_separate_lo": lo,
                "n_separate_hi": hi,
                "tk_exact_top3_frac": float(tk["mean"]),
                "top3_divergence": 1.0 - float(tk["mean"]),
                "exceeds_full_library": bool(mean > library),
            }
        )
    rows.sort(key=lambda r: r["n_separate_mean"])

    x = np.arange(len(rows), dtype=float)
    mean = np.array([r["n_separate_mean"] for r in rows])
    lo = np.array([r["n_separate_lo"] for r in rows])
    hi = np.array([r["n_separate_hi"] for r in rows])
    div = np.array([r["top3_divergence"] for r in rows])

    # Asymmetric error bars, clipped so a lower bound at or below zero cannot
    # produce a negative coordinate on a log axis.
    floor = min(mean.min(), lo[lo > 0].min() if (lo > 0).any() else mean.min()) * 0.4
    lo_clipped = np.maximum(lo, floor)
    yerr = np.vstack([mean - lo_clipped, hi - mean])

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.set_yscale("log")

    ax.errorbar(
        x,
        mean,
        yerr=yerr,
        fmt="o",
        ms=5,
        mfc="#0072B2",
        mec="white",
        mew=0.6,
        ecolor="#0072B2",
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
        label="$n$ to separate",
    )

    top = max(hi.max(), library) * 2.5
    ax.axhspan(library, top, color="0.88", alpha=0.45, lw=0, zorder=0)
    ax.axhline(library, color="0.25", lw=0.9, ls="--", zorder=2)
    ax.text(
        -0.35,
        library * 1.6,
        f"full library ({library:,.0f})",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.15",
    )

    ax.set_ylim(floor, top)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["cell"] for r in rows], rotation=45, ha="right")
    ax.set_ylabel("held-out measurements to separate\na fit from its twin")
    ax.set_xlabel("cell (width $\\times$ depth), ordered by $n$ to separate")
    # Horizontal rules only: the y axis spans nine decades, so the decade grid is
    # load-bearing for reading a value. No vertical grid; x is categorical.
    ax.grid(axis="y", which="major", color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    x_off = x + 0.22
    ax2.plot(
        x_off,
        div,
        linestyle="none",
        marker="D",
        ms=4.2,
        mfc=SECOND_SERIES_COLOR,
        mec="0.25",
        mew=0.5,
        zorder=4,
    )
    ax2.set_ylabel("top-3 attributed set differs", color="0.25")
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(axis="y", labelsize=8, colors="0.25")

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            mfc="#0072B2",
            mec="white",
            ms=5,
            label="$n$ to separate (left, log)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="none",
            mfc=SECOND_SERIES_COLOR,
            mec="0.25",
            ms=4.2,
            label="top-3 divergence (right)",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        handletextpad=0.3,
        borderpad=0.2,
    )

    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), format="png")
    plt.close(fig)

    payload = {
        "figure": out.name,
        "source": _describe(run),
        "y_left": "cells[*].n_separate.{mean,lo,hi}",
        "y_right": "1 - cells[*].tk_exact_top3_frac.mean",
        "reference_line": {"value": library, "read_from": library_src},
        "n_cells_above_reference": sum(r["exceeds_full_library"] for r in rows),
        "rows": rows,
    }
    payload["sidecar"] = _write_sidecar(out, payload).name
    return payload


# ----------------------------------------------------------------------- main


def _print_source(tag: str, src: dict) -> None:
    dirty = " (DIRTY TREE)" if src.get("git_dirty") else ""
    print(f"  {tag} run   : {src['run']}{dirty}")
    print(f"  git sha    : {src['git_sha']}    config hash: {src['config_hash']}")
    print(f"  seeds      : {src['seeds']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--occurrence-run", default=None)
    ap.add_argument("--separation-run", default=None)
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("resolving runs")
    occ_run = resolve_run("occurrence", OCC_CELL_KEYS, args.occurrence_run)
    sep_run = resolve_run("separation", SEP_CELL_KEYS, args.separation_run)

    f1 = figure1(occ_run, out_dir / "fig1_not_weak_models.pdf")
    f3 = figure3(sep_run, out_dir / "fig3_separation.pdf")

    print("\n" + "=" * 78)
    print("FIGURE 1  paper/figures/fig1_not_weak_models.pdf")
    print("=" * 78)
    _print_source("occurrence", f1["source"])
    print(f"  x = {f1['x']}")
    print(f"  y = {f1['y']}")
    print(f"  annotated  : {f1['annotated_cell']}")
    print()
    print(
        f"  {'cell':>7}  {'depth':>5}  {'x heldout R2':>13}  {'pooled_tk_ex3':>14}  {'y divergence':>13}"
    )
    for r in f1["rows"]:
        print(
            f"  {r['cell']:>7}  {r['depth']:>5}  {r['heldout_r2_mean']:>13.6f}"
            f"  {r['pooled_tk_ex3']:>14.6f}  {r['top3_divergence']:>13.6f}"
        )

    print("\n" + "=" * 78)
    print("FIGURE 3  paper/figures/fig3_separation.pdf")
    print("=" * 78)
    _print_source("separation", f3["source"])
    print(f"  y left  = {f3['y_left']}  (log scale)")
    print(f"  y right = {f3['y_right']}")
    ref = f3["reference_line"]
    print(f"  reference line = {ref['value']:,.0f}  read from {ref['read_from']}")
    print(
        f"  cells above the full library: {f3['n_cells_above_reference']} of {len(f3['rows'])}"
    )
    print()
    print(
        f"  {'cell':>7}  {'depth':>5}  {'n_separate mean':>17}  {'lo':>15}  {'hi':>16}"
        f"  {'top3 diverg.':>12}  above"
    )
    for r in f3["rows"]:
        print(
            f"  {r['cell']:>7}  {r['depth']:>5}  {r['n_separate_mean']:>17,.3f}"
            f"  {r['n_separate_lo']:>15,.3f}  {r['n_separate_hi']:>16,.3f}"
            f"  {r['top3_divergence']:>12.4f}  {'yes' if r['exceeds_full_library'] else '-'}"
        )

    print(f"\nwrote {out_dir/'fig1_not_weak_models.pdf'}")
    print(f"wrote {out_dir/'fig3_separation.pdf'}")
    print(f"wrote sidecars {f1['sidecar']}, {f3['sidecar']}")


if __name__ == "__main__":
    main()
