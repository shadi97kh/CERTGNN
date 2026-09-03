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

Figure 3  paper/figures/fig3_occurrence_separability.pdf   [MAIN TEXT]
    y  held-out measurements to separate two accuracy-matched
       fits, log scale                                  (occurrence: n_separate_median)
    twin axis  top-3 divergence                         (occurrence: 1 - pooled_tk_ex3)

Appendix  paper/figures/figA_twin_separability.pdf      [APPENDIX]
    the same quantities for a fit against its REPARAMETERIZED TWIN
                                                        (separation: n_separate,
                                                         1 - tk_exact_top3_frac)

Both main-text figures therefore describe the occurrence experiment. The twin
comparison is a separate result and is never mixed into a main-text figure: the
two populations differ in top-3 divergence by up to 50 points.
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

# The paper includes these as PNG rather than PDF. At the widths they are set
# to (0.86 to 0.98 of a 5.5in line) a 200 dpi raster lands near 220 dpi on the
# page, which is visibly soft in print. 400 keeps every figure above 400 dpi
# effective. The workflow figure stays vector and is unaffected.
PNG_DPI = 400

# Keys each figure needs. A run that lacks one of these is rejected loudly
# rather than silently plotted with a substituted quantity.
OCC_CELL_KEYS = (
    "hidden",
    "depth",
    "heldout_r2_mean",
    "pooled_tk_ex3",
    "n_separate_median",
)
SEP_CELL_KEYS = ("hidden", "depth", "n_separate", "tk_exact_top3_frac")

# The two separability figures measure the SAME quantity on DIFFERENT populations,
# and conflating them is the error this module exists to prevent. The occurrence
# figure compares independently trained models that the data cannot tell apart.
# The twin figure compares one fit against its own reparameterized twin. Their
# top-3 divergences differ by up to 50 points, so a caption that names the wrong
# population misstates the result rather than merely mislabelling it.
SEPARABILITY = {
    "occurrence": {
        "n_key": "n_separate_median",
        "div_key": "pooled_tk_ex3",
        "div_is_dict": False,
        "population": "independently trained, accuracy-matched pairs",
        "ylabel": "held-out measurements to separate\ntwo accuracy-matched fits",
        "divlabel": "top-3 attributed set differs",
    },
    "twin": {
        "n_key": "n_separate",
        "div_key": "tk_exact_top3_frac",
        "div_is_dict": True,
        "population": "a fit against its reparameterized twin",
        "ylabel": "held-out measurements to separate\na fit from its reparameterized twin",
        "divlabel": "top-3 attributed set differs",
    },
}


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
    # A dump run covers a subset of the grid and says so in its own record.
    # Refusing it here is what stops a figure or a table silently acquiring a
    # two-cell denominator that looks perfectly well-formed.
    if res.get("dump_only"):
        return "dump_only run (partial grid)"
    cells = res.get("cells") or {}
    if not cells:
        return "no cells"
    if len(cells) < 12:
        return f"partial grid ({len(cells)} cells, expected 12)"
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


def n_floor(v: float) -> float:
    """Report a sample size as a whole number of measurements, at least one.

    The bound of Proposition 3 divides by a per-point average, so it can return
    a value below 1. Such a value does not name a number of measurements -- it
    says the average standardized difference already clears the threshold at a
    single point -- and 0 cannot be placed on a log axis at all. Every reported
    requirement is therefore ceil'd with a floor of 1.
    """
    import math

    if not math.isfinite(v):
        return v
    return float(max(1, math.ceil(v)))


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
    # Say what is actually plotted. pooled_tk_ex3 is a MEDIAN OVER PAIRS of a
    # per-pair fraction, not a fraction pooled over sequences, and the earlier
    # label claimed the latter.
    ax.set_ylabel("top-3 set differs\n(median over tied pairs,\nfraction of sequences)")
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
    fig.savefig(out.with_suffix(".png"), format="png", dpi=PNG_DPI)
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


def figure_separability(run: pathlib.Path, out: pathlib.Path, kind: str) -> dict:
    """Measurements needed to separate two fits, for one population of pairs.

    `kind` selects the population: "occurrence" for independently trained,
    accuracy-matched pairs, "twin" for a fit against its reparameterized twin.
    Everything the caption depends on comes from SEPARABILITY[kind], so the
    axis label and the data can never come from different experiments.
    """
    sp = SEPARABILITY[kind]
    cells = load_cells(run)
    library, library_src = read_library_size()

    rows = []
    for name, c in cells.items():
        mean, lo, hi = (n_floor(v) for v in _mean_lo_hi(c[sp["n_key"]]))
        raw = c[sp["div_key"]]
        agree = float(raw["mean"]) if sp["div_is_dict"] else float(raw)
        rows.append(
            {
                "cell": name,
                "hidden": int(c["hidden"]),
                "depth": int(c["depth"]),
                "n_separate_mean": mean,
                "n_separate_lo": lo,
                "n_separate_hi": hi,
                "top3_agreement": agree,
                "top3_divergence": 1.0 - agree,
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

    # A log axis earns its place only when the data spans decades. The twin
    # population spans nine and needs it; the accuracy-matched population spans
    # about half of one, where a log axis shows a single tick and reads as an
    # error. Choose from the data rather than fixing it per figure.
    decades = np.log10(hi.max() / max(lo_clipped.min(), 1e-12))
    logscale = decades >= 1.5

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    if logscale:
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

    # The library reference is only drawn when it is within two decades of the
    # data. For the accuracy-matched population every cell separates with a
    # couple of measurements, four orders of magnitude below the library, so
    # plotting the line would compress all twelve cells into a sliver and the
    # figure would carry less information, not more. There it is stated instead.
    on_scale = hi.max() >= library / 100.0
    if on_scale:
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
    else:
        top = hi.max() * 1.18
        ax.text(
            len(rows) - 0.5,
            top * 0.985,
            f"every cell separates with fewer than {np.ceil(hi.max()):.0f}"
            f" measurements;\nthe full library holds {library:,.0f}",
            ha="right",
            va="top",
            fontsize=7.5,
            color="0.15",
        )

    ax.set_ylim(0.0 if not logscale else floor, top)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["cell"] for r in rows], rotation=45, ha="right")
    ax.set_ylabel(sp["ylabel"])
    ax.set_xlabel("cell (width $\\times$ depth), ordered by $n$ to separate")
    # The population belongs on the figure itself, not only in the caption:
    # these two figures are indistinguishable at a glance and are routinely
    # confused, which is the error this whole module guards against.
    ax.set_title(sp["population"], fontsize=8, loc="left", color="0.25", pad=6)
    # Horizontal rules only. On the twin figure the y axis spans nine decades and
    # the decade grid is load-bearing for reading a value; on the occurrence
    # figure it still helps because the axis stays logarithmic for comparability
    # between the two. No vertical grid; x is categorical.
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
    ax2.set_ylabel(sp["divlabel"], color="0.25")
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
            label="$n$ to separate (left, log)"
            if logscale
            else "$n$ to separate (left)",
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
        loc="upper left" if logscale else "lower right",
        handletextpad=0.3,
        borderpad=0.2,
        labelspacing=0.3,
    )

    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=PNG_DPI)
    plt.close(fig)

    payload = {
        "figure": out.name,
        "kind": kind,
        "population": sp["population"],
        "source": _describe(run),
        "y_scale": "log" if logscale else "linear",
        "y_left": f"cells[*].{sp['n_key']}.{{mean,lo,hi}}",
        "y_right": f"1 - cells[*].{sp['div_key']}"
        + (".mean" if sp["div_is_dict"] else ""),
        "reference_line": {
            "value": library,
            "read_from": library_src,
            "drawn": bool(on_scale),
        },
        "n_cells_above_reference": sum(r["exceeds_full_library"] for r in rows),
        "rows": rows,
    }
    payload["sidecar"] = _write_sidecar(out, payload).name
    return payload


# ---------------------------------------------------------------------- fig 2


def resolve_dump() -> pathlib.Path | None:
    """Newest occurrence run that carries a per-instance dump.

    Deliberately the mirror image of `resolve_run`: that one refuses dump runs
    because they hold a partial grid, this one requires them. Neither can pick
    up the other's runs by accident.
    """
    best = None
    for run in sorted(RUNS.glob("*/")):
        if not (run / "per_instance.npz").exists():
            continue
        try:
            meta = json.load(open(run / "meta.json"))
            res = json.load(open(run / "results.json"))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("experiment") != "occurrence" or not res.get("dump_only"):
            continue
        key = (not bool(meta.get("git_dirty")), meta.get("timestamp_utc", ""))
        if best is None or key > best[0]:
            best = (key, run)
    return None if best is None else best[1]


def _dump_arrays(run: pathlib.Path, cell: str) -> dict[str, np.ndarray]:
    """Pool a cell's per-sequence arrays over every dumped seed."""
    z = np.load(run / "per_instance.npz")
    seeds = sorted(
        {
            k.split("__")[0].split("_seed")[1]
            for k in z.files
            if k.startswith(f"{cell}_seed")
        },
        key=int,
    )
    if not seeds:
        raise SystemExit(f"no dumped seeds for {cell} in {run}")
    rho, jac, prof = [], [], []
    for sd in seeds:
        base = f"{cell}_seed{sd}__"
        rho.append(np.asarray(z[base + "rho"]).ravel())
        jac.append(np.asarray(z[base + "jaccard"]).ravel())
        prof.append(np.asarray(z[base + "rankprofile"]).reshape(-1, 9))
    return {
        "rho": np.concatenate(rho),
        "jaccard": np.concatenate(jac),
        "profile": np.concatenate(prof, axis=0),
        "seeds": np.asarray([int(x) for x in seeds]),
    }


def figure2(
    run: pathlib.Path, out: pathlib.Path, cell: str = "128x1", rho_star: float = 0.708
) -> dict:
    """Two panels: the rank profile, and the panel the argument rests on."""
    d = _dump_arrays(run, cell)
    prof = d["profile"]
    ok = np.isfinite(prof).all(axis=1)
    prof = prof[ok]
    ranks = np.arange(1, prof.shape[1] + 1)
    p25, p50, p75 = (np.percentile(prof, q, axis=0) for q in (25, 50, 75))
    top3 = float(np.median(prof[:, :3].sum(axis=1)))

    rho, jac = d["rho"], d["jaccard"]
    m = np.isfinite(rho) & np.isfinite(jac)
    rho, jac = rho[m], jac[m]
    left = rho < rho_star
    exact = jac >= 1.0
    n_arg = int((left & exact).sum())
    frac_arg = n_arg / rho.size if rho.size else float("nan")
    frac_left = n_arg / max(int(left.sum()), 1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.5, 2.6))

    # ---- (a) rank profile
    ax_a.fill_between(ranks, p25, p75, color="#0072B2", alpha=0.22, lw=0)
    ax_a.plot(ranks, p50, "-o", color="#0072B2", ms=3.4, lw=1.3, mec="white", mew=0.5)
    ax_a.axvspan(0.5, 3.5, color="0.88", alpha=0.5, lw=0, zorder=0)
    ax_a.annotate(
        f"top-3 mass\n{top3 * 100:.0f}%",
        xy=(2, p50[:3].max()),
        xytext=(4.4, max(p75) * 0.82),
        fontsize=7.5,
        color="0.15",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="0.45"),
    )
    ax_a.set_xticks(ranks)
    ax_a.set_xlim(0.5, prof.shape[1] + 0.5)
    ax_a.set_ylim(0.0, None)
    ax_a.set_xlabel("rank position")
    ax_a.set_ylabel("normalized attribution mass")
    ax_a.set_title("(a)", fontsize=9, loc="left")

    # ---- (b) the key panel
    # Top-3 Jaccard on 9 positions takes only {0, 0.2, 0.5, 1}, so an honest
    # scatter would draw four lines and hide all density. Jitter spreads each
    # level enough to read the mass without moving a point across a level.
    rng = np.random.default_rng(0)
    yj = jac + rng.uniform(-0.035, 0.035, size=jac.size)
    ax_b.scatter(rho, yj, s=4, c="#0072B2", alpha=0.10, lw=0, rasterized=True, zorder=2)
    hi = left & exact
    ax_b.scatter(
        rho[hi], yj[hi], s=5, c="#D55E00", alpha=0.30, lw=0, rasterized=True, zorder=3
    )
    ax_b.axvline(rho_star, color="0.2", ls="--", lw=1.0, zorder=4)
    # The rho* label sits above the cloud, not across it: rotated into the dense
    # region it was unreadable and hid the very points it refers to.
    ax_b.annotate(
        "perfect top-3 agreement\nexpected here (Prop. 2)",
        xy=(rho_star, 1.16),
        xytext=(rho_star - 0.10, 1.30),
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="0.15",
        annotation_clip=False,
        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45", shrinkA=0, shrinkB=2),
    )
    # Anchor the count to the highlighted block. Both fractions are given: the
    # share of all points is small, but the share AMONG points the rank
    # correlation calls disagreement is the number the argument turns on.
    if n_arg:
        anchor_x = float(np.median(rho[hi]))
        ax_b.annotate(
            f"{frac_arg * 100:.1f}% of all points,\n"
            f"{frac_left * 100:.0f}% of those left of the line",
            xy=(anchor_x, 1.0),
            xytext=(0.03, 0.62),
            textcoords="axes fraction",
            fontsize=6.6,
            color="#8a3d00",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#D55E00"),
        )
    ax_b.set_xlabel("per-sequence full-rank $\\rho$")
    ax_b.set_ylabel("per-sequence top-3 Jaccard")
    ax_b.set_yticks([0, 0.2, 0.5, 1.0])
    ax_b.set_ylim(-0.12, 1.14)
    ax_b.set_title(f"(b) {cell}", fontsize=9, loc="left")

    fig.tight_layout(pad=0.4, w_pad=1.4)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=PNG_DPI)
    plt.close(fig)

    payload = {
        "figure": out.name,
        "cell": cell,
        "source": _describe(run),
        "dumped_seeds": [int(v) for v in d["seeds"]],
        "panel_a": {
            "y": "per-sequence normalized attribution mass, sorted descending",
            "n_sequence_profiles": int(prof.shape[0]),
            "median_by_rank": [float(v) for v in p50],
            "p25_by_rank": [float(v) for v in p25],
            "p75_by_rank": [float(v) for v in p75],
            "median_top3_mass": top3,
        },
        "panel_b": {
            "x": "per-sequence full-rank Spearman, accuracy-matched pairs",
            "y": "per-sequence top-3 Jaccard",
            "rho_star": rho_star,
            "n_points": int(rho.size),
            "n_left_of_line": int(left.sum()),
            "n_exact_top3": int(exact.sum()),
            "n_left_and_exact": n_arg,
            "frac_left_and_exact": frac_arg,
            "frac_exact_among_left": frac_left,
        },
    }
    payload["sidecar"] = _write_sidecar(out, payload).name
    return payload


# --------------------------------------------------------------------- fig A.2


def _per_seed(run: pathlib.Path, key: str) -> dict[str, dict[int, float]]:
    """Per-cell, per-seed values of one field, read from the seed_*.json files.

    The aggregated results.json keeps only mean and bootstrap bounds, which
    cannot show the basin spread: a cell whose cold restarts land anywhere
    between 0.21 and 0.87 looks the same as a tight one once summarized.
    """
    out: dict[str, dict[int, float]] = {}
    for f in sorted(run.glob("seed_*.json")):
        d = json.load(open(f))
        seed = int(d["seed"])
        cells = (
            d["cells"] if isinstance(d["cells"], list) else list(d["cells"].values())
        )
        for c in cells:
            v = c.get(key)
            if v is None or not np.isfinite(float(v)):
                continue
            out.setdefault(f"{c['hidden']}x{c['depth']}", {})[seed] = float(v)
    return out


def figureA2(warm_run: pathlib.Path, cold_run: pathlib.Path, out: pathlib.Path) -> dict:
    """The vacuous ceiling: warm-start returns 1.000000 without optimizing."""
    warm = _per_seed(warm_run, "null_heldout")
    cold = _per_seed(cold_run, "arm1_cold_heldout")
    cells = sorted(
        set(warm) & set(cold),
        key=lambda k: (int(k.split("x")[0]), int(k.split("x")[1])),
    )
    if not cells:
        raise SystemExit("no cells shared between the warm and cold runs")

    rows = []
    for c in cells:
        w = np.array(sorted(warm[c].values()))
        k = np.array(sorted(cold[c].values()))
        rows.append(
            {
                "cell": c,
                "warm_mean": float(w.mean()),
                "warm_min": float(w.min()),
                "warm_max": float(w.max()),
                "warm_all_exactly_one": bool(np.all(w == 1.0)),
                "warm_seeds": int(w.size),
                "cold_mean": float(k.mean()),
                "cold_min": float(k.min()),
                "cold_max": float(k.max()),
                "cold_seeds": int(k.size),
            }
        )

    x = np.arange(len(cells), dtype=float)
    wdt = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 2.9))

    ax.bar(
        x - wdt / 2,
        [r["warm_mean"] for r in rows],
        wdt,
        color="#999999",
        edgecolor="white",
        lw=0.5,
        zorder=2,
        label="warm start (begins at the target)",
    )
    ax.bar(
        x + wdt / 2,
        [r["cold_mean"] for r in rows],
        wdt,
        color="#0072B2",
        edgecolor="white",
        lw=0.5,
        zorder=2,
        label="cold start (must search)",
    )

    # Per-seed points. The warm arm is a single value repeated, so its strip is
    # a flat line at 1.0 -- which is the point of the figure, not a defect.
    rng = np.random.default_rng(0)
    for i, c in enumerate(cells):
        for off, src, col in (
            (-wdt / 2, warm[c], "0.35"),
            (wdt / 2, cold[c], "#00436b"),
        ):
            v = np.array(list(src.values()))
            jx = i + off + rng.uniform(-wdt * 0.28, wdt * 0.28, size=v.size)
            ax.scatter(jx, v, s=5, c=col, alpha=0.75, lw=0, zorder=4)

    ax.axhline(1.0, color="0.2", lw=0.7, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cells, rotation=45, ha="right")
    ax.set_ylabel(
        "held-out $R^2$ recovering a target\nthe architecture realizes exactly"
    )
    ax.set_xlabel("cell (width $\\times$ depth)")
    ax.set_ylim(0.0, 1.04)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Bars fill the whole 0-1 range, so any in-axes legend sits on data.
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        fontsize=7,
        handletextpad=0.5,
        borderpad=0.2,
        ncol=2,
        columnspacing=1.4,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=PNG_DPI)
    plt.close(fig)

    payload = {
        "figure": out.name,
        "warm_source": _describe(warm_run),
        "cold_source": _describe(cold_run),
        "warm_field": "closure_heldout seed_*.json cells[*].null_heldout",
        "cold_field": "closure_search seed_*.json cells[*].arm1_cold_heldout",
        "warm_all_exactly_one_everywhere": all(r["warm_all_exactly_one"] for r in rows),
        "n_warm_values": sum(r["warm_seeds"] for r in rows),
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


def _print_sep(title: str, f: dict) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"  population : {f['population']}")
    _print_source(f["kind"], f["source"])
    print(f"  y left  = {f['y_left']}  (log scale)")
    print(f"  y right = {f['y_right']}")
    ref = f["reference_line"]
    print(
        f"  library reference = {ref['value']:,.0f} from {ref['read_from']}"
        f"  (drawn: {ref['drawn']})"
    )
    print(
        f"  cells above the full library: {f['n_cells_above_reference']}"
        f" of {len(f['rows'])}"
    )
    print()
    print(
        f"  {'cell':>7}  {'depth':>5}  {'n to separate':>16}  {'lo':>15}  {'hi':>16}"
        f"  {'top3 diverg.':>12}"
    )
    for r in f["rows"]:
        print(
            f"  {r['cell']:>7}  {r['depth']:>5}  {r['n_separate_mean']:>16,.4f}"
            f"  {r['n_separate_lo']:>15,.4f}  {r['n_separate_hi']:>16,.4f}"
            f"  {r['top3_divergence']:>12.4f}"
        )


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

    # Both MAIN-TEXT figures come from the occurrence run, so the body of the
    # paper describes one population throughout. The twin comparison is a real
    # and separate result and keeps its own figure, in the appendix, named for
    # what it measures.
    f1 = figure1(occ_run, out_dir / "fig1_not_weak_models.pdf")
    f3 = figure_separability(
        occ_run, out_dir / "fig3_occurrence_separability.pdf", "occurrence"
    )
    fa = figure_separability(sep_run, out_dir / "figA_twin_separability.pdf", "twin")

    ch_run = resolve_run("closure_heldout", ("hidden", "depth"), None)
    cs_run = resolve_run(
        "closure_search", ("hidden", "depth", "arm1_cold_heldout"), None
    )
    fa2 = figureA2(ch_run, cs_run, out_dir / "figA2_vacuous_ceiling.pdf")

    # Figure 2 needs the per-instance dump. If none exists the rest of the
    # figures still build, and the caller is told plainly rather than getting a
    # stale or silently-missing panel.
    dump = resolve_dump()
    f2 = None
    if dump is None:
        print(
            "\nNOTE: no per-instance dump found; Figure 2 skipped.\n"
            "      run: python -m experiments.occurrence --dump-per-instance"
        )
    else:
        f2 = figure2(dump, out_dir / "fig2_concentration.pdf")

    print("\n" + "=" * 78)
    print("FIGURE 1 (main)  paper/figures/fig1_not_weak_models.pdf")
    print("=" * 78)
    print("  population : independently trained, accuracy-matched pairs")
    _print_source("occurrence", f1["source"])
    print(f"  x = {f1['x']}")
    print(f"  y = {f1['y']}")
    print(f"  annotated  : {f1['annotated_cell']}")
    print()
    print(
        f"  {'cell':>7}  {'depth':>5}  {'x heldout R2':>13}"
        f"  {'pooled_tk_ex3':>14}  {'y divergence':>13}"
    )
    for r in f1["rows"]:
        print(
            f"  {r['cell']:>7}  {r['depth']:>5}  {r['heldout_r2_mean']:>13.6f}"
            f"  {r['pooled_tk_ex3']:>14.6f}  {r['top3_divergence']:>13.6f}"
        )

    if f2 is not None:
        print("\n" + "=" * 78)
        print("FIGURE 2 (main)  paper/figures/fig2_concentration.pdf")
        print("=" * 78)
        print(f"  population : {f2['cell']}, accuracy-matched pairs")
        _print_source("dump", f2["source"])
        print(f"  dumped seeds: {f2['dumped_seeds']}")
        a, b = f2["panel_a"], f2["panel_b"]
        print(f"\n  (a) {a['n_sequence_profiles']:,} per-sequence profiles")
        print(f"      median top-3 mass      : {a['median_top3_mass']:.4f}")
        print(f"      {'rank':>5} {'p25':>9} {'median':>9} {'p75':>9}")
        for i, (q1, md, q3) in enumerate(
            zip(a["p25_by_rank"], a["median_by_rank"], a["p75_by_rank"]), start=1
        ):
            print(f"      {i:>5} {q1:>9.4f} {md:>9.4f} {q3:>9.4f}")
        print(f"\n  (b) {b['n_points']:,} (sequence, pair) points")
        print(f"      rho* from Prop. 2      : {b['rho_star']}")
        print(f"      left of rho*           : {b['n_left_of_line']:,}")
        print(f"      exact top-3 match      : {b['n_exact_top3']:,}")
        print(f"      LEFT and EXACT         : {b['n_left_and_exact']:,}")
        print(f"      fraction of all points : {b['frac_left_and_exact'] * 100:.1f}%")
        print(f"      fraction among left    : {b['frac_exact_among_left'] * 100:.1f}%")

    _print_sep("FIGURE 3 (main)  paper/figures/fig3_occurrence_separability.pdf", f3)
    _print_sep("APPENDIX FIGURE  paper/figures/figA_twin_separability.pdf", fa)

    print("\n" + "=" * 78)
    print("APPENDIX FIGURE  paper/figures/figA2_vacuous_ceiling.pdf")
    print("=" * 78)
    print(f"  warm : {fa2['warm_source']['run']}  git {fa2['warm_source']['git_sha']}")
    print(f"  cold : {fa2['cold_source']['run']}  git {fa2['cold_source']['git_sha']}")
    print(f"  warm field : {fa2['warm_field']}")
    print(f"  cold field : {fa2['cold_field']}")
    print(
        f"  warm arm is EXACTLY 1.0 in all {fa2['n_warm_values']} values: "
        f"{fa2['warm_all_exactly_one_everywhere']}"
    )
    print()
    print(
        f"  {'cell':>7} {'warm mean':>10} {'warm min':>9} {'warm max':>9}"
        f" {'cold mean':>10} {'cold min':>9} {'cold max':>9}"
    )
    for r in fa2["rows"]:
        print(
            f"  {r['cell']:>7} {r['warm_mean']:>10.6f} {r['warm_min']:>9.6f}"
            f" {r['warm_max']:>9.6f} {r['cold_mean']:>10.4f} {r['cold_min']:>9.4f}"
            f" {r['cold_max']:>9.4f}"
        )

    print("\nwrote:")
    names = ["fig1_not_weak_models.pdf"]
    if f2 is not None:
        names.append("fig2_concentration.pdf")
    names += [
        "fig3_occurrence_separability.pdf",
        "figA_twin_separability.pdf",
        "figA2_vacuous_ceiling.pdf",
    ]
    for name in names:
        print(f"  {out_dir / name}")
    side = [f1["sidecar"], f3["sidecar"], fa["sidecar"], fa2["sidecar"]]
    if f2 is not None:
        side.insert(1, f2["sidecar"])
    print("sidecars: " + ", ".join(side))


if __name__ == "__main__":
    main()
