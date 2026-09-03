#!/usr/bin/env python
"""What do the never-trained input weights do to the reported top-k sets?

The BRCA2 library holds position 4 at G in all 30,483 rows and position 5 to
{C,U}. One-hot encoding therefore leaves five of 36 input units identically
zero, so their weights receive no gradient and keep their N(0, 0.3^2)
initialization for the whole of training. In-silico mutagenesis (Eq. 1)
iterates over all nine positions regardless, so part of every attribution
vector is a function of the initialization seed alone.

This reads the position-indexed `__ism` array from a committed dump and asks
how much of the reported disagreement that accounts for.

    python -m experiments.dead_positions [--run RUNDIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEAD = (3, 4)  # zero-indexed: positions 4 and 5
FREE = (0, 1, 2, 5, 6, 7, 8)


def cells_of(z) -> list[str]:
    names = {k.split("_seed")[0] for k in z.files if k.endswith("__ism")}
    return sorted(names, key=lambda c: (int(c.split("x")[0]), int(c.split("x")[1])))


def seeds_of(z, cell: str) -> list[int]:
    return sorted(
        int(k.split("_seed")[1].split("__")[0])
        for k in z.files
        if k.startswith(f"{cell}_seed") and k.endswith("__ism")
    )


def topk(a: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest entries per row, of the columns given."""
    return np.argsort(-a, axis=-1)[..., :k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default="paper/figures/dead_positions_data.json")
    args = ap.parse_args()

    run = pathlib.Path(args.run) if args.run else None
    if run is None:
        cands = [
            d
            for d in sorted((ROOT / "results" / "runs").glob("*/"))
            if (d / "per_instance.npz").exists()
            and any(k.endswith("__ism") for k in np.load(d / "per_instance.npz").files)
        ]
        if not cands:
            raise SystemExit("no committed dump carries a position-indexed __ism array")
        run = cands[-1]
    if not run.is_absolute():
        run = ROOT / run
    z = np.load(run / "per_instance.npz")
    meta = json.load(open(run / "meta.json"))
    print(f"run {run.name}  sha={meta['git_sha']}  seeds={meta['seeds']}\n")

    rows = {}
    for cell in cells_of(z):
        sd = seeds_of(z, cell)
        inc4 = inc5 = nmod = 0
        dis = dis_dead = 0
        m9 = {"t1": 0, "t2": 0, "t3": 0, "ex3": 0, "jac": 0.0, "n": 0}
        m7 = {"t1": 0, "t2": 0, "t3": 0, "ex3": 0, "jac": 0.0, "n": 0}
        mass9, eff9, mass7, eff7 = [], [], [], []
        for s in sd:
            A = z[f"{cell}_seed{s}__ism"].astype(float)  # [models, inst, 9]
            pairs = z[f"{cell}_seed{s}__pairs"]
            # (1) per-model top-3 membership of the dead positions
            t3 = topk(A, 3)
            inc4 += int((t3 == DEAD[0]).any(axis=-1).sum())
            inc5 += int((t3 == DEAD[1]).any(axis=-1).sum())
            nmod += A.shape[0] * A.shape[1]
            # (4) concentration on nine vs seven positions
            for M, mass, eff in ((A, mass9, eff9), (A[..., FREE], mass7, eff7)):
                tot = M.sum(axis=-1, keepdims=True)
                q = np.where(tot > 0, M / np.where(tot == 0, 1, tot), 0.0)
                srt = np.sort(q, axis=-1)[..., ::-1]
                mass.append(srt[..., :3].sum(axis=-1).ravel())
                with np.errstate(divide="ignore", invalid="ignore"):
                    h = -(q * np.where(q > 0, np.log(q), 0.0)).sum(axis=-1)
                eff.append(np.exp(h).ravel())
            # (2, 3) pairwise agreement, nine vs seven positions
            for i, j in pairs:
                a, b = A[i], A[j]
                sa, sb = topk(a, 3), topk(b, 3)
                for r in range(a.shape[0]):
                    A3, B3 = set(sa[r].tolist()), set(sb[r].tolist())
                    same = A3 == B3
                    m9["t1"] += sa[r, 0] != sb[r, 0]
                    m9["t2"] += set(sa[r, :2].tolist()) != set(sb[r, :2].tolist())
                    m9["t3"] += not same
                    m9["ex3"] += same
                    m9["jac"] += len(A3 & B3) / len(A3 | B3)
                    m9["n"] += 1
                    if not same:
                        dis += 1
                        if (A3 ^ B3) & set(DEAD):
                            dis_dead += 1
                af, bf = a[:, FREE], b[:, FREE]
                sa7, sb7 = topk(af, 3), topk(bf, 3)
                for r in range(af.shape[0]):
                    A3, B3 = set(sa7[r].tolist()), set(sb7[r].tolist())
                    same = A3 == B3
                    m7["t1"] += sa7[r, 0] != sb7[r, 0]
                    m7["t2"] += set(sa7[r, :2].tolist()) != set(sb7[r, :2].tolist())
                    m7["t3"] += not same
                    m7["ex3"] += same
                    m7["jac"] += len(A3 & B3) / len(A3 | B3)
                    m7["n"] += 1
        rows[cell] = {
            "p4_in_top3": inc4 / nmod,
            "p5_in_top3": inc5 / nmod,
            "disagreements": dis,
            "disagree_involving_dead": dis_dead / dis if dis else float("nan"),
            "nine": {k: (v / m9["n"] if k != "n" else v) for k, v in m9.items()},
            "seven": {k: (v / m7["n"] if k != "n" else v) for k, v in m7.items()},
            "mass9": float(np.median(np.concatenate(mass9))),
            "mass7": float(np.median(np.concatenate(mass7))),
            "eff9": float(np.median(np.concatenate(eff9))),
            "eff7": float(np.median(np.concatenate(eff7))),
        }
        print(f"  {cell} done")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"run": run.name, "git_sha": meta["git_sha"], "seeds": meta["seeds"],
         "dead_positions": [d + 1 for d in DEAD], "cells": rows},
        open(out, "w"), indent=2,
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
