#!/usr/bin/env python
"""Expand ABLATIONS.md rows into configs and launch them.

    python scripts/launch_sweep.py --matrix ABLATIONS.md --tier 0 [--rows 0.6,0.7]
        [--seeds 10] [--dry-run] [--allow-dirty] [--partial] [--include-draft]

Every launched run gets results/runs/<timestamp>_<gitsha>_<confighash>/ with
the resolved config, git SHA and dirty flag, seed, environment lockfile and
raw per-seed values (written by the experiment via experiments/_common).
The launcher writes a manifest to results/sweeps/<sweep_id>.json listing
every planned run_id, so scripts/aggregate.py can check completeness
against the plan rather than against what happened to finish.

Refuses to launch on a dirty working tree unless --allow-dirty; the flag is
recorded in the manifest and in every run's meta.json. Refuses to launch a
tier in which some requested row has no registered runner unless
--partial; those rows are always listed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments._common import git_dirty, git_sha, load_config  # noqa: E402
from experiments.registry import runner_for  # noqa: E402


def parse_matrix(path: pathlib.Path) -> list[dict[str, Any]]:
    text = path.read_text()
    rows = [yaml.safe_load(b) for b in re.findall(r"```yaml\n(.*?)```", text, re.S)]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row ids in matrix")
    return rows


def select_rows(
    rows: list[dict[str, Any]], tier: int, only: list[str] | None, include_draft: bool
) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r["status"] == "candidate":
            continue
        if int(r["tier"]) != tier:
            continue
        if only and r["id"] not in only:
            continue
        if r["status"] == "draft" and not include_draft:
            print(
                f"skipping draft row {r['id']} ({r['name']}); pass --include-draft to launch it",
                file=sys.stderr,
            )
            continue
        out.append(r)
    if only:
        missing = set(only) - {r["id"] for r in out}
        if missing:
            raise SystemExit(
                f"requested rows not in tier {tier} / not launchable: {sorted(missing)}"
            )
    return out


def expand_cells(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Cells of a row: dotted-override dicts. One-factor from the reference
    config, two-factor cross product of exactly two axes, or a single fixed cell."""
    axes: dict[str, list[Any]] = dict(row["axes"] or {})
    if row["sweep"] == "fixed":
        return [{}]
    if row["sweep"] == "one_factor":
        return [{k: v} for k, vals in axes.items() for v in vals]
    if row["sweep"] == "two_factor":
        if len(axes) != 2:
            raise ValueError(f"row {row['id']}: two_factor needs exactly 2 axes")
        (k1, v1), (k2, v2) = axes.items()
        return [{k1: a, k2: b} for a, b in itertools.product(v1, v2)]
    raise ValueError(f"row {row['id']}: unknown sweep {row['sweep']!r}")


def _fmt(v: Any) -> str:
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def plan(
    rows: list[dict[str, Any]], seeds: int, base: str, sweep_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (launchable runs, unlaunchable rows)."""
    runs: list[dict[str, Any]] = []
    unlaunchable: list[dict[str, Any]] = []
    seen_modules: set[str] = set()
    for row in rows:
        rr = runner_for(row["id"])
        n_seeds = max(int(seeds), int(row["seeds"]))
        if rr is None:
            unlaunchable.append(
                {"id": row["id"], "name": row["name"], "cells": expand_cells(row)}
            )
            continue
        module, mode = rr
        if mode == "row_experiment":
            if module in seen_modules:
                continue  # one run per seed covers every row this module implements
            seen_modules.add(module)
            covered = [
                r["id"] for r in rows if (runner_for(r["id"]) or ("",))[0] == module
            ]
            for k in range(n_seeds):
                run_id = f"{sweep_id}/{module.split('.')[-1]}_seed{k}"
                runs.append(
                    {
                        "run_id": run_id,
                        "module": module,
                        "rows": covered,
                        "cell": {},
                        "seed": k,
                        "overrides": [f"seed={k}", f"run_id={run_id}"],
                    }
                )
        else:
            for ci, cell in enumerate(expand_cells(row)):
                for k in range(n_seeds):
                    run_id = f"{sweep_id}/{row['id']}_c{ci}_seed{k}"
                    ov = [f"{key}={_fmt(val)}" for key, val in cell.items()] + [
                        f"seed={k}",
                        f"run_id={run_id}",
                    ]
                    runs.append(
                        {
                            "run_id": run_id,
                            "module": module,
                            "rows": [row["id"]],
                            "cell": cell,
                            "seed": k,
                            "overrides": ov,
                        }
                    )
    return runs, unlaunchable


def write_config(
    base: str, run: dict[str, Any], out_dir: pathlib.Path, sweep_meta: dict[str, Any]
) -> pathlib.Path:
    cfg = load_config(base, run["overrides"])
    merged = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "_resolved": True,
                "sweep": {**sweep_meta, "rows": run["rows"], "cell": run["cell"]},
            }
        ),
    )
    assert isinstance(merged, DictConfig)
    cfg = merged
    path = out_dir / (run["run_id"].split("/", 1)[1] + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--matrix", default="ABLATIONS.md")
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument(
        "--rows", default=None, help="comma-separated row ids to restrict to"
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="minimum seeds per cell (rows may require more)",
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="configs/generated")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="generate configs and manifest, launch nothing",
    )
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument(
        "--partial",
        action="store_true",
        help="launch runnable rows even if some rows have no runner",
    )
    ap.add_argument("--include-draft", action="store_true")
    args = ap.parse_args(argv)

    dirty = git_dirty()
    if dirty and not args.allow_dirty and not args.dry_run:
        raise SystemExit(
            "REFUSING TO LAUNCH: working tree is dirty. Commit, or pass --allow-dirty (recorded in the manifest)."
        )

    rows = select_rows(
        parse_matrix(pathlib.Path(args.matrix)),
        args.tier,
        args.rows.split(",") if args.rows else None,
        args.include_draft,
    )
    if not rows:
        raise SystemExit(f"no launchable rows in tier {args.tier}")
    sweep_id = f"tier{args.tier}_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}_{git_sha()}"
    runs, unlaunchable = plan(rows, args.seeds, args.config, sweep_id)
    out_dir = pathlib.Path(args.out) / sweep_id
    sweep_meta = {
        "id": sweep_id,
        "tier": args.tier,
        "matrix": args.matrix,
        "allow_dirty": bool(args.allow_dirty),
        "git_dirty": dirty,
    }

    for run in runs:
        run["config_path"] = str(write_config(args.config, run, out_dir, sweep_meta))
    for (
        u
    ) in unlaunchable:  # expansion is written for inspection even though it cannot run
        cells_path = out_dir / f"{u['id']}_cells.NO_RUNNER.json"
        cells_path.parent.mkdir(parents=True, exist_ok=True)
        cells_path.write_text(json.dumps(u, indent=1))

    if unlaunchable:
        print("ROWS WITH NO REGISTERED RUNNER (not launched):", file=sys.stderr)
        for u in unlaunchable:
            print(
                f"  {u['id']} {u['name']}: {len(u['cells'])} cell(s) expanded to {out_dir}",
                file=sys.stderr,
            )
        if not args.partial and not args.dry_run:
            raise SystemExit(
                "REFUSING TO LAUNCH: some requested rows cannot run. Pass --partial to launch the rest; the manifest will list the gap."
            )

    manifest = {
        **sweep_meta,
        "rows_requested": [r["id"] for r in rows],
        "rows_unlaunchable": [u["id"] for u in unlaunchable],
        "runs": runs,
        "dry_run": bool(args.dry_run),
    }
    man_dir = pathlib.Path("results/sweeps")
    man_dir.mkdir(parents=True, exist_ok=True)
    man_path = man_dir / f"{sweep_id}.json"
    man_path.write_text(json.dumps(manifest, indent=1))
    print(
        f"sweep {sweep_id}: {len(runs)} run(s) planned, {len(unlaunchable)} row(s) without runner; manifest {man_path}"
    )
    if args.dry_run:
        return 0

    for run in runs:
        cmd = [sys.executable, "-m", run["module"], "--config", run["config_path"]] + (
            ["--allow-dirty"] if args.allow_dirty else []
        )
        print(f"launch {run['run_id']}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd)
        run["exit_code"] = proc.returncode
        man_path.write_text(json.dumps(manifest, indent=1))
        if proc.returncode != 0:
            print(
                f"RUN FAILED (exit {proc.returncode}): {run['run_id']}; recorded in manifest, continuing",
                file=sys.stderr,
            )
    failed = [r["run_id"] for r in runs if r.get("exit_code", 1) != 0]
    print(
        f"done: {len(runs) - len(failed)} ok, {len(failed)} failed"
        + (f": {failed}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
