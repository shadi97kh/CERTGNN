#!/usr/bin/env python
"""Aggregate run directories into tables. Fails loudly on any incomplete cell.

    python scripts/aggregate.py --runs results/runs --out paper/tables
        [--manifest results/sweeps/<id>.json] [--expected-seeds N] [--tuning-budget]

Per cell (experiment x cell overrides): mean, 95% bootstrap CI and n for
every metric in per_seed_values.json. Holm-Bonferroni across the gate
family from gate_stats.json (latest run per gate). LaTeX (booktabs) and
Markdown are written to --out.

Completeness is checked against the manifest when given (every planned
run_id must have finished with exit code 0 and a results directory), else
every cell must hold at least --expected-seeds seeds (default 5, the
project minimum). On any gap the script writes nothing and exits 1 with the
gap listed. It never backfills from a neighbouring cell, never drops a
cell, and never emits a partial table as if complete.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments._common import fmt_ci, mean_ci  # noqa: E402


# ------------------------------------------------------------------ loading


def load_runs(runs_dir: pathlib.Path) -> list[dict[str, Any]]:
    runs = []
    for d in sorted(runs_dir.iterdir()):
        meta_p = d / "meta.json"
        if not d.is_dir() or not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        run = {"dir": d, "meta": meta}
        for name in ("per_seed_values", "gate_stats", "tuning_budget"):
            p = d / f"{name}.json"
            run[name] = json.loads(p.read_text()) if p.exists() else None
        run["complete"] = (d / "results.json").exists() and run[
            "per_seed_values"
        ] is not None
        runs.append(run)
    return runs


def cell_key(run: dict[str, Any]) -> tuple[str, str]:
    meta = run["meta"]
    sweep = meta.get("sweep") or {}
    cell = (
        json.dumps(sweep.get("cell", {}), sort_keys=True)
        if sweep
        else meta["config_hash"]
    )
    return meta["experiment"], cell


# ------------------------------------------------------------ completeness


def check_manifest(runs: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    by_id = {r["meta"].get("run_id"): r for r in runs if r["meta"].get("run_id")}
    problems = []
    for planned in manifest["runs"]:
        rid = planned["run_id"]
        if planned.get("exit_code", None) not in (0,):
            problems.append(
                f"{rid}: exit code {planned.get('exit_code', 'not launched')}"
            )
        r = by_id.get(rid)
        if r is None:
            problems.append(f"{rid}: no run directory")
        elif not r["complete"]:
            problems.append(
                f"{rid}: run directory {r['dir']} has no results.json / per_seed_values.json"
            )
    if manifest.get("rows_unlaunchable"):
        problems.append(
            f"rows with no runner were never run: {manifest['rows_unlaunchable']}"
        )
    return problems


def collect_cells(
    runs: list[dict[str, Any]], expected_seeds: int
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    problems = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if r["complete"]:
            grouped[cell_key(r)].append(r)
    for key, rs in grouped.items():
        metrics: dict[str, dict[str, float]] = defaultdict(dict)
        for r in rs:
            for metric, per_seed in r["per_seed_values"].items():
                for seed, val in per_seed.items():
                    if seed in metrics[metric] and metrics[metric][seed] != val:
                        problems.append(
                            f"{key}: seed {seed} of {metric} appears twice with different values"
                        )
                    metrics[metric][seed] = val
        n_by_metric = {m: len(v) for m, v in metrics.items()}
        short = {m: n for m, n in n_by_metric.items() if n < expected_seeds}
        if short:
            problems.append(
                f"{key[0]} cell {key[1]}: fewer than {expected_seeds} seeds for {short}"
            )
        cells[key] = {
            "experiment": key[0],
            "cell": key[1],
            "git_shas": sorted(
                {
                    r["meta"]["git_sha"] + ("-dirty" if r["meta"]["git_dirty"] else "")
                    for r in rs
                }
            ),
            "metrics": {
                m: mean_ci([v for v in vals.values() if v is not None])
                for m, vals in metrics.items()
            },
        }
    return cells, problems


# ------------------------------------------------------------------- Holm


def holm_bonferroni(
    pvals: dict[str, float], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Holm step-down: adjusted p-values and rejections at ``alpha``."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # enforce monotonicity
        out[k] = {"p": p, "p_adjusted": running, "reject": running <= alpha}
    return out


def gate_table(runs: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for r in runs:
        gs = r.get("gate_stats")
        if gs:
            latest[gs["gate"]] = {
                **gs,
                "run": r["dir"].name,
                "git_sha": r["meta"]["git_sha"],
            }
    if not latest:
        return {}
    holm = holm_bonferroni({g: float(v["p_value"]) for g, v in latest.items()})
    return {g: {**latest[g], **holm[g]} for g in latest}


# ----------------------------------------------------------------- output


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def cells_markdown(cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    L = [
        "# Aggregated cells\n",
        "| experiment | cell | metric | mean [95% CI] (n) | git |",
        "|---|---|---|---|---|",
    ]
    for c in cells.values():
        for m, s in c["metrics"].items():
            L.append(
                f"| {c['experiment']} | `{c['cell']}` | {m} | {fmt_ci(s)} | {','.join(c['git_shas'])} |"
            )
    return "\n".join(L) + "\n"


def cells_latex(cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    L = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"experiment & metric & mean & 95\% CI & $n$ \\",
        r"\midrule",
    ]
    for c in cells.values():
        for m, s in c["metrics"].items():
            L.append(
                f"{_tex_escape(c['experiment'])} & {_tex_escape(m)} & {s['mean']:.3f} & [{s['lo']:.3f}, {s['hi']:.3f}] & {s['n']} \\\\"
            )
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def gates_markdown(gates: dict[str, Any]) -> str:
    if not gates:
        return "# Gates\n\nNo gate_stats.json found.\n"
    L = [
        "# Gate family (Holm-Bonferroni)\n",
        "| gate | verdict | p (bootstrap vs threshold) | Holm-adjusted p | reject at 0.05 | seeds | run |",
        "|---|---|---|---|---|---|---|",
    ]
    for g, v in sorted(gates.items()):
        L.append(
            f"| {g} | {v['verdict']} | {v['p']:.4f} | {v['p_adjusted']:.4f} | {v['reject']} | {v['n_seeds']} | {v['run']} |"
        )
    L.append(
        "\nVerdicts are the pre-registered threshold comparisons; p is the one-sided bootstrap probability that the seed-mean fails its threshold, (b+1)/(B+1) convention; it is a CI inversion, not a null-calibrated test, and saturates near 1/(B+1) whenever the threshold lies outside the seed range."
    )
    return "\n".join(L) + "\n"


def gates_latex(gates: dict[str, Any]) -> str:
    L = [
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"gate & verdict & $p$ & Holm $p$ & reject \\",
        r"\midrule",
    ]
    for g, v in sorted(gates.items()):
        L.append(
            f"{g} & {v['verdict']} & {v['p']:.4f} & {v['p_adjusted']:.4f} & {'yes' if v['reject'] else 'no'} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def tuning_budget_markdown(runs: list[dict[str, Any]]) -> str:
    L = [
        "# Tuning budget\n",
        "Search effort per model and experiment, from tuning_budget.json in every run directory. "
        "Baselines must receive equal effort (PREREGISTRATION analysis plan); unequal rows are flagged.\n",
        "| experiment | substrate | model | configs tried | epochs | gradient steps | search space | selection | runs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in sorted(
        runs, key=lambda r: r["meta"].get("timestamp_utc", "")
    ):  # latest entry wins
        for e in r.get("tuning_budget") or []:
            key = (r["meta"]["experiment"], str(e.get("substrate", "-")), e["model"])
            prev = agg.get(key)
            agg[key] = {**e, "runs": (prev["runs"] if prev else 0) + 1}
    flags = []
    by_exp: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for (exp, sub, model), e in sorted(agg.items()):
        L.append(
            f"| {exp} | {sub} | {model} | {e['configs_tried']} | {e['epochs']} | {e.get('gradient_steps', '-')} | {e['search_space']} | {e['selection']} | {e['runs']} |"
        )
        if model != "none":
            by_exp[(exp, sub)].add((int(e["configs_tried"]), int(e["epochs"])))
    for (exp, sub), effort in by_exp.items():
        if len(effort) > 1:
            flags.append(
                f"- UNEQUAL effort in {exp}/{sub}: {sorted(effort)} (configs tried, epochs)"
            )
    L.append("")
    L.extend(
        flags
        or [
            "All models within each experiment received equal search effort by the recorded counts."
        ]
    )
    notes = pathlib.Path("results/tuning_budget_notes.md")
    if notes.exists():
        L += [
            "",
            "## Known discrepancies (results/tuning_budget_notes.md)",
            "",
            notes.read_text().strip(),
        ]
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--expected-seeds", type=int, default=5)
    ap.add_argument("--tuning-budget", default="results/tuning_budget.md")
    ap.add_argument("--prefix", default="aggregate")
    args = ap.parse_args(argv)

    all_runs = load_runs(pathlib.Path(args.runs))
    runs = all_runs
    problems: list[str] = []
    if args.manifest:
        manifest = json.loads(pathlib.Path(args.manifest).read_text())
        problems += check_manifest(runs, manifest)
        ids = {p["run_id"] for p in manifest["runs"]}
        runs = [r for r in runs if r["meta"].get("run_id") in ids]
    expected = args.expected_seeds
    if args.manifest:
        per_module: dict[str, int] = defaultdict(int)
        for planned in manifest["runs"]:
            per_module[planned.get("module", "?")] += 1
        expected = max(expected, min(per_module.values()))  # planned seeds per module
    cells, cell_problems = collect_cells(runs, expected)
    problems += cell_problems
    if problems:
        print(
            "AGGREGATION REFUSED: incomplete cells. Nothing was written.",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if not cells:
        print("AGGREGATION REFUSED: no complete runs found.", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gates = gate_table(all_runs)  # the gate family spans sweeps
    (out / f"{args.prefix}_cells.md").write_text(cells_markdown(cells))
    (out / f"{args.prefix}_cells.tex").write_text(cells_latex(cells))
    (out / f"{args.prefix}_gates.md").write_text(gates_markdown(gates))
    (out / f"{args.prefix}_gates.tex").write_text(gates_latex(gates))
    tb = pathlib.Path(args.tuning_budget)
    tb.parent.mkdir(parents=True, exist_ok=True)
    tb.write_text(tuning_budget_markdown(all_runs))
    print(
        f"wrote {out / (args.prefix + '_cells.{md,tex}')}, {out / (args.prefix + '_gates.{md,tex}')}, {tb}"
    )
    print(cells_markdown(cells))
    print(gates_markdown(gates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
