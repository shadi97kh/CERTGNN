"""Tier 0 controls: ABLATIONS rows 0.6, 0.7 and (brain only) 0.8.

    0.6  degree-preserving topology shuffle: same node features, edges
         rewired by double-edge swaps, GNN retrained.
    0.7  MLP on node features with all edges removed.
    0.8  Hadamard / BQN baseline (Yang et al., ICML 2025) on brain substrates.

For every substrate: GNN, shuffled-topology GNN, edge-free MLP and, where
applicable, BQN test performance with 95% bootstrap CIs over seeds, paired
seed-wise differences against the GNN, the pre-stated interpretation, and a
substrate verdict. A cross-substrate recommendation (SpliceCert vs
ConnectomeCert) closes the report, including the case where no substrate
passes or a substrate could not be evaluated.

Interpretation, stated in the output rather than left to the reader:
- MLP matches GNN            -> not a graph problem; cannot carry a topology paper.
- shuffled matches real      -> topology is decorative.
- BQN beats GNN (brain)      -> the ICML 2025 critique applies; connectome not viable as framed.

"Matches" means the 95% CI of the paired seed-wise difference (GNN minus
control) includes zero or lies below it. "Beats" means the CI of
(control minus GNN) lies strictly above zero.

Usage:
    python -m experiments.tier0_controls --config configs/base.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import pathlib
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from certgnn.eval.sanity import degree_preserving_rewire
from certgnn.models import (
    BrainQuadraticNetwork,
    NodeFeatureMLP,
    TargetReadoutGCN,
    fit,
    primary_metric,
)
from certgnn.substrates.base import Substrate
from certgnn.substrates.synthetic import (
    SyntheticConfig,
    SyntheticSubstrate,
    TopologySpec,
)
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

MODELS = ("gnn", "shuffled", "mlp", "mlp_mean", "bqn")


# ------------------------------------------------------------ substrates


class SubstrateUnavailable(RuntimeError):
    """The substrate cannot be evaluated: not implemented or no data."""


def make_substrate(name: str, cfg: Any, seed: int = 0) -> Substrate:
    if name == "synthetic":
        s = (
            cfg.substrate
            if cfg.substrate.get("name") == "synthetic"
            else load_config("configs/base.yaml", ["substrate=synthetic"]).substrate
        )
        return SyntheticSubstrate(
            SyntheticConfig(
                topology=TopologySpec(**dict(s.topology)),
                regulator=str(s.regulator),
                delta=float(s.delta),
                link=str(s.link),
                p0=(None if s.get("p0") is None else float(s.p0)),
                noise=float(s.noise),
                structural_features=bool(s.structural_features),
                target_indicator=bool(s.get("target_indicator", False)),
                n_train=int(s.n_train),
                n_val=int(s.n_val),
                n_test=int(s.n_test),
                seed=seed,  # each sweep seed draws its own dataset (stats audit S5)
            )
        )
    if name in ("splice", "connectome"):
        mod = importlib.import_module(f"certgnn.substrates.{name}")
        cls_name = f"{name.capitalize()}Substrate"
        cls = getattr(mod, cls_name, None)
        if cls is not None and name == "splice":
            from certgnn.substrates.splice import SpliceConfig

            sc = cfg.get("splice", {}) or {}
            return cls(SpliceConfig(seed=seed, **{k: v for k, v in dict(sc).items()}))
        if cls is None:
            data_dir = pathlib.Path("data/raw")
            n_files = sum(1 for _ in data_dir.rglob("*")) if data_dir.exists() else 0
            raise SubstrateUnavailable(
                f"{name}: certgnn/substrates/{name}/ defines no {cls_name} (package is an empty stub) "
                f"and data/raw holds {n_files} files. The substrate has not been implemented; "
                "nothing was run for it."
            )
        return cls()
    raise ValueError(f"unknown substrate {name!r}")


# --------------------------------------------------------------- controls


def _strip_edges(data: list[Data]) -> list[Data]:
    out = []
    for d in data:
        e = copy.copy(d)
        e.edge_index = torch.empty(2, 0, dtype=torch.long)
        out.append(e)
    return out


def _rewire(data: list[Data], seed: int, swaps_per_edge: float) -> list[Data]:
    out = []
    for i, d in enumerate(data):
        e = copy.copy(d)
        try:
            e.edge_index = degree_preserving_rewire(
                d.edge_index,
                int(d.num_nodes),
                seed=seed * 100003 + i,
                swaps_per_edge=swaps_per_edge,
            )
        except ValueError:
            pass  # fewer than 2 edges: nothing to rewire
        out.append(e)
    return out


def _num_nodes(data: list[Data]) -> int | None:
    sizes = {int(d.num_nodes) for d in data}
    return sizes.pop() if len(sizes) == 1 else None


def run_substrate(
    name: str, sub: Substrate, cfg: Any, seeds: list[int], log: Any
) -> dict[str, Any]:
    t0 = time.time()
    mc, bc = cfg.model, cfg.tier0.bqn
    run_bqn = name in list(cfg.tier0.run_bqn_on)
    per_seed: dict[str, list[float]] = {m: [] for m in MODELS}
    task = None
    notes: list[str] = []
    in_dim = 0
    n_fixed: int | None = None
    n_tr = 0

    def gcn() -> torch.nn.Module:
        return TargetReadoutGCN(
            in_dim,
            int(mc.hidden),
            int(mc.depth),
            str(mc.readout),
            str(mc.get("arch", "gcn")),
        )

    def mlp() -> torch.nn.Module:
        return NodeFeatureMLP(in_dim, int(mc.hidden), int(mc.depth), str(mc.readout))

    def mlp_mean() -> torch.nn.Module:
        # Mean-pool over every node with the edges removed: a bag of windows.
        # The target-readout MLP above sees ONLY the readout node, which is a
        # severe handicap and makes the GNN look good for the wrong reason.
        # This arm is the one that tests whether topology adds anything beyond
        # the node features taken as an unordered set.
        return NodeFeatureMLP(in_dim, int(mc.hidden), int(mc.depth), "mean")

    for seed in seeds:
        sub = make_substrate(name, cfg, seed)
        splits = {s: sub.load(s) for s in ("train", "val", "test")}
        in_dim = int(splits["train"][0].x.size(1))
        n_fixed = _num_nodes(splits["train"] + splits["val"] + splits["test"])
        n_tr = len(splits["train"])
        kw: dict[str, Any] = dict(
            seed=seed,
            epochs=int(mc.epochs),
            lr=float(mc.lr),
            batch_size=int(mc.batch_size),
        )
        r = fit(gcn, splits["train"], splits["val"], splits["test"], **kw)
        task = r.task
        key = primary_metric(task)
        per_seed["gnn"].append(r.test_metrics[key])

        shuffled = {
            s: _rewire(splits[s], seed, float(cfg.tier0.rewire_swaps_per_edge))
            for s in splits
        }
        r = fit(gcn, shuffled["train"], shuffled["val"], shuffled["test"], **kw)
        per_seed["shuffled"].append(r.test_metrics[key])

        edgeless = {s: _strip_edges(splits[s]) for s in splits}
        r = fit(mlp, edgeless["train"], edgeless["val"], edgeless["test"], **kw)
        per_seed["mlp"].append(r.test_metrics[key])

        r = fit(mlp_mean, edgeless["train"], edgeless["val"], edgeless["test"], **kw)
        per_seed["mlp_mean"].append(r.test_metrics[key])

        if run_bqn:
            if n_fixed is None:
                notes.append(
                    "BQN skipped: graphs have varying node counts; BQN needs a fixed ROI count"
                )
                run_bqn = False
            else:
                r = fit(
                    lambda: BrainQuadraticNetwork(
                        n_fixed, int(bc.layers), int(bc.clusters), float(bc.dropout)
                    ),
                    splits["train"],
                    splits["val"],
                    splits["test"],
                    seed=seed,
                    epochs=int(bc.epochs),
                    lr=float(bc.lr),
                    batch_size=int(mc.batch_size),
                    weight_decay=float(bc.weight_decay),
                )
                per_seed["bqn"].append(r.test_metrics[key])
        log(
            f"  {name} seed {seed}: "
            + ", ".join(f"{m}={per_seed[m][-1]:.3f}" for m in MODELS if per_seed[m])
            + f"  [{time.time() - t0:.0f}s]"
        )

    nb = int(cfg.bootstrap_resamples)
    assert task is not None
    steps_per_epoch = -(-n_tr // int(mc.batch_size))
    tuning_budget = [
        {
            "model": m,
            "configs_tried": (
                3 if m in ("gnn", "shuffled") else 1
            ),  # backbone chosen among 3 (results/diagnostics/)
            "epochs": int(ep),
            "lr": float(lr),
            "gradient_steps": int(ep) * steps_per_epoch,
            "search_space": (
                "backbone chosen among gcn/sage_sum/gin in results/diagnostics/ (test-split R2 was inspected; disclosed there)"
                if m in ("gnn", "shuffled")
                else "none: one pre-specified configuration (configs/model/mlp.yaml, tier0.bqn)"
            ),
            "selection": "epoch with best validation metric; test split touched once",
        }
        for m, ep, lr in (
            ("gnn", mc.epochs, mc.lr),
            ("shuffled", mc.epochs, mc.lr),
            ("mlp", mc.epochs, mc.lr),
            ("bqn", bc.epochs, bc.lr),
        )
        if per_seed[m]
    ]
    summary = {m: mean_ci(v, n_boot=nb) for m, v in per_seed.items() if v}
    gnn = np.array(per_seed["gnn"])
    paired = {}
    for m in ("shuffled", "mlp", "mlp_mean", "bqn"):
        if per_seed[m]:
            paired[m] = mean_ci(
                (gnn - np.array(per_seed[m])).tolist(), n_boot=nb
            )  # GNN minus control
    return {
        "substrate": name,
        "status": "evaluated",
        "task": task,
        "metric": primary_metric(task),
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "per_seed": per_seed,
        "summary": summary,
        "paired_gnn_minus_control": paired,
        "bqn_run": bool(per_seed["bqn"]),
        "tuning_budget": tuning_budget,
        "notes": notes,
        "seconds": time.time() - t0,
    }


# ----------------------------------------------------------- interpretation


def interpret(res: dict[str, Any], is_candidate: bool) -> dict[str, Any]:
    """Apply the pre-stated rules to one evaluated substrate."""
    p = res["paired_gnn_minus_control"]
    findings: list[str] = []
    flags = {
        "mlp_matches_gnn": False,
        "topology_decorative": False,
        "bqn_beats_gnn": False,
    }

    d = p.get("mlp_mean", p["mlp"])
    if d["lo"] <= 0.0:
        flags["mlp_matches_gnn"] = True
        findings.append(
            f"Edge-free MLP matches the GNN (GNN−MLP = {d['mean']:+.3f} [{d['lo']:+.3f}, "
            f"{d['hi']:+.3f}] includes 0): the substrate is NOT a graph problem and cannot "
            "carry a topology paper. The comparison uses the mean-pool arm, which sees every "
            "node's features as an unordered set; the target-readout arm sees only the readout "
            "node and would flatter the GNN."
        )
    else:
        findings.append(
            f"GNN beats the edge-free MLP by {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]: the edges carry information the model uses."
        )

    d = p["shuffled"]
    if d["lo"] <= 0.0:
        flags["topology_decorative"] = True
        findings.append(
            f"Shuffled topology matches real topology (GNN−shuffled = {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] includes 0): "
            "the topology is decorative."
        )
    else:
        findings.append(
            f"Real topology beats degree-preserving shuffle by {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]: the specific wiring matters, not just the degree sequence."
        )

    if res["bqn_run"]:
        d = p["bqn"]
        if d["hi"] < 0.0:
            flags["bqn_beats_gnn"] = True
            findings.append(
                f"BQN beats the GNN (GNN−BQN = {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] < 0): "
                "the ICML 2025 critique applies directly; the connectome substrate is not viable as framed."
            )
        else:
            findings.append(
                f"BQN does not beat the GNN (GNN−BQN = {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}])."
            )

    passes = not (
        flags["mlp_matches_gnn"]
        or flags["topology_decorative"]
        or flags["bqn_beats_gnn"]
    )
    return {
        "flags": flags,
        "findings": findings,
        "passes_tier0": passes,
        "candidate": is_candidate,
    }


def recommend(
    results: dict[str, dict[str, Any]], candidates: list[str]
) -> dict[str, Any]:
    """Cross-substrate verdict on the SpliceCert vs ConnectomeCert question."""
    lines: list[str] = []
    evaluated = {
        n: r
        for n, r in results.items()
        if n in candidates and r["status"] == "evaluated"
    }
    unavailable = {
        n: r
        for n, r in results.items()
        if n in candidates and r["status"] != "evaluated"
    }
    passing = [n for n, r in evaluated.items() if r["interpretation"]["passes_tier0"]]
    failing = [
        n for n, r in evaluated.items() if not r["interpretation"]["passes_tier0"]
    ]

    for n, r in unavailable.items():
        lines.append(f"{n}: NOT EVALUATED — {r['reason']}")
    for n in failing:
        lines.append(
            f"{n}: FAILS tier 0 — "
            + " ".join(r_ for r_ in results[n]["interpretation"]["findings"])
        )
    for n in passing:
        lines.append(f"{n}: passes tier 0.")

    if len(passing) == 1 and not unavailable:
        rec = f"RECOMMEND {passing[0]}: it is the only candidate that passes tier 0."
    elif len(passing) >= 2:
        best = max(
            passing, key=lambda n: results[n]["paired_gnn_minus_control"]["mlp"]["mean"]
        )
        rec = f"BOTH PASS tier 0; {best} shows the larger GNN-over-MLP margin. Choose on gate G1/G2 evidence, not tier 0."
    elif len(passing) == 1 and unavailable:
        rec = (
            f"PROVISIONAL: {passing[0]} passes tier 0; {', '.join(unavailable)} could not be evaluated. "
            "The comparison is incomplete until the missing substrate is implemented and run."
        )
    elif evaluated and not passing and not unavailable:
        rec = "NEITHER SUBSTRATE PASSES tier 0. Do not proceed with a topology paper on either as framed."
    elif evaluated and not passing:
        rec = (
            f"NO CANDIDATE PASSES among those evaluated ({', '.join(evaluated)}); "
            f"{', '.join(unavailable)} could not be evaluated. No substrate can be adopted on current evidence."
        )
    else:
        rec = (
            "UNDECIDED: no candidate substrate could be evaluated. The SpliceCert vs ConnectomeCert question is "
            "blocked on implementing the substrates and obtaining their data, not on results."
        )
    return {
        "lines": lines,
        "recommendation": rec,
        "passing": passing,
        "failing": failing,
        "unavailable": list(unavailable),
    }


# ------------------------------------------------------------------ report


def make_table(
    results: dict[str, dict[str, Any]],
    rec: dict[str, Any],
    cfg: Any,
    meta: dict[str, Any],
) -> str:
    L = ["# Tier 0 controls (ABLATIONS 0.6, 0.7, 0.8)\n"]
    L.append(
        f"git {meta['git_sha']}{' (DIRTY)' if meta['git_dirty'] else ''}, config {meta['config_hash']}, "
        f"{len(resolve_seeds(cfg))} seed(s), mean [95% bootstrap CI]. Test-split performance; model selection on validation only.\n"
    )
    for name, r in results.items():
        role = (
            "candidate"
            if name in list(cfg.tier0.candidates)
            else "pipeline validation (not a candidate)"
        )
        L.append(f"## {name} — {role}\n")
        if r["status"] != "evaluated":
            L.append(f"**NOT EVALUATED.** {r['reason']}\n")
            continue
        L.append(f"task: {r['task']}, metric: {r['metric']}, seeds: {r['n_seeds']}\n")
        L.append("| model | row | test metric | paired GNN − model |")
        L.append("|---|---|---|---|")
        rows = {
            "gnn": "reference",
            "shuffled": "0.6 degree-preserving shuffle",
            "mlp": "0.7 edge-free MLP",
            "bqn": "0.8 Hadamard/BQN",
        }
        for m in MODELS:
            if m in r["summary"]:
                pd = r["paired_gnn_minus_control"].get(m)
                L.append(
                    f"| {m} | {rows[m]} | {fmt_ci(r['summary'][m])} | {fmt_ci(pd) if pd else '—'} |"
                )
        if not r["bqn_run"]:
            L.append(
                f"| bqn | 0.8 Hadamard/BQN | not run ({'brain-substrate row' if name not in list(cfg.tier0.run_bqn_on) else 'see notes'}) | — |"
            )
        L.append("")
        for f in r["interpretation"]["findings"]:
            L.append(f"- {f}")
        for n_ in r["notes"]:
            L.append(f"- note: {n_}")
        L.append(
            f"\n**Tier 0: {'PASS' if r['interpretation']['passes_tier0'] else 'FAIL'}**\n"
        )
    L.append("## Verdict: SpliceCert vs ConnectomeCert\n")
    for line in rec["lines"]:
        L.append(f"- {line}")
    L.append(f"\n**{rec['recommendation']}**\n")
    return "\n".join(L)


# -------------------------------------------------------------------- main


def aggregate_manifest(manifest_path: str, cfg: Any) -> int:
    """Rebuild the cross-seed verdict from the per-seed run directories of a sweep."""
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    by_id: dict[str, pathlib.Path] = {}
    for d in pathlib.Path(cfg.output.runs).iterdir():
        mp = d / "meta.json"
        if mp.exists():
            rid = json.loads(mp.read_text()).get("run_id")
            if rid:
                by_id[rid] = d
    missing = [
        r["run_id"]
        for r in manifest["runs"]
        if r["run_id"] not in by_id
        or not (by_id[r["run_id"]] / "results.json").exists()
    ]
    if missing:
        raise SystemExit(
            f"AGGREGATION REFUSED: {len(missing)} planned run(s) have no results: {missing}"
        )
    merged: dict[str, dict[str, Any]] = {}
    for planned in manifest["runs"]:
        d = by_id[planned["run_id"]]
        res = json.loads((d / "results.json").read_text())["results"]
        for name, r in res.items():
            if r["status"] != "evaluated":
                merged.setdefault(name, r)
                continue
            m = merged.setdefault(
                name,
                {
                    **r,
                    "per_seed": {k: [] for k in MODELS},
                    "seeds": [],
                    "tuning_budget": r["tuning_budget"],
                },
            )
            for k in MODELS:
                m["per_seed"][k].extend(r["per_seed"].get(k, []))
            m["seeds"].extend(r["seeds"])
    nb = int(cfg.bootstrap_resamples)
    for name, r in merged.items():
        if r["status"] != "evaluated":
            continue
        gnn = np.array(r["per_seed"]["gnn"])
        r["n_seeds"] = len(r["seeds"])
        r["summary"] = {m: mean_ci(v, n_boot=nb) for m, v in r["per_seed"].items() if v}
        r["paired_gnn_minus_control"] = {
            m: mean_ci((gnn - np.array(r["per_seed"][m])).tolist(), n_boot=nb)
            for m in ("shuffled", "mlp", "mlp_mean", "bqn")
            if r["per_seed"][m]
        }
        r["bqn_run"] = bool(r["per_seed"]["bqn"])
        r["interpretation"] = interpret(r, name in list(cfg.tier0.candidates))
    rec = recommend(merged, list(cfg.tier0.candidates))
    meta = {
        "git_sha": manifest["id"].split("_")[-1],
        "git_dirty": manifest["git_dirty"],
        "config_hash": "sweep:" + manifest["id"],
    }
    table = make_table(merged, rec, cfg, meta)
    tab_dir = pathlib.Path(cfg.output.tables)
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "tier0_controls.md").write_text(table)
    out = pathlib.Path("results") / "tier0_verdict.md"
    out.write_text(
        f"sweep: {manifest['id']}\nruns: {len(manifest['runs'])}\n\n"
        + "## Verdict\n\n"
        + "\n".join(f"- {line}" for line in rec["lines"])
        + f"\n\n**{rec['recommendation']}**\n"
    )
    (pathlib.Path("results/sweeps") / f"{manifest['id']}.aggregate.json").write_text(
        json.dumps(to_jsonable({"results": merged, "recommendation": rec}), indent=1)
    )
    print(table)
    print(f"table: {tab_dir / 'tier0_controls.md'}\nverdict: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run on a dirty tree (recorded in meta.json)",
    )
    ap.add_argument(
        "--aggregate",
        default=None,
        metavar="MANIFEST",
        help="rebuild the cross-seed verdict from a sweep manifest instead of running",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.aggregate:
        return aggregate_manifest(args.aggregate, cfg)
    configure_torch(cfg)
    run = make_run_dir(cfg, "tier0_controls", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run dir: {run}")
    results: dict[str, dict[str, Any]] = {}
    for name in list(cfg.tier0.substrates):
        try:
            sub = make_substrate(name, cfg)
        except SubstrateUnavailable as e:
            log(f"{name}: NOT EVALUATED — {e}")
            results[name] = {
                "substrate": name,
                "status": "unavailable",
                "reason": str(e),
            }
            continue
        except Exception:
            tb = traceback.format_exc()
            log(f"{name}: NOT EVALUATED — substrate construction raised:\n{tb}")
            results[name] = {
                "substrate": name,
                "status": "error",
                "reason": f"substrate construction raised: {tb.strip().splitlines()[-1]}",
            }
            continue
        seeds = resolve_seeds(cfg)
        log(f"{name}: running seeds {seeds}")
        r = run_substrate(name, sub, cfg, seeds, log)
        r["interpretation"] = interpret(r, name in list(cfg.tier0.candidates))
        results[name] = r
        (run / f"{name}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    rec = recommend(results, list(cfg.tier0.candidates))
    budget = [
        dict(e, substrate=name)
        for name, r in results.items()
        if r["status"] == "evaluated"
        for e in r["tuning_budget"]
    ]
    write_tuning_budget(run, budget)
    per_seed_values = {
        f"{name}/{m}": dict(zip(map(str, r["seeds"]), vals))
        for name, r in results.items()
        if r["status"] == "evaluated"
        for m, vals in r["per_seed"].items()
        if vals
    }
    (run / "per_seed_values.json").write_text(json.dumps(per_seed_values, indent=1))
    table = make_table(results, rec, cfg, meta)
    (run / "table.md").write_text(table)
    if cfg.get("sweep") is None:
        # a direct multi-seed run publishes its table; a per-seed sweep run does not,
        # the cross-seed aggregate (--aggregate MANIFEST) publishes instead
        tab_dir = pathlib.Path(cfg.output.tables)
        tab_dir.mkdir(parents=True, exist_ok=True)
        (tab_dir / "tier0_controls.md").write_text(table)
    (run / "results.json").write_text(
        json.dumps(to_jsonable({"results": results, "recommendation": rec}), indent=1)
    )
    (run / "log.txt").write_text("\n".join(log_lines))
    print("\n" + table)
    print(
        f"run: {run}"
        + (
            ""
            if cfg.get("sweep") is not None
            else f"\ntable: {pathlib.Path(cfg.output.tables) / 'tier0_controls.md'}"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
