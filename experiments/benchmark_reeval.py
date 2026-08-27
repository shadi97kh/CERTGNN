"""Do explainer RANKINGS change between probability and latent space? (ABLATIONS C.5-C.8)

THE QUESTION

Theorem 2 says probability-space faithfulness inherits the link's Jacobian and
so disagrees with latent-space faithfulness across instances at different
operating points. Gate G2 shows that on a synthetic substrate. What it cannot
show is whether the confound *matters*: if every explainer moves by the same
amount, the ranking is unchanged and the field loses nothing by ignoring it.

So the reported quantity here is not a gap, it is a **ranking**. Explainers are
scored on published benchmarks in both spaces, ranked in each, and the two
rankings compared by Kendall tau. If tau is at or near 1 everywhere, Theorem 2
is real and inconsequential for practice, and that is what the paper must say.

WHAT WOULD MAKE A NULL RESULT UNINTERPRETABLE, AND HOW IT IS GUARDED

A ranking can only move if ``p(1-p)`` varies across instances. A model whose
predictions sit near ``p = 0.5`` has an almost constant Jacobian and cannot
produce a ranking change whatever Theorem 2 says. ``--precheck`` runs that
measurement alone (ABLATIONS C.8), and the full run reports the same statistics
beside every tau, so a tau of 1 can be attributed to a narrow operating-point
distribution rather than to the theorem being wrong.

BINARY VERSUS MULTICLASS

Theorem 2 is written for the logistic link, whose Jacobian is the scalar
``p(1-p)``. MUTAG, BA-2Motifs and Tree-Cycles are binary and test it directly.
BA-Shapes and BA-Community are multiclass: the output map is a softmax, whose
Jacobian is ``diag(p) - p pT``, and the scalar argument does not carry over.
Those datasets are scored on the explained class's logit against its softmax
probability and are reported as an extension (ABLATIONS C.7), never folded into
the binary result.

STATUS: the ABLATIONS rows this implements are `status: candidate` and are not
approved. Running the full matrix needs the PI's sign-off on the dataset and
explainer list; ``--precheck`` is the cheap part that decides whether the rest
is worth it.

Usage:
    python -m experiments.benchmark_reeval --config configs/base.yaml --precheck
    python -m experiments.benchmark_reeval --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections.abc import Callable
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from scipy.stats import kendalltau  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from certgnn.eval.faithfulness import (  # noqa: E402
    aopc,
    comprehensiveness,
    normalized_aopc,
    sufficiency,
)
from certgnn.models.gnn import TargetReadoutGCN  # noqa: E402
from experiments._common import (
    configure_torch,  # noqa: E402
    fmt_ci,
    load_config,
    make_run_dir,
    mean_ci,
    resolve_seeds,
    to_jsonable,
    write_tuning_budget,
)

METRICS = ("comprehensiveness", "sufficiency", "aopc", "normalized_aopc")


# ------------------------------------------------------------------ data


def load_benchmark(name: str, root: str, seed: int) -> dict[str, Any]:
    """One benchmark as ``{data, task, n_classes, level}``.

    Rebuilt from ``torch_geometric.datasets`` rather than from GraphXAI, which
    is not installed and is not on PyPI. Raises ``NotImplementedError`` with the
    reason for any name that cannot be built from what is installed -- an
    unavailable dataset is reported as unavailable, never silently swapped for a
    neighbour (ABLATIONS rule 3).
    """
    import torch_geometric.datasets as D
    from torch_geometric.datasets.graph_generator import BAGraph, TreeGraph
    from torch_geometric.datasets.motif_generator import CycleMotif, HouseMotif

    if name == "mutag":
        ds = D.TUDataset(root=f"{root}/TUDataset", name="MUTAG")
        return {"data": list(ds), "level": "graph", "n_classes": int(ds.num_classes)}
    if name == "ba_2motifs":
        ds = D.BA2MotifDataset(root=f"{root}/BA2Motif")
        return {"data": list(ds), "level": "graph", "n_classes": int(ds.num_classes)}
    if name == "tree_cycles":
        ds = D.ExplainerDataset(
            graph_generator=TreeGraph(depth=8),
            motif_generator=CycleMotif(6),
            num_motifs=60,
            num_graphs=1,
        )
        return {"data": list(ds), "level": "node", "n_classes": 2}
    if name == "ba_shapes":
        ds = D.ExplainerDataset(
            graph_generator=BAGraph(num_nodes=300, num_edges=5),
            motif_generator=HouseMotif(),
            num_motifs=80,
            num_graphs=1,
        )
        return {"data": list(ds), "level": "node", "n_classes": 4}
    if name == "ba_community":
        raise NotImplementedError(
            "BA-Community is two feature-distinguished BA-Shapes graphs joined at "
            "random; torch_geometric ships no generator for it and GraphXAI is not "
            "installed. It needs either an approved GraphXAI pin or a generator "
            "written and reviewed here. Reported as unavailable rather than "
            "substituted."
        )
    raise NotImplementedError(f"unknown benchmark {name!r}")


def to_instances(bundle: dict[str, Any], cfg: Any, seed: int) -> list[Any]:
    """Per-explanation instances, each carrying the node whose output is explained.

    Graph-level datasets explain the pooled readout, so ``target_idx`` is a
    formality (node 0) and the readout is ``mean``. Node-level datasets explain
    one node each, so a single large graph becomes many instances that share it.
    """
    rng = np.random.default_rng(seed)
    n_max = int(cfg.benchmark.max_instances)
    if bundle["level"] == "graph":
        data = bundle["data"]
        idx = rng.permutation(len(data))[:n_max]
        out = []
        for i in idx:
            d = data[int(i)].clone()
            d.target_idx = torch.tensor(0)
            out.append(d)
        return out
    big = bundle["data"][0]
    nodes = rng.permutation(int(big.num_nodes))[:n_max]
    out = []
    for v in nodes:
        d = big.clone()
        d.target_idx = torch.tensor(int(v))
        d.y = big.y[int(v)].reshape(1)
        out.append(d)
    return out


# ----------------------------------------------------------------- model


def train_classifier(
    instances: list[Any], bundle: dict[str, Any], cfg: Any, seed: int
) -> tuple[torch.nn.Module, dict[str, float]]:
    """A standard GNN classifier, trained with cross-entropy on logits.

    The head emits logits and nothing here applies a sigmoid or softmax: that
    conversion happens only inside the metrics, which is the whole point
    (CLAUDE.md, and ``certgnn.eval.faithfulness``).
    """
    mc, bc = cfg.model, cfg.benchmark
    torch.manual_seed(seed)
    n_cls = int(bundle["n_classes"])
    model = TargetReadoutGCN(
        int(instances[0].x.size(1)),
        int(mc.hidden),
        int(mc.depth),
        "mean" if bundle["level"] == "graph" else "target",
        str(mc.get("arch", "gcn")),
        out_dim=1 if n_cls == 2 else n_cls,
    )
    n = len(instances)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    cut = int(0.8 * n)
    tr = [instances[i] for i in perm[:cut]]
    te = [instances[i] for i in perm[cut:]]
    opt = torch.optim.Adam(model.parameters(), lr=float(mc.lr))
    binary = n_cls == 2
    loss_fn = torch.nn.BCEWithLogitsLoss() if binary else torch.nn.CrossEntropyLoss()
    loader = DataLoader(
        tr,
        batch_size=int(bc.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(int(bc.epochs)):
        model.train()
        for batch in loader:
            opt.zero_grad()
            out = model(
                batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx
            )
            y = batch.y.reshape(-1)
            loss = loss_fn(out, y.float() if binary else y.long())
            loss.backward()
            opt.step()
    return model, evaluate(model, te, binary)


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: list[Any], binary: bool) -> dict[str, float]:
    model.eval()
    correct = total = 0
    for batch in DataLoader(data, batch_size=128, shuffle=False):
        out = model(batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx)
        y = batch.y.reshape(-1)
        pred = (out > 0).long() if binary else out.argmax(-1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return {"test_accuracy": correct / max(total, 1), "n_test": total}


def make_latent_fn(model: torch.nn.Module, d: Any, cls: int) -> Callable:
    """``latent_fn(mask) -> 0-d tensor``: the explained class's logit.

    Binary models emit one logit, which is already the log-odds of class 1;
    multiclass models emit ``k`` and the explained class's logit is used, whose
    softmax is not a logistic link (ABLATIONS C.7).
    """
    ptr = torch.tensor([0, int(d.num_nodes)])
    batch = torch.zeros(int(d.num_nodes), dtype=torch.long)
    tgt = d.target_idx.reshape(1)

    def latent_fn(mask: torch.Tensor) -> torch.Tensor:
        out = model(d.x, d.edge_index, batch, ptr, tgt, mask.to(torch.float32))
        return out.reshape(-1)[0] if out.dim() == 1 else out.reshape(-1)[cls]

    return latent_fn


# ------------------------------------------------------------- explainers


def explainer_masks(
    name: str,
    model: torch.nn.Module,
    instances: list[Any],
    bundle: dict[str, Any],
    cfg: Any,
    seed: int,
) -> list[torch.Tensor]:
    """A soft node mask per instance from one explainer.

    Every mask is left soft: rule 6 of ABLATIONS and CLAUDE.md forbid a hard
    binary mask outside an explicit negative control. The target node is pinned
    to 1 because ``certgnn.eval.faithfulness`` refuses to score a masked target.
    """
    from torch_geometric.explain import Explainer, ModelConfig
    from torch_geometric.explain.algorithm import (
        DummyExplainer,
        GNNExplainer,
        GraphMaskExplainer,
        PGExplainer,
    )

    torch.manual_seed(seed)
    epochs = int(cfg.benchmark.explainer_epochs)
    binary = int(bundle["n_classes"]) == 2
    algos = {
        "gnn_explainer": lambda: GNNExplainer(epochs=epochs),
        "graphmask": lambda: GraphMaskExplainer(int(cfg.model.depth), epochs=epochs),
        "pg_explainer": lambda: PGExplainer(epochs=epochs),
        "dummy": DummyExplainer,
    }
    if name not in algos:
        raise NotImplementedError(
            f"explainer {name!r} is not wired. torch_geometric.explain ships "
            "AttentionExplainer (needs an attention backbone this paper does not "
            "use) and CaptumExplainer (needs `captum`, not installed). Both are "
            "flagged in ABLATIONS under candidate additions."
        )
    model_config = ModelConfig(
        mode="binary_classification" if binary else "multiclass_classification",
        task_level=bundle["level"],
        return_type="raw",
    )
    explainer = Explainer(
        model=_PyGAdapter(model),
        algorithm=algos[name](),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=model_config,
    )
    masks = []
    for d in instances:
        kw: dict[str, Any] = {}
        if bundle["level"] == "node":
            kw["index"] = int(d.target_idx)
        try:
            exp = explainer(d.x, d.edge_index, **kw)
            m = exp.get("node_mask")
            m = (
                torch.ones(int(d.num_nodes))
                if m is None
                else m.reshape(int(d.num_nodes), -1).mean(-1)
            )
        except Exception as err:  # noqa: BLE001 - reported, never silently skipped
            raise RuntimeError(f"{name} failed on an instance: {err}") from err
        m = _to_soft(m)
        m[int(d.target_idx)] = 1.0
        masks.append(m.to(torch.float64))
    return masks


def _to_soft(m: torch.Tensor) -> torch.Tensor:
    """Rescale an explainer's raw scores into ``(0, 1)`` without thresholding.

    Min-max onto ``[0.02, 0.98]``: keeping both ends strictly inside the open
    interval means no node is ever fully deleted or fully forced, which is what
    "soft mask" is required to mean here.
    """
    m = m.detach().reshape(-1).to(torch.float64)
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-12:
        return torch.full_like(m, 0.5)
    return 0.02 + 0.96 * (m - lo) / (hi - lo)


class _PyGAdapter(torch.nn.Module):
    """Adapts ``TargetReadoutGCN`` to the ``(x, edge_index)`` signature PyG's
    Explainer calls, filling in the single-graph batch bookkeeping."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        n = x.size(0)
        batch = torch.zeros(n, dtype=torch.long)
        ptr = torch.tensor([0, n])
        out = self.model(x, edge_index, batch, ptr, torch.tensor([0]))
        return out.reshape(1, -1) if out.dim() > 1 else out.reshape(1, 1)


# ---------------------------------------------------------------- scoring


def score_instances(
    model: torch.nn.Module, instances: list[Any], masks: list[torch.Tensor], cfg: Any
) -> dict[str, dict[str, list[float]]]:
    """Every metric, both spaces, per instance. Also the operating point."""
    k = cfg.benchmark.get("aopc_k")
    out: dict[str, dict[str, list[float]]] = {
        m: {"probability": [], "latent": []} for m in METRICS
    }
    out["operating_point"] = {"probability": [], "latent": []}
    fns = {
        "comprehensiveness": comprehensiveness,
        "sufficiency": sufficiency,
        "aopc": lambda f, m, t: aopc(f, m, t, k=k),
        "normalized_aopc": lambda f, m, t: normalized_aopc(f, m, t, k=k),
    }
    for d, mask in zip(instances, masks, strict=True):
        cls = int(d.y.reshape(-1)[0])
        fn = make_latent_fn(model, d, cls)
        t = int(d.target_idx)
        with torch.no_grad():
            z = float(fn(torch.ones(int(d.num_nodes), dtype=torch.float64)))
        p = 1.0 / (1.0 + np.exp(-z))
        out["operating_point"]["probability"].append(p)
        out["operating_point"]["latent"].append(z)
        for name, f in fns.items():
            s = f(fn, mask, t)
            out[name]["probability"].append(s.probability)
            out[name]["latent"].append(s.latent)
    return out


def jacobian_stats(p: np.ndarray) -> dict[str, float]:
    """Spread of the logistic Jacobian at the operating points.

    ``spread_ratio`` is the 95th over the 5th percentile of ``p(1-p)``: the
    factor by which the same latent change is inflated or shrunk between the
    most and least sensitive instances. It bounds how large a ranking effect
    can be, so it is reported next to every tau.
    """
    j = p * (1.0 - p)
    lo, hi = float(np.quantile(j, 0.05)), float(np.quantile(j, 0.95))
    return {
        "jacobian_p05": lo,
        "jacobian_p95": hi,
        "jacobian_spread_ratio": float(hi / lo) if lo > 0 else float("inf"),
        "saturated_fraction": float(np.mean((p < 0.01) | (p > 0.99))),
        "operating_point_mean": float(p.mean()),
        "operating_point_std": float(p.std()),
    }


def rankings_and_tau(per_explainer: dict[str, dict], metric: str) -> dict[str, Any]:
    """Rank explainers by mean score in each space, then compare the rankings.

    Kendall tau over four or five explainers is coarse by construction -- it can
    only take a handful of values -- so the per-explainer means are reported
    beside it and tau is never quoted alone.
    """
    names = sorted(per_explainer)
    means = {
        space: np.array([np.mean(per_explainer[n][metric][space]) for n in names])
        for space in ("probability", "latent")
    }
    higher_is_better = metric != "sufficiency"
    order = {
        space: [names[i] for i in np.argsort(-v if higher_is_better else v)]
        for space, v in means.items()
    }
    tau, p = kendalltau(means["probability"], means["latent"])
    return {
        "explainers": names,
        "mean_probability": {
            n: float(v) for n, v in zip(names, means["probability"], strict=True)
        },
        "mean_latent": {
            n: float(v) for n, v in zip(names, means["latent"], strict=True)
        },
        "ranking_probability": order["probability"],
        "ranking_latent": order["latent"],
        "kendall_tau": float(tau) if np.isfinite(tau) else float("nan"),
        "kendall_p": float(p) if np.isfinite(p) else float("nan"),
        "ranking_changed": order["probability"] != order["latent"],
    }


# ------------------------------------------------------------------- run


def run_precheck(cfg: Any, seed: int) -> dict[str, Any]:
    """ABLATIONS C.8: operating-point distributions only. No explainers."""
    out: dict[str, Any] = {}
    for name in list(cfg.benchmark.datasets):
        try:
            bundle = load_benchmark(name, str(cfg.benchmark.root), seed)
        except NotImplementedError as err:
            out[name] = {"available": False, "reason": str(err)}
            continue
        inst = to_instances(bundle, cfg, seed)
        model, metrics = train_classifier(inst, bundle, cfg, seed)
        z = np.array(
            [
                float(
                    make_latent_fn(model, d, int(d.y.reshape(-1)[0]))(
                        torch.ones(int(d.num_nodes), dtype=torch.float64)
                    )
                )
                for d in inst
            ]
        )
        p = 1.0 / (1.0 + np.exp(-z))
        out[name] = {
            "available": True,
            "level": bundle["level"],
            "n_classes": bundle["n_classes"],
            "n_instances": len(inst),
            **metrics,
            **jacobian_stats(p),
        }
    return out


def run_dataset(cfg: Any, name: str, seed: int) -> dict[str, Any]:
    bundle = load_benchmark(name, str(cfg.benchmark.root), seed)
    inst = to_instances(bundle, cfg, seed)
    model, metrics = train_classifier(inst, bundle, cfg, seed)
    per_explainer: dict[str, dict] = {}
    for ex in list(cfg.benchmark.explainers):
        masks = explainer_masks(ex, model, inst, bundle, cfg, seed)
        per_explainer[ex] = score_instances(model, inst, masks, cfg)
    p = np.array(
        per_explainer[list(per_explainer)[0]]["operating_point"]["probability"]
    )
    ranked = {m: rankings_and_tau(per_explainer, m) for m in METRICS}
    return {
        "dataset": name,
        "level": bundle["level"],
        "n_classes": bundle["n_classes"],
        "binary": bundle["n_classes"] == 2,
        "n_instances": len(inst),
        **metrics,
        **jacobian_stats(p),
        "rankings": ranked,
        "per_explainer_means": {
            ex: {
                m: {s: float(np.mean(v[m][s])) for s in ("probability", "latent")}
                for m in METRICS
            }
            for ex, v in per_explainer.items()
        },
    }


# --------------------------------------------------------------- outputs


def make_table(agg: dict[str, Any], meta: dict, precheck: bool) -> str:
    L = [
        "# Benchmark re-evaluation: do explainer rankings change between spaces?\n",
        f"git SHA `{meta['git_sha']}` (dirty: {meta['git_dirty']}), config hash "
        f"`{meta['config_hash']}`, {agg['n_seeds']} seeds, mean [95% bootstrap CI].\n",
    ]
    if precheck:
        L += [
            "ABLATIONS C.8 only: operating points, no explainers. A ranking change "
            "needs the logistic Jacobian to vary across instances; the spread ratio "
            "below bounds how large any such change can be.\n",
            "| dataset | level | classes | test acc | mean p | Jacobian p95/p05 | saturated |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, d in agg["datasets"].items():
            if not d.get("available", True):
                L.append(f"| {name} | — | — | UNAVAILABLE | | | |")
                continue
            L.append(
                f"| {name} | {d['level']} | {d['n_classes']} | {fmt_ci(d['test_accuracy'])} | "
                f"{fmt_ci(d['operating_point_mean'])} | {fmt_ci(d['jacobian_spread_ratio'])} | "
                f"{fmt_ci(d['saturated_fraction'])} |"
            )
        L.append(
            "\nA spread ratio near 1 means the link cannot move a ranking on that "
            "dataset, whatever Theorem 2 says, and a tau of 1 there is not evidence "
            "against the theorem.\n"
        )
        return "\n".join(L)

    L += [
        "| dataset | binary | metric | tau(prob, latent) | ranking changed | Jacobian p95/p05 |",
        "|---|---|---|---|---|---|",
    ]
    for name, d in agg["datasets"].items():
        for m in METRICS:
            r = d["rankings"][m]
            L.append(
                f"| {name} | {d['binary']} | {m} | {fmt_ci(r['kendall_tau'])} | "
                f"{fmt_ci(r['ranking_changed'])} | {fmt_ci(d['jacobian_spread_ratio'])} |"
            )
    L.append(
        "\nRows with `binary = False` use a softmax output map, whose Jacobian is "
        "not the scalar p(1-p); they are an extension (ABLATIONS C.7) and are not "
        "evidence for Theorem 2 as written.\n"
    )
    L.append(f"\n**{agg['statement']}**\n")
    return "\n".join(L)


def summarise(agg: dict[str, Any]) -> str:
    binary = [d for d in agg["datasets"].values() if d.get("binary")]
    if not binary:
        return "No binary dataset completed, so Theorem 2's practical consequence is untested."
    changed = [
        n
        for n, d in agg["datasets"].items()
        if d.get("binary")
        and any(d["rankings"][m]["ranking_changed"]["mean"] > 0 for m in METRICS)
    ]
    if changed:
        return (
            "Explainer rankings differ between probability and latent space on "
            f"{', '.join(changed)}: the link confound changes the answer the "
            "benchmark gives, not just its magnitude."
        )
    return (
        "Rankings agree in both spaces on every binary benchmark. Theorem 2 is "
        "real but inconsequential for explainer comparison at these operating "
        "points, and the paper must report that as this section's headline."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    ap.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree")
    ap.add_argument(
        "--precheck",
        action="store_true",
        help="ABLATIONS C.8 only: operating-point distributions, no explainers",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    configure_torch(cfg)
    run = make_run_dir(cfg, "benchmark_reeval", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    print(f"run dir: {run}")

    B = int(cfg.bootstrap_resamples)
    per_seed: list[dict[str, Any]] = []
    t0 = time.time()
    for seed in resolve_seeds(cfg):
        if args.precheck:
            r = {"seed": seed, "datasets": run_precheck(cfg, seed)}
        else:
            ds: dict[str, Any] = {}
            for name in list(cfg.benchmark.datasets):
                try:
                    ds[name] = run_dataset(cfg, name, seed)
                except NotImplementedError as err:
                    ds[name] = {"available": False, "reason": str(err)}
            r = {"seed": seed, "datasets": ds}
        per_seed.append(r)
        (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))
        print(f"seed {seed} done [{time.time() - t0:.0f}s]", flush=True)

    names = list(cfg.benchmark.datasets)
    agg: dict[str, Any] = {"n_seeds": len(per_seed), "datasets": {}}
    for name in names:
        rows = [
            r["datasets"][name]
            for r in per_seed
            if r["datasets"][name].get("available", True)
        ]
        if not rows:
            agg["datasets"][name] = {
                "available": False,
                "reason": per_seed[0]["datasets"][name].get("reason", ""),
            }
            continue
        d: dict[str, Any] = {
            "available": True,
            "level": rows[0]["level"],
            "n_classes": rows[0]["n_classes"],
            "binary": rows[0]["n_classes"] == 2,
        }
        for k in (
            "test_accuracy",
            "jacobian_spread_ratio",
            "saturated_fraction",
            "operating_point_mean",
            "operating_point_std",
        ):
            d[k] = mean_ci([x[k] for x in rows], n_boot=B)
        if not args.precheck:
            d["rankings"] = {
                m: {
                    "kendall_tau": mean_ci(
                        [x["rankings"][m]["kendall_tau"] for x in rows], n_boot=B
                    ),
                    "ranking_changed": mean_ci(
                        [float(x["rankings"][m]["ranking_changed"]) for x in rows],
                        n_boot=B,
                    ),
                    "ranking_probability": rows[0]["rankings"][m][
                        "ranking_probability"
                    ],
                    "ranking_latent": rows[0]["rankings"][m]["ranking_latent"],
                }
                for m in METRICS
            }
        agg["datasets"][name] = d
    agg["statement"] = "precheck only" if args.precheck else summarise(agg)

    table = make_table(agg, meta, args.precheck)
    stem = "benchmark_precheck" if args.precheck else "benchmark_reeval"
    tab_dir = pathlib.Path(cfg.output.tables)
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / f"{stem}.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(
        json.dumps(to_jsonable({"aggregate": agg}), indent=1)
    )
    write_tuning_budget(
        run,
        [
            {
                "model": str(cfg.model.get("arch", "gcn")),
                "configs_tried": 1,
                "epochs": int(cfg.benchmark.epochs),
                "gradient_steps": "epochs x ceil(0.8*max_instances/batch_size) per dataset per seed",
                "search_space": "none: configs/model used as is, no benchmark-specific tuning",
                "selection": "no model selection; fixed epoch budget, reported test accuracy",
            }
        ],
    )
    print(table)
    print(f"\ntable: {tab_dir / stem}.md\nrun: {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
