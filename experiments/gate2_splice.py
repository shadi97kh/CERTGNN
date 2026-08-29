"""ABLATIONS 1.9 on the REAL splice substrate, scored by a trained model.

Gate 2 as recorded rests on `gate2_link.py`, which takes its fidelity
scores from the synthetic generator's own latent function. The data are
made by pushing a latent effect through a sigmoid and the score is read
back off that same sigmoid, so the result cannot separate "the link
matters for a model" from "we generated through a link". `gate2_model.py`
removed half of that circularity by swapping in a trained GNN, but stayed
on the synthetic substrate. This removes the rest: real measurements, a
real baseline rate, a trained model, and no oracle anywhere.

Substrate: MFASS (Cheung et al. 2019) through `certgnn.substrates.splice`.
2,339 exons, splits grouped by `ensembl_id`. The measured phenotype is a
splicing index in [0, 1]; the baseline rate is the natural (unmutated)
exon's index, which is a property of the exon rather than an artefact of
a generator.

Design decisions, stated because each could bias the answer:

- The model is trained on the LATENT scale, `logit(index)`, per CLAUDE.md.
  Indices at exactly 0 or 1 have no finite logit, so they are clamped to
  `[eps, 1-eps]` and the clamped fraction is reported. Training on the
  probability directly would put the head in probability space and make a
  "latent gap" a probability gap with a latent label.
- Explanations are the Theorem 4 resistance-ball soft mask, the same
  construction `gate2_link` scores, so only the scorer differs.
- Scores come from the model under that soft mask, threaded through every
  message-passing layer, never from a generator.
- Strata are quantiles of `|logit(baseline_rate)|`, as in row 1.9.
- The sanity controls that CLAUDE.md requires ship with the result:
  model-parameter randomization and label randomization.

This run does not re-decide G2. G2's verdict rests on its pre-registered
thresholds and its own run; this is a separate entry.

Usage:
    python -m experiments.gate2_splice --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch

from certgnn.certify.conformal import (
    conditional_coverage_gap,
    empirical_coverage,
    split_conformal_quantile,
)
from certgnn.certify.link import stratify_by_baseline
from certgnn.eval.sanity import degenerate_explanation_check, randomize_model_parameters
from certgnn.explain.masks import resistance_ball_init
from certgnn.models import TargetReadoutGCN, fit
from certgnn.substrates.splice import SpliceConfig, SpliceSubstrate
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

EPS = 1e-3


def to_latent(y: torch.Tensor) -> tuple[torch.Tensor, float]:
    """logit of a rate, with the clamped fraction reported rather than hidden."""
    p = y.reshape(-1).double()
    clamped = float(((p <= EPS) | (p >= 1 - EPS)).float().mean())
    p = p.clamp(EPS, 1 - EPS)
    return torch.log(p / (1 - p)).float(), clamped


def relabel_latent(data: list) -> tuple[list, float]:
    out, fracs = [], []
    for d in data:
        e = copy.copy(d)
        z, c = to_latent(d.y)
        e.y = z
        fracs.append(c)
        out.append(e)
    return out, float(np.mean(fracs))


def explanation_masks(
    sub: SpliceSubstrate, data: list, frac: float
) -> list[torch.Tensor]:
    """Theorem 4 resistance-ball soft masks, one per graph."""
    masks = []
    for d in data:
        g = sub.to_networkx(d)
        k = max(1, int(round(frac * (int(d.num_nodes) - 1))))
        masks.append(
            resistance_ball_init(g, sub.target_node(d), ball_size=k, exact=False)
        )
    return masks


@torch.no_grad()
def model_scores(
    model: torch.nn.Module, data: list, masks: list[torch.Tensor], batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """|z_full - z_masked| in latent space and in probability space."""
    from torch_geometric.loader import DataLoader

    model.eval()
    lat, prob = [], []
    start = 0
    for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
        n = int(batch.num_graphs)
        m = torch.cat([masks[i].float() for i in range(start, start + n)])
        start += n
        z_full = model(
            batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx
        )
        z_mask = model(
            batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx, m
        )
        lat.append((z_full - z_mask).abs().cpu())
        prob.append((torch.sigmoid(z_full) - torch.sigmoid(z_mask)).abs().cpu())
    return torch.cat(lat).numpy(), torch.cat(prob).numpy()


def coverage_profile(
    cal_lat: np.ndarray,
    cal_prob: np.ndarray,
    te_lat: np.ndarray,
    te_prob: np.ndarray,
    te_p0: np.ndarray,
    cfg: Any,
) -> dict[str, Any]:
    alpha, n_strata = float(cfg.alpha), int(cfg.n_strata)
    q_lat = split_conformal_quantile(torch.from_numpy(cal_lat).double(), alpha)
    q_prob = split_conformal_quantile(torch.from_numpy(cal_prob).double(), alpha)
    strata = stratify_by_baseline(torch.from_numpy(te_p0).double(), n_strata)
    c_lat = conditional_coverage_gap(torch.from_numpy(te_lat).double(), strata, q_lat)
    c_prob = conditional_coverage_gap(
        torch.from_numpy(te_prob).double(), strata, q_prob
    )
    pl = np.array([c_lat[k] for k in range(n_strata)])
    pp = np.array([c_prob[k] for k in range(n_strata)])
    return {
        "profile_latent": pl,
        "profile_probability": pp,
        "gap_latent": float(pl.max() - pl.min()),
        "gap_probability": float(pp.max() - pp.min()),
        "marginal_latent": empirical_coverage(torch.from_numpy(te_lat).double(), q_lat),
        "marginal_probability": empirical_coverage(
            torch.from_numpy(te_prob).double(), q_prob
        ),
        "infinite_quantile": bool(np.isinf(q_lat) or np.isinf(q_prob)),
    }


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    gs = cfg.gate2_splice
    sub = SpliceSubstrate(SpliceConfig(seed=seed, window=int(cfg.splice.window)))
    splits = {s: sub.load(s) for s in ("train", "val", "cal", "test")}
    n_cap = int(gs.max_per_split)
    for k in splits:
        if len(splits[k]) > n_cap:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(splits[k]), n_cap, replace=False)
            splits[k] = [splits[k][i] for i in sorted(idx)]

    lat_splits, clamped = {}, {}
    for k, v in splits.items():
        lat_splits[k], clamped[k] = relabel_latent(v)
    in_dim = int(splits["train"][0].x.size(1))
    mc = cfg.model

    r = fit(
        lambda: TargetReadoutGCN(
            in_dim, int(mc.hidden), int(mc.depth), str(mc.readout), str(mc.arch)
        ),
        lat_splits["train"],
        lat_splits["val"],
        lat_splits["test"],
        seed=seed,
        epochs=int(mc.epochs),
        lr=float(mc.lr),
        batch_size=int(mc.batch_size),
    )
    model = r.model
    assert model is not None
    informative = r.test_metrics.get("r2", float("-inf")) >= float(gs.min_test_r2)

    frac = float(gs.explanation_ball_fraction)
    cal_masks = explanation_masks(sub, splits["cal"], frac)
    te_masks = explanation_masks(sub, splits["test"], frac)
    degen = float(
        np.mean([degenerate_explanation_check(m) for m in cal_masks + te_masks])
    )

    te_p0 = np.array([sub.baseline_rate(d) for d in splits["test"]])
    ok = np.isfinite(te_p0)
    te_p0 = np.clip(te_p0[ok], EPS, 1 - EPS)

    c_lat, c_prob = model_scores(model, splits["cal"], cal_masks)
    t_lat, t_prob = model_scores(model, splits["test"], te_masks)
    main = coverage_profile(c_lat, c_prob, t_lat[ok], t_prob[ok], te_p0, cfg)

    # sanity controls, mandatory per CLAUDE.md
    rand_model = randomize_model_parameters(model, layers="all")
    rc_lat, rc_prob = model_scores(rand_model, splits["cal"], cal_masks)
    rt_lat, rt_prob = model_scores(rand_model, splits["test"], te_masks)
    model_rand = coverage_profile(rc_lat, rc_prob, rt_lat[ok], rt_prob[ok], te_p0, cfg)

    g = torch.Generator().manual_seed(seed + 77)
    perm = torch.randperm(len(lat_splits["train"]), generator=g)
    shuffled = []
    for i, d in enumerate(lat_splits["train"]):
        e = copy.copy(d)
        e.y = lat_splits["train"][int(perm[i])].y
        shuffled.append(e)
    r2 = fit(
        lambda: TargetReadoutGCN(
            in_dim, int(mc.hidden), int(mc.depth), str(mc.readout), str(mc.arch)
        ),
        shuffled,
        lat_splits["val"],
        lat_splits["test"],
        seed=seed,
        epochs=int(mc.epochs),
        lr=float(mc.lr),
        batch_size=int(mc.batch_size),
    )
    assert r2.model is not None
    lc_lat, lc_prob = model_scores(r2.model, splits["cal"], cal_masks)
    lt_lat, lt_prob = model_scores(r2.model, splits["test"], te_masks)
    label_rand = coverage_profile(lc_lat, lc_prob, lt_lat[ok], lt_prob[ok], te_p0, cfg)

    # does the model track the baseline at all? if not, a flat profile is
    # uninformative rather than evidence about the link
    with torch.no_grad():
        from torch_geometric.loader import DataLoader

        zs = []
        for b in DataLoader(splits["test"], batch_size=256, shuffle=False):
            zs.append(model(b.x, b.edge_index, b.batch, b.ptr, b.target_idx).cpu())
        z_test = torch.cat(zs).numpy()[ok]
    lg = np.log(te_p0 / (1 - te_p0))
    r_base = (
        float(np.corrcoef(z_test, lg)[0, 1]) if z_test.std() > 1e-9 else float("nan")
    )

    return {
        "seed": seed,
        "n": {k: len(v) for k, v in splits.items()},
        "clamped_fraction": clamped,
        "val_r2": float(r.val_metric),
        "test_r2": float(r.test_metrics.get("r2", float("nan"))),
        "informative": bool(informative),
        "r_model_latent_vs_logit_baseline": r_base,
        "degenerate_fraction": degen,
        "main": main,
        "model_randomization": model_rand,
        "label_randomization": label_rand,
    }


def make_table(agg: dict[str, Any], cfg: Any, meta: dict[str, Any]) -> str:
    L = ["# ABLATIONS 1.9 on the real splice substrate, scored by a trained model\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}, config "
        f"`{meta['config_hash']}`, {agg['n_seeds']} seeds, mean [95% bootstrap CI].\n"
    )
    L.append(
        "Substrate: MFASS via `certgnn.substrates.splice`, splits grouped by exon. The scorer "
        "is a trained GNN under a Theorem 4 resistance-ball soft mask threaded through every "
        "message-passing layer. There is no oracle anywhere in this experiment, which is the "
        "circularity it exists to remove.\n"
    )
    L.append("## Trained model\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    for k, lab in (
        ("val_r2", "validation R² (latent scale)"),
        ("test_r2", "test R² (latent scale)"),
        ("r_model_latent_vs_logit_baseline", "r(model latent, logit baseline)"),
        ("degenerate_fraction", "degenerate explanations"),
    ):
        L.append(f"| {lab} | {fmt_ci(agg[k])} |")
    L.append(
        f"| clamped at the logit boundary (train) | {fmt_ci(agg['clamped_train'])} |"
    )
    L.append(f"| informative every seed | {agg['informative_all']} |")
    L.append("")
    L.append("## Conditional coverage, max−min across baseline strata\n")
    L.append(
        "| arm | probability-space gap | latent-space gap | marginal (prob / latent) |"
    )
    L.append("|---|---|---|---|")
    for arm, lab in (
        ("main", "trained model"),
        ("model_randomization", "model randomization"),
        ("label_randomization", "label randomization"),
    ):
        a = agg[arm]
        L.append(
            f"| {lab} | {fmt_ci(a['gap_probability'])} | {fmt_ci(a['gap_latent'])} | "
            f"{a['marginal_probability']['mean']:.3f} / {a['marginal_latent']['mean']:.3f} |"
        )
    L.append("")
    L.append("## Per-stratum coverage (mean over seeds)\n")
    L.append(
        "| arm | space | "
        + " | ".join(f"s{i}" for i in range(int(cfg.n_strata)))
        + " |"
    )
    L.append("|---" * (int(cfg.n_strata) + 2) + "|")
    for arm, lab in (
        ("main", "trained model"),
        ("model_randomization", "model rand."),
        ("label_randomization", "label rand."),
    ):
        for sp in ("probability", "latent"):
            L.append(
                f"| {lab} | {sp} | "
                + " | ".join(f"{v:.3f}" for v in agg[arm][f"profile_{sp}"])
                + " |"
            )
    L.append("")
    L.append("## Reading\n")
    L.append(agg["verdict"])
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    configure_torch(cfg)
    run = make_run_dir(cfg, "gate2_splice", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    print(f"run dir: {run}")

    per_seed = []
    for seed in resolve_seeds(cfg):
        r = run_seed(cfg, seed)
        per_seed.append(r)
        print(
            f"seed {seed}: test R2 {r['test_r2']:+.3f} | prob gap {r['main']['gap_probability']:.3f} "
            f"| latent gap {r['main']['gap_latent']:.3f} | r(base) {r['r_model_latent_vs_logit_baseline']:+.3f}",
            flush=True,
        )
        (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    nb = int(cfg.bootstrap_resamples)
    agg: dict[str, Any] = {"n_seeds": len(per_seed)}
    for k in (
        "val_r2",
        "test_r2",
        "r_model_latent_vs_logit_baseline",
        "degenerate_fraction",
    ):
        agg[k] = mean_ci([r[k] for r in per_seed], n_boot=nb)
    agg["clamped_train"] = mean_ci(
        [r["clamped_fraction"]["train"] for r in per_seed], n_boot=nb
    )
    agg["informative_all"] = all(r["informative"] for r in per_seed)
    for arm in ("main", "model_randomization", "label_randomization"):
        a: dict[str, Any] = {}
        for k in (
            "gap_probability",
            "gap_latent",
            "marginal_probability",
            "marginal_latent",
        ):
            a[k] = mean_ci([r[arm][k] for r in per_seed], n_boot=nb)
        for sp in ("probability", "latent"):
            a[f"profile_{sp}"] = np.mean(
                [r[arm][f"profile_{sp}"] for r in per_seed], axis=0
            )
        a["any_infinite"] = any(r[arm]["infinite_quantile"] for r in per_seed)
        agg[arm] = a

    m, mr, lr_ = agg["main"], agg["model_randomization"], agg["label_randomization"]
    prob_gt_lat = m["gap_probability"]["lo"] > m["gap_latent"]["hi"]
    above_controls = m["gap_probability"]["lo"] > max(
        mr["gap_probability"]["hi"], lr_["gap_probability"]["hi"]
    )
    if not agg["informative_all"]:
        verdict = (
            f"**Uninformative.** The trained model did not reach the configured test R² on every "
            f"seed (mean {agg['test_r2']['mean']:+.3f}), so a flat coverage profile carries no "
            "information about the link and this run cannot support or refute Theorem 2(iii)."
        )
    elif prob_gt_lat and above_controls:
        verdict = (
            f"**The pathology reproduces on real data with a trained model.** The "
            f"probability-space conditional coverage gap is {m['gap_probability']['mean']:.3f} "
            f"[{m['gap_probability']['lo']:.3f}, {m['gap_probability']['hi']:.3f}] against a "
            f"latent-space gap of {m['gap_latent']['mean']:.3f} "
            f"[{m['gap_latent']['lo']:.3f}, {m['gap_latent']['hi']:.3f}], and the "
            "probability-space gap exceeds both randomization controls. No oracle is involved, "
            "so this is not the generator detecting its own sigmoid."
        )
    elif prob_gt_lat:
        verdict = (
            f"**Probability space is worse than latent space, but the controls are not clear.** "
            f"Gaps: probability {m['gap_probability']['mean']:.3f}, latent "
            f"{m['gap_latent']['mean']:.3f}; model randomization "
            f"{mr['gap_probability']['mean']:.3f}, label randomization "
            f"{lr_['gap_probability']['mean']:.3f}. A gap that its own null controls also "
            "produce is not evidence about the link."
        )
    else:
        verdict = (
            f"**The pathology does NOT reproduce on real data.** The probability-space gap is "
            f"{m['gap_probability']['mean']:.3f} [{m['gap_probability']['lo']:.3f}, "
            f"{m['gap_probability']['hi']:.3f}] and the latent-space gap "
            f"{m['gap_latent']['mean']:.3f} [{m['gap_latent']['lo']:.3f}, "
            f"{m['gap_latent']['hi']:.3f}]; the probability-space gap is not reliably the larger. "
            "On this evidence Theorem 2(iii) is not supported once the oracle is removed and the "
            "substrate is real, and the paper must say so."
        )
    agg["verdict"] = verdict

    table = make_table(agg, cfg, meta)
    tab = pathlib.Path(cfg.output.tables)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "gate2_splice.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(json.dumps(to_jsonable(agg), indent=1))
    (run / "per_seed_values.json").write_text(
        json.dumps(
            {
                "gap_probability": {
                    str(r["seed"]): r["main"]["gap_probability"] for r in per_seed
                },
                "gap_latent": {
                    str(r["seed"]): r["main"]["gap_latent"] for r in per_seed
                },
                "test_r2": {str(r["seed"]): r["test_r2"] for r in per_seed},
            },
            indent=1,
        )
    )
    write_tuning_budget(
        run,
        [
            {
                "model": str(cfg.model.arch),
                "configs_tried": 1,
                "epochs": int(cfg.model.epochs),
                "gradient_steps": int(cfg.model.epochs),
                "search_space": "the pre-selected configs/model config, used as is",
                "selection": "best validation epoch; test split touched once",
            }
        ],
    )
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
