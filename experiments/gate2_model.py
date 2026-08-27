"""ABLATIONS 1.9 with a TRAINED model in the loop, as a check on Gate G2.

WHY THIS EXISTS

``gate2_link.py`` computes its fidelity scores from ``Oracle.latent``, the
generator's own latent function. Data are produced by pushing a latent effect
through a sigmoid, and the score is then read back off that same sigmoid. The
conditional-coverage pathology it reports is therefore a property of the
generative process, and cannot by construction distinguish "the link matters
for a model" from "we generated through a link". This experiment replaces the
oracle scorer with a trained GNN and leaves everything else alone -- same
topology, same link, same delta, same resistance-ball explanation, same
split-conformal machinery, same stratification -- so the only thing that
differs between the two coverage profiles is who computes the score.

WHAT HAD TO CHANGE UNDERNEATH

Two things, both reported here because both could have silently produced a
meaningless answer.

*The head must emit logits.* CLAUDE.md requires model outputs in latent space,
but the substrate's shipped label ``y`` is ``mu``, a probability, so the task
inferred from it is regression onto a probability and the trained head would
live in probability space. A "latent gap" read off such a model is a
probability gap with a latent label -- the original circularity rebuilt one
level up. The models here are trained on ``eta`` instead, and the choice is
recorded in the run directory.

*The model must be able to see the baseline.* ``eta0`` is drawn per instance
and appears in no node feature, so a model trained on the substrate as shipped
cannot represent the baseline rate at all. Its scores are then flat across
strata of ``|logit p0|`` whatever the link does, and "no pathology" would be
indistinguishable from "no information". Both arms are therefore run: one with
the baseline hidden, as shipped, and one with it appended as a node feature,
which is the situation in a real substrate where the baseline rate is a
function of the input. The second arm is the one that can carry evidence.

WHAT IT DOES NOT DO

It does not re-decide G2. G2's verdict stands on its pre-registered thresholds
and is not touched here. This run reports both profiles side by side and states
what the comparison supports.

Usage:
    python -m experiments.gate2_model --config configs/base.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from certgnn.certify.conformal import (  # noqa: E402
    conditional_coverage_gap,
    empirical_coverage,
    split_conformal_quantile,
)
from certgnn.certify.link import stratify_by_baseline  # noqa: E402
from certgnn.eval.sanity import (  # noqa: E402
    degenerate_explanation_check,
    randomize_labels,
    randomize_model_parameters,
)
from certgnn.explain.masks import resistance_ball_init  # noqa: E402
from certgnn.models.gnn import TargetReadoutGCN  # noqa: E402
from certgnn.models.train import fit, primary_metric  # noqa: E402
from certgnn.substrates.synthetic import (  # noqa: E402
    SyntheticConfig,
    SyntheticSubstrate,
    TopologySpec,
)
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

ARMS = ("baseline_hidden", "baseline_observable")


# ------------------------------------------------------------------ data


def build_substrate(
    cfg: Any, seed: int, *, baseline_feature: bool
) -> SyntheticSubstrate:
    """Substrate for one arm, matching ``gate2.coverage`` except where stated."""
    gm = cfg.gate2_model
    cov = cfg.gate2.coverage
    return SyntheticSubstrate(
        SyntheticConfig(
            topology=TopologySpec(**dict(cov.topology)),
            regulator=str(gm.regulator),
            link="logit",
            delta=float(gm.delta),
            p0=None,
            eta0_mean=float(cfg.gate2.eta0_mean),
            eta0_std=float(cfg.gate2.eta0_std),
            baseline_feature=baseline_feature,
            n_train=int(gm.n_train),
            n_val=int(gm.n_val),
            n_cal=int(gm.n_cal),
            n_test=int(gm.n_test),
            seed=seed,
        )
    )


def retarget(data: list, field: str) -> list:
    """Copy ``data`` with ``y`` replaced by another stored field.

    ``field="latent"`` puts ``eta`` in ``y`` so the head is trained in latent
    space; ``field="probability"`` leaves the shipped ``mu``-derived label.
    """
    if field == "probability":
        return list(data)
    if field != "latent":
        raise ValueError(f"target must be 'latent' or 'probability', got {field!r}")
    out = []
    for d in data:
        e = d.clone()
        e.y = d.eta.reshape(1).to(torch.float32)
        out.append(e)
    return out


# --------------------------------------------------------------- scoring


def explanation_masks(
    sub: SyntheticSubstrate, data: list, frac: float
) -> list[torch.Tensor]:
    """The resistance-ball explanation for each instance.

    Identical to the one ``gate2_link`` scores with the oracle, so the two
    coverage profiles differ only in the scorer and never in the explanation.
    """
    masks = []
    for d in data:
        o = sub.oracle(d)
        k = max(1, int(round(frac * (o.n - 1))))
        masks.append(resistance_ball_init(o.graph, o.target, ball_size=k, exact=False))
    return masks


@torch.no_grad()
def model_latents(
    model: torch.nn.Module,
    data: list,
    masks: list[torch.Tensor] | None,
    batch_size: int = 128,
) -> np.ndarray:
    """Latent output per instance under a per-instance soft node mask.

    ``masks=None`` is the unmasked pass. Batched: the per-graph masks are
    concatenated in the same order the loader concatenates nodes.
    """
    model.eval()
    if masks is not None:
        data = [d.clone() for d in data]
        for d, m in zip(data, masks, strict=True):
            d.node_mask = m.reshape(-1).to(torch.float32)
    outs = []
    for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
        mask = getattr(batch, "node_mask", None) if masks is not None else None
        outs.append(
            model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.ptr,
                batch.target_idx,
                mask,
            ).cpu()
        )
    return torch.cat(outs).numpy()


def model_scores(
    model: torch.nn.Module, data: list, masks: list[torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    """Latent and probability fidelity gaps of the explanation, from the model."""
    z_full = model_latents(model, data, [torch.ones_like(m) for m in masks])
    z_mask = model_latents(model, data, masks)
    s_lat = np.abs(z_full - z_mask)
    p_full = 1.0 / (1.0 + np.exp(-z_full))
    p_mask = 1.0 / (1.0 + np.exp(-z_mask))
    return s_lat, np.abs(p_full - p_mask)


def oracle_scores(
    sub: SyntheticSubstrate, data: list, masks: list[torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    """The same two gaps from the generator's latent function (the gate2_link scorer)."""
    s_lat, s_prob = [], []
    for d, m in zip(data, masks, strict=True):
        o = sub.oracle(d)
        z_full = o.latent_full()
        z_mask = float(o.latent(m))
        s_lat.append(abs(z_full - z_mask))
        s_prob.append(
            abs(
                float(torch.sigmoid(torch.tensor(z_full)))
                - float(torch.sigmoid(torch.tensor(z_mask)))
            )
        )
    return np.array(s_lat), np.array(s_prob)


def coverage_profile(
    cal_lat: np.ndarray,
    cal_prob: np.ndarray,
    te_lat: np.ndarray,
    te_prob: np.ndarray,
    te_p0: np.ndarray,
    cfg: Any,
) -> dict[str, Any]:
    """Split-conformal conditional coverage in both score spaces.

    Exactly the 1.9 analysis of ``gate2_link.run_coverage``; only the arrays
    handed to it come from somewhere else.
    """
    alpha = float(cfg.alpha)
    n_strata = int(cfg.n_strata)
    q_lat = split_conformal_quantile(torch.from_numpy(cal_lat), alpha)
    q_prob = split_conformal_quantile(torch.from_numpy(cal_prob), alpha)
    strata = stratify_by_baseline(torch.from_numpy(te_p0), n_strata)
    cov_lat = conditional_coverage_gap(torch.from_numpy(te_lat), strata, q_lat)
    cov_prob = conditional_coverage_gap(torch.from_numpy(te_prob), strata, q_prob)
    prof_lat = np.array([cov_lat[k] for k in range(n_strata)])
    prof_prob = np.array([cov_prob[k] for k in range(n_strata)])
    return {
        "coverage_profile_latent": prof_lat,
        "coverage_profile_probability": prof_prob,
        "gap_latent": float(prof_lat.max() - prof_lat.min()),
        "gap_probability": float(prof_prob.max() - prof_prob.min()),
        "marginal_latent": empirical_coverage(torch.from_numpy(te_lat), q_lat),
        "marginal_probability": empirical_coverage(torch.from_numpy(te_prob), q_prob),
        "q_latent": float(q_lat),
        "q_probability": float(q_prob),
        "infinite_quantile": bool(np.isinf(q_lat) or np.isinf(q_prob)),
        "n_cal": int(cal_lat.size),
        "n_test": int(te_lat.size),
    }


# ------------------------------------------------------------------ arm


def make_model_factory(cfg: Any, in_dim: int):
    mc = cfg.model
    return lambda: TargetReadoutGCN(
        in_dim,
        int(mc.hidden),
        int(mc.depth),
        str(mc.readout),
        str(mc.get("arch", "gcn")),
    )


def run_arm(cfg: Any, seed: int, arm: str) -> dict[str, Any]:
    """Train one model and score the 1.9 analysis with it and with the oracle."""
    gm = cfg.gate2_model
    mc = cfg.model
    sub = build_substrate(cfg, seed, baseline_feature=(arm == "baseline_observable"))
    splits = {s: sub.load(s) for s in ("train", "val", "cal", "test")}
    tgt = str(gm.target)
    tr, va = retarget(splits["train"], tgt), retarget(splits["val"], tgt)
    te_for_fit = retarget(splits["test"], tgt)
    in_dim = int(tr[0].x.size(1))
    kw: dict[str, Any] = dict(
        seed=seed, epochs=int(mc.epochs), lr=float(mc.lr), batch_size=int(mc.batch_size)
    )
    r = fit(make_model_factory(cfg, in_dim), tr, va, te_for_fit, **kw)
    model = r.model
    assert model is not None, "fit must return the trained model"

    frac = float(gm.explanation_ball_fraction)
    cal_masks = explanation_masks(sub, splits["cal"], frac)
    te_masks = explanation_masks(sub, splits["test"], frac)
    te_p0 = np.array([float(d.p0) for d in splits["test"]])

    m_cal_lat, m_cal_prob = model_scores(model, splits["cal"], cal_masks)
    m_te_lat, m_te_prob = model_scores(model, splits["test"], te_masks)
    o_cal_lat, o_cal_prob = oracle_scores(sub, splits["cal"], cal_masks)
    o_te_lat, o_te_prob = oracle_scores(sub, splits["test"], te_masks)

    key = primary_metric(r.task)
    out: dict[str, Any] = {
        "arm": arm,
        "task": r.task,
        "target": tgt,
        "in_dim": in_dim,
        "best_epoch": r.best_epoch,
        "val_metric": float(r.val_metric),
        "test_metrics": {k: float(v) for k, v in r.test_metrics.items()},
        "primary_metric": key,
        "learned": bool(float(r.val_metric) >= float(gm.min_val_r2)),
        "model": coverage_profile(
            m_cal_lat, m_cal_prob, m_te_lat, m_te_prob, te_p0, cfg
        ),
        "oracle": coverage_profile(
            o_cal_lat, o_cal_prob, o_te_lat, o_te_prob, te_p0, cfg
        ),
    }

    # Does the model's latent output track the baseline at all? If it does not,
    # a flat profile says nothing about the link.
    z_full = model_latents(model, splits["test"], None)
    logit_p0 = np.log(te_p0 / (1.0 - te_p0))
    out["latent_vs_baseline_pearson_r"] = float(np.corrcoef(z_full, logit_p0)[0, 1])
    out["latent_std"] = float(z_full.std())

    out["sanity"] = run_sanity(
        cfg, seed, arm, sub, splits, cal_masks, te_masks, te_p0, model
    )
    out["degenerate_explanations"] = float(
        np.mean([degenerate_explanation_check(m) for m in te_masks])
    )
    return out


def run_sanity(
    cfg: Any,
    seed: int,
    arm: str,
    sub: SyntheticSubstrate,
    splits: dict[str, list],
    cal_masks: list[torch.Tensor],
    te_masks: list[torch.Tensor],
    te_p0: np.ndarray,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Model- and label-randomization controls (CLAUDE.md: not optional).

    Both re-run the whole 1.9 analysis against a model that should carry no
    signal. A pathology that survives them is a property of the input or the
    explanation, not of the trained model.
    """
    gm, mc = cfg.gate2_model, cfg.model
    tgt = str(gm.target)

    rand = randomize_model_parameters(model)
    rc_lat, rc_prob = model_scores(rand, splits["cal"], cal_masks)
    rt_lat, rt_prob = model_scores(rand, splits["test"], te_masks)
    randomized = coverage_profile(rc_lat, rc_prob, rt_lat, rt_prob, te_p0, cfg)

    tr = retarget(splits["train"], tgt)
    ys = torch.cat([d.y.reshape(-1) for d in tr])
    permuted = randomize_labels(ys, seed=seed)
    tr_perm = []
    for d, y in zip(tr, permuted, strict=True):
        e = d.clone()
        e.y = y.reshape(1)
        tr_perm.append(e)
    va = retarget(splits["val"], tgt)
    r = fit(
        make_model_factory(cfg, int(tr[0].x.size(1))),
        tr_perm,
        va,
        va,
        seed=seed,
        epochs=int(mc.epochs),
        lr=float(mc.lr),
        batch_size=int(mc.batch_size),
    )
    assert r.model is not None
    lc_lat, lc_prob = model_scores(r.model, splits["cal"], cal_masks)
    lt_lat, lt_prob = model_scores(r.model, splits["test"], te_masks)
    label_rand = coverage_profile(lc_lat, lc_prob, lt_lat, lt_prob, te_p0, cfg)
    return {
        "model_randomization": {
            "gap_probability": randomized["gap_probability"],
            "gap_latent": randomized["gap_latent"],
        },
        "label_randomization": {
            "val_metric": float(r.val_metric),
            "gap_probability": label_rand["gap_probability"],
            "gap_latent": label_rand["gap_latent"],
        },
    }


# ------------------------------------------------------------ aggregation


def aggregate(per_seed: list[dict[str, Any]], cfg: Any) -> dict[str, Any]:
    n_strata = int(cfg.n_strata)
    B = int(cfg.bootstrap_resamples)
    out: dict[str, Any] = {"n_seeds": len(per_seed), "arms": {}}
    for arm in ARMS:
        rows = [r["arms"][arm] for r in per_seed]
        a: dict[str, Any] = {
            "val_metric": mean_ci([x["val_metric"] for x in rows], n_boot=B),
            "test_primary": mean_ci(
                [x["test_metrics"][x["primary_metric"]] for x in rows], n_boot=B
            ),
            "learned_all_seeds": all(x["learned"] for x in rows),
            "latent_vs_baseline_pearson_r": mean_ci(
                [x["latent_vs_baseline_pearson_r"] for x in rows], n_boot=B
            ),
            "degenerate_explanations": mean_ci(
                [x["degenerate_explanations"] for x in rows], n_boot=B
            ),
        }
        for scorer in ("model", "oracle"):
            for space in ("probability", "latent"):
                a[f"{scorer}_gap_{space}"] = mean_ci(
                    [x[scorer][f"gap_{space}"] for x in rows], n_boot=B
                )
                a[f"{scorer}_marginal_{space}"] = mean_ci(
                    [x[scorer][f"marginal_{space}"] for x in rows], n_boot=B
                )
                a[f"{scorer}_profile_{space}_mean"] = [
                    float(
                        np.mean(
                            [x[scorer][f"coverage_profile_{space}"][k] for x in rows]
                        )
                    )
                    for k in range(n_strata)
                ]
            a[f"{scorer}_any_infinite_quantile"] = any(
                x[scorer]["infinite_quantile"] for x in rows
            )
        for ctrl in ("model_randomization", "label_randomization"):
            a[ctrl] = {
                k: mean_ci([x["sanity"][ctrl][k] for x in rows], n_boot=B)
                for k in ("gap_probability", "gap_latent")
            }
        out["arms"][arm] = a
    return out


def interpret(agg: dict[str, Any], cfg: Any) -> dict[str, Any]:
    """What the side-by-side supports. Deliberately not a gate verdict."""
    gm = cfg.gate2_model
    findings = {}
    for arm in ARMS:
        a = agg["arms"][arm]
        o_gap, m_gap = a["oracle_gap_probability"], a["model_gap_probability"]
        m_lat = a["model_gap_latent"]
        informative = (
            a["learned_all_seeds"]
            and abs(a["latent_vs_baseline_pearson_r"]["mean"]) > 0.1
        )
        reproduces = m_gap["lo"] > m_lat["hi"]
        findings[arm] = {
            "informative": bool(informative),
            "why_uninformative": (
                None
                if informative
                else (
                    "the trained model does not track the baseline "
                    f"(|r| = {abs(a['latent_vs_baseline_pearson_r']['mean']):.3f}) "
                    "or did not reach the configured val threshold, so a flat "
                    "profile carries no information about the link"
                )
            ),
            "model_reproduces_pathology": bool(reproduces),
            "oracle_prob_gap": o_gap,
            "model_prob_gap": m_gap,
            "model_latent_gap": m_lat,
        }
    ev = [arm for arm in ARMS if findings[arm]["informative"]]
    supported = bool(ev) and all(findings[a]["model_reproduces_pathology"] for a in ev)
    return {
        "arms": findings,
        "informative_arms": ev,
        "model_derived_support": supported,
        "min_val_threshold": float(gm.min_val_r2),
        "statement": (
            "Theorem 2's conditional-coverage pathology reproduces with a trained "
            "model in the loop on every informative arm."
            if supported
            else (
                "The pathology does NOT reproduce with a trained model on the "
                "informative arms; on this evidence Theorem 2's empirical support "
                "is limited to the generative process and the paper must say so."
                if ev
                else "No arm was informative: no trained model here tracks the "
                "baseline, so this run cannot decide the question either way."
            )
        ),
    }


# --------------------------------------------------------------- outputs


def make_figure(agg: dict[str, Any], cfg: Any, stem: pathlib.Path) -> None:
    n = int(cfg.n_strata)
    xs = np.arange(n)
    fig, axes = plt.subplots(1, len(ARMS), figsize=(9.5, 3.4), sharey=True)
    for ax, arm in zip(np.atleast_1d(axes), ARMS, strict=True):
        a = agg["arms"][arm]
        for scorer, style in (("oracle", "--"), ("model", "-")):
            for space, colour in (("probability", "#b2182b"), ("latent", "#2166ac")):
                ax.plot(
                    xs,
                    a[f"{scorer}_profile_{space}_mean"],
                    style,
                    color=colour,
                    marker="o" if scorer == "model" else "s",
                    ms=4,
                    lw=1.4,
                    label=f"{scorer}, {space}",
                )
        ax.axhline(1.0 - float(cfg.alpha), color="0.4", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xlabel(r"stratum of $|\mathrm{logit}\ p_0|$")
        ax.set_title(
            f"{arm.replace('_', ' ')}\n"
            f"val={a['val_metric']['mean']:.3f}, "
            f"r(latent, logit $p_0$)={a['latent_vs_baseline_pearson_r']['mean']:.2f}",
            fontsize=9,
        )
    np.atleast_1d(axes)[0].set_ylabel("conditional coverage")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def make_table(
    agg: dict[str, Any], interp: dict[str, Any], cfg: Any, meta: dict
) -> str:
    L = [
        "# ABLATIONS 1.9 with a trained model (experiments/gate2_model.py)\n",
        f"git SHA `{meta['git_sha']}` (dirty: {meta['git_dirty']}), config hash "
        f"`{meta['config_hash']}`, {agg['n_seeds']} seeds, mean [95% bootstrap CI].\n",
        "Scores come from a trained GNN under a soft node mask, and from the "
        "generator's latent function, on the *same* instances with the *same* "
        "resistance-ball explanation.\n",
        "## Trained models\n",
        "| arm | node features | val (latent R²) | test R² | r(model latent, logit p₀) | learned every seed |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        a = agg["arms"][arm]
        feats = "z, degree, distance" + (
            ", eta0" if arm == "baseline_observable" else ""
        )
        L.append(
            f"| {arm.replace('_', ' ')} | {feats} | {fmt_ci(a['val_metric'])} | "
            f"{fmt_ci(a['test_primary'])} | {fmt_ci(a['latent_vs_baseline_pearson_r'])} | "
            f"{a['learned_all_seeds']} |"
        )
    L.append("\n## Conditional coverage, max−min across strata\n")
    L.append(
        "| arm | scorer | probability-space gap | latent-space gap | marginal (prob / latent) |"
    )
    L.append("|---|---|---|---|---|")
    for arm in ARMS:
        a = agg["arms"][arm]
        for scorer in ("oracle", "model"):
            L.append(
                f"| {arm.replace('_', ' ')} | {scorer} | {fmt_ci(a[f'{scorer}_gap_probability'])} | "
                f"{fmt_ci(a[f'{scorer}_gap_latent'])} | "
                f"{a[f'{scorer}_marginal_probability']['mean']:.3f} / "
                f"{a[f'{scorer}_marginal_latent']['mean']:.3f} |"
            )
    L.append("\n## Per-stratum coverage (mean over seeds)\n")
    L.append(
        "| arm | scorer | space | "
        + " | ".join(f"s{k}" for k in range(int(cfg.n_strata)))
        + " |"
    )
    L.append("|---|---|---|" + "---|" * int(cfg.n_strata))
    for arm in ARMS:
        for scorer in ("oracle", "model"):
            for space in ("probability", "latent"):
                prof = agg["arms"][arm][f"{scorer}_profile_{space}_mean"]
                L.append(
                    f"| {arm.replace('_', ' ')} | {scorer} | {space} | "
                    + " | ".join(f"{x:.3f}" for x in prof)
                    + " |"
                )
    L.append("\n## Sanity controls (CLAUDE.md: mandatory)\n")
    L.append("| arm | control | probability-space gap | latent-space gap |")
    L.append("|---|---|---|---|")
    for arm in ARMS:
        a = agg["arms"][arm]
        for ctrl in ("model_randomization", "label_randomization"):
            L.append(
                f"| {arm.replace('_', ' ')} | {ctrl.replace('_', ' ')} | "
                f"{fmt_ci(a[ctrl]['gap_probability'])} | {fmt_ci(a[ctrl]['gap_latent'])} |"
            )
    L.append("\n## Reading\n")
    for arm in ARMS:
        f = interp["arms"][arm]
        L.append(
            f"- **{arm.replace('_', ' ')}**: informative = {f['informative']}"
            + (f" ({f['why_uninformative']})" if f["why_uninformative"] else "")
            + f"; model reproduces the pathology = {f['model_reproduces_pathology']}."
        )
    L.append(f"\n**{interp['statement']}**\n")
    L.append(
        "This run does not re-decide G2. G2's verdict rests on its pre-registered "
        "thresholds and is unchanged.\n"
    )
    return "\n".join(L)


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    ap.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    configure_torch(cfg)
    run = make_run_dir(cfg, "gate2_model", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    print(f"run dir: {run}")

    per_seed: list[dict[str, Any]] = []
    t0 = time.time()
    for seed in resolve_seeds(cfg):
        r: dict[str, Any] = {"seed": seed, "arms": {}}
        for arm in ARMS:
            r["arms"][arm] = run_arm(cfg, seed, arm)
            a = r["arms"][arm]
            print(
                f"seed {seed} {arm:<20} val={a['val_metric']:.3f} "
                f"r(z,logit p0)={a['latent_vs_baseline_pearson_r']:+.3f}  "
                f"model prob/lat gap {a['model']['gap_probability']:.3f}/"
                f"{a['model']['gap_latent']:.3f}  "
                f"oracle {a['oracle']['gap_probability']:.3f}/"
                f"{a['oracle']['gap_latent']:.3f}  [{time.time() - t0:.0f}s]",
                flush=True,
            )
        per_seed.append(r)
        (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    agg = aggregate(per_seed, cfg)
    interp = interpret(agg, cfg)

    fig_dir = pathlib.Path(cfg.output.figures)
    tab_dir = pathlib.Path(cfg.output.tables)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    make_figure(agg, cfg, fig_dir / "gate2_model")
    table = make_table(agg, interp, cfg, meta)
    (tab_dir / "gate2_model.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(
        json.dumps(to_jsonable({"aggregate": agg, "interpretation": interp}), indent=1)
    )
    mc = cfg.model
    write_tuning_budget(
        run,
        [
            {
                "model": str(mc.get("arch", "gcn")),
                "configs_tried": 1,
                "epochs": int(mc.epochs),
                "gradient_steps": "not counted; epochs x ceil(n_train/batch_size) per fit",
                "search_space": "none: the pre-selected configs/model config was used as is",
                "selection": "best validation epoch, validation split only",
            }
        ],
    )
    for src in (fig_dir / "gate2_model.png", fig_dir / "gate2_model.pdf"):
        shutil.copy(src, run / src.name)

    print(table)
    print(
        f"\nfigure: {fig_dir / 'gate2_model.png'}\ntable: {tab_dir / 'gate2_model.md'}\nrun: {run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
