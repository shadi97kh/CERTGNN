"""Gate 2: link-function correction (Theorem 2), ABLATIONS rows 1.6 to 1.9.

Runs on the synthetic substrate, where the latent effect of the regulator
is known by construction, so any baseline dependence of the *observed*
(probability-space) effect is attributable to the link and nothing else.

    1.6  link sweep: the non-comparability effect must be absent under the
         identity link and present under logit / probit / cloglog.
    1.7  observed effect vs baseline rate: fitted exponent gamma in
         ratio = C * [p0 (1 - p0)]^gamma against the predicted gamma = 1.
    1.8  explicit cross-instance rank reversal at fixed baseline pairs.
    1.9  conditional coverage profile of a split-conformal fidelity
         certificate, probability-space vs latent-space score, stratified on
         |logit p0| in 5 strata. Feeds Gate G2.

Outputs: one four-panel figure in paper/figures/, a results table in
paper/tables/, and everything (raw per-seed numbers, verdict) in the run
directory under results/runs/. The G2 verdict compares the seed-mean gaps
to the pre-registered thresholds and is written to verdict.md; appending
to results/GATE_LOG.md is done by the gate-check step, not here.

Usage:
    python -m experiments.gate2_link --config configs/base.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import shutil
import sys
import time
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from certgnn.certify.conformal import (  # noqa: E402
    conditional_coverage_gap,
    empirical_coverage,
    split_conformal_quantile,
)
from certgnn.certify.link import stratify_by_baseline  # noqa: E402
from certgnn.explain.masks import resistance_ball_init  # noqa: E402
from certgnn.substrates.synthetic import (  # noqa: E402
    SyntheticConfig,
    SyntheticSubstrate,
    TopologySpec,
    get_link,
)
from experiments._common import (
    configure_torch,  # noqa: E402
    bootstrap_p_value,
    fmt_ci,
    load_config,
    make_run_dir,
    mean_ci,
    resolve_seeds,
    to_jsonable,
    write_tuning_budget,
)

GATE = "G2"
PREREG = pathlib.Path("PREREGISTRATION.md")


# ----------------------------------------------------------------- helpers


def _effects(
    sub: SyntheticSubstrate, data: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(eta0, p0, latent_effect, observed_effect) arrays for a split."""
    eta0 = np.array([float(d.eta0) for d in data])
    p0 = np.array([float(d.p0) for d in data])
    lat = np.array([sub.latent_effect(d) for d in data])
    obs = np.array([sub.observed_effect(d) for d in data])
    return eta0, p0, lat, obs


def _strata_means(x: np.ndarray, y: np.ndarray, n_strata: int) -> np.ndarray:
    """Mean of y within quantile strata of x."""
    edges = np.quantile(x, np.linspace(0, 1, n_strata + 1)[1:-1])
    s = np.searchsorted(edges, x, side="right")
    return np.array(
        [y[s == k].mean() if np.any(s == k) else np.nan for k in range(n_strata)]
    )


def _reversal_rate(
    lat_a: np.ndarray, obs_a: np.ndarray, lat_b: np.ndarray, obs_b: np.ndarray
) -> tuple[float, int]:
    """Among pairs with |lat_a| > |lat_b|, the fraction with |obs_a| < |obs_b|."""
    la, lb = np.abs(lat_a)[:, None], np.abs(lat_b)[None, :]
    oa, ob = np.abs(obs_a)[:, None], np.abs(obs_b)[None, :]
    qualifies = la > lb
    n = int(qualifies.sum())
    if n == 0:
        return float("nan"), 0
    return float((oa < ob)[qualifies].mean()), n


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-30:
        return float("nan")
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot


# ------------------------------------------------------------ 1.6 link sweep


def run_link_sweep(cfg: Any, seed: int, rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    g2 = cfg.gate2
    for link in g2.links:
        sub = SyntheticSubstrate(
            SyntheticConfig(
                topology=TopologySpec("path", length=2),
                link=link,
                delta=float(g2.link_sweep.delta),
                p0=None,
                eta0_mean=float(g2.eta0_mean),
                eta0_std=float(g2.eta0_std),
                n_train=int(g2.link_sweep.n_instances),
                seed=seed,
            )
        )
        eta0, p0, lat, obs = _effects(sub, sub.load("train"))
        ratio = obs / lat
        jac = get_link(link).jacobian(torch.from_numpy(eta0)).numpy()
        strata_ratio = _strata_means(np.abs(eta0), ratio, int(cfg.n_strata))
        i = rng.integers(0, len(lat), int(g2.link_sweep.n_pairs))
        j = rng.integers(0, len(lat), int(g2.link_sweep.n_pairs))
        keep = i != j
        rev, n_pairs = _reversal_rate(
            lat[i[keep]], obs[i[keep]], lat[j[keep]], obs[j[keep]]
        )
        # binned curve for the figure
        bins = np.linspace(-2.5 * float(g2.eta0_std), 2.5 * float(g2.eta0_std), 25)
        idx = np.clip(np.digitize(eta0, bins) - 1, 0, len(bins) - 2)
        curve = np.array(
            [
                ratio[idx == k].mean() if np.any(idx == k) else np.nan
                for k in range(len(bins) - 1)
            ]
        )
        out[link] = {
            "ratio_range_across_strata": float(
                np.nanmax(strata_ratio) - np.nanmin(strata_ratio)
            ),
            "r2_vs_predicted_jacobian": _r2(ratio, jac),
            "reversal_rate": rev,
            "n_pairs": n_pairs,
            "curve_bins": bins,
            "curve": curve,
        }
    return out


# ------------------------------------------------------------ 1.7 curve fit


def run_curve_fit(cfg: Any, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    g2 = cfg.gate2
    for delta in g2.curve_fit.deltas:
        sub = SyntheticSubstrate(
            SyntheticConfig(
                topology=TopologySpec("path", length=2),
                link="logit",
                delta=float(delta),
                p0=None,
                eta0_mean=float(g2.eta0_mean),
                eta0_std=float(g2.eta0_std),
                n_train=int(g2.curve_fit.n_instances),
                seed=seed,
            )
        )
        eta0, p0, lat, obs = _effects(sub, sub.load("train"))
        x = p0 * (1.0 - p0)
        y = np.abs(obs / lat)
        ok = (x > 1e-12) & (y > 1e-12)
        X = np.stack([np.ones(ok.sum()), np.log(x[ok])], axis=1)
        coef, *_ = np.linalg.lstsq(X, np.log(y[ok]), rcond=None)
        log_c, gamma = float(coef[0]), float(coef[1])
        yhat = np.exp(log_c + gamma * np.log(x[ok]))
        out[str(delta)] = {
            "gamma_hat": gamma,
            "c_hat": math.exp(log_c),
            "r2_log_fit": _r2(np.log(y[ok]), np.log(yhat)),
            "r2_vs_predicted_form": _r2(
                y[ok], x[ok]
            ),  # ratio vs p0(1-p0) with gamma=1, C=1
            "sample_x": x[ok][:300],
            "sample_y": y[ok][:300],
        }
    return out


# -------------------------------------------------------- 1.8 rank reversal


def run_rank_reversal(cfg: Any, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    g2 = cfg.gate2

    def gen(p0: float, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sub = SyntheticSubstrate(
            SyntheticConfig(
                topology=TopologySpec("path", length=2),
                link="logit",
                delta=float(g2.rank_reversal.delta),
                p0=float(p0),
                n_train=int(g2.rank_reversal.n_per_baseline),
                seed=s,
            )
        )
        _, pp, lat, obs = _effects(sub, sub.load("train"))
        return pp, lat, obs

    for k, (p_mid, p_ext) in enumerate(g2.rank_reversal.baseline_pairs):
        p_b, lat_b, obs_b = gen(float(p_mid), seed * 100 + 2 * k)
        p_a, lat_a, obs_a = gen(float(p_ext), seed * 100 + 2 * k + 1)
        rate_prob, n_pairs = _reversal_rate(lat_a, obs_a, lat_b, obs_b)
        rate_lat, _ = _reversal_rate(
            lat_a, lat_a, lat_b, lat_b
        )  # latent importance = |Delta|
        # explicit example: the reversed pair with the largest latent-effect ratio
        la, lb = np.abs(lat_a)[:, None], np.abs(lat_b)[None, :]
        oa, ob = np.abs(obs_a)[:, None], np.abs(obs_b)[None, :]
        rev = (la > lb) & (oa < ob)
        example = None
        if rev.any():
            ratio = np.where(rev, la / np.maximum(lb, 1e-12), 0.0)
            ia, ib = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
            example = {
                "a": {
                    "p0": float(p_a[ia]),
                    "latent_effect": float(lat_a[ia]),
                    "observed_effect": float(obs_a[ia]),
                },
                "b": {
                    "p0": float(p_b[ib]),
                    "latent_effect": float(lat_b[ib]),
                    "observed_effect": float(obs_b[ib]),
                },
            }
        out[f"{p_mid}_vs_{p_ext}"] = {
            "reversal_rate_probability": rate_prob,
            "reversal_rate_latent": rate_lat,
            "n_pairs": n_pairs,
            "example": example,
        }
    return out


# ------------------------------------------------- 1.9 conditional coverage


def _fidelity_scores(
    sub: SyntheticSubstrate, data: list, frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latent and probability fidelity gaps of the resistance-ball explanation."""
    s_lat, s_prob, p0 = [], [], []
    for d in data:
        o = sub.oracle(d)
        k = max(1, int(round(frac * (o.n - 1))))
        mask = resistance_ball_init(o.graph, o.target, ball_size=k, exact=False)
        z_full = o.latent_full()
        z_masked = float(o.latent(mask))
        s_lat.append(abs(z_full - z_masked))
        s_prob.append(
            abs(
                float(torch.sigmoid(torch.tensor(z_full)))
                - float(torch.sigmoid(torch.tensor(z_masked)))
            )
        )
        p0.append(float(d.p0))
    return np.array(s_lat), np.array(s_prob), np.array(p0)


def run_coverage(cfg: Any, seed: int) -> dict[str, Any]:
    g2 = cfg.gate2
    cov = g2.coverage
    sub = SyntheticSubstrate(
        SyntheticConfig(
            topology=TopologySpec(**dict(cov.topology)),
            regulator=str(cov.regulator),
            link="logit",
            delta=float(cov.delta),
            p0=None,
            eta0_mean=float(g2.eta0_mean),
            eta0_std=float(g2.eta0_std),
            n_train=0,
            n_val=0,
            n_cal=int(cov.n_cal),
            n_test=int(cov.n_test),
            seed=seed,
        )
    )
    frac = float(cov.explanation_ball_fraction)
    cal_lat, cal_prob, _ = _fidelity_scores(sub, sub.load("cal"), frac)
    te_lat, te_prob, te_p0 = _fidelity_scores(sub, sub.load("test"), frac)
    alpha = float(cfg.alpha)
    q_lat = split_conformal_quantile(torch.from_numpy(cal_lat), alpha)
    q_prob = split_conformal_quantile(torch.from_numpy(cal_prob), alpha)
    strata = stratify_by_baseline(torch.from_numpy(te_p0), int(cfg.n_strata))
    cov_lat = conditional_coverage_gap(torch.from_numpy(te_lat), strata, q_lat)
    cov_prob = conditional_coverage_gap(torch.from_numpy(te_prob), strata, q_prob)
    prof_lat = np.array([cov_lat[k] for k in range(int(cfg.n_strata))])
    prof_prob = np.array([cov_prob[k] for k in range(int(cfg.n_strata))])
    abs_logit = np.abs(np.log(te_p0 / (1 - te_p0)))
    edges = np.quantile(abs_logit, np.linspace(0, 1, int(cfg.n_strata) + 1))
    return {
        "coverage_profile_latent": prof_lat,
        "coverage_profile_probability": prof_prob,
        "gap_latent": float(prof_lat.max() - prof_lat.min()),
        "gap_probability": float(prof_prob.max() - prof_prob.min()),
        "marginal_latent": empirical_coverage(torch.from_numpy(te_lat), q_lat),
        "marginal_probability": empirical_coverage(torch.from_numpy(te_prob), q_prob),
        "q_latent": q_lat,
        "q_probability": q_prob,
        "infinite_quantile": bool(math.isinf(q_lat) or math.isinf(q_prob)),
        "stratum_edges_abs_logit_p0": edges,
        "n_cal": int(cal_lat.size),
        "n_test": int(te_lat.size),
    }


# ------------------------------------------------------------ aggregation


def aggregate(per_seed: list[dict[str, Any]], cfg: Any) -> dict[str, Any]:
    nb = int(cfg.bootstrap_resamples)

    def ci(path: list[str]) -> dict[str, float]:
        vals = []
        for r in per_seed:
            v: Any = r
            for p in path:
                v = v[p]
            vals.append(float(v) if v is not None else float("nan"))
        return mean_ci(vals, n_boot=nb)

    def stack(path: list[str]) -> np.ndarray:
        arrs = []
        for r in per_seed:
            v: Any = r
            for p in path:
                v = v[p]
            arrs.append(np.asarray(v, dtype=np.float64))
        return np.stack(arrs)

    agg: dict[str, Any] = {"n_seeds": len(per_seed)}
    agg["link_sweep"] = {
        link: {
            "ratio_range_across_strata": ci(
                ["link_sweep", link, "ratio_range_across_strata"]
            ),
            "r2_vs_predicted_jacobian": ci(
                ["link_sweep", link, "r2_vs_predicted_jacobian"]
            ),
            "reversal_rate": ci(["link_sweep", link, "reversal_rate"]),
            "curve_bins": per_seed[0]["link_sweep"][link]["curve_bins"],
            "curve_mean": np.nanmean(stack(["link_sweep", link, "curve"]), axis=0),
        }
        for link in cfg.gate2.links
    }
    agg["curve_fit"] = {
        str(d): {
            "gamma_hat": ci(["curve_fit", str(d), "gamma_hat"]),
            "c_hat": ci(["curve_fit", str(d), "c_hat"]),
            "r2_log_fit": ci(["curve_fit", str(d), "r2_log_fit"]),
            "r2_vs_predicted_form": ci(["curve_fit", str(d), "r2_vs_predicted_form"]),
        }
        for d in cfg.gate2.curve_fit.deltas
    }
    agg["rank_reversal"] = {}
    for key in per_seed[0]["rank_reversal"]:
        examples = [
            r["rank_reversal"][key]["example"]
            for r in per_seed
            if r["rank_reversal"][key]["example"]
        ]
        agg["rank_reversal"][key] = {
            "reversal_rate_probability": ci(
                ["rank_reversal", key, "reversal_rate_probability"]
            ),
            "reversal_rate_latent": ci(["rank_reversal", key, "reversal_rate_latent"]),
            "example": examples[0] if examples else None,
            "seeds_with_reversal": len(examples),
        }
    prof_lat = stack(["coverage", "coverage_profile_latent"])
    prof_prob = stack(["coverage", "coverage_profile_probability"])
    agg["coverage"] = {
        "gap_latent": ci(["coverage", "gap_latent"]),
        "gap_probability": ci(["coverage", "gap_probability"]),
        "marginal_latent": ci(["coverage", "marginal_latent"]),
        "marginal_probability": ci(["coverage", "marginal_probability"]),
        "profile_latent_mean": prof_lat.mean(axis=0),
        "profile_latent_lo": np.quantile(prof_lat, 0.025, axis=0),
        "profile_latent_hi": np.quantile(prof_lat, 0.975, axis=0),
        "profile_probability_mean": prof_prob.mean(axis=0),
        "profile_probability_lo": np.quantile(prof_prob, 0.025, axis=0),
        "profile_probability_hi": np.quantile(prof_prob, 0.975, axis=0),
        "any_infinite_quantile": any(
            r["coverage"]["infinite_quantile"] for r in per_seed
        ),
        "per_seed_gap_probability": [
            r["coverage"]["gap_probability"] for r in per_seed
        ],
        "per_seed_gap_latent": [r["coverage"]["gap_latent"] for r in per_seed],
        "stratum_edges_abs_logit_p0": np.mean(
            stack(["coverage", "stratum_edges_abs_logit_p0"]), axis=0
        ),
    }
    return agg


# ----------------------------------------------------------------- verdict


def read_g2_threshold() -> dict[str, Any]:
    """Parse the pre-registered G2 thresholds from PREREGISTRATION.md."""
    text = PREREG.read_text()
    row = next(line for line in text.splitlines() if line.startswith("| G2 "))
    cell = row.split("|")[3]
    m_prob = re.search(r"prob-space max-min gap > ([0-9.]+)", cell)
    m_lat = re.search(r"latent-space gap < ([0-9.]+)", cell)
    if not (m_prob and m_lat):
        raise RuntimeError(f"could not parse G2 threshold from: {cell!r}")
    return {
        "prob_gap_gt": float(m_prob.group(1)),
        "latent_gap_lt": float(m_lat.group(1)),
        "row": row.strip(),
    }


def verdict(agg: dict[str, Any]) -> dict[str, Any]:
    thr = read_g2_threshold()
    gp, gl = agg["coverage"]["gap_probability"], agg["coverage"]["gap_latent"]
    prob_ok = gp["mean"] > thr["prob_gap_gt"]
    lat_ok = gl["mean"] < thr["latent_gap_lt"]
    passed = bool(prob_ok and lat_ok and not agg["coverage"]["any_infinite_quantile"])
    p_prob = bootstrap_p_value(
        agg["coverage"]["per_seed_gap_probability"],
        threshold=thr["prob_gap_gt"],
        direction="greater",
    )
    p_lat = bootstrap_p_value(
        agg["coverage"]["per_seed_gap_latent"],
        threshold=thr["latent_gap_lt"],
        direction="less",
    )
    return {
        "p_values": {
            "prob_gap_gt_threshold": p_prob,
            "latent_gap_lt_threshold": p_lat,
            "gate": max(p_prob, p_lat),
        },
        "gate": GATE,
        "threshold": thr,
        "measured": {"prob_gap_mean": gp, "latent_gap_mean": gl},
        "prob_condition_met": bool(prob_ok),
        "latent_condition_met": bool(lat_ok),
        "ci_clears_threshold": bool(
            gp["lo"] > thr["prob_gap_gt"] and gl["hi"] < thr["latent_gap_lt"]
        ),
        "vacuous": bool(agg["coverage"]["any_infinite_quantile"]),
        "verdict": "PASS" if passed else "FAIL",
    }


# ------------------------------------------------------------------ figure


def make_figure(agg: dict[str, Any], cfg: Any, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    (axA, axB), (axC, axD) = axes

    # A: 1.6 link sweep
    grid = np.linspace(
        -2.5 * float(cfg.gate2.eta0_std), 2.5 * float(cfg.gate2.eta0_std), 200
    )
    for link in cfg.gate2.links:
        d = agg["link_sweep"][link]
        centers = 0.5 * (d["curve_bins"][1:] + d["curve_bins"][:-1])
        ln = axA.plot(centers, d["curve_mean"], "o", ms=3, label=f"{link} (measured)")
        jac = get_link(link).jacobian(torch.from_numpy(grid)).numpy()
        axA.plot(grid, jac, "-", lw=1, color=ln[0].get_color(), alpha=0.7)
    axA.set_xlabel(r"baseline latent $\eta_0$")
    axA.set_ylabel("observed effect / latent effect")
    axA.set_title(
        f"1.6 link sweep (δ={cfg.gate2.link_sweep.delta}); lines = predicted g'(η₀)"
    )
    axA.legend(fontsize=7)

    # B: 1.7 curve fit
    d01 = str(cfg.gate2.curve_fit.deltas[0])
    dmax = str(cfg.gate2.curve_fit.deltas[-1])
    xs = np.linspace(1e-3, 0.25, 200)
    axB.plot(xs, xs, "k--", lw=1, label="predicted: ratio = p₀(1−p₀)")
    for d, mk in ((d01, "."), (dmax, "x")):
        fit = agg["curve_fit"][d]
        axB.plot(
            xs,
            fit["c_hat"]["mean"] * xs ** fit["gamma_hat"]["mean"],
            "-",
            lw=1,
            label=f"δ={d}: γ̂={fit['gamma_hat']['mean']:.3f} [{fit['gamma_hat']['lo']:.3f}, {fit['gamma_hat']['hi']:.3f}]",
        )
    axB.set_xlabel(r"$p_0(1-p_0)$")
    axB.set_ylabel("|observed / latent effect|")
    axB.set_title("1.7 observed effect vs baseline (logit link)")
    axB.legend(fontsize=7)

    # C: 1.8 rank reversal
    keys = list(agg["rank_reversal"].keys())
    means = [agg["rank_reversal"][k]["reversal_rate_probability"]["mean"] for k in keys]
    los = [agg["rank_reversal"][k]["reversal_rate_probability"]["lo"] for k in keys]
    his = [agg["rank_reversal"][k]["reversal_rate_probability"]["hi"] for k in keys]
    lat = [agg["rank_reversal"][k]["reversal_rate_latent"]["mean"] for k in keys]
    xpos = np.arange(len(keys))
    axC.bar(
        xpos - 0.2,
        means,
        0.4,
        yerr=[np.subtract(means, los), np.subtract(his, means)],
        capsize=3,
        label="probability space",
    )
    axC.bar(xpos + 0.2, lat, 0.4, label="latent space")
    axC.set_xticks(xpos, [k.replace("_vs_", " vs ") for k in keys])
    axC.set_ylabel("cross-instance reversal rate")
    axC.set_xlabel("baseline pair (p₀ of b vs p₀ of a), |Δ_a| > |Δ_b|")
    axC.set_title("1.8 rank reversal rate")
    axC.legend(fontsize=7)

    # D: 1.9 conditional coverage
    cv = agg["coverage"]
    s = np.arange(1, int(cfg.n_strata) + 1)
    for name, color in (("probability", "C3"), ("latent", "C0")):
        m, lo, hi = (
            cv[f"profile_{name}_mean"],
            cv[f"profile_{name}_lo"],
            cv[f"profile_{name}_hi"],
        )
        axD.plot(
            s,
            m,
            "o-",
            color=color,
            label=f"{name} space (gap {cv[f'gap_{name}']['mean']:.3f})",
        )
        axD.fill_between(s, lo, hi, color=color, alpha=0.2)
    axD.axhline(
        1 - float(cfg.alpha),
        color="k",
        ls="--",
        lw=1,
        label=f"1 − α = {1 - float(cfg.alpha):.2f}",
    )
    edges = cv["stratum_edges_abs_logit_p0"]
    axD.set_xticks(
        s, [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(len(s))], fontsize=7
    )
    axD.set_xlabel(r"stratum of $|\mathrm{logit}\,p_0|$ (quintiles)")
    axD.set_ylabel("conditional coverage")
    axD.set_title(f"1.9 split-conformal coverage by baseline stratum (α={cfg.alpha})")
    axD.legend(fontsize=7)

    fig.suptitle(
        f"Gate 2 / Theorem 2 on the synthetic substrate, {agg['n_seeds']} seeds, mean and 95% CI"
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ------------------------------------------------------------------- table


def make_table(
    agg: dict[str, Any], v: dict[str, Any], cfg: Any, meta: dict[str, Any]
) -> str:
    L = []
    L.append("# Gate 2 results (ABLATIONS 1.6 to 1.9), synthetic substrate\n")
    L.append(
        f"git {meta['git_sha']}{' (DIRTY)' if meta['git_dirty'] else ''}, config {meta['config_hash']}, "
        f"{agg['n_seeds']} seeds, mean [95% bootstrap CI].\n"
    )
    L.append("## 1.6 link sweep\n")
    L.append(
        "| link | ratio range across |η₀| strata | R² vs predicted Jacobian | cross-instance reversal rate |"
    )
    L.append("|---|---|---|---|")
    for link in cfg.gate2.links:
        d = agg["link_sweep"][link]
        L.append(
            f"| {link} | {fmt_ci(d['ratio_range_across_strata'])} | {fmt_ci(d['r2_vs_predicted_jacobian'])} | {fmt_ci(d['reversal_rate'])} |"
        )
    L.append(
        "\nExpected: identity range and reversal rate at 0; non-identity R² > 0.9 (R² undefined for identity: ratio ≡ 1).\n"
    )
    L.append("## 1.7 curve fit, ratio = C·[p₀(1−p₀)]^γ, predicted γ = 1, C = 1\n")
    L.append("| δ | γ̂ | Ĉ | R² (log fit) | R² vs predicted form |")
    L.append("|---|---|---|---|---|")
    for d in cfg.gate2.curve_fit.deltas:
        f = agg["curve_fit"][str(d)]
        L.append(
            f"| {d} | {fmt_ci(f['gamma_hat'])} | {fmt_ci(f['c_hat'])} | {fmt_ci(f['r2_log_fit'])} | {fmt_ci(f['r2_vs_predicted_form'])} |"
        )
    L.append(
        "\n## 1.8 cross-instance rank reversal (|Δ_a| > |Δ_b|, |obs_a| < |obs_b|)\n"
    )
    L.append(
        "| baseline pair (b vs a) | reversal rate, probability | reversal rate, latent | example (seed 0) |"
    )
    L.append("|---|---|---|---|")
    for k, d in agg["rank_reversal"].items():
        ex = d["example"]
        ex_s = (
            "none"
            if ex is None
            else (
                f"a: p₀={ex['a']['p0']:.3f}, Δ={ex['a']['latent_effect']:+.3f}, obs={ex['a']['observed_effect']:+.4f}; "
                f"b: p₀={ex['b']['p0']:.3f}, Δ={ex['b']['latent_effect']:+.3f}, obs={ex['b']['observed_effect']:+.4f}"
            )
        )
        L.append(
            f"| {k.replace('_vs_', ' vs ')} | {fmt_ci(d['reversal_rate_probability'])} | {fmt_ci(d['reversal_rate_latent'])} | {ex_s} |"
        )
    L.append(
        "\n## 1.9 conditional coverage (split conformal, α = %s, %d strata of |logit p₀|)\n"
        % (cfg.alpha, cfg.n_strata)
    )
    cv = agg["coverage"]
    L.append(
        "| score space | max−min coverage gap | marginal coverage | per-stratum coverage (mean) |"
    )
    L.append("|---|---|---|---|")
    for name in ("probability", "latent"):
        prof = ", ".join(f"{x:.3f}" for x in cv[f"profile_{name}_mean"])
        L.append(
            f"| {name} | {fmt_ci(cv[f'gap_{name}'])} | {fmt_ci(cv[f'marginal_{name}'])} | {prof} |"
        )
    L.append(f"\nInfinite quantile in any seed: {cv['any_infinite_quantile']}\n")
    L.append("## Gate G2 verdict\n")
    L.append(f"Pre-registered row: `{v['threshold']['row']}`\n")
    L.append(
        f"- probability-space gap mean {v['measured']['prob_gap_mean']['mean']:.4f} > {v['threshold']['prob_gap_gt']}: **{v['prob_condition_met']}**"
    )
    L.append(
        f"- latent-space gap mean {v['measured']['latent_gap_mean']['mean']:.4f} < {v['threshold']['latent_gap_lt']}: **{v['latent_condition_met']}**"
    )
    L.append(f"- 95% CI bounds also clear both thresholds: {v['ci_clears_threshold']}")
    L.append(f"- vacuous (infinite quantile): {v['vacuous']}")
    L.append(f"\n**{v['verdict']}**\n")
    return "\n".join(L)


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run on a dirty tree (recorded in meta.json)",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    configure_torch(cfg)
    run = make_run_dir(cfg, "gate2_link", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    print(f"run dir: {run}")

    per_seed: list[dict[str, Any]] = []
    t0 = time.time()
    for seed in resolve_seeds(cfg):
        rng = np.random.default_rng([seed, 1606])
        r: dict[str, Any] = {
            "seed": seed,
            "link_sweep": run_link_sweep(cfg, seed, rng),
            "curve_fit": run_curve_fit(cfg, seed),
            "rank_reversal": run_rank_reversal(cfg, seed),
            "coverage": run_coverage(cfg, seed),
        }
        per_seed.append(r)
        c = r["coverage"]
        print(
            f"seed {seed}: prob gap {c['gap_probability']:.3f}, latent gap {c['gap_latent']:.3f}, "
            f"marginal prob {c['marginal_probability']:.3f} / latent {c['marginal_latent']:.3f}  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
        (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    agg = aggregate(per_seed, cfg)
    v = verdict(agg)

    fig_dir = pathlib.Path(cfg.output.figures)
    tab_dir = pathlib.Path(cfg.output.tables)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    make_figure(agg, cfg, fig_dir / "gate2_link")
    table = make_table(agg, v, cfg, meta)
    (tab_dir / "gate2_link.md").write_text(table)
    (run / "results.json").write_text(
        json.dumps(to_jsonable({"aggregate": agg, "verdict": v}), indent=1)
    )
    (run / "table.md").write_text(table)
    (run / "per_seed_values.json").write_text(
        json.dumps(
            {
                "gap_probability": {
                    str(r["seed"]): r["coverage"]["gap_probability"] for r in per_seed
                },
                "gap_latent": {
                    str(r["seed"]): r["coverage"]["gap_latent"] for r in per_seed
                },
                "marginal_probability": {
                    str(r["seed"]): r["coverage"]["marginal_probability"]
                    for r in per_seed
                },
                "marginal_latent": {
                    str(r["seed"]): r["coverage"]["marginal_latent"] for r in per_seed
                },
            },
            indent=1,
        )
    )
    write_tuning_budget(
        run,
        [
            {
                "model": "none",
                "configs_tried": 0,
                "epochs": 0,
                "gradient_steps": 0,
                "search_space": "no trained model: oracle substrate",
                "selection": "n/a",
            }
        ],
    )
    (run / "gate_stats.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    "gate": GATE,
                    "verdict": v["verdict"],
                    "p_value": v["p_values"]["gate"],
                    "p_values": v["p_values"],
                    "measured": v["measured"],
                    "threshold": v["threshold"],
                    "n_seeds": agg["n_seeds"],
                }
            ),
            indent=1,
        )
    )
    (run / "verdict.md").write_text(
        f"gate: {GATE}\ngit_sha: {meta['git_sha']}\ngit_dirty: {meta['git_dirty']}\n"
        f"config_hash: {meta['config_hash']}\ntimestamp_utc: {meta['timestamp_utc']}\n"
        f"measured: prob_gap={v['measured']['prob_gap_mean']['mean']:.4f} "
        f"[{v['measured']['prob_gap_mean']['lo']:.4f}, {v['measured']['prob_gap_mean']['hi']:.4f}], "
        f"latent_gap={v['measured']['latent_gap_mean']['mean']:.4f} "
        f"[{v['measured']['latent_gap_mean']['lo']:.4f}, {v['measured']['latent_gap_mean']['hi']:.4f}]\n"
        f"threshold: prob gap > {v['threshold']['prob_gap_gt']} AND latent gap < {v['threshold']['latent_gap_lt']}\n"
        f"verdict: {v['verdict']}\n"
    )
    for src in (fig_dir / "gate2_link.png", fig_dir / "gate2_link.pdf"):
        shutil.copy(src, run / src.name)

    print(table)
    print(
        f"\nfigure: {fig_dir / 'gate2_link.png'}\ntable: {tab_dir / 'gate2_link.md'}\nrun: {run}"
    )
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
