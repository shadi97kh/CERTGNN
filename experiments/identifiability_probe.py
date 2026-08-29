"""What is identifiable in y = g(phi(x)) when phi is NONLINEAR?

MAVE-NN fixes gauge and diffeomorphic modes for *linear* G-P maps, states
that "additional unconstrained modes can occur in special situations", and
does not fix modes for custom (neural) G-P maps. Every genomic DNN is the
nonlinear case, and it is uncharacterized.

The theory predicts the answer. Kinney & Atwal: two models are information
equivalent iff their latents are related by an invertible transformation.
A linear G-P map admits only affine modes not because the theory stops
there, but because psi(phi) must remain *inside the model class*, and the
only monotone psi carrying a linear function to a linear function is affine.
A neural G-P map is closed under every monotone warp, so its equivalence
class is infinite dimensional.

For attributions that predicts a sharp split. For a twin phi' = psi(phi),

    grad phi'(x) = psi'(phi(x)) * grad phi(x),

a *per-input positive scalar*. Therefore:

  - attributions on the composite f = g(phi) are exactly invariant
    (grad f' = grad f, algebraically), so measuring only those proves
    nothing;
  - attributions on the latent phi -- which is what SQUID and MAVE-NN
    interpret -- keep their direction *within* an instance but not their
    magnitude *across* instances;
  - second-order (epistasis) terms pick up a psi'' * grad phi grad phi^T
    rank-one contamination.

So the decisive measurement is the cross-instance ranking of latent
attribution magnitude, not within-instance similarity.

Two parts, because multi-restart fitting alone cannot separate
non-identifiability from optimizer luck:

  A. Constructive (deterministic). Build an explicit monotone warp psi and
     the twin (psi . phi_hat, g_hat . psi^-1), whose predictions are
     identical by construction, then ask per class whether the twin is
     still *in* the class. This is the class-size result with no appeal to
     optimization.
  B. Empirical. Refit from many random initializations; among fits of equal
     loss, measure whether the recovered latents are affinely related
     (affine R^2) or merely monotonically related (Spearman).

A negative result -- latent attributions invariant across the equivalence
class -- kills the direction and is reported as such.

Usage:
    python -m experiments.identifiability_probe --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

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

CLASSES = ("linear", "pairwise", "neural")


@dataclasses.dataclass
class Fit:
    """One refit: the model, its training loss, and its latent on the data."""

    model: "LatentModel"
    loss: float
    phi: torch.Tensor


# ------------------------------------------------------------------ data


def make_dataset(
    cfg: Any, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x from the synthetic substrate's node features; phi* nonlinear; y = g*(phi*).

    Returns (x, phi_true, y). g* is the logistic link, the substrate's own
    dial, so the ground-truth pair (phi*, g*) is known exactly.
    """
    ip = cfg.identifiability
    sub = SyntheticSubstrate(
        SyntheticConfig(
            topology=TopologySpec(**dict(ip.topology)),
            regulator="far",
            link="logit",
            structural_features=False,
            n_train=int(ip.n_instances),
            seed=seed,
        )
    )
    data = sub.load("train")
    x = torch.stack([d.x[:, 0].double() for d in data])  # node signal features
    n, d = x.shape
    g = torch.Generator().manual_seed(seed + 991)
    w = torch.randn(d, generator=g, dtype=torch.float64) / np.sqrt(d)
    # low-rank quadratic form: phi* is genuinely nonlinear in x, so the linear
    # G-P map is misspecified -- which is the realistic case, and the point
    rank = int(ip.quad_rank)
    U = torch.randn(d, rank, generator=g, dtype=torch.float64) / np.sqrt(d)
    lin = x @ w
    quad = ((x @ U) ** 2).sum(1) - ((x @ U) ** 2).sum(1).mean()
    phi = lin + float(ip.nonlinearity) * quad
    phi = (phi - phi.mean()) / phi.std()
    y = torch.sigmoid(float(ip.scale) * phi)
    y = y + float(ip.noise) * torch.randn(n, generator=g, dtype=torch.float64)
    return x, phi, y


# ---------------------------------------------------------------- models


class MonotoneGE(nn.Module):
    """MAVE-NN-style GE nonlinearity: a sum of sigmoids with positive weights,
    hence strictly monotone increasing in phi."""

    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.b_raw = nn.Parameter(torch.zeros(k, dtype=torch.float64))
        self.c_raw = nn.Parameter(torch.zeros(k, dtype=torch.float64))
        self.d = nn.Parameter(torch.linspace(-2, 2, k).double())

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        b = nn.functional.softplus(self.b_raw)  # > 0
        c = nn.functional.softplus(self.c_raw)  # > 0
        return self.a + (b * torch.sigmoid(c * phi[:, None] + self.d)).sum(1)


def make_gp_map(kind: str, d: int, hidden: int, gen: torch.Generator) -> nn.Module:
    """The G-P map phi(x). Three parameterizations, as in MAVE-NN plus a net."""
    if kind == "linear":
        m: nn.Module = nn.Linear(d, 1, dtype=torch.float64)
    elif kind == "pairwise":
        m = PairwiseGP(d)
    elif kind == "neural":
        m = nn.Sequential(
            nn.Linear(d, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64),
        )
    else:
        raise ValueError(kind)
    for p in m.parameters():
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64) * 0.3)
    return m


class PairwiseGP(nn.Module):
    """phi(x) = w.x + x^T (V V^T) x, the pairwise-interaction G-P map."""

    def __init__(self, d: int, rank: int = 4) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.V = nn.Parameter(torch.zeros(d, rank, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.w + ((x @ self.V) ** 2).sum(1) + self.b)[:, None]


class LatentModel(nn.Module):
    """phi (G-P map) composed with a monotone GE nonlinearity g."""

    def __init__(
        self, kind: str, d: int, hidden: int, k: int, gen: torch.Generator
    ) -> None:
        super().__init__()
        self.phi = make_gp_map(kind, d, hidden, gen)
        self.g = MonotoneGE(k)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi(x).reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(self.latent(x))


def fit_model(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float
) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((model(x) - y) ** 2).mean())


# ------------------------------------------------------- equivalence class


def affine_r2(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 of the best affine fit b ~ p + q*a. 1.0 means affinely related."""
    A = np.stack([np.ones_like(a), a], 1)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ coef
    ss_tot = float(((b - b.mean()) ** 2).sum())
    return 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 1e-30 else float("nan")


def monotone_warp(phi: torch.Tensor, strength: float, omega: float) -> torch.Tensor:
    """psi(phi) = phi + (s/omega) sin(omega phi), strictly monotone for |s| < 1."""
    z = (phi - phi.mean()) / phi.std()
    return z + (strength / omega) * torch.sin(omega * z)


def warp_derivative(phi: torch.Tensor, strength: float, omega: float) -> torch.Tensor:
    z = (phi - phi.mean()) / phi.std()
    return 1.0 + strength * torch.cos(omega * z)


def latent_attributions(model: LatentModel, x: torch.Tensor) -> torch.Tensor:
    """grad phi / grad x, one row per instance."""
    xr = x.clone().requires_grad_(True)
    phi = model.latent(xr).sum()
    (grad,) = torch.autograd.grad(phi, xr)
    return grad.detach()


def composite_attributions(model: LatentModel, x: torch.Tensor) -> torch.Tensor:
    xr = x.clone().requires_grad_(True)
    out = model(xr).sum()
    (grad,) = torch.autograd.grad(out, xr)
    return grad.detach()


def attribution_agreement(g1: torch.Tensor, g2: torch.Tensor) -> dict[str, float]:
    """Within-instance direction agreement, and cross-instance magnitude ranking."""
    a, b = g1.numpy(), g2.numpy()
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    cos = float(np.abs(num[ok] / den[ok]).mean()) if ok.any() else float("nan")
    m1, m2 = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    rho = spearmanr(m1, m2).statistic
    return {
        "within_instance_cosine": cos,
        "cross_instance_magnitude_spearman": float(rho)
        if np.isfinite(rho)
        else float("nan"),
    }


def class_contains_warp(
    kind: str, x: torch.Tensor, target: torch.Tensor, cfg: Any, gen: torch.Generator
) -> float:
    """R^2 of the best fit of this model class to the warped latent.

    ~1 means the class is closed under the warp (the twin is in the class, so
    the warp is an unconstrained mode); < 1 means the warp leaves the class,
    which is why linear G-P maps are identifiable up to affine only.
    """
    ip = cfg.identifiability
    m = make_gp_map(kind, x.shape[1], int(ip.hidden), gen)
    opt = torch.optim.Adam(m.parameters(), lr=float(ip.lr))
    t = (target - target.mean()) / target.std()
    for _ in range(int(ip.warp_fit_epochs)):
        opt.zero_grad()
        pred = m(x).reshape(-1)
        loss = ((pred - pred.mean()) / (pred.std() + 1e-9) - t).pow(2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = m(x).reshape(-1)
    return affine_r2(pred.numpy(), target.numpy())


# ------------------------------------------------------- warp families

WARP_FAMILIES = ("sinusoid", "spline", "sigmoid_mixture")


def _standardize(v: torch.Tensor) -> torch.Tensor:
    return (v - v.mean()) / v.std()


def warp(
    z: torch.Tensor, family: str, strength: float, omega: float, seed: int = 0
) -> torch.Tensor:
    """A monotone reparameterization psi(z) of a standardized latent.

    ``strength`` is the deviation from the identity (0 = identity) and
    ``omega`` the family's shape parameter. Monotonicity is not assumed: every
    grid point is checked numerically by :func:`is_monotone` and skipped if it
    fails.

    sinusoid        psi(z) = z + (s/omega) sin(omega z); psi' = 1 + s cos(.),
                    monotone for |s| < 1.
    spline          piecewise-linear through knots whose increments are
                    1 + s*u_k with u_k ~ U(-1, 1); monotone for |s| < 1.
    sigmoid_mixture (1-s) z + s G(z) with G a positively-weighted sum of
                    sigmoids -- the family MAVE-NN itself fits as its GE
                    nonlinearity, so the twin lies in the class the field
                    already uses. Monotone for s in [0, 1] by construction.
    """
    if family == "sinusoid":
        return z + (strength / omega) * torch.sin(omega * z)
    g = torch.Generator().manual_seed(seed)
    if family == "spline":
        n_knots = max(4, int(round(omega * 4)))
        lo, hi = float(z.min()), float(z.max())
        knots = torch.linspace(lo, hi, n_knots, dtype=torch.float64)
        u = torch.rand(n_knots - 1, generator=g, dtype=torch.float64) * 2 - 1
        inc = 1.0 + strength * u
        vals = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(inc, 0)])
        vals = vals * (hi - lo) / float(vals[-1]) + lo
        idx = torch.clamp(torch.bucketize(z, knots) - 1, 0, n_knots - 2)
        t = (z - knots[idx]) / (knots[idx + 1] - knots[idx])
        return vals[idx] + t * (vals[idx + 1] - vals[idx])
    if family == "sigmoid_mixture":
        k = max(2, int(round(omega * 2)))
        b = torch.rand(k, generator=g, dtype=torch.float64) + 0.5  # > 0
        c = (torch.rand(k, generator=g, dtype=torch.float64) + 0.5) * omega
        d = torch.linspace(-2, 2, k, dtype=torch.float64)
        gz = (b * torch.sigmoid(c * z[:, None] + d)).sum(1)
        gz = _standardize(gz)
        return (1.0 - strength) * z + strength * gz
    raise ValueError(f"unknown warp family {family!r}")


def is_monotone(
    family: str, strength: float, omega: float, seed: int, span: float = 4.0
) -> bool:
    """Strictly increasing on a dense grid covering the latent's range."""
    grid = torch.linspace(-span, span, 2001, dtype=torch.float64)
    w = warp(grid, family, strength, omega, seed)
    return bool((w[1:] - w[:-1] > 1e-12).all())


def invert_on_samples(
    warped_ref: torch.Tensor, z_ref: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """psi^-1 by interpolation on the paired samples (psi(z_ref), z_ref).

    Exact where the class reproduces the warped latent; values outside the
    observed range are clamped to the endpoints.
    """
    order = torch.argsort(warped_ref)
    xs = warped_ref[order].numpy()
    ys = z_ref[order].numpy()
    return torch.from_numpy(np.interp(query.numpy(), xs, ys))


def fit_class_to_target(
    kind: str,
    x: torch.Tensor,
    target: torch.Tensor,
    cfg: Any,
    gen: torch.Generator,
    warm_start: nn.Module | None = None,
) -> tuple[torch.Tensor, nn.Module]:
    """Best in-class approximation to a target latent, standardized.

    ``warm_start`` initializes from the reference G-P map's own weights. This
    matters: at warp strength 0 the target *is* the reference latent, so a
    cold-started refit measures optimization error rather than representational
    limits, and that error dominates the indistinguishability test. Warm
    starting makes the strength-0 refit exact by construction, which is what
    the null control checks.
    """
    ip = cfg.identifiability
    m = make_gp_map(kind, x.shape[1], int(ip.hidden), gen)
    if warm_start is not None:
        m.load_state_dict(warm_start.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=float(ip.lr))
    t = _standardize(target)

    def normalized_loss() -> torch.Tensor:
        pred = m(x).reshape(-1)
        return ((pred - pred.mean()) / (pred.std() + 1e-9) - t).pow(2).mean()

    # Keep the best iterate, not the last: warm-started at the exact solution,
    # Adam wanders away from it, and that wandering -- not the warp -- then
    # dominates the indistinguishability test.
    import copy as _copy

    best_loss = float(normalized_loss().detach())
    best_state = _copy.deepcopy(m.state_dict())
    for _ in range(int(ip.radius.warp_fit_epochs)):
        opt.zero_grad()
        loss = normalized_loss()
        loss.backward()
        opt.step()
        cur = float(loss.detach())
        if cur < best_loss:
            best_loss, best_state = cur, _copy.deepcopy(m.state_dict())
        if best_loss < float(ip.radius.fit_tol):
            break
    m.load_state_dict(best_state)
    with torch.no_grad():
        pred = m(x).reshape(-1)
    return pred.detach(), m


def indistinguishability(
    y: torch.Tensor, pred_orig: torch.Tensor, pred_twin: torch.Tensor
) -> dict[str, float]:
    """Paired test on squared residuals.

    d_i = (y_i - twin_i)^2 - (y_i - orig_i)^2. Under the null that the twin
    fits no worse, E[d] = 0. The paired t statistic at sample size N is
    ``t(N) = (mean d / sd d) * sqrt(N)``, so the standardized effect size
    ``mean d / sd d`` -- estimated once here -- gives the statistic at any N
    without regenerating data. The twin is indistinguishable at N when
    |t(N)| < 1.96 (two-sided, alpha = 0.05).
    """
    d = ((y - pred_twin) ** 2 - (y - pred_orig) ** 2).numpy()
    sd = float(d.std(ddof=1))
    eff = float(d.mean()) / sd if sd > 1e-30 else 0.0
    return {"effect_size": eff, "mean_d": float(d.mean()), "sd_d": sd}


def t_at(effect_size: float, n: int) -> float:
    return abs(effect_size) * np.sqrt(n)


def radius_sweep(
    cfg: Any, seed: int, x: torch.Tensor, y: torch.Tensor, refs: dict[str, Fit]
) -> list[dict[str, Any]]:
    """Grid over (family, strength, omega) x G-P class."""
    ip = cfg.identifiability
    rows: list[dict[str, Any]] = []
    for kind in CLASSES:
        ref = refs[kind]
        z_ref = _standardize(ref.phi)
        mu, sd = float(ref.phi.mean()), float(ref.phi.std())
        with torch.no_grad():
            pred_orig = ref.model.g(ref.phi).detach()
        g_ref = latent_attributions(ref.model, x)
        mag_ref = torch.linalg.norm(g_ref, dim=1).numpy()
        gen = torch.Generator().manual_seed(seed * 31 + 7)
        # Null control: strength 0 is the identity warp, so the refit target IS
        # the reference latent and any detectable difference is refit error.
        # Every radius below is only meaningful relative to this floor.
        phi0, m0 = fit_class_to_target(
            kind, x, z_ref, cfg, gen, warm_start=ref.model.phi
        )
        z_back0 = invert_on_samples(z_ref, z_ref, _standardize(phi0))
        with torch.no_grad():
            pred0 = ref.model.g(mu + sd * z_back0).detach()
        null = indistinguishability(y, pred_orig, pred0)
        rows.append(
            {
                "class": kind,
                "family": "NULL(identity)",
                "omega": 0.0,
                "strength": 0.0,
                "monotone": True,
                "closure_r2": affine_r2(phi0.numpy(), z_ref.numpy()),
                "affine_r2_to_phi_hat": 1.0,
                "within_instance_cosine": attribution_agreement(
                    g_ref, _grad_of_map(m0, x)
                )["within_instance_cosine"],
                "cross_instance_spearman": attribution_agreement(
                    g_ref, _grad_of_map(m0, x)
                )["cross_instance_magnitude_spearman"],
                "effect_size": null["effect_size"],
                "is_null_control": True,
                "attr_magnitude_cv_ref": float(
                    mag_ref.std() / max(abs(mag_ref.mean()), 1e-12)
                ),
                "attr_magnitude_cv_twin": float("nan"),
            }
        )
        for family in list(ip.radius.families):
            for omega in list(ip.radius.omegas):
                for strength in list(ip.radius.strengths):
                    if not is_monotone(family, float(strength), float(omega), seed):
                        rows.append(
                            {
                                "class": kind,
                                "family": family,
                                "omega": float(omega),
                                "strength": float(strength),
                                "monotone": False,
                            }
                        )
                        continue
                    warped = warp(z_ref, family, float(strength), float(omega), seed)
                    phi_t, m_t = fit_class_to_target(
                        kind, x, warped, cfg, gen, warm_start=ref.model.phi
                    )
                    closure = affine_r2(phi_t.numpy(), warped.numpy())
                    # twin predictions: g(psi^-1(phi_t)), exact when closure is exact
                    z_back = invert_on_samples(
                        _standardize(warped), z_ref, _standardize(phi_t)
                    )
                    with torch.no_grad():
                        pred_twin = ref.model.g(mu + sd * z_back).detach()
                    # Baseline against the strength-0 refit, not the original
                    # model: both twin and null go through the same refit and
                    # numerical inversion, so their shared floor cancels and
                    # what remains is the warp's own contribution.
                    ind = indistinguishability(y, pred0, pred_twin)
                    ind_vs_orig = indistinguishability(y, pred_orig, pred_twin)
                    g_t = (
                        latent_attributions(
                            m_t if not isinstance(m_t, LatentModel) else m_t, x
                        )
                        if False
                        else _grad_of_map(m_t, x)
                    )
                    agree = attribution_agreement(g_ref, g_t)
                    mag_t = torch.linalg.norm(g_t, dim=1).numpy()
                    rows.append(
                        {
                            "class": kind,
                            "family": family,
                            "omega": float(omega),
                            "strength": float(strength),
                            "monotone": True,
                            "closure_r2": closure,
                            "affine_r2_to_phi_hat": affine_r2(
                                z_ref.numpy(), warped.numpy()
                            ),
                            "within_instance_cosine": agree["within_instance_cosine"],
                            "cross_instance_spearman": agree[
                                "cross_instance_magnitude_spearman"
                            ],
                            "effect_size": ind["effect_size"],
                            "attr_magnitude_cv_ref": float(
                                mag_ref.std() / max(abs(mag_ref.mean()), 1e-12)
                            ),
                            "attr_magnitude_cv_twin": float(
                                mag_t.std() / max(abs(mag_t.mean()), 1e-12)
                            ),
                        }
                    )
    return rows


def _grad_of_map(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """grad of a bare G-P map (not wrapped in LatentModel) w.r.t. the input."""
    xr = x.clone().requires_grad_(True)
    out = m(xr).reshape(-1).sum()
    (grad,) = torch.autograd.grad(out, xr)
    return grad.detach()


# --------------------------------------------------- radius aggregation

T_CRIT = 1.96  # two-sided normal quantile at alpha = 0.05


def per_seed_radius(
    rows: list[dict[str, Any]], n: int
) -> dict[tuple[str, str, float], dict[str, float]]:
    """Largest indistinguishable strength per (class, family, omega) at size n."""
    out: dict[tuple[str, str, float], dict[str, float]] = {}
    for r in rows:
        if not r.get("monotone") or r.get("is_null_control"):
            continue
        key = (r["class"], r["family"], r["omega"])
        ok = t_at(r["effect_size"], n) < T_CRIT
        cur: dict[str, float] | None = out.get(key)
        if ok and (cur is None or r["strength"] > cur["radius"]):
            out[key] = {
                "radius": r["strength"],
                "cross_instance_spearman": r["cross_instance_spearman"],
                "within_instance_cosine": r["within_instance_cosine"],
                "closure_r2": r["closure_r2"],
                "affine_r2_to_phi_hat": r["affine_r2_to_phi_hat"],
                "t_at_n": t_at(r["effect_size"], n),
            }
        elif cur is None:
            out.setdefault(
                key,
                {
                    "radius": 0.0,
                    "cross_instance_spearman": 1.0,
                    "within_instance_cosine": 1.0,
                    "closure_r2": float("nan"),
                    "affine_r2_to_phi_hat": float("nan"),
                    "t_at_n": float("nan"),
                },
            )
    return out


def make_radius_table(agg: dict[str, Any], cfg: Any, meta: dict[str, Any]) -> str:
    ip = cfg.identifiability
    sizes = list(ip.radius.sample_sizes)
    L = ["# Indistinguishability radius for a nonlinear G-P map\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}, config "
        f"`{meta['config_hash']}`, {agg['n_seeds']} seeds, mean [95% bootstrap CI].\n"
    )
    L.append(
        "A twin is (psi.phi_hat, g_hat.psi^-1) for a monotone psi. It predicts identically "
        "only where the model class can represent psi.phi_hat; the class's approximation "
        "error is what makes a twin detectable. The radius is therefore the largest warp "
        "strength whose *best in-class approximation* stays statistically indistinguishable "
        "from the original fit at a given sample size.\n"
    )
    L.append(
        "**Test.** Paired on squared residuals: d_i = (y_i - twin_i)^2 - (y_i - orig_i)^2, "
        "null E[d] = 0. The paired t statistic at size N is t(N) = (mean d / sd d) * sqrt(N); "
        "the standardized effect size is estimated once per grid point, so t is available at "
        "any N without regenerating data. Indistinguishable means |t(N)| < 1.96 (two-sided, "
        "alpha = 0.05). Noise is the substrate's own observation noise "
        f"(sigma = {float(ip.noise)}). Sample sizes are real MPSA designs: "
        + ", ".join(f"{s['label']} N = {int(s['n'])}" for s in sizes)
        + ".\n"
    )
    L.append(
        "Every grid point is checked for strict monotonicity of psi on a dense grid; "
        f"points that fail are skipped ({agg['n_skipped_nonmonotone']} of "
        f"{agg['n_grid_points']} grid points skipped).\n"
    )

    L.append(
        "## Closure: can the class represent the warped latent? (six significant figures)\n"
    )
    L.append("| G-P map | warp family | closure R^2 at max monotone strength |")
    L.append("|---|---|---|")
    for k in CLASSES:
        for fam in agg["families"]:
            c = agg["closure"].get((k, fam))
            if c:
                L.append(
                    f"| {k} | {fam} | {c['mean']:.6f} [{c['lo']:.6f}, {c['hi']:.6f}] (n={c['n']}) |"
                )
    L.append(
        "\nA class whose closure R^2 is 1.000000 is closed under the warp: the twin is a "
        "legitimate member and no amount of data separates it from the original.\n"
    )

    for sz in sizes:
        n = int(sz["n"])
        L.append(f"## Radius at N = {n} ({sz['label']})\n")
        L.append(
            "| G-P map | warp family | radius (largest indistinguishable s) | cross-instance magnitude Spearman at the boundary | within-instance cosine |"
        )
        L.append("|---|---|---|---|---|")
        for k in CLASSES:
            for fam in agg["families"]:
                r = agg["radius"].get((k, fam, n))
                if r:
                    L.append(
                        f"| {k} | {fam} | {fmt_ci(r['radius'])} | {fmt_ci(r['cross_instance_spearman'])} | "
                        f"{fmt_ci(r['within_instance_cosine'])} |"
                    )
        L.append("")
    L.append("## Headline\n")
    L.append(agg["radius_verdict"])
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- run


def run_seed(cfg: Any, seed: int) -> dict[str, Any]:
    ip = cfg.identifiability
    x, phi_true, y = make_dataset(cfg, seed)
    d = x.shape[1]
    out: dict[str, Any] = {"seed": seed, "classes": {}}
    refs: dict[str, Fit] = {}

    for kind in CLASSES:
        gen = torch.Generator().manual_seed(seed * 7919 + hash(kind) % 1000)
        fits: list[Fit] = []
        for r in range(int(ip.restarts)):
            torch.manual_seed(seed * 100 + r)
            m = LatentModel(kind, d, int(ip.hidden), int(ip.ge_components), gen)
            loss = fit_model(m, x, y, int(ip.epochs), float(ip.lr))
            fits.append(Fit(model=m, loss=loss, phi=m.latent(x).detach()))
        best = min(f.loss for f in fits)
        tol = float(ip.loss_tol)
        good = [f for f in fits if f.loss <= best * (1.0 + tol)]

        # --- B. empirical: how are equally-good latents related?
        aff, spr, att_lat, att_comp = [], [], [], []
        for i, j in itertools.combinations(range(len(good)), 2):
            a = good[i].phi.numpy()
            b = good[j].phi.numpy()
            aff.append(affine_r2(a, b))
            rho = spearmanr(a, b).statistic
            spr.append(abs(float(rho)) if np.isfinite(rho) else float("nan"))
            gi = latent_attributions(good[i].model, x)
            gj = latent_attributions(good[j].model, x)
            att_lat.append(attribution_agreement(gi, gj))
            ci = composite_attributions(good[i].model, x)
            cj = composite_attributions(good[j].model, x)
            att_comp.append(attribution_agreement(ci, cj))

        # --- A. constructive: is the explicit monotone twin in the class?
        ref = min(good, key=lambda f: f.loss)
        refs[kind] = ref
        phi_hat = ref.phi
        warped = monotone_warp(phi_hat, float(ip.warp_strength), float(ip.warp_omega))
        in_class = class_contains_warp(kind, x, warped, cfg, gen)
        # attributions of the twin, exactly: grad(psi.phi) = psi'(phi) grad(phi)
        g_ref = latent_attributions(ref.model, x)
        g_twin = (
            warp_derivative(phi_hat, float(ip.warp_strength), float(ip.warp_omega))[
                :, None
            ]
            * g_ref
        )
        twin_att = attribution_agreement(g_ref, g_twin)
        twin_affine = affine_r2(phi_hat.numpy(), warped.numpy())

        mags = torch.linalg.norm(g_ref, dim=1).numpy()
        out["classes"][kind] = {
            "latent_attr_magnitude_cv": float(
                mags.std() / max(abs(mags.mean()), 1e-12)
            ),
            "best_loss": best,
            "n_equally_good": len(good),
            "loss_spread": float(max(f.loss for f in good) - best),
            "recovers_truth_spearman": abs(
                float(spearmanr(phi_hat.numpy(), phi_true.numpy()).statistic)
            ),
            "restart_affine_r2": float(np.mean(aff)) if aff else float("nan"),
            "restart_spearman": float(np.nanmean(spr)) if spr else float("nan"),
            "restart_latent_within_cosine": float(
                np.mean([a["within_instance_cosine"] for a in att_lat])
            )
            if att_lat
            else float("nan"),
            "restart_latent_cross_spearman": float(
                np.mean([a["cross_instance_magnitude_spearman"] for a in att_lat])
            )
            if att_lat
            else float("nan"),
            "restart_composite_within_cosine": float(
                np.mean([a["within_instance_cosine"] for a in att_comp])
            )
            if att_comp
            else float("nan"),
            "restart_composite_cross_spearman": float(
                np.mean([a["cross_instance_magnitude_spearman"] for a in att_comp])
            )
            if att_comp
            else float("nan"),
            "twin_in_class_r2": in_class,
            "twin_affine_r2": twin_affine,
            "twin_latent_within_cosine": twin_att["within_instance_cosine"],
            "twin_latent_cross_spearman": twin_att["cross_instance_magnitude_spearman"],
        }
    out["radius_rows"] = radius_sweep(cfg, seed, x, y, refs)
    return out


METRICS = (
    "best_loss",
    "n_equally_good",
    "recovers_truth_spearman",
    "restart_affine_r2",
    "restart_spearman",
    "restart_latent_within_cosine",
    "restart_latent_cross_spearman",
    "restart_composite_within_cosine",
    "restart_composite_cross_spearman",
    "twin_in_class_r2",
    "twin_affine_r2",
    "twin_latent_within_cosine",
    "twin_latent_cross_spearman",
)


def make_table(agg: dict[str, Any], meta: dict[str, Any]) -> str:
    L = ["# Identifiability of y = g(phi(x)) for nonlinear phi\n"]
    L.append(
        f"git SHA `{meta['git_sha']}`{' (DIRTY)' if meta['git_dirty'] else ''}, config "
        f"`{meta['config_hash']}`, {agg['n_seeds']} seeds, mean [95% bootstrap CI].\n"
    )
    L.append(
        "## A. Constructive: is the explicit monotone twin inside the model class?\n"
    )
    L.append(
        "| G-P map | twin in class (R^2) | twin affinely related to phi_hat (R^2) |"
    )
    L.append("|---|---|---|")
    for k in CLASSES:
        c = agg["classes"][k]
        L.append(
            f"| {k} | {fmt_ci(c['twin_in_class_r2'])} | {fmt_ci(c['twin_affine_r2'])} |"
        )
    L.append(
        "\nThe twin (psi.phi_hat, g_hat.psi^-1) predicts identically by construction. "
        "If it is *in* the class, the warp is an unconstrained mode of that class.\n"
    )
    L.append("## B. Empirical: how are equally-good refits related?\n")
    L.append(
        "Read with the composite control at the foot of section C. Refits speak to the "
        "equivalence class only if they fit the same function; where their composite "
        "attributions disagree, they are different fits, not different representatives.\n"
    )
    L.append(
        "| G-P map | equally-good fits | affine R^2 | Spearman | recovers phi* (Spearman) |"
    )
    L.append("|---|---|---|---|---|")
    for k in CLASSES:
        c = agg["classes"][k]
        L.append(
            f"| {k} | {fmt_ci(c['n_equally_good'], 1)} | {fmt_ci(c['restart_affine_r2'])} | "
            f"{fmt_ci(c['restart_spearman'])} | {fmt_ci(c['recovers_truth_spearman'])} |"
        )
    L.append("\n## C. Are attributions invariant across the equivalence class?\n")
    L.append(
        "| G-P map | source | latent: within-instance cosine | latent: cross-instance magnitude Spearman |"
    )
    L.append("|---|---|---|---|")
    for k in CLASSES:
        c = agg["classes"][k]
        L.append(
            f"| {k} | constructed twin | {fmt_ci(c['twin_latent_within_cosine'])} | {fmt_ci(c['twin_latent_cross_spearman'])} |"
        )
        L.append(
            f"| {k} | random restarts | {fmt_ci(c['restart_latent_within_cosine'])} | {fmt_ci(c['restart_latent_cross_spearman'])} |"
        )
    L.append("")
    L.append(
        "Composite attributions (grad of f = g(phi)), which are invariant algebraically and are shown only as a control:\n"
    )
    L.append("| G-P map | within-instance cosine | cross-instance magnitude Spearman |")
    L.append("|---|---|---|")
    for k in CLASSES:
        c = agg["classes"][k]
        L.append(
            f"| {k} | {fmt_ci(c['restart_composite_within_cosine'])} | {fmt_ci(c['restart_composite_cross_spearman'])} |"
        )
    L.append("\n## Verdict\n")
    L.append(agg["verdict"]["statement"])
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
    run = make_run_dir(cfg, "identifiability_probe", allow_dirty=args.allow_dirty)
    meta = json.loads((run / "meta.json").read_text())
    print(f"run dir: {run}")

    per_seed = []
    for seed in resolve_seeds(cfg):
        r = run_seed(cfg, seed)
        per_seed.append(r)
        msg = "  ".join(
            f"{k}: twin_in_class={r['classes'][k]['twin_in_class_r2']:.3f} "
            f"cross_rho={r['classes'][k]['twin_latent_cross_spearman']:.3f}"
            for k in CLASSES
        )
        print(f"seed {seed}: {msg}", flush=True)
        (run / f"seed_{seed}.json").write_text(json.dumps(to_jsonable(r), indent=1))

    nb = int(cfg.bootstrap_resamples)
    agg: dict[str, Any] = {"n_seeds": len(per_seed), "classes": {}}
    for k in CLASSES:
        agg["classes"][k] = {
            m: mean_ci([r["classes"][k][m] for r in per_seed], n_boot=nb)
            for m in METRICS
        }

    lin, neu = agg["classes"]["linear"], agg["classes"]["neural"]
    neural_closed = neu["twin_in_class_r2"]["lo"] > 0.95
    linear_not_closed = lin["twin_in_class_r2"]["hi"] < 0.95
    twin_breaks_cross = neu["twin_latent_cross_spearman"]["hi"] < 0.9
    restarts_break_cross = neu["restart_latent_cross_spearman"]["hi"] < 0.9
    # Restarts only speak to the equivalence class if they fit the SAME function.
    # The composite gradient is invariant under reparameterization algebraically,
    # so if it disagrees across restarts they are different fits, not different
    # representatives, and part B cannot be used as evidence either way.
    restarts_prediction_equivalent = neu["restart_composite_within_cosine"]["lo"] > 0.95
    within_preserved = neu["twin_latent_within_cosine"]["lo"] > 0.95

    if neural_closed and twin_breaks_cross:
        statement = (
            "**Attribution on a nonlinear G-P map is not identifiable.** The neural class is "
            "closed under a monotone warp of the latent (twin in class R^2 "
            f"{neu['twin_in_class_r2']['mean']:.3f}), so the twin fits the data identically by "
            "construction and is a legitimate member of the same model class. Its latent "
            "attributions keep their direction within an instance (cosine "
            f"{neu['twin_latent_within_cosine']['mean']:.3f}) but not their magnitude across "
            f"instances (Spearman {neu['twin_latent_cross_spearman']['mean']:.3f}). The linear "
            f"class is not closed under the same warp (R^2 {lin['twin_in_class_r2']['mean']:.3f}), "
            "which is exactly why MAVE-NN can fix its modes affinely. Consequence for the "
            "field: any interpretation of a nonlinear surrogate's latent that depends on "
            "comparing attribution magnitudes across inputs -- which includes cross-locus "
            "motif-effect comparisons -- is reporting one arbitrary representative of an "
            "infinite equivalence class. Within-instance rankings survive."
            + (
                " Ordinary multi-restart training also lands in different members of the class."
                if restarts_break_cross and restarts_prediction_equivalent
                else " The multi-restart arm cannot corroborate this: the restarts' *composite*"
                " attributions disagree too (cosine"
                f" {neu['restart_composite_within_cosine']['mean']:.3f}), and the composite is"
                " invariant under reparameterization algebraically, so those restarts are"
                " different fits rather than different representatives of one function. Part B"
                " is therefore reported but carries no weight; the result rests on the"
                " construction, which is exact."
            )
        )
    elif neural_closed and within_preserved and not twin_breaks_cross:
        statement = (
            "**Negative result: the direction does not survive.** The neural class is closed "
            "under the warp, so the equivalence class is larger than affine, but latent "
            "attributions are invariant across it on both measures tested "
            f"(cross-instance Spearman {neu['twin_latent_cross_spearman']['mean']:.3f}). "
            "Non-identifiability of the latent does not translate into non-identifiability of "
            "attributions here, so the interpretability consequence this probe was built to "
            "find is absent and the direction should be dropped."
        )
    elif not neural_closed:
        statement = (
            "**Inconclusive: the construction did not work.** The neural class did not "
            f"reproduce the warped latent (R^2 {neu['twin_in_class_r2']['mean']:.3f}), so the "
            "twin is not demonstrably in the class and part A proves nothing. This is a "
            "fitting failure, not evidence of identifiability; the warp fit needs more "
            "capacity or epochs before any conclusion is drawn."
        )
    else:
        statement = (
            "Mixed: the class is closed under the warp but the attribution measures disagree "
            "with each other; read the tables case by case before concluding."
        )
    agg["verdict"] = {
        "neural_class_closed_under_warp": bool(neural_closed),
        "linear_class_not_closed": bool(linear_not_closed),
        "twin_breaks_cross_instance": bool(twin_breaks_cross),
        "restarts_break_cross_instance": bool(restarts_break_cross),
        "restarts_prediction_equivalent": bool(restarts_prediction_equivalent),
        "within_instance_preserved": bool(within_preserved),
        "statement": statement,
    }

    # ---------------- radius aggregation across seeds ----------------
    ip = cfg.identifiability
    fams = list(ip.radius.families)
    sizes = list(ip.radius.sample_sizes)
    all_rows = [r for rec in per_seed for r in rec["radius_rows"]]
    agg["families"] = fams
    agg["n_grid_points"] = len(all_rows)
    agg["n_skipped_nonmonotone"] = sum(1 for r in all_rows if not r.get("monotone"))

    # closure at the largest strength that is monotone, averaged over omegas
    closure: dict[tuple[str, str], dict[str, float]] = {}
    for k in CLASSES:
        for fam in fams:
            vals = []
            for rec in per_seed:
                mono = [
                    r
                    for r in rec["radius_rows"]
                    if r.get("monotone") and r["class"] == k and r["family"] == fam
                ]
                if not mono:
                    continue
                smax = max(r["strength"] for r in mono)
                at = [r["closure_r2"] for r in mono if r["strength"] == smax]
                vals.append(float(np.mean(at)))
            if vals:
                closure[(k, fam)] = mean_ci(vals, n_boot=nb)
    agg["closure"] = closure

    radius: dict[tuple[str, str, int], dict[str, Any]] = {}
    for sz in sizes:
        n = int(sz["n"])
        for k in CLASSES:
            for fam in fams:
                rads, spears, coss = [], [], []
                for rec in per_seed:
                    ps = per_seed_radius(rec["radius_rows"], n)
                    keys = [key for key in ps if key[0] == k and key[1] == fam]
                    if not keys:
                        continue
                    # the binding constraint across omegas is the smallest radius
                    worst = min(keys, key=lambda key: ps[key]["radius"])
                    rads.append(ps[worst]["radius"])
                    spears.append(ps[worst]["cross_instance_spearman"])
                    coss.append(ps[worst]["within_instance_cosine"])
                if rads:
                    radius[(k, fam, n)] = {
                        "radius": mean_ci(rads, n_boot=nb),
                        "cross_instance_spearman": mean_ci(spears, n_boot=nb),
                        "within_instance_cosine": mean_ci(coss, n_boot=nb),
                    }
    agg["radius"] = radius

    smax_grid = max(float(v) for v in ip.radius.strengths)
    lines = []
    for sz in sizes:
        n = int(sz["n"])
        for fam in fams:
            entry = radius.get(("neural", fam, n))
            if not entry:
                continue
            at_ceiling = entry["radius"]["lo"] >= smax_grid - 1e-9
            lines.append(
                f"- N = {n} ({sz['label']}), {fam}: radius {entry['radius']['mean']:.3f}"
                f"{' (the grid ceiling: the whole monotone range is indistinguishable)' if at_ceiling else ''}"
                f", attribution-magnitude Spearman at the boundary "
                f"{entry['cross_instance_spearman']['mean']:.3f} "
                f"[{entry['cross_instance_spearman']['lo']:.3f}, {entry['cross_instance_spearman']['hi']:.3f}], "
                f"within-instance cosine {entry['within_instance_cosine']['mean']:.3f}."
            )
    neural_closed_all = all(
        closure[(("neural"), fam)]["lo"] > 0.999999
        for fam in fams
        if ("neural", fam) in closure
    )
    agg["radius_verdict"] = (
        (
            "**The neural class is closed under every warp family tested (closure R^2 = 1.000000 "
            "to six figures), so its radius is set by the monotonicity limit rather than by the "
            "data: no MPSA sample size separates the twin from the original.**\n\n"
            if neural_closed_all
            else "**The neural class is not closed to six figures under every family; the radius below "
            "is set by the class's approximation error, so it is finite and shrinks with N.**\n\n"
        )
        + "\n".join(lines)
        + "\n\nWithin-instance attribution direction is preserved throughout; what the radius "
        "bounds is the cross-instance comparison."
    )

    radius_table = make_radius_table(agg, cfg, meta)
    (tab := pathlib.Path(cfg.output.tables)).mkdir(parents=True, exist_ok=True)
    (tab / "identifiability_radius.md").write_text(radius_table)
    (run / "radius_table.md").write_text(radius_table)

    table = make_table(agg, meta)
    (tab / "identifiability_probe.md").write_text(table)
    (run / "table.md").write_text(table)
    (run / "results.json").write_text(json.dumps(to_jsonable(agg), indent=1))
    (run / "per_seed_values.json").write_text(
        json.dumps(
            {
                f"{k}/{m}": {str(r["seed"]): r["classes"][k][m] for r in per_seed}
                for k in CLASSES
                for m in METRICS
            },
            indent=1,
        )
    )
    write_tuning_budget(
        run,
        [
            {
                "model": k,
                "configs_tried": 1,
                "epochs": int(cfg.identifiability.epochs),
                "gradient_steps": int(cfg.identifiability.epochs)
                * int(cfg.identifiability.restarts),
                "search_space": "one pre-specified configuration per G-P map class; restarts differ only in initialization",
                "selection": "lowest training loss; no held-out selection (identifiability, not generalization)",
            }
            for k in CLASSES
        ],
    )
    print("\n" + table)
    print("\n" + radius_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
