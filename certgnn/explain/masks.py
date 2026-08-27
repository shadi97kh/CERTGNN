"""Soft node masks optimized for latent-space epsilon-sufficiency.

Assumption 5 (soft masks). Every explanation mask in this project is a
continuous vector in [0, 1]^n. The Jacobian arguments in Theorems 3, 4 and 5
differentiate the target's latent output with respect to the mask; a hard
binary mask has no such derivative, so those theorems say nothing about it.
Hard masking exists here only so the ablation can show what the theory
buys (ABLATIONS C.4), and it cannot be reached without a flag that warns.

Model interface. Core code must not depend on a model class, so the model
is a callable ``latent_fn(mask) -> Tensor`` returning the target's *latent*
(logit-space) output under node mask ``mask``; the caller binds the graph,
features and target. It must return logits, never probabilities (see
``certgnn.certify.link`` and Theorem 2).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import torch

from certgnn.topology.resistance import resistance_to_target

LatentFn = Callable[[torch.Tensor], torch.Tensor]
"""``latent_fn(mask) -> 0-d tensor``: target latent output under a node mask."""

MaskType = Literal["soft", "hard"]
WarmStart = Literal["resistance_ball", "cold"]


class HardMaskWarning(UserWarning):
    """Emitted whenever a hard binary mask is produced."""


warnings.simplefilter("always", HardMaskWarning)

HARD_MASK_MESSAGE = (
    "HARD BINARY MASK REQUESTED. This violates Assumption 5 (soft masks in "
    "[0, 1]^n). The Jacobian arguments in Theorem 3 (certified radius), "
    "Theorem 4 (resistance-ball bound) and Theorem 5 (rewiring budget) do not "
    "hold for this mask, and no certificate derived from it is valid. Hard "
    "masking exists only for the negative-control ablation (ABLATIONS C.4)."
)


@dataclass(frozen=True)
class MaskResult:
    """Output of the mask optimizer.

    Attributes
    ----------
    mask : torch.Tensor
        Node mask in [0, 1]^n; the target entry is 1. Binary iff
        ``mask_type == "hard"``.
    latent_gap : float
        Achieved ``|z(1) - z(mask)|`` in latent space, evaluated on the
        returned mask (after binarization for hard masks).
    iterations : int
        Optimizer steps taken.
    initial_gap : float
        Latent gap of the initial mask, before any optimization. The
        resistance-ball warm start is expected to lower this relative to a
        cold start when the ball contains the minimal set (Theorem 4).
    first_sufficient : int | None
        First iteration at which ``gap <= eps`` held, or ``None`` if never.
        This is the warm-start vs cold-start comparison the ablation reports;
        ``iterations`` also reflects the annealing schedule.
    converged : bool
        Whether the stopping rule fired before ``max_iter``.
    sufficient : bool
        ``latent_gap <= eps``.
    mask_type : str
    warm_start : str
    sparsity : float
        Sum of the mask over candidate nodes (the L1 term at the optimum).
    """

    mask: torch.Tensor
    latent_gap: float
    iterations: int
    initial_gap: float
    first_sufficient: int | None
    converged: bool
    sufficient: bool
    mask_type: str
    warm_start: str
    sparsity: float


def _as_scalar(z: torch.Tensor) -> torch.Tensor:
    if z.numel() != 1:
        raise ValueError(
            f"latent_fn must return one latent value, got shape {tuple(z.shape)}"
        )
    return z.reshape(())


def latent_gap(latent_fn: LatentFn, mask: torch.Tensor) -> float:
    """``|z(1) - z(mask)|`` in latent space (Theorem 2: never in probability)."""
    full = torch.ones_like(mask)
    with torch.no_grad():
        return float((_as_scalar(latent_fn(full)) - _as_scalar(latent_fn(mask))).abs())


def default_candidates(num_nodes: int, target: int) -> torch.Tensor:
    return torch.tensor([v for v in range(num_nodes) if v != target], dtype=torch.long)


# ------------------------------------------------------------------ inits


def cold_init(num_nodes: int, target: int, value: float = 0.5) -> torch.Tensor:
    """Uniform soft initialization; the cold-start arm of ABLATIONS 3.x."""
    if not 0.0 < value < 1.0:
        raise ValueError("cold init value must lie strictly inside (0, 1)")
    m = torch.full((num_nodes,), value, dtype=torch.float64)
    m[target] = 1.0
    return m


def resistance_ball_init(
    G: nx.Graph,
    target: int,
    *,
    radius: float | None = None,
    ball_size: int | None = None,
    inside: float = 0.9,
    outside: float = 0.1,
) -> torch.Tensor:
    """Warm start from the resistance ball of Theorem 4.

    Nodes with effective resistance to the target at most ``radius`` start at
    ``inside``; all others at ``outside``. Give either ``radius`` or
    ``ball_size`` (the ball then contains the ``ball_size`` nearest
    non-target nodes). Both init levels are strictly inside (0, 1): the warm
    start is a soft mask, not a hard pre-selection.
    """
    if (radius is None) == (ball_size is None):
        raise ValueError("give exactly one of radius or ball_size")
    if not (0.0 < outside < inside < 1.0):
        raise ValueError("need 0 < outside < inside < 1")
    R = resistance_to_target(G, target)
    nodes = sorted(G.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError("graph nodes must be labelled 0..n-1")
    if radius is None:
        assert ball_size is not None
        others = sorted(R[v] for v in nodes if v != target)
        if ball_size < 1 or ball_size > len(others):
            raise ValueError(f"ball_size must be in [1, {len(others)}]")
        radius = others[ball_size - 1]
    m = torch.full((len(nodes),), outside, dtype=torch.float64)
    for v in nodes:
        if R[v] <= radius + 1e-12:
            m[v] = inside
    m[target] = 1.0
    return m


# -------------------------------------------------------------- optimizer


def optimize_soft_mask(
    latent_fn: LatentFn,
    init: torch.Tensor,
    target: int,
    eps: float,
    *,
    candidates: Sequence[int] | torch.Tensor | None = None,
    sparsity_weight: float = 0.05,
    gap_weight: float = 10.0,
    lr: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-3,
    patience: int = 10,
    dual_every: int = 20,
    slack: float = 0.05,
    lr_final_ratio: float = 0.01,
    mask_type: MaskType = "soft",
    warm_start_name: str = "custom",
) -> MaskResult:
    """Minimize ``sparsity_weight * ||m||_1`` subject to ``gap <= eps``.

    The constraint is handled by an augmented Lagrangian on
    ``v = relu(gap - eps)``: the loss is
    ``sparsity_weight * ||m||_1 + mu * v + (gap_weight / 2) * v^2`` and the
    multiplier ``mu`` is raised by ``gap_weight * v`` every ``dual_every``
    steps. A fixed quadratic penalty alone settles where the L1 gradient
    balances the penalty gradient, i.e. slightly *outside* the constraint;
    the multiplier removes that bias so the returned mask is actually
    epsilon-sufficient when the problem is feasible. The constraint is
    enforced at ``(1 - slack) * eps`` so that the optimizer's hovering around
    its boundary stays inside the true tolerance; ``sufficient`` in the
    result is judged against ``eps`` itself. The learning rate decays
    exponentially to ``lr * lr_final_ratio`` at ``max_iter`` so the optimizer
    stops hovering on the constraint boundary; convergence is declared when
    the mask is sufficient and ``max |delta m| < tol`` for ``patience``
    consecutive steps.

    ``gap = |z(1) - z(m)|`` is the latent fidelity gap; the mask is
    parameterized as ``sigmoid(theta)`` so it stays strictly inside (0, 1)
    during optimization. Non-candidate nodes and the target are fixed at 1.

    Parameters
    ----------
    latent_fn : callable
        See module docstring. Must be differentiable in the mask.
    init : torch.Tensor
        Initial soft mask in (0, 1)^n (target entry ignored, forced to 1).
    target : int
    eps : float
        Latent-space sufficiency tolerance.
    candidates : sequence of int, optional
        Nodes allowed in the explanation; default all but the target.
    mask_type : {"soft", "hard"}
        ``"hard"`` binarizes the optimized soft mask at 0.5 and emits
        :class:`HardMaskWarning`. Use only for the negative-control ablation.

    Returns
    -------
    MaskResult
    """
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if not 0.0 <= slack < 1.0:
        raise ValueError("slack must lie in [0, 1)")
    eps_inner = (1.0 - slack) * eps
    n = init.numel()
    if not (0 <= target < n):
        raise ValueError("target out of range")
    cand = (
        default_candidates(n, target)
        if candidates is None
        else torch.as_tensor(candidates, dtype=torch.long).reshape(-1)
    )
    if bool((cand == target).any()):
        raise ValueError("the target cannot be a candidate")
    if bool((init[cand] <= 0).any()) or bool((init[cand] >= 1).any()):
        raise ValueError("init values on candidates must lie strictly inside (0, 1)")

    fixed = torch.ones(n, dtype=torch.float64)
    theta = torch.logit(init[cand].to(torch.float64).clone()).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    if not 0.0 < lr_final_ratio <= 1.0:
        raise ValueError("lr_final_ratio must lie in (0, 1]")
    sched = torch.optim.lr_scheduler.ExponentialLR(
        opt, gamma=lr_final_ratio ** (1.0 / max(max_iter, 1))
    )

    with torch.no_grad():
        z_full = _as_scalar(latent_fn(fixed)).detach().to(torch.float64)

    def assemble(th: torch.Tensor) -> torch.Tensor:
        m = fixed.clone()
        m[cand] = torch.sigmoid(th)
        return m

    still = 0
    converged = False
    iterations = 0
    first_sufficient: int | None = None
    mu = 0.0
    prev_m = assemble(theta).detach()
    initial_gap = latent_gap(latent_fn, prev_m)
    for it in range(1, max_iter + 1):
        opt.zero_grad()
        m = assemble(theta)
        l1 = m[cand].sum()
        gap = (z_full - _as_scalar(latent_fn(m)).to(torch.float64)).abs()
        viol = torch.relu(gap - eps_inner)
        loss = sparsity_weight * l1 + mu * viol + 0.5 * gap_weight * viol**2
        loss.backward()
        opt.step()
        sched.step()
        iterations = it
        gap_now = float(gap.detach())
        if gap_now <= eps and first_sufficient is None:
            first_sufficient = it
        cur_m = m.detach()
        if gap_now <= eps and float((cur_m - prev_m).abs().max()) < tol:
            still += 1
            if still >= patience:
                converged = True
                break
        else:
            still = 0
        if it % dual_every == 0 and gap_now > eps_inner:
            mu += gap_weight * float(viol.detach())
        prev_m = cur_m

    with torch.no_grad():
        mask = assemble(theta).detach()

    if mask_type == "hard":
        warnings.warn(HARD_MASK_MESSAGE, HardMaskWarning, stacklevel=2)
        mask = binarize(mask, acknowledge_assumption_5=True)
    elif mask_type != "soft":
        raise ValueError("mask_type must be 'soft' or 'hard'")

    gap_final = latent_gap(latent_fn, mask)
    return MaskResult(
        mask=mask,
        latent_gap=gap_final,
        iterations=iterations,
        initial_gap=initial_gap,
        first_sufficient=first_sufficient,
        converged=converged,
        sufficient=gap_final <= eps + 1e-12,
        mask_type=mask_type,
        warm_start=warm_start_name,
        sparsity=float(mask[cand].sum()),
    )


def binarize(
    mask: torch.Tensor,
    threshold: float = 0.5,
    *,
    acknowledge_assumption_5: bool = False,
) -> torch.Tensor:
    """Threshold a soft mask into a hard one. Refuses without the flag."""
    if not acknowledge_assumption_5:
        raise ValueError(
            "binarize() produces a hard mask, which violates Assumption 5; pass "
            "acknowledge_assumption_5=True only for the negative-control ablation"
        )
    warnings.warn(HARD_MASK_MESSAGE, HardMaskWarning, stacklevel=2)
    return (mask >= threshold).to(mask.dtype)


def explain(
    latent_fn: LatentFn,
    G: nx.Graph,
    target: int,
    eps: float,
    *,
    warm_start: WarmStart = "resistance_ball",
    radius: float | None = None,
    ball_size: int | None = None,
    cold_value: float = 0.5,
    mask_type: MaskType = "soft",
    **opt_kwargs: object,
) -> MaskResult:
    """Warm-start (Theorem 4 resistance ball) or cold-start, then optimize.

    With ``warm_start="resistance_ball"`` and neither ``radius`` nor
    ``ball_size`` given, the ball holds the nearest half of the non-target
    nodes.
    """
    n = G.number_of_nodes()
    if warm_start == "resistance_ball":
        if radius is None and ball_size is None:
            ball_size = max(1, (n - 1) // 2)
        init = resistance_ball_init(G, target, radius=radius, ball_size=ball_size)
    elif warm_start == "cold":
        init = cold_init(n, target, cold_value)
    else:
        raise ValueError("warm_start must be 'resistance_ball' or 'cold'")
    return optimize_soft_mask(
        latent_fn,
        init,
        target,
        eps,
        mask_type=mask_type,
        warm_start_name=warm_start,
        **opt_kwargs,  # type: ignore[arg-type]
    )
