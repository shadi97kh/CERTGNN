"""Faithfulness metrics, always reported in probability AND latent space.

Theorem 2 says probability-space faithfulness scores inherit the link's
Jacobian ``p(1-p)`` and therefore disagree systematically with latent-space
scores across instances with different baseline rates. Every metric here
returns a :class:`PairedScore` holding both. There is deliberately no
function that returns the probability-space number alone: that is the
mistake the paper is about.

Normalized AOPC (Edin et al., ACL 2025) is included as a *baseline*
correction: it rescales AOPC by the best and worst achievable AOPC for the
instance. It is computed in both spaces too, so the paper can show that a
range normalization in probability space is not the same fix as the link
correction.

Every metric runs ``degenerate_explanation_check`` on the explanation mask
and surfaces the flag in its return value.

Perturbation curves (AOPC, deletion, insertion) are defined by stepwise
removal or insertion of nodes in rank order. Those steps are evaluation
perturbations of the *model input*, not explanation masks, so they do not
touch Assumption 5; the explanation itself stays soft.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from certgnn.eval.sanity import degenerate_explanation_check
from certgnn.explain.masks import (
    HARD_MASK_MESSAGE,
    HardMaskWarning,
    LatentFn,
    default_candidates,
)


@dataclass(frozen=True)
class PairedScore:
    """One metric in both output spaces.

    Attributes
    ----------
    metric : str
    probability : float
        Computed on ``sigmoid(z)``.
    latent : float
        Computed on ``z`` (logits).
    degenerate : bool
        ``degenerate_explanation_check`` on the explanation mask.
    curve_probability, curve_latent : tuple[float, ...]
        The perturbation curve, when the metric has one.
    """

    metric: str
    probability: float
    latent: float
    degenerate: bool
    curve_probability: tuple[float, ...] = ()
    curve_latent: tuple[float, ...] = ()


def _latent(latent_fn: LatentFn, mask: torch.Tensor) -> float:
    with torch.no_grad():
        z = latent_fn(mask)
    if z.numel() != 1:
        raise ValueError(
            f"latent_fn must return one latent value, got shape {tuple(z.shape)}"
        )
    return float(z.reshape(()))


def _prob(z: float) -> float:
    return float(torch.sigmoid(torch.tensor(z, dtype=torch.float64)))


def _check_mask(mask: torch.Tensor, target: int) -> None:
    if mask.dim() != 1:
        raise ValueError("mask must be 1-d")
    if bool((mask < 0).any()) or bool((mask > 1).any()):
        raise ValueError("mask entries must lie in [0, 1]")
    if float(mask[target]) != 1.0:
        raise ValueError("the target entry of the mask must be 1")


def _candidates(
    mask: torch.Tensor, target: int, candidates: Sequence[int] | torch.Tensor | None
) -> torch.Tensor:
    if candidates is None:
        c = default_candidates(mask.numel(), target)
    else:
        c = torch.as_tensor(candidates, dtype=torch.long).reshape(-1)
        if bool((c == target).any()):
            raise ValueError("the target cannot be a candidate")
    vals = mask[c]
    if bool(((vals == 0.0) | (vals == 1.0)).all()):
        # Evaluating a hard mask is allowed (the negative-control ablation
        # needs it) but never silent: Assumption 5 is violated upstream.
        warnings.warn(
            "Evaluating a HARD (binary) explanation mask. " + HARD_MASK_MESSAGE,
            HardMaskWarning,
            stacklevel=3,
        )
    return c


def _ranking(mask: torch.Tensor, cand: torch.Tensor) -> list[int]:
    """Candidates ordered by mask value, descending; ties by node id."""
    vals = mask[cand].tolist()
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], int(cand[i])))
    return [int(cand[i]) for i in order]


def _removal_mask(n: int, removed: Sequence[int]) -> torch.Tensor:
    m = torch.ones(n, dtype=torch.float64)
    m[list(removed)] = 0.0
    return m


def _keep_mask(
    n: int, target: int, cand: torch.Tensor, kept: Sequence[int]
) -> torch.Tensor:
    m = torch.ones(n, dtype=torch.float64)
    m[cand] = 0.0
    m[list(kept)] = 1.0
    m[target] = 1.0
    return m


# ------------------------------------------------------------ ERASER-style


def comprehensiveness(
    latent_fn: LatentFn,
    mask: torch.Tensor,
    target: int,
    *,
    candidates: Sequence[int] | torch.Tensor | None = None,
) -> PairedScore:
    """``f(x) - f(x without the explanation)`` using the soft complement mask.

    Larger is better. The complement of a soft mask ``m`` on candidates is
    ``1 - m``; the target and non-candidates stay at 1.
    """
    _check_mask(mask, target)
    cand = _candidates(mask, target, candidates)
    comp = torch.ones_like(mask, dtype=torch.float64)
    comp[cand] = 1.0 - mask[cand].to(torch.float64)
    z_full = _latent(latent_fn, torch.ones_like(comp))
    z_comp = _latent(latent_fn, comp)
    return PairedScore(
        "comprehensiveness",
        _prob(z_full) - _prob(z_comp),
        z_full - z_comp,
        degenerate_explanation_check(mask),
    )


def sufficiency(
    latent_fn: LatentFn,
    mask: torch.Tensor,
    target: int,
    *,
    candidates: Sequence[int] | torch.Tensor | None = None,
) -> PairedScore:
    """``f(x) - f(explanation only)`` using the soft mask itself. Smaller is
    better; its absolute value in latent space is the latent fidelity gap."""
    _check_mask(mask, target)
    cand = _candidates(mask, target, candidates)
    keep = torch.ones_like(mask, dtype=torch.float64)
    keep[cand] = mask[cand].to(torch.float64)
    z_full = _latent(latent_fn, torch.ones_like(keep))
    z_keep = _latent(latent_fn, keep)
    return PairedScore(
        "sufficiency",
        _prob(z_full) - _prob(z_keep),
        z_full - z_keep,
        degenerate_explanation_check(mask),
    )


# -------------------------------------------------------- perturbation curves


def _aopc_curves(
    latent_fn: LatentFn, n: int, order: Sequence[int], k: int
) -> tuple[list[float], list[float]]:
    """``f(x) - f(x minus top-j)`` for j = 1..k, in both spaces."""
    z_full = _latent(latent_fn, torch.ones(n, dtype=torch.float64))
    p_full = _prob(z_full)
    cp: list[float] = []
    cl: list[float] = []
    for j in range(1, k + 1):
        z = _latent(latent_fn, _removal_mask(n, order[:j]))
        cp.append(p_full - _prob(z))
        cl.append(z_full - z)
    return cp, cl


def aopc(
    latent_fn: LatentFn,
    mask: torch.Tensor,
    target: int,
    *,
    candidates: Sequence[int] | torch.Tensor | None = None,
    k: int | None = None,
) -> PairedScore:
    """Area over the perturbation curve: mean over ``j = 1..k`` of
    ``f(x) - f(x minus the top-j ranked nodes)``."""
    _check_mask(mask, target)
    cand = _candidates(mask, target, candidates)
    order = _ranking(mask, cand)
    k = len(order) if k is None else min(k, len(order))
    if k < 1:
        raise ValueError("need at least one candidate")
    cp, cl = _aopc_curves(latent_fn, mask.numel(), order, k)
    return PairedScore(
        "aopc",
        sum(cp) / k,
        sum(cl) / k,
        degenerate_explanation_check(mask),
        tuple(cp),
        tuple(cl),
    )


def deletion_auc(
    latent_fn: LatentFn,
    mask: torch.Tensor,
    target: int,
    *,
    candidates: Sequence[int] | torch.Tensor | None = None,
) -> PairedScore:
    """Area under ``f`` as ranked nodes are deleted one at a time from the
    full input (Petsiuk et al. 2018). Lower is better. Trapezoid rule over
    a unit x-axis, ``K + 1`` points."""
    _check_mask(mask, target)
    cand = _candidates(mask, target, candidates)
    order = _ranking(mask, cand)
    n = mask.numel()
    zs = [
        _latent(latent_fn, _removal_mask(n, order[:j])) for j in range(len(order) + 1)
    ]
    ps = [_prob(z) for z in zs]
    return PairedScore(
        "deletion_auc",
        _trapz(ps),
        _trapz(zs),
        degenerate_explanation_check(mask),
        tuple(ps),
        tuple(zs),
    )


def insertion_auc(
    latent_fn: LatentFn,
    mask: torch.Tensor,
    target: int,
    *,
    candidates: Sequence[int] | torch.Tensor | None = None,
) -> PairedScore:
    """Area under ``f`` as ranked nodes are inserted one at a time into the
    empty candidate set. Higher is better."""
    _check_mask(mask, target)
    cand = _candidates(mask, target, candidates)
    order = _ranking(mask, cand)
    n = mask.numel()
    zs = [
        _latent(latent_fn, _keep_mask(n, target, cand, order[:j]))
        for j in range(len(order) + 1)
    ]
    ps = [_prob(z) for z in zs]
    return PairedScore(
        "insertion_auc",
        _trapz(ps),
        _trapz(zs),
        degenerate_explanation_check(mask),
        tuple(ps),
        tuple(zs),
    )


def _trapz(ys: Sequence[float]) -> float:
    if len(ys) < 2:
        return float(ys[0]) if ys else float("nan")
    h = 1.0 / (len(ys) - 1)
    return h * (sum(ys) - 0.5 * (ys[0] + ys[-1]))


# ---------------------------------------------------------- normalized AOPC


def _aopc_bounds(
    latent_fn: LatentFn,
    n: int,
    cand: torch.Tensor,
    k: int,
    *,
    exact_max_nodes: int,
    beam_width: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(lo, hi) AOPC over all rankings, in (probability, latent) spaces.

    Exact dynamic program over subsets when ``len(cand) <= exact_max_nodes``
    (``2^m`` model calls); otherwise beam search of width ``beam_width``
    over rank prefixes, as in Edin et al. (2025).
    """
    nodes = [int(v) for v in cand]
    z_full = _latent(latent_fn, torch.ones(n, dtype=torch.float64))
    p_full = _prob(z_full)
    cache: dict[frozenset[int], tuple[float, float]] = {}

    def gain(S: frozenset[int]) -> tuple[float, float]:
        if S not in cache:
            z = _latent(latent_fn, _removal_mask(n, sorted(S)))
            cache[S] = (p_full - _prob(z), z_full - z)
        return cache[S]

    if len(nodes) <= exact_max_nodes:
        # best[S] = extreme over orderings of S of the summed prefix gains
        best_hi: dict[frozenset[int], tuple[float, float]] = {frozenset(): (0.0, 0.0)}
        best_lo: dict[frozenset[int], tuple[float, float]] = {frozenset(): (0.0, 0.0)}
        for size in range(1, k + 1):
            for combo in itertools.combinations(nodes, size):
                S = frozenset(combo)
                g = gain(S)
                hi_p = max(best_hi[S - {v}][0] for v in S) + g[0]
                hi_l = max(best_hi[S - {v}][1] for v in S) + g[1]
                lo_p = min(best_lo[S - {v}][0] for v in S) + g[0]
                lo_l = min(best_lo[S - {v}][1] for v in S) + g[1]
                best_hi[S] = (hi_p, hi_l)
                best_lo[S] = (lo_p, lo_l)
        full_sets = [frozenset(c) for c in itertools.combinations(nodes, k)]
        hi = (
            max(best_hi[S][0] for S in full_sets) / k,
            max(best_hi[S][1] for S in full_sets) / k,
        )
        lo = (
            min(best_lo[S][0] for S in full_sets) / k,
            min(best_lo[S][1] for S in full_sets) / k,
        )
        return lo, hi

    def beam(maximize: bool, space: int) -> float:
        beams: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
        for _ in range(k):
            expanded: list[tuple[float, tuple[int, ...]]] = []
            for total, prefix in beams:
                used = set(prefix)
                for v in nodes:
                    if v in used:
                        continue
                    g = gain(frozenset(prefix) | {v})[space]
                    expanded.append((total + g, (*prefix, v)))
            expanded.sort(key=lambda t: t[0], reverse=maximize)
            beams = expanded[:beam_width]
        return beams[0][0] / k

    hi = (beam(True, 0), beam(True, 1))
    lo = (beam(False, 0), beam(False, 1))
    return lo, hi


def normalized_aopc(
    latent_fn: LatentFn,
    mask: torch.Tensor,
    target: int,
    *,
    candidates: Sequence[int] | torch.Tensor | None = None,
    k: int | None = None,
    exact_max_nodes: int = 10,
    beam_width: int = 5,
) -> PairedScore:
    """Normalized AOPC (Edin et al., ACL 2025): ``(AOPC - lo) / (hi - lo)``
    with ``lo``/``hi`` the worst/best AOPC achievable by any ranking of the
    same instance. Lies in [0, 1] when the bounds are exact. Returns ``nan``
    in a space where ``hi == lo`` (the instance cannot discriminate rankings
    in that space).

    This is a per-instance *range* correction. Theorem 2's link correction
    is a *Jacobian* correction; the two are computed separately here so the
    paper can show they are not the same fix.
    """
    _check_mask(mask, target)
    cand = _candidates(mask, target, candidates)
    order = _ranking(mask, cand)
    k = len(order) if k is None else min(k, len(order))
    if k < 1:
        raise ValueError("need at least one candidate")
    cp, cl = _aopc_curves(latent_fn, mask.numel(), order, k)
    raw = (sum(cp) / k, sum(cl) / k)
    lo, hi = _aopc_bounds(
        latent_fn,
        mask.numel(),
        cand,
        k,
        exact_max_nodes=exact_max_nodes,
        beam_width=beam_width,
    )

    def norm(i: int) -> float:
        span = hi[i] - lo[i]
        return float("nan") if abs(span) < 1e-15 else (raw[i] - lo[i]) / span

    return PairedScore(
        "normalized_aopc",
        norm(0),
        norm(1),
        degenerate_explanation_check(mask),
        tuple(cp),
        tuple(cl),
    )


# ------------------------------------------------------------------ report


def faithfulness_report(
    latent_fn: LatentFn,
    mask: torch.Tensor,
    target: int,
    *,
    candidates: Sequence[int] | torch.Tensor | None = None,
    k: int | None = None,
    exact_max_nodes: int = 10,
    beam_width: int = 5,
) -> dict[str, PairedScore]:
    """All metrics, each as a probability/latent pair."""
    return {
        "comprehensiveness": comprehensiveness(
            latent_fn, mask, target, candidates=candidates
        ),
        "sufficiency": sufficiency(latent_fn, mask, target, candidates=candidates),
        "aopc": aopc(latent_fn, mask, target, candidates=candidates, k=k),
        "deletion_auc": deletion_auc(latent_fn, mask, target, candidates=candidates),
        "insertion_auc": insertion_auc(latent_fn, mask, target, candidates=candidates),
        "normalized_aopc": normalized_aopc(
            latent_fn,
            mask,
            target,
            candidates=candidates,
            k=k,
            exact_max_nodes=exact_max_nodes,
            beam_width=beam_width,
        ),
    }
