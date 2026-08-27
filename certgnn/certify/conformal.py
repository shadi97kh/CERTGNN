"""Split, Mondrian, and weighted conformal calibration for explanation fidelity."""

from __future__ import annotations

import numpy as np
import torch


def split_conformal_quantile(scores: torch.Tensor, alpha: float = 0.1) -> float:
    """Standard split-conformal quantile with the finite-sample correction."""
    n = scores.numel()
    if n == 0:
        raise ValueError("empty calibration set")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")  # cannot certify at this alpha with this n
    return float(torch.sort(scores).values[k - 1])


def mondrian_quantiles(
    scores: torch.Tensor, strata: torch.Tensor, alpha: float = 0.1
) -> dict[int, float]:
    """Per-stratum quantiles. Restores conditional coverage (Theorem 2 iii)."""
    return {
        int(s): split_conformal_quantile(scores[strata == s], alpha)
        for s in torch.unique(strata)
    }


def weighted_conformal_quantile(
    scores: torch.Tensor, weights: torch.Tensor, alpha: float = 0.1
) -> float:
    """Covariate-shift-corrected quantile (Tibshirani et al. 2019)."""
    order = torch.argsort(scores)
    s, w = scores[order], weights[order]
    w = w / (w.sum() + 1.0)
    cum = torch.cumsum(w, 0)
    idx = int(torch.searchsorted(cum, torch.tensor(1.0 - alpha)).item())
    idx = min(idx, s.numel() - 1)
    return float(s[idx])


def empirical_coverage(
    test_scores: torch.Tensor,
    q: float | dict[int, float],
    strata: torch.Tensor | None = None,
) -> float:
    if isinstance(q, dict):
        assert strata is not None
        covered = torch.stack(
            [test_scores[i] <= q[int(strata[i])] for i in range(test_scores.numel())]
        )
    else:
        covered = test_scores <= q
    return float(covered.float().mean())


def conditional_coverage_gap(
    test_scores: torch.Tensor, strata: torch.Tensor, q: float | dict[int, float]
) -> dict[int, float]:
    """Per-stratum coverage. A flat profile is the goal; a sloped one is the
    Theorem 2(iii) failure."""
    out = {}
    for s in torch.unique(strata):
        m = strata == s
        qq = q[int(s)] if isinstance(q, dict) else q
        out[int(s)] = float((test_scores[m] <= qq).float().mean())
    return out
