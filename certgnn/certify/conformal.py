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


def _as_f64(x: torch.Tensor) -> np.ndarray:
    """Flatten any tensor / array-like to a 1-D float64 numpy array."""
    return torch.as_tensor(x).detach().cpu().to(torch.float64).numpy().reshape(-1)


def weighted_conformal_quantile(
    scores: torch.Tensor,
    cal_weights: torch.Tensor,
    test_weight: float = 1.0,
    alpha: float = 0.1,
) -> float:
    """Weighted split-conformal quantile under covariate shift.

    Implements Tibshirani, Foygel Barber, Candes and Ramdas (NeurIPS 2019),
    "Conformal Prediction Under Covariate Shift". The weights are likelihood
    ratios ``w(x) = dP_test(x) / dP_train(x)`` of the covariate
    distributions, evaluated at each calibration point (``cal_weights``) and
    at the test point (``test_weight``). Only ratios between weights matter:
    rescaling all of them by a common positive constant leaves the result
    unchanged.

    The returned value is the ``(1 - alpha)`` quantile of the discrete
    distribution

        sum_i p_i * delta_{V_i}  +  p_{n+1} * delta_{+inf},

    with ``p_i = w_i / (sum_j w_j + w_test)`` and
    ``p_{n+1} = w_test / (sum_j w_j + w_test)``. The atom at ``+inf`` carries
    the test point's own mass and is the weighted analogue of the ``(n + 1)``
    finite-sample correction in :func:`split_conformal_quantile`. Dropping
    that atom, or normalising by ``sum_j w_j`` alone, inflates every
    calibration atom's mass, selects a too-small quantile, and yields
    systematic undercoverage that grows with ``w_test``. Under site shift
    such an artifact is indistinguishable from a genuine degradation of
    conformal validity, which is exactly the effect the leave-one-site-out
    ablations are meant to measure.

    All arithmetic is performed in float64. With unit weights the result is
    bit-identical to :func:`split_conformal_quantile`.

    Parameters
    ----------
    scores : torch.Tensor
        Nonconformity scores ``V_1, ..., V_n`` of the calibration set.
    cal_weights : torch.Tensor
        Likelihood ratios ``w(X_i)`` at the calibration points, one per
        score. Must be strictly positive.
    test_weight : float, default 1.0
        Likelihood ratio ``w(X_{n+1})`` at the test point. Must be strictly
        positive. The default is only correct when there is no shift.
    alpha : float, default 0.1
        Miscoverage level, strictly inside ``(0, 1)``.

    Returns
    -------
    float
        The quantile, or ``float("inf")`` when the ``(1 - alpha)`` level falls
        inside the ``+inf`` atom, i.e. the calibration set cannot certify at
        this ``alpha`` (the analogue of ``k > n`` in split conformal).

    Raises
    ------
    ValueError
        If the calibration set is empty, ``cal_weights`` does not match
        ``scores`` in length, any weight is non-positive or NaN, or ``alpha``
        lies outside ``(0, 1)``.
    """
    v = _as_f64(scores)
    w = _as_f64(cal_weights)
    w_test = float(test_weight)
    n = v.size
    if n == 0:
        raise ValueError("empty calibration set")
    if w.size != n:
        raise ValueError(
            f"cal_weights has {w.size} entries but scores has {n}; "
            "one likelihood ratio per calibration score is required"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    # ``not (x > 0)`` also rejects NaN, which compares False against everything.
    if not (bool(np.all(w > 0.0)) and w_test > 0.0):
        raise ValueError(
            "weights must be strictly positive likelihood ratios "
            "dP_test / dP_train (got a non-positive or NaN weight)"
        )

    order = np.argsort(v, kind="stable")
    v_sorted, w_sorted = v[order], w[order]
    # Compare *unnormalised* cumulative mass against the (1 - alpha) share of
    # the total mass, calibration atoms plus the test atom. Dividing each
    # weight by the total first would introduce rounding that breaks the exact
    # reduction to split conformal under unit weights (e.g. nine additions of
    # 0.1 fall short of 0.9 in binary floating point).
    total = float(np.sum(w_sorted)) + w_test
    threshold = (1.0 - alpha) * total
    cum = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cum, threshold, side="left"))
    if idx >= n:
        return float("inf")  # level sits inside the +inf atom
    return float(v_sorted[idx])


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
