"""Tests for ``certgnn.certify.conformal``.

The weighted quantile follows Tibshirani et al. (NeurIPS 2019). The exact
reduction to split conformal under unit weights is the regression guard for
the three original defects (hardcoded test weight, missing test weight in the
normaliser, missing ``+inf`` atom); the Monte Carlo tests check the coverage
guarantee itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from certgnn.certify.conformal import (
    split_conformal_quantile,
    weighted_conformal_quantile,
)

ALPHAS = [0.05, 0.1, 0.2]
N_TRIALS = 1000


def _two_se_band(alpha: float, n_trials: int) -> tuple[float, float]:
    se = math.sqrt(alpha * (1.0 - alpha) / n_trials)
    return 1.0 - alpha - 2.0 * se, 1.0 - alpha + 2.0 * se


# --------------------------------------------------------------------------- #
# weighted_conformal_quantile
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", ALPHAS)
def test_weighted_reduces_to_split_under_unit_weights(alpha: float) -> None:
    """Regression guard: with ``w_i = w_test = 1`` the weighted quantile must
    equal ``split_conformal_quantile`` *exactly* (same float, or both inf).

    The sweep over ``n`` deliberately includes sizes where ``(n + 1)(1 - a)``
    is an integer (the ``ceil`` boundary), sizes where ``k > n`` (both must
    return inf), and sizes where naive normalised cumsums drift below the
    level in binary floating point (e.g. ``n = 9, alpha = 0.1``)."""
    gen = torch.Generator().manual_seed(0)
    for n in [1, 2, 3, 4, 5, 9, 10, 18, 19, 20, 39, 40, 50, 99, 100, 199, 200, 257]:
        scores = torch.randn(n, generator=gen)
        expected = split_conformal_quantile(scores, alpha)
        got = weighted_conformal_quantile(scores, torch.ones(n), 1.0, alpha)
        assert got == expected, (n, alpha, got, expected)  # inf == inf is True


def test_weighted_matches_hand_computation() -> None:
    """Three atoms with masses 1, 1, 2 plus a unit test atom (total 5)."""
    scores = torch.tensor([1.0, 2.0, 3.0])
    weights = torch.tensor([1.0, 1.0, 2.0])
    # cumulative unnormalised mass in sorted order: [1, 2, 4]; threshold 5(1-a)
    assert weighted_conformal_quantile(scores, weights, 1.0, 0.7) == 2.0  # 1.5
    assert weighted_conformal_quantile(scores, weights, 1.0, 0.5) == 3.0  # 2.5
    assert weighted_conformal_quantile(scores, weights, 1.0, 0.3) == 3.0  # 3.5
    assert math.isinf(weighted_conformal_quantile(scores, weights, 1.0, 0.1))  # 4.5


def test_weighted_is_invariant_to_common_rescaling() -> None:
    """Only ratios matter. Power-of-two scaling keeps arithmetic exact."""
    gen = torch.Generator().manual_seed(1)
    scores = torch.randn(50, generator=gen)
    weights = torch.rand(50, generator=gen) + 0.1
    for alpha in ALPHAS:
        base = weighted_conformal_quantile(scores, weights, 0.7, alpha)
        scaled = weighted_conformal_quantile(scores, 4.0 * weights, 4.0 * 0.7, alpha)
        assert scaled == base


def test_weighted_is_computed_in_float64() -> None:
    """float32 inputs must not be summed in float32: a calibration set of
    2**24 + 1 unit weights is exactly representable only in float64."""
    n = 2**24 + 1
    scores = torch.arange(n, dtype=torch.float32)
    q = weighted_conformal_quantile(
        scores, torch.ones(n, dtype=torch.float32), 1.0, 0.5
    )
    # threshold = 0.5 * (n + 1) = 2**23 + 1 -> index 2**23 -> score 2**23
    assert q == float(2**23)


def test_returns_inf_when_test_weight_dominates() -> None:
    """When the test atom alone holds more than ``alpha`` of the total mass,
    the ``(1 - alpha)`` level lands inside the ``+inf`` atom."""
    gen = torch.Generator().manual_seed(2)
    n = 100
    scores = torch.randn(n, generator=gen)
    assert math.isfinite(weighted_conformal_quantile(scores, torch.ones(n), 1.0, 0.1))
    # test mass = 1e6 / (1e6 + 100) > 0.1
    assert math.isinf(weighted_conformal_quantile(scores, torch.ones(n), 1e6, 0.1))
    # exact boundary: test mass = alpha exactly is still inside the atom only
    # when strictly greater; here 12/(100+12) > 0.1 so inf, 11/111 < 0.1 finite
    assert math.isinf(weighted_conformal_quantile(scores, torch.ones(n), 12.0, 0.1))
    assert math.isfinite(weighted_conformal_quantile(scores, torch.ones(n), 11.0, 0.1))


def test_rejects_nonpositive_weights() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3])
    good = torch.ones(3)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, torch.tensor([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, torch.tensor([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, torch.tensor([1.0, float("nan"), 1.0]))
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, good, test_weight=0.0)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, good, test_weight=-2.0)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, good, test_weight=float("nan"))


def test_weighted_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        weighted_conformal_quantile(torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError):
        weighted_conformal_quantile(torch.ones(3), torch.ones(2))
    for alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            weighted_conformal_quantile(torch.ones(3), torch.ones(3), 1.0, alpha)


@pytest.mark.slow
def test_weighted_coverage_under_known_covariate_shift() -> None:
    """Monte Carlo check of the Tibshirani et al. guarantee.

    Covariate shift with a closed-form likelihood ratio:
        train  X ~ N(0, 1),   test  X ~ N(mu, 1)
        w(x) = dP_test/dP_train (x) = exp(mu * x - mu^2 / 2).
    Scores are heteroscedastic in X, ``V = |Z| * exp(X / 2)``, so the shift
    changes the score distribution and an unweighted quantile undercovers.

    Empirical coverage of the weighted quantile must lie within two standard
    errors of ``1 - alpha`` over ``N_TRIALS`` trials. As a power check, the
    unweighted split quantile on the same draws must fall *below* the band.

    ``mu`` is kept mild (0.5) on purpose: heavier shifts shrink the effective
    calibration size and the correct estimator then over-covers by more than
    the band allows, which would be a false failure."""
    alpha, mu, n_cal = 0.1, 0.5, 100
    rng = np.random.default_rng(0)

    def lr(x: np.ndarray) -> np.ndarray:
        return np.exp(mu * x - mu**2 / 2.0)

    covered_weighted = 0
    covered_unweighted = 0
    for _ in range(N_TRIALS):
        x_cal = rng.standard_normal(n_cal)
        v_cal = np.abs(rng.standard_normal(n_cal)) * np.exp(x_cal / 2.0)
        x_test = rng.normal(mu, 1.0)
        v_test = abs(rng.standard_normal()) * math.exp(x_test / 2.0)

        scores = torch.from_numpy(v_cal)
        q_w = weighted_conformal_quantile(
            scores, torch.from_numpy(lr(x_cal)), float(lr(x_test)), alpha
        )
        q_u = split_conformal_quantile(scores, alpha)
        covered_weighted += v_test <= q_w
        covered_unweighted += v_test <= q_u

    lo, hi = _two_se_band(alpha, N_TRIALS)
    cov_w = covered_weighted / N_TRIALS
    cov_u = covered_unweighted / N_TRIALS
    assert (
        lo <= cov_w <= hi
    ), f"weighted coverage {cov_w:.4f} outside [{lo:.4f}, {hi:.4f}]"
    assert (
        cov_u < lo
    ), f"unweighted coverage {cov_u:.4f} should undercover; test has no power"


# --------------------------------------------------------------------------- #
# split_conformal_quantile
# --------------------------------------------------------------------------- #


def test_split_returns_inf_when_k_exceeds_n() -> None:
    # n = 5, alpha = 0.05: k = ceil(6 * 0.95) = 6 > 5
    assert math.isinf(split_conformal_quantile(torch.randn(5), 0.05))
    with pytest.raises(ValueError):
        split_conformal_quantile(torch.empty(0))


@pytest.mark.slow
@pytest.mark.parametrize("alpha", ALPHAS)
def test_split_coverage_iid(alpha: float) -> None:
    """Exchangeable calibration and test scores: empirical coverage must lie
    within two standard errors of ``1 - alpha`` over ``N_TRIALS`` trials.
    (For continuous scores the exact value is ``k / (n + 1)``.)"""
    n_cal = 100
    rng = np.random.default_rng(3)
    covered = 0
    for _ in range(N_TRIALS):
        v_cal = np.abs(rng.standard_normal(n_cal))
        v_test = abs(rng.standard_normal())
        covered += v_test <= split_conformal_quantile(torch.from_numpy(v_cal), alpha)
    lo, hi = _two_se_band(alpha, N_TRIALS)
    cov = covered / N_TRIALS
    assert (
        lo <= cov <= hi
    ), f"coverage {cov:.4f} outside [{lo:.4f}, {hi:.4f}] at alpha={alpha}"
