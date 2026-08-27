import pytest
import torch

from certgnn.certify.link import (
    latent_fidelity_gap,
    predicted_observed_effect,
    sigmoid_jacobian,
)


def test_jacobian_peaks_at_half():
    # linspace(0.01, 0.99, 99) has step 0.01, so p = 0.50 sits at index 49.
    p = torch.linspace(0.01, 0.99, 99)
    assert torch.argmax(sigmoid_jacobian(p)).item() == 49
    assert p[49].item() == pytest.approx(0.5)


def test_observed_effect_vanishes_at_extremes():
    """Theorem 2(i)."""
    d = 1.0
    assert predicted_observed_effect(d, 0.5) > 10 * predicted_observed_effect(d, 0.01)


def test_latent_gap_is_baseline_invariant():
    """Theorem 2(iv): equal latent shifts give equal latent scores regardless
    of baseline, which is exactly what probability space fails to do."""
    for z0 in [-4.0, 0.0, 4.0]:
        p0 = torch.sigmoid(torch.tensor([z0]))
        p1 = torch.sigmoid(torch.tensor([z0 + 0.7]))
        assert latent_fidelity_gap(p0, p1).item() == pytest.approx(0.7, abs=1e-4)


def test_stratify_by_baseline_accepts_any_float_dtype():
    """Regression: torch.quantile requires the probs tensor to match the
    input dtype; float64 inputs used to raise."""
    from certgnn.certify.link import stratify_by_baseline

    for dtype in (torch.float32, torch.float64):
        p0 = torch.linspace(0.01, 0.99, 200, dtype=dtype)
        s = stratify_by_baseline(p0, n_strata=5)
        assert s.min().item() == 0 and s.max().item() == 4
        counts = torch.bincount(s, minlength=5)
        assert counts.min().item() >= 30  # roughly balanced quantile strata
