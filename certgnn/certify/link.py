"""Link-function corrections (Theorem 2).

Attribution and faithfulness scores on bounded outputs inherit the link's
Jacobian, which varies with the operating point. Comparing raw scores across
instances with different baselines is therefore not meaningful. These helpers
move scores into a space where they are comparable.
"""

from __future__ import annotations

import torch

EPS = 1e-6


def logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(EPS, 1.0 - EPS)
    return torch.log(p / (1.0 - p))


def sigmoid_jacobian(p: torch.Tensor) -> torch.Tensor:
    """d(prob)/d(logit) = p(1-p). Maximal at 0.5, vanishing at 0 and 1."""
    return p * (1.0 - p)


def arcsin_sqrt(p: torch.Tensor) -> torch.Tensor:
    """Variance-stabilizing transform for proportions. Alternative to logit."""
    return torch.asin(torch.sqrt(p.clamp(0.0, 1.0)))


def latent_fidelity_gap(p_full: torch.Tensor, p_masked: torch.Tensor) -> torch.Tensor:
    """Nonconformity score for Proposition 22. Requires no ground truth."""
    return (logit(p_masked) - logit(p_full)).abs()


def probability_fidelity_gap(
    p_full: torch.Tensor, p_masked: torch.Tensor
) -> torch.Tensor:
    """The naive score. Provided so the ablation can show it fails."""
    return (p_masked - p_full).abs()


def predicted_observed_effect(delta_latent: float, p0: float) -> float:
    """Theorem 2(i): first-order observed effect for a latent effect delta."""
    return abs(delta_latent) * p0 * (1.0 - p0)


def stratify_by_baseline(p0: torch.Tensor, n_strata: int = 5) -> torch.Tensor:
    """Mondrian strata on |logit p0|. Theorem 2(iii) says conditional coverage
    fails without this."""
    z = logit(p0).abs()
    qs = torch.quantile(z, torch.linspace(0, 1, n_strata + 1)[1:-1])
    return torch.bucketize(z, qs)
