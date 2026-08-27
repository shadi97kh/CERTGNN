"""Closed-form covariate shift on the baseline latent eta0.

Train, validation and calibration instances draw ``eta0 ~ N(mu_train,
sigma_train^2)``; test instances draw ``eta0 ~ N(mu_test, sigma_test^2)``.
Every other step of the generative process is identical, so the likelihood
ratio of the full instance distribution is exactly the density ratio in
``eta0``. This is the only setting in the project where weighted conformal
can be checked against a *known* ratio, separating ratio-estimation error
from calibration bugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianShift:
    """``w(eta0) = N(eta0; mu_test, sigma_test) / N(eta0; mu_train, sigma_train)``."""

    mu_train: float
    sigma_train: float
    mu_test: float
    sigma_test: float

    def __post_init__(self) -> None:
        if self.sigma_train <= 0 or self.sigma_test <= 0:
            raise ValueError("sigmas must be positive")

    def log_ratio(self, eta0: torch.Tensor) -> torch.Tensor:
        """Log likelihood ratio, computed in float64."""
        e = eta0.to(torch.float64)
        a = -0.5 * ((e - self.mu_test) / self.sigma_test) ** 2
        b = -0.5 * ((e - self.mu_train) / self.sigma_train) ** 2
        return a - b + math.log(self.sigma_train / self.sigma_test)

    def ratio(self, eta0: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_ratio(eta0))
