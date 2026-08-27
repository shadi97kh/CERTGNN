"""Link functions g: latent eta -> mean mu, with inverses and Jacobians.

The synthetic substrate generates the mean through a named link so that
Theorem 2's Jacobian argument can be checked against every link the paper
mentions, not only the logistic one the real substrates use.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

LINK_NAMES = ("identity", "logit", "probit", "cloglog")


@dataclass(frozen=True)
class Link:
    """A monotone link g with inverse and derivative.

    Attributes
    ----------
    name : str
    forward : callable
        ``mu = g(eta)``.
    inverse : callable
        ``eta = g^{-1}(mu)``; for bounded links the domain is ``(0, 1)``.
    jacobian : callable
        ``d mu / d eta`` evaluated at ``eta``.
    bounded : bool
        Whether ``mu`` lives in ``(0, 1)``.
    """

    name: str
    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]
    jacobian: Callable[[torch.Tensor], torch.Tensor]
    bounded: bool


def _identity(e: torch.Tensor) -> torch.Tensor:
    return e


def _ones(e: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(e)


def _logit_jac(e: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(e)
    return s * (1.0 - s)


def _probit_jac(e: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * e * e) / math.sqrt(2.0 * math.pi)


def _cloglog(e: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.exp(-torch.exp(e))


def _cloglog_inv(p: torch.Tensor) -> torch.Tensor:
    return torch.log(-torch.log1p(-p))


def _cloglog_jac(e: torch.Tensor) -> torch.Tensor:
    return torch.exp(e - torch.exp(e))


_LINKS: dict[str, Link] = {
    "identity": Link("identity", _identity, _identity, _ones, bounded=False),
    "logit": Link("logit", torch.sigmoid, torch.logit, _logit_jac, bounded=True),
    "probit": Link(
        "probit", torch.special.ndtr, torch.special.ndtri, _probit_jac, bounded=True
    ),
    "cloglog": Link("cloglog", _cloglog, _cloglog_inv, _cloglog_jac, bounded=True),
}


def get_link(name: str) -> Link:
    """Look up a link by name.

    Raises
    ------
    ValueError
        If ``name`` is not one of ``LINK_NAMES``.
    """
    try:
        return _LINKS[name]
    except KeyError:
        raise ValueError(
            f"unknown link {name!r}; expected one of {LINK_NAMES}"
        ) from None
