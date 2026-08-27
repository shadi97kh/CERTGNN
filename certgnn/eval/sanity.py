"""Sanity checks for explanations (Adebayo et al. 2018, adapted to graphs).

An explanation method that passes faithfulness metrics but fails these is
measuring the input, not the model. Every explanation result must ship with
these controls.
"""

from __future__ import annotations

import copy

import torch


def randomize_model_parameters(
    model: torch.nn.Module, layers: str = "all"
) -> torch.nn.Module:
    """Cascading or independent parameter randomization. Explanations should
    degrade toward chance. If they do not, the explainer is edge-detecting."""
    m = copy.deepcopy(model)
    mods = list(m.modules())
    if layers == "last":
        mods = mods[-2:]
    with torch.no_grad():
        for mod in mods:
            reset = getattr(mod, "reset_parameters", None)
            if callable(reset):
                reset()
    return m


def randomize_labels(y: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Permute labels; retrain; explanations should carry no signal."""
    g = torch.Generator().manual_seed(seed)
    return y[torch.randperm(y.numel(), generator=g)]


def shuffle_topology(edge_index: torch.Tensor, num_nodes: int, seed: int = 0):
    """Degree-preserving edge rewiring. Isolates topology contribution.
    Critical for brain graphs given the message-passing critique."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    return perm[edge_index]


def random_explainer_baseline(num_nodes: int, k: int, seed: int = 0):
    """The floor any explainer must beat."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(num_nodes, generator=g)[:k]


def degenerate_explanation_check(mask: torch.Tensor, tol: float = 1e-3) -> bool:
    """Flags near-constant or near-full masks, which score well on faithfulness
    metrics while explaining nothing (Azzolin et al.)."""
    frac = float((mask > 0.5).float().mean())
    return frac > 1.0 - tol or frac < tol or float(mask.std()) < tol
