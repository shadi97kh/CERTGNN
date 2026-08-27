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
    """Node relabelling: decouples features from graph position by permuting
    node ids. This is NOT a degree-preserving rewiring of the graph itself
    (the graph is isomorphic to the original); use
    ``degree_preserving_rewire`` for ABLATIONS row 0.6."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    return perm[edge_index]


def degree_preserving_rewire(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int = 0,
    swaps_per_edge: float = 10.0,
) -> torch.Tensor:
    """Degree-preserving rewiring by double-edge swaps (ABLATIONS 0.6).

    Every node keeps its degree and its features; only *which* nodes are
    connected changes. Isolates the contribution of the specific wiring from
    that of the degree sequence. Critical for brain graphs given the
    message-passing critique. Returns an undirected ``edge_index`` with both
    directions. Raises if the swap procedure cannot run (fewer than 2 edges).
    """
    import networkx as nx

    ei = edge_index.cpu()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from((int(u), int(v)) for u, v in ei.t().tolist() if u != v)
    m = G.number_of_edges()
    if m < 2:
        raise ValueError("degree-preserving rewire needs at least 2 edges")
    n_swaps = max(1, int(swaps_per_edge * m))
    nx.double_edge_swap(G, nswap=n_swaps, max_tries=100 * n_swaps, seed=seed)
    und = torch.tensor(sorted(G.edges()), dtype=torch.long).t()
    return torch.cat([und, und.flip(0)], dim=1)


def random_explainer_baseline(num_nodes: int, k: int, seed: int = 0):
    """The floor any explainer must beat."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(num_nodes, generator=g)[:k]


def degenerate_explanation_check(mask: torch.Tensor, tol: float = 1e-3) -> bool:
    """Flags near-constant or near-full masks, which score well on faithfulness
    metrics while explaining nothing (Azzolin et al.)."""
    frac = float((mask > 0.5).float().mean())
    return frac > 1.0 - tol or frac < tol or float(mask.std()) < tol
