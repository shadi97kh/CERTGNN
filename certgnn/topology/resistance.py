"""Effective resistance on graphs.

Two paths: an exact O(E) series-parallel reduction for graphs with no K4 minor
(splice graphs qualify), and a general Laplacian pseudo-inverse. The solver is
validated against the pseudo-inverse in tests; any disagreement is a bug in the
reduction, not a numerical tolerance issue.
"""

from __future__ import annotations

import networkx as nx
import torch


def resistance_via_pinv(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Exact effective resistance for all node pairs via Laplacian pseudo-inverse.

    R(u,v) = L+[u,u] + L+[v,v] - 2 L+[u,v]

    O(n^3). Use for validation and for graphs that are not series-parallel.
    """
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    A[edge_index[0], edge_index[1]] = 1.0
    A = torch.maximum(A, A.T)
    L = torch.diag(A.sum(1)) - A
    Lp = torch.linalg.pinv(L)
    d = torch.diag(Lp)
    return d[:, None] + d[None, :] - 2.0 * Lp


def is_series_parallel(G: nx.Graph) -> bool:
    """Series-parallel iff no K4 minor. Checked by iterative reduction."""
    H = G.copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    changed = True
    while changed and H.number_of_nodes() > 2:
        changed = False
        for v in list(H.nodes()):
            if H.degree(v) == 1:
                H.remove_node(v)
                changed = True
            elif H.degree(v) == 2:
                a, b = list(H.neighbors(v))
                if a != b:
                    H.remove_node(v)
                    H.add_edge(a, b)
                    changed = True
    return H.number_of_nodes() <= 2


def resistance_series_parallel(G: nx.Graph, source: int, target: int) -> float:
    """Exact R(source, target) by series-parallel reduction.

    Raises ValueError if the graph is not reducible, rather than returning a
    wrong answer. Do not silently fall back to the pinv here; the caller should
    decide.
    """
    H = nx.MultiGraph()
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, r=float(data.get("resistance", 1.0)))

    changed = True
    while changed:
        changed = False
        # Dangling reduction: degree-1 nodes carry no current between the
        # terminals, so prune them. Required for pendant regulatory windows.
        for w in list(H.nodes()):
            if w in (source, target):
                continue
            if H.degree(w) == 1:
                H.remove_node(w)
                changed = True
        # Parallel reduction
        for u, v in {tuple(sorted((a, b))) for a, b, _ in H.edges(keys=True)}:
            keys = list(H[u][v].keys()) if H.has_edge(u, v) else []
            if len(keys) > 1:
                inv = sum(1.0 / H[u][v][k]["r"] for k in keys)
                for k in keys:
                    H.remove_edge(u, v, k)
                H.add_edge(u, v, r=1.0 / inv)
                changed = True
        # Series reduction
        for w in list(H.nodes()):
            if w in (source, target) or H.degree(w) != 2:
                continue
            nbrs = [(u, k) for u in H.neighbors(w) for k in H[w][u]]
            if len(nbrs) != 2:
                continue
            (a, ka), (b, kb) = nbrs
            if a == b:
                continue
            r = H[w][a][ka]["r"] + H[w][b][kb]["r"]
            H.remove_node(w)
            H.add_edge(a, b, r=r)
            changed = True

    H.remove_nodes_from([n for n in list(H.nodes()) if n not in (source, target)])

    if H.number_of_nodes() != 2 or not H.has_edge(source, target):
        raise ValueError(
            "Graph did not reduce to a single edge; it is not series-parallel "
            "between these terminals. Use resistance_via_pinv."
        )
    keys = list(H[source][target].keys())
    inv = sum(1.0 / H[source][target][k]["r"] for k in keys)
    return 1.0 / inv


def resistance_to_target(
    G: nx.Graph, target: int, exact: bool = True
) -> dict[int, float]:
    """R(v, target) for every node v. Tries exact reduction, falls back to pinv."""
    out: dict[int, float] = {}
    if exact and is_series_parallel(G):
        for v in G.nodes():
            if v == target:
                out[v] = 0.0
                continue
            try:
                out[v] = resistance_series_parallel(G, v, target)
            except ValueError:
                exact = False
                break
        if exact:
            return out

    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    ei = torch.tensor(
        [[idx[u] for u, _ in G.edges()], [idx[v] for _, v in G.edges()]],
        dtype=torch.long,
    )
    R = resistance_via_pinv(ei, len(nodes))
    return {n: float(R[idx[n], idx[target]]) for n in nodes}
