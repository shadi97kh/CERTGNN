"""The series-parallel solver must agree with the Laplacian pseudo-inverse.
Disagreement is a correctness bug, not a tolerance issue."""

import networkx as nx
import numpy as np
import pytest
import torch

from certgnn.topology.resistance import (
    is_series_parallel,
    resistance_series_parallel,
    resistance_to_target,
    resistance_via_pinv,
)


def _pinv_pair(G, u, v):
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    ei = torch.tensor(
        [[idx[a] for a, _ in G.edges()], [idx[b] for _, b in G.edges()]],
        dtype=torch.long,
    )
    R = resistance_via_pinv(ei, len(nodes))
    return float(R[idx[u], idx[v]])


def test_path_resistance_equals_length():
    """Theorem 1(iii): pendant path gives R = d exactly."""
    G = nx.path_graph(6)
    assert resistance_series_parallel(G, 0, 5) == pytest.approx(5.0)


def test_parallel_paths():
    """Theorem 1(ii): k parallel paths give the harmonic combination."""
    G = nx.Graph()
    nx.add_path(G, [0, 10, 1])  # length 2
    nx.add_path(G, [0, 20, 21, 1])  # length 3
    expected = 1.0 / (1.0 / 2 + 1.0 / 3)
    assert resistance_series_parallel(G, 0, 1) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(20))
def test_agrees_with_pinv_on_random_sp_graphs(seed):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    nx.add_path(G, [0, 1])
    nxt = 2
    for _ in range(rng.integers(2, 6)):
        L = int(rng.integers(1, 4))
        prev = 0
        for _ in range(L):
            G.add_edge(prev, nxt)
            prev, nxt = nxt, nxt + 1
        G.add_edge(prev, 1)
    if not is_series_parallel(G):
        pytest.skip("not series-parallel")
    assert resistance_series_parallel(G, 0, 1) == pytest.approx(
        _pinv_pair(G, 0, 1), rel=1e-6
    )


def test_k4_is_not_series_parallel():
    assert not is_series_parallel(nx.complete_graph(4))


def test_resistance_to_target_falls_back():
    """Non-SP graphs must still return correct values via pinv."""
    G = nx.complete_graph(4)
    r = resistance_to_target(G, target=0)
    assert r[1] == pytest.approx(_pinv_pair(G, 1, 0), rel=1e-6)
