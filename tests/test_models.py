"""Model forward shapes, the training loop, and the tier-0 controls."""

from __future__ import annotations

import networkx as nx
import pytest
import torch
from torch_geometric.loader import DataLoader

from certgnn.eval.sanity import degree_preserving_rewire, shuffle_topology
from certgnn.models import (
    BrainQuadraticNetwork,
    NodeFeatureMLP,
    TargetReadoutGCN,
    fit,
    infer_task,
)
from certgnn.substrates.synthetic import (
    SyntheticConfig,
    SyntheticSubstrate,
    TopologySpec,
)


def _data(n=40, structural=False, seed=0):
    sub = SyntheticSubstrate(
        SyntheticConfig(
            topology=TopologySpec("bubble", length=2, n_distractors=2),
            regulator="random",
            structural_features=structural,
            n_train=n,
            n_val=n // 2,
            n_test=n // 2,
            seed=seed,
        )
    )
    return sub, sub.load("train"), sub.load("val"), sub.load("test")


def test_structural_features_flag_controls_feature_width():
    _, tr, _, _ = _data(structural=False)
    assert tr[0].x.size(1) == 1
    _, tr, _, _ = _data(structural=True)
    assert tr[0].x.size(1) == 3


def test_models_forward_shapes_and_latent_output():
    _, tr, _, _ = _data()
    batch = next(iter(DataLoader(tr, batch_size=8)))
    n = int(tr[0].num_nodes)
    for model in (
        TargetReadoutGCN(1, 16, 2),
        TargetReadoutGCN(1, 16, 2, readout="mean"),
        NodeFeatureMLP(1, 16, 2),
        BrainQuadraticNetwork(n, layers=2, clusters=3),
        BrainQuadraticNetwork(n, layers=1, variant="hadamard", pooling="mean"),
    ):
        out = model(batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        BrainQuadraticNetwork(n + 1)(
            batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx
        )


def test_target_readout_gathers_the_right_node():
    """With one GCN layer of depth 0 (identity features), a target readout
    must return the target's own feature row."""
    _, tr, _, _ = _data()
    batch = next(iter(DataLoader(tr, batch_size=4)))
    model = NodeFeatureMLP(1, 4, 1)
    with torch.no_grad():
        model.mlp[0].weight.fill_(1.0)
        model.mlp[0].bias.zero_()
        model.head.weight.fill_(1.0)
        model.head.bias.zero_()
    out = model(batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx)
    expected = torch.stack([4 * torch.relu(d.x[int(d.target_idx), 0]) for d in tr[:4]])
    assert torch.allclose(out, expected)


def test_fit_runs_and_selects_on_validation():
    _, tr, va, te = _data(n=60)
    assert infer_task(tr) == "regression"
    r = fit(lambda: TargetReadoutGCN(1, 16, 2), tr, va, te, seed=0, epochs=3, lr=0.01)
    assert r.task == "regression" and 1 <= r.best_epoch <= 3
    assert set(r.test_metrics) == {"r2", "mse"}
    labelled = []
    for d in tr:
        e = d.clone()
        e.y = (d.y > d.y.median()).float()
        labelled.append(e)
    assert infer_task(labelled) == "classification"


def test_degree_preserving_rewire_keeps_degrees_and_features_but_not_edges():
    G = nx.random_regular_graph(3, 30, seed=2)
    und = torch.tensor(list(G.edges())).t()
    ei = torch.cat([und, und.flip(0)], 1)
    out = degree_preserving_rewire(ei, 30, seed=0)
    H = nx.Graph()
    H.add_edges_from(out.t().tolist())
    assert sorted(d for _, d in G.degree()) == sorted(d for _, d in H.degree())
    assert [H.degree(v) for v in range(30)] == [
        G.degree(v) for v in range(30)
    ]  # per node, not just the multiset
    overlap = (
        len(set(G.edges()) & {tuple(sorted(e)) for e in H.edges()})
        / G.number_of_edges()
    )
    assert overlap < 0.5
    assert out.size(1) == ei.size(1)
    # the legacy permutation is isomorphic to the input, so it is not a rewire
    perm = shuffle_topology(ei, 30, seed=0)
    P = nx.Graph()
    P.add_edges_from(perm.t().tolist())
    assert nx.is_isomorphic(G, P)
    with pytest.raises(ValueError):
        degree_preserving_rewire(torch.tensor([[0], [1]]), 2)
