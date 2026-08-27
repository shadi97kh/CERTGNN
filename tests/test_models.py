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


# ------------------------------------------------------ soft node masking


def _one_batch(data, k=4):
    return next(iter(DataLoader(data[:k], batch_size=k, shuffle=False)))


def _model_and_batch(depth=3, seed=0):
    _, tr, _, _ = _data(n=40, seed=seed)
    batch = _one_batch(tr)
    torch.manual_seed(seed)
    model = TargetReadoutGCN(int(batch.x.size(1)), 16, depth)
    model.eval()
    return model, batch


def _fwd(model, batch, mask=None):
    return model(batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx, mask)


def test_mask_of_ones_is_the_unmasked_forward():
    """The masked path must contain the unmasked one, or the two coverage
    profiles in gate2_model would differ for a reason unrelated to masking."""
    model, batch = _model_and_batch()
    with torch.no_grad():
        assert torch.allclose(
            _fwd(model, batch), _fwd(model, batch, torch.ones(batch.num_nodes)), atol=0
        )


def test_mask_actually_reaches_the_output():
    """Zeroing every non-target node must change the target's latent output.

    Before the mask argument existed no mask had ever entered a GNN in this
    repo; this is the test that would have caught a mask silently ignored.
    """
    model, batch = _model_and_batch()
    m = torch.ones(batch.num_nodes)
    keep = batch.ptr[:-1] + batch.target_idx.reshape(-1)
    m[:] = 0.0
    m[keep] = 1.0
    with torch.no_grad():
        assert not torch.allclose(_fwd(model, batch), _fwd(model, batch, m))


def test_mask_applies_at_every_layer_not_just_the_input():
    """A node two hops from the target must matter at depth 2 but not depth 1.

    If the mask were applied only to the input features, a masked node would
    still relay its neighbours' messages and the depth-2 model would be
    insensitive to it in the same way the depth-1 model is.
    """
    _, tr, _, _ = _data(n=40)
    batch = _one_batch(tr, k=1)
    G = nx.Graph()
    G.add_nodes_from(range(int(batch.num_nodes)))
    G.add_edges_from(batch.edge_index.t().tolist())
    target = int(batch.target_idx)
    dist = nx.single_source_shortest_path_length(G, target)
    two_hop = [v for v, d in dist.items() if d == 2]
    if not two_hop:
        pytest.skip("no node exactly two hops from the target in this instance")
    m = torch.ones(int(batch.num_nodes))
    m[two_hop[0]] = 0.0
    changed = {}
    for depth in (1, 2):
        torch.manual_seed(0)
        model = TargetReadoutGCN(int(batch.x.size(1)), 16, depth)
        model.eval()
        with torch.no_grad():
            changed[depth] = not torch.allclose(
                _fwd(model, batch), _fwd(model, batch, m), atol=1e-7
            )
    assert changed[2], "a two-hop node must reach the target through two layers"


def test_mask_is_differentiable_soft_not_thresholded():
    """Theorems 3 to 5 need a gradient path through the mask (CLAUDE.md)."""
    model, batch = _model_and_batch()
    m = torch.full((int(batch.num_nodes),), 0.5, requires_grad=True)
    _fwd(model, batch, m).sum().backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()
    assert float(m.grad.abs().sum()) > 0.0


def test_mask_rejects_the_wrong_number_of_entries():
    model, batch = _model_and_batch()
    with pytest.raises(ValueError, match="one value per node"):
        _fwd(model, batch, torch.ones(int(batch.num_nodes) + 1))


def test_mlp_accepts_a_mask_for_interface_parity():
    _, tr, _, _ = _data(n=40)
    batch = _one_batch(tr)
    torch.manual_seed(0)
    mlp = NodeFeatureMLP(int(batch.x.size(1)), 16, 2)
    mlp.eval()
    with torch.no_grad():
        base = mlp(batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx)
        same = mlp(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.ptr,
            batch.target_idx,
            torch.ones(int(batch.num_nodes)),
        )
    assert torch.allclose(base, same, atol=0)


def test_fit_returns_the_trained_model_and_it_is_usable():
    """gate2_model needs the fitted weights, not just the metrics."""
    _, tr, va, te = _data(n=60)
    r = fit(lambda: TargetReadoutGCN(int(tr[0].x.size(1)), 16, 2), tr, va, te, seed=0, epochs=2)
    assert r.model is not None
    batch = _one_batch(te)
    with torch.no_grad():
        out = _fwd(r.model, batch)
    assert out.shape == (min(4, len(te)),) and torch.isfinite(out).all()


def test_baseline_feature_appends_the_instance_baseline():
    """The observable-baseline arm of gate2_model depends on this column."""
    kw = dict(
        topology=TopologySpec("bubble", length=2, n_distractors=2),
        regulator="random",
        p0=None,
        n_train=3,
        n_val=0,
        n_cal=0,
        n_test=0,
        seed=0,
    )
    off = SyntheticSubstrate(SyntheticConfig(**kw)).load("train")[0]
    on = SyntheticSubstrate(SyntheticConfig(**kw, baseline_feature=True)).load("train")[0]
    assert on.x.size(1) == off.x.size(1) + 1
    col = on.x[:, -1]
    assert torch.allclose(col, col[0].expand_as(col))
    assert float(col[0]) == pytest.approx(float(on.eta0), abs=1e-5)
