"""Tests for the synthetic substrate. Ground truth is known by construction,
so every check here is against an exact value, not a tolerance band."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest
import torch

from certgnn.certify.link import predicted_observed_effect
from certgnn.substrates.base import Substrate
from certgnn.substrates.synthetic import (
    LINK_NAMES,
    SyntheticConfig,
    SyntheticSubstrate,
    TopologySpec,
    get_link,
)
from certgnn.topology.resistance import is_series_parallel, resistance_series_parallel

SMALL_TOPOLOGIES = [
    TopologySpec("path", length=4),
    TopologySpec("path", length=3, n_distractors=2, pendant_chains=1, pendant_depth=2),
    TopologySpec("bubble", length=2),
    TopologySpec("bubble", length=3, n_distractors=2),
    TopologySpec("bubble", length=2, stem=2, pendant_chains=1),
    TopologySpec("multi_parallel", n_branches=3, length=2, n_distractors=1),
    TopologySpec("nested_bubbles", n_branches=2, depth=1, n_distractors=3),
    TopologySpec("pendant", n_branches=2, length=2, depth=2, n_distractors=1),
]


def _sub(**kw) -> SyntheticSubstrate:
    return SyntheticSubstrate(SyntheticConfig(**kw))


# ---------------------------------------------------------------- protocol


def test_implements_substrate_protocol_without_modifying_base() -> None:
    assert isinstance(_sub(), Substrate)


def test_all_motifs_are_series_parallel_with_target_zero() -> None:
    for spec in SMALL_TOPOLOGIES:
        sub = _sub(topology=spec, n_train=2)
        for d in sub.load("train"):
            G = sub.to_networkx(d)
            assert sub.target_node(d) == 0
            assert is_series_parallel(G)
            assert nx.is_connected(G)
            assert int(d.regulator) in sub.candidate_nodes(d).tolist()


# ------------------------------------------------- 1. label recovery


@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_label_recovery_under_identity_link(noise: float) -> None:
    """Under the identity link ``y - eta0 = delta * z_r`` (+ noise). A linear
    probe on the node signal features must put its weight on the regulator
    and nothing on the distractors, and argmax must recover the regulator on
    every instance-independent fit."""
    spec = TopologySpec("bubble", length=2, n_distractors=3)
    sub = _sub(
        topology=spec, link="identity", delta=1.5, noise=noise, n_train=300, seed=3
    )
    data = sub.load("train")
    Z = torch.stack([d.x[:, 0] for d in data]).double()
    resid = torch.tensor([float(d.y) - float(d.eta0) for d in data]).double()
    coef = torch.linalg.lstsq(Z, resid.unsqueeze(1)).solution.squeeze(1)
    reg = int(data[0].regulator)
    assert all(int(d.regulator) == reg for d in data)
    tol = 1e-6 if noise == 0.0 else 0.05
    assert coef[reg].item() == pytest.approx(1.5, abs=tol)
    others = torch.cat([coef[:reg], coef[reg + 1 :]])
    assert others.abs().max().item() < tol
    assert int(torch.argmax(coef.abs())) == reg
    # ground_truth_window reports the same node
    assert sub.ground_truth_window(data[0]).tolist() == [reg]


# ------------------------------- 2. exact p0(1-p0) scaling under logit


def test_observed_effect_scales_exactly_as_p0_1_minus_p0_under_logit() -> None:
    """Theorem 2(i). With delta -> 0 the probability-space effect divided by
    the latent effect is the sigmoid Jacobian ``p0 (1 - p0)`` *exactly*;
    at finite delta the relative error is O(delta)."""
    delta = 1e-4
    for p0 in [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]:
        sub = _sub(
            topology=TopologySpec("path", length=2),
            link="logit",
            p0=p0,
            delta=delta,
            n_train=20,
        )
        for d in sub.load("train"):
            assert sub.baseline_rate(d) == pytest.approx(p0, abs=1e-12)
            lat = sub.latent_effect(d)
            obs = sub.observed_effect(d)
            assert obs / lat == pytest.approx(p0 * (1.0 - p0), rel=2e-4)
            # agrees with the core's first-order prediction
            assert abs(obs) == pytest.approx(
                predicted_observed_effect(lat, p0), rel=2e-4
            )
    # and the Jacobian object itself is exact, not approximate
    e = torch.linspace(-6, 6, 25, dtype=torch.float64)
    s = torch.sigmoid(e)
    assert torch.allclose(get_link("logit").jacobian(e), s * (1 - s))


@pytest.mark.parametrize("name", LINK_NAMES)
def test_link_inverse_and_jacobian_are_consistent(name: str) -> None:
    link = get_link(name)
    e = torch.linspace(-2.5, 2.5, 41, dtype=torch.float64)
    assert torch.allclose(link.inverse(link.forward(e)), e, atol=1e-9)
    h = 1e-6
    fd = (link.forward(e + h) - link.forward(e - h)) / (2 * h)
    assert torch.allclose(link.jacobian(e), fd, atol=1e-7)


def test_identity_link_has_no_baseline_dependence() -> None:
    for eta0 in [-3.0, 0.0, 3.0]:
        sub = _sub(
            topology=TopologySpec("path", length=2),
            link="identity",
            p0=eta0,
            delta=0.7,
            n_train=5,
        )
        for d in sub.load("train"):
            assert sub.observed_effect(d) == pytest.approx(
                sub.latent_effect(d), abs=1e-12
            )


# ---------------------- 3. brute force vs optimizer on < 12 nodes


def test_bruteforce_minimal_set_agrees_with_optimizer_under_12_nodes() -> None:
    """Both searches are exact, so they must return sets of identical size
    that are both sufficient, on every motif, seed and epsilon. This is the
    ground truth Theorem 4's bound is tested against (ABLATIONS 1.10, 1.12)."""
    n_checked = 0
    for spec, seed in itertools.product(SMALL_TOPOLOGIES, range(3)):
        sub = _sub(topology=spec, regulator="random", delta=1.0, n_train=2, seed=seed)
        for d in sub.load("train"):
            assert int(d.num_nodes) < 12, spec
            o = sub.oracle(d)
            effect = abs(o.latent_effect())
            for frac in [0.0, 0.1, 0.4, 0.6, 0.9, 1.01]:
                eps = frac * effect
                brute = o.minimal_sufficient_set_bruteforce(eps)
                fast = o.minimal_sufficient_set(eps)
                assert len(brute) == len(fast), (spec, seed, frac, brute, fast)
                assert o.is_sufficient(brute, eps) and o.is_sufficient(fast, eps)
                if frac > 1.0:
                    assert brute == [] and fast == []
                elif brute:
                    assert int(d.regulator) in brute and int(d.regulator) in fast
                n_checked += 1
    assert n_checked >= 100


def test_minimal_sets_have_the_structure_the_theorems_predict() -> None:
    """Hand-checkable cases: a path needs every interior node; a bubble needs
    both branches iff eps < effect/2; pendant chains and distractor leaves
    are never selected."""
    # path t-1-2-3-r: r plus every interior node
    sub = _sub(topology=TopologySpec("path", length=4), delta=1.0, n_train=1)
    d = sub.load("train")[0]
    o = sub.oracle(d)
    assert o.minimal_sufficient_set(0.0) == sorted(o.candidates())
    # bubble with 2 distractor leaves: branches are {2,3} and {4,5}? -> check sizes
    sub = _sub(
        topology=TopologySpec("bubble", length=3, n_distractors=2), delta=1.0, n_train=1
    )
    d = sub.load("train")[0]
    o = sub.oracle(d)
    eff = abs(o.latent_effect())
    both = o.minimal_sufficient_set(0.49 * eff)
    one = o.minimal_sufficient_set(0.51 * eff)
    assert (
        len(both) == 5 and len(one) == 3
    )  # r + 2 branches x 2 interior ; r + 1 branch
    leaves = [v for v in o.nodes if o.graph.degree(v) == 1 and v != o.target]
    assert leaves and not set(leaves) & set(both)
    # pendant: chain of depth 3 from the junction (junction + 2 interior + r = 4
    # nodes, all in series) plus the bubble. Full R = 3 + 1 = 4; dropping one
    # branch gives R = 3 + 2 = 5, kappa = 4/5, so the gap is 20% of the effect:
    # one branch interior suffices above 0.2, both are needed below it.
    sub = _sub(
        topology=TopologySpec("pendant", n_branches=2, length=2, depth=3),
        delta=1.0,
        n_train=1,
    )
    d = sub.load("train")[0]
    o = sub.oracle(d)
    eff = abs(o.latent_effect())
    assert len(o.minimal_sufficient_set(0.21 * eff)) == 4 + 1
    assert len(o.minimal_sufficient_set(0.19 * eff)) == 4 + 2


def test_kappa_is_rayleigh_monotone_and_matches_core_solver() -> None:
    """Removing any node never increases kappa; kappa(full) == 1; the
    full-graph resistance agrees with the exact series-parallel solver."""
    for spec in SMALL_TOPOLOGIES:
        sub = _sub(topology=spec, n_train=1, seed=7)
        d = sub.load("train")[0]
        o = sub.oracle(d)
        assert float(o.kappa(o.full_mask())) == pytest.approx(1.0, abs=1e-12)
        cands = o.candidates()
        for v in cands:
            k1 = float(o.kappa(o.set_mask([u for u in cands if u != v])))
            assert k1 <= 1.0 + 1e-12
            for w in cands:
                if w == v:
                    continue
                k2 = float(o.kappa(o.set_mask([u for u in cands if u not in (v, w)])))
                assert k2 <= k1 + 1e-12
        r_core = resistance_series_parallel(o.graph, o.regulator, o.target)
        assert 1.0 / o._c_full == pytest.approx(r_core, rel=1e-10)


def test_soft_mask_is_differentiable() -> None:
    sub = _sub(topology=TopologySpec("bubble", length=2), n_train=1)
    d = sub.load("train")[0]
    o = sub.oracle(d)
    m = torch.full((o.n,), 0.7, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        m[o.t] = 1.0
    eta = o.latent(m)
    eta.backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()
    # every non-target node in a plain bubble carries current, so each has a
    # non-zero gradient with the sign of the latent effect
    sign = 1.0 if o.latent_effect() > 0 else -1.0
    for v in o.candidates():
        assert sign * m.grad[o.index[v]].item() > 0


# --------------------------------------------------- 4. determinism


def test_generation_is_deterministic_in_seed() -> None:
    cfg = dict(
        topology=TopologySpec("bubble", length=2, n_distractors=2),
        regulator="random",
        noise=0.3,
        shift_eta0_mean=1.0,
        n_train=20,
        n_cal=10,
        n_test=10,
    )
    a = SyntheticSubstrate(SyntheticConfig(seed=11, **cfg))
    b = SyntheticSubstrate(SyntheticConfig(seed=11, **cfg))
    c = SyntheticSubstrate(SyntheticConfig(seed=12, **cfg))
    for split in ("train", "cal", "test"):
        da, db, dc = a.load(split), b.load(split), c.load(split)
        assert len(da) == len(db) == len(dc)
        for x, y in zip(da, db):
            assert torch.equal(x.x, y.x) and torch.equal(x.edge_index, y.edge_index)
            assert torch.equal(x.y, y.y) and torch.equal(x.eta0, y.eta0)
            assert int(x.regulator) == int(y.regulator)
            assert x.likelihood_ratio.item() == y.likelihood_ratio.item()
        assert any(not torch.equal(x.x, z.x) for x, z in zip(da, dc))
    # splits are independent streams: train and cal differ under the same seed
    assert not torch.equal(a.load("train")[0].x, a.load("cal")[0].x)


# ------------------------------------------ closed-form likelihood ratio


def test_likelihood_ratio_is_closed_form_and_integrates_to_one() -> None:
    """w(eta0) must equal the Gaussian density ratio exactly, be 1 without a
    shift, and satisfy E_train[w] = 1 (Monte Carlo, 2 SE)."""
    sub = _sub(
        topology=TopologySpec("path", length=2),
        eta0_mean=0.0,
        eta0_std=2.0,
        shift_eta0_mean=1.0,
        shift_eta0_std=1.5,
        n_train=4000,
        n_test=5,
    )
    train = sub.load("train")
    f64 = torch.float64
    p_train = torch.distributions.Normal(
        torch.tensor(0.0, dtype=f64), torch.tensor(2.0, dtype=f64)
    )
    p_test = torch.distributions.Normal(
        torch.tensor(1.0, dtype=f64), torch.tensor(1.5, dtype=f64)
    )
    for d in train[:50] + sub.load("test"):
        expected = torch.exp(p_test.log_prob(d.eta0) - p_train.log_prob(d.eta0))
        assert sub.likelihood_ratio(d) == pytest.approx(expected.item(), rel=1e-10)
    w = np.array([sub.likelihood_ratio(d) for d in train])
    se = w.std(ddof=1) / np.sqrt(len(w))
    assert abs(w.mean() - 1.0) < 2 * se + 1e-3
    # test-split eta0 really comes from the shifted law
    test = _sub(
        topology=TopologySpec("path", length=2),
        shift_eta0_mean=3.0,
        shift_eta0_std=0.1,
        n_test=200,
    ).load("test")
    e = np.array([float(d.eta0) for d in test])
    assert abs(e.mean() - 3.0) < 0.05 and e.std() < 0.2
    assert all(sub.likelihood_ratio(d) == 1.0 for d in _sub(n_train=3).load("train"))


def test_config_rejects_shift_with_fixed_p0_and_bad_links() -> None:
    with pytest.raises(ValueError):
        SyntheticConfig(p0=0.5, shift_eta0_mean=1.0)
    with pytest.raises(ValueError):
        SyntheticConfig(link="cauchit")
    with pytest.raises(ValueError):
        SyntheticConfig(link="logit", p0=1.0)
    with pytest.raises(ValueError):
        TopologySpec("bubble", length=1)
    with pytest.raises(ValueError):
        _sub().load("calibration")
