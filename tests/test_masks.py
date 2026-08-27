"""Soft-mask optimizer against the synthetic oracle (exact ground truth)."""

from __future__ import annotations

import warnings

import pytest
import torch

from certgnn.explain.masks import (
    HardMaskWarning,
    MaskResult,
    binarize,
    cold_init,
    explain,
    latent_gap,
    optimize_soft_mask,
    resistance_ball_init,
)
from certgnn.substrates.synthetic import (
    SyntheticConfig,
    SyntheticSubstrate,
    TopologySpec,
)
from certgnn.topology.resistance import resistance_to_target


def _instance(spec: TopologySpec, seed: int = 0, delta: float = 2.0):
    sub = SyntheticSubstrate(
        SyntheticConfig(topology=spec, delta=delta, p0=0.5, n_train=1, seed=seed)
    )
    d = sub.load("train")[0]
    o = sub.oracle(d)
    return sub, d, o


def test_soft_mask_reaches_epsilon_and_drops_distractors() -> None:
    sub, d, o = _instance(TopologySpec("bubble", length=2, n_distractors=3))
    eps = 0.3 * abs(o.latent_effect())
    res = explain(o.latent, o.graph, o.target, eps, warm_start="cold", max_iter=400)
    assert isinstance(res, MaskResult)
    assert res.mask_type == "soft" and res.warm_start == "cold"
    assert res.sufficient and res.latent_gap <= eps + 1e-9
    assert float(res.mask[o.target]) == 1.0
    assert bool((res.mask > 0).all()) and bool(
        (res.mask < 1).all() or res.mask[o.target] == 1
    )
    leaves = [v for v in o.nodes if o.graph.degree(v) == 1 and v != o.target]
    assert leaves and max(float(res.mask[v]) for v in leaves) < 0.1
    # soft optimum: the regulator is the top-ranked candidate, not necessarily ~1
    assert int(torch.argmax(res.mask[o.candidates()])) == o.candidates().index(
        o.regulator
    )
    assert res.sparsity < len(o.candidates()) - len(leaves) + 0.5
    assert 1 <= res.iterations <= 400 and res.converged
    assert res.first_sufficient is not None and res.first_sufficient <= res.iterations
    assert latent_gap(o.latent, res.mask) == pytest.approx(res.latent_gap)


def test_soft_path_emits_no_hard_mask_warning() -> None:
    _, _, o = _instance(TopologySpec("path", length=3))
    with warnings.catch_warnings():
        warnings.simplefilter("error", HardMaskWarning)
        explain(o.latent, o.graph, o.target, 0.1, max_iter=50)


def test_hard_mask_flag_warns_loudly_naming_assumption_5() -> None:
    _, _, o = _instance(TopologySpec("bubble", length=2, n_distractors=2))
    eps = 0.3 * abs(o.latent_effect())
    with pytest.warns(HardMaskWarning) as rec:
        res = explain(o.latent, o.graph, o.target, eps, mask_type="hard", max_iter=300)
    msg = " ".join(str(w.message) for w in rec)
    assert "Assumption 5" in msg
    for thm in ("Theorem 3", "Theorem 4", "Theorem 5"):
        assert thm in msg
    assert res.mask_type == "hard"
    assert set(res.mask.unique().tolist()) <= {0.0, 1.0}
    # the gap reported is for the binarized mask, not the soft one
    assert res.latent_gap == pytest.approx(latent_gap(o.latent, res.mask))
    with pytest.raises(ValueError):
        binarize(res.mask)
    with pytest.warns(HardMaskWarning):
        binarize(torch.tensor([0.2, 0.7, 1.0]), acknowledge_assumption_5=True)


def test_resistance_ball_warm_start_is_soft_and_ordered_by_resistance() -> None:
    _, _, o = _instance(TopologySpec("path", length=4, n_distractors=2))
    init = resistance_ball_init(o.graph, o.target, ball_size=2)
    assert float(init[o.target]) == 1.0
    inside = [v for v in o.candidates() if float(init[v]) > 0.5]
    assert len(inside) == 2
    assert 0.0 < init.min() and init[o.candidates()].max() < 1.0
    # by radius
    init_r = resistance_ball_init(o.graph, o.target, radius=1.0)
    assert sorted(v for v in o.candidates() if float(init_r[v]) > 0.5) == [
        v for v in o.candidates() if o.graph.has_edge(v, o.target)
    ]
    with pytest.raises(ValueError):
        resistance_ball_init(o.graph, o.target)
    with pytest.raises(ValueError):
        resistance_ball_init(o.graph, o.target, radius=1.0, ball_size=1)


def test_warm_start_from_theorem_4_ball_starts_closer_and_reaches_sufficiency_first() -> (
    None
):
    """Theorem 4's ball is the resistance ball whose radius reaches the
    regulator. Starting there lowers the initial latent gap relative to a
    uniform cold start on every motif, and on a bubble at moderate eps the
    warm start is sufficient from the first iteration while the cold start
    is not. (Iterations-to-feasibility in general is an ablation measurement,
    not a guarantee: a far-from-feasible start gets a larger penalty
    gradient. A ball chosen by *count* can also fill with low-resistance
    distractor leaves before it reaches the regulator's path.)"""

    def theorem4_radius(o) -> float:
        return resistance_to_target(o.graph, o.target)[o.regulator]

    for spec in [
        TopologySpec("path", length=4, n_distractors=3),
        TopologySpec("bubble", length=2, n_distractors=3),
        TopologySpec("pendant", n_branches=2, length=2, depth=2, n_distractors=2),
    ]:
        _, _, o = _instance(spec)
        eps = 0.05 * abs(o.latent_effect())
        warm = explain(
            o.latent,
            o.graph,
            o.target,
            eps,
            warm_start="resistance_ball",
            radius=theorem4_radius(o),
            max_iter=600,
        )
        cold = explain(
            o.latent, o.graph, o.target, eps, warm_start="cold", max_iter=600
        )
        assert warm.warm_start == "resistance_ball" and cold.warm_start == "cold"
        assert warm.sufficient and cold.sufficient and warm.converged and cold.converged
        assert warm.initial_gap < cold.initial_gap, spec
        assert warm.first_sufficient is not None and cold.first_sufficient is not None
    # bubble at a looser eps: the Theorem 4 ball start is sufficient immediately
    _, _, o = _instance(TopologySpec("bubble", length=2, n_distractors=3))
    eps = 0.3 * abs(o.latent_effect())
    warm = explain(
        o.latent,
        o.graph,
        o.target,
        eps,
        warm_start="resistance_ball",
        radius=theorem4_radius(o),
        max_iter=600,
    )
    cold = explain(o.latent, o.graph, o.target, eps, warm_start="cold", max_iter=600)
    assert warm.first_sufficient == 1
    assert (
        cold.first_sufficient is not None
        and cold.first_sufficient > warm.first_sufficient
    )


def test_optimizer_validates_inputs() -> None:
    _, _, o = _instance(TopologySpec("path", length=2))
    with pytest.raises(ValueError):
        optimize_soft_mask(o.latent, cold_init(o.n, o.target), o.target, -0.1)
    with pytest.raises(ValueError):
        optimize_soft_mask(
            o.latent, torch.ones(o.n), o.target, 0.1
        )  # init not inside (0,1)
    with pytest.raises(ValueError):
        optimize_soft_mask(
            o.latent, cold_init(o.n, o.target), o.target, 0.1, candidates=[o.target]
        )
    with pytest.raises(ValueError):
        cold_init(o.n, o.target, value=1.0)
    with pytest.raises(ValueError):
        explain(o.latent, o.graph, o.target, 0.1, warm_start="random")  # type: ignore[arg-type]
