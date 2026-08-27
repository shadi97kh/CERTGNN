"""Faithfulness metrics: paired spaces, Theorem 2 disagreement, degeneracy."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

import certgnn.eval.faithfulness as F
from certgnn.eval.faithfulness import (
    PairedScore,
    aopc,
    comprehensiveness,
    deletion_auc,
    faithfulness_report,
    insertion_auc,
    normalized_aopc,
    sufficiency,
)
from certgnn.substrates.synthetic import (
    SyntheticConfig,
    SyntheticSubstrate,
    TopologySpec,
)


def _oracle(spec: TopologySpec, p0: float, delta: float = 2.0, seed: int = 0):
    sub = SyntheticSubstrate(
        SyntheticConfig(topology=spec, delta=delta, p0=p0, n_train=1, seed=seed)
    )
    d = sub.load("train")[0]
    return sub.oracle(d)


def _true_mask(o) -> torch.Tensor:
    """Soft mask that is high on the exact minimal set and low elsewhere."""
    m = torch.full((o.n,), 0.05, dtype=torch.float64)
    for v in o.minimal_sufficient_set(0.0):
        m[v] = 0.95
    m[o.t] = 1.0
    return m


def test_every_public_metric_returns_a_pair_and_nothing_returns_probability_only() -> (
    None
):
    o = _oracle(TopologySpec("bubble", length=2, n_distractors=2), p0=0.5)
    m = _true_mask(o)
    report = faithfulness_report(o.latent, m, o.target)
    assert set(report) == {
        "comprehensiveness",
        "sufficiency",
        "aopc",
        "deletion_auc",
        "insertion_auc",
        "normalized_aopc",
    }
    for name, score in report.items():
        assert isinstance(score, PairedScore) and score.metric == name
        assert isinstance(score.probability, float) and isinstance(score.latent, float)
        assert isinstance(score.degenerate, bool)
    # static guard: no public function in the module is annotated to return a bare float
    for name, fn in inspect.getmembers(F, inspect.isfunction):
        if name.startswith("_") or fn.__module__ != F.__name__:
            continue
        ret = inspect.signature(fn).return_annotation
        assert "PairedScore" in str(ret) or "dict" in str(ret), name


def test_probability_and_latent_scores_disagree_across_baselines_theorem_2() -> None:
    """Same graph, same regulator, same latent effect at p0 = 0.5 and 0.97:
    latent-space comprehensiveness is identical, probability-space
    comprehensiveness collapses at the extreme baseline."""
    spec = TopologySpec("path", length=2)
    mid = _oracle(spec, p0=0.5)
    ext = _oracle(spec, p0=0.97)
    assert mid.latent_effect() == pytest.approx(
        ext.latent_effect()
    )  # same seed -> same z_r
    m = _true_mask(mid)
    c_mid = comprehensiveness(mid.latent, m, mid.target)
    c_ext = comprehensiveness(ext.latent, m, ext.target)
    assert c_mid.latent == pytest.approx(c_ext.latent, abs=1e-9)
    assert abs(c_ext.probability) < 0.5 * abs(c_mid.probability)
    # the same holds for AOPC
    a_mid = aopc(mid.latent, m, mid.target)
    a_ext = aopc(ext.latent, m, ext.target)
    assert a_mid.latent == pytest.approx(a_ext.latent, abs=1e-9)
    assert abs(a_ext.probability) < 0.5 * abs(a_mid.probability)


def test_sufficiency_latent_is_the_fidelity_gap_and_comprehensiveness_signs() -> None:
    o = _oracle(TopologySpec("bubble", length=2, n_distractors=2), p0=0.5)
    m = _true_mask(o)
    s = sufficiency(o.latent, m, o.target)
    keep = torch.ones(o.n, dtype=torch.float64)
    keep[o.candidates()] = m[o.candidates()]
    assert s.latent == pytest.approx(o.latent_full() - float(o.latent(keep)))
    c = comprehensiveness(o.latent, m, o.target)
    # removing the true explanation removes (almost) the whole effect
    assert abs(c.latent) > 0.8 * abs(o.latent_effect())
    # keeping only the true explanation retains most of it
    assert abs(s.latent) < 0.2 * abs(o.latent_effect())


def test_insertion_deletion_curves_favor_the_true_ranking() -> None:
    o = _oracle(TopologySpec("bubble", length=2, n_distractors=3), p0=0.5, delta=3.0)
    good = _true_mask(o)
    bad = torch.ones(o.n, dtype=torch.float64) - good
    bad[o.t] = 1.0
    sign = 1.0 if o.latent_effect() > 0 else -1.0
    ins_good, ins_bad = (
        insertion_auc(o.latent, good, o.target),
        insertion_auc(o.latent, bad, o.target),
    )
    del_good, del_bad = (
        deletion_auc(o.latent, good, o.target),
        deletion_auc(o.latent, bad, o.target),
    )
    for space in ("latent", "probability"):
        assert sign * (getattr(ins_good, space) - getattr(ins_bad, space)) > 0
        assert sign * (getattr(del_good, space) - getattr(del_bad, space)) < 0
    assert len(ins_good.curve_latent) == len(o.candidates()) + 1
    # endpoints: deletion starts at full, insertion ends at full
    assert del_good.curve_latent[0] == pytest.approx(o.latent_full())
    assert ins_good.curve_latent[-1] == pytest.approx(o.latent_full())


def test_normalized_aopc_is_one_for_the_best_ranking_and_in_unit_interval() -> None:
    o = _oracle(TopologySpec("bubble", length=2, n_distractors=2), p0=0.5, delta=3.0)
    best = _true_mask(o)
    n_best = normalized_aopc(o.latent, best, o.target, k=3)
    assert (
        0.0 <= n_best.latent <= 1.0 + 1e-12 and 0.0 <= n_best.probability <= 1.0 + 1e-12
    )
    worst = torch.ones(o.n, dtype=torch.float64) - best
    worst[o.t] = 1.0
    n_worst = normalized_aopc(o.latent, worst, o.target, k=3)
    assert n_best.latent > n_worst.latent and n_best.probability > n_worst.probability
    # exact and beam bounds agree on a small instance
    n_beam = normalized_aopc(
        o.latent, best, o.target, k=3, exact_max_nodes=0, beam_width=50
    )
    assert n_beam.latent == pytest.approx(n_best.latent, abs=1e-9)
    assert n_beam.probability == pytest.approx(n_best.probability, abs=1e-9)


def test_normalized_aopc_is_not_the_link_correction() -> None:
    """A range normalization in probability space does not recover the
    latent-space ordering across baselines: NAOPC in probability space
    still differs from NAOPC in latent space on the same instance."""
    spec = TopologySpec("multi_parallel", n_branches=3, length=2, n_distractors=1)
    o = _oracle(spec, p0=0.9, delta=3.0)
    # sub-optimal ranking: branch interiors first, regulator late, leaf last,
    # so the AOPC is interior to its [worst, best] range in both spaces
    branch = [v for v in o.candidates() if v != o.regulator and o.graph.degree(v) == 2]
    m = torch.full((o.n,), 0.1, dtype=torch.float64)
    for i, v in enumerate(branch):
        m[v] = 0.9 - 0.1 * i
    m[o.regulator] = 0.3
    m[o.t] = 1.0
    s = normalized_aopc(o.latent, m, o.target, k=3)
    assert not math.isnan(s.latent) and not math.isnan(s.probability)
    assert 0.0 < s.latent < 1.0 and 0.0 < s.probability < 1.0
    assert s.latent != pytest.approx(s.probability, abs=1e-3)


def test_degenerate_flag_is_surfaced_by_every_metric() -> None:
    o = _oracle(TopologySpec("bubble", length=2, n_distractors=2), p0=0.5)
    full = torch.ones(o.n, dtype=torch.float64)
    for score in faithfulness_report(o.latent, full, o.target).values():
        assert score.degenerate is True
    for score in faithfulness_report(o.latent, _true_mask(o), o.target).values():
        assert score.degenerate is False


def test_mask_validation() -> None:
    o = _oracle(TopologySpec("path", length=2), p0=0.5)
    bad = torch.ones(o.n, dtype=torch.float64)
    bad[o.t] = 0.5
    with pytest.raises(ValueError):
        comprehensiveness(o.latent, bad, o.target)
    with pytest.raises(ValueError):
        aopc(o.latent, torch.full((o.n,), 1.5), o.target)


def test_evaluating_a_hard_mask_warns_but_a_soft_mask_does_not() -> None:
    import warnings

    from certgnn.explain.masks import HardMaskWarning

    o = _oracle(TopologySpec("bubble", length=2, n_distractors=2), p0=0.5)
    hard = torch.zeros(o.n, dtype=torch.float64)
    hard[o.regulator] = 1.0
    hard[o.t] = 1.0
    with pytest.warns(HardMaskWarning, match="Assumption 5"):
        comprehensiveness(o.latent, hard, o.target)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HardMaskWarning)
        faithfulness_report(o.latent, _true_mask(o), o.target)
