"""Sweep launcher and aggregator: expansion, refusal on gaps, Holm."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate as agg  # noqa: E402
import launch_sweep as ls  # noqa: E402

from experiments._common import load_config  # noqa: E402


def test_matrix_parses_and_tier_selection_excludes_candidates_and_drafts():
    rows = ls.parse_matrix(ROOT / "ABLATIONS.md")
    assert {r["id"] for r in rows} >= {"0.6", "0.7", "1.6", "1.9", "C.1"}
    t0 = ls.select_rows(rows, 0, None, include_draft=False)
    assert [r["id"] for r in t0] == [f"0.{i}" for i in range(1, 11)]
    t2 = ls.select_rows(rows, 2, None, include_draft=False)
    assert t2 == []  # drafts excluded by default
    assert len(ls.select_rows(rows, 2, None, include_draft=True)) == 5
    with pytest.raises(SystemExit):
        ls.select_rows(rows, 0, ["9.9"], include_draft=False)


def test_cell_expansion_follows_sweep_discipline():
    one = {"id": "x", "sweep": "one_factor", "axes": {"model.depth": [1, 2, 3]}}
    assert ls.expand_cells(one) == [
        {"model.depth": 1},
        {"model.depth": 2},
        {"model.depth": 3},
    ]
    two = {"id": "y", "sweep": "two_factor", "axes": {"a": [1, 2], "b": ["p", "q"]}}
    assert len(ls.expand_cells(two)) == 4
    assert ls.expand_cells({"id": "z", "sweep": "fixed", "axes": {}}) == [{}]
    with pytest.raises(ValueError):
        ls.expand_cells(
            {"id": "w", "sweep": "two_factor", "axes": {"a": [1], "b": [2], "c": [3]}}
        )


def test_plan_covers_registered_rows_once_per_seed_and_lists_the_rest():
    rows = ls.select_rows(ls.parse_matrix(ROOT / "ABLATIONS.md"), 0, None, False)
    runs, unlaunchable = ls.plan(rows, seeds=3, base="configs/base.yaml", sweep_id="t")
    modules = {r["module"] for r in runs}
    assert modules == {"experiments.tier0_controls"}
    assert (
        len(runs) == 10
    )  # rows 0.6/0.7 require 10 seeds; one run per seed for the module
    assert set(runs[0]["rows"]) == {"0.6", "0.7", "0.8"}
    assert {u["id"] for u in unlaunchable} == {
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.9",
        "0.10",
    }
    assert all(f"seed={r['seed']}" in r["overrides"] for r in runs)


def test_generated_config_round_trips_without_remerging_groups(tmp_path):
    run = {
        "run_id": "t/x_seed0",
        "module": "m",
        "rows": ["0.6"],
        "cell": {"model.hidden": 32},
        "seed": 0,
        "overrides": ["model.hidden=32", "seed=0", "run_id=t/x_seed0"],
    }
    p = ls.write_config(
        "configs/base.yaml",
        run,
        tmp_path,
        {
            "id": "t",
            "tier": 0,
            "matrix": "ABLATIONS.md",
            "allow_dirty": False,
            "git_dirty": False,
        },
    )
    cfg = load_config(str(p), [])
    assert cfg.model.hidden == 32 and cfg.seed == 0 and cfg.run_id == "t/x_seed0"
    assert cfg.sweep.cell["model.hidden"] == 32 and cfg._resolved is True


def test_holm_bonferroni_matches_hand_computation():
    out = agg.holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"]["p_adjusted"] == pytest.approx(0.03)
    assert out["c"]["p_adjusted"] == pytest.approx(0.06)
    assert out["b"]["p_adjusted"] == pytest.approx(0.06)  # monotone: max(0.04*1, 0.06)
    assert out["a"]["reject"] and not out["b"]["reject"]


def _fake_run(
    root: pathlib.Path,
    name: str,
    experiment: str,
    seeds: dict[str, float],
    run_id: str | None = None,
    complete=True,
):
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "git_sha": "abc1234",
                "git_dirty": False,
                "config_hash": "h",
                "run_id": run_id,
                "sweep": None,
            }
        )
    )
    (d / "per_seed_values.json").write_text(json.dumps({"metric": seeds}))
    if complete:
        (d / "results.json").write_text("{}")
    return d


def test_aggregate_refuses_incomplete_cells_and_writes_nothing(tmp_path):
    runs_dir = tmp_path / "runs"
    _fake_run(runs_dir, "r1", "exp", {"0": 1.0, "1": 1.1, "2": 0.9})
    out = tmp_path / "tables"
    rc = agg.main(
        [
            "--runs",
            str(runs_dir),
            "--out",
            str(out),
            "--expected-seeds",
            "5",
            "--tuning-budget",
            str(tmp_path / "tb.md"),
        ]
    )
    assert rc == 1 and not out.exists()
    # complete cell aggregates
    _fake_run(runs_dir, "r2", "exp", {"3": 1.2, "4": 1.0})
    rc = agg.main(
        [
            "--runs",
            str(runs_dir),
            "--out",
            str(out),
            "--expected-seeds",
            "5",
            "--tuning-budget",
            str(tmp_path / "tb.md"),
        ]
    )
    assert rc == 0 and (out / "aggregate_cells.tex").exists()
    assert r"\begin{tabular}" in (out / "aggregate_cells.tex").read_text()
    # manifest with a planned run that never happened -> refuse
    man = tmp_path / "m.json"
    man.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_id": "s/a_seed0", "exit_code": 0},
                    {"run_id": "s/a_seed1"},
                ],
                "rows_unlaunchable": [],
            }
        )
    )
    rc = agg.main(
        [
            "--runs",
            str(runs_dir),
            "--out",
            str(tmp_path / "t2"),
            "--manifest",
            str(man),
            "--tuning-budget",
            str(tmp_path / "tb.md"),
        ]
    )
    assert rc == 1 and not (tmp_path / "t2").exists()


def test_git_dirty_ignores_untracked_outputs_only(monkeypatch):
    import experiments._common as C

    def fake(status_text):
        monkeypatch.setattr(C.subprocess, "check_output", lambda *a, **k: status_text)

    fake(
        "?? results/sweeps/x.json\n?? paper/tables/t.md\n?? configs/generated/a.yaml\n"
    )
    assert C.git_dirty() is False
    fake("?? certgnn/new_module.py\n")
    assert C.git_dirty() is True
    fake(" M paper/tables/t.md\n")
    assert C.git_dirty() is True  # a modified *tracked* file always counts
    fake("")
    assert C.git_dirty() is False
