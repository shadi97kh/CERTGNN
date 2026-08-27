"""End-to-end smoke tests: each experiment must finish a tiny run and write
the full provenance set. Catches wiring regressions between aggregate(),
verdict() and the writers that unit tests of the pieces cannot."""

from __future__ import annotations

import json
import pathlib

import pytest

from experiments import gate2_link, tier0_controls

PROVENANCE = {
    "meta.json",
    "config.yaml",
    "environment.lock",
    "results.json",
    "per_seed_values.json",
    "tuning_budget.json",
}


def _run_dir(root: pathlib.Path) -> pathlib.Path:
    dirs = [d for d in root.iterdir() if d.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


@pytest.mark.slow
def test_gate2_link_runs_end_to_end(tmp_path):
    out = tmp_path / "runs"
    rc = gate2_link.main(
        [
            "--config",
            "configs/base.yaml",
            "--allow-dirty",
            "seeds=1",
            "gate2.link_sweep.n_instances=60",
            "gate2.link_sweep.n_pairs=500",
            "gate2.curve_fit.n_instances=60",
            "gate2.rank_reversal.n_per_baseline=20",
            "gate2.coverage.n_cal=40",
            "gate2.coverage.n_test=100",
            f"output.runs={out}",
            f"output.figures={tmp_path / 'fig'}",
            f"output.tables={tmp_path / 'tab'}",
        ]
    )
    assert rc in (0, 1)  # PASS or FAIL are both valid completions; a crash is not
    d = _run_dir(out)
    assert PROVENANCE <= {p.name for p in d.iterdir()}
    assert (d / "gate_stats.json").exists() and (d / "verdict.md").exists()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["seeds"] == [0] and "git_sha" in meta
    per_seed = json.loads((d / "per_seed_values.json").read_text())
    assert set(per_seed) >= {"gap_probability", "gap_latent"} and list(
        per_seed["gap_probability"]
    ) == ["0"]
    assert (tmp_path / "fig" / "gate2_link.png").exists()


@pytest.mark.slow
def test_tier0_controls_runs_end_to_end(tmp_path):
    out = tmp_path / "runs"
    rc = tier0_controls.main(
        [
            "--config",
            "configs/base.yaml",
            "--allow-dirty",
            "seeds=1",
            "model.epochs=2",
            "substrate.n_train=40",
            "substrate.n_val=10",
            "substrate.n_test=20",
            "tier0.substrates=[synthetic]",
            f"output.runs={out}",
            f"output.tables={tmp_path / 'tab'}",
        ]
    )
    assert rc == 0
    d = _run_dir(out)
    assert PROVENANCE <= {p.name for p in d.iterdir()}
    per_seed = json.loads((d / "per_seed_values.json").read_text())
    assert set(per_seed) == {"synthetic/gnn", "synthetic/shuffled", "synthetic/mlp"}
    budget = json.loads((d / "tuning_budget.json").read_text())
    assert {b["model"] for b in budget} == {"gnn", "shuffled", "mlp"}
    tried = {b["model"]: b["configs_tried"] for b in budget}
    assert tried == {
        "gnn": 3,
        "shuffled": 3,
        "mlp": 1,
    }  # honest count: backbone chosen among 3
