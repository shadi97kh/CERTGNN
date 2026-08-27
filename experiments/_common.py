"""Shared plumbing for experiments: config, run directories, bootstrap CIs.

Every experiment writes to ``results/runs/<timestamp>_<gitsha>_<confighash>/``.
A number without a git SHA and a config hash does not exist.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


def load_config(path: str, overrides: Sequence[str] = ()) -> DictConfig:
    """YAML config merged with Hydra-style dotlist overrides (``a.b=1``)."""
    base = OmegaConf.load(path)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(base, DictConfig)
    return base


def config_hash(cfg: DictConfig) -> str:
    text = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], text=True
    ).strip()


def git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(out.strip())


def make_run_dir(cfg: DictConfig, name: str) -> pathlib.Path:
    """Create the run directory and record config + provenance."""
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    sha, dirty, h = git_sha(), git_dirty(), config_hash(cfg)
    run = pathlib.Path(cfg.output.runs) / f"{ts}_{sha}{'-dirty' if dirty else ''}_{h}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    meta = {
        "experiment": name,
        "git_sha": sha,
        "git_dirty": dirty,
        "config_hash": h,
        "timestamp_utc": ts,
        "argv": sys.argv,
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    if dirty:
        print(
            "WARNING: git tree is dirty; this run's SHA does not identify its code",
            file=sys.stderr,
        )
    return run


def mean_ci(
    values: Sequence[float], *, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> dict[str, float]:
    """Mean with a percentile-bootstrap CI over the given per-seed values."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, n), replace=True).mean(axis=1)
    a = (1.0 - level) / 2.0
    return {
        "mean": float(v.mean()),
        "lo": float(np.quantile(boots, a)),
        "hi": float(np.quantile(boots, 1.0 - a)),
        "n": n,
    }


def fmt_ci(d: dict[str, float], nd: int = 3) -> str:
    if d["n"] == 0:
        return "n/a"
    return f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}] (n={d['n']})"


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
