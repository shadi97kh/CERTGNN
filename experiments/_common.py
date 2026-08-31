"""Shared plumbing for experiments: config, run directories, bootstrap CIs.

Every experiment writes to ``results/runs/<timestamp>_<gitsha>_<confighash>/``
holding the resolved config, git SHA and dirty flag, the seed(s), an
environment lockfile, and the raw per-seed values. A number without a git
SHA and a config hash does not exist.

Config layout: ``configs/base.yaml`` names a selection per group under
``groups:`` (substrate, model, explainer, rewire, conformal); each selection
is merged from ``configs/<group>/<name>.yaml`` under the key ``<group>``.
Overrides are Hydra-style dotlists: ``model=gin`` changes a selection,
``model.hidden=128`` changes a value, ``gate2.coverage.n_test=2000`` any key.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


def configure_torch(cfg: DictConfig) -> None:
    """Pin the CPU thread count; oversubscription makes scatter ops pathologically slow."""
    import torch

    torch.set_num_threads(int(cfg.get("torch_threads", 1)))


GROUPS = ("substrate", "model", "explainer", "rewire", "conformal")


def _split_overrides(overrides: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Separate ``group=name`` selections from dotted value overrides."""
    selections: dict[str, str] = {}
    dotlist: list[str] = []
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} is not key=value")
        key, val = ov.split("=", 1)
        if key in GROUPS:
            selections[key] = val
        else:
            dotlist.append(ov)
    return selections, dotlist


def load_config(path: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Base YAML + group files + dotlist overrides, fully resolved."""
    base = OmegaConf.load(path)
    assert isinstance(base, DictConfig)
    selections, dotlist = _split_overrides(overrides)
    groups = dict(base.get("groups", {}) or {})
    groups.update(selections)
    root = pathlib.Path(path).parent
    merged: Any = base
    if bool(base.get("_resolved", False)):
        groups = {}  # a generated config already carries its group contents
    for group, name in groups.items():
        gp = root / group / f"{name}.yaml"
        if not gp.exists():
            raise FileNotFoundError(
                f"config group file {gp} for {group}={name} does not exist"
            )
        merged = OmegaConf.merge(merged, OmegaConf.create({group: OmegaConf.load(gp)}))
    if groups:
        merged = OmegaConf.merge(merged, OmegaConf.create({"groups": groups}))
    if dotlist:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(dotlist))
    assert isinstance(merged, DictConfig)
    return merged


def resolve_seeds(cfg: DictConfig) -> list[int]:
    """``[cfg.seed]`` when a single seed is set (sweep launches), else ``range(cfg.seeds)``."""
    if cfg.get("seed") is not None:
        return [int(cfg.seed)]
    return list(range(int(cfg.seeds)))


def config_hash(cfg: DictConfig) -> str:
    text = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], text=True
    ).strip()


OUTPUT_PREFIXES = ("results/", "paper/", "configs/generated/")


def git_dirty() -> bool:
    """True if any tracked file is modified, or any file outside the output
    directories is untracked. Run artifacts (results/, paper/,
    configs/generated/) do not count: the sweep launcher writes its manifest
    before launching, and outputs of one run must not block the next."""
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    for line in out.splitlines():
        status, path = line[:2], line[3:]
        if status == "??" and path.startswith(OUTPUT_PREFIXES):
            continue
        return True
    return False


def environment_lock() -> str:
    """Interpreter, platform and the frozen package list."""
    try:
        frozen = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:  # pragma: no cover - pip missing
        frozen = f"# pip freeze failed: {e}\n"
    return f"# python {sys.version.split()[0]} on {platform.platform()}\n{frozen}"


def make_run_dir(
    cfg: DictConfig, name: str, *, allow_dirty: bool | None = None
) -> pathlib.Path:
    """Create the run directory and record config, provenance and environment."""
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    sha, dirty, h = git_sha(), git_dirty(), config_hash(cfg)
    if dirty and not allow_dirty:
        raise SystemExit(
            "REFUSING TO RUN: git working tree is dirty, so no SHA identifies this code. "
            "Commit first, or pass --allow-dirty (the flag is recorded in meta.json)."
        )
    stem = f"{ts}_{sha}{'-dirty' if dirty else ''}_{h}"
    root = pathlib.Path(cfg.output.runs)
    # Timestamps have one-second resolution, so two runs of the same code and
    # config started in the same second would collide; never reuse or silently
    # share a run directory, take the next free suffix instead.
    run = root / stem
    for n in range(1, 100):
        try:
            run.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run = root / f"{stem}-{n}"
    else:
        raise RuntimeError(f"could not create a fresh run directory for {stem}")
    (run / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (run / "environment.lock").write_text(environment_lock())
    meta = {
        "experiment": name,
        "git_sha": sha,
        "git_dirty": dirty,
        "allow_dirty": allow_dirty,
        "config_hash": h,
        "seeds": resolve_seeds(cfg),
        "timestamp_utc": ts,
        "argv": sys.argv,
        "run_id": cfg.get("run_id"),
        "sweep": OmegaConf.to_container(cfg.sweep)
        if cfg.get("sweep") is not None
        else None,
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    if dirty:
        print(
            "WARNING: git tree is dirty; this run's SHA does not identify its code",
            file=sys.stderr,
        )
    return run


def write_tuning_budget(run: pathlib.Path, entries: list[dict[str, Any]]) -> None:
    """Record the hyperparameter search effort per model for results/tuning_budget.md.

    Each entry: ``{"model": str, "configs_tried": int, "epochs": int,
    "gradient_steps": int, "search_space": str, "selection": str}``.
    """
    (run / "tuning_budget.json").write_text(json.dumps(entries, indent=2))


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


def bootstrap_p_value(
    values: Sequence[float],
    *,
    threshold: float,
    direction: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """One-sided bootstrap p-value that the seed-mean fails a threshold.

    ``direction="greater"``: H1 is mean > threshold; p = fraction of bootstrap
    means <= threshold. ``"less"``: H1 is mean < threshold; p = fraction >=.
    Uses the (b+1)/(B+1) convention so a zero never masquerades as exact.
    This is a CI-inversion bootstrap (resampling the observed seed values),
    not a null-calibrated test; it saturates near 1/(B+1) whenever the
    threshold lies outside the range of the seed values.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    if direction == "greater":
        b = int((boots <= threshold).sum())
    elif direction == "less":
        b = int((boots >= threshold).sum())
    else:
        raise ValueError("direction must be 'greater' or 'less'")
    return (b + 1.0) / (
        n_boot + 1.0
    )  # Davison & Hinkley (b+1)/(B+1); never exactly zero


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


def void_if_unsearched(n_examined: int, what: str, detail: str = "") -> str | None:
    """A VOID verdict when a search examined nothing, else None.

    Distinguishes two situations a verdict must never conflate:

    * **searched and found nothing** -- a real negative, and reportable;
    * **never actually searched** -- no information at all.

    `signlayer`'s depth-witness run hit the second and printed "No witness
    found", a false negative, because every cell it was asked to search had an
    empty candidate set for structural reasons. A negative asserted from zero
    candidates is not weak evidence, it is none, and it is worse than silence
    because it reads as a result.

    Call this at the top of any verdict computed from a filtered or sampled
    candidate set, and return its value when it is not None.
    """
    if n_examined > 0:
        return None
    tail = f" {detail}" if detail else ""
    return (
        f"**VOID -- nothing was searched.** The candidate set for {what} was "
        f"empty, so this run supports no conclusion in either direction; in "
        f"particular it is NOT a negative result.{tail} Reporting one from an "
        "empty search would be a false negative."
    )
