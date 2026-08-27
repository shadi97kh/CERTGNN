"""Minimal, seed-deterministic training and evaluation for control experiments.

Task is inferred from the labels: binary {0, 1} targets are classified with
BCE-with-logits and scored by AUROC and accuracy; anything else is regressed
with MSE and scored by R^2. Model selection uses the validation split only.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader


@dataclass(frozen=True)
class FitResult:
    """Outcome of one fit.

    ``model`` carries the trained network with the selected epoch's weights
    already loaded. It defaults to ``None`` so existing callers that only read
    metrics are unaffected; experiments that need to *evaluate* the trained
    model -- rather than an oracle standing in for it -- read this field.
    """

    task: str
    best_epoch: int
    val_metric: float
    test_metrics: dict[str, float]
    model: torch.nn.Module | None = None


def infer_task(data: list) -> str:
    ys = torch.cat([d.y.reshape(-1).float() for d in data])
    return "classification" if bool(((ys == 0) | (ys == 1)).all()) else "regression"


def _forward(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    return model(batch.x, batch.edge_index, batch.batch, batch.ptr, batch.target_idx)


@torch.no_grad()
def predict(
    model: torch.nn.Module, data: list, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outs, ys = [], []
    for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
        outs.append(_forward(model, batch).detach().cpu())
        ys.append(batch.y.reshape(-1).float().cpu())
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def metrics(pred_latent: np.ndarray, y: np.ndarray, task: str) -> dict[str, float]:
    if task == "classification":
        p = 1.0 / (1.0 + np.exp(-pred_latent))
        out = {"accuracy": float(((p >= 0.5) == (y >= 0.5)).mean())}
        out["auroc"] = (
            float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
        )
        return out
    ss_res = float(((y - pred_latent) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mse": ss_res / len(y),
    }


def primary_metric(task: str) -> str:
    return "auroc" if task == "classification" else "r2"


def fit(
    make_model: Callable[[], torch.nn.Module],
    train: list,
    val: list,
    test: list,
    *,
    seed: int,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 0.0,
) -> FitResult:
    """Train with Adam, select the epoch by validation metric, score on test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    task = infer_task(train)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = (
        torch.nn.BCEWithLogitsLoss() if task == "classification" else torch.nn.MSELoss()
    )
    loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    key = primary_metric(task)
    best_val, best_state, best_epoch = (
        -float("inf"),
        copy.deepcopy(model.state_dict()),
        0,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            opt.zero_grad()
            loss = loss_fn(_forward(model, batch), batch.y.reshape(-1).float())
            loss.backward()
            opt.step()
        pv, yv = predict(model, val)
        mv = metrics(pv, yv, task)[key]
        if np.isfinite(mv) and mv > best_val:
            best_val, best_state, best_epoch = (
                mv,
                copy.deepcopy(model.state_dict()),
                epoch,
            )
    model.load_state_dict(best_state)
    pt, yt = predict(model, test)
    return FitResult(
        task=task,
        best_epoch=best_epoch,
        val_metric=best_val,
        test_metrics=metrics(pt, yt, task),
        model=model,
    )
