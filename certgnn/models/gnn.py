"""Message-passing GNN and its edge-free MLP counterpart.

Both models return a single *latent* (logit-space) output per graph. The
head has no sigmoid: probabilities are for display only (Theorem 2). The
MLP is the ABLATIONS 0.7 control: identical per-node processing and readout
with message passing removed, so any gap between the two is attributable to
the edges.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool

Readout = Literal["target", "mean"]


def _readout(
    h: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
    target_idx: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "target":
        return h[ptr[:-1] + target_idx.reshape(-1)]
    if mode == "mean":
        return global_mean_pool(h, batch)
    raise ValueError(f"unknown readout {mode!r}")


class TargetReadoutGCN(nn.Module):
    """Message-passing GNN with target-node (or mean-pool) readout and a linear logit head.

    Parameters
    ----------
    in_dim : int
    hidden : int
    depth : int
        Number of message-passing layers.
    readout : {"target", "mean"}
    arch : {"gcn", "sage_sum", "gin"}
        Backbone: symmetric-normalized GCN, GraphSAGE with sum aggregation, or
        GIN (sum aggregation with an MLP update). Sum aggregation preserves
        degree/position information that GCN's normalization averages away.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        depth: int = 3,
        readout: str = "target",
        arch: str = "gcn",
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        convs: list[nn.Module]
        if arch == "gcn":
            convs = [GCNConv(dims[i], dims[i + 1]) for i in range(depth)]
        elif arch == "sage_sum":
            convs = [SAGEConv(dims[i], dims[i + 1], aggr="add") for i in range(depth)]
        elif arch == "gin":
            convs = [
                GINConv(
                    nn.Sequential(
                        nn.Linear(dims[i], hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, dims[i + 1]),
                    )
                )
                for i in range(depth)
            ]
        else:
            raise ValueError("arch must be gcn, sage_sum or gin")
        self.convs = nn.ModuleList(convs)
        self.arch = arch
        self.head = nn.Linear(hidden, 1)
        self.readout = readout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        return self.head(_readout(h, batch, ptr, target_idx, self.readout)).squeeze(-1)


class NodeFeatureMLP(nn.Module):
    """Per-node MLP with the same readout and head; no message passing.

    ``edge_index`` is accepted and ignored, so the model can be evaluated on
    the same batches as the GNN. This is the 'edges removed' control.
    """

    def __init__(
        self, in_dim: int, hidden: int = 64, depth: int = 3, readout: str = "target"
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
        self.readout = readout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        h = self.mlp(x)
        return self.head(_readout(h, batch, ptr, target_idx, self.readout)).squeeze(-1)
