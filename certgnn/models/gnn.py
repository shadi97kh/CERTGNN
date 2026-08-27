"""Message-passing GNN and its edge-free MLP counterpart.

Both models return a single *latent* (logit-space) output per graph. The
head has no sigmoid: probabilities are for display only (Theorem 2). The
MLP is the ABLATIONS 0.7 control: identical per-node processing and readout
with message passing removed, so any gap between the two is attributable to
the edges.

Both accept an optional soft node ``mask``. It multiplies the node states at
the input and after every layer, so a node held near 0 contributes almost
nothing to its neighbours at any hop -- the smooth analogue of deleting it.
Soft only: the mask is never thresholded here, because the Jacobian arguments
of Theorems 3 to 5 require a differentiable path (see ``certgnn/explain/masks.py``).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool

Readout = Literal["target", "mean"]


def _apply_mask(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Scale node states by a soft mask.

    Parameters
    ----------
    h : torch.Tensor
        Node states, shape ``(num_nodes, dim)``.
    mask : torch.Tensor or None
        Per-node multipliers in ``[0, 1]``, shape ``(num_nodes,)``. ``None``
        leaves ``h`` untouched, which is the unmasked forward pass.

    Returns
    -------
    torch.Tensor
        ``h`` scaled row-wise, in ``h``'s dtype so the mask may be float64
        while the model runs in float32.
    """
    if mask is None:
        return h
    if mask.dim() != 1 or mask.numel() != h.size(0):
        raise ValueError(
            f"mask must be one value per node: got {tuple(mask.shape)} "
            f"for {h.size(0)} nodes"
        )
    return h * mask.to(h.dtype).unsqueeze(-1)


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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Latent output per graph, optionally under a soft node mask.

        The mask is applied to the input and after every message-passing layer.
        Applying it once at the input would only attenuate a node's own
        features while leaving it free to relay its neighbours' messages, which
        is not what masking a node means.
        """
        h = _apply_mask(x, mask)
        for conv in self.convs:
            h = _apply_mask(torch.relu(conv(h, edge_index)), mask)
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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Latent output per graph, optionally under a soft node mask.

        Accepted for interface parity with the GNN. With no message passing a
        mask can only attenuate the readout node itself, so this control is
        insensitive to masking anything else -- which is the point of it.
        """
        h = _apply_mask(self.mlp(_apply_mask(x, mask)), mask)
        return self.head(_readout(h, batch, ptr, target_idx, self.readout)).squeeze(-1)
