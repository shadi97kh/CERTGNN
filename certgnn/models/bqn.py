"""Brain Quadratic Network (BQN) and the Hadamard-product encoder.

Reference: Yang, Liu, Zhuo, Jin, Wang, Wang & Cao, "Do We Really Need
Message Passing in Brain Network Modeling?", ICML 2025 (PMLR v267), code at
github.com/LYWJUN/BQN-demo. The paper's finding is that on brain networks
built from Pearson correlations, replacing the matrix product of message
passing, ``H_l = sigma(A H_{l-1} W_l)``, by the Hadamard product,
``H_l = H_{l-1} ⊙ (A W_l)``, improves performance; BQN adds a quadratic
term. This is the ABLATIONS 0.8 baseline: if it beats the GNN on the
connectome substrate, the critique applies directly.

What follows the reference implementation (``model/BQN.py``):
- input is the dense n x n connectivity matrix ``A`` only (no node
  features); the initial representation is ``A`` itself;
- each quadratic-perceptron layer computes
  ``act(H ⊙ (A W_R + b_R) + (H ⊙ H) W_G + b_G)`` with ``W_R, W_G`` in
  ``R^{n x n}`` applied row-wise, then dropout. (The reference defines a
  third branch ``MLP_B`` but never uses it in ``forward``; it is omitted.)
- readout: cluster pooling into ``clusters`` groups, ``Linear(n -> 8)``,
  flatten, ``256 -> 32 -> out`` MLP with LeakyReLU.

What is simplified: the reference readout is BrainNetTF's DEC orthogonal
clustering module with a KL clustering loss; here the soft cluster
assignment is a learned row-wise softmax over ``Linear(n -> clusters)``
without the auxiliary loss. ``pooling="mean"`` gives the plain variant.
The head returns one latent (logit) rather than two class scores, per the
project convention.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import to_dense_adj

_ACTS = {"gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}


class QuadraticPerceptron(nn.Module):
    """One BQN layer: ``act(H ⊙ (A W_R + b_R) + (H ⊙ H) W_G + b_G)``."""

    def __init__(
        self, n: int, activation: str = "leaky_relu", dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.mlp_r = nn.Linear(n, n)
        self.mlp_g = nn.Linear(n, n)
        self.act = _ACTS[activation]()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(h * self.mlp_r(corr) + self.mlp_g(h * h)))


class HadamardEncoderLayer(nn.Module):
    """The paper's QNN comparison encoder: ``H_l = H_{l-1} ⊙ (A W_l)``."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.w = nn.Linear(n, n, bias=False)

    def forward(self, h: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        return h * self.w(corr)


class BrainQuadraticNetwork(nn.Module):
    """BQN on a dense connectivity matrix; returns one latent per graph.

    Parameters
    ----------
    num_nodes : int
        Fixed ROI count ``n``; every graph must have exactly this many nodes.
    layers : int
        Quadratic-perceptron depth (paper: stable for 1 to 3).
    clusters : int
        Cluster-pooling groups (reference default 4).
    variant : {"bqn", "hadamard"}
        ``"hadamard"`` uses the plain Hadamard encoder layers instead.
    pooling : {"cluster", "mean"}
    """

    def __init__(
        self,
        num_nodes: int,
        layers: int = 2,
        clusters: int = 4,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
        variant: str = "bqn",
        pooling: str = "cluster",
    ) -> None:
        super().__init__()
        n = num_nodes
        self.n = n
        if variant == "bqn":
            self.layers = nn.ModuleList(
                QuadraticPerceptron(n, activation, dropout) for _ in range(layers)
            )
        elif variant == "hadamard":
            self.layers = nn.ModuleList(HadamardEncoderLayer(n) for _ in range(layers))
        else:
            raise ValueError("variant must be 'bqn' or 'hadamard'")
        self.pooling = pooling
        if pooling == "cluster":
            self.assign = nn.Linear(n, clusters)
            groups = clusters
        elif pooling == "mean":
            groups = 1
        else:
            raise ValueError("pooling must be 'cluster' or 'mean'")
        self.dim_reduction = nn.Sequential(nn.Linear(n, 8), nn.LeakyReLU())
        self.fc = nn.Sequential(
            nn.Linear(8 * groups, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        target_idx: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sizes = ptr[1:] - ptr[:-1]
        if bool((sizes != self.n).any()):
            raise ValueError(
                f"BQN needs every graph to have exactly {self.n} nodes; got sizes {sizes.tolist()[:5]}"
            )
        corr = to_dense_adj(
            edge_index, batch, edge_attr=edge_weight, max_num_nodes=self.n
        )
        if corr.dim() == 4:  # edge_attr with a trailing feature dim of 1
            corr = corr.squeeze(-1)
        h = corr.clone()
        for layer in self.layers:
            h = layer(h, corr)
        if self.pooling == "cluster":
            a = torch.softmax(
                self.assign(h), dim=1
            )  # [B, n, clusters], soft assignment per node
            a = a / a.sum(dim=1, keepdim=True).clamp_min(1e-9)
            pooled = a.transpose(1, 2) @ h  # [B, clusters, n]
        else:
            pooled = h.mean(dim=1, keepdim=True)  # [B, 1, n]
        z = self.dim_reduction(pooled).reshape(pooled.size(0), -1)
        return self.fc(z).squeeze(-1)
