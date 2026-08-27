"""Substrate protocol. Core code depends on this and nothing more concrete."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx
import torch
from torch_geometric.data import Data


@runtime_checkable
class Substrate(Protocol):
    name: str

    def load(self, split: str) -> list[Data]:
        """Return graphs. Node features in `x`, target index in `target_idx`."""
        ...

    def target_node(self, data: Data) -> int:
        """Node whose readout the model predicts."""
        ...

    def candidate_nodes(self, data: Data) -> torch.Tensor:
        """Nodes eligible to appear in an explanation. For splice, intronic
        windows. For connectome, edges/regions of interest."""
        ...

    def ground_truth_window(self, data: Data) -> torch.Tensor | None:
        """Annotated functional region, if known. None for most instances.
        Only instances returning non-None enter the restricted-scope certificate."""
        ...

    def to_networkx(self, data: Data) -> nx.Graph: ...

    def baseline_rate(self, data: Data) -> float:
        """Baseline output level, used for Mondrian stratification."""
        ...
