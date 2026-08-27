"""Series-parallel motif generators with known regulator-to-target structure.

Every graph built here is series-parallel (no K4 minor), so the exact
reduction in ``certgnn.topology.resistance`` applies. The target is always
node 0. Distractor leaves and pendant chains carry no current between the
regulator and the target and therefore have zero effect by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np

TopologyKind = Literal["path", "bubble", "nested_bubbles", "multi_parallel", "pendant"]
TOPOLOGY_KINDS: tuple[str, ...] = (
    "path",
    "bubble",
    "nested_bubbles",
    "multi_parallel",
    "pendant",
)


@dataclass(frozen=True)
class TopologySpec:
    """Dials for one motif family.

    Parameters
    ----------
    kind : str
        One of ``TOPOLOGY_KINDS``.
    length : int
        ``path``: number of edges from target to regulator. Bubble-like kinds:
        edges per branch (must be >= 2 so branches are not parallel edges).
    n_branches : int
        Parallel branches for ``multi_parallel``, ``nested_bubbles`` and
        ``pendant``. ``bubble`` always uses 2.
    depth : int
        ``nested_bubbles``: nesting depth (1 is a plain bubble of 2-edge
        branches). ``pendant``: edges in the pendant chain that carries the
        regulator beyond the bubble.
    stem : int
        Edges between the target and the bubble entry (0 means the target is
        the entry).
    n_distractors : int
        Zero-effect leaf nodes attached to motif nodes.
    pendant_chains : int
        Zero-effect chains of ``pendant_depth`` edges attached to motif nodes.
    pendant_depth : int
        Length of each zero-effect chain.
    """

    kind: str = "bubble"
    length: int = 2
    n_branches: int = 2
    depth: int = 2
    stem: int = 0
    n_distractors: int = 0
    pendant_chains: int = 0
    pendant_depth: int = 2

    def __post_init__(self) -> None:
        if self.kind not in TOPOLOGY_KINDS:
            raise ValueError(
                f"unknown topology {self.kind!r}; expected {TOPOLOGY_KINDS}"
            )
        if self.length < 1:
            raise ValueError("length must be >= 1")
        if self.kind != "path" and self.length < 2:
            raise ValueError("branch length must be >= 2 for bubble-like motifs")
        if self.n_branches < 2:
            raise ValueError("n_branches must be >= 2")
        if self.kind == "multi_parallel" and self.n_branches < 3:
            raise ValueError("multi_parallel needs n_branches >= 3; use 'bubble' for 2")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if min(self.stem, self.n_distractors, self.pendant_chains) < 0:
            raise ValueError("stem, n_distractors, pendant_chains must be >= 0")
        if self.pendant_depth < 1:
            raise ValueError("pendant_depth must be >= 1")


@dataclass(frozen=True)
class Motif:
    """A built motif.

    Attributes
    ----------
    graph : nx.Graph
        Unit-resistance graph; node 0 is the target.
    target : int
    regulator : int
        Default ("far") regulator position.
    motif_nodes : tuple[int, ...]
        Nodes of the core motif, i.e. excluding distractors and pendant
        chains. Candidates for a randomly placed regulator.
    """

    graph: nx.Graph
    target: int
    regulator: int
    motif_nodes: tuple[int, ...]


class _Builder:
    def __init__(self) -> None:
        self.G = nx.Graph()
        self.G.add_node(0)
        self._next = 1

    def new(self) -> int:
        v = self._next
        self._next += 1
        self.G.add_node(v)
        return v

    def path(self, u: int, v: int, edges: int) -> list[int]:
        """Connect u to v through ``edges`` edges; returns interior nodes."""
        if edges < 1:
            raise ValueError("path needs >= 1 edge")
        interior: list[int] = []
        prev = u
        for _ in range(edges - 1):
            w = self.new()
            self.G.add_edge(prev, w)
            interior.append(w)
            prev = w
        self.G.add_edge(prev, v)
        return interior

    def chain(self, u: int, edges: int) -> int:
        """Append a chain of ``edges`` edges hanging off u; returns its end."""
        end = u
        for _ in range(edges):
            w = self.new()
            self.G.add_edge(end, w)
            end = w
        return end

    def bubble(self, u: int, v: int, n_branches: int, length: int) -> None:
        for _ in range(n_branches):
            self.path(u, v, length)

    def nested(self, u: int, v: int, n_branches: int, depth: int) -> None:
        """Bubbles of bubbles: parallel composition of series compositions."""
        if depth == 0:
            self.G.add_edge(u, v)
            return
        for _ in range(n_branches):
            m = self.new()
            self.nested(u, m, n_branches, depth - 1)
            self.nested(m, v, n_branches, depth - 1)


def build_motif(spec: TopologySpec, rng: np.random.Generator) -> Motif:
    """Build the motif described by ``spec``.

    Distractor and pendant-chain attachment points are drawn from ``rng``;
    the core motif is deterministic given ``spec``.
    """
    b = _Builder()
    target = 0
    entry = target
    if spec.stem > 0:
        entry = b.new()
        b.path(target, entry, spec.stem)

    if spec.kind == "path":
        regulator = b.new()
        b.path(entry, regulator, spec.length)
    elif spec.kind in ("bubble", "multi_parallel"):
        k = 2 if spec.kind == "bubble" else spec.n_branches
        regulator = b.new()
        b.bubble(entry, regulator, k, spec.length)
    elif spec.kind == "nested_bubbles":
        regulator = b.new()
        b.nested(entry, regulator, spec.n_branches, spec.depth)
    elif spec.kind == "pendant":
        junction = b.new()
        b.bubble(entry, junction, spec.n_branches, spec.length)
        regulator = b.chain(junction, spec.depth)
    else:  # pragma: no cover - guarded by TopologySpec
        raise ValueError(spec.kind)

    motif_nodes = tuple(sorted(b.G.nodes()))
    attach_pool = [v for v in motif_nodes]
    for _ in range(spec.n_distractors):
        host = int(rng.choice(attach_pool))
        b.chain(host, 1)
    for _ in range(spec.pendant_chains):
        host = int(rng.choice(attach_pool))
        b.chain(host, spec.pendant_depth)

    return Motif(graph=b.G, target=target, regulator=regulator, motif_nodes=motif_nodes)
