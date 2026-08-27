"""Ground-truth mechanism: electrical propagation from regulator to target.

The regulator ``r`` carries a signal ``z_r``. Under a soft node mask ``m``
(``m_t = 1`` always) every edge ``(u, v)`` has conductance ``m_u * m_v`` and
the target's latent is

    eta(m) = eta0 + delta * z_r * kappa(m),
    kappa(m) = C_m(r, t) / C_1(r, t),

where ``C`` is the effective conductance (reciprocal effective resistance)
between regulator and target. Consequences that the theorems rely on hold by
construction: pendant nodes carry no current, so masking them changes
nothing; parallel branches combine harmonically; ``kappa`` is monotone in
the unmasked set (Rayleigh); the empty circuit gives ``kappa = 0``.

An epsilon-sufficient explanation is a node set ``S`` (target excluded) with
``|eta(1_S) - eta(1)| <= eps`` in latent space. Minimal such sets are
computed exactly here, by brute force and by a pruned exact search.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import networkx as nx
import torch

_TOL = 1e-9


class Oracle:
    """Exact ground truth for one instance.

    Parameters
    ----------
    graph : nx.Graph
        Unit-resistance graph.
    regulator, target : int
    eta0 : float
        Baseline latent.
    delta : float
        Latent effect size of the regulator.
    z_r : float
        Regulator signal feature.
    """

    def __init__(
        self,
        graph: nx.Graph,
        regulator: int,
        target: int,
        eta0: float,
        delta: float,
        z_r: float,
    ) -> None:
        if regulator == target:
            raise ValueError("regulator must differ from target")
        self.graph = graph
        self.nodes: list[int] = sorted(graph.nodes())
        self.index = {v: i for i, v in enumerate(self.nodes)}
        self.n = len(self.nodes)
        self.edges: list[tuple[int, int]] = [
            (self.index[u], self.index[v]) for u, v in graph.edges()
        ]
        self.regulator = regulator
        self.target = target
        self.r = self.index[regulator]
        self.t = self.index[target]
        self.eta0 = float(eta0)
        self.delta = float(delta)
        self.z_r = float(z_r)
        self._c_full = float(self._conductance(torch.ones(self.n, dtype=torch.float64)))
        if self._c_full <= 0:
            raise ValueError("regulator is not connected to target")

    # ---------------------------------------------------------------- masks

    def full_mask(self) -> torch.Tensor:
        return torch.ones(self.n, dtype=torch.float64)

    def set_mask(self, keep: Iterable[int]) -> torch.Tensor:
        """Binary mask keeping ``keep`` (node labels) and the target."""
        m = torch.zeros(self.n, dtype=torch.float64)
        for v in keep:
            m[self.index[v]] = 1.0
        m[self.t] = 1.0
        return m

    # ------------------------------------------------------------ mechanism

    def _conductance(self, mask: torch.Tensor) -> torch.Tensor:
        """Effective conductance r->t with edge conductance ``m_u m_v``.

        Solves the grounded Laplacian restricted to the support component of
        the regulator (edges with positive conductance), which is positive
        definite; returns 0 when regulator and target are disconnected.
        """
        m = mask.to(torch.float64)
        cond = (
            torch.stack([m[u] * m[v] for u, v in self.edges]) if self.edges else m[:0]
        )
        support = nx.Graph()
        support.add_nodes_from(range(self.n))
        support.add_edges_from(
            (u, v) for (u, v), c in zip(self.edges, cond.tolist()) if c > 0.0
        )
        if not nx.has_path(support, self.r, self.t):
            return torch.zeros((), dtype=torch.float64)
        comp = sorted(nx.node_connected_component(support, self.r))
        keep = [v for v in comp if v != self.t]
        pos = {v: i for i, v in enumerate(keep)}
        k = len(keep)
        L = torch.zeros(k, k, dtype=torch.float64)
        for (u, v), c in zip(self.edges, cond):
            if u in pos:
                L[pos[u], pos[u]] = L[pos[u], pos[u]] + c
            if v in pos:
                L[pos[v], pos[v]] = L[pos[v], pos[v]] + c
            if u in pos and v in pos:
                L[pos[u], pos[v]] = L[pos[u], pos[v]] - c
                L[pos[v], pos[u]] = L[pos[v], pos[u]] - c
        rhs = torch.zeros(k, dtype=torch.float64)
        rhs[pos[self.r]] = 1.0
        x = torch.linalg.solve(L, rhs)
        return 1.0 / x[pos[self.r]]

    def kappa(self, mask: torch.Tensor) -> torch.Tensor:
        """Fraction of full-graph conductance retained under ``mask``."""
        if mask.numel() != self.n:
            raise ValueError(f"mask has {mask.numel()} entries, graph has {self.n}")
        if float(mask[self.t].detach()) != 1.0:
            raise ValueError("the target node must not be masked")
        if bool((mask < 0).any()) or bool((mask > 1).any()):
            raise ValueError("mask entries must lie in [0, 1]")
        return self._conductance(mask) / self._c_full

    def latent(self, mask: torch.Tensor) -> torch.Tensor:
        """``eta(m)``; differentiable in a soft ``mask``."""
        return self.eta0 + self.delta * self.z_r * self.kappa(mask)

    def latent_full(self) -> float:
        return self.eta0 + self.delta * self.z_r

    def latent_effect(self) -> float:
        """``eta(1) - eta0``, the regulator's full latent effect."""
        return self.delta * self.z_r

    # ------------------------------------------------------- sufficiency

    def is_sufficient(self, keep: Iterable[int], eps: float) -> bool:
        gap = abs(self.latent_full() - float(self.latent(self.set_mask(keep))))
        return gap <= eps + _TOL

    def candidates(self) -> list[int]:
        return [v for v in self.nodes if v != self.target]

    def relevant_nodes(self) -> list[int]:
        """Nodes on some simple regulator-target path (Menger criterion).

        A node ``v`` lies on a simple r-t path iff there are two internally
        node-disjoint paths from ``v`` to a super-sink adjacent only to ``r``
        and ``t``. Nodes not on any such path carry no current in any
        sub-circuit, so a minimal sufficient set never needs them.
        """
        H = self.graph.copy()
        sink = ("sink",)
        H.add_edge(self.regulator, sink)
        H.add_edge(self.target, sink)
        rel = [self.regulator]
        for v in self.nodes:
            if v in (self.regulator, self.target):
                continue
            if nx.node_connectivity(H, v, sink) >= 2:
                rel.append(v)
        return sorted(rel)

    def minimal_sufficient_set_bruteforce(self, eps: float) -> list[int]:
        """Smallest epsilon-sufficient set by exhaustive enumeration.

        Enumerates all subsets of the candidates in increasing size, so the
        first feasible subset is minimal. Exponential; intended for graphs
        under roughly 16 nodes.
        """
        if abs(self.latent_effect()) <= eps + _TOL:
            return []
        cands = self.candidates()
        for size in range(1, len(cands) + 1):
            for subset in itertools.combinations(cands, size):
                if self.is_sufficient(subset, eps):
                    return sorted(subset)
        raise RuntimeError("full candidate set is not sufficient; eps < 0?")

    def minimal_sufficient_set(self, eps: float) -> list[int]:
        """Smallest epsilon-sufficient set by pruned exact search.

        Prunes to nodes on some r-t path, requires the regulator, and skips
        subsets that do not connect regulator to target before solving the
        circuit. Still exact: it enumerates subsets of the pruned pool in
        increasing size.
        """
        if abs(self.latent_effect()) <= eps + _TOL:
            return []
        pool = [v for v in self.relevant_nodes() if v != self.regulator]
        for size in range(0, len(pool) + 1):
            for extra in itertools.combinations(pool, size):
                subset = (self.regulator, *extra)
                sub = self.graph.subgraph((*subset, self.target))
                if not nx.has_path(sub, self.regulator, self.target):
                    continue
                if self.is_sufficient(subset, eps):
                    return sorted(subset)
        raise RuntimeError("relevant node set is not sufficient; eps < 0?")


def subset_mask(oracle: Oracle, keep: Sequence[int]) -> torch.Tensor:
    """Convenience alias for ``oracle.set_mask``."""
    return oracle.set_mask(keep)
