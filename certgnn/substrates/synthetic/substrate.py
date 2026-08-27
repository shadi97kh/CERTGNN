"""The synthetic substrate: ground truth by construction.

Implements the ``Substrate`` protocol from ``certgnn.substrates.base`` and
adds substrate-specific accessors (oracle, minimal sufficient sets,
likelihood ratio) that the protocol has no hook for. ``base.py`` is not
modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from certgnn.substrates.synthetic.links import LINK_NAMES, Link, get_link
from certgnn.substrates.synthetic.oracle import Oracle
from certgnn.substrates.synthetic.shift import GaussianShift
from certgnn.substrates.synthetic.topology import Motif, TopologySpec, build_motif

SPLITS: tuple[str, ...] = ("train", "val", "cal", "test")


@dataclass(frozen=True)
class SyntheticConfig:
    """All dials of the generator.

    Parameters
    ----------
    topology : TopologySpec
    regulator : int | str
        ``"far"`` places the regulator at the motif's far end; ``"random"``
        draws it per instance from the motif nodes (never the target); an
        int fixes a node label.
    delta : float
        Latent effect size of the regulator.
    link : str
        One of ``identity, logit, probit, cloglog``.
    p0 : float | None
        Fixed baseline rate. ``None`` samples ``eta0 ~ N(eta0_mean, eta0_std)``
        so instances span the full range of ``p0``.
    eta0_mean, eta0_std : float
        Train/val/cal distribution of ``eta0`` when ``p0`` is ``None``.
    noise : float
        Observation noise scale: ``y = link(eta) + noise * N(0, 1)``.
    shift_eta0_mean, shift_eta0_std : float | None
        Test-split distribution of ``eta0``; ``None`` means no shift. Requires
        ``p0 is None``. The likelihood ratio is then closed form.
    target_indicator : bool
        Append a one-hot ``is_target`` column. The target is known to every
        model (it is the readout node), so this leaks nothing; message
        passing can turn it into distance-to-target, an edge-free model
        cannot.
    structural_features : bool
        Append ``degree / max degree`` and ``distance to target / max
        distance`` to the node features. These leak topology into the
        features, which defeats the edge-free MLP control (ABLATIONS 0.7);
        set ``False`` for that control.
    n_train, n_val, n_cal, n_test : int
    seed : int
    """

    topology: TopologySpec = field(default_factory=TopologySpec)
    regulator: int | str = "far"
    delta: float = 1.0
    link: str = "logit"
    p0: float | None = None
    eta0_mean: float = 0.0
    eta0_std: float = 2.0
    noise: float = 0.0
    shift_eta0_mean: float | None = None
    shift_eta0_std: float | None = None
    structural_features: bool = True
    target_indicator: bool = False
    n_train: int = 200
    n_val: int = 50
    n_cal: int = 200
    n_test: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        if self.link not in LINK_NAMES:
            raise ValueError(f"link must be one of {LINK_NAMES}, got {self.link!r}")
        if self.p0 is not None and get_link(self.link).bounded:
            if not 0.0 < self.p0 < 1.0:
                raise ValueError("fixed p0 must lie strictly inside (0, 1)")
        if self.p0 is not None and (
            self.shift_eta0_mean is not None or self.shift_eta0_std is not None
        ):
            raise ValueError("a shift on eta0 requires p0=None (sampled baseline)")
        if self.eta0_std <= 0:
            raise ValueError("eta0_std must be positive")
        if self.shift_eta0_std is not None and self.shift_eta0_std <= 0:
            raise ValueError("shift_eta0_std must be positive")
        if self.noise < 0:
            raise ValueError("noise must be >= 0")
        if isinstance(self.regulator, str) and self.regulator not in ("far", "random"):
            raise ValueError("regulator must be 'far', 'random', or an int node label")
        if min(self.n_train, self.n_val, self.n_cal, self.n_test) < 0:
            raise ValueError("split sizes must be >= 0")


class SyntheticSubstrate:
    """Synthetic substrate with exact ground truth.

    Instances carry, beyond the protocol's ``x``, ``edge_index`` and
    ``target_idx``: ``y`` (noisy observation), ``mu`` (noise-free mean),
    ``eta`` (latent), ``eta0``, ``p0``, ``delta``, ``regulator``,
    ``likelihood_ratio``, ``link``, ``topology`` and ``split``. Node feature
    columns are ``[z, degree / max degree, distance to target / max
    distance]`` (structural columns and the optional ``is_target`` column
    per config); the regulator's ``z`` is the signal.
    """

    name = "synthetic"

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.link: Link = get_link(self.config.link)
        self.shift: GaussianShift | None = None
        if (
            self.config.shift_eta0_mean is not None
            or self.config.shift_eta0_std is not None
        ):
            self.shift = GaussianShift(
                mu_train=self.config.eta0_mean,
                sigma_train=self.config.eta0_std,
                mu_test=(
                    self.config.shift_eta0_mean
                    if self.config.shift_eta0_mean is not None
                    else self.config.eta0_mean
                ),
                sigma_test=(
                    self.config.shift_eta0_std
                    if self.config.shift_eta0_std is not None
                    else self.config.eta0_std
                ),
            )

    # ---------------------------------------------------------- protocol

    def load(self, split: str) -> list[Data]:
        """Generate the split deterministically from ``(seed, split)``.

        ``test`` draws ``eta0`` from the shifted distribution when a shift is
        configured; ``train``, ``val`` and ``cal`` never do.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        n = {
            "train": self.config.n_train,
            "val": self.config.n_val,
            "cal": self.config.n_cal,
            "test": self.config.n_test,
        }[split]
        rng = np.random.default_rng([self.config.seed, SPLITS.index(split)])
        shifted = split == "test" and self.shift is not None
        return [self._instance(rng, split, shifted) for _ in range(n)]

    def target_node(self, data: Data) -> int:
        return int(data.target_idx)

    def candidate_nodes(self, data: Data) -> torch.Tensor:
        t = int(data.target_idx)
        return torch.tensor([v for v in range(int(data.num_nodes)) if v != t])

    def ground_truth_window(self, data: Data) -> torch.Tensor | None:
        """The true regulator. The minimal epsilon-sufficient set depends on
        epsilon, which the protocol does not pass; see
        ``minimal_sufficient_set``."""
        return data.regulator.reshape(1)

    def to_networkx(self, data: Data) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(int(data.num_nodes)))
        ei = data.edge_index
        for u, v in zip(ei[0].tolist(), ei[1].tolist()):
            if u < v:
                G.add_edge(u, v)
        return G

    def baseline_rate(self, data: Data) -> float:
        """``p0 = link(eta0)``. Under the identity link this is ``eta0`` itself."""
        return float(data.p0)

    # ------------------------------------------------- substrate-specific

    def oracle(self, data: Data) -> Oracle:
        return Oracle(
            graph=self.to_networkx(data),
            regulator=int(data.regulator),
            target=int(data.target_idx),
            eta0=float(data.eta0),
            delta=float(data.delta),
            z_r=float(data.x[int(data.regulator), 0]),
        )

    def minimal_sufficient_set(self, data: Data, eps: float) -> list[int]:
        """Exact minimal epsilon-sufficient set (pruned search)."""
        return self.oracle(data).minimal_sufficient_set(eps)

    def minimal_sufficient_set_bruteforce(self, data: Data, eps: float) -> list[int]:
        """Exact minimal epsilon-sufficient set by exhaustive enumeration."""
        return self.oracle(data).minimal_sufficient_set_bruteforce(eps)

    def likelihood_ratio(self, data: Data) -> float:
        """``dP_test / dP_train`` at this instance's covariate ``eta0``.

        Exactly 1 when no shift is configured. Valid for instances of any
        split, which is what weighted conformal needs: calibration weights
        are ``w(eta0_i)`` and the test weight is ``w(eta0_test)``.
        """
        return float(data.likelihood_ratio)

    def observed_effect(self, data: Data) -> float:
        """Noise-free probability-space effect ``link(eta) - link(eta0)``."""
        return float(data.mu) - float(self.link.forward(data.eta0.to(torch.float64)))

    def latent_effect(self, data: Data) -> float:
        """``eta - eta0 = delta * z_r``."""
        return float(data.eta) - float(data.eta0)

    # ---------------------------------------------------------- internals

    def _pick_regulator(self, motif: Motif, rng: np.random.Generator) -> int:
        reg = self.config.regulator
        if reg == "far":
            return motif.regulator
        if reg == "random":
            pool = [v for v in motif.motif_nodes if v != motif.target]
            return int(rng.choice(pool))
        assert isinstance(reg, int)
        if reg == motif.target or reg not in motif.graph:
            raise ValueError(f"regulator {reg} is the target or not in the motif")
        return reg

    def _instance(self, rng: np.random.Generator, split: str, shifted: bool) -> Data:
        motif = build_motif(self.config.topology, rng)
        G = motif.graph
        n = G.number_of_nodes()
        regulator = self._pick_regulator(motif, rng)
        target = motif.target

        if self.config.p0 is not None:
            eta0 = float(
                self.link.inverse(torch.tensor(self.config.p0, dtype=torch.float64))
            )
        else:
            mean = self.config.eta0_mean
            std = self.config.eta0_std
            if shifted and self.shift is not None:
                mean, std = self.shift.mu_test, self.shift.sigma_test
            eta0 = float(rng.normal(mean, std))

        z = rng.standard_normal(n)
        eta = eta0 + self.config.delta * float(z[regulator])  # kappa(full) == 1
        mu = float(self.link.forward(torch.tensor(eta, dtype=torch.float64)))
        y = mu + self.config.noise * float(rng.standard_normal())
        p0 = float(self.link.forward(torch.tensor(eta0, dtype=torch.float64)))

        deg = np.array([G.degree(v) for v in range(n)], dtype=np.float64)
        dist = nx.single_source_shortest_path_length(G, target)
        d = np.array([dist[v] for v in range(n)], dtype=np.float64)
        if self.config.structural_features:
            x = np.stack([z, deg / max(deg.max(), 1.0), d / max(d.max(), 1.0)], axis=1)
        else:
            x = z[:, None]
        if self.config.target_indicator:
            ind = np.zeros((n, 1))
            ind[target, 0] = 1.0
            x = np.concatenate([x, ind], axis=1)

        edges = np.array(list(G.edges()), dtype=np.int64).T
        edge_index = torch.from_numpy(np.concatenate([edges, edges[::-1]], axis=1))

        e0 = torch.tensor(eta0, dtype=torch.float64)
        lr = (
            self.shift.ratio(e0)
            if self.shift is not None
            else torch.ones((), dtype=torch.float64)
        )

        return Data(
            x=torch.from_numpy(x).to(torch.float32),
            edge_index=edge_index,
            y=torch.tensor([y], dtype=torch.float32),
            mu=torch.tensor(mu, dtype=torch.float64),
            eta=torch.tensor(eta, dtype=torch.float64),
            eta0=e0,
            p0=torch.tensor(p0, dtype=torch.float64),
            delta=torch.tensor(self.config.delta, dtype=torch.float64),
            regulator=torch.tensor(regulator),
            target_idx=torch.tensor(target),
            likelihood_ratio=lr,
            num_nodes=n,
            link=self.link.name,
            topology=self.config.topology.kind,
            split=split,
        )
