"""Synthetic substrate: theorems are falsifiable here because ground truth is
known by construction, not annotated."""

from certgnn.substrates.synthetic.links import LINK_NAMES, Link, get_link
from certgnn.substrates.synthetic.oracle import Oracle
from certgnn.substrates.synthetic.shift import GaussianShift
from certgnn.substrates.synthetic.substrate import (
    SPLITS,
    SyntheticConfig,
    SyntheticSubstrate,
)
from certgnn.substrates.synthetic.topology import TOPOLOGY_KINDS, Motif, TopologySpec

__all__ = [
    "LINK_NAMES",
    "Link",
    "get_link",
    "Oracle",
    "GaussianShift",
    "SPLITS",
    "SyntheticConfig",
    "SyntheticSubstrate",
    "TOPOLOGY_KINDS",
    "Motif",
    "TopologySpec",
]
