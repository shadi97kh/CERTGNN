"""Splice substrate over the MFASS exon-recognition assay.

Data: Cheung et al., *Molecular Cell* 73:183-194 (2019), the MFASS single
nucleotide variant library. 32,669 variants across 2,339 distinct exons,
each assayed in a 170 nt minigene fragment laid out as
``[intron1][exon][intron2]``, with a splicing index in [0, 1]. Provenance,
sizes and checksums are in ``data/raw/PROVENANCE.md``.

Graph, per Definition 1. The fragment is tiled into fixed-width windows.
Exon windows and intronic regulatory windows are nodes; edges are the
backbone (consecutive windows in genomic order) plus the *skipping*
junction that connects the last upstream intronic window directly to the
first downstream intronic window. The skipping junction is what makes this
a graph rather than a path: its endpoints move with the intron and exon
lengths, so topology varies across exons even though the window grammar
does not.

**Deviation from Definition 1, recorded rather than hidden.** Definition 1
tiles intronic windows to a radius of 300 nt. MFASS assays a 170 nt
fragment whose flanking introns have a median length of 45 nt, so a 300 nt
radius exceeds the sequence that was actually measured. Windows are
therefore tiled only within the assayed construct. The alternative --
drawing windows from hg38 beyond the minigene -- would build a graph over
sequence the assay never interrogated, so the measurement could not speak
to those nodes. ``radius_nt`` records the requested radius and
``effective_radius_nt`` what the construct actually supports.

Splits are grouped by exon: no ``ensembl_id`` appears in more than one
split. Variants of one exon share almost all of their sequence, so a
random split would leak.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

BASES = "ACGT"
SPLITS = ("train", "val", "cal", "test")
DEFAULT_PATH = "data/raw/mfass_snv_data_clean.txt"


@dataclass(frozen=True)
class SpliceConfig:
    """Dials for the splice substrate.

    Parameters
    ----------
    path : str
        MFASS ``snv_data_clean.txt``.
    window : int
        Tiling width in nucleotides. Each window is one node and carries its
        own sequence one-hot, so a single-nucleotide variant remains visible.
    radius_nt : int
        Definition 1's intronic radius. Clipped to the assayed construct;
        see the module docstring.
    target_col : str
        Measured phenotype. ``v2_index`` is the library-2 splicing index.
    baseline_col : str
        Reference (natural) exon's index, used by ``baseline_rate``.
    min_reads : int
        Drop rows whose read support is below this in either replicate.
    split_fracs : tuple[float, float, float, float]
        train / val / cal / test, assigned by hashing ``ensembl_id`` so the
        grouping is deterministic and no exon straddles splits.
    """

    path: str = DEFAULT_PATH
    window: int = 10
    radius_nt: int = 300
    target_col: str = "v2_index"
    baseline_col: str = "nat_v2_index"
    min_reads: int = 10
    split_fracs: tuple[float, float, float, float] = (0.6, 0.1, 0.1, 0.2)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if abs(sum(self.split_fracs) - 1.0) > 1e-9:
            raise ValueError("split_fracs must sum to 1")


def _one_hot(seq: str, width: int) -> np.ndarray:
    """One-hot a window, padded to ``width``; unknown bases are all-zero."""
    out = np.zeros((width, 4), dtype=np.float32)
    for i, ch in enumerate(seq[:width]):
        j = BASES.find(ch.upper())
        if j >= 0:
            out[i, j] = 1.0
    return out.reshape(-1)


def _split_of(exon_id: str, fracs: tuple[float, float, float, float], seed: int) -> str:
    """Deterministic grouped assignment: all variants of an exon land together."""
    h = hashlib.sha256(f"{seed}:{exon_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    acc = 0.0
    for name, f in zip(SPLITS, fracs):
        acc += f
        if u < acc:
            return name
    return SPLITS[-1]


class SpliceSubstrate:
    """MFASS exon-recognition assay as a graph substrate.

    Implements the ``Substrate`` protocol in ``certgnn/substrates/base.py``
    without modifying it.
    """

    name = "splice"

    def __init__(self, config: SpliceConfig | None = None) -> None:
        self.config = config or SpliceConfig()
        p = pathlib.Path(self.config.path)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/fetch_data.sh; see data/raw/PROVENANCE.md."
            )
        df = pd.read_csv(p, sep="\t", low_memory=False)
        c = self.config
        need = [
            "ensembl_id",
            "sequence",
            "intron1_len",
            "exon_len",
            "intron2_len",
            "rel_position",
            c.target_col,
            c.baseline_col,
        ]
        missing = [k for k in need if k not in df.columns]
        if missing:
            raise ValueError(f"MFASS file is missing columns {missing}")
        df = df.dropna(subset=[c.target_col, "sequence", "exon_len"])
        for col in ("v2_R1_sum", "v2_R2_sum"):
            if col in df.columns:
                df = df[
                    pd.to_numeric(df[col], errors="coerce").fillna(0) >= c.min_reads
                ]
        df = df.reset_index(drop=True)
        df["_split"] = [_split_of(str(e), c.split_fracs, c.seed) for e in df.ensembl_id]
        self._df = df
        self.effective_radius_nt = int(
            min(c.radius_nt, max(df.intron1_len.max(), df.intron2_len.max()))
        )

    # ------------------------------------------------------------ protocol

    def load(self, split: str) -> list[Data]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        sub = self._df[self._df._split == split]
        return [self._build(r) for _, r in sub.iterrows()]

    def target_node(self, data: Data) -> int:
        return int(data.target_idx)

    def candidate_nodes(self, data: Data) -> torch.Tensor:
        t = int(data.target_idx)
        return torch.tensor(
            [v for v in range(int(data.num_nodes)) if v != t], dtype=torch.long
        )

    def ground_truth_window(self, data: Data) -> torch.Tensor | None:
        """The window containing the assayed variant, when its position is known."""
        w = int(data.variant_window)
        return None if w < 0 else torch.tensor([w], dtype=torch.long)

    def to_networkx(self, data: Data) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(int(data.num_nodes)))
        ei = data.edge_index
        g.add_edges_from(
            (int(u), int(v)) for u, v in zip(ei[0].tolist(), ei[1].tolist()) if u < v
        )
        return g

    def baseline_rate(self, data: Data) -> float:
        """Splicing index of the natural (unmutated) exon: a rate in [0, 1]."""
        return float(data.baseline)

    # ----------------------------------------------------------- internals

    def _build(self, r: pd.Series) -> Data:
        c = self.config
        seq = str(r.sequence).upper()
        i1, ex = int(r.intron1_len), int(r.exon_len)
        w = c.window
        # window boundaries, clipped to the assayed construct (see docstring)
        bounds = list(range(0, len(seq), w))
        kinds, feats = [], []
        for b in bounds:
            piece = seq[b : b + w]
            mid = b + len(piece) / 2.0
            if mid < i1:
                k = 0  # upstream intron
            elif mid < i1 + ex:
                k = 1  # exon
            else:
                k = 2  # downstream intron
            kinds.append(k)
            onehot = _one_hot(piece, w)
            flags = np.zeros(3, dtype=np.float32)
            flags[k] = 1.0
            feats.append(
                np.concatenate([onehot, flags, [b / max(len(seq), 1)]]).astype(
                    np.float32
                )
            )
        n = len(bounds)
        x = torch.from_numpy(np.stack(feats))

        exon_idx = [i for i, k in enumerate(kinds) if k == 1]
        up_idx = [i for i, k in enumerate(kinds) if k == 0]
        dn_idx = [i for i, k in enumerate(kinds) if k == 2]
        # backbone: consecutive windows
        edges = [(i, i + 1) for i in range(n - 1)]
        # skipping junction: last upstream intronic window to first downstream one.
        # Its endpoints move with the intron and exon lengths, so the topology
        # varies across exons; without it the graph is a bare path.
        if up_idx and dn_idx:
            edges.append((up_idx[-1], dn_idx[0]))
        und = (
            torch.tensor(edges, dtype=torch.long).t()
            if edges
            else torch.empty(2, 0, dtype=torch.long)
        )
        edge_index = torch.cat([und, und.flip(0)], dim=1) if edges else und

        target_idx = exon_idx[len(exon_idx) // 2] if exon_idx else 0
        rel = r.get("rel_position", np.nan)
        vw = -1
        if pd.notna(rel):
            pos = int(i1 + float(rel)) if float(rel) >= 0 else int(float(rel))
            if 0 <= pos < len(seq):
                vw = min(pos // w, n - 1)

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([float(r[c.target_col])], dtype=torch.float32),
            baseline=torch.tensor(
                float(r[c.baseline_col])
                if pd.notna(r[c.baseline_col])
                else float("nan")
            ),
            target_idx=torch.tensor(target_idx),
            variant_window=torch.tensor(vw),
            num_nodes=n,
            exon_id=str(r.ensembl_id),
            exon_len=int(ex),
            intron1_len=int(i1),
            intron2_len=int(r.intron2_len),
        )
