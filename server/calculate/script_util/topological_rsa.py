"""Topological Representational Similarity Analysis (tRSA).

Augments the geometric layer-similarity statistic (CKA) with a *topological*
view of each layer's representation. For every layer we build the
Representational Dissimilarity Matrix (RDM) -- the pairwise distances between
stimulus responses -- replace it by its single-linkage (sub-dominant)
ultrametric, and then compare layers by RSA-style correlation of the upper
triangles of those topologically-transformed RDMs.

The single-linkage ultrametric *is* the 0-th persistent homology (merge tree)
of the dissimilarity space: two stimuli merge at the smallest threshold at
which a chain of sub-threshold edges connects them. This is the same
topological primitive the rest of this repo extracts from loss landscapes via
the Topology ToolKit (TTK) merge trees, now applied to representational
geometry.

Adapted from "Topological Representational Similarity Analysis in Brains and
Beyond", arXiv:2408.11948 (Mode 2 -- adapted port):

  * Core mechanism kept at full fidelity -- a topological transform of the RDM
    followed by second-order RSA comparison of the transformed matrices.
  * The paper's persistent-homology estimator over RDMs (which needs a TDA
    library such as gudhi/ripser, not in this repo's dependencies) is replaced
    by a parameter-free scipy single-linkage cophenetic ultrametric, i.e. the
    0-D / merge-tree component of the transform. Higher-dimensional persistence
    (loops, voids) is intentionally out of scope: it would require TDA tooling
    the repo does not depend on.
  * The paper's neuroscience benchmark / evaluation suite is cut -- evaluation
    belongs in a downstream PR. This analyzer reuses the repo's existing
    forward-hook feature-extraction pattern (mirroring ``torch_cka.CKA``).
"""

from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader


class TopologicalRSA:
    """Parallel ``.compare()/.export()`` analyzer that produces a layer x layer
    topological similarity matrix, mirroring the geometric CKA matrix produced
    by ``torch_cka.CKA``.

    Parameters mirror ``torch_cka.CKA`` so the two analyzers can be driven from
    the same call site on the same pair of models and the same dataloader.
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        device: str = "cpu",
        max_stimuli: int = 128,
        model1_layers: Optional[List[str]] = None,
        model2_layers: Optional[List[str]] = None,
    ) -> None:
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.max_stimuli = max_stimuli
        self.model1_layers = model1_layers
        self.model2_layers = model2_layers

        self.model1_features: Dict[str, torch.Tensor] = {}
        self.model2_features: Dict[str, torch.Tensor] = {}

        self._hooks: List = []
        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)
        self.model1.eval()
        self.model2.eval()

    # -- forward-hook feature extraction (same pattern as torch_cka.CKA) -------
    def _log_layer(
        self,
        which: str,
        name: str,
        module: nn.Module,
        inp: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        feat = out
        if not torch.is_tensor(feat):
            feat = feat[0] if isinstance(feat, (tuple, list)) else None
        if feat is None or not torch.is_tensor(feat):
            return
        store = self.model1_features if which == "model1" else self.model2_features
        store[name] = feat.detach()

    def _insert_hooks(self) -> None:
        for name, layer in self.model1.named_modules():
            if self.model1_layers is None or name in self.model1_layers:
                handle = layer.register_forward_hook(
                    partial(self._log_layer, "model1", name)
                )
                self._hooks.append(handle)
        for name, layer in self.model2.named_modules():
            if self.model2_layers is None or name in self.model2_layers:
                handle = layer.register_forward_hook(
                    partial(self._log_layer, "model2", name)
                )
                self._hooks.append(handle)

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    # -- topological transform ------------------------------------------------
    @staticmethod
    def _topological_rdm(features: np.ndarray) -> np.ndarray:
        """RDM -> single-linkage cophenetic ultrametric (0-D merge tree).

        Returns the condensed (upper-triangular) ultrametric distance vector,
        aligned element-wise with ``scipy.spatial.distance.pdist`` ordering so
        two layers can be compared directly.
        """
        rdm = pdist(features, metric="euclidean")
        if rdm.size < 2 or np.allclose(rdm, 0.0):
            # Degenerate representation (e.g. constant layer): no topology to
            # compare; caller falls back to a zero / cosine comparison.
            return np.zeros_like(rdm)
        tree = linkage(rdm, method="single")
        return cophenet(tree)

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Second-order RSA statistic: Pearson correlation between two
        topologically-transformed (condensed) RDMs. Falls back to cosine
        similarity when one side is degenerate (zero variance)."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return 0.0
        sa, sb = a.std(), b.std()
        if sa > 0 and sb > 0:
            am = a - a.mean()
            bm = b - b.mean()
            return float(np.dot(am, bm) / (a.size * sa * sb))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # -- public API -----------------------------------------------------------
    def compare(self, dataloader: DataLoader) -> None:
        """Run both models over ``dataloader`` and aggregate up to
        ``max_stimuli`` stimulus responses per layer, then build the
        topological similarity matrix."""
        pooled1: Dict[str, List[np.ndarray]] = {}
        pooled2: Dict[str, List[np.ndarray]] = {}
        collected = 0

        for item in dataloader:
            (x, *_) = item
            x = x.to(self.device)
            batch = x.shape[0]

            self.model1_features = {}
            self.model2_features = {}
            with torch.no_grad():
                _ = self.model1(x)
                _ = self.model2(x)

            for name, feat in self.model1_features.items():
                row = feat.reshape(batch, -1).cpu().numpy()
                pooled1.setdefault(name, []).append(row)
            for name, feat in self.model2_features.items():
                row = feat.reshape(batch, -1).cpu().numpy()
                pooled2.setdefault(name, []).append(row)

            collected += batch
            if collected >= self.max_stimuli:
                break

        self.layer_names1 = list(pooled1.keys())
        self.layer_names2 = list(pooled2.keys())
        self.features1 = {
            name: np.concatenate(pooled1[name], axis=0)[: self.max_stimuli]
            for name in self.layer_names1
        }
        self.features2 = {
            name: np.concatenate(pooled2[name], axis=0)[: self.max_stimuli]
            for name in self.layer_names2
        }

        ultra1 = [self._topological_rdm(self.features1[n]) for n in self.layer_names1]
        ultra2 = [self._topological_rdm(self.features2[n]) for n in self.layer_names2]

        matrix = np.zeros((len(self.layer_names1), len(self.layer_names2)))
        for i, u1 in enumerate(ultra1):
            for j, u2 in enumerate(ultra2):
                matrix[i, j] = self._similarity(u1, u2)
        self.trsa_matrix = matrix

    def export(self) -> Dict:
        """Export the topological similarity matrix and layer names."""
        return {
            "tRSA": self.trsa_matrix,
            "model1_layers": list(self.layer_names1),
            "model2_layers": list(self.layer_names2),
        }
