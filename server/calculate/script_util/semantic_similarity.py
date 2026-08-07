"""Decoupled semantic vs. spatial representational similarity.

Adapted from "Decoupling Semantic Similarity from Spatial Alignment for Neural
Networks" (https://arxiv.org/abs/2410.23107). The paper observes that a standard
representational similarity matrix (RSM) -- and the CKA score computed from it --
conflates two distinct things:

  * **semantic similarity**: do two representations encode the same relational
    structure between inputs (i.e. are the same inputs clustered together),
    *regardless* of how the underlying units are arranged?
  * **spatial alignment**: do corresponding spatial locations / feature axes
    line up between the two representations?

The paper formalises this by treating each input's representation as a set of
``S`` concept vectors ``v_s in R^C`` (channels ``C`` across spatial locations
``S``). Two RSM kernels are then defined for an input pair ``(i, j)``:

  * **spatio-semantic** (Eq. 3, spatial-alignment *sensitive*): sum the inner
    products at *matching* spatial indices,
    ``K_ss[i, j] = sum_s <v_{i,s}, v_{j,s}>``. This is exactly the linear kernel
    the repo's existing ``CKA.linear_CKA`` already compares, so the repo's CKA
    score *is* the spatio-semantic RSM compared via CKA.
  * **semantic** (Eqs. 4-5, permutation *invariant*): solve a bipartite matching
    (Hungarian) over the ``S`` spatial locations to *maximise* the inner product,
    ``K_sem[i, j] = max_P sum_s <v_{i,s}, v_{j,P(s)}> >= K_ss[i, j]``. This is
    invariant to any permutation of the spatial axis.

Each representation therefore yields two ``N x N`` RSMs (spatio-semantic and
semantic); comparing the two models' RSMs with the same biased centering-HSIC
CKA used elsewhere in this repo gives two decoupled scores. Their gap measures
how much of the apparent similarity is owed to coincidental spatial alignment
versus genuine relational (semantic) structure.

This module is intentionally dependency-light (numpy + scipy only) so it can be
exercised without the repo's full torch / model-loading stack.
"""

from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# Cap on the number of spatial locations considered per input. Bounds the
# O(S^3) Hungarian solve and the RSM memory; the paper's spatial axis is the
# within-input location dimension, so a modest S captures the decoupling signal.
_MAX_SPATIAL = 16
# Cap on the number of inputs (rows) used to build an RSM. RSMs are O(N^2); for
# the weight-matrix comparisons this module is fed from compute_cka_similarity
# we subsample rows deterministically rather than materialising a huge matrix.
_MAX_SAMPLES = 128


def _centering(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix: ``H K H`` with ``H = I - 11^T / n``.

    Mirrors ``CKA.centering`` in ``core_functions`` so that RSMs are compared
    with the same biased CKA the repo already uses.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Biased HSIC between two kernel matrices (matches ``CKA.linear_HSIC``)."""
    return float(np.sum(_centering(K) * _centering(L)))


def cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    """CKA between two (already computed) ``N x N`` kernel/RSM matrices.

    Identical convention to ``CKA.linear_CKA`` applied to the sample Gram
    matrices, so ``cka_from_grams(X @ X.T, Y @ Y.T)`` reproduces the repo's
    existing spatio-semantic CKA score for representations ``X, Y``.
    """
    denom = np.sqrt(_hsic(K, K) * _hsic(L, L))
    if denom == 0.0:
        return 0.0
    return _hsic(K, L) / denom


def pick_spatial_dim(num_features: int, cap: int = _MAX_SPATIAL) -> int:
    """Choose a spatial-axis size ``S`` that divides ``num_features``.

    Picks the divisor of ``num_features`` (with ``2 <= S <= cap``) closest to
    ``sqrt(num_features)``, so the factored concept vectors stay reasonably
    balanced. Falls back to ``1`` (no spatial axis -> semantic == spatio
    semantic, gap zero) when no non-trivial divisor exists.
    """
    target = int(np.sqrt(num_features))
    best = 1
    for s in range(2, cap + 1):
        if num_features % s == 0:
            if abs(s - target) < abs(best - target) or best == 1:
                best = s
    return best


def _to_concept_sets(X: np.ndarray, spatial_dim: int) -> np.ndarray:
    """Reshape each row of ``X`` (length ``P = C * S``) into ``S`` concept vectors.

    Returns an ``(N, S, C)`` array: for input ``i``, location ``s``, channel
    ``c`` the value is ``X[i, s * C + c]``.
    """
    n, p = X.shape
    c = p // spatial_dim
    usable = spatial_dim * c
    if usable != p:
        X = X[:, :usable]
    return X.reshape(n, spatial_dim, c)


def _spatio_semantic_rsm(X: np.ndarray) -> np.ndarray:
    """Spatio-semantic RSM: the plain sample Gram matrix ``X @ X.T``.

    Summing inner products at matching spatial indices over the factored view is
    algebraically identical to the full dot product (Eq. 3 == linear kernel), so
    the spatio-semantic RSM is simply ``X @ X.T``.
    """
    return X @ X.T


def _semantic_rsm(V: np.ndarray) -> np.ndarray:
    """Permutation-invariant (semantic) RSM from concept sets ``V`` (``N, S, C``).

    Entry ``[i, j]`` is the maximum, over permutations of the ``S`` spatial
    locations, of ``sum_s <v_{i,s}, v_{j,P(s)}>`` -- found per pair by Hungarian
    matching on the ``S x S`` affinity matrix ``V[i] @ V[j].T`` (paper Eqs. 4-5).
    Guaranteed ``>=`` the spatio-semantic entry for the same pair.
    """
    n = V.shape[0]
    rsm = np.zeros((n, n))
    for i in range(n):
        vi = V[i]
        # Diagonal: matching a set with itself -> sum of squared norms (max over
        # permutations of identical vectors is the trivial identity match).
        rsm[i, i] = float(np.sum(vi * vi))
        for j in range(i + 1, n):
            affinity = vi @ V[j].T  # (S, S)
            row_ind, col_ind = linear_sum_assignment(-affinity)
            value = float(affinity[row_ind, col_ind].sum())
            rsm[i, j] = value
            rsm[j, i] = value  # symmetric: matching on the transpose is identical
    return rsm


def decoupled_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    spatial_dim: Optional[int] = None,
    max_samples: int = _MAX_SAMPLES,
) -> Dict[str, float]:
    """Decoupled semantic / spatial-alignment similarity for two representations.

    Parameters
    ----------
    X, Y : array-like, shape ``(N, P)``
        Two representations over the same ``N`` inputs (``P`` features each).
        Rows are paired by index, matching how ``compute_cka_similarity`` feeds
        two flattened weight tensors into ``CKA.linear_CKA``.
    spatial_dim : int, optional
        Number of spatial locations ``S`` used to factor each ``P``-length row
        into ``S`` concept vectors (``P = S * C``). Auto-chosen by
        :func:`pick_spatial_dim` when omitted.
    max_samples : int
        Cap on ``N`` for building the ``N x N`` RSMs; rows are subsampled
        deterministically (even stride) when exceeded.

    Returns
    -------
    dict with keys:
        ``semantic``            -- CKA of the permutation-invariant RSMs.
        ``spatio_semantic``     -- CKA of the spatial-alignment-sensitive RSMs
                                   (reproduces the repo's existing linear CKA).
        ``spatial_alignment``   -- ``spatio_semantic - semantic``; a large
                                   magnitude means apparent similarity hinges on
                                   spatial alignment rather than relational
                                   structure (negative when matching recovers
                                   similarity that positional comparison misses).
        ``spatial_dim``         -- the ``S`` used.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must share shape, got {X.shape} and {Y.shape}"
        )
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {X.shape}")

    n, p = X.shape
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples).round().astype(int)
        X = X[idx]
        Y = Y[idx]
        n = X.shape[0]

    s = pick_spatial_dim(p) if spatial_dim is None else spatial_dim
    if s < 1:
        raise ValueError(f"spatial_dim must be >= 1, got {s}")
    c = p // s
    if c < 1:
        s = 1
        c = p

    vx = _to_concept_sets(X, s)
    vy = _to_concept_sets(Y, s)

    kx_ss = _spatio_semantic_rsm(X)
    ky_ss = _spatio_semantic_rsm(Y)
    spatio_semantic = cka_from_grams(kx_ss, ky_ss)

    if s == 1:
        # No spatial axis to permute: the two kernels coincide.
        semantic = spatio_semantic
    else:
        kx_sem = _semantic_rsm(vx)
        ky_sem = _semantic_rsm(vy)
        semantic = cka_from_grams(kx_sem, ky_sem)

    return {
        "semantic": float(semantic),
        "spatio_semantic": float(spatio_semantic),
        "spatial_alignment": float(spatio_semantic - semantic),
        "spatial_dim": int(s),
    }
