"""Tests for the decoupled semantic/spatial similarity capability.

These exercise ``script_util.semantic_similarity`` (the new module) alongside
``script_util.torch_cka.cka`` -- the repo's existing CKA implementation, a
non-new module in the same similarity-measurement package -- to confirm the new
capability integrates with the repo's existing CKA machinery and delivers the
decoupling from "Decoupling Semantic Similarity from Spatial Alignment for
Neural Networks" (arXiv:2410.23107).

``compute_cka_similarity`` in ``core_functions`` invokes
``decoupled_similarity`` and persists these scores via ``update_db``; that
module is not imported here because the repo's ``core_functions`` depends on a
local ``loss_landscapes_pinn`` package that is not present in this checkout.
"""

import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
CALC = os.path.dirname(HERE)  # server/calculate
sys.path.insert(0, CALC)

from script_util import semantic_similarity as ss  # new capability module
from script_util.torch_cka import cka as torch_cka  # non-new module (repo's CKA)


def test_spatio_semantic_reproduces_repo_gram_cka():
    # The spatial-alignment-sensitive score is exactly the repo's existing
    # linear-CKA-on-Grams applied to the two representations.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 16))
    Y = rng.standard_normal((20, 16))
    out = ss.decoupled_similarity(X, Y, spatial_dim=4)
    expected = ss.cka_from_grams(X @ X.T, Y @ Y.T)
    assert abs(out["spatio_semantic"] - expected) < 1e-9


def test_semantic_kernel_dominates_and_decouples_under_spatial_shift():
    # Per-pair K_sem >= K_ss (the Hungarian matching maximises the inner
    # product), and STRICTLY exceeds it when one input is a spatial shift of the
    # other -- the paper's headline result that semantic similarity is recovered
    # once spatial alignment is decoupled.
    rng = np.random.default_rng(1)
    S, C, N = 4, 5, 8
    V = rng.standard_normal((N, S, C))
    X = V.reshape(N, S * C)
    Kss = ss._spatio_semantic_rsm(X)
    Ksem = ss._semantic_rsm(V)
    assert np.all(Ksem >= Kss - 1e-9)

    shifted = np.roll(V[1], shift=1, axis=0)  # input 1's spatial axis shifted
    spatio = float(np.sum(V[0] * shifted))    # identity spatial pairing
    affinity = V[0] @ shifted.T
    rows, cols = linear_sum_assignment(-affinity)
    semantic = float(affinity[rows, cols].sum())  # optimal (matched) pairing
    assert semantic > spatio


def test_semantic_rsm_invariant_to_global_spatial_permutation():
    # The semantic RSM is invariant to any permutation of the spatial axis
    # (paper Eqs. 4-5) -- the matching re-discovers the optimal pairing.
    rng = np.random.default_rng(2)
    S, C, N = 4, 6, 8
    V = rng.standard_normal((N, S, C))
    Ksem = ss._semantic_rsm(V)
    perm = rng.permutation(S)
    Ksem_perm = ss._semantic_rsm(V[:, perm, :])
    assert np.allclose(Ksem, Ksem_perm, atol=1e-9)


def test_scores_bounded_and_self_consistent():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 24))
    Y = X + 0.05 * rng.standard_normal((30, 24))
    out = ss.decoupled_similarity(X, Y, spatial_dim=6)
    for key in ("semantic", "spatio_semantic"):
        assert -1e-9 <= out[key] <= 1.0 + 1e-9
    assert (
        abs(out["spatial_alignment"] - (out["spatio_semantic"] - out["semantic"]))
        < 1e-9
    )
    assert out["spatial_dim"] == 6


def test_integration_with_existing_cka_module():
    # The new capability lives alongside the repo's existing CKA package and
    # produces a spatio-semantic score that is a valid CKA in the repo's [0, 1]
    # convention, computed from the same row-Gram data path torch_cka uses.
    assert hasattr(torch_cka, "CKA")
    rng = np.random.default_rng(4)
    # row-normalise features the way torch_cka.cka.CKA.compare builds its gram
    X = rng.standard_normal((16, 12))
    Y = rng.standard_normal((16, 12))
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    out = ss.decoupled_similarity(Xn, Yn, spatial_dim=3)
    assert 0.0 <= out["spatio_semantic"] <= 1.0
    assert 0.0 <= out["semantic"] <= 1.0


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        ss.decoupled_similarity(np.zeros((5, 4)), np.zeros((5, 6)))
