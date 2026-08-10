"""Integration test for the Gap Index wiring in the TTK pipeline.

Exercises ``loss_landscape_to_vtu`` (the existing projection call site in
``script_util.ttk_functions``) with ``gap_index=True`` and asserts that the
2D projection is scored for empty-region distortion. Also checks the default
``gap_index=False`` path is unchanged (still returns just the output file).
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from script_util.ttk_functions import loss_landscape_to_vtu


def _make_coords(n=40, d=3, seed=0):
    rng = np.random.default_rng(seed)
    # Three loose clusters in d-dim space so the 2D projection has real
    # empty-region structure between clusters.
    centers = rng.normal(size=(3, d)) * 4.0
    pts = np.vstack(
        [centers[i] + rng.normal(size=(n // 3, d)) for i in range(3)]
    )
    return pts


def test_loss_landscape_to_vtu_returns_gap_index(tmp_path):
    high_coords = _make_coords()
    out = loss_landscape_to_vtu(
        loss_coords=high_coords,
        loss_values=np.zeros(high_coords.shape[0]),
        output_path=str(tmp_path / "landscape"),
        graph_kwargs="delaunay",
        gap_index=True,
    )

    # Opt-in path returns (output_file, gi_result) and wrote the .vtu.
    assert isinstance(out, tuple) and len(out) == 2
    output_file, gi_result = out
    assert output_file.endswith(".vtu")
    assert os.path.exists(output_file)

    # The Gap Index itself is a finite, non-negative scalar over real triangles.
    assert isinstance(gi_result["gap_index"], float)
    assert np.isfinite(gi_result["gap_index"])
    assert gi_result["gap_index"] >= 0.0
    assert gi_result["n_triangles"] > 0
    assert gi_result["per_triangle"].shape[0] == gi_result["n_triangles"]


def test_default_path_unchanged_without_flag(tmp_path):
    """gap_index defaults to False -> returns just the output file (str)."""
    high_coords = _make_coords(n=30)
    out = loss_landscape_to_vtu(
        loss_coords=high_coords,
        loss_values=np.zeros(high_coords.shape[0]),
        output_path=str(tmp_path / "landscape_default"),
        graph_kwargs="delaunay",
    )
    assert isinstance(out, str)
    assert out.endswith(".vtu")


def test_gap_index_flags_distortion():
    """A faithful projection scores lower than a deliberately distorted one."""
    from script_util.gap_index import compute_gap_index
    from sklearn.decomposition import PCA

    high_coords = _make_coords(n=60, d=5)
    proj_pca = PCA(n_components=2).fit_transform(high_coords)
    # Shrink one axis of the projection to introduce empty-region deformation.
    proj_squashed = proj_pca * np.array([0.05, 1.0])

    gi_pca = compute_gap_index(high_coords, proj_pca)["gap_index"]
    gi_squashed = compute_gap_index(high_coords, proj_squashed)["gap_index"]

    assert gi_pca < gi_squashed
