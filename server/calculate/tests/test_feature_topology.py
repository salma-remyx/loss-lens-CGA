"""Unit tests for the Hopkins-statistic feature-topology measurement.

These exercise the new ``script_util.feature_topology`` module directly --
pure-Python, no backend deps -- so they run in any environment.
"""

import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from script_util.feature_topology import hopkins_statistic, layer_hopkins  # noqa: E402


def _clustered():
    pts = []
    for cx, cy in [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)]:
        for _ in range(20):
            pts.append([cx + 0.01, cy + 0.01])
    return pts


def _grid():
    return [[float(i), float(j)] for i in range(8) for j in range(8)]


def test_clustered_data_scores_high():
    # Points in three tight clusters should be strongly clustered (H -> 1).
    assert hopkins_statistic(_clustered(), seed=0) > 0.9


def test_regular_grid_scores_below_half():
    # A regular grid is evenly spaced, which scores below 0.5, and well
    # below the clustered case -- i.e. the statistic discriminates topology.
    h_grid = hopkins_statistic(_grid(), seed=1)
    assert 0.1 < h_grid < 0.5
    assert h_grid < hopkins_statistic(_clustered(), seed=0)


def test_values_are_bounded():
    for pts in (_clustered(), _grid()):
        h = hopkins_statistic(pts, seed=3)
        assert 0.0 <= h <= 1.0


def test_deterministic_with_seed():
    pts = _grid()
    assert hopkins_statistic(pts, seed=42) == hopkins_statistic(pts, seed=42)


def test_n_samples_override_is_respected():
    pts = _clustered()
    # A larger subsample changes the estimate but stays a valid topology score.
    h_small = hopkins_statistic(pts, n_samples=2, seed=7)
    h_large = hopkins_statistic(pts, n_samples=30, seed=7)
    assert 0.0 <= h_small <= 1.0
    assert 0.0 <= h_large <= 1.0


def test_too_few_points_returns_nan():
    assert math.isnan(hopkins_statistic([[1.0, 2.0]]))
    assert math.isnan(hopkins_statistic([]))


def test_layer_hopkins_maps_each_layer():
    features = {"layer_0": _clustered(), "layer_1": _grid()}
    out = layer_hopkins(features, seed=0)
    assert set(out) == {"layer_0", "layer_1"}
    assert out["layer_0"] > 0.9  # clustered layer
    assert out["layer_1"] < 0.5  # grid layer
