"""Integration tests for the spectral (effective-resistance) topology option.

Exercises the ``"spectral"`` ``graph_kwargs`` branch wired into the existing
topology pipeline (``ttk_functions``), plus the dependency-light persistence
barcode from ``spectral_persistence`` (adapted from arXiv:2311.03087v3).
"""

import os
import sys

import networkx as nx
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # .../server/calculate
sys.path.append(parent_dir)

from script_util import ttk_functions  # existing (non-new) call-site module
from script_util import spectral_persistence  # new capability module


def _grid_point_cloud(steps=5):
    """A small 2D loss-landscape point cloud with a scalar height field."""
    grid_x, grid_y = np.meshgrid(np.arange(steps), np.arange(steps))
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(float)
    rng = np.random.RandomState(0)
    values = rng.rand(coords.shape[0])
    return coords, values


def test_spectral_branch_runs_through_loss_landscape_pipeline(tmp_path):
    """The spectral graph option flows through the existing VTK pipeline."""
    coords, values = _grid_point_cloud(steps=5)

    output = ttk_functions.loss_landscape_to_vtu(
        loss_coords=coords,
        loss_values=values,
        output_path=str(tmp_path / "spectral"),
        graph_kwargs="spectral",
    )

    # the spectral branch produced the expected unstructured-grid artifact,
    # i.e. the new graph builder was actually invoked by the existing pipeline
    assert output.endswith("_spectral.vtu")
    assert os.path.exists(output)


def test_spectral_graph_connects_point_cloud():
    """Effective resistance is finite across a connected base graph."""
    coords, _ = _grid_point_cloud(steps=5)

    adjacency, graph = spectral_persistence.effective_resistance_graph(
        loss_coords=coords, verbose=0
    )

    assert graph.number_of_nodes() == coords.shape[0]
    assert adjacency.shape == (coords.shape[0], coords.shape[0])
    # the spectral neighbourhood graph spans the whole sampled landscape
    assert nx.number_connected_components(graph) == 1
    # every edge carries the spectral (effective-resistance) distance
    assert all("eff_resistance" in data for _, _, data in graph.edges(data=True))


def test_rips_barcode_count_for_connected_graph():
    """Connected spectral graph -> exactly n-1 finite H0 pairs (Rips filtration)."""
    coords, _ = _grid_point_cloud(steps=5)  # 25 connected points

    barcode = spectral_persistence.spectral_persistence_barcode(coords)

    assert len(barcode) == coords.shape[0] - 1  # n-1 finite merges for 1 component
    assert all(item["y0"] == 0.0 for item in barcode)  # all vertices born at 0
    assert all(item["y1"] >= item["y0"] for item in barcode)  # death >= birth


def test_lower_star_barcode_detects_local_minimum():
    """A second (local) loss minimum on a connected graph yields a finite pair."""
    coords, _ = _grid_point_cloud(steps=5)
    n_points = coords.shape[0]

    # global minimum at node 0 (corner), a separate local minimum at the
    # opposite corner (node 24); the two are not adjacent in the spectral
    # graph, so node 24 is a genuine second basin that later merges in
    values = np.full(n_points, 5.0)
    values[0] = 0.0
    values[n_points - 1] = 1.0

    barcode = spectral_persistence.spectral_persistence_barcode(
        coords, loss_values=values
    )

    assert len(barcode) >= 1  # the local basin merges into the global one
    assert all(set(item.keys()) == {"x", "y0", "y1"} for item in barcode)
    assert all(item["y1"] >= item["y0"] for item in barcode)  # death >= birth
