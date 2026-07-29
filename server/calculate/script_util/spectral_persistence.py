"""
Spectral (effective-resistance) persistent homology for loss landscapes.

Adapted port (Mode 2) of:

    Persistent Homology for High-dimensional Data Based on Spectral Methods
    (arXiv:2311.03087v3)

The paper's core observation is that for point clouds whose intrinsic
dimension is much smaller than their ambient dimension (exactly the regime
of sampled loss-landscape coordinates), the Euclidean distances used to
build neighbourhood graphs for persistent homology concentrate and become
dominated by noise, so the resulting barcodes miss the true topology. The
remedy is to measure neighbourhoods with a *spectral* distance derived from
the graph Laplacian instead.

This module implements that core mechanism -- the **effective resistance**

    R_eff(i, j) = (e_i - e_j)^T L^+ (e_i - e_j)

with ``L^+`` the Moore-Penrose pseudo-inverse of the (combinatorial) graph
Laplacian -- as the neighbourhood metric for the loss-landscape point cloud
and exposes it as a ``"spectral"`` graph-construction option for the existing
topology pipeline (``loss_landscape_to_vtu`` / ``compute_persistence_barcode``
in ``ttk_functions``).

Target-native substitutions (Mode 2), called out for honesty:

  * The paper's persistent homology over the spectral metric is delivered
    two ways:
      (1) wired into the repo's own TTK pipeline through the new
          ``"spectral"`` ``graph_kwargs`` branch in ``ttk_functions`` (full
          fidelity to the repo's topology pillar -- the spectral graph is
          the domain TTK computes persistence over), and
      (2) a parameter-free 0-dimensional (connected-component) barcode
          computed here with a union-find filtration, for dependency-light /
          offline analysis.
  * Substitute (2) replaces the external ParaView/TTK ``pvpython`` engine (a
    non-Python binary, unavailable outside the repo's pvpython environment)
    with a dependency-light proxy. It preserves the paper's spectral-distance
    *metric* while removing the bespoke PH engine; higher-dimensional
    barcodes and the full benchmark suite are intentionally out of scope.

The spectral metric itself -- the paper's actual contribution -- is kept at
full fidelity; only the PH *engine* is target-native.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "effective_resistance_distances",
    "effective_resistance_graph",
    "spectral_persistence_barcode",
]


def _knn_adjacency(
    loss_coords: np.ndarray, n_neighbors: int, metric: str
) -> np.ndarray:
    """Symmetric (unweighted) kNN adjacency matrix over the point cloud."""

    n_points = loss_coords.shape[0]
    k = max(1, min(n_neighbors, n_points - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(loss_coords)
    _, indices = nbrs.kneighbors(loss_coords)

    adjacency = np.zeros((n_points, n_points), dtype=float)
    for node in range(n_points):
        for neighbor in indices[node, 1:]:
            adjacency[node, neighbor] = 1.0
    return np.maximum(adjacency, adjacency.T)


def effective_resistance_distances(
    loss_coords=None,
    n_neighbors: Optional[int] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Spectral (effective-resistance) distance matrix of the point cloud.

    Builds a Euclidean kNN graph, takes its combinatorial Laplacian ``L`` and
    returns the effective-resistance distance between every pair of points

        R_eff[i, j] = L^+[i, i] + L^+[j, j] - 2 * L^+[i, j]

    which is the spectral distance the paper prescribes in place of the
    noise-dominated Euclidean distance. ``n_neighbors`` defaults to
    ``4 * intrinsic_dim`` to match the repo's existing kNN convention.
    """

    loss_coords = np.asarray(loss_coords, dtype=float)
    if loss_coords.ndim == 1:
        loss_coords = loss_coords.reshape(-1, 1)
    n_points = loss_coords.shape[0]
    if n_points < 2:
        return np.zeros((n_points, n_points))

    if n_neighbors is None:
        n_neighbors = 4 * loss_coords.shape[1]

    adjacency = _knn_adjacency(loss_coords, n_neighbors, metric)
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency

    # Moore-Penrose pseudo-inverse (L is symmetric positive semi-definite).
    laplacian_pinv = np.linalg.pinv(laplacian)
    diag = np.diag(laplacian_pinv)
    resistance = diag[:, None] + diag[None, :] - 2.0 * laplacian_pinv
    # guard tiny floating-point asymmetry / negatives from the pinv
    resistance = np.maximum(0.5 * (resistance + resistance.T), 0.0)
    np.fill_diagonal(resistance, 0.0)

    # Effective resistance is infinite across separate components of the base
    # graph; the pinv formula otherwise under-estimates those pairs and would
    # spuriously bridge disconnected neighbourhoods.
    base_graph = nx.from_numpy_array(adjacency)
    component = np.zeros(n_points, dtype=int)
    for cid, nodes in enumerate(nx.connected_components(base_graph)):
        component[list(nodes)] = cid
    same_component = component[:, None] == component[None, :]
    resistance = np.where(same_component, resistance, np.inf)
    np.fill_diagonal(resistance, 0.0)
    return resistance


def effective_resistance_graph(
    loss_coords=None,
    n_neighbors: Optional[int] = None,
    metric: str = "euclidean",
    return_graph: bool = True,
    verbose: int = 1,
):
    """kNN graph in effective-resistance (spectral) space.

    Mirrors the ``(adjacency, graph)`` contract of ``compute_aknn`` /
    ``compute_gabriel`` in ``ttk_functions`` so it slots directly into the
    existing persistence / merge-tree dispatch. The returned ``networkx``
    graph carries an ``"eff_resistance"`` weight on every edge.
    """

    loss_coords = np.asarray(loss_coords, dtype=float)
    if loss_coords.ndim == 1:
        loss_coords = loss_coords.reshape(-1, 1)
    n_points = loss_coords.shape[0]
    if n_neighbors is None:
        n_neighbors = 4 * loss_coords.shape[1]
    k = max(1, min(n_neighbors, n_points - 1))

    resistance = effective_resistance_distances(
        loss_coords, n_neighbors=n_neighbors, metric=metric
    )

    graph = nx.Graph()
    graph.add_nodes_from(range(n_points))
    for node in range(n_points):
        # k nearest neighbours by spectral distance (self has distance 0);
        # cross-component pairs are at +inf and never bridged
        order = np.argsort(resistance[node], kind="stable")
        for neighbor in order[1 : k + 1]:
            distance = resistance[node, neighbor]
            if not np.isfinite(distance):
                continue
            graph.add_edge(node, int(neighbor), eff_resistance=float(distance))

    adjacency = nx.adjacency_matrix(graph)

    if verbose > 0:
        print("\n... Computing spectral (effective-resistance) graph")
        print(f"    G.number_of_nodes() = {graph.number_of_nodes()}")
        print(f"    G.number_of_edges() = {graph.number_of_edges()}")
        print(f"    A.shape = {adjacency.shape}")

    if return_graph:
        return adjacency, graph
    return adjacency


def _lower_star_h0_barcode(graph: nx.Graph, heights: np.ndarray) -> List[Tuple]:
    """0-dim sublevel-set persistence of ``heights`` over the spectral graph.

    Pairs are ``(birth, death)`` in the units of ``heights``; the essential
    class (the global-minimum basin that never merges) is omitted, matching
    the finite-pair convention of the repo's persistence barcode.
    """

    n_nodes = len(heights)
    order = list(np.argsort(heights, kind="stable"))
    rank = [0] * n_nodes
    for position, node in enumerate(order):
        rank[node] = position

    parent = list(range(n_nodes))

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    component_birth: Dict[int, float] = {}
    pairs: List[Tuple[float, float]] = []

    for node in order:
        node_height = heights[node]
        active_roots = set()
        for neighbor in graph.neighbors(node):
            if rank[neighbor] < rank[node]:
                active_roots.add(find(neighbor))

        if not active_roots:
            parent[node] = node
            component_birth[node] = node_height
            continue

        # the oldest component (smallest birth) survives; the rest die here
        survivor = min(active_roots, key=lambda root: component_birth[root])
        for root in active_roots:
            if root == survivor:
                continue
            pairs.append((component_birth[root], node_height))
            parent[root] = survivor
        parent[node] = survivor

    return pairs


def _rips_h0_barcode(graph: nx.Graph) -> List[Tuple]:
    """0-dim persistence of the effective-resistance edge filtration.

    Pure spectral filtration (no external scalar field): vertices are born at
    ``0`` and components merge at their connecting effective-resistance
    distance. Finite pairs ``(0, merge_distance)`` are returned.
    """

    n_nodes = graph.number_of_nodes()
    edges = sorted(
        (data["eff_resistance"], u, v) for u, v, data in graph.edges(data=True)
    )

    parent = list(range(n_nodes))
    tree_rank = [0] * n_nodes

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    pairs: List[Tuple[float, float]] = []
    for weight, u, v in edges:
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            continue
        pairs.append((0.0, float(weight)))
        if tree_rank[root_u] < tree_rank[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        if tree_rank[root_u] == tree_rank[root_v]:
            tree_rank[root_u] += 1

    return pairs


def spectral_persistence_barcode(
    loss_coords=None,
    loss_values=None,
    n_neighbors: Optional[int] = None,
    metric: str = "euclidean",
) -> List[Dict[str, float]]:
    """Persistence barcode computed over the spectral (effective-resistance) graph.

    When ``loss_values`` is given, returns 0-dimensional sublevel-set
    persistence of the loss heights over the spectral graph (same semantics
    as the repo's loss-landscape persistence barcode). When ``loss_values``
    is ``None``, returns 0-dim persistence of the pure effective-resistance
    edge filtration. Pairs are emitted in the repo's ``{x, y0, y1}`` barcode
    shape with ``y0`` the birth, ``y1`` the death and ``x`` the birth.
    """

    _, graph = effective_resistance_graph(
        loss_coords=loss_coords,
        n_neighbors=n_neighbors,
        metric=metric,
        return_graph=True,
        verbose=0,
    )

    if loss_values is None:
        pairs = _rips_h0_barcode(graph)
    else:
        heights = np.asarray(loss_values, dtype=float).ravel()
        pairs = _lower_star_h0_barcode(graph, heights)

    return [
        {"x": float(birth), "y0": float(birth), "y1": float(death)}
        for birth, death in pairs
    ]
