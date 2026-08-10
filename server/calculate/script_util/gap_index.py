"""Gap Index (GI): distortion of the *empty regions* of a 2D projection.

Adapted from "Measuring Distortion in the Empty Regions of Dimensionality
Reduction Scatterplots with the Gap Index" (arXiv:2607.28324).

Most dimensionality-reduction (DR) quality metrics (trustworthiness, continuity,
neighborhood hit, stress, ...) only score relationships between *points* and are
blind to distortion in the empty space between points. The Gap Index fills that
gap: it partitions the 2D projection into empty triangles via a Delaunay
triangulation, compares each empty triangle's shape to the shape of its
high-dimensional counterpart, and aggregates the per-triangle deformation into a
single scalar. A high GI warns that structures a user reads in the empty regions
of the scatterplot (cluster separations, corridors between modes) are artifacts
of the projection rather than of the data.

Paper mechanism kept at full fidelity
-------------------------------------
  * Decompose the projection into empty triangles with a Delaunay triangulation.
  * For each empty triangle, compare its 2D shape to its high-dimensional
    counterpart and accumulate a deformation.
  * Aggregate area-weighted (larger empty regions are more visually salient) and
    also expose the per-triangle deformation for a regional overlay.

Per-triangle deformation (concrete realization)
-----------------------------------------------
The paper's mechanism is "compare each empty triangle to its high-dimensional
counterpart"; it does not fix a single closed-form deformation. We realize it
with a parameter-free, scale-invariant triangle *shape distance*: the L1
difference of the perimeter-normalized edge lengths between the 2D triangle and
its high-dimensional counterpart. Perimeter normalization makes the score
invariant to a global rescaling of either space (a uniform scale is not
"distortion"), so the GI measures pure shape deformation of the empty regions.
"""

import numpy as np
from scipy.spatial import Delaunay, QhullError


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Element-wise divide that yields 0 where the denominator is 0."""
    out = np.zeros_like(numer, dtype=float)
    nonzero = denom != 0
    out[nonzero] = numer[nonzero] / denom[nonzero]
    return out


def _triangle_areas(coords2d: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Signed-magnitude area of each triangle via the cross-product formula."""
    a = coords2d[simplices[:, 0]]
    b = coords2d[simplices[:, 1]]
    c = coords2d[simplices[:, 2]]
    return 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )


def compute_gap_index(
    high_coords: np.ndarray,
    proj_coords: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the Gap Index of a 2D projection against its high-dim source.

    Parameters
    ----------
    high_coords : (n, d) array
        High-dimensional coordinates of the ``n`` points.
    proj_coords : (n, 2) array
        The same ``n`` points after dimensionality reduction to 2D.

    Returns
    -------
    dict with keys
        ``gap_index``       : float  -- area-weighted mean empty-triangle
                                         deformation (0 = no shape distortion).
        ``per_triangle``    : (m,)   -- deformation of each empty triangle
                                         (for a regional overlay).
        ``areas``           : (m,)   -- 2D area of each empty triangle.
        ``simplices``       : (m, 3) -- Delaunay triangles (point indices).
        ``n_triangles``     : int    -- number of empty triangles (0 if GI
                                         could not be computed).

    Notes
    -----
    Returns ``gap_index = nan`` and ``n_triangles = 0`` when a Delaunay
    triangulation of the projection cannot be formed (fewer than 3 points or
    all points collinear).
    """
    high_coords = np.asarray(high_coords, dtype=float)
    proj_coords = np.asarray(proj_coords, dtype=float)

    empty = {
        "gap_index": float("nan"),
        "per_triangle": np.empty((0,), dtype=float),
        "areas": np.empty((0,), dtype=float),
        "simplices": np.empty((0, 3), dtype=int),
        "n_triangles": 0,
    }

    if high_coords.ndim != 2 or proj_coords.ndim != 2:
        raise ValueError("high_coords and proj_coords must both be 2D arrays")
    if high_coords.shape[0] != proj_coords.shape[0]:
        raise ValueError("high_coords and proj_coords must share the first axis")
    if proj_coords.shape[1] != 2:
        raise ValueError("proj_coords must be 2-dimensional (shape (n, 2))")
    if proj_coords.shape[0] < 3:
        return empty

    # Decompose the projection into empty triangles.
    try:
        tri = Delaunay(proj_coords)
    except QhullError:
        # Points are collinear / degenerate -- no empty regions to score.
        return empty
    simplices = tri.simplices  # (m, 3) int
    if simplices.shape[0] == 0:
        return empty

    i, j, k = simplices[:, 0], simplices[:, 1], simplices[:, 2]

    # Edge lengths in 2D and in the high-dimensional space.
    def edge(a, b, coords):
        return np.linalg.norm(coords[a] - coords[b], axis=1)

    e2_ij, e2_jk, e2_ki = edge(i, j, proj_coords), edge(j, k, proj_coords), edge(k, i, proj_coords)
    eh_ij, eh_jk, eh_ki = edge(i, j, high_coords), edge(j, k, high_coords), edge(k, i, high_coords)

    perim2 = e2_ij + e2_jk + e2_ki
    perimh = eh_ij + eh_jk + eh_ki

    # Perimeter-normalized edge lengths -> scale-invariant triangle shape.
    # Deformation = L1 distance between the 2D and HD normalized edge vectors.
    per_triangle = (
        np.abs(_safe_divide(e2_ij, perim2) - _safe_divide(eh_ij, perimh))
        + np.abs(_safe_divide(e2_jk, perim2) - _safe_divide(eh_jk, perimh))
        + np.abs(_safe_divide(e2_ki, perim2) - _safe_divide(eh_ki, perimh))
    )

    areas = _triangle_areas(proj_coords, simplices)
    total_area = float(areas.sum())
    if total_area > 0:
        gap_index = float((areas * per_triangle).sum() / total_area)
    elif per_triangle.size:
        # All triangles degenerate to zero area; fall back to the plain mean.
        gap_index = float(per_triangle.mean())
    else:
        gap_index = float("nan")

    return {
        "gap_index": gap_index,
        "per_triangle": per_triangle,
        "areas": areas,
        "simplices": simplices,
        "n_triangles": int(simplices.shape[0]),
    }


def gap_index_scalar(
    high_coords: np.ndarray, proj_coords: np.ndarray
) -> float | None:
    """Convenience wrapper returning only the scalar Gap Index."""
    return compute_gap_index(high_coords, proj_coords)["gap_index"]
