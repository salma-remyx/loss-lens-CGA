"""Widest-path (Max-Min) reachability fields for connectivity-bottleneck analysis.

Adapted from "Widest-Path Reachability Fields for Connectivity-Preserving
Slender Structure Segmentation" (arXiv:2607.07123). That work introduces a
differentiable Max-Min reachability objective that re-routes gradient flow to
connectivity *bottlenecks*, countering **topological gradient starvation (TGS)**:
point-wise objectives spread gradients uniformly, yet connectivity hinges on a
sparse set of bottleneck pixels whose aggregate gradient contribution is
negligible. Optimising the widest-path (maximum-bottleneck) reachability
concentrates the signal exactly on those pixels.

LossLens studies loss landscapes as scalar fields rather than training
segmentation models, so this module exposes the paper's CORE mechanism -- the
widest-path reachability field and the bottleneck pixels it isolates -- as a
topology *descriptor* on a 2D scalar field (e.g. a planar loss-landscape slice).
It is a companion to the TTK merge-tree / persistence-diagram analysis, with the
same "scalar field -> topology" shape.

Adaptation notes (Mode 2)
-------------------------
* Core mechanism kept at full fidelity: the widest-path reachability
  ``R(p) = max over source->p paths of min(vertex weight along the path)``,
  i.e. the maximum-bottleneck (widest) path value, computed by a Dijkstra-style
  maximising traversal over the 4-neighbour grid (dynamic programming on a
  domain-restricted graph, as in the paper).
* Substituted auxiliary: the *differentiable soft-relaxation* the paper uses to
  make the objective trainable is replaced by the EXACT widest-path reachability
  -- the faithful core object, and the correct choice for analysis (the soft
  variant only exists to keep the objective differentiable for gradient descent,
  which LossLens does not perform here).
* Field polarity is parameterised: ``maximize=True`` recovers the paper's
  segmentation semantics (capacity = field value, high = traversable; seeds at
  the global maximum); ``maximize=False`` gives loss-landscape semantics
  (capacity = -loss, seeds at the global minimum) so that low reachability marks
  the saddles / mountain passes -- the barriers between basins.
* Intentionally out of scope: the segmentation training loop, the OMVIS dataset,
  backbone integration, and clDice evaluation (LossLens has no segmentation
  trainer to host them).

This module is dependency-free (stdlib only) so it imports anywhere; it accepts
any 2D indexable field (list of lists or a numpy 2-D array).
"""

import heapq


def _shape(field):
    rows = len(field)
    if rows == 0:
        return 0, 0
    cols = len(field[0])
    return rows, cols


def _neighbors(i, j, h, w):
    if i > 0:
        yield i - 1, j
    if i < h - 1:
        yield i + 1, j
    if j > 0:
        yield i, j - 1
    if j < w - 1:
        yield i, j + 1


def _argmax_2d(values, h, w):
    """Index of the maximum entry of a flat ``values`` list over an (h, w) grid."""
    best_i = 0
    best_v = values[0]
    for k in range(1, len(values)):
        if values[k] > best_v:
            best_v = values[k]
            best_i = k
    return best_i // w, best_i % w


def widest_path_reachability_field(field, source=None, maximize=True):
    """Exact widest-path (Max-Min) reachability from ``source`` over a 2D field.

    Parameters
    ----------
    field : 2D indexable of numbers
        The scalar field (a loss-landscape slice, or a predicted probability map).
    source : tuple(int, int), optional
        Seed pixel. Defaults to the highest-capacity pixel: the global maximum of
        ``field`` when ``maximize`` else the global minimum.
    maximize : bool
        Field polarity. ``True`` -> capacity is the field value (paper's
        segmentation setting); ``False`` -> capacity is ``-field`` (loss-landscape
        setting, where low reachability flags inter-basin barriers).

    Returns
    -------
    list of list of float
        ``R[p]`` = the bottleneck (minimum capacity) of the widest source->p path.
        Unreachable pixels get ``-inf``.
    """
    h, w = _shape(field)
    if h == 0 or w == 0:
        return []
    flat = [float(field[i][j]) for i in range(h) for j in range(w)]
    capacity = flat if maximize else [-v for v in flat]
    if source is None:
        source = _argmax_2d(capacity, h, w)
    src = source[0] * w + source[1]

    neg_inf = float("-inf")
    reach = [neg_inf] * (h * w)
    reach[src] = capacity[src]
    # Max-heap keyed by reachability (store negatives for heapq's min-heap).
    heap = [(-capacity[src], src)]
    while heap:
        neg_val, u = heapq.heappop(heap)
        val = -neg_val
        if val < reach[u]:
            continue  # stale entry; a better reachability was already recorded
        ui, uj = u // w, u % w
        for vi, vj in _neighbors(ui, uj, h, w):
            v = vi * w + vj
            # widest path to v through u is bottlenecked by v's own capacity
            candidate = val if val < capacity[v] else capacity[v]
            if candidate > reach[v]:
                reach[v] = candidate
                heapq.heappush(heap, (-candidate, v))

    return [[reach[i * w + j] for j in range(w)] for i in range(h)]


def bottleneck_pixels(field, source=None, maximize=True, fraction=0.1):
    """Return the lowest-reachability pixels -- the connectivity bottlenecks.

    These are the pixels whose widest-path reachability is smallest: the sparse
    set on which connectivity hinges (the TGS targets in the paper's framing, or
    the saddle/barrier pixels of a loss landscape).

    Parameters
    ----------
    fraction : float
        Fraction of pixels (by count, at least one) to return, sorted from the
        lowest reachability upward.

    Returns
    -------
    list of [row, col]
    """
    h, w = _shape(field)
    reach = widest_path_reachability_field(field, source=source, maximize=maximize)
    ranked = sorted(
        ((reach[i][j], i, j) for i in range(h) for j in range(w)),
        key=lambda t: t[0],
    )
    count = max(1, int(round(fraction * len(ranked))))
    return [[i, j] for _, i, j in ranked[:count]]


def widest_path_reachability_descriptor(field, source=None, maximize=False, fraction=0.1):
    """Scalar-field -> topology descriptor summarising widest-path reachability.

    Intended as a companion descriptor to the TTK merge tree / persistence
    diagram of the same field. ``maximize`` defaults to ``False`` (loss-landscape
    semantics) since LossLens analyses loss landscapes.

    Returns a dict with:
    * ``reachabilityField`` -- the full ``R`` field (list of lists);
    * ``bottleneckValue`` -- ``min R`` (worst barrier / weakest link);
    * ``bottleneckPixels`` -- the lowest-reachability pixels (connectivity
      bottlenecks);
    * ``connectivity`` -- mean normalised reachability in [0, 1]; higher means the
      field is more uniformly reachable from the seed (better connected).
    """
    h, w = _shape(field)
    reach = widest_path_reachability_field(field, source=source, maximize=maximize)

    flat = [reach[i][j] for i in range(h) for j in range(w)]
    finite = [v for v in flat if v != float("-inf")]
    r_min = min(finite) if finite else 0.0
    r_max = max(finite) if finite else 0.0
    span = (r_max - r_min) or 1.0
    connectivity = sum((v - r_min) / span for v in finite) / len(finite) if finite else 0.0

    return {
        "reachabilityField": reach,
        "bottleneckValue": float(r_min),
        "bottleneckPixels": bottleneck_pixels(
            field, source=source, maximize=maximize, fraction=fraction
        ),
        "connectivity": float(connectivity),
    }
