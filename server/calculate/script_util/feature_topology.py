"""Feature-space topology measurement via the Hopkins statistic.

The Hopkins statistic (Hopkins & Skellam, 1954) scores the *clustering
tendency* of a point set on the interval ``[0, 1]``:

  * near ``1``  -- samples are tightly clustered,
  * near ``0.5``-- samples are spread roughly uniformly at random,
  * near ``0``  -- samples are regularly (evenly) spaced.

Applied per layer to a network's activation matrix (``batch x features``)
it yields a scalar summary of that layer's feature-space *topology* -- a
per-layer companion signal to the CKA similarity matrix that
``compute_layer_similarity`` already produces.

Adapted from "Feature Space Topology Control via Hopkins Loss"
(arXiv:2509.11154). The paper introduces *Hopkins loss*, a training
objective built on the target-H form of the statistic. LossLens is a
visual *analysis* framework with no training loop, so this ports the
statistic itself as a passive, per-layer measurement (the paper's core
mechanism) and intentionally drops the training-loss wrapper, optimizer,
and the target-H control surface -- those have no host here.

The implementation is dependency-free (standard library only) so it can be
exercised without the heavy backend stack; torch / numpy arrays are coerced
via their ``.tolist()`` methods.
"""

import math
import random
from typing import Dict, List, Optional


def _flatten_recursive(x) -> List[float]:
    """Flatten an arbitrarily nested scalar/list/tuple into a flat float list."""
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for e in x:
            if isinstance(e, (list, tuple)):
                out.extend(_flatten_recursive(e))
            else:
                out.append(float(e))
        return out
    return [float(x)]


def _coerce_rows(X) -> List[List[float]]:
    """Coerce a torch/numpy array or nested list into ``[n_points][n_features]``."""
    if hasattr(X, "tolist"):
        X = X.tolist()
    if isinstance(X, (list, tuple)):
        return [_flatten_recursive(row) for row in X]
    raise TypeError(f"Unsupported activation type: {type(X)!r}")


def hopkins_statistic(
    X,
    sampling_ratio: float = 0.1,
    n_samples: Optional[int] = None,
    seed: int = 0,
) -> float:
    """Hopkins clustering-tendency statistic for a ``(batch, features)`` matrix.

    ``X`` is a ``(n_points, n_features)`` array-like (torch tensor, numpy
    array, or nested list). Returns a float in ``[0, 1]`` (``nan`` when the
    input is too small to define the statistic). ``n_samples`` overrides the
    default subsample size of ``round(sampling_ratio * n_points)``.
    """
    rows = _coerce_rows(X)
    n = len(rows)
    if n < 2:
        return float("nan")
    d = len(rows[0])
    if d < 1:
        return float("nan")

    if n_samples is None:
        m = max(1, min(n, int(round(sampling_ratio * n))))
    else:
        m = max(1, min(n, int(n_samples)))

    rng = random.Random(seed)

    # W: sum of nearest-neighbour distances for m sampled real points.
    sample_idx = rng.sample(range(n), m)
    w_sum = 0.0
    for i in sample_idx:
        xi = rows[i]
        best = math.inf
        for j in range(n):
            if j == i:
                continue
            dist = _sq_dist(xi, rows[j])
            if dist < best:
                best = dist
        w_sum += math.sqrt(best)

    # Bounding box of the data; constant (zero-span) dims get a unit span.
    columns = list(zip(*rows))
    lo = [min(col) for col in columns]
    hi = [max(col) for col in columns]
    span = [(hi[k] - lo[k]) if hi[k] > lo[k] else 1.0 for k in range(d)]

    # U: sum of nearest-neighbour distances for m uniform random points.
    u_sum = 0.0
    for _ in range(m):
        r = [lo[k] + rng.random() * span[k] for k in range(d)]
        best = math.inf
        for j in range(n):
            dist = _sq_dist(r, rows[j])
            if dist < best:
                best = dist
        u_sum += math.sqrt(best)

    denom = u_sum + w_sum
    if denom == 0.0:
        return float("nan")
    return float(u_sum / denom)


def _sq_dist(a: List[float], b: List[float]) -> float:
    s = 0.0
    for ak, bk in zip(a, b):
        diff = ak - bk
        s += diff * diff
    return s


def layer_hopkins(
    features,
    sampling_ratio: float = 0.1,
    n_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Per-layer Hopkins statistic for a ``{layer_name: activations}`` mapping.

    ``features`` is a dict keyed by layer name (or any iterable of
    ``(name, activations)`` pairs, e.g. a CKA object's ``model1_features``).
    Each activation entry is a ``(batch, ...)`` array-like; it is flattened
    row-wise so each row is one sample point in that layer's feature space.
    """
    if hasattr(features, "items"):
        items = list(features.items())
    else:
        items = list(features)
    return {
        str(name): hopkins_statistic(
            act,
            sampling_ratio=sampling_ratio,
            n_samples=n_samples,
            seed=seed,
        )
        for name, act in items
    }
