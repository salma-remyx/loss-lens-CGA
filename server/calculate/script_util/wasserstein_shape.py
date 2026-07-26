"""Distributional-shape metric based on the squared 2-Wasserstein distance
to the standard Gaussian.

Adapted from: "Contrast-Free ICA and Causal Inference via Wasserstein
Distances to the Gaussian" (arXiv:2607.12832).

That paper studies the squared 2-Wasserstein distance to ``N(0, 1)`` as a
*non-Gaussianity criterion*. Its core observation is a strict inequality:
independent standardized sources are strictly *further* from the Gaussian
(in W2) than any unit-norm linear combination that mixes them. In other
words, "non-Gaussianity = structure" -- the more a standardized
distribution departs from the Gaussian, the more structural information it
carries. The paper uses that signal for linear ICA / LiNGAM causal
inference; those tasks (unmixing matrix, causal order, Picard-style
optimizer, dynamic-programming order search) are not applicable to a
loss-landscape analysis repo and are intentionally out of scope here.

What we keep at full fidelity is the measure itself. For a 1-D sample the
squared 2-Wasserstein distance has the closed form

    W2^2(mu, nu) = integral_0^1 ( F_mu^-1(u) - F_nu^-1(u) )^2 du

with ``nu = N(0, 1)`` and ``F_nu^-1 = Phi^-1`` the Gaussian quantile
function. Standardizing the sample (zero mean, unit variance) first turns
this into a pure *shape* criterion -- a parameter-free plug-in estimator
of the paper's Wasserstein non-Gaussianity. This module applies that
estimator to the critical-point value distributions the repo already
computes (persistence-barcode birth/death values), complementing the
topological (TTK) and Hessian lenses with a distributional one.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm

__all__ = [
    "wasserstein_squared_to_standard_gaussian",
    "persistence_barcode_shape",
]


def wasserstein_squared_to_standard_gaussian(samples) -> float:
    """Squared 2-Wasserstein distance of a 1-D sample to ``N(0, 1)``.

    The sample is standardized (zero mean, unit variance) before the
    distance is measured, so the result is a scale- and location-invariant
    measure of how non-Gaussian the distribution *shape* is -- the paper's
    Wasserstein non-Gaussianity. Returns ``0.0`` for a Gaussian (in the
    limit) and strictly positive values for non-Gaussian shapes.

    Degenerate inputs (fewer than two samples, or zero variance) return
    ``0.0`` so the metric never crashes a pipeline run; callers that need
    to distinguish "not enough data" should check ``persistence_barcode_shape``.
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    if n < 2:
        return 0.0

    std = x.std(ddof=0)
    if std == 0:
        return 0.0

    # standardized order statistics (empirical quantiles)
    z = np.sort((x - x.mean()) / std)

    # Gaussian quantiles at the matching plotting positions
    u = (np.arange(n) + 0.5) / n
    gauss_q = norm.ppf(u)

    return float(np.mean((z - gauss_q) ** 2))


def persistence_barcode_shape(
    persistence_barcode: List[Dict[str, float]],
    value_key: str = "y1",
) -> Optional[Dict[str, object]]:
    """Distributional-shape descriptor for a persistence barcode.

    A persistence barcode (as produced by ``process_persistence_barcode``
    in either ``read_csv_to_db.py`` or ``ttk_functions.py``) is a list of
    ``{"x", "y0", "y1"}`` entries, where ``y0``/``y1`` are the birth/death
    scalar values of each topological pair. This collects the chosen
    per-pair scalar (the death value ``y1`` by default -- i.e. the
    distribution of critical-point values) and summarizes its departure
    from the Gaussian via the squared 2-Wasserstein non-Gaussianity.

    Returns ``None`` when too few finite values are available to form a
    meaningful distribution, so callers can decide whether to store the
    descriptor at all.
    """
    if not persistence_barcode:
        return None

    values = []
    for entry in persistence_barcode:
        if not isinstance(entry, dict):
            continue
        if value_key not in entry:
            continue
        try:
            value = float(entry[value_key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)

    if len(values) < 2:
        return None

    non_gaussianity = wasserstein_squared_to_standard_gaussian(values)

    return {
        "wassersteinNonGaussianity": non_gaussianity,
        "nSamples": len(values),
        "valueKey": value_key,
    }
