"""HESD-type-aware generalization assessment for the loss-landscape pipeline.

Adapted from: "The effects of Hessian eigenvalue spectral density type on the
applicability of Hessian analysis to generalization capability assessment of
neural networks" (arXiv:2504.17618). That work observes that Hessian-based
generalization criteria rely on the Hessian eigenvalue spectral density (HESD)
behaving consistently across networks, and shows that the *type* of HESD
(positive-semidefinite bulk vs. indefinite/saddle vs. outlier-dominated) governs
whether those criteria are actually applicable for a given model.

This module delivers that insight as a measurement that slots into the repo's
existing Hessian pipeline. Mode 2 (adapted port):

  * Faithful core: the HESD is the real spectral density estimated by
    ``pyhessian``'s stochastic-Lanczos ``density()`` method, which the pipeline
    already constructs (see ``core_functions.compute_mode_hessian``) but did not
    previously invoke. The HESD-type classification and the applicability flag
    follow the paper's framing directly.
  * Target-native substitution: the paper studies a *family* of generalization
    criteria and the factors that change HESD type rather than prescribing one
    universal formula for arbitrary networks. The scalar ``generalizationCriterion``
    below is a target-native composite of the HESD features the paper identifies
    as carrying the generalization signal (top curvature / sharpness and the
    negative-eigenvalue mass that marks a non-minimum). The paper's full
    benchmark / cross-architecture study apparatus is intentionally out of scope.

All public functions are pure (NumPy only) so the criterion logic is testable
without loading a model or torch.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Fraction of spectral mass that must lie on negative eigenvalues before the
# HESD is treated as indefinite (i.e. the evaluated point is not a minimum).
NEGATIVE_MASS_THRESHOLD = 0.05

# A HESD is "outlier-dominated" when the single largest eigenvalue holds more
# than this share of the total positive curvature -- a few sharp directions
# dominating a small bulk, the regime the paper flags as degrading curvature-
# based criteria.
OUTLIER_CURVATURE_SHARE = 0.25


def flatten_density(
    eigen_list_full: Sequence[Sequence[complex]],
    weight_list_full: Sequence[Sequence[complex]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten ``pyhessian``'s ``density()`` output into real eigenvalue/weight arrays.

    ``density()`` returns one (eigenvalues, weights) pair per stochastic Lanczos
    run. The Lanczos tridiagonal ``T`` is symmetric, so its eigenpairs are real
    up to numerical noise; the imaginary parts are discarded here. Weights are
    the squared eigenvector components and are clipped to be non-negative.

    Returns two 1-D float arrays of equal length (the combined HESD sample).
    """
    if not eigen_list_full:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    eigenvalues = np.concatenate(
        [np.real(np.asarray(ev, dtype=complex)).ravel() for ev in eigen_list_full]
    )
    weights = np.concatenate(
        [np.real(np.asarray(w, dtype=complex)).ravel() for w in weight_list_full]
    )

    # Numerical noise can push squared weights slightly negative.
    weights = np.clip(weights, 0.0, None)

    return eigenvalues.astype(float), weights.astype(float)


def classify_hesd_type(eigenvalues: np.ndarray, weights: np.ndarray) -> str:
    """Classify the HESD type from a weighted eigenvalue sample.

    Returns one of:
      * ``"psd_bulk"``          -- mass concentrated on non-negative eigenvalues
                                    with no dominant outlier; the regime where the
                                    paper says Hessian-based generalization
                                    criteria are applicable.
      * ``"outlier_dominated"`` -- a few sharp positive directions dominate a
                                    small bulk; criteria only partially applicable.
      * ``"indefinite"``        -- substantial mass on negative eigenvalues, i.e.
                                    the evaluated point is not a minimum; criteria
                                    questionable per the paper.
      * ``"unknown"``           -- empty / degenerate spectrum.
    """
    if eigenvalues.size == 0:
        return "unknown"

    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return "unknown"
    norm_weights = weights / total_weight

    negative_mass = float(norm_weights[eigenvalues < 0.0].sum())
    if negative_mass > NEGATIVE_MASS_THRESHOLD:
        return "indefinite"

    # Outlier dominance: does the single largest eigenvalue carry a
    # disproportionate share of the positive curvature? This is robust to the
    # outlier itself inflating global spread statistics.
    max_index = int(np.argmax(eigenvalues))
    max_eigenvalue = float(eigenvalues[max_index])
    positive_mask = eigenvalues > 0.0
    positive_curvature = float(
        (norm_weights[positive_mask] * eigenvalues[positive_mask]).sum()
    )
    if positive_curvature > 0.0:
        top_share = (
            max_eigenvalue * float(norm_weights[max_index])
        ) / positive_curvature
        if top_share > OUTLIER_CURVATURE_SHARE:
            return "outlier_dominated"

    return "psd_bulk"


def _applicability(hesd_type: str) -> str:
    """Map an HESD type to the paper's applicability verdict for curvature criteria."""
    return {
        "psd_bulk": "applicable",
        "outlier_dominated": "partial",
        "indefinite": "questionable",
    }.get(hesd_type, "unknown")


def generalization_criterion(
    eigenvalues: np.ndarray, weights: np.ndarray
) -> Dict[str, object]:
    """Compute an HESD-type-aware generalization assessment.

    Combines the curvature statistics the paper's criteria rely on with an
    explicit HESD-type / applicability verdict. The scalar
    ``generalizationCriterion`` is the top eigenvalue (sharpness) scaled up by
    the negative-eigenvalue mass: lower is flatter and more minimum-like, which
    the paper's family of criteria associates with better generalization.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    weights = np.asarray(weights, dtype=float)

    hesd_type = classify_hesd_type(eigenvalues, weights)

    if eigenvalues.size == 0:
        return {
            "hesdType": hesd_type,
            "applicability": _applicability(hesd_type),
            "generalizationCriterion": None,
            "negativeMass": None,
            "maxEigenvalue": None,
            "minEigenvalue": None,
            "bulkStd": None,
            "weightedMean": None,
            "numEigenvalues": 0,
        }

    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        norm_weights = np.full_like(weights, 1.0 / weights.size)
    else:
        norm_weights = weights / total_weight

    negative_mass = float(norm_weights[eigenvalues < 0.0].sum())
    weighted_mean = float((norm_weights * eigenvalues).sum())
    variance = float((norm_weights * (eigenvalues - weighted_mean) ** 2).sum())
    bulk_std = float(np.sqrt(max(variance, 0.0)))
    max_eigenvalue = float(eigenvalues.max())
    min_eigenvalue = float(eigenvalues.min())

    # Target-native composite (see module docstring): sharpness amplified by the
    # share of the spectrum that is negative, since a non-minimum point makes
    # curvature-based generalization estimates less reliable.
    criterion = max_eigenvalue * (1.0 + negative_mass)

    return {
        "hesdType": hesd_type,
        "applicability": _applicability(hesd_type),
        "generalizationCriterion": float(criterion),
        "negativeMass": negative_mass,
        "maxEigenvalue": max_eigenvalue,
        "minEigenvalue": min_eigenvalue,
        "bulkStd": bulk_std,
        "weightedMean": weighted_mean,
        "numEigenvalues": int(eigenvalues.size),
    }


def assess_density(
    eigen_list_full: Sequence[Sequence[complex]],
    weight_list_full: Sequence[Sequence[complex]],
) -> Dict[str, object]:
    """Convenience wrapper: flatten ``density()`` output then run the criterion."""
    eigenvalues, weights = flatten_density(eigen_list_full, weight_list_full)
    return generalization_criterion(eigenvalues, weights)


__all__: List[str] = [
    "flatten_density",
    "classify_hesd_type",
    "generalization_criterion",
    "assess_density",
]
