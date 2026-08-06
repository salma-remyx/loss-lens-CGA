"""Condition-number spectral metric for weight matrices.

Implements the portable core of kappa-LoRA: the condition number of a weight
matrix (the ratio of its largest to smallest singular value) measures how
"underdeveloped" its directions are. kappa-LoRA's central finding is that
matrices with *larger* condition numbers span richer subspaces and are far more
critical for adaptation than well-balanced (low-kappa) matrices, which
contribute only marginally. Restricting updates to the top-50% of matrices by
condition number halves the trainable parameter count with negligible accuracy
loss.

This framework computes loss-landscape / model-geometry metrics rather than
training models, so the training-time selective-update procedure and benchmark
suite from the paper are intentionally out of scope here. What is ported at full
fidelity is the signal itself: per-matrix condition numbers, a ranking by that
signal, and the kappa-LoRA top-fraction selection rule -- exposed as an
inspectable metric alongside the existing Hessian / curvature metrics.

Adapted from "kappa-LoRA: Condition Numbers Reveal Which LoRA Matrices Worth
Updating" (arXiv:2607.22489).
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np

# kappa-LoRA's headline setting: restrict updates to the top half of weight
# matrices ranked by condition number.
DEFAULT_SELECTION_FRACTION = 0.5


def _singular_extremes(weight: np.ndarray) -> Tuple[float, float]:
    """Return (sigma_max, sigma_min) of a 2D matrix via economy SVD."""
    singular_values = np.linalg.svd(weight, compute_uv=False)
    return float(singular_values[0]), float(singular_values[-1])


def condition_number(weight: np.ndarray) -> float:
    """Condition number ``kappa = sigma_max / sigma_min`` of a 2D weight matrix.

    This is the paper's definition. A rank-deficient matrix (sigma_min ~= 0) is
    treated as infinitely ill-conditioned -- i.e. maximally underdeveloped and
    therefore maximally critical for adaptation -- and reported as ``+inf``.
    """
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError(
            f"condition_number expects a 2D matrix, got shape {weight.shape}"
        )
    sigma_max, sigma_min = _singular_extremes(weight)
    if sigma_min <= 0.0 or not np.isfinite(sigma_min):
        return float("inf")
    return sigma_max / sigma_min


def analyze_layers(
    named_weights: Iterable[Tuple[str, np.ndarray]],
) -> List[Dict]:
    """Compute a condition-number descriptor per 2D weight matrix.

    ``named_weights`` is an iterable of ``(layer_name, 2D ndarray)`` pairs.
    Returns one descriptor per matrix, sorted by condition number descending
    (most-adaptation-critical first), matching kappa-LoRA's importance ordering.
    Non-finite condition numbers are stored as ``None`` so the result stays
    JSON-serializable for storage alongside the other model metrics.
    """
    entries: List[Dict] = []
    for layer_name, weight in named_weights:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.ndim != 2:
            raise ValueError(
                f"analyze_layers expects 2D matrices, layer {layer_name!r} "
                f"got shape {weight.shape}"
            )
        sigma_max, sigma_min = _singular_extremes(weight)
        kappa = (
            float("inf")
            if sigma_min <= 0.0 or not np.isfinite(sigma_min)
            else sigma_max / sigma_min
        )
        entries.append(
            {
                "layer": layer_name,
                "conditionNumber": None if not np.isfinite(kappa) else kappa,
                "singularMax": sigma_max,
                "singularMin": sigma_min,
                "shape": [int(weight.shape[0]), int(weight.shape[1])],
                "_kappa": kappa,
            }
        )
    entries.sort(key=lambda entry: entry["_kappa"], reverse=True)
    for entry in entries:
        del entry["_kappa"]
    return entries


def select_top_fraction(entries: List[Dict], fraction: float) -> Dict:
    """kappa-LoRA selection rule: keep the top ``fraction`` of matrices by
    condition number.

    ``entries`` must already be sorted by condition number descending (as
    returned by :func:`analyze_layers`). At least one matrix is selected when
    ``fraction > 0`` and matrices exist, so a model with a single weight still
    surfaces its one critical matrix.
    """
    total = len(entries)
    if total == 0 or fraction <= 0.0:
        return {
            "selected": [],
            "selectedCount": 0,
            "totalCount": total,
            "fraction": fraction,
            "meanConditionNumber": 0.0,
        }
    selected_count = max(1, int(round(total * fraction)))
    selected_entries = entries[:selected_count]
    selected_kappas = [
        entry["conditionNumber"]
        for entry in selected_entries
        if entry["conditionNumber"] is not None
    ]
    mean_kappa = float(np.mean(selected_kappas)) if selected_kappas else 0.0
    return {
        "selected": [entry["layer"] for entry in selected_entries],
        "selectedCount": selected_count,
        "totalCount": total,
        "fraction": fraction,
        "meanConditionNumber": mean_kappa,
    }


def analyze(
    named_weights: Iterable[Tuple[str, np.ndarray]],
    fraction: float = DEFAULT_SELECTION_FRACTION,
) -> Dict:
    """Full kappa-LoRA signal for a set of weight matrices.

    Returns per-layer condition numbers (ranked), the top-``fraction`` selection
    (the paper's trainable-parameter halving rule), and aggregate summary
    statistics. The whole structure is JSON-serializable so it can be stored as
    a model metric next to the Hessian eigenvalues.
    """
    entries = analyze_layers(named_weights)
    finite_kappas = [
        entry["conditionNumber"]
        for entry in entries
        if entry["conditionNumber"] is not None
    ]
    summary = {
        "mean": float(np.mean(finite_kappas)) if finite_kappas else 0.0,
        "max": float(np.max(finite_kappas)) if finite_kappas else 0.0,
        "min": float(np.min(finite_kappas)) if finite_kappas else 0.0,
    }
    return {
        "layers": entries,
        "selection": select_top_fraction(entries, fraction),
        "summary": summary,
    }
