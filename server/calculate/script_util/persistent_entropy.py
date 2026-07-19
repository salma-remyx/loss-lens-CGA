"""Persistent entropy summaries for loss-landscape persistence barcodes.

This module provides two scalar topological summaries computed from a
persistence barcode (the list of birth/death pairs produced by the TTK
persistence-diagram pipeline):

  * Persistent entropy (PE) -- Eq. (1)
  * Length-weighted persistent entropy (LWPE) -- Eq. (2)

Adapted from: V. Toscano-Duran, R. Gonzalez-Diaz, M. A. Gutierrez-Naranjo,
"Barycentric Neural Networks and Length-Weighted Persistent Entropy Loss:
A Green Geometric and Topological Framework for Function Approximation",
arXiv:2509.06694 (2025).

Adaptation note (Mode 2). The paper introduces LWPE as a *training loss*
(|LWPE_ref - LWPE_pred|) used to optimise the base points of a Barycentric
Neural Network. LossLens is an analysis tool -- it does not train BNNs -- so
only the two scalar summaries are ported here, at full fidelity to the
paper's definitions. The BNN architecture, the base-point optimisation loop,
and the loss-vs-reference formulation are intentionally out of scope for this
tool; what remains is a pair of parameter-free topological metrics over the
barcode the repo already computes.

Definitions (arXiv:2509.06694, Sec. 2-3). Given a persistence diagram
D = {(b_i, e_i) | i in I} of finite pairs, with persistence length
ell_i = e_i - b_i and total persistence L = sum_i ell_i:

    PE   = - sum_i  p_i * ln(p_i),        p_i = ell_i / L      ... (Eq. 1)
    LWPE = - sum_i  ell_i * ln(p_i),      p_i = ell_i / L      ... (Eq. 2)

PE measures the uniformity of bar lengths (scale-invariant; maximal at
ln(#I) when all bars are equal). LWPE re-weights each entropy term by the
absolute persistence ell_i, so it is sensitive to the *scale* of the
topological features as well as their distribution -- the property the paper
shows distinguishes faithful loss-landscape approximations from coarse ones.

Infinite / non-positive-length pairs (the essential class, diagonal noise)
are excluded, since their length is undefined in the entropy. With no finite
bars the summaries are defined to be 0.0.
"""

from math import isfinite, log
from typing import Dict, Iterable, List, Tuple

Pair = Tuple[float, float]


def filter_finite_pairs(pairs: Iterable[Pair]) -> List[Pair]:
    """Keep only finite pairs with strictly positive persistence (e_i > b_i)."""
    result: List[Pair] = []
    for pair in pairs:
        try:
            b, e = float(pair[0]), float(pair[1])
        except (TypeError, IndexError, ValueError):
            continue
        if e > b and isfinite(b) and isfinite(e):  # drops NaN and the essential/infinite bar
            result.append((b, e))
    return result


def _lengths(pairs: Iterable[Pair]) -> List[float]:
    return [e - b for b, e in filter_finite_pairs(pairs)]


def persistent_entropy(pairs: Iterable[Pair]) -> float:
    """Persistent entropy (Eq. 1): -sum p_i ln p_i, with p_i = ell_i / L."""
    lengths = _lengths(pairs)
    total = sum(lengths)
    if total <= 0.0:
        return 0.0
    return -sum((ell / total) * _ln(ell / total) for ell in lengths)


def length_weighted_persistent_entropy(pairs: Iterable[Pair]) -> float:
    """Length-weighted persistent entropy (Eq. 2): -sum ell_i ln p_i.

    Unlike :func:`persistent_entropy`, each entropy term is weighted by the
    absolute persistence ``ell_i`` rather than the normalised proportion
    ``p_i``, so the value scales with the magnitude of the topological
    features (it is not invariant under rescaling of the function values).
    """
    lengths = _lengths(pairs)
    total = sum(lengths)
    if total <= 0.0:
        return 0.0
    return -sum(ell * _ln(ell / total) for ell in lengths)


def persistent_entropy_summary(pairs: Iterable[Pair]) -> Dict[str, float]:
    """Return PE, LWPE and supporting counts for a set of persistence pairs."""
    finite = filter_finite_pairs(pairs)
    lengths = [e - b for b, e in finite]
    total = sum(lengths)
    return {
        "persistentEntropy": persistent_entropy(finite),
        "lengthWeightedPersistentEntropy": length_weighted_persistent_entropy(finite),
        "numBars": float(len(finite)),
        "totalPersistence": float(total),
    }


def _ln(x: float) -> float:
    """Natural log guarded against the p_i -> 0 limit (lim p ln p = 0)."""
    if x <= 0.0:
        return 0.0
    return log(x)
