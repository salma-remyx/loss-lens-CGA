"""Basin sharpness / basin volume descriptors for loss-landscape merge trees.

Adapted from: "Loss Landscape Topology Reveals Why Simple Baselines are
Competitive at 3D Point Cloud Segmentation Under Class Imbalance"
(arXiv:2607.21089). That paper's central mechanistic claim is that the severity
of class imbalance shapes the *topology* of the loss landscape: extreme
imbalance carves narrow, sharp solution basins while moderate imbalance leaves
flat plateaus, and that geometric constraint is what blunts loss-level
mitigations. LossLens already extracts merge trees and persistence diagrams
from loss surfaces via TTK; this module turns those descriptors into exactly
the per-basin geometry the paper argues is diagnostic -- basin depth
(persistence), volume (region size), width (region span), and a sharpness /
concentration summary -- plus a landscape-level "narrow-basin vs flat-plateau"
regime score, and correlates that score with a dataset's imbalance ratio.

Implementation mode: Mode 2 (adapted port). The paper's core topological
mechanism -- basin sharpness and volume explaining imbalance-driven topology
-- is implemented at full fidelity on the merge-tree / persistence-diagram
CSVs the TTK pipeline already emits. The paper's auxiliary apparatus is
intentionally out of scope here and belongs in a downstream case study: the 11
imbalance-aware training methods, the 3D point-cloud segmentation datasets at
controlled 641:1 / 56:1 imbalance, and the mIoU benchmark suite, none of which
this repo (a loss-landscape visual analytics framework) can host.

Inputs are the ``MergeTree.csv`` and ``PersistenceDiagram.csv`` files written
by ``calculate_ttk_merge_tree.py`` / ``calculate_ttk_persistence_diagram.py``.
``MergeTree.csv`` columns used: ``Scalar`` (loss value at the critical point),
``CriticalType`` (0 = local minimum / basin, 1 = saddle, 3 = global maximum),
``RegionSize`` (basin volume in grid cells), ``RegionSpan`` (basin width).
``PersistenceDiagram.csv`` columns used: ``Points:0`` (birth), ``Points:1``
(death); a basin's persistence is ``death - birth``.
"""

import csv
import math
import os
from typing import Dict, List, Optional

CRITICAL_MINIMUM = 0
CRITICAL_MAXIMUM = 3


def parse_merge_tree_csv(path: str) -> List[Dict[str, float]]:
    """Read a TTK ``MergeTree.csv`` into per-critical-point records."""
    points: List[Dict[str, float]] = []
    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            points.append(
                {
                    "scalar": float(row["Scalar"]),
                    "critical_type": int(float(row["CriticalType"])),
                    "region_size": int(float(row["RegionSize"])),
                    "region_span": int(float(row["RegionSpan"])),
                    "x": float(row["Points:0"]),
                    "y": float(row["Points:1"]),
                }
            )
    return points


def parse_persistence_diagram_csv(path: str) -> List[Dict[str, float]]:
    """Read a TTK ``PersistenceDiagram.csv`` into per-basin pairs.

    A meaningful 0-dimensional pair (one basin) is any row whose death
    (``Points:1``) exceeds its birth (``Points:0``); trivial on-diagonal points
    (birth == death) carry no persistence and are dropped.
    """
    pairs: List[Dict[str, float]] = []
    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            birth = float(row["Points:0"])
            death = float(row["Points:1"])
            if death > birth:
                pairs.append(
                    {
                        "birth": birth,
                        "death": death,
                        "persistence": death - birth,
                        "critical_type": int(float(row["CriticalType"])),
                    }
                )
    return pairs


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _shannon_entropy_normalized(probs: List[float]) -> float:
    """Shannon entropy of a distribution, normalized to ``[0, 1]``.

    High entropy (many comparable basins) reads as a flat plateau; low entropy
    (one dominant basin) reads as a narrow-basin regime.
    """
    probs = [p for p in probs if p > 0]
    if len(probs) < 2:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(probs))


def compute_basin_descriptors(
    critical_points: List[Dict[str, float]],
    persistence_pairs: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Compute the paper's basin-geometry descriptors for one loss landscape.

    Returns the basin count, depth (persistence), volume, and width, a
    landscape-level ``narrow_basin_score`` in ``[0, 1]`` (high = the narrow,
    sharp, single-dominant-basin regime the paper links to extreme imbalance;
    low = the flat, many-comparable-basin plateau it links to moderate
    imbalance), and a coarse ``regime`` label.
    """
    minima = [c for c in critical_points if c["critical_type"] == CRITICAL_MINIMUM]
    n_basins = len(minima)

    scalars = [c["scalar"] for c in critical_points]
    loss_range = (max(scalars) - min(scalars)) if scalars else 0.0

    region_spans = [c["region_span"] for c in critical_points]
    region_sizes = [c["region_size"] for c in critical_points if c["region_size"] > 0]
    dominant_region_size = max(region_sizes) if region_sizes else 0
    dominant_region_span = max(region_spans) if region_spans else 0

    # Basin depths come from the persistence diagram's exact birth/death pairs;
    # when no diagram is available we fall back to the merge-tree scalar relief
    # (a single coarse basin) and flag the source.
    if persistence_pairs:
        persistences = [p["persistence"] for p in persistence_pairs]
        persistence_source = "persistence_diagram"
    else:
        persistences = [loss_range] if loss_range > 0 else []
        persistence_source = "merge_tree"

    persistence_total = sum(persistences)
    max_persistence = max(persistences) if persistences else 0.0
    mean_persistence = _mean(persistences)

    # dominance: share of total persistence locked in the single deepest basin.
    dominance = (
        max_persistence / persistence_total if persistence_total > 0 else 0.0
    )
    probs = (
        [p / persistence_total for p in persistences]
        if persistence_total > 0
        else []
    )
    persistence_entropy = _shannon_entropy_normalized(probs)

    # persistence_gap: how much deeper the dominant basin is than the average
    # basin -- a sharpness / concentration proxy (>>1 = one sharp dominant
    # basin; ~1 = all basins comparably shallow = flat plateau).
    persistence_gap = (
        max_persistence / mean_persistence if mean_persistence > 0 else 1.0
    )

    # Raw sharpness: depth of the dominant basin per unit of its width.
    basin_sharpness = (
        max_persistence / dominant_region_span if dominant_region_span > 0 else 0.0
    )

    if n_basins > 0 and persistence_total > 0:
        gap_score = persistence_gap / (persistence_gap + n_basins)
        narrow_basin_score = (
            0.45 * dominance
            + 0.25 * gap_score
            + 0.30 * (1.0 - persistence_entropy)
        )
        if narrow_basin_score >= 0.6:
            regime = "narrow_basin"
        elif narrow_basin_score <= 0.4:
            regime = "flat_plateau"
        else:
            regime = "mixed"
    else:
        narrow_basin_score = 0.0
        regime = "unknown"

    return {
        "n_basins": n_basins,
        "loss_range": loss_range,
        "max_persistence": max_persistence,
        "mean_persistence": mean_persistence,
        "persistence_total": persistence_total,
        "dominance": dominance,
        "persistence_entropy": persistence_entropy,
        "persistence_gap": persistence_gap,
        "dominant_region_size": dominant_region_size,
        "dominant_region_span": dominant_region_span,
        "basin_sharpness": basin_sharpness,
        "narrow_basin_score": narrow_basin_score,
        "regime": regime,
        "persistence_source": persistence_source,
    }


def descriptors_from_files(
    merge_tree_csv: str,
    persistence_diagram_csv: Optional[str] = None,
) -> Dict[str, object]:
    """Compute basin descriptors from a ``MergeTree.csv`` (+ optional diagram).

    If no persistence-diagram path is given, the sibling
    ``..._PersistenceDiagram.csv`` next to the merge tree is used when present.
    """
    if persistence_diagram_csv is None:
        candidate = merge_tree_csv.replace(
            "MergeTree.csv", "PersistenceDiagram.csv"
        )
        persistence_diagram_csv = candidate if os.path.exists(candidate) else None

    critical_points = parse_merge_tree_csv(merge_tree_csv)
    persistence_pairs = (
        parse_persistence_diagram_csv(persistence_diagram_csv)
        if persistence_diagram_csv and os.path.exists(persistence_diagram_csv)
        else None
    )
    return compute_basin_descriptors(critical_points, persistence_pairs)


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else 0.0


def _average_ranks(values: List[float]) -> List[float]:
    """1-indexed ranks with ties resolved by averaging (for Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def imbalance_correlation(records: List[Dict[str, object]]) -> Dict[str, object]:
    """Correlate basin-regime descriptors with dataset imbalance ratio.

    ``records`` is one entry per landscape: ``{"imbalance_ratio": float,
    **compute_basin_descriptors(...)}``. The paper predicts that more imbalanced
    datasets exhibit more narrow-basin topology, so a *positive* correlation
    between ``imbalance_ratio`` and ``narrow_basin_score`` (and ``dominance``)
    supports its mechanism. Returns Pearson and Spearman correlations plus the
    observed regime progression ordered by increasing imbalance.
    """
    ratios = [float(r["imbalance_ratio"]) for r in records]
    scores = [float(r.get("narrow_basin_score", 0.0)) for r in records]
    dominance = [float(r.get("dominance", 0.0)) for r in records]
    sharpness = [float(r.get("basin_sharpness", 0.0)) for r in records]

    order = sorted(range(len(ratios)), key=lambda i: ratios[i])
    progression = [
        {
            "imbalance_ratio": ratios[i],
            "regime": records[i].get("regime", "unknown"),
            "narrow_basin_score": scores[i],
        }
        for i in order
    ]

    return {
        "n_imbalance_levels": len(ratios),
        "narrow_basin_score_pearson": _pearson(ratios, scores),
        "narrow_basin_score_spearman": _spearman(ratios, scores),
        "dominance_pearson": _pearson(ratios, dominance),
        "sharpness_pearson": _pearson(ratios, sharpness),
        "regime_progression": progression,
    }
