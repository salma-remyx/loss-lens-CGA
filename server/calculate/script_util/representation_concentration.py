"""Representation-change concentration analysis for the Layer-Similarity (CKA) pillar.

This module operationalises the core finding of:

    "Gradient Concentration, Not Weight Saliency, Explains Representation-Level
     Class Unlearning" (arXiv:2607.21353v1)

that *representation-level change is governed by concentration and geometry,
not by the identity of selected weights* -- in the studied setting the change
is strongly concentrated in a few layers (~92% of the squared gradient energy
in the final layers), so *where* the change lives matters more than *which*
weights carry it.

LossLens compares two models through per-layer CKA (``compute_layer_similarity``
in ``core_functions.py``), so this module answers the paper's question on the
repo's native surface: given a per-layer CKA matrix, *where across depth is the
representation change concentrated?*

This is an **adapted port (Mode 2)** -- the paper's core mechanism is kept at
full fidelity while two auxiliary components are replaced with target-native
equivalents:

* The paper measures concentration with the **squared energy of the forget
  gradient** per layer. LossLens never computes a forget/retain gradient split;
  it compares two models via activations. We substitute a parameter-free proxy
  of the same signal: the **representation-distance energy ``1 - CKA``** per
  layer (the paper itself uses layer-wise CKA as its primary representation
  evaluation). ``concentration_metrics`` reports the fraction of that energy in
  the final layers -- the direct analog of the paper's "92% in the final
  layers" -- plus a layer-count-independent entropy concentration index.

* The paper's **prototype-recovery** metric needs CIFAR class labels and a
  forget/retain split. LossLens's flagship use case (the PINN case study) has
  neither. We substitute a **label-free prototype-separation** metric: per
  layer, how far apart are the two models' feature centroids relative to their
  pooled internal spread. Same geometry-first question, target-native channel.

Only numpy is required -- the module never imports torch, so it can run on the
plain CKA matrix that ``compute_layer_similarity`` already returns.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np

__all__ = [
    "representation_change_profile",
    "concentration_metrics",
    "concentration_summary",
    "prototype_separation",
    "prototype_separation_by_layer",
]

# A CKA matrix: rows index model1 layers, columns index model2 layers.
LayerSimilarity = Union[np.ndarray, Sequence[Sequence[float]]]


def _as_matrix(layer_similarity: LayerSimilarity) -> np.ndarray:
    """Coerce a (possibly nested-list / tensor-backed) CKA matrix to float ndarray."""
    return np.asarray(layer_similarity, dtype=float)


def _to_numpy(x) -> np.ndarray:
    """Best-effort conversion of a tensor or array to a float numpy array.

    Duck-typed so the module stays torch-free: torch tensors expose
    ``.detach()``/``.cpu()``/``.numpy()``, plain arrays fall through to
    ``np.asarray``.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=float)


def representation_change_profile(
    layer_similarity: LayerSimilarity,
    layer_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Per-layer representation-change energy derived from a CKA matrix.

    CKA is 1.0 for identical representations and 0.0 for orthogonal ones, so
    the representation *distance* per layer-pair is ``1 - CKA``. Marginalising
    that distance over the other axis yields, for each layer of a model, how
    much its representations diverge from the other model -- the layer-localised
    change-energy profile.

    :param layer_similarity: CKA matrix with model1 layers as rows and model2
        layers as columns.
    :param layer_names: optional names for the model1 (row) layers; used only to
        label the returned profile.
    :return: dict with per-layer change energy along model1 (rows) and model2
        (columns), plus layer names when supplied.
    """
    matrix = _as_matrix(layer_similarity)
    distance = 1.0 - np.clip(matrix, 0.0, 1.0)

    # Row/column marginals: mean representation distance per layer of each model.
    model1_energy = distance.mean(axis=1)
    model2_energy = distance.mean(axis=0)

    profile: Dict[str, object] = {
        "model1_change_energy": model1_energy.tolist(),
        "model2_change_energy": model2_energy.tolist(),
        "total_change_energy": float(model1_energy.sum()),
    }
    if layer_names is not None:
        profile["layer_names"] = list(layer_names)
    return profile


def concentration_metrics(
    layer_similarity: LayerSimilarity,
    final_layers: Optional[int] = None,
) -> Dict[str, object]:
    """Concentration summary for a per-layer CKA matrix.

    Captures the paper's headline claim -- that representation change is
    *concentrated* rather than spread evenly -- with two complementary numbers:

    * ``final_layer_fraction``: fraction of the total model1 (row) change energy
      located in the last ``final_layers`` layers. This is the direct analog of
      the paper's "fraction of squared gradient energy in the final layers".
    * ``concentration_index``: ``1 - H/H_max`` over the normalised per-layer
      energy, where ``H`` is Shannon entropy. It is 0 for a perfectly uniform
      change profile and 1 when all change concentrates in a single layer,
      independent of *where* that layer sits (robust when the peak is not at the
      end, e.g. mid-network in a PINN).

    :param layer_similarity: CKA matrix (model1 layers as rows).
    :param final_layers: how many trailing layers count as "final". Defaults to
        the last quarter of the model1 layers (at least one).
    """
    matrix = _as_matrix(layer_similarity)
    distance = 1.0 - np.clip(matrix, 0.0, 1.0)
    energy = distance.mean(axis=1)  # per model1 (row) layer
    n_layers = int(energy.shape[0])

    if final_layers is None:
        final_layers = max(1, n_layers // 4)
    final_layers = int(min(final_layers, n_layers))

    total_energy = float(energy.sum())
    if total_energy > 0.0:
        profile = energy / total_energy
        entropy = float(-np.sum(profile[profile > 0] * np.log(profile[profile > 0])))
        max_entropy = float(np.log(n_layers)) if n_layers > 1 else 1.0
        concentration_index = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)
        final_fraction = float(energy[-final_layers:].sum() / total_energy) if final_layers else 0.0
        dominant_index = int(np.argmax(energy))
        dominant_fraction = float(energy[dominant_index] / total_energy)
    else:
        # Identical representations across every layer: no change to concentrate.
        concentration_index = 0.0
        final_fraction = 0.0
        dominant_index = 0
        dominant_fraction = 0.0

    return {
        "layer_count": n_layers,
        "final_layers_used": final_layers,
        "total_change_energy": total_energy,
        "concentration_index": concentration_index,
        "final_layer_fraction": final_fraction,
        "dominant_layer_index": dominant_index,
        "dominant_layer_energy_fraction": dominant_fraction,
    }


def concentration_summary(
    layer_similarity: LayerSimilarity,
    layer_names: Optional[Sequence[str]] = None,
    final_layers: Optional[int] = None,
    prototype_separation: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """JSON-serialisable concentration summary for storage / the UI.

    Combines :func:`representation_change_profile` and
    :func:`concentration_metrics`, rounding floats for clean persistence, and
    optionally attaches the per-layer :func:`prototype_separation_by_layer`
    result.
    """
    profile = representation_change_profile(layer_similarity, layer_names)
    metrics = concentration_metrics(layer_similarity, final_layers)

    summary: Dict[str, object] = {
        "profile": {k: v for k, v in profile.items() if k != "layer_names"},
        "metrics": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                    for k, v in metrics.items()},
        "layer_names": profile.get("layer_names", []),
    }
    if prototype_separation:
        summary["prototype_separation"] = {
            k: round(float(v), 6) for k, v in prototype_separation.items()
        }
    return summary


def prototype_separation(act_a: np.ndarray, act_b: np.ndarray) -> Dict[str, float]:
    """Label-free prototype-separation between two same-input activation sets.

    Adapted (Mode 2) stand-in for the paper's class-prototype *recovery* metric:
    instead of asking whether a class prototype survives forgetting, we ask how
    far apart the two models' representation *centroids* sit relative to their
    pooled internal spread. Larger values mean the two models diverge more at
    that representation -- exactly the geometry-first signal the paper argues
    governs representation-level change.

    :param act_a: activations of model1 for ``N`` shared inputs, shape
        ``[N, D]`` (any extra dimensions are flattened).
    :param act_b: activations of model2 for the same ``N`` inputs, shape
        ``[N, D]``.
    :return: dict with the centroid distance (``between_centroid_distance``),
        the pooled per-feature spread (``within_spread``) and their ratio
        (``separation``).
    """
    a = _to_numpy(act_a).reshape(_to_numpy(act_a).shape[0], -1)
    b = _to_numpy(act_b).reshape(_to_numpy(act_b).shape[0], -1)

    centroid_a = a.mean(axis=0)
    centroid_b = b.mean(axis=0)
    between = float(np.linalg.norm(centroid_a - centroid_b))

    # Pooled within-centroid spread: average per-feature std across both models.
    within = float(0.5 * (a.std(axis=0).mean() + b.std(axis=0).mean()))
    eps = 1e-12
    separation = between / (within + eps)

    return {
        "between_centroid_distance": round(between, 6),
        "within_spread": round(within, 6),
        "separation": round(separation, 6),
    }


def prototype_separation_by_layer(
    features_a: Dict[str, object],
    features_b: Dict[str, object],
) -> Dict[str, float]:
    """Per-layer prototype separation from two CKA-engine feature dicts.

    ``features_a`` / ``features_b`` map layer name -> activations ``[N, D]`` for
    the two models on the same inputs (e.g. the ``cka.model1_features`` /
    ``cka.model2_features`` dicts the existing CKA engines populate during
    ``compare()``). Only layers present in both are compared; the per-layer
    ``separation`` ratio is returned.

    :return: dict mapping layer name to its separation ratio.
    """
    common = [name for name in features_a if name in features_b]
    return {name: prototype_separation(features_a[name], features_b[name])["separation"]
            for name in common}
