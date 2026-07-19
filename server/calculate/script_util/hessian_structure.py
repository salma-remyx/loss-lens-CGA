"""Hessian block-diagonal structure analysis.

Quantifies the near-block-diagonal structure of a neural-network Hessian and
decomposes it into a *static force* (rooted in architecture, present even at
random initialization) and a *dynamic force* (arising from training), after:

    "Towards Quantifying the Hessian Structure of Neural Networks"
    arXiv:2505.02809 (Yao et al.).

Adapted port (Mode 2). The paper estimates the Hessian spectral density with
a stochastic-Lanczos (SLQ) procedure and derives closed-form static-force
spectra for linear / 1-hidden-layer networks. Those auxiliary estimators are
substituted here by a direct, parameter-free Frobenius block-energy ratio
computable from any (sub-)Hessian matrix the caller supplies, while the
paper's core contribution -- the two-force decomposition of block-diagonal
structure -- is kept intact. Per-block spectra (``layer_block_spectrum``)
mirror the paper's block-level spectral analysis.

The Hessian of a network whose parameters are grouped into per-layer blocks
is exactly block-diagonal when inter-layer coupling vanishes.
``block_diagonal_ratio`` measures how close a given Hessian is to that ideal
(1.0 = perfectly block-diagonal). ``static_dynamic_forces`` attributes the
structure to architecture (a randomly re-initialized Hessian of the same
shape) vs training (the trained Hessian).
"""

from typing import Dict, List, Sequence

import numpy as np


def _block_bounds(block_sizes: Sequence[int]) -> List[int]:
    """Cumulative ``[0, s0, s0+s1, ...]`` offsets delimiting each block."""
    bounds = [0]
    running = 0
    for size in block_sizes:
        running += int(size)
        bounds.append(running)
    return bounds


def _validate(hessian: np.ndarray, block_sizes: Sequence[int]) -> None:
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square 2-D matrix")
    total = _block_bounds(block_sizes)[-1]
    if total != hessian.shape[0]:
        raise ValueError(
            f"block sizes sum to {total} but Hessian is "
            f"{hessian.shape[0]}x{hessian.shape[0]}"
        )


def block_diagonal_ratio(hessian, block_sizes: Sequence[int]) -> float:
    """Frobenius energy fraction held by diagonal (intra-layer) blocks.

    Returns a value in ``[0, 1]``. ``1.0`` means perfectly block-diagonal (no
    inter-layer coupling); smaller values mean stronger cross-layer coupling.
    A zero Hessian is defined as ratio ``1.0`` (vacuously block-diagonal).
    """
    H = np.asarray(hessian, dtype=float)
    _validate(H, block_sizes)
    bounds = _block_bounds(block_sizes)
    diag_energy = 0.0
    for start, end in zip(bounds[:-1], bounds[1:]):
        block = H[start:end, start:end]
        diag_energy += float(np.square(block).sum())
    total_energy = float(np.square(H).sum())
    if total_energy == 0.0:
        return 1.0
    return diag_energy / total_energy


def off_diagonal_coupling(hessian, block_sizes: Sequence[int]) -> float:
    """Frobenius energy fraction in off-diagonal (inter-layer) blocks."""
    return 1.0 - block_diagonal_ratio(hessian, block_sizes)


def layer_block_spectrum(hessian, block_sizes: Sequence[int]) -> List[List[float]]:
    """Eigenvalues of each diagonal block, sorted descending.

    Mirrors the paper's analysis of block structure through per-block spectra.
    """
    H = np.asarray(hessian, dtype=float)
    _validate(H, block_sizes)
    bounds = _block_bounds(block_sizes)
    spectra: List[List[float]] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        block = H[start:end, start:end]
        # symmetrize defensively; Hessian blocks are symmetric in theory.
        block = 0.5 * (block + block.T)
        eigvals = np.linalg.eigvalsh(block)
        spectra.append(sorted((float(v) for v in eigvals), reverse=True))
    return spectra


def static_dynamic_forces(
    hessian_trained, hessian_random, block_sizes: Sequence[int]
) -> Dict[str, float]:
    """Two-force decomposition of Hessian block-diagonal structure.

    ``static_force`` is the block-diagonal ratio at random initialization --
    the architecture's intrinsic decoupling (the "static force" of
    arXiv:2505.02809). ``dynamic_force`` is ``trained_ratio - static_ratio``,
    i.e. how much training moved the structure: positive => training made the
    Hessian *more* block-diagonal, negative => training increased inter-layer
    coupling. ``trained_ratio`` is the block-diagonal ratio of the trained
    Hessian.
    """
    trained_ratio = block_diagonal_ratio(hessian_trained, block_sizes)
    static_ratio = block_diagonal_ratio(hessian_random, block_sizes)
    return {
        "static_force": static_ratio,
        "dynamic_force": trained_ratio - static_ratio,
        "trained_ratio": trained_ratio,
    }
