"""Low-variance stochastic Hessian trace estimation via Hutch++.

Adapted from "Optimal Stochastic Trace Estimation in Generative Modeling"
(arXiv:2502.18808), which applies the Hutch++ estimator to trace queries
where Hutchinson's estimator suffers from high variance. Hutch++ splits the
trace into an exactly-computed low-rank component (from a handful of
matrix-vector products) plus a Hutchinson estimate over the orthogonal
residual, sharply reducing variance for ill-conditioned matrices with
decaying spectra -- the regime of neural-network loss Hessians.

Only the trace-estimation core is ported. It runs against the repo's
existing Hessian-vector-product oracle contract: vectors are lists of
parameter-shaped tensors, as used by pyhessian's ``hessian`` /
``hessian_pinn`` classes (see ``compute_mode_hessian_trace`` in
``core_functions.py`` for the wiring). The paper's diffusion-model training
loop and divergence-based likelihood objective are intentionally not ported.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

# A "group vector" in the pyhessian sense: one tensor per model parameter.
GroupVector = List[torch.Tensor]

# Hessian-vector-product oracle: maps a group vector to H @ v.
HVPOracle = Callable[[GroupVector], GroupVector]


@dataclass
class TraceEstimate:
    """Result of a Hutch++ trace estimate.

    ``trace`` is the headline estimate (``low_rank_trace + residual_trace``);
    the components are exposed so callers can see how much of the curvature
    mass was captured by the deterministic low-rank part versus the
    stochastic residual. ``num_matvecs`` is the total HVP budget spent.
    """

    trace: float
    low_rank_trace: float
    residual_trace: float
    num_matvecs: int


def _rademacher_like(
    params: GroupVector, generator: Optional[torch.Generator] = None
) -> GroupVector:
    """Rademacher (+1/-1) probe with the same shapes as ``params``.

    Mirrors the probe construction in pyhessian's ``trace()`` so estimates
    are directly comparable with the existing Hutchinson path.
    """
    v = [torch.randint_like(p, high=2, generator=generator) for p in params]
    for v_i in v:
        v_i[v_i == 0] = -1
    return v


def _dot(xs: GroupVector, ys: GroupVector) -> float:
    """Inner product of two group vectors (pyhessian's group_product)."""
    return sum(torch.sum(x * y) for (x, y) in zip(xs, ys)).item()


def _axpy(xs: GroupVector, ys: GroupVector, alpha: float) -> GroupVector:
    """Return ``xs + alpha * ys`` as a new group vector."""
    return [x + alpha * y for (x, y) in zip(xs, ys)]


def _orthonormalize(vectors: List[GroupVector]) -> List[GroupVector]:
    """Modified Gram-Schmidt (with one reorthogonalization pass) over group
    vectors; numerically dependent directions are dropped."""
    basis: List[GroupVector] = []
    for v in vectors:
        for _ in range(2):
            for q in basis:
                v = _axpy(v, q, -_dot(v, q))
        norm = math.sqrt(max(_dot(v, v), 0.0))
        if norm < 1e-12:
            continue
        basis.append([v_i / norm for v_i in v])
    return basis


def hutchpp_trace(
    hvp: HVPOracle,
    params: GroupVector,
    n_lowrank: int = 8,
    n_probes: Optional[int] = None,
    seed: Optional[int] = None,
) -> TraceEstimate:
    """Estimate tr(H) for a symmetric matrix given only an HVP oracle.

    Hutch++ (Meyer et al. 2021, applied to generative-model trace queries in
    arXiv:2502.18808):

      1. Sketch: apply H to ``n_lowrank`` Rademacher vectors and
         orthonormalize -> Q spanning (approximately) the dominant eigenspace.
      2. Low-rank term: tr(Q^T H Q), computed exactly with one HVP per
         basis vector.
      3. Residual term: Hutchinson estimate of tr(H - Q Q^T H Q) using
         ``n_probes`` Rademacher probes projected orthogonal to Q.

    Total cost is ``2 * n_lowrank + n_probes`` HVPs (fewer if the sketch is
    rank-deficient). For Hessians whose spectrum decays quickly, step 2
    captures most of the trace deterministically, leaving a small-variance
    residual -- this is the variance reduction the paper reports over plain
    Hutchinson.

    ``n_probes`` defaults to ``n_lowrank``; pass ``seed`` for a
    reproducible estimate.
    """
    if n_lowrank < 0:
        raise ValueError("n_lowrank must be >= 0")
    if n_probes is None:
        n_probes = n_lowrank
    if n_probes < 1:
        raise ValueError("n_probes must be >= 1")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=params[0].device)
        generator.manual_seed(seed)

    num_matvecs = 0

    # Steps 1-2: sketch + exact low-rank trace.
    basis: List[GroupVector] = []
    low_rank_trace = 0.0
    if n_lowrank > 0:
        sketched = []
        for _ in range(n_lowrank):
            sketched.append(hvp(_rademacher_like(params, generator)))
            num_matvecs += 1
        basis = _orthonormalize(sketched)
        for q in basis:
            hq = hvp(q)
            num_matvecs += 1
            low_rank_trace += _dot(q, hq)

    # Step 3: Hutchinson estimate over the residual (I - Q Q^T) H (I - Q Q^T).
    residual_samples = []
    for _ in range(n_probes):
        g = _rademacher_like(params, generator)
        for q in basis:
            g = _axpy(g, q, -_dot(g, q))
        hg = hvp(g)
        num_matvecs += 1
        residual_samples.append(_dot(g, hg))
    residual_trace = sum(residual_samples) / len(residual_samples)

    return TraceEstimate(
        trace=low_rank_trace + residual_trace,
        low_rank_trace=low_rank_trace,
        residual_trace=residual_trace,
        num_matvecs=num_matvecs,
    )
