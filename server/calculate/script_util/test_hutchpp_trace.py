"""Integration test for the Hutch++ Hessian trace estimator.

Imports the existing pyhessian ``hessian`` oracle (the module the
``compute_mode_hessian_trace`` wiring in ``core_functions.py`` builds its
HVP closure on) and checks the estimator against exact Hessian traces.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_dir + "/training_scripts/pinn")

import torch
from torch import nn

from pyhessian import hessian  # existing HVP-oracle module (not new)
from pyhessian.utils import hessian_vector_product
from script_util.hutchpp_trace import hutchpp_trace


def _make_hvp(hessian_comp):
    """Same closure shape as ``compute_mode_hessian_trace`` in
    core_functions.py: single-batch HVP through the pyhessian oracle."""

    def hvp(v):
        hessian_comp.model.zero_grad()
        return hessian_vector_product(
            hessian_comp.gradsH, hessian_comp.params, v
        )

    return hvp


def _exact_trace(hvp, params):
    """Exact tr(H) by applying the HVP oracle to each standard basis vector."""
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]
    n = sum(sizes)

    def flat_to_group(flat):
        group, offset = [], 0
        for shape, size in zip(shapes, sizes):
            group.append(flat[offset : offset + size].reshape(shape))
            offset += size
        return group

    trace = 0.0
    for i in range(n):
        e = torch.zeros(n)
        e[i] = 1.0
        hv = hvp(flat_to_group(e))
        flat_hv = torch.cat([h.reshape(-1) for h in hv])
        trace += flat_hv[i].item()
    return trace


def test_hutchpp_recovers_trace_of_low_rank_hessian():
    """Core Hutch++ property: when the Hessian is (numerically) low-rank,
    the sketch captures its range and the trace is recovered almost exactly,
    with the stochastic residual left at ~0."""
    torch.manual_seed(0)
    # Rank-2 data -> Hessian of the MSE has rank <= rank(X^T X) * out_dim = 4.
    basis = torch.randn(6, 2)
    mixing = torch.randn(2, 6)
    x = basis @ mixing  # rank-2 design matrix
    y = torch.randn(6, 2)

    model = nn.Linear(6, 2, bias=False)
    hessian_comp = hessian(model, nn.MSELoss(), data=(x, y), cuda=False)
    hvp = _make_hvp(hessian_comp)

    estimate = hutchpp_trace(hvp, hessian_comp.params, n_lowrank=6, n_probes=4, seed=0)
    exact = _exact_trace(hvp, hessian_comp.params)

    assert abs(exact) > 1e-6
    assert abs(estimate.trace - exact) / abs(exact) < 1e-3
    assert abs(estimate.residual_trace) / abs(exact) < 1e-3


def test_hutchpp_matches_exact_trace_on_mlp():
    """Integration with the pyhessian ``hessian`` oracle on a small MLP:
    the Hutch++ estimate lands near the exact trace, and beats a plain
    Hutchinson estimate given the same HVP budget."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 4))
    x = torch.randn(16, 6)
    y = torch.randint(0, 4, (16,))

    hessian_comp = hessian(model, nn.CrossEntropyLoss(), data=(x, y), cuda=False)
    hvp = _make_hvp(hessian_comp)

    exact = _exact_trace(hvp, hessian_comp.params)

    # Hutch++ budget: 2 * n_lowrank + n_probes = 48 HVPs.
    estimate = hutchpp_trace(hvp, hessian_comp.params, n_lowrank=16, n_probes=16, seed=0)
    hutchpp_err = abs(estimate.trace - exact) / abs(exact)
    assert hutchpp_err < 0.1

    # Plain Hutchinson with the same 48-probe budget (fixed seed -> deterministic).
    generator = torch.Generator().manual_seed(1)
    samples = []
    for _ in range(48):
        v = [torch.randint_like(p, high=2, generator=generator) for p in hessian_comp.params]
        for v_i in v:
            v_i[v_i == 0] = -1
        hv = hvp(v)
        samples.append(sum(torch.sum(a * b) for a, b in zip(hv, v)).item())
    hutchinson_err = abs(sum(samples) / len(samples) - exact) / abs(exact)

    assert hutchpp_err <= hutchinson_err
