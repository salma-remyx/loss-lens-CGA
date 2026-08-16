"""Tests for the KronPSGD entry in the optimizer factory."""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from choose_optimizer import choose_optimizer


def _ill_conditioned_quadratic():
    """A problem whose Hessian has a 1e4 condition number."""
    hessian = torch.diag(torch.tensor([100.0, 1.0, 0.1, 0.01]))
    target = torch.ones(4, 1)
    return hessian, target


def test_factory_returns_kron_optimizer():
    hessian, target = _ill_conditioned_quadratic()
    param = torch.zeros(4, 1, requires_grad=True)
    optimizer = choose_optimizer("KronPSGD", [param], 0.02)

    # the factory must hand back the Kron-factored preconditioner, not the
    # None that an unknown name produces
    assert optimizer is not None
    assert type(optimizer).__name__ == "KronPSGD"
    assert optimizer.param_groups[0]["lr"] == 0.02


def test_factory_still_returns_existing_choices():
    param = torch.zeros(4, 1, requires_grad=True)
    for name in ("SGD", "Adam"):
        optimizer = choose_optimizer(name, [param], 0.02)
        assert optimizer is not None, name


def test_kron_beats_first_order_methods_on_ill_conditioned_problem():
    """Within a fixed step budget, the curvature fit closes the gap that a
    first-order method has to walk out one stiff direction at a time."""
    torch.manual_seed(0)
    hessian, target = _ill_conditioned_quadratic()
    optimum = torch.linalg.solve(hessian, target)

    errors = {}
    for name in ("SGD", "Adam", "KronPSGD"):
        param = torch.zeros(4, 1, requires_grad=True)
        optimizer = choose_optimizer(name, [param], 0.02)

        def closure():
            optimizer.zero_grad()
            loss = (0.5 * (param.t() @ hessian @ param) - target.t() @ param).squeeze()
            # the Kron fit probes curvature, so it needs the gradient graph
            loss.backward(create_graph=name == "KronPSGD")
            return loss

        for _ in range(2000):
            loss = optimizer.step(closure)
        errors[name] = (param - optimum).norm().item()

    assert torch.isfinite(torch.tensor(errors["KronPSGD"]))
    # the preconditioner recovers the curvature that plain SGD cannot see
    assert errors["KronPSGD"] < errors["SGD"] / 100.0
    assert errors["KronPSGD"] < errors["Adam"]


def test_preconditioner_conditions_the_hessian():
    """The fitted P should shrink cond(H) by an order of magnitude."""
    from kron_preconditioner import balance_kron_factors, woodfulk_kron_update

    torch.manual_seed(0)
    hessian, _ = _ill_conditioned_quadratic()
    factors = (torch.eye(4), torch.eye(1))

    for i in range(2000):
        probe = torch.randn(4, 1)
        factors = woodfulk_kron_update(*factors, probe, hessian @ probe)
        if i % 10 == 0:
            factors = balance_kron_factors(factors)

    left, right = factors
    preconditioned = left.t() @ left @ hessian
    fitted = torch.linalg.svdvals(preconditioned)
    raw = torch.linalg.svdvals(hessian)
    assert (fitted[0] / fitted[-1]) < (raw[0] / raw[-1]) / 10.0


def test_non_matrix_parameters_are_handled():
    """Biases and other 1D parameters take the scalar fit, not the Kron fit."""
    torch.manual_seed(0)
    weight = torch.zeros(4, 3, requires_grad=True)
    bias = torch.zeros(4, requires_grad=True)
    optimizer = choose_optimizer("KronPSGD", [weight, bias], 0.01)

    data = torch.rand(8, 3)
    target = torch.rand(8, 4)

    def closure():
        optimizer.zero_grad()
        loss = ((data @ weight.t() + bias - target) ** 2).mean()
        loss.backward(create_graph=True)
        return loss

    for _ in range(20):
        loss = optimizer.step(closure)

    assert torch.isfinite(loss)
    assert all(p.isfinite().all() for p in (weight, bias))
