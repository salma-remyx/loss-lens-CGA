"""Integration tests for the SAM / LateSAM optimizer wiring.

These import the existing :mod:`choose_optimizer` factory (the non-new call
site edited for this change) and exercise the two new branches through the
closure-based ``step(closure)`` interface that ``PhysicsInformedNN_pbc``
(``net_pbc.py``) drives.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports from a NON-NEW module: this is the call site edited in this change.
from choose_optimizer import choose_optimizer
from sharpness_aware_optimizer import SAMOptimizer, LatePhaseSAM


def _model_and_closure():
    torch.manual_seed(0)
    model = nn.Linear(4, 1, bias=False)
    x = torch.randn(16, 4)
    target = torch.randn(16, 1)

    def make_closure(opt):
        def closure():
            opt.zero_grad()
            loss = ((model(x) - target) ** 2).sum()
            loss.backward()
            return loss
        return closure

    return model, make_closure


def test_sam_selectable_via_factory_and_reduces_loss():
    """choose_optimizer('SAM', ...) returns a working SAMOptimizer that descends."""
    model, make_closure = _model_and_closure()
    opt = choose_optimizer('SAM', model.parameters(), 1e-2)
    assert isinstance(opt, SAMOptimizer)

    closure = make_closure(opt)
    first = opt.step(closure).item()
    for _ in range(9):
        last = opt.step(closure).item()
    assert last < first, "SAM should reduce the loss over a few steps"


def test_sam_step_calls_closure_twice():
    """A single SAM update evaluates the closure once for ascent and once for descent."""
    model, make_closure = _model_and_closure()
    opt = choose_optimizer('SAM', model.parameters(), 1e-2)
    closure = make_closure(opt)

    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return closure()

    opt.step(counting)
    assert calls["n"] == 2


def test_late_phase_sam_switches_to_sam_in_final_phase():
    """LateSAM runs the plain base early and SAM only in the configured final phase.

    With total=5 / late=2 the switch happens after step 3. Plain SGD calls the
    closure once per step; SAM calls it twice (ascent + descent), so the
    per-step marginal closure-call pattern is [1, 1, 1, 2, 2].
    """
    os.environ["PINN_SAM_TOTAL_ITERS"] = "5"
    os.environ["PINN_SAM_LATE_ITERS"] = "2"
    os.environ["PINN_SAM_RHO"] = "0.1"
    try:
        model, make_closure = _model_and_closure()
        opt = choose_optimizer('LateSAM', model.parameters(), 1e-2)
        assert isinstance(opt, LatePhaseSAM)
        assert opt.total_iters == 5
        assert opt.late_iters == 2

        closure = make_closure(opt)
        marginals = []
        for _ in range(opt.total_iters):
            calls = {"n": 0}

            def counting():
                calls["n"] += 1
                return closure()

            opt.step(counting)
            marginals.append(calls["n"])

        assert marginals == [1, 1, 1, 2, 2], marginals
    finally:
        os.environ.pop("PINN_SAM_TOTAL_ITERS", None)
        os.environ.pop("PINN_SAM_LATE_ITERS", None)
        os.environ.pop("PINN_SAM_RHO", None)
