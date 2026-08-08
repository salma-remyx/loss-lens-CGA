"""Tests for the FlowAdam (shared-scalar / gauge-equivariant Adam) wiring.

``test_choose_optimizer_wires_flowadam`` exercises the live call site -- the
``choose_optimizer`` factory -- so it proves the new optimizer is reachable
from existing code, not just self-consistent. ``test_gauge_equivariance``
asserts the paper's actual mechanism: a shared-scalar preconditioner commutes
with the gauge rotation ``(U, V) -> (U Q, V Q)`` (its step rotates
equivariantly), while a coordinate-wise one (Adam) does not.
"""

import os
import sys

# Make ``pinn.pbc_examples`` importable the same way the repo's script_util
# modules do: put ``training_scripts`` on the path. The test sits inside the
# package, so ``..`` twice reaches ``training_scripts``.
_TRAINING_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _TRAINING_SCRIPTS not in sys.path:
    sys.path.insert(0, _TRAINING_SCRIPTS)

import torch  # noqa: E402

from pinn.pbc_examples.choose_optimizer import choose_optimizer  # noqa: E402
from pinn.pbc_examples.flow_adam import FlowAdam  # noqa: E402


def test_choose_optimizer_wires_flowadam():
    """The factory returns a working FlowAdam optimizer that reduces loss."""
    torch.manual_seed(1)
    x = torch.randn(16, 4)
    y = torch.randn(16, 1)
    model = torch.nn.Linear(4, 1)

    opt = choose_optimizer("FlowAdam", model.parameters(), 1e-2)
    assert isinstance(opt, torch.optim.Optimizer)
    assert isinstance(opt, FlowAdam)

    loss_fn = torch.nn.MSELoss()
    before = loss_fn(model(x), y).item()
    for _ in range(50):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    after = loss_fn(model(x), y).item()
    assert after < before, f"FlowAdam did not reduce loss: {before} -> {after}"


def test_gauge_equivariance():
    """Shared-scalar Adam's step is rotation-equivariant; coordinate-wise Adam's is not.

    With ``betas=(0, 0)`` the rule is memoryless: the step is ``-lr * g / d``
    where ``d`` is the preconditioner. For FlowAdam ``d`` is a single scalar
    (sqrt of mean squared gradient), which is invariant under any orthogonal
    rotation of ``g`` -- so ``step(g Q) == step(g) Q`` exactly. For Adam ``d``
    is per-coordinate (sqrt of squared gradient), which does NOT commute with
    ``Q`` -- so the equality breaks. That distinction is the paper's point.
    """
    torch.manual_seed(0)
    g = torch.randn(8, 5)  # gradient on a factored (matrix) parameter
    Q, _ = torch.linalg.qr(torch.randn(5, 5))  # random orthogonal gauge rotation

    def step_of(opt_cls, grad):
        p = torch.zeros_like(grad)
        p.grad = grad
        opt = opt_cls([p], lr=1.0, betas=(0.0, 0.0))  # memoryless regime
        opt.step()
        return p.detach()  # update applied to a zero parameter

    # Equivariance holds iff the preconditioner is a scalar.
    d_flow = step_of(FlowAdam, g)
    d_flow_rot = step_of(FlowAdam, g @ Q)
    err_flow = (d_flow_rot - d_flow @ Q).abs().max()
    assert err_flow < 1e-5, f"FlowAdam broke gauge equivariance: {err_flow}"

    # Coordinate-wise Adam does not satisfy it -- the contrast the paper draws.
    d_adam = step_of(torch.optim.Adam, g)
    d_adam_rot = step_of(torch.optim.Adam, g @ Q)
    err_adam = (d_adam_rot - d_adam @ Q).abs().max()
    assert err_adam > 0.01, f"Adam unexpectedly gauge-equivariant: {err_adam}"


def test_unknown_optimizer_is_unchanged():
    """An optimizer name the factory does not know still returns None (no stray branch)."""
    assert choose_optimizer("DefinitelyNotAnOptimizer", iter([]), 1e-2) is None
