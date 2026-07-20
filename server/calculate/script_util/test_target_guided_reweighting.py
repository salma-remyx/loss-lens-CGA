"""Integration tests for the TGSR target-guided reweighting capability.

These build a real ``PhysicsInformedNN_pbc`` from the repo's existing
``net_pbc`` module (a non-new module under ``training_scripts``) and exercise
the reweighting wiring on it -- proving the capability integrates with the
actual model class used by the PINN case study, not just self-consistency.
"""

import os
import sys

import pytest

# Make the sibling capability module and the repo's training_scripts importable.
HERE = os.path.dirname(os.path.abspath(__file__))
CALCULATE = os.path.dirname(HERE)
TRAINING_SCRIPTS = os.path.join(CALCULATE, "training_scripts")
for _path in (HERE, TRAINING_SCRIPTS):
    if _path not in sys.path:
        sys.path.append(_path)

torch = pytest.importorskip("torch")  # repo hard-depends on torch
np = pytest.importorskip("numpy")

from net_pbc import PhysicsInformedNN_pbc  # noqa: E402  (non-new module under src)

from target_guided_reweighting import (  # noqa: E402
    apply_target_guided_reweighting,
    compute_neuron_scores,
    weak_adaptation_signals,
)


def _make_pinn(layers=(2, 12, 12, 1), seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.linspace(0, 2 * np.pi, 24, endpoint=False).reshape(-1, 1)
    t = np.linspace(0, 1, x.shape[0]).reshape(-1, 1)
    X_u = np.hstack([x, t])
    u = np.sin(x)
    X_f = X_u
    bc_lb = X_u[:4]
    bc_ub = X_u[-4:]
    G = np.zeros(X_f.shape[0])
    # convection system: nu=0, beta=1, rho=0 (same PDE family as the case study).
    return PhysicsInformedNN_pbc(
        "convection", X_u, u, X_f, bc_lb, bc_ub, list(layers), G,
        0.0, 1.0, 0.0, "Adam", 1e-3, "DNN",
    )


def _hidden_row_norms(pinn):
    """Per-neuron weight-row norms for each hidden Linear layer (by position)."""
    import torch.nn as nn

    linears = [
        m for m in pinn.dnn.layers.modules() if isinstance(m, nn.Linear)
    ][:-1]
    return [m.weight.detach().norm(dim=1).clone() for m in linears]


def test_scores_and_factors_have_expected_shape_and_range():
    pinn = _make_pinn()
    scores = compute_neuron_scores(
        pinn.dnn, pinn.net_u, pinn.x_f, pinn.t_f, residual_fn=pinn.net_f
    )
    # Two hidden layers (12, 12 neurons); output layer excluded from scoring.
    assert sorted(len(v["raw"]) for v in scores.values()) == [12, 12]
    factors = weak_adaptation_signals(scores, decay=0.2, use_gmm=True)
    for f in factors.values():
        assert torch.all(f >= 0.2 - 1e-6)
        assert torch.all(f <= 1.0 + 1e-6)


def test_selective_soft_decay_preserves_output_shape_and_protects_first_layer():
    pinn = _make_pinn()
    before = _hidden_row_norms(pinn)
    out_before = pinn.net_u(pinn.x_f, pinn.t_f).detach().clone()

    returned = apply_target_guided_reweighting(
        pinn, pinn.x_f, pinn.t_f, decay=0.2, protected_layers=(0,)
    )

    after = _hidden_row_norms(pinn)
    out_after = pinn.net_u(pinn.x_f, pinn.t_f).detach()

    assert returned is pinn  # operates in place on the source model
    assert out_after.shape == out_before.shape  # model still a valid PINN

    # Protected first layer (position 0) is untouched.
    assert torch.allclose(before[0], after[0])
    # Some non-protected hidden neuron actually got soft-decayed (row shrinks),
    # so the reweighting is non-trivial.
    total_before = sum(float(b.sum()) for b in before[1:])
    total_after = sum(float(a.sum()) for a in after[1:])
    assert total_after <= total_before + 1e-6
    assert total_after < total_before


def test_rank_fallback_path_runs_without_sklearn(monkeypatch):
    import target_guided_reweighting as tgr

    pinn = _make_pinn()
    # Force the GMM path off -> exercises the rank-based fallback directly.
    monkeypatch.setattr(tgr, "_HAS_SKLEARN", False)
    returned = apply_target_guided_reweighting(
        pinn, pinn.x_f, pinn.t_f, decay=0.3, use_gmm=True
    )
    assert returned is pinn
    # Rank fallback applied real soft decay; the model still evaluates cleanly.
    assert pinn.net_u(pinn.x_f, pinn.t_f).shape[0] == pinn.x_f.shape[0]
