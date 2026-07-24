"""Integration test for the B-spline field parametrization (PI-Splines).

Exercises the contract that the ``--pi_spline`` wiring in
``compute_pinn_loss_landscape.py`` relies on: the new ``BSplineField`` drops into
the existing PINN pipeline (``PhysicsInformedNN_pbc`` from ``net_pbc``) in place of
the ``DNN``, so the parameter-perturbation / residual-loss / Hessian steps -- which
operate generically on ``model.parameters()`` -- run over the trainable control
coefficients.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("torch_optimizer")  # imported at top of net_pbc

# Make the repo's PINN module (training_scripts/) and this package importable
# without assuming the pytest invocation directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "training_scripts"))
sys.path.insert(0, _HERE)

from net_pbc import PhysicsInformedNN_pbc  # non-new module: defines the .dnn slot
from b_spline_field import BSplineField  # new module: the B-spline field


def _tiny_pinn(seed=0):
    """A minimal convection-system PINN for exercising the parametrization slot."""
    rng = np.random.RandomState(seed)
    X_u = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.3], [4.0, 0.6], [5.0, 1.0]], dtype=np.float32
    )
    u_tr = (np.sin(X_u[:, 0:1]) * 0.3).astype(np.float32)
    X_f = (rng.rand(12, 2) * np.array([2 * np.pi, 1.0])).astype(np.float32)
    bc_lb = np.hstack(
        [np.zeros((4, 1)), np.linspace(0, 1, 4)[:, None]]
    ).astype(np.float32)
    bc_ub = np.hstack(
        [np.full((4, 1), 2 * np.pi), np.linspace(0, 1, 4)[:, None]]
    ).astype(np.float32)
    G = np.zeros(X_f.shape[0], dtype=np.float32)
    return PhysicsInformedNN_pbc(
        "convection", X_u, u_tr, X_f, bc_lb, bc_ub, [2, 8, 8, 1], G,
        nu=0.0, beta=1.0, rho=0.0, optimizer_name="Adam", lr=1e-3,
        net="DNN", L=1.0, activation="tanh", loss_style="mean",
    )


def _pinn_loss(model):
    """The data + residual PINN loss, evaluated through model.dnn (the field)."""
    u_pred = model.net_u(model.x_u, model.t_u)
    f_pred = model.net_f(model.x_f, model.t_f)
    return torch.mean((model.u - u_pred) ** 2) + model.L * torch.mean(f_pred ** 2)


def test_bspline_field_drops_into_pinn_pipeline():
    """The B-spline field satisfies the DNN interface so the loss landscape is
    well defined over control coefficients."""
    model = _tiny_pinn()

    field = BSplineField((0.0, 2 * np.pi), (0.0, 1.0), 8, 3)
    model.dnn = field  # the wiring edit in compute_pinn_loss_landscape.py
    model.dnn.eval()

    # net_u / net_f must flow through the B-spline field with autograd derivatives.
    u_pred = model.net_u(model.x_u, model.t_u)
    f_pred = model.net_f(model.x_f, model.t_f)
    assert u_pred.shape == (model.x_u.shape[0], 1)
    assert f_pred.shape == (model.x_f.shape[0], 1)
    assert u_pred.requires_grad and f_pred.requires_grad

    # model.dnn.parameters() are exactly the trainable control coefficients.
    params = list(model.dnn.parameters())
    assert len(params) == 1
    assert params[0] is field.C
    assert field.C.shape == (8, 8)
    assert field.n_params() == 64

    # The PINN loss must respond to a control-coefficient perturbation -- i.e. the
    # landscape machinery samples a non-degenerate surface over control coefficients.
    loss_before = _pinn_loss(model).item()
    with torch.no_grad():
        field.C.add_(1.0)
    loss_after = _pinn_loss(model).item()
    assert loss_after != pytest.approx(loss_before)


def test_bspline_field_fits_known_solution():
    """Trainable control coefficients can represent a smooth reference solution
    (the analytical solution the PINN case study loads)."""
    xs = np.linspace(0, 2 * np.pi, 40)
    ts = np.linspace(0, 1, 20)
    Xg, Tg = np.meshgrid(xs, ts)
    X_star = np.hstack([Xg.flatten()[:, None], Tg.flatten()[:, None]])
    u_star = np.sin(X_star[:, 0]) * np.exp(-X_star[:, 1])  # reference field

    field = BSplineField((0.0, 2 * np.pi), (0.0, 1.0), 24, 3)
    field.fit(X_star, u_star)
    with torch.no_grad():
        pred = field(torch.tensor(X_star, dtype=torch.float32)).numpy().flatten()
    err = np.abs(pred - u_star)
    assert err.mean() < 1e-3
