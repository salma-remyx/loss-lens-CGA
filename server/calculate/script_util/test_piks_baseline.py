"""Integration tests for the PIKS closed-form baseline.

These import the repo's existing PDE solution library ``systems_pbc`` -- the
same module ``compute_pinn_loss_landscape.py`` consumes via
``from systems_pbc import *`` -- and exercise the ``--piks`` branch wired into
that script through ``piks_solver.run_piks_baseline``. The convection problem
is the repo's default ``--system`` and is linear, so it is PIKS-eligible.
"""

import argparse
import os
import sys

import numpy as np

# The repo spreads these scripts across two directories; put both on the path
# so the test is importable regardless of pytest's rootdir.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(
    0,
    os.path.join(_HERE, "..", "training_scripts", "pinn", "pbc_examples"),
)

import systems_pbc
from piks_solver import run_piks_baseline


def _build_data(system, nu, beta, rho, source=0.0, xgrid=64, nt=33, N_f=120,
                seed=0):
    """Replicate the data assembly in compute_pinn_loss_landscape.py."""
    x = np.linspace(0, 2 * np.pi, xgrid, endpoint=False).reshape(-1, 1)
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    x_noboundary, t_noinitial = x[1:], t[1:]
    Xn, Tn = np.meshgrid(x_noboundary, t_noinitial)
    interior = np.hstack((Xn.flatten()[:, None], Tn.flatten()[:, None]))
    rng = np.random.RandomState(seed)
    X_f_train = interior[rng.choice(interior.shape[0], size=N_f, replace=False)]

    u_vals = systems_pbc.convection_diffusion(
        "sin(x)", nu, beta, source, xgrid, nt
    )
    u_star = u_vals.reshape(-1, 1)
    Exact = u_star.reshape(nt, xgrid)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    bc_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    bc_ub = np.hstack((np.full((nt, 1), 2 * np.pi), t))
    G = np.full(X_f_train.shape[0], float(source))
    return (
        {
            "X_star": X_star, "u_star": u_star, "X_f_train": X_f_train, "G": G,
            "xx1": xx1, "uu1": uu1, "bc_lb": bc_lb, "bc_ub": bc_ub, "uu2": uu2,
            "nu": nu, "beta": beta, "rho": rho,
        },
        argparse.Namespace(
            system=system, u0_str="sin(x)", N_f=N_f,
            piks_sigma=1.0, piks_ridge=1e-6,
        ),
    )


def test_piks_recovers_convection_solution():
    """PIKS solves the linear convection PDE in closed form and tracks sin(x-beta t)."""
    data, args = _build_data("convection", nu=0.0, beta=1.0, rho=0.0)
    out = run_piks_baseline(save=False, args=args, **data)
    assert out["u_pred"].shape == data["u_star"].shape
    # Far better than a trivial zero guess (norm-ratio ~ 1) -- the closed-form
    # kernel estimator recovers the analytic solution to ~1e-5.
    assert out["l2_rel"] < 1e-2, out["l2_rel"]


def test_piks_recovers_diffusion_solution():
    """The second-derivative operator image (nu != 0) is also correct."""
    data, args = _build_data("diffusion", nu=0.1, beta=0.0, rho=0.0, source=0.5)
    out = run_piks_baseline(save=False, args=args, **data)
    assert out["l2_rel"] < 1e-2, out["l2_rel"]


def test_piks_rejects_nonlinear_pde():
    """Reaction (rho != 0) is nonlinear and outside PIKS's linear guarantees."""
    data, args = _build_data("reaction", nu=0.0, beta=0.0, rho=1.0)
    try:
        run_piks_baseline(save=False, args=args, **data)
    except ValueError as exc:
        assert "rho" in str(exc)
        return
    raise AssertionError("PIKS should refuse the nonlinear reaction PDE")
