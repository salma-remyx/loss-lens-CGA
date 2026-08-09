"""Integration test for the CL-PINN continual-learning wiring.

Imports the repo's existing modules (``net_pbc``, ``systems_pbc``) and drives
``continual_param_pinn.run_continual`` -- the function that ``main_pbc.py``'s
``--continual`` hook delegates to -- end-to-end on a tiny config.
"""

import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from net_pbc import DNN  # existing module (non-new)
import systems_pbc  # existing module (non-new)
from continual_param_pinn import run_continual  # new module


def _tiny_args(**overrides):
    base = dict(
        system="convection",
        param="",  # inferred -> beta
        param_min=0.5,
        param_max=3.0,
        n_tasks=3,
        steps_per_task=160,
        replay_size=40,
        active_selection=True,
        layers="32,32,1",
        activation="tanh",
        xgrid=32,
        nt=20,
        N_f=80,
        source=0.0,
        u0_str="sin(x)",
        lr=1e-2,
        L=1.0,
        seed=0,
        save_model=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_continual_uses_repo_dnn_and_systems_pbc():
    args = _tiny_args()
    model = run_continual(args)

    # Integration: the shared model is the repo's own net_pbc.DNN.
    assert isinstance(model.dnn, DNN)
    # systems_pbc is exercised (ground-truth family solvers); sanity-check import.
    assert hasattr(systems_pbc, "convection_diffusion")


def test_continual_reduces_error_across_domain():
    args = _tiny_args()
    model = run_continual(args)

    init_mean = float(np.mean(list(model.initial_errors.values())))
    final_mean = float(np.mean(list(model.final_errors.values())))
    # Sequential training + replay reduced mean relative error across the domain.
    assert final_mean < init_mean

    # Every candidate parameter was visited (the task pool fully drained).
    assert len(model.task_order) == args.n_tasks
    # Error-greedy selection picked the worst-initial parameter first.
    worst_init = max(model.initial_errors, key=model.initial_errors.get)
    assert abs(model.task_order[0] - worst_init) < 1e-6


def test_model_is_parameter_conditioned():
    """A single (x,t) maps to different u for different mu -> a family, not one fn."""
    args = _tiny_args()
    model = run_continual(args)

    x = np.linspace(0, 2 * np.pi, args.xgrid, endpoint=False).reshape(-1, 1)
    t = np.zeros_like(x)
    X = np.hstack((x, t))
    u_lo = model.predict(X, args.param_min).ravel()
    u_hi = model.predict(X, args.param_max).ravel()
    assert not np.allclose(u_lo, u_hi)
