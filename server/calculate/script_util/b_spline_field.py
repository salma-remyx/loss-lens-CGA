"""Tensor-product B-spline field parametrization with trainable control coefficients.

This module implements the core approximation architecture of *Physics-Informed
Splines (PI-Splines)* ("Trainable Spline Representations for Physics-Informed
Learning", arXiv:2607.15751): instead of representing the unknown solution of a
differential equation with a neural network, the field is parametrized directly
through a tensor-product B-spline expansion

    u(x, t) = sum_{i, j} C[i, j] * B_i(x) * B_j(t)

whose control coefficients ``C`` are the trainable parameters. This gives compact
support, explicit smoothness control, and a direct geometric interpretation of the
parameters, in contrast to a dense feed-forward network.

Adaptation note (Mode 2, adapted port)
---------------------------------------
Only the *approximation architecture* is ported here. The auxiliary pieces of the
paper are intentionally target-native:

* Derivatives of ``u`` are obtained through ``torch.autograd`` (the existing
  ``PhysicsInformedNN_pbc.net_f`` path already differentiates ``model.dnn``
  symbolically) rather than via a hand-coded analytical B-spline derivative basis.
  The B-spline basis is piecewise polynomial and autograd-recoverable, so the
  residual / Hessian computations are exact within each knot span.
* No training loop or benchmark suite is ported. The field is a drop-in
  ``torch.nn.Module`` exposing ``parameters()`` (the control coefficients), so it
  slots into ``compute_pinn_loss_landscape.py`` in place of the ``DNN``: the
  parameter-perturbation, Hessian, and residual-loss steps there operate generically
  on ``model.parameters()`` and run unchanged, now over control coefficients.

The deliverable for LossLens is therefore a second, structured field parametrization
whose loss landscape can be sampled and compared against the existing PINN (DNN)
landscapes -- precisely the comparative study the paper motivates.
"""

import numpy as np
import torch
import torch.nn as nn


def _clamped_uniform_knots(lo, hi, n_ctrl, degree):
    """Clamped (open) uniform knot vector over ``[lo, hi]``.

    The first ``degree + 1`` knots equal ``lo`` and the last ``degree + 1`` equal
    ``hi``, so the spline interpolates the boundary control coefficients -- the
    mechanism the paper uses to impose boundary conditions strongly.
    """
    lo = float(lo)
    hi = float(hi)
    n_interior = n_ctrl - degree - 1
    if n_interior > 0:
        interior = torch.linspace(lo, hi, n_interior + 2)[1:-1]
    else:
        interior = torch.empty(0, dtype=torch.get_default_dtype())
    ends = torch.full((degree + 1,), lo)
    knots = torch.cat(
        [ends, interior.to(ends.dtype), torch.full((degree + 1,), hi)]
    )
    return knots


def _bspline_basis_1d(x, knots, degree):
    """Evaluate the 1-D B-spline basis at ``x`` via the Cox-de Boor recursion.

    Parameters
    ----------
    x : torch.Tensor, shape ``(N,)``
        Query points.
    knots : torch.Tensor, shape ``(K,)``
        Knot vector (non-decreasing).
    degree : int
        Spline degree ``p``; ``n_ctrl = K - degree - 1`` basis functions result.

    Returns
    -------
    torch.Tensor, shape ``(N, n_ctrl)`` -- the basis matrix, differentiable in
    ``x`` (so autograd can recover derivatives of the field).
    """
    x = x.reshape(-1)
    knots = knots.reshape(-1)
    K = knots.shape[0]
    n_ctrl = K - degree - 1

    # Last knot span with positive width -- the global upper endpoint belongs here
    # (so x == hi evaluates to the final control point, matching clamped behaviour).
    widths = knots[1:] - knots[:-1]
    nondeg = torch.nonzero(widths > 0).reshape(-1)
    last_span = int(nondeg[-1])

    left = knots[:-1].unsqueeze(0)  # (1, K-1)
    right = knots[1:].unsqueeze(0)  # (1, K-1)
    xu = x.unsqueeze(1)  # (N, 1)

    # degree-0 indicator functions; piecewise constant (no gradient w.r.t. x).
    mask = (xu >= left) & (xu < right)
    # Make the upper boundary inclusive on the last non-degenerate span.
    boundary = torch.zeros_like(mask)
    boundary[:, last_span] = (xu[:, 0] >= left[0, last_span]) & (
        xu[:, 0] <= right[0, last_span]
    )
    basis = (mask | boundary).to(knots.dtype)

    for d in range(1, degree + 1):
        m = (K - 1) - d  # number of degree-d basis functions
        denom_l = knots[d:K - 1] - knots[0:K - 1 - d]
        denom_r = knots[d + 1:K] - knots[1:K - d]
        # Guard the degenerate (zero-width) spans; their basis values are zero
        # there, so the 0/0 is replaced by a harmless 0/1.
        denom_l = torch.where(denom_l == 0, torch.ones_like(denom_l), denom_l)
        denom_r = torch.where(denom_r == 0, torch.ones_like(denom_r), denom_r)

        term_l = (xu[:, :m] - knots[:m]) / denom_l.unsqueeze(0) * basis[:, :m]
        term_r = (
            (knots[d + 1:d + 1 + m] - xu[:, :m]) / denom_r.unsqueeze(0)
            * basis[:, 1:m + 1]
        )
        basis = term_l + term_r

    return basis  # (N, n_ctrl)


class BSplineField(nn.Module):
    """A 2-D field parametrized as a tensor-product B-spline.

    The field maps ``(x, t)`` -> ``u`` and is differentiable in its inputs, so it
    is a drop-in replacement for the feed-forward ``DNN`` used as ``model.dnn`` in
    the PINN pipeline: its ``parameters()`` are the trainable control coefficients
    ``C`` of shape ``(n_ctrl_x, n_ctrl_t)``.

    Parameters
    ----------
    x_range, t_range : tuple(float, float)
        Domain bounds along each axis.
    n_ctrl_per_axis : int
        Number of control points (basis functions) along each axis.
    degree : int
        B-spline degree (same on both axes).
    """

    def __init__(self, x_range, t_range, n_ctrl_per_axis, degree=3):
        super().__init__()
        if n_ctrl_per_axis < degree + 1:
            raise ValueError(
                f"need at least degree+1={degree + 1} control points per axis, "
                f"got {n_ctrl_per_axis}"
            )
        self.x_range = x_range
        self.t_range = t_range
        self.n_ctrl_x = n_ctrl_per_axis
        self.n_ctrl_t = n_ctrl_per_axis
        self.degree = int(degree)

        knots_x = _clamped_uniform_knots(x_range[0], x_range[1], self.n_ctrl_x, self.degree)
        knots_t = _clamped_uniform_knots(t_range[0], t_range[1], self.n_ctrl_t, self.degree)
        self.register_buffer("knots_x", knots_x)
        self.register_buffer("knots_t", knots_t)

        # Trainable control coefficients -- the PI-Spline parameters.
        self.C = nn.Parameter(torch.zeros(self.n_ctrl_x, self.n_ctrl_t))
        self.reset_parameters()

    def reset_parameters(self):
        """Small random control coefficients, matching the DNN's scale roughly."""
        nn.init.uniform_(self.C, -1e-2, 1e-2)

    def n_params(self):
        """Number of trainable control coefficients."""
        return int(self.n_ctrl_x * self.n_ctrl_t)

    def forward(self, xt):
        """Map a batch of coordinates to field values.

        Parameters
        ----------
        xt : torch.Tensor, shape ``(N, 2)``
            Columns are ``(x, t)`` -- the same layout ``PhysicsInformedNN_pbc.net_u``
            feeds to ``self.dnn``.

        Returns
        -------
        torch.Tensor, shape ``(N, 1)`` -- field values ``u(x, t)``.
        """
        xt = xt.float()
        x = xt[:, 0]
        t = xt[:, 1]
        bx = _bspline_basis_1d(x, self.knots_x, self.degree)  # (N, n_ctrl_x)
        bt = _bspline_basis_1d(t, self.knots_t, self.degree)  # (N, n_ctrl_t)
        # u_k = sum_{i,j} C[i,j] * bx[k,i] * bt[k,j]
        u = torch.einsum("ni,ij,nj->n", bx, self.C, bt)
        return u.unsqueeze(1)

    def fit(self, X, u, damping=1e-8):
        """Least-squares fit of the control coefficients to sampled field values.

        One-shot linear solve (B-splines are linear in their control coefficients).
        Used to initialize the field to a known solution -- e.g. the analytical
        reference solution available in the PINN case study -- so the sampled loss
        landscape is taken over a field that actually represents a solution.

        Parameters
        ----------
        X : array-like, shape ``(M, 2)``
            Sample coordinates ``(x, t)``.
        u : array-like, shape ``(M,)`` or ``(M, 1)``
            Target field values.
        """
        X = np.asarray(X, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        with torch.no_grad():
            x = torch.tensor(X[:, 0], dtype=torch.float64)
            t = torch.tensor(X[:, 1], dtype=torch.float64)
            bx = _bspline_basis_1d(x, self.knots_x.double(), self.degree).numpy()
            bt = _bspline_basis_1d(t, self.knots_t.double(), self.degree).numpy()
        # Design matrix: Phi[m, i*n_ctrl_t + j] = bx[m, i] * bt[m, j]
        phi = (bx[:, :, None] * bt[:, None, :]).reshape(X.shape[0], -1)
        # Damped normal equations keep the solve stable at coarse grids.
        gram = phi.T @ phi + damping * np.eye(phi.shape[1])
        coef = np.linalg.solve(gram, phi.T @ u)
        with torch.no_grad():
            self.C.copy_(
                torch.tensor(coef.reshape(self.n_ctrl_x, self.n_ctrl_t), dtype=self.C.dtype)
            )
        return self
