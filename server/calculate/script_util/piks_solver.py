"""Physics-Informed Kernel methodS (PIKS): a closed-form PDE solver.

Adapted from "PIKS: Universal Physics-Informed Kernel Methods"
(arXiv:2607.27062v1). Provides a kernel baseline against the repo's trained
PINNs for the linear-PDE subset of the PINN case study.

Implementation mode -- Mode 2 (adapted port):

  * Core mechanism kept at full fidelity. A universal Gaussian kernel and the
    operator-theoretic block system that enforces linear differential
    constraints in closed form -- value/value, value/PDE, PDE/PDE blocks --
    are solved exactly as a Gaussian-process posterior mean. This is the
    central object the paper analyzes.

  * Auxiliary apparatus NOT ported (intentionally out of scope): the
    universal-consistency and finite-sample source-condition proofs, and the
    separate PINN/FEM benchmark suite. Those are theory and downstream
    evaluation; this module delivers the solver the theory describes.

  * The linear operator ``L = d/dt + beta*d/dx - nu*d^2/dx^2`` matches the
    repo's PINN residual (``net_pbc.PhysicsInformedNN_pbc.net_f``). PIKS
    guarantees cover *linear* differential constraints, so the solver is
    gated to the linear PDE subset (``rho == 0``); the reaction term
    ``rho*u*(1-u)`` is nonlinear and stays on the PINN.

The differential-operator images of the kernel (``G = L_b K`` and
``H = L_a L_b K``) are derived in closed form for this constant-coefficient
operator, so the block system assembles without autograd. With differences
``u = x_a - x_b``, ``v = t_a - t_b`` and bandwidth squared ``s = sigma**2``::

    K(a, b) = exp(-(u^2 + v^2) / (2 s))
    G(a, b) = L_b  K = K / s * [v + beta*u + nu*(1 - u^2 / s)]
    H(a, b) = L_a L_b K = K * [(1 + beta^2)/s + (3*nu^2 - P^2)/s^2
                               - 6*nu^2*u^2/s^3 + nu^2*u^4/s^4]
    where P = v + beta*u. ``G`` reduces (for nu=0) to the convection image
    ``K/s * (v + beta*u)`` and ``H`` to ``K * [(1+beta^2)/s - P^2/s^2]``.
"""

import os

import numpy as np


def _kernel_blocks(A, B, beta, nu, s):
    """Gaussian kernel ``K`` and its operator images ``G = L_b K``, ``H = L_a L_b K``.

    ``A`` (n, 2) and ``B`` (m, 2) hold columns ``[x, t]``. Returns ``(K, G, H)``
    each of shape (n, m). ``G(a, b)`` is the value/PDE cross-covariance
    (operator on the second/``b`` argument); ``H(a, b)`` is the PDE/PDE
    covariance (operator on both arguments) and is symmetric in ``a, b``.
    """
    d = A[:, None, :] - B[None, :, :]  # (n, m, 2)
    u = d[..., 0]  # x_a - x_b
    v = d[..., 1]  # t_a - t_b
    K = np.exp(-(u * u + v * v) / (2.0 * s))
    P = v + beta * u
    G = K / s * (v + beta * u + nu * (1.0 - u * u / s))
    H = K * (
        (1.0 + beta * beta) / s
        + (3.0 * nu * nu - P * P) / (s * s)
        - 6.0 * nu * nu * u * u / (s ** 3)
        + nu * nu * u ** 4 / (s ** 4)
    )
    return K, G, H


def piks_predict(X_obs, u_obs, X_col, f_col, X_query, beta, nu, sigma,
                 ridge=1e-6):
    """Closed-form PIKS estimate of ``u`` at ``X_query``.

    Solves the block system ``M [alpha; gamma] = [u_obs; f_col]`` with

    ::

        M = [[K(obs, obs),   G(obs, col)],
             [G(obs, col)^T, H(col, col)]]

    and predicts ``u(z) = K(z, obs) . alpha + G(z, col) . gamma`` -- the
    posterior mean of a GP whose observations are the Dirichlet data
    (``u_obs`` at ``X_obs``) and the PDE residual (``L u = f_col`` at
    ``X_col``). ``ridge * I`` is added to ``M`` for numerical stability
    (Tikhonov / source-condition regularization).

    Returns ``(u_pred, coeffs)`` where ``u_pred`` has shape (n_query, 1) and
    ``coeffs = [alpha; gamma]``.
    """
    X_obs = np.asarray(X_obs, dtype=float)
    X_col = np.asarray(X_col, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    u_obs = np.asarray(u_obs, dtype=float).reshape(-1)
    f_col = np.asarray(f_col, dtype=float).reshape(-1)
    s = float(sigma) ** 2

    K_oo = _kernel_blocks(X_obs, X_obs, beta, nu, s)[0]
    G_oc = _kernel_blocks(X_obs, X_col, beta, nu, s)[1]
    H_cc = _kernel_blocks(X_col, X_col, beta, nu, s)[2]

    n_o = K_oo.shape[0]
    n_c = H_cc.shape[0]
    M = np.zeros((n_o + n_c, n_o + n_c))
    M[:n_o, :n_o] = K_oo
    M[:n_o, n_o:] = G_oc
    M[n_o:, :n_o] = G_oc.T
    M[n_o:, n_o:] = H_cc
    M = M + ridge * np.eye(n_o + n_c)

    rhs = np.concatenate([u_obs, f_col])
    try:
        coeffs = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(M, rhs, rcond=None)

    alpha = coeffs[:n_o]
    gamma = coeffs[n_o:]

    K_qo = _kernel_blocks(X_query, X_obs, beta, nu, s)[0]
    G_qc = _kernel_blocks(X_query, X_col, beta, nu, s)[1]
    u_pred = K_qo @ alpha + G_qc @ gamma
    return u_pred.reshape(-1, 1), coeffs


def assemble_observations(xx1, uu1, bc_lb, uu2, bc_ub):
    """Stack the Dirichlet data: initial condition + periodic boundaries.

    Mirrors the boundary/initial-condition arrays built in
    ``compute_pinn_loss_landscape.py`` (``xx1``/``uu1`` for the t=0 initial
    condition; ``bc_lb``/``bc_ub`` with shared values ``uu2`` for the periodic
    x=0 / x=2*pi boundaries).
    """
    X_obs = np.vstack([xx1, bc_lb, bc_ub])
    u_obs = np.vstack([uu1, uu2, uu2])
    return X_obs, u_obs


def run_piks_baseline(args, X_star, u_star, X_f_train, G,
                      xx1, uu1, bc_lb, bc_ub, uu2,
                      nu, beta, rho, save=True):
    """Solve PIKS for the configured PDE and report L2 error vs ``u_star``.

    This is the entry point wired into ``compute_pinn_loss_landscape.py`` via
    the ``--piks`` flag. It assembles the observation/collocation data the
    PINN pipeline already produces, solves the kernel block system in closed
    form, and stores the prediction + L2 error as a comparison baseline.

    Returns a dict with ``u_pred``, ``l2_abs``, ``l2_rel``, ``n_obs``, ``n_col``.
    """
    if rho != 0:
        raise ValueError(
            f"PIKS supports only linear PDEs (rho == 0); got rho={rho!r}. "
            "Reaction / reaction-diffusion stay on the PINN."
        )

    sigma = float(getattr(args, "piks_sigma", 1.0))
    ridge = float(getattr(args, "piks_ridge", 1e-6))

    X_obs, u_obs = assemble_observations(xx1, uu1, bc_lb, uu2, bc_ub)
    f_col = np.asarray(G, dtype=float).reshape(-1)

    u_pred, _ = piks_predict(
        X_obs, u_obs, X_f_train, f_col, X_star,
        beta=beta, nu=nu, sigma=sigma, ridge=ridge,
    )

    u_star = np.asarray(u_star, dtype=float).reshape(-1, 1)
    l2_abs = float(np.linalg.norm(u_pred - u_star))
    l2_rel = l2_abs / float(np.linalg.norm(u_star))

    print("=" * 60)
    print("PIKS closed-form baseline (linear PDE subset)")
    print(
        f"  system={getattr(args, 'system', '?')}  beta={beta:g}  "
        f"nu={nu:g}  sigma={sigma:g}  ridge={ridge:g}"
    )
    print(f"  n_obs={X_obs.shape[0]}  n_col={X_f_train.shape[0]}")
    print(f"  L2 abs error vs u_star: {l2_abs:.6e}")
    print(f"  L2 rel error vs u_star: {l2_rel:.6e}")
    print("=" * 60)

    if save:
        out_dir = "../analyze_loss_cubes/loss_landscape_files_highdim"
        os.makedirs(out_dir, exist_ok=True)
        fname = (
            f"piks_baseline_{getattr(args, 'system', '?')}"
            f"_u0{getattr(args, 'u0_str', '?')}"
            f"_nu{nu:g}_beta{beta:g}"
            f"_Nf{getattr(args, 'N_f', -1)}_sigma{sigma:g}.npy"
        )
        np.save(os.path.join(out_dir, fname), u_pred)

    return {
        "u_pred": u_pred,
        "l2_abs": l2_abs,
        "l2_rel": l2_rel,
        "n_obs": X_obs.shape[0],
        "n_col": int(np.asarray(X_f_train).shape[0]),
    }
