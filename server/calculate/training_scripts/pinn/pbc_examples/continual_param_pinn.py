"""Continual learning over a PDE parameter domain for PINNs.

Implements the core mechanism of Continual-Learning PINNs (CL-PINN,
arXiv:2608.04778) adapted to this repo's ``pbc_examples``. Whereas
``main_pbc`` trains one PINN for a single scalar parameter setting, this
module learns a *family* of PDE solutions across a parameter domain
*sequentially as related tasks* on one shared model:

  * a single SHARED, parameter-conditioned ``DNN`` (the paper's "optional
    parameter subnetwork", made load-bearing here so one net represents
    ``u(x, t; mu)``) is trained task-by-task;
  * SPARSE PHYSICS-CONSTRAINED REPLAY rehearses a small buffer of earlier
    tasks' collocation residuals every step to mitigate forgetting;
  * ACTIVE PARAMETER SELECTION picks the next task by an error-greedy
    acquisition (worst-current-residual parameter) instead of a fixed grid.

Mode 2 (adapted port) -- substitutions, cited:
  * Bayesian-optimization active selector -> parameter-free error-greedy
    acquisition (the paper's "grid-greedy" shape); active *behaviour*
    preserved, BO query-efficiency gain is not.
  * Separate five-benchmark eval suite cut; this emits one trained model
    plus a per-parameter accuracy report for downstream LossLens analysis.
  * Task-wise dynamic loss weighting -> parameter-free uniform average of
    replay-task residuals vs. the current task.

Reuses ``net_pbc.DNN``, the ``systems_pbc`` solvers, ``choose_optimizer``,
and ``utils.set_seed`` / ``utils.sample_random`` unchanged; the residual
mirrors ``PhysicsInformedNN_pbc.net_f`` so the physics is identical to the
single-task path.
"""

import os

import numpy as np
import torch
from torch import autograd

from net_pbc import DNN
from systems_pbc import (
    convection_diffusion,
    reaction_diffusion_discrete_solution,
    reaction_solution,
)
from choose_optimizer import choose_optimizer
from utils import set_seed, sample_random

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(array):
    return torch.tensor(np.asarray(array), dtype=torch.float32, device=_DEVICE)


def _default_param(system):
    """Physical coefficient swept for each supported system."""
    return {"convection": "beta", "diffusion": "nu", "rd": "rho", "reaction": "rho"}[
        system
    ]


def _task_coeffs(system, param, mu):
    """Per-task (nu, beta, rho) with the swept coefficient set to ``mu``.

    Mirrors the per-system coefficient zeroing in ``main_pbc``.
    """
    nu = beta = rho = 0.0
    if system == "convection":
        if param == "beta":
            beta = mu
    elif system == "diffusion":
        if param == "nu":
            nu = mu
    elif system == "rd":
        nu, rho = 1.0, 1.0
        if param == "rho":
            rho = mu
        elif param == "nu":
            nu = mu
    elif system == "reaction":
        if param == "rho":
            rho = mu
    else:
        raise ValueError("unsupported system: %s" % system)
    return nu, beta, rho


def build_task(system, param, mu, xgrid, nt, n_f, source, u0_str):
    """IC/BC/collocation data + exact solution for one parameter value.

    Reuses the ``systems_pbc`` ground-truth solvers and
    ``utils.sample_random``, paralleling the data section of ``main_pbc``
    but parameterised by ``mu``.
    """
    nu, beta, rho = _task_coeffs(system, param, mu)
    x = np.linspace(0, 2 * np.pi, xgrid, endpoint=False).reshape(-1, 1)
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    grid_X, grid_T = np.meshgrid(x, t)
    X_star = np.hstack((grid_X.flatten()[:, None], grid_T.flatten()[:, None]))

    if system in ("convection", "diffusion"):
        u_vals = convection_diffusion(u0_str, nu, beta, source, xgrid, nt)
    elif system == "rd":
        u_vals = reaction_diffusion_discrete_solution(u0_str, nu, rho, xgrid, nt)
    elif system == "reaction":
        u_vals = reaction_solution(u0_str, rho, xgrid, nt)
    else:
        raise ValueError("unsupported system: %s" % system)

    u_star = u_vals.reshape(-1, 1)
    exact = u_star.reshape(len(t), len(x))
    xx1 = np.hstack((grid_X[0:1, :].T, grid_T[0:1, :].T))  # IC at t=0
    bc_lb = np.hstack((grid_X[:, 0:1], grid_T[:, 0:1]))  # BC at x=0
    bc_ub = np.hstack((np.full_like(t, 2 * np.pi), t))  # BC at x=2pi

    interior_X, interior_T = np.meshgrid(x[1:], t[1:])  # PDE enforced on interior
    interior = np.hstack((interior_X.flatten()[:, None], interior_T.flatten()[:, None]))
    X_f = sample_random(interior, min(n_f, interior.shape[0]))

    return {
        "mu": float(mu),
        "nu": nu,
        "beta": beta,
        "rho": rho,
        "x_ic": xx1[:, 0:1], "t_ic": xx1[:, 1:2], "u_ic": exact[0:1, :].T,
        "x_lb": bc_lb[:, 0:1], "t_lb": bc_lb[:, 1:2],
        "x_ub": bc_ub[:, 0:1], "t_ub": bc_ub[:, 1:2],
        "x_f": X_f[:, 0:1], "t_f": X_f[:, 1:2],
        "X_star": X_star, "u_star": u_star, "source": float(source),
    }


class ContinualParamPINN:
    """One shared parameter-conditioned DNN learned across a parameter domain."""

    def __init__(
        self, system, param, param_lo, param_hi, layers_str, activation="tanh",
        xgrid=256, nt=100, n_f=100, source=0.0, u0_str="sin(x)", lr=1e-3,
        L=1.0, replay_size=200, replay_weight=1.0, seed=0,
    ):
        set_seed(seed)
        self.system = system
        self.param = param or _default_param(system)
        self.param_lo, self.param_hi = float(param_lo), float(param_hi)
        self.xgrid, self.nt, self.n_f = xgrid, nt, n_f
        self.source, self.u0_str, self.L = float(source), u0_str, L
        self.replay_size, self.replay_weight = replay_size, replay_weight

        layers = [3] + [int(v) for v in layers_str.split(",")]  # (x, t, mu) -> u
        self.dnn = DNN(layers, activation).to(_DEVICE)
        self.optimizer = choose_optimizer("Adam", self.dnn.parameters(), lr)

        self.buffer = []  # sparse collocation from prior tasks (replay)
        self.history = []  # (mu, err_before, err_after) per trained task
        self.initial_errors, self.final_errors, self.task_order = {}, {}, []

    @staticmethod
    def _grad(y, x):
        return autograd.grad(
            y, x, torch.ones_like(y), retain_graph=True, create_graph=True
        )[0]

    def _norm(self, mu):
        span = self.param_hi - self.param_lo
        return 2.0 * (mu - self.param_lo) / (span + 1e-12) - 1.0

    def _net_u(self, x, t, mu_norm):
        mu = torch.full((x.shape[0], 1), float(mu_norm), device=_DEVICE)
        return self.dnn(torch.cat([x, t, mu], dim=1))

    def _residual(self, x, t, mu_norm, coeffs):
        """PDE residual f(x, t; mu), mirroring PhysicsInformedNN_pbc.net_f."""
        nu, beta, rho = coeffs
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = self._net_u(x, t, mu_norm)
        u_t = self._grad(u, t)
        if self.system == "reaction":
            return u_t - rho * u + rho * u ** 2
        u_x = self._grad(u, x)
        if self.system in ("convection", "diffusion"):
            f = u_t + beta * u_x - self.source
            return f if nu == 0 else f - nu * self._grad(u_x, x)
        if self.system == "rd":
            return u_t - nu * self._grad(u_x, x) - rho * u + rho * u ** 2
        raise ValueError("unsupported system: %s" % self.system)

    def _replay_loss(self):
        """Sparse physics-constrained replay: rehearse prior tasks' residuals."""
        if not self.buffer:
            return torch.zeros((), device=_DEVICE)
        losses = [
            torch.mean(self._residual(e["x_f"], e["t_f"], e["mu_norm"], e["coeffs"]) ** 2)
            for e in self.buffer
        ]
        return torch.stack(losses).mean()  # uniform task weighting (parameter-free)

    def _add_replay(self, task, coeffs, mu_norm):
        x_f, t_f = _to_tensor(task["x_f"]), _to_tensor(task["t_f"])
        idx = torch.randperm(x_f.shape[0])[: min(self.replay_size, x_f.shape[0])]
        self.buffer.append(
            {"x_f": x_f[idx], "t_f": t_f[idx], "mu_norm": mu_norm, "coeffs": coeffs}
        )

    def _error_on_task(self, task, mu_norm):
        self.dnn.eval()
        with torch.no_grad():
            xs = _to_tensor(task["X_star"])
            u_pred = self._net_u(xs[:, 0:1], xs[:, 1:2], mu_norm).cpu().numpy()
        self.dnn.train()
        u_star = task["u_star"]
        return float(np.linalg.norm(u_star - u_pred, 2) / (np.linalg.norm(u_star, 2) + 1e-12))

    def eval_error(self, mu):
        """Relative L2 error of the shared model at parameter value ``mu``."""
        task = build_task(
            self.system, self.param, mu, self.xgrid, self.nt, self.n_f,
            self.source, self.u0_str,
        )
        return self._error_on_task(task, self._norm(mu))

    def select_next_parameter(self, candidates):
        """Error-greedy active acquisition: worst-current-error parameter."""
        errors = [self.eval_error(m) for m in candidates]
        return candidates[int(np.argmax(errors))], errors

    def predict(self, X, mu):
        self.dnn.eval()
        with torch.no_grad():
            xs = _to_tensor(X)
            u = self._net_u(xs[:, 0:1], xs[:, 1:2], self._norm(mu)).cpu().numpy()
        self.dnn.train()
        return u

    def train_task(self, mu, steps, verbose=False):
        """Train the shared model on one task (parameter value) with replay."""
        mu_norm = self._norm(mu)
        task = build_task(
            self.system, self.param, mu, self.xgrid, self.nt, self.n_f,
            self.source, self.u0_str,
        )
        coeffs = (task["nu"], task["beta"], task["rho"])
        err_before = self._error_on_task(task, mu_norm)

        x_ic, t_ic, u_ic = _to_tensor(task["x_ic"]), _to_tensor(task["t_ic"]), _to_tensor(task["u_ic"])
        x_lb, t_lb = _to_tensor(task["x_lb"]), _to_tensor(task["t_lb"])
        x_ub, t_ub = _to_tensor(task["x_ub"]), _to_tensor(task["t_ub"])
        x_f, t_f = _to_tensor(task["x_f"]), _to_tensor(task["t_f"])

        for step in range(steps):
            self.optimizer.zero_grad()
            loss_ic = torch.mean((self._net_u(x_ic, t_ic, mu_norm) - u_ic) ** 2)

            u_lb, u_ub = self._net_u(x_lb, t_lb, mu_norm), self._net_u(x_ub, t_ub, mu_norm)
            loss_bc = torch.mean((u_lb - u_ub) ** 2)
            if coeffs[0] != 0:  # nu != 0: enforce periodic derivative too
                x_lb_g = x_lb.detach().requires_grad_(True)
                t_lb_g = t_lb.detach().requires_grad_(True)
                x_ub_g = x_ub.detach().requires_grad_(True)
                t_ub_g = t_ub.detach().requires_grad_(True)
                u_lb_x = self._grad(self._net_u(x_lb_g, t_lb_g, mu_norm), x_lb_g)
                u_ub_x = self._grad(self._net_u(x_ub_g, t_ub_g, mu_norm), x_ub_g)
                loss_bc = loss_bc + torch.mean((u_lb_x - u_ub_x) ** 2)

            loss_f = torch.mean(self._residual(x_f, t_f, mu_norm, coeffs) ** 2)
            loss_replay = self._replay_loss()
            loss = loss_ic + loss_bc + self.L * loss_f + self.replay_weight * loss_replay
            loss.backward()
            self.optimizer.step()

            if verbose and step % max(1, steps // 5) == 0:
                print(
                    "  task mu=%.4f step %d/%d loss=%.3e loss_f=%.3e replay=%.3e"
                    % (mu, step, steps, loss.item(), loss_f.item(), loss_replay.item())
                )

        self._add_replay(task, coeffs, mu_norm)
        err_after = self._error_on_task(task, mu_norm)
        self.history.append((float(mu), err_before, err_after))
        return err_before, err_after


def run_continual(args):
    """Entry point invoked by ``main_pbc --continual``.

    Trains one shared parameter-conditioned model across the parameter
    domain, then reports per-parameter accuracy for downstream LossLens
    comparison against static-sampling PINNs.
    """
    param = args.param or _default_param(args.system)
    lo, hi = args.param_min, args.param_max
    candidates = [float(v) for v in np.linspace(lo, hi, args.n_tasks)]

    model = ContinualParamPINN(
        system=args.system, param=param, param_lo=lo, param_hi=hi,
        layers_str=args.layers, activation=args.activation, xgrid=args.xgrid,
        nt=args.nt, n_f=args.N_f, source=args.source, u0_str=args.u0_str,
        lr=args.lr, L=args.L, replay_size=args.replay_size, seed=args.seed,
    )
    model.initial_errors = {round(c, 6): model.eval_error(c) for c in candidates}

    pool = list(candidates)
    while pool:
        mu = model.select_next_parameter(pool)[0] if args.active_selection else pool[0]
        pool.remove(mu)
        model.task_order.append(mu)
        print("CL-PINN task mu=%.4f" % mu)
        model.train_task(mu, args.steps_per_task, verbose=True)

    model.final_errors = {round(c, 6): model.eval_error(c) for c in candidates}
    print("=== CL-PINN continual-learning report ===")
    print(
        "system=%s param=%s domain=[%g,%g] n_tasks=%d active=%s order=%s"
        % (args.system, param, lo, hi, args.n_tasks, args.active_selection,
           [round(m, 4) for m in model.task_order])
    )
    for c in candidates:
        k = round(c, 6)
        print(
            "  param=%.4f init=%.3e final=%.3e"
            % (c, model.initial_errors[k], model.final_errors[k])
        )
    arr = np.array(list(model.final_errors.values()))
    print("mean final=%.3e max/min=%.3e (lower max/min = more balanced)" % (arr.mean(), arr.max() / (arr.min() + 1e-12)))

    if args.save_model:
        out = "saved_models"
        os.makedirs(out, exist_ok=True)
        fn = "%s/continual_%s_%s_%gto%g_ntasks%d_Nf%d_%s_seed%d.pt" % (
            out, args.system, param, lo, hi, args.n_tasks, args.N_f, args.layers, args.seed
        )
        torch.save(model, fn)
        print("saved shared parameter-conditioned model -> %s" % fn)

    return model
