"""Kron-factored preconditioned SGD (PSGD with a stochastic Hessian fit).

Adapted from "Stochastic Hessian Fittings with Lie Groups"
(arXiv:2402.11858v6). The paper's core result is that P*grad is a descent
direction whenever the preconditioner under-estimates the inverse Hessian
in the PSD-order sense (0 < P <= inv(H)), and that Kronecker-factor
preconditioners -- Lie-group parameters on the SPD cone, updated
multiplicatively as Q <- Q - eta*G*Q -- are the most reliable way to reach
that regime stochastically. That is what is implemented here, written from
the paper's fitting criterion

    min_Q  || Q dG ||_F^2 - || Q^-T dX ||_F^2

probed with a pair (dX, dG) where dG = H dX is obtained by one extra
backward pass (Hessian-vector product) on the same minibatch. This is the
paper's "stochastic Hessian fitting": the pair is random, so the fit sees
the local curvature without ever forming the Hessian.

The eta normalization uses a lower bound on the spectral norm (max of the
1- and inf-induced norms), which keeps the Lie-group step scale-free.

Out of scope: the parameter-free trust region and step-size scaling, the
low-rank and affine preconditioner families, and the paper's benchmark
suite. This is the dense Kron fit plus the preconditioned step, expressed
as a drop-in ``torch.optim`` optimizer for the PINN training scripts.
"""

import torch

_tiny = 1e-9


def _norm_lower_bound(grad: torch.Tensor) -> torch.Tensor:
    """Cheap lower bound on the spectral norm used to normalize the step."""
    m, n = grad.shape
    if m == 1 and n == 1:
        return grad.abs()
    return torch.maximum(
        torch.linalg.matrix_norm(grad, ord=1) / max(m, n),
        torch.linalg.matrix_norm(grad, ord=float("inf")),
    )


def balance_kron_factors(factors: tuple) -> tuple:
    """Rescale the two factors so they carry comparable weight.

    P = kron(Qr^T Qr, Ql^T Ql) is invariant under Ql -> Ql/s, Qr -> Qr*s,
    so the split between the factors is a free gauge. Left alone it drifts
    toward one factor collapsing, which degrades the triangular solves.
    """
    left, right = factors
    scale = (left.diag().norm() / (right.diag().norm() + _tiny)).sqrt()
    return left / scale, right * scale


def woodfulk_kron_update(
    left: torch.Tensor,
    right: torch.Tensor,
    probe: torch.Tensor,
    hess_probe: torch.Tensor,
    step: float = 0.5,
):
    """One multiplicative (Lie-group) update of a dense Kron-factor pair.

    ``left`` and ``right`` are the factors of the preconditioner
    P = kron(right^T right, left^T left). The pair (probe, hess_probe)
    must satisfy hess_probe ~= H @ probe, i.e. be a genuine
    Hessian-vector product for the same random direction.

    Args:
        left: factor of shape (out_dim, out_dim).
        right: factor of shape (in_dim, in_dim).
        probe: random probing direction, shape (out_dim, in_dim).
        hess_probe: Hessian acting on ``probe``, same shape.
        step: normalized step size for the Lie-group update.

    Returns:
        The updated (left, right) factors.
    """
    a = left @ hess_probe @ right.t()
    b = torch.linalg.solve_triangular(
        right,
        torch.linalg.solve_triangular(left.t(), probe, upper=False),
        upper=True,
        left=False,
    )

    grad_left = torch.triu(a @ a.t() - b @ b.t())
    grad_right = torch.triu(a.t() @ a - b.t() @ b)

    step_left = step / (_norm_lower_bound(grad_left) + _tiny)
    step_right = step / (_norm_lower_bound(grad_right) + _tiny)
    return left - step_left * (grad_left @ left), right - step_right * (grad_right @ right)


def precond_grad_kron(factors: tuple, grad: torch.Tensor) -> torch.Tensor:
    """Apply P = kron(right^T right, left^T left) to a matrix gradient."""
    left, right = factors
    return torch.linalg.multi_dot([left.t(), left, grad, right.t(), right])


def _scalar_update(scale: torch.Tensor, probe: torch.Tensor, hess_probe: torch.Tensor, step: float):
    """Scalar inverse-Hessian fit for parameters that are not Kron-factored.

    The same fitting criterion with a 1x1 preconditioner reduces to a
    scalar multiple of the identity, which is the paper's treatment of
    biases and other non-matrix parameters.
    """
    grad_q = (scale * probe).square().sum() - (hess_probe / (scale + _tiny)).square().sum()
    return scale - (step / (grad_q.abs() + _tiny)) * grad_q * scale


class KronPSGD(torch.optim.Optimizer):
    """Preconditioned SGD with a Kronecker-factored stochastic Hessian fit.

    2D parameters get a dense Kron fit P = kron(Qr^T Qr, Ql^T Ql); anything
    else (biases, normalization scales, conv filters) gets the scalar fit.
    Both are updated multiplicatively, so P stays on the SPD cone.

    The fit needs one extra backward pass per step to form the
    Hessian-vector product, so the closure passed to ``step`` is evaluated
    with ``create_graph=True``. Optimizers that do not need curvature can
    ignore the closure entirely; this one requires it.
    """

    def __init__(self, params, lr=1e-3, step=0.5, weight_decay=0.0, balance_every=10):
        defaults = dict(lr=lr, step=step, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.balance_every = balance_every

    def _init_state(self, p, grad):
        state = self.state[p]
        if "factors" in state:
            return state
        if grad.dim() == 2:
            state["factors"] = (
                torch.eye(grad.shape[0], device=grad.device, dtype=grad.dtype),
                torch.eye(grad.shape[1], device=grad.device, dtype=grad.dtype),
            )
        else:
            state["factors"] = torch.ones((), device=grad.device, dtype=grad.dtype)
        state["count"] = 0
        return state

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError(
                "KronPSGD needs a closure: the Hessian fit is probed with one "
                "extra backward pass per step (Hessian-vector product)."
            )

        probes = {}
        with torch.enable_grad():
            loss = closure()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self._init_state(p, p.grad)
                    probe = torch.randn(p.shape, device=p.device, dtype=p.dtype)
                    # the Hessian-vector product, only defined if the closure
                    # built a graph for its gradients
                    if p.grad.grad_fn is None:
                        raise ValueError(
                            "KronPSGD needs gradients with a grad_fn; call "
                            "loss.backward(create_graph=True) in the closure."
                        )
                    hess_p = torch.autograd.grad(
                        p.grad, p, grad_outputs=probe, retain_graph=True
                    )[0]
                    probes[p] = (probe, hess_p.detach())

        for group in self.param_groups:
            lr = group["lr"]
            step = group["step"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad
                probe, hess_probe = probes[p]

                if grad.dim() == 2:
                    left, right = state["factors"]
                    state["factors"] = woodfulk_kron_update(left, right, probe, hess_probe, step)
                    state["count"] += 1
                    if self.balance_every and state["count"] % self.balance_every == 0:
                        state["factors"] = balance_kron_factors(state["factors"])
                    update = precond_grad_kron(state["factors"], grad)
                else:
                    scale = state["factors"]
                    state["factors"] = _scalar_update(scale, probe, hess_probe, step)
                    update = grad * state["factors"]

                if wd:
                    update = update.add(p, alpha=wd)
                p.add_(update.reshape_as(p), alpha=-lr)

        return loss
