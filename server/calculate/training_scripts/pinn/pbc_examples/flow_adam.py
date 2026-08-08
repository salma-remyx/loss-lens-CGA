"""Gauge-equivariant (shared-scalar) Adam optimizer.

Standard Adam / RMSProp scale every parameter coordinate by its own running
second moment. That coordinate-wise preconditioning breaks the gauge
symmetry ``(U, V) -> (U Q, V Q)`` of a factored model ``W = U V^T`` -- the
loss does not see the basis, but Adam does -- so the coordinate-wise rules do
NOT inherit gradient flow's low-rank implicit bias.

``FlowAdam`` keeps Adam's first moment (a linear EMA, which commutes with any
orthogonal rotation and is therefore gauge-equivariant) but replaces the
per-coordinate second moment with a single SHARED scalar: the running mean of
the squared gradient over the whole parameter tensor. A uniform scalar
preconditioner commutes with every orthogonal ``Q``, so the update rule is
gauge-equivariant and keeps the low-rank bias that coordinate-wise Adam
loses. This is the "shared-scalar Adam" endpoint of the paper's
coordinate-wise -> shared-scalar family.

Adapted (Mode 2) from "The Loss Does Not See the Basis, but Adam Does"
(arXiv:2608.05136). Substitutions: only the shared-scalar Adam rule is
ported. The paper's full one-parameter anisotropy family, the "spectral
schedule", and the Muon / Shampoo comparison rules are out of scope here --
the repo's optimizer zoo already ships Shampoo, and the other rules are
orthogonal axes a downstream PR can add. The matrix-sensing recovery demo
below exercises the paper's core result on a *factored* model (the one place
the gauge contribution is actually live) rather than claiming landscape
changes on the repo's non-factored PINN / MLP / ResNet case studies.
"""

import torch


class FlowAdam(torch.optim.Optimizer):
    """Adam with a shared-scalar (gauge-equivariant) second moment.

    The preconditioner is a single scalar per parameter tensor -- the running
    mean of the squared gradient -- so it commutes with every orthogonal
    rotation of the parameter. That is the property the paper shows is
    necessary for an optimizer to carry gradient flow's pathwise low-rank
    mechanism on a factored ``W = U V^T``.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr <= 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros((), dtype=p.dtype, device=p.device)
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]
                # Per-coordinate first moment: a linear EMA commutes with any
                # orthogonal rotation, so it is gauge-equivariant on its own.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Shared-scalar second moment: mean of squared gradient over the
                # whole tensor. Invariant under any orthogonal rotation of the
                # parameter, so the resulting preconditioner is a uniform scalar.
                v.mul_(beta2).add_(grad.pow(2).mean(), alpha=1 - beta2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                denom = v_hat.sqrt().add(eps)
                p.addcdiv_(m_hat, denom, value=-lr)
        return loss


def matrix_sensing_recovery(optimizer_name="FlowAdam", m=15, n=15, rank=1,
                            n_samples=60, steps=4000, lr=1e-2, seed=0):
    """Underdetermined matrix-sensing illustration of the paper's core result.

    Fits a factored model ``W_hat = U V^T`` from a small initialization to
    ``n_samples`` random linear measurements of a planted rank-``rank`` ground
    truth ``W*``. With ``n_samples < m * n`` the problem is underdetermined,
    so the optimizer -- not the loss -- selects the interpolant. Returns the
    relative Frobenius recovery error ``||U V^T - W*||_F / ||W*||_F`` and the
    effective rank ``(sum s_i^2)^2 / sum s_i^4`` of the recovered product
    (equals ``rank`` for an exactly rank-``rank`` matrix).

    Routed through the repo's ``choose_optimizer`` factory so the FlowAdam
    branch added there is the live code path. This is an exploratory
    illustration, not a benchmark reproduction: the recovery gap is real but
    seed/stack-sensitive in the narrow underdetermined band, so
    ``compare_gauge_bias`` averages over several seeds rather than trusting a
    single run. The deterministic statement of the paper's mechanism -- that a
    shared-scalar preconditioner is gauge-equivariant while a coordinate-wise
    one is not -- is asserted directly in ``test_flow_adam``.
    """
    from pinn.pbc_examples.choose_optimizer import choose_optimizer

    g = torch.Generator().manual_seed(seed + 1)

    # Planted low-rank ground truth.
    u_star = torch.randn(m, rank, generator=g) * 0.1
    v_star = torch.randn(n, rank, generator=g) * 0.1
    w_star = u_star @ v_star.t()

    # Random Gaussian sensing tensors; fewer than m*n measurements -> underdetermined.
    A = torch.randn(n_samples, m, n, generator=g)
    y = (A * w_star).sum(dim=(1, 2))

    # Small initialization: required for gradient flow's low-rank mechanism.
    U = (torch.randn(m, rank, generator=g) * 1e-3).requires_grad_(True)
    V = (torch.randn(n, rank, generator=g) * 1e-3).requires_grad_(True)

    opt = choose_optimizer(optimizer_name, [U, V], lr)

    for _ in range(steps):
        opt.zero_grad()
        w_hat = U @ V.t()
        pred = (A * w_hat).sum(dim=(1, 2))
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        w_hat = U @ V.t()
        rel_err = (w_hat - w_star).norm() / w_star.norm().clamp_min(1e-12)
        sv = torch.linalg.svdvals(w_hat).clamp_min(1e-12)
        eff_rank = (sv ** 2).sum() ** 2 / (sv ** 4).sum().clamp_min(1e-12)
    return float(rel_err), float(eff_rank)


def compare_gauge_bias(seeds=(0, 1, 2, 3, 4, 5, 6, 7), **kwargs):
    """Run the sensing demo across the optimizer zoo, averaged over ``seeds``.

    Returns the mean relative recovery error and effective rank per optimizer.
    In the underdetermined regime the gauge-equivariant rules (SGD, FlowAdam)
    typically recover the planted low-rank target with markedly lower error
    than coordinate-wise Adam -- the paper's central observation that basis
    choice selects which interpolant the optimizer reaches. Averaging over
    seeds tames the per-run noise of this finicky optimization; the result is
    still an illustration of the gauge axis, not a benchmark claim.
    """
    rows = {}
    for name in ["Adam", "SGD", "FlowAdam"]:
        errs, ranks = [], []
        for seed in seeds:
            rel_err, eff_rank = matrix_sensing_recovery(
                optimizer_name=name, seed=seed, **kwargs)
            errs.append(rel_err)
            ranks.append(eff_rank)
        n = len(seeds)
        rows[name] = {"rel_err": sum(errs) / n, "eff_rank": sum(ranks) / n}
    return rows


if __name__ == "__main__":
    import json

    print(json.dumps(compare_gauge_bias(), indent=2))
