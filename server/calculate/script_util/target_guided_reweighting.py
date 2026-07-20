"""Target-guided selective reweighting for PINN transfer learning.

Implements the core representation-correction mechanism of TGSR-PINN
("Target-Guided Selective Reweighting for Physics-Informed Neural Network
Inverse Problems: A Transfer Learning Approach", arXiv:2607.05271), operating
on the repo's existing ``PhysicsInformedNN_pbc`` (a tanh-MLP ``DNN`` trained
for the convection / diffusion / reaction PDE family).

Given a *source* PINN and a fixed *target* scoring batch, the method re-weights
the source using target evidence only:

1. **Neuron target scores** from first-order Taylor sensitivity of a
   target-evidence scalar (the PDE residual energy when the model exposes
   ``net_f``) w.r.t. each neuron's pre-activation, combined with the
   pre-activation variance over the scoring batch.
2. **Continuous weak-adaptation signals** per neuron, estimated with a
   2-component Gaussian mixture model over the scores, falling back to a
   rank-based normalization when the GMM is unavailable or degenerate.
3. **Selective soft decay** of the input-weight rows and biases of low-scoring
   neurons (continuous scaling, not hard pruning or random reset), skipping
   protected layers (the input-proximal layer by default).

This is an **adapted port (Mode 2)**. Substituted / scoped out, intentionally:
the source-PINN *training* pipeline (LossLens consumes already-trained PINNs as
sources, so reweighting is applied to a loaded model); the inverse-problem
parameter-recovery benchmark (the rewritten model is fed straight into the
existing loss-landscape / Hessian / topological pipeline in
``compute_pinn_loss_landscape.py``); and the paper's short target-adaptation
phase (exposed via ``adapt_steps``, defaulting to ``0``). The kept-at-full
-fidelity core is neuron scoring + GMM-with-rank-fallback signal estimation +
selective soft decay + layer protection.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import torch
import torch.nn as nn

try:  # GMM is the paper-faithful signal estimator; rank fallback covers its absence.
    from sklearn.mixture import GaussianMixture  # type: ignore

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    GaussianMixture = None  # type: ignore
    _HAS_SKLEARN = False


def iter_hidden_linears(dnn) -> Iterable[Tuple[int, str, nn.Linear]]:
    """Yield ``(position, name, module)`` for the hidden ``nn.Linear`` layers.

    The repo's ``DNN`` stacks ``nn.Linear`` + activation modules inside an
    ``nn.Sequential`` named ``layers``. The final ``nn.Linear`` is the output
    layer and is excluded from scoring (it has a single neuron and is not a
    representation neuron).
    """
    seq = getattr(dnn, "layers", dnn)
    linears = [
        (name, mod)
        for name, mod in seq.named_modules()
        if isinstance(mod, nn.Linear)
    ]
    for position, (name, module) in enumerate(linears[:-1]):
        yield position, name, module


def compute_neuron_scores(
    dnn,
    predict_fn: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    residual_fn: Callable = None,
) -> Dict[str, Dict[str, object]]:
    """Score each hidden neuron by Taylor sensitivity x pre-activation variance.

    A forward hook captures every hidden Linear layer's pre-activation and
    ``retain_grad()``-s it; a single backward from a target-evidence scalar
    then populates the per-neuron sensitivity. The target-evidence scalar is
    the mean-squared PDE residual when ``residual_fn`` is given (the PINN
    objective, faithful to "target-evidence-driven"), otherwise the mean output
    energy. Sensitivity and variance are reduced over the scoring batch.
    """
    dnn.zero_grad(set_to_none=True)
    handles = []
    store: Dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            output.retain_grad()
            store[name] = output

        return hook

    for _pos, name, module in iter_hidden_linears(dnn):
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        output = predict_fn(x, t)
        if residual_fn is not None:
            residual = residual_fn(x, t)
            scalar = (residual ** 2).mean()
        else:
            scalar = (output ** 2).mean()
        scalar.backward()
    finally:
        for handle in handles:
            handle.remove()

    scores: Dict[str, Dict[str, object]] = {}
    for position, name, _module in iter_hidden_linears(dnn):
        pre_activation = store[name]
        pre = pre_activation.detach()
        grad = pre_activation.grad
        if grad is None:
            sensitivity = torch.zeros(pre.shape[1], device=pre.device)
        else:
            sensitivity = grad.detach().abs().mean(dim=0)
        variance = pre.var(dim=0, unbiased=False)
        raw = sensitivity * torch.sqrt(variance.clamp_min(0.0))
        scores[name] = {
            "sensitivity": sensitivity,
            "variance": variance,
            "raw": raw,
            "position": position,
        }
    return scores


def _rank_factors(raw: torch.Tensor, decay: float) -> torch.Tensor:
    """Continuous factors in ``[decay, 1]`` from score rank (parameter-free)."""
    n = raw.numel()
    if n <= 1:
        return torch.ones(n, device=raw.device, dtype=raw.dtype)
    order = torch.argsort(raw, stable=True)
    ranks = torch.empty(n, device=raw.device, dtype=raw.dtype)
    ranks[order] = torch.arange(n, device=raw.device, dtype=raw.dtype)
    norm = ranks / (n - 1)  # 0 for the lowest-scoring neuron .. 1 for the highest
    return decay + (1.0 - decay) * norm


def _gmm_factors(raw: torch.Tensor, decay: float, random_state: int) -> torch.Tensor:
    """Continuous factors from a 2-component GMM over the score distribution.

    The low-mean component carries the weak-adaptation (decayed) neurons; each
    neuron's factor is its responsibility for the high-mean component, mapped to
    ``[decay, 1]``. Raises on any degeneracy so the caller falls back to rank.
    """
    import numpy as np

    n = raw.numel()
    if n < 2:
        raise ValueError("need at least 2 neurons to fit a GMM")
    data = raw.detach().cpu().numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_state, n_init=1)
    gmm.fit(data)
    responsibilities = gmm.predict_proba(data)
    high = int(np.argmax(gmm.means_.reshape(-1)))
    p_high = responsibilities[:, high]
    if not np.all(np.isfinite(p_high)):
        raise ValueError("non-finite GMM responsibilities")
    p_high_t = torch.tensor(p_high, device=raw.device, dtype=raw.dtype)
    return decay + (1.0 - decay) * p_high_t


def weak_adaptation_signals(
    scores: Dict[str, Dict[str, object]],
    *,
    decay: float = 0.1,
    use_gmm: bool = True,
    random_state: int = 0,
) -> Dict[str, torch.Tensor]:
    """Convert neuron scores into continuous weak-adaptation factors.

    Per hidden layer, fit a GMM over the raw scores and map each neuron's
    high-evidence responsibility to ``[decay, 1]``; fall back to rank-based
    factors when the GMM is unavailable, degenerate, or the layer is too small.
    """
    factors: Dict[str, torch.Tensor] = {}
    for name, info in scores.items():
        raw = info["raw"]
        if use_gmm and _HAS_SKLEARN:
            try:
                factors[name] = _gmm_factors(raw, decay, random_state)
                continue
            except Exception:
                pass
        factors[name] = _rank_factors(raw, decay)
    return factors


def selective_soft_decay(
    dnn,
    factors: Dict[str, torch.Tensor],
    *,
    protected_layers: Tuple[int, ...] = (0,),
) -> Dict[str, torch.Tensor]:
    """Soft-decay input-weight rows and biases of low-scoring neurons in place.

    For each non-protected hidden layer, neuron ``j``'s weight row and bias are
    scaled by its factor ``f[j]`` in ``[decay, 1]`` (soft decay, no pruning).
    Returns the pre-decay per-neuron weight norms for diagnostics.
    """
    protected = set(protected_layers)
    pre_norms: Dict[str, torch.Tensor] = {}
    for position, name, module in iter_hidden_linears(dnn):
        if position in protected or name not in factors:
            continue
        factor = factors[name].to(module.weight.device, module.weight.dtype)
        pre_norms[name] = module.weight.detach().norm(dim=1).clone()
        with torch.no_grad():
            module.weight.mul_(factor.unsqueeze(1))
            module.bias.mul_(factor)
    return pre_norms


def apply_target_guided_reweighting(
    pinn,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    decay: float = 0.1,
    adapt_steps: int = 0,
    protected_layers: Tuple[int, ...] = (0,),
    use_gmm: bool = True,
    residual_based: bool = True,
    random_state: int = 0,
):
    """Apply TGSR target-guided selective reweighting to ``pinn`` in place.

    ``pinn`` is a ``PhysicsInformedNN_pbc`` (must expose ``.dnn`` and
    ``.net_u(x, t)``). The scoring batch ``(x, t)`` is target evidence; in the
    loss-landscape pipeline the model's own collocation tensors
    (``model.x_f``, ``model.t_f``) are used. Returns ``pinn`` for convenience.
    """
    dnn = getattr(pinn, "dnn", None)
    predict_fn = getattr(pinn, "net_u", None)
    if dnn is None or predict_fn is None:
        raise TypeError(
            "pinn must expose .dnn and .net_u(x, t) (PhysicsInformedNN_pbc)"
        )
    residual_fn = getattr(pinn, "net_f", None) if residual_based else None

    for _ in range(int(adapt_steps)):
        if hasattr(pinn, "train"):
            pinn.train()

    dnn.eval()
    scores = compute_neuron_scores(dnn, predict_fn, x, t, residual_fn=residual_fn)
    factors = weak_adaptation_signals(
        scores, decay=decay, use_gmm=use_gmm, random_state=random_state
    )
    selective_soft_decay(dnn, factors, protected_layers=protected_layers)
    dnn.zero_grad(set_to_none=True)
    return pinn
