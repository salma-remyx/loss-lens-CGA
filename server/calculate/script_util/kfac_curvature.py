"""Kronecker-Factored Approximate Curvature (K-FAC) Fisher eigenvalue spectrum.

Adapted from Eschenhagen et al., "Kronecker-Factored Approximate Curvature for
Modern Neural Network Architectures" (arxiv:2311.00636).

K-FAC approximates a layer's Fisher information matrix -- a curvature object --
as the Kronecker product of two small factors estimated from forward
activations and backward gradients::

    F_layer ~= E[g g^T] (x) E[a a^T] = B (x) A

where ``a`` is the input activation to the layer and ``g = dL/dz`` is the
gradient w.r.t. its pre-activation output. The paper's central contribution is a
framework that reduces weight-shared operations (convolutions, attention) to
this same linear form; the canonical linear-layer factorisation is the building
block every architecture hosted here (MLP, ViT, ResNet, PINN) is assembled from,
and is what the weight-sharing reduction targets. We apply that canonical case.

Because the eigenvalues of a Kronecker product are the pairwise products of the
factors' eigenvalues, a layer's K-FAC curvature spectrum is cheap to obtain
(eigendecompose two small matrices and take their outer product) relative to the
exact Fisher / Hessian, and it is the standard complementary curvature object to
the Hessian eigenvalue spectrum already produced by ``compute_mode_hessian``.

Implementation scope (Mode 2 adapted port): the core K-FAC factorisation and
Fisher eigenvalue spectrum are implemented at full fidelity. The paper's generic
weight-sharing framework for arbitrary conv/attention layouts, its second-order
optimizer, and its separate benchmark suite are intentionally out of scope --
this module supplies the curvature *metric* (the Fisher eigenvalue spectrum)
that slots into the existing ``load_mode``/``load_data`` -> eigenvalues curvature
pipeline, not the training method. Evaluation against the paper's reported
optimization-speedup numbers belongs in a downstream PR.
"""

from collections.abc import Callable

import torch
from torch import nn

LossFn = Callable[[nn.Module], torch.Tensor]


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Average a matrix with its transpose so eigvalsh sees a symmetric input."""
    return 0.5 * (matrix + matrix.t())


def _linear_kfac_factors(
    inputs: torch.Tensor, grad_outputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical K-FAC factors A and B for one linear layer over a mini-batch.

    ``inputs`` is the layer input activation ``a`` with leading batch axis;
    ``grad_outputs`` is the pre-activation gradient ``g = dL/dz``. A bias unit is
    appended to ``a`` so the input factor spans the full parameter block.
    Returns (A, B) with A in ``[in+1, in+1]`` and B in ``[out, out]``.
    """
    a = inputs.reshape(inputs.shape[0], -1).detach()
    g = grad_outputs.reshape(grad_outputs.shape[0], -1).detach()
    ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
    a = torch.cat([a, ones], dim=1)
    batch = a.shape[0]
    A = (a.t() @ a) / batch
    B = (g.t() @ g) / batch
    return A, B


def _layer_spectrum(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of the Kronecker block ``B (x) A`` as a flat 1-D tensor.

    They are the pairwise products of the eigenvalues of B and of A. Gram-style
    factors are PSD, so their eigenvalues are non-negative in theory; we take the
    absolute value to absorb tiny numerical negatives from eigvalsh and report
    curvature magnitudes consistently.
    """
    eig_A = torch.linalg.eigvalsh(_symmetrize(A.double()))
    eig_B = torch.linalg.eigvalsh(_symmetrize(B.double()))
    return torch.outer(eig_B, eig_A).reshape(-1).abs()


def kfac_fisher_spectrum(
    model: nn.Module, loss_fn: LossFn, top_n: int = 10
) -> list[float]:
    """Top-N K-FAC Fisher curvature eigenvalues across ``model``'s linear layers.

    Runs one forward + backward pass; for each ``nn.Linear`` layer it captures the
    input activation ``a`` (forward hook) and the pre-activation gradient ``g``
    (full backward hook), forms that layer's Fisher block as ``B (x) A``, and
    collects its eigenvalues. Returns the ``top_n`` largest eigenvalues across all
    layers, sorted descending -- mirroring the contract of
    ``compute_mode_hessian``.
    """
    model.zero_grad(set_to_none=True)
    captured: dict[int, dict[str, torch.Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def forward_hook(module: nn.Module, inputs: tuple, _output: torch.Tensor) -> None:
        captured.setdefault(id(module), {})["a"] = inputs[0]

    def backward_hook(
        module: nn.Module, _grad_input: tuple, grad_output: tuple
    ) -> None:
        captured[id(module)]["g"] = grad_output[0]

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_hook(backward_hook))

    try:
        loss = loss_fn(model)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    spectra: list[float] = []
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        seen = captured.get(id(module))
        if seen is None or "a" not in seen or "g" not in seen:
            continue
        A, B = _linear_kfac_factors(seen["a"], seen["g"])
        spectra.extend(_layer_spectrum(A, B).tolist())

    if not spectra:
        return []
    spectra.sort(reverse=True)
    return [float(value) for value in spectra[:top_n]]
