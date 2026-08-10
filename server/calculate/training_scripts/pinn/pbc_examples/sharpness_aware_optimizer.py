"""Sharpness-aware optimizers for the PINN training factory.

Two optimizers that drop into :func:`choose_optimizer.choose_optimizer`:

* :class:`SAMOptimizer` -- Sharpness-Aware Minimization (Foret et al. 2021).
  Each ``step(closure)`` evaluates the closure twice: once to find the worst
  neighbour within a radius-``rho`` ball, and once at that perturbed point to
  obtain the descent gradient. The base optimizer then descends from the
  *original* weights using the perturbed-point gradient, which is what biases
  training towards flatter minima.

* :class:`LatePhaseSAM` -- runs a plain base optimizer for most of training
  and switches to SAM only for the final phase. This delivers the core
  finding of Andriushchenko et al. (2024), "Sharpness-Aware Minimization
  Efficiently Selects Flatter Minima Late in Training" (arXiv:2410.10373):
  SAM's flat-minima selection is concentrated late in training, so a short
  late-phase SAM schedule recovers most of the benefit of running SAM for
  the whole run.

Both expose the closure-based ``step(closure)`` / ``zero_grad()`` interface
that ``PhysicsInformedNN_pbc`` (``net_pbc.py``) already drives, so they need
no new training-loop contracts. Flatness of the resulting minima can then be
checked with the repo's existing Hessian-curvature pipeline
(``hessian_values_pbc.py`` / ``hessian_contour_pbc.py``).

Schedule knobs are read from the environment so the existing
``choose_optimizer(name, params, lr)`` call site is unchanged:

* ``PINN_SAM_TOTAL_ITERS`` (default 2000) -- length of the training run, used
  by :class:`LatePhaseSAM` to locate the late phase.
* ``PINN_SAM_LATE_ITERS`` (default 200) -- how many final steps use SAM.
* ``PINN_SAM_RHO`` (default 0.05) -- SAM neighbourhood radius.
"""

import os

import torch


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class SAMOptimizer(torch.optim.Optimizer):
    """Sharpness-Aware Minimization over a base optimizer.

    Parameters
    ----------
    params : iterable of parameters or parameter-group dicts.
    base_optimizer : a ``torch.optim.Optimizer`` subclass used for the
        descent step (e.g. ``torch.optim.SGD``).
    rho : neighbourhood radius controlling flatness-seeking strength.
        ``None`` resolves to ``PINN_SAM_RHO`` (default ``0.05``).
    **kwargs : forwarded to ``base_optimizer`` (e.g. ``lr``, ``momentum``).
    """

    def __init__(self, params, base_optimizer, rho=None, **kwargs):
        rho = _env_float("PINN_SAM_RHO", 0.05) if rho is None else rho
        if rho <= 0:
            raise ValueError("rho must be positive, got {}".format(rho))
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.base_optimizer.defaults.setdefault("rho", rho)

    @torch.no_grad()
    def _grad_norm(self):
        sq_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    sq_sum += float(p.grad.detach().norm(2)) ** 2
        return sq_sum ** 0.5

    @torch.no_grad()
    def first_step(self):
        """Ascend to the worst neighbour within the rho-ball and stash the step."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        """Restore the original weights and descend using the perturbed gradient."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # undo the ascent
        self.base_optimizer.step()

    def step(self, closure):
        """Run one SAM update, evaluating ``closure`` twice."""
        if closure is None:
            raise ValueError("SAM requires a closure")
        with torch.enable_grad():
            closure()  # gradient at the original weights
        self.first_step()
        with torch.enable_grad():
            loss = closure()  # gradient at the perturbed weights
        self.second_step()
        return loss


class LatePhaseSAM:
    """Plain base optimizer that switches to SAM for the final training phase.

    Delegates every step to a base optimizer until ``total_iters - late_iters``
    steps have run, then delegates to an internal :class:`SAMOptimizer` for the
    last ``late_iters`` steps. This is the late-phase flat-minima selection of
    Andriushchenko et al. (2024): even a few SAM steps late in training push
    the solution towards a flatter minimum.

    Parameters
    ----------
    params : iterable of parameters.
    base_optimizer : base optimizer class used both for the plain phase and as
        the descent step inside SAM.
    lr : learning rate forwarded to both phases.
    rho : SAM radius (``None`` -> ``PINN_SAM_RHO``).
    total_iters : total run length (``None`` -> ``PINN_SAM_TOTAL_ITERS``).
    late_iters : number of final steps run as SAM (``None`` ->
        ``PINN_SAM_LATE_ITERS``).
    **kwargs : extra base-optimizer kwargs (e.g. ``momentum``).
    """

    def __init__(self, params, base_optimizer, lr=1e-3, rho=None,
                 total_iters=None, late_iters=None, **kwargs):
        self.total_iters = total_iters if total_iters is not None else _env_int("PINN_SAM_TOTAL_ITERS", 2000)
        self.late_iters = late_iters if late_iters is not None else _env_int("PINN_SAM_LATE_ITERS", 200)
        if self.late_iters <= 0:
            raise ValueError("late_iters must be positive")
        if self.late_iters >= self.total_iters:
            raise ValueError("late_iters must be smaller than total_iters")
        self._step_count = 0
        param_list = list(params)  # materialize once; the caller passes a generator
        self.base = base_optimizer(param_list, lr=lr, **kwargs)
        self.sam = SAMOptimizer(param_list, base_optimizer, rho=rho, lr=lr, **kwargs)

    def _in_late_phase(self):
        return self._step_count >= self.total_iters - self.late_iters

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def defaults(self):
        return self.base.defaults

    def zero_grad(self, set_to_none=True):
        # base and sam reference the same parameter tensors, so zeroing via
        # the base clears the shared .grad tensors used by both phases.
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "step_count": self._step_count,
            "total_iters": self.total_iters,
            "late_iters": self.late_iters,
            "base": self.base.state_dict(),
            "sam": self.sam.state_dict(),
        }

    def step(self, closure):
        if self._in_late_phase():
            optimizer = self.sam
        else:
            optimizer = self.base
        self._step_count += 1
        return optimizer.step(closure)
