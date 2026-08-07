"""Integration tests for the K-FAC Fisher curvature pipeline.

Covers both the pure K-FAC mechanism (`script_util.kfac_curvature`) and the
wiring that lands it as a sibling curvature metric to the Hessian pipeline in the
existing `script_util.core_functions` module (the call site). The call-site test
monkeypatches `load_mode`/`load_data` so the (model_id, mode_id) entry runs on a
synthetic model + batch without needing trained checkpoints or dataset downloads.
"""

import os
import sys

import pytest

# Match the bootstrap used by the neighbouring test_add_mode_to_db.py so that
# `script_util` resolves regardless of where pytest is invoked from.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

torch = pytest.importorskip("torch")
from script_util.kfac_curvature import kfac_fisher_spectrum
from torch import nn


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 3))


def test_kfac_spectrum_shape_and_order() -> None:
    model = _tiny_model()
    x = torch.randn(8, 6)
    y = torch.randint(0, 3, (8,))
    criterion = nn.CrossEntropyLoss()

    spectrum = kfac_fisher_spectrum(model, lambda m: criterion(m(x), y), top_n=5)

    assert len(spectrum) == 5
    assert all(isinstance(value, float) for value in spectrum)
    assert spectrum == sorted(spectrum, reverse=True)
    assert all(value >= 0.0 for value in spectrum)


def test_kfac_spectrum_scales_quadratically_with_loss() -> None:
    # B = E[g g^T]; scaling the loss by k scales g by k and the spectrum by k^2.
    model = _tiny_model()
    x = torch.randn(8, 6)
    y = torch.randint(0, 3, (8,))
    criterion = nn.CrossEntropyLoss()

    base = kfac_fisher_spectrum(model, lambda m: criterion(m(x), y), top_n=1)[0]
    scaled = kfac_fisher_spectrum(model, lambda m: 3.0 * criterion(m(x), y), top_n=1)[0]

    assert abs(scaled / base - 9.0) < 1e-3


def test_compute_mode_kfac_curvature_wiring() -> None:
    """The new entry in the existing core_functions module runs end-to-end.

    Patches load_mode/load_data so the (model_id, mode_id) contract executes on a
    synthetic model + batch, then asserts a valid Fisher eigenvalue spectrum is
    returned through the call-site wiring.
    """
    try:
        from script_util import core_functions
    except ImportError as exc:  # pragma: no cover - needs the full ML stack
        pytest.skip(f"core_functions import needs full ML stack: {exc!r}")

    assert callable(core_functions.compute_mode_kfac_curvature)

    model = _tiny_model()
    x = torch.randn(8, 6)
    y = torch.randint(0, 3, (8,))
    core_functions.load_mode = lambda model_id, mode_id: model
    core_functions.load_data = lambda model_id, train=False: [(x, y)]

    spectrum = core_functions.compute_mode_kfac_curvature("mnist_mlp", "0")

    assert isinstance(spectrum, list)
    assert 0 < len(spectrum) <= 10
    assert all(isinstance(value, float) for value in spectrum)
    assert spectrum == sorted(spectrum, reverse=True)
