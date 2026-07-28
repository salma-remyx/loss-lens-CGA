"""Integration test for the Hopkins topology wiring in ``core_functions``.

Imports the (non-new) ``script_util.core_functions`` module and exercises the
``feature_topology_from_cka`` call-site wiring added alongside
``compute_layer_similarity``. Builds a real torch model, captures its
per-layer activations exactly the way the CKA forward pass does, and asserts
the wiring assembles a bounded per-layer topology score for both models.

Skipped when the backend stack (torch + the repo's model deps) is absent.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")  # noqa: F841 -- skip without backend deps

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:  # core_functions pulls in the full backend stack; skip if incomplete
    from script_util.core_functions import feature_topology_from_cka  # noqa: E402
except ImportError as exc:  # pragma: no cover - depends on local env
    pytest.skip(
        f"core_functions backend stack unavailable: {exc}", allow_module_level=True
    )


class _CkaStub:
    """Minimal stand-in exposing the feature dicts a CKA object carries."""

    def __init__(self, model1_features, model2_features):
        self.model1_features = model1_features
        self.model2_features = model2_features


def _capture_layers(model, x):
    """Run a forward pass and capture each submodule's output via hooks."""
    features = {}
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue

        def hook(_module, _inputs, output, _name=name):
            features[_name] = output

        handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for handle in handles:
        handle.remove()
    return features


def test_feature_topology_from_cka_assembles_per_layer_scalars():
    model1 = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
    )
    model2 = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
    )
    x = torch.randn(32, 4)

    feats1 = _capture_layers(model1, x)
    feats2 = _capture_layers(model2, x)
    stub = _CkaStub(feats1, feats2)

    topology = feature_topology_from_cka(stub)

    assert set(topology) == {"model1", "model2"}
    assert set(topology["model1"]) == set(feats1)
    assert set(topology["model2"]) == set(feats2)
    for branch in topology.values():
        for value in branch.values():
            assert value == value  # not NaN
            assert 0.0 <= value <= 1.0
