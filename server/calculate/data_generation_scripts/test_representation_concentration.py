"""Tests for the representation-change concentration analysis.

Two layers of coverage:

* Pure tests of ``representation_concentration`` (numpy only) -- the math behind
  the layer-localized concentration and prototype-separation metrics.
* An integration test that drives the EXISTING ``torch_cka`` CKA engine on two
  small models and feeds its real output into the new module, proving the new
  capability consumes the engine's contract (skipped where torch is absent).
"""

import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pytest

from script_util import representation_concentration as rc


def test_concentration_identical_representations_is_zero():
    # CKA == 1 everywhere means identical representations -> no change energy.
    cka = np.ones((6, 6))
    m = rc.concentration_metrics(cka)
    assert m["total_change_energy"] == 0.0
    assert m["concentration_index"] == 0.0
    assert m["final_layer_fraction"] == 0.0


def test_concentration_localizes_final_layer_change():
    # Only the last two model1 layers differ (low CKA) -> change is concentrated
    # in the final layers, the analog of the paper's "92% in final layers".
    cka = np.ones((8, 8))
    cka[-2:, :] = 0.1
    m = rc.concentration_metrics(cka, final_layers=2)
    assert m["final_layer_fraction"] > 0.9
    assert m["concentration_index"] > 0.5
    assert m["dominant_layer_index"] >= 6


def test_concentration_uniform_change_is_not_concentrated():
    # Equal distance at every layer -> flat profile -> ~zero concentration.
    cka = np.full((8, 8), 0.5)
    m = rc.concentration_metrics(cka)
    assert abs(m["concentration_index"]) < 1e-9


def test_change_energy_is_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    cka = rng.random((5, 5))
    profile = rc.representation_change_profile(cka)
    energies = np.asarray(profile["model1_change_energy"])
    assert (energies >= 0.0).all()
    assert (energies <= 1.0 + 1e-9).all()


def test_summary_is_json_serializable_and_carries_names():
    cka = np.ones((4, 4))
    cka[-1, :] = 0.2
    summary = rc.concentration_summary(
        cka, layer_names=["in", "h1", "h2", "out"], final_layers=1
    )
    # Must round-trip through JSON so it can be stored in the layer-similarity
    # document by update_db.
    text = json.dumps(summary)
    restored = json.loads(text)
    assert restored["layer_names"] == ["in", "h1", "h2", "out"]
    assert "metrics" in restored and "profile" in restored
    assert restored["metrics"]["final_layer_fraction"] > 0.5


def test_prototype_separation_extremes():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((30, 5))
    # Same inputs, same model -> coincident centroids -> ~0 separation.
    assert rc.prototype_separation(a, a.copy())["separation"] < 1e-3
    # Shifted centroid relative to fixed spread -> large separation.
    b = a + np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    assert rc.prototype_separation(a, b)["separation"] > 1.0


def test_prototype_separation_by_layer_aligns_common_layers():
    rng = np.random.default_rng(2)
    fa = {"layer_a": rng.standard_normal((20, 4)), "layer_b": rng.standard_normal((20, 3))}
    fb = {"layer_a": rng.standard_normal((20, 4)), "other": rng.standard_normal((20, 3))}
    result = rc.prototype_separation_by_layer(fa, fb)
    # Only layers present in both models are compared.
    assert list(result.keys()) == ["layer_a"]
    assert result["layer_a"] >= 0.0


def test_concentration_wired_through_cka_engine():
    """Integration: the existing torch_cka CKA engine's output feeds the new
    concentration module. Where the full calculate env is importable, also
    confirm the canonical call site (compute_layer_similarity) exposes the
    concentration hook added for this capability (arXiv:2607.21353v1)."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from script_util.torch_cka import cka as torch_cka

    # The call-site module pulls a large dep surface; only assert the hook when
    # that surface is actually importable so this stays runnable in minimal envs.
    try:
        import inspect

        from script_util.core_functions import compute_layer_similarity
    except ImportError:
        compute_layer_similarity = None
    if compute_layer_similarity is not None:
        assert "with_concentration" in inspect.signature(compute_layer_similarity).parameters

    torch.manual_seed(0)

    def toy_model():
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

    model1, model2 = toy_model(), toy_model()
    x = torch.randn(40, 4)
    loader = DataLoader(TensorDataset(x), batch_size=10)

    engine = torch_cka.CKA(model1, model2, device="cpu")
    engine.compare(loader)
    matrix = engine.export()["CKA"].numpy()

    summary = rc.concentration_summary(matrix, final_layers=1)
    assert 0.0 <= summary["metrics"]["concentration_index"] <= 1.0
    assert 0.0 <= summary["metrics"]["final_layer_fraction"] <= 1.0
    assert summary["metrics"]["layer_count"] == matrix.shape[0]

