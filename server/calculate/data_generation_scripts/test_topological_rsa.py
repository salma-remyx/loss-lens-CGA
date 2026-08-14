import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Imports from the existing (non-new) call-site module, proving the wiring
# edit lands inside the repo's real layer-similarity pipeline.
from script_util import core_functions  # noqa: E402


def _tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def _tiny_loader():
    torch.manual_seed(123)
    x = torch.randn(48, 4)
    y = torch.zeros(48, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def _patch_pipeline(monkeypatch):
    """Drive compute_layer_similarity without the DB / dataset download path."""
    shared = _tiny_model()
    monkeypatch.setattr(core_functions, "load_mode", lambda *a, **k: shared)
    monkeypatch.setattr(core_functions, "load_data", lambda *a, **k: _tiny_loader())
    return shared


def test_compute_layer_similarity_returns_topological_matrix(monkeypatch):
    """The opt-in flag wires in the tRSA analyzer and returns both the
    geometric CKA grid and the topological tRSA grid."""
    _patch_pipeline(monkeypatch)

    result = core_functions.compute_layer_similarity(
        "mnist_mlp", "mnist_mlp", "m0", "m1", topological=True
    )

    assert isinstance(result, dict)
    assert set(result) == {"CKA", "tRSA"}
    cka, trsa = result["CKA"], result["tRSA"]

    # The topological grid shares the layer-pair shape of the CKA grid.
    assert len(cka) == len(trsa) and len(cka) > 0
    assert all(len(row) == len(trsa) for row in trsa)

    n = len(trsa)
    # Identical models: a layer's topological RDM correlates perfectly with
    # itself on the diagonal, and the matrix is symmetric.
    for i in range(n):
        assert abs(trsa[i][i] - 1.0) < 1e-6
    for i in range(n):
        for j in range(n):
            assert abs(trsa[i][j] - trsa[j][i]) < 1e-9


def test_compute_layer_similarity_default_contract_unchanged(monkeypatch):
    """Without the opt-in flag the function keeps its original contract:
    a bare list-of-lists (the CKA grid), so existing update_db callers are
    unaffected."""
    _patch_pipeline(monkeypatch)

    grid = core_functions.compute_layer_similarity(
        "mnist_mlp", "mnist_mlp", "m0", "m1"
    )

    assert isinstance(grid, list)
    assert grid and all(isinstance(row, list) for row in grid)
