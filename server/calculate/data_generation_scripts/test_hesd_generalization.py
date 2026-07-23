"""Tests for the HESD-type-aware generalization assessment.

The pure-logic tests exercise ``hesd_generalization`` directly (NumPy only) and
run anywhere. The wiring test imports the existing ``core_functions`` call-site
module, injects a canned HESD through the shared ``_mode_hessian_comp`` hook,
and asserts that ``compute_mode_generalization`` delegates to the new module --
so it only runs where the repo's full torch stack is importable.
"""

import os
import sys

import pytest

# Make ``script_util`` importable, matching the convention in test_add_mode_to_db.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeHessianComp:
    """Stand-in for a pyhessian hessian object with a canned ``density()``."""

    def __init__(self, eigen_list_full, weight_list_full):
        self._eigen = eigen_list_full
        self._weight = weight_list_full

    def density(self, iter=100, n_v=1):  # noqa: A002 (match pyhessian signature)
        return self._eigen, self._weight


def test_flatten_density_handles_pyhessian_output():
    pytest.importorskip("numpy")
    from script_util.hesd_generalization import flatten_density

    eigenvalues, weights = flatten_density(
        [[1.0 + 0j, 2.0 + 0j], [-3.0 + 0j]],
        [[0.5 + 0j, 0.3 + 0j], [0.2 + 0j]],
    )

    assert eigenvalues.tolist() == [1.0, 2.0, -3.0]
    assert weights.tolist() == [0.5, 0.3, 0.2]


def test_flatten_density_empty():
    pytest.importorskip("numpy")
    from script_util.hesd_generalization import flatten_density

    eigenvalues, weights = flatten_density([], [])
    assert eigenvalues.size == 0 and weights.size == 0


def test_classify_hesd_type_covers_three_regimes():
    pytest.importorskip("numpy")
    import numpy as np
    from script_util.hesd_generalization import classify_hesd_type

    # PSD bulk: spread of positive eigenvalues, no dominant direction.
    psd = classify_hesd_type(
        np.array([1.0, 2.0, 0.5, 1.5, 0.8]), np.array([0.3, 0.1, 0.25, 0.15, 0.2])
    )
    assert psd == "psd_bulk"

    # Indefinite: most mass on negative eigenvalues -> not a minimum.
    indef = classify_hesd_type(
        np.array([-3.0, -1.0, 0.5, 2.0]), np.array([0.3, 0.3, 0.2, 0.2])
    )
    assert indef == "indefinite"

    # Outlier-dominated: a single sharp direction holds most positive curvature.
    outlier = classify_hesd_type(
        np.array([0.01, 0.02, 0.015, 0.012, 50.0]),
        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    )
    assert outlier == "outlier_dominated"

    assert classify_hesd_type(np.array([]), np.array([])) == "unknown"


def test_generalization_criterion_monotonic_in_negative_mass():
    pytest.importorskip("numpy")
    import numpy as np
    from script_util.hesd_generalization import generalization_criterion

    flat = generalization_criterion(
        np.array([2.0, 1.0]), np.array([0.5, 0.5])
    )["generalizationCriterion"]
    with_negative = generalization_criterion(
        np.array([2.0, -1.0]), np.array([0.5, 0.5])
    )["generalizationCriterion"]

    # Same top curvature, but the negative-mass spectrum scores worse (higher).
    assert with_negative > flat


def test_compute_mode_generalization_wiring(monkeypatch):
    """Exercises the call-site edit: core_functions -> hesd_generalization."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from script_util import core_functions

    # Indefinite HESD -> questionable applicability, non-null criterion.
    fake = _FakeHessianComp([[-3.0, -1.0, 0.5, 2.0]], [[0.3, 0.3, 0.2, 0.2]])
    monkeypatch.setattr(core_functions, "_mode_hessian_comp", lambda *args: fake)

    result = core_functions.compute_mode_generalization("any_model", "any_mode")

    assert result["hesdType"] == "indefinite"
    assert result["applicability"] == "questionable"
    assert result["generalizationCriterion"] is not None
    assert result["numEigenvalues"] == 4
