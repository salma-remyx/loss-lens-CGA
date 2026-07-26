"""Tests for the Wasserstein-to-Gaussian distributional-shape metric.

Covers two things:
  1. the core estimator (new ``wasserstein_shape`` module) behaves as a
     non-Gaussianity criterion -- Gaussian samples score ~0 and strictly
     below clearly non-Gaussian shapes (the paper's strict-inequality
     motivation);
  2. the wiring: the existing ingestion driver ``read_csv_to_db`` invokes
     the new metric and persists the descriptor alongside the barcode
     (exercised with the database layer stubbed out).
"""

import os
import sys

import numpy as np
import pytest

# Make sibling modules importable regardless of pytest's rootdir.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import read_csv_to_db  # noqa: E402  (non-new call-site module)
from wasserstein_shape import (  # noqa: E402
    persistence_barcode_shape,
    wasserstein_squared_to_standard_gaussian,
)


# --------------------------------------------------------------------------- #
# core estimator (new module)
# --------------------------------------------------------------------------- #


def test_gaussian_samples_score_near_zero():
    rng = np.random.default_rng(0)
    gaussian = rng.standard_normal(4000)
    metric = wasserstein_squared_to_standard_gaussian(gaussian)
    assert metric == pytest.approx(0.0, abs=0.05)


def test_non_gaussian_strictly_above_gaussian():
    """The paper's core motivation: non-Gaussian => strictly larger W2-to-Gaussian."""
    rng = np.random.default_rng(1)
    gaussian = rng.standard_normal(4000)
    g_metric = wasserstein_squared_to_standard_gaussian(gaussian)

    # bimodal (two separated clusters) and exponential (heavy one-sided tail)
    bimodal = np.concatenate(
        [rng.standard_normal(2000) - 4.0, rng.standard_normal(2000) + 4.0]
    )
    exponential = rng.exponential(scale=1.0, size=4000)

    b_metric = wasserstein_squared_to_standard_gaussian(bimodal)
    e_metric = wasserstein_squared_to_standard_gaussian(exponential)

    assert b_metric > g_metric
    assert e_metric > g_metric
    # clearly structured, not a rounding fluctuation (Gaussian scores ~1e-3)
    assert b_metric > 0.1
    assert e_metric > 0.1


def test_standardization_is_scale_and_location_invariant():
    """Non-Gaussianity is a shape criterion: shifting/scaling must not change it."""
    rng = np.random.default_rng(2)
    base = rng.exponential(scale=1.0, size=2000)
    metric_base = wasserstein_squared_to_standard_gaussian(base)
    metric_shifted = wasserstein_squared_to_standard_gaussian(100.0 + 7.0 * base)
    assert metric_shifted == pytest.approx(metric_base, rel=1e-9, abs=1e-9)


def test_degenerate_inputs_do_not_crash():
    assert wasserstein_squared_to_standard_gaussian([3.0]) == 0.0
    assert wasserstein_squared_to_standard_gaussian([]) == 0.0
    assert wasserstein_squared_to_standard_gaussian([2.0, 2.0, 2.0]) == 0.0


# --------------------------------------------------------------------------- #
# barcode applier (new module)
# --------------------------------------------------------------------------- #


def test_barcode_descriptor_non_gaussian():
    barcode = [
        {"y0": 0.0, "y1": float(v), "x": float(i)}
        for i, v in enumerate([0, 0, 0, 0, 10, 10, 10, 10])
    ]
    shape = persistence_barcode_shape(barcode)
    assert shape is not None
    assert shape["nSamples"] == 8
    assert shape["valueKey"] == "y1"
    assert shape["wassersteinNonGaussianity"] > 0.0


def test_barcode_descriptor_none_when_insufficient():
    assert persistence_barcode_shape([]) is None
    assert persistence_barcode_shape([{"y0": 1.0, "y1": 2.0}]) is None  # one sample
    # missing / non-finite values are skipped, leaving too few samples
    assert persistence_barcode_shape([{"y0": 1.0}, {"x": 0.0}]) is None


# --------------------------------------------------------------------------- #
# integration: the existing call-site module invokes the new metric
# --------------------------------------------------------------------------- #


def _stub_db(monkeypatch, captured):
    """Stub the MongoDB layer imported into read_csv_to_db (no DB needed)."""
    monkeypatch.setattr(read_csv_to_db, "dbExists", lambda: True)
    monkeypatch.setattr(read_csv_to_db, "collectionExists", lambda name: True)
    monkeypatch.setattr(read_csv_to_db, "createDB", lambda: None)
    monkeypatch.setattr(read_csv_to_db, "createCollection", lambda name: None)

    def fake_add_or_update(collection, query, record):
        captured.append((collection, query, record))

    monkeypatch.setattr(read_csv_to_db, "addOrUpdateDocument", fake_add_or_update)


def test_update_mode_persistence_barcode_persists_shape(monkeypatch):
    captured = []
    _stub_db(monkeypatch, captured)

    barcode = [
        {"y0": 0.0, "y1": float(v), "x": float(i)}
        for i, v in enumerate([0, 1, 2, 3, 20, 21, 22, 23])
    ]
    read_csv_to_db.update_mode_persistence_barcode("pinn", "m1", "s1", barcode)

    assert len(captured) == 1
    collection, query, record = captured[0]
    assert collection == read_csv_to_db.PERSISTENCE_BARCODE
    assert record["edges"] == barcode

    # the wiring computed and attached the shape descriptor
    assert "wassersteinShape" in record
    expected = persistence_barcode_shape(barcode)
    assert record["wassersteinShape"] == expected
    assert record["wassersteinShape"]["wassersteinNonGaussianity"] > 0.0


def test_update_mode_persistence_barcode_omits_shape_when_empty(monkeypatch):
    captured = []
    _stub_db(monkeypatch, captured)

    read_csv_to_db.update_mode_persistence_barcode("pinn", "m1", "s1", [])

    assert len(captured) == 1
    _, _, record = captured[0]
    assert record["edges"] == []
    assert "wassersteinShape" not in record  # graceful: no descriptor for empty input
