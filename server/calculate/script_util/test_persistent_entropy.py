"""Tests for the persistent-entropy loss-landscape metric.

Covers three layers:
  1. the pure math in ``persistent_entropy`` (Eq. 1 / Eq. 2 of
     arXiv:2509.06694),
  2. the wiring in the existing call-site module ``read_csv_to_db``
     (``persistence_barcode_summary``), and
  3. an end-to-end pass that reads a real TTK persistence-diagram CSV with
     the repo's own ``process_persistence_barcode`` parser and summarises it.
"""

import math
import os
import sys

import pytest

# Make ``server/calculate`` importable, the same way the repo's own scripts do.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from script_util import read_csv_to_db  # non-new, call-site module
from script_util import persistent_entropy  # the new module


# --------------------------------------------------------------------------
# 1. Pure math (Eq. 1 and Eq. 2)
# --------------------------------------------------------------------------


def test_persistent_entropy_two_equal_bars():
    # Two equal-length bars: p_i = 0.5 each -> PE = -ln(0.5) = ln(2).
    pairs = [(0.0, 1.0), (0.0, 1.0)]
    assert math.isclose(
        persistent_entropy.persistent_entropy(pairs), math.log(2), rel_tol=1e-12
    )
    # LWPE weights each term by the absolute persistence (1.0) -> 2 * ln(2).
    assert math.isclose(
        persistent_entropy.length_weighted_persistent_entropy(pairs),
        2.0 * math.log(2),
        rel_tol=1e-12,
    )


def test_single_bar_and_empty_are_zero():
    # A single bar => p_i = 1 => ln(1) = 0 => both summaries are 0.
    assert persistent_entropy.persistent_entropy([(0.0, 5.0)]) == 0.0
    assert persistent_entropy.length_weighted_persistent_entropy([(0.0, 5.0)]) == 0.0
    # No finite bars => defined to be 0.
    assert persistent_entropy.persistent_entropy([]) == 0.0
    assert persistent_entropy.length_weighted_persistent_entropy([]) == 0.0


def test_infinite_and_diagonal_pairs_are_excluded():
    # Birth-only / diagonal (e == b) and infinite-death pairs must be dropped.
    pairs = [(1.0, 1.0), (0.0, float("inf")), (0.0, 1.0)]
    summary = persistent_entropy.persistent_entropy_summary(pairs)
    assert summary["numBars"] == 1.0
    assert summary["totalPersistence"] == 1.0
    assert summary["persistentEntropy"] == 0.0  # single finite bar


def test_lwpe_is_scale_sensitive_while_pe_is_not():
    # The central claim of arXiv:2509.06694 Sec. 3: rescaling the function
    # leaves PE unchanged (it depends only on bar-length proportions) but
    # scales LWPE (each term is weighted by absolute persistence).
    base = [(0.0, 1.0), (0.0, 1.0)]
    scaled = [(0.0, 10.0), (0.0, 10.0)]
    assert math.isclose(
        persistent_entropy.persistent_entropy(base),
        persistent_entropy.persistent_entropy(scaled),
        rel_tol=1e-12,
    )
    assert math.isclose(
        persistent_entropy.length_weighted_persistent_entropy(scaled),
        10.0 * persistent_entropy.length_weighted_persistent_entropy(base),
        rel_tol=1e-12,
    )


# --------------------------------------------------------------------------
# 2. Wiring through the existing call-site module
# --------------------------------------------------------------------------


def test_persistence_barcode_summary_reads_repo_barcode_shape():
    # ``process_persistence_barcode`` emits dicts {"y0", "y1", "x"} where y0 is
    # the birth-axis (Points:0) and y1 is the death-axis (Points:1). Birth rows
    # sit on the diagonal (y1 == y0) and must be ignored.
    barcode = [
        {"y0": 0.0, "y1": 0.0, "x": 0.0},  # birth row
        {"y0": 0.0, "y1": 1.0, "x": 0.0},  # death row -> (0, 1)
        {"y0": 0.0, "y1": 0.0, "x": 0.0},  # birth row
        {"y0": 0.0, "y1": 1.0, "x": 0.0},  # death row -> (0, 1)
    ]
    summary = read_csv_to_db.persistence_barcode_summary(barcode)
    assert set(summary) == {
        "persistentEntropy",
        "lengthWeightedPersistentEntropy",
        "numBars",
        "totalPersistence",
    }
    assert summary["numBars"] == 2.0
    assert summary["totalPersistence"] == 2.0
    assert math.isclose(summary["persistentEntropy"], math.log(2), rel_tol=1e-12)
    assert math.isclose(
        summary["lengthWeightedPersistentEntropy"], 2.0 * math.log(2), rel_tol=1e-12
    )


# --------------------------------------------------------------------------
# 3. End-to-end on a real TTK persistence-diagram CSV
# --------------------------------------------------------------------------


_SAMPLE_PD = os.path.join(
    current_dir,
    "temp_data",
    "loss_landscapes_MT_PD",
    "resnet20_batch_norm_True_residual_False_seed_123456_"
    "net_hessian_False_batch_size_512_distance_0.5_steps_40_"
    "norm_layer_random_normal_PersistenceDiagram.csv",
)


def test_end_to_end_on_real_ttk_csv():
    if not os.path.exists(_SAMPLE_PD):
        pytest.skip("sample persistence-diagram CSV not available in this layout")

    # Drive the repo's own CSV parser, then the wired summary helper.
    barcode = read_csv_to_db.process_persistence_barcode(_SAMPLE_PD)
    summary = read_csv_to_db.persistence_barcode_summary(barcode)

    n = int(summary["numBars"])
    assert n > 1, "expected a non-trivial barcode from the sample loss landscape"
    # PE is bounded above by ln(numBars) (the uniform-bar maximum from the paper).
    assert 0.0 <= summary["persistentEntropy"] <= math.log(n) + 1e-9
    # LWPE weights each term by a positive persistence and p_i in (0, 1) -> >= 0.
    assert summary["lengthWeightedPersistentEntropy"] >= 0.0
    assert summary["totalPersistence"] > 0.0
