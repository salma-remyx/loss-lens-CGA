"""Integration tests for the Hessian block-diagonal structure analysis.

These exercise the wiring in the (non-new) ``read_csv_to_db`` ingestion module,
which builds the structure record by calling into ``hessian_structure``. They
do not touch the database: ``process_hessian_structure`` is a pure
CSV -> metrics transform, mirroring ``process_persistence_barcode``.
"""

import csv
import os
import sys
import tempfile

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # server/calculate
sys.path.append(parent_dir)

from script_util import hessian_structure  # noqa: E402
from script_util import read_csv_to_db  # noqa: E402


def _write_matrix_csv(path: str, matrix: np.ndarray) -> None:
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        for row in matrix:
            writer.writerow([f"{v:.10f}" for v in row])


# A perfectly block-diagonal 4x4 Hessian: two 2x2 layer blocks, no coupling.
BLOCK_SIZES = [2, 2]
H_BLOCK = np.array(
    [
        [3.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, -1.0],
        [0.0, 0.0, -1.0, 5.0],
    ]
)
# Same diagonal blocks, but with off-diagonal (inter-layer) coupling added.
H_COUPLED = np.array(
    [
        [3.0, 1.0, 0.5, 0.0],
        [1.0, 2.0, 0.0, 0.5],
        [0.5, 0.0, 4.0, -1.0],
        [0.0, 0.5, -1.0, 5.0],
    ]
)


def test_module_metric_block_diagonal_is_one():
    # Direct unit check of the core metric on a perfectly block-diagonal matrix.
    assert hessian_structure.block_diagonal_ratio(H_BLOCK, BLOCK_SIZES) == 1.0
    assert hessian_structure.off_diagonal_coupling(H_BLOCK, BLOCK_SIZES) == 0.0


def test_wiring_ingests_block_diagonal_matrix():
    # The non-new read_csv_to_db module invokes the new module via CSV ingest.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "h.csv")
        _write_matrix_csv(path, H_BLOCK)
        result = read_csv_to_db.process_hessian_structure(path, BLOCK_SIZES)

    assert result["block_diagonal_ratio"] == 1.0
    assert result["off_diagonal_coupling"] == 0.0
    # Two layer blocks -> two per-block spectra.
    assert len(result["layer_block_spectrum"]) == 2


def test_wiring_coupling_lowers_ratio():
    with tempfile.TemporaryDirectory() as tmp:
        coupled_path = os.path.join(tmp, "coupled.csv")
        block_path = os.path.join(tmp, "block.csv")
        _write_matrix_csv(coupled_path, H_COUPLED)
        _write_matrix_csv(block_path, H_BLOCK)
        coupled = read_csv_to_db.process_hessian_structure(coupled_path, BLOCK_SIZES)
        block = read_csv_to_db.process_hessian_structure(block_path, BLOCK_SIZES)

    assert coupled["block_diagonal_ratio"] < block["block_diagonal_ratio"]


def test_wiring_static_dynamic_decomposition():
    # Trained Hessian is more coupled than the random-init (block-diagonal) one,
    # so the static force should exceed the trained ratio and the dynamic force
    # (training's effect) should be negative.
    with tempfile.TemporaryDirectory() as tmp:
        trained_path = os.path.join(tmp, "trained.csv")
        random_path = os.path.join(tmp, "random.csv")
        _write_matrix_csv(trained_path, H_COUPLED)
        _write_matrix_csv(random_path, H_BLOCK)
        result = read_csv_to_db.process_hessian_structure(
            trained_path, BLOCK_SIZES, baseline_csv=random_path
        )

    forces = result["forces"]
    assert forces["static_force"] > forces["trained_ratio"]
    assert forces["dynamic_force"] < 0
