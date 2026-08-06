"""Integration tests for the condition-number metric (the kappa-LoRA signal).

Covers (a) the numpy spectral core in ``condition_number_metric`` and (b) the
storage contract used by ``update_mode_condition_number``, which writes into the
``semi-global-local-structure`` collection exposed by the existing
``database.db_util`` module. The torch-backed wrappers in ``core_functions`` /
``update_db`` are exercised via guarded imports: they only run where the full
model stack is importable; elsewhere they skip, since this repo's compute
functions (like ``compute_mode_hessian``) require trained-model artifacts.
"""

import json
import os
import sys

import numpy as np
import pytest

# Make ``script_util`` / ``database`` importable regardless of cwd, mirroring the
# sys.path manipulation used by the other scripts in this directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # .../server/calculate
sys.path.insert(0, parent_dir)

from script_util import condition_number_metric  # new module under test


def test_condition_number_matches_definition():
    # diag([1, 1000]) -> kappa = sigma_max / sigma_min = 1000 / 1
    assert condition_number_metric.condition_number(np.diag([1.0, 1000.0])) == pytest.approx(
        1000.0
    )
    # a scaled identity is perfectly balanced across directions -> kappa = 1
    assert condition_number_metric.condition_number(5.0 * np.eye(3)) == pytest.approx(1.0)


def test_condition_number_rejects_non_2d():
    with pytest.raises(ValueError):
        condition_number_metric.condition_number(np.zeros((2, 2, 2)))


def test_high_condition_number_matrices_rank_first_and_are_selected():
    # kappa-LoRA's central claim: larger condition number => more critical.
    well_balanced = np.eye(4)  # kappa = 1
    critical = np.diag([1.0, 1e3, 1.0, 1.0])  # kappa = 1000
    mid = np.diag([1.0, 5.0, 1.0, 1.0])  # kappa = 5
    result = condition_number_metric.analyze(
        [("well_balanced", well_balanced), ("critical", critical), ("mid", mid)]
    )

    ranked = [entry["layer"] for entry in result["layers"]]
    assert ranked[0] == "critical"  # most underdeveloped matrix ranks first
    assert ranked[-1] == "well_balanced"

    selection = result["selection"]
    assert selection["selected"][0] == "critical"  # top pick is the high-kappa matrix
    assert selection["fraction"] == 0.5
    assert selection["selectedCount"] == 2  # round(3 * 0.5) = 2
    assert selection["totalCount"] == 3


def test_rank_deficient_matrix_is_flagged_as_critical():
    # A singular matrix (sigma_min = 0) is maximally critical -> inf -> None.
    result = condition_number_metric.analyze([("singular", np.array([[1.0, 1.0], [1.0, 1.0]]))])
    entry = result["layers"][0]
    assert entry["conditionNumber"] is None
    assert entry["singularMin"] == pytest.approx(0.0)


def test_metric_uses_real_db_collection_contract():
    # update_mode_condition_number stores the metric under a node's
    # "conditionNumber" field in the existing semi-global-local-structure
    # collection; verify the structure round-trips through JSON and that the
    # real collection constant from the existing db_util module is in play.
    try:
        from database import db_util
    except Exception as exc:  # missing pymongo / credentials -> skip, not fail
        pytest.skip(f"database.db_util not importable here: {exc}")

    result = condition_number_metric.analyze(
        [("a", np.diag([1.0, 2.0])), ("b", np.diag([1.0, 50.0]))]
    )
    node = {"modelId": "test_model", "modeId": "0", "conditionNumber": result}
    serialized = json.loads(json.dumps(node))  # must not raise (MongoDB storage)
    assert serialized["conditionNumber"]["selection"]["totalCount"] == 2
    assert db_util.SEMI_GLOBAL_LOCAL_STRUCTURE == "semi-global-local-structure"


def test_compute_mode_condition_number_is_wired():
    # Guarded: only runs where the full torch / model stack is importable.
    try:
        from script_util.core_functions import compute_mode_condition_number
    except Exception as exc:  # missing torch / pinn stack -> skip, not fail
        pytest.skip(f"core_functions not importable here: {exc}")
    assert callable(compute_mode_condition_number)


def test_update_mode_condition_number_is_wired():
    try:
        from script_util.update_db import update_mode_condition_number
    except Exception as exc:  # missing torch / pinn stack -> skip, not fail
        pytest.skip(f"update_db not importable here: {exc}")
    assert callable(update_mode_condition_number)
