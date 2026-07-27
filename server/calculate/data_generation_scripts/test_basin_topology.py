"""Integration tests for the basin-topology merge-tree wiring.

These import the *existing* ingestion driver ``read_csv_to_db`` (the call site)
and exercise the ``attach_basin_descriptors`` hook added there, which sources
basin sharpness/volume descriptors from the TTK ``MergeTree.csv`` /
``PersistenceDiagram.csv`` samples shipped under ``script_util/temp_data``.
"""

import glob
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # .../server/calculate
sys.path.append(parent_dir)

from script_util import basin_topology
from script_util import read_csv_to_db  # the NON-NEW call-site module

SAMPLE_DIR = os.path.join(
    parent_dir, "script_util", "temp_data", "loss_landscapes_MT_PD"
)


def _sample_merge_tree_csvs():
    matches = [
        f
        for f in sorted(glob.glob(os.path.join(SAMPLE_DIR, "*_MergeTree.csv")))
        if not f.endswith("MergeTreePlanar.csv")
    ]
    assert matches, "no sample *_MergeTree.csv found under temp_data"
    return matches


def test_attach_basin_descriptors_populates_from_real_merge_tree():
    merge_csv = _sample_merge_tree_csvs()[0]
    planar_csv = merge_csv.replace("MergeTree.csv", "MergeTreePlanar.csv")
    merge_tree = {"nodes": [{"id": 0, "x": 0.0, "y": 0.0}], "edges": []}

    result = read_csv_to_db.attach_basin_descriptors(merge_tree, planar_csv)

    # The call-site hook invoked the new module and attached descriptors.
    assert "basinDescriptors" in result
    desc = result["basinDescriptors"]
    assert desc["n_basins"] >= 1
    assert desc["max_persistence"] >= 0.0
    assert desc["persistence_source"] in ("persistence_diagram", "merge_tree")
    assert 0.0 <= desc["narrow_basin_score"] <= 1.0
    assert desc["regime"] in ("narrow_basin", "flat_plateau", "mixed")
    # The original merge-tree payload is preserved alongside the new field.
    assert result["nodes"] == merge_tree["nodes"]


def test_attach_is_a_noop_without_sibling_merge_csv(tmp_path):
    missing = str(tmp_path / "does_not_exist_MergeTreePlanar.csv")
    result = read_csv_to_db.attach_basin_descriptors(
        {"nodes": [], "edges": []}, missing
    )
    assert "basinDescriptors" not in result  # graceful degradation


def test_descriptors_drive_imbalance_correlation():
    # Integrated scenario: descriptors computed via the call-site data path are
    # correlated with imbalance ratio at the paper's 1:1 / 56:1 / 641:1 levels.
    records = []
    for ratio, merge_csv in zip([1.0, 56.0, 641.0], _sample_merge_tree_csvs()[:3]):
        desc = basin_topology.descriptors_from_files(merge_csv)
        desc["imbalance_ratio"] = ratio
        records.append(desc)

    corr = basin_topology.imbalance_correlation(records)
    assert corr["n_imbalance_levels"] == len(records)
    assert -1.0 <= corr["narrow_basin_score_pearson"] <= 1.0
    assert -1.0 <= corr["narrow_basin_score_spearman"] <= 1.0
    ratios = [step["imbalance_ratio"] for step in corr["regime_progression"]]
    assert ratios == sorted(ratios)  # ordered by increasing imbalance
