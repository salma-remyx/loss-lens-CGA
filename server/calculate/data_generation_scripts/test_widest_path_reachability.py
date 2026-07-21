"""Tests for widest-path reachability analysis (adapted from WPRF, arXiv:2607.07123).

The first two tests exercise the core mechanism directly (stdlib-only, so they
run anywhere). The third test imports the NON-NEW call-site module
``script_util.ttk_functions`` (the confirmed ``server/calculate/script_util/``
landing zone) and drives ``compute_widest_path_reachability`` -- the descriptor
wired into ``compute_merge_tree_planar`` -- proving the integration.
"""

import os
import sys

import pytest

calc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, calc_dir)

# Two low-loss basins (-5) separated by a high-loss saddle (9); border at 4.
TWO_BASINS_WITH_SADDLE = [
    [4, 4, 4, 4, 4],
    [4, -5, 9, -5, 4],
    [4, 4, 4, 4, 4],
]


def test_reachability_localizes_the_saddle_bottleneck():
    """Core mechanism: widest-path reachability is lowest at the saddle that
    separates two basins -- the connectivity bottleneck (the TGS target)."""
    from script_util.widest_path_reachability import (
        bottleneck_pixels,
        widest_path_reachability_descriptor,
        widest_path_reachability_field,
    )

    reach = widest_path_reachability_field(TWO_BASINS_WITH_SADDLE, maximize=False)
    # Loss-landscape polarity: capacity = -loss, seed = a global minimum, so the
    # high-loss saddle is the hardest pixel to reach with a low-loss path.
    assert reach[1][1] > reach[1][2]  # basin more reachable than the saddle
    assert reach[1][3] > reach[1][2]
    assert reach[1][2] == min(min(row) for row in reach)  # saddle is the global bottleneck

    bottlenecks = [tuple(p) for p in bottleneck_pixels(TWO_BASINS_WITH_SADDLE, maximize=False, fraction=0.2)]
    assert (1, 2) in bottlenecks

    desc = widest_path_reachability_descriptor(TWO_BASINS_WITH_SADDLE, maximize=False)
    assert 0.0 <= desc["connectivity"] <= 1.0
    assert desc["bottleneckValue"] == pytest.approx(-9.0)


def test_maximize_recovers_the_papers_segmentation_polarity():
    """maximize=True is the paper's setting: capacity is the field value, so a
    low-value cell bottlenecks reachability from a high-value source."""
    from script_util.widest_path_reachability import widest_path_reachability_field

    field = [
        [0.2, 0.9, 0.2],
        [0.2, 0.2, 0.2],
    ]
    reach = widest_path_reachability_field(field, source=(0, 1), maximize=True)
    assert reach[0][1] == pytest.approx(0.9)  # source keeps its own value
    assert reach[1][1] == pytest.approx(min(0.9, 0.2))  # capped by the low cell


def test_compute_widest_path_reachability_is_reachable_from_ttk_functions():
    """Integration: the descriptor is wired into the existing call-site module
    ``script_util.ttk_functions`` (companion to ``compute_merge_tree_planar``)."""
    pytest.importorskip("numpy")
    try:
        from script_util.ttk_functions import compute_widest_path_reachability
    except ModuleNotFoundError as exc:
        pytest.skip(f"ttk_functions dependencies unavailable: {exc.name}")

    desc = compute_widest_path_reachability(loss_landscape=TWO_BASINS_WITH_SADDLE)
    assert desc is not None
    assert set(desc) == {
        "reachabilityField",
        "bottleneckValue",
        "bottleneckPixels",
        "connectivity",
    }
    assert desc["bottleneckValue"] == pytest.approx(-9.0)
    assert [1, 2] in desc["bottleneckPixels"]

    # The same field supplied flat + grid steps (how turn_landscape_to_csv calls
    # the planar pipeline) must agree with the 2-D form.
    import numpy as np

    flat = np.array(TWO_BASINS_WITH_SADDLE, dtype=float).reshape(-1)
    desc_flat = compute_widest_path_reachability(
        loss_values=flat, loss_steps_dim1=3, loss_steps_dim2=5
    )
    assert np.allclose(
        np.array(desc_flat["reachabilityField"]),
        np.array(desc["reachabilityField"]),
    )
