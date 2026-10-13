import copy
import json
from dataclasses import replace

import pytest

from offline_evaluation.server_v2.core import Grid, Limits, ValidationError
from offline_evaluation.server_v2.oracle import (
    decode_exact_spec, encode_exact_spec, executor_control, gt_to_actions, partial_oracle,
)


def fixture_grid():
    return Grid({(2, 5, -2): "stone", (2, 5, -1): "stone", (2, 5, 0): "glass",
                 (-1, 6, 0): "wood", (-1, 6, 2): "wood"}, origin=(-3, 4, -4))


def test_exact_spec_expected_runs_and_lossless_deterministic_roundtrip():
    grid = fixture_grid()
    spec = encode_exact_spec(grid)
    assert spec["palette"] == ["glass", "stone", "wood"]
    assert spec["runs"] == [[5, 2, -2, -1, 1], [5, 2, 0, 0, 0],
                            [6, -1, 0, 0, 2], [6, -1, 2, 2, 2]]
    decoded = decode_exact_spec(json.dumps(spec))
    assert decoded.to_dict() == grid.to_dict()
    assert encode_exact_spec(decoded) == spec
    reversed_grid = Grid(dict(reversed(list(grid.voxels.items()))), origin=grid.origin)
    assert encode_exact_spec(reversed_grid) == spec


def test_executor_control_reproduces_every_material_and_position():
    grid = fixture_grid()
    actions = gt_to_actions(grid)
    assert len(actions) == 4
    assert actions[0] == {"op": "fill", "x1": 2, "y1": 5, "z1": -2,
                          "x2": 2, "y2": 5, "z2": -1, "block": "stone"}
    control = executor_control(grid)
    assert control.to_dict() == grid.to_dict()


def test_local_exact_spec_preserves_frame_and_control_maps_to_world():
    grid = Grid({(0, 0, 0): "stone", (0, 0, 1): "glass"}, "local", (-5, 10, 2))
    decoded = decode_exact_spec(encode_exact_spec(grid))
    assert decoded.to_dict() == grid.to_dict()
    assert executor_control(grid).voxels == {(-5, 10, 2): "stone", (-5, 10, 3): "glass"}


def test_empty_exact_spec_and_control():
    spec = encode_exact_spec(Grid())
    assert spec["palette"] == spec["runs"] == []
    assert decode_exact_spec(spec).voxels == executor_control(Grid()).voxels == {}


def test_partial_oracle_is_explicit_diagnostic_incomplete_spec():
    partial = partial_oracle(fixture_grid())
    assert partial["diagnostic_only"] is True
    assert partial["source"] == "ground_truth"
    assert partial["condition"] == "partial_oracle"
    assert partial["occupied_dimensions_xyz"] == [4, 2, 5]
    assert partial["exact_material_set"] == ["glass", "stone", "wood"]
    assert "individual_voxel_coordinates" in partial["not_provided"]
    assert "runs" not in partial


@pytest.mark.parametrize("mutation", [
    lambda s: s["runs"].append([6, -1, 0, 1, 2]),
    lambda s: s["runs"].reverse(),
    lambda s: s["runs"][0].__setitem__(4, 999),
    lambda s: s["runs"][0].__setitem__(2, True),
    lambda s: s["palette"].append("air"),
    lambda s: s.__setitem__("endpoints", "exclusive"),
    lambda s: s.__setitem__("unexpected", "metadata"),
])
def test_decoder_rejects_ambiguous_or_invalid_specs(mutation):
    spec = copy.deepcopy(encode_exact_spec(fixture_grid()))
    mutation(spec)
    with pytest.raises(ValidationError):
        decode_exact_spec(spec)


def test_decoder_preflights_expanded_volume():
    spec = encode_exact_spec(Grid({(0, 0, 0): "stone"}))
    spec["runs"][0][3] = 1000
    with pytest.raises(ValidationError, match="expanded voxels"):
        decode_exact_spec(spec, replace(Limits(), max_voxels=10))


def test_oracle_limits_report_incompatible_instead_of_truncating():
    with pytest.raises(ValidationError, match="operation limit"):
        encode_exact_spec(fixture_grid(), replace(Limits(), max_operations=3))
    with pytest.raises(ValidationError, match="byte limit"):
        encode_exact_spec(fixture_grid(), replace(Limits(), max_json_bytes=20))
    with pytest.raises(ValidationError, match="voxel limit"):
        encode_exact_spec(fixture_grid(), replace(Limits(), max_voxels=4))
