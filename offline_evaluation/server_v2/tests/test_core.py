import json
from dataclasses import replace

import numpy as np
import pytest

from offline_evaluation.server_v2 import core
from offline_evaluation.server_v2.core import (
    Grid, Limits, ValidationError, evaluate_pair, execute, metrics, parse_actions,
    payload_to_grid, validate_actions,
)
from offline_evaluation.server_v2.data import inspect_npy, load_grid, load_scene


def fill(x1=0, x2=1, material="stone", **kwargs):
    return {"op": "fill", "x1": x1, "y1": 0, "z1": 0, "x2": x2,
            "y2": 0, "z2": 0, "block": material, **kwargs}


def test_inclusive_reversed_negative_fill_and_sequential_deletes():
    actions = [fill(x1=1, x2=-1, z1=0, z2=-1),
               {"op": "set", "x": -1, "y": 0, "z": -1, "block": "glass"},
               {"op": "carve", "x1": 0, "y1": 0, "z1": 0, "x2": 0, "y2": 0, "z2": 0},
               {"op": "set", "x": 1, "y": 0, "z": -1, "block": "minecraft:air"}]
    grid = execute(actions)
    assert grid.voxels == {(-1, 0, -1): "glass", (-1, 0, 0): "stone",
                           (0, 0, -1): "stone", (1, 0, 0): "stone"}
    assert grid.dimensions() == (3, 1, 2)


def test_exact_material_ids_are_not_coarsened():
    result = execute([{"op": "set", "x": x, "y": 0, "z": 0, "block": m}
                      for x, m in enumerate(["minecraft:oak_planks", "planks", "brick_block", "stonebrick"])])
    assert list(result.voxels.values()) == ["oak_planks", "planks", "brick_block", "stonebrick"]


def test_world_local_origin_and_bounds():
    local = execute([fill(0, 1)], coordinate_frame="local", origin=(-5, 7, 2))
    assert local.coordinate_frame == "world"
    assert local.voxels == {(-5, 7, 2): "stone", (-4, 7, 2): "stone"}
    world = execute([fill(0, 1)], origin=(-5, 7, 2))
    assert set(world.voxels) == {(0, 0, 0), (1, 0, 0)}
    with pytest.raises(ValidationError, match="coordinate bound"):
        execute([fill(0, 1)], coordinate_frame="local", origin=(10, 0, 0),
                limits=replace(Limits(), max_coordinate_abs=10))


@pytest.mark.parametrize("text", [
    '{"operations":[],"notes":[]}',
    '{"operations":[],"operations":[]}',
    '[{"op":"set","x":true,"y":0,"z":0,"block":"stone"}]',
    '[{"op":"set","x":0.0,"y":0,"z":0,"block":"stone"}]',
    '[{"op":"set","x":NaN,"y":0,"z":0,"block":"stone"}]',
    '[{"op":"shell","command":"echo hi"}]',
    '[{"op":"set","x":0,"y":0,"z":0,"block":"stone","state":{}}]',
    '```json\n[]\n```',
])
def test_strict_json_and_allowed_fields(text):
    with pytest.raises(ValidationError):
        parse_actions(text)


def test_limits_before_fill_enumeration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("voxel enumeration happened before validation")
    monkeypatch.setattr(core.itertools, "product", forbidden)
    with pytest.raises(ValidationError, match="live voxel limit"):
        execute([fill(0, 1000)], replace(Limits(), max_voxels=10))
    with pytest.raises(ValidationError, match="touch limit"):
        execute([{k: v for k, v in fill(0, 1000).items() if k != "block"} | {"op": "carve"}],
                replace(Limits(), max_touched_voxels=10))
    with pytest.raises(ValidationError, match="unsupported"):
        execute([fill(), {"op": "teleport"}])


def test_operation_coordinate_json_and_accumulated_live_limits():
    with pytest.raises(ValidationError, match="operation limit"):
        validate_actions([fill()], replace(Limits(), max_operations=0))
    with pytest.raises(ValidationError, match="coordinate bound"):
        validate_actions([fill(-20, 0)], replace(Limits(), max_coordinate_abs=10))
    with pytest.raises(ValidationError, match="byte limit"):
        parse_actions('[ ]', replace(Limits(), max_json_bytes=2))
    with pytest.raises(ValidationError, match="accumulated"):
        execute([fill(0, 1), fill(2, 3)], replace(Limits(), max_voxels=3))


def test_independent_numeric_metric_example():
    gt = Grid({(0, 0, 0): "stone", (1, 0, 0): "glass", (3, 0, 0): "wood"})
    pred = Grid({(0, 0, 0): "stone", (1, 0, 0): "brick", (2, 0, 0): "sandstone"})
    result = metrics(gt, pred)
    # Independently: occupancy intersection=2, union=4, labelled match=1.
    assert result["occupancy_iou"] == 1 / 2
    assert result["occupancy_f1"] == 2 / 3
    assert result["dimension_mae"] == 1 / 3
    assert result["material_set_f1"] == 1 / 3
    assert result["material_aware_iou"] == 1 / 5
    assert [result[k] for k in ("add_count", "delete_count", "replace_count")] == [1, 1, 1]
    assert result["total_normalized_repair_cost"] == 1


def test_known_translation_and_same_shift_for_every_metric():
    gt = Grid({(-2, 3, 0): "stone", (0, 4, 1): "glass", (1, 3, 1): "wood"})
    prediction = gt.shifted((1, -1, 1))
    result = evaluate_pair(gt, prediction, alignment_radius=1)
    assert result["absolute"]["occupancy_iou"] == 0
    assert result["alignment"]["shift_xyz"] == [-1, 1, -1]
    assert result["alignment"]["candidates_evaluated"] == 27
    assert result["aligned"]["occupancy_iou"] == 1
    assert result["aligned"]["material_aware_iou"] == 1
    assert result["aligned"]["total_normalized_repair_cost"] == 0


def test_alignment_tie_does_not_choose_material_favorable_shift():
    gt = Grid({(0, 0, 0): "stone", (2, 0, 0): "glass"})
    pred = Grid({(0, 0, 0): "brick", (1, 0, 0): "stone"})
    result = evaluate_pair(gt, pred, 1)
    assert result["alignment"]["shift_xyz"] == [0, 0, 0]
    assert result["aligned"]["material_aware_iou"] == 0
    assert result["aligned"]["replace_count"] == 1
    assert metrics(gt, pred.shifted((-1, 0, 0)))["material_aware_iou"] > 0


def test_prediction_outside_gt_is_not_clamped():
    gt = Grid({(0, 0, 0): "stone"})
    pred = Grid({(0, 0, 0): "stone", (20, 0, 0): "stone"})
    result = evaluate_pair(gt, pred, 0)["absolute"]
    assert result["occupancy_iou"] == 1 / 2
    assert result["delete_count"] == 1
    assert result["dimension_mae"] == 20 / 3


def test_empty_conventions_and_absent_gt_repair_denominator():
    both_empty = metrics(Grid(), Grid())
    for key in ("occupancy_iou", "occupancy_f1", "material_set_f1", "material_aware_iou"):
        assert both_empty[key] == 1
    assert both_empty["dimension_mae"] == both_empty["total_normalized_repair_cost"] == 0
    extra = metrics(Grid(), Grid({(0, 0, 0): "stone", (1, 0, 0): "stone"}))
    assert extra["occupancy_f1"] == 0
    assert extra["total_normalized_repair_cost"] == 2


def test_alignment_work_is_bounded():
    gt = Grid({(0, 0, 0): "stone"})
    with pytest.raises(ValidationError, match="radius"):
        evaluate_pair(gt, gt, 9)
    with pytest.raises(ValidationError, match="work limit"):
        evaluate_pair(gt, gt, 1, replace(Limits(), max_alignment_work=20))


def test_grid_payload_is_deterministic_and_rejects_duplicate_coordinates():
    grid = Grid({(1, 2, 3): "glass", (-1, 0, 2): "stone"}, origin=(-2, 0, -3))
    payload = grid.to_dict()
    assert payload["voxels"] == [[-1, 0, 2, "stone"], [1, 2, 3, "glass"]]
    assert payload_to_grid(payload).to_dict() == payload
    payload["voxels"].append(payload["voxels"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        payload_to_grid(payload)


def write_fixture(tmp_path, array=None, **bbox_updates):
    building = tmp_path / "buildings_100_v1" / "building_999"
    (building / "gt").mkdir(parents=True)
    if array is None:
        array = np.full((2, 3, 4), "air", dtype="U16")
        array[1, 2, 3] = "stone"
    np.save(building / "gt/voxels.npy", array)
    bbox = {"xmin": 10, "xmax": 12, "ymin": -3, "ymax": -2,
            "zmin": 2, "zmax": 5, "voxel_axis_order": "Y,X,Z", **bbox_updates}
    (building / "gt/bbox.json").write_text(json.dumps(bbox), encoding="utf-8")
    (building / "meta.json").write_text('{"generation":{"origin":{"x":100,"y":100,"z":100}}}')
    return building


def test_gt_axis_bbox_world_origin_are_independently_checked(tmp_path):
    building = write_fixture(tmp_path)
    scene = load_scene(building)
    assert scene.scene_id == "buildings_100_v1_building_999"
    assert scene.grid.voxels == {(12, -2, 5): "stone"}
    assert scene.grid.origin == (10, -3, 2)
    assert scene.header["shape_yxz"] == [2, 3, 4]
    assert scene.header["allow_pickle"] is False
    assert load_grid(building / "gt/voxels.npy", building / "gt/bbox.json").voxels == scene.grid.voxels


def test_object_dtype_rejected_before_load(tmp_path, monkeypatch):
    building = write_fixture(tmp_path, np.full((2, 3, 4), "stone", dtype=object))
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("np.load called for object dtype"))
    with pytest.raises(ValidationError, match="pickle"):
        load_scene(building)


@pytest.mark.parametrize("updates", [{"xmax": 13}, {"voxel_axis_order": "X,Y,Z"}, {"xmin": True}])
def test_shape_axes_and_bbox_types_rejected(tmp_path, updates):
    with pytest.raises(ValidationError):
        load_scene(write_fixture(tmp_path, **updates))


def test_npy_storage_limit_and_truncated_payload(tmp_path):
    building = write_fixture(tmp_path)
    path = building / "gt/voxels.npy"
    with pytest.raises(ValidationError, match="cell count"):
        inspect_npy(path, replace(Limits(), max_voxels=23))
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    with pytest.raises(ValidationError, match="truncated"):
        inspect_npy(path)
