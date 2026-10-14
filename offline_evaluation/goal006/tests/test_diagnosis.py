"""CPU-only fixtures for the new saved-result diagnostic semantics."""
import socket

import pytest

from offline_evaluation.goal006 import diagnosis as d
from offline_evaluation.server_v2 import core


@pytest.fixture(autouse=True)
def network_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("research network forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def row(scene="a", condition="O", description="openai", provider="openai", value=0.5, failure=None):
    metrics = {frame: {"occupancy_iou": value, "material_aware_iou": value,
                      "dimension_mae": value, "material_set_f1": value,
                      "total_normalized_repair_cost": value} for frame in ("absolute", "aligned")}
    return {"scene_id": scene, "condition": condition, "description_model": description,
            "provider": provider, "status": "failed" if failure else "success", "failure_kind": failure,
            "metrics": None if failure else metrics, "attempted": True, "completed_request_outcome": True}


def spec():
    return {"provided_attributes": ["occupied_dimensions_xyz", "world_bbox", "exact_material_set", "voxel_count"],
            "occupied_dimensions_xyz": [3, 1, 1],
            "world_bbox": {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0},
            "exact_material_set": ["stone", "brick_block"], "voxel_count": 2}


def test_p_matching_dimensions_and_materials_do_not_imply_matching_voxel_count():
    found = d.p_attributes(spec(), core.Grid({(0, 0, 0): "stone", (1, 0, 0): "stone", (2, 0, 0): "brick_block"}))
    assert found["attribute_satisfied"] == {"occupied_dimensions_xyz": True, "world_bbox": True,
                                            "exact_material_set": True, "voxel_count": False}
    assert found["signed_errors_predicted_minus_input"]["voxel_count"] == 1
    assert found["all_four_attributes_satisfied"] is False


def test_p_signed_bbox_material_presence_and_world_offset():
    grid = core.Grid({(0, 0, 0): "stone", (2, 0, 0): "glass"}, coordinate_frame="local", origin=(5, 0, 0))
    found = d.p_attributes(spec(), grid)
    assert found["signed_errors_predicted_minus_input"]["world_bbox"]["xmin"] == 5
    assert found["signed_errors_predicted_minus_input"]["material_presence"] == {"brick_block": -1, "glass": 1, "stone": 0}
    assert found["attribute_satisfied"]["voxel_count"] is True
    assert found["all_four_attributes_satisfied"] is False


def test_p_rejects_undeclared_attribute_contract():
    changed = spec(); changed["provided_attributes"].remove("voxel_count")
    with pytest.raises(ValueError, match="four-attribute"):
        d.p_attributes(changed, core.Grid({(0, 0, 0): "stone"}))


def test_placement_evidence_uses_last_write_without_executor(monkeypatch):
    monkeypatch.setattr(core, "execute", lambda *args, **kwargs: pytest.fail("must not execute"))
    actions = [{"op": "fill", "x1": 0, "x2": 1, "y1": 0, "y2": 0, "z1": 0, "z2": 0, "block": "stone"},
               {"op": "set", "x": 1, "y": 0, "z": 0, "block": "glass"}]
    found = d.placement_evidence({(0, 0, 0): "stone"}, {(0, 0, 0): "stone", (1, 0, 0): "glass"}, actions)
    assert found["examples"][0]["last_operation_index_zero_based"] == 1
    assert found["examples"][0]["GT_material"] == "air"


def test_building_mean_does_not_treat_descriptions_as_buildings():
    rows = [row(value=0), row(description="gemini", value=0), row(scene="b", value=1)]
    found = d.building_summary(rows, "aligned_occupancy_iou")
    assert found["equal_building_mean"] == 0.5
    assert found["buildings"] == 2 and found["eligible_rows"] == 3


def test_oj_group_difference_is_separate_from_common_valid_paired_difference():
    rows = [row(value=1), row(condition="J", value=0.5),
            row(description="gemini", value=0), row(condition="J", description="gemini", failure="validation_failure")]
    found = d.denominators(rows)
    contrast = next(x for x in found["OJ_common_valid_contrasts"] if x["provider"] == "openai" and x["metric"] == "aligned_occupancy_iou")
    assert contrast["unpaired_valid_group_mean_difference_J_minus_O"] == 0
    assert contrast["common_valid_paired_equal_building_mean_J_minus_O"] == -0.5
    assert contrast["common_valid_pair_count"] == 1


def test_failure0_rejects_unknown_or_unattempted_as_zero():
    completed = row(failure="truncation")
    unknown = {**completed, "completed_request_outcome": False}
    unattempted = {**completed, "status": "NOT_RUN", "attempted": False}
    found = d.building_summary([completed, unknown, unattempted], "aligned_occupancy_iou", failure0=True)
    assert found["eligible_rows"] == 1 and found["equal_building_mean"] == 0
    assert d.metric_value(completed, "aligned_dimension_mae") is None


def test_e_common_valid_is_intersection_and_failure0_keeps_all_attempts():
    rows = [row(scene="a", condition="E", description=None, provider="openai"),
            row(scene="b", condition="E", description=None, provider="openai", failure="truncation"),
            row(scene="a", condition="E", description=None, provider="anthropic", failure="truncation"),
            row(scene="b", condition="E", description=None, provider="anthropic")]
    details = [{**r, "absolute": {"occupancy_exact": True, "labelled_voxel_exact": False},
                "aligned_exact": {"occupancy_exact": True, "labelled_voxel_exact": False}} for r in rows]
    found = d.e_summary(rows, details)
    assert found["common_valid_buildings"] == 0
    group = found["groups"][0]
    assert group["valid_buildings"] == 1 and group["planned_buildings"] == 2
    assert group["absolute_occupancy_exact"] == 1 and group["absolute_labelled_voxel_exact"] == 0
    assert group["metrics"]["aligned_occupancy_iou"]["failure0"]["equal_building_mean"] == 0.25


def test_representatives_are_first_success_per_provider_condition_not_highest_score():
    rows = [row(value=0.1), row(value=1), row(provider="anthropic", failure="truncation"), row(provider="anthropic", value=0.2)]
    assert d.representatives(rows, rows) == [0, 3]


def test_actual_truncation_reason_not_reference_cap_inference():
    entry = {"charged_input_tokens": 5, "charged_output_tokens": 32768, "usage": {"output_tokens": 32768}}
    raw = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
           "usage": {"output_tokens": 32768, "output_tokens_details": {"reasoning_tokens": 384}}}
    reference = {"reference_action_tokens_local_o200k_base": 100, "reference_body_within_output_cap_local_openai": True}
    found = d.truncation_evidence("openai", raw, entry, reference)
    assert found["provider_stop_reason"] == {"reason": "max_output_tokens"}
    assert found["provider_reasoning_tokens"] == 384
    assert found["reference_body_within_output_cap_local_openai"] is True


def test_inventory_change_is_rejected_without_reexecution(tmp_path):
    path = tmp_path / "raw.json"; path.write_text("{}")
    inventory = [{"path": str(path), "sha256": d.sha(path), "bytes": 2}]
    d.check_inventory(inventory)
    path.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        d.check_inventory(inventory)
