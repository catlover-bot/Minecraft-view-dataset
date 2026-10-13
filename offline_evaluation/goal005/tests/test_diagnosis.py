"""Small CPU fixtures for diagnostic semantics, never paid generation."""
import json
import socket

import pytest

from offline_evaluation.goal005 import diagnosis as d
from offline_evaluation.resmoke_v2 import cuboids
from offline_evaluation.server_v2 import core


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU diagnosis attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def fill(block="stone", **changes):
    return dict(op="fill", x1=0, y1=0, z1=0, x2=0, y2=0, z2=0, block=block, **changes)


def test_semantic_invalid_identifier_passes_shape_but_identifies_first_operation():
    actions = [fill(), fill("quartz (white)"), fill("Stone")]
    assert d.schema.validate_document({"operations": actions}) == actions
    found = d.first_local_violation(actions)
    assert found["operation_index_zero_based"] == 1
    assert found["field"] == "block"
    assert found["actual_value"] == "quartz (white)"
    assert found["rule_category"] == "material_identifier_syntax"
    assert found["classification"] == "model_output_violates_predeclared_specification"


def test_frozen_rule_order_coordinates_precede_material_and_touch_is_cumulative():
    action = fill("invalid spaces")
    action["x1"] = 100001
    result = d.first_local_violation([action])
    assert result["field"] == "x1"
    assert result["rule_category"] == "coordinate"
    result = d.first_local_violation([fill(), fill()], core.Limits(max_touched_voxels=1))
    assert result["operation_index_zero_based"] == 1
    assert result["actual_value"] == 2
    assert result["rule_category"] == "total_touched_voxels"


def test_independent_decode_inclusive_origin_and_exact_materials():
    grid = core.Grid({(0, 0, 0): "stone_slab", (0, 0, 1): "stone_slab", (1, 0, 0): "stone"},
                     coordinate_frame="local", origin=(5, -2, 3))
    spec = cuboids.encode_exact(grid)
    assert d.independent_decode(spec) == grid.as_world().voxels
    spec["cuboids"].append(spec["cuboids"][0])
    with pytest.raises(ValueError, match="overlapping"):
        d.independent_decode(spec)


def test_exact_material_confusion_preserves_name_differences_and_counts_missing():
    gt = {(0, 0, 0): "stone_slab", (1, 0, 0): "stone", (2, 0, 0): "stone"}
    pred = {(0, 0, 0): "stone", (1, 0, 0): "stone", (3, 0, 0): "stone"}
    diff = d.voxel_difference(gt, pred)
    assert (diff["add_count"], diff["delete_count"], diff["replace_count"]) == (1, 1, 1)
    assert diff["labelled_voxel_exact"] is False
    assert sum(c["count"] for c in diff["confusion_including_missing_air"]) == 4


def test_same_bounds_wrong_palette_material_and_air_are_reported():
    grid = core.Grid({(0, 0, 0): "stone_slab", (1, 0, 0): "stone"})
    spec = cuboids.encode_exact(grid)
    actions = [{"op": "set", "x": 0, "y": 0, "z": 0, "block": "stone"},
               {"op": "set", "x": 1, "y": 0, "z": 0, "block": "air"}]
    result = d.cuboid_difference(spec, actions, core.execute(actions).voxels)
    assert result["one_exact_same_bounds_material_operation"] == 0
    assert len(result["mismatching_input_cuboids"]) == 2
    assert result["output_materials_not_in_palette"] == ["air"]


def test_common_valid_sets_are_intersected_not_assumed_from_equal_n():
    rows = [{"provider": "openai", "scene_id": s, "condition": c,
             "description_model": "openai", "metrics": {}} for s, c in [("a", "O"), ("b", "J")]]
    result = d.common_valid_oj(rows)["openai"]
    assert result["O_valid_scene_ids"] == ["a"]
    assert result["J_valid_scene_ids"] == ["b"]
    assert result["common_valid_scene_ids"] == []


@pytest.mark.parametrize("status,completed,expected", [("failed", True, 0.0), ("NOT_RUN", False, None), ("failed", False, None)])
def test_only_identified_completed_output_failure_gets_bounded_zero(status, completed, expected):
    row = {"status": status, "attempted": status != "NOT_RUN", "failure_kind": "validation_failure",
           "completed_request_outcome": completed, "metrics": None}
    flat = d.flat_row(row)
    assert flat["aligned_occupancy_iou_failure0"] == expected
    assert flat["aligned_dimension_mae"] is None
    assert flat["aligned_total_normalized_repair_cost"] is None


def test_snapshot_observes_content_change_and_output_does_not_overwrite(tmp_path):
    path = tmp_path / "original.json"
    d.write(path, {"value": 1})
    before = d.snapshot([path])
    with pytest.raises(FileExistsError):
        d.write(path, {"value": 2})
    path.write_text(json.dumps({"value": 2}))
    assert d.snapshot([path]) != before


def archive_fixture(tmp_path):
    archive = tmp_path / "restored"
    root = archive / "original/plan_root"
    enabled = root / "enabled"
    enabled.mkdir(parents=True)
    authority_root = archive / "original/authority"
    authority_root.mkdir()
    identity = {"plan_path": "/canonical/original/plan.json", "implementation_hashes": {}, "limits": {},
                "authority_path": "/canonical/.authority.json", "prompt": "fixture",
                "budgets": {"max_output_tokens": 32768}, "schedule": []}
    entries = []
    for index in range(16):
        record = {"input_id": f"input-{index}", "provider": "openai", "schedule_index": index,
                  "model": "gpt-5-mini", "input_text": "fixture-input"}
        identity["schedule"].append(record)
        operations = [fill("quartz (white)" if index < 2 else "stone")]
        text = json.dumps({"operations": operations})
        raw = {"id": f"response-{index}", "model": "gpt-5-mini-2025-08-07", "status": "completed",
               "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
               "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        path = enabled / "attempts" / f"attempt-{index}"
        path.mkdir(parents=True)
        d.write(path / "raw.json", raw)
        parsed_response = d.adapter.parse_response("openai", (path / "raw.json").read_bytes())
        names = ["raw.json"]
        if index >= 2:
            d.write(path / "parsed.json", {"operations": operations})
            d.write(path / "voxels.json", core.grid_to_payload(core.execute(operations)))
            names += ["parsed.json", "voxels.json"]
        artifacts = {name: {"path": str((path / name).relative_to(enabled)),
                            "sha256": d.sha(path / name), "bytes": (path / name).stat().st_size} for name in names}
        payload = d.adapter.build_payload("openai", record["model"], identity["prompt"], record["input_text"], 32768)
        entries.append({"input_id": record["input_id"], "provider": "openai", "schedule_index": index,
                        "request_payload_hash": d.object_hash(payload), "response_id": raw["id"],
                        "response_model": raw["model"], "natural_stop": True, "usage": parsed_response.usage,
                        "attempt_id": f"attempt-{index}", "artifacts": artifacts,
                        "status": "failed" if index < 2 else "success",
                        "failure_stage": "local_validator" if index < 2 else None})
    identity_hash = d.object_hash(identity)
    approval = {"plan_identity_hash": identity_hash}
    d.write(root / "plan.json", {"identity": identity, "identity_hash": identity_hash})
    d.write(enabled / "ledger.json", {"plan_identity_hash": identity_hash, "entries": entries})
    d.write(enabled / "approval.json", approval)
    d.write(authority_root / ".authority.json", {"plan_identity_hash": identity_hash, "approval_hash": d.object_hash(approval)})
    return archive


def test_archive_verifies_raw14_execution_and2_failures_without_metric_search_or_activation(tmp_path, monkeypatch):
    archive = archive_fixture(tmp_path)
    before = d.snapshot([archive])
    def forbidden(*args, **kwargs):
        pytest.fail("archive verification must not rerun metric search")
    monkeypatch.setattr(core, "evaluate_pair", forbidden)
    result = d.verify_archive(archive)
    assert result["successful_execution_chains"] == 14
    assert len(result["reproduced_failures"]) == 2
    assert result["metric_searches_repeated"] == 0
    assert result["archive_activation_performed"] is False
    assert d.snapshot([archive]) == before


@pytest.mark.parametrize("tamper", ["raw", "payload", "authority"])
def test_archive_rejects_corruption_in_raw_request_or_control(tmp_path, tamper):
    archive = archive_fixture(tmp_path)
    enabled = archive / "original/plan_root/enabled"
    if tamper == "raw":
        path = enabled / "attempts/attempt-0/raw.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif tamper == "payload":
        path = enabled / "ledger.json"
        ledger = d.read(path)
        ledger["entries"][0]["request_payload_hash"] = "0" * 64
        path.write_text(json.dumps(ledger))
    else:
        path = archive / "original/authority/.authority.json"
        authority = d.read(path)
        authority["approval_hash"] = "0" * 64
        path.write_text(json.dumps(authority))
    with pytest.raises(ValueError, match="mismatch"):
        d.verify_archive(archive)
