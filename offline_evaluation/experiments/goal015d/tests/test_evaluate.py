import copy
import json

import pytest

from offline_evaluation.experiments.goal015d import evaluate as e

DOCUMENT = {"operations": [
    {"op": "fill", "x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 1, "z2": 1, "block": "stone"},
    {"op": "carve", "x1": 0, "y1": 0, "z1": 0, "x2": 0, "y2": 0, "z2": 0},
    {"op": "set", "x": 3, "y": 0, "z": 0, "block": "glass"},
]}
MODELS = {"openai": "gpt-5-mini-2025-08-07", "anthropic": "claude-haiku-4-5-20251001"}


def record(builder="openai", condition=None, index=1, scene="scene1", repeat=1, pair="pair1"):
    return {"call_id": f"fixture_{index:03d}", "scene_id": scene, "pair_id": pair,
            "builder": builder, "condition": condition or e.CONDITIONS[0],
            "repeat_index": repeat, "wire": None}


def observation(builder="openai", document=None, *, text=None, truncated=False):
    text = json.dumps(DOCUMENT if document is None else document) if text is None else text
    usage = {"input_tokens": 10, "output_tokens": 20}
    if builder == "openai":
        raw = {"id": "resp_fixture", "model": MODELS[builder],
               "status": "incomplete" if truncated else "completed",
               "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
               "usage": usage, "output": [{"type": "message", "role": "assistant",
                  "status": "incomplete" if truncated else "completed",
                  "content": [{"type": "output_text", "text": text}]}]}
    else:
        raw = {"id": "msg_fixture", "model": MODELS[builder], "role": "assistant",
               "type": "message", "usage": usage, "stop_reason": "max_tokens" if truncated else "end_turn",
               "content": [{"type": "text", "text": text}]}
    return {"state": "SETTLED", "send_status": "POST_RETURNED", "usage_known": True,
            "known_cost_usd": "0.0000425" if builder == "openai" else "0.000110",
            "http_status": 200, "raw_bytes": json.dumps(raw).encode(), "text_bytes": text.encode()}


def frame():
    rows = []
    for scene in ("scene1", "scene2"):
        for builder in MODELS:
            for repeat in (1, 2):
                for condition in e.CONDITIONS:
                    rows.append(record(builder, condition, len(rows)+1, scene, repeat,
                                       f"{scene}-{builder}-{repeat}"))
    return rows


@pytest.mark.parametrize("builder", MODELS)
@pytest.mark.parametrize("condition", e.CONDITIONS)
def test_mixed_program_uses_frozen_metrics_and_independent_trace(builder, condition):
    gt = e.core.execute(DOCUMENT["operations"])
    row = e.evaluate_one(record(builder, condition), observation(builder), gt)
    assert row["protocol_valid"] and row["nonempty"] and row["final_voxel_count"] == 8
    assert row["independent_executor_matches"] and row["write_operation_present"]
    assert row["non_air_write_observed"] and not row["carve_only"]
    assert [row[k] for k in ("set_count", "fill_count", "carve_count")] == [1, 1, 1]
    assert row["execution_trace"]["deleted_occupied_voxels"] == 1
    assert row["metrics"] == e.frozen_metrics.evaluate_grids(gt, gt)
    assert row["metrics"]["values"]["aligned_occupancy_iou"] == 1
    assert row["metrics"]["values"]["aligned_surface_f1_at_1"] == 1
    assert "task_pass" not in row and "grid_exact" not in row


def test_translated_material_error_shares_original_occupancy_shift():
    gt = e.core.execute(DOCUMENT["operations"])
    doc = copy.deepcopy(DOCUMENT)
    for op in doc["operations"]:
        for key in ("x", "x1", "x2"):
            if key in op:
                op[key] += 7
    doc["operations"][2]["block"] = "gold_block"
    row = e.evaluate_one(record(), observation(document=doc), gt)
    values = row["metrics"]["values"]
    # Frozen material IoU counts the wrong material as a missed and an extra item.
    assert values["aligned_occupancy_iou"] == 1 and values["aligned_material_aware_iou"] == 7/9
    assert values["aligned_occupancy_precision"] == values["aligned_occupancy_recall"] == 1
    assert values["aligned_dimension_mae"] == 0
    assert row["metrics"]["extended_provenance"]["stored_alignment_shift_xyz"] == row["metrics"]["alignment"]["shift_xyz"]
    assert row["metrics"]["extended_provenance"]["alignment_searched"] is False


@pytest.mark.parametrize("operations,carve_only", [([], False), ([DOCUMENT["operations"][1]], True)])
def test_valid_empty_is_not_a_missing_or_failed_grid(operations, carve_only):
    gt = e.core.execute(DOCUMENT["operations"])
    row = e.evaluate_one(record(), observation(document={"operations": operations}), gt)
    assert row["protocol_valid"] and row["final_voxel_count"] == 0 and row["empty_output"]
    assert not row["nonempty"] and row["carve_only"] is carve_only
    assert not row["model_output_failure"] and row["metrics"] is not None
    assert row["metrics"]["values"]["aligned_occupancy_iou"] == 0
    assert row["metrics"]["values"]["aligned_occupancy_precision"] is None


@pytest.mark.parametrize("text", ['```json\n{"operations":[]}\n```', '{"operations":[',
                                  '{"operations":[],"operations":[]}',
                                  '{"operations":[{"op":"set","x":true,"y":0,"z":0,"block":"stone"}]}'])
def test_strict_failure_is_never_repaired_or_given_grid(text):
    row = e.evaluate_one(record(), observation(text=text), lambda: pytest.fail("failure read GT"))
    assert row["status"] == "KNOWN_OUTPUT_FAILURE" and row["model_output_failure"]
    assert row["protocol_valid"] is False and row["metrics"] is None
    assert row["final_voxel_count"] is None and row["nonempty"] is None


@pytest.mark.parametrize("builder", MODELS)
@pytest.mark.parametrize("text", [None, '{"operations":['])
def test_truncation_never_executes_even_when_prefix_is_valid(builder, text, monkeypatch):
    monkeypatch.setattr(e.core, "execute", lambda *a, **k: pytest.fail("truncated output executed"))
    row = e.evaluate_one(record(builder), observation(builder, text=text, truncated=True))
    assert row["truncation"] and row["model_output_failure"] and row["final_voxel_count"] is None
    assert row["operation_count"] == (3 if text is None else None)


def test_two_million_touch_guard_records_index_without_enumeration(monkeypatch):
    monkeypatch.setattr(e.core, "execute", lambda *a, **k: pytest.fail("invalid output executed"))
    doc = {"operations": [{"op": "carve", "x1": 0, "y1": 0, "z1": 0,
                           "x2": 127, "y2": 127, "z2": 127}]}
    row = e.evaluate_one(record(), observation(document=doc))
    assert row["touch_limit_rejected"] and row["rejected_operation_index"] == 0
    assert row["carve_only"] and row["carve_count"] == 1 and row["final_voxel_count"] is None


@pytest.mark.parametrize("state", ["NOT_SENT", "SENT_PENDING", "UNKNOWN_USAGE"])
def test_not_sent_and_unknown_remain_null_without_extraction_or_gt(state, monkeypatch):
    monkeypatch.setattr(e, "extract", lambda *a: pytest.fail("uncertain raw evaluated"))
    row = e.evaluate_one(record(), {"state": state, "usage_known": False},
                         lambda: pytest.fail("uncertain output read GT"))
    assert row["status"] == ("NOT_SENT" if state == "NOT_SENT" else "UNKNOWN")
    assert row["protocol_valid"] is None and row["final_voxel_count"] is None and row["metrics"] is None


def test_executor_disagreement_is_local_unknown(monkeypatch):
    monkeypatch.setattr(e, "execute_independent", lambda *a: ({}, {}))
    row = e.evaluate_one(record(), observation())
    assert row["status"] == "CPU_EVALUATION_FAILURE" and not row["model_output_failure"]
    assert row["final_voxel_count"] is None


def test_raw_text_binding_mismatch_and_model_contract_mismatch():
    obs = observation()
    obs["text_bytes"] = b"changed"
    assert e.evaluate_one(record(), obs)["failure_kind"] == "saved_text_binding_mismatch"
    obs = observation()
    raw = json.loads(obs["raw_bytes"])
    raw["model"] = "unknown-alias"
    obs["raw_bytes"] = json.dumps(raw).encode()
    row = e.evaluate_one(record(), obs)
    assert row["status"] == "OPERATIONAL_FAILURE" and row["metrics"] is None


def test_http_error_body_is_not_evaluated_even_if_it_looks_like_success(monkeypatch):
    obs = observation()
    obs.update(http_status=503, failure_kind="http_failure", completed_response=False)
    monkeypatch.setattr(e, "extract", lambda *a: pytest.fail("HTTP error evaluated"))
    row = e.evaluate_one(record(), obs)
    assert row["status"] == "OPERATIONAL_FAILURE" and row["metrics"] is None
    assert row["protocol_valid"] is None and row["final_voxel_count"] is None
    plan = {"pricing_contract": {"providers": {"openai": {
        "input_usd_per_million": "0.25", "output_usd_per_million": "2"}}}}
    e._accounting_check(plan, record(), obs, row)
    assert row["known_cost_verified_against_raw_usage"] is True


def test_all_16_rows_eight_cells_and_eight_pairs_preserve_missing():
    rows = [e.evaluate_one(r, {"state": "NOT_SENT"}) for r in frame()]
    gt = e.core.execute(DOCUMENT["operations"])
    rows[0] = e.evaluate_one(frame()[0], observation(document={"operations": []}), gt)
    rows[1] = e.evaluate_one(frame()[1], observation(), gt)
    rows[2] = e.evaluate_one(frame()[2], observation(text="failure"), gt)
    result = e.summarize(rows)
    assert result["all"]["planned"] == 16 and len(result["by_cell"]) == len(result["paired_differences"]) == 8
    assert all(cell["planned"] == 2 for cell in result["by_cell"])
    first = result["paired_differences"][0]["metrics"]
    assert first["aligned_occupancy_iou"]["B_minus_A"] == 1
    assert first["aligned_occupancy_precision"]["B_minus_A"] is None
    assert first["aligned_occupancy_precision"]["A_null_reason"]
    assert all(metric["B_minus_A"] is None for metric in result["paired_differences"][1]["metrics"].values())
    assert result["all"]["known_output_failure"] == 1 and result["all"]["normal_empty"] == 1
    assert result["inference"] == {"bootstrap": False, "p_values": False, "ranking": False, "failure0_imputation": False}


def test_pair_identity_and_missing_scheduled_row_rejected():
    rows = [e.evaluate_one(r, {"state": "NOT_SENT"}) for r in frame()]
    with pytest.raises(ValueError, match="sixteen"):
        e.summarize(rows[:-1])
    rows[1]["pair_id"] = "wrong"
    with pytest.raises(ValueError, match="adjacent"):
        e.summarize(rows)


def test_actual_plan_not_sent_evaluation_is_read_only(tmp_path):
    from offline_evaluation.experiments.goal015d.tests.test_runtime import build_fixture
    path = build_fixture(tmp_path)
    summary = e.evaluate_plan(path)
    assert summary["row_count"] == 16 and summary["all"]["known_cost_rows"] == 0
    assert not (path.parent / "enabled").exists() and not (path.parent / "execution.lock").exists()
    assert e.find_existing_evaluation(path) == summary
    csv_path = summary["files"]["cells_csv"]["path"]
    with open(csv_path, "a") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError, match="source descriptor"):
        e.find_existing_evaluation(path)


def test_gt_tamper_rejected_before_use(tmp_path):
    from offline_evaluation.experiments.goal015d.tests.test_runtime import build_fixture
    path = build_fixture(tmp_path)
    plan = json.loads(path.read_text())
    scene = plan["schedule"][0]["scene_id"]
    source = plan["gt_sources"][scene]
    from pathlib import Path
    p = Path(source["building_directory"]) / "gt/bbox.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="GT bytes"):
        e._gt_loader(plan)(scene)
