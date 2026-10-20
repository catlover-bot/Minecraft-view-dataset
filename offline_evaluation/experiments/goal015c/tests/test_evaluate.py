import copy
import json
from types import SimpleNamespace
import sys

import pytest

from offline_evaluation.experiments.goal015c import adapter, evaluate as e


def record(builder="openai", condition=None, index=1):
    return {"call_id": f"fixture_{index:03d}", "builder": builder,
            "condition": condition or adapter.CONDITIONS[0], "repeat_index": 1,
            "wire": None}


def observation(builder, document=None, *, text=None, truncated=False):
    if text is None:
        text = json.dumps(e.EXPECTED_DOCUMENT if document is None else document)
    usage = {"input_tokens": 10, "output_tokens": 20}
    if builder == "openai":
        raw = {"id": "resp_fixture", "model": adapter.MODELS[builder],
               "status": "incomplete" if truncated else "completed",
               "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
               "usage": usage, "output": [{"type": "message", "role": "assistant",
                   "status": "incomplete" if truncated else "completed",
                   "content": [{"type": "output_text", "text": text}]}]}
    else:
        raw = {"id": "msg_fixture", "model": adapter.MODELS[builder], "role": "assistant",
               "type": "message", "usage": usage, "stop_reason": "max_tokens" if truncated else "end_turn",
               "content": [{"type": "text", "text": text}]}
    return {"state": "SETTLED", "send_status": "POST_RETURNED", "usage_known": True,
            "known_cost_usd": "0.01", "http_status": 200,
            "raw_bytes": json.dumps(raw).encode(), "text_bytes": text.encode()}


@pytest.mark.parametrize("builder", adapter.MODELS)
@pytest.mark.parametrize("condition", adapter.CONDITIONS)
def test_three_operations_eight_exact_cells(builder, condition):
    assert adapter.EXPECTED_DOCUMENT == e.EXPECTED_DOCUMENT
    row = e.evaluate_one(record(builder, condition), observation(builder))
    assert row["status"] == "COMPLETE" and row["protocol_valid"]
    assert row["task_pass"] and row["grid_exact"] and row["operation_sequence_conforms"]
    assert row["final_voxel_count"] == 8 and row["independent_executor_matches"]
    assert [row[k] for k in ("set_count", "fill_count", "carve_count")] == [1, 1, 1]
    assert row["execution_trace"]["deleted_occupied_voxels"] == 1


def test_whitespace_and_object_key_order_do_not_change_answer():
    text = json.dumps(e.EXPECTED_DOCUMENT, sort_keys=True, indent=4)
    row = e.evaluate_one(record(), observation("openai", text=text))
    assert row["task_pass"] and row["operation_sequence_conforms"]


@pytest.mark.parametrize("operations", [[], [e.EXPECTED_DOCUMENT["operations"][1]]])
def test_empty_and_carve_only_valid_but_not_task_success(operations):
    row = e.evaluate_one(record(), observation("openai", {"operations": operations}))
    assert row["protocol_valid"] and row["final_voxel_count"] == 0
    assert row["task_pass"] is False and row["operation_sequence_conforms"] is False


def test_exact_grid_is_separate_from_operation_sequence():
    doc = {"operations": [{"op": "set", "x": x, "y": y, "z": z, "block": block}
                          for (x, y, z), block in e.EXPECTED_GRID.items()]}
    row = e.evaluate_one(record(), observation("openai", doc))
    assert row["task_pass"] and row["grid_exact"]
    assert row["operation_sequence_conforms"] is False


def test_material_alias_not_repaired_into_task_success():
    doc = copy.deepcopy(e.EXPECTED_DOCUMENT)
    doc["operations"][0]["block"] = "minecraft:stone"
    row = e.evaluate_one(record(), observation("openai", doc))
    assert row["protocol_valid"] and row["grid_exact"]
    assert row["literal_material_spelling_preserved"] is False and row["task_pass"] is False


@pytest.mark.parametrize("builder", adapter.MODELS)
def test_truncated_json_stays_failure_with_null_grid(builder):
    row = e.evaluate_one(record(builder), observation(builder, text='{"operations":[', truncated=True))
    assert row["status"] == "KNOWN_OUTPUT_FAILURE" and row["truncation"]
    assert row["json_valid"] is False and row["task_pass"] is False
    assert row["final_voxel_count"] is None


@pytest.mark.parametrize("text", ['```json\n{"operations":[]}\n```',
                                 '{"operations":[],"operations":[]}', '{"operations": [',
                                 '{"operations":[{"op":"set","x":true,"y":0,"z":0,"block":"stone"}]}'])
def test_no_fence_removal_json_repair_or_type_coercion(text):
    row = e.evaluate_one(record(), observation("openai", text=text))
    assert row["status"] == "KNOWN_OUTPUT_FAILURE"
    assert row["task_pass"] is False and row["final_voxel_count"] is None


def test_touch_limit_rejection_without_execution(monkeypatch):
    doc = {"operations": [{"op": "carve", "x1": 0, "y1": 0, "z1": 0, "x2": 127, "y2": 127, "z2": 127}]}
    monkeypatch.setattr(e.core, "execute", lambda *a, **k: pytest.fail("invalid program executed"))
    row = e.evaluate_one(record(), observation("openai", doc))
    assert row["touch_limit_rejected"] and row["rejected_operation_index"] == 0
    assert row["failure_kind"] == "validation_failure" and row["final_voxel_count"] is None


@pytest.mark.parametrize("state", ["NOT_SENT", "SENT_PENDING", "UNKNOWN_USAGE"])
def test_unsent_uncertain_not_scored_as_failure(state, monkeypatch):
    monkeypatch.setattr(e, "extract", lambda *a: pytest.fail("uncertain raw evaluated"))
    row = e.evaluate_one(record(), {"state": state, "usage_known": False})
    assert row["status"] == ("NOT_SENT" if state == "NOT_SENT" else "UNKNOWN")
    assert row["task_pass"] is None and row["final_voxel_count"] is None


def test_cpu_error_not_known_model_output_failure(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("fixture CPU failure")
    monkeypatch.setattr(e, "execute_independent", failed)
    row = e.evaluate_one(record(), observation("openai"))
    assert row["status"] == "CPU_EVALUATION_FAILURE" and row["task_pass"] is None


def test_saved_text_mismatch_cannot_be_success():
    obs = observation("openai")
    obs["text_bytes"] = b'{"operations":[]}'
    row = e.evaluate_one(record(), obs)
    assert row["status"] == "CPU_EVALUATION_FAILURE" and row["task_pass"] is None


def test_model_mismatch_operational_not_task_failure():
    obs = observation("openai")
    raw = json.loads(obs["raw_bytes"])
    raw["model"] = "different-model"
    obs["raw_bytes"] = json.dumps(raw).encode()
    row = e.evaluate_one(record(), obs)
    assert row["status"] == "OPERATIONAL_FAILURE" and row["task_pass"] is None


def test_provider_completion_is_separate_from_protocol_validity():
    row = e.evaluate_one(record(), observation("openai", text="not JSON"))
    group = e.summarize([row])["all"]
    assert group["post_returned"] == group["settled_known_usage"] == group["completed_response"] == 1
    assert group["protocol_valid"] == group["strict_success"] == 0


def test_parseable_but_truncated_program_is_never_executed(monkeypatch):
    monkeypatch.setattr(e.core, "execute", lambda *a, **k: pytest.fail("truncated output executed"))
    row = e.evaluate_one(record("anthropic"), observation("anthropic", truncated=True))
    assert row["json_valid"] and row["operation_count"] == 3
    assert row["status"] == "KNOWN_OUTPUT_FAILURE" and row["final_voxel_count"] is None


def test_known_cost_reconciles_against_raw_usage_and_rejects_changed_ledger_cost():
    plan = {"pricing_contract": {"providers": {"openai": {
        "input_usd_per_million": "0.25", "output_usd_per_million": "2"}}}}
    obs = observation("openai")
    obs["known_cost_usd"] = "0.0000425"
    row = e.evaluate_one(record(), obs)
    e._accounting_check(plan, record(), obs, row)
    assert row["known_cost_verified_against_raw_usage"] is True
    obs["known_cost_usd"] = "0.01"
    with pytest.raises(ValueError, match="usage/cost"):
        e._accounting_check(plan, record(), obs, row)


def test_all_planned_conditions_preserved_in_descriptive_summary():
    rows = [e.evaluate_one(record(b, c, i), {"state": "NOT_SENT"})
            for i, (b, c) in enumerate((b, c) for _ in range(2) for b in adapter.MODELS for c in adapter.CONDITIONS)]
    summary = e.summarize(rows)
    assert summary["all"]["planned"] == 12 and summary["all"]["strict_success"] == 0
    assert summary["all"]["known_cost_rows"] == 0
    assert all(g["planned"] == 4 for g in summary["by_condition"])
    assert all(g["planned"] == 2 for g in summary["by_builder_condition"])
