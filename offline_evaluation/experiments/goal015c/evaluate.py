"""Offline exact-grid evaluation for the fixed Goal015C miniature task.

No alignment, repair, API calls, statistical tests, or inferred usage.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import uuid

from offline_evaluation.analysis.goal015b.audit import decode, extract
from offline_evaluation.analysis.goal015b.independent import execute_independent, IndependentExecutionError
from offline_evaluation.experiments.goal015 import adapter as frozen_adapter
from offline_evaluation.resmoke_v2.schema import validate_document
from offline_evaluation.server_v2 import core

EXPECTED_DOCUMENT = {"operations": [
    {"op": "fill", "x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 1, "z2": 1, "block": "stone"},
    {"op": "carve", "x1": 0, "y1": 0, "z1": 0, "x2": 0, "y2": 0, "z2": 0},
    {"op": "set", "x": 3, "y": 0, "z": 0, "block": "glass"},
]}
EXPECTED_GRID = {(x, y, z): "stone" for x in range(2) for y in range(2)
                 for z in range(2) if (x, y, z) != (0, 0, 0)}
EXPECTED_GRID[(3, 0, 0)] = "glass"
CONDITIONS = ("NATIVE_CURRENT_ORDER", "NATIVE_ORIGINAL_PROPERTY_ORDER", "TEXT_JSON_INSTRUCTION_ONLY")
IDENTITY = ("call_id", "builder", "condition", "repeat_index")


def descriptor(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def bound(desc):
    raw = Path(desc["path"]).read_bytes()
    if len(raw) != desc["bytes"] or hashlib.sha256(raw).hexdigest() != desc["sha256"]:
        raise ValueError("evaluation source descriptor mismatch")
    return raw


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return descriptor(path)


def _failure(row, status, reason, *, known_output=False):
    row.update(status=status, failure_kind=reason)
    if known_output:
        row.update(protocol_valid=False, task_pass=False, operation_sequence_conforms=False)
    return row


def evaluate_one(record, observation):
    """Evaluate a saved observation; raw_bytes/text_bytes are private memory input.

    NOT_SENT and uncertain usage never become task failures. A known output
    failure has task_pass=False but no invented final grid. CPU failures remain
    unknown. Material namespace/air aliases are not repaired into task success.
    """
    row = {name: record[name] for name in IDENTITY}
    row.update(state=observation.get("state", "NOT_SENT"), status="NOT_SENT",
               send_status=observation.get("send_status", "NOT_SENT"),
               wire=record.get("wire"), raw=observation.get("raw"),
               response_text_path=observation.get("response_text_path"),
               http_status=observation.get("http_status"),
               response_id=None, response_model=None, raw_status=None,
               completed_response=None,
               stop_reason=None, incomplete_details=None, usage=None,
               usage_known=observation.get("usage_known"), known_cost_usd=observation.get("known_cost_usd"),
               known_cost_verified_against_raw_usage=None,
               json_valid=None, protocol_valid=None, operation_count=None,
               set_count=None, fill_count=None, carve_count=None,
               operation_sequence_conforms=None, final_voxel_count=None,
               grid_exact=None, task_pass=None, literal_material_spelling_preserved=None,
               truncation=None, touch_limit_rejected=None, rejected_operation_index=None,
               failure_kind=None, independent_executor_matches=None,
               extracted_text_sha256=None, text_bytes=None, extraction=None,
               grid_sha256=None, execution_trace=None)
    state = row["state"]
    if state == "NOT_SENT":
        return row
    if state in ("SENT_PENDING", "UNKNOWN_USAGE") or observation.get("usage_known") is False:
        return _failure(row, "UNKNOWN", "uncertain_transmission_or_usage")
    if state != "SETTLED":
        return _failure(row, "CPU_EVALUATION_FAILURE", "unsupported_ledger_state")
    raw = observation.get("raw_bytes")
    if not isinstance(raw, bytes):
        return _failure(row, "CPU_EVALUATION_FAILURE", "missing_saved_raw")
    try:
        envelope, text, metadata = extract(record["builder"], raw)
        response = frozen_adapter.parse_response(record["builder"], raw)
    except (ValueError, UnicodeError, frozen_adapter.ProviderError):
        return _failure(row, "KNOWN_OUTPUT_FAILURE", "response_parse_failure", known_output=True)
    row.update(response_id=envelope.get("id"), response_model=envelope.get("model"),
               raw_status=envelope.get("status"), stop_reason=envelope.get("stop_reason"),
               incomplete_details=envelope.get("incomplete_details"), usage=envelope.get("usage"),
               extraction=metadata, completed_response=response.complete)
    if response.text != text:
        return _failure(row, "CPU_EVALUATION_FAILURE", "independent_extraction_disagreement")
    for key, actual in (("response_id", response.response_id), ("response_model", response.model)):
        if observation.get(key) is not None and observation[key] != actual:
            return _failure(row, "CPU_EVALUATION_FAILURE", key + "_binding_mismatch")
    saved_text = observation.get("text_bytes")
    if saved_text is not None and (text is None or text.encode("utf-8") != saved_text):
        return _failure(row, "CPU_EVALUATION_FAILURE", "saved_text_binding_mismatch")
    if response.failure_kind in ("response_model_mismatch", "unexpected_response_feature", "incomplete_response"):
        return _failure(row, "OPERATIONAL_FAILURE", response.failure_kind)
    if text is not None:
        row.update(extracted_text_sha256=hashlib.sha256(text.encode()).hexdigest(), text_bytes=len(text.encode()))
    try:
        document = core.strict_json(text) if isinstance(text, str) else None
        row["json_valid"] = text is not None
    except (core.ValidationError, UnicodeError):
        document = None
        row["json_valid"] = False
    if isinstance(document, dict) and isinstance(document.get("operations"), list):
        ops = document["operations"]
        counts = Counter(op.get("op") for op in ops
                         if isinstance(op, dict) and isinstance(op.get("op"), str))
        row.update(operation_count=len(ops), set_count=counts["set"],
                   fill_count=counts["fill"], carve_count=counts["carve"])
    row["truncation"] = response.failure_kind == "truncation"
    if not response.complete:
        return _failure(row, "KNOWN_OUTPUT_FAILURE", response.failure_kind or "incomplete_output",
                        known_output=True)
    if not row["json_valid"]:
        return _failure(row, "KNOWN_OUTPUT_FAILURE", "json_failure", known_output=True)
    try:
        operations = validate_document(document)
        counts = Counter(op["op"] for op in operations)
        row.update(operation_count=len(operations), set_count=counts["set"],
                   fill_count=counts["fill"], carve_count=counts["carve"])
        # Preserve the original document: the old validator creates normalized copies.
        normalized = core.validate_actions(operations)
        literal_materials = all("block" not in original or original["block"] == norm["block"]
                                for original, norm in zip(operations, normalized))
    except core.ValidationError as exc:
        reason = str(exc)
        match = re.search(r"operation (\d+)", reason)
        row.update(touch_limit_rejected="touch limit" in reason,
                   rejected_operation_index=int(match.group(1)) if match else None,
                   validator_reason=reason)
        return _failure(row, "KNOWN_OUTPUT_FAILURE", "validation_failure", known_output=True)
    try:
        old_grid = core.execute(operations).voxels
        independent_grid, trace = execute_independent(operations)
    except core.ValidationError as exc:
        row["validator_reason"] = str(exc)
        return _failure(row, "KNOWN_OUTPUT_FAILURE", "execution_limit_failure", known_output=True)
    except (IndependentExecutionError, ArithmeticError, MemoryError, RuntimeError, ValueError):
        return _failure(row, "CPU_EVALUATION_FAILURE", "execution_verification_failed")
    if old_grid != independent_grid:
        return _failure(row, "CPU_EVALUATION_FAILURE", "executor_disagreement")
    exact = old_grid == EXPECTED_GRID
    row.update(status="COMPLETE", protocol_valid=True, operation_sequence_conforms=document == EXPECTED_DOCUMENT,
               final_voxel_count=len(old_grid), grid_exact=exact,
               task_pass=exact and literal_materials, literal_material_spelling_preserved=literal_materials,
               truncation=False, touch_limit_rejected=False, independent_executor_matches=True,
               grid_sha256=trace["final_grid_sha256"], execution_trace=trace)
    return row


def summarize(rows):
    """Descriptive counts only, with planned denominators and known-cost coverage."""
    def group(selected, **identity):
        known = [Decimal(str(r["known_cost_usd"])) for r in selected if r["known_cost_usd"] is not None]
        return {**identity, "planned": len(selected),
                "sent_or_possible": sum(r["state"] != "NOT_SENT" for r in selected),
                "post_returned": sum(r["send_status"] == "POST_RETURNED" for r in selected),
                "possibly_sent": sum(r["send_status"] == "POSSIBLY_SENT" for r in selected),
                "settled_known_usage": sum(r["state"] == "SETTLED" for r in selected),
                "completed_response": sum(r["completed_response"] is True for r in selected),
                "protocol_valid": sum(r["protocol_valid"] is True for r in selected),
                "strict_success": sum(r["task_pass"] is True for r in selected),
                "operation_sequence_conforming": sum(r["operation_sequence_conforms"] is True for r in selected),
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "known_cost_usd": str(sum(known, Decimal(0))), "known_cost_rows": len(known)}
    return {"all": group(rows),
            "by_condition": [group([r for r in rows if r["condition"] == c], condition=c) for c in CONDITIONS],
            "by_builder_condition": [group([r for r in rows if r["builder"] == b and r["condition"] == c],
                                          builder=b, condition=c)
                                     for b in ("openai", "anthropic") for c in CONDITIONS]}


def _state(plan_path, plan):
    root = Path(plan["canonical_root"])
    ledger_path = root / "enabled/ledger.json"
    binding = {"plan_id": plan["plan_id"], "plan": descriptor(plan_path), "ledger": None}
    if ledger_path.exists():
        from . import runtime
        enabled = runtime.verify_enabled(plan)
        binding["ledger"] = descriptor(ledger_path)
        ledger = enabled["ledger"]
        if len(ledger["calls"]) != len(plan["schedule"]):
            raise ValueError("evaluation ledger scope mismatch")
        return ledger["calls"], binding
    if any((root / "enabled" / name).exists() for name in ("approval.json", "authority.json")):
        raise ValueError("incomplete enabled evaluation state")
    return [{**record, "state": "NOT_SENT", "send_status": "NOT_SENT"} for record in plan["schedule"]], binding


def _accounting_check(plan, record, saved, row):
    if saved["state"] != "SETTLED":
        return
    classified = frozen_adapter.classify_usage(
        record["builder"], row["usage"], plan["pricing_contract"]["providers"][record["builder"]])
    if (classified["usage_known"] is not True
            or Decimal(str(classified["cost_usd"])) != Decimal(str(saved["known_cost_usd"]))):
        raise ValueError("saved usage/cost binding mismatch")
    row["known_cost_verified_against_raw_usage"] = True


def evaluate_plan(plan_path, output_dir=None):
    from . import runtime
    plan_path = Path(plan_path)
    plan = runtime.verify_plan(plan_path)
    with runtime.locks(plan, create=False):
        observations, binding = _state(plan_path, plan)
        if len(plan["schedule"]) != 12:
            raise ValueError("evaluation requires fixed twelve-call scope")
        rows = []
        for record, saved in zip(plan["schedule"], observations):
            if any(record[k] != saved[k] for k in IDENTITY) or record.get("wire") != saved.get("wire"):
                raise ValueError("evaluation call binding mismatch")
            obs = dict(saved)
            if obs.get("raw"):
                obs["raw_bytes"] = bound(obs["raw"])
            if obs.get("response_text_path"):
                obs["text_bytes"] = bound(obs["response_text_path"])
            row = evaluate_one(record, obs)
            _accounting_check(plan, record, saved, row)
            rows.append(row)
        root = Path(plan["canonical_root"])
        output = Path(output_dir) if output_dir else root / "evaluations" / (
            "goal015c-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
        if not output.resolve().is_relative_to(root.resolve() / "evaluations"):
            raise ValueError("evaluation output must be in canonical evaluations directory")
        output.mkdir(parents=True, exist_ok=False)
        artifact = write_json(output / "evaluation_12.json",
                              {"schema": "goal015c_evaluation_v1", "plan_binding": binding, "rows": rows})
        summary = {"schema": "goal015c_evaluation_summary_v1", "plan_binding": binding,
                   "evaluation_artifact": artifact, "row_count": len(rows), **summarize(rows),
                   "new_research_api_requests_sent": 0,
                   "interpretation": "Descriptive miniature-task diagnostics; no mechanism or generalization proof."}
        summary_artifact = write_json(output / "summary.json", summary)
        return {**summary, "summary_artifact": summary_artifact}


def find_existing_evaluation(plan_path):
    """Reuse only a fully bound evaluation of the exact current ledger."""
    from . import runtime
    plan_path = Path(plan_path)
    plan = runtime.verify_plan(plan_path)
    with runtime.locks(plan, create=False):
        return _find_existing_evaluation_locked(plan_path, plan)


def _find_existing_evaluation_locked(plan_path, plan):
    """Checkpoint internal helper; caller must already hold runtime paired locks."""
    observations, binding = _state(plan_path, plan)
    candidates = sorted((Path(plan["canonical_root"]) / "evaluations").glob("*/summary.json"), reverse=True)
    for path in candidates:
        summary = decode(path.read_bytes())
        if summary.get("schema") != "goal015c_evaluation_summary_v1" or summary.get("plan_binding") != binding:
            continue
        if Path(summary["evaluation_artifact"]["path"]).resolve() != (path.parent / "evaluation_12.json").resolve():
            raise ValueError("cached evaluation path mismatch")
        evaluation = decode(bound(summary["evaluation_artifact"]))
        if evaluation.get("plan_binding") != binding or len(evaluation.get("rows", [])) != 12:
            raise ValueError("cached evaluation binding mismatch")
        for record, saved, row in zip(plan["schedule"], observations, evaluation["rows"]):
            if any(record[k] != row[k] or saved[k] != row[k] for k in IDENTITY):
                raise ValueError("cached evaluation identity mismatch")
            for key in ("raw", "response_text_path"):
                if row.get(key) != saved.get(key):
                    raise ValueError("cached evaluation source mismatch")
                if row.get(key):
                    bound(row[key])
            if (row["state"] != saved["state"] or row.get("wire") != record.get("wire")
                    or row.get("known_cost_usd") != saved.get("known_cost_usd")
                    or row.get("http_status") != saved.get("http_status")):
                raise ValueError("cached evaluation state mismatch")
            if saved["state"] == "SETTLED":
                raw_usage = decode(bound(saved["raw"])).get("usage")
                if row["usage"] != raw_usage or row["known_cost_verified_against_raw_usage"] is not True:
                    raise ValueError("cached raw usage binding mismatch")
                _accounting_check(plan, record, saved, row)
        if any(summary.get(k) != v for k, v in summarize(evaluation["rows"]).items()):
            raise ValueError("cached evaluation aggregation mismatch")
        return {**summary, "summary_artifact": descriptor(path)}
    return None
