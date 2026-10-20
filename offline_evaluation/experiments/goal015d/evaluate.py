"""Strict saved-output evaluation for the Goal015D two-building canary.

Frozen Goal015 geometry metrics, one shared occupancy-selected shift; no repair,
miniature task scoring, failure-zero imputation, statistical tests, or transport.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import csv
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
from offline_evaluation.server_v2.alignment import AlignmentResourceLimit
from offline_evaluation.experiments.goal015 import evaluate as frozen_metrics

CONDITIONS = ("NATIVE_CURRENT_ORDER", "NATIVE_ORIGINAL_PROPERTY_ORDER")
IDENTITY = ("call_id", "scene_id", "pair_id", "builder", "condition", "repeat_index")
METRICS = frozen_metrics.METRICS


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
    row.update(status=status, failure_kind=reason, metric_null_reason=reason)
    if known_output:
        row.update(protocol_valid=False, model_output_failure=True, metric_null_reason=reason)
    return row


def evaluate_one(record, observation, gt=None):
    """Evaluate strict raw→text→operations→grid, with a lazy evaluation-only GT.

    Missing/uncertain/failed output has no invented grid or zero-valued metric.
    A valid empty grid remains valid and uses the frozen metric null rules.
    """
    row = {name: record[name] for name in IDENTITY}
    row.update(state=observation.get("state", "NOT_SENT"), status="NOT_SENT",
               send_status=observation.get("send_status", "NOT_SENT"),
               response_received=observation.get("response_received",
                                                 observation.get("send_status") == "POST_RETURNED"),
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
               final_voxel_count=None, nonempty=None, empty_output=None,
               write_operation_present=None, non_air_write_observed=None, carve_only=None,
               model_output_failure=False, metrics=None, evaluation_status="NOT_COMPUTED",
               metric_null_reason="NOT_SENT", gt_source=record.get("gt_source"),
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
        envelope_usage = decode(raw)
        if isinstance(envelope_usage, dict):
            row["usage"] = envelope_usage.get("usage")
    except (ValueError, UnicodeError):
        pass
    if row["http_status"] != 200 or observation.get("failure_kind") == "http_failure":
        row["completed_response"] = False
        return _failure(row, "OPERATIONAL_FAILURE", "http_failure")
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
                   fill_count=counts["fill"], carve_count=counts["carve"],
                   write_operation_present=bool(counts["set"] + counts["fill"]),
                   carve_only=bool(ops) and counts["carve"] == len(ops))
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
        core.validate_actions(operations)
        row.update(write_operation_present=bool(counts["set"] + counts["fill"]),
                   carve_only=bool(operations) and counts["carve"] == len(operations))
    except core.ValidationError as exc:
        reason = str(exc)
        match = re.search(r"operation (\d+)", reason)
        row.update(touch_limit_rejected="touch limit" in reason,
                   rejected_operation_index=int(match.group(1)) if match else None,
                   validator_reason=reason)
        return _failure(row, "KNOWN_OUTPUT_FAILURE", "validation_failure", known_output=True)
    try:
        prediction = core.execute(operations)
        old_grid = prediction.voxels
        independent_grid, trace = execute_independent(operations)
    except core.ValidationError as exc:
        row["validator_reason"] = str(exc)
        return _failure(row, "KNOWN_OUTPUT_FAILURE", "execution_limit_failure", known_output=True)
    except (IndependentExecutionError, ArithmeticError, MemoryError, RuntimeError, ValueError):
        return _failure(row, "CPU_EVALUATION_FAILURE", "execution_verification_failed")
    if old_grid != independent_grid:
        return _failure(row, "CPU_EVALUATION_FAILURE", "executor_disagreement")
    row.update(status="COMPLETE", protocol_valid=True,
               final_voxel_count=len(old_grid), nonempty=bool(old_grid), empty_output=not bool(old_grid),
               non_air_write_observed=trace["non_air_write_visits"] > 0,
               truncation=False, touch_limit_rejected=False, independent_executor_matches=True,
               grid_sha256=trace["final_grid_sha256"], execution_trace=trace,
               metric_null_reason="GT_NOT_AVAILABLE")
    if gt is not None:
        try:
            truth = gt() if callable(gt) else gt
            if not isinstance(truth, core.Grid):
                raise ValueError("evaluation GT must use the frozen Grid reader")
            row["metrics"] = frozen_metrics.evaluate_grids(truth, prediction)
            row.update(evaluation_status="PASS", metric_null_reason=None)
        except AlignmentResourceLimit as exc:
            row.update(evaluation_status="ALIGNMENT_RESOURCE_LIMIT", metric_null_reason="ALIGNMENT_RESOURCE_LIMIT",
                       evaluation_failure_metadata=exc.metadata)
        except (frozen_metrics.extended.MetricResourceLimit, MemoryError, OSError, core.ValidationError) as exc:
            row.update(evaluation_status="EVALUATION_UNAVAILABLE", metric_null_reason=type(exc).__name__)
    return row


def summarize(rows):
    """Eight planned cells and eight adjacent B−A pairs; no inferential statistics."""
    if len(rows) != 16 or len({r["call_id"] for r in rows}) != 16:
        raise ValueError("exact sixteen-row evaluation frame required")
    scenes = list(dict.fromkeys(r["scene_id"] for r in rows))
    keys = {(r["scene_id"], r["builder"], r["condition"], r["repeat_index"]) for r in rows}
    expected = {(s, b, c, n) for s in scenes for b in ("openai", "anthropic")
                for c in CONDITIONS for n in (1, 2)}
    if len(scenes) != 2 or len(keys) != 16 or keys != expected:
        raise ValueError("fixed two-scene/Builder/condition/repeat frame differs")
    def group(selected, **identity):
        known = [Decimal(str(r["known_cost_usd"])) for r in selected if r["known_cost_usd"] is not None]
        return {**identity, "planned": len(selected),
                "sent_or_possible": sum(r["state"] != "NOT_SENT" for r in selected),
                "post_returned": sum(r["send_status"] == "POST_RETURNED" for r in selected),
                "possibly_sent": sum(r["send_status"] == "POSSIBLY_SENT" for r in selected),
                "settled_known_usage": sum(r["state"] == "SETTLED" for r in selected),
                "completed_response": sum(r["completed_response"] is True for r in selected),
                "protocol_valid": sum(r["protocol_valid"] is True for r in selected),
                "nonempty": sum(r["nonempty"] is True for r in selected),
                "normal_empty": sum(r["protocol_valid"] is True and r["empty_output"] is True for r in selected),
                "carve_only": sum(r["carve_only"] is True for r in selected),
                "known_output_failure": sum(r["model_output_failure"] is True for r in selected),
                "unknown_or_operational_failure": sum(r["status"] in ("UNKNOWN", "OPERATIONAL_FAILURE") for r in selected),
                "cpu_failure": sum(r["status"] == "CPU_EVALUATION_FAILURE" or r["evaluation_status"] in
                                   ("ALIGNMENT_RESOURCE_LIMIT", "EVALUATION_UNAVAILABLE") for r in selected),
                "metric_rows": sum(r["metrics"] is not None for r in selected),
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "known_cost_usd": str(sum(known, Decimal(0))), "known_cost_rows": len(known)}
    pairs = []
    for index in range(0, 16, 2):
        adjacent = rows[index:index+2]
        a, b = sorted(adjacent, key=lambda r: CONDITIONS.index(r["condition"]))
        if (a["condition"], b["condition"]) != CONDITIONS or any(
                a[k] != b[k] for k in ("pair_id", "scene_id", "builder", "repeat_index")):
            raise ValueError("adjacent A/B pair differs")
        values = {}
        for metric in METRICS:
            av = a["metrics"]["values"][metric] if a["metrics"] is not None else None
            bv = b["metrics"]["values"][metric] if b["metrics"] is not None else None
            def reason(row, value):
                if value is not None:
                    return None
                return (row["metrics"]["null_reasons"].get(metric, "UNDEFINED_SUCCESS_METRIC")
                        if row["metrics"] is not None else row["metric_null_reason"])
            values[metric] = {"A": av, "B": bv, "B_minus_A": bv-av if av is not None and bv is not None else None,
                              "A_null_reason": reason(a, av), "B_null_reason": reason(b, bv)}
        pairs.append({**{k: a[k] for k in ("pair_id", "scene_id", "builder", "repeat_index")},
                      "A_call_id": a["call_id"], "B_call_id": b["call_id"], "metrics": values,
                      "failure0_applied": False})
    if len({p["pair_id"] for p in pairs}) != 8:
        raise ValueError("exactly eight unique adjacent pair IDs required")
    return {"all": group(rows), "independent_buildings": 2,
            "by_cell": [group([r for r in rows if (r["scene_id"], r["builder"], r["condition"]) == (s, b, c)],
                              scene_id=s, builder=b, condition=c)
                        for s in scenes for b in ("openai", "anthropic") for c in CONDITIONS],
            "paired_differences": pairs,
            "inference": {"bootstrap": False, "p_values": False, "ranking": False, "failure0_imputation": False}}


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


def _gt_loader(plan):
    from offline_evaluation.server_v2.data import load_scene
    sources = plan["gt_sources"]
    if set(sources) != {r["scene_id"] for r in plan["schedule"]}:
        raise ValueError("all planned GT source descriptors required")
    grids = {}

    def load_gt(scene):
        if scene not in grids:
            source = sources[scene]
            building = Path(source["building_directory"])
            if not building.is_absolute() or any(p.is_symlink() for p in (building, *building.parents)):
                raise ValueError("absolute nonsymlink original GT source required")
            for relative, field in (("gt/voxels.npy", "voxels_sha256"), ("gt/bbox.json", "bbox_sha256")):
                path = building / relative
                if path.is_symlink() or descriptor(path)["sha256"] != source[field]:
                    raise ValueError("original GT bytes differ")
            loaded = load_scene(building)
            if loaded.scene_id != scene or frozen_metrics.frozen_grid_sha256(loaded.grid) != source["grid_sha256"]:
                raise ValueError("original GT normalized grid or scene differs")
            grids[scene] = loaded.grid
        return grids[scene]
    return load_gt


def _csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NULL" if row.get(key) is None else
                             json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
                             if isinstance(row[key], (dict, list, bool)) else row[key] for key in keys})
    return descriptor(path)


def evaluate_plan(plan_path, output_dir=None):
    from . import runtime
    plan_path = Path(plan_path)
    plan = runtime.verify_plan(plan_path)
    with runtime.locks(plan, create=False):
        observations, binding = _state(plan_path, plan)
        if len(plan["schedule"]) != 16:
            raise ValueError("evaluation requires fixed sixteen-call scope")
        load_gt = _gt_loader(plan)
        rows = []
        for record, saved in zip(plan["schedule"], observations):
            if any(record[k] != saved[k] for k in IDENTITY) or record.get("wire") != saved.get("wire"):
                raise ValueError("evaluation call binding mismatch")
            obs = dict(saved)
            if obs.get("raw"):
                obs["raw_bytes"] = bound(obs["raw"])
            if obs.get("response_text_path"):
                obs["text_bytes"] = bound(obs["response_text_path"])
            row = evaluate_one(record, obs, lambda: load_gt(record["scene_id"]))
            row["gt_source"] = plan["gt_sources"][record["scene_id"]]
            _accounting_check(plan, record, saved, row)
            rows.append(row)
        root = Path(plan["canonical_root"])
        output = Path(output_dir) if output_dir else root / "evaluations" / (
            "goal015d-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
        if not output.resolve().is_relative_to(root.resolve() / "evaluations"):
            raise ValueError("evaluation output must be in canonical evaluations directory")
        output.mkdir(parents=True, exist_ok=False)
        artifact = write_json(output / "evaluation_16.json",
                              {"schema": "goal015d_evaluation_v1", "plan_binding": binding, "rows": rows})
        aggregates = summarize(rows)
        files = {
            "rows_csv": _csv(output / "results_16.csv", rows),
            "cells_csv": _csv(output / "cells_8.csv", aggregates["by_cell"]),
            "pairs_json": write_json(output / "pairs_8.json", aggregates["paired_differences"]),
        }
        summary = {"schema": "goal015d_evaluation_summary_v1", "plan_binding": binding,
                   "evaluation_artifact": artifact, "row_count": len(rows), "files": files, **aggregates,
                   "new_research_api_requests_sent": 0,
                   "interpretation": "Two-building development canary; nonempty is not reconstruction accuracy; no mechanism or generalization proof."}
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
        if summary.get("schema") != "goal015d_evaluation_summary_v1" or summary.get("plan_binding") != binding:
            continue
        if Path(summary["evaluation_artifact"]["path"]).resolve() != (path.parent / "evaluation_16.json").resolve():
            raise ValueError("cached evaluation path mismatch")
        evaluation = decode(bound(summary["evaluation_artifact"]))
        if evaluation.get("plan_binding") != binding or len(evaluation.get("rows", [])) != 16:
            raise ValueError("cached evaluation binding mismatch")
        for item in summary.get("files", {}).values():
            if Path(item["path"]).resolve().parent != path.parent.resolve():
                raise ValueError("cached table path mismatch")
            bound(item)
        for record, saved, row in zip(plan["schedule"], observations, evaluation["rows"]):
            if any(record[k] != row[k] or saved[k] != row[k] for k in IDENTITY):
                raise ValueError("cached evaluation identity mismatch")
            for key in ("raw", "response_text_path"):
                if row.get(key) != saved.get(key):
                    raise ValueError("cached evaluation source mismatch")
                if row.get(key):
                    bound(row[key])
            if row.get("gt_source") != plan["gt_sources"][record["scene_id"]]:
                raise ValueError("cached GT source binding mismatch")
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


_locked_latest_evaluation = _find_existing_evaluation_locked
