"""Offline Goal015B diagnostics. Never sends, repairs, or executes partial output."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time

PARENT = Path("/root/minecraft-runs/goal015-20261017T170546Z")
EVALUATION = PARENT / "evaluations/goal015a-parent72-20261019T141842Z-13cfbc79/evaluation_80.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_pairs(items):
    result = {}
    for k, v in items:
        if k in result:
            raise ValueError("duplicate JSON key: " + k)
        result[k] = v
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def bound(desc):
    raw = Path(desc["path"]).read_bytes()
    if len(raw) != desc["bytes"] or sha(raw) != desc["sha256"]:
        raise ValueError("descriptor mismatch: " + desc["path"])
    return raw


def extract(provider, raw):
    """Independent, typed envelope traversal; metadata records every block type."""
    data = decode(raw)
    if not isinstance(data, dict) or provider not in ("openai", "anthropic"):
        raise ValueError("invalid envelope/provider")
    parts, locations, events, block_types, roles = [], [], [], [], []
    source = data.get("output" if provider == "openai" else "content")
    if not isinstance(source, list):
        raise ValueError("missing or non-array envelope content")
    for i, item in enumerate(source):
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise ValueError("non-object or untyped envelope item")
        if provider == "openai":
            block_types.append("output:" + item["type"])
            if item["type"] != "message":
                events.append("nontext_output:" + item["type"])
                continue
            roles.append(item.get("role"))
            if item.get("role") != "assistant":
                events.append("non_assistant_message")
            if item.get("status") != "completed":
                events.append("message_status:" + str(item.get("status")))
            blocks = item.get("content")
            if not isinstance(blocks, list):
                raise ValueError("message content is not an array")
        else:
            blocks = [item]
        for j, block in enumerate(blocks):
            if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                raise ValueError("invalid content block")
            kind = block["type"]
            block_types.append("content:" + kind)
            if kind == ("output_text" if provider == "openai" else "text"):
                if not isinstance(block.get("text"), str):
                    raise ValueError("non-string text")
                parts.append(block["text"])
                locations.append([i, j])
            else:
                events.append("refusal" if kind == "refusal" else "nontext_content:" + kind)
    if not parts:
        events.append("missing_text")
    if len(parts) > 1:
        events.append("multiple_text_blocks")
    if provider == "anthropic":
        roles.append(data.get("role"))
        if data.get("role") != "assistant":
            events.append("non_assistant_message")
    text = "".join(parts) if parts else None
    return data, text, {
        "text_block_count": len(parts), "text_block_locations": locations,
        "block_types": block_types, "envelope_events": events, "roles": roles,
        "extraction_order": "API array order; concatenate text blocks without inserted separators",
    }


def prefix_observations(text):
    """Observe complete JSON operation objects in a prefix; NEVER repair/execute."""
    opening = re.match(r'\s*\{\s*"operations"\s*:\s*\[', text)
    if not opening:
        return {"prefix_operations_observed": None, "prefix_observation_status": "UNRECOGNIZED_PREFIX"}
    decoder = json.JSONDecoder(object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    cursor, ops = opening.end(), []
    while True:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] == "]":
            break
        try:
            op, end = decoder.raw_decode(text, cursor)
        except ValueError:
            break
        if not isinstance(op, dict):
            break
        ops.append(op)
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != ",":
            break
        cursor += 1
    hashes = [sha(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()) for o in ops]
    counts = Counter(o.get("op") for o in ops)
    longest = run = 0
    previous = None
    for h in hashes:
        run = run + 1 if h == previous else 1
        longest = max(longest, run)
        previous = h
    return {
        "prefix_operations_observed": len(ops), "prefix_observation_status": "OBSERVATION_ONLY_NOT_EXECUTED",
        "prefix_set_count": counts["set"], "prefix_fill_count": counts["fill"],
        "prefix_carve_count": counts["carve"], "prefix_unique_operations": len(set(hashes)),
        "prefix_repeated_operations": len(hashes) - len(set(hashes)),
        "prefix_longest_identical_run": longest,
        "text_ends_with_object_close": text.rstrip().endswith("}"),
        "text_ends_with_operations_close": bool(re.search(r"\]\s*\}\s*$", text)),
    }


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                             if isinstance(v, (dict, list)) else ("NULL" if v is None else v)
                             for k in keys for v in [row.get(k)]})


def validation_reason(operations):
    from offline_evaluation.server_v2 import core
    try:
        core.validate_actions(operations)
        return None, None
    except core.ValidationError as exc:
        reason = str(exc)
    explicit = re.search(r"operation (\d+)", reason)
    if explicit:
        return reason, int(explicit.group(1))
    if isinstance(operations, list):
        for i, op in enumerate(operations):
            try:
                core.validate_actions([op])
            except core.ValidationError:
                return reason, i
    return reason, None


def openai_echo_matches(envelope, payload):
    """Keep API-added format metadata separate from the actual schema tree."""
    actual = envelope.get("text", {}).get("format", {})
    requested = payload.get("text", {}).get("format", {})
    return {
        "raw_echo_prompt_matches_payload": envelope.get("instructions") == payload.get("instructions"),
        "raw_echo_format_exact_matches_payload": actual == requested,
        "raw_echo_schema_matches_payload": actual.get("schema") == requested.get("schema"),
        "raw_echo_format_contract_matches_payload": all(actual.get(k) == requested.get(k)
                                                      for k in ("name", "strict", "type", "schema")),
        "raw_echo_format_added_keys": sorted(set(actual)-set(requested)),
        "raw_echo_output_cap_matches_payload": envelope.get("max_output_tokens") == payload.get("max_output_tokens"),
    }


def run(run_root, report_root):
    from offline_evaluation.analysis.goal015b.independent import execute_independent
    from offline_evaluation.experiments.goal015 import adapter
    from offline_evaluation.server_v2 import core
    started, cpu_start = time.monotonic(), time.process_time()
    run_root, report_root = Path(run_root), Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    private = run_root / "audit"
    private.mkdir()
    inputs = {}

    def read(path):
        p = Path(path)
        b = p.read_bytes()
        inputs[str(p)] = {"path": str(p), "bytes": len(b), "sha256": sha(b)}
        return b

    def read_bound(desc):
        b = read(desc["path"])
        if len(b) != desc["bytes"] or sha(b) != desc["sha256"]:
            raise ValueError("descriptor mismatch")
        return b

    plan = decode(read(PARENT / "plan.json"))
    ledger = decode(read(PARENT / "enabled/ledger.json"))
    evaluation_bytes = read(EVALUATION)
    evaluation = decode(evaluation_bytes)
    old_estimate = decode(read(PARENT / "COST_ESTIMATE.json"))["old_O_reference80_usd"]
    assert len(plan["schedule"]) == len(ledger["calls"]) == len(evaluation["rows"]) == 80
    rows, failures, traces = [], [], []
    raw_ids, raw_hashes, text_hashes = [], [], []
    for record, saved, evaluated in zip(plan["schedule"], ledger["calls"], evaluation["rows"]):
        for name in ("call_id", "scene_id", "builder", "condition", "repeat_index", "paired_block_id"):
            assert record[name] == saved[name] == evaluated[name], (name, record["call_id"])
        assert record["payload"] == saved["payload"] == evaluated["artifacts"]["request_payload"]
        payload_bytes = read_bound(record["payload"])
        payload = decode(payload_bytes)
        adapter.validate_payload(record["builder"], payload)
        row = {k: record[k] for k in ("call_id", "scene_id", "builder", "condition", "repeat_index")}
        row.update(state=saved["state"], payload_sha256=sha(payload_bytes),
                   evaluation_sha256=sha(evaluation_bytes), requested_model=payload["model"],
                   payload_matches_frozen_prompt_schema_and_input=True,
                   raw_sha256=None, text_sha256=None, returned_model=None, response_id=None,
                   http_status=None, http_status_source="ledger_only_not_raw_HTTP_headers",
                   raw_status=None, stop_reason=None, incomplete_details=None,
                   input_tokens=None, output_tokens=None, reasoning_tokens=None,
                   cache_read_tokens=None, cache_write_tokens=None, extracted_text_bytes=None,
                   extracted_text_sha256=None, text_match=None, operation_count=None,
                   set_count=None, fill_count=None, carve_count=None, air_write_visits=None,
                   non_air_write_visits=None, final_occupied_voxels=None, classification="NOT_SENT",
                   known_cost_usd=None, evidence="NOT_SENT; no output or metric imputation")
        if saved["state"] == "NOT_SENT":
            assert evaluated["status"] == "NOT_SENT" and evaluated["metrics"] is None
            rows.append(row)
            continue
        assert saved["state"] == "SETTLED"
        raw = read_bound(saved["raw"])
        text_bytes = read_bound(saved["response_text_path"])
        assert len(raw) <= adapter.MAX_RESPONSE_BYTES
        assert len(text_bytes) <= core.Limits().max_json_bytes
        envelope, text, metadata = extract(record["builder"], raw)
        assert text is not None and text.encode() == text_bytes, record["call_id"]
        assert envelope["id"] == saved["response_id"]
        assert envelope["model"] == saved["response_model"] == payload["model"]
        assert saved["raw"] == evaluated["artifacts"]["raw"]
        assert saved["response_text_path"] == evaluated["artifacts"]["response_text_path"]
        raw_ids.append(envelope["id"])
        raw_hashes.append(sha(raw))
        text_hashes.append(sha(text_bytes))
        usage = envelope["usage"]
        assert isinstance(usage, dict)
        provider = record["builder"]
        it, ot = usage["input_tokens"], usage["output_tokens"]
        assert type(it) is type(ot) is int and it >= 0 and ot >= 0
        cache_read = (usage.get("input_tokens_details", {}).get("cached_tokens")
                      if provider == "openai" else usage.get("cache_read_input_tokens"))
        cache_write = (usage.get("input_tokens_details", {}).get("cache_write_tokens")
                       if provider == "openai" else usage.get("cache_creation_input_tokens"))
        assert cache_write in (None, 0)
        billable_it = it + ((cache_read or 0) + (cache_write or 0) if provider == "anthropic" else 0)
        rates = plan["pricing_contract"]["rates_per_million"][provider]
        cost = ((Decimal(billable_it)*Decimal(rates["input"]) + Decimal(ot)*Decimal(rates["output"]))
                / Decimal(1_000_000)).quantize(Decimal("0.000000001"), rounding=ROUND_CEILING)
        assert cost == Decimal(saved["known_cost_usd"])
        assert saved["usage"]["input_tokens"] == billable_it and saved["usage"]["output_tokens"] == ot
        row.update(raw_sha256=sha(raw), text_sha256=sha(text_bytes), returned_model=envelope["model"],
                   response_id=envelope["id"], http_status=saved["http_status"],
                   raw_status=envelope.get("status"), stop_reason=envelope.get("stop_reason"),
                   incomplete_details=envelope.get("incomplete_details"), input_tokens=it,
                   output_tokens=ot, reasoning_tokens=usage.get("output_tokens_details", {}).get("reasoning_tokens"),
                   cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                   extracted_text_bytes=len(text.encode()), extracted_text_sha256=sha(text.encode()),
                   text_match=True, known_cost_usd=str(cost), **metadata)
        row["call_mapping_evidence"] = "schedule/ledger/evaluation identities and exact descriptor bytes; API has no local call_id"
        if provider == "openai":
            row.update(openai_echo_matches(envelope, payload))
        try:
            document = decode(text)
            row["text_json_complete"] = True
        except ValueError:
            document = None
            row["text_json_complete"] = False
        if isinstance(document, dict) and isinstance(document.get("operations"), list):
            ops = document["operations"]
            counts = Counter(x.get("op") if isinstance(x, dict) else "nonobject" for x in ops)
            row.update(operation_count=len(ops), set_count=counts["set"],
                       fill_count=counts["fill"], carve_count=counts["carve"])
        else:
            ops = None
        if evaluated["protocol_valid"]:
            assert isinstance(document, dict) and set(document) == {"operations"}
            assert ops is not None
            assert decode(read_bound(evaluated["artifacts"]["parsed.json"])) == document
            grid, trace = execute_independent(ops)
            prediction = decode(read_bound(evaluated["artifacts"]["voxels.json"]))
            frozen_grid = core.payload_to_grid(prediction)
            assert grid == frozen_grid.voxels, record["call_id"]
            assert len(grid) == evaluated["final_voxel_count"]
            row.update(final_occupied_voxels=len(grid), air_write_visits=trace["air_write_visits"],
                       non_air_write_visits=trace["non_air_write_visits"],
                       first_occupied_operation_index=trace["first_occupied_operation_index"],
                       max_occupied_voxels=trace["max_occupied_voxels"],
                       last_zero_transition_operation_index=trace["last_zero_transition_operation_index"],
                       independent_grid_sha256=trace["final_grid_sha256"],
                       parsed_json_sha256=evaluated["artifacts"]["parsed.json"]["sha256"],
                       saved_voxel_sha256=evaluated["artifacts"]["voxels.json"]["sha256"],
                       prediction_coordinates_materials_match=True)
            if grid:
                classification = "NONEMPTY"
            elif not ops:
                classification = "RAW_EMPTY_OPERATIONS"
            elif trace["non_air_write_visits"] == 0:
                classification = "DELETE_ONLY_FROM_EMPTY"
            else:
                classification = "PLACED_THEN_ALL_DELETED"
            row.update(classification=classification,
                       evidence="independent envelope text == saved text == parsed document; independent XYZ/material grid == saved prediction")
            traces.append({"call_id": record["call_id"], **trace})
        else:
            kind = evaluated["failure_kind"]
            row["classification"] = kind.upper()
            row["evidence"] = "saved provider stop signal and frozen validation; no repair or partial execution"
            fail = {k: row[k] for k in ("call_id", "scene_id", "builder", "condition", "repeat_index",
                                        "raw_sha256", "text_sha256", "classification", "raw_status", "stop_reason",
                                        "incomplete_details", "output_tokens", "reasoning_tokens",
                                        "extracted_text_bytes", "text_json_complete", "operation_count")}
            fail.update(prefix_observations(text))
            if kind == "truncation":
                signal = (envelope.get("incomplete_details") or {}).get("reason") if provider == "openai" else envelope.get("stop_reason")
                assert signal == ("max_output_tokens" if provider == "openai" else "max_tokens")
                fail["confirmed_stop_at_output_cap"] = ot == plan["pricing_contract"]["output_cap"]
            elif kind == "validation_failure":
                reason, index = validation_reason(ops)
                assert reason is not None
                fail.update(validator_reason=reason, operation_index_zero_based=index,
                            rule_category="RESOURCE_LIMIT" if "limit" in reason or "bound" in reason else "SCHEMA_MATERIAL_OR_TYPE")
                row.update(validator_reason=reason, operation_index_zero_based=index)
                if "total voxel touch limit" in reason and index is not None:
                    # Arithmetic only: never execute the invalid program prefix.
                    volumes = [1 if op["op"] == "set" else
                               (abs(op["x2"]-op["x1"])+1)*(abs(op["y2"]-op["y1"])+1)*(abs(op["z2"]-op["z1"])+1)
                               for op in ops[:index+1]]
                    fail.update(touch_limit=core.Limits().max_touched_voxels,
                                cumulative_touches_before_rejected_operation=sum(volumes[:-1]),
                                cumulative_touches_including_rejected_operation=sum(volumes))
            else:
                raise AssertionError("unexpected output failure: " + str(kind))
            failures.append(fail)
        rows.append(row)
    assert len(set(raw_ids)) == len(raw_ids), "duplicate API response IDs"

    def aggregate(keys):
        groups = {}
        for row in rows:
            if row["state"] == "NOT_SENT":
                continue
            key = tuple(row[k] for k in keys)
            g = groups.setdefault(key, {**dict(zip(keys, key)), "calls": 0, "empty": 0,
                                       "nonempty": 0, "truncation": 0, "validation": 0,
                                       "known_cost_usd": Decimal(0), "output_tokens": 0,
                                       "reasoning_tokens_raw_present_sum": 0, "reasoning_usage_rows": 0})
            g["calls"] += 1
            cls = row["classification"]
            g["empty"] += cls in ("RAW_EMPTY_OPERATIONS", "DELETE_ONLY_FROM_EMPTY", "PLACED_THEN_ALL_DELETED")
            g["nonempty"] += cls == "NONEMPTY"
            g["truncation"] += cls == "TRUNCATION"
            g["validation"] += cls == "VALIDATION_FAILURE"
            g["known_cost_usd"] += Decimal(row["known_cost_usd"])
            g["output_tokens"] += row["output_tokens"]
            if row["reasoning_tokens"] is not None:
                g["reasoning_tokens_raw_present_sum"] += row["reasoning_tokens"]
                g["reasoning_usage_rows"] += 1
        result = []
        for key in sorted(groups):
            g = groups[key]
            g["known_cost_usd"] = str(g["known_cost_usd"])
            result.append(g)
        return result

    total = sum((Decimal(row["known_cost_usd"]) for row in rows if row["known_cost_usd"] is not None), Decimal(0))
    summary = {
        "status": "PASS", "planned_calls": len(rows), "settled": len(raw_ids),
        "classifications": dict(Counter(r["classification"] for r in rows)),
        "raw_text_exact_matches": sum(r["text_match"] is True for r in rows),
        "unique_response_ids": len(set(raw_ids)), "unique_raw_hashes": len(set(raw_hashes)),
        "unique_text_hashes": len(set(text_hashes)), "independent_executions": len(traces),
        "all_independent_grids_match_saved": all(r.get("prediction_coordinates_materials_match") for r in rows if r.get("final_occupied_voxels") is not None),
        "known_cost_usd": str(total), "old_O_reference80_usd": old_estimate,
        "actual72_divided_by_old80_extrapolation": str(total/Decimal(old_estimate)),
        "cost_is_existing_no_cache_discount_accounting_not_new_invoice": True,
        "failure_partial_outputs_executed": 0, "research_api_calls": 0,
        "provider_totals": aggregate(["builder"]),
        "elapsed_seconds": time.monotonic()-started, "cpu_seconds": time.process_time()-cpu_start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    for name, data in [("RAW_RESPONSE_AUDIT_80.csv", rows),
                       ("EMPTY_CLASSIFICATION_48.csv", [r for r in rows if r.get("final_occupied_voxels") == 0]),
                       ("TRUNCATION_VALIDATION_AUDIT.csv", failures),
                       ("BY_CELL_8.csv", aggregate(["builder", "condition", "repeat_index"])),
                       ("BY_BUILDING_9.csv", aggregate(["scene_id"]))]:
        write_csv(report_root/name, data)
    write_json(private/"ROWS.json", rows)
    write_json(private/"EXECUTION_TRACES.json", traces)
    write_json(private/"INPUT_DESCRIPTORS.json", inputs)
    write_json(report_root/"AUDIT_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-root", required=True)
    args = parser.parse_args()
    network = {"attempts": 0}
    def guard(event, arguments):
        if event in ("socket.connect", "socket.getaddrinfo", "urllib.Request"):
            network["attempts"] += 1
            raise RuntimeError("network prohibited in Goal015B diagnosis")
    sys.addaudithook(guard)
    resource.setrlimit(resource.RLIMIT_CPU, (450, 450))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    run(args.run_root, args.report_root)
    assert network["attempts"] == 0


if __name__ == "__main__":
    main()
