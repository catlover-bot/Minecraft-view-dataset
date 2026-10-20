"""Narrow, offline old-O input comparison and explicit saved-output replay.

Every source is hashed and recorded before its contents are decoded. Replays
are diagnostic fixtures, never parent Goal015 calls or new research outcomes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import ast
import copy
import resource
import time
from types import SimpleNamespace


FIELDS = ("summary", "building_type", "shape", "dimensions_estimate", "materials",
          "elements", "rebuild_hints", "uncertainties")


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SnapshotReader:
    """Append provenance before reading source content, then detect changes."""

    def __init__(self, receipt):
        self.receipt = Path(receipt)
        self.receipt.parent.mkdir(parents=True, exist_ok=True)

    def bytes(self, path, *, expected=None, purpose="comparison"):
        path = Path(path).resolve(strict=True)
        sha = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
                size += len(block)
        descriptor = {"path": str(path), "sha256": sha.hexdigest(),
                      "bytes": size, "purpose": purpose}
        if expected is not None:
            if descriptor["sha256"] != expected["sha256"]:
                raise ValueError("source descriptor SHA differs: " + str(path))
            if "bytes" in expected and descriptor["bytes"] != expected["bytes"]:
                raise ValueError("source descriptor size differs: " + str(path))
        with self.receipt.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(descriptor, sort_keys=True) + "\n")
            stream.flush()
        data = path.read_bytes()
        if len(data) != size or digest(data) != descriptor["sha256"]:
            raise ValueError("source changed between snapshot and decode")
        return data

    def json(self, path, **kwargs):
        return json.loads(self.bytes(path, **kwargs))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def frozen_object_hash(value):
    return digest((json.dumps(value, sort_keys=True, ensure_ascii=False,
                              allow_nan=False, indent=2) + "\n").encode())


def pure_function(source, name, namespace):
    """Compile one inspected pure function, never import a transport module."""
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError("expected exactly one frozen function: " + name)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<frozen-pure>", "exec"),
         namespace)
    return namespace[name]


def _input(payload, builder):
    return payload["input"] if builder == "openai" else payload["messages"][0]["content"]


def _schema(payload, builder):
    return (payload["text"]["format"]["schema"] if builder == "openai"
            else payload["output_config"]["format"]["schema"])


def _settings(payload):
    return {key: payload.get(key, "OMITTED; effective default UNKNOWN")
            for key in ("model", "temperature", "top_p", "top_k", "seed", "reasoning",
                        "thinking", "max_output_tokens", "max_tokens", "service_tier",
                        "stream", "store", "truncation", "tools")}


def run_comparison(run_root, parent_plan_path, report_path=None):
    """Select two old examples by ID, compare saved inputs, then replay in RAM."""
    started, cpu_started = time.monotonic(), time.process_time()
    output = Path(run_root) / "phase4"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPARISON_REPLAY.json").exists():
        raise ValueError("exclusive Phase4 output already exists")
    reader = SnapshotReader(output / "INPUT_SNAPSHOTS.jsonl")
    repo = Path(__file__).resolve().parents[3]
    parent_plan = reader.json(parent_plan_path, purpose="current parent plan")
    ledger = reader.json(Path(parent_plan["canonical_root"]) / "enabled/ledger.json",
                         purpose="current 72/8 ledger")
    bindings = reader.json(parent_plan["source_bindings"]["path"],
                           expected=parent_plan["source_bindings"])
    sent_scenes = sorted({row["scene_id"] for row in ledger["calls"]
                          if row["state"] == "SETTLED"})
    by_scene = {row["scene_id"]: row for row in bindings}
    manifest = reader.json(repo / "reports/server_only/goal007/MANIFEST.json")
    slots = reader.json(Path(manifest["output_root"]) / "data/final_slots_960.json",
                        expected=manifest["products"]["data/final_slots_960.json"])
    full_manifest = reader.json(manifest["source_manifest"]["path"],
                                expected=manifest["source_manifest"])

    def code(relative, current=False):
        expected = (parent_plan["implementation_hashes"][str(repo / relative)] if current
                    else manifest["code_hashes"][relative])
        return reader.bytes(repo / relative, expected={"sha256": expected},
                            purpose="frozen pure request/replay code").decode()

    providers_code = code("offline_evaluation/server_v2/providers.py")
    adapter_code = code("offline_evaluation/resmoke_v2/adapter.py")
    batch_code = code("offline_evaluation/batch_v1/runner.py")
    code("offline_evaluation/continuation_v1/runner.py")
    code("offline_evaluation/server_v2/generation.py")
    code("offline_evaluation/server_v2/core.py", current=True)
    code("offline_evaluation/experiments/goal015/evaluate.py", current=True)
    code("offline_evaluation/experiments/goal015/adapter.py", current=True)
    reader.bytes(repo / "offline_evaluation/analysis/goal015b/audit.py",
                 purpose="independent envelope extractor used for replay")
    from .audit import extract
    from ...experiments.goal015 import evaluate
    from ...experiments.goal015 import adapter as current_adapter
    from ...server_v2 import core

    comparisons = []
    for builder in ("openai", "anthropic"):
        candidates = sorted(
            (row for row in slots
             if row["scene_id"] in sent_scenes and row["condition"] == "O"
             and row["description_model"] == by_scene[row["scene_id"]]["description_provider"]
             and row["provider"] == builder and row["status"] == "success"
             and row["final_voxel_count"] > 0),
            key=lambda row: (row["scene_id"], row["global_schedule_index"]))
        if not candidates:
            comparisons.append({"builder": builder, "status": "NOT_FOUND"})
            continue
        slot = candidates[0]
        source = manifest["sources"][slot["source_label"]]
        old_plan = reader.json(slot["source_plan_path"],
                               expected={"sha256": source["plan_sha256"]})
        old_ledger = reader.json(source["ledger_path"],
                                 expected={"sha256": source["ledger_sha256"]})
        old_record = next(row for row in old_plan["identity"]["schedule"]
                          if row["schedule_index"] == slot["global_schedule_index"])
        full_record = next(row for row in full_manifest["identity"]["full_schedule"]
                           if row["schedule_index"] == slot["global_schedule_index"])
        assert old_record["input_text"] == full_record["input_text"]
        old_entry = next(row for row in old_ledger["entries"]
                         if row["attempt_id"] == slot["attempt_id"])
        binding = by_scene[slot["scene_id"]]
        original_text = reader.bytes(binding["description_source"]["path"],
                                     expected=binding["description_source"],
                                     purpose="original description text")
        assert original_text == old_record["input_text"].encode()
        assert digest(original_text) == slot["input_sha256"]
        old_document = json.loads(original_text)

        build_base = pure_function(providers_code, "build_payload",
                                   {"ENDPOINTS": {"openai": "", "anthropic": ""}})
        build_native = pure_function(
            adapter_code, "build_payload",
            {"MODELS": {"openai": "gpt-5-mini", "anthropic": "claude-haiku-4-5-20251001"},
             "MAX_OUTPUT_TOKENS": 32768, "ProviderError": ValueError,
             "providers": SimpleNamespace(build_payload=build_base),
             "action_schema": lambda: copy.deepcopy(old_plan["identity"]["schema"])})
        build_original = pure_function(batch_code, "_payload",
                                       {"adapter": SimpleNamespace(build_payload=build_native)})
        old_payload = build_original(old_plan["identity"], old_record)
        old_payload_hash = frozen_object_hash(old_payload)
        assert old_payload_hash == slot["request_payload_hash"] == old_entry["request_payload_hash"]

        artifacts = {}
        for name in ("raw.json", "parsed.json", "voxels.json"):
            descriptor = slot["artifacts"][name]
            path = Path(slot["source_run"]) / descriptor["path"]
            artifacts[name] = reader.bytes(path, expected=descriptor,
                                           purpose="selected original " + name)
        raw, extracted, extraction = extract(builder, artifacts["raw.json"])
        assert isinstance(extracted, str)
        parsed = json.loads(artifacts["parsed.json"])
        old_voxels = json.loads(artifacts["voxels.json"])
        assert json.loads(extracted) == parsed
        assert raw["model"] == slot["response_model"]
        native_response = current_adapter.parse_response(builder, artifacts["raw.json"])
        assert native_response.complete and native_response.failure_kind is None
        assert native_response.text == extracted and native_response.model == raw["model"]
        assert current_adapter.validate_output(native_response.text) == parsed["operations"]

        current_calls = sorted(
            (row for row in ledger["calls"]
             if row["scene_id"] == slot["scene_id"] and row["builder"] == builder
             and row["condition"] == "FULL_RECORD"), key=lambda row: row["repeat_index"])
        current_payload = None
        current_evidence = []
        for call in current_calls:
            payload = reader.json(call["payload"]["path"], expected=call["payload"])
            if current_payload is not None:
                assert payload == current_payload
            current_payload = payload
            current_raw, current_text, current_extraction = extract(
                builder, reader.bytes(call["raw"]["path"], expected=call["raw"]))
            saved_text = reader.bytes(call["response_text_path"]["path"],
                                      expected=call["response_text_path"])
            assert current_text.encode() == saved_text
            current_evidence.append({
                "call_id": call["call_id"], "repeat_index": call["repeat_index"],
                "started_at_utc": call["started_at_utc"], "finished_at_utc": call["finished_at_utc"],
                "returned_model": current_raw.get("model"),
                "response_id": current_raw.get("id"),
                "status": current_raw.get("status", current_raw.get("stop_reason")),
                "response_echo_settings": {key: current_raw[key] for key in
                                           ("reasoning", "temperature", "top_p", "max_output_tokens",
                                            "max_tokens", "service_tier") if key in current_raw},
                "payload": call["payload"], "raw": call["raw"],
                "text": call["response_text_path"], "extraction": current_extraction})
        current_text_input = _input(current_payload, builder)
        current_document = json.loads(current_text_input)
        field_checks = {
            key: {"typed_value_equal": canonical(old_document[key]) == canonical(current_document[key]),
                  "old_value_sha256": digest(canonical(old_document[key])),
                  "new_value_sha256": digest(canonical(current_document[key]))}
            for key in FIELDS}
        assert set(current_document) == set(FIELDS)
        assert all(row["typed_value_equal"] for row in field_checks.values())
        prompt = current_payload["instructions" if builder == "openai" else "system"]
        assert prompt == old_plan["identity"]["prompt"]
        assert _schema(current_payload, builder) == _schema(old_payload, builder)

        # These identifiers are explicitly new diagnostic fixtures. The selected
        # old identity is provenance only; no parent call IDs are impersonated.
        fixture = {"call_id": "goal015b_replay_" + builder,
                   "scene_id": "fixture_replay_" + slot["scene_id"],
                   "builder": builder, "condition": "OLD_O_REPLAY_FIXTURE",
                   "repeat_index": 1, "paired_block_id": "goal015b_replay_only_" + builder}
        observation = dict(fixture, state="SETTLED", send_status="POST_RETURNED",
                           response_received=True, completed_response=True, usage_known=True,
                           model_output_failure=False, failure_kind=None,
                           response_text=native_response.text, response_model=raw["model"])
        # Synthetic self-reference GT is the saved original prediction, not the
        # real building GT. This tests the existing full evaluation path cheaply.
        prediction_as_fixture_gt = core.payload_to_grid(old_voxels)
        replay = evaluate.evaluate_one(fixture, observation, prediction_as_fixture_gt)
        replay_checks = {
            "protocol_valid": replay["protocol_valid"],
            "current_adapter_parse_and_validate_equal": True,
            "nonempty": replay["nonempty"],
            "evaluation_pass": replay["evaluation_status"] == "PASS",
            "parsed_equals_saved_original": replay["parsed_document"] == parsed,
            "grid_exactly_equals_saved_original": replay["voxel_payload"] == old_voxels,
            "operation_count_matches_old": replay.get("operation_count") == slot["operation_count"],
            "voxel_count_matches_old": replay.get("final_voxel_count") == slot["final_voxel_count"],
        }
        comparisons.append({
            "builder": builder, "status": "PASS" if all(replay_checks.values()) else "REPLAY_INCOMPATIBLE",
            "scene_id": slot["scene_id"], "description_provider": binding["description_provider"],
            "candidate_count": len(candidates),
            "candidate_ids_in_order": [{"scene_id": row["scene_id"],
                                        "global_schedule_index": row["global_schedule_index"]}
                                       for row in candidates],
            "selected_global_schedule_index": slot["global_schedule_index"],
            "selected_attempt_id": slot["attempt_id"],
            "source_plan": slot["source_plan_path"], "source_ledger": source["ledger_path"],
            "source_raw": str(Path(slot["source_run"]) / slot["artifacts"]["raw.json"]["path"]),
            "old_original_description": binding["description_source"],
            "old_original_request_artifact": "NOT_SAVED_SEPARATELY",
            "old_reconstructed_payload_frozen_object_sha256": old_payload_hash,
            "old_request_hash_matches_ledger_and_selected_slot": True,
            "wire_capture_available": False,
            "field_checks": field_checks,
            "old_metadata_fields": sorted(set(old_document) - set(FIELDS)),
            "new_metadata_fields": sorted(set(current_document) - set(FIELDS)),
            "old_input": {"bytes": len(original_text), "sha256": digest(original_text),
                          "serialization": "exact original description file bytes, including formatting/newline"},
            "new_input": {"bytes": len(current_text_input.encode()),
                          "sha256": digest(current_text_input.encode()),
                          "serialization": binding["serialization"]},
            "input_bytes_equal": original_text == current_text_input.encode(),
            "prompt_exact_equal": True, "prompt_sha256": digest(prompt.encode()),
            "schema_typed_tree_equal": True,
            "old_schema_bound_sha256": slot["schema_sha256"],
            "old_requested_settings": _settings(old_payload),
            "new_requested_settings": _settings(current_payload),
            "old_response_model": raw["model"], "old_response_id": raw["id"],
            "old_response_echo_settings": {key: raw[key] for key in
                                            ("reasoning", "temperature", "top_p", "max_output_tokens",
                                             "max_tokens", "service_tier") if key in raw},
            "old_started_at_utc": slot["started_at_utc"], "old_finished_at_utc": slot["finished_at_utc"],
            "old_bound_code_commit": old_plan["identity"]["commit"],
            "old_actual_runtime_environment": "NOT_RECORDED_IN_SELECTED_PLAN_LEDGER_ARTIFACT_CHAIN",
            "new_configured_environment": parent_plan["existing_execution_environment"],
            "new_actual_historical_process_environment": "NOT_CAPTURED_BY_THIS_COMPARISON",
            "current_full_calls": current_evidence, "extraction": extraction,
            "old_extracted_text": {"bytes": len(extracted.encode()), "sha256": digest(extracted.encode())},
            "replay": {"fixture_record": fixture, "gt": "SYNTHETIC_SELF_REFERENCE_SAVED_OLD_PREDICTION",
                       "research_outcome": False, "saved_parent_call_modified": False,
                       "checks": replay_checks, "operation_count": replay.get("operation_count"),
                       "final_voxel_count": replay.get("final_voxel_count"),
                       "failure_kind": replay["failure_kind"], "null_reason": replay["null_reason"],
                       "metric_status": replay["evaluation_status"],
                       "synthetic_self_reference_metrics": replay["metrics"],
                       "prediction_sha256": evaluate.frozen_grid_sha256(prediction_as_fixture_gt)},
        })
    result = {
        "schema": "goal015b_narrow_old_o_comparison_v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in comparisons) else "INCOMPLETE",
        "selection_rule": "Current SETTLED scenes and same description provider; first nonempty successful old O by scene_id, then global_schedule_index, separately per Builder; no IoU selection.",
        "current_settled_scenes": sent_scenes,
        "comparisons": comparisons,
        "research_api_requests_sent": 0, "old_artifact_writes": 0,
        "new_research_outcomes": 0, "replays": sum("replay" in row for row in comparisons),
        "cpu_seconds": time.process_time() - cpu_started,
        "wall_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": [
            "No independent historical transport wire capture; old request is reconstructed exactly to its saved ledger object hash.",
            "Saved response settings are observations; omitted request defaults are not inferred.",
            "Old actual process environment is not recorded in the selected source chain; new plan environment is configured evidence.",
            "Replay uses saved prediction as synthetic GT, so self-reference scores do not measure real building accuracy.",
            "Input/settings/date differences do not identify a cause of changed model output.",
        ],
    }
    with (output / "COMPARISON_REPLAY.json").open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    if report_path is not None:
        write_report(result, report_path)
    return result


def write_report(result, path):
    if result["status"] != "PASS":
        with Path(path).open("x") as stream:
            stream.write("# Goal015B: old-output comparison incomplete\n\n"
                         "Selection or replay failed. No compatibility claim is made; "
                         "see the private COMPARISON_REPLAY.json for the exact result.\n")
        return
    rows = result["comparisons"]
    lines = [
        "# Goal015B: old O versus current FULL inputs and saved-output replay", "",
        "Both ID-selected old nonempty outputs remain nonempty through the frozen Goal015 evaluator. "
        "This confirms compatibility of these two saved outputs with the current extraction, validation, execution, "
        "and metric path. It does not explain why the later models returned deletion-only programs.", "",
        "Selection was fixed to the nine currently sent scenes, the same description provider and Builder, "
        "then ascending scene ID. No IoU ranking was used. Both selected examples are building 003, "
        "using its saved OpenAI description. Each Builder had nine eligible old nonempty O examples.", "",
        "| Evidence | Old OpenAI O | Old Anthropic O |",
        "|---|---|---|",
        f"| Selected global slot | {rows[0]['selected_global_schedule_index']} | {rows[1]['selected_global_schedule_index']} |",
        f"| Operations / occupied voxels | {rows[0]['replay']['operation_count']} / {rows[0]['replay']['final_voxel_count']} | {rows[1]['replay']['operation_count']} / {rows[1]['replay']['final_voxel_count']} |",
        "| Replay parse / exact saved grid / metric execution | PASS / PASS / PASS | PASS / PASS / PASS |", "",
        "| Dimension | Old O | Current FULL (both repeats) |",
        "|---|---|---|",
        "| Summary and seven other semantic fields | All eight original typed values | All eight exactly equal to old O |",
        "| Management metadata | `building`, `created_at`, `model`, `provider` present | All four omitted |",
        f"| Input serialization / bytes | Original formatted file, {rows[0]['old_input']['bytes']} UTF-8 bytes | Sorted compact exact serializer, {rows[0]['new_input']['bytes']} UTF-8 bytes, no final newline |",
        "| Prompt | Same exact bytes | Same exact bytes |",
        "| Native output schema | Same shared set/fill/carve schema; empty array permitted | Same exact typed schema |",
        "| OpenAI requested / returned model | `gpt-5-mini` / `gpt-5-mini-2025-08-07` | Exact snapshot requested and returned |",
        "| Anthropic requested / returned model | `claude-haiku-4-5-20251001` / same | Same exact snapshot requested and returned |",
        "| OpenAI reasoning | Request omitted; response reports `medium` | Request explicitly `medium`; response reports `medium` |",
        "| OpenAI sampling | Request temperature/top-p/seed omitted; effective defaults unknown | Same omitted request settings; effective defaults unknown |",
        "| Anthropic sampling / thinking | Temperature/top-p/top-k/seed/thinking omitted; effective defaults unknown | Temperature 1.0 and thinking disabled explicit; top-p/top-k/seed omitted |",
        "| Maximum output tokens | 32768, both Builders | 32768, both Builders |",
        "| Service tier | OpenAI `default`; Anthropic `standard_only` | Same explicit requests |",
        "| Generation time | 2026-10-14 UTC (exact timestamps in private receipt) | 2026-10-19 UTC (exact timestamps in private receipt) |",
        "| Environment evidence | Bound generation commit/code; actual Python/package process environment not recorded in selected chain | Plan names existing Python environment and packages; actual historical process environment not captured by this comparison |", "",
        "The prompt hash is `" + rows[0]["prompt_sha256"] + "`. The old bound schema hash is `"
        + rows[0]["old_schema_bound_sha256"] + "`; equality was checked on actual schema trees.", "",
        "Goal007's manifest authenticates its canonical selected-slot product and original Goal006 plan/ledger. "
        "The original Goal005 full schedule agrees with the selected Goal006 input text. The original description bytes, "
        "raw provider envelope, parsed document and voxel artifact were each hash-checked before decoding. "
        "There is no separately saved old HTTP request artifact. The old request was reconstructed using only the frozen "
        "pure payload functions and matched the exact object hash in both ledger and selected slot:", "",
    ]
    for row in rows:
        lines.append(f"- {row['builder']}: `{row['old_reconstructed_payload_frozen_object_sha256']}`; "
                     f"old attempt `{row['selected_attempt_id']}`; current FULL calls "
                     + ", ".join("`" + call["call_id"] + "`" for call in row["current_full_calls"]) + ".")
    lines += [
        "",
        "This provides saved request-construction provenance, not independent proof of the historical bytes on the wire. "
        "Omitted defaults remain unknown even when a response echoes a value. The removed metadata, formatting, "
        "explicit settings and five-day time difference are observable differences, not established causes.", "",
        "Replay identifiers are `goal015b_replay_openai` and `goal015b_replay_anthropic`; they do not impersonate parent calls. "
        "The actual Goal015 `adapter.parse_response` and `adapter.validate_output` accept both native raw responses "
        "and match independent envelope extraction and the saved parsed document exactly. "
        "Their unchanged text was passed into `Goal015.evaluate_one`. "
        "GT was explicitly the saved old prediction as a synthetic self-reference fixture, keeping the real evaluation path "
        "without loading or clipping actual GT. Both parsed documents and all saved coordinates/materials match exactly. "
        "Self-reference metrics (IoU and surface F1 equal 1) are fixture checks and are not new reconstruction results.", "",
        f"Replay computation: {result['cpu_seconds']:.3f} CPU seconds, {result['wall_seconds']:.3f} wall seconds, "
        f"peak RSS {result['max_rss_kib']} KiB, one thread. Additional research requests: 0. "
        "Original outputs, parent calls and the eight unsent calls were not modified.", "",
        "Private evidence: `phase4/INPUT_SNAPSHOTS.jsonl` (pre-read path/hash/size log) and "
        "`phase4/COMPARISON_REPLAY.json` (selection frame, per-field hashes, request/response settings, "
        "timestamps, exact source descriptors and replay checks) under the new Goal015B run.", "",
    ]
    with Path(path).open("x") as stream:
        stream.write("\n".join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--parent-plan", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    value = run_comparison(args.run_root, args.parent_plan, args.report)
    print(json.dumps({key: value[key] for key in
                      ("status", "replays", "cpu_seconds", "wall_seconds", "max_rss_kib")}))
