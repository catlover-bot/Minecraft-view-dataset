"""Read-only Goal005 diagnosis of saved resmoke outputs; never calls a sender.

Production execution verifies the frozen implementation hashes before using
its parser, validator, executor and metrics. New output directories are required.
No tokenizer, API key, approval, activation, or network access is needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time

from offline_evaluation.resmoke_v2 import adapter, schema
from offline_evaluation.server_v2 import core
from offline_evaluation.server_v2.analysis_v4 import BOUNDED_IOU, bounded_all_attempted_value, summarize_v4
from offline_evaluation.server_v2.data import load_scene
from offline_evaluation.server_v2.generation import object_hash

ROOT = Path(__file__).resolve().parents[2]
METRIC_FIELDS = ("occupancy_iou", "material_aware_iou", "dimension_mae", "add_count",
                 "delete_count", "replace_count", "total_normalized_repair_cost")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def snapshot(paths):
    files = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files.add(path)
        else:
            raise ValueError("required original missing")
    if any(p.is_symlink() for p in files):
        raise ValueError("original symlink rejected")
    return {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(files)}


def first_local_violation(actions, limits=core.Limits()):
    """Locate the first frozen validator failure without changing its decision.

    Every prefix uses the frozen validator, preserving its operation and rule
    order. Only a bounded operation is included, never raw response metadata.
    """
    try:
        core.validate_actions(actions, limits)
        return None
    except core.ValidationError as failure:
        cause = str(failure)
    if not isinstance(actions, list) or len(actions) > limits.max_operations:
        return {"operation_index_zero_based": None, "field": "operations", "actual_value": len(actions),
                "rule_category": "operation_count", "expected_rule": f"at most {limits.max_operations} operations",
                "exception_type": "ValidationError", "cause": cause}
    for index in range(len(actions)):
        try:
            core.validate_actions(actions[:index + 1], limits)
        except core.ValidationError:
            action = actions[index]
            category, field, value, rule = "unclassified", None, None, cause
            if cause == "block must be a lower-case Minecraft material identifier":
                category, field, value = "material_identifier_syntax", "block", action["block"]
                rule = core.MATERIAL_RE.pattern + "; optional minecraft: namespace only; no spaces or parentheses"
            elif "coordinate bound" in cause or "must be an integer" in cause:
                field = cause.split()[0]
                category, value = "coordinate", action.get(field)
            elif "touch limit" in cause:
                category, field = "total_touched_voxels", "cumulative inclusive operation volume"
                value = sum(operation_volume(a) for a in actions[:index + 1])
            elif "live voxel limit" in cause:
                category, field, value = "live_voxels", "inclusive operation volume", operation_volume(action)
            excerpt = json.dumps(action, sort_keys=True, ensure_ascii=True)
            if len(excerpt.encode()) > 1024:
                excerpt = {"omitted": True, "operation_sha256": object_hash(action)}
            else:
                excerpt = action
            return {"operation_index_zero_based": index, "operation_number_one_based": index + 1,
                    "field": field, "actual_value": value, "expected_rule": rule,
                    "rule_category": category, "exception_type": "ValidationError", "cause": cause,
                    "safe_operation_excerpt": excerpt, "excerpt_max_bytes": 1024,
                    "classification": "model_output_violates_predeclared_specification",
                    "validator_bug_found": False, "spec_validator_mismatch_found": False}
    raise AssertionError("frozen validator failure was not reproducible")


def operation_bounds(action):
    if action["op"] == "set":
        return tuple(action[a] for a in "xyz") * 2
    return tuple(min(action[a + "1"], action[a + "2"]) for a in "xyz") + tuple(
        max(action[a + "1"], action[a + "2"]) for a in "xyz")


def operation_volume(action):
    bounds = operation_bounds(action)
    result = 1
    for i in range(3):
        result *= bounds[i + 3] - bounds[i] + 1
    return result


def independent_decode(spec):
    """Independent expansion of the actual supplied palette/cuboid JSON.

    Does not call encode_exact, decode_exact, reference_actions, or execute.
    The complete metadata remains checked by the frozen plan verifier elsewhere.
    """
    if (spec["endpoints"] != "inclusive" or spec["absent_coordinates"] != "air"
            or spec["palette_semantics"] != "zero_based_exact_non_air_material_identifiers"
            or spec["coordinate_frame"] not in {"world", "local"}):
        raise ValueError("unsupported discrete specification")
    offset = spec["origin_xyz"] if spec["coordinate_frame"] == "local" else [0, 0, 0]
    result = {}
    for row in spec["cuboids"]:
        if len(row) != 7 or any(type(v) is not int for v in row):
            raise ValueError("invalid cuboid")
        if not 0 <= row[6] < len(spec["palette"]) or any(row[i] > row[i + 3] for i in range(3)):
            raise ValueError("invalid bounds or palette index")
        material = spec["palette"][row[6]]
        if core.material_id(material) != material or material == "air":
            raise ValueError("palette identifier is not exact non-air")
        for coordinate in itertools.product(*(range(row[i], row[i + 3] + 1) for i in range(3))):
            world = tuple(coordinate[i] + offset[i] for i in range(3))
            if world in result:
                raise ValueError("overlapping cuboids")
            result[world] = material
    return result


def voxel_difference(gt, pred):
    """Exact-ID absolute comparison; air denotes missing occupied coordinates."""
    confusion = Counter((gt.get(c, "air"), pred.get(c, "air")) for c in set(gt) | set(pred))
    differences = []
    for (g, p), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0])):
        if g != p:
            examples = [list(c) for c in sorted(set(gt) | set(pred))
                        if gt.get(c, "air") == g and pred.get(c, "air") == p][:5]
            differences.append({"gt_material": g, "pred_material": p, "count": count,
                                "example_coordinates_xyz": examples})
    return {"occupancy_exact": set(gt) == set(pred), "labelled_voxel_exact": gt == pred,
            "gt_voxel_count": len(gt), "pred_voxel_count": len(pred),
            "add_count": len(set(gt) - set(pred)), "delete_count": len(set(pred) - set(gt)),
            "replace_count": sum(gt[c] != pred[c] for c in set(gt) & set(pred)),
            "gt_materials": sorted(set(gt.values())), "pred_materials": sorted(set(pred.values())),
            "confusion_including_missing_air": [{"gt_material": g, "pred_material": p, "count": n}
                                                 for (g, p), n in sorted(confusion.items())],
            "differences": differences}


def cuboid_difference(spec, actions, pred):
    offset = spec["origin_xyz"] if spec["coordinate_frame"] == "local" else [0, 0, 0]
    bounds_to_actions = defaultdict(list)
    visits = Counter()
    for index, action in enumerate(actions):
        bounds = operation_bounds(action)
        bounds_to_actions[bounds].append({"operation_index_zero_based": index,
                                          "material": action.get("block", "air")})
        for c in itertools.product(*(range(bounds[i], bounds[i + 3] + 1) for i in range(3))):
            visits[c] += 1
    matched, mismatches = 0, []
    for index, row in enumerate(spec["cuboids"]):
        bounds = tuple(row[i] + offset[i % 3] for i in range(6))
        material = spec["palette"][row[6]]
        matches = bounds_to_actions.get(bounds, [])
        if len(matches) == 1 and matches[0]["material"] == material:
            matched += 1
        else:
            output = Counter(pred.get(c, "air") for c in itertools.product(
                *(range(bounds[i], bounds[i + 3] + 1) for i in range(3))))
            mismatches.append({"input_cuboid_index_zero_based": index, "input_cuboid": row,
                               "expected_material": material, "same_bounds_output_operations": matches,
                               "final_material_counts_inside_input_bounds": dict(sorted(output.items()))})
    return {"input_cuboids": len(spec["cuboids"]), "output_operations": len(actions),
            "one_exact_same_bounds_material_operation": matched,
            "mismatching_input_cuboids": mismatches,
            "output_coordinates_visited_more_than_once": sum(n > 1 for n in visits.values()),
            "duplicate_operation_visits": sum(n - 1 for n in visits.values()),
            "output_materials_not_in_palette": sorted({a.get("block", "air") for a in actions} - set(spec["palette"])),
            "interpretation": "Exact IDs only; no alias mapping. Same bounds with another material proves a transcription discrepancy, not its cognitive cause."}


def transcription_details(spec, actions):
    """Bounded operation-level evidence for changed bounds and palette mappings."""
    offset = spec["origin_xyz"] if spec["coordinate_frame"] == "local" else [0, 0, 0]
    originals = {tuple(row[i] + offset[i % 3] for i in range(6)): (index, row)
                 for index, row in enumerate(spec["cuboids"])}
    palette_map, nonmatching_bounds = Counter(), []
    seen, overlap_examples = {}, []
    for index, action in enumerate(actions):
        bounds = operation_bounds(action)
        material = action.get("block", "air")
        if bounds in originals:
            _, row = originals[bounds]
            expected = spec["palette"][row[6]]
            if expected != material:
                palette_map[(row[6], expected, material)] += 1
        else:
            nearby = []
            for reference, (input_index, row) in originals.items():
                overlap = 1
                for i in range(3):
                    overlap *= max(0, min(bounds[i + 3], reference[i + 3]) - max(bounds[i], reference[i]) + 1)
                if overlap:
                    nearby.append({"input_cuboid_index_zero_based": input_index, "input_cuboid": row,
                                   "expected_material": spec["palette"][row[6]], "intersection_voxels": overlap})
            nonmatching_bounds.append({"operation_index_zero_based": index, "operation": action,
                                       "overlapping_input_cuboids": nearby[:20]})
        for coordinate in itertools.product(*(range(bounds[i], bounds[i + 3] + 1) for i in range(3))):
            if coordinate in seen and len(overlap_examples) < 20:
                overlap_examples.append({"coordinate_xyz": list(coordinate), "earlier_operation": seen[coordinate],
                                         "later_operation_index_zero_based": index, "later_material": material})
            seen[coordinate] = {"operation_index_zero_based": index, "material": material}
    return {"same_bounds_palette_transcriptions": [
                {"input_palette_index": i, "expected_material": expected, "output_material": actual,
                 "affected_cuboids": count} for (i, expected, actual), count in sorted(palette_map.items())],
            "changed_bounds_output_operations": nonmatching_bounds,
            "overlap_examples": overlap_examples,
            "classification": "observed output transcription differences; no inference about hidden reasoning",
            "material_alias_sensitivity": "not performed; exact IDs retained; no evidence requires an alias merge"}


def common_valid_oj(rows):
    result = {}
    for provider in ("openai", "anthropic"):
        sides = {c: {(r["scene_id"], r["description_model"]) for r in rows
                     if r["provider"] == provider and r["condition"] == c and r["metrics"] is not None}
                 for c in "OJ"}
        result[provider] = {"O_valid_scene_ids": sorted(s for s, _ in sides["O"]),
                            "J_valid_scene_ids": sorted(s for s, _ in sides["J"]),
                            "common_valid_scene_ids": sorted({s for s, _ in sides["O"] & sides["J"]}),
                            "common_valid_pairs": [{"scene_id": s, "description_model": m}
                                                   for s, m in sorted(sides["O"] & sides["J"])]}
    return result


def flat_row(row):
    result = {k: row.get(k) for k in ("provider", "scene_id", "condition", "description_model", "input_id",
              "attempt_id", "status", "failure_kind", "completed_request_outcome", "valid_nonempty_reconstruction")}
    for frame in ("absolute", "aligned"):
        for field in METRIC_FIELDS:
            result[frame + "_" + field] = row["metrics"][frame][field] if row["metrics"] else None
    for metric in sorted(BOUNDED_IOU):
        result[metric + "_failure0"] = bounded_all_attempted_value(row, metric)
    return result


def verify_archive(archive_root):
    """Verify diagnostic-only restored originals without activating/reserving.

    Expected layout: original/plan_root/{plan.json,enabled/...} and
    original/authority/<authority basename>. Canonical paths INSIDE the old
    plan are never changed. Frozen-code and GT references use this repository.
    Historical metric objects are hash-preserved by the archive manifest; this
    check replays raw->schema->validator->executor, not the 14 metric searches.
    """
    archive_root = Path(archive_root).resolve()
    plan_root = archive_root / "original/plan_root"
    plan = read(plan_root / "plan.json")
    identity = plan["identity"]
    if object_hash(identity) != plan["identity_hash"]:
        raise ValueError("archive plan identity mismatch")
    if str(plan_root / "plan.json") == identity["plan_path"]:
        raise ValueError("archive verification must not target active originals")
    for relative, expected in identity["implementation_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("archive frozen implementation mismatch")
    enabled = plan_root / "enabled"
    ledger, approval = read(enabled / "ledger.json"), read(enabled / "approval.json")
    authority = read(archive_root / "original/authority" / Path(identity["authority_path"]).name)
    if (ledger["plan_identity_hash"] != plan["identity_hash"]
            or approval["plan_identity_hash"] != plan["identity_hash"]
            or authority["approval_hash"] != object_hash(approval)
            or authority["plan_identity_hash"] != plan["identity_hash"]):
        raise ValueError("archive control identity mismatch")
    if len(ledger["entries"]) != 16 or len(identity["schedule"]) != 16:
        raise ValueError("archive fixed16 count mismatch")
    limits, successes, failures = core.Limits(**identity["limits"]), 0, []
    for record, entry in zip(identity["schedule"], ledger["entries"]):
        if (record["input_id"], record["provider"], record["schedule_index"]) != (
                entry["input_id"], entry["provider"], entry["schedule_index"]):
            raise ValueError("archive schedule mismatch")
        for artifact in entry["artifacts"].values():
            path = enabled / artifact["path"]
            if not path.resolve().is_relative_to(enabled.resolve()) or path.is_symlink():
                raise ValueError("archive artifact traversal")
            if sha(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
                raise ValueError("archive artifact hash mismatch")
        payload = adapter.build_payload(record["provider"], record["model"], identity["prompt"],
                                        record["input_text"], identity["budgets"]["max_output_tokens"])
        if object_hash(payload) != entry["request_payload_hash"]:
            raise ValueError("archive request payload mismatch")
        response = adapter.parse_response(record["provider"], (enabled / entry["artifacts"]["raw.json"]["path"]).read_bytes())
        if (response.response_id != entry["response_id"] or response.model != entry["response_model"]
                or response.usage != entry["usage"] or response.complete != entry["natural_stop"]):
            raise ValueError("archive raw metadata mismatch")
        actions = schema.validate_document(core.strict_json(response.text, limits))
        violation = first_local_violation(actions, limits)
        if violation:
            if entry["status"] != "failed" or entry["failure_stage"] != "local_validator":
                raise ValueError("archive failure status mismatch")
            failures.append({"attempt_id": entry["attempt_id"], "first_violation": violation})
        else:
            normalized = core.validate_actions(actions, limits)
            grid = core.execute(normalized, limits)
            if (entry["status"] != "success"
                    or read(enabled / entry["artifacts"]["parsed.json"]["path"]) != {"operations": normalized}
                    or read(enabled / entry["artifacts"]["voxels.json"]["path"]) != core.grid_to_payload(grid)):
                raise ValueError("archive raw parsed voxel chain mismatch")
            successes += 1
    return {"archive_root": str(archive_root), "diagnostic_only": True,
            "canonical_original_plan_paths_preserved": True, "archive_activation_performed": False,
            "raw_responses_verified": len(ledger["entries"]), "successful_execution_chains": successes,
            "reproduced_failures": failures, "metric_searches_repeated": 0, "research_api_sends": 0}


def run(plan_path, output):
    started = time.monotonic()
    plan_path, output = Path(plan_path), Path(output)
    if output.exists():
        raise ValueError("a new unique diagnosis output directory is required")
    plan = read(plan_path)
    identity = plan["identity"]
    if object_hash(identity) != plan["identity_hash"]:
        raise ValueError("plan identity mismatch")
    for relative, expected in identity["implementation_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("frozen implementation mismatch: " + relative)
    enabled = Path(identity["enabled_directory"])
    originals = [plan_path, enabled, identity["authority_path"], identity["parent"]["campaign_directory"]]
    before_at = datetime.now(timezone.utc).isoformat()
    before = snapshot(originals)
    ledger, original = read(enabled / "ledger.json"), read(enabled / "evaluation.json")
    summary = read(enabled / "summary.json")
    if len(identity["schedule"]) != 16 or len(ledger["entries"]) != 16 or len(original["rows"]) != 16:
        raise ValueError("expected complete fixed16 original")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "original_hashes.before.json", {"observed_at_utc": before_at, "files": before})
    grids = {scene: load_scene(desc["building_directory"]).grid for scene, desc in identity["gt_by_scene"].items()}
    for scene, grid in grids.items():
        if object_hash(core.grid_to_payload(grid)) != identity["gt_by_scene"][scene]["grid_sha256"]:
            raise ValueError("GT identity mismatch")
    rows, failures, e_rows, chains, comparisons, timings, e_details = [], [], [], [], [], [], []
    limits = core.Limits(**identity["limits"])
    for record, entry, stored in zip(identity["schedule"], ledger["entries"], original["rows"]):
        key = (record["provider"], record["input_id"], entry["attempt_id"])
        if key != (stored["provider"], stored["input_id"], stored["attempt_id"]):
            raise ValueError("row identity mismatch")
        for artifact in entry["artifacts"].values():
            path = enabled / artifact["path"]
            if sha(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
                raise ValueError("saved artifact mismatch")
        payload = adapter.build_payload(record["provider"], record["model"], identity["prompt"],
                                        record["input_text"], identity["budgets"]["max_output_tokens"])
        if (object_hash(payload) != entry["request_payload_hash"]
                or hashlib.sha256(record["input_text"].encode()).hexdigest() != record["input_sha256"]
                or record["prompt_sha256"] != identity["prompt_sha256"]
                or record["schema_sha256"] != object_hash(identity["schema"])):
            raise ValueError("frozen request mismatch")
        response = adapter.parse_response(record["provider"], (enabled / entry["artifacts"]["raw.json"]["path"]).read_bytes())
        if (response.response_id != entry["response_id"] or response.model != entry["response_model"]
                or response.usage != entry["usage"] or response.complete != entry["natural_stop"]):
            raise ValueError("raw metadata mismatch")
        document = core.strict_json(response.text, limits)
        actions = schema.validate_document(document)
        violation = first_local_violation(actions, limits)
        chain = {"provider": record["provider"], "scene_id": record["scene_id"], "condition": record["condition"],
                 "input_id": record["input_id"], "attempt_id": entry["attempt_id"],
                 "response_id": response.response_id, "response_model": response.model,
                 "raw_sha256": entry["artifacts"]["raw.json"]["sha256"],
                 "natural_stop": response.complete, "json_success": True, "schema_success": True,
                 "local_validator_success": violation is None, "executor_success": "NOT_RUN",
                 "payload_hash_matches_ledger": True, "input_prompt_schema_match_frozen_plan": True}
        row = dict(stored)
        if violation:
            if entry["status"] != "failed" or entry["failure_stage"] != "local_validator" or stored["metrics"] is not None:
                raise ValueError("stored failure differs from replay")
            failures.append({**chain, "first_violation": violation,
                             "rule_evidence": {"validator": "offline_evaluation/server_v2/core.py:material_id",
                                               "prompt": "Preserve exact lower-case material identifiers; do not change their spelling or case.",
                                               "schema_contract": "Shape-only schema permits strings; semantic checks deliberately remain local."}})
        else:
            normalized = core.validate_actions(actions, limits)
            pred = core.execute(normalized, limits)
            if (read(enabled / entry["artifacts"]["parsed.json"]["path"]) != {"operations": normalized}
                    or read(enabled / entry["artifacts"]["voxels.json"]["path"]) != core.grid_to_payload(pred)):
                raise ValueError("raw parsed voxel chain mismatch")
            chain.update(executor_success=True, raw_parsed_voxel_chain_exact=True)
            tick = time.monotonic()
            calculated = core.evaluate_pair(grids[record["scene_id"]], pred, alignment_mode="strict_translation")
            timings.append({"attempt_id": entry["attempt_id"], "seconds": time.monotonic() - tick})
            comparisons.append({"provider": record["provider"], "input_id": record["input_id"],
                                "attempt_id": entry["attempt_id"], "entire_metric_object_exact": calculated == stored["metrics"]})
            if calculated != stored["metrics"]:
                raise ValueError("metric replay mismatch; preserve diagnosis and investigate")
            row["metrics"] = calculated
            if record["condition"] == "E":
                spec = json.loads(record["input_text"].split("\n", 1)[1])
                decoded = independent_decode(spec)
                gt = grids[record["scene_id"]].as_world().voxels
                if decoded != gt:
                    raise ValueError("actual E specification fails independent GT comparison")
                e_rows.append({**chain, "independent_input_decode_labelled_gt_exact": True,
                    "input_spec_sha256": object_hash(spec), "input_palette": spec["palette"],
                    "absolute": voxel_difference(gt, pred.voxels),
                    "aligned_shift_xyz": calculated["alignment"]["shift_xyz"],
                    "aligned_metrics": calculated["aligned"],
                    "cuboid_differences": cuboid_difference(spec, normalized, pred.voxels)})
                e_details.append({"provider": record["provider"], "scene_id": record["scene_id"],
                                  "attempt_id": entry["attempt_id"], **transcription_details(spec, normalized)})
        chains.append(chain)
        rows.append(row)
    derived = summarize_v4(rows, smoke=True)
    if derived["groups"] != summary["groups"]:
        raise ValueError("group summary replay mismatch")
    flat = [flat_row(row) for row in rows]
    write(output / "all16.json", {"rows": rows, "flat_rows": flat, "failure_MAE_repair": None})
    with (output / "all16.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    write(output / "failures.json", {"failures": failures, "measurement_bug_found": False})
    write(output / "e_differences.json", {"rows": e_rows, "normalization": "frozen optional minecraft: namespace stripping and air normalization only; no alias mapping"})
    write(output / "e_transcription_details.json", {"rows": e_details})
    write(output / "metric_replay.json", {"comparisons": comparisons, "metric_evaluations": len(comparisons),
          "all_exact": all(c["entire_metric_object_exact"] for c in comparisons), "timings": timings})
    write(output / "request_chains.json", {"rows": chains})
    write(output / "summary.json", {**derived, "common_valid_OJ": common_valid_oj(rows), "used": summary["used"],
          "original_stage_counts": summary["stage_counts"], "old_smoke_comparison": "Historical context only; prompt/schema/E/cap differ.",
          "interpretation": "Descriptive two-building smoke only. E is discrete-specification conversion; P is incomplete attributes. No p-values or causal/generalized superiority claim."})
    after_at, after = datetime.now(timezone.utc).isoformat(), snapshot(originals)
    write(output / "original_hashes.after.json", {"observed_at_utc": after_at, "files": after,
                                                 "before_observed_at_utc": before_at, "all_unchanged": before == after})
    if before != after:
        raise ValueError("original changed during diagnosis")
    write(output / "execution.json", {"version": "goal005_offline_diagnosis_v1", "plan_path": str(plan_path),
          "plan_identity_hash": plan["identity_hash"], "plan_file_sha256": sha(plan_path),
          "python": platform.python_version(), "research_api_sends": 0, "tokenizer": "none",
          "cpu_thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
          "elapsed_seconds": time.monotonic() - started, "rows": len(rows), "valid_rows": len(comparisons),
          "failures": len(failures), "E_outputs": len(e_rows), "original_files_unchanged": len(before),
          "frozen_implementation_verified": True, "measurement_bug_found": False})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan")
    mode.add_argument("--verify-archive")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.verify_archive:
        if args.output:
            parser.error("archive verification is read-only; --output is not accepted")
        print(json.dumps(verify_archive(args.verify_archive), sort_keys=True, indent=2))
    else:
        if not args.output:
            parser.error("--plan requires a new --output directory")
        print(run(args.plan, args.output))
