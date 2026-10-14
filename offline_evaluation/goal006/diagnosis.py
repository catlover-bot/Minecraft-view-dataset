"""Minimal saved-result diagnostics after the Goal006 results checkpoint.

The completed-batch audit already executed every successful raw->voxel chain.
This module binds that evidence to unchanged originals, rechecks only failed
local validation, and replays eight fixed representative metric objects once.
It contains no activation, provider transport, repair, or inference entrypoint.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import time

from offline_evaluation.goal005.diagnosis import (
    first_local_violation, independent_decode, operation_bounds, read, sha,
    voxel_difference, write,
)
from offline_evaluation.resmoke_v2 import adapter, schema
from offline_evaluation.server_v2 import core
from offline_evaluation.server_v2.analysis_v4 import BOUNDED_IOU, bounded_all_attempted_value
from offline_evaluation.server_v2.data import load_scene
from offline_evaluation.server_v2.generation import object_hash

ROOT = Path(__file__).resolve().parents[2]
METRICS = tuple(sorted(BOUNDED_IOU)) + (
    "aligned_dimension_mae", "aligned_material_set_f1", "aligned_total_normalized_repair_cost")
REPRESENTATIVE_RULE = "first successful schedule row per provider and condition, in frozen schedule order"


def metric_value(row, name, *, failure0=False):
    if failure0:
        return bounded_all_attempted_value(row, name)
    if row["status"] != "success" or row.get("metrics") is None:
        return None
    frame, field = name.split("_", 1)
    return row["metrics"][frame][field]


def building_summary(rows, name, *, failure0=False):
    """Average descriptions within each building, then weight buildings equally."""
    by_scene = defaultdict(list)
    for row in rows:
        value = metric_value(row, name, failure0=failure0)
        if value is not None:
            by_scene[row["scene_id"]].append(value)
    buildings = [{"scene_id": scene, "rows": len(values), "mean": mean(values)}
                 for scene, values in sorted(by_scene.items())]
    return {"eligible_rows": sum(b["rows"] for b in buildings), "buildings": len(buildings),
            "equal_building_mean": mean(b["mean"] for b in buildings) if buildings else None,
            "by_building": buildings}


def denominators(rows):
    groups, contrasts, descriptions = [], [], []
    for provider in ("openai", "anthropic"):
        for condition in "OJPE":
            selected = [r for r in rows if r["provider"] == provider and r["condition"] == condition]
            group = {"provider": provider, "condition": condition, "planned_rows": len(selected),
                     "planned_buildings": len({r["scene_id"] for r in selected}),
                     "valid_rows": sum(r["status"] == "success" and r["metrics"] is not None for r in selected),
                     "failure_rows": sum(r["status"] == "failed" for r in selected),
                     "metrics": {name: {"valid_only": building_summary(selected, name),
                         **({"failure0": building_summary(selected, name, failure0=True)} if name in BOUNDED_IOU else {})}
                         for name in METRICS}}
            groups.append(group)
            if condition in "OJ":
                for description in ("openai", "anthropic", "gemini"):
                    subset = [r for r in selected if r["description_model"] == description]
                    descriptions.append({"provider": provider, "condition": condition, "description_model": description,
                        "planned_rows": len(subset), "metrics": {name: building_summary(subset, name) for name in METRICS}})
        sides = {condition: {(r["scene_id"], r["description_model"]): r for r in rows
                  if r["provider"] == provider and r["condition"] == condition} for condition in "OJ"}
        for name in METRICS:
            valid_keys = sorted(key for key in sides["O"].keys() & sides["J"].keys()
                                if all(metric_value(sides[c][key], name) is not None for c in "OJ"))
            building_deltas = defaultdict(list)
            pairs = []
            for scene, description in valid_keys:
                a, b = (metric_value(sides[c][(scene, description)], name) for c in "OJ")
                building_deltas[scene].append(b - a)
                pairs.append({"scene_id": scene, "description_model": description, "O": a, "J": b, "J_minus_O": b - a})
            means = {c: building_summary(list(sides[c].values()), name)["equal_building_mean"] for c in "OJ"}
            contrasts.append({"provider": provider, "metric": name,
                "unpaired_valid_group_mean_difference_J_minus_O": means["J"] - means["O"] if None not in means.values() else None,
                "common_valid_pair_count": len(pairs), "common_valid_buildings": sorted(building_deltas),
                "common_valid_building_count": len(building_deltas), "pairs": pairs,
                "common_valid_paired_equal_building_mean_J_minus_O": mean(mean(v) for v in building_deltas.values()) if building_deltas else None})
    return {"independent_unit": "building", "analysis": "interim descriptive; no p-values or superiority conclusion",
            "O_J_semantics": "J is information-preserving JSON reserialization of the saved O description, not the old ARR Structured IR.",
            "failure_MAE_repair": None, "groups": groups, "OJ_common_valid_contrasts": contrasts,
            "description_provider_tables": descriptions}


def p_attributes(spec, pred):
    required = {"occupied_dimensions_xyz", "world_bbox", "exact_material_set", "voxel_count"}
    if set(spec["provided_attributes"]) != required:
        raise ValueError("P attributes differ from frozen four-attribute contract")
    pred = pred.as_world()
    expected_materials, actual_materials = set(spec["exact_material_set"]), set(pred.voxels.values())
    expected = {k: spec[k] for k in sorted(required)}
    predicted = {"occupied_dimensions_xyz": list(pred.dimensions()), "world_bbox": pred.bbox(),
                 "exact_material_set": sorted(actual_materials), "voxel_count": len(pred.voxels)}
    errors = {"occupied_dimensions_xyz": [p - e for p, e in zip(predicted["occupied_dimensions_xyz"], expected["occupied_dimensions_xyz"])],
              "world_bbox": {k: predicted["world_bbox"][k] - value for k, value in expected["world_bbox"].items()} if pred.voxels else None,
              "voxel_count": len(pred.voxels) - expected["voxel_count"],
              "material_presence": {m: int(m in actual_materials) - int(m in expected_materials)
                                    for m in sorted(expected_materials | actual_materials)},
              "material_set_size": len(actual_materials) - len(expected_materials)}
    exact = {k: (actual_materials == expected_materials if k == "exact_material_set" else predicted[k] == expected[k]) for k in required}
    return {"input_attributes": expected, "predicted_attributes": predicted,
            "signed_errors_predicted_minus_input": errors, "attribute_satisfied": exact,
            "all_four_attributes_satisfied": all(exact.values()),
            "missing_materials": sorted(expected_materials - actual_materials),
            "extra_materials": sorted(actual_materials - expected_materials),
            "material_set_f1": 2 * len(expected_materials & actual_materials) / (len(expected_materials) + len(actual_materials))
                if expected_materials or actual_materials else 1.0}


def placement_evidence(gt, pred, actions):
    """Cite last saved operation for bounded examples, without executing outputs."""
    examples = []
    for label, coordinates in (
        ("predicted_occupied_GT_air", sorted(set(pred) - set(gt))[:5]),
        ("predicted_wrong_material", sorted(c for c in set(gt) & set(pred) if gt[c] != pred[c])[:5]),
    ):
        for coordinate in coordinates:
            for index in range(len(actions) - 1, -1, -1):
                bounds = operation_bounds(actions[index])
                if all(bounds[i] <= coordinate[i] <= bounds[i + 3] for i in range(3)):
                    if actions[index].get("block", "air") != pred[coordinate]:
                        raise ValueError("saved last-operation evidence disagrees with saved voxel")
                    examples.append({"kind": label, "coordinate_xyz": list(coordinate),
                        "GT_material": gt.get(coordinate, "air"), "predicted_material": pred[coordinate],
                        "last_operation_index_zero_based": index, "last_operation": actions[index]})
                    break
            else:
                raise ValueError("saved voxel has no covering operation")
    return {"examples": examples, "interpretation": "Observed final writes and GT disagreements only; no hidden reasoning or causal inference."}


def truncation_evidence(provider, raw, entry, reference):
    reason = raw.get("incomplete_details") if provider == "openai" else raw.get("stop_reason")
    return {"provider_status": raw.get("status"), "provider_stop_reason": reason,
            "provider_output_tokens": raw["usage"]["output_tokens"],
            "provider_reasoning_tokens": raw["usage"].get("output_tokens_details", {}).get("reasoning_tokens"),
            "charged_input_tokens": entry["charged_input_tokens"], "charged_output_tokens": entry["charged_output_tokens"],
            "usage": entry["usage"], "reference_action_tokens_local_o200k_base": reference["reference_action_tokens_local_o200k_base"],
            "reference_body_within_output_cap_local_openai": reference["reference_body_within_output_cap_local_openai"],
            "reference_count_is_provider_measurement": False,
            "interpretation": "Truncation is established by the actual provider finish reason. The reference is not a shortest representation; its local token count does not itself establish output failure or Anthropic token count."}


def representatives(schedule, rows):
    selected, seen = [], set()
    for index, (record, row) in enumerate(zip(schedule, rows)):
        key = record["provider"], record["condition"]
        if row["status"] == "success" and key not in seen:
            selected.append(index)
            seen.add(key)
    return selected


def e_summary(rows, details):
    selected = [r for r in rows if r["condition"] == "E"]
    valid = {p: {r["scene_id"] for r in selected if r["provider"] == p and r["status"] == "success"}
             for p in ("openai", "anthropic")}
    common = sorted(valid["openai"] & valid["anthropic"])
    groups = []
    for provider in ("openai", "anthropic"):
        subset = [r for r in selected if r["provider"] == provider]
        exact = [r for r in details if r["provider"] == provider and r["status"] == "success"]
        groups.append({"provider": provider, "planned_buildings": len(subset), "valid_buildings": len(valid[provider]),
            "valid_scene_ids": sorted(valid[provider]), "truncation_buildings": sum(r["failure_kind"] == "truncation" for r in subset),
            "absolute_occupancy_exact": sum(r["absolute"]["occupancy_exact"] for r in exact),
            "absolute_labelled_voxel_exact": sum(r["absolute"]["labelled_voxel_exact"] for r in exact),
            "aligned_occupancy_exact": sum(r["aligned_exact"]["occupancy_exact"] for r in exact),
            "aligned_labelled_voxel_exact": sum(r["aligned_exact"]["labelled_voxel_exact"] for r in exact),
            "metrics": {name: {"valid_only": building_summary(subset, name),
                "failure0": building_summary(subset, name, failure0=True),
                "common_valid": building_summary([r for r in subset if r["scene_id"] in common], name)} for name in sorted(BOUNDED_IOU)}})
    return {"groups": groups, "common_valid_scene_ids": common, "common_valid_buildings": len(common),
        "interpretation": "E provides the complete discrete labelled GT including interior and coordinates. E-O is not pure natural-language information loss. Unequal valid-only denominators do not establish provider superiority.",
        "rows": details}


def check_inventory(files):
    for item in files:
        path = Path(item["path"])
        if path.is_symlink() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("original evidence changed: " + str(path))


def run(plan_path, output, verified_evidence):
    started = time.monotonic()
    output, plan_path, verified_evidence = Path(output), Path(plan_path), Path(verified_evidence)
    if output.exists():
        raise ValueError("new diagnosis directory required")
    proof, plan = read(verified_evidence), read(plan_path)
    identity = plan["identity"]
    if (proof["status"] != "VERIFIED_COMPLETE_BATCH160" or proof["successful_raw_parsed_voxel_replays"] != 154
            or proof["plan_sha256"] != sha(plan_path) or proof["plan_identity_hash"] != plan["identity_hash"]
            or object_hash(identity) != plan["identity_hash"] or proof["raw_responses"] != 160):
        raise ValueError("completed execution-chain audit required")
    check_inventory(proof["files"])
    for relative, expected in identity["implementation_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("frozen implementation changed")
    enabled = Path(identity["enabled_directory"])
    ledger, original = read(enabled / "ledger.json"), read(enabled / "evaluation.json")
    rows, schedule = original["rows"], identity["schedule"]
    if len(rows) != 160 or len(ledger["entries"]) != 160 or len(schedule) != 160:
        raise ValueError("complete160 required")
    lengths_path = ROOT / "reports/server_only/goal005/planning/lengths.json"
    references = {r["scene_id"]: r for r in read(lengths_path)["rows"]}
    grids = {scene: load_scene(desc["building_directory"]).grid for scene, desc in identity["gt_by_scene"].items()}
    for scene, grid in grids.items():
        if object_hash(core.grid_to_payload(grid)) != identity["gt_by_scene"][scene]["grid_sha256"]:
            raise ValueError("GT identity mismatch")
    chosen = representatives(schedule, rows)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "representatives.json", {"rule": REPRESENTATIVE_RULE,
        "schedule_indices": [schedule[i]["schedule_index"] for i in chosen],
        "selection_uses_metric_values": False, "planned_replays": len(chosen)})
    failures, p_rows, e_rows, comparisons = [], [], [], []
    decoded_scenes = set()
    for index, (record, entry, row) in enumerate(zip(schedule, ledger["entries"], rows)):
        for key in ("provider", "input_id"):
            if record[key] != entry[key] or record[key] != row[key]:
                raise ValueError("row identity mismatch")
        if entry["attempt_id"] != row["attempt_id"] or entry["status"] != row["status"]:
            raise ValueError("attempt outcome mismatch")
        meta = {k: record.get(k) for k in ("provider", "scene_id", "condition", "description_model", "input_id", "schedule_index")}
        meta.update(attempt_id=entry["attempt_id"], raw_sha256=entry["artifacts"]["raw.json"]["sha256"], status=entry["status"])
        if entry["status"] == "failed":
            raw_path = enabled / entry["artifacts"]["raw.json"]["path"]
            raw = read(raw_path)
            failed = {**meta, "failure_kind": entry["failure_kind"], "failure_stage": entry["failure_stage"]}
            if entry["failure_kind"] == "validation_failure":
                response = adapter.parse_response(record["provider"], raw_path.read_bytes())
                actions = schema.validate_document(core.strict_json(response.text, core.Limits(**identity["limits"])))
                violation = first_local_violation(actions, core.Limits(**identity["limits"]))
                if violation is None:
                    raise ValueError("failed validator result not reproduced")
                failed["first_violation"] = violation
            elif entry["failure_kind"] == "truncation":
                failed.update(truncation_evidence(record["provider"], raw, entry, references[record["scene_id"]]))
            else:
                raise ValueError("unexpected failure category requires investigation")
            failures.append(failed)
            if record["condition"] == "E":
                e_rows.append({**meta, "failure_kind": entry["failure_kind"], "metrics": None})
            continue
        if index not in chosen and record["condition"] not in {"P", "E"}:
            continue
        pred = core.payload_to_grid(read(enabled / entry["artifacts"]["voxels.json"]["path"])).as_world()
        gt = grids[record["scene_id"]].as_world()
        if index in chosen:
            tick = time.monotonic()
            recalculated = core.evaluate_pair(grids[record["scene_id"]], pred, alignment_mode="strict_translation")
            comparisons.append({**meta, "entire_metric_dictionary_exact": recalculated == row["metrics"], "seconds": time.monotonic() - tick})
            if recalculated != row["metrics"]:
                write(output / "metric_mismatch.json", {**meta, "original": row["metrics"], "recalculated": recalculated})
                raise ValueError("measurement mismatch; separate versioned investigation required")
        if record["condition"] == "P":
            attrs = p_attributes(json.loads(record["input_text"]), pred)
            actions = read(enabled / entry["artifacts"]["parsed.json"]["path"])["operations"]
            p_rows.append({**meta, **attrs, "saved_metrics": row["metrics"],
                "parsed_sha256": entry["artifacts"]["parsed.json"]["sha256"],
                "voxels_sha256": entry["artifacts"]["voxels.json"]["sha256"],
                "placement_evidence": placement_evidence(gt.voxels, pred.voxels, actions)})
        if record["condition"] == "E":
            spec = json.loads(record["input_text"].split("\n", 1)[1])
            if record["scene_id"] not in decoded_scenes:
                if independent_decode(spec) != gt.voxels:
                    raise ValueError("E input differs from labelled GT")
                decoded_scenes.add(record["scene_id"])
            shifted = pred.shifted(tuple(row["metrics"]["alignment"]["shift_xyz"]))
            e_rows.append({**meta, "failure_kind": None, "input_spec_sha256": object_hash(spec),
                "absolute": voxel_difference(gt.voxels, pred.voxels),
                "aligned_shift_xyz": row["metrics"]["alignment"]["shift_xyz"],
                "aligned_exact": {"occupancy_exact": set(gt.voxels) == set(shifted.voxels), "labelled_voxel_exact": gt.voxels == shifted.voxels},
                "metrics": row["metrics"], "interpretation": "Actual saved output differs from complete labelled input; no material alias mapping or hidden-cause inference."})
    p_groups = []
    for provider in ("openai", "anthropic"):
        selected = [r for r in p_rows if r["provider"] == provider]
        p_groups.append({"provider": provider, "buildings": len(selected),
            "attribute_satisfied_counts": {k: sum(r["attribute_satisfied"][k] for r in selected) for k in selected[0]["attribute_satisfied"]},
            "all_four_attributes_satisfied": sum(r["all_four_attributes_satisfied"] for r in selected),
            "voxel_count_signed_errors": [{"scene_id": r["scene_id"], "error": r["signed_errors_predicted_minus_input"]["voxel_count"]} for r in selected],
            "mean_material_set_f1": mean(r["material_set_f1"] for r in selected),
            "mean_aligned_occupancy_iou": mean(r["saved_metrics"]["aligned"]["occupancy_iou"] for r in selected),
            "mean_aligned_material_aware_iou": mean(r["saved_metrics"]["aligned"]["material_aware_iou"] for r in selected)})
    write(output / "failures.json", {"rows": failures, "measurement_bug_found": False})
    write(output / "p_attributes.json", {"rows": p_rows, "groups": p_groups,
        "interpretation": "P contains only the four declared GT attributes, without original description, individual coordinates, or per-coordinate materials. Attribute agreement does not establish shape/material-placement agreement."})
    with (output / "p_attributes.csv").open("x", newline="") as stream:
        flat = [{"provider": r["provider"], "scene_id": r["scene_id"],
            "input_voxel_count": r["input_attributes"]["voxel_count"], "predicted_voxel_count": r["predicted_attributes"]["voxel_count"],
            "voxel_count_signed_error": r["signed_errors_predicted_minus_input"]["voxel_count"],
            **{k + "_satisfied": v for k, v in r["attribute_satisfied"].items()},
            "all_four_satisfied": r["all_four_attributes_satisfied"]} for r in p_rows]
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(flat)
    write(output / "e_diagnostics.json", e_summary(rows, e_rows))
    write(output / "denominators.json", denominators(rows))
    write(output / "metric_replay.json", {"selection_rule": REPRESENTATIVE_RULE, "metric_recalculations": len(comparisons),
        "all_exact": all(r["entire_metric_dictionary_exact"] for r in comparisons), "rows": comparisons,
        "successful_execution_replays_repeated": 0, "all154_metric_recalculation_performed": False})
    check_inventory(proof["files"])
    write(output / "execution.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path), "plan_sha256": sha(plan_path), "plan_identity_hash": plan["identity_hash"],
        "source_verification": str(verified_evidence), "source_verification_sha256": sha(verified_evidence),
        "source_inventory_sha256": object_hash(proof["files"]), "all_original_files_unchanged": len(proof["files"]),
        "prior_raw_parsed_voxel_replays_reused": 154, "new_successful_executor_replays": 0,
        "failures": len(failures), "P_rows": len(p_rows), "E_rows": len(e_rows), "metric_recalculations": len(comparisons),
        "measurement_bug_found": False, "research_api_additional_sends": 0,
        "reference_lengths_sha256": sha(lengths_path), "wall_seconds": time.monotonic() - started})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--verified-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args.plan, args.output, args.verified_evidence))
