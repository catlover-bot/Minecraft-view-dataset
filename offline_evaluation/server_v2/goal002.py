"""Finite CPU controls and separate diagnostic re-evaluation for Goal002.

No generation transport is used. Preserved v5 artifacts are not outputs of
either new Builder. Original hashes are checked before and after reading.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import csv
import json
from pathlib import Path
import resource
import time

from .core import Limits, execute, metrics, evaluate_pair
from .data import load_scene
from .experiment import _scenes
from .geometry_data import load_geometry, geometry_metrics, storage_truncation_evidence
from .io_utils import ROOT, atomic_json, read_json, sha256, writable
from .oracle import encode_exact_spec, decode_exact_spec, gt_to_actions


def _originals():
    return {r["path"]: r for r in map(json.loads,
        (ROOT / "reports/server_only/originals_manifest.jsonl").read_text().splitlines())}


def _verified_sources(paths, originals):
    sources = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Original source escaped repository")
        original = originals[relative]
        actual = sha256(path)
        if actual != original["sha256"]:
            raise ValueError("Original source hash differs from preserved reference")
        sources.append({"path": relative, "sha256": actual,
                        "ref": original["ref"], "blob_hash": original["blob_hash"]})
    return sources


def _pipeline_hashes():
    return {name: sha256(Path(__file__).parent / name)
            for name in ("goal002.py", "geometry_data.py", "core.py", "alignment.py", "data.py", "oracle.py")}


def _new_task(output_dir, filename, workers):
    output = writable(output_dir)
    if workers not in (1, 2):
        raise ValueError("worker count must be 1 or 2")
    if (output / filename).exists():
        raise FileExistsError("Completed CPU task is immutable; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def verify_strict_controls(output_dir, workers=2):
    from .alignment import select_translation
    output = _new_task(output_dir, "strict_gt_summary.json", workers)
    started = time.monotonic()
    originals, pipeline = _originals(), _pipeline_hashes()
    shifts = ((1, -1, 1), (20, -3, -17), (-100, 7, 80))
    far = (50_000, -20_000, 70_000)

    def one(path):
        gt_sources = _verified_sources([path / "gt/voxels.npy", path / "gt/bbox.json"], originals)
        scene = load_scene(path)
        source_hash = sha256(scene.paths["voxels"])
        spec = encode_exact_spec(scene.grid)
        actions = gt_to_actions(scene.grid)
        checks = []
        for displacement in shifts:
            prediction = scene.grid.shifted(displacement)
            result = evaluate_pair(scene.grid, prediction, alignment_mode="strict_translation")
            expected = [-v for v in displacement]
            checks.append({"prediction_displacement_xyz": list(displacement),
                           "expected_alignment_shift_xyz": expected, "metrics": result,
                           "pass": result["alignment"]["shift_xyz"] == expected and
                                   result["aligned"]["occupancy_iou"] == 1 and
                                   result["alignment"]["search_complete"]})
        far_gt = scene.grid.shifted(far)
        far_pred = scene.grid.shifted(tuple(far[i] + shifts[1][i] for i in range(3)))
        far_result = select_translation(far_gt.voxels, far_pred.voxels)
        row = {"scene_id": scene.scene_id, "gt_sha256": source_hash, "gt_original_sources": gt_sources,
               "gt_nonempty": bool(scene.grid.voxels), "gt_occupied": len(scene.grid.voxels),
               "exact_roundtrip": decode_exact_spec(spec).voxels == scene.grid.voxels,
               "executor_control": execute(actions).voxels == scene.grid.voxels,
               "reference_actions": len(actions), "reference_spec_bytes": len(json.dumps(spec).encode()),
               "limits": vars(Limits()), "known_shifts": checks,
               "far_origin_shift_xyz": list(far), "far_origin_alignment": far_result,
               "far_origin_pass": far_result["shift_xyz"] == [-v for v in shifts[1]] and
                                  far_result["maximum_occupancy_intersection"] == len(scene.grid.voxels),
               "source_unchanged": all(sha256(ROOT / r["path"]) == r["sha256"] for r in gt_sources),
               "llm_builder_success": None}
        atomic_json(output / "strict_controls" / (scene.scene_id + ".json"), row)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, _scenes()))
    summary = {"metric_version": "server_v2.2_strict_translation", "provenance": "new_implementation_cpu_controls",
               "scenes": len(rows), "nonempty_gt": sum(r["gt_nonempty"] for r in rows),
               "oracle_roundtrip_pass": sum(r["exact_roundtrip"] for r in rows),
               "executor_control_pass": sum(r["executor_control"] for r in rows),
               "known_shift_checks": len(rows) * len(shifts),
               "known_shift_pass": sum(c["pass"] for r in rows for c in r["known_shifts"]),
               "per_shift_pass": {str(shift): sum(r["known_shifts"][i]["pass"] for r in rows)
                                  for i, shift in enumerate(shifts)},
               "far_origin_pass": sum(r["far_origin_pass"] for r in rows),
               "source_hash_preserved": all(r["source_unchanged"] for r in rows),
               "workers": workers, "blas_threads": 1,
               "wall_seconds": time.monotonic() - started,
               "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "research_api_requests_sent": 0, "pipeline_source_hashes": pipeline,
               "pipeline_source_unchanged": _pipeline_hashes() == pipeline}
    atomic_json(output / "strict_gt_results.json", rows)
    atomic_json(output / "strict_gt_summary.json", summary)
    return summary


def diagnose_legacy(output_dir, workers=2):
    from .alignment import AlignmentResourceLimit, select_translation
    output = _new_task(output_dir, "legacy_geometry_summary.json", workers)
    started = time.monotonic()
    inventory = read_json(ROOT / "reports/server_only/available_predictions.json")
    originals, pipeline = _originals(), _pipeline_hashes()
    scenes, gt_sources = {}, {}
    for path in _scenes():
        sources = _verified_sources([path / "gt/voxels.npy", path / "gt/bbox.json"], originals)
        scene = load_scene(path)
        if not all(sha256(ROOT / r["path"]) == r["sha256"] for r in sources):
            raise ValueError("GT source changed during load")
        scenes[scene.scene_id], gt_sources[scene.scene_id] = scene, sources

    def one(item):
        path = ROOT / item["voxel_path"]
        directory = path.parent
        scene_id = directory.parents[1].name + "_" + directory.parent.name
        scene = scenes[scene_id]
        source_paths = [path, directory / "bbox.json", directory / "actions.json"]
        sources = _verified_sources(source_paths, originals)
        read = load_geometry(path, directory / "bbox.json")
        evidence = storage_truncation_evidence(read, read_json(directory / "actions.json"))
        row = {"scene_id": scene_id, "dataset_version": directory.parents[1].name,
               "condition": directory.name, "original_sources": sources, "gt_original_sources": gt_sources[scene_id],
               "provenance": "new_implementation_diagnostic_on_preserved_legacy_i2t2b_v5",
               "historical_arr_replay": "BLOCKED_MISSING_ORIGINALS",
               "raw_preserved": True,
               "raw_preserved_definition": "original saved voxel/bbox/action bytes; not provider raw response availability",
               "provider_raw_response_availability": "NOT_ESTABLISHED",
               "geometry_valid": read.geometry_valid, "material_valid": read.material_valid,
               "material_validity": "VALID" if read.material_valid else "UNKNOWN",
               "unresolved": not (read.geometry_valid and read.material_valid),
               "invalid_labels": read.invalid_labels, "confirmed_storage_truncation": evidence,
               "npy_header": read.header, "bbox": read.bbox, "raw_label_counts": read.raw_labels,
               "reconstructed_data_created": False,
               "reconstruction_disposition": "full truncated action labels are still ambiguous; no GT filling or material remapping" if evidence else "original data retained",
               "absolute": None, "bounded_radius1_aligned": None,
               "strict_translation_aligned": None, "alignment": None,
               "status": "GEOMETRY_UNKNOWN", "llm_execution_failure_claimed": False}
        if read.geometry_valid:
            metric_fn = (lambda shift: metrics(scene.grid, read.material_grid.shifted(shift))) if read.material_valid else (
                lambda shift: geometry_metrics(scene.grid.voxels, read.coordinates, shift))
            row["absolute"] = metric_fn((0, 0, 0))
            bounded = select_translation(scene.grid.voxels, read.coordinates, mode="bounded", radius=1)
            row["bounded_radius1_alignment"] = bounded
            row["bounded_radius1_aligned"] = metric_fn(tuple(bounded["shift_xyz"]))
            try:
                alignment = select_translation(scene.grid.voxels, read.coordinates)
                row["alignment"] = alignment
                row["strict_translation_aligned"] = metric_fn(tuple(alignment["shift_xyz"]))
                row["status"] = "MATERIAL_AND_GEOMETRY_VALID" if read.material_valid else "GEOMETRY_ONLY_MATERIAL_UNKNOWN"
            except AlignmentResourceLimit as exc:
                row["status"], row["alignment"] = "ALIGNMENT_RESOURCE_LIMIT", exc.metadata
        if not all(sha256(ROOT / r["path"]) == r["sha256"] for r in sources + gt_sources[scene_id]):
            raise ValueError("Original source changed during diagnostic read")
        atomic_json(output / "legacy_geometry" / scene_id / (directory.name + ".json"), row)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, inventory))
    atomic_json(output / "legacy_geometry_results.json", rows)
    csv_fields = ["scene_id", "condition", "raw_preserved", "geometry_valid", "material_valid", "unresolved",
                  "absolute_occupancy_iou", "absolute_dimension_mae", "bounded_radius1_aligned_occupancy_iou",
                  "strict_translation_aligned_occupancy_iou", "strict_minus_bounded1_iou",
                  "strict_translation_shift_x", "strict_translation_shift_y", "strict_translation_shift_z",
                  "strict_translation_aligned_material_aware_iou", "strict_translation_aligned_total_normalized_repair_cost"]
    with (output / "alignment_comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in csv_fields if key in row}
            for prefix, key in (("absolute", "absolute"), ("bounded_radius1_aligned", "bounded_radius1_aligned"),
                                ("strict_translation_aligned", "strict_translation_aligned")):
                for metric, value in (row[key] or {}).items():
                    if prefix + "_" + metric in csv_fields:
                        flat[prefix + "_" + metric] = value
            if row["strict_translation_aligned"]:
                flat["strict_minus_bounded1_iou"] = row["strict_translation_aligned"]["occupancy_iou"] - row["bounded_radius1_aligned"]["occupancy_iou"]
                for axis, shift in zip("xyz", row["alignment"]["shift_xyz"]):
                    flat["strict_translation_shift_" + axis] = shift
            writer.writerow(flat)
    invalid = [r for r in rows if not r["material_valid"]]
    atomic_json(output / "material_diagnostics.json", invalid)
    conditions = sorted({r["condition"] for r in rows})
    groups, paired = [], []
    for condition in conditions:
        group = [r for r in rows if r["condition"] == condition]
        geometry = [r for r in group if r["strict_translation_aligned"]]
        groups.append({"condition": condition, "total": len(group),
                       "geometry_valid": sum(r["geometry_valid"] for r in group),
                       "material_valid": sum(r["material_valid"] for r in group),
                       "unresolved": sum(r["unresolved"] for r in group),
                       "strict_alignment_completed": len(geometry),
                       "strict_occupancy_iou_mean": sum(r["strict_translation_aligned"]["occupancy_iou"] for r in geometry) / len(geometry) if geometry else None,
                       "bounded1_occupancy_iou_mean_same_subset": sum(r["bounded_radius1_aligned"]["occupancy_iou"] for r in geometry) / len(geometry) if geometry else None})
    by_condition = {condition: {r["scene_id"]: r for r in rows if r["condition"] == condition} for condition in conditions}
    for a, b in combinations(conditions, 2):
        common = sorted(set(by_condition[a]) & set(by_condition[b]))
        valid = [s for s in common if by_condition[a][s]["material_valid"] and by_condition[b][s]["material_valid"]]
        paired.append({"condition_a": a, "condition_b": b, "available_pair_denominator": len(common),
                       "common_geometry_valid": sum(by_condition[a][s]["geometry_valid"] and by_condition[b][s]["geometry_valid"] for s in common),
                       "common_material_valid": len(valid), "common_material_valid_scene_ids": valid})
        paired[-1]["common_strict_geometry_metric_available"] = sum(bool(by_condition[a][s]["strict_translation_aligned"]) and bool(by_condition[b][s]["strict_translation_aligned"]) for s in common)
        paired[-1]["common_strict_material_metric_available"] = sum(bool(by_condition[a][s]["strict_translation_aligned"]) and bool(by_condition[b][s]["strict_translation_aligned"]) for s in valid)
    summary = {"total": len(rows), "raw_preserved": sum(r["raw_preserved"] for r in rows),
               "geometry_valid": sum(r["geometry_valid"] for r in rows),
               "material_valid": sum(r["material_valid"] for r in rows),
               "unresolved": len(invalid), "affected_buildings": len({r["scene_id"] for r in invalid}),
               "confirmed_storage_truncation_artifacts": sum(bool(r["confirmed_storage_truncation"]) for r in rows),
               "strict_alignment_completed": sum(bool(r["strict_translation_aligned"]) for r in rows),
               "alignment_resource_limits": sum(r["status"] == "ALIGNMENT_RESOURCE_LIMIT" for r in rows),
               "strict_better_than_radius1": sum(bool(r["strict_translation_aligned"]) and r["strict_translation_aligned"]["occupancy_iou"] > r["bounded_radius1_aligned"]["occupancy_iou"] + 1e-12 for r in rows),
               "all_six_common_material_valid_buildings": len(set.intersection(*({s for s,r in by_condition[c].items() if r["material_valid"]} for c in conditions))),
               "groups": groups, "pairwise_common_valid_subsets": paired,
               "workers": workers, "blas_threads": 1, "wall_seconds": time.monotonic() - started,
               "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "originals_unchanged": True, "reconstructed_outputs": 0,
               "historical_model_or_minecraft_execution_failures_inferred": False,
               "research_api_requests_sent": 0, "pipeline_source_hashes": pipeline,
               "pipeline_source_unchanged": _pipeline_hashes() == pipeline}
    atomic_json(output / "legacy_geometry_summary.json", summary)
    return summary
