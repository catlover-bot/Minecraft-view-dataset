#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from tools.evaluate_rebuild_metrics import (
    DEFAULT_THRESHOLDS,
    component_match_metrics,
    load_json,
    load_voxels,
    material_metrics,
    occupancy_metrics,
    search_best_shift,
    shift_map,
    voxel_maps,
)
from tools.render_rebuild_from_plan import _apply_fill, _int, _op_bounds, _resolve_bbox

Coord3D = Tuple[int, int, int]


@dataclass
class SubmissionPayload:
    map_data: Dict[Coord3D, str]
    bbox: Dict[str, int]
    mode: str
    derived_paths: Dict[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score human image->rebuild submissions against GT with repair metrics.")
    p.add_argument(
        "--cases_manifest",
        default="reports/final/original_benchmark_human_image_rebuild_cases.json",
        help="Manifest created by prepare_original_benchmark_human_image_rebuild_pilot.py",
    )
    p.add_argument("--submission_root", default="outputs/human_image_rebuild/submissions")
    p.add_argument("--out_root", default="outputs/human_image_rebuild/scored_submissions")
    p.add_argument("--thresholds_json", default="tools/thresholds_levels.example.json")
    p.add_argument("--max_dim", type=int, default=256)
    p.add_argument("--allow_plan_fallback", action="store_true", help="Allow plan.json fallback if voxels.npy is missing.")
    p.add_argument("--allow_voxels_json", action="store_true", help="Allow voxels.json fallback if voxels.npy is missing.")
    p.add_argument("--participant_glob", default="*")
    p.add_argument("--condition_glob", default="*")
    p.add_argument("--case_glob", default="*")
    return p.parse_args()


def _safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else float(n) / float(d)


def _neighbors6(c: Coord3D) -> Iterable[Coord3D]:
    x, y, z = c
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _component_count(points: Set[Coord3D]) -> int:
    rem = set(points)
    n = 0
    while rem:
        n += 1
        s = rem.pop()
        stack = [s]
        while stack:
            cur = stack.pop()
            for nxt in _neighbors6(cur):
                if nxt in rem:
                    rem.remove(nxt)
                    stack.append(nxt)
    return n


def _render_plan_to_voxels(plan_path: Path, out_dir: Path, max_dim: int) -> SubmissionPayload:
    plan = load_json(plan_path)
    bbox = _resolve_bbox(plan)

    sx = int(bbox["xmax"]) - int(bbox["xmin"]) + 1
    sy = int(bbox["ymax"]) - int(bbox["ymin"]) + 1
    sz = int(bbox["zmax"]) - int(bbox["zmin"]) + 1
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"invalid bbox from plan: {plan_path}")
    if max(sx, sy, sz) > int(max_dim):
        raise ValueError(f"plan bbox too large ({sx}x{sy}x{sz}) > max_dim={max_dim}: {plan_path}")

    vox = np.full((sy, sx, sz), "air", dtype="<U32")
    ops = plan.get("operations", []) if isinstance(plan.get("operations"), list) else []

    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op", "")).strip().lower()
        if kind not in {"fill", "carve", "set"}:
            continue

        if kind in {"fill", "carve"}:
            x1, x2, y1, y2, z1, z2 = _op_bounds(op)
            block = "air" if kind == "carve" else str(op.get("block", "air"))
            _apply_fill(vox, bbox, x1, y1, z1, x2, y2, z2, block)
        else:
            x = _int(op.get("x", 0), 0)
            y = _int(op.get("y", 0), 0)
            z = _int(op.get("z", 0), 0)
            block = str(op.get("block", "air"))
            _apply_fill(vox, bbox, x, y, z, x, y, z, block)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_vox = out_dir / "voxels.npy"
    out_bbox = out_dir / "bbox.json"
    np.save(out_vox, vox)
    out_bbox.write_text(json.dumps(bbox, ensure_ascii=False, indent=2), encoding="utf-8")

    return SubmissionPayload(
        map_data=voxel_maps(vox, bbox),
        bbox=bbox,
        mode="plan_json",
        derived_paths={"voxels": str(out_vox), "bbox": str(out_bbox)},
    )


def _voxels_json_to_payload(vox_json_path: Path, bbox_path: Optional[Path], out_dir: Path) -> SubmissionPayload:
    raw = load_json(vox_json_path)
    bbox: Dict[str, Any]
    voxels_arr: Any
    if isinstance(raw, dict) and isinstance(raw.get("voxels"), list):
        voxels_arr = raw["voxels"]
        if isinstance(raw.get("bbox"), dict):
            bbox = raw["bbox"]
        elif bbox_path and bbox_path.is_file():
            bbox = load_json(bbox_path)
        else:
            raise ValueError(f"voxels.json missing bbox and bbox.json not found: {vox_json_path}")
    elif isinstance(raw, list):
        voxels_arr = raw
        if not (bbox_path and bbox_path.is_file()):
            raise ValueError(f"list-form voxels.json requires bbox.json: {vox_json_path}")
        bbox = load_json(bbox_path)
    else:
        raise ValueError(f"unsupported voxels.json schema: {vox_json_path}")

    arr = np.array(voxels_arr)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_vox = out_dir / "voxels.npy"
    out_bbox = out_dir / "bbox.json"
    np.save(out_vox, arr)
    out_bbox.write_text(json.dumps(bbox, ensure_ascii=False, indent=2), encoding="utf-8")

    return SubmissionPayload(
        map_data=voxel_maps(arr, bbox),
        bbox={k: int(v) for k, v in bbox.items() if k in {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}},
        mode="voxels_json",
        derived_paths={"voxels": str(out_vox), "bbox": str(out_bbox)},
    )


def _load_submission_payload(
    case_dir: Path,
    derived_dir: Path,
    allow_plan_fallback: bool,
    allow_voxels_json: bool,
    max_dim: int,
) -> SubmissionPayload:
    vox_path = case_dir / "voxels.npy"
    bbox_path = case_dir / "bbox.json"
    if vox_path.is_file() and bbox_path.is_file():
        bbox = load_json(bbox_path)
        arr = load_voxels(vox_path)
        return SubmissionPayload(
            map_data=voxel_maps(arr, bbox),
            bbox={k: int(v) for k, v in bbox.items() if k in {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}},
            mode="voxels_npy",
            derived_paths={"voxels": str(vox_path), "bbox": str(bbox_path)},
        )

    if allow_plan_fallback:
        plan_path = case_dir / "plan.json"
        if plan_path.is_file():
            return _render_plan_to_voxels(plan_path, derived_dir, max_dim=max_dim)

    if allow_voxels_json:
        vox_json_path = case_dir / "voxels.json"
        if vox_json_path.is_file():
            return _voxels_json_to_payload(vox_json_path, bbox_path if bbox_path.is_file() else None, derived_dir)

    raise FileNotFoundError(
        f"submission files not found (expected bbox+voxels.npy or plan.json fallback): {case_dir}"
    )


def _repair_metrics(gt_map: Dict[Coord3D, str], pred_shifted: Dict[Coord3D, str]) -> Dict[str, Any]:
    gt_occ = set(gt_map.keys())
    pred_occ = set(pred_shifted.keys())

    gt_only = gt_occ - pred_occ
    pred_only = pred_occ - gt_occ
    overlap = gt_occ & pred_occ
    replacements = {c for c in overlap if gt_map[c] != pred_shifted[c]}

    additions = len(gt_only)
    deletions = len(pred_only)
    repl = len(replacements)
    total = additions + deletions + repl
    gt_non_air = len(gt_occ)
    correct = len(overlap) - repl

    k_steps = [50, 100, 200, 500]
    completion_after_k = {
        str(k): min(1.0, _safe_div(correct + min(k, total), gt_non_air)) for k in k_steps
    }

    return {
        "counts": {
            "gt_non_air": gt_non_air,
            "pred_non_air": len(pred_occ),
            "correct_after_shift": correct,
            "additions_needed": additions,
            "deletions_needed": deletions,
            "replacements_needed": repl,
            "total_edit_operations": total,
        },
        "normalized": {
            "edit_distance_over_gt": _safe_div(total, gt_non_air),
            "additions_over_gt": _safe_div(additions, gt_non_air),
            "deletions_over_gt": _safe_div(deletions, gt_non_air),
            "replacements_over_gt": _safe_div(repl, gt_non_air),
        },
        "completion_after_k_edits": completion_after_k,
        "approx_cuboid_repair_count": {
            "additions_components": _component_count(gt_only) if gt_only else 0,
            "deletions_components": _component_count(pred_only) if pred_only else 0,
            "replacements_components": _component_count(replacements) if replacements else 0,
        },
    }


def _mean(xs: List[float]) -> float:
    return 0.0 if not xs else float(sum(xs) / len(xs))


def _agg(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "n": float(len(rows)),
        "iou": _mean([float(r["metrics"]["iou"]) for r in rows]),
        "f1": _mean([float(r["metrics"]["f1"]) for r in rows]),
        "material_match": _mean([float(r["metrics"]["material_match"]) for r in rows]),
        "coarse_material_match": _mean([float(r["metrics"]["coarse_material_match"]) for r in rows]),
        "correct_placement_rate": _mean([float(r["metrics"]["correct_placement_rate"]) for r in rows]),
        "edit_distance_over_gt": _mean([float(r["repair"]["normalized"]["edit_distance_over_gt"]) for r in rows]),
        "additions_over_gt": _mean([float(r["repair"]["normalized"]["additions_over_gt"]) for r in rows]),
        "deletions_over_gt": _mean([float(r["repair"]["normalized"]["deletions_over_gt"]) for r in rows]),
        "replacements_over_gt": _mean([float(r["repair"]["normalized"]["replacements_over_gt"]) for r in rows]),
    }


def _load_thresholds(path: Path) -> Dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.setdefault("max_shift_xy", 48.0)
    thresholds.setdefault("max_shift_y", 8.0)
    thresholds.setdefault("top_shift_candidates", 24.0)
    if path.is_file():
        payload = load_json(path)
        if isinstance(payload, dict):
            for k, v in payload.items():
                if k in thresholds:
                    thresholds[k] = float(v)
    return thresholds


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.cases_manifest).resolve()
    submission_root = Path(args.submission_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not manifest_path.is_file():
        raise SystemExit(f"cases_manifest not found: {manifest_path}")
    if not submission_root.is_dir():
        raise SystemExit(f"submission_root not found: {submission_root}")

    manifest = load_json(manifest_path)
    cases = manifest.get("cases", []) if isinstance(manifest.get("cases"), list) else []
    case_map: Dict[str, Dict[str, Any]] = {str(c.get("case_id")): c for c in cases}
    if not case_map:
        raise SystemExit(f"no cases found in manifest: {manifest_path}")

    thresholds = _load_thresholds(Path(args.thresholds_json))

    out_root.mkdir(parents=True, exist_ok=True)
    derived_root = out_root / "derived_from_secondary"
    derived_root.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    invalid_items: List[Dict[str, Any]] = []

    participant_dirs = sorted([p for p in submission_root.glob(args.participant_glob) if p.is_dir()])
    for pdir in participant_dirs:
        participant_id = pdir.name
        condition_dirs = sorted([c for c in pdir.glob(args.condition_glob) if c.is_dir()])
        for cdir in condition_dirs:
            condition = cdir.name
            case_dirs = sorted([d for d in cdir.glob(args.case_glob) if d.is_dir()])
            for case_dir in case_dirs:
                case_id = case_dir.name
                case_info = case_map.get(case_id)
                if case_info is None:
                    invalid_items.append(
                        {
                            "participant_id": participant_id,
                            "condition": condition,
                            "case_id": case_id,
                            "submission_path": str(case_dir),
                            "status": "invalid_case_id",
                            "reason": "case_id not found in manifest",
                        }
                    )
                    continue

                gt_bbox_path = Path(str(case_info.get("gt_bbox_path", "")))
                gt_vox_path = Path(str(case_info.get("gt_voxels_path", "")))
                if not (gt_bbox_path.is_file() and gt_vox_path.is_file()):
                    invalid_items.append(
                        {
                            "participant_id": participant_id,
                            "condition": condition,
                            "case_id": case_id,
                            "submission_path": str(case_dir),
                            "status": "missing_gt",
                            "reason": f"gt artifact missing for case {case_id}",
                        }
                    )
                    continue

                try:
                    payload = _load_submission_payload(
                        case_dir=case_dir,
                        derived_dir=derived_root / participant_id / condition / case_id,
                        allow_plan_fallback=bool(args.allow_plan_fallback),
                        allow_voxels_json=bool(args.allow_voxels_json),
                        max_dim=int(args.max_dim),
                    )
                except Exception as exc:  # noqa: BLE001
                    invalid_items.append(
                        {
                            "participant_id": participant_id,
                            "condition": condition,
                            "case_id": case_id,
                            "submission_path": str(case_dir),
                            "status": "invalid_submission",
                            "reason": str(exc),
                        }
                    )
                    continue

                gt_map = voxel_maps(load_voxels(gt_vox_path), load_json(gt_bbox_path))
                pred_map = payload.map_data

                gt_occ = set(gt_map.keys())
                pred_occ = set(pred_map.keys())

                shift = search_best_shift(
                    gt_occ,
                    pred_occ,
                    max_shift_xy=int(thresholds["max_shift_xy"]),
                    max_shift_y=int(thresholds["max_shift_y"]),
                    top_candidates=int(thresholds["top_shift_candidates"]),
                )
                pred_shifted = shift_map(pred_map, shift.dx, shift.dy, shift.dz)
                pred_occ_s = set(pred_shifted.keys())

                occ = occupancy_metrics(gt_occ, pred_occ_s)
                mat = material_metrics(gt_map, pred_shifted)
                comp = component_match_metrics(
                    gt_occ,
                    pred_occ_s,
                    iou_threshold=float(thresholds["component_iou_match_threshold"]),
                )
                strict_match_count = float(mat.get("strict_match_count", 0.0))
                relaxed_match_count = float(mat.get("relaxed_match_count", 0.0))
                mat["correct_placement_rate"] = _safe_div(strict_match_count, len(pred_occ_s))
                mat["correct_placement_coverage"] = _safe_div(strict_match_count, len(gt_occ))
                mat["correct_placement_rate_relaxed_id"] = _safe_div(relaxed_match_count, len(pred_occ_s))
                mat["correct_placement_coverage_relaxed_id"] = _safe_div(relaxed_match_count, len(gt_occ))
                repair = _repair_metrics(gt_map, pred_shifted)

                llm_baselines = case_info.get("llm_baselines", {}) if isinstance(case_info.get("llm_baselines"), dict) else {}
                item = {
                    "participant_id": participant_id,
                    "condition": condition,
                    "case_id": case_id,
                    "dataset_split": case_info.get("dataset_split"),
                    "building_id": case_info.get("building_id"),
                    "difficulty": case_info.get("difficulty"),
                    "submission_path": str(case_dir),
                    "submission_mode": payload.mode,
                    "derived_paths": payload.derived_paths,
                    "shift": {"dx": shift.dx, "dy": shift.dy, "dz": shift.dz},
                    "metrics": {**occ, **mat, **comp},
                    "repair": repair,
                    "llm_baselines_main": llm_baselines,
                    "status": "ok",
                }
                items.append(item)

    # Aggregates
    by_participant_condition: Dict[str, Dict[str, float]] = {}
    for key in sorted({f"{x['participant_id']}::{x['condition']}" for x in items}):
        rows = [x for x in items if f"{x['participant_id']}::{x['condition']}" == key]
        by_participant_condition[key] = _agg(rows)

    by_condition: Dict[str, Dict[str, float]] = {}
    for cond in sorted({x["condition"] for x in items}):
        rows = [x for x in items if x["condition"] == cond]
        by_condition[cond] = _agg(rows)

    by_case: Dict[str, Dict[str, float]] = {}
    for cid in sorted({x["case_id"] for x in items}):
        rows = [x for x in items if x["case_id"] == cid]
        by_case[cid] = _agg(rows)

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases_manifest": str(manifest_path),
        "submission_root": str(submission_root),
        "note": "Scoring infrastructure output. May include placeholder/validation submissions; not a human study result claim.",
        "coverage": {
            "participants_detected": len(participant_dirs),
            "scored_submissions": len(items),
            "invalid_submissions": len(invalid_items),
        },
        "aggregates": {
            "by_participant_condition": by_participant_condition,
            "by_condition": by_condition,
            "by_case": by_case,
        },
        "items": items,
        "invalid_items": invalid_items,
    }

    out_json = out_root / "human_scores.json"
    out_csv = out_root / "human_scores.csv"
    out_summary_md = out_root / "human_scores_summary.md"
    out_compare_csv = out_root / "human_vs_llm_case_table.csv"

    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Flat CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "participant_id",
            "condition",
            "case_id",
            "dataset_split",
            "building_id",
            "difficulty",
            "submission_mode",
            "iou",
            "f1",
            "material_match",
            "coarse_material_match",
            "correct_placement_rate",
            "edit_distance_over_gt",
            "additions_over_gt",
            "deletions_over_gt",
            "replacements_over_gt",
            "submission_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for it in items:
            writer.writerow(
                {
                    "participant_id": it["participant_id"],
                    "condition": it["condition"],
                    "case_id": it["case_id"],
                    "dataset_split": it["dataset_split"],
                    "building_id": it["building_id"],
                    "difficulty": it["difficulty"],
                    "submission_mode": it["submission_mode"],
                    "iou": it["metrics"]["iou"],
                    "f1": it["metrics"]["f1"],
                    "material_match": it["metrics"]["material_match"],
                    "coarse_material_match": it["metrics"]["coarse_material_match"],
                    "correct_placement_rate": it["metrics"]["correct_placement_rate"],
                    "edit_distance_over_gt": it["repair"]["normalized"]["edit_distance_over_gt"],
                    "additions_over_gt": it["repair"]["normalized"]["additions_over_gt"],
                    "deletions_over_gt": it["repair"]["normalized"]["deletions_over_gt"],
                    "replacements_over_gt": it["repair"]["normalized"]["replacements_over_gt"],
                    "submission_path": it["submission_path"],
                }
            )

    # Comparison CSV against LLM baselines (case-level rows per submission)
    with out_compare_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "participant_id",
            "condition",
            "case_id",
            "dataset_split",
            "building_id",
            "human_iou",
            "human_f1",
            "human_material_match",
            "human_correct_placement_rate",
            "human_edit_distance_over_gt",
            "openai_direct_iou",
            "openai_structured_iou",
            "openai_direct_edit_distance_over_gt",
            "openai_structured_edit_distance_over_gt",
            "claude_direct_iou",
            "claude_structured_iou",
            "claude_direct_edit_distance_over_gt",
            "claude_structured_edit_distance_over_gt",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for it in items:
            baselines = it.get("llm_baselines_main", {})
            o = baselines.get("openai_main", {}) if isinstance(baselines, dict) else {}
            c = baselines.get("claude_main", {}) if isinstance(baselines, dict) else {}
            writer.writerow(
                {
                    "participant_id": it["participant_id"],
                    "condition": it["condition"],
                    "case_id": it["case_id"],
                    "dataset_split": it["dataset_split"],
                    "building_id": it["building_id"],
                    "human_iou": it["metrics"]["iou"],
                    "human_f1": it["metrics"]["f1"],
                    "human_material_match": it["metrics"]["material_match"],
                    "human_correct_placement_rate": it["metrics"]["correct_placement_rate"],
                    "human_edit_distance_over_gt": it["repair"]["normalized"]["edit_distance_over_gt"],
                    "openai_direct_iou": o.get("direct_iou", ""),
                    "openai_structured_iou": o.get("structured_iou", ""),
                    "openai_direct_edit_distance_over_gt": o.get("direct_edit_distance_over_gt", ""),
                    "openai_structured_edit_distance_over_gt": o.get("structured_edit_distance_over_gt", ""),
                    "claude_direct_iou": c.get("direct_iou", ""),
                    "claude_structured_iou": c.get("structured_iou", ""),
                    "claude_direct_edit_distance_over_gt": c.get("direct_edit_distance_over_gt", ""),
                    "claude_structured_edit_distance_over_gt": c.get("structured_edit_distance_over_gt", ""),
                }
            )

    summary_lines = [
        "# Human Image->Rebuild Scoring Summary",
        "",
        "このファイルは採点パイプライン出力です。人間被験者結果の主張には使いません（プレースホルダ提出を含む可能性があります）。",
        "",
        f"- participants_detected: {len(participant_dirs)}",
        f"- scored_submissions: {len(items)}",
        f"- invalid_submissions: {len(invalid_items)}",
        "",
        "## Condition aggregates",
    ]
    for cond, vals in sorted(by_condition.items()):
        summary_lines.append(
            f"- {cond}: n={int(vals['n'])}, IoU={vals['iou']:.4f}, F1={vals['f1']:.4f}, "
            f"material={vals['material_match']:.4f}, correct={vals['correct_placement_rate']:.4f}, edit={vals['edit_distance_over_gt']:.4f}"
        )
    out_summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[score_human_image_rebuild_submissions] wrote {out_json}")
    print(f"[score_human_image_rebuild_submissions] wrote {out_csv}")
    print(f"[score_human_image_rebuild_submissions] wrote {out_summary_md}")
    print(f"[score_human_image_rebuild_submissions] wrote {out_compare_csv}")


if __name__ == "__main__":
    main()
