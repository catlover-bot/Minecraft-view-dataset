#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools.evaluate_rebuild_metrics import (
    component_match_metrics,
    load_json,
    load_voxels,
    material_metrics,
    occupancy_metrics,
    parse_thresholds,
    search_best_shift,
    shift_map,
    voxel_maps,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score human submissions against GT using rebuild metric logic.")
    p.add_argument("--gt_root", required=True)
    p.add_argument("--submission_root", required=True, help="submissions/<participant>/<condition>/<case>")
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--thresholds_json", default="tools/thresholds_levels.example.json")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _list_participants(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root).resolve()
    sub_root = Path(args.submission_root).resolve()
    if not gt_root.is_dir():
        raise SystemExit(f"gt_root not found: {gt_root}")
    if not sub_root.is_dir():
        raise SystemExit(f"submission_root not found: {sub_root}")

    thresholds = parse_thresholds(Path(args.thresholds_json))
    items: List[Dict[str, Any]] = []

    for participant_dir in _list_participants(sub_root):
        pid = participant_dir.name
        for cond_dir in sorted([p for p in participant_dir.iterdir() if p.is_dir()]):
            cond = cond_dir.name
            for case_dir in sorted([p for p in cond_dir.glob(args.building_pattern) if p.is_dir()]):
                case_id = case_dir.name
                gt_bbox = gt_root / case_id / "gt" / "bbox.json"
                gt_vox = gt_root / case_id / "gt" / "voxels.npy"
                pred_bbox = case_dir / "bbox.json"
                pred_vox = case_dir / "voxels.npy"
                if not (gt_bbox.is_file() and gt_vox.is_file() and pred_bbox.is_file() and pred_vox.is_file()):
                    continue

                gt_map = voxel_maps(load_voxels(gt_vox), load_json(gt_bbox))
                pred_map = voxel_maps(load_voxels(pred_vox), load_json(pred_bbox))
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

                metrics = {
                    **occ,
                    **mat,
                    **comp,
                }
                item = {
                    "participant_id": pid,
                    "condition": cond,
                    "building": case_id,
                    "submission_path": str(case_dir),
                    "metrics": metrics,
                }
                items.append(item)

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    group_summary: Dict[str, Dict[str, float]] = {}
    for key in sorted({f"{it['participant_id']}::{it['condition']}" for it in items}):
        rows = [it for it in items if f"{it['participant_id']}::{it['condition']}" == key]
        group_summary[key] = {
            "n": float(len(rows)),
            "iou": _mean([float(r["metrics"]["iou"]) for r in rows]),
            "f1": _mean([float(r["metrics"]["f1"]) for r in rows]),
            "material_match": _mean([float(r["metrics"]["material_match"]) for r in rows]),
            "coarse_material_match": _mean([float(r["metrics"]["coarse_material_match"]) for r in rows]),
            "component_f1": _mean([float(r["metrics"]["component_f1"]) for r in rows]),
        }

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gt_root": str(gt_root),
        "submission_root": str(sub_root),
        "summary": {
            "scored_submissions": len(items),
            "group_summary": group_summary,
            "note": "Protocol scoring only. No claims about human outcomes beyond submitted files.",
        },
        "items": items,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate_human_rebuild_submissions] wrote {out_path}")


if __name__ == "__main__":
    main()
