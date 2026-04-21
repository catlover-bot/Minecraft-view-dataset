#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from tools.evaluate_rebuild_metrics import (
    load_json,
    load_voxels,
    search_best_shift,
    shift_map,
    voxel_maps,
)

Coord3D = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate repair effort from prediction to GT.")
    p.add_argument("--gt_root", required=True)
    p.add_argument("--pred_root", required=True)
    p.add_argument("--pred_subdir", required=True)
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_shift_xy", type=int, default=6)
    p.add_argument("--max_shift_y", type=int, default=4)
    p.add_argument("--top_shift_candidates", type=int, default=20)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _list_buildings(root: Path, pattern: str, limit: int) -> List[Path]:
    xs = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if limit > 0:
        xs = xs[:limit]
    return xs


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


def _per_material_breakdown(
    gt_map: Dict[Coord3D, str],
    pred_map: Dict[Coord3D, str],
    add_coords: Set[Coord3D],
    del_coords: Set[Coord3D],
    rep_coords: Set[Coord3D],
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}

    def bump(mat: str, key: str, c: int = 1) -> None:
        rec = out.setdefault(mat, {"additions": 0, "deletions": 0, "replacements": 0})
        rec[key] += int(c)

    for c in add_coords:
        bump(gt_map.get(c, "unknown"), "additions")
    for c in del_coords:
        bump(pred_map.get(c, "unknown"), "deletions")
    for c in rep_coords:
        bump(gt_map.get(c, "unknown"), "replacements")
    return out


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root).resolve()
    pred_root = Path(args.pred_root).resolve()
    if not gt_root.is_dir():
        raise SystemExit(f"gt_root not found: {gt_root}")
    if not pred_root.is_dir():
        raise SystemExit(f"pred_root not found: {pred_root}")

    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for gdir in _list_buildings(gt_root, args.building_pattern, args.limit):
        bname = gdir.name
        gt_bbox = gdir / "gt" / "bbox.json"
        gt_vox = gdir / "gt" / "voxels.npy"
        pred_bbox = pred_root / bname / args.pred_subdir / "bbox.json"
        pred_vox = pred_root / bname / args.pred_subdir / "voxels.npy"
        if not (gt_bbox.is_file() and gt_vox.is_file() and pred_bbox.is_file() and pred_vox.is_file()):
            missing.append(bname)
            continue

        gt_map = voxel_maps(load_voxels(gt_vox), load_json(gt_bbox))
        pred_map = voxel_maps(load_voxels(pred_vox), load_json(pred_bbox))

        gt_occ = set(gt_map.keys())
        pred_occ = set(pred_map.keys())
        shift = search_best_shift(
            gt_occ,
            pred_occ,
            max_shift_xy=int(args.max_shift_xy),
            max_shift_y=int(args.max_shift_y),
            top_candidates=int(args.top_shift_candidates),
        )
        pred_shifted = shift_map(pred_map, shift.dx, shift.dy, shift.dz)
        pred_occ_s = set(pred_shifted.keys())

        gt_only = gt_occ - pred_occ_s
        pred_only = pred_occ_s - gt_occ
        overlap = gt_occ & pred_occ_s
        replacements = {c for c in overlap if gt_map[c] != pred_shifted[c]}

        additions = len(gt_only)
        deletions = len(pred_only)
        repl = len(replacements)
        total = additions + deletions + repl
        gt_non_air = len(gt_occ)
        correct = len(overlap) - repl

        k_steps = [50, 100, 200, 500]
        completion_after_k = {}
        for k in k_steps:
            completion_after_k[str(k)] = min(1.0, _safe_div(correct + min(k, total), gt_non_air))

        add_comp = _component_count(gt_only) if gt_only else 0
        del_comp = _component_count(pred_only) if pred_only else 0
        rep_comp = _component_count(replacements) if replacements else 0

        item = {
            "building": bname,
            "pred_subdir": args.pred_subdir,
            "shift": {"dx": shift.dx, "dy": shift.dy, "dz": shift.dz},
            "counts": {
                "gt_non_air": gt_non_air,
                "pred_non_air": len(pred_occ_s),
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
                "additions_components": add_comp,
                "deletions_components": del_comp,
                "replacements_components": rep_comp,
                "total_components": add_comp + del_comp + rep_comp,
            },
            "per_material": _per_material_breakdown(gt_map, pred_shifted, gt_only, pred_only, replacements),
            "notes": {
                "approximation": [
                    "completion_after_k_edits is optimistic one-edit-one-voxel approximation",
                    "approx_cuboid_repair_count uses 6-neighbor connected components as cuboid-op proxy",
                ]
            },
        }
        items.append(item)

    def mean(key: str) -> float:
        if not items:
            return 0.0
        vals = []
        for it in items:
            if key in it["normalized"]:
                vals.append(float(it["normalized"][key]))
        return sum(vals) / len(vals) if vals else 0.0

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gt_root": str(gt_root),
        "pred_root": str(pred_root),
        "pred_subdir": args.pred_subdir,
        "summary": {
            "evaluated_buildings": len(items),
            "missing_predictions": missing,
            "mean_edit_distance_over_gt": mean("edit_distance_over_gt"),
            "mean_additions_over_gt": mean("additions_over_gt"),
            "mean_deletions_over_gt": mean("deletions_over_gt"),
            "mean_replacements_over_gt": mean("replacements_over_gt"),
        },
        "items": items,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate_repair_effort] wrote {out_path}")


if __name__ == "__main__":
    main()
