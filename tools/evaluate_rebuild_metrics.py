#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Coord3D = Tuple[int, int, int]
Coord2D = Tuple[int, int]

DEFAULT_THRESHOLDS = {
    "level0_min_intersection": 32,
    "level1_min_iou": 0.35,
    "level1_min_f1": 0.50,
    "level2_min_coarse_material_match": 0.55,
    "level3_min_material_match": 0.40,
    "level4_min_component_f1": 0.45,
    "component_iou_match_threshold": 0.30,
}

DIRECT_TYPE_NORM = {
    # Stone / brick aliases.
    "stone_bricks": "stonebrick",
    "stone_brick": "stonebrick",
    "stonebrick": "stonebrick",
    "minecraft_stone_bricks": "stonebrick",
    "minecraft_stone_brick": "stonebrick",
    "brick_block": "brick",
    "bricks": "brick",
    # Fence aliases.
    "oak_fence": "fence",
    "spruce_fence": "fence",
    "birch_fence": "fence",
    "jungle_fence": "fence",
    "acacia_fence": "fence",
    "dark_oak_fence": "fence",
    "wooden_fence": "fence",
    "nether_brick_fence": "fence",
    # Plank aliases.
    "planks": "wood",
    "oak_planks": "wood",
    "spruce_planks": "wood",
    "birch_planks": "wood",
    "jungle_planks": "wood",
    "acacia_planks": "wood",
    "dark_oak_planks": "wood",
    "wooden_planks": "wood",
    # Slab aliases.
    "stone_slab": "slab_stone",
    "stone_slab2": "slab_stone",
    "double_stone_slab": "slab_stone",
    "double_stone_slab2": "slab_stone",
    "wooden_slab": "slab_wood",
    "double_wooden_slab": "slab_wood",
    # Misc.
    "glass_pane": "glass",
    "stained_glass": "glass",
    "stained_glass_pane": "glass",
    "air": "air",
    "cave_air": "air",
    "void_air": "air",
}

STONE_HINTS = (
    "stone",
    "cobble",
    "andesite",
    "diorite",
    "granite",
    "quartz",
    "sandstone",
    "deepslate",
    "slab_stone",
)
WOOD_HINTS = (
    "wood",
    "plank",
    "log",
    "bark",
    "fence",
    "slab_wood",
    "stairs_wood",
    "door",
    "trapdoor",
)
BRICK_HINTS = (
    "brick",
    "terracotta",
)


@dataclass
class ShiftResult:
    dx: int
    dy: int
    dz: int
    intersection: int
    footprint_intersection: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GT vs predicted rebuilt voxels with robust material normalization "
            "and level-wise metrics (0-4)."
        )
    )
    parser.add_argument("--gt_root", required=True, help="GT dataset root (contains building_xxx/gt).")
    parser.add_argument(
        "--pred_root",
        required=True,
        help="Prediction root (contains building_xxx/rebuild_world by default).",
    )
    parser.add_argument(
        "--out",
        default="metrics_levels.json",
        help="Output JSON path. Default: metrics_levels.json",
    )
    parser.add_argument(
        "--pred_source",
        choices=["rebuild_world", "pred", "gt"],
        default="rebuild_world",
        help="Prediction source preference. Default: rebuild_world",
    )
    parser.add_argument(
        "--pred_subdir",
        default="",
        help="Optional explicit subdir under each building dir for pred voxels/bbox.",
    )
    parser.add_argument(
        "--pred_voxels_name",
        default="voxels.npy",
        help="Pred voxel filename. Default: voxels.npy",
    )
    parser.add_argument(
        "--pred_bbox_name",
        default="bbox.json",
        help="Pred bbox filename. Default: bbox.json",
    )
    parser.add_argument(
        "--allow_gt_fallback",
        action="store_true",
        help="If pred path is missing, allow fallback to building_xxx/gt/* under pred_root.",
    )
    parser.add_argument(
        "--fail_on_missing_pred",
        action="store_true",
        help="Fail if any building has no prediction voxels.",
    )
    parser.add_argument(
        "--max_shift_xy",
        type=int,
        default=48,
        help="Max |dx| and |dz| for shift search. Default: 48",
    )
    parser.add_argument(
        "--max_shift_y",
        type=int,
        default=8,
        help="Max |dy| for shift search. Default: 8",
    )
    parser.add_argument(
        "--top_shift_candidates",
        type=int,
        default=24,
        help="Top 2D shift candidates to refine in 3D. Default: 24",
    )
    parser.add_argument(
        "--thresholds_json",
        default="",
        help="Optional JSON file overriding level thresholds.",
    )
    parser.add_argument(
        "--building_pattern",
        default="building_*",
        help="Building glob pattern under gt_root. Default: building_*",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of buildings to evaluate (0 means all).",
    )
    return parser.parse_args()


def _clean_token(raw: object) -> str:
    token = str(raw).strip().lower()
    token = token.split("[", 1)[0]
    token = token.split("{", 1)[0]
    if ":" in token:
        token = token.split(":", 1)[1]
    token = token.replace(" ", "_").replace("-", "_").replace(".", "_")
    token = re.sub(r"__+", "_", token)
    token = token.strip("_")
    token = re.sub(r"(stone_slab)\d+$", r"\1", token)
    token = re.sub(r"(double_stone_slab)\d+$", r"\1", token)
    return token


def normalize_block_type(raw: object) -> str:
    token = _clean_token(raw)
    if not token:
        return "air"

    mapped = DIRECT_TYPE_NORM.get(token)
    if mapped is not None:
        return mapped

    if token.endswith("_fence") or "_fence_" in token or token == "fence":
        return "fence"

    if "plank" in token:
        return "wood"

    if "slab" in token:
        if any(h in token for h in ("wood", "oak", "spruce", "birch", "jungle", "acacia", "dark_oak")):
            return "slab_wood"
        if any(
            h in token
            for h in (
                "stone",
                "cobble",
                "brick",
                "sandstone",
                "quartz",
                "nether",
                "deepslate",
            )
        ):
            return "slab_stone"
        return "slab"

    if token in {"stone_brick", "stone_bricks", "minecraft_stone_bricks"}:
        return "stonebrick"

    if "stone_brick" in token:
        return "stonebrick"

    if token.endswith("_stairs"):
        base = token[: -len("_stairs")]
        if any(w in base for w in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "wood")):
            return "stairs_wood"
        return f"stairs_{base}" if base else "stairs"

    if "glass" in token:
        return "glass"

    if token in {"cave_air", "void_air"}:
        return "air"

    return token


def coarse_material(norm_type: str) -> str:
    if norm_type == "air":
        return "AIR"
    if any(h in norm_type for h in WOOD_HINTS):
        return "WOOD"
    if norm_type == "stonebrick":
        return "BRICK"
    if any(h in norm_type for h in BRICK_HINTS):
        return "BRICK"
    if any(h in norm_type for h in STONE_HINTS):
        return "STONE"
    if "glass" in norm_type:
        return "GLASS"
    if "wool" in norm_type or "carpet" in norm_type:
        return "TEXTILE"
    if "leaf" in norm_type or "vine" in norm_type:
        return "PLANT"
    return "OTHER"


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def load_voxels(path: Path) -> np.ndarray:
    vox = np.load(path, allow_pickle=True)
    if vox.ndim != 3:
        raise ValueError(f"Expected 3D voxel array at {path}, got shape={vox.shape}")
    return vox


def voxel_maps(voxels: np.ndarray, bbox: Dict[str, int]) -> Dict[Coord3D, str]:
    ymin = int(bbox["ymin"])
    xmin = int(bbox["xmin"])
    zmin = int(bbox["zmin"])

    result: Dict[Coord3D, str] = {}
    sy, sx, sz = voxels.shape
    for yi in range(sy):
        for xi in range(sx):
            for zi in range(sz):
                norm = normalize_block_type(voxels[yi, xi, zi])
                if norm == "air":
                    continue
                result[(xmin + xi, ymin + yi, zmin + zi)] = norm
    return result


def _intersection_2d(foot_gt: Set[Coord2D], foot_pred: Set[Coord2D], dx: int, dz: int) -> int:
    if len(foot_pred) <= len(foot_gt):
        return sum(1 for (x, z) in foot_pred if (x + dx, z + dz) in foot_gt)
    return sum(1 for (x, z) in foot_gt if (x - dx, z - dz) in foot_pred)


def _intersection_3d(gt_occ: Set[Coord3D], pred_points: Sequence[Coord3D], dx: int, dy: int, dz: int) -> int:
    count = 0
    for x, y, z in pred_points:
        if (x + dx, y + dy, z + dz) in gt_occ:
            count += 1
    return count


def search_best_shift(
    gt_occ: Set[Coord3D],
    pred_occ: Set[Coord3D],
    max_shift_xy: int,
    max_shift_y: int,
    top_candidates: int,
) -> ShiftResult:
    if not gt_occ or not pred_occ:
        return ShiftResult(dx=0, dy=0, dz=0, intersection=0, footprint_intersection=0)

    foot_gt: Set[Coord2D] = {(x, z) for (x, _y, z) in gt_occ}
    foot_pred: Set[Coord2D] = {(x, z) for (x, _y, z) in pred_occ}

    candidates: List[Tuple[int, int, int]] = []
    for dx in range(-max_shift_xy, max_shift_xy + 1):
        for dz in range(-max_shift_xy, max_shift_xy + 1):
            inter2d = _intersection_2d(foot_gt, foot_pred, dx=dx, dz=dz)
            candidates.append((inter2d, dx, dz))

    candidates.sort(key=lambda t: (t[0], -(abs(t[1]) + abs(t[2]))), reverse=True)
    candidates = candidates[: max(1, top_candidates)]

    pred_points = list(pred_occ)
    best = ShiftResult(dx=0, dy=0, dz=0, intersection=-1, footprint_intersection=-1)

    for inter2d, dx, dz in candidates:
        for dy in range(-max_shift_y, max_shift_y + 1):
            inter3d = _intersection_3d(gt_occ, pred_points, dx=dx, dy=dy, dz=dz)
            if inter3d > best.intersection:
                best = ShiftResult(dx=dx, dy=dy, dz=dz, intersection=inter3d, footprint_intersection=inter2d)
                continue
            if inter3d == best.intersection:
                if inter2d > best.footprint_intersection:
                    best = ShiftResult(dx=dx, dy=dy, dz=dz, intersection=inter3d, footprint_intersection=inter2d)
                    continue
                if inter2d == best.footprint_intersection:
                    current_norm = abs(dx) + abs(dy) + abs(dz)
                    best_norm = abs(best.dx) + abs(best.dy) + abs(best.dz)
                    if current_norm < best_norm:
                        best = ShiftResult(dx=dx, dy=dy, dz=dz, intersection=inter3d, footprint_intersection=inter2d)

    if best.intersection < 0:
        return ShiftResult(dx=0, dy=0, dz=0, intersection=0, footprint_intersection=0)
    return best


def shift_map(src: Dict[Coord3D, str], dx: int, dy: int, dz: int) -> Dict[Coord3D, str]:
    shifted: Dict[Coord3D, str] = {}
    for (x, y, z), value in src.items():
        shifted[(x + dx, y + dy, z + dz)] = value
    return shifted


def safe_div(n: float, d: float) -> float:
    if d == 0:
        return 0.0
    return float(n) / float(d)


def occupancy_metrics(gt_occ: Set[Coord3D], pred_occ: Set[Coord3D]) -> Dict[str, float]:
    inter = len(gt_occ & pred_occ)
    union = len(gt_occ | pred_occ)
    prec = safe_div(inter, len(pred_occ))
    rec = safe_div(inter, len(gt_occ))
    f1 = safe_div(2.0 * prec * rec, prec + rec)
    iou = safe_div(inter, union)
    return {
        "intersection": float(inter),
        "union": float(union),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "iou": iou,
    }


def material_metrics(gt_map: Dict[Coord3D, str], pred_map: Dict[Coord3D, str]) -> Dict[str, float]:
    inter_coords = gt_map.keys() & pred_map.keys()
    inter_count = len(inter_coords)
    if inter_count == 0:
        return {
            "intersection_count": 0.0,
            "material_match": 0.0,
            "coarse_material_match": 0.0,
        }

    strict_match = 0
    coarse_match = 0
    for coord in inter_coords:
        gt_t = gt_map[coord]
        pr_t = pred_map[coord]
        if gt_t == pr_t:
            strict_match += 1
        if coarse_material(gt_t) == coarse_material(pr_t):
            coarse_match += 1

    return {
        "intersection_count": float(inter_count),
        "material_match": safe_div(strict_match, inter_count),
        "coarse_material_match": safe_div(coarse_match, inter_count),
    }


def connected_components_2d(footprint: Set[Coord2D]) -> List[Set[Coord2D]]:
    remaining = set(footprint)
    comps: List[Set[Coord2D]] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            x, z = stack.pop()
            for dx, dz in neighbors:
                nxt = (x + dx, z + dz)
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def component_match_metrics(
    gt_occ: Set[Coord3D],
    pred_occ: Set[Coord3D],
    iou_threshold: float,
) -> Dict[str, float]:
    gt_fp = {(x, z) for (x, _y, z) in gt_occ}
    pred_fp = {(x, z) for (x, _y, z) in pred_occ}
    gt_comps = connected_components_2d(gt_fp)
    pred_comps = connected_components_2d(pred_fp)

    if not gt_comps and not pred_comps:
        return {
            "gt_components": 0.0,
            "pred_components": 0.0,
            "matched_components": 0.0,
            "component_precision": 1.0,
            "component_recall": 1.0,
            "component_f1": 1.0,
            "component_mean_iou": 1.0,
        }

    if not gt_comps or not pred_comps:
        return {
            "gt_components": float(len(gt_comps)),
            "pred_components": float(len(pred_comps)),
            "matched_components": 0.0,
            "component_precision": 0.0,
            "component_recall": 0.0,
            "component_f1": 0.0,
            "component_mean_iou": 0.0,
        }

    edges: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_comps):
        for pi, p in enumerate(pred_comps):
            inter = len(g & p)
            if inter == 0:
                continue
            union = len(g | p)
            iou = safe_div(inter, union)
            if iou >= iou_threshold:
                edges.append((iou, gi, pi))

    edges.sort(key=lambda t: t[0], reverse=True)

    used_g: Set[int] = set()
    used_p: Set[int] = set()
    matched_ious: List[float] = []
    for iou, gi, pi in edges:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched_ious.append(iou)

    matched = len(matched_ious)
    p = safe_div(matched, len(pred_comps))
    r = safe_div(matched, len(gt_comps))
    f1 = safe_div(2.0 * p * r, p + r)

    return {
        "gt_components": float(len(gt_comps)),
        "pred_components": float(len(pred_comps)),
        "matched_components": float(matched),
        "component_precision": p,
        "component_recall": r,
        "component_f1": f1,
        "component_mean_iou": safe_div(sum(matched_ious), len(matched_ious)),
    }


def level_passes(metrics: Dict[str, float], thresholds: Dict[str, float]) -> Dict[str, bool]:
    level0 = metrics["intersection"] >= thresholds["level0_min_intersection"]
    level1 = (
        metrics["iou"] >= thresholds["level1_min_iou"]
        and metrics["f1"] >= thresholds["level1_min_f1"]
    )
    level2 = metrics["coarse_material_match"] >= thresholds["level2_min_coarse_material_match"]
    level3 = metrics["material_match"] >= thresholds["level3_min_material_match"]
    level4 = metrics["component_f1"] >= thresholds["level4_min_component_f1"]

    return {
        "level0_shift": bool(level0),
        "level1_shape": bool(level1),
        "level2_coarse_material": bool(level2),
        "level3_strict_material": bool(level3),
        "level4_structure_components": bool(level4),
        "all_levels": bool(level0 and level1 and level2 and level3 and level4),
    }


def _building_dirs(root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def resolve_pred_paths(
    pred_root: Path,
    building_name: str,
    pred_source: str,
    pred_subdir: str,
    pred_voxels_name: str,
    pred_bbox_name: str,
    allow_gt_fallback: bool,
) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    bdir = pred_root / building_name
    candidates: List[Tuple[Path, str]] = []

    if pred_subdir:
        candidates.append((bdir / pred_subdir, f"{pred_subdir}"))
    elif pred_source == "rebuild_world":
        candidates.append((bdir / "rebuild_world", "rebuild_world"))
        candidates.append((bdir / "pred", "pred"))
    elif pred_source == "pred":
        candidates.append((bdir / "pred", "pred"))
        candidates.append((bdir / "rebuild_world", "rebuild_world"))
    else:
        candidates.append((bdir / "gt", "gt"))

    if allow_gt_fallback and (bdir / "gt") not in [p for p, _ in candidates]:
        candidates.append((bdir / "gt", "gt_fallback"))

    candidates.append((bdir, "building_root"))

    for base, tag in candidates:
        vox = base / pred_voxels_name
        bb = base / pred_bbox_name
        if vox.is_file() and bb.is_file():
            return vox, bb, tag

    return None, None, None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def main() -> None:
    args = parse_args()

    gt_root = Path(args.gt_root).resolve()
    pred_root = Path(args.pred_root).resolve()
    out_path = Path(args.out).resolve()

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds_json:
        overrides = load_json(Path(args.thresholds_json).resolve())
        for key, value in overrides.items():
            if key in thresholds:
                thresholds[key] = float(value)

    gt_dirs = _building_dirs(gt_root, pattern=args.building_pattern, limit=args.limit)
    if not gt_dirs:
        raise SystemExit(f"No building directories found under gt_root: {gt_root}")

    items: List[Dict[str, object]] = []
    missing_pred: List[str] = []

    for gdir in gt_dirs:
        building_name = gdir.name
        gt_vox_path = gdir / "gt" / "voxels.npy"
        gt_bbox_path = gdir / "gt" / "bbox.json"
        if not gt_vox_path.is_file() or not gt_bbox_path.is_file():
            continue

        pred_vox_path, pred_bbox_path, pred_source_used = resolve_pred_paths(
            pred_root=pred_root,
            building_name=building_name,
            pred_source=args.pred_source,
            pred_subdir=args.pred_subdir,
            pred_voxels_name=args.pred_voxels_name,
            pred_bbox_name=args.pred_bbox_name,
            allow_gt_fallback=args.allow_gt_fallback,
        )
        if pred_vox_path is None or pred_bbox_path is None:
            missing_pred.append(building_name)
            continue

        gt_bbox = load_json(gt_bbox_path)
        pred_bbox = load_json(pred_bbox_path)
        gt_vox = load_voxels(gt_vox_path)
        pred_vox = load_voxels(pred_vox_path)

        gt_map = voxel_maps(gt_vox, gt_bbox)
        pred_map = voxel_maps(pred_vox, pred_bbox)
        gt_occ = set(gt_map.keys())
        pred_occ = set(pred_map.keys())

        shift = search_best_shift(
            gt_occ=gt_occ,
            pred_occ=pred_occ,
            max_shift_xy=max(0, int(args.max_shift_xy)),
            max_shift_y=max(0, int(args.max_shift_y)),
            top_candidates=max(1, int(args.top_shift_candidates)),
        )

        pred_shifted_map = shift_map(pred_map, dx=shift.dx, dy=shift.dy, dz=shift.dz)
        pred_shifted_occ = set(pred_shifted_map.keys())

        occ_m = occupancy_metrics(gt_occ=gt_occ, pred_occ=pred_shifted_occ)
        mat_m = material_metrics(gt_map=gt_map, pred_map=pred_shifted_map)
        comp_m = component_match_metrics(
            gt_occ=gt_occ,
            pred_occ=pred_shifted_occ,
            iou_threshold=float(thresholds["component_iou_match_threshold"]),
        )

        merged_metrics = {
            **occ_m,
            **mat_m,
            **comp_m,
        }
        pass_flags = level_passes(merged_metrics, thresholds=thresholds)

        item = {
            "building": building_name,
            "gt_voxels": str(gt_vox_path),
            "pred_voxels": str(pred_vox_path),
            "pred_source_used": pred_source_used,
            "counts": {
                "gt_non_air": len(gt_occ),
                "pred_non_air": len(pred_occ),
                "pred_non_air_after_shift": len(pred_shifted_occ),
            },
            "shift": {
                "dx": shift.dx,
                "dy": shift.dy,
                "dz": shift.dz,
                "intersection_at_shift": shift.intersection,
                "footprint_intersection": shift.footprint_intersection,
            },
            "metrics": merged_metrics,
            "levels": {
                "level0_shift": {
                    "metric": "intersection",
                    "value": merged_metrics["intersection"],
                    "threshold": thresholds["level0_min_intersection"],
                    "pass": pass_flags["level0_shift"],
                },
                "level1_shape": {
                    "metrics": {
                        "iou": merged_metrics["iou"],
                        "f1": merged_metrics["f1"],
                    },
                    "thresholds": {
                        "iou": thresholds["level1_min_iou"],
                        "f1": thresholds["level1_min_f1"],
                    },
                    "pass": pass_flags["level1_shape"],
                },
                "level2_coarse_material": {
                    "metric": "coarse_material_match",
                    "value": merged_metrics["coarse_material_match"],
                    "threshold": thresholds["level2_min_coarse_material_match"],
                    "pass": pass_flags["level2_coarse_material"],
                },
                "level3_strict_material": {
                    "metric": "material_match",
                    "value": merged_metrics["material_match"],
                    "threshold": thresholds["level3_min_material_match"],
                    "pass": pass_flags["level3_strict_material"],
                },
                "level4_structure_components": {
                    "metric": "component_f1",
                    "value": merged_metrics["component_f1"],
                    "threshold": thresholds["level4_min_component_f1"],
                    "component_iou_match_threshold": thresholds["component_iou_match_threshold"],
                    "pass": pass_flags["level4_structure_components"],
                },
                "all_levels_pass": pass_flags["all_levels"],
            },
        }
        items.append(item)

    if args.fail_on_missing_pred and missing_pred:
        raise SystemExit(
            "Missing prediction voxels for: " + ", ".join(missing_pred)
        )

    metrics_keys = [
        "intersection",
        "union",
        "precision",
        "recall",
        "f1",
        "iou",
        "material_match",
        "coarse_material_match",
        "component_f1",
        "component_precision",
        "component_recall",
        "component_mean_iou",
    ]

    aggregate_metrics: Dict[str, float] = {}
    for key in metrics_keys:
        aggregate_metrics[key] = mean(item["metrics"][key] for item in items) if items else 0.0

    aggregate_pass_rates = {
        "level0_shift_pass_rate": mean(1.0 if item["levels"]["level0_shift"]["pass"] else 0.0 for item in items),
        "level1_shape_pass_rate": mean(1.0 if item["levels"]["level1_shape"]["pass"] else 0.0 for item in items),
        "level2_coarse_material_pass_rate": mean(
            1.0 if item["levels"]["level2_coarse_material"]["pass"] else 0.0 for item in items
        ),
        "level3_strict_material_pass_rate": mean(
            1.0 if item["levels"]["level3_strict_material"]["pass"] else 0.0 for item in items
        ),
        "level4_structure_components_pass_rate": mean(
            1.0 if item["levels"]["level4_structure_components"]["pass"] else 0.0 for item in items
        ),
        "all_levels_pass_rate": mean(1.0 if item["levels"]["all_levels_pass"] else 0.0 for item in items),
    }

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gt_root": str(gt_root),
        "pred_root": str(pred_root),
        "pred_source": args.pred_source,
        "thresholds": thresholds,
        "settings": {
            "max_shift_xy": int(args.max_shift_xy),
            "max_shift_y": int(args.max_shift_y),
            "top_shift_candidates": int(args.top_shift_candidates),
            "pred_subdir": args.pred_subdir,
            "allow_gt_fallback": bool(args.allow_gt_fallback),
        },
        "summary": {
            "evaluated_buildings": len(items),
            "missing_predictions": missing_pred,
        },
        "aggregate": {
            "metrics": aggregate_metrics,
            "pass_rates": aggregate_pass_rates,
        },
        "items": items,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[evaluate_rebuild_metrics] wrote: {out_path}")
    print(f"[evaluate_rebuild_metrics] evaluated_buildings={len(items)} missing_predictions={len(missing_pred)}")
    if items:
        print(
            "[evaluate_rebuild_metrics] aggregate: "
            f"IoU={aggregate_metrics['iou']:.4f}, "
            f"F1={aggregate_metrics['f1']:.4f}, "
            f"coarse_material={aggregate_metrics['coarse_material_match']:.4f}, "
            f"material_match={aggregate_metrics['material_match']:.4f}, "
            f"component_f1={aggregate_metrics['component_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
