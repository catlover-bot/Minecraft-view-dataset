#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.generate_rebuild_plans import (
    DEFAULT_ROLE_BLOCKS,
    REQUIRED_PALETTE_ROLES,
    _coerce_plan,
    _expand_roof_template,
    _expand_window_pattern,
    _infer_shape_preferences,
    _int,
    _normalize_block_type,
    _normalize_role,
    _validate_and_repair_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-GT self-refinement pass after render simulation.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing building_xxx")
    parser.add_argument("--plan_subdir", required=True, help="Input rebuild plan subdir")
    parser.add_argument("--description_subdir", default="description", help="Description subdir")
    parser.add_argument("--out_plan_subdir", required=True, help="Output plan subdir")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern")
    parser.add_argument("--limit", type=int, default=0, help="Max building count (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--max_dim", type=int, default=192, help="Safety bound for voxel axis size")
    parser.add_argument("--max_iterations", type=int, default=2, help="Refinement iterations")
    parser.add_argument("--min_score_gain", type=float, default=0.01, help="Minimum score gain to accept")
    parser.add_argument("--max_added_ops_per_iter", type=int, default=64, help="Max correction ops per iteration")
    parser.add_argument(
        "--roof_search_variants",
        type=int,
        default=6,
        help="Number of roof-template variants to explore per iteration.",
    )
    parser.add_argument(
        "--window_search_variants",
        type=int,
        default=6,
        help="Number of window-pattern variants to explore per iteration.",
    )
    parser.add_argument(
        "--max_search_candidates",
        type=int,
        default=16,
        help="Maximum correction candidates evaluated per iteration.",
    )
    parser.add_argument(
        "--material_budget_reprojection_strength",
        type=float,
        default=0.5,
        help="Strength of render-after material budget reprojection.",
    )
    parser.add_argument(
        "--material_budget_reprojection_min_deficit_ratio",
        type=float,
        default=0.03,
        help="Minimum deficit ratio to trigger budget reprojection for a role.",
    )
    parser.add_argument(
        "--material_budget_reprojection_trigger_material_score",
        type=float,
        default=0.78,
        help="Apply budget reprojection only when material component is below this score.",
    )
    parser.add_argument(
        "--selection_op_penalty",
        type=float,
        default=0.001,
        help="Candidate selection penalty per added operation to avoid over-correction.",
    )
    parser.add_argument(
        "--enable_material_budget_reprojection",
        dest="enable_material_budget_reprojection",
        action="store_true",
        help="Enable material budget reprojection using render feedback.",
    )
    parser.add_argument(
        "--no_enable_material_budget_reprojection",
        dest="enable_material_budget_reprojection",
        action="store_false",
        help="Disable material budget reprojection.",
    )

    parser.add_argument("--strict_schema", dest="strict_schema", action="store_true")
    parser.add_argument("--no_strict_schema", dest="strict_schema", action="store_false")
    parser.add_argument("--enforce_role_fixed", dest="enforce_role_fixed", action="store_true")
    parser.add_argument("--no_enforce_role_fixed", dest="enforce_role_fixed", action="store_false")
    parser.add_argument("--require_material_budget", dest="require_material_budget", action="store_true")
    parser.add_argument("--no_require_material_budget", dest="require_material_budget", action="store_false")
    parser.add_argument("--prefer_description_palette", dest="prefer_description_palette", action="store_true")
    parser.add_argument("--no_prefer_description_palette", dest="prefer_description_palette", action="store_false")
    parser.add_argument("--material_budget_tolerance", type=float, default=0.35)
    parser.add_argument("--role_fix_min_confidence", type=float, default=0.78)
    parser.add_argument("--max_operations", type=int, default=260)
    parser.add_argument(
        "--required_palette_roles",
        nargs="*",
        default=list(REQUIRED_PALETTE_ROLES),
        help="Required palette roles subset",
    )
    parser.set_defaults(
        strict_schema=True,
        enforce_role_fixed=True,
        require_material_budget=True,
        prefer_description_palette=True,
        enable_material_budget_reprojection=True,
    )
    return parser.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in dataset_root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _normalize_required_roles(raw_roles: List[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for role in raw_roles:
        nr = _normalize_role(role)
        if nr in REQUIRED_PALETTE_ROLES and nr not in out:
            out.append(nr)
    return tuple(out) if out else REQUIRED_PALETTE_ROLES


def _resolve_bbox(plan: Dict[str, Any]) -> Dict[str, int]:
    bbox_raw = plan.get("bbox", {}) if isinstance(plan.get("bbox"), dict) else {}
    bbox = {
        "xmin": _int(bbox_raw.get("xmin", 0), 0),
        "xmax": _int(bbox_raw.get("xmax", 15), 15),
        "ymin": _int(bbox_raw.get("ymin", 0), 0),
        "ymax": _int(bbox_raw.get("ymax", 12), 12),
        "zmin": _int(bbox_raw.get("zmin", 0), 0),
        "zmax": _int(bbox_raw.get("zmax", 15), 15),
    }
    if bbox["xmax"] < bbox["xmin"]:
        bbox["xmin"], bbox["xmax"] = bbox["xmax"], bbox["xmin"]
    if bbox["ymax"] < bbox["ymin"]:
        bbox["ymin"], bbox["ymax"] = bbox["ymax"], bbox["ymin"]
    if bbox["zmax"] < bbox["zmin"]:
        bbox["zmin"], bbox["zmax"] = bbox["zmax"], bbox["zmin"]

    for op in plan.get("operations", []) if isinstance(plan.get("operations"), list) else []:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op", "")).strip().lower()
        if kind in {"fill", "carve"}:
            x1 = _int(op.get("x1", 0), 0)
            y1 = _int(op.get("y1", 0), 0)
            z1 = _int(op.get("z1", 0), 0)
            x2 = _int(op.get("x2", x1), x1)
            y2 = _int(op.get("y2", y1), y1)
            z2 = _int(op.get("z2", z1), z1)
            bbox["xmin"] = min(bbox["xmin"], x1, x2)
            bbox["xmax"] = max(bbox["xmax"], x1, x2)
            bbox["ymin"] = min(bbox["ymin"], y1, y2)
            bbox["ymax"] = max(bbox["ymax"], y1, y2)
            bbox["zmin"] = min(bbox["zmin"], z1, z2)
            bbox["zmax"] = max(bbox["zmax"], z1, z2)
        elif kind == "set":
            x = _int(op.get("x", 0), 0)
            y = _int(op.get("y", 0), 0)
            z = _int(op.get("z", 0), 0)
            bbox["xmin"] = min(bbox["xmin"], x)
            bbox["xmax"] = max(bbox["xmax"], x)
            bbox["ymin"] = min(bbox["ymin"], y)
            bbox["ymax"] = max(bbox["ymax"], y)
            bbox["zmin"] = min(bbox["zmin"], z)
            bbox["zmax"] = max(bbox["zmax"], z)
    return bbox


def _apply_fill(vox: np.ndarray, bbox: Dict[str, int], x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, block: str) -> None:
    xmin, ymin, zmin = bbox["xmin"], bbox["ymin"], bbox["zmin"]
    ix1, ix2 = x1 - xmin, x2 - xmin
    iy1, iy2 = y1 - ymin, y2 - ymin
    iz1, iz2 = z1 - zmin, z2 - zmin
    vox[iy1 : iy2 + 1, ix1 : ix2 + 1, iz1 : iz2 + 1] = block


def _render_plan(plan: Dict[str, Any], *, max_dim: int) -> Tuple[np.ndarray, Dict[str, int]]:
    bbox = _resolve_bbox(plan)
    sx = bbox["xmax"] - bbox["xmin"] + 1
    sy = bbox["ymax"] - bbox["ymin"] + 1
    sz = bbox["zmax"] - bbox["zmin"] + 1
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise RuntimeError("Invalid bbox dimensions")
    if max(sx, sy, sz) > int(max_dim):
        raise RuntimeError(f"Bbox too large for self-refine render: {sx}x{sy}x{sz}")

    vox = np.full((sy, sx, sz), "air", dtype="<U40")
    ops = plan.get("operations", []) if isinstance(plan.get("operations"), list) else []
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op", "")).strip().lower()
        if kind == "fill":
            x1 = _int(op.get("x1", 0), 0)
            y1 = _int(op.get("y1", 0), 0)
            z1 = _int(op.get("z1", 0), 0)
            x2 = _int(op.get("x2", x1), x1)
            y2 = _int(op.get("y2", y1), y1)
            z2 = _int(op.get("z2", z1), z1)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if z2 < z1:
                z1, z2 = z2, z1
            block = _normalize_block_type(op.get("block", "stonebrick"))
            _apply_fill(vox, bbox, x1, y1, z1, x2, y2, z2, block)
        elif kind == "carve":
            x1 = _int(op.get("x1", 0), 0)
            y1 = _int(op.get("y1", 0), 0)
            z1 = _int(op.get("z1", 0), 0)
            x2 = _int(op.get("x2", x1), x1)
            y2 = _int(op.get("y2", y1), y1)
            z2 = _int(op.get("z2", z1), z1)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if z2 < z1:
                z1, z2 = z2, z1
            _apply_fill(vox, bbox, x1, y1, z1, x2, y2, z2, "air")
        elif kind == "set":
            x = _int(op.get("x", 0), 0)
            y = _int(op.get("y", 0), 0)
            z = _int(op.get("z", 0), 0)
            block = _normalize_block_type(op.get("block", "stonebrick"))
            _apply_fill(vox, bbox, x, y, z, x, y, z, block)
    return vox, bbox


def _role_block_map(palette: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for role in REQUIRED_PALETTE_ROLES:
        out[role] = _normalize_block_type(palette.get(role, DEFAULT_ROLE_BLOCKS[role]))
    return out


def _counts_by_role(vox: np.ndarray, role_blocks: Dict[str, str]) -> Dict[str, int]:
    counts = {r: 0 for r in REQUIRED_PALETTE_ROLES}
    # Priority prevents double counting when same block is reused across roles.
    priority = ("glass", "light", "roof", "floor", "wall", "trim")
    total = int(np.count_nonzero(vox != "air"))
    for role in priority:
        block = role_blocks[role]
        counts[role] = int(np.count_nonzero(vox == block))

    assigned = sum(counts.values())
    if assigned < total:
        # Remaining non-air goes to wall bucket.
        counts["wall"] += total - assigned
    return counts


def _target_role_counts(plan: Dict[str, Any], desc: Dict[str, Any], total_non_air: int) -> Dict[str, int]:
    budget = plan.get("material_budget", {})
    out = {r: 0 for r in REQUIRED_PALETTE_ROLES}
    if isinstance(budget, dict):
        found = False
        for role in REQUIRED_PALETTE_ROLES:
            item = budget.get(role, {})
            if not isinstance(item, dict):
                continue
            target = max(0, _int(item.get("target_blocks", 0), 0))
            if target > 0:
                found = True
            out[role] = target
        if found:
            return out

    # Fallback priors from description context.
    text = " ".join(
        str(x)
        for x in (
            desc.get("summary", ""),
            " ".join(desc.get("elements", [])) if isinstance(desc.get("elements"), list) else "",
            " ".join(desc.get("rebuild_hints", [])) if isinstance(desc.get("rebuild_hints"), list) else "",
        )
    ).lower()
    priors = {
        "wall": 0.45,
        "roof": 0.20,
        "trim": 0.10,
        "glass": 0.08,
        "light": 0.04,
        "floor": 0.13,
    }
    if "window" in text:
        priors["glass"] += 0.04
        priors["wall"] -= 0.02
        priors["trim"] -= 0.02
    if "tower" in text:
        priors["wall"] += 0.06
        priors["roof"] -= 0.03
        priors["floor"] -= 0.03
    s = max(1e-6, sum(max(0.0, v) for v in priors.values()))
    for role in REQUIRED_PALETTE_ROLES:
        out[role] = int(round(total_non_air * max(0.0, priors[role]) / s))
    return out


def _compute_dim_score(vox: np.ndarray, desc: Dict[str, Any]) -> float:
    dims = desc.get("dimensions_estimate", {}) if isinstance(desc.get("dimensions_estimate"), dict) else {}
    exp_w = max(1, _int(dims.get("width", vox.shape[1]), vox.shape[1]))
    exp_d = max(1, _int(dims.get("depth", vox.shape[2]), vox.shape[2]))
    exp_h = max(1, _int(dims.get("height", vox.shape[0]), vox.shape[0]))
    pred_h, pred_w, pred_d = vox.shape

    def axis_score(pred: int, exp: int) -> float:
        err = abs(pred - exp) / float(max(1, exp))
        return max(0.0, 1.0 - err)

    return (axis_score(pred_w, exp_w) + axis_score(pred_d, exp_d) + axis_score(pred_h, exp_h)) / 3.0


def _compute_window_score(vox: np.ndarray, role_blocks: Dict[str, str], desc_text: str) -> float:
    if "window" not in desc_text and "glass" not in desc_text:
        return 1.0
    sy, sx, sz = vox.shape
    glass = role_blocks["glass"]
    y1 = max(1, int(round(sy * 0.25)))
    y2 = min(sy - 2, int(round(sy * 0.75)))
    if y2 < y1:
        y1, y2 = 1, max(1, sy - 2)
    sub = vox[y1 : y2 + 1, :, :]
    boundary = np.zeros_like(sub, dtype=bool)
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    actual = int(np.count_nonzero((sub == glass) & boundary))
    perimeter = max(8, 2 * (sx + sz))
    expected = max(6, perimeter // 8)
    return min(1.0, actual / float(max(1, expected)))


def _compute_entrance_score(vox: np.ndarray, desc_text: str) -> float:
    if ("entrance" not in desc_text) and ("door" not in desc_text) and ("entry" not in desc_text):
        return 1.0
    sy, sx, sz = vox.shape
    y_bot = 1
    y_top = min(sy - 1, 3)
    if y_top <= y_bot:
        return 0.0

    openings = 0
    # North/south faces
    for x in range(1, sx - 1):
        if np.all(vox[y_bot:y_top, x, 0] == "air"):
            openings += 1
        if np.all(vox[y_bot:y_top, x, sz - 1] == "air"):
            openings += 1
    # West/east faces
    for z in range(1, sz - 1):
        if np.all(vox[y_bot:y_top, 0, z] == "air"):
            openings += 1
        if np.all(vox[y_bot:y_top, sx - 1, z] == "air"):
            openings += 1
    return 1.0 if openings >= 2 else (0.5 if openings == 1 else 0.0)


def _compute_roof_score(vox: np.ndarray, role_blocks: Dict[str, str], roof_type: str) -> float:
    sy = vox.shape[0]
    top_layers = min(3, sy)
    roof_block = role_blocks["roof"]
    top = vox[sy - top_layers : sy, :, :]
    top_non_air = int(np.count_nonzero(top != "air"))
    if top_non_air <= 0:
        return 0.0
    roof_ratio = int(np.count_nonzero(top == roof_block)) / float(top_non_air)
    if roof_type not in {"gable", "hip", "dome"}:
        return roof_ratio

    # Taper preference: upper occupied area should not grow.
    areas: List[int] = []
    for y in range(max(0, sy - 6), sy):
        areas.append(int(np.count_nonzero(vox[y, :, :] != "air")))
    violations = 0
    for i in range(1, len(areas)):
        if areas[i] > areas[i - 1]:
            violations += 1
    taper = 1.0 if len(areas) <= 1 else max(0.0, 1.0 - violations / float(len(areas) - 1))
    return 0.7 * roof_ratio + 0.3 * taper


def _compute_material_score(
    counts: Dict[str, int],
    target: Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    total_actual = float(max(1, sum(counts.values())))
    total_target = float(max(1, sum(target.values())))
    l1 = 0.0
    deficits: Dict[str, float] = {}
    for role in REQUIRED_PALETTE_ROLES:
        ar = counts[role] / total_actual
        tr = target[role] / total_target
        l1 += abs(ar - tr)
        deficits[role] = tr - ar
    score = max(0.0, 1.0 - 0.5 * l1)
    return score, deficits


def _self_consistency_score(
    plan: Dict[str, Any],
    vox: np.ndarray,
    bbox: Dict[str, int],
    desc: Dict[str, Any],
) -> Dict[str, Any]:
    palette = plan.get("palette", {}) if isinstance(plan.get("palette"), dict) else {}
    role_blocks = _role_block_map(palette)
    counts = _counts_by_role(vox, role_blocks)
    non_air = int(np.count_nonzero(vox != "air"))
    target = _target_role_counts(plan, desc, non_air)
    material_score, deficits = _compute_material_score(counts, target)
    prefs = _infer_shape_preferences(desc if isinstance(desc, dict) else {})
    desc_text = str(prefs.get("text", ""))

    dim_score = _compute_dim_score(vox, desc)
    window_score = _compute_window_score(vox, role_blocks, desc_text)
    entrance_score = _compute_entrance_score(vox, desc_text)
    roof_score = _compute_roof_score(vox, role_blocks, str(prefs.get("roof_type", "flat")))

    weights = {
        "material": 0.35,
        "dim": 0.20,
        "roof": 0.20,
        "window": 0.15,
        "entrance": 0.10,
    }
    score = (
        weights["material"] * material_score
        + weights["dim"] * dim_score
        + weights["roof"] * roof_score
        + weights["window"] * window_score
        + weights["entrance"] * entrance_score
    )
    return {
        "score": float(score),
        "components": {
            "material": float(material_score),
            "dim": float(dim_score),
            "roof": float(roof_score),
            "window": float(window_score),
            "entrance": float(entrance_score),
        },
        "role_counts": counts,
        "target_role_counts": target,
        "role_deficits": deficits,
        "bbox": bbox,
        "role_blocks": role_blocks,
        "shape_preferences": prefs,
    }


def _add_op_if_room(ops: List[Dict[str, Any]], op: Dict[str, Any], max_operations: int) -> bool:
    if len(ops) >= max_operations:
        return False
    ops.append(op)
    return True


def _tag_ops(ops: List[Dict[str, Any]], *, role: str, reason: str, confidence: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        fixed = dict(op)
        kind = str(fixed.get("op", "")).strip().lower()
        if kind != "carve":
            fixed["role"] = role
            fixed["role_confidence"] = float(confidence)
            fixed["role_reason"] = reason
        out.append(fixed)
    return out


def _trim_perimeter_ops(bbox: Dict[str, int], block: str, band_y: int, purpose: str) -> List[Dict[str, Any]]:
    x1, x2 = bbox["xmin"], bbox["xmax"]
    z1, z2 = bbox["zmin"], bbox["zmax"]
    if x2 <= x1 or z2 <= z1:
        return []
    return [
        {"op": "fill", "x1": x1, "y1": band_y, "z1": z1, "x2": x2, "y2": band_y, "z2": z1, "block": block, "purpose": purpose},
        {"op": "fill", "x1": x1, "y1": band_y, "z1": z2, "x2": x2, "y2": band_y, "z2": z2, "block": block, "purpose": purpose},
        {"op": "fill", "x1": x1, "y1": band_y, "z1": z1, "x2": x1, "y2": band_y, "z2": z2, "block": block, "purpose": purpose},
        {"op": "fill", "x1": x2, "y1": band_y, "z1": z1, "x2": x2, "y2": band_y, "z2": z2, "block": block, "purpose": purpose},
    ]


def _propose_base_balance_ops(
    metrics: Dict[str, Any],
    *,
    max_operations: int,
) -> List[Dict[str, Any]]:
    bbox = metrics["bbox"]
    role_blocks = metrics["role_blocks"]
    comps = metrics["components"]
    deficits = metrics["role_deficits"]
    ops: List[Dict[str, Any]] = []

    if comps["entrance"] < 0.65:
        cx = (bbox["xmin"] + bbox["xmax"]) // 2
        _add_op_if_room(
            ops,
            {
                "op": "carve",
                "x1": cx,
                "y1": bbox["ymin"] + 1,
                "z1": bbox["zmin"],
                "x2": cx,
                "y2": min(bbox["ymax"] - 1, bbox["ymin"] + 3),
                "z2": bbox["zmin"],
                "block": "air",
                "purpose": "self_refine_entrance",
            },
            max_operations,
        )

    if deficits.get("wall", 0.0) > 0.06:
        _add_op_if_room(
            ops,
            {
                "op": "fill",
                "x1": bbox["xmin"],
                "y1": bbox["ymin"] + 1,
                "z1": bbox["zmin"],
                "x2": bbox["xmax"],
                "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                "z2": bbox["zmax"],
                "block": role_blocks["wall"],
                "role": "wall",
                "role_confidence": 0.85,
                "role_reason": "self_refine_wall_balance",
                "purpose": "self_refine_wall_balance",
            },
            max_operations,
        )

    if deficits.get("floor", 0.0) > 0.04:
        _add_op_if_room(
            ops,
            {
                "op": "fill",
                "x1": bbox["xmin"],
                "y1": bbox["ymin"],
                "z1": bbox["zmin"],
                "x2": bbox["xmax"],
                "y2": bbox["ymin"],
                "z2": bbox["zmax"],
                "block": role_blocks["floor"],
                "role": "floor",
                "role_confidence": 0.88,
                "role_reason": "self_refine_floor_balance",
                "purpose": "self_refine_floor_balance",
            },
            max_operations,
        )

    if deficits.get("trim", 0.0) > 0.03:
        band_y = max(bbox["ymin"] + 2, min(bbox["ymax"] - 1, bbox["ymax"] - 3))
        trim_ops = _trim_perimeter_ops(bbox, role_blocks["trim"], band_y, "self_refine_trim_perimeter")
        for op in _tag_ops(trim_ops, role="trim", reason="self_refine_trim_balance", confidence=0.84):
            if not _add_op_if_room(ops, op, max_operations):
                break

    if deficits.get("light", 0.0) > 0.02:
        for x, z in (
            (bbox["xmin"], bbox["zmin"]),
            (bbox["xmax"], bbox["zmin"]),
            (bbox["xmin"], bbox["zmax"]),
            (bbox["xmax"], bbox["zmax"]),
        ):
            light = {
                "op": "set",
                "x": x,
                "y": min(bbox["ymax"], bbox["ymin"] + 2),
                "z": z,
                "block": role_blocks["light"],
                "role": "light",
                "role_confidence": 0.9,
                "role_reason": "self_refine_light_balance",
                "purpose": "self_refine_corner_light",
            }
            if not _add_op_if_room(ops, light, max_operations):
                break
    return ops


def _roof_variant_specs(
    bbox: Dict[str, int],
    prefs: Dict[str, Any],
    *,
    max_variants: int,
) -> List[Dict[str, Any]]:
    pref_type = str(prefs.get("roof_type", "flat")).strip().lower() or "flat"
    roof_types = [pref_type, "gable", "hip", "flat", "dome"]
    roof_types = list(dict.fromkeys(roof_types))
    layers_opts = [4, 5, 3, 6]
    base_opts = [
        max(bbox["ymin"] + 3, bbox["ymax"] - 3),
        max(bbox["ymin"] + 3, bbox["ymax"] - 2),
        max(bbox["ymin"] + 2, bbox["ymax"] - 1),
    ]
    base_opts = list(dict.fromkeys(base_opts))

    out: List[Dict[str, Any]] = []
    for rt in roof_types:
        for layers in layers_opts:
            for by in base_opts:
                out.append({"roof_type": rt, "layers": int(layers), "base_y": int(by)})
                if len(out) >= max(1, int(max_variants)):
                    return out
    return out


def _window_variant_specs(
    bbox: Dict[str, int],
    deficits: Dict[str, float],
    *,
    max_variants: int,
) -> List[Dict[str, Any]]:
    glass_def = float(deficits.get("glass", 0.0) or 0.0)
    if glass_def > 0.08:
        spacing_opts = [2, 3, 4]
    elif glass_def > 0.04:
        spacing_opts = [3, 2, 4]
    else:
        spacing_opts = [4, 3, 2]
    height_opts = [2, 3, 1]
    y_opts = [bbox["ymin"] + 2, bbox["ymin"] + 3, max(bbox["ymin"] + 1, (bbox["ymin"] + bbox["ymax"]) // 2 - 1)]
    face_opts = ["all", "north", "south", "east", "west"]

    out: List[Dict[str, Any]] = []
    for spacing in spacing_opts:
        for win_h in height_opts:
            for yv in y_opts:
                for face in face_opts:
                    out.append({"face": face, "spacing": int(spacing), "window_height": int(win_h), "y": int(yv)})
                    if len(out) >= max(1, int(max_variants)):
                        return out
    return out


def _bundle_signature(ops: List[Dict[str, Any]]) -> str:
    try:
        return json.dumps(ops, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(len(ops))


def _propose_correction_candidates(
    plan: Dict[str, Any],
    desc: Dict[str, Any],
    metrics: Dict[str, Any],
    *,
    max_operations: int,
    max_added_ops_per_iter: int,
    roof_search_variants: int,
    window_search_variants: int,
    max_search_candidates: int,
) -> List[List[Dict[str, Any]]]:
    bbox = metrics["bbox"]
    role_blocks = metrics["role_blocks"]
    comps = metrics["components"]
    deficits = metrics["role_deficits"]
    prefs = metrics["shape_preferences"]

    need_roof = comps["roof"] < 0.62 or deficits.get("roof", 0.0) > 0.05
    need_window = comps["window"] < 0.60 or deficits.get("glass", 0.0) > 0.03

    roof_variants = max(1, int(roof_search_variants))
    window_variants = max(1, int(window_search_variants))
    max_candidates = max(1, int(max_search_candidates))

    roof_ops_list: List[List[Dict[str, Any]]] = [[]]
    if need_roof:
        roof_ops_list = []
        for idx, spec in enumerate(_roof_variant_specs(bbox, prefs, max_variants=roof_variants)):
            rops = _expand_roof_template(
                {**spec, "block": role_blocks["roof"]},
                bbox,
                {k: role_blocks[k] for k in role_blocks},
                "roof",
                f"self_refine_roof_v{idx}",
            )
            rops = _tag_ops(rops, role="roof", reason=f"self_refine_roof_v{idx}", confidence=0.9)
            roof_ops_list.append(rops)
        if not roof_ops_list:
            roof_ops_list = [[]]

    window_ops_list: List[List[Dict[str, Any]]] = [[]]
    if need_window:
        window_ops_list = []
        for idx, spec in enumerate(_window_variant_specs(bbox, deficits, max_variants=window_variants)):
            wops = _expand_window_pattern(
                {**spec, "block": role_blocks["glass"]},
                bbox,
                {k: role_blocks[k] for k in role_blocks},
                "glass",
                f"self_refine_window_v{idx}",
            )
            wops = _tag_ops(wops, role="glass", reason=f"self_refine_window_v{idx}", confidence=0.9)
            window_ops_list.append(wops)
        if not window_ops_list:
            window_ops_list = [[]]

    base_ops = _propose_base_balance_ops(metrics, max_operations=max_operations)
    bundles: List[List[Dict[str, Any]]] = []
    seen = set()
    for rops in roof_ops_list:
        for wops in window_ops_list:
            ops: List[Dict[str, Any]] = []
            ops.extend(rops)
            ops.extend(wops)
            ops.extend(base_ops)
            if not ops:
                continue
            if len(ops) > max_added_ops_per_iter:
                ops = ops[:max_added_ops_per_iter]
            sig = _bundle_signature(ops)
            if sig in seen:
                continue
            seen.add(sig)
            bundles.append(ops)
            if len(bundles) >= max_candidates:
                return bundles
    return bundles


def _sample_block_positions(
    vox: np.ndarray,
    bbox: Dict[str, int],
    block: str,
    *,
    max_points: int,
    prefer_top: bool = False,
) -> List[Tuple[int, int, int]]:
    idx = np.argwhere(vox == block)
    if idx.size == 0:
        return []
    if prefer_top:
        order = np.argsort(idx[:, 0])[::-1]
        idx = idx[order]
    n = int(max(1, min(int(max_points), idx.shape[0])))
    step = max(1, idx.shape[0] // n)
    picked = idx[::step][:n]
    out: List[Tuple[int, int, int]] = []
    for iy, ix, iz in picked:
        out.append((bbox["xmin"] + int(ix), bbox["ymin"] + int(iy), bbox["zmin"] + int(iz)))
    return out


def _propose_material_budget_reprojection_ops(
    plan: Dict[str, Any],
    metrics: Dict[str, Any],
    vox: np.ndarray,
    bbox: Dict[str, int],
    *,
    max_operations: int,
    max_added_ops_per_iter: int,
    strength: float,
    min_deficit_ratio: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    role_blocks = metrics["role_blocks"]
    deficits = metrics["role_deficits"]
    target = metrics["target_role_counts"]
    prefs = metrics["shape_preferences"]
    total_target = int(max(1, sum(int(v) for v in target.values())))
    ops: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"added_by_role": {}, "reduced_by_role": {}}

    def add_many(new_ops: List[Dict[str, Any]]) -> None:
        for op in new_ops:
            if len(ops) >= max_operations or len(ops) >= max_added_ops_per_iter:
                break
            ops.append(op)

    # Positive deficits: add role-constrained ops.
    for role in REQUIRED_PALETTE_ROLES:
        deficit = float(deficits.get(role, 0.0) or 0.0)
        if deficit <= float(min_deficit_ratio):
            continue
        demand_blocks = int(round(deficit * total_target * float(strength)))
        if demand_blocks <= 0:
            continue

        added_before = len(ops)
        if role == "roof":
            layers = max(2, min(6, 2 + demand_blocks // 150))
            rops = _expand_roof_template(
                {
                    "roof_type": str(prefs.get("roof_type", "flat")),
                    "base_y": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                    "layers": layers,
                    "block": role_blocks["roof"],
                },
                bbox,
                {k: role_blocks[k] for k in role_blocks},
                "roof",
                "budget_reproject_roof",
            )
            add_many(_tag_ops(rops, role="roof", reason="budget_reproject_roof", confidence=0.9))
        elif role == "glass":
            spacing = 2 if deficit > 0.08 else (3 if deficit > 0.05 else 4)
            wops = _expand_window_pattern(
                {
                    "face": "all",
                    "spacing": spacing,
                    "window_height": 2 if deficit < 0.1 else 3,
                    "y": bbox["ymin"] + 2,
                    "block": role_blocks["glass"],
                },
                bbox,
                {k: role_blocks[k] for k in role_blocks},
                "glass",
                "budget_reproject_glass",
            )
            add_many(_tag_ops(wops, role="glass", reason="budget_reproject_glass", confidence=0.9))
        elif role == "wall":
            wall_fill = [
                {
                    "op": "fill",
                    "x1": bbox["xmin"],
                    "y1": bbox["ymin"] + 1,
                    "z1": bbox["zmin"],
                    "x2": bbox["xmax"],
                    "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                    "z2": bbox["zmin"],
                    "block": role_blocks["wall"],
                    "purpose": "budget_reproject_wall_north",
                },
                {
                    "op": "fill",
                    "x1": bbox["xmin"],
                    "y1": bbox["ymin"] + 1,
                    "z1": bbox["zmax"],
                    "x2": bbox["xmax"],
                    "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                    "z2": bbox["zmax"],
                    "block": role_blocks["wall"],
                    "purpose": "budget_reproject_wall_south",
                },
            ]
            if deficit > 0.08:
                wall_fill.extend(
                    [
                        {
                            "op": "fill",
                            "x1": bbox["xmin"],
                            "y1": bbox["ymin"] + 1,
                            "z1": bbox["zmin"],
                            "x2": bbox["xmin"],
                            "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                            "z2": bbox["zmax"],
                            "block": role_blocks["wall"],
                            "purpose": "budget_reproject_wall_west",
                        },
                        {
                            "op": "fill",
                            "x1": bbox["xmax"],
                            "y1": bbox["ymin"] + 1,
                            "z1": bbox["zmin"],
                            "x2": bbox["xmax"],
                            "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                            "z2": bbox["zmax"],
                            "block": role_blocks["wall"],
                            "purpose": "budget_reproject_wall_east",
                        },
                    ]
                )
            add_many(_tag_ops(wall_fill, role="wall", reason="budget_reproject_wall", confidence=0.88))
        elif role == "floor":
            add_many(
                _tag_ops(
                    [
                        {
                            "op": "fill",
                            "x1": bbox["xmin"],
                            "y1": bbox["ymin"],
                            "z1": bbox["zmin"],
                            "x2": bbox["xmax"],
                            "y2": bbox["ymin"],
                            "z2": bbox["zmax"],
                            "block": role_blocks["floor"],
                            "purpose": "budget_reproject_floor",
                        }
                    ],
                    role="floor",
                    reason="budget_reproject_floor",
                    confidence=0.9,
                )
            )
        elif role == "trim":
            band_y = max(bbox["ymin"] + 2, min(bbox["ymax"] - 1, bbox["ymax"] - 3))
            tops = _trim_perimeter_ops(bbox, role_blocks["trim"], band_y, "budget_reproject_trim")
            add_many(_tag_ops(tops, role="trim", reason="budget_reproject_trim", confidence=0.86))
        elif role == "light":
            light_ops = []
            for x, z in (
                (bbox["xmin"], bbox["zmin"]),
                (bbox["xmax"], bbox["zmin"]),
                (bbox["xmin"], bbox["zmax"]),
                (bbox["xmax"], bbox["zmax"]),
            ):
                light_ops.append(
                    {
                        "op": "set",
                        "x": x,
                        "y": min(bbox["ymax"], bbox["ymin"] + 2),
                        "z": z,
                        "block": role_blocks["light"],
                        "purpose": "budget_reproject_light",
                    }
                )
            add_many(_tag_ops(light_ops, role="light", reason="budget_reproject_light", confidence=0.9))
        report["added_by_role"][role] = len(ops) - added_before

    # Negative deficits: reduce oversupplied non-core roles by converting to wall.
    wall_block = role_blocks.get("wall", "stonebrick")
    for role in ("glass", "trim", "light", "roof"):
        deficit = float(deficits.get(role, 0.0) or 0.0)
        over_ratio = -deficit
        if over_ratio <= float(min_deficit_ratio):
            continue
        over_blocks = int(round(over_ratio * total_target * float(strength) * 0.4))
        if over_blocks <= 0:
            continue
        sample_n = int(max(2, min(24, over_blocks)))
        coords = _sample_block_positions(
            vox,
            bbox,
            role_blocks.get(role, ""),
            max_points=sample_n,
            prefer_top=(role == "roof"),
        )
        reduced = 0
        for x, y, z in coords:
            set_op = {
                "op": "set",
                "x": int(x),
                "y": int(y),
                "z": int(z),
                "block": wall_block,
                "role": "wall",
                "role_confidence": 0.82,
                "role_reason": f"budget_reproject_reduce_{role}",
                "purpose": f"budget_reproject_reduce_{role}",
            }
            if not _add_op_if_room(ops, set_op, max_operations):
                break
            if len(ops) >= max_added_ops_per_iter:
                break
            reduced += 1
        report["reduced_by_role"][role] = reduced

    if len(ops) > max_added_ops_per_iter:
        ops = ops[:max_added_ops_per_iter]
    report["total_added_ops"] = len(ops)
    return ops, report


def _merge_with_stage_order(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def rank(op: Dict[str, Any]) -> int:
        kind = str(op.get("op", "")).strip().lower()
        if kind == "carve":
            return 2
        role = _normalize_role(op.get("role", ""))
        if role == "floor":
            return 0
        if role in {"wall", "roof"}:
            return 1
        if role in {"trim", "glass", "light"}:
            return 3
        return 2

    indexed = list(enumerate(ops))
    indexed.sort(key=lambda it: (rank(it[1]), it[0]))
    return [op for _idx, op in indexed]


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    required_roles = _normalize_required_roles(args.required_palette_roles)
    buildings = _list_buildings(dataset_root, args.building_pattern, args.limit)

    for bdir in buildings:
        src_plan_path = bdir / args.plan_subdir / "plan.json"
        desc_path = bdir / args.description_subdir / "description.json"
        if not src_plan_path.is_file():
            print(f"[self_refine_no_gt] skip {bdir.name} (missing plan)")
            continue
        if not desc_path.is_file():
            print(f"[self_refine_no_gt] skip {bdir.name} (missing description)")
            continue

        out_dir = bdir / args.out_plan_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_plan_path = out_dir / "plan.json"
        out_req_path = out_dir / "plan.request.json"
        if out_plan_path.is_file() and not args.overwrite:
            print(f"[self_refine_no_gt] skip {bdir.name} (exists)")
            continue

        raw_plan = json.loads(src_plan_path.read_text(encoding="utf-8"))
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        coerced, coerce_report = _coerce_plan(raw_plan)
        plan, validation = _validate_and_repair_plan(
            coerced,
            desc=desc if isinstance(desc, dict) else {},
            strict_schema=bool(args.strict_schema),
            enforce_role_fixed=bool(args.enforce_role_fixed),
            require_material_budget=bool(args.require_material_budget),
            material_budget_tolerance=float(args.material_budget_tolerance),
            role_fix_min_confidence=float(args.role_fix_min_confidence),
            prefer_description_palette=bool(args.prefer_description_palette),
            max_operations=int(args.max_operations),
            required_palette_roles=required_roles,
        )

        history: List[Dict[str, Any]] = []
        accepted_iterations = 0
        total_added_ops = 0

        try:
            vox0, bbox0 = _render_plan(plan, max_dim=int(args.max_dim))
            metrics = _self_consistency_score(plan, vox0, bbox0, desc)
        except Exception as exc:  # noqa: BLE001
            print(f"[self_refine_no_gt] {bdir.name} render failed: {exc}")
            plan["self_refine_no_gt"] = {
                "enabled": True,
                "error": str(exc),
                "accepted_iterations": 0,
                "history": [],
            }
            out_plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            out_req_path.write_text(
                json.dumps(
                    {
                        "building": bdir.name,
                        "source_plan": str(src_plan_path.relative_to(bdir)),
                        "description_path": str(desc_path.relative_to(bdir)),
                        "coerce_report": coerce_report,
                        "validation_report": validation,
                        "self_refine_error": str(exc),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue

        history.append({"iter": 0, "accepted": True, "score": metrics["score"], "components": metrics["components"], "added_ops": 0})

        for it in range(1, max(0, int(args.max_iterations)) + 1):
            corr_candidates = _propose_correction_candidates(
                plan,
                desc,
                metrics,
                max_operations=int(args.max_operations),
                max_added_ops_per_iter=int(args.max_added_ops_per_iter),
                roof_search_variants=int(args.roof_search_variants),
                window_search_variants=int(args.window_search_variants),
                max_search_candidates=int(args.max_search_candidates),
            )
            if not corr_candidates:
                history.append({"iter": it, "accepted": False, "reason": "no_corrections"})
                break

            best_result: Optional[Dict[str, Any]] = None
            render_fail_count = 0
            for cand_idx, corr_ops in enumerate(corr_candidates):
                candidate = dict(plan)
                c_ops = list(plan.get("operations", [])) + list(corr_ops)
                c_ops = _merge_with_stage_order(c_ops)
                if len(c_ops) > int(args.max_operations):
                    c_ops = c_ops[: int(args.max_operations)]
                candidate["operations"] = c_ops

                candidate, cand_validation = _validate_and_repair_plan(
                    candidate,
                    desc=desc if isinstance(desc, dict) else {},
                    strict_schema=bool(args.strict_schema),
                    enforce_role_fixed=bool(args.enforce_role_fixed),
                    require_material_budget=bool(args.require_material_budget),
                    material_budget_tolerance=float(args.material_budget_tolerance),
                    role_fix_min_confidence=float(args.role_fix_min_confidence),
                    prefer_description_palette=bool(args.prefer_description_palette),
                    max_operations=int(args.max_operations),
                    required_palette_roles=required_roles,
                )

                try:
                    c_vox, c_bbox = _render_plan(candidate, max_dim=int(args.max_dim))
                    c_metrics = _self_consistency_score(candidate, c_vox, c_bbox, desc)
                except Exception:  # noqa: BLE001
                    render_fail_count += 1
                    continue

                budget_reproject_ops: List[Dict[str, Any]] = []
                budget_reproject_report: Dict[str, Any] = {}
                c_material = float(c_metrics.get("components", {}).get("material", 1.0))
                if bool(args.enable_material_budget_reprojection) and c_material < float(
                    args.material_budget_reprojection_trigger_material_score
                ):
                    budget_reproject_ops, budget_reproject_report = _propose_material_budget_reprojection_ops(
                        candidate,
                        c_metrics,
                        c_vox,
                        c_bbox,
                        max_operations=int(args.max_operations),
                        max_added_ops_per_iter=max(8, int(args.max_added_ops_per_iter) // 2),
                        strength=float(args.material_budget_reprojection_strength),
                        min_deficit_ratio=float(args.material_budget_reprojection_min_deficit_ratio),
                    )
                    if budget_reproject_ops:
                        c2 = dict(candidate)
                        c2_ops = list(candidate.get("operations", [])) + budget_reproject_ops
                        c2_ops = _merge_with_stage_order(c2_ops)
                        if len(c2_ops) > int(args.max_operations):
                            c2_ops = c2_ops[: int(args.max_operations)]
                        c2["operations"] = c2_ops
                        c2, c2_validation = _validate_and_repair_plan(
                            c2,
                            desc=desc if isinstance(desc, dict) else {},
                            strict_schema=bool(args.strict_schema),
                            enforce_role_fixed=bool(args.enforce_role_fixed),
                            require_material_budget=bool(args.require_material_budget),
                            material_budget_tolerance=float(args.material_budget_tolerance),
                            role_fix_min_confidence=float(args.role_fix_min_confidence),
                            prefer_description_palette=bool(args.prefer_description_palette),
                            max_operations=int(args.max_operations),
                            required_palette_roles=required_roles,
                        )
                        try:
                            c2_vox, c2_bbox = _render_plan(c2, max_dim=int(args.max_dim))
                            c2_metrics = _self_consistency_score(c2, c2_vox, c2_bbox, desc)
                            if float(c2_metrics["score"]) >= float(c_metrics["score"]):
                                candidate = c2
                                cand_validation = c2_validation
                                c_metrics = c2_metrics
                                c_vox = c2_vox
                                c_bbox = c2_bbox
                        except Exception:  # noqa: BLE001
                            pass

                total_added_ops = int(len(corr_ops) + len(budget_reproject_ops))
                selection_score = float(c_metrics["score"]) - float(args.selection_op_penalty) * float(total_added_ops)
                result = {
                    "candidate": candidate,
                    "metrics": c_metrics,
                    "validation": cand_validation,
                    "corr_ops_count": len(corr_ops),
                    "candidate_index": int(cand_idx),
                    "budget_reproject_ops_count": len(budget_reproject_ops),
                    "budget_reproject_report": budget_reproject_report,
                    "selection_score": selection_score,
                }
                if best_result is None or float(result["selection_score"]) > float(best_result["selection_score"]):
                    best_result = result

            if best_result is None:
                history.append({"iter": it, "accepted": False, "reason": f"all_candidates_failed_render:{render_fail_count}"})
                break

            gain = float(best_result["metrics"]["score"]) - float(metrics["score"])
            accepted = gain >= float(args.min_score_gain)
            history.append(
                {
                    "iter": it,
                    "accepted": bool(accepted),
                    "gain": float(gain),
                    "score": best_result["metrics"]["score"],
                    "components": best_result["metrics"]["components"],
                    "added_ops": int(best_result["corr_ops_count"] + best_result["budget_reproject_ops_count"]),
                    "candidate_index": int(best_result["candidate_index"]),
                    "candidates_tested": len(corr_candidates),
                    "render_fail_count": int(render_fail_count),
                    "budget_reproject_ops_count": int(best_result["budget_reproject_ops_count"]),
                    "budget_reproject_report": best_result["budget_reproject_report"],
                    "selection_score": float(best_result["selection_score"]),
                    "validation_report": {
                        "strict_blocking_issues": best_result["validation"].get("strict_blocking_issues", []),
                        "budget_violations_count": len(best_result["validation"].get("budget_violations", [])),
                    },
                }
            )
            if not accepted:
                break

            plan = best_result["candidate"]
            validation = best_result["validation"]
            metrics = best_result["metrics"]
            accepted_iterations += 1
            total_added_ops += int(best_result["corr_ops_count"] + best_result["budget_reproject_ops_count"])

        plan["self_refine_no_gt"] = {
            "enabled": True,
            "accepted_iterations": int(accepted_iterations),
            "total_added_ops": int(total_added_ops),
            "final_score": float(metrics["score"]),
            "final_components": metrics["components"],
            "history": history,
            "profile": {
                "max_iterations": int(args.max_iterations),
                "min_score_gain": float(args.min_score_gain),
                "max_added_ops_per_iter": int(args.max_added_ops_per_iter),
                "roof_search_variants": int(args.roof_search_variants),
                "window_search_variants": int(args.window_search_variants),
                "max_search_candidates": int(args.max_search_candidates),
                "enable_material_budget_reprojection": bool(args.enable_material_budget_reprojection),
                "material_budget_reprojection_strength": float(args.material_budget_reprojection_strength),
                "material_budget_reprojection_min_deficit_ratio": float(args.material_budget_reprojection_min_deficit_ratio),
                "material_budget_reprojection_trigger_material_score": float(args.material_budget_reprojection_trigger_material_score),
                "selection_op_penalty": float(args.selection_op_penalty),
                "strict_schema": bool(args.strict_schema),
                "enforce_role_fixed": bool(args.enforce_role_fixed),
                "require_material_budget": bool(args.require_material_budget),
                "material_budget_tolerance": float(args.material_budget_tolerance),
                "role_fix_min_confidence": float(args.role_fix_min_confidence),
                "prefer_description_palette": bool(args.prefer_description_palette),
            },
        }
        plan["validation_report"] = validation
        plan["building"] = bdir.name
        plan["created_at"] = datetime.now(timezone.utc).isoformat()

        out_plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        out_req_path.write_text(
            json.dumps(
                {
                    "building": bdir.name,
                    "source_plan": str(src_plan_path.relative_to(bdir)),
                    "description_path": str(desc_path.relative_to(bdir)),
                    "coerce_report": coerce_report,
                    "validation_report": validation,
                    "self_refine_no_gt": plan["self_refine_no_gt"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[self_refine_no_gt] wrote {out_plan_path}")


if __name__ == "__main__":
    main()
