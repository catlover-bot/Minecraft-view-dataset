#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate deterministic rebuild plan from structured intermediate.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--intermediate_subdir", default="structured_intermediate")
    p.add_argument("--out_subdir", default="rebuild_plan_structured")
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    xs = sorted([p for p in dataset_root.glob(pattern) if p.is_dir()])
    if limit > 0:
        xs = xs[:limit]
    return xs


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _roof_ops(
    *,
    xmin: int,
    xmax: int,
    zmin: int,
    zmax: int,
    roof_y: int,
    roof_type: str,
    roof_h: int,
    roof_block: str,
) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    roof_type = (roof_type or "flat").strip().lower()
    if roof_type == "flat":
        ops.append({"op": "fill", "x1": xmin, "y1": roof_y, "z1": zmin, "x2": xmax, "y2": roof_y, "z2": zmax, "block": roof_block, "purpose": "roof_flat"})
        return ops

    if roof_type in {"gable_x", "gable"}:
        half = max(1, (xmax - xmin + 1) // 2)
        for x in range(xmin, xmax + 1):
            rel = min(x - xmin, xmax - x)
            h = max(0, min(roof_h, int(round((rel / half) * roof_h))))
            ops.append({"op": "fill", "x1": x, "y1": roof_y, "z1": zmin, "x2": x, "y2": roof_y + h, "z2": zmax, "block": roof_block, "purpose": "roof_gable_x"})
        return ops

    if roof_type == "gable_z":
        half = max(1, (zmax - zmin + 1) // 2)
        for z in range(zmin, zmax + 1):
            rel = min(z - zmin, zmax - z)
            h = max(0, min(roof_h, int(round((rel / half) * roof_h))))
            ops.append({"op": "fill", "x1": xmin, "y1": roof_y, "z1": z, "x2": xmax, "y2": roof_y + h, "z2": z, "block": roof_block, "purpose": "roof_gable_z"})
        return ops

    # hip
    for x in range(xmin, xmax + 1):
        for z in range(zmin, zmax + 1):
            dist = min(x - xmin, xmax - x, z - zmin, zmax - z)
            h = max(0, min(roof_h, dist))
            ops.append({"op": "fill", "x1": x, "y1": roof_y, "z1": z, "x2": x, "y2": roof_y + h, "z2": z, "block": roof_block, "purpose": "roof_hip"})
    return ops


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    for bdir in _list_buildings(dataset_root, args.building_pattern, args.limit):
        inter_path = bdir / args.intermediate_subdir / "intermediate.json"
        out_dir = bdir / args.out_subdir
        out_plan = out_dir / "plan.json"
        out_req = out_dir / "plan.request.json"

        if out_plan.is_file() and not args.overwrite:
            print(f"[generate_plan_from_intermediate] skip {bdir.name} (exists)")
            continue
        if not inter_path.is_file():
            print(f"[generate_plan_from_intermediate] skip {bdir.name} (missing intermediate)")
            continue

        inter = _load_json(inter_path)
        intent = inter.get("rebuild_intent", {}) if isinstance(inter.get("rebuild_intent"), dict) else {}
        dims = intent.get("dimensions", {}) if isinstance(intent.get("dimensions"), dict) else {}
        foot = intent.get("footprint", {}) if isinstance(intent.get("footprint"), dict) else {}
        mats = intent.get("materials", {}) if isinstance(intent.get("materials"), dict) else {}
        openings = intent.get("openings", {}) if isinstance(intent.get("openings"), dict) else {}
        hp = intent.get("height_profile", {}) if isinstance(intent.get("height_profile"), dict) else {}

        width = max(8, int(round(float(dims.get("width", 12)))))
        depth = max(8, int(round(float(dims.get("depth", 10)))))
        floors = max(1, int(round(float(dims.get("floors", 1)))))
        floor_h = max(3, int(round(float(dims.get("floor_height", 4)))))

        xmin, zmin = 0, 0
        xmax, zmax = width - 1, depth - 1
        ymin = 0
        ymax = floors * floor_h + 4

        wall = str(mats.get("wall", "planks"))
        floor_block = str(mats.get("floor", "planks"))
        roof_block = str(mats.get("roof", "nether_brick"))
        win_block = str(mats.get("window", "glass"))
        pillar_block = str(mats.get("pillar", str(mats.get("foundation", "stonebrick"))))
        trim_block = str(mats.get("trim", "stone_slab"))

        ops: List[Dict[str, Any]] = []
        # foundation and floor slabs
        ops.append({"op": "fill", "x1": xmin, "y1": ymin - 1, "z1": zmin, "x2": xmax, "y2": ymin - 1, "z2": zmax, "block": str(mats.get("foundation", "stonebrick")), "purpose": "foundation"})
        for f in range(floors):
            fy = ymin + f * floor_h
            ops.append({"op": "fill", "x1": xmin, "y1": fy, "z1": zmin, "x2": xmax, "y2": fy, "z2": zmax, "block": floor_block, "purpose": "floor"})

            # walls shell
            ops.append({"op": "fill", "x1": xmin, "y1": fy + 1, "z1": zmin, "x2": xmax, "y2": fy + floor_h - 1, "z2": zmin, "block": wall, "purpose": "wall_north"})
            ops.append({"op": "fill", "x1": xmin, "y1": fy + 1, "z1": zmax, "x2": xmax, "y2": fy + floor_h - 1, "z2": zmax, "block": wall, "purpose": "wall_south"})
            ops.append({"op": "fill", "x1": xmin, "y1": fy + 1, "z1": zmin, "x2": xmin, "y2": fy + floor_h - 1, "z2": zmax, "block": wall, "purpose": "wall_west"})
            ops.append({"op": "fill", "x1": xmax, "y1": fy + 1, "z1": zmin, "x2": xmax, "y2": fy + floor_h - 1, "z2": zmax, "block": wall, "purpose": "wall_east"})

            # pillars + trim ring
            for x, z in [(xmin, zmin), (xmin, zmax), (xmax, zmin), (xmax, zmax)]:
                ops.append({"op": "fill", "x1": x, "y1": fy + 1, "z1": z, "x2": x, "y2": fy + floor_h - 1, "z2": z, "block": pillar_block, "purpose": "pillar"})
            ops.append({"op": "fill", "x1": xmin, "y1": fy + floor_h - 1, "z1": zmin, "x2": xmax, "y2": fy + floor_h - 1, "z2": zmin, "block": trim_block, "purpose": "trim_north"})
            ops.append({"op": "fill", "x1": xmin, "y1": fy + floor_h - 1, "z1": zmax, "x2": xmax, "y2": fy + floor_h - 1, "z2": zmax, "block": trim_block, "purpose": "trim_south"})

            # window rows
            w_y0 = fy + 2
            w_y1 = min(fy + floor_h - 2, w_y0 + max(1, int(openings.get("window_height", 2))) - 1)
            spacing = max(2, int(openings.get("window_spacing", 3)))
            for x in range(xmin + 1, xmax):
                if x % spacing == 0:
                    ops.append({"op": "fill", "x1": x, "y1": w_y0, "z1": zmin, "x2": x, "y2": w_y1, "z2": zmin, "block": win_block, "purpose": "window_north"})
                    ops.append({"op": "fill", "x1": x, "y1": w_y0, "z1": zmax, "x2": x, "y2": w_y1, "z2": zmax, "block": win_block, "purpose": "window_south"})
            for z in range(zmin + 1, zmax):
                if z % spacing == 0:
                    ops.append({"op": "fill", "x1": xmin, "y1": w_y0, "z1": z, "x2": xmin, "y2": w_y1, "z2": z, "block": win_block, "purpose": "window_west"})
                    ops.append({"op": "fill", "x1": xmax, "y1": w_y0, "z1": z, "x2": xmax, "y2": w_y1, "z2": z, "block": win_block, "purpose": "window_east"})

        # entrance carve
        door_side = str(openings.get("door_side", "south")).lower()
        door_h = max(2, int(openings.get("door_height", 2)))
        dx = (xmin + xmax) // 2
        dz = zmin
        if door_side == "north":
            dz = zmax
        elif door_side == "west":
            dx = xmin
            dz = (zmin + zmax) // 2
        elif door_side == "east":
            dx = xmax
            dz = (zmin + zmax) // 2

        ops.append({"op": "fill", "x1": dx, "y1": ymin + 1, "z1": dz, "x2": dx, "y2": ymin + door_h, "z2": dz, "block": "air", "purpose": "entrance"})

        roof_y = ymin + floors * floor_h
        roof_type = str(hp.get("roof_type", "flat"))
        roof_h = max(1, int(hp.get("roof_height", 2)))
        ops.extend(_roof_ops(xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, roof_y=roof_y, roof_type=roof_type, roof_h=roof_h, roof_block=roof_block))

        plan = {
            "bbox": {"xmin": xmin, "xmax": xmax, "ymin": ymin - 1, "ymax": ymax, "zmin": zmin, "zmax": zmax},
            "palette": {
                "wall": wall,
                "floor": floor_block,
                "roof": roof_block,
                "window": win_block,
                "trim": trim_block,
                "pillar": pillar_block,
            },
            "operations": ops,
            "material_budget": {
                "wall": {"block": wall, "target_ratio": 0.35},
                "roof": {"block": roof_block, "target_ratio": 0.20},
                "floor": {"block": floor_block, "target_ratio": 0.20},
                "window": {"block": win_block, "target_ratio": 0.10},
                "trim": {"block": trim_block, "target_ratio": 0.15},
            },
            "notes": ["diagnostic deterministic plan from structured intermediate"],
            "plan_version": "structured_intermediate_v1",
            "building": bdir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": "diagnostic_structured_ir",
            "model": "deterministic_translator",
        }

        request = {
            "building": bdir.name,
            "mode": "structured_intermediate_deterministic",
            "intermediate_path": str(inter_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fallback_triggered": False,
            "llm_attempted": False,
            "llm_failed": False,
            "validation_report": {
                "strict_schema": True,
                "valid_strict": True,
                "schema_violations": [],
                "strict_blocking_issues": [],
                "operations_trimmed": False,
            },
            "coerce_report": {
                "repaired_operations_count": 0,
                "expanded_operations_count": 0,
                "dropped_operations_count": 0,
            },
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        out_req.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[generate_plan_from_intermediate] wrote {out_plan}")


if __name__ == "__main__":
    main()
