#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from tools.evaluate_rebuild_metrics import normalize_block_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render rebuild plan JSON into rebuild_world voxels.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root with building_xxx dirs")
    parser.add_argument("--plan_subdir", default="rebuild_plan", help="Subdir containing plan.json")
    parser.add_argument("--out_subdir", default="rebuild_world", help="Output subdir for rendered voxels")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern")
    parser.add_argument("--limit", type=int, default=0, help="Max building count (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--max_dim", type=int, default=192, help="Max dimension per axis for safety")
    return parser.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in dataset_root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _op_bounds(op: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    kind = str(op.get("op", "")).strip().lower()
    if kind in {"fill", "carve"}:
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
        return x1, x2, y1, y2, z1, z2

    x = _int(op.get("x", 0), 0)
    y = _int(op.get("y", 0), 0)
    z = _int(op.get("z", 0), 0)
    return x, x, y, y, z, z


def _resolve_bbox(plan: Dict[str, Any]) -> Dict[str, int]:
    bbox = plan.get("bbox") if isinstance(plan.get("bbox"), dict) else {}
    xmin = _int(bbox.get("xmin", 0), 0)
    xmax = _int(bbox.get("xmax", 0), 0)
    ymin = _int(bbox.get("ymin", 0), 0)
    ymax = _int(bbox.get("ymax", 0), 0)
    zmin = _int(bbox.get("zmin", 0), 0)
    zmax = _int(bbox.get("zmax", 0), 0)

    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    if zmax < zmin:
        zmin, zmax = zmax, zmin

    for op in plan.get("operations", []) if isinstance(plan.get("operations"), list) else []:
        if not isinstance(op, dict):
            continue
        ox1, ox2, oy1, oy2, oz1, oz2 = _op_bounds(op)
        xmin = min(xmin, ox1)
        xmax = max(xmax, ox2)
        ymin = min(ymin, oy1)
        ymax = max(ymax, oy2)
        zmin = min(zmin, oz1)
        zmax = max(zmax, oz2)

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }


def _apply_fill(vox: np.ndarray, bbox: Dict[str, int], x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, block: str) -> None:
    xmin, ymin, zmin = bbox["xmin"], bbox["ymin"], bbox["zmin"]
    ix1, ix2 = x1 - xmin, x2 - xmin
    iy1, iy2 = y1 - ymin, y2 - ymin
    iz1, iz2 = z1 - zmin, z2 - zmin
    vox[iy1 : iy2 + 1, ix1 : ix2 + 1, iz1 : iz2 + 1] = block


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    buildings = _list_buildings(dataset_root, args.building_pattern, args.limit)
    for bdir in buildings:
        plan_path = bdir / args.plan_subdir / "plan.json"
        if not plan_path.is_file():
            print(f"[render_rebuild_from_plan] skip {bdir.name} (no plan.json)")
            continue

        out_dir = bdir / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_bbox = out_dir / "bbox.json"
        out_vox = out_dir / "voxels.npy"
        out_actions = out_dir / "actions.json"

        if out_vox.is_file() and out_bbox.is_file() and not args.overwrite:
            print(f"[render_rebuild_from_plan] skip {bdir.name} (exists)")
            continue

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        bbox = _resolve_bbox(plan)

        sx = bbox["xmax"] - bbox["xmin"] + 1
        sy = bbox["ymax"] - bbox["ymin"] + 1
        sz = bbox["zmax"] - bbox["zmin"] + 1
        if sx <= 0 or sy <= 0 or sz <= 0:
            print(f"[render_rebuild_from_plan] skip {bdir.name} (invalid bbox)")
            continue
        if max(sx, sy, sz) > int(args.max_dim):
            print(f"[render_rebuild_from_plan] skip {bdir.name} (bbox too large: {sx}x{sy}x{sz})")
            continue

        vox = np.full((sy, sx, sz), "air", dtype="<U32")
        applied_ops: List[Dict[str, Any]] = []

        for op in plan.get("operations", []) if isinstance(plan.get("operations"), list) else []:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "")).strip().lower()
            if kind not in {"fill", "carve", "set"}:
                continue

            if kind in {"fill", "carve"}:
                x1, x2, y1, y2, z1, z2 = _op_bounds(op)
                block = "air" if kind == "carve" else normalize_block_type(op.get("block", "air"))
                _apply_fill(vox, bbox, x1, y1, z1, x2, y2, z2, block)
                applied_ops.append(
                    {
                        "op": kind,
                        "x1": x1,
                        "y1": y1,
                        "z1": z1,
                        "x2": x2,
                        "y2": y2,
                        "z2": z2,
                        "block": block,
                        "purpose": str(op.get("purpose", "")),
                    }
                )
            else:
                x = _int(op.get("x", 0), 0)
                y = _int(op.get("y", 0), 0)
                z = _int(op.get("z", 0), 0)
                block = normalize_block_type(op.get("block", "air"))
                _apply_fill(vox, bbox, x, y, z, x, y, z, block)
                applied_ops.append(
                    {
                        "op": "set",
                        "x": x,
                        "y": y,
                        "z": z,
                        "block": block,
                        "purpose": str(op.get("purpose", "")),
                    }
                )

        np.save(out_vox, vox)
        bbox_payload = {
            **bbox,
            "order": "xmin,xmax,ymin,ymax,zmin,zmax",
            "voxel_axis_order": "Y,X,Z",
            "source": f"{args.plan_subdir}/plan.json",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        out_bbox.write_text(json.dumps(bbox_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        out_actions.write_text(json.dumps({"operations": applied_ops}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[render_rebuild_from_plan] wrote {out_vox}")


if __name__ == "__main__":
    main()
