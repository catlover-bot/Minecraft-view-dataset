#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


KNOWN_BLOCKS = {
    "air",
    "stone",
    "cobblestone",
    "stonebrick",
    "planks",
    "log",
    "sandstone",
    "brick_block",
    "nether_brick",
    "quartz_block",
    "glass",
    "stained_glass",
    "glowstone",
    "sea_lantern",
    "stone_slab",
    "wooden_slab",
    "double_stone_slab",
    "double_wooden_slab",
    "oak_stairs",
    "spruce_stairs",
    "birch_stairs",
    "jungle_stairs",
    "acacia_stairs",
    "dark_oak_stairs",
    "stone_stairs",
    "brick_stairs",
    "nether_brick_stairs",
    "sandstone_stairs",
    "quartz_stairs",
    "fence",
    "spruce_fence",
    "birch_fence",
    "jungle_fence",
    "acacia_fence",
    "dark_oak_fence",
    "torch",
    "redstone_lamp",
}


BLOCK_ALIASES = {
    "stone_bricks": "stonebrick",
    "stone_brick": "stonebrick",
    "stone brick": "stonebrick",
    "minecraft:stone_bricks": "stonebrick",
    "netherbrick": "nether_brick",
    "nether_bricks": "nether_brick",
    "wood": "planks",
    "oak_planks": "planks",
    "spruce_planks": "planks",
    "birch_planks": "planks",
    "jungle_planks": "planks",
    "acacia_planks": "planks",
    "dark_oak_planks": "planks",
    "fence": "fence",
    "wooden_fence": "fence",
    "oakfence": "fence",
    "oak_fence": "fence",
    "minecraft:oak_fence": "fence",
    "window": "glass",
    "light": "glowstone",
    "slab": "stone_slab",
    "slab_stone": "stone_slab",
    "stone_slab2": "stone_slab",
    "brick": "brick_block",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate proxy agent-executed rebuild worlds by applying the same "
            "block sanitization rules used in Malmo capture."
        )
    )
    parser.add_argument("--dataset_root", required=True, help="Root containing building_xxx dirs (usually outputs/i2t2b/<dataset>).")
    parser.add_argument("--source_subdir", required=True, help="Source rebuild world subdir name.")
    parser.add_argument("--out_subdir", required=True, help="Destination agent-exec subdir name.")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern.")
    parser.add_argument("--limit", type=int, default=0, help="Max buildings (0=all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def sanitize_block_name(raw: Any) -> str:
    name = str(raw).strip().lower()
    if not name or name == "air":
        return "air"
    if ":" in name:
        name = name.split(":", 1)[1]
    name = name.replace("-", "_").replace(" ", "_")
    name = BLOCK_ALIASES.get(name, name)

    if name.endswith("_planks"):
        name = "planks"
    elif name == "oak_fence":
        name = "fence"
    elif name.endswith("_fence"):
        name = name
    elif name.endswith("_slab"):
        name = "stone_slab"
    elif name.startswith("stone_slab"):
        name = "stone_slab"

    if name in KNOWN_BLOCKS:
        return name

    if "glass" in name:
        return "glass"
    if "brick" in name and "nether" in name:
        return "nether_brick"
    if "brick" in name:
        return "brick_block"
    if "fence" in name:
        return "fence"
    if "slab" in name:
        return "stone_slab"
    if "stair" in name:
        return "stone_stairs"
    if "wood" in name or "plank" in name:
        return "planks"
    if "light" in name or "glow" in name:
        return "glowstone"
    if "stone" in name:
        return "stonebrick"
    return "stone"


def _list_buildings(root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _load_bbox(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_non_air(arr: np.ndarray) -> int:
    return int(np.sum(arr != "air"))


def convert_voxels(voxels: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    out = np.empty_like(voxels)
    alias_or_normalized_count = 0
    fallback_to_stone_count = 0

    it = np.nditer(voxels, flags=["multi_index", "refs_ok"])
    for raw in it:
        raw_s = str(raw.item())
        raw_l = raw_s.strip().lower()
        norm = sanitize_block_name(raw_s)
        out[it.multi_index] = norm
        if norm != raw_l:
            alias_or_normalized_count += 1
        if norm == "stone" and raw_l not in {"stone", "minecraft:stone"}:
            fallback_to_stone_count += 1

    stats = {
        "alias_or_normalized_count": alias_or_normalized_count,
        "fallback_to_stone_count": fallback_to_stone_count,
    }
    return out, stats


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    buildings = _list_buildings(dataset_root, args.building_pattern, int(args.limit))
    done = 0
    skipped = 0
    missing = 0

    for bdir in buildings:
        src_dir = bdir / args.source_subdir
        out_dir = bdir / args.out_subdir
        src_vox = src_dir / "voxels.npy"
        src_bbox = src_dir / "bbox.json"
        src_actions = src_dir / "actions.json"
        if not src_vox.is_file() or not src_bbox.is_file():
            missing += 1
            continue

        out_vox = out_dir / "voxels.npy"
        out_bbox = out_dir / "bbox.json"
        out_actions = out_dir / "actions.json"
        out_report = out_dir / "agentexec_proxy_report.json"
        if out_vox.is_file() and out_bbox.is_file() and not args.overwrite:
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        vox = np.load(src_vox, allow_pickle=False)
        converted, convert_stats = convert_voxels(vox)
        np.save(out_vox, converted)

        bbox = _load_bbox(src_bbox)
        bbox_out = {
            **bbox,
            "source": str(src_bbox),
            "generated_by": "tools/generate_agentexec_world_proxy.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "agentexec_mode": "proxy_sanitize_only",
        }
        out_bbox.write_text(json.dumps(bbox_out, ensure_ascii=False, indent=2), encoding="utf-8")
        if src_actions.is_file():
            out_actions.write_text(src_actions.read_text(encoding="utf-8"), encoding="utf-8")

        report = {
            "building": bdir.name,
            "source_subdir": args.source_subdir,
            "out_subdir": args.out_subdir,
            "source_voxels": str(src_vox),
            "out_voxels": str(out_vox),
            "source_non_air_blocks": _count_non_air(vox),
            "out_non_air_blocks": _count_non_air(converted),
            **convert_stats,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1

    print(
        "[generate_agentexec_world_proxy] "
        f"dataset_root={dataset_root} source_subdir={args.source_subdir} out_subdir={args.out_subdir} "
        f"done={done} skipped={skipped} missing_source={missing}"
    )


if __name__ == "__main__":
    main()
