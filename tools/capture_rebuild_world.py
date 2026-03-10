#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.capture_one_building import (  # noqa: E402
    CaptureError,
    Logger,
    Pose,
    build_mission_xml,
    capture_views,
    compute_view_poses,
    expected_min_non_air_count,
    load_malmo,
    prewarm_viewpoints,
    start_mission,
    wait_for_generation_stable,
    wait_for_mission_begin,
    wait_for_observation,
    wait_for_stable_video_frame,
)


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
    "stone_brick_stairs": "stone_stairs",
    "brick": "brick_block",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RGB screenshots from an existing rebuild_world.")
    parser.add_argument(
        "--rebuild_world_dir",
        required=True,
        help="Path to directory containing bbox.json and voxels.npy.",
    )
    parser.add_argument("--out", required=True, help="Output directory path.")
    parser.add_argument("--port", type=int, default=10000, help="Malmo client port.")
    parser.add_argument("--views", type=int, default=8, help="Number of viewpoints.")
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=[960, 540],
        help="Image size: --image_size W H",
    )
    parser.add_argument("--fov", type=float, default=70.0, help="FOV metadata.")
    parser.add_argument("--shift_x", type=int, default=0, help="World shift for x.")
    parser.add_argument("--shift_y", type=int, default=4, help="World shift for y.")
    parser.add_argument("--shift_z", type=int, default=0, help="World shift for z.")
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


def load_rebuild_voxels(rebuild_world_dir: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    vox_path = rebuild_world_dir / "voxels.npy"
    bbox_path = rebuild_world_dir / "bbox.json"
    if not vox_path.is_file():
        raise CaptureError(f"voxels.npy not found: {vox_path}")
    if not bbox_path.is_file():
        raise CaptureError(f"bbox.json not found: {bbox_path}")
    voxels = np.load(vox_path, allow_pickle=False)
    bbox_raw = json.loads(bbox_path.read_text(encoding="utf-8"))
    bbox = {
        "xmin": int(bbox_raw["xmin"]),
        "xmax": int(bbox_raw["xmax"]),
        "ymin": int(bbox_raw["ymin"]),
        "ymax": int(bbox_raw["ymax"]),
        "zmin": int(bbox_raw["zmin"]),
        "zmax": int(bbox_raw["zmax"]),
    }
    sy, sx, sz = voxels.shape
    bx = bbox["xmax"] - bbox["xmin"] + 1
    by = bbox["ymax"] - bbox["ymin"] + 1
    bz = bbox["zmax"] - bbox["zmin"] + 1
    if (sy, sx, sz) != (by, bx, bz):
        raise CaptureError(
            "bbox と voxels の shape が不一致です: "
            f"voxels={voxels.shape}, bbox_size={(by, bx, bz)}"
        )
    return voxels, bbox


def voxels_to_blocks(
    voxels: np.ndarray,
    bbox: Dict[str, int],
    shift: Tuple[int, int, int],
) -> Tuple[List[Tuple[int, int, int, str]], Dict[str, int], Dict[str, int]]:
    xmin, ymin, zmin = bbox["xmin"], bbox["ymin"], bbox["zmin"]
    sx, sy, sz = shift[0], shift[1], shift[2]
    blocks: List[Tuple[int, int, int, str]] = []
    fallback_count = 0
    alias_count = 0
    unique_raw: Dict[str, int] = {}
    for yi in range(voxels.shape[0]):
        for xi in range(voxels.shape[1]):
            for zi in range(voxels.shape[2]):
                raw = str(voxels[yi, xi, zi])
                unique_raw[raw] = unique_raw.get(raw, 0) + 1
                block = sanitize_block_name(raw)
                if block == "air":
                    continue
                if block == "stone" and raw.strip().lower() not in {"stone", "minecraft:stone"}:
                    fallback_count += 1
                if block != raw.strip().lower():
                    alias_count += 1
                x = xmin + xi + sx
                y = ymin + yi + sy
                z = zmin + zi + sz
                blocks.append((x, y, z, block))

    shifted_bbox = {
        "xmin": bbox["xmin"] + sx,
        "xmax": bbox["xmax"] + sx,
        "ymin": bbox["ymin"] + sy,
        "ymax": bbox["ymax"] + sy,
        "zmin": bbox["zmin"] + sz,
        "zmax": bbox["zmax"] + sz,
    }
    stats = {
        "non_air_blocks": len(blocks),
        "raw_unique_block_types": len(unique_raw),
        "alias_or_normalized_count": alias_count,
        "fallback_to_stone_count": fallback_count,
    }
    return blocks, shifted_bbox, stats


def cleanup_previous_images(out_images_dir: Path, logger: Logger) -> None:
    removed = 0
    if out_images_dir.exists():
        for p in out_images_dir.glob("*.png"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    out_images_dir.mkdir(parents=True, exist_ok=True)
    if removed > 0:
        logger.log(f"cleaned previous images: removed {removed} png files")


def make_anchor_pose_from_bbox(bbox: Dict[str, int]) -> Pose:
    cx = (bbox["xmin"] + bbox["xmax"]) / 2.0
    cz = (bbox["zmin"] + bbox["zmax"]) / 2.0
    y = max(4.0, float(bbox["ymin"] + 2))
    return Pose(x=cx + 0.5, y=y, z=cz + 0.5, yaw=0.0, pitch=0.0)


def build_meta(
    rebuild_world_dir: Path,
    shifted_bbox: Dict[str, int],
    fov: float,
    image_w: int,
    image_h: int,
    radius: float,
    target: Tuple[float, float, float],
    stable_non_air_count: int,
    block_stats: Dict[str, int],
    views: List[Dict[str, Any]],
    shift: Tuple[int, int, int],
) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rebuild_world_dir": str(rebuild_world_dir),
        "bbox": shifted_bbox,
        "shift": {"x": shift[0], "y": shift[1], "z": shift[2]},
        "image_size": {"width": image_w, "height": image_h},
        "camera_planner": {
            "radius": round(radius, 4),
            "target": {"x": round(target[0], 4), "y": round(target[1], 4), "z": round(target[2], 4)},
            "fov": float(fov),
        },
        "generation": {
            "stable_non_air_count": int(stable_non_air_count),
            **block_stats,
        },
        "views": views,
    }


def run() -> int:
    args = parse_args()
    rebuild_world_dir = Path(args.rebuild_world_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    image_w, image_h = int(args.image_size[0]), int(args.image_size[1])
    shift = (int(args.shift_x), int(args.shift_y), int(args.shift_z))
    out_images_dir = out_dir / "images"
    out_logs_dir = out_dir / "logs"
    out_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_logs_dir / "capture.log"
    logger = Logger(log_path)
    agent_host: Optional[Any] = None

    try:
        if not rebuild_world_dir.is_dir():
            raise CaptureError(f"rebuild_world_dir not found: {rebuild_world_dir}")
        cleanup_previous_images(out_images_dir=out_images_dir, logger=logger)

        voxels, bbox = load_rebuild_voxels(rebuild_world_dir)
        blocks, shifted_bbox, block_stats = voxels_to_blocks(voxels=voxels, bbox=bbox, shift=shift)
        if not blocks:
            raise CaptureError("rebuild_world has no non-air blocks after conversion.")
        logger.log(
            "rebuild loaded: "
            f"source_bbox={bbox}, shifted_bbox={shifted_bbox}, "
            f"non_air_blocks={block_stats['non_air_blocks']}, "
            f"raw_unique={block_stats['raw_unique_block_types']}, "
            f"alias_or_normalized={block_stats['alias_or_normalized_count']}, "
            f"fallback_to_stone={block_stats['fallback_to_stone_count']}"
        )

        poses, radius, target = compute_view_poses(
            shifted_bbox,
            views=args.views,
            fov=args.fov,
            image_w=image_w,
            image_h=image_h,
        )
        logger.log(f"computed {len(poses)} camera poses with radius={radius:.2f}")

        anchor_pose = make_anchor_pose_from_bbox(shifted_bbox)
        anchor_block = (
            int(math.floor(anchor_pose.x)),
            int(math.floor(anchor_pose.y)),
            int(math.floor(anchor_pose.z)),
        )
        grid_name = "build_grid"
        grid_min = (
            shifted_bbox["xmin"] - anchor_block[0],
            shifted_bbox["ymin"] - anchor_block[1],
            shifted_bbox["zmin"] - anchor_block[2],
        )
        grid_max = (
            shifted_bbox["xmax"] - anchor_block[0],
            shifted_bbox["ymax"] - anchor_block[1],
            shifted_bbox["zmax"] - anchor_block[2],
        )

        mission_xml = build_mission_xml(
            image_w=image_w,
            image_h=image_h,
            blocks=blocks,
            bbox=shifted_bbox,
            grid_name=grid_name,
            grid_min=grid_min,
            grid_max=grid_max,
            start_pose=anchor_pose,
        )

        MalmoPython = load_malmo()
        agent_host = MalmoPython.AgentHost()
        start_mission(MalmoPython, agent_host, mission_xml, port=args.port, logger=logger)
        wait_for_mission_begin(agent_host, timeout_sec=180.0, logger=logger)
        _ = wait_for_observation(agent_host, timeout_sec=20.0)
        logger.log("Initial observation received.")

        min_non_air_count = expected_min_non_air_count(len(blocks))
        logger.log(
            "generation stability threshold: "
            f"min_non_air_count={min_non_air_count} / authored_blocks={len(blocks)}"
        )
        stable_count = wait_for_generation_stable(
            agent_host=agent_host,
            grid_name=grid_name,
            logger=logger,
            min_non_air_count=min_non_air_count,
            stable_k=4,
            sample_interval_sec=0.5,
            max_samples=180,
            max_seconds=90.0,
        )

        logger.log("Waiting extra render stabilization at anchor pose.")
        try:
            wait_for_stable_video_frame(
                agent_host,
                timeout_sec=12.0,
                stable_k=2,
                diff_threshold=3.5,
                min_elapsed_sec=1.0,
                min_center_std=2.5,
                allow_timeout_fallback=True,
                hold_pose=anchor_pose,
                hold_interval_sec=0.65,
            )
        except CaptureError as exc:
            logger.log(f"WARN: anchor stabilization skipped: {exc}")

        try:
            prewarm_viewpoints(agent_host=agent_host, poses=poses, logger=logger)
        except CaptureError as exc:
            logger.log(f"WARN: prewarm partially skipped: {exc}")

        view_records = capture_views(
            agent_host=agent_host,
            out_images_dir=out_images_dir,
            poses=poses,
            fov=args.fov,
            image_w=image_w,
            image_h=image_h,
            logger=logger,
        )

        meta = build_meta(
            rebuild_world_dir=rebuild_world_dir,
            shifted_bbox=shifted_bbox,
            fov=args.fov,
            image_w=image_w,
            image_h=image_h,
            radius=radius,
            target=target,
            stable_non_air_count=stable_count,
            block_stats=block_stats,
            views=view_records,
            shift=shift,
        )
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.log(f"meta saved: {meta_path}")

        agent_host.sendCommand("quit")
        logger.log("capture completed.")
        return 0
    except CaptureError as exc:
        logger.log(f"ERROR: {exc}")
        print(f"[capture_rebuild_world] ERROR: {exc}", file=sys.stderr)
        print(f"[capture_rebuild_world] capture log: {log_path}", file=sys.stderr)
        print("[capture_rebuild_world] also check Malmo client logs: ./logs/malmo_client.log", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.log(f"UNEXPECTED ERROR: {exc}")
        print(f"[capture_rebuild_world] UNEXPECTED ERROR: {exc}", file=sys.stderr)
        print(f"[capture_rebuild_world] capture log: {log_path}", file=sys.stderr)
        print("[capture_rebuild_world] also check Malmo client logs: ./logs/malmo_client.log", file=sys.stderr)
        return 1
    finally:
        if agent_host is not None:
            try:
                agent_host.sendCommand("quit")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(run())
