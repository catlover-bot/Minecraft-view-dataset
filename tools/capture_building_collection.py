#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from building.generator import BuildingSpec, generate_building
from building.gt_export import export_ground_truth
from tools.capture_one_building import (
    CaptureError,
    Logger,
    Pose,
    compute_view_poses,
    frame_to_image,
    load_malmo,
    start_mission,
    wait_for_frame,
    wait_for_generation_stable,
    wait_for_mission_begin,
    wait_for_observation,
    wait_for_stable_video_frame,
    wait_until_pose_reached,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a collection of buildings in one mission.")
    parser.add_argument("--out", required=True, help="Output dataset root directory.")
    parser.add_argument("--port", type=int, default=10000, help="Malmo client port.")
    parser.add_argument("--count", type=int, default=100, help="Number of building styles to capture.")
    parser.add_argument("--start_style_id", type=int, default=0, help="First style id.")
    parser.add_argument("--views", type=int, default=12, help="Views per building.")
    parser.add_argument("--image_size", nargs=2, type=int, metavar=("W", "H"), default=[960, 540], help="Image size")
    parser.add_argument("--fov", type=float, default=70.0, help="FOV metadata for camera planning.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for deterministic variations.")
    parser.add_argument("--spacing", type=int, default=180, help="Spacing between buildings in world.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip building dirs that already have complete outputs.")
    return parser.parse_args()


def build_collection_mission_xml(
    image_w: int,
    image_h: int,
    specs: List[BuildingSpec],
    start_pose: Pose,
    grid_name: str,
    grid_min: Tuple[int, int, int],
    grid_max: Tuple[int, int, int],
) -> str:
    draw_lines: List[str] = []
    clear_margin = 10
    clear_top_margin = 14
    for spec in specs:
        b = spec.bbox
        draw_lines.append(
            f'<DrawCuboid x1="{b["xmin"] - clear_margin}" y1="{max(2, b["ymin"] - 1)}" z1="{b["zmin"] - clear_margin}" '
            f'x2="{b["xmax"] + clear_margin}" y2="{b["ymax"] + clear_top_margin}" z2="{b["zmax"] + clear_margin}" type="air"/>'
        )
    for spec in specs:
        for x, y, z, block in spec.blocks:
            draw_lines.append(f'<DrawBlock x="{x}" y="{y}" z="{z}" type="{block}"/>')
    drawing_xml = "\n        ".join(draw_lines)

    gx0, gy0, gz0 = grid_min
    gx1, gy1, gz1 = grid_max

    mission_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>Multi building collection capture</Summary>
  </About>
  <ServerSection>
    <ServerInitialConditions>
      <Time>
        <StartTime>1000</StartTime>
        <AllowPassageOfTime>false</AllowPassageOfTime>
      </Time>
      <Weather>clear</Weather>
      <AllowSpawning>false</AllowSpawning>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2*3,2;1;"/>
      <DrawingDecorator>
        {drawing_xml}
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="14400000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
  <AgentSection mode="Creative">
    <Name>CaptureBot</Name>
    <AgentStart>
      <Placement x="{start_pose.x:.3f}" y="{start_pose.y:.3f}" z="{start_pose.z:.3f}" yaw="{start_pose.yaw:.3f}" pitch="{start_pose.pitch:.3f}"/>
    </AgentStart>
    <AgentHandlers>
      <AbsoluteMovementCommands/>
      <ObservationFromFullStats/>
      <ObservationFromGrid>
        <Grid name="{grid_name}">
          <min x="{gx0}" y="{gy0}" z="{gz0}"/>
          <max x="{gx1}" y="{gy1}" z="{gz1}"/>
        </Grid>
      </ObservationFromGrid>
      <VideoProducer want_depth="false">
        <Width>{image_w}</Width>
        <Height>{image_h}</Height>
      </VideoProducer>
      <MissionQuitCommands/>
    </AgentHandlers>
  </AgentSection>
</Mission>
"""
    return mission_xml


def build_filename(style_id: int, view_id: int, yaw: float, pitch: float) -> str:
    return f"rgb_style{style_id:03d}_view{view_id:02d}_yaw{yaw:+07.2f}_pitch{pitch:+07.2f}.png"


def generate_specs(count: int, start_style_id: int, seed: int, spacing: int) -> List[BuildingSpec]:
    grid_cols = math.ceil(math.sqrt(count))
    specs: List[BuildingSpec] = []
    for i in range(count):
        style_id = start_style_id + i
        gx = i % grid_cols
        gz = i // grid_cols
        origin = (gx * spacing, 4, gz * spacing)
        local_rng = random.Random(seed * 1000003 + style_id * 9167 + 17)
        spec = generate_building(local_rng, origin=origin, style_id=style_id)
        specs.append(spec)
    return specs


def make_anchor_pose(spec: BuildingSpec) -> Pose:
    bbox = spec.bbox
    cx = (bbox["xmin"] + bbox["xmax"]) / 2.0
    cz = (bbox["zmin"] + bbox["zmax"]) / 2.0
    y = max(4.0, float(bbox["ymin"] + 2))
    return Pose(x=cx + 0.5, y=y, z=cz + 0.5, yaw=0.0, pitch=0.0)


def capture_building_views(
    agent_host: Any,
    spec: BuildingSpec,
    out_dir: Path,
    views: int,
    image_w: int,
    image_h: int,
    fov: float,
    seed: int,
    grid_name: str,
    grid_min: Tuple[int, int, int],
    grid_max: Tuple[int, int, int],
    logger: Logger,
) -> Dict[str, Any]:
    out_images_dir = out_dir / "images"
    out_gt_dir = out_dir / "gt"
    out_logs_dir = out_dir / "logs"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_logs_dir.mkdir(parents=True, exist_ok=True)

    export_ground_truth(out_gt_dir, spec.bbox, spec.blocks)
    poses, radius, target = compute_view_poses(spec.bbox, views=views, fov=fov, image_w=image_w, image_h=image_h)
    logger.log(
        f"capturing style_id={spec.style_id} style={spec.style} palette={spec.palette_name} "
        f"views={views} radius={radius:.2f}"
    )

    view_records: List[Dict[str, Any]] = []
    # Preload nearby chunks before taking the first view for this building.
    cx = (spec.bbox["xmin"] + spec.bbox["xmax"] + 1) / 2.0
    cy = (spec.bbox["ymin"] + spec.bbox["ymax"] + 1) / 2.0
    cz = (spec.bbox["zmin"] + spec.bbox["zmax"] + 1) / 2.0
    preload_pose = Pose(x=cx, y=max(float(spec.bbox["ymin"] + 3), 7.0), z=cz, yaw=45.0, pitch=10.0)
    agent_host.sendCommand(f"tp {preload_pose.x:.3f} {preload_pose.y:.3f} {preload_pose.z:.3f}")
    agent_host.sendCommand(f"setYaw {preload_pose.yaw:.3f}")
    agent_host.sendCommand(f"setPitch {preload_pose.pitch:.3f}")
    wait_until_pose_reached(agent_host, preload_pose, timeout_sec=10.0)
    anchor_block = (
        int(math.floor(preload_pose.x)),
        int(math.floor(preload_pose.y)),
        int(math.floor(preload_pose.z)),
    )
    expected_in_grid = 0
    for x, y, z, _block in spec.blocks:
        rx = x - anchor_block[0]
        ry = y - anchor_block[1]
        rz = z - anchor_block[2]
        if grid_min[0] <= rx <= grid_max[0] and grid_min[1] <= ry <= grid_max[1] and grid_min[2] <= rz <= grid_max[2]:
            expected_in_grid += 1
    min_non_air = max(40, int(expected_in_grid * 0.75))
    stable_non_air_count = -1
    try:
        stable_non_air_count = wait_for_generation_stable(
            agent_host=agent_host,
            grid_name=grid_name,
            logger=logger,
            min_non_air_count=min_non_air,
            stable_k=3,
            sample_interval_sec=0.5,
            max_samples=80,
            max_seconds=45.0,
        )
        logger.log(
            f"style_id={spec.style_id} generation stable in local grid: "
            f"non_air={stable_non_air_count}, expected>={min_non_air}"
        )
    except CaptureError as exc:
        logger.log(
            f"style_id={spec.style_id} grid stability fallback: {exc}. "
            "Proceeding with video-frame stability check."
        )
    try:
        _ = wait_for_stable_video_frame(
            agent_host,
            timeout_sec=20.0,
            stable_k=3,
            diff_threshold=2.0,
            min_elapsed_sec=2.0,
            min_center_std=6.0,
            allow_timeout_fallback=True,
            hold_pose=preload_pose,
        )
    except CaptureError as exc:
        logger.log(
            f"style_id={spec.style_id} preload stable-frame fallback: {exc}. "
            "Continuing with latest frame stream."
        )
        _ = wait_for_frame(agent_host, timeout_sec=8.0)

    # Chunk warmup sweep before actual captures.
    warmup_count = min(4, max(1, len(poses) // 3))
    for warm_pose in poses[:warmup_count]:
        agent_host.sendCommand(f"tp {warm_pose.x:.3f} {warm_pose.y:.3f} {warm_pose.z:.3f}")
        agent_host.sendCommand(f"setYaw {warm_pose.yaw:.3f}")
        agent_host.sendCommand(f"setPitch {warm_pose.pitch:.3f}")
        wait_until_pose_reached(agent_host, warm_pose, timeout_sec=8.0)
        try:
            _ = wait_for_stable_video_frame(
                agent_host,
                timeout_sec=14.0,
                stable_k=3,
                diff_threshold=2.0,
                min_elapsed_sec=1.8,
                min_center_std=5.5,
                allow_timeout_fallback=True,
                hold_pose=warm_pose,
            )
        except CaptureError:
            pass

    for i, pose in enumerate(poses):
        frame = None
        last_exc: Exception | None = None
        for _attempt in range(4):
            try:
                agent_host.sendCommand(f"tp {pose.x:.3f} {pose.y:.3f} {pose.z:.3f}")
                agent_host.sendCommand(f"setYaw {pose.yaw:.3f}")
                agent_host.sendCommand(f"setPitch {pose.pitch:.3f}")
                wait_until_pose_reached(agent_host, pose, timeout_sec=10.0)
                frame = wait_for_stable_video_frame(
                    agent_host,
                    timeout_sec=18.0,
                    stable_k=2,
                    diff_threshold=4.0,
                    min_elapsed_sec=1.0,
                    min_center_std=4.0,
                    allow_timeout_fallback=True,
                    hold_pose=pose,
                )
                obs_pose = wait_for_observation(agent_host, timeout_sec=0.8)
                x_obs = obs_pose.get("XPos")
                y_obs = obs_pose.get("YPos")
                z_obs = obs_pose.get("ZPos")
                if not (isinstance(x_obs, (int, float)) and isinstance(y_obs, (int, float)) and isinstance(z_obs, (int, float))):
                    raise CaptureError("Pose check failed: missing XPos/YPos/ZPos.")
                if abs(float(x_obs) - pose.x) > 2.2 or abs(float(y_obs) - pose.y) > 2.2 or abs(float(z_obs) - pose.z) > 2.2:
                    raise CaptureError(
                        f"Pose drift too large before capture: "
                        f"target=({pose.x:.2f},{pose.y:.2f},{pose.z:.2f}) "
                        f"actual=({float(x_obs):.2f},{float(y_obs):.2f},{float(z_obs):.2f})"
                    )
                break
            except CaptureError as exc:
                last_exc = exc
                # Short detour to center tends to kick chunk mesh updates.
                agent_host.sendCommand(f"tp {preload_pose.x:.3f} {preload_pose.y:.3f} {preload_pose.z:.3f}")
                wait_until_pose_reached(agent_host, preload_pose, timeout_sec=6.0)
                time.sleep(0.4)
        if frame is None:
            if last_exc is not None:
                raise last_exc
            raise CaptureError(f"Failed to capture frame for style_id={spec.style_id}, view={i}.")
        image = frame_to_image(frame)
        filename = build_filename(spec.style_id, i, pose.yaw, pose.pitch)
        image_path = out_images_dir / filename
        image.save(image_path)

        record = {
            "path": str(Path("images") / filename),
            "x": round(pose.x, 4),
            "y": round(pose.y, 4),
            "z": round(pose.z, 4),
            "yaw": round(pose.yaw, 4),
            "pitch": round(pose.pitch, 4),
            "fov": float(fov),
            "width": int(image_w),
            "height": int(image_h),
        }
        try:
            obs = wait_for_observation(agent_host, timeout_sec=0.8)
            x_obs = obs.get("XPos")
            y_obs = obs.get("YPos")
            z_obs = obs.get("ZPos")
            if isinstance(x_obs, (int, float)) and isinstance(y_obs, (int, float)) and isinstance(z_obs, (int, float)):
                record["actual_x"] = round(float(x_obs), 4)
                record["actual_y"] = round(float(y_obs), 4)
                record["actual_z"] = round(float(z_obs), 4)
        except CaptureError:
            pass
        view_records.append(record)

    meta = {
        "seed": seed,
        "style_id": spec.style_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bbox": spec.bbox,
        "palette": spec.palette_name,
        "style": spec.style,
        "image_size": {"width": image_w, "height": image_h},
        "camera_planner": {
            "radius": round(radius, 4),
            "target": {"x": round(target[0], 4), "y": round(target[1], 4), "z": round(target[2], 4)},
            "fov": float(fov),
        },
        "generation": {
            "origin": {"x": spec.origin[0], "y": spec.origin[1], "z": spec.origin[2]},
            "num_blocks": len(spec.blocks),
            "stable_non_air_count": stable_non_air_count,
            "grid_expected_non_air_min": min_non_air,
        },
        "views": view_records,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def is_building_complete(building_dir: Path, expected_views: int) -> bool:
    meta_path = building_dir / "meta.json"
    voxels_path = building_dir / "gt" / "voxels.npy"
    bbox_path = building_dir / "gt" / "bbox.json"
    images_dir = building_dir / "images"
    if not (meta_path.exists() and voxels_path.exists() and bbox_path.exists() and images_dir.exists()):
        return False
    image_count = len(list(images_dir.glob("*.png")))
    if image_count < expected_views:
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return len(meta.get("views", [])) >= expected_views
    except Exception:
        return False


def run() -> int:
    args = parse_args()
    out_root = Path(args.out).expanduser().resolve()
    image_w, image_h = int(args.image_size[0]), int(args.image_size[1])
    out_root.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_root / "logs" / "capture_collection.log")

    try:
        specs = generate_specs(
            count=args.count,
            start_style_id=args.start_style_id,
            seed=args.seed,
            spacing=args.spacing,
        )
        logger.log(
            f"generated specs: count={len(specs)} start_style_id={args.start_style_id} "
            f"seed={args.seed} spacing={args.spacing}"
        )

        anchor_pose = make_anchor_pose(specs[0])
        grid_name = "local_building_grid"
        grid_min = (-12, -2, -12)
        grid_max = (12, 28, 12)
        mission_xml = build_collection_mission_xml(
            image_w=image_w,
            image_h=image_h,
            specs=specs,
            start_pose=anchor_pose,
            grid_name=grid_name,
            grid_min=grid_min,
            grid_max=grid_max,
        )

        MalmoPython = load_malmo()
        agent_host = MalmoPython.AgentHost()
        start_mission(MalmoPython, agent_host, mission_xml, port=args.port, logger=logger)
        wait_for_mission_begin(agent_host, timeout_sec=180.0, logger=logger)
        _ = wait_for_observation(agent_host, timeout_sec=45.0)
        _ = wait_for_frame(agent_host, timeout_sec=45.0)
        logger.log("initial observation and frame ready.")

        collection_records: List[Dict[str, Any]] = []
        for i, spec in enumerate(specs):
            building_dir = out_root / f"building_{spec.style_id:03d}"
            if args.skip_existing and is_building_complete(building_dir, expected_views=args.views):
                existing = json.loads((building_dir / "meta.json").read_text(encoding="utf-8"))
                logger.log(f"skipping existing style_id={spec.style_id} ({building_dir.name})")
                collection_records.append(
                    {
                        "index": i,
                        "style_id": spec.style_id,
                        "style": existing.get("style", spec.style),
                        "path": str(building_dir.relative_to(out_root)),
                        "palette": existing.get("palette", spec.palette_name),
                        "bbox": existing.get("bbox", spec.bbox),
                        "views": len(existing.get("views", [])),
                    }
                )
                continue
            meta = capture_building_views(
                agent_host=agent_host,
                spec=spec,
                out_dir=building_dir,
                views=args.views,
                image_w=image_w,
                image_h=image_h,
                fov=args.fov,
                seed=args.seed,
                grid_name=grid_name,
                grid_min=grid_min,
                grid_max=grid_max,
                logger=logger,
            )
            collection_records.append(
                {
                    "index": i,
                    "style_id": spec.style_id,
                    "style": spec.style,
                    "path": str(building_dir.relative_to(out_root)),
                    "palette": spec.palette_name,
                    "bbox": spec.bbox,
                    "views": len(meta.get("views", [])),
                }
            )

        collection_meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "count": args.count,
            "start_style_id": args.start_style_id,
            "views_per_building": args.views,
            "image_size": {"width": image_w, "height": image_h},
            "fov": args.fov,
            "spacing": args.spacing,
            "items": collection_records,
        }
        (out_root / "meta_collection.json").write_text(
            json.dumps(collection_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.log(f"collection meta saved: {out_root / 'meta_collection.json'}")

        # Graceful mission end.
        agent_host.sendCommand("quit")
        time.sleep(2.0)
        logger.log("collection capture completed.")
        return 0
    except CaptureError as exc:
        logger.log(f"ERROR: {exc}")
        print(f"[capture_building_collection] ERROR: {exc}", file=sys.stderr)
        print(f"[capture_building_collection] capture log: {out_root / 'logs' / 'capture_collection.log'}", file=sys.stderr)
        print("[capture_building_collection] also check Malmo client logs: ./logs/malmo_client.log", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.log(f"UNEXPECTED ERROR: {exc}")
        print(f"[capture_building_collection] UNEXPECTED ERROR: {exc}", file=sys.stderr)
        print(f"[capture_building_collection] capture log: {out_root / 'logs' / 'capture_collection.log'}", file=sys.stderr)
        print("[capture_building_collection] also check Malmo client logs: ./logs/malmo_client.log", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
