#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from building.generator import BuildingSpec, generate_building
from building.gt_export import export_ground_truth


class CaptureError(RuntimeError):
    pass


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one building with Malmo.")
    parser.add_argument("--out", required=True, help="Output directory path.")
    parser.add_argument("--port", type=int, default=10000, help="Malmo client port.")
    parser.add_argument("--views", type=int, default=12, help="Number of viewpoints.")
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=[960, 540],
        help="Image size: --image_size W H",
    )
    parser.add_argument("--fov", type=float, default=70.0, help="Requested FOV metadata and radius tuning hint.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--style_id", type=int, default=None, help="Building style id (0-99 recommended).")
    parser.add_argument(
        "--profile",
        choices=("complex", "simple"),
        default="complex",
        help="Building profile. Use 'simple' for lightweight/basic architecture.",
    )
    return parser.parse_args()


def cleanup_previous_outputs(out_images_dir: Path, out_gt_dir: Path, logger: Logger) -> None:
    removed = 0
    if out_images_dir.exists():
        for p in out_images_dir.glob("*.png"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    out_images_dir.mkdir(parents=True, exist_ok=True)

    if out_gt_dir.exists():
        for p in out_gt_dir.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    if removed > 0:
        logger.log(f"cleaned previous outputs: removed {removed} files")


def load_malmo():
    try:
        import MalmoPython  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local Malmo setup
        raise CaptureError(
            "MalmoPython の import に失敗しました。"
            " `source scripts/malmo_env_mac.sh` 実行後、"
            " `python3 -c \"import MalmoPython\"` が成功するか確認してください。"
        ) from exc
    return MalmoPython


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def as_float(obs: Dict[str, Any], key: str) -> Optional[float]:
    v = obs.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def look_at(camera: Tuple[float, float, float], target: Tuple[float, float, float]) -> Tuple[float, float]:
    dx = target[0] - camera[0]
    dy = target[1] - camera[1]
    dz = target[2] - camera[2]
    horiz = max(1e-6, math.hypot(dx, dz))
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = -math.degrees(math.atan2(dy, horiz))
    pitch = clamp(pitch, -80.0, 80.0)
    return yaw, pitch


def bbox_size(bbox: Dict[str, int]) -> Tuple[int, int, int]:
    width = bbox["xmax"] - bbox["xmin"] + 1
    height = bbox["ymax"] - bbox["ymin"] + 1
    depth = bbox["zmax"] - bbox["zmin"] + 1
    return width, height, depth


def _hfov_vfov_rad(fov_deg: float, image_w: int, image_h: int) -> Tuple[float, float]:
    hfov = math.radians(clamp(fov_deg, 45.0, 100.0))
    aspect = max(1e-6, float(image_h) / max(1.0, float(image_w)))
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * aspect)
    return hfov, vfov


def compute_capture_radius(
    bbox: Dict[str, int],
    fov: float,
    image_w: int,
    image_h: int,
    margin_blocks: float = 1.0,
) -> float:
    width, height, depth = bbox_size(bbox)
    half_planar = 0.5 * max(width, depth) + margin_blocks
    half_vertical = 0.55 * height + margin_blocks
    hfov, vfov = _hfov_vfov_rad(fov, image_w, image_h)
    h_allow = max(math.radians(10.0), hfov / 2.0 - math.radians(3.0))
    v_allow = max(math.radians(10.0), vfov / 2.0 - math.radians(2.0))

    r_from_width = half_planar / max(1e-6, math.tan(h_allow))
    r_from_height = half_vertical / max(1e-6, math.tan(v_allow))
    heuristic = max(width, depth) * 0.85 + 2.0
    return clamp(max(r_from_width, r_from_height, heuristic), 8.0, 72.0)


def compute_view_poses(
    bbox: Dict[str, int],
    views: int,
    fov: float,
    image_w: int = 960,
    image_h: int = 540,
) -> Tuple[List[Pose], float, Tuple[float, float, float]]:
    views = max(1, views)
    width, height, depth = bbox_size(bbox)
    _ = width, depth  # kept for readability when tuning formulas
    cx = (bbox["xmin"] + bbox["xmax"] + 1) / 2.0
    cy = (bbox["ymin"] + bbox["ymax"] + 1) / 2.0
    cz = (bbox["zmin"] + bbox["zmax"] + 1) / 2.0
    target = (cx, cy, cz)
    low_y = max(float(bbox["ymin"]) + height * 0.52, float(bbox["ymin"]) + 6.0)
    high_y = max(float(bbox["ymin"]) + height * 0.74, float(bbox["ymin"]) + 10.0)
    ring_levels = (low_y, high_y)
    topdown_views = 0
    if views >= 10:
        topdown_views = 2
    elif views >= 5:
        topdown_views = 1
    topdown_views = min(topdown_views, max(0, views - 3))
    ring_views = max(1, views - topdown_views)
    radius = compute_capture_radius(
        bbox=bbox,
        fov=fov,
        image_w=image_w,
        image_h=image_h,
        margin_blocks=1.0,
    )

    poses: List[Pose] = []
    for i in range(ring_views):
        theta = (2.0 * math.pi * i) / ring_views
        x = cx + radius * math.cos(theta)
        z = cz + radius * math.sin(theta)
        y = ring_levels[i % len(ring_levels)]
        yaw, pitch = look_at((x, y, z), target)
        poses.append(Pose(x=x, y=y, z=z, yaw=yaw, pitch=pitch))

    if topdown_views > 0:
        top_radius = clamp(max(width, depth) * 0.22, 4.0, 14.0)
        top_y_base = max(float(bbox["ymax"]) + height * 1.1 + 12.0, cy + 26.0)
        top_pitch_target_deg = 80.0
        top_pitch_target_rad = math.radians(top_pitch_target_deg)
        for j in range(topdown_views):
            theta = (2.0 * math.pi * j) / topdown_views
            x = cx + top_radius * math.cos(theta)
            z = cz + top_radius * math.sin(theta)
            horiz = max(1e-6, math.hypot(x - cx, z - cz))
            target_y = cy + math.tan(top_pitch_target_rad) * horiz
            y = max(top_y_base + j * 1.8, target_y)
            yaw, pitch = look_at((x, y, z), target)
            pitch = clamp(max(pitch, 78.0), -89.0, 89.0)
            poses.append(Pose(x=x, y=y, z=z, yaw=yaw, pitch=pitch))
    return poses, radius, target


def build_mission_xml(
    image_w: int,
    image_h: int,
    blocks: List[Tuple[int, int, int, str]],
    bbox: Dict[str, int],
    grid_name: str,
    grid_min: Tuple[int, int, int],
    grid_max: Tuple[int, int, int],
    start_pose: Pose,
) -> str:
    clear_margin = 8
    clear_top_margin = 12
    global_clear_radius = 192
    global_clear_top = 200
    draw_lines: List[str] = [
        (
            f'<DrawCuboid x1="{-global_clear_radius}" y1="3" z1="{-global_clear_radius}" '
            f'x2="{global_clear_radius}" y2="{global_clear_top}" z2="{global_clear_radius}" type="air"/>'
        ),
        (
            f'<DrawCuboid x1="{-global_clear_radius}" y1="3" z1="{-global_clear_radius}" '
            f'x2="{global_clear_radius}" y2="3" z2="{global_clear_radius}" type="grass"/>'
        ),
        (
            f'<DrawCuboid x1="{bbox["xmin"] - clear_margin}" y1="{max(2, bbox["ymin"] - 1)}" z1="{bbox["zmin"] - clear_margin}" '
            f'x2="{bbox["xmax"] + clear_margin}" y2="{bbox["ymax"] + clear_top_margin}" z2="{bbox["zmax"] + clear_margin}" type="air"/>'
        )
    ]
    for x, y, z, block in blocks:
        draw_lines.append(f'<DrawBlock x="{x}" y="{y}" z="{z}" type="{block}"/>')
    drawing_xml = "\n        ".join(draw_lines)

    gx0, gy0, gz0 = grid_min
    gx1, gy1, gz1 = grid_max

    mission_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>One building multi-view capture</Summary>
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
      <ServerQuitFromTimeUp timeLimitMs="900000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
  <AgentSection mode="Spectator">
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


def start_mission(MalmoPython: Any, agent_host: Any, mission_xml: str, port: int, logger: Logger) -> None:
    mission_spec = MalmoPython.MissionSpec(mission_xml, True)
    mission_record = MalmoPython.MissionRecordSpec()
    client_pool = MalmoPython.ClientPool()
    client_pool.add(MalmoPython.ClientInfo("127.0.0.1", port))

    last_exc: Optional[Exception] = None
    experiment_id = f"one-building-{uuid.uuid4()}"
    max_attempts = 30
    for attempt in range(1, max_attempts + 1):
        try:
            agent_host.startMission(mission_spec, client_pool, mission_record, 0, experiment_id)
            logger.log(
                f"startMission succeeded (attempt {attempt}/{max_attempts}, "
                f"experiment_id={experiment_id})."
            )
            return
        except RuntimeError as exc:
            last_exc = exc
            msg = str(exc)
            retry_wait = 3.0
            if (
                "Failed to find an available client for this mission" in msg
                or "All ports in range were busy" in msg
            ):
                # Client can be LISTEN but not yet fully mission-accepting.
                retry_wait = min(12.0, 3.0 + attempt * 0.8)
            logger.log(
                f"startMission failed (attempt {attempt}/{max_attempts}): {msg} "
                f"(retry in {retry_wait:.1f}s)"
            )
            time.sleep(retry_wait)

    raise CaptureError(
        "startMission に失敗しました。"
        " Malmo クライアントが :10000 で待受しているか、"
        " `logs/malmo_client.log` を確認してください。"
    ) from last_exc


def _check_world_state_errors(world_state: Any) -> None:
    if world_state.errors:
        joined = "; ".join(err.text for err in world_state.errors)
        raise CaptureError(f"Malmo world state error: {joined}")


def wait_for_mission_begin(agent_host: Any, timeout_sec: float, logger: Logger) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        world_state = agent_host.getWorldState()
        _check_world_state_errors(world_state)
        if world_state.has_mission_begun:
            logger.log("Mission has begun.")
            return
        time.sleep(0.1)
    raise CaptureError("Mission start timeout. `logs/malmo_client.log` を確認してください。")


def wait_for_observation(agent_host: Any, timeout_sec: float) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    latest: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        world_state = agent_host.getWorldState()
        _check_world_state_errors(world_state)
        if world_state.observations:
            for obs in world_state.observations:
                if not obs.text:
                    continue
                try:
                    latest = json.loads(obs.text)
                except json.JSONDecodeError:
                    continue
            if latest is not None:
                return latest
        if not world_state.is_mission_running and world_state.has_mission_begun:
            break
        time.sleep(0.05)
    if latest is not None:
        return latest
    raise CaptureError("Observation timeout.")


def wait_for_frame(agent_host: Any, timeout_sec: float) -> Any:
    deadline = time.time() + timeout_sec
    last_frame = None
    while time.time() < deadline:
        world_state = agent_host.getWorldState()
        _check_world_state_errors(world_state)
        if world_state.video_frames:
            last_frame = world_state.video_frames[-1]
            return last_frame
        if not world_state.is_mission_running and world_state.has_mission_begun:
            break
        time.sleep(0.05)
    raise CaptureError("Video frame timeout.")


def _frame_rgb_flat(frame: Any) -> np.ndarray:
    raw = bytes(frame.pixels)
    w = int(frame.width)
    h = int(frame.height)
    rgb_size = w * h * 3
    rgba_size = w * h * 4
    if len(raw) == rgb_size:
        return np.frombuffer(raw, dtype=np.uint8)
    if len(raw) == rgba_size:
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape((h * w, 4))
        return rgba[:, :3].reshape(-1)
    raise CaptureError(
        f"Unexpected frame size: {len(raw)} bytes for {w}x{h}. "
        f"Expected {rgb_size} or {rgba_size}."
    )


def wait_for_stable_video_frame(
    agent_host: Any,
    timeout_sec: float = 12.0,
    stable_k: int = 3,
    diff_threshold: float = 2.0,
    min_elapsed_sec: float = 1.2,
    min_center_std: float = 10.0,
    allow_timeout_fallback: bool = False,
    hold_pose: Optional[Pose] = None,
    hold_interval_sec: float = 0.30,
) -> Any:
    """
    Wait until consecutive frames are visually stable.
    This reduces captures taken before nearby chunks/meshes settle after teleport.
    """
    start_time = time.time()
    deadline = start_time + timeout_sec
    last_vec: Optional[np.ndarray] = None
    last_frame = None
    streak = 0
    best_center_std = 0.0
    best_diff = float("inf")
    next_hold_time = 0.0

    def center_std(vec: np.ndarray, w: int, h: int) -> float:
        rgb = vec.reshape((h, w, 3)).astype(np.float32)
        y0, y1 = h // 4, h - h // 4
        x0, x1 = w // 4, w - w // 4
        patch = rgb[y0:y1, x0:x1, :]
        gray = patch.mean(axis=2)
        return float(gray.std())

    while time.time() < deadline:
        now = time.time()
        if hold_pose is not None and now >= next_hold_time:
            agent_host.sendCommand(f"tp {hold_pose.x:.3f} {hold_pose.y:.3f} {hold_pose.z:.3f}")
            agent_host.sendCommand(f"setYaw {hold_pose.yaw:.3f}")
            agent_host.sendCommand(f"setPitch {hold_pose.pitch:.3f}")
            next_hold_time = now + max(0.05, hold_interval_sec)
        frame = wait_for_frame(agent_host, timeout_sec=2.0)
        vec = _frame_rgb_flat(frame)
        cstd = center_std(vec, int(frame.width), int(frame.height))
        best_center_std = max(best_center_std, cstd)
        quality_ok = cstd >= min_center_std
        if last_vec is None or vec.shape != last_vec.shape:
            streak = 1 if quality_ok else 0
        else:
            diff = float(np.mean(np.abs(vec.astype(np.int16) - last_vec.astype(np.int16))))
            best_diff = min(best_diff, diff)
            if diff <= diff_threshold and quality_ok and (time.time() - start_time) >= min_elapsed_sec:
                streak += 1
            else:
                streak = 1 if quality_ok else 0
        last_vec = vec
        last_frame = frame
        if streak >= stable_k:
            return frame

    if (
        allow_timeout_fallback
        and last_frame is not None
        and best_center_std >= min_center_std * 0.8
        and best_diff <= diff_threshold * 1.5
    ):
        return last_frame
    raise CaptureError("Stable video frame timeout.")


def count_non_air_blocks(obs: Dict[str, Any], grid_name: str) -> Optional[int]:
    blocks = obs.get(grid_name)
    if not isinstance(blocks, list):
        return None
    count = 0
    for block in blocks:
        if isinstance(block, str) and block != "air":
            count += 1
    return count


def wait_for_generation_stable(
    agent_host: Any,
    grid_name: str,
    logger: Logger,
    min_non_air_count: Optional[int] = None,
    stable_k: int = 3,
    sample_interval_sec: float = 0.5,
    max_samples: int = 120,
    max_seconds: float = 60.0,
) -> int:
    """Wait until non-air count inside bbox-grid is unchanged for K consecutive samples."""
    deadline = time.time() + max_seconds
    last_count: Optional[int] = None
    streak = 0
    samples = 0

    while time.time() < deadline and samples < max_samples:
        try:
            obs = wait_for_observation(agent_host, timeout_sec=6.0)
        except CaptureError:
            logger.log("stability sample: observation not ready yet; retrying.")
            time.sleep(sample_interval_sec)
            continue
        count = count_non_air_blocks(obs, grid_name)
        if count is None:
            logger.log("stability sample: grid not found yet; retrying.")
            time.sleep(sample_interval_sec)
            continue

        samples += 1
        if min_non_air_count is not None and count < min_non_air_count:
            logger.log(
                f"stability sample={samples}, non_air={count} "
                f"(waiting for >= {min_non_air_count})"
            )
            last_count = count
            streak = 0
            time.sleep(sample_interval_sec)
            continue

        if last_count == count:
            streak += 1
        else:
            streak = 1
            last_count = count

        logger.log(f"stability sample={samples}, non_air={count}, streak={streak}/{stable_k}")
        if streak >= stable_k:
            logger.log(f"Generation stable: non-air count fixed at {count}.")
            return count

        time.sleep(sample_interval_sec)

    raise CaptureError(
        "建築生成の安定判定に失敗しました。"
        " 生成が完了していない可能性があります。"
    )


def expected_min_non_air_count(num_blocks: int) -> int:
    """
    Conservative lower-bound for completion checks.
    Some Malmo grid observations can be slightly lower than the authored block count
    (eg. placement quantization / edge sampling), so we avoid exact-equality gating.
    """
    n = max(1, int(num_blocks))
    if n <= 256:
        return max(1, int(n * 0.90))
    if n <= 2048:
        return max(1, int(n * 0.93))
    return max(1, int(n * 0.95))


def wait_until_pose_reached(agent_host: Any, pose: Pose, timeout_sec: float = 4.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_obs: Dict[str, Any] = {}
    while time.time() < deadline:
        try:
            obs = wait_for_observation(agent_host, timeout_sec=0.6)
        except CaptureError:
            continue
        last_obs = obs
        x = as_float(obs, "XPos")
        y = as_float(obs, "YPos")
        z = as_float(obs, "ZPos")
        yaw = as_float(obs, "Yaw")
        pitch = as_float(obs, "Pitch")
        if None in (x, y, z, yaw, pitch):
            continue
        assert x is not None and y is not None and z is not None and yaw is not None and pitch is not None
        if (
            abs(x - pose.x) <= 1.0
            and abs(y - pose.y) <= 1.0
            and abs(z - pose.z) <= 1.0
            and angle_diff_deg(yaw, pose.yaw) <= 10.0
            and abs(pitch - pose.pitch) <= 10.0
        ):
            return obs
    return last_obs


def frame_to_image(frame: Any) -> Image.Image:
    raw = bytes(frame.pixels)
    rgb_size = frame.width * frame.height * 3
    rgba_size = frame.width * frame.height * 4

    if len(raw) == rgb_size:
        return Image.frombytes("RGB", (frame.width, frame.height), raw)
    if len(raw) == rgba_size:
        return Image.frombytes("RGBA", (frame.width, frame.height), raw).convert("RGB")
    raise CaptureError(
        f"Unexpected frame size: {len(raw)} bytes for {frame.width}x{frame.height}. "
        f"Expected {rgb_size} or {rgba_size}."
    )


def build_filename(view_id: int, yaw: float, pitch: float) -> str:
    return f"rgb_view{view_id:02d}_yaw{yaw:+07.2f}_pitch{pitch:+07.2f}.png"


def prewarm_viewpoints(agent_host: Any, poses: List[Pose], logger: Logger) -> None:
    """
    Visit each planned viewpoint once before real capture.
    This helps chunk/mesh loading settle and reduces cut-off building frames.
    """
    if not poses:
        return
    logger.log(f"Prewarming {len(poses)} viewpoints for chunk/render readiness.")
    for i, pose in enumerate(poses):
        logger.log(
            f"Prewarm view {i + 1}/{len(poses)} at "
            f"x={pose.x:.2f}, y={pose.y:.2f}, z={pose.z:.2f}, yaw={pose.yaw:.2f}, pitch={pose.pitch:.2f}"
        )
        ok = False
        for _attempt in range(2):
            try:
                agent_host.sendCommand(f"tp {pose.x:.3f} {pose.y:.3f} {pose.z:.3f}")
                agent_host.sendCommand(f"setYaw {pose.yaw:.3f}")
                agent_host.sendCommand(f"setPitch {pose.pitch:.3f}")
                wait_until_pose_reached(agent_host, pose, timeout_sec=5.0)
                wait_for_stable_video_frame(
                    agent_host,
                    timeout_sec=7.0,
                    stable_k=2,
                    diff_threshold=5.0,
                    min_elapsed_sec=0.7,
                    min_center_std=1.5,
                    allow_timeout_fallback=True,
                    hold_pose=pose,
                    hold_interval_sec=0.65,
                )
                ok = True
                break
            except CaptureError:
                time.sleep(0.25)
        if not ok:
            raise CaptureError(f"Prewarm failed at view {i + 1}/{len(poses)}.")


def capture_views(
    agent_host: Any,
    out_images_dir: Path,
    poses: List[Pose],
    fov: float,
    image_w: int,
    image_h: int,
    logger: Logger,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    out_images_dir.mkdir(parents=True, exist_ok=True)

    def ensure_pose_lock(pose: Pose) -> Dict[str, Any]:
        obs = wait_for_observation(agent_host, timeout_sec=0.8)
        ax = as_float(obs, "XPos")
        ay = as_float(obs, "YPos")
        az = as_float(obs, "ZPos")
        ayaw = as_float(obs, "Yaw")
        apitch = as_float(obs, "Pitch")
        if None in (ax, ay, az):
            raise CaptureError("Pose check failed: missing XPos/YPos/ZPos.")
        if ayaw is None or apitch is None:
            raise CaptureError("Pose check failed: missing Yaw/Pitch.")
        assert ax is not None and ay is not None and az is not None and ayaw is not None and apitch is not None
        if (
            abs(ax - pose.x) > 1.6
            or abs(ay - pose.y) > 1.0
            or abs(az - pose.z) > 1.6
            or angle_diff_deg(ayaw, pose.yaw) > 8.0
            or abs(apitch - pose.pitch) > 8.0
        ):
            raise CaptureError(
                f"Pose drift too large before capture: "
                f"target=({pose.x:.2f},{pose.y:.2f},{pose.z:.2f}) "
                f"actual=({ax:.2f},{ay:.2f},{az:.2f}) "
                f"target_yaw_pitch=({pose.yaw:.2f},{pose.pitch:.2f}) "
                f"actual_yaw_pitch=({ayaw:.2f},{apitch:.2f})"
            )
        return obs

    for i, pose in enumerate(poses):
        logger.log(
            f"Capturing view {i + 1}/{len(poses)} at "
            f"x={pose.x:.2f}, y={pose.y:.2f}, z={pose.z:.2f}, yaw={pose.yaw:.2f}, pitch={pose.pitch:.2f}"
        )
        frame = None
        obs_for_record: Optional[Dict[str, Any]] = None
        last_exc: Optional[Exception] = None
        for _attempt in range(4):
            try:
                agent_host.sendCommand(f"tp {pose.x:.3f} {pose.y:.3f} {pose.z:.3f}")
                agent_host.sendCommand(f"setYaw {pose.yaw:.3f}")
                agent_host.sendCommand(f"setPitch {pose.pitch:.3f}")
                wait_until_pose_reached(agent_host, pose, timeout_sec=8.0)
                frame = wait_for_stable_video_frame(
                    agent_host,
                    timeout_sec=12.0,
                    stable_k=2,
                    diff_threshold=3.5,
                    min_elapsed_sec=1.0,
                    min_center_std=3.0,
                    allow_timeout_fallback=True,
                    hold_pose=pose,
                    hold_interval_sec=0.55,
                )
                obs_for_record = ensure_pose_lock(pose)
                break
            except CaptureError as exc:
                last_exc = exc
                time.sleep(0.35)
        if frame is None:
            if last_exc is not None:
                raise last_exc
            raise CaptureError(f"Failed to capture frame for view {i}.")

        image = frame_to_image(frame)
        filename = build_filename(i, pose.yaw, pose.pitch)
        output_path = out_images_dir / filename
        image.save(output_path)

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
        if obs_for_record is not None:
            ax = as_float(obs_for_record, "XPos")
            ay = as_float(obs_for_record, "YPos")
            az = as_float(obs_for_record, "ZPos")
            if None not in (ax, ay, az):
                record["actual_x"] = round(float(ax), 4)
                record["actual_y"] = round(float(ay), 4)
                record["actual_z"] = round(float(az), 4)
        records.append(record)

    return records


def make_anchor_pose(spec: BuildingSpec) -> Pose:
    bbox = spec.bbox
    cx = (bbox["xmin"] + bbox["xmax"]) / 2.0
    cz = (bbox["zmin"] + bbox["zmax"]) / 2.0
    y = max(4.0, float(bbox["ymin"] + 2))
    return Pose(x=cx + 0.5, y=y, z=cz + 0.5, yaw=0.0, pitch=0.0)


def build_meta(
    spec: BuildingSpec,
    seed: int,
    profile: str,
    image_w: int,
    image_h: int,
    fov: float,
    radius: float,
    stable_non_air_count: int,
    target: Tuple[float, float, float],
    views: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "seed": seed,
        "profile": profile,
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
        },
        "views": views,
    }


def run() -> int:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    image_w, image_h = int(args.image_size[0]), int(args.image_size[1])
    out_images_dir = out_dir / "images"
    out_gt_dir = out_dir / "gt"
    out_logs_dir = out_dir / "logs"
    out_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_logs_dir / "capture.log"
    logger = Logger(log_path)
    agent_host: Optional[Any] = None

    try:
        cleanup_previous_outputs(out_images_dir=out_images_dir, out_gt_dir=out_gt_dir, logger=logger)
        rng = random.Random(args.seed)
        logger.log(f"seed={args.seed}")
        spec = generate_building(rng, origin=(0, 4, 0), style_id=args.style_id, profile=args.profile)
        logger.log(
            "building generated: "
            f"profile={args.profile}, style={spec.style}, style_id={spec.style_id}, palette={spec.palette_name}, "
            f"blocks={len(spec.blocks)}, bbox={spec.bbox}"
        )
        export_ground_truth(out_gt_dir, spec.bbox, spec.blocks)
        logger.log(f"ground truth exported to {out_gt_dir}")

        poses, radius, target = compute_view_poses(
            spec.bbox,
            views=args.views,
            fov=args.fov,
            image_w=image_w,
            image_h=image_h,
        )
        logger.log(f"computed {len(poses)} camera poses with radius={radius:.2f}")

        anchor_pose = make_anchor_pose(spec)
        anchor_block = (int(math.floor(anchor_pose.x)), int(math.floor(anchor_pose.y)), int(math.floor(anchor_pose.z)))
        grid_name = "build_grid"
        grid_min = (
            spec.bbox["xmin"] - anchor_block[0],
            spec.bbox["ymin"] - anchor_block[1],
            spec.bbox["zmin"] - anchor_block[2],
        )
        grid_max = (
            spec.bbox["xmax"] - anchor_block[0],
            spec.bbox["ymax"] - anchor_block[1],
            spec.bbox["zmax"] - anchor_block[2],
        )

        mission_xml = build_mission_xml(
            image_w=image_w,
            image_h=image_h,
            blocks=spec.blocks,
            bbox=spec.bbox,
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
        min_non_air_count = expected_min_non_air_count(len(spec.blocks))
        logger.log(
            "generation stability threshold: "
            f"min_non_air_count={min_non_air_count} / authored_blocks={len(spec.blocks)}"
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
            spec=spec,
            seed=args.seed,
            profile=args.profile,
            image_w=image_w,
            image_h=image_h,
            fov=args.fov,
            radius=radius,
            stable_non_air_count=stable_count,
            target=target,
            views=view_records,
        )
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.log(f"meta saved: {meta_path}")

        agent_host.sendCommand("quit")
        logger.log("capture completed.")
        return 0
    except CaptureError as exc:
        logger.log(f"ERROR: {exc}")
        print(f"[capture_one_building] ERROR: {exc}", file=sys.stderr)
        print(f"[capture_one_building] capture log: {log_path}", file=sys.stderr)
        print("[capture_one_building] also check Malmo client logs: ./logs/malmo_client.log", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.log(f"UNEXPECTED ERROR: {exc}")
        print(f"[capture_one_building] UNEXPECTED ERROR: {exc}", file=sys.stderr)
        print(f"[capture_one_building] capture log: {log_path}", file=sys.stderr)
        print("[capture_one_building] also check Malmo client logs: ./logs/malmo_client.log", file=sys.stderr)
        return 1
    finally:
        if agent_host is not None:
            try:
                agent_host.sendCommand("quit")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(run())
