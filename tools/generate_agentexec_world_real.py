#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.capture_one_building import (  # noqa: E402
    CaptureError,
    Logger,
    Pose,
    load_malmo,
    start_mission,
    wait_for_generation_stable,
    wait_for_mission_begin,
    wait_for_observation,
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
    "brick": "brick_block",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate real agent-executed rebuild worlds by starting Malmo mission, "
            "executing placement commands, and re-reading final grid voxels."
        )
    )
    parser.add_argument("--dataset_root", required=True, help="Root containing building_xxx dirs (usually outputs/i2t2b/<dataset>).")
    parser.add_argument("--source_subdir", required=True, help="Source rebuild world subdir name.")
    parser.add_argument("--out_subdir", required=True, help="Destination real agent-exec subdir name.")
    parser.add_argument("--port", type=int, default=10000, help="Malmo client port.")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern.")
    parser.add_argument("--limit", type=int, default=0, help="Max buildings (0=all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--placement_mode",
        choices=["chat_commands", "hand_place"],
        default="chat_commands",
        help="chat_commands: /setblock,/fill via chat. hand_place: creative hand-placement via use.",
    )
    parser.add_argument("--command_interval_sec", type=float, default=0.04, help="Pause between chat commands.")
    parser.add_argument("--post_command_wait_sec", type=float, default=1.2, help="Wait after all commands sent.")
    parser.add_argument("--max_operations", type=int, default=5000, help="Safety cap for operations.")
    parser.add_argument("--hand_place_hotbar_slot", type=int, default=0, help="0-8 slot index for hand-place mode.")
    parser.add_argument("--hand_place_use_pulse_sec", type=float, default=0.06, help="Duration of use=1 pulse.")
    parser.add_argument(
        "--hand_place_tp_height_offset",
        type=float,
        default=2.2,
        help="Feet-height offset above target block during hand-place teleport.",
    )
    parser.add_argument(
        "--hand_place_max_passes",
        type=int,
        default=3,
        help="Max support-order retry passes for deferred hand-place blocks.",
    )
    parser.add_argument(
        "--min_non_air_ratio",
        type=float,
        default=0.0,
        help=(
            "Minimum observed non-air ratio against source_non_air for stability gating. "
            "0 disables hard minimum (recommended for real placement)."
        ),
    )
    parser.add_argument("--stability_k", type=int, default=3, help="Consecutive equal non-air samples for stable verdict.")
    parser.add_argument("--stability_interval_sec", type=float, default=0.5, help="Sampling interval for stability check.")
    parser.add_argument("--stability_max_samples", type=int, default=160, help="Maximum stability samples.")
    parser.add_argument("--stability_max_seconds", type=float, default=90.0, help="Maximum seconds for stability wait.")
    parser.add_argument(
        "--mission_quit_wait_sec",
        type=float,
        default=8.0,
        help="How long to wait for mission shutdown after sending quit.",
    )
    parser.add_argument(
        "--inter_building_cooldown_sec",
        type=float,
        default=1.2,
        help="Cooldown between buildings to avoid client-pool busy states.",
    )
    return parser.parse_args()


def _append_candidate_malmo_paths() -> None:
    candidates: List[Path] = []
    malmo_dir = Path(os.environ.get("MALMO_DIR", "")).expanduser().resolve() if os.environ.get("MALMO_DIR") else None
    if malmo_dir and malmo_dir.is_dir():
        candidates.extend(
            [
                malmo_dir / "build" / "install" / "Python_Examples",
                malmo_dir / "build" / "Malmo" / "src" / "PythonWrapper",
                malmo_dir / "Python_Examples",
            ]
        )
    candidates.extend(
        [
            ROOT / "MalmoPlatform" / "build" / "install" / "Python_Examples",
            ROOT / "MalmoPlatform" / "build" / "Malmo" / "src" / "PythonWrapper",
            ROOT / "MalmoPlatform" / "Python_Examples",
        ]
    )
    for c in candidates:
        if c.is_dir():
            s = str(c)
            if s not in sys.path:
                sys.path.insert(0, s)


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


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _list_buildings(root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _load_bbox(path: Path) -> Dict[str, int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {
        "xmin": _int(obj.get("xmin", 0), 0),
        "xmax": _int(obj.get("xmax", 0), 0),
        "ymin": _int(obj.get("ymin", 0), 0),
        "ymax": _int(obj.get("ymax", 0), 0),
        "zmin": _int(obj.get("zmin", 0), 0),
        "zmax": _int(obj.get("zmax", 0), 0),
    }


def _bbox_dims(bbox: Dict[str, int]) -> Tuple[int, int, int]:
    bx = int(bbox["xmax"] - bbox["xmin"] + 1)
    by = int(bbox["ymax"] - bbox["ymin"] + 1)
    bz = int(bbox["zmax"] - bbox["zmin"] + 1)
    return bx, by, bz


def _make_anchor_pose(bbox: Dict[str, int]) -> Pose:
    cx = (bbox["xmin"] + bbox["xmax"]) / 2.0 + 0.5
    cz = (bbox["zmin"] + bbox["zmax"]) / 2.0 + 0.5
    y = max(float(bbox["ymax"] + 10), 12.0)
    return Pose(x=cx, y=y, z=cz, yaw=0.0, pitch=50.0)


def _build_agentexec_mission_xml(
    bbox: Dict[str, int],
    grid_name: str,
    grid_min: Tuple[int, int, int],
    grid_max: Tuple[int, int, int],
    start_pose: Pose,
) -> str:
    clear_margin = 12
    clear_top_margin = 18
    global_clear_radius = 192
    global_clear_top = 200
    draw_lines = [
        (
            f'<DrawCuboid x1="{-global_clear_radius}" y1="3" z1="{-global_clear_radius}" '
            f'x2="{global_clear_radius}" y2="{global_clear_top}" z2="{global_clear_radius}" type="air"/>'
        ),
        (
            f'<DrawCuboid x1="{-global_clear_radius}" y1="3" z1="{-global_clear_radius}" '
            f'x2="{global_clear_radius}" y2="3" z2="{global_clear_radius}" type="grass"/>'
        ),
        (
            f'<DrawCuboid x1="{bbox["xmin"] - clear_margin}" y1="{max(2, bbox["ymin"] - 2)}" z1="{bbox["zmin"] - clear_margin}" '
            f'x2="{bbox["xmax"] + clear_margin}" y2="{bbox["ymax"] + clear_top_margin}" z2="{bbox["zmax"] + clear_margin}" type="air"/>'
        ),
        (
            f'<DrawCuboid x1="{bbox["xmin"] - 2}" y1="{max(2, bbox["ymin"] - 1)}" z1="{bbox["zmin"] - 2}" '
            f'x2="{bbox["xmax"] + 2}" y2="{max(2, bbox["ymin"] - 1)}" z2="{bbox["zmax"] + 2}" type="stone"/>'
        ),
    ]
    drawing_xml = "\n        ".join(draw_lines)
    gx0, gy0, gz0 = grid_min
    gx1, gy1, gz1 = grid_max

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>Agent execution rebuild capture</Summary>
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
    <AgentSection mode="Creative">
    <Name>BuilderBot</Name>
    <AgentStart>
      <Placement x="{start_pose.x:.3f}" y="{start_pose.y:.3f}" z="{start_pose.z:.3f}" yaw="{start_pose.yaw:.3f}" pitch="{start_pose.pitch:.3f}"/>
    </AgentStart>
    <AgentHandlers>
      <AbsoluteMovementCommands/>
      <ContinuousMovementCommands/>
      <InventoryCommands/>
      <ChatCommands/>
      <ObservationFromFullStats/>
      <ObservationFromGrid>
        <Grid name="{grid_name}">
          <min x="{gx0}" y="{gy0}" z="{gz0}"/>
          <max x="{gx1}" y="{gy1}" z="{gz1}"/>
        </Grid>
      </ObservationFromGrid>
      <MissionQuitCommands/>
    </AgentHandlers>
  </AgentSection>
</Mission>
"""


def _sort_bounds(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> Tuple[int, int, int, int, int, int]:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if z2 < z1:
        z1, z2 = z2, z1
    return x1, y1, z1, x2, y2, z2


def _load_ops_from_actions(actions_path: Path, max_operations: int) -> List[Dict[str, Any]]:
    if not actions_path.is_file():
        return []
    obj = json.loads(actions_path.read_text(encoding="utf-8"))
    ops = obj.get("operations", [])
    out: List[Dict[str, Any]] = []
    if isinstance(ops, list):
        for op in ops:
            if isinstance(op, dict):
                out.append(dict(op))
            if len(out) >= max_operations:
                break
    return out


def _ops_from_voxels(voxels: np.ndarray, bbox: Dict[str, int], max_operations: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    xmin, ymin, zmin = bbox["xmin"], bbox["ymin"], bbox["zmin"]
    sy, sx, sz = voxels.shape
    for yi in range(sy):
        for xi in range(sx):
            for zi in range(sz):
                raw = str(voxels[yi, xi, zi])
                b = sanitize_block_name(raw)
                if b == "air":
                    continue
                out.append(
                    {
                        "op": "set",
                        "x": xmin + xi,
                        "y": ymin + yi,
                        "z": zmin + zi,
                        "block": b,
                        "source": "voxels_fallback",
                    }
                )
                if len(out) >= max_operations:
                    return out
    return out


def _command_from_op(op: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    kind = str(op.get("op", "")).strip().lower()
    if kind == "carve":
        x1 = _int(op.get("x1", 0), 0)
        y1 = _int(op.get("y1", 0), 0)
        z1 = _int(op.get("z1", 0), 0)
        x2 = _int(op.get("x2", x1), x1)
        y2 = _int(op.get("y2", y1), y1)
        z2 = _int(op.get("z2", z1), z1)
        x1, y1, z1, x2, y2, z2 = _sort_bounds(x1, y1, z1, x2, y2, z2)
        cmd = f"/fill {x1} {y1} {z1} {x2} {y2} {z2} air"
        rec = {"op": "carve", "x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2, "block": "air"}
        return cmd, rec

    if kind == "fill":
        x1 = _int(op.get("x1", 0), 0)
        y1 = _int(op.get("y1", 0), 0)
        z1 = _int(op.get("z1", 0), 0)
        x2 = _int(op.get("x2", x1), x1)
        y2 = _int(op.get("y2", y1), y1)
        z2 = _int(op.get("z2", z1), z1)
        x1, y1, z1, x2, y2, z2 = _sort_bounds(x1, y1, z1, x2, y2, z2)
        block = sanitize_block_name(op.get("block", "air"))
        cmd = f"/fill {x1} {y1} {z1} {x2} {y2} {z2} {block}"
        rec = {"op": "fill", "x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2, "block": block}
        return cmd, rec

    if kind == "set":
        x = _int(op.get("x", 0), 0)
        y = _int(op.get("y", 0), 0)
        z = _int(op.get("z", 0), 0)
        block = sanitize_block_name(op.get("block", "air"))
        cmd = f"/setblock {x} {y} {z} {block}"
        rec = {"op": "set", "x": x, "y": y, "z": z, "block": block}
        return cmd, rec

    return None


def _source_non_air_count(voxels: np.ndarray) -> int:
    count = 0
    it = np.nditer(voxels, flags=["refs_ok"])
    for v in it:
        if sanitize_block_name(v.item()) != "air":
            count += 1
    return count


def _equip_hotbar_block(agent_host: Any, block: str, slot: int, current: Optional[str], interval_sec: float) -> Tuple[Optional[str], int]:
    if current == block:
        return current, 0
    slot = max(0, min(8, int(slot)))
    hotbar_index = slot + 1  # Malmo command uses 1-based hotbar numbering.
    send_count = 0
    agent_host.sendCommand(f"chat /replaceitem entity @p slot.hotbar.{slot} minecraft:{block} 64")
    send_count += 1
    time.sleep(max(0.01, interval_sec))
    agent_host.sendCommand(f"hotbar.{hotbar_index} 1")
    send_count += 1
    time.sleep(0.03)
    agent_host.sendCommand(f"hotbar.{hotbar_index} 0")
    send_count += 1
    time.sleep(0.03)
    return block, send_count


def _hand_place_single_block(
    agent_host: Any,
    x: int,
    y: int,
    z: int,
    tp_height_offset: float,
    use_pulse_sec: float,
    interval_sec: float,
) -> int:
    # Place from above: look straight down so the block is placed at (x,y,z)
    # against support at (x,y-1,z) when available.
    px = x + 0.5
    py = y + float(tp_height_offset)
    pz = z + 0.5
    send_count = 0
    agent_host.sendCommand(f"tp {px:.3f} {py:.3f} {pz:.3f}")
    send_count += 1
    agent_host.sendCommand("setYaw 0")
    send_count += 1
    agent_host.sendCommand("setPitch 89")
    send_count += 1
    time.sleep(max(0.01, interval_sec * 0.5))
    agent_host.sendCommand("use 1")
    send_count += 1
    time.sleep(max(0.02, use_pulse_sec))
    agent_host.sendCommand("use 0")
    send_count += 1
    time.sleep(max(0.0, interval_sec))
    return send_count


def _grid_to_voxels(grid: List[Any], bbox: Dict[str, int]) -> np.ndarray:
    bx, by, bz = _bbox_dims(bbox)
    expected = bx * by * bz
    if len(grid) != expected:
        raise CaptureError(
            f"Grid size mismatch: got={len(grid)} expected={expected} for bbox dims (bx,by,bz)=({bx},{by},{bz})"
        )
    vox = np.empty((by, bx, bz), dtype="<U32")
    idx = 0
    for yi in range(by):
        for zi in range(bz):
            for xi in range(bx):
                raw = grid[idx]
                idx += 1
                vox[yi, xi, zi] = sanitize_block_name(raw)
    return vox


def _extract_grid(obs: Dict[str, Any], grid_name: str) -> Optional[List[Any]]:
    arr = obs.get(grid_name)
    if isinstance(arr, list):
        return arr
    return None


def _poll_grid(agent_host: Any, grid_name: str, timeout_sec: float = 12.0) -> List[Any]:
    deadline = time.time() + timeout_sec
    last: Optional[List[Any]] = None
    while time.time() < deadline:
        obs = wait_for_observation(agent_host, timeout_sec=1.2)
        arr = _extract_grid(obs, grid_name)
        if arr is not None:
            last = arr
            if len(arr) > 0:
                return arr
        time.sleep(0.08)
    if last is not None:
        return last
    raise CaptureError("grid observation not available.")


def _maybe_set_malmo_dir() -> None:
    if os.environ.get("MALMO_DIR"):
        malmo_dir = Path(os.environ["MALMO_DIR"]).expanduser()
    else:
        malmo_dir = ROOT / "MalmoPlatform"
        if malmo_dir.is_dir():
            os.environ["MALMO_DIR"] = str(malmo_dir.resolve())
    if not malmo_dir.is_dir():
        return

    xsd_candidates = [
        malmo_dir / "Schemas",
        malmo_dir / "Malmo" / "Schemas",
    ]
    current_xsd = Path(os.environ.get("MALMO_XSD_PATH", "")).expanduser() if os.environ.get("MALMO_XSD_PATH") else None
    current_ok = bool(current_xsd and (current_xsd / "Mission.xsd").is_file())
    if not current_ok:
        for c in xsd_candidates:
            if (c / "Mission.xsd").is_file():
                os.environ["MALMO_XSD_PATH"] = str(c.resolve())
                break


def _safe_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())
    return s or "unknown"


def _execute_building(
    bdir: Path,
    source_subdir: str,
    out_subdir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    src_dir = bdir / source_subdir
    out_dir = bdir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(log_dir / "agentexec_real.log")

    src_vox_path = src_dir / "voxels.npy"
    src_bbox_path = src_dir / "bbox.json"
    src_actions_path = src_dir / "actions.json"
    if not src_vox_path.is_file() or not src_bbox_path.is_file():
        raise CaptureError(f"missing source files: {src_vox_path} / {src_bbox_path}")

    out_vox = out_dir / "voxels.npy"
    out_bbox = out_dir / "bbox.json"
    out_actions = out_dir / "actions.json"
    out_report = out_dir / "agentexec_real_report.json"
    if out_vox.is_file() and out_bbox.is_file() and not args.overwrite:
        return {"building": bdir.name, "status": "skipped_exists"}

    bbox = _load_bbox(src_bbox_path)
    bx, by, bz = _bbox_dims(bbox)
    if bx <= 0 or by <= 0 or bz <= 0:
        raise CaptureError(f"invalid bbox dims: {bbox}")

    src_vox = np.load(src_vox_path, allow_pickle=False)
    source_non_air = _source_non_air_count(src_vox)
    placement_mode = str(args.placement_mode).strip().lower()
    hand_place_mode = placement_mode == "hand_place"

    if hand_place_mode:
        # Hand-place mode must operate on explicit per-block set operations.
        ops = _ops_from_voxels(src_vox, bbox=bbox, max_operations=int(args.max_operations))
        fallback_ops_used = True
    else:
        ops = _load_ops_from_actions(src_actions_path, max_operations=int(args.max_operations))
        fallback_ops_used = False
        if not ops:
            ops = _ops_from_voxels(src_vox, bbox=bbox, max_operations=int(args.max_operations))
            fallback_ops_used = True

    commands: List[str] = []
    executed_ops: List[Dict[str, Any]] = []
    dropped_ops = 0
    if not hand_place_mode:
        for op in ops:
            converted = _command_from_op(op)
            if converted is None:
                dropped_ops += 1
                continue
            cmd, rec = converted
            commands.append(cmd)
            executed_ops.append(rec)
            if len(commands) >= int(args.max_operations):
                break
        if not commands:
            raise CaptureError("no executable operations after conversion.")
    elif not ops:
        raise CaptureError("no block placements for hand_place mode.")

    anchor = _make_anchor_pose(bbox)
    anchor_block = (
        int(math.floor(anchor.x)),
        int(math.floor(anchor.y)),
        int(math.floor(anchor.z)),
    )
    grid_name = "build_grid"
    grid_min = (bbox["xmin"] - anchor_block[0], bbox["ymin"] - anchor_block[1], bbox["zmin"] - anchor_block[2])
    grid_max = (bbox["xmax"] - anchor_block[0], bbox["ymax"] - anchor_block[1], bbox["zmax"] - anchor_block[2])
    mission_xml = _build_agentexec_mission_xml(
        bbox=bbox,
        grid_name=grid_name,
        grid_min=grid_min,
        grid_max=grid_max,
        start_pose=anchor,
    )

    _append_candidate_malmo_paths()
    _maybe_set_malmo_dir()
    MalmoPython = load_malmo()
    agent_host = MalmoPython.AgentHost()

    def _graceful_quit() -> None:
        try:
            agent_host.sendCommand("quit")
        except Exception:
            return
        deadline = time.time() + max(0.0, float(args.mission_quit_wait_sec))
        while time.time() < deadline:
            try:
                ws = agent_host.getWorldState()
            except Exception:
                break
            is_running = bool(getattr(ws, "is_mission_running", False))
            if not is_running:
                break
            time.sleep(0.15)
        # Small extra wait helps the next startMission() find an available client.
        time.sleep(0.25)

    try:
        start_mission(MalmoPython, agent_host, mission_xml, port=int(args.port), logger=logger)
        wait_for_mission_begin(agent_host, timeout_sec=180.0, logger=logger)
        _ = wait_for_observation(agent_host, timeout_sec=20.0)

        # Keep anchor fixed for deterministic relative-grid observation.
        agent_host.sendCommand(f"tp {anchor.x:.3f} {anchor.y:.3f} {anchor.z:.3f}")
        agent_host.sendCommand("setYaw 0")
        agent_host.sendCommand("setPitch 50")
        time.sleep(0.25)

        # Quiet logs + ensure deterministic environment.
        agent_host.sendCommand("chat /gamerule commandBlockOutput false")
        agent_host.sendCommand("chat /gamerule doMobSpawning false")
        agent_host.sendCommand("chat /gamerule randomTickSpeed 0")
        time.sleep(0.1)

        t0 = time.time()
        commands_sent = 0
        hand_unplaced_ops = 0
        if hand_place_mode:
            remaining = list(ops)
            placed_coords: set[Tuple[int, int, int]] = set()
            equip_block: Optional[str] = None
            pass_count = max(1, int(args.hand_place_max_passes))
            for pass_idx in range(pass_count):
                if not remaining:
                    break
                next_remaining: List[Dict[str, Any]] = []
                pass_progress = 0
                for op in remaining:
                    if str(op.get("op", "")).lower() != "set":
                        dropped_ops += 1
                        continue
                    x = _int(op.get("x", 0), 0)
                    y = _int(op.get("y", 0), 0)
                    z = _int(op.get("z", 0), 0)
                    block = sanitize_block_name(op.get("block", "air"))
                    if block == "air":
                        dropped_ops += 1
                        continue

                    has_vertical_support = (y <= bbox["ymin"]) or ((x, y - 1, z) in placed_coords)
                    # Defer floating blocks first; final pass tries best-effort placement anyway.
                    if not has_vertical_support and pass_idx < pass_count - 1:
                        next_remaining.append(op)
                        continue

                    equip_block, sent = _equip_hotbar_block(
                        agent_host=agent_host,
                        block=block,
                        slot=int(args.hand_place_hotbar_slot),
                        current=equip_block,
                        interval_sec=max(0.01, float(args.command_interval_sec)),
                    )
                    commands_sent += sent
                    commands_sent += _hand_place_single_block(
                        agent_host=agent_host,
                        x=x,
                        y=y,
                        z=z,
                        tp_height_offset=float(args.hand_place_tp_height_offset),
                        use_pulse_sec=float(args.hand_place_use_pulse_sec),
                        interval_sec=max(0.01, float(args.command_interval_sec)),
                    )
                    executed_ops.append(
                        {
                            "op": "set",
                            "x": x,
                            "y": y,
                            "z": z,
                            "block": block,
                            "placement_mode": "hand_place",
                            "pass": pass_idx + 1,
                        }
                    )
                    placed_coords.add((x, y, z))
                    pass_progress += 1
                    if len(executed_ops) % 200 == 0:
                        logger.log(
                            f"hand-place progress: executed={len(executed_ops)}/{len(ops)} "
                            f"pass={pass_idx + 1}/{pass_count}"
                        )
                    if len(executed_ops) >= int(args.max_operations):
                        break

                if len(executed_ops) >= int(args.max_operations):
                    break
                if not next_remaining:
                    remaining = []
                    break
                if pass_progress == 0 and pass_idx == pass_count - 1:
                    remaining = next_remaining
                    break
                remaining = next_remaining

            hand_unplaced_ops = max(0, len(ops) - len(executed_ops) - dropped_ops)
            dropped_ops += hand_unplaced_ops
        else:
            for cmd in commands:
                agent_host.sendCommand(f"chat {cmd}")
                commands_sent += 1
                time.sleep(max(0.0, float(args.command_interval_sec)))
        time.sleep(max(0.0, float(args.post_command_wait_sec)))

        ratio = max(0.0, float(args.min_non_air_ratio))
        min_non_air: Optional[int] = None
        if ratio > 0.0:
            min_non_air = max(1, int(source_non_air * ratio))
        stable_non_air = -1
        try:
            stable_non_air = wait_for_generation_stable(
                agent_host=agent_host,
                grid_name=grid_name,
                logger=logger,
                min_non_air_count=min_non_air,
                stable_k=max(2, int(args.stability_k)),
                sample_interval_sec=max(0.05, float(args.stability_interval_sec)),
                max_samples=max(10, int(args.stability_max_samples)),
                max_seconds=max(5.0, float(args.stability_max_seconds)),
            )
        except CaptureError as exc:
            logger.log(f"WARN: stability check fallback: {exc}")

        agent_host.sendCommand(f"tp {anchor.x:.3f} {anchor.y:.3f} {anchor.z:.3f}")
        time.sleep(0.2)
        grid = _poll_grid(agent_host=agent_host, grid_name=grid_name, timeout_sec=12.0)
        vox = _grid_to_voxels(grid=grid, bbox=bbox)

        np.save(out_vox, vox)
        bbox_payload = {
            **bbox,
            "order": "xmin,xmax,ymin,ymax,zmin,zmax",
            "voxel_axis_order": "Y,X,Z",
            "source": str(src_bbox_path),
            "generated_by": "tools/generate_agentexec_world_real.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "agentexec_mode": "real_hand_place" if hand_place_mode else "real_chat_commands",
        }
        out_bbox.write_text(json.dumps(bbox_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        out_actions.write_text(json.dumps({"operations": executed_ops}, ensure_ascii=False, indent=2), encoding="utf-8")

        out_non_air = int(np.sum(vox != "air"))
        report = {
            "building": bdir.name,
            "source_subdir": source_subdir,
            "out_subdir": out_subdir,
            "source_actions_exists": src_actions_path.is_file(),
            "fallback_ops_used": fallback_ops_used,
            "source_non_air_blocks": int(source_non_air),
            "output_non_air_blocks": int(out_non_air),
            "operations_loaded": int(len(ops)),
            "operations_executed": int(len(executed_ops)),
            "operations_dropped": int(dropped_ops),
            "commands_sent": int(commands_sent),
            "placement_mode": placement_mode,
            "hand_place_unplaced_ops": int(hand_unplaced_ops) if hand_place_mode else None,
            "stable_non_air_count": int(stable_non_air),
            "port": int(args.port),
            "command_interval_sec": float(args.command_interval_sec),
            "post_command_wait_sec": float(args.post_command_wait_sec),
            "stability_min_non_air_ratio": ratio,
            "stability_min_non_air_count": int(min_non_air) if min_non_air is not None else None,
            "stability_k": int(max(2, int(args.stability_k))),
            "stability_interval_sec": float(max(0.05, float(args.stability_interval_sec))),
            "stability_max_samples": int(max(10, int(args.stability_max_samples))),
            "stability_max_seconds": float(max(5.0, float(args.stability_max_seconds))),
            "elapsed_sec": round(time.time() - t0, 3),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.log(
            f"real agentexec done: mode={placement_mode} commands={commands_sent} "
            f"source_non_air={source_non_air} output_non_air={out_non_air}"
        )
        return {"building": bdir.name, "status": "done", **report}
    finally:
        _graceful_quit()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    buildings = _list_buildings(dataset_root, args.building_pattern, int(args.limit))
    if not buildings:
        raise SystemExit(f"no buildings found: root={dataset_root} pattern={args.building_pattern}")

    done = 0
    skipped = 0
    failed = 0
    failed_buildings: List[str] = []
    started_at = time.time()

    for i, bdir in enumerate(buildings, start=1):
        try:
            result = _execute_building(
                bdir=bdir,
                source_subdir=args.source_subdir,
                out_subdir=args.out_subdir,
                args=args,
            )
            status = result.get("status", "done")
            if status == "skipped_exists":
                skipped += 1
            else:
                done += 1
            print(f"[generate_agentexec_world_real] {i}/{len(buildings)} {bdir.name} status={status}")
        except Exception as exc:
            failed += 1
            failed_buildings.append(bdir.name)
            print(f"[generate_agentexec_world_real] {i}/{len(buildings)} {bdir.name} FAILED: {exc}", file=sys.stderr)
        if i < len(buildings):
            time.sleep(max(0.0, float(args.inter_building_cooldown_sec)))

    elapsed = time.time() - started_at
    print(
        "[generate_agentexec_world_real] summary: "
        f"root={dataset_root} source={args.source_subdir} out={args.out_subdir} "
        f"done={done} skipped={skipped} failed={failed} elapsed_sec={elapsed:.2f}"
    )
    if failed_buildings:
        print("[generate_agentexec_world_real] failed_buildings=" + ",".join(_safe_name(x) for x in failed_buildings), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
