#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.cost_logger import append_cost_event, estimate_usage_cost
from tools.llm_client import LLMCompletion, complete_multimodal_with_meta, extract_json_object
from tools.llm_config import load_llm_config, require_provider_key


BASELINE_SYSTEM_PROMPT = (
    "You are a Minecraft reconstruction planner. "
    "Return only one strict JSON object with a compact build plan."
)

BASELINE_USER_PROMPT_TEMPLATE = """
You are given a textual building description.
Create an executable build plan using ONLY these operation types:
- fill: axis-aligned cuboid fill
- carve: axis-aligned cuboid set to air
- set: single block placement

Output STRICT JSON with this schema:
{
  "plan_version": "v1",
  "bbox": {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0},
  "palette": {
    "wall": "stonebrick",
    "roof": "wood",
    "trim": "brick",
    "glass": "glass",
    "light": "glowstone",
    "floor": "wood"
  },
  "operations": [
    {"op": "fill", "x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 1, "z2": 1, "block": "stonebrick", "purpose": "base"},
    {"op": "carve", "x1": 0, "y1": 0, "z1": 0, "x2": 0, "y2": 0, "z2": 0, "purpose": "hollow"},
    {"op": "set", "x": 0, "y": 0, "z": 0, "block": "glowstone", "purpose": "light"}
  ],
  "notes": ["optional"]
}

Rules:
- Coordinates must be integers.
- Keep bbox anchored near origin and positive when possible.
- operations count <= 260.
- Create a complete, coherent building: floor, walls, roof, windows, entrance.
- Use normalized material IDs when possible.

Description JSON:
[[DESCRIPTION_JSON]]
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rebuild plans from description JSON.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root with building_xxx dirs.")
    parser.add_argument("--description_subdir", default="description", help="Subdir containing description.json")
    parser.add_argument("--plan_subdir", default="rebuild_plan", help="Subdir to save plan outputs")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern")
    parser.add_argument("--limit", type=int, default=0, help="Max building count (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plan")
    parser.add_argument("--use_heuristic_only", action="store_true", help="Skip LLM and use heuristic plan")
    parser.add_argument(
        "--no_fallback_heuristic",
        action="store_true",
        help="If LLM fails, do not fallback to heuristic (skip building instead)",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1800)
    parser.add_argument(
        "--prompt_profile",
        default="",
        help="Optional JSON prompt profile path for rebuild planning.",
    )
    parser.add_argument(
        "--critic_revise",
        action="store_true",
        help="Enable second-pass critic/revise stage if profile includes critic prompts.",
    )
    parser.add_argument("--dotenv", default="")
    parser.add_argument("--provider", default="", help="Optional override: openai|anthropic|mock")
    return parser.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in dataset_root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _default_prompt_profile() -> Dict[str, Any]:
    return {
        "name": "baseline_v1",
        "version": "1",
        "planner_system_prompt": BASELINE_SYSTEM_PROMPT,
        "planner_user_template": BASELINE_USER_PROMPT_TEMPLATE,
        "critic_system_prompt": "",
        "critic_user_template": "",
        "few_shots": [],
    }


def _load_prompt_profile(path_value: str) -> Dict[str, Any]:
    profile = _default_prompt_profile()
    if not path_value:
        return profile

    profile_path = Path(path_value).expanduser().resolve()
    if not profile_path.is_file():
        raise SystemExit(f"prompt_profile not found: {profile_path}")

    try:
        loaded = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to parse prompt_profile JSON: {profile_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"prompt_profile must be a JSON object: {profile_path}")

    for key in (
        "name",
        "version",
        "planner_system_prompt",
        "planner_user_template",
        "critic_system_prompt",
        "critic_user_template",
        "few_shots",
    ):
        if key in loaded:
            profile[key] = loaded[key]

    profile["name"] = str(profile.get("name", "custom_profile"))
    profile["version"] = str(profile.get("version", "1"))
    profile["planner_system_prompt"] = str(profile.get("planner_system_prompt", BASELINE_SYSTEM_PROMPT))
    profile["planner_user_template"] = str(profile.get("planner_user_template", BASELINE_USER_PROMPT_TEMPLATE))
    profile["critic_system_prompt"] = str(profile.get("critic_system_prompt", ""))
    profile["critic_user_template"] = str(profile.get("critic_user_template", ""))
    if not isinstance(profile.get("few_shots"), list):
        profile["few_shots"] = []
    profile["profile_path"] = str(profile_path)
    return profile


def _format_few_shots_block(few_shots: List[Any]) -> str:
    blocks: List[str] = []
    for i, raw in enumerate(few_shots, start=1):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", f"example_{i}")).strip() or f"example_{i}"
        desc_obj = raw.get("description_json", {})
        plan_obj = raw.get("plan_json", {})
        blocks.append(
            (
                f"[Few-shot {i}: {title}]\n"
                f"Description JSON:\n{json.dumps(desc_obj, ensure_ascii=False, indent=2)}\n\n"
                f"Target Plan JSON:\n{json.dumps(plan_obj, ensure_ascii=False, indent=2)}"
            )
        )
    return "\n\n".join(blocks).strip()


def _render_prompt_template(
    template: str,
    *,
    description_json: str,
    few_shots_block: str = "",
    draft_plan_json: str = "",
) -> str:
    text = str(template)
    text = text.replace("[[DESCRIPTION_JSON]]", description_json)
    text = text.replace("[[FEW_SHOTS]]", few_shots_block)
    text = text.replace("[[DRAFT_PLAN_JSON]]", draft_plan_json)
    return text


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _pick_material(desc: Dict[str, Any], role_hint: str, default: str) -> str:
    mats = desc.get("materials", [])
    if not isinstance(mats, list):
        return default
    best = None
    for m in mats:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name", "")).strip()
        role = str(m.get("role", "")).strip().lower()
        if not name:
            continue
        if role_hint in role:
            return name
        if best is None:
            best = name
    return best or default


def _heuristic_plan(desc: Dict[str, Any]) -> Dict[str, Any]:
    dims = desc.get("dimensions_estimate", {}) if isinstance(desc.get("dimensions_estimate"), dict) else {}
    width = _clamp(_int(dims.get("width", 14), 14), 8, 56)
    depth = _clamp(_int(dims.get("depth", 12), 12), 8, 56)
    height = _clamp(_int(dims.get("height", 10), 10), 6, 36)

    shape = desc.get("shape", {}) if isinstance(desc.get("shape"), dict) else {}
    floors = _clamp(_int(shape.get("floors_estimate", 1), 1), 1, 4)
    roof_type = str(shape.get("roof_type", "flat")).strip().lower() or "flat"

    wall = _pick_material(desc, "wall", "stonebrick")
    roof = _pick_material(desc, "roof", "wood")
    trim = _pick_material(desc, "trim", "brick")
    glass = _pick_material(desc, "glass", "glass")
    floor = _pick_material(desc, "floor", "wood")
    light = _pick_material(desc, "light", "glowstone")

    base_floor_h = max(3, min(6, height // max(1, floors)))
    wall_top = max(3, height - 2)

    x0, z0, y0 = 0, 0, 0
    x1, z1 = width - 1, depth - 1
    y1 = max(wall_top, 4)

    ops: List[Dict[str, Any]] = []

    # Floor and shell.
    ops.append({"op": "fill", "x1": x0, "y1": y0, "z1": z0, "x2": x1, "y2": y0, "z2": z1, "block": floor, "purpose": "floor"})
    ops.append({"op": "fill", "x1": x0, "y1": y0 + 1, "z1": z0, "x2": x1, "y2": y1, "z2": z1, "block": wall, "purpose": "outer_shell"})
    if x1 - x0 >= 2 and z1 - z0 >= 2 and y1 - (y0 + 1) >= 1:
        ops.append({"op": "carve", "x1": x0 + 1, "y1": y0 + 1, "z1": z0 + 1, "x2": x1 - 1, "y2": y1 - 1, "z2": z1 - 1, "purpose": "hollow_interior"})

    # Door.
    door_x = (x0 + x1) // 2
    ops.append({"op": "carve", "x1": door_x, "y1": y0 + 1, "z1": z0, "x2": door_x, "y2": y0 + 3, "z2": z0, "purpose": "entrance_opening"})

    # Windows.
    win_y1 = y0 + 2
    win_y2 = min(y1 - 1, y0 + 3)
    if win_y2 >= win_y1:
        for x in range(x0 + 2, x1 - 1, 3):
            ops.append({"op": "fill", "x1": x, "y1": win_y1, "z1": z0, "x2": x, "y2": win_y2, "z2": z0, "block": glass, "purpose": "front_windows"})
            ops.append({"op": "fill", "x1": x, "y1": win_y1, "z1": z1, "x2": x, "y2": win_y2, "z2": z1, "block": glass, "purpose": "rear_windows"})
        for z in range(z0 + 2, z1 - 1, 3):
            ops.append({"op": "fill", "x1": x0, "y1": win_y1, "z1": z, "x2": x0, "y2": win_y2, "z2": z, "block": glass, "purpose": "side_windows"})
            ops.append({"op": "fill", "x1": x1, "y1": win_y1, "z1": z, "x2": x1, "y2": win_y2, "z2": z, "block": glass, "purpose": "side_windows"})

    # Decorative corner lights.
    corners = [(x0, z0), (x1, z0), (x0, z1), (x1, z1)]
    for cx, cz in corners:
        ops.append({"op": "set", "x": cx, "y": y0 + 2, "z": cz, "block": light, "purpose": "corner_light"})

    # Roof.
    if roof_type in {"gable", "hip"}:
        layers = min((min(width, depth) // 2), 6)
        for i in range(layers):
            rx0 = x0 + i
            rx1 = x1 - i
            rz0 = z0 + (i if roof_type == "hip" else 0)
            rz1 = z1 - (i if roof_type == "hip" else 0)
            ry = y1 + 1 + i
            if rx1 < rx0 or rz1 < rz0:
                break
            ops.append({"op": "fill", "x1": rx0, "y1": ry, "z1": rz0, "x2": rx1, "y2": ry, "z2": rz1, "block": roof, "purpose": "roof_layer"})
        ops.append({"op": "fill", "x1": x0, "y1": y1 + 1, "z1": z0, "x2": x1, "y2": y1 + 1, "z2": z1, "block": trim, "purpose": "roof_trim"})
    else:
        ops.append({"op": "fill", "x1": x0, "y1": y1 + 1, "z1": z0, "x2": x1, "y2": y1 + 1, "z2": z1, "block": roof, "purpose": "flat_roof"})

    bbox = {"xmin": x0, "xmax": x1, "ymin": y0, "ymax": max(y1 + 1, y0 + 4), "zmin": z0, "zmax": z1}
    plan = {
        "plan_version": "v1",
        "bbox": bbox,
        "palette": {
            "wall": wall,
            "roof": roof,
            "trim": trim,
            "glass": glass,
            "light": light,
            "floor": floor,
        },
        "operations": ops,
        "notes": ["heuristic fallback plan"],
    }
    return plan


def _plan_candidates(plan: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    candidates: List[Tuple[str, Dict[str, Any]]] = [("root", plan)]
    for key in ("plan", "final_plan", "result", "output", "revised_plan", "build_plan", "json"):
        value = plan.get(key)
        if isinstance(value, dict):
            candidates.append((f"root.{key}", value))
    return candidates


def _find_dict_by_keys(plan: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[Dict[str, Any]], str]:
    for source, cand in _plan_candidates(plan):
        for key in keys:
            value = cand.get(key)
            if isinstance(value, dict):
                return value, f"{source}.{key}"
    return None, ""


def _extract_notes(plan: Dict[str, Any]) -> List[str]:
    for _source, cand in _plan_candidates(plan):
        notes = cand.get("notes")
        if isinstance(notes, list):
            return [str(x) for x in notes]
    return []


def _extract_palette(plan: Dict[str, Any]) -> Dict[str, str]:
    pal_obj, _ = _find_dict_by_keys(plan, ("palette",))
    if not isinstance(pal_obj, dict):
        return {}
    return {str(k): str(v) for k, v in pal_obj.items()}


def _get_first_value(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return None


def _parse_xyz(value: Any) -> Optional[Tuple[int, int, int]]:
    if isinstance(value, dict):
        x = _get_first_value(value, ("x", "X", "x1", "xmin", "min_x"))
        y = _get_first_value(value, ("y", "Y", "y1", "ymin", "min_y"))
        z = _get_first_value(value, ("z", "Z", "z1", "zmin", "min_z"))
        if x is None or y is None or z is None:
            return None
        return _int(x, 0), _int(y, 0), _int(z, 0)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return _int(value[0], 0), _int(value[1], 0), _int(value[2], 0)
    if isinstance(value, str):
        raw = value.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 3:
            try:
                return _int(parts[0], 0), _int(parts[1], 0), _int(parts[2], 0)
            except Exception:
                return None
    return None


def _parse_bbox_from_dict(raw: Dict[str, Any]) -> Optional[Dict[str, int]]:
    xmin = _get_first_value(raw, ("xmin", "x_min", "min_x", "x1"))
    xmax = _get_first_value(raw, ("xmax", "x_max", "max_x", "x2"))
    ymin = _get_first_value(raw, ("ymin", "y_min", "min_y", "y1"))
    ymax = _get_first_value(raw, ("ymax", "y_max", "max_y", "y2"))
    zmin = _get_first_value(raw, ("zmin", "z_min", "min_z", "z1"))
    zmax = _get_first_value(raw, ("zmax", "z_max", "max_z", "z2"))

    if None in (xmin, xmax, ymin, ymax, zmin, zmax):
        min_xyz = _parse_xyz(raw.get("min"))
        max_xyz = _parse_xyz(raw.get("max"))
        if min_xyz is not None and max_xyz is not None:
            xmin, ymin, zmin = min_xyz
            xmax, ymax, zmax = max_xyz
        else:
            from_xyz = _parse_xyz(raw.get("from"))
            to_xyz = _parse_xyz(raw.get("to"))
            if from_xyz is not None and to_xyz is not None:
                xmin, ymin, zmin = from_xyz
                xmax, ymax, zmax = to_xyz

    if None in (xmin, xmax, ymin, ymax, zmin, zmax):
        return None

    bxmin, bxmax = _int(xmin, 0), _int(xmax, 0)
    bymin, bymax = _int(ymin, 0), _int(ymax, 0)
    bzmin, bzmax = _int(zmin, 0), _int(zmax, 0)
    if bxmax < bxmin:
        bxmin, bxmax = bxmax, bxmin
    if bymax < bymin:
        bymin, bymax = bymax, bymin
    if bzmax < bzmin:
        bzmin, bzmax = bzmax, bzmin
    return {"xmin": bxmin, "xmax": bxmax, "ymin": bymin, "ymax": bymax, "zmin": bzmin, "zmax": bzmax}


def _extract_bbox(plan: Dict[str, Any]) -> Tuple[Dict[str, int], str]:
    bbox_obj, bbox_source = _find_dict_by_keys(plan, ("bbox", "bounds", "bounding_box", "box"))
    if isinstance(bbox_obj, dict):
        parsed = _parse_bbox_from_dict(bbox_obj)
        if parsed is not None:
            return parsed, bbox_source

    top = _parse_bbox_from_dict(plan)
    if top is not None:
        return top, "root.inline_bbox"

    # Safe default if neither bbox nor inline bounds exists.
    return {"xmin": 0, "xmax": 15, "ymin": 0, "ymax": 12, "zmin": 0, "zmax": 15}, "default"


def _extract_raw_operations(plan: Dict[str, Any]) -> Tuple[List[Any], str]:
    op_keys = ("operations", "ops", "actions", "steps", "commands", "build_ops")
    for source, cand in _plan_candidates(plan):
        for key in op_keys:
            raw = cand.get(key)
            if isinstance(raw, list):
                return raw, f"{source}.{key}"
            if isinstance(raw, str):
                txt = raw.strip()
                if txt.startswith("[") and txt.endswith("]"):
                    try:
                        loaded = json.loads(txt)
                    except Exception:
                        loaded = None
                    if isinstance(loaded, list):
                        return loaded, f"{source}.{key}(json)"
        # Wrapper case: {"plan": {"operations": [...]}} already handled via _plan_candidates.
    return [], "missing"


def _extract_purpose(op: Dict[str, Any]) -> str:
    for key in ("purpose", "description", "desc", "note", "label", "reason"):
        value = op.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_block(op: Dict[str, Any], default: str) -> str:
    for key in (
        "block",
        "material",
        "block_type",
        "type_id",
        "id",
        "name",
        "to",
        "to_block",
        "set_to",
        "with",
        "replace_with",
        "outer_block",
        "inner_block",
        "block_glass",
        "block_frame",
    ):
        value = op.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _parse_op_kind(op: Dict[str, Any]) -> Tuple[str, str, bool]:
    kind_key = ""
    raw_kind = ""
    for key in ("op", "type", "action", "kind", "cmd", "command"):
        value = op.get(key)
        if value is None:
            continue
        txt = str(value).strip()
        if not txt:
            continue
        kind_key = key
        raw_kind = txt.lower().replace("-", "_").replace(" ", "_")
        break

    if not raw_kind:
        return "", "", False

    alias = {
        "fill": "fill",
        "cuboid": "fill",
        "box": "fill",
        "build": "fill",
        "place_cuboid": "fill",
        "roof_fill": "fill",
        "roof_coping": "fill",
        "trim_detail": "fill",
        "step": "fill",
        "window_row": "fill",
        "windows_row": "fill",
        "window_line": "fill",
        "windows_column": "fill",
        "replace_layer": "fill",
        "replace_edges": "fill",
        "walls": "fill",
        "wall": "fill",
        "replace": "replace",
        "set": "set",
        "setblock": "set",
        "place": "set",
        "put": "set",
        "block": "set",
        "carve": "carve",
        "clear": "carve",
        "remove": "carve",
        "delete": "carve",
        "erase": "carve",
        "air": "carve",
        "entrance_cut": "carve",
        "clear_above": "carve",
        "outline": "outline",
        "shell": "outline",
        "hollow": "outline",
        "hollow_box": "outline",
        "frame": "outline",
    }
    normalized = alias.get(raw_kind, raw_kind)
    was_repaired = normalized != raw_kind or kind_key != "op"
    return normalized, raw_kind, was_repaired


def _parse_cuboid(op: Dict[str, Any], bbox: Dict[str, int]) -> Optional[Tuple[int, int, int, int, int, int, bool]]:
    repaired = False

    x1_raw = _get_first_value(op, ("x1", "xmin", "x_min", "min_x"))
    y1_raw = _get_first_value(op, ("y1", "ymin", "y_min", "min_y"))
    z1_raw = _get_first_value(op, ("z1", "zmin", "z_min", "min_z"))
    x2_raw = _get_first_value(op, ("x2", "xmax", "x_max", "max_x"))
    y2_raw = _get_first_value(op, ("y2", "ymax", "y_max", "max_y"))
    z2_raw = _get_first_value(op, ("z2", "zmax", "z_max", "max_z"))

    if None in (x1_raw, y1_raw, z1_raw, x2_raw, y2_raw, z2_raw):
        for key in ("bbox", "box", "cube", "bounds"):
            raw_box = op.get(key)
            if isinstance(raw_box, dict):
                parsed = _parse_bbox_from_dict(raw_box)
                if parsed is not None:
                    x1_raw, y1_raw, z1_raw = parsed["xmin"], parsed["ymin"], parsed["zmin"]
                    x2_raw, y2_raw, z2_raw = parsed["xmax"], parsed["ymax"], parsed["zmax"]
                    repaired = True
                    break
            if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 6:
                x1_raw, y1_raw, z1_raw = raw_box[0], raw_box[1], raw_box[2]
                x2_raw, y2_raw, z2_raw = raw_box[3], raw_box[4], raw_box[5]
                repaired = True
                break

    if None in (x1_raw, y1_raw, z1_raw, x2_raw, y2_raw, z2_raw):
        # Cases like x1,x2,y1,y2,z (constant z) or x,y1,y2,z1,z2 (constant x).
        z_raw = _get_first_value(op, ("z",))
        x_raw = _get_first_value(op, ("x",))
        if None not in (x1_raw, y1_raw, x2_raw, y2_raw, z_raw):
            z1_raw = z2_raw = z_raw
            repaired = True
        elif None not in (z1_raw, y1_raw, z2_raw, y2_raw, x_raw):
            x1_raw = x2_raw = x_raw
            repaired = True

    if None in (x1_raw, y1_raw, z1_raw, x2_raw, y2_raw, z2_raw):
        point_pairs = (
            ("from", "to"),
            ("start", "end"),
            ("pos1", "pos2"),
            ("p1", "p2"),
            ("corner1", "corner2"),
            ("min", "max"),
            ("a", "b"),
        )
        parsed_pair = None
        for k1, k2 in point_pairs:
            p1 = _parse_xyz(op.get(k1))
            p2 = _parse_xyz(op.get(k2))
            if p1 is not None and p2 is not None:
                parsed_pair = (p1, p2)
                repaired = True
                break
        if parsed_pair is not None:
            (x1, y1, z1), (x2, y2, z2) = parsed_pair
            x1_raw, y1_raw, z1_raw, x2_raw, y2_raw, z2_raw = x1, y1, z1, x2, y2, z2

    if None in (x1_raw, y1_raw, z1_raw, x2_raw, y2_raw, z2_raw):
        center = _parse_xyz(op.get("pos")) or _parse_xyz(op.get("position")) or _parse_xyz(op.get("at")) or _parse_xyz(op.get("point"))
        if center is not None:
            cx, cy, cz = center
            w = max(1, _int(_get_first_value(op, ("width", "w", "dx", "size_x")), 1))
            h = max(1, _int(_get_first_value(op, ("height", "h", "dy", "size_y")), 1))
            d = max(1, _int(_get_first_value(op, ("depth", "d", "dz", "size_z")), 1))
            x1_raw, y1_raw, z1_raw = cx, cy, cz
            x2_raw, y2_raw, z2_raw = cx + w - 1, cy + h - 1, cz + d - 1
            repaired = True

    if None in (x1_raw, y1_raw, z1_raw, x2_raw, y2_raw, z2_raw):
        return None

    x1 = _int(x1_raw, bbox["xmin"])
    y1 = _int(y1_raw, bbox["ymin"])
    z1 = _int(z1_raw, bbox["zmin"])
    x2 = _int(x2_raw, bbox["xmax"])
    y2 = _int(y2_raw, bbox["ymax"])
    z2 = _int(z2_raw, bbox["zmax"])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if z2 < z1:
        z1, z2 = z2, z1
    return x1, y1, z1, x2, y2, z2, repaired


def _parse_set_point(op: Dict[str, Any], bbox: Dict[str, int]) -> Tuple[int, int, int, bool]:
    x_raw = _get_first_value(op, ("x", "px"))
    y_raw = _get_first_value(op, ("y", "py"))
    z_raw = _get_first_value(op, ("z", "pz"))
    repaired = False

    if None in (x_raw, y_raw, z_raw):
        point = _parse_xyz(op.get("pos")) or _parse_xyz(op.get("position")) or _parse_xyz(op.get("at")) or _parse_xyz(op.get("point"))
        if point is not None:
            x_raw, y_raw, z_raw = point
            repaired = True

    if None in (x_raw, y_raw, z_raw):
        if _get_first_value(op, ("x1", "y1", "z1")) is not None:
            x_raw = _get_first_value(op, ("x1",))
            y_raw = _get_first_value(op, ("y1",))
            z_raw = _get_first_value(op, ("z1",))
            repaired = True

    x = _int(x_raw, bbox["xmin"])
    y = _int(y_raw, bbox["ymin"])
    z = _int(z_raw, bbox["zmin"])
    return x, y, z, repaired


def _expand_outline_to_faces(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> List[Tuple[int, int, int, int, int, int]]:
    faces = [
        (x1, y1, z1, x2, y1, z2),  # bottom
        (x1, y2, z1, x2, y2, z2),  # top
        (x1, y1, z1, x1, y2, z2),  # west
        (x2, y1, z1, x2, y2, z2),  # east
        (x1, y1, z1, x2, y2, z1),  # north
        (x1, y1, z2, x2, y2, z2),  # south
    ]
    dedup: List[Tuple[int, int, int, int, int, int]] = []
    seen = set()
    for face in faces:
        if face in seen:
            continue
        seen.add(face)
        dedup.append(face)
    return dedup


def _ops_bbox(ops: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not ops:
        return None
    xmin = ymin = zmin = None
    xmax = ymax = zmax = None
    for op in ops:
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
        else:
            x1 = x2 = _int(op.get("x", 0), 0)
            y1 = y2 = _int(op.get("y", 0), 0)
            z1 = z2 = _int(op.get("z", 0), 0)
        if xmin is None:
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            zmin, zmax = min(z1, z2), max(z1, z2)
        else:
            xmin = min(xmin, x1, x2)
            xmax = max(xmax, x1, x2)
            ymin = min(ymin, y1, y2)
            ymax = max(ymax, y1, y2)
            zmin = min(zmin, z1, z2)
            zmax = max(zmax, z1, z2)
    if None in (xmin, xmax, ymin, ymax, zmin, zmax):
        return None
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax}


def _coerce_plan(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = dict(plan)
    bbox, bbox_source = _extract_bbox(plan)
    palette = _extract_palette(plan)
    raw_ops, ops_source = _extract_raw_operations(plan)

    report: Dict[str, Any] = {
        "bbox_source": bbox_source,
        "operations_source": ops_source,
        "input_operations_count": len(raw_ops),
        "accepted_operations_count": 0,
        "repaired_operations_count": 0,
        "expanded_operations_count": 0,
        "dropped_operations_count": 0,
        "dropped_examples": [],
    }

    fixed_ops: List[Dict[str, Any]] = []
    for idx, raw_op in enumerate(raw_ops):
        if not isinstance(raw_op, dict):
            report["dropped_operations_count"] += 1
            if len(report["dropped_examples"]) < 5:
                report["dropped_examples"].append({"index": idx, "reason": "operation_not_object"})
            continue

        kind, raw_kind, kind_repaired = _parse_op_kind(raw_op)
        if not kind:
            report["dropped_operations_count"] += 1
            if len(report["dropped_examples"]) < 5:
                report["dropped_examples"].append({"index": idx, "reason": "missing_op_kind"})
            continue

        repaired = bool(kind_repaired)
        purpose = _extract_purpose(raw_op)

        if kind in {"fill", "carve", "replace", "outline"}:
            cuboid = _parse_cuboid(raw_op, bbox)
            if cuboid is None:
                report["dropped_operations_count"] += 1
                if len(report["dropped_examples"]) < 5:
                    report["dropped_examples"].append({"index": idx, "reason": "missing_cuboid_coords", "raw_kind": raw_kind})
                continue
            x1, y1, z1, x2, y2, z2, coords_repaired = cuboid
            repaired = repaired or coords_repaired

            eff_kind = kind
            if kind == "replace":
                block_raw = _extract_block(raw_op, "stonebrick")
                if str(block_raw).strip().lower().endswith(":air") or str(block_raw).strip().lower() == "air":
                    eff_kind = "carve"
                else:
                    eff_kind = "fill"
                repaired = True

            if eff_kind == "outline":
                block = _extract_block(raw_op, "stonebrick")
                faces = _expand_outline_to_faces(x1, y1, z1, x2, y2, z2)
                for fi, (fx1, fy1, fz1, fx2, fy2, fz2) in enumerate(faces):
                    fixed_ops.append(
                        {
                            "op": "fill",
                            "x1": fx1,
                            "y1": fy1,
                            "z1": fz1,
                            "x2": fx2,
                            "y2": fy2,
                            "z2": fz2,
                            "block": str(block),
                            "purpose": f"{purpose}_outline_face_{fi}" if purpose else f"outline_face_{fi}",
                        }
                    )
                report["accepted_operations_count"] += 1
                report["expanded_operations_count"] += max(0, len(faces) - 1)
                if repaired:
                    report["repaired_operations_count"] += 1
                continue

            block = "air" if eff_kind == "carve" else _extract_block(raw_op, "stonebrick")
            fixed_ops.append(
                {
                    "op": eff_kind,
                    "x1": x1,
                    "y1": y1,
                    "z1": z1,
                    "x2": x2,
                    "y2": y2,
                    "z2": z2,
                    "block": str(block),
                    "purpose": purpose,
                }
            )
            report["accepted_operations_count"] += 1
            if repaired:
                report["repaired_operations_count"] += 1
            continue

        if kind == "set":
            x, y, z, point_repaired = _parse_set_point(raw_op, bbox)
            repaired = repaired or point_repaired
            block = _extract_block(raw_op, "stonebrick")
            fixed_ops.append({"op": "set", "x": x, "y": y, "z": z, "block": str(block), "purpose": purpose})
            report["accepted_operations_count"] += 1
            if repaired:
                report["repaired_operations_count"] += 1
            continue

        # Generic auto-repair: unknown op but coordinates + block exist.
        inferred_cuboid = _parse_cuboid(raw_op, bbox)
        if inferred_cuboid is not None:
            x1, y1, z1, x2, y2, z2, coords_repaired = inferred_cuboid
            block = _extract_block(raw_op, "stonebrick")
            block_l = str(block).strip().lower()
            inferred_kind = "carve" if (block_l == "air" or block_l.endswith(":air") or "clear" in raw_kind or "cut" in raw_kind) else "fill"
            fixed_ops.append(
                {
                    "op": inferred_kind,
                    "x1": x1,
                    "y1": y1,
                    "z1": z1,
                    "x2": x2,
                    "y2": y2,
                    "z2": z2,
                    "block": "air" if inferred_kind == "carve" else str(block),
                    "purpose": purpose,
                }
            )
            report["accepted_operations_count"] += 1
            report["repaired_operations_count"] += 1
            continue

        report["dropped_operations_count"] += 1
        if len(report["dropped_examples"]) < 5:
            report["dropped_examples"].append({"index": idx, "reason": "unsupported_op_kind", "raw_kind": raw_kind})

    fixed_ops = fixed_ops[:3000]
    op_bbox = _ops_bbox(fixed_ops)
    if op_bbox is not None:
        if bbox_source == "default":
            bbox = op_bbox
            report["bbox_source"] = "derived_from_operations"
        else:
            bbox = {
                "xmin": min(bbox["xmin"], op_bbox["xmin"]),
                "xmax": max(bbox["xmax"], op_bbox["xmax"]),
                "ymin": min(bbox["ymin"], op_bbox["ymin"]),
                "ymax": max(bbox["ymax"], op_bbox["ymax"]),
                "zmin": min(bbox["zmin"], op_bbox["zmin"]),
                "zmax": max(bbox["zmax"], op_bbox["zmax"]),
            }

    out["bbox"] = bbox
    out["palette"] = palette
    out["operations"] = fixed_ops
    out["notes"] = _extract_notes(plan)
    out.setdefault("plan_version", "v1")
    out["coerce_report"] = report
    return out, report


def _extract_all_json_dicts(text: str, max_objects: int = 600) -> List[Dict[str, Any]]:
    src = text.strip()
    out: List[Dict[str, Any]] = []
    start = src.find("{")
    while start >= 0 and len(out) < max_objects:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(src)):
            ch = src[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = src[start : i + 1]
                    try:
                        obj = json.loads(cand)
                    except Exception:
                        break
                    if isinstance(obj, dict):
                        out.append(obj)
                    break
        start = src.find("{", start + 1)
    return out


def _extract_json_list_after_key(text: str, key: str) -> Optional[List[Any]]:
    token = f'"{key}"'
    idx = 0
    while True:
        k = text.find(token, idx)
        if k < 0:
            return None
        colon = text.find(":", k + len(token))
        if colon < 0:
            return None
        lb = text.find("[", colon + 1)
        if lb < 0:
            return None

        depth = 0
        in_str = False
        esc = False
        end_idx = -1
        for i in range(lb, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx < 0:
            idx = k + len(token)
            continue

        cand = text[lb : end_idx + 1]
        try:
            loaded = json.loads(cand)
        except Exception:
            idx = k + len(token)
            continue
        if isinstance(loaded, list):
            return loaded
        idx = k + len(token)


def _looks_like_operation_obj(obj: Dict[str, Any]) -> bool:
    has_kind = any(k in obj for k in ("op", "type", "action", "kind", "cmd", "command"))
    has_single = all(k in obj for k in ("x", "y", "z"))
    has_cuboid = all(k in obj for k in ("x1", "y1", "z1")) and any(k in obj for k in ("x2", "y2", "z2", "xmax", "ymax", "zmax"))
    has_pair = any(k in obj for k in ("from", "start", "pos1", "corner1")) and any(k in obj for k in ("to", "end", "pos2", "corner2"))
    return has_kind and (has_single or has_cuboid or has_pair)


def _plan_candidate_score(plan_obj: Dict[str, Any]) -> int:
    score = 0
    raw_ops, _ = _extract_raw_operations(plan_obj)
    if raw_ops:
        score += 100 + min(500, len(raw_ops))
    _bbox, bbox_source = _extract_bbox(plan_obj)
    if bbox_source != "default":
        score += 20
    palette = _extract_palette(plan_obj)
    if palette:
        score += 10
    if "plan_version" in plan_obj:
        score += 5
    return score


def _extract_plan_from_response_text(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parse_report: Dict[str, Any] = {
        "method": "extract_json_object",
        "candidate_count": 0,
        "selected_candidate_score": 0,
        "used_ops_array_repair": False,
        "used_op_objects_repair": False,
    }

    candidates: List[Dict[str, Any]] = []
    selected: Dict[str, Any] = {}
    try:
        first = extract_json_object(text)
        if isinstance(first, dict):
            candidates.append(first)
    except Exception:
        pass

    scanned = _extract_all_json_dicts(text)
    parse_report["candidate_count"] = len(scanned)
    candidates.extend(scanned)

    best_score = -1
    for cand in candidates:
        score = _plan_candidate_score(cand)
        if score > best_score:
            best_score = score
            selected = cand

    if selected:
        parse_report["selected_candidate_score"] = max(0, best_score)

    raw_ops, _ops_source = _extract_raw_operations(selected) if selected else ([], "missing")
    if not raw_ops:
        for key in ("operations", "ops", "actions", "steps", "commands", "build_ops"):
            repaired_ops = _extract_json_list_after_key(text, key)
            if repaired_ops:
                selected = dict(selected) if selected else {}
                selected["operations"] = repaired_ops
                parse_report["method"] = f"ops_array:{key}"
                parse_report["used_ops_array_repair"] = True
                break

    raw_ops, _ops_source = _extract_raw_operations(selected) if selected else ([], "missing")
    if not raw_ops:
        op_like = [obj for obj in scanned if _looks_like_operation_obj(obj)]
        if len(op_like) >= 2:
            selected = dict(selected) if selected else {}
            selected["operations"] = op_like
            parse_report["method"] = "operation_object_scan"
            parse_report["used_op_objects_repair"] = True

    if not selected:
        selected = {}

    _bbox, bbox_source = _extract_bbox(selected)
    if bbox_source == "default":
        for cand in candidates:
            cbbox, csource = _extract_bbox(cand)
            if csource != "default":
                selected = dict(selected)
                selected["bbox"] = cbbox
                break

    if not _extract_palette(selected):
        for cand in candidates:
            pal = _extract_palette(cand)
            if pal:
                selected = dict(selected)
                selected["palette"] = pal
                break

    return selected, parse_report


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    cfg = load_llm_config(args.dotenv or None)
    if args.provider:
        cfg.provider = args.provider

    use_llm = not args.use_heuristic_only
    if use_llm and cfg.provider != "mock":
        require_provider_key(cfg)

    prompt_profile = _load_prompt_profile(args.prompt_profile)
    profile_name = str(prompt_profile.get("name", "baseline_v1"))
    profile_version = str(prompt_profile.get("version", "1"))
    profile_path = str(prompt_profile.get("profile_path", "")) if args.prompt_profile else ""
    few_shots = prompt_profile.get("few_shots", []) if isinstance(prompt_profile.get("few_shots"), list) else []
    critic_revise_enabled = bool(
        args.critic_revise
        and str(prompt_profile.get("critic_system_prompt", "")).strip()
        and str(prompt_profile.get("critic_user_template", "")).strip()
    )
    print(
        "[generate_rebuild_plans] "
        f"prompt_profile={profile_name}@{profile_version} "
        f"critic_revise={critic_revise_enabled} "
        f"few_shots={len(few_shots)}"
    )

    buildings = _list_buildings(dataset_root, args.building_pattern, args.limit)
    for bdir in buildings:
        desc_path = bdir / args.description_subdir / "description.json"
        if not desc_path.is_file():
            print(f"[generate_rebuild_plans] skip {bdir.name} (no description.json)")
            continue

        out_dir = bdir / args.plan_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_plan = out_dir / "plan.json"
        out_raw = out_dir / "plan.raw.txt"
        out_req = out_dir / "plan.request.json"
        out_usage = out_dir / "llm_usage.json"

        if out_plan.is_file() and not args.overwrite:
            print(f"[generate_rebuild_plans] skip {bdir.name} (exists)")
            continue

        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        llm_failed = False
        response_text = ""
        completion: LLMCompletion | None = None
        planner_system_prompt = ""
        planner_prompt = ""
        critic_system_prompt = ""
        critic_prompt = ""
        draft_parse_report: Dict[str, Any] = {}
        revise_parse_report: Dict[str, Any] = {}
        stage_records: List[Dict[str, Any]] = []
        plan_obj: Dict[str, Any]
        draft_plan_obj: Dict[str, Any] = {}

        if use_llm:
            desc_json_text = json.dumps(desc, ensure_ascii=False, indent=2)
            few_shots_block = _format_few_shots_block(few_shots)
            planner_system_prompt = str(prompt_profile.get("planner_system_prompt", BASELINE_SYSTEM_PROMPT))
            planner_prompt = _render_prompt_template(
                str(prompt_profile.get("planner_user_template", BASELINE_USER_PROMPT_TEMPLATE)),
                description_json=desc_json_text,
                few_shots_block=few_shots_block,
            )
            try:
                draft_completion = complete_multimodal_with_meta(
                    cfg=cfg,
                    system_prompt=planner_system_prompt,
                    user_prompt=planner_prompt,
                    image_paths=[],
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                )
                completion = draft_completion
                response_text = "[DRAFT_RESPONSE]\n" + draft_completion.text
                draft_plan_obj, draft_parse_report = _extract_plan_from_response_text(draft_completion.text)
                plan_obj = draft_plan_obj
                usage = draft_completion.usage
                cost = estimate_usage_cost(draft_completion.provider, draft_completion.model, usage)
                stage_records.append(
                    {
                        "stage": "plan_draft",
                        "provider": draft_completion.provider,
                        "model": draft_completion.model,
                        "endpoint": draft_completion.endpoint,
                        "usage": usage,
                        "cost": cost,
                        "parse_report": draft_parse_report,
                    }
                )
                append_cost_event(
                    dataset_root=dataset_root,
                    event={
                        "building": bdir.name,
                        "stage": "plan_draft",
                        "provider": draft_completion.provider,
                        "model": draft_completion.model,
                        "endpoint": draft_completion.endpoint,
                        "usage": usage,
                        "cost": cost,
                        "status": "ok",
                        "output_path": str(out_plan),
                    },
                )
                if critic_revise_enabled:
                    critic_system_prompt = str(prompt_profile.get("critic_system_prompt", ""))
                    critic_prompt = _render_prompt_template(
                        str(prompt_profile.get("critic_user_template", "")),
                        description_json=desc_json_text,
                        few_shots_block=few_shots_block,
                        draft_plan_json=json.dumps(draft_plan_obj, ensure_ascii=False, indent=2),
                    )
                    try:
                        revise_completion = complete_multimodal_with_meta(
                            cfg=cfg,
                            system_prompt=critic_system_prompt,
                            user_prompt=critic_prompt,
                            image_paths=[],
                            temperature=float(args.temperature),
                            max_tokens=int(args.max_tokens),
                        )
                        completion = revise_completion
                        plan_obj, revise_parse_report = _extract_plan_from_response_text(revise_completion.text)
                        response_text += "\n\n[REVISED_RESPONSE]\n" + revise_completion.text
                        revise_usage = revise_completion.usage
                        revise_cost = estimate_usage_cost(revise_completion.provider, revise_completion.model, revise_usage)
                        stage_records.append(
                            {
                                "stage": "plan_revise",
                                "provider": revise_completion.provider,
                                "model": revise_completion.model,
                                "endpoint": revise_completion.endpoint,
                                "usage": revise_usage,
                                "cost": revise_cost,
                                "parse_report": revise_parse_report,
                            }
                        )
                        append_cost_event(
                            dataset_root=dataset_root,
                            event={
                                "building": bdir.name,
                                "stage": "plan_revise",
                                "provider": revise_completion.provider,
                                "model": revise_completion.model,
                                "endpoint": revise_completion.endpoint,
                                "usage": revise_usage,
                                "cost": revise_cost,
                                "status": "ok",
                                "output_path": str(out_plan),
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        response_text += f"\n\n[REVISED_RESPONSE_ERROR]\n{exc}\n"
                        append_cost_event(
                            dataset_root=dataset_root,
                            event={
                                "building": bdir.name,
                                "stage": "plan_revise",
                                "provider": completion.provider if completion is not None else cfg.provider,
                                "model": completion.model if completion is not None else (cfg.openai_model if cfg.provider == "openai" else cfg.anthropic_model),
                                "endpoint": "unknown",
                                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                                "cost": {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                    "input_price_per_mtok": None,
                                    "output_price_per_mtok": None,
                                    "input_cost_usd": None,
                                    "output_cost_usd": None,
                                    "estimated_cost_usd": None,
                                    "pricing_source": "unknown",
                                },
                                "status": "error",
                                "error": str(exc),
                            },
                        )
                        print(f"[generate_rebuild_plans] {bdir.name} critic-revise failed, keep draft: {exc}")
            except Exception as exc:  # noqa: BLE001
                llm_failed = True
                if cfg.provider == "openai":
                    model_name = cfg.openai_model
                elif cfg.provider == "anthropic":
                    model_name = cfg.anthropic_model
                else:
                    model_name = "mock-model"
                append_cost_event(
                    dataset_root=dataset_root,
                    event={
                        "building": bdir.name,
                        "stage": "plan_draft",
                        "provider": cfg.provider,
                        "model": model_name,
                        "endpoint": "unknown",
                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        "cost": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "input_price_per_mtok": None,
                            "output_price_per_mtok": None,
                            "input_cost_usd": None,
                            "output_cost_usd": None,
                            "estimated_cost_usd": None,
                            "pricing_source": "unknown",
                        },
                        "status": "error",
                        "error": str(exc),
                    },
                )
                print(f"[generate_rebuild_plans] {bdir.name} LLM plan failed: {exc}")
                plan_obj = {}
        else:
            plan_obj = {}

        if stage_records:
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens = 0
            total_estimated_cost = 0.0
            known_cost_calls = 0
            for rec in stage_records:
                usage = rec.get("usage", {}) if isinstance(rec.get("usage"), dict) else {}
                cost = rec.get("cost", {}) if isinstance(rec.get("cost"), dict) else {}
                total_input_tokens += int(usage.get("input_tokens", 0) or 0)
                total_output_tokens += int(usage.get("output_tokens", 0) or 0)
                total_tokens += int(usage.get("total_tokens", 0) or 0)
                ec = cost.get("estimated_cost_usd")
                if isinstance(ec, (int, float)):
                    known_cost_calls += 1
                    total_estimated_cost += float(ec)
            out_usage.write_text(
                json.dumps(
                    {
                        "building": bdir.name,
                        "stage": "plan",
                        "provider": stage_records[-1]["provider"],
                        "model": stage_records[-1]["model"],
                        "calls": stage_records,
                        "totals": {
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                            "total_tokens": total_tokens,
                            "known_cost_calls": known_cost_calls,
                            "estimated_cost_usd": total_estimated_cost if known_cost_calls > 0 else None,
                        },
                        "prompt_profile": {
                            "name": profile_name,
                            "version": profile_version,
                            "path": profile_path,
                            "critic_revise_enabled": bool(critic_revise_enabled),
                            "few_shots_count": len(few_shots),
                        },
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if not plan_obj:
            if args.no_fallback_heuristic and llm_failed:
                print(f"[generate_rebuild_plans] skip {bdir.name} (llm failed and fallback disabled)")
                continue
            plan_obj = _heuristic_plan(desc)
            if not response_text:
                response_text = json.dumps(plan_obj, ensure_ascii=False, indent=2)

        coerced, coerce_report = _coerce_plan(plan_obj)
        fallback_triggered = False
        fallback_report: Dict[str, Any] | None = None
        if not coerced.get("operations") and draft_plan_obj:
            draft_coerced, draft_coerce_report = _coerce_plan(draft_plan_obj)
            if draft_coerced.get("operations"):
                coerced = draft_coerced
                coerced["notes"] = list(coerced.get("notes", [])) + ["used_draft_plan_due_to_bad_or_empty_revise"]
                coerced["coerce_report_revise"] = coerce_report
                coerced["coerce_report_draft"] = draft_coerce_report
                coerce_report = {
                    **coerce_report,
                    "draft_salvage_used": True,
                    "draft_accepted_operations_count": draft_coerce_report.get("accepted_operations_count", 0),
                }
        if not coerced.get("operations"):
            if args.no_fallback_heuristic and use_llm:
                print(
                    f"[generate_rebuild_plans] skip {bdir.name} "
                    "(empty operations after coercion and fallback disabled)"
                )
                out_raw.write_text(response_text, encoding="utf-8")
                out_req.write_text(
                    json.dumps(
                        {
                            "building": bdir.name,
                            "provider": completion.provider if completion is not None else cfg.provider,
                            "model": completion.model if completion is not None else (cfg.openai_model if cfg.provider == "openai" else cfg.anthropic_model),
                            "mode": "llm",
                            "llm_attempted": True,
                            "llm_failed": bool(llm_failed),
                            "empty_operations_after_coerce": True,
                            "fallback_disabled": True,
                            "coerce_report": coerce_report,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                continue
            fallback_triggered = True
            coerced, fallback_report = _coerce_plan(_heuristic_plan(desc))
            coerced["notes"] = list(coerced.get("notes", [])) + ["fallback_used_due_to_empty_operations"]
            coerced["coerce_report_before_fallback"] = coerce_report
            coerced["coerce_report_fallback"] = fallback_report

        coerced["building"] = bdir.name
        coerced["created_at"] = datetime.now(timezone.utc).isoformat()
        if completion is not None:
            coerced["provider"] = completion.provider
            coerced["model"] = completion.model
        else:
            coerced["provider"] = cfg.provider
            coerced["model"] = cfg.openai_model if cfg.provider == "openai" else cfg.anthropic_model

        out_plan.write_text(json.dumps(coerced, ensure_ascii=False, indent=2), encoding="utf-8")
        out_raw.write_text(response_text, encoding="utf-8")
        out_req.write_text(
            json.dumps(
                {
                    "building": bdir.name,
                    "provider": coerced["provider"],
                    "model": coerced["model"],
                    "description_path": str(desc_path.relative_to(bdir)),
                    "mode": "llm" if use_llm else "heuristic_only",
                    "llm_attempted": bool(use_llm),
                    "llm_failed": bool(llm_failed),
                    "prompt_profile": {
                        "name": profile_name,
                        "version": profile_version,
                        "path": profile_path,
                        "critic_revise_enabled": bool(critic_revise_enabled),
                        "few_shots_count": len(few_shots),
                    },
                    "system_prompt": planner_system_prompt if use_llm else "",
                    "prompt": planner_prompt if use_llm else "",
                    "planner_system_prompt": planner_system_prompt if use_llm else "",
                    "planner_prompt": planner_prompt if use_llm else "",
                    "critic_system_prompt": critic_system_prompt if use_llm else "",
                    "critic_prompt": critic_prompt if use_llm else "",
                    "draft_parse_report": draft_parse_report if use_llm else {},
                    "revise_parse_report": revise_parse_report if use_llm else {},
                    "temperature": float(args.temperature),
                    "max_tokens": int(args.max_tokens),
                    "fallback_triggered": bool(fallback_triggered),
                    "coerce_report": coerce_report,
                    "fallback_coerce_report": fallback_report if fallback_triggered else None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[generate_rebuild_plans] wrote {out_plan}")


if __name__ == "__main__":
    main()
