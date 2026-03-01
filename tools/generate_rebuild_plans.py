#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

REQUIRED_PALETTE_ROLES: Tuple[str, ...] = ("wall", "roof", "trim", "glass", "light", "floor")
DEFAULT_ROLE_BLOCKS: Dict[str, str] = {
    "wall": "stonebrick",
    "roof": "wood",
    "trim": "brick",
    "glass": "glass",
    "light": "glowstone",
    "floor": "wood",
}
ROLE_ALIASES: Dict[str, str] = {
    "walls": "wall",
    "exterior_wall": "wall",
    "outer_wall": "wall",
    "facade": "wall",
    "roofing": "roof",
    "ceiling": "roof",
    "eave": "roof",
    "eaves": "roof",
    "frame": "trim",
    "frames": "trim",
    "pillar": "trim",
    "pillars": "trim",
    "column": "trim",
    "columns": "trim",
    "edge": "trim",
    "edges": "trim",
    "window": "glass",
    "windows": "glass",
    "glazing": "glass",
    "pane": "glass",
    "panes": "glass",
    "lighting": "light",
    "lights": "light",
    "lamp": "light",
    "lamps": "light",
    "lantern": "light",
    "lanterns": "light",
    "base": "floor",
    "foundation": "floor",
    "ground": "floor",
}
PURPOSE_ROLE_HINTS: Tuple[Tuple[str, str], ...] = (
    ("roof", "roof"),
    ("gable", "roof"),
    ("hip", "roof"),
    ("ridge", "roof"),
    ("wall", "wall"),
    ("facade", "wall"),
    ("shell", "wall"),
    ("trim", "trim"),
    ("frame", "trim"),
    ("column", "trim"),
    ("pillar", "trim"),
    ("edge", "trim"),
    ("window", "glass"),
    ("glass", "glass"),
    ("pane", "glass"),
    ("light", "light"),
    ("lantern", "light"),
    ("torch", "light"),
    ("entrance_light", "light"),
    ("floor", "floor"),
    ("foundation", "floor"),
    ("base", "floor"),
)
DIRECT_BLOCK_NORM: Dict[str, str] = {
    "stone_bricks": "stonebrick",
    "stone_brick": "stonebrick",
    "stonebrick": "stonebrick",
    "minecraft_stone_bricks": "stonebrick",
    "minecraft_stone_brick": "stonebrick",
    "bricks": "brick",
    "brick_block": "brick",
    "planks": "wood",
    "oak_planks": "wood",
    "spruce_planks": "wood",
    "birch_planks": "wood",
    "jungle_planks": "wood",
    "acacia_planks": "wood",
    "dark_oak_planks": "wood",
    "wooden_planks": "wood",
    "oak_fence": "fence",
    "spruce_fence": "fence",
    "birch_fence": "fence",
    "jungle_fence": "fence",
    "acacia_fence": "fence",
    "dark_oak_fence": "fence",
    "wooden_fence": "fence",
    "stone_slab": "slab_stone",
    "stone_slab2": "slab_stone",
    "double_stone_slab": "slab_stone",
    "double_stone_slab2": "slab_stone",
    "wooden_slab": "slab_wood",
    "double_wooden_slab": "slab_wood",
    "stained_glass": "glass",
    "glass_pane": "glass",
    "stained_glass_pane": "glass",
    "cave_air": "air",
    "void_air": "air",
    "air": "air",
}
ROLE_KEYWORD_HINTS: Dict[str, Tuple[str, ...]] = {
    "wall": ("wall", "facade", "exterior", "outer"),
    "roof": ("roof", "gable", "hip", "eave", "ridge"),
    "trim": ("trim", "accent", "band", "cornice", "border", "frame", "pillar", "column", "ledge", "detail"),
    "glass": ("window", "glass", "pane", "glazing"),
    "light": ("light", "lamp", "lantern", "torch"),
    "floor": ("floor", "foundation", "base", "ground", "platform"),
}
BLOCK_FAMILY_HINTS: Dict[str, str] = {
    "glass": "glass",
    "glowstone": "light",
    "lantern": "light",
    "sea_lantern": "light",
    "redstone_lamp": "light",
    "torch": "light",
    "fence": "trim",
    "slab_stone": "trim",
    "slab_wood": "trim",
    "stairs_wood": "trim",
    "brick": "wall",
    "stonebrick": "wall",
    "stone": "wall",
    "cobblestone": "wall",
    "quartz_block": "trim",
    "wood": "floor",
}


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
    parser.add_argument("--llm_seed", type=int, default=-1, help="Optional LLM seed (OpenAI only, -1 disables).")
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
    parser.add_argument("--strict_schema", dest="strict_schema", action="store_true", help="Enable strict schema/material validation.")
    parser.add_argument("--no_strict_schema", dest="strict_schema", action="store_false", help="Disable strict schema/material validation.")
    parser.add_argument(
        "--enforce_role_fixed",
        dest="enforce_role_fixed",
        action="store_true",
        help="Force non-carve operation block to match palette role block.",
    )
    parser.add_argument(
        "--no_enforce_role_fixed",
        dest="enforce_role_fixed",
        action="store_false",
        help="Do not force role-fixed block replacement.",
    )
    parser.add_argument(
        "--require_material_budget",
        dest="require_material_budget",
        action="store_true",
        help="Require and repair material_budget section.",
    )
    parser.add_argument(
        "--no_require_material_budget",
        dest="require_material_budget",
        action="store_false",
        help="Do not require material_budget section.",
    )
    parser.add_argument(
        "--material_budget_tolerance",
        type=float,
        default=0.25,
        help="Allowed relative error for budget-vs-actual per role (default: 0.25).",
    )
    parser.add_argument(
        "--role_fix_min_confidence",
        type=float,
        default=0.7,
        help="Min confidence to force op.block = palette[role] (default: 0.7).",
    )
    parser.add_argument(
        "--prefer_description_palette",
        dest="prefer_description_palette",
        action="store_true",
        help="Use description material hints to repair/override weak palette assignments.",
    )
    parser.add_argument(
        "--no_prefer_description_palette",
        dest="prefer_description_palette",
        action="store_false",
        help="Do not use description materials for palette repair.",
    )
    parser.add_argument("--max_operations", type=int, default=260, help="Maximum operations kept in final plan.")
    parser.set_defaults(
        strict_schema=True,
        enforce_role_fixed=True,
        require_material_budget=True,
        prefer_description_palette=True,
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
        "strict_schema": True,
        "enforce_role_fixed": True,
        "require_material_budget": True,
        "material_budget_tolerance": 0.25,
        "role_fix_min_confidence": 0.7,
        "prefer_description_palette": True,
        "max_operations": 260,
        "required_palette_roles": list(REQUIRED_PALETTE_ROLES),
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
        "strict_schema",
        "enforce_role_fixed",
        "require_material_budget",
        "material_budget_tolerance",
        "role_fix_min_confidence",
        "prefer_description_palette",
        "max_operations",
        "required_palette_roles",
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
    profile["strict_schema"] = bool(profile.get("strict_schema", True))
    profile["enforce_role_fixed"] = bool(profile.get("enforce_role_fixed", True))
    profile["require_material_budget"] = bool(profile.get("require_material_budget", True))
    try:
        profile["material_budget_tolerance"] = max(0.0, float(profile.get("material_budget_tolerance", 0.25)))
    except Exception:
        profile["material_budget_tolerance"] = 0.25
    try:
        profile["role_fix_min_confidence"] = max(0.0, min(1.0, float(profile.get("role_fix_min_confidence", 0.7))))
    except Exception:
        profile["role_fix_min_confidence"] = 0.7
    profile["prefer_description_palette"] = bool(profile.get("prefer_description_palette", True))
    try:
        profile["max_operations"] = max(1, int(profile.get("max_operations", 260)))
    except Exception:
        profile["max_operations"] = 260
    if not isinstance(profile.get("required_palette_roles"), list):
        profile["required_palette_roles"] = list(REQUIRED_PALETTE_ROLES)
    else:
        profile["required_palette_roles"] = [str(x) for x in profile.get("required_palette_roles", []) if str(x).strip()]
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


def _normalize_role(value: Any) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    token = token.split(":", 1)[-1]
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        return ""
    return ROLE_ALIASES.get(token, token)


def _normalize_block_type(value: Any) -> str:
    token = str(value).strip().lower()
    token = token.split("[", 1)[0].split("{", 1)[0]
    if ":" in token:
        token = token.split(":", 1)[1]
    token = token.replace("-", "_").replace(" ", "_").replace(".", "_")
    token = re.sub(r"__+", "_", token).strip("_")
    token = re.sub(r"(stone_slab)\d+$", r"\1", token)
    token = re.sub(r"(double_stone_slab)\d+$", r"\1", token)
    if not token:
        return "air"

    mapped = DIRECT_BLOCK_NORM.get(token)
    if mapped is not None:
        return mapped

    if token.endswith("_fence") or "_fence_" in token or token == "fence":
        return "fence"
    if "plank" in token:
        return "wood"
    if "stone_brick" in token:
        return "stonebrick"
    if "glass" in token:
        return "glass"
    if "slab" in token:
        if any(w in token for w in ("wood", "oak", "spruce", "birch", "jungle", "acacia", "dark_oak")):
            return "slab_wood"
        return "slab_stone" if any(w in token for w in ("stone", "brick", "sand", "quartz", "deepslate", "cobble")) else "slab"
    return token


def _block_family_role(block: str) -> str:
    b = _normalize_block_type(block)
    if not b:
        return ""
    if b in BLOCK_FAMILY_HINTS:
        return BLOCK_FAMILY_HINTS[b]
    if "glass" in b:
        return "glass"
    if any(k in b for k in ("lantern", "lamp", "torch", "glow", "light")):
        return "light"
    if any(k in b for k in ("slab", "stairs", "fence")):
        return "trim"
    if any(k in b for k in ("roof", "tile", "shingle")):
        return "roof"
    if any(k in b for k in ("brick", "stone", "cobble", "concrete")):
        return "wall"
    if any(k in b for k in ("wood", "plank", "log", "bark")):
        return "floor"
    return ""


def _extract_desc_material_hints(desc: Dict[str, Any], required_roles: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    mats = desc.get("materials", []) if isinstance(desc.get("materials"), list) else []
    hints: Dict[str, Dict[str, Any]] = {}
    for m in mats:
        if not isinstance(m, dict):
            continue
        raw_name = m.get("name", "")
        block = _normalize_block_type(raw_name)
        if not block or block == "air":
            continue

        raw_role = str(m.get("role", "")).strip().lower()
        try:
            conf = float(m.get("confidence", 0.6))
        except Exception:
            conf = 0.6
        conf = max(0.0, min(1.0, conf))

        role_scores: List[Tuple[str, float]] = []
        for role in required_roles:
            kws = ROLE_KEYWORD_HINTS.get(role, ())
            if any(kw in raw_role for kw in kws):
                role_scores.append((role, conf))

        if not role_scores:
            fam = _block_family_role(block)
            if fam in required_roles:
                role_scores.append((fam, conf * 0.7))

        if not role_scores:
            continue

        for role, score in role_scores:
            if role not in required_roles:
                continue
            prev = hints.get(role)
            if prev is None or score > float(prev.get("score", 0.0)):
                hints[role] = {
                    "block": block,
                    "score": float(score),
                    "source_role": raw_role,
                }
    return hints


def _normalize_palette(
    raw: Dict[str, Any],
    required_roles: Tuple[str, ...],
    *,
    desc_hints: Optional[Dict[str, Dict[str, Any]]] = None,
    prefer_description_palette: bool = True,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    normalized: Dict[str, str] = {}
    dropped_roles: List[str] = []
    desc_overrides: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        for rk, rv in raw.items():
            role = _normalize_role(rk)
            if not role:
                continue
            if role not in required_roles:
                dropped_roles.append(str(rk))
                continue
            block = _normalize_block_type(rv if rv is not None else DEFAULT_ROLE_BLOCKS.get(role, "stonebrick"))
            normalized[role] = block

    desc_hints = desc_hints or {}
    if prefer_description_palette:
        for role in required_roles:
            hint = desc_hints.get(role)
            if not isinstance(hint, dict):
                continue
            hinted_block = _normalize_block_type(hint.get("block", DEFAULT_ROLE_BLOCKS[role]))
            hinted_score = float(hint.get("score", 0.0) or 0.0)
            if role not in normalized:
                normalized[role] = hinted_block
                desc_overrides.append({"role": role, "from": "<missing>", "to": hinted_block, "score": hinted_score, "reason": "fill_missing"})
                continue

            current = _normalize_block_type(normalized.get(role, DEFAULT_ROLE_BLOCKS[role]))
            current_family = _block_family_role(current)
            role_family_match = current_family == role
            current_generic = current in {"stonebrick", "brick", "wood"} or current == DEFAULT_ROLE_BLOCKS.get(role, "")
            if hinted_block != current and hinted_score >= 0.72 and (not role_family_match or current_generic):
                normalized[role] = hinted_block
                desc_overrides.append({"role": role, "from": current, "to": hinted_block, "score": hinted_score, "reason": "description_preferred"})

    missing_before = [r for r in required_roles if r not in normalized]
    for role in missing_before:
        normalized[role] = DEFAULT_ROLE_BLOCKS.get(role, "stonebrick")

    coherence_repairs: List[Dict[str, str]] = []

    glass_block = _normalize_block_type(normalized.get("glass", DEFAULT_ROLE_BLOCKS["glass"]))
    if "glass" not in glass_block:
        to_block = _normalize_block_type(desc_hints.get("glass", {}).get("block", DEFAULT_ROLE_BLOCKS["glass"]))
        coherence_repairs.append({"role": "glass", "from": glass_block, "to": to_block})
        normalized["glass"] = to_block

    light_block = _normalize_block_type(normalized.get("light", DEFAULT_ROLE_BLOCKS["light"]))
    if not any(k in light_block for k in ("glowstone", "lantern", "lamp", "torch", "light")):
        to_block = _normalize_block_type(desc_hints.get("light", {}).get("block", DEFAULT_ROLE_BLOCKS["light"]))
        coherence_repairs.append({"role": "light", "from": light_block, "to": to_block})
        normalized["light"] = to_block

    wall_block = _normalize_block_type(normalized.get("wall", DEFAULT_ROLE_BLOCKS["wall"]))
    roof_block = _normalize_block_type(normalized.get("roof", DEFAULT_ROLE_BLOCKS["roof"]))
    floor_block = _normalize_block_type(normalized.get("floor", DEFAULT_ROLE_BLOCKS["floor"]))
    trim_block = _normalize_block_type(normalized.get("trim", DEFAULT_ROLE_BLOCKS["trim"]))

    if roof_block == wall_block:
        to_block = _normalize_block_type(desc_hints.get("roof", {}).get("block", DEFAULT_ROLE_BLOCKS["roof"]))
        coherence_repairs.append({"role": "roof", "from": roof_block, "to": to_block})
        normalized["roof"] = to_block
    if floor_block == wall_block:
        to_block = _normalize_block_type(desc_hints.get("floor", {}).get("block", DEFAULT_ROLE_BLOCKS["floor"]))
        coherence_repairs.append({"role": "floor", "from": floor_block, "to": to_block})
        normalized["floor"] = to_block
    if trim_block in {wall_block, _normalize_block_type(normalized["roof"]), _normalize_block_type(normalized["floor"])}:
        to_block = _normalize_block_type(desc_hints.get("trim", {}).get("block", DEFAULT_ROLE_BLOCKS["trim"]))
        coherence_repairs.append({"role": "trim", "from": trim_block, "to": to_block})
        normalized["trim"] = to_block

    return normalized, {
        "missing_roles_before_fill": missing_before,
        "dropped_unknown_roles": dropped_roles[:12],
        "description_overrides": desc_overrides[:12],
        "coherence_repairs": coherence_repairs,
    }


def _extract_material_budget(plan: Dict[str, Any], required_roles: Tuple[str, ...]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    def parse_budget_obj(raw: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                role = _normalize_role(item.get("role", ""))
                if role not in required_roles:
                    continue
                block_raw = item.get("block", item.get("material", item.get("id", DEFAULT_ROLE_BLOCKS[role])))
                tgt_raw = item.get("target_blocks", item.get("max_blocks", item.get("budget", item.get("count", item.get("blocks", 0)))))
                out[role] = {
                    "role": role,
                    "block": _normalize_block_type(block_raw),
                    "target_blocks": max(0, _int(tgt_raw, 0)),
                }
            return out

        if not isinstance(raw, dict):
            return out

        if isinstance(raw.get("material_budget"), (dict, list)):
            return parse_budget_obj(raw.get("material_budget"))

        for rk, rv in raw.items():
            role = _normalize_role(rk)
            if role not in required_roles:
                continue
            if isinstance(rv, dict):
                block_raw = rv.get("block", rv.get("material", rv.get("id", DEFAULT_ROLE_BLOCKS[role])))
                tgt_raw = rv.get("target_blocks", rv.get("max_blocks", rv.get("budget", rv.get("count", rv.get("blocks", 0)))))
                out[role] = {
                    "role": role,
                    "block": _normalize_block_type(block_raw),
                    "target_blocks": max(0, _int(tgt_raw, 0)),
                }
            else:
                out[role] = {
                    "role": role,
                    "block": DEFAULT_ROLE_BLOCKS[role],
                    "target_blocks": max(0, _int(rv, 0)),
                }
        return out

    for source, cand in _plan_candidates(plan):
        if not isinstance(cand, dict):
            continue
        for key in ("material_budget", "materials_budget", "budget"):
            if key not in cand:
                continue
            parsed = parse_budget_obj(cand.get(key))
            if parsed:
                return parsed, f"{source}.{key}"
    return {}, "missing"


def _op_volume(op: Dict[str, Any]) -> int:
    kind = str(op.get("op", "")).strip().lower()
    if kind in {"fill", "carve"}:
        x1 = _int(op.get("x1", 0), 0)
        y1 = _int(op.get("y1", 0), 0)
        z1 = _int(op.get("z1", 0), 0)
        x2 = _int(op.get("x2", x1), x1)
        y2 = _int(op.get("y2", y1), y1)
        z2 = _int(op.get("z2", z1), z1)
        return max(0, abs(x2 - x1) + 1) * max(0, abs(y2 - y1) + 1) * max(0, abs(z2 - z1) + 1)
    return 1


def _infer_role_from_purpose(purpose: str) -> str:
    p = purpose.strip().lower().replace("-", "_").replace(" ", "_")
    for hint, role in PURPOSE_ROLE_HINTS:
        if hint in p:
            return role
    return ""


def _infer_role_from_block(block: str) -> str:
    b = _normalize_block_type(block)
    if b == "glass" or "glass" in b:
        return "glass"
    if b in {"glowstone", "lantern", "sea_lantern", "torch", "redstone_lamp"} or "light" in b:
        return "light"
    if b in {"fence", "slab_stone", "slab_wood", "stairs_wood"}:
        return "trim"
    if "slab" in b or "stairs" in b or "fence" in b:
        return "trim"
    if b in {"brick", "stonebrick", "stone", "cobblestone", "concrete"} or "brick" in b or "stone" in b:
        return "wall"
    if b in {"wood", "slab_wood"} or "wood" in b or "plank" in b:
        return "floor"
    return ""


def _infer_operation_role(op: Dict[str, Any], palette: Dict[str, str]) -> str:
    explicit = _normalize_role(
        _get_first_value(
            op,
            (
                "role",
                "material_role",
                "palette_role",
                "semantic_role",
            ),
        )
    )
    if explicit in palette:
        return explicit

    purpose_role = _infer_role_from_purpose(str(op.get("purpose", "")))
    if purpose_role in palette:
        return purpose_role

    block_role = _infer_role_from_block(str(op.get("block", "")))
    if block_role in palette:
        return block_role

    # If the block exactly matches one palette role, use it.
    b = _normalize_block_type(op.get("block", ""))
    for role, p_block in palette.items():
        if _normalize_block_type(p_block) == b:
            return role
    return ""


def _op_bounds_for_role_infer(op: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    kind = str(op.get("op", "")).strip().lower()
    if kind in {"fill", "carve"}:
        x1 = _int(op.get("x1", 0), 0)
        y1 = _int(op.get("y1", 0), 0)
        z1 = _int(op.get("z1", 0), 0)
        x2 = _int(op.get("x2", x1), x1)
        y2 = _int(op.get("y2", y1), y1)
        z2 = _int(op.get("z2", z1), z1)
        lo_x, hi_x = min(x1, x2), max(x1, x2)
        lo_y, hi_y = min(y1, y2), max(y1, y2)
        lo_z, hi_z = min(z1, z2), max(z1, z2)
        return lo_x, hi_x, lo_y, hi_y, lo_z, hi_z
    x = _int(op.get("x", 0), 0)
    y = _int(op.get("y", 0), 0)
    z = _int(op.get("z", 0), 0)
    return x, x, y, y, z, z


def _infer_operation_role_with_confidence(op: Dict[str, Any], palette: Dict[str, str], bbox: Dict[str, int]) -> Tuple[str, float, str]:
    explicit = _normalize_role(
        _get_first_value(
            op,
            ("role", "material_role", "palette_role", "semantic_role"),
        )
    )
    if explicit in palette:
        return explicit, 1.0, "explicit_role"

    purpose = str(op.get("purpose", "")).strip().lower()
    purpose_role = _infer_role_from_purpose(purpose)
    if purpose_role in palette:
        return purpose_role, 0.9, "purpose_hint"

    kind = str(op.get("op", "")).strip().lower()
    block = _normalize_block_type(op.get("block", "air"))
    block_role = _infer_role_from_block(block)
    if kind == "set" and block_role == "light":
        return "light", 0.95, "set_light"
    if kind == "set" and block_role in palette:
        return block_role, 0.7, "set_block_family"

    x1, x2, y1, y2, z1, z2 = _op_bounds_for_role_infer(op)
    dx, dy, dz = (x2 - x1 + 1), (y2 - y1 + 1), (z2 - z1 + 1)
    top_y = bbox["ymax"]
    bottom_y = bbox["ymin"]
    y_center = 0.5 * (y1 + y2)
    y_span = max(1.0, float(top_y - bottom_y + 1))
    y_ratio = (y_center - bottom_y) / y_span
    touches_top = y2 >= top_y - 1
    near_ground = y1 <= bottom_y + 1
    boundary_touches = int(x1 <= bbox["xmin"]) + int(x2 >= bbox["xmax"]) + int(z1 <= bbox["zmin"]) + int(z2 >= bbox["zmax"])
    flat_layer = dy <= 1 and dx * dz >= 9
    thin_strip = min(dx, dy, dz) == 1

    if kind == "fill":
        if block_role == "glass" or "window" in purpose:
            return "glass", 0.9, "window_or_glass"
        if block_role == "light":
            return "light", 0.9, "light_block"
        if near_ground and flat_layer:
            return "floor", 0.88, "ground_flat_layer"
        if touches_top and (flat_layer or y_ratio > 0.62):
            return "roof", 0.86, "top_layer"
        if boundary_touches >= 2 and dy >= 2:
            return "wall", 0.8, "perimeter_shell"
        if thin_strip and boundary_touches >= 1:
            return "trim", 0.68, "thin_outer_detail"
        if block_role in palette:
            conf = 0.62
            if block_role == "floor" and y_ratio > 0.55:
                return "roof", 0.58, "wood_upper_ambiguous"
            return block_role, conf, "block_family"

    if kind == "carve":
        if "window" in purpose:
            return "glass", 0.75, "carve_window"
        if "entrance" in purpose or "door" in purpose:
            return "wall", 0.7, "carve_entrance"

    inferred = _infer_operation_role(op, palette)
    if inferred in palette:
        return inferred, 0.55, "fallback_infer"
    return "", 0.0, "unknown"


def _is_inside_bbox(op: Dict[str, Any], bbox: Dict[str, int]) -> bool:
    kind = str(op.get("op", "")).strip().lower()
    if kind in {"fill", "carve"}:
        x1 = _int(op.get("x1", 0), 0)
        y1 = _int(op.get("y1", 0), 0)
        z1 = _int(op.get("z1", 0), 0)
        x2 = _int(op.get("x2", x1), x1)
        y2 = _int(op.get("y2", y1), y1)
        z2 = _int(op.get("z2", z1), z1)
        lo_x, hi_x = min(x1, x2), max(x1, x2)
        lo_y, hi_y = min(y1, y2), max(y1, y2)
        lo_z, hi_z = min(z1, z2), max(z1, z2)
        return (
            lo_x >= bbox["xmin"]
            and hi_x <= bbox["xmax"]
            and lo_y >= bbox["ymin"]
            and hi_y <= bbox["ymax"]
            and lo_z >= bbox["zmin"]
            and hi_z <= bbox["zmax"]
        )

    x = _int(op.get("x", 0), 0)
    y = _int(op.get("y", 0), 0)
    z = _int(op.get("z", 0), 0)
    return (
        bbox["xmin"] <= x <= bbox["xmax"]
        and bbox["ymin"] <= y <= bbox["ymax"]
        and bbox["zmin"] <= z <= bbox["zmax"]
    )


def _expand_bbox_with_op(bbox: Dict[str, int], op: Dict[str, Any]) -> Dict[str, int]:
    out = dict(bbox)
    kind = str(op.get("op", "")).strip().lower()
    if kind in {"fill", "carve"}:
        x1 = _int(op.get("x1", 0), 0)
        y1 = _int(op.get("y1", 0), 0)
        z1 = _int(op.get("z1", 0), 0)
        x2 = _int(op.get("x2", x1), x1)
        y2 = _int(op.get("y2", y1), y1)
        z2 = _int(op.get("z2", z1), z1)
        out["xmin"] = min(out["xmin"], x1, x2)
        out["xmax"] = max(out["xmax"], x1, x2)
        out["ymin"] = min(out["ymin"], y1, y2)
        out["ymax"] = max(out["ymax"], y1, y2)
        out["zmin"] = min(out["zmin"], z1, z2)
        out["zmax"] = max(out["zmax"], z1, z2)
    else:
        x = _int(op.get("x", 0), 0)
        y = _int(op.get("y", 0), 0)
        z = _int(op.get("z", 0), 0)
        out["xmin"] = min(out["xmin"], x)
        out["xmax"] = max(out["xmax"], x)
        out["ymin"] = min(out["ymin"], y)
        out["ymax"] = max(out["ymax"], y)
        out["zmin"] = min(out["zmin"], z)
        out["zmax"] = max(out["zmax"], z)
    return out


def _desc_text_blob(desc: Dict[str, Any]) -> str:
    parts: List[str] = []
    if isinstance(desc.get("summary"), str):
        parts.append(str(desc.get("summary", "")))
    if isinstance(desc.get("building_type"), str):
        parts.append(str(desc.get("building_type", "")))
    for key in ("elements", "rebuild_hints", "uncertainties"):
        values = desc.get(key, [])
        if isinstance(values, list):
            parts.extend(str(x) for x in values)
    shape = desc.get("shape")
    if isinstance(shape, dict):
        parts.extend(str(v) for v in shape.values())
    return " ".join(parts).lower()


def _infer_shape_preferences(desc: Dict[str, Any]) -> Dict[str, Any]:
    shape = desc.get("shape", {}) if isinstance(desc.get("shape"), dict) else {}
    roof_type = str(shape.get("roof_type", "")).strip().lower()
    floors = _clamp(_int(shape.get("floors_estimate", 1), 1), 1, 5)
    text = _desc_text_blob(desc)
    if not roof_type or roof_type == "unknown":
        if "gable" in text:
            roof_type = "gable"
        elif "hip" in text:
            roof_type = "hip"
        elif "flat" in text:
            roof_type = "flat"
        else:
            roof_type = "flat"
    return {"roof_type": roof_type, "floors": floors, "text": text}


def _coarse_stage_rank(op: Dict[str, Any], bbox: Dict[str, int]) -> int:
    kind = str(op.get("op", "")).strip().lower()
    role = _normalize_role(op.get("role", ""))
    purpose = str(op.get("purpose", "")).strip().lower()
    x1, x2, y1, y2, _z1, _z2 = _op_bounds_for_role_infer(op)
    top = bbox["ymax"]
    bottom = bbox["ymin"]

    if kind == "carve":
        if "entrance" in purpose or "door" in purpose:
            return 2
        return 1

    if role in {"floor", "wall", "roof"}:
        if role == "floor":
            return 0
        if role == "wall":
            return 1
        return 2

    if role in {"trim", "glass", "light"}:
        return 4

    if y2 <= bottom + 1:
        return 0
    if y1 >= top - 1:
        return 2
    return 3


def _expand_roof_template(
    op: Dict[str, Any],
    bbox: Dict[str, int],
    palette: Dict[str, str],
    role: str,
    purpose: str,
) -> List[Dict[str, Any]]:
    roof_type = str(_get_first_value(op, ("roof_type", "template", "style", "kind")) or "gable").strip().lower()
    base_y = _int(_get_first_value(op, ("base_y", "y", "y1")), bbox["ymax"] - 1)
    x1 = _int(_get_first_value(op, ("x1", "xmin")), bbox["xmin"])
    x2 = _int(_get_first_value(op, ("x2", "xmax")), bbox["xmax"])
    z1 = _int(_get_first_value(op, ("z1", "zmin")), bbox["zmin"])
    z2 = _int(_get_first_value(op, ("z2", "zmax")), bbox["zmax"])
    if x2 < x1:
        x1, x2 = x2, x1
    if z2 < z1:
        z1, z2 = z2, z1
    block = _normalize_block_type(_extract_block(op, palette.get("roof", DEFAULT_ROLE_BLOCKS["roof"])))
    role_fixed = role or "roof"
    layers = max(1, min(8, _int(op.get("layers", 4), 4)))

    out: List[Dict[str, Any]] = []
    if roof_type in {"flat", "unknown"}:
        out.append(
            {
                "op": "fill",
                "x1": x1,
                "y1": base_y,
                "z1": z1,
                "x2": x2,
                "y2": base_y,
                "z2": z2,
                "block": block,
                "role": role_fixed,
                "purpose": purpose or "roof_template_flat",
            }
        )
        return out

    if roof_type in {"gable", "hip", "dome", "other"}:
        for i in range(layers):
            rx1 = x1 + i
            rx2 = x2 - i
            if roof_type == "gable":
                rz1 = z1
                rz2 = z2
            else:
                rz1 = z1 + i
                rz2 = z2 - i
            ry = base_y + i
            if rx2 < rx1 or rz2 < rz1:
                break
            out.append(
                {
                    "op": "fill",
                    "x1": rx1,
                    "y1": ry,
                    "z1": rz1,
                    "x2": rx2,
                    "y2": ry,
                    "z2": rz2,
                    "block": block,
                    "role": role_fixed,
                    "purpose": purpose or f"roof_template_{roof_type}",
                }
            )
        return out

    # Fallback as one top layer.
    out.append(
        {
            "op": "fill",
            "x1": x1,
            "y1": base_y,
            "z1": z1,
            "x2": x2,
            "y2": base_y,
            "z2": z2,
            "block": block,
            "role": role_fixed,
            "purpose": purpose or "roof_template_fallback",
        }
    )
    return out


def _expand_window_pattern(
    op: Dict[str, Any],
    bbox: Dict[str, int],
    palette: Dict[str, str],
    role: str,
    purpose: str,
) -> List[Dict[str, Any]]:
    face = str(_get_first_value(op, ("face", "side")) or "all").strip().lower()
    spacing = max(2, _int(op.get("spacing", 3), 3))
    win_h = max(1, min(3, _int(op.get("window_height", 2), 2)))
    y_base = _int(op.get("y", bbox["ymin"] + 2), bbox["ymin"] + 2)
    y1 = max(bbox["ymin"] + 1, y_base)
    y2 = min(bbox["ymax"] - 1, y1 + win_h - 1)
    block = _normalize_block_type(_extract_block(op, palette.get("glass", DEFAULT_ROLE_BLOCKS["glass"])))
    role_fixed = role or "glass"

    faces = ["north", "south", "east", "west"] if face in {"all", "perimeter", ""} else [face]
    out: List[Dict[str, Any]] = []
    for f in faces:
        if f in {"north", "south"}:
            z = bbox["zmin"] if f == "north" else bbox["zmax"]
            for x in range(bbox["xmin"] + 1, bbox["xmax"], spacing):
                out.append(
                    {
                        "op": "fill",
                        "x1": x,
                        "y1": y1,
                        "z1": z,
                        "x2": x,
                        "y2": y2,
                        "z2": z,
                        "block": block,
                        "role": role_fixed,
                        "purpose": purpose or f"window_pattern_{f}",
                    }
                )
        elif f in {"east", "west"}:
            x = bbox["xmin"] if f == "west" else bbox["xmax"]
            for z in range(bbox["zmin"] + 1, bbox["zmax"], spacing):
                out.append(
                    {
                        "op": "fill",
                        "x1": x,
                        "y1": y1,
                        "z1": z,
                        "x2": x,
                        "y2": y2,
                        "z2": z,
                        "block": block,
                        "role": role_fixed,
                        "purpose": purpose or f"window_pattern_{f}",
                    }
                )
    return out


def _expand_slope(
    op: Dict[str, Any],
    bbox: Dict[str, int],
    palette: Dict[str, str],
    role: str,
    purpose: str,
) -> List[Dict[str, Any]]:
    axis = str(_get_first_value(op, ("axis", "ridge_axis")) or "x").strip().lower()
    steps = max(1, min(8, _int(op.get("steps", 4), 4)))
    base_y = _int(_get_first_value(op, ("base_y", "y", "y1")), bbox["ymax"] - 1)
    block = _normalize_block_type(_extract_block(op, palette.get("roof", DEFAULT_ROLE_BLOCKS["roof"])))
    role_fixed = role or "roof"
    out: List[Dict[str, Any]] = []

    for i in range(steps):
        if axis == "z":
            z1 = bbox["zmin"] + i
            z2 = bbox["zmax"] - i
            if z2 < z1:
                break
            out.append(
                {
                    "op": "fill",
                    "x1": bbox["xmin"],
                    "y1": base_y + i,
                    "z1": z1,
                    "x2": bbox["xmax"],
                    "y2": base_y + i,
                    "z2": z2,
                    "block": block,
                    "role": role_fixed,
                    "purpose": purpose or "slope_roof",
                }
            )
        else:
            x1 = bbox["xmin"] + i
            x2 = bbox["xmax"] - i
            if x2 < x1:
                break
            out.append(
                {
                    "op": "fill",
                    "x1": x1,
                    "y1": base_y + i,
                    "z1": bbox["zmin"],
                    "x2": x2,
                    "y2": base_y + i,
                    "z2": bbox["zmax"],
                    "block": block,
                    "role": role_fixed,
                    "purpose": purpose or "slope_roof",
                }
            )
    return out


def _self_repair_pass(
    ops: List[Dict[str, Any]],
    *,
    bbox: Dict[str, int],
    palette: Dict[str, str],
    desc: Dict[str, Any],
    max_operations: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    report: Dict[str, Any] = {
        "added_operations": 0,
        "added_tags": [],
        "reordered_two_stage": False,
        "window_block_fixes": 0,
        "entrance_fixes": 0,
    }
    out = [dict(op) for op in ops if isinstance(op, dict)]

    has_role = {r: False for r in REQUIRED_PALETTE_ROLES}
    has_window = False
    has_entrance = False
    has_light = False
    for op in out:
        kind = str(op.get("op", "")).strip().lower()
        if kind not in {"fill", "set", "carve"}:
            continue
        role = _normalize_role(op.get("role", "")) or _infer_operation_role(op, palette)
        if kind != "carve" and role in has_role:
            has_role[role] = True
        purpose = str(op.get("purpose", "")).strip().lower()
        if role == "glass" or "window" in purpose:
            has_window = True
        if kind == "carve" and ("entrance" in purpose or "door" in purpose):
            has_entrance = True
        if role == "light":
            has_light = True

    prefs = _infer_shape_preferences(desc)
    text = prefs["text"]
    want_windows = ("window" in text) or has_window
    want_entrance = ("entrance" in text or "door" in text or "entry" in text) or has_entrance

    def add_op(op: Dict[str, Any], tag: str) -> None:
        nonlocal out
        if len(out) >= max_operations:
            return
        out.append(op)
        report["added_operations"] += 1
        if len(report["added_tags"]) < 20:
            report["added_tags"].append(tag)

    if not has_role.get("floor", False):
        add_op(
            {
                "op": "fill",
                "x1": bbox["xmin"],
                "y1": bbox["ymin"],
                "z1": bbox["zmin"],
                "x2": bbox["xmax"],
                "y2": bbox["ymin"],
                "z2": bbox["zmax"],
                "block": palette["floor"],
                "role": "floor",
                "role_confidence": 0.95,
                "role_reason": "self_repair_floor",
                "purpose": "self_repair_floor",
            },
            "floor",
        )

    if not has_role.get("wall", False):
        add_op(
            {
                "op": "fill",
                "x1": bbox["xmin"],
                "y1": bbox["ymin"] + 1,
                "z1": bbox["zmin"],
                "x2": bbox["xmax"],
                "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                "z2": bbox["zmax"],
                "block": palette["wall"],
                "role": "wall",
                "role_confidence": 0.9,
                "role_reason": "self_repair_wall",
                "purpose": "self_repair_outer_shell",
            },
            "wall",
        )

    if not has_role.get("roof", False):
        roof_layers = _expand_roof_template(
            {"roof_type": prefs["roof_type"], "base_y": max(bbox["ymin"] + 3, bbox["ymax"] - 2), "layers": 4},
            bbox,
            palette,
            "roof",
            "self_repair_roof",
        )
        for rop in roof_layers:
            add_op(rop, "roof")

    if want_windows and not has_window:
        for wop in _expand_window_pattern({"face": "all", "spacing": 3, "window_height": 2, "y": bbox["ymin"] + 2}, bbox, palette, "glass", "self_repair_window"):
            add_op(wop, "window")

    if want_entrance and not has_entrance:
        door_x = (bbox["xmin"] + bbox["xmax"]) // 2
        add_op(
            {
                "op": "carve",
                "x1": door_x,
                "y1": bbox["ymin"] + 1,
                "z1": bbox["zmin"],
                "x2": door_x,
                "y2": min(bbox["ymin"] + 3, bbox["ymax"] - 1),
                "z2": bbox["zmin"],
                "block": "air",
                "purpose": "self_repair_entrance",
            },
            "entrance",
        )
        report["entrance_fixes"] += 1

    if not has_light:
        for cx, cz in (
            (bbox["xmin"], bbox["zmin"]),
            (bbox["xmax"], bbox["zmin"]),
            (bbox["xmin"], bbox["zmax"]),
            (bbox["xmax"], bbox["zmax"]),
        ):
            add_op(
                {
                    "op": "set",
                    "x": cx,
                    "y": min(bbox["ymax"], bbox["ymin"] + 2),
                    "z": cz,
                    "block": palette["light"],
                    "role": "light",
                    "role_confidence": 0.9,
                    "role_reason": "self_repair_light",
                    "purpose": "self_repair_light",
                },
                "light",
            )

    # Material cleanup for obvious window mismatches.
    for op in out:
        kind = str(op.get("op", "")).strip().lower()
        if kind == "carve":
            continue
        purpose = str(op.get("purpose", "")).strip().lower()
        role = _normalize_role(op.get("role", "")) or _infer_operation_role(op, palette)
        if (role == "glass" or "window" in purpose) and _normalize_block_type(op.get("block", "")) != _normalize_block_type(palette["glass"]):
            op["block"] = palette["glass"]
            op["role"] = "glass"
            op["role_confidence"] = max(float(op.get("role_confidence", 0.0) or 0.0), 0.92)
            op["role_reason"] = "self_repair_window_material"
            report["window_block_fixes"] += 1

    indexed = list(enumerate(out))
    indexed.sort(key=lambda it: (_coarse_stage_rank(it[1], bbox), it[0]))
    out_sorted = [op for _idx, op in indexed]
    report["reordered_two_stage"] = True

    if len(out_sorted) > max_operations:
        out_sorted = out_sorted[:max_operations]
    return out_sorted, report


def _validate_and_repair_plan(
    plan: Dict[str, Any],
    *,
    desc: Optional[Dict[str, Any]],
    strict_schema: bool,
    enforce_role_fixed: bool,
    require_material_budget: bool,
    material_budget_tolerance: float,
    role_fix_min_confidence: float,
    prefer_description_palette: bool,
    max_operations: int,
    required_palette_roles: Tuple[str, ...],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = dict(plan)
    desc_hints = _extract_desc_material_hints(desc if isinstance(desc, dict) else {}, required_palette_roles)
    palette_raw = out.get("palette", {}) if isinstance(out.get("palette"), dict) else {}
    palette, palette_report = _normalize_palette(
        palette_raw,
        required_palette_roles,
        desc_hints=desc_hints,
        prefer_description_palette=bool(prefer_description_palette),
    )
    out["palette"] = palette

    raw_ops = out.get("operations", []) if isinstance(out.get("operations"), list) else []
    trimmed = False
    if len(raw_ops) > max_operations:
        raw_ops = raw_ops[:max_operations]
        trimmed = True

    bbox = out.get("bbox", {}) if isinstance(out.get("bbox"), dict) else {}
    bbox_fixed = {
        "xmin": _int(bbox.get("xmin", 0), 0),
        "xmax": _int(bbox.get("xmax", 15), 15),
        "ymin": _int(bbox.get("ymin", 0), 0),
        "ymax": _int(bbox.get("ymax", 12), 12),
        "zmin": _int(bbox.get("zmin", 0), 0),
        "zmax": _int(bbox.get("zmax", 15), 15),
    }
    if bbox_fixed["xmax"] < bbox_fixed["xmin"]:
        bbox_fixed["xmin"], bbox_fixed["xmax"] = bbox_fixed["xmax"], bbox_fixed["xmin"]
    if bbox_fixed["ymax"] < bbox_fixed["ymin"]:
        bbox_fixed["ymin"], bbox_fixed["ymax"] = bbox_fixed["ymax"], bbox_fixed["ymin"]
    if bbox_fixed["zmax"] < bbox_fixed["zmin"]:
        bbox_fixed["zmin"], bbox_fixed["zmax"] = bbox_fixed["zmax"], bbox_fixed["zmin"]

    role_actual_blocks: Dict[str, int] = {r: 0 for r in required_palette_roles}
    assigned_role_count = 0
    unknown_role_count = 0
    role_fixed_block_count = 0
    role_fixed_skipped_low_conf_count = 0
    bbox_outside_count = 0
    role_infer_reasons: Dict[str, int] = {}
    role_infer_conf_sum = 0.0
    role_infer_conf_count = 0
    normalized_ops: List[Dict[str, Any]] = []

    for op in raw_ops:
        if not isinstance(op, dict):
            continue
        fixed = dict(op)
        kind = str(fixed.get("op", "")).strip().lower()
        if kind not in {"fill", "carve", "set"}:
            continue
        fixed["op"] = kind

        if kind == "carve":
            fixed["block"] = "air"
        else:
            fixed["block"] = _normalize_block_type(fixed.get("block", "stonebrick"))

        role, role_conf, role_reason = _infer_operation_role_with_confidence(fixed, palette, bbox_fixed)
        role_infer_reasons[role_reason] = int(role_infer_reasons.get(role_reason, 0)) + 1
        role_infer_conf_sum += float(role_conf)
        role_infer_conf_count += 1
        if not role and kind != "carve":
            # Prefer wall as deterministic fallback for untagged structural fills.
            role = "wall"
            role_conf = max(role_conf, 0.35)
            role_reason = "fallback_wall"

        if role in palette and kind != "carve":
            assigned_role_count += 1
            fixed["role"] = role
            fixed["role_confidence"] = round(float(role_conf), 4)
            fixed["role_reason"] = role_reason
            if enforce_role_fixed and _normalize_block_type(fixed.get("block", "")) != _normalize_block_type(palette[role]):
                if float(role_conf) >= float(role_fix_min_confidence):
                    fixed["block"] = palette[role]
                    role_fixed_block_count += 1
                else:
                    role_fixed_skipped_low_conf_count += 1
            role_actual_blocks[role] += _op_volume(fixed)
        elif kind != "carve":
            unknown_role_count += 1

        if not _is_inside_bbox(fixed, bbox_fixed):
            bbox_outside_count += 1
            if strict_schema:
                bbox_fixed = _expand_bbox_with_op(bbox_fixed, fixed)

        normalized_ops.append(fixed)

    normalized_ops, self_repair_report = _self_repair_pass(
        normalized_ops,
        bbox=bbox_fixed,
        palette=palette,
        desc=desc if isinstance(desc, dict) else {},
        max_operations=max_operations,
    )
    role_fixed_block_count += int(self_repair_report.get("window_block_fixes", 0) or 0)

    # Recompute role/material stats after self-repair pass.
    role_actual_blocks = {r: 0 for r in required_palette_roles}
    assigned_role_count = 0
    unknown_role_count = 0
    for op in normalized_ops:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op", "")).strip().lower()
        if kind == "carve":
            continue
        role = _normalize_role(op.get("role", "")) or _infer_operation_role(op, palette)
        if role in required_palette_roles:
            assigned_role_count += 1
            op["role"] = role
            role_actual_blocks[role] += _op_volume(op)
        else:
            unknown_role_count += 1

    out["operations"] = normalized_ops
    out["bbox"] = bbox_fixed

    input_budget, budget_source = _extract_material_budget(plan, required_palette_roles)
    budget_present = bool(input_budget)
    material_budget: Dict[str, Dict[str, Any]] = {}
    budget_violations: List[Dict[str, Any]] = []

    for role in required_palette_roles:
        input_item = input_budget.get(role, {}) if isinstance(input_budget.get(role), dict) else {}
        target = max(0, _int(input_item.get("target_blocks", role_actual_blocks.get(role, 0)), role_actual_blocks.get(role, 0)))
        if not budget_present and require_material_budget:
            target = role_actual_blocks.get(role, 0)
        block = _normalize_block_type(input_item.get("block", palette.get(role, DEFAULT_ROLE_BLOCKS[role])))
        if enforce_role_fixed:
            block = palette.get(role, block)

        actual = role_actual_blocks.get(role, 0)
        rel_err = 0.0 if target == 0 else abs(actual - target) / float(max(1, target))
        within = rel_err <= material_budget_tolerance
        if target == 0 and actual > 0:
            within = False
            rel_err = 1.0
        if not within:
            budget_violations.append(
                {
                    "role": role,
                    "target_blocks": target,
                    "actual_blocks": actual,
                    "relative_error": round(rel_err, 6),
                }
            )
        material_budget[role] = {
            "role": role,
            "block": block,
            "target_blocks": target,
            "actual_blocks": actual,
            "within_tolerance": within,
        }

    out["material_budget"] = material_budget

    self_check = out.get("self_check", {})
    if not isinstance(self_check, dict):
        self_check = {}
    existing_issues = self_check.get("issues", [])
    if not isinstance(existing_issues, list):
        existing_issues = []
    issues = [str(x) for x in existing_issues]
    if trimmed:
        issues.append(f"operations_trimmed_to_{max_operations}")
    if bbox_outside_count > 0:
        issues.append(f"bbox_expanded_for_{bbox_outside_count}_ops")
    if budget_violations:
        issues.append(f"material_budget_violations={len(budget_violations)}")
    if unknown_role_count > 0:
        issues.append(f"unknown_role_operations={unknown_role_count}")

    self_check["operation_count"] = len(normalized_ops)
    self_check["operation_count_ok"] = len(normalized_ops) <= max_operations
    self_check["bbox_consistency"] = "ok" if bbox_outside_count == 0 else "expanded_to_fit_operations"
    self_check["role_fixed_consistency"] = "ok" if role_fixed_block_count == 0 else f"repaired_{role_fixed_block_count}"
    self_check["material_budget_consistency"] = "ok" if not budget_violations else "warning"
    self_check["required_roles_present"] = all(r in palette for r in required_palette_roles)
    self_check["issues"] = issues
    out["self_check"] = self_check

    schema_violations: List[str] = []
    if not normalized_ops:
        schema_violations.append("operations_empty")
    if trimmed:
        schema_violations.append("operations_exceeded_max_and_trimmed")
    if require_material_budget and not budget_present:
        schema_violations.append("material_budget_missing_in_input_repaired")
    if any(r not in palette for r in required_palette_roles):
        schema_violations.append("palette_missing_required_roles")

    strict_blocking_issues: List[str] = []
    if not normalized_ops:
        strict_blocking_issues.append("operations_empty")
    if strict_schema and require_material_budget and budget_violations:
        strict_blocking_issues.append("material_budget_violation")

    validation_report = {
        "strict_schema": bool(strict_schema),
        "enforce_role_fixed": bool(enforce_role_fixed),
        "require_material_budget": bool(require_material_budget),
        "material_budget_tolerance": float(material_budget_tolerance),
        "role_fix_min_confidence": float(role_fix_min_confidence),
        "prefer_description_palette": bool(prefer_description_palette),
        "max_operations": int(max_operations),
        "required_palette_roles": list(required_palette_roles),
        "description_material_hints": desc_hints,
        "palette_report": palette_report,
        "budget_source": budget_source,
        "budget_present_in_input": budget_present,
        "operations_trimmed": bool(trimmed),
        "operations_assigned_role_count": assigned_role_count,
        "operations_unknown_role_count": unknown_role_count,
        "role_fixed_block_count": role_fixed_block_count,
        "role_fixed_skipped_low_conf_count": role_fixed_skipped_low_conf_count,
        "role_infer_confidence_mean": (role_infer_conf_sum / role_infer_conf_count) if role_infer_conf_count > 0 else 0.0,
        "role_infer_reasons": role_infer_reasons,
        "self_repair_report": self_repair_report,
        "bbox_outside_operation_count": bbox_outside_count,
        "budget_violations": budget_violations,
        "schema_violations": schema_violations,
        "strict_blocking_issues": strict_blocking_issues,
        "valid_strict": len(strict_blocking_issues) == 0,
    }
    out["validation_report"] = validation_report
    return out, validation_report


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


def _strip_markdown_fence(text: str) -> str:
    raw = text.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip().startswith("```"):
        body = "\n".join(lines[1:-1]).strip()
        if body:
            return body
    return raw


def _strip_json_comments(text: str) -> str:
    # Remove JS-style comments conservatively outside of quoted strings.
    out: List[str] = []
    i = 0
    n = len(text)
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _remove_trailing_commas(text: str) -> str:
    prev = text
    while True:
        curr = re.sub(r",(\s*[}\]])", r"\1", prev)
        if curr == prev:
            return curr
        prev = curr


def _to_python_literal_jsonish(text: str) -> str:
    out = text
    out = re.sub(r"\btrue\b", "True", out, flags=re.IGNORECASE)
    out = re.sub(r"\bfalse\b", "False", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnull\b", "None", out, flags=re.IGNORECASE)
    return out


def _try_load_jsonish(text: str) -> Optional[Any]:
    raw = _strip_markdown_fence(text.strip())
    if not raw:
        return None

    candidates = [raw, _strip_json_comments(raw)]
    candidates.append(_remove_trailing_commas(candidates[-1]))
    # Common quote variants from LLM answers.
    candidates.append(candidates[-1].replace("“", '"').replace("”", '"').replace("’", "'"))

    seen: Set[str] = set()
    uniq: List[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            uniq.append(cand)

    for cand in uniq:
        try:
            return json.loads(cand)
        except Exception:
            continue

    for cand in uniq:
        try:
            obj = ast.literal_eval(_to_python_literal_jsonish(cand))
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            return obj

    return None


def _is_likely_plan_dict(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    if not keys:
        return False
    if keys & {"operations", "ops", "actions", "steps", "commands", "build_ops"}:
        return True
    if keys & {"bbox", "bounds", "bounding_box", "box"} and keys & {"palette", "materials"}:
        return True
    if keys & {"plan_version", "material_budget", "self_check"} and keys & {"bbox", "palette"}:
        return True
    if "structure_spec" in keys and keys & {"bbox", "palette"}:
        return True
    return False


def _iter_nested_plan_dicts(root: Dict[str, Any], max_depth: int = 4) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    queue: List[Tuple[str, Any, int]] = [("root", root, 0)]
    seen_ids: Set[int] = set()
    while queue:
        path, node, depth = queue.pop(0)
        if isinstance(node, dict):
            oid = id(node)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            out.append((path, node))
            if depth >= max_depth:
                continue
            for k, v in node.items():
                if isinstance(v, dict):
                    queue.append((f"{path}.{k}", v, depth + 1))
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            queue.append((f"{path}.{k}[{i}]", item, depth + 1))
        elif isinstance(node, list) and depth < max_depth:
            for i, item in enumerate(node):
                if isinstance(item, (dict, list)):
                    queue.append((f"{path}[{i}]", item, depth + 1))
    return out


def _plan_candidates(plan: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    # Keep deterministic order: root, common wrappers, then nested likely plan dicts.
    candidates: List[Tuple[str, Dict[str, Any]]] = [("root", plan)]
    seen_ids: Set[int] = {id(plan)}
    for key in (
        "plan",
        "final_plan",
        "result",
        "output",
        "revised_plan",
        "build_plan",
        "json",
        "data",
        "response",
        "content",
    ):
        value = plan.get(key)
        if isinstance(value, dict) and id(value) not in seen_ids:
            seen_ids.add(id(value))
            candidates.append((f"root.{key}", value))

    for path, node in _iter_nested_plan_dicts(plan, max_depth=4):
        if id(node) in seen_ids:
            continue
        if _is_likely_plan_dict(node):
            seen_ids.add(id(node))
            candidates.append((path, node))
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

    def _extract_from_dict_payload(raw: Dict[str, Any], source: str) -> Optional[Tuple[List[Any], str]]:
        # Common wrappers: {"operations":{"items":[...]}} or {"operations":{"main":[...], "detail":[...]}}
        for sub in ("items", "list", "value", "values", "data", "ops", "operations", "steps", "commands"):
            v = raw.get(sub)
            if isinstance(v, list):
                return v, f"{source}.{sub}"
            if isinstance(v, str):
                loaded = _try_load_jsonish(v)
                if isinstance(loaded, list):
                    return loaded, f"{source}.{sub}(jsonish)"

        merged: List[Any] = []
        for v in raw.values():
            if isinstance(v, list):
                if any(isinstance(x, dict) for x in v):
                    merged.extend(v)
            elif isinstance(v, dict):
                nested = _extract_from_dict_payload(v, source)
                if nested is not None:
                    merged.extend(nested[0])
        if merged:
            return merged, f"{source}.dict_merged_lists"
        return None

    for source, cand in _plan_candidates(plan):
        for key in op_keys:
            raw = cand.get(key)
            if isinstance(raw, list):
                return raw, f"{source}.{key}"
            if isinstance(raw, dict):
                extracted = _extract_from_dict_payload(raw, f"{source}.{key}")
                if extracted is not None:
                    return extracted
            if isinstance(raw, str):
                loaded = _try_load_jsonish(raw)
                if isinstance(loaded, list):
                    return loaded, f"{source}.{key}(jsonish)"
                if isinstance(loaded, dict):
                    extracted = _extract_from_dict_payload(loaded, f"{source}.{key}(jsonish_dict)")
                    if extracted is not None:
                        return extracted
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


def _extract_role(op: Dict[str, Any]) -> str:
    for key in ("role", "material_role", "palette_role", "semantic_role"):
        value = op.get(key)
        if value is None:
            continue
        role = _normalize_role(value)
        if role:
            return role
    return ""


def _extract_block(op: Dict[str, Any], default: str) -> str:
    def _block_from_any(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for sub in ("block", "material", "id", "name", "type", "value", "to", "target"):
                vv = value.get(sub)
                if vv is not None:
                    out = _block_from_any(vv)
                    if out:
                        return out
            return ""
        if isinstance(value, (list, tuple)):
            for vv in value:
                out = _block_from_any(vv)
                if out:
                    return out
            return ""
        text = str(value).strip()
        return text

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
        text = _block_from_any(value)
        if text:
            return text
    return default


def _normalize_raw_operation(raw_op: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    op = dict(raw_op)
    repaired = False

    # Unwrap frequent arg containers.
    for args_key in ("params", "arguments", "args", "payload", "data", "value"):
        args_val = op.get(args_key)
        if isinstance(args_val, dict):
            for k, v in args_val.items():
                if k not in op:
                    op[k] = v
            repaired = True

    # Style: {"operation": {...}} or {"step": {...}}.
    for wrap in ("operation", "step", "command"):
        wrap_val = op.get(wrap)
        if isinstance(wrap_val, dict):
            merged = dict(wrap_val)
            for k, v in op.items():
                if k == wrap:
                    continue
                merged.setdefault(k, v)
            op = merged
            repaired = True
            break

    kind_keys = ("op", "type", "action", "kind", "cmd", "command")
    has_kind = any(k in op and str(op.get(k)).strip() for k in kind_keys)

    # Style: {"fill": {...}} / {"carve": {...}} / {"set": {...}}
    if not has_kind and len(op) == 1:
        k, v = next(iter(op.items()))
        key_norm = str(k).strip().lower().replace("-", "_").replace(" ", "_")
        if isinstance(v, dict):
            merged = dict(v)
            merged.setdefault("op", key_norm)
            op = merged
            repaired = True
        elif isinstance(v, (list, tuple)) and len(v) >= 6:
            op = {
                "op": key_norm,
                "x1": v[0],
                "y1": v[1],
                "z1": v[2],
                "x2": v[3],
                "y2": v[4],
                "z2": v[5],
            }
            repaired = True

    # Style: {"target":"fill", ...}
    if not has_kind and "target" in op and isinstance(op.get("target"), str):
        op["op"] = op.get("target")
        repaired = True

    return op, repaired


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
        "place_box": "fill",
        "fill_box": "fill",
        "fill_cuboid": "fill",
        "add_cuboid": "fill",
        "add_box": "fill",
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
        "set_block": "set",
        "place_block": "set",
        "place": "set",
        "put": "set",
        "block": "set",
        "carve": "carve",
        "clear": "carve",
        "remove": "carve",
        "delete": "carve",
        "erase": "carve",
        "cut": "carve",
        "dig": "carve",
        "air": "carve",
        "entrance_cut": "carve",
        "clear_above": "carve",
        "outline": "outline",
        "shell": "outline",
        "hollow": "outline",
        "hollow_box": "outline",
        "frame": "outline",
        # High-level macro ops (expanded to fill/set/carve in _coerce_plan).
        "roof_template": "roof_template",
        "roof_pattern": "roof_template",
        "window_pattern": "window_pattern",
        "window_grid": "window_pattern",
        "slope": "slope",
        "slope_roof": "slope",
        "ramp": "slope",
    }
    normalized = alias.get(raw_kind, raw_kind)
    was_repaired = normalized != raw_kind or kind_key != "op"
    return normalized, raw_kind, was_repaired


def _parse_cuboid(op: Dict[str, Any], bbox: Dict[str, int]) -> Optional[Tuple[int, int, int, int, int, int, bool]]:
    repaired = False

    x1_raw = _get_first_value(op, ("x1", "xmin", "x_min", "min_x", "x0", "x_start", "start_x"))
    y1_raw = _get_first_value(op, ("y1", "ymin", "y_min", "min_y", "y0", "y_start", "start_y"))
    z1_raw = _get_first_value(op, ("z1", "zmin", "z_min", "min_z", "z0", "z_start", "start_z"))
    x2_raw = _get_first_value(op, ("x2", "xmax", "x_max", "max_x", "x_end", "end_x"))
    y2_raw = _get_first_value(op, ("y2", "ymax", "y_max", "max_y", "y_end", "end_y"))
    z2_raw = _get_first_value(op, ("z2", "zmax", "z_max", "max_z", "z_end", "end_z"))

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
        origin = _parse_xyz(op.get("origin")) or _parse_xyz(op.get("anchor"))
        size = op.get("size")
        if origin is not None and isinstance(size, (list, tuple)) and len(size) >= 3:
            ox, oy, oz = origin
            sx, sy, sz = max(1, _int(size[0], 1)), max(1, _int(size[1], 1)), max(1, _int(size[2], 1))
            x1_raw, y1_raw, z1_raw = ox, oy, oz
            x2_raw, y2_raw, z2_raw = ox + sx - 1, oy + sy - 1, oz + sz - 1
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


def _operations_from_structure_spec(plan: Dict[str, Any], bbox: Dict[str, int], palette: Dict[str, str]) -> Tuple[List[Dict[str, Any]], str]:
    for source, cand in _plan_candidates(plan):
        spec = cand.get("structure_spec")
        if not isinstance(spec, dict):
            continue

        x0, x1 = bbox["xmin"], bbox["xmax"]
        y0, y1 = bbox["ymin"], bbox["ymax"]
        z0, z1 = bbox["zmin"], bbox["zmax"]
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue

        wall = _normalize_block_type(palette.get("wall", "stonebrick"))
        roof = _normalize_block_type(palette.get("roof", "wood"))
        floor = _normalize_block_type(palette.get("floor", "wood"))
        glass = _normalize_block_type(palette.get("glass", "glass"))

        ops: List[Dict[str, Any]] = []
        # Coarse shell synthesis.
        ops.append({"op": "fill", "x1": x0, "y1": y0, "z1": z0, "x2": x1, "y2": y0, "z2": z1, "block": floor, "purpose": "spec_floor"})
        if y1 - y0 >= 2:
            ops.append({"op": "fill", "x1": x0, "y1": y0 + 1, "z1": z0, "x2": x1, "y2": y1 - 1, "z2": z0, "block": wall, "purpose": "spec_wall_north"})
            ops.append({"op": "fill", "x1": x0, "y1": y0 + 1, "z1": z1, "x2": x1, "y2": y1 - 1, "z2": z1, "block": wall, "purpose": "spec_wall_south"})
            ops.append({"op": "fill", "x1": x0, "y1": y0 + 1, "z1": z0, "x2": x0, "y2": y1 - 1, "z2": z1, "block": wall, "purpose": "spec_wall_west"})
            ops.append({"op": "fill", "x1": x1, "y1": y0 + 1, "z1": z0, "x2": x1, "y2": y1 - 1, "z2": z1, "block": wall, "purpose": "spec_wall_east"})
            if x1 - x0 >= 2 and z1 - z0 >= 2 and y1 - y0 >= 3:
                ops.append(
                    {
                        "op": "carve",
                        "x1": x0 + 1,
                        "y1": y0 + 1,
                        "z1": z0 + 1,
                        "x2": x1 - 1,
                        "y2": y1 - 1,
                        "z2": z1 - 1,
                        "block": "air",
                        "purpose": "spec_hollow_core",
                    }
                )
        ops.append({"op": "fill", "x1": x0, "y1": y1, "z1": z0, "x2": x1, "y2": y1, "z2": z1, "block": roof, "purpose": "spec_roof"})

        # Openings from structure_spec.openings
        openings = spec.get("openings")
        if isinstance(openings, list):
            for oi, opening in enumerate(openings[:96]):
                if not isinstance(opening, dict):
                    continue
                cub = _parse_cuboid(opening, bbox)
                if cub is None:
                    p = _parse_xyz(opening.get("pos")) or _parse_xyz(opening.get("position"))
                    if p is None:
                        continue
                    x, y, z = p
                    cub = (x, y, z, x, y, z, True)
                ox1, oy1, oz1, ox2, oy2, oz2, _ = cub
                kind = str(opening.get("kind", "")).strip().lower()
                if "door" in kind or "entrance" in kind:
                    ops.append(
                        {
                            "op": "carve",
                            "x1": ox1,
                            "y1": oy1,
                            "z1": oz1,
                            "x2": ox2,
                            "y2": oy2,
                            "z2": oz2,
                            "block": "air",
                            "purpose": f"spec_opening_{oi}",
                        }
                    )
                else:
                    ops.append(
                        {
                            "op": "fill",
                            "x1": ox1,
                            "y1": oy1,
                            "z1": oz1,
                            "x2": ox2,
                            "y2": oy2,
                            "z2": oz2,
                            "block": glass,
                            "purpose": f"spec_window_{oi}",
                        }
                    )

        # Roof profile explicit layers (if provided).
        roof_profile = spec.get("roof_profile")
        if isinstance(roof_profile, dict):
            layers = roof_profile.get("layers")
            if isinstance(layers, list):
                for li, layer in enumerate(layers[:32]):
                    if not isinstance(layer, dict):
                        continue
                    cub = _parse_cuboid(layer, bbox)
                    if cub is None:
                        continue
                    lx1, ly1, lz1, lx2, ly2, lz2, _ = cub
                    block = _extract_block(layer, roof)
                    ops.append(
                        {
                            "op": "fill",
                            "x1": lx1,
                            "y1": ly1,
                            "z1": lz1,
                            "x2": lx2,
                            "y2": ly2,
                            "z2": lz2,
                            "block": block,
                            "purpose": f"spec_roof_layer_{li}",
                        }
                    )

        if len(ops) >= 4:
            return ops, f"{source}.structure_spec"
    return [], "missing"


def _coerce_plan(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = dict(plan)
    bbox, bbox_source = _extract_bbox(plan)
    palette = _extract_palette(plan)
    raw_ops, ops_source = _extract_raw_operations(plan)
    if not raw_ops:
        spec_ops, spec_source = _operations_from_structure_spec(plan, bbox, palette)
        if spec_ops:
            raw_ops = spec_ops
            ops_source = spec_source

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

        op_obj, op_unwrapped = _normalize_raw_operation(raw_op)

        kind, raw_kind, kind_repaired = _parse_op_kind(op_obj)
        if not kind:
            report["dropped_operations_count"] += 1
            if len(report["dropped_examples"]) < 5:
                report["dropped_examples"].append({"index": idx, "reason": "missing_op_kind"})
            continue

        repaired = bool(kind_repaired or op_unwrapped)
        purpose = _extract_purpose(op_obj)
        role = _extract_role(op_obj)

        if kind in {"roof_template", "window_pattern", "slope"}:
            if kind == "roof_template":
                expanded = _expand_roof_template(op_obj, bbox, palette, role, purpose)
            elif kind == "window_pattern":
                expanded = _expand_window_pattern(op_obj, bbox, palette, role, purpose)
            else:
                expanded = _expand_slope(op_obj, bbox, palette, role, purpose)
            if not expanded:
                report["dropped_operations_count"] += 1
                if len(report["dropped_examples"]) < 5:
                    report["dropped_examples"].append({"index": idx, "reason": f"empty_{kind}_expansion"})
                continue
            fixed_ops.extend(expanded)
            report["accepted_operations_count"] += 1
            report["expanded_operations_count"] += max(0, len(expanded) - 1)
            report["repaired_operations_count"] += 1
            continue

        if kind in {"fill", "carve", "replace", "outline"}:
            cuboid = _parse_cuboid(op_obj, bbox)
            if cuboid is None:
                report["dropped_operations_count"] += 1
                if len(report["dropped_examples"]) < 5:
                    report["dropped_examples"].append({"index": idx, "reason": "missing_cuboid_coords", "raw_kind": raw_kind})
                continue
            x1, y1, z1, x2, y2, z2, coords_repaired = cuboid
            repaired = repaired or coords_repaired

            eff_kind = kind
            if kind == "replace":
                block_raw = _extract_block(op_obj, "stonebrick")
                if str(block_raw).strip().lower().endswith(":air") or str(block_raw).strip().lower() == "air":
                    eff_kind = "carve"
                else:
                    eff_kind = "fill"
                repaired = True

            if eff_kind == "outline":
                block = _extract_block(op_obj, "stonebrick")
                faces = _expand_outline_to_faces(x1, y1, z1, x2, y2, z2)
                for fi, (fx1, fy1, fz1, fx2, fy2, fz2) in enumerate(faces):
                    op_fixed = {
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
                    if role:
                        op_fixed["role"] = role
                    fixed_ops.append(op_fixed)
                report["accepted_operations_count"] += 1
                report["expanded_operations_count"] += max(0, len(faces) - 1)
                if repaired:
                    report["repaired_operations_count"] += 1
                continue

            block = "air" if eff_kind == "carve" else _extract_block(op_obj, "stonebrick")
            op_fixed = {
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
            if role:
                op_fixed["role"] = role
            fixed_ops.append(op_fixed)
            report["accepted_operations_count"] += 1
            if repaired:
                report["repaired_operations_count"] += 1
            continue

        if kind == "set":
            x, y, z, point_repaired = _parse_set_point(op_obj, bbox)
            repaired = repaired or point_repaired
            block = _extract_block(op_obj, "stonebrick")
            op_fixed = {"op": "set", "x": x, "y": y, "z": z, "block": str(block), "purpose": purpose}
            if role:
                op_fixed["role"] = role
            fixed_ops.append(op_fixed)
            report["accepted_operations_count"] += 1
            if repaired:
                report["repaired_operations_count"] += 1
            continue

        # Generic auto-repair: unknown op but coordinates + block exist.
        inferred_cuboid = _parse_cuboid(op_obj, bbox)
        if inferred_cuboid is not None:
            x1, y1, z1, x2, y2, z2, coords_repaired = inferred_cuboid
            block = _extract_block(op_obj, "stonebrick")
            block_l = str(block).strip().lower()
            inferred_kind = "carve" if (block_l == "air" or block_l.endswith(":air") or "clear" in raw_kind or "cut" in raw_kind) else "fill"
            op_fixed = {
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
            if role:
                op_fixed["role"] = role
            fixed_ops.append(op_fixed)
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
                    obj = _try_load_jsonish(cand)
                    if not isinstance(obj, dict):
                        break
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
        loaded = _try_load_jsonish(cand)
        if loaded is None:
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
        "jsonish_loader_used": False,
    }

    candidates: List[Dict[str, Any]] = []
    candidate_seen: Set[str] = set()

    def _append_candidate(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        sig = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:8000]
        if sig in candidate_seen:
            return
        candidate_seen.add(sig)
        candidates.append(obj)

    selected: Dict[str, Any] = {}
    try:
        first = extract_json_object(text)
        _append_candidate(first)
    except Exception:
        pass

    jsonish = _try_load_jsonish(text)
    if isinstance(jsonish, dict):
        _append_candidate(jsonish)
        parse_report["jsonish_loader_used"] = True

    # JSON fenced blocks.
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE):
        maybe_obj = _try_load_jsonish(m.group(1))
        if isinstance(maybe_obj, dict):
            _append_candidate(maybe_obj)
            parse_report["jsonish_loader_used"] = True

    scanned = _extract_all_json_dicts(text)
    parse_report["candidate_count"] = len(scanned)
    for cand in scanned:
        _append_candidate(cand)

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
        # Try to rescue operations from any candidate before scanning loose objects.
        for cand in candidates:
            cand_ops, _ = _extract_raw_operations(cand)
            if cand_ops:
                selected = dict(selected) if selected else {}
                selected["operations"] = cand_ops
                parse_report["method"] = "candidate_ops_rescue"
                parse_report["used_ops_array_repair"] = True
                break

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

    # Rescue structure_spec if selected plan lost it during operation extraction.
    if "structure_spec" not in selected:
        for cand in candidates:
            spec = cand.get("structure_spec")
            if isinstance(spec, dict):
                selected = dict(selected)
                selected["structure_spec"] = spec
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
    strict_schema = bool(prompt_profile.get("strict_schema", args.strict_schema))
    enforce_role_fixed = bool(prompt_profile.get("enforce_role_fixed", args.enforce_role_fixed))
    require_material_budget = bool(prompt_profile.get("require_material_budget", args.require_material_budget))
    try:
        material_budget_tolerance = max(0.0, float(prompt_profile.get("material_budget_tolerance", args.material_budget_tolerance)))
    except Exception:
        material_budget_tolerance = float(args.material_budget_tolerance)
    try:
        role_fix_min_confidence = max(0.0, min(1.0, float(prompt_profile.get("role_fix_min_confidence", args.role_fix_min_confidence))))
    except Exception:
        role_fix_min_confidence = float(args.role_fix_min_confidence)
    prefer_description_palette = bool(prompt_profile.get("prefer_description_palette", args.prefer_description_palette))
    try:
        max_operations = max(1, int(prompt_profile.get("max_operations", args.max_operations)))
    except Exception:
        max_operations = int(args.max_operations)
    profile_required_roles = prompt_profile.get("required_palette_roles", list(REQUIRED_PALETTE_ROLES))
    if isinstance(profile_required_roles, list):
        required_palette_roles = tuple(
            r for r in (_normalize_role(x) for x in profile_required_roles) if r in REQUIRED_PALETTE_ROLES
        )
    else:
        required_palette_roles = REQUIRED_PALETTE_ROLES
    if not required_palette_roles:
        required_palette_roles = REQUIRED_PALETTE_ROLES
    critic_revise_enabled = bool(
        args.critic_revise
        and str(prompt_profile.get("critic_system_prompt", "")).strip()
        and str(prompt_profile.get("critic_user_template", "")).strip()
    )
    print(
        "[generate_rebuild_plans] "
        f"prompt_profile={profile_name}@{profile_version} "
        f"critic_revise={critic_revise_enabled} "
        f"few_shots={len(few_shots)} "
        f"strict_schema={strict_schema} "
        f"role_fixed={enforce_role_fixed} "
        f"role_fix_min_conf={role_fix_min_confidence} "
        f"require_budget={require_material_budget} "
        f"prefer_desc_palette={prefer_description_palette} "
        f"max_ops={max_operations}"
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
                    llm_seed=(int(args.llm_seed) if int(args.llm_seed) >= 0 else None),
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
                            llm_seed=(int(args.llm_seed) if int(args.llm_seed) >= 0 else None),
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
                            "strict_schema": bool(strict_schema),
                            "enforce_role_fixed": bool(enforce_role_fixed),
                            "require_material_budget": bool(require_material_budget),
                            "material_budget_tolerance": float(material_budget_tolerance),
                            "role_fix_min_confidence": float(role_fix_min_confidence),
                            "prefer_description_palette": bool(prefer_description_palette),
                            "max_operations": int(max_operations),
                            "required_palette_roles": list(required_palette_roles),
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
        validation_report: Dict[str, Any] | None = None
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

        coerced, validation_report = _validate_and_repair_plan(
            coerced,
            desc=desc if isinstance(desc, dict) else {},
            strict_schema=bool(strict_schema),
            enforce_role_fixed=bool(enforce_role_fixed),
            require_material_budget=bool(require_material_budget),
            material_budget_tolerance=float(material_budget_tolerance),
            role_fix_min_confidence=float(role_fix_min_confidence),
            prefer_description_palette=bool(prefer_description_palette),
            max_operations=int(max_operations),
            required_palette_roles=tuple(required_palette_roles),
        )

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
                        "strict_schema": bool(strict_schema),
                        "enforce_role_fixed": bool(enforce_role_fixed),
                        "require_material_budget": bool(require_material_budget),
                        "material_budget_tolerance": float(material_budget_tolerance),
                        "role_fix_min_confidence": float(role_fix_min_confidence),
                        "prefer_description_palette": bool(prefer_description_palette),
                        "max_operations": int(max_operations),
                        "required_palette_roles": list(required_palette_roles),
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
                    "llm_seed": int(args.llm_seed),
                    "fallback_triggered": bool(fallback_triggered),
                    "coerce_report": coerce_report,
                    "fallback_coerce_report": fallback_report if fallback_triggered else None,
                    "validation_report": validation_report if validation_report is not None else {},
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
