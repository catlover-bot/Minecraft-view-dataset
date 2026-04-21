#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools.cost_logger import append_cost_event, estimate_usage_cost
from tools.llm_client import LLMError, complete_multimodal_with_meta, extract_json_object
from tools.llm_config import load_llm_config, model_for_provider, require_provider_key

SYSTEM_PROMPT = (
    "You are a Minecraft architectural source-spec author for a controlled diagnostic dataset. "
    "Return only one strict JSON object."
)

USER_PROMPT = """
Create exactly 10 Minecraft building source specifications for a diagnostic study.

Hard constraints:
- Return JSON object: {"cases": [ ... ]}
- Exactly 10 items in `cases`
- Difficulty distribution must be exactly:
  - 3 simple
  - 4 medium
  - 3 complex
- case_id must be llm_case_001 ... llm_case_010 (stable order)
- footprint.kind must be one of: rectangle, l_shape, u_shape, plus, ring
- roof.type must be one of: flat, gable_x, gable_z, hip
- all dimensions must be positive integers
- materials are Minecraft-like block ids without namespace

Schema for each case:
{
  "case_id": "llm_case_001",
  "title": "short title",
  "difficulty": "simple|medium|complex",
  "width": 14,
  "depth": 12,
  "floors": 1,
  "floor_height": 4,
  "footprint": {
    "kind": "rectangle|l_shape|u_shape|plus|ring",
    "notch_width": 0,
    "notch_depth": 0,
    "corner": "nw|ne|sw|se",
    "opening": "north|south|east|west",
    "gap_width": 0,
    "thickness": 3,
    "arm_width": 0,
    "arm_depth": 0
  },
  "roof": {"type": "flat|gable_x|gable_z|hip", "height": 2},
  "entrance": {"side": "north|south|east|west", "width": 1, "height": 2},
  "windows": {"pattern": "checker|stripe_x|stripe_z", "spacing": 3, "height": 2},
  "features": {"tower": false, "porch": false, "balcony": false},
  "materials": {
    "foundation": "stonebrick",
    "wall": "planks",
    "roof": "nether_brick",
    "window": "glass",
    "accent": "quartz_block",
    "trim": "stone_slab",
    "floor": "planks",
    "light": "glowstone",
    "pillar": "stonebrick"
  },
  "notes": "one short sentence"
}

Use varied shapes and roof styles.
Make simple/medium/complex progression obvious.
Do not include markdown. JSON only.
""".strip()
RETRY_SUFFIX = (
    "\n\nYour previous output did not satisfy the required schema. "
    "Return ONLY one JSON object with top-level key \"cases\" containing exactly 10 items."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 10 LLM-authored source-building specs for diagnostic study.")
    p.add_argument("--out_dir", default="datasets/llm_authored_10/source_specs")
    p.add_argument("--provider", default="", help="openai|anthropic|mock (optional override)")
    p.add_argument("--dotenv", default="", help="Optional .env path")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=2600)
    p.add_argument("--llm_seed", type=int, default=-1)
    p.add_argument("--allow_template_fallback", action="store_true")
    return p.parse_args()


def _int(v: Any, d: int) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return d


def _coerce_case(i: int, raw: Dict[str, Any]) -> Dict[str, Any]:
    case_id = f"llm_case_{i:03d}"
    difficulty = str(raw.get("difficulty", "medium")).strip().lower()
    if difficulty not in {"simple", "medium", "complex"}:
        difficulty = "medium"

    width = max(8, _int(raw.get("width", 14), 14))
    depth = max(8, _int(raw.get("depth", 12), 12))
    floors = max(1, _int(raw.get("floors", 1), 1))
    floor_height = max(3, _int(raw.get("floor_height", 4), 4))

    fp = raw.get("footprint", {}) if isinstance(raw.get("footprint"), dict) else {}
    kind = str(fp.get("kind", "rectangle")).strip().lower()
    if kind not in {"rectangle", "l_shape", "u_shape", "plus", "ring"}:
        kind = "rectangle"

    roof = raw.get("roof", {}) if isinstance(raw.get("roof"), dict) else {}
    roof_type = str(roof.get("type", "flat")).strip().lower()
    if roof_type not in {"flat", "gable_x", "gable_z", "hip"}:
        roof_type = "flat"

    ent = raw.get("entrance", {}) if isinstance(raw.get("entrance"), dict) else {}
    ent_side = str(ent.get("side", "south")).strip().lower()
    if ent_side not in {"north", "south", "east", "west"}:
        ent_side = "south"

    win = raw.get("windows", {}) if isinstance(raw.get("windows"), dict) else {}
    win_pattern = str(win.get("pattern", "checker")).strip().lower()
    if win_pattern not in {"checker", "stripe_x", "stripe_z"}:
        win_pattern = "checker"

    feat = raw.get("features", {}) if isinstance(raw.get("features"), dict) else {}
    mats = raw.get("materials", {}) if isinstance(raw.get("materials"), dict) else {}

    def _m(name: str, default: str) -> str:
        v = mats.get(name, default)
        return str(v).strip().lower().replace("minecraft:", "") or default

    return {
        "case_id": case_id,
        "title": str(raw.get("title", case_id)).strip() or case_id,
        "difficulty": difficulty,
        "width": width,
        "depth": depth,
        "floors": floors,
        "floor_height": floor_height,
        "footprint": {
            "kind": kind,
            "notch_width": max(0, _int(fp.get("notch_width", 0), 0)),
            "notch_depth": max(0, _int(fp.get("notch_depth", 0), 0)),
            "corner": str(fp.get("corner", "nw")).strip().lower() or "nw",
            "opening": str(fp.get("opening", "south")).strip().lower() or "south",
            "gap_width": max(0, _int(fp.get("gap_width", 0), 0)),
            "thickness": max(1, _int(fp.get("thickness", 3), 3)),
            "arm_width": max(0, _int(fp.get("arm_width", 0), 0)),
            "arm_depth": max(0, _int(fp.get("arm_depth", 0), 0)),
        },
        "roof": {
            "type": roof_type,
            "height": max(1, _int(roof.get("height", 2), 2)),
        },
        "entrance": {
            "side": ent_side,
            "width": max(1, min(3, _int(ent.get("width", 1), 1))),
            "height": max(2, min(4, _int(ent.get("height", 2), 2))),
        },
        "windows": {
            "pattern": win_pattern,
            "spacing": max(2, _int(win.get("spacing", 3), 3)),
            "height": max(1, min(3, _int(win.get("height", 2), 2))),
        },
        "features": {
            "tower": bool(feat.get("tower", False)),
            "porch": bool(feat.get("porch", False)),
            "balcony": bool(feat.get("balcony", False)),
        },
        "materials": {
            "foundation": _m("foundation", "stonebrick"),
            "wall": _m("wall", "planks"),
            "roof": _m("roof", "nether_brick"),
            "window": _m("window", "glass"),
            "accent": _m("accent", "quartz_block"),
            "trim": _m("trim", "stone_slab"),
            "floor": _m("floor", "planks"),
            "light": _m("light", "glowstone"),
            "pillar": _m("pillar", "stonebrick"),
        },
        "notes": str(raw.get("notes", "")).strip(),
    }


def _template_fallback() -> List[Dict[str, Any]]:
    base: List[Dict[str, Any]] = []
    difficulties = ["simple"] * 3 + ["medium"] * 4 + ["complex"] * 3
    kinds = ["rectangle", "rectangle", "l_shape", "u_shape", "plus", "ring", "l_shape", "plus", "ring", "u_shape"]
    roofs = ["flat", "gable_x", "gable_z", "hip", "gable_x", "hip", "flat", "gable_z", "hip", "gable_x"]
    for i in range(10):
        d = difficulties[i]
        base.append(
            _coerce_case(
                i + 1,
                {
                    "difficulty": d,
                    "title": f"Template {d} case {i+1}",
                    "width": 12 + i,
                    "depth": 10 + (i % 4),
                    "floors": 1 if d == "simple" else (2 if d == "medium" else 3),
                    "floor_height": 4,
                    "footprint": {"kind": kinds[i], "thickness": 3, "gap_width": 6, "arm_width": 6, "arm_depth": 6},
                    "roof": {"type": roofs[i], "height": 2 + (i % 2)},
                    "entrance": {"side": "south", "width": 1, "height": 2},
                    "windows": {"pattern": "checker", "spacing": 3, "height": 2},
                    "features": {"tower": i >= 6, "porch": i % 2 == 0, "balcony": i >= 4},
                },
            )
        )
    return base


def _extract_cases(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _from_any(x: Any, depth: int = 0) -> List[Dict[str, Any]]:
        if depth > 3:
            return []
        if isinstance(x, str):
            try:
                parsed = json.loads(x)
                return _from_any(parsed, depth + 1)
            except Exception:
                return []
        if isinstance(x, list):
            out = [v for v in x if isinstance(v, dict)]
            for v in x:
                if isinstance(v, dict):
                    continue
                got = _from_any(v, depth + 1)
                if got:
                    out.extend(got)
            # de-dup by object id order-preserving enough for this use.
            if out:
                return out
            return []
        if isinstance(x, dict):
            keys = ("cases", "buildings", "items", "source_buildings", "data", "result", "output")
            for k in keys:
                if k in x:
                    got = _from_any(x.get(k), depth + 1)
                    if got:
                        return got
            # dict keyed by id -> spec object
            vals = [v for v in x.values() if isinstance(v, dict)]
            if len(vals) >= 3:
                return vals
            for v in x.values():
                got = _from_any(v, depth + 1)
                if got:
                    return got
            return []
        return []

    if not isinstance(obj, dict):
        return []
    got = _from_any(obj, 0)
    if got:
        return got
    # final fallback: if top-level already resembles one case
    if {"width", "depth", "roof"}.issubset(set(obj.keys())):
        return [obj]
    return []


def _ensure_difficulty_distribution(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target = ["simple"] * 3 + ["medium"] * 4 + ["complex"] * 3
    out = []
    for i in range(10):
        c = dict(cases[i])
        c["difficulty"] = target[i]
        c["case_id"] = f"llm_case_{i+1:03d}"
        out.append(c)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_llm_config(args.dotenv or None)
    if args.provider:
        cfg.provider = args.provider
    if cfg.provider != "mock":
        require_provider_key(cfg)

    request_payload = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "llm_seed": int(args.llm_seed),
        "provider": cfg.provider,
        "model": model_for_provider(cfg),
    }

    raw_text = ""
    spec_payload: Dict[str, Any] = {}
    usage_out: Dict[str, Any] = {}
    last_exc: Exception | None = None

    for attempt in range(1, 4):
        try:
            completion = complete_multimodal_with_meta(
                cfg=cfg,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=(USER_PROMPT if attempt == 1 else USER_PROMPT + RETRY_SUFFIX),
                image_paths=[],
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                llm_seed=(int(args.llm_seed) if int(args.llm_seed) >= 0 else None),
            )
            raw_text = completion.text
            spec_payload = extract_json_object(raw_text)
            raw_cases = _extract_cases(spec_payload)
            cost = estimate_usage_cost(completion.provider, completion.model, completion.usage)
            usage_out = {
                "provider": completion.provider,
                "model": completion.model,
                "endpoint": completion.endpoint,
                "usage": completion.usage,
                "cost": cost,
                "attempt": attempt,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            append_cost_event(
                dataset_root=out_dir.parent,
                event={
                    "building": "llm_authored_10",
                    "stage": "source_spec_generation",
                    "provider": completion.provider,
                    "model": completion.model,
                    "endpoint": completion.endpoint,
                    "usage": completion.usage,
                    "cost": cost,
                    "attempt": attempt,
                    "output_path": str(out_dir / "source_specs.json"),
                },
            )
            if len(raw_cases) >= 10:
                break
            last_exc = RuntimeError(f"insufficient cases on attempt {attempt}: {len(raw_cases)}")
        except Exception as exc:
            last_exc = exc
            continue
    else:
        if not args.allow_template_fallback:
            raise SystemExit(f"LLM spec generation failed: {last_exc}")
        usage_out = {
            "provider": cfg.provider,
            "model": model_for_provider(cfg),
            "error": str(last_exc),
            "fallback": "template",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        spec_payload = {"cases": _template_fallback()}

    raw_cases = _extract_cases(spec_payload)
    if len(raw_cases) < 10 and not args.allow_template_fallback:
        # Persist diagnostic artifacts even on hard failure.
        (out_dir / "source_specs.request.json").write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "source_specs.raw.txt").write_text(raw_text, encoding="utf-8")
        (out_dir / "source_specs.usage.json").write_text(json.dumps(usage_out, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"LLM returned insufficient cases: {len(raw_cases)}")
    if len(raw_cases) < 10:
        raw_cases = _template_fallback()

    coerced = [_coerce_case(i + 1, raw_cases[i] if i < len(raw_cases) else {}) for i in range(10)]
    coerced = _ensure_difficulty_distribution(coerced)

    final_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "llm_authored_10_diagnostic",
        "authoring_mode": "llm" if not usage_out.get("fallback") else "template_fallback",
        "source_condition": "shared_source",
        "author_provider": cfg.provider,
        "author_model": model_for_provider(cfg),
        "cases": coerced,
    }

    (out_dir / "source_specs.request.json").write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "source_specs.raw.txt").write_text(raw_text, encoding="utf-8")
    (out_dir / "source_specs.usage.json").write_text(json.dumps(usage_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "source_specs.json").write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[generate_llm_authored_specs] wrote {out_dir / 'source_specs.json'}")


if __name__ == "__main__":
    main()
