#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools.cost_logger import append_cost_event, estimate_usage_cost
from tools.llm_client import LLMError, complete_multimodal_with_meta, extract_json_object
from tools.llm_config import load_llm_config, require_provider_key


SYSTEM_PROMPT = (
    "You are an expert Minecraft architecture analyst. "
    "Return only one JSON object. No markdown, no code block."
)

USER_PROMPT_TEMPLATE = """
You are given multiple views of ONE Minecraft building.
Create a faithful structural description for reconstruction.

Requirements:
- Use ONLY what is visible in the images.
- If uncertain, write uncertainty explicitly.
- Normalize material names to simple IDs when possible (stonebrick, brick, wood, glass, fence, slab_stone, slab_wood, air).
- Output STRICT JSON with this schema:
{
  "summary": "short paragraph",
  "building_type": "string",
  "shape": {
    "overall_form": "string",
    "footprint_hint": "rectangle|l_shape|u_shape|complex|unknown",
    "symmetry": "string",
    "floors_estimate": 1,
    "roof_type": "flat|gable|hip|dome|other|unknown"
  },
  "dimensions_estimate": {
    "width": 0,
    "depth": 0,
    "height": 0
  },
  "materials": [
    {"name": "stonebrick", "role": "wall", "confidence": 0.0}
  ],
  "elements": ["windows", "entrance", "roof"],
  "rebuild_hints": ["bullet 1", "bullet 2"],
  "uncertainties": ["string"]
}

- confidence range: 0.0 to 1.0
- dimensions_estimate values must be integers >= 1
- materials must include at least 3 candidates when possible
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate building descriptions from multi-view images.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing building_xxx directories.")
    parser.add_argument("--out_subdir", default="description", help="Subdirectory per building for description output.")
    parser.add_argument("--building_pattern", default="building_*", help="Glob pattern under dataset_root.")
    parser.add_argument("--limit", type=int, default=0, help="Max buildings to process (0 means all).")
    parser.add_argument("--max_images", type=int, default=6, help="Max images per building sent to model.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing descriptions.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--max_tokens", type=int, default=1800, help="LLM max output tokens.")
    parser.add_argument("--llm_seed", type=int, default=-1, help="Optional LLM seed (OpenAI only, -1 disables).")
    parser.add_argument("--dotenv", default="", help="Optional .env path.")
    parser.add_argument("--provider", default="", help="Optional provider override: openai|anthropic|mock")
    return parser.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in dataset_root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _pick_images(meta: Dict[str, Any], building_dir: Path, max_images: int) -> List[Path]:
    views = meta.get("views", [])
    if not isinstance(views, list):
        return []

    top = []
    ring = []
    for view in views:
        if not isinstance(view, dict):
            continue
        pitch = float(view.get("pitch", 0.0))
        if pitch >= 55.0:
            top.append(view)
        else:
            ring.append(view)

    selected = []

    ring_budget = max(1, max_images - min(2, len(top)))
    if ring:
        step = max(1, len(ring) // ring_budget)
        selected.extend(ring[::step][:ring_budget])

    selected.extend(top[:2])

    dedup: List[Path] = []
    seen = set()
    for view in selected:
        rel = view.get("path")
        if not isinstance(rel, str):
            continue
        p = (building_dir / rel).resolve()
        if not p.is_file():
            continue
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)

    return dedup[:max_images]


def _coerce_description_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    out.setdefault("summary", "")
    out.setdefault("building_type", "unknown")

    shape = out.get("shape")
    if not isinstance(shape, dict):
        shape = {}
    shape.setdefault("overall_form", "unknown")
    shape.setdefault("footprint_hint", "unknown")
    shape.setdefault("symmetry", "unknown")
    floors = shape.get("floors_estimate", 1)
    try:
        floors_int = max(1, int(round(float(floors))))
    except Exception:
        floors_int = 1
    shape["floors_estimate"] = floors_int
    shape.setdefault("roof_type", "unknown")
    out["shape"] = shape

    dims = out.get("dimensions_estimate")
    if not isinstance(dims, dict):
        dims = {}
    for k in ("width", "depth", "height"):
        v = dims.get(k, 1)
        try:
            v_int = max(1, int(round(float(v))))
        except Exception:
            v_int = 1
        dims[k] = v_int
    out["dimensions_estimate"] = dims

    mats = out.get("materials")
    if not isinstance(mats, list):
        mats = []
    fixed_mats = []
    for m in mats:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name", "")).strip() or "unknown"
        role = str(m.get("role", "")).strip() or "unknown"
        conf = m.get("confidence", 0.5)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.5
        conf_f = max(0.0, min(1.0, conf_f))
        fixed_mats.append({"name": name, "role": role, "confidence": conf_f})
    out["materials"] = fixed_mats

    for key in ("elements", "rebuild_hints", "uncertainties"):
        v = out.get(key)
        if not isinstance(v, list):
            out[key] = []
        else:
            out[key] = [str(x) for x in v]

    return out


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    cfg = load_llm_config(args.dotenv or None)
    if args.provider:
        cfg.provider = args.provider
    if cfg.provider != "mock":
        require_provider_key(cfg)

    buildings = _list_buildings(dataset_root, args.building_pattern, args.limit)
    if not buildings:
        raise SystemExit(f"No buildings found in {dataset_root} with pattern {args.building_pattern}")

    for bdir in buildings:
        meta_path = bdir / "meta.json"
        if not meta_path.is_file():
            continue

        out_dir = bdir / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "description.json"
        out_raw = out_dir / "description.raw.txt"
        out_req = out_dir / "description.request.json"
        out_usage = out_dir / "llm_usage.json"

        if out_json.is_file() and not args.overwrite:
            print(f"[generate_building_descriptions] skip {bdir.name} (exists)")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        images = _pick_images(meta, bdir, max_images=max(1, args.max_images))
        if not images:
            print(f"[generate_building_descriptions] skip {bdir.name} (no images)")
            continue

        user_prompt = USER_PROMPT_TEMPLATE
        try:
            completion = complete_multimodal_with_meta(
                cfg=cfg,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_paths=images,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                llm_seed=(int(args.llm_seed) if int(args.llm_seed) >= 0 else None),
            )
            response_text = completion.text
            desc_obj = extract_json_object(response_text)
            usage = completion.usage
            cost = estimate_usage_cost(completion.provider, completion.model, usage)
            out_usage.write_text(
                json.dumps(
                    {
                        "building": bdir.name,
                        "stage": "description",
                        "provider": completion.provider,
                        "model": completion.model,
                        "endpoint": completion.endpoint,
                        "usage": usage,
                        "cost": cost,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            append_cost_event(
                dataset_root=dataset_root,
                event={
                    "building": bdir.name,
                    "stage": "description",
                    "provider": completion.provider,
                    "model": completion.model,
                    "endpoint": completion.endpoint,
                    "usage": usage,
                    "cost": cost,
                    "status": "ok",
                    "output_path": str(out_json),
                },
            )
        except Exception as exc:  # noqa: BLE001
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
                    "stage": "description",
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
            if isinstance(exc, LLMError):
                print(f"[generate_building_descriptions] {bdir.name} LLM error: {exc}")
            else:
                print(f"[generate_building_descriptions] {bdir.name} error: {exc}")
            continue

        desc_obj = _coerce_description_schema(desc_obj)
        desc_obj["provider"] = cfg.provider
        desc_obj["model"] = cfg.openai_model if cfg.provider == "openai" else cfg.anthropic_model
        desc_obj["building"] = bdir.name
        desc_obj["created_at"] = datetime.now(timezone.utc).isoformat()
        desc_obj["llm_seed"] = int(args.llm_seed)

        out_json.write_text(json.dumps(desc_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        out_raw.write_text(response_text, encoding="utf-8")
        out_req.write_text(
            json.dumps(
                {
                    "building": bdir.name,
                    "provider": cfg.provider,
                    "model": completion.model,
                    "images": [str(p.relative_to(bdir)) for p in images],
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt": user_prompt,
                    "temperature": float(args.temperature),
                    "max_tokens": int(args.max_tokens),
                    "llm_seed": int(args.llm_seed),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[generate_building_descriptions] wrote {out_json}")


if __name__ == "__main__":
    main()
