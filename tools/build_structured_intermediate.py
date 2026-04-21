#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build reconstruction-oriented structured IR from description JSON.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--description_subdir", required=True)
    p.add_argument("--out_subdir", default="structured_intermediate")
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    xs = sorted([p for p in dataset_root.glob(pattern) if p.is_dir()])
    if limit > 0:
        xs = xs[:limit]
    return xs


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _guess_substructures(elements: List[str], hints: List[str], summary: str) -> List[str]:
    text = " ".join(elements + hints + [summary]).lower()
    out: List[str] = []
    for name in ["tower", "balcony", "wing", "gate", "porch", "annex"]:
        if name in text:
            out.append(name)
    return sorted(set(out))


def _coerce_material_roles(materials: List[Dict[str, Any]]) -> Dict[str, str]:
    role_map = {
        "foundation": "stonebrick",
        "wall": "planks",
        "roof": "nether_brick",
        "window": "glass",
        "accent": "quartz_block",
        "trim": "stone_slab",
        "floor": "planks",
        "light": "glowstone",
        "pillar": "stonebrick",
    }
    for m in materials:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name", "")).strip().lower().replace("minecraft:", "")
        role = str(m.get("role", "")).strip().lower()
        if not name:
            continue
        if role in role_map:
            role_map[role] = name
    if role_map["window"] == role_map["wall"]:
        role_map["window"] = "glass"
    return role_map


def _footprint_from_hint(hint: str) -> str:
    h = (hint or "").strip().lower()
    if h in {"rectangle", "l_shape", "u_shape", "plus", "ring"}:
        return h
    if h == "complex":
        return "ring"
    return "rectangle"


def _roof_from_shape(shape: Dict[str, Any]) -> str:
    rt = str(shape.get("roof_type", "flat")).strip().lower()
    if rt in {"flat", "gable", "gable_x", "gable_z", "hip"}:
        if rt == "gable":
            return "gable_x"
        return rt
    return "flat"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    for bdir in _list_buildings(dataset_root, args.building_pattern, args.limit):
        desc_path = bdir / args.description_subdir / "description.json"
        out_dir = bdir / args.out_subdir
        out_json = out_dir / "intermediate.json"
        out_md = out_dir / "intermediate.md"
        if out_json.is_file() and not args.overwrite:
            print(f"[build_structured_intermediate] skip {bdir.name} (exists)")
            continue
        if not desc_path.is_file():
            print(f"[build_structured_intermediate] skip {bdir.name} (missing description)")
            continue

        desc = _load_json(desc_path)
        shape = desc.get("shape", {}) if isinstance(desc.get("shape"), dict) else {}
        dims = desc.get("dimensions_estimate", {}) if isinstance(desc.get("dimensions_estimate"), dict) else {}
        mats = desc.get("materials", []) if isinstance(desc.get("materials"), list) else []
        elements = [str(x).strip().lower() for x in (desc.get("elements") or []) if str(x).strip()]
        hints = [str(x).strip() for x in (desc.get("rebuild_hints") or []) if str(x).strip()]
        summary = str(desc.get("summary", "")).strip()

        width = max(8, int(round(float(dims.get("width", 12)))))
        depth = max(8, int(round(float(dims.get("depth", 10)))))
        height = max(5, int(round(float(dims.get("height", 8)))))
        floors = max(1, int(round(float(shape.get("floors_estimate", 1)))))
        floor_height = max(3, int(round(height / floors)))

        footprint_kind = _footprint_from_hint(str(shape.get("footprint_hint", "rectangle")))
        roof_type = _roof_from_shape(shape)
        materials = _coerce_material_roles(mats)

        openings = {
            "has_door": ("entrance" in elements) or ("door" in summary.lower()),
            "door_side": "south",
            "door_width": 1,
            "door_height": 2,
            "window_pattern": "checker" if "windows" in elements else "stripe_x",
            "window_spacing": 3,
            "window_height": 2,
        }

        ir = {
            "case_id": bdir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "description_path": str(desc_path),
                "description_subdir": args.description_subdir,
            },
            "rebuild_intent": {
                "footprint": {
                    "kind": footprint_kind,
                    "width": width,
                    "depth": depth,
                    "symmetry": str(shape.get("symmetry", "unknown")),
                },
                "dimensions": {
                    "width": width,
                    "depth": depth,
                    "height": height,
                    "floors": floors,
                    "floor_height": floor_height,
                },
                "height_profile": {
                    "roof_type": roof_type,
                    "roof_height": 2 if roof_type == "flat" else 3,
                    "tower_hint": "tower" in _guess_substructures(elements, hints, summary),
                },
                "materials": {
                    "foundation": materials["foundation"],
                    "wall": materials["wall"],
                    "roof": materials["roof"],
                    "window": materials["window"],
                    "accent": materials["accent"],
                    "trim": materials["trim"],
                    "floor": materials["floor"],
                    "light": materials["light"],
                    "pillar": materials["pillar"],
                },
                "openings": openings,
                "substructures": _guess_substructures(elements, hints, summary),
                "repeated_patterns": [h for h in hints if any(k in h.lower() for k in ["repeat", "symmetry", "row", "pattern"])],
                "raw_elements": elements,
                "raw_hints": hints,
                "summary": summary,
            },
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
        out_md.write_text(
            "\n".join(
                [
                    f"# Structured Intermediate: {bdir.name}",
                    "",
                    "- This file is diagnostic-only structured build intent.",
                    f"- footprint={footprint_kind}, dims=({width},{depth},{height}), floors={floors}",
                    f"- roof={roof_type}, substructures={', '.join(ir['rebuild_intent']['substructures']) or 'none'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[build_structured_intermediate] wrote {out_json}")


if __name__ == "__main__":
    main()
