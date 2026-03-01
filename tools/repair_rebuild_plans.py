#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from tools.generate_rebuild_plans import (
    REQUIRED_PALETTE_ROLES,
    _coerce_plan,
    _normalize_role,
    _validate_and_repair_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair existing rebuild_plan outputs with strict schema/material alignment.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing building_xxx directories")
    parser.add_argument("--source_plan_subdir", required=True, help="Existing plan subdir to read plan.json from")
    parser.add_argument("--out_plan_subdir", required=True, help="Output plan subdir for repaired plan.json")
    parser.add_argument("--description_subdir", default="description", help="Description subdir containing description.json")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern")
    parser.add_argument("--limit", type=int, default=0, help="Max buildings (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing repaired plans")

    parser.add_argument("--strict_schema", dest="strict_schema", action="store_true")
    parser.add_argument("--no_strict_schema", dest="strict_schema", action="store_false")
    parser.add_argument("--enforce_role_fixed", dest="enforce_role_fixed", action="store_true")
    parser.add_argument("--no_enforce_role_fixed", dest="enforce_role_fixed", action="store_false")
    parser.add_argument("--require_material_budget", dest="require_material_budget", action="store_true")
    parser.add_argument("--no_require_material_budget", dest="require_material_budget", action="store_false")
    parser.add_argument("--prefer_description_palette", dest="prefer_description_palette", action="store_true")
    parser.add_argument("--no_prefer_description_palette", dest="prefer_description_palette", action="store_false")
    parser.add_argument("--material_budget_tolerance", type=float, default=0.35)
    parser.add_argument("--role_fix_min_confidence", type=float, default=0.78)
    parser.add_argument("--max_operations", type=int, default=260)
    parser.add_argument(
        "--required_palette_roles",
        nargs="*",
        default=list(REQUIRED_PALETTE_ROLES),
        help="Subset of required palette roles. Default: wall roof trim glass light floor",
    )

    parser.set_defaults(
        strict_schema=True,
        enforce_role_fixed=True,
        require_material_budget=True,
        prefer_description_palette=True,
    )
    return parser.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in dataset_root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _normalize_required_roles(raw_roles: List[str]) -> Tuple[str, ...]:
    out = []
    for r in raw_roles:
        nr = _normalize_role(r)
        if nr in REQUIRED_PALETTE_ROLES and nr not in out:
            out.append(nr)
    return tuple(out) if out else REQUIRED_PALETTE_ROLES


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    required_roles = _normalize_required_roles(args.required_palette_roles)
    buildings = _list_buildings(dataset_root, args.building_pattern, args.limit)

    for bdir in buildings:
        src_plan = bdir / args.source_plan_subdir / "plan.json"
        desc_path = bdir / args.description_subdir / "description.json"
        if not src_plan.is_file():
            print(f"[repair_rebuild_plans] skip {bdir.name} (missing source plan: {src_plan})")
            continue
        if not desc_path.is_file():
            print(f"[repair_rebuild_plans] skip {bdir.name} (missing description: {desc_path})")
            continue

        out_dir = bdir / args.out_plan_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_plan = out_dir / "plan.json"
        out_req = out_dir / "plan.request.json"

        if out_plan.is_file() and not args.overwrite:
            print(f"[repair_rebuild_plans] skip {bdir.name} (exists)")
            continue

        src_obj = json.loads(src_plan.read_text(encoding="utf-8"))
        desc = json.loads(desc_path.read_text(encoding="utf-8"))

        coerced, coerce_report = _coerce_plan(src_obj)
        repaired, validation_report = _validate_and_repair_plan(
            coerced,
            desc=desc if isinstance(desc, dict) else {},
            strict_schema=bool(args.strict_schema),
            enforce_role_fixed=bool(args.enforce_role_fixed),
            require_material_budget=bool(args.require_material_budget),
            material_budget_tolerance=float(args.material_budget_tolerance),
            role_fix_min_confidence=float(args.role_fix_min_confidence),
            prefer_description_palette=bool(args.prefer_description_palette),
            max_operations=int(args.max_operations),
            required_palette_roles=required_roles,
        )

        repaired["building"] = bdir.name
        repaired["created_at"] = datetime.now(timezone.utc).isoformat()
        repaired["repair_source_plan_subdir"] = args.source_plan_subdir
        repaired["repair_profile"] = {
            "strict_schema": bool(args.strict_schema),
            "enforce_role_fixed": bool(args.enforce_role_fixed),
            "require_material_budget": bool(args.require_material_budget),
            "prefer_description_palette": bool(args.prefer_description_palette),
            "material_budget_tolerance": float(args.material_budget_tolerance),
            "role_fix_min_confidence": float(args.role_fix_min_confidence),
            "max_operations": int(args.max_operations),
            "required_palette_roles": list(required_roles),
        }
        if isinstance(src_obj.get("provider"), str):
            repaired["provider"] = src_obj["provider"]
        if isinstance(src_obj.get("model"), str):
            repaired["model"] = src_obj["model"]

        out_plan.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
        out_req.write_text(
            json.dumps(
                {
                    "building": bdir.name,
                    "source_plan": str(src_plan.relative_to(bdir)),
                    "description_path": str(desc_path.relative_to(bdir)),
                    "coerce_report": coerce_report,
                    "validation_report": validation_report,
                    "repair_profile": repaired["repair_profile"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[repair_rebuild_plans] wrote {out_plan}")


if __name__ == "__main__":
    main()
