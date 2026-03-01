#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools.generate_rebuild_plans import (
    REQUIRED_PALETTE_ROLES,
    _coerce_plan,
    _extract_plan_from_response_text,
    _heuristic_plan,
    _validate_and_repair_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-parse existing plan.raw.txt with upgraded parser (no LLM call).")
    parser.add_argument("--dataset_root", required=True, help="Dataset root with building_xxx dirs.")
    parser.add_argument("--source_plan_subdir", required=True, help="Source plan subdir that contains plan.raw.txt.")
    parser.add_argument("--out_plan_subdir", required=True, help="Output plan subdir.")
    parser.add_argument("--description_subdir", default="description", help="Description subdir.")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern.")
    parser.add_argument("--limit", type=int, default=0, help="Max buildings (0=all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output plan.json if exists.")
    parser.add_argument("--use_heuristic_if_empty", action="store_true", help="Fallback to heuristic plan if parsed operations are empty.")
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
        help="Required role subset.",
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


def _normalize_roles(raw_roles: List[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for r in raw_roles:
        t = str(r).strip().lower()
        if t in REQUIRED_PALETTE_ROLES and t not in out:
            out.append(t)
    return tuple(out) if out else REQUIRED_PALETTE_ROLES


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    bdirs = _list_buildings(dataset_root, args.building_pattern, int(args.limit))
    if not bdirs:
        raise SystemExit("no building dirs found")

    required_roles = _normalize_roles(list(args.required_palette_roles))
    parsed_ok = 0
    heuristic_fallback = 0
    skipped = 0

    for bdir in bdirs:
        src_plan_dir = bdir / args.source_plan_subdir
        src_raw_path = src_plan_dir / "plan.raw.txt"
        src_plan_path = src_plan_dir / "plan.json"
        desc_path = bdir / args.description_subdir / "description.json"
        if not src_raw_path.exists() or not desc_path.exists():
            skipped += 1
            continue

        out_dir = bdir / args.out_plan_subdir
        out_plan = out_dir / "plan.json"
        out_raw = out_dir / "plan.raw.txt"
        out_req = out_dir / "plan.request.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_plan.exists() and not args.overwrite:
            continue

        raw_text = src_raw_path.read_text(encoding="utf-8")
        desc = _load_json(desc_path)
        source_plan = _load_json(src_plan_path) if src_plan_path.exists() else {}

        parsed_obj, parse_report = _extract_plan_from_response_text(raw_text)
        coerced, coerce_report = _coerce_plan(parsed_obj)
        fallback_used = False
        if not coerced.get("operations") and bool(args.use_heuristic_if_empty):
            fallback_used = True
            coerced, _ = _coerce_plan(_heuristic_plan(desc))
            coerced["notes"] = list(coerced.get("notes", [])) + ["fallback_used_due_to_empty_operations_reparse"]
            heuristic_fallback += 1

        coerced, validation_report = _validate_and_repair_plan(
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

        provider = str(source_plan.get("provider", "")).strip() or "unknown"
        model = str(source_plan.get("model", "")).strip() or "unknown"
        coerced["building"] = bdir.name
        coerced["created_at"] = datetime.now(timezone.utc).isoformat()
        coerced["provider"] = provider
        coerced["model"] = model
        parsed_ok += 1

        out_plan.write_text(json.dumps(coerced, ensure_ascii=False, indent=2), encoding="utf-8")
        out_raw.write_text(raw_text, encoding="utf-8")
        out_req.write_text(
            json.dumps(
                {
                    "building": bdir.name,
                    "source_plan_subdir": args.source_plan_subdir,
                    "source_plan_path": str(src_plan_path.relative_to(bdir)) if src_plan_path.exists() else None,
                    "description_path": str(desc_path.relative_to(bdir)),
                    "provider": provider,
                    "model": model,
                    "parse_report": parse_report,
                    "coerce_report": coerce_report,
                    "validation_report": validation_report,
                    "fallback_triggered": bool(fallback_used),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "settings": {
                        "strict_schema": bool(args.strict_schema),
                        "enforce_role_fixed": bool(args.enforce_role_fixed),
                        "require_material_budget": bool(args.require_material_budget),
                        "material_budget_tolerance": float(args.material_budget_tolerance),
                        "role_fix_min_confidence": float(args.role_fix_min_confidence),
                        "prefer_description_palette": bool(args.prefer_description_palette),
                        "max_operations": int(args.max_operations),
                        "required_palette_roles": list(required_roles),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[reparse_rebuild_plans_from_raw] wrote {out_plan}")

    print(
        "[reparse_rebuild_plans_from_raw] done "
        f"parsed_ok={parsed_ok} heuristic_fallback={heuristic_fallback} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
