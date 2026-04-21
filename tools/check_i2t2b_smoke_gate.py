#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Check smoke-run quality gates for i2t2b outputs "
            "(missing outputs, plan fallback/empty ops, and aggregate rebuild metrics)."
        )
    )
    p.add_argument("--gt_root", required=True, help="GT dataset root (contains building_xxx).")
    p.add_argument("--pred_root", required=True, help="Prediction/output root (contains building_xxx).")
    p.add_argument("--description_subdir", required=True)
    p.add_argument("--plan_subdir", required=True)
    p.add_argument("--rebuild_subdir", required=True)
    p.add_argument("--metrics_json", required=True, help="evaluate_rebuild_metrics output json path.")
    p.add_argument("--building_pattern", default="building_*")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max_missing_description_rate", type=float, default=0.00)
    p.add_argument("--max_missing_plan_rate", type=float, default=0.00)
    p.add_argument("--max_missing_rebuild_rate", type=float, default=0.00)
    p.add_argument("--max_empty_operations_rate", type=float, default=0.10)
    p.add_argument("--max_fallback_rate", type=float, default=0.60)
    p.add_argument("--max_strict_blocking_rate", type=float, default=0.60)
    p.add_argument("--min_iou", type=float, default=0.18)
    p.add_argument("--min_f1", type=float, default=0.30)
    p.add_argument("--min_material_match", type=float, default=0.10)
    p.add_argument("--out_json", default="", help="Optional path to save gate result json.")
    return p.parse_args()


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b <= 0.0 else a / b


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _list_buildings(root: Path, pattern: str, limit: int) -> List[Path]:
    xs = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if limit > 0:
        xs = xs[:limit]
    return xs


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root).resolve()
    pred_root = Path(args.pred_root).resolve()
    metrics_json = Path(args.metrics_json).resolve()

    if not gt_root.is_dir():
        raise SystemExit(f"gt_root not found: {gt_root}")
    if not pred_root.is_dir():
        raise SystemExit(f"pred_root not found: {pred_root}")
    if not metrics_json.is_file():
        raise SystemExit(f"metrics_json not found: {metrics_json}")

    buildings = _list_buildings(gt_root, args.building_pattern, int(args.limit))
    if not buildings:
        raise SystemExit("No buildings found under gt_root with given pattern/limit.")

    total = len(buildings)
    missing_desc = 0
    missing_plan = 0
    missing_rebuild = 0
    empty_ops = 0
    fallback_count = 0
    strict_blocking_count = 0
    inspected = 0
    examples: Dict[str, List[str]] = {
        "missing_description": [],
        "missing_plan": [],
        "missing_rebuild": [],
        "empty_operations": [],
        "fallback_triggered": [],
        "strict_blocking": [],
    }

    for b in buildings:
        name = b.name
        pred_b = pred_root / name
        desc_path = pred_b / args.description_subdir / "description.json"
        plan_path = pred_b / args.plan_subdir / "plan.json"
        req_path = pred_b / args.plan_subdir / "plan.request.json"
        vox_path = pred_b / args.rebuild_subdir / "voxels.npy"

        if not desc_path.is_file():
            missing_desc += 1
            if len(examples["missing_description"]) < 6:
                examples["missing_description"].append(name)
        if not plan_path.is_file():
            missing_plan += 1
            if len(examples["missing_plan"]) < 6:
                examples["missing_plan"].append(name)
        if not vox_path.is_file():
            missing_rebuild += 1
            if len(examples["missing_rebuild"]) < 6:
                examples["missing_rebuild"].append(name)

        plan_obj: Dict[str, Any] = {}
        req_obj: Dict[str, Any] = {}
        if plan_path.is_file():
            try:
                plan_obj = _load_json(plan_path)
            except Exception:
                plan_obj = {}
        if req_path.is_file():
            try:
                req_obj = _load_json(req_path)
            except Exception:
                req_obj = {}

        ops = plan_obj.get("operations")
        if isinstance(ops, list):
            inspected += 1
            if len(ops) == 0:
                empty_ops += 1
                if len(examples["empty_operations"]) < 6:
                    examples["empty_operations"].append(name)

        if bool(req_obj.get("fallback_triggered")):
            fallback_count += 1
            if len(examples["fallback_triggered"]) < 6:
                examples["fallback_triggered"].append(name)

        val = req_obj.get("validation_report", {})
        if isinstance(val, dict):
            sbi = val.get("strict_blocking_issues", [])
            if isinstance(sbi, list) and len(sbi) > 0:
                strict_blocking_count += 1
                if len(examples["strict_blocking"]) < 6:
                    examples["strict_blocking"].append(name)

    desc_missing_rate = _safe_div(float(missing_desc), float(total))
    plan_missing_rate = _safe_div(float(missing_plan), float(total))
    rebuild_missing_rate = _safe_div(float(missing_rebuild), float(total))
    empty_ops_rate = _safe_div(float(empty_ops), float(max(1, inspected)))
    fallback_rate = _safe_div(float(fallback_count), float(total))
    strict_blocking_rate = _safe_div(float(strict_blocking_count), float(total))

    metrics_obj = _load_json(metrics_json)
    agg = metrics_obj.get("aggregate", {})
    agg_metrics = agg.get("metrics", {}) if isinstance(agg, dict) else {}
    iou = _to_float(agg_metrics.get("iou"))
    f1 = _to_float(agg_metrics.get("f1"))
    material = _to_float(agg_metrics.get("material_match_relaxed_id"))
    if material <= 0.0:
        material = _to_float(agg_metrics.get("material_match"))

    checks: List[Dict[str, Any]] = [
        {
            "name": "missing_description_rate",
            "value": desc_missing_rate,
            "threshold": float(args.max_missing_description_rate),
            "comparator": "<=",
            "pass": desc_missing_rate <= float(args.max_missing_description_rate),
        },
        {
            "name": "missing_plan_rate",
            "value": plan_missing_rate,
            "threshold": float(args.max_missing_plan_rate),
            "comparator": "<=",
            "pass": plan_missing_rate <= float(args.max_missing_plan_rate),
        },
        {
            "name": "missing_rebuild_rate",
            "value": rebuild_missing_rate,
            "threshold": float(args.max_missing_rebuild_rate),
            "comparator": "<=",
            "pass": rebuild_missing_rate <= float(args.max_missing_rebuild_rate),
        },
        {
            "name": "empty_operations_rate",
            "value": empty_ops_rate,
            "threshold": float(args.max_empty_operations_rate),
            "comparator": "<=",
            "pass": empty_ops_rate <= float(args.max_empty_operations_rate),
        },
        {
            "name": "fallback_rate",
            "value": fallback_rate,
            "threshold": float(args.max_fallback_rate),
            "comparator": "<=",
            "pass": fallback_rate <= float(args.max_fallback_rate),
        },
        {
            "name": "strict_blocking_rate",
            "value": strict_blocking_rate,
            "threshold": float(args.max_strict_blocking_rate),
            "comparator": "<=",
            "pass": strict_blocking_rate <= float(args.max_strict_blocking_rate),
        },
        {
            "name": "iou",
            "value": iou,
            "threshold": float(args.min_iou),
            "comparator": ">=",
            "pass": iou >= float(args.min_iou),
        },
        {
            "name": "f1",
            "value": f1,
            "threshold": float(args.min_f1),
            "comparator": ">=",
            "pass": f1 >= float(args.min_f1),
        },
        {
            "name": "material_match",
            "value": material,
            "threshold": float(args.min_material_match),
            "comparator": ">=",
            "pass": material >= float(args.min_material_match),
        },
    ]

    passed = all(bool(x.get("pass")) for x in checks)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gt_root": str(gt_root),
        "pred_root": str(pred_root),
        "description_subdir": args.description_subdir,
        "plan_subdir": args.plan_subdir,
        "rebuild_subdir": args.rebuild_subdir,
        "metrics_json": str(metrics_json),
        "building_pattern": args.building_pattern,
        "limit": int(args.limit),
        "total_buildings": total,
        "inspected_plans": inspected,
        "counts": {
            "missing_description": missing_desc,
            "missing_plan": missing_plan,
            "missing_rebuild": missing_rebuild,
            "empty_operations": empty_ops,
            "fallback_triggered": fallback_count,
            "strict_blocking": strict_blocking_count,
        },
        "rates": {
            "missing_description_rate": desc_missing_rate,
            "missing_plan_rate": plan_missing_rate,
            "missing_rebuild_rate": rebuild_missing_rate,
            "empty_operations_rate": empty_ops_rate,
            "fallback_rate": fallback_rate,
            "strict_blocking_rate": strict_blocking_rate,
        },
        "aggregate_metrics": {
            "iou": iou,
            "f1": f1,
            "material_match": material,
        },
        "checks": checks,
        "examples": examples,
        "passed": passed,
    }

    if args.out_json:
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
