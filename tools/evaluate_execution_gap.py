#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


METRICS_PRIORITY = [
    "iou",
    "f1",
    "precision",
    "recall",
    "correct_placement_rate",
    "correct_placement_coverage",
    "correct_placement_rate_relaxed_id",
    "correct_placement_coverage_relaxed_id",
    "material_match",
    "material_match_relaxed_id",
    "coarse_material_match",
    "component_f1",
    "component_precision",
    "component_recall",
    "intersection",
]

LEVEL_PASS_KEYS = [
    "level0_shift",
    "level1_shape",
    "level2_coarse_material",
    "level3_strict_material",
    "level4_structure_components",
    "all_levels_pass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare renderer upper-bound score vs agent execution score and "
            "report execution_gap."
        )
    )
    parser.add_argument("--gt_root", required=True, help="GT root (building_xxx/gt).")
    parser.add_argument("--pred_root", required=True, help="Prediction root (building_xxx/*).")
    parser.add_argument("--renderer_pred_subdir", required=True, help="Renderer prediction subdir under each building.")
    parser.add_argument("--agent_pred_subdir", required=True, help="Agent-execution prediction subdir under each building.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--renderer_metrics_out", default="", help="Optional path to save renderer metrics JSON.")
    parser.add_argument("--agent_metrics_out", default="", help="Optional path to save agent metrics JSON.")
    parser.add_argument("--thresholds_json", default="", help="Optional threshold override JSON.")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern.")
    parser.add_argument("--limit", type=int, default=0, help="Max buildings (0=all).")
    parser.add_argument("--max_shift_xy", type=int, default=48)
    parser.add_argument("--max_shift_y", type=int, default=8)
    parser.add_argument("--top_shift_candidates", type=int, default=24)
    parser.add_argument("--allow_gt_fallback", action="store_true")
    parser.add_argument("--fail_on_missing_renderer_pred", dest="fail_on_missing_renderer_pred", action="store_true")
    parser.add_argument(
        "--no_fail_on_missing_renderer_pred",
        dest="fail_on_missing_renderer_pred",
        action="store_false",
    )
    parser.add_argument("--fail_on_missing_agent_pred", dest="fail_on_missing_agent_pred", action="store_true")
    parser.add_argument(
        "--no_fail_on_missing_agent_pred",
        dest="fail_on_missing_agent_pred",
        action="store_false",
    )
    parser.set_defaults(
        fail_on_missing_renderer_pred=True,
        fail_on_missing_agent_pred=True,
    )
    return parser.parse_args()


def _safe_div(n: float, d: float) -> float:
    if d == 0.0:
        return 0.0
    return float(n) / float(d)


def _retention(agent_value: float, renderer_value: float) -> float:
    if renderer_value == 0.0 and agent_value == 0.0:
        return 1.0
    return _safe_div(agent_value, renderer_value)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _numeric_metrics_from_aggregate(payload: Dict[str, Any]) -> Dict[str, float]:
    src = payload.get("aggregate", {}).get("metrics", {})
    out: Dict[str, float] = {}
    if isinstance(src, dict):
        for k, v in src.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
    return out


def _metric_keys(renderer_metrics: Dict[str, float], agent_metrics: Dict[str, float]) -> List[str]:
    shared = set(renderer_metrics.keys()) & set(agent_metrics.keys())
    ordered = [k for k in METRICS_PRIORITY if k in shared]
    ordered += sorted(k for k in shared if k not in ordered)
    return ordered


def _extract_level_passes(item: Dict[str, Any]) -> Dict[str, bool]:
    levels = item.get("levels", {})
    result: Dict[str, bool] = {}
    if not isinstance(levels, dict):
        return {k: False for k in LEVEL_PASS_KEYS}

    for key in LEVEL_PASS_KEYS:
        if key == "all_levels_pass":
            value = levels.get("all_levels_pass")
            if isinstance(value, bool):
                result[key] = value
            elif isinstance(value, dict):
                result[key] = bool(value.get("pass", False))
            else:
                result[key] = False
            continue
        obj = levels.get(key, {})
        result[key] = bool(obj.get("pass", False)) if isinstance(obj, dict) else bool(obj)
    return result


def _run_rebuild_eval(
    gt_root: Path,
    pred_root: Path,
    pred_subdir: str,
    out_path: Path,
    args: argparse.Namespace,
    fail_on_missing: bool,
) -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "tools" / "evaluate_rebuild_metrics.py"),
        "--gt_root",
        str(gt_root),
        "--pred_root",
        str(pred_root),
        "--pred_source",
        "rebuild_world",
        "--pred_subdir",
        str(pred_subdir),
        "--out",
        str(out_path),
        "--building_pattern",
        str(args.building_pattern),
        "--max_shift_xy",
        str(int(args.max_shift_xy)),
        "--max_shift_y",
        str(int(args.max_shift_y)),
        "--top_shift_candidates",
        str(int(args.top_shift_candidates)),
    ]
    if args.thresholds_json:
        cmd += ["--thresholds_json", str(Path(args.thresholds_json).resolve())]
    if args.limit > 0:
        cmd += ["--limit", str(int(args.limit))]
    if args.allow_gt_fallback:
        cmd += ["--allow_gt_fallback"]
    if fail_on_missing:
        cmd += ["--fail_on_missing_pred"]
    subprocess.run(cmd, check=True)


def _mean(values: Iterable[float]) -> float:
    seq = list(values)
    if not seq:
        return 0.0
    return float(sum(seq)) / float(len(seq))


def _to_item_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = payload.get("items", [])
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("building", "")).strip()
        if not name:
            continue
        result[name] = item
    return result


def _aggregate_gap(
    renderer_payload: Dict[str, Any],
    agent_payload: Dict[str, Any],
) -> Dict[str, Any]:
    renderer_metrics = _numeric_metrics_from_aggregate(renderer_payload)
    agent_metrics = _numeric_metrics_from_aggregate(agent_payload)
    metric_keys = _metric_keys(renderer_metrics, agent_metrics)

    metric_gap = {
        key: float(renderer_metrics[key] - agent_metrics[key])
        for key in metric_keys
    }
    metric_retention = {
        key: _retention(agent_metrics[key], renderer_metrics[key])
        for key in metric_keys
    }

    renderer_pass = renderer_payload.get("aggregate", {}).get("pass_rates", {})
    agent_pass = agent_payload.get("aggregate", {}).get("pass_rates", {})
    shared_pass = sorted(set(renderer_pass.keys()) & set(agent_pass.keys()))
    pass_gap = {
        key: float(renderer_pass[key] - agent_pass[key])
        for key in shared_pass
        if isinstance(renderer_pass[key], (int, float)) and isinstance(agent_pass[key], (int, float))
    }
    pass_retention = {
        key: _retention(float(agent_pass[key]), float(renderer_pass[key]))
        for key in pass_gap.keys()
    }

    return {
        "renderer": {
            "metrics": {k: renderer_metrics[k] for k in metric_keys},
            "pass_rates": {k: float(renderer_pass[k]) for k in pass_gap.keys()},
        },
        "agent": {
            "metrics": {k: agent_metrics[k] for k in metric_keys},
            "pass_rates": {k: float(agent_pass[k]) for k in pass_gap.keys()},
        },
        "execution_gap": {
            "metrics": metric_gap,
            "metrics_retention_ratio": metric_retention,
            "pass_rates": pass_gap,
            "pass_rates_retention_ratio": pass_retention,
        },
    }


def _item_level_gap(renderer_item: Dict[str, Any], agent_item: Dict[str, Any]) -> Dict[str, float]:
    r = _extract_level_passes(renderer_item)
    a = _extract_level_passes(agent_item)
    return {k: float(int(r[k]) - int(a[k])) for k in LEVEL_PASS_KEYS}


def _item_level_retention(renderer_item: Dict[str, Any], agent_item: Dict[str, Any]) -> Dict[str, float]:
    r = _extract_level_passes(renderer_item)
    a = _extract_level_passes(agent_item)
    out: Dict[str, float] = {}
    for k in LEVEL_PASS_KEYS:
        rv = 1.0 if r[k] else 0.0
        av = 1.0 if a[k] else 0.0
        out[k] = _retention(av, rv)
    return out


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root).resolve()
    pred_root = Path(args.pred_root).resolve()
    out_path = Path(args.out).resolve()

    if not gt_root.is_dir():
        raise SystemExit(f"gt_root not found: {gt_root}")
    if not pred_root.is_dir():
        raise SystemExit(f"pred_root not found: {pred_root}")

    renderer_out = Path(args.renderer_metrics_out).resolve() if args.renderer_metrics_out else (
        out_path.parent / f"{out_path.stem}.renderer_metrics.json"
    )
    agent_out = Path(args.agent_metrics_out).resolve() if args.agent_metrics_out else (
        out_path.parent / f"{out_path.stem}.agent_metrics.json"
    )
    renderer_out.parent.mkdir(parents=True, exist_ok=True)
    agent_out.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[evaluate_execution_gap] renderer_eval:",
        f"pred_subdir={args.renderer_pred_subdir}",
        f"out={renderer_out}",
    )
    _run_rebuild_eval(
        gt_root=gt_root,
        pred_root=pred_root,
        pred_subdir=args.renderer_pred_subdir,
        out_path=renderer_out,
        args=args,
        fail_on_missing=bool(args.fail_on_missing_renderer_pred),
    )

    print(
        "[evaluate_execution_gap] agent_eval:",
        f"pred_subdir={args.agent_pred_subdir}",
        f"out={agent_out}",
    )
    _run_rebuild_eval(
        gt_root=gt_root,
        pred_root=pred_root,
        pred_subdir=args.agent_pred_subdir,
        out_path=agent_out,
        args=args,
        fail_on_missing=bool(args.fail_on_missing_agent_pred),
    )

    renderer_payload = _load_json(renderer_out)
    agent_payload = _load_json(agent_out)
    aggregate = _aggregate_gap(renderer_payload=renderer_payload, agent_payload=agent_payload)

    renderer_items = _to_item_map(renderer_payload)
    agent_items = _to_item_map(agent_payload)
    renderer_names = set(renderer_items.keys())
    agent_names = set(agent_items.keys())
    common_names = sorted(renderer_names & agent_names)

    metric_keys = list(aggregate["renderer"]["metrics"].keys())
    items: List[Dict[str, Any]] = []
    for name in common_names:
        r_item = renderer_items[name]
        a_item = agent_items[name]
        r_metrics = r_item.get("metrics", {})
        a_metrics = a_item.get("metrics", {})
        if not isinstance(r_metrics, dict) or not isinstance(a_metrics, dict):
            continue
        gap = {
            key: float(r_metrics[key] - a_metrics[key])
            for key in metric_keys
            if isinstance(r_metrics.get(key), (int, float)) and isinstance(a_metrics.get(key), (int, float))
        }
        retention = {
            key: _retention(float(a_metrics[key]), float(r_metrics[key]))
            for key in gap.keys()
        }
        items.append(
            {
                "building": name,
                "renderer": {
                    "metrics": {k: float(r_metrics[k]) for k in gap.keys()},
                    "levels_pass": _extract_level_passes(r_item),
                },
                "agent": {
                    "metrics": {k: float(a_metrics[k]) for k in gap.keys()},
                    "levels_pass": _extract_level_passes(a_item),
                },
                "execution_gap": {
                    "metrics": gap,
                    "metrics_retention_ratio": retention,
                    "levels_pass": _item_level_gap(r_item, a_item),
                    "levels_pass_retention_ratio": _item_level_retention(r_item, a_item),
                },
            }
        )

    level_gap_mean = {
        key: _mean(item["execution_gap"]["levels_pass"][key] for item in items) if items else 0.0
        for key in LEVEL_PASS_KEYS
    }
    level_retention_mean = {
        key: _mean(item["execution_gap"]["levels_pass_retention_ratio"][key] for item in items) if items else 0.0
        for key in LEVEL_PASS_KEYS
    }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gt_root": str(gt_root),
        "pred_root": str(pred_root),
        "renderer_metrics_path": str(renderer_out),
        "agent_metrics_path": str(agent_out),
        "settings": {
            "renderer_pred_subdir": args.renderer_pred_subdir,
            "agent_pred_subdir": args.agent_pred_subdir,
            "building_pattern": args.building_pattern,
            "limit": int(args.limit),
            "max_shift_xy": int(args.max_shift_xy),
            "max_shift_y": int(args.max_shift_y),
            "top_shift_candidates": int(args.top_shift_candidates),
            "allow_gt_fallback": bool(args.allow_gt_fallback),
            "fail_on_missing_renderer_pred": bool(args.fail_on_missing_renderer_pred),
            "fail_on_missing_agent_pred": bool(args.fail_on_missing_agent_pred),
            "thresholds_json": str(Path(args.thresholds_json).resolve()) if args.thresholds_json else "",
        },
        "summary": {
            "renderer_evaluated_buildings": int(renderer_payload.get("summary", {}).get("evaluated_buildings", 0)),
            "agent_evaluated_buildings": int(agent_payload.get("summary", {}).get("evaluated_buildings", 0)),
            "common_buildings": len(common_names),
            "renderer_only_buildings": sorted(renderer_names - agent_names),
            "agent_only_buildings": sorted(agent_names - renderer_names),
            "renderer_missing_predictions": list(renderer_payload.get("summary", {}).get("missing_predictions", [])),
            "agent_missing_predictions": list(agent_payload.get("summary", {}).get("missing_predictions", [])),
        },
        "aggregate": {
            **aggregate,
            "execution_gap_levels_mean": level_gap_mean,
            "execution_gap_levels_retention_ratio_mean": level_retention_mean,
        },
        "items": items,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    iou_gap = payload["aggregate"]["execution_gap"]["metrics"].get("iou", 0.0)
    f1_gap = payload["aggregate"]["execution_gap"]["metrics"].get("f1", 0.0)
    cpr_gap = payload["aggregate"]["execution_gap"]["metrics"].get("correct_placement_rate", 0.0)
    print(f"[evaluate_execution_gap] wrote: {out_path}")
    print(
        "[evaluate_execution_gap] aggregate execution_gap: "
        f"IoU={iou_gap:.4f}, "
        f"F1={f1_gap:.4f}, "
        f"correct_placement_rate={cpr_gap:.4f}"
    )


if __name__ == "__main__":
    main()
