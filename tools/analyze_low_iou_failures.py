#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Config:
    name: str
    metrics_path: Path
    dataset_root: Path
    plan_subdir: str


def _parse_default_configs() -> List[Config]:
    return [
        Config(
            name="v1/openai",
            metrics_path=Path("datasets/buildings_100_v1/metrics_levels_ablation_full_openai_gpt_5_mini.json"),
            dataset_root=Path("datasets/buildings_100_v1"),
            plan_subdir="rebuild_plan_ablation_full_openai_gpt_5_mini",
        ),
        Config(
            name="v1/claude",
            metrics_path=Path("datasets/buildings_100_v1/metrics_levels_ablation_full_anthropic_claude_haiku_4_5_20251001.json"),
            dataset_root=Path("datasets/buildings_100_v1"),
            plan_subdir="rebuild_plan_ablation_full_anthropic_claude_haiku_4_5_20251001",
        ),
        Config(
            name="v4/openai",
            metrics_path=Path("datasets/buildings_100_v4/metrics_levels_ablation_full_openai_gpt_5_mini.json"),
            dataset_root=Path("datasets/buildings_100_v4"),
            plan_subdir="rebuild_plan_ablation_full_openai_gpt_5_mini",
        ),
        Config(
            name="v4/claude",
            metrics_path=Path("datasets/buildings_100_v4/metrics_levels_ablation_full_anthropic_claude_haiku_4_5_20251001.json"),
            dataset_root=Path("datasets/buildings_100_v4"),
            plan_subdir="rebuild_plan_ablation_full_anthropic_claude_haiku_4_5_20251001",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify low-IoU failures into fallback / plan_collapse / material_mismatch.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Optional config override. Format: name|metrics_json|dataset_root|plan_subdir. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--low_iou_threshold", type=float, default=0.20, help="Low IoU threshold.")
    parser.add_argument(
        "--material_match_threshold",
        type=float,
        default=0.20,
        help="material_match below this is treated as material mismatch signal.",
    )
    parser.add_argument(
        "--plan_collapse_ratio_low",
        type=float,
        default=0.55,
        help="pred/gt non-air ratio lower bound for non-collapse.",
    )
    parser.add_argument(
        "--plan_collapse_ratio_high",
        type=float,
        default=1.80,
        help="pred/gt non-air ratio upper bound for non-collapse.",
    )
    parser.add_argument(
        "--min_operations_for_non_collapse",
        type=int,
        default=25,
        help="Operation count below this is treated as collapse signal.",
    )
    parser.add_argument("--top_k_examples", type=int, default=8, help="Top-K low IoU examples per failure type.")
    parser.add_argument("--out_json", required=True, help="Output JSON path.")
    parser.add_argument("--out_md", required=True, help="Output markdown path.")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_config_list(args: argparse.Namespace) -> List[Config]:
    if not args.config:
        return _parse_default_configs()

    out: List[Config] = []
    for raw in args.config:
        parts = raw.split("|")
        if len(parts) != 4:
            raise SystemExit(f"Invalid --config: {raw}")
        name, metrics_path, dataset_root, plan_subdir = parts
        out.append(
            Config(
                name=name.strip(),
                metrics_path=Path(metrics_path.strip()),
                dataset_root=Path(dataset_root.strip()),
                plan_subdir=plan_subdir.strip(),
            )
        )
    return out


def _safe_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def _safe_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _classify_failure(
    *,
    metrics: Dict[str, Any],
    counts: Dict[str, Any],
    plan: Dict[str, Any],
    material_match_threshold: float,
    plan_collapse_ratio_low: float,
    plan_collapse_ratio_high: float,
    min_operations_for_non_collapse: int,
) -> Tuple[str, Dict[str, Any]]:
    parse_report = _safe_dict(plan.get("parse_report"))
    coerce_report = _safe_dict(plan.get("coerce_report"))
    validation_report = _safe_dict(plan.get("validation_report"))
    strict_blocking_issues = _safe_list(validation_report.get("strict_blocking_issues"))
    budget_violations = _safe_list(validation_report.get("budget_violations"))
    operations = _safe_list(plan.get("operations"))

    fallback_signal = bool(parse_report.get("used_ops_array_repair")) or bool(
        parse_report.get("used_op_objects_repair")
    ) or int(coerce_report.get("repaired_operations_count") or 0) > 0 or int(
        coerce_report.get("dropped_operations_count") or 0
    ) > 0

    material_match = float(metrics.get("material_match") or 0.0)
    material_signal = (
        material_match < material_match_threshold
        or "material_budget_violation" in strict_blocking_issues
        or len(budget_violations) > 0
    )

    gt_non_air = int(counts.get("gt_non_air") or 0)
    pred_non_air = int(counts.get("pred_non_air_after_shift") or counts.get("pred_non_air") or 0)
    ratio = (float(pred_non_air) / float(gt_non_air)) if gt_non_air > 0 else 0.0
    op_count = len(operations)
    collapse_signal = ratio < plan_collapse_ratio_low or ratio > plan_collapse_ratio_high or op_count < int(
        min_operations_for_non_collapse
    )

    # primary cause precedence
    if fallback_signal:
        cause = "fallback"
    elif material_signal:
        cause = "material_mismatch"
    else:
        cause = "plan_collapse"

    # If classified as collapse but no collapse evidence and material evidence exists, prefer material.
    if cause == "plan_collapse" and (not collapse_signal) and material_signal:
        cause = "material_mismatch"

    signals = {
        "fallback_signal": fallback_signal,
        "material_signal": material_signal,
        "collapse_signal": collapse_signal,
        "parse_method": str(parse_report.get("method") or ""),
        "used_ops_array_repair": bool(parse_report.get("used_ops_array_repair")),
        "used_op_objects_repair": bool(parse_report.get("used_op_objects_repair")),
        "repaired_operations_count": int(coerce_report.get("repaired_operations_count") or 0),
        "dropped_operations_count": int(coerce_report.get("dropped_operations_count") or 0),
        "strict_blocking_issues_count": len(strict_blocking_issues),
        "budget_violations_count": len(budget_violations),
        "material_match": material_match,
        "gt_non_air": gt_non_air,
        "pred_non_air_after_shift": pred_non_air,
        "pred_gt_ratio": ratio,
        "operation_count": op_count,
    }
    return cause, signals


def _recommendation_for_cause(cause: str) -> str:
    if cause == "material_mismatch":
        return "材質budget再投影と role固定の閾値再調整（budget違反削減を最優先）"
    if cause == "plan_collapse":
        return "粗形状の下限保証（最小wall/roof/floor密度）と operation最小本数ガード"
    if cause == "fallback":
        return "出力スキーマ拘束強化と parser自動修復ルール拡張（ops配列復旧依存を減らす）"
    return "原因不明"


def _priority_score(cases: List[Dict[str, Any]], low_iou_threshold: float) -> float:
    if not cases:
        return 0.0
    mean_iou = mean(float(c["iou"]) for c in cases)
    return float(len(cases) * max(0.0, low_iou_threshold - mean_iou))


def _analyze_config(config: Config, args: argparse.Namespace) -> Dict[str, Any]:
    metrics_obj = _load_json(config.metrics_path)
    items = metrics_obj.get("items", [])
    if not isinstance(items, list):
        raise RuntimeError(f"Invalid items list: {config.metrics_path}")

    low_cases: List[Dict[str, Any]] = []
    per_cause = defaultdict(list)

    for item in items:
        if not isinstance(item, dict):
            continue
        building = str(item.get("building") or "").strip()
        if not building:
            continue
        metrics = _safe_dict(item.get("metrics"))
        iou = float(metrics.get("iou") or 0.0)
        if iou >= float(args.low_iou_threshold):
            continue

        plan_path = config.dataset_root / building / config.plan_subdir / "plan.json"
        if not plan_path.exists():
            continue
        plan = _load_json(plan_path)
        cause, signals = _classify_failure(
            metrics=metrics,
            counts=_safe_dict(item.get("counts")),
            plan=plan,
            material_match_threshold=float(args.material_match_threshold),
            plan_collapse_ratio_low=float(args.plan_collapse_ratio_low),
            plan_collapse_ratio_high=float(args.plan_collapse_ratio_high),
            min_operations_for_non_collapse=int(args.min_operations_for_non_collapse),
        )

        row = {
            "config": config.name,
            "building": building,
            "iou": iou,
            "f1": float(metrics.get("f1") or 0.0),
            "material_match": float(metrics.get("material_match") or 0.0),
            "coarse_material_match": float(metrics.get("coarse_material_match") or 0.0),
            "cause": cause,
            "signals": signals,
            "metrics_path": str(config.metrics_path),
            "plan_path": str(plan_path),
        }
        low_cases.append(row)
        per_cause[cause].append(row)

    cause_counts = {k: len(v) for k, v in sorted(per_cause.items(), key=lambda kv: (-len(kv[1]), kv[0]))}
    cause_stats: Dict[str, Dict[str, Any]] = {}
    for cause, rows in per_cause.items():
        cause_stats[cause] = {
            "count": len(rows),
            "mean_iou": mean(float(r["iou"]) for r in rows),
            "mean_f1": mean(float(r["f1"]) for r in rows),
            "mean_material_match": mean(float(r["material_match"]) for r in rows),
            "mean_pred_gt_ratio": mean(float(r["signals"]["pred_gt_ratio"]) for r in rows),
            "mean_operation_count": mean(float(r["signals"]["operation_count"]) for r in rows),
            "priority_score": _priority_score(rows, float(args.low_iou_threshold)),
            "recommendation": _recommendation_for_cause(cause),
        }

    examples_by_cause: Dict[str, List[Dict[str, Any]]] = {}
    top_k = int(args.top_k_examples)
    for cause, rows in per_cause.items():
        rows_sorted = sorted(rows, key=lambda r: (r["iou"], r["material_match"]))
        examples_by_cause[cause] = rows_sorted[:top_k]

    return {
        "config": config.name,
        "metrics_path": str(config.metrics_path),
        "plan_subdir": config.plan_subdir,
        "low_iou_threshold": float(args.low_iou_threshold),
        "total_items": len(items),
        "low_iou_count": len(low_cases),
        "low_iou_ratio": (len(low_cases) / len(items)) if items else 0.0,
        "cause_counts": cause_counts,
        "cause_stats": cause_stats,
        "examples_by_cause": examples_by_cause,
        "low_iou_cases": low_cases,
    }


def _aggregate(config_results: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    all_cases: List[Dict[str, Any]] = []
    for r in config_results:
        all_cases.extend(r.get("low_iou_cases", []))

    by_cause = defaultdict(list)
    by_config_cause: Dict[str, Counter] = {}
    for r in config_results:
        c = Counter()
        for row in r.get("low_iou_cases", []):
            cause = row.get("cause")
            c[cause] += 1
            by_cause[cause].append(row)
        by_config_cause[r["config"]] = c

    total_low = len(all_cases)
    cause_counts = {k: len(v) for k, v in sorted(by_cause.items(), key=lambda kv: (-len(kv[1]), kv[0]))}
    cause_stats: Dict[str, Dict[str, Any]] = {}
    for cause, rows in by_cause.items():
        cause_stats[cause] = {
            "count": len(rows),
            "share_of_low_iou": (len(rows) / total_low) if total_low else 0.0,
            "mean_iou": mean(float(r["iou"]) for r in rows),
            "mean_f1": mean(float(r["f1"]) for r in rows),
            "mean_material_match": mean(float(r["material_match"]) for r in rows),
            "mean_pred_gt_ratio": mean(float(r["signals"]["pred_gt_ratio"]) for r in rows),
            "mean_operation_count": mean(float(r["signals"]["operation_count"]) for r in rows),
            "priority_score": _priority_score(rows, float(args.low_iou_threshold)),
            "recommendation": _recommendation_for_cause(cause),
        }

    priority_list = sorted(
        (
            {
                "cause": cause,
                **stats,
            }
            for cause, stats in cause_stats.items()
        ),
        key=lambda x: (-float(x["priority_score"]), -int(x["count"]), str(x["cause"])),
    )

    return {
        "total_low_iou_cases": total_low,
        "cause_counts": cause_counts,
        "cause_stats": cause_stats,
        "by_config_cause_counts": {k: dict(v) for k, v in by_config_cause.items()},
        "priority_ranking": priority_list,
    }


def _write_markdown(
    *,
    out_path: Path,
    config_results: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines: List[str] = []
    lines.append("# 低IoU失敗ケースの系統分析")
    lines.append("")
    lines.append("## 判定設定")
    lines.append(f"- low_iou_threshold: `{float(args.low_iou_threshold):.3f}`")
    lines.append(f"- material_match_threshold: `{float(args.material_match_threshold):.3f}`")
    lines.append(
        f"- plan_collapse_ratio: `< {float(args.plan_collapse_ratio_low):.2f}` または `> {float(args.plan_collapse_ratio_high):.2f}`"
    )
    lines.append(f"- min_operations_for_non_collapse: `{int(args.min_operations_for_non_collapse)}`")
    lines.append("")
    lines.append("## 失敗タイプ別件数（全体）")
    lines.append("")
    lines.append("| cause | count | share of low-IoU | mean IoU | mean F1 | mean material |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    total_low = int(aggregate.get("total_low_iou_cases", 0))
    for cause, stats in aggregate.get("cause_stats", {}).items():
        lines.append(
            "| {c} | {n} | {s:.1%} | {iou:.4f} | {f1:.4f} | {mat:.4f} |".format(
                c=cause,
                n=int(stats.get("count", 0)),
                s=float(stats.get("share_of_low_iou", 0.0)),
                iou=float(stats.get("mean_iou", 0.0)),
                f1=float(stats.get("mean_f1", 0.0)),
                mat=float(stats.get("mean_material_match", 0.0)),
            )
        )
    lines.append("")
    lines.append(f"- total low-IoU cases: `{total_low}`")
    lines.append("")
    lines.append("## 失敗タイプ別件数（設定別）")
    lines.append("")
    lines.append("| config | low-IoU count | fallback | plan_collapse | material_mismatch |")
    lines.append("|---|---:|---:|---:|---:|")
    by_cfg = aggregate.get("by_config_cause_counts", {})
    for r in config_results:
        cfg = r.get("config", "")
        cc = by_cfg.get(cfg, {})
        lines.append(
            "| {cfg} | {low} | {fb} | {pc} | {mm} |".format(
                cfg=cfg,
                low=int(r.get("low_iou_count", 0)),
                fb=int(cc.get("fallback", 0)),
                pc=int(cc.get("plan_collapse", 0)),
                mm=int(cc.get("material_mismatch", 0)),
            )
        )
    lines.append("")
    lines.append("## 次の修正優先順位")
    lines.append("")
    lines.append("| priority | cause | score | rationale | next fix |")
    lines.append("|---:|---|---:|---|---|")
    for idx, row in enumerate(aggregate.get("priority_ranking", []), start=1):
        lines.append(
            "| {idx} | {cause} | {score:.4f} | count={count}, mean_iou={iou:.4f} | {fix} |".format(
                idx=idx,
                cause=row.get("cause"),
                score=float(row.get("priority_score", 0.0)),
                count=int(row.get("count", 0)),
                iou=float(row.get("mean_iou", 0.0)),
                fix=row.get("recommendation", ""),
            )
        )
    lines.append("")
    lines.append("## 参考: 最悪ケース例（各原因Top）")
    lines.append("")
    for cause in ("fallback", "plan_collapse", "material_mismatch"):
        lines.append(f"### {cause}")
        lines.append("")
        lines.append("| config | building | iou | f1 | material | parse_method | budget_viol | op_count | pred/gt |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|")
        ex_rows: List[Dict[str, Any]] = []
        for r in config_results:
            ex_rows.extend(r.get("examples_by_cause", {}).get(cause, []))
        ex_rows = sorted(ex_rows, key=lambda x: (x["iou"], x["material_match"]))[: int(args.top_k_examples)]
        for e in ex_rows:
            sig = e.get("signals", {})
            lines.append(
                "| {cfg} | {b} | {iou:.4f} | {f1:.4f} | {mat:.4f} | {pm} | {bv} | {ops} | {ratio:.3f} |".format(
                    cfg=e.get("config", ""),
                    b=e.get("building", ""),
                    iou=float(e.get("iou", 0.0)),
                    f1=float(e.get("f1", 0.0)),
                    mat=float(e.get("material_match", 0.0)),
                    pm=str(sig.get("parse_method", "")),
                    bv=int(sig.get("budget_violations_count", 0)),
                    ops=int(sig.get("operation_count", 0)),
                    ratio=float(sig.get("pred_gt_ratio", 0.0)),
                )
            )
        if not ex_rows:
            lines.append("| (none) | - | - | - | - | - | - | - | - |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configs = _get_config_list(args)

    config_results: List[Dict[str, Any]] = []
    for cfg in configs:
        if not cfg.metrics_path.exists():
            raise SystemExit(f"metrics file not found: {cfg.metrics_path}")
        if not cfg.dataset_root.exists():
            raise SystemExit(f"dataset root not found: {cfg.dataset_root}")
        config_results.append(_analyze_config(cfg, args))

    aggregate = _aggregate(config_results, args)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "low_iou_threshold": float(args.low_iou_threshold),
            "material_match_threshold": float(args.material_match_threshold),
            "plan_collapse_ratio_low": float(args.plan_collapse_ratio_low),
            "plan_collapse_ratio_high": float(args.plan_collapse_ratio_high),
            "min_operations_for_non_collapse": int(args.min_operations_for_non_collapse),
            "top_k_examples": int(args.top_k_examples),
        },
        "configs": [
            {
                "name": c.name,
                "metrics_path": str(c.metrics_path),
                "dataset_root": str(c.dataset_root),
                "plan_subdir": c.plan_subdir,
            }
            for c in configs
        ],
        "config_results": config_results,
        "aggregate": aggregate,
    }

    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_path=out_md, config_results=config_results, aggregate=aggregate, args=args)

    print(f"[analyze_low_iou_failures] wrote: {out_json}")
    print(f"[analyze_low_iou_failures] wrote: {out_md}")
    print("[analyze_low_iou_failures] low-IoU total:", aggregate.get("total_low_iou_cases", 0))
    print("[analyze_low_iou_failures] cause counts:", aggregate.get("cause_counts", {}))


if __name__ == "__main__":
    main()
