#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from tools.plot_experiment_figures import _draw_grouped_bars_svg


ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cases() -> List[Tuple[str, Path, Path, Path]]:
    return [
        (
            "v1/OpenAI",
            ROOT / "datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini.json",
            ROOT / "datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt.json",
            ROOT / "datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned.json",
        ),
        (
            "v1/Claude",
            ROOT / "datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001.json",
            ROOT / "datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt.json",
            ROOT
            / "datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned.json",
        ),
        (
            "v4/OpenAI",
            ROOT / "datasets/buildings_100_v4/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini.json",
            ROOT / "datasets/buildings_100_v4/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt.json",
            ROOT / "datasets/buildings_100_v4/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned.json",
        ),
        (
            "v4/Claude",
            ROOT / "datasets/buildings_100_v4/metrics_levels_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001.json",
            ROOT / "datasets/buildings_100_v4/metrics_levels_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt.json",
            ROOT
            / "datasets/buildings_100_v4/metrics_levels_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned.json",
        ),
    ]


def _metric_values(objs: List[Dict], key: str) -> List[float]:
    return [float(o["aggregate"]["metrics"][key]) for o in objs]


def _evaluated_count(obj: Dict) -> int:
    summary = obj.get("summary", {})
    if isinstance(summary, dict):
        n = int(summary.get("evaluated_buildings", 0) or 0)
        if n > 0:
            return n
    items = obj.get("items", [])
    if isinstance(items, list) and items:
        return len(items)
    return 100


def _weighted_delta(
    baseline: List[Dict],
    compared: List[Dict],
    indices: List[int],
    key: str,
) -> float:
    num = 0.0
    den = 0
    for idx in indices:
        b = baseline[idx]
        c = compared[idx]
        w = _evaluated_count(b)
        dv = float(c["aggregate"]["metrics"][key]) - float(b["aggregate"]["metrics"][key])
        num += dv * w
        den += w
    if den <= 0:
        return 0.0
    return num / float(den)


def main() -> None:
    out_dir = ROOT / "reports/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ja_font = "Hiragino Sans, Yu Gothic, Meiryo, Noto Sans CJK JP, sans-serif"

    categories: List[str] = []
    baseline: List[Dict] = []
    refined_old: List[Dict] = []
    refined_tuned: List[Dict] = []
    for label, b_path, old_path, tuned_path in _cases():
        categories.append(label)
        baseline.append(_load(b_path))
        refined_old.append(_load(old_path))
        refined_tuned.append(_load(tuned_path))

    # Main metric comparisons.
    for key, title, out_name, y_max in [
        ("iou", "self_refine 比較: IoU", "self_refine_iou_ja.svg", 0.4),
        ("f1", "self_refine 比較: F1", "self_refine_f1_ja.svg", 0.6),
        ("material_match", "self_refine 比較: 材質一致率", "self_refine_material_ja.svg", 0.4),
    ]:
        _draw_grouped_bars_svg(
            out_path=out_dir / out_name,
            title=title,
            categories=categories,
            series_names=["baseline(v5_repair)", "self_refine_no_gt(old)", "self_refine_no_gt_tuned"],
            values=[_metric_values(baseline, key), _metric_values(refined_old, key), _metric_values(refined_tuned, key)],
            y_min=0.0,
            y_max=y_max,
            font_family=ja_font,
        )

    # Per-condition deltas (old and tuned).
    d_iou_old = [rv - bv for rv, bv in zip(_metric_values(refined_old, "iou"), _metric_values(baseline, "iou"))]
    d_f1_old = [rv - bv for rv, bv in zip(_metric_values(refined_old, "f1"), _metric_values(baseline, "f1"))]
    d_mat_old = [rv - bv for rv, bv in zip(_metric_values(refined_old, "material_match"), _metric_values(baseline, "material_match"))]
    d_iou_tuned = [rv - bv for rv, bv in zip(_metric_values(refined_tuned, "iou"), _metric_values(baseline, "iou"))]
    d_f1_tuned = [rv - bv for rv, bv in zip(_metric_values(refined_tuned, "f1"), _metric_values(baseline, "f1"))]
    d_mat_tuned = [
        rv - bv for rv, bv in zip(_metric_values(refined_tuned, "material_match"), _metric_values(baseline, "material_match"))
    ]
    _draw_grouped_bars_svg(
        out_path=out_dir / "self_refine_delta_per_condition_ja.svg",
        title="self_refine 差分（old/tuned - baseline）",
        categories=categories,
        series_names=[
            "old IoU差分",
            "tuned IoU差分",
            "old F1差分",
            "tuned F1差分",
            "old 材質差分",
            "tuned 材質差分",
        ],
        values=[d_iou_old, d_iou_tuned, d_f1_old, d_f1_tuned, d_mat_old, d_mat_tuned],
        y_min=-0.05,
        y_max=0.12,
        font_family=ja_font,
    )

    # Weighted deltas over 200/400 samples (old and tuned).
    openai_idx = [0, 2]
    claude_idx = [1, 3]
    all_idx = [0, 1, 2, 3]
    agg_categories = ["OpenAI(200)", "Claude(200)", "全条件(400)"]
    agg_iou_old = [
        _weighted_delta(baseline, refined_old, openai_idx, "iou"),
        _weighted_delta(baseline, refined_old, claude_idx, "iou"),
        _weighted_delta(baseline, refined_old, all_idx, "iou"),
    ]
    agg_f1_old = [
        _weighted_delta(baseline, refined_old, openai_idx, "f1"),
        _weighted_delta(baseline, refined_old, claude_idx, "f1"),
        _weighted_delta(baseline, refined_old, all_idx, "f1"),
    ]
    agg_mat_old = [
        _weighted_delta(baseline, refined_old, openai_idx, "material_match"),
        _weighted_delta(baseline, refined_old, claude_idx, "material_match"),
        _weighted_delta(baseline, refined_old, all_idx, "material_match"),
    ]
    agg_iou_tuned = [
        _weighted_delta(baseline, refined_tuned, openai_idx, "iou"),
        _weighted_delta(baseline, refined_tuned, claude_idx, "iou"),
        _weighted_delta(baseline, refined_tuned, all_idx, "iou"),
    ]
    agg_f1_tuned = [
        _weighted_delta(baseline, refined_tuned, openai_idx, "f1"),
        _weighted_delta(baseline, refined_tuned, claude_idx, "f1"),
        _weighted_delta(baseline, refined_tuned, all_idx, "f1"),
    ]
    agg_mat_tuned = [
        _weighted_delta(baseline, refined_tuned, openai_idx, "material_match"),
        _weighted_delta(baseline, refined_tuned, claude_idx, "material_match"),
        _weighted_delta(baseline, refined_tuned, all_idx, "material_match"),
    ]
    _draw_grouped_bars_svg(
        out_path=out_dir / "self_refine_weighted_delta_ja.svg",
        title="self_refine 加重平均差分",
        categories=agg_categories,
        series_names=[
            "old IoU差分",
            "tuned IoU差分",
            "old F1差分",
            "tuned F1差分",
            "old 材質差分",
            "tuned 材質差分",
        ],
        values=[agg_iou_old, agg_iou_tuned, agg_f1_old, agg_f1_tuned, agg_mat_old, agg_mat_tuned],
        y_min=-0.01,
        y_max=0.08,
        font_family=ja_font,
    )

    payload = {
        "categories": categories,
        "baseline": baseline,
        "self_refine_no_gt_old": refined_old,
        "self_refine_no_gt_tuned": refined_tuned,
        "delta_per_condition": {
            "iou_old": d_iou_old,
            "f1_old": d_f1_old,
            "material_match_old": d_mat_old,
            "iou_tuned": d_iou_tuned,
            "f1_tuned": d_f1_tuned,
            "material_match_tuned": d_mat_tuned,
        },
        "weighted_delta": {
            "categories": agg_categories,
            "iou_old": agg_iou_old,
            "f1_old": agg_f1_old,
            "material_match_old": agg_mat_old,
            "iou_tuned": agg_iou_tuned,
            "f1_tuned": agg_f1_tuned,
            "material_match_tuned": agg_mat_tuned,
        },
    }
    (out_dir / "self_refine_data_2026-03-01.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[plot_self_refine_comparison] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
