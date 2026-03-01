#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from tools.plot_experiment_figures import _draw_grouped_bars_svg


ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_runs(summary: Dict) -> List[Dict]:
    run_map: Dict[str, Dict] = {}
    for run in summary.get("runs", []):
        provider = "openai" if str(run.get("model", "")).startswith("openai") else "claude"
        key = f"{run.get('dataset')}/{provider}"
        run_map[key] = run
    order = ["v1/openai", "v1/claude", "v4/openai", "v4/claude"]
    return [run_map[k] for k in order if k in run_map]


def _label(run: Dict) -> str:
    provider = "OpenAI" if str(run.get("model", "")).startswith("openai") else "Claude"
    return f"{run.get('dataset')}/{provider}"


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> None:
    summary_path = ROOT / "reports/fallback_reduction_parser_v6_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary JSON: {summary_path}")

    summary = _load(summary_path)
    runs = _ordered_runs(summary)
    if not runs:
        raise SystemExit("No runs found in fallback summary.")

    out_dir = ROOT / "reports/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ja_font = "Hiragino Sans, Yu Gothic, Meiryo, Noto Sans CJK JP, sans-serif"

    categories = [_label(r) for r in runs]

    base_missing_rate = [float(r["plan_baseline"]["missing_or_none_rate"]) for r in runs]
    new_missing_rate = [float(r["plan_parser_v6"]["missing_or_none_rate"]) for r in runs]
    base_empty_rate = [float(r["plan_baseline"]["accepted_zero_rate"]) for r in runs]
    new_empty_rate = [float(r["plan_parser_v6"]["accepted_zero_rate"]) for r in runs]

    _draw_grouped_bars_svg(
        out_path=out_dir / "parser_v6_fallback_rates_ja.svg",
        title="parser_v6導入前後: fallback関連率",
        categories=categories,
        series_names=["baseline missing/none率", "parser_v6 missing/none率", "baseline 空operations率", "parser_v6 空operations率"],
        values=[base_missing_rate, new_missing_rate, base_empty_rate, new_empty_rate],
        y_min=0.0,
        y_max=0.5,
        font_family=ja_font,
    )

    base_accept_avg = [float(r["plan_baseline"]["accepted_avg"]) for r in runs]
    new_accept_avg = [float(r["plan_parser_v6"]["accepted_avg"]) for r in runs]
    y_max_ops = max(base_accept_avg + new_accept_avg) + 5.0
    _draw_grouped_bars_svg(
        out_path=out_dir / "parser_v6_operation_acceptance_ja.svg",
        title="parser_v6導入前後: 平均採用operations数",
        categories=categories,
        series_names=["baseline accepted_avg", "parser_v6 accepted_avg"],
        values=[base_accept_avg, new_accept_avg],
        y_min=0.0,
        y_max=y_max_ops,
        font_family=ja_font,
    )

    d_iou = [float(r["metrics_delta"]["iou"]) for r in runs]
    d_f1 = [float(r["metrics_delta"]["f1"]) for r in runs]
    d_mat = [float(r["metrics_delta"]["material_match"]) for r in runs]
    d_coarse = [float(r["metrics_delta"]["coarse_material_match"]) for r in runs]
    _draw_grouped_bars_svg(
        out_path=out_dir / "parser_v6_metric_delta_ja.svg",
        title="parser_v6導入差分（parser_v6 - baseline）",
        categories=categories,
        series_names=["IoU差分", "F1差分", "material差分", "coarse_material差分"],
        values=[d_iou, d_f1, d_mat, d_coarse],
        y_min=-0.2,
        y_max=0.08,
        font_family=ja_font,
    )

    openai = [r for r in runs if str(r.get("model", "")).startswith("openai")]
    claude = [r for r in runs if str(r.get("model", "")).startswith("anthropic")]
    all_runs = runs

    agg_categories = ["OpenAI(200)", "Claude(200)", "全条件(400)"]
    agg_iou = [_mean([r["metrics_delta"]["iou"] for r in openai]), _mean([r["metrics_delta"]["iou"] for r in claude]), _mean([r["metrics_delta"]["iou"] for r in all_runs])]
    agg_f1 = [_mean([r["metrics_delta"]["f1"] for r in openai]), _mean([r["metrics_delta"]["f1"] for r in claude]), _mean([r["metrics_delta"]["f1"] for r in all_runs])]
    agg_mat = [_mean([r["metrics_delta"]["material_match"] for r in openai]), _mean([r["metrics_delta"]["material_match"] for r in claude]), _mean([r["metrics_delta"]["material_match"] for r in all_runs])]
    agg_coarse = [
        _mean([r["metrics_delta"]["coarse_material_match"] for r in openai]),
        _mean([r["metrics_delta"]["coarse_material_match"] for r in claude]),
        _mean([r["metrics_delta"]["coarse_material_match"] for r in all_runs]),
    ]
    _draw_grouped_bars_svg(
        out_path=out_dir / "parser_v6_weighted_delta_ja.svg",
        title="parser_v6導入差分（モデル別/全体平均）",
        categories=agg_categories,
        series_names=["IoU差分", "F1差分", "material差分", "coarse_material差分"],
        values=[agg_iou, agg_f1, agg_mat, agg_coarse],
        y_min=-0.14,
        y_max=0.07,
        font_family=ja_font,
    )

    payload = {
        "created_at": summary.get("created_at"),
        "categories": categories,
        "fallback_rates": {
            "baseline_missing_or_none_rate": base_missing_rate,
            "parser_v6_missing_or_none_rate": new_missing_rate,
            "baseline_accepted_zero_rate": base_empty_rate,
            "parser_v6_accepted_zero_rate": new_empty_rate,
        },
        "accepted_avg": {
            "baseline": base_accept_avg,
            "parser_v6": new_accept_avg,
        },
        "metric_delta_per_condition": {
            "iou": d_iou,
            "f1": d_f1,
            "material_match": d_mat,
            "coarse_material_match": d_coarse,
        },
        "metric_delta_aggregate": {
            "categories": agg_categories,
            "iou": agg_iou,
            "f1": agg_f1,
            "material_match": agg_mat,
            "coarse_material_match": agg_coarse,
        },
        "notes": summary.get("notes", []),
    }
    (out_dir / "parser_v6_data_2026-03-02.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[plot_fallback_reduction_parser_v6] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
