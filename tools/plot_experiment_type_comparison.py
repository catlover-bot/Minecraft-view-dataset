#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from tools.plot_experiment_figures import _draw_grouped_bars_svg


ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_paths() -> List[Tuple[str, Path, Path, Path]]:
    return [
        (
            "v1/OpenAI",
            ROOT / "datasets/buildings_100_v1/metrics_levels_openai_gpt_5_mini.json",
            ROOT / "datasets/buildings_100_v1/metrics_levels_pe_v2_openai_gpt_5_mini.json",
            ROOT / "datasets/buildings_100_v1/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json",
        ),
        (
            "v1/Claude",
            ROOT / "datasets/buildings_100_v1/metrics_levels_anthropic_claude_haiku_4_5_20251001.json",
            ROOT / "datasets/buildings_100_v1/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001.json",
            ROOT / "datasets/buildings_100_v1/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001_schema_repair_20260301.json",
        ),
        (
            "v4/OpenAI",
            ROOT / "datasets/buildings_100_v4/metrics_levels_openai_gpt_5_mini.json",
            ROOT / "datasets/buildings_100_v4/metrics_levels_pe_v2_openai_gpt_5_mini.json",
            ROOT / "datasets/buildings_100_v4/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json",
        ),
        (
            "v4/Claude",
            ROOT / "datasets/buildings_100_v4/metrics_levels_anthropic_claude_haiku_4_5_20251001.json",
            ROOT / "datasets/buildings_100_v4/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001.json",
            ROOT / "datasets/buildings_100_v4/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001_schema_repair_20260301.json",
        ),
    ]


def main() -> None:
    out_dir = ROOT / "reports/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ja_font = "Hiragino Sans, Yu Gothic, Meiryo, Noto Sans CJK JP, sans-serif"

    categories: List[str] = []
    baseline: List[Dict] = []
    pe_v2: List[Dict] = []
    repaired: List[Dict] = []

    for label, p_base, p_pe, p_rep in _case_paths():
        categories.append(label)
        baseline.append(_load(p_base))
        pe_v2.append(_load(p_pe))
        repaired.append(_load(p_rep))

    def metric_values(objs: List[Dict], key: str) -> List[float]:
        return [float(o["aggregate"]["metrics"][key]) for o in objs]

    def pass_values(objs: List[Dict], key: str) -> List[float]:
        return [float(o["aggregate"]["pass_rates"][key]) for o in objs]

    series_names = ["ベースライン", "強化プロンプト(pe_v2)", "pe_v2+スキーマ修復"]

    # IoU
    _draw_grouped_bars_svg(
        out_path=out_dir / "two_types_iou_ja.svg",
        title="2種類の実験比較: IoU",
        categories=categories,
        series_names=series_names,
        values=[metric_values(baseline, "iou"), metric_values(pe_v2, "iou"), metric_values(repaired, "iou")],
        y_min=0.0,
        y_max=0.4,
        font_family=ja_font,
    )

    # F1
    _draw_grouped_bars_svg(
        out_path=out_dir / "two_types_f1_ja.svg",
        title="2種類の実験比較: F1",
        categories=categories,
        series_names=series_names,
        values=[metric_values(baseline, "f1"), metric_values(pe_v2, "f1"), metric_values(repaired, "f1")],
        y_min=0.0,
        y_max=0.55,
        font_family=ja_font,
    )

    # Material
    _draw_grouped_bars_svg(
        out_path=out_dir / "two_types_material_ja.svg",
        title="2種類の実験比較: 材質一致率",
        categories=categories,
        series_names=series_names,
        values=[
            metric_values(baseline, "material_match"),
            metric_values(pe_v2, "material_match"),
            metric_values(repaired, "material_match"),
        ],
        y_min=0.0,
        y_max=0.4,
        font_family=ja_font,
    )

    # All-level pass
    _draw_grouped_bars_svg(
        out_path=out_dir / "two_types_all_levels_pass_ja.svg",
        title="2種類の実験比較: 全レベル合格率",
        categories=categories,
        series_names=series_names,
        values=[
            pass_values(baseline, "all_levels_pass_rate"),
            pass_values(pe_v2, "all_levels_pass_rate"),
            pass_values(repaired, "all_levels_pass_rate"),
        ],
        y_min=0.0,
        y_max=0.1,
        font_family=ja_font,
    )

    # Delta (pe_v2 - baseline / repaired - baseline)
    base_iou = metric_values(baseline, "iou")
    pe_iou = metric_values(pe_v2, "iou")
    rep_iou = metric_values(repaired, "iou")
    base_f1 = metric_values(baseline, "f1")
    pe_f1 = metric_values(pe_v2, "f1")
    rep_f1 = metric_values(repaired, "f1")
    base_mat = metric_values(baseline, "material_match")
    pe_mat = metric_values(pe_v2, "material_match")
    rep_mat = metric_values(repaired, "material_match")

    d_pe_iou = [pe_iou[i] - base_iou[i] for i in range(len(categories))]
    d_rep_iou = [rep_iou[i] - base_iou[i] for i in range(len(categories))]
    d_pe_f1 = [pe_f1[i] - base_f1[i] for i in range(len(categories))]
    d_rep_f1 = [rep_f1[i] - base_f1[i] for i in range(len(categories))]
    d_pe_mat = [pe_mat[i] - base_mat[i] for i in range(len(categories))]
    d_rep_mat = [rep_mat[i] - base_mat[i] for i in range(len(categories))]

    _draw_grouped_bars_svg(
        out_path=out_dir / "two_types_delta_vs_baseline_ja.svg",
        title="ベースライン比の差分（+が改善）",
        categories=categories,
        series_names=["pe_v2 IoU差", "修復 IoU差", "pe_v2 F1差", "修復 F1差", "pe_v2 材質差", "修復 材質差"],
        values=[d_pe_iou, d_rep_iou, d_pe_f1, d_rep_f1, d_pe_mat, d_rep_mat],
        y_min=-0.2,
        y_max=0.1,
        font_family=ja_font,
    )

    payload = {
        "categories": categories,
        "baseline": baseline,
        "pe_v2": pe_v2,
        "pe_v2_schema_repaired": repaired,
    }
    (out_dir / "two_types_data_2026-03-01.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[plot_experiment_type_comparison] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
