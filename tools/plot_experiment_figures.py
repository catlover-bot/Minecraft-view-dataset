#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _draw_grouped_bars_svg(
    out_path: Path,
    title: str,
    categories: Sequence[str],
    series_names: Sequence[str],
    values: Sequence[Sequence[float]],
    y_min: float,
    y_max: float,
    y_ticks: int = 5,
    font_family: str = "Arial, sans-serif",
) -> None:
    width, height = 1800, 1080
    margin_l, margin_r, margin_t, margin_b = 140, 80, 120, 220
    x0, y0 = margin_l, margin_t
    x1, y1 = width - margin_r, height - margin_b

    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#14B8A6", "#F97316"]
    n_cat = len(categories)
    n_series = len(series_names)

    def y_of(v: float) -> float:
        if y_max == y_min:
            return float(y1)
        t = (v - y_min) / (y_max - y_min)
        return y1 - t * (y1 - y0)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{(x0 + x1) / 2:.1f}" y="58" text-anchor="middle" font-size="40" font-family="{_esc(font_family)}" fill="#111111">{_esc(title)}</text>'
    )

    # Plot frame.
    lines.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="#333333" stroke-width="2"/>')

    # Grid + y ticks.
    for i in range(y_ticks + 1):
        t = i / max(1, y_ticks)
        y = y1 - t * (y1 - y0)
        v = y_min + t * (y_max - y_min)
        lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="16" y="{y + 6:.1f}" font-size="20" font-family="{_esc(font_family)}" fill="#374151">{v:.2f}</text>'
        )

    # Axes.
    lines.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#374151" stroke-width="2"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#374151" stroke-width="2"/>')

    zero_y = y_of(0.0)
    if y_min < 0 < y_max:
        lines.append(f'<line x1="{x0}" y1="{zero_y:.1f}" x2="{x1}" y2="{zero_y:.1f}" stroke="#9CA3AF" stroke-width="2"/>')

    if n_cat > 0 and n_series > 0:
        plot_w = x1 - x0
        group_w = plot_w / n_cat
        bar_group_w = group_w * 0.72
        bar_w = bar_group_w / n_series
        for ci, cat in enumerate(categories):
            gx = x0 + ci * group_w
            start_x = gx + (group_w - bar_group_w) / 2.0
            lines.append(
                f'<text x="{gx + group_w / 2.0:.1f}" y="{y1 + 48:.1f}" text-anchor="middle" font-size="22" font-family="{_esc(font_family)}" fill="#111827">{_esc(cat)}</text>'
            )
            for si, _name in enumerate(series_names):
                v = float(values[si][ci])
                bx0 = start_x + si * bar_w + 2
                bx1 = start_x + (si + 1) * bar_w - 2
                by = y_of(v)
                top = min(by, zero_y)
                bottom = max(by, zero_y)
                color = colors[si % len(colors)]
                lines.append(
                    f'<rect x="{bx0:.1f}" y="{top:.1f}" width="{max(1.0, bx1 - bx0):.1f}" height="{max(1.0, bottom - top):.1f}" fill="{color}" stroke="#374151" stroke-width="1"/>'
                )
                label_y = top - 10 if v >= 0 else bottom + 20
                lines.append(
                    f'<text x="{(bx0 + bx1) / 2.0:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="17" font-family="{_esc(font_family)}" fill="#111827">{v:.3f}</text>'
                )

    # Legend.
    lx, ly = x1 - 440, y0 + 20
    for i, name in enumerate(series_names):
        yy = ly + i * 34
        color = colors[i % len(colors)]
        lines.append(f'<rect x="{lx}" y="{yy}" width="24" height="20" fill="{color}" stroke="#374151" stroke-width="1"/>')
        lines.append(
            f'<text x="{lx + 34}" y="{yy + 16}" font-size="21" font-family="{_esc(font_family)}" fill="#111827">{_esc(name)}</text>'
        )

    lines.append("</svg>")
    _ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _collect_rebuild_pair_data() -> Dict[str, Dict[str, Dict]]:
    pairs = {
        "v1/openai": {
            "baseline": ROOT / "datasets/buildings_100_v1/metrics_levels_pe_v2_openai_gpt_5_mini.json",
            "repaired": ROOT / "datasets/buildings_100_v1/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json",
        },
        "v1/claude": {
            "baseline": ROOT / "datasets/buildings_100_v1/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001.json",
            "repaired": ROOT / "datasets/buildings_100_v1/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001_schema_repair_20260301.json",
        },
        "v4/openai": {
            "baseline": ROOT / "datasets/buildings_100_v4/metrics_levels_pe_v2_openai_gpt_5_mini.json",
            "repaired": ROOT / "datasets/buildings_100_v4/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json",
        },
        "v4/claude": {
            "baseline": ROOT / "datasets/buildings_100_v4/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001.json",
            "repaired": ROOT / "datasets/buildings_100_v4/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001_schema_repair_20260301.json",
        },
    }
    out: Dict[str, Dict[str, Dict]] = {}
    for key, paths in pairs.items():
        out[key] = {
            "baseline": _load_json(paths["baseline"]),
            "repaired": _load_json(paths["repaired"]),
        }
    return out


def _collect_description_data() -> Dict[str, Dict]:
    files = {
        "v1/openai": ROOT / "datasets/buildings_100_v1/description_metrics_openai_gpt_5_mini.json",
        "v1/claude": ROOT / "datasets/buildings_100_v1/description_metrics_anthropic_claude_haiku_4_5_20251001.json",
        "v4/openai": ROOT / "datasets/buildings_100_v4/description_metrics_openai_gpt_5_mini.json",
        "v4/claude": ROOT / "datasets/buildings_100_v4/description_metrics_anthropic_claude_haiku_4_5_20251001.json",
    }
    return {k: _load_json(v) for k, v in files.items()}


def _collect_plan_note_stats() -> Dict[str, Dict[str, float]]:
    targets = {
        "v1/openai": (ROOT / "datasets/buildings_100_v1", "rebuild_plan_pe_v2_openai_gpt_5_mini"),
        "v1/claude": (ROOT / "datasets/buildings_100_v1", "rebuild_plan_pe_v2_anthropic_claude_haiku_4_5_20251001"),
        "v4/openai": (ROOT / "datasets/buildings_100_v4", "rebuild_plan_pe_v2_openai_gpt_5_mini"),
        "v4/claude": (ROOT / "datasets/buildings_100_v4", "rebuild_plan_pe_v2_anthropic_claude_haiku_4_5_20251001"),
    }
    out: Dict[str, Dict[str, float]] = {}
    for key, (root, subdir) in targets.items():
        total = fallback = repaired = draft_salvage = 0
        for p in root.glob(f"building_*/{subdir}/plan.json"):
            total += 1
            obj = _load_json(p)
            notes = {str(x) for x in obj.get("notes", []) if isinstance(obj.get("notes", []), list)}
            if "fallback_used_due_to_empty_operations" in notes:
                fallback += 1
            if "schema_repair_from_raw_response" in notes:
                repaired += 1
            if "schema_repair_used_draft_when_revise_invalid" in notes:
                draft_salvage += 1
        if total == 0:
            out[key] = {
                "total": 0.0,
                "fallback_rate": 0.0,
                "repaired_rate": 0.0,
                "draft_salvage_rate": 0.0,
            }
        else:
            out[key] = {
                "total": float(total),
                "fallback_rate": fallback / total,
                "repaired_rate": repaired / total,
                "draft_salvage_rate": draft_salvage / total,
            }
    return out


def main() -> None:
    out_dir = ROOT / "reports/figures"
    _ensure_dir(out_dir)

    rebuild_pairs = _collect_rebuild_pair_data()
    desc_data = _collect_description_data()
    note_stats = _collect_plan_note_stats()

    categories = ["v1/OpenAI", "v1/Claude", "v4/OpenAI", "v4/Claude"]
    key_order = ["v1/openai", "v1/claude", "v4/openai", "v4/claude"]
    ja_font = "Hiragino Sans, Yu Gothic, Meiryo, Noto Sans CJK JP, sans-serif"

    iou_vals = [rebuild_pairs[k]["repaired"]["aggregate"]["metrics"]["iou"] for k in key_order]
    f1_vals = [rebuild_pairs[k]["repaired"]["aggregate"]["metrics"]["f1"] for k in key_order]
    mat_vals = [rebuild_pairs[k]["repaired"]["aggregate"]["metrics"]["material_match"] for k in key_order]
    _draw_grouped_bars_svg(
        out_path=out_dir / "rebuild_latest_metrics.svg",
        title="Rebuild Metrics (Latest Repaired Plans)",
        categories=categories,
        series_names=["IoU", "F1", "Material Match"],
        values=[iou_vals, f1_vals, mat_vals],
        y_min=0.0,
        y_max=0.5,
    )
    _draw_grouped_bars_svg(
        out_path=out_dir / "rebuild_latest_metrics_ja.svg",
        title="再建築メトリクス（最新・修復後）",
        categories=categories,
        series_names=["IoU", "F1", "材質一致率"],
        values=[iou_vals, f1_vals, mat_vals],
        y_min=0.0,
        y_max=0.5,
        font_family=ja_font,
    )

    l0 = [rebuild_pairs[k]["repaired"]["aggregate"]["pass_rates"]["level0_shift_pass_rate"] for k in key_order]
    l1 = [rebuild_pairs[k]["repaired"]["aggregate"]["pass_rates"]["level1_shape_pass_rate"] for k in key_order]
    l2 = [rebuild_pairs[k]["repaired"]["aggregate"]["pass_rates"]["level2_coarse_material_pass_rate"] for k in key_order]
    l3 = [rebuild_pairs[k]["repaired"]["aggregate"]["pass_rates"]["level3_strict_material_pass_rate"] for k in key_order]
    l4 = [rebuild_pairs[k]["repaired"]["aggregate"]["pass_rates"]["level4_structure_components_pass_rate"] for k in key_order]
    all_levels = [rebuild_pairs[k]["repaired"]["aggregate"]["pass_rates"]["all_levels_pass_rate"] for k in key_order]
    _draw_grouped_bars_svg(
        out_path=out_dir / "rebuild_level_pass_rates.svg",
        title="Rebuild Level Pass Rates",
        categories=categories,
        series_names=["L0 Shift", "L1 Shape", "L2 CoarseMat", "L3 StrictMat", "L4 Components", "All Levels"],
        values=[l0, l1, l2, l3, l4, all_levels],
        y_min=0.0,
        y_max=1.0,
    )
    _draw_grouped_bars_svg(
        out_path=out_dir / "rebuild_level_pass_rates_ja.svg",
        title="再建築 レベル別合格率",
        categories=categories,
        series_names=["L0 位置合わせ", "L1 形状", "L2 粗材質", "L3 厳密材質", "L4 構造成分", "全レベル"],
        values=[l0, l1, l2, l3, l4, all_levels],
        y_min=0.0,
        y_max=1.0,
        font_family=ja_font,
    )

    auto_vals = [desc_data[k]["aggregate"]["auto_score_mean"] for k in key_order]
    strict_vals = [desc_data[k]["aggregate"]["strict_material_f1_mean"] for k in key_order]
    coarse_vals = [desc_data[k]["aggregate"]["coarse_material_f1_mean"] for k in key_order]
    dim_vals = [desc_data[k]["aggregate"]["dimension_score_mean"] for k in key_order]
    _draw_grouped_bars_svg(
        out_path=out_dir / "description_quality_metrics.svg",
        title="Description Quality Metrics",
        categories=categories,
        series_names=["Auto Score", "StrictMat F1", "CoarseMat F1", "Dimension Score"],
        values=[auto_vals, strict_vals, coarse_vals, dim_vals],
        y_min=0.0,
        y_max=1.0,
    )
    _draw_grouped_bars_svg(
        out_path=out_dir / "description_quality_metrics_ja.svg",
        title="説明文品質メトリクス",
        categories=categories,
        series_names=["総合スコア", "厳密材質F1", "粗材質F1", "寸法スコア"],
        values=[auto_vals, strict_vals, coarse_vals, dim_vals],
        y_min=0.0,
        y_max=1.0,
        font_family=ja_font,
    )

    diou = [rebuild_pairs[k]["repaired"]["aggregate"]["metrics"]["iou"] - rebuild_pairs[k]["baseline"]["aggregate"]["metrics"]["iou"] for k in key_order]
    df1 = [rebuild_pairs[k]["repaired"]["aggregate"]["metrics"]["f1"] - rebuild_pairs[k]["baseline"]["aggregate"]["metrics"]["f1"] for k in key_order]
    dmat = [rebuild_pairs[k]["repaired"]["aggregate"]["metrics"]["material_match"] - rebuild_pairs[k]["baseline"]["aggregate"]["metrics"]["material_match"] for k in key_order]
    _draw_grouped_bars_svg(
        out_path=out_dir / "baseline_vs_repaired_deltas.svg",
        title="Delta: Repaired - Baseline Rebuild Metrics",
        categories=categories,
        series_names=["Delta IoU", "Delta F1", "Delta Material"],
        values=[diou, df1, dmat],
        y_min=-0.2,
        y_max=0.08,
    )
    _draw_grouped_bars_svg(
        out_path=out_dir / "baseline_vs_repaired_deltas_ja.svg",
        title="差分：修復後 - ベースライン（再建築）",
        categories=categories,
        series_names=["IoU差分", "F1差分", "材質一致差分"],
        values=[diou, df1, dmat],
        y_min=-0.2,
        y_max=0.08,
        font_family=ja_font,
    )

    fallback_rate = [note_stats[k]["fallback_rate"] for k in key_order]
    repaired_rate = [note_stats[k]["repaired_rate"] for k in key_order]
    draft_rate = [note_stats[k]["draft_salvage_rate"] for k in key_order]
    _draw_grouped_bars_svg(
        out_path=out_dir / "plan_repair_and_fallback_rates.svg",
        title="Plan Schema Repair Coverage and Remaining Fallback",
        categories=categories,
        series_names=["Schema-Repaired Rate", "Draft-Salvage Rate", "Fallback Rate"],
        values=[repaired_rate, draft_rate, fallback_rate],
        y_min=0.0,
        y_max=1.0,
    )
    _draw_grouped_bars_svg(
        out_path=out_dir / "plan_repair_and_fallback_rates_ja.svg",
        title="Planスキーマ修復率とFallback率",
        categories=categories,
        series_names=["スキーマ修復率", "Draft救済率", "Fallback率"],
        values=[repaired_rate, draft_rate, fallback_rate],
        y_min=0.0,
        y_max=1.0,
        font_family=ja_font,
    )

    all_pass_count = [
        sum(1 for item in rebuild_pairs[k]["repaired"]["items"] if bool(item.get("levels", {}).get("all_levels_pass", False)))
        for k in key_order
    ]
    all_pass_rate = [x / 100.0 for x in all_pass_count]
    _draw_grouped_bars_svg(
        out_path=out_dir / "all_levels_pass_rate.svg",
        title="All-Levels Pass Rate (Count / 100)",
        categories=categories,
        series_names=["All-Levels Pass Rate"],
        values=[all_pass_rate],
        y_min=0.0,
        y_max=0.2,
    )
    _draw_grouped_bars_svg(
        out_path=out_dir / "all_levels_pass_rate_ja.svg",
        title="全レベル合格率（100件あたり）",
        categories=categories,
        series_names=["全レベル合格率"],
        values=[all_pass_rate],
        y_min=0.0,
        y_max=0.2,
        font_family=ja_font,
    )

    bundle = {
        "rebuild_pairs": rebuild_pairs,
        "description": desc_data,
        "plan_note_stats": note_stats,
        "categories": categories,
        "key_order": key_order,
    }
    (out_dir / "figure_data_2026-03-01.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[plot_experiment_figures] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
