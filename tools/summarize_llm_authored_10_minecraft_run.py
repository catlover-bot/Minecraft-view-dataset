#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "final"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(v: float) -> float:
    return round(float(v) * 100.0, 4)


def _collect_provenance(dataset_root: Path) -> Dict[str, Any]:
    rows = []
    valid = 0
    for bdir in sorted(p for p in dataset_root.glob("llm_case_*") if p.is_dir()):
        prov_path = bdir / "provenance.json"
        if not prov_path.is_file():
            continue
        p = _load(prov_path)
        sbo = p.get("source_build_origin")
        sio = p.get("source_image_origin")
        dbo = p.get("direct_rebuild_build_origin")
        dio = p.get("direct_rebuild_image_origin")
        tbo = p.get("structured_rebuild_build_origin")
        tio = p.get("structured_rebuild_image_origin")
        ok = (
            sbo == "minecraft_instantiated"
            and sio == "minecraft_capture"
            and dbo == "minecraft_instantiated"
            and dio == "minecraft_capture"
            and tbo == "minecraft_instantiated"
            and tio == "minecraft_capture"
        )
        if ok:
            valid += 1
        rows.append(
            {
                "case_id": bdir.name,
                "source_build_origin": sbo,
                "source_image_origin": sio,
                "direct_rebuild_build_origin": dbo,
                "direct_rebuild_image_origin": dio,
                "structured_rebuild_build_origin": tbo,
                "structured_rebuild_image_origin": tio,
                "source_capture_script": p.get("source_capture_script"),
                "source_capture_timestamp": p.get("source_capture_timestamp"),
                "direct_rebuild_capture_script": p.get("direct_rebuild_capture_script"),
                "direct_rebuild_capture_timestamp": p.get("direct_rebuild_capture_timestamp"),
                "structured_rebuild_capture_script": p.get("structured_rebuild_capture_script"),
                "structured_rebuild_capture_timestamp": p.get("structured_rebuild_capture_timestamp"),
                "active_python_interpreter": p.get("active_python_interpreter"),
                "active_python_version": p.get("active_python_version"),
                "malmo_python_path": p.get("malmo_python_path"),
                "fallback_used": p.get("any_fallback_used"),
                "valid_minecraft_grounded": ok,
            }
        )
    return {
        "dataset_root": str(dataset_root),
        "total_cases": len(rows),
        "valid_minecraft_grounded_cases": valid,
        "invalid_or_missing_cases": len(rows) - valid,
        "cases": rows,
    }


def _model_block(results_json: Path) -> Dict[str, Any]:
    r = _load(results_json)
    return {
        "provider_tag": r.get("provider_tag"),
        "source_dataset_root": r.get("source_dataset_root"),
        "description_summary": r.get("description_summary", {}),
        "direct_rebuild_summary": r.get("direct_rebuild_summary", {}),
        "structured_rebuild_summary": r.get("structured_rebuild_summary", {}),
        "difficulty_breakdown": r.get("difficulty_breakdown", {}),
        "delta_structured_minus_direct": r.get("comparison_delta_structured_minus_direct", {}),
    }


def _synthetic_block(results_json: Path) -> Dict[str, Any]:
    r = _load(results_json)
    return {
        "provider_tag": r.get("provider_tag"),
        "source_dataset_root": r.get("source_dataset_root"),
        "description_summary": r.get("description_summary", {}),
        "direct_rebuild_summary": r.get("direct_rebuild_summary", {}),
        "structured_rebuild_summary": r.get("structured_rebuild_summary", {}),
        "delta_structured_minus_direct": r.get("comparison_delta_structured_minus_direct", {}),
    }


def main() -> None:
    openai_res = REPORTS / "llm_authored_10_results_openai_gpt_5_mini.json"
    claude_res = REPORTS / "llm_authored_10_results_anthropic_claude_haiku_4_5_20251001.json"
    openai_prev = REPORTS / "llm_authored_10_results_openai_gpt_5_mini_synthetic_prev.json"
    claude_prev = REPORTS / "llm_authored_10_results_anthropic_claude_haiku_4_5_20251001_synthetic_prev.json"

    now = datetime.now(timezone.utc).isoformat()

    openai_block = _model_block(openai_res)
    claude_block = _model_block(claude_res)
    openai_prev_block = _synthetic_block(openai_prev)
    claude_prev_block = _synthetic_block(claude_prev)

    prov_openai = _collect_provenance(Path(openai_block["source_dataset_root"]))
    prov_claude = _collect_provenance(Path(claude_block["source_dataset_root"]))

    result = {
        "created_at": now,
        "study": "llm_authored_10_minecraft_grounded_diagnostic",
        "separation_note": "Diagnostic set only. Not a replacement for the 200-building main benchmark.",
        "openai": openai_block,
        "claude": claude_block,
        "provenance_verification": {
            "openai_dataset": prov_openai,
            "claude_dataset": prov_claude,
        },
        "prior_ambiguous_or_synthetic_reference": {
            "openai": openai_prev_block,
            "claude": claude_prev_block,
        },
    }

    out_json = REPORTS / "llm_authored_10_minecraft_results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    o_d = openai_block["direct_rebuild_summary"]
    o_s = openai_block["structured_rebuild_summary"]
    c_d = claude_block["direct_rebuild_summary"]
    c_s = claude_block["structured_rebuild_summary"]

    md = []
    md.append("# llm_authored_10 Minecraft Grounded Summary")
    md.append("")
    md.append("このレポートは `llm_authored_10` 診断セットの **Minecraft実画像入力版** です。")
    md.append("メイン200件ベンチ（Main/Supplementary/Execution-gap）とは混ぜません。")
    md.append("")
    md.append("## 実行条件")
    md.append(f"- OpenAI dataset: `{openai_block['source_dataset_root']}`")
    md.append(f"- Claude dataset: `{claude_block['source_dataset_root']}`")
    md.append("- source image origin: `minecraft_capture`")
    md.append("- source build origin: `minecraft_instantiated`")
    md.append("- direct rebuild image origin: `minecraft_capture`")
    md.append("- direct rebuild build origin: `minecraft_instantiated`")
    md.append("- structured rebuild image origin: `minecraft_capture`")
    md.append("- structured rebuild build origin: `minecraft_instantiated`")
    md.append("")
    md.append("## Provenance検証")
    md.append(
        f"- OpenAI valid cases (source+direct+structured all Minecraft grounded): "
        f"{prov_openai['valid_minecraft_grounded_cases']}/{prov_openai['total_cases']}"
    )
    md.append(
        f"- Claude valid cases (source+direct+structured all Minecraft grounded): "
        f"{prov_claude['valid_minecraft_grounded_cases']}/{prov_claude['total_cases']}"
    )
    md.append("")
    md.append("## OpenAI (gpt-5-mini)")
    md.append(
        f"- Description: auto {openai_block['description_summary']['auto_score_pct']:.2f}%, "
        f"strict material F1 {openai_block['description_summary']['strict_material_f1_pct']:.2f}%, "
        f"coarse material F1 {openai_block['description_summary']['coarse_material_f1_pct']:.2f}%, "
        f"dimension {openai_block['description_summary']['dimension_score_pct']:.2f}%"
    )
    md.append(
        f"- Direct rebuild: IoU {o_d['iou_pct']:.2f}%, F1 {o_d['f1_pct']:.2f}%, "
        f"material {o_d['material_match_pct']:.2f}%, correct placement {o_d['correct_placement_pct']:.2f}%, "
        f"repair edit {o_d['repair_edit_distance']:.3f}"
    )
    md.append(
        f"- Structured rebuild: IoU {o_s['iou_pct']:.2f}%, F1 {o_s['f1_pct']:.2f}%, "
        f"material {o_s['material_match_pct']:.2f}%, correct placement {o_s['correct_placement_pct']:.2f}%, "
        f"repair edit {o_s['repair_edit_distance']:.3f}"
    )
    md.append(
        f"- Structured - Direct: IoU {o_s['iou_pct']-o_d['iou_pct']:+.2f} pt, "
        f"F1 {o_s['f1_pct']-o_d['f1_pct']:+.2f} pt, material {o_s['material_match_pct']-o_d['material_match_pct']:+.2f} pt, "
        f"correct placement {o_s['correct_placement_pct']-o_d['correct_placement_pct']:+.2f} pt"
    )
    md.append("")
    md.append("## Claude (claude-haiku-4-5)")
    md.append(
        f"- Description: auto {claude_block['description_summary']['auto_score_pct']:.2f}%, "
        f"strict material F1 {claude_block['description_summary']['strict_material_f1_pct']:.2f}%, "
        f"coarse material F1 {claude_block['description_summary']['coarse_material_f1_pct']:.2f}%, "
        f"dimension {claude_block['description_summary']['dimension_score_pct']:.2f}%"
    )
    md.append(
        f"- Direct rebuild: IoU {c_d['iou_pct']:.2f}%, F1 {c_d['f1_pct']:.2f}%, "
        f"material {c_d['material_match_pct']:.2f}%, correct placement {c_d['correct_placement_pct']:.2f}%, "
        f"repair edit {c_d['repair_edit_distance']:.3f}"
    )
    md.append(
        f"- Structured rebuild: IoU {c_s['iou_pct']:.2f}%, F1 {c_s['f1_pct']:.2f}%, "
        f"material {c_s['material_match_pct']:.2f}%, correct placement {c_s['correct_placement_pct']:.2f}%, "
        f"repair edit {c_s['repair_edit_distance']:.3f}"
    )
    md.append(
        f"- Structured - Direct: IoU {c_s['iou_pct']-c_d['iou_pct']:+.2f} pt, "
        f"F1 {c_s['f1_pct']-c_d['f1_pct']:+.2f} pt, material {c_s['material_match_pct']-c_d['material_match_pct']:+.2f} pt, "
        f"correct placement {c_s['correct_placement_pct']-c_d['correct_placement_pct']:+.2f} pt"
    )
    md.append("")
    md.append("## OpenAI vs Claude（同一 shared-source）")
    md.append(
        f"- Direct: OpenAI-claude差 IoU {o_d['iou_pct']-c_d['iou_pct']:+.2f} pt, "
        f"F1 {o_d['f1_pct']-c_d['f1_pct']:+.2f} pt"
    )
    md.append(
        f"- Structured: OpenAI-claude差 IoU {o_s['iou_pct']-c_s['iou_pct']:+.2f} pt, "
        f"F1 {o_s['f1_pct']-c_s['f1_pct']:+.2f} pt"
    )
    md.append("")
    md.append("## 以前の ambiguous/synthetic-run との差")
    op = openai_prev_block["structured_rebuild_summary"]
    cp = claude_prev_block["structured_rebuild_summary"]
    md.append(
        f"- OpenAI structured IoU: {op['iou_pct']:.2f}% -> {o_s['iou_pct']:.2f}% "
        f"({o_s['iou_pct']-op['iou_pct']:+.2f} pt)"
    )
    md.append(
        f"- Claude structured IoU: {cp['iou_pct']:.2f}% -> {c_s['iou_pct']:.2f}% "
        f"({c_s['iou_pct']-cp['iou_pct']:+.2f} pt)"
    )
    md.append("")
    md.append("## 注意")
    md.append("- これは10件の診断実験で、統計的確定ではなく傾向確認です。")
    md.append("- 人手実験の結果は含みません（プロトコルのみ）。")

    out_md = REPORTS / "llm_authored_10_minecraft_summary.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[summarize_llm_authored_10_minecraft_run] wrote {out_json}")
    print(f"[summarize_llm_authored_10_minecraft_run] wrote {out_md}")


if __name__ == "__main__":
    main()
