#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create shared-source cross-provider summary for llm_authored_10.")
    p.add_argument("--dataset_root", default="datasets/llm_authored_10")
    p.add_argument("--reports_dir", default="reports/final")
    p.add_argument("--openai_tag", default="openai_gpt_5_mini")
    p.add_argument("--claude_tag", default="anthropic_claude_haiku_4_5_20251001")
    p.add_argument("--out_summary", default="llm_authored_10_summary.md")
    p.add_argument("--out_json", default="llm_authored_10_results.json")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(a: float, b: float) -> float:
    return float(a) - float(b)


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _extract_core(d: Dict[str, Any]) -> Dict[str, float]:
    desc = d["description_summary"]
    direct = d["direct_rebuild_summary"]
    structured = d["structured_rebuild_summary"]
    return {
        "desc_auto": float(desc["auto_score_pct"]),
        "desc_strict_mat_f1": float(desc["strict_material_f1_pct"]),
        "desc_coarse_mat_f1": float(desc["coarse_material_f1_pct"]),
        "desc_dim": float(desc["dimension_score_pct"]),
        "direct_iou": float(direct["iou_pct"]),
        "direct_f1": float(direct["f1_pct"]),
        "direct_mat": float(direct["material_match_pct"]),
        "direct_correct": float(direct["correct_placement_pct"]),
        "direct_repair": float(direct["repair_edit_distance"]),
        "structured_iou": float(structured["iou_pct"]),
        "structured_f1": float(structured["f1_pct"]),
        "structured_mat": float(structured["material_match_pct"]),
        "structured_correct": float(structured["correct_placement_pct"]),
        "structured_repair": float(structured["repair_edit_distance"]),
    }


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_manifest = _read_json(dataset_root / "dataset_manifest.json")

    openai_path = reports_dir / f"llm_authored_10_results_{args.openai_tag}.json"
    claude_path = reports_dir / f"llm_authored_10_results_{args.claude_tag}.json"
    openai = _read_json(openai_path)
    claude = _read_json(claude_path)
    o = _extract_core(openai)
    c = _extract_core(claude)

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "llm_authored_10_diagnostic_shared_source",
        "separation_note": "Diagnostic-only set. Keep separate from main/supplementary/execution-gap benchmark claims.",
        "dataset_manifest": {
            "source_author": dataset_manifest.get("source_author", {}),
            "difficulty_count": dataset_manifest.get("difficulty_count", {}),
        },
        "providers": {
            args.openai_tag: openai,
            args.claude_tag: claude,
        },
        "cross_provider_shared_source": {
            "direct_openai_minus_claude": {
                "iou_pct_point": _delta(o["direct_iou"], c["direct_iou"]),
                "f1_pct_point": _delta(o["direct_f1"], c["direct_f1"]),
                "material_pct_point": _delta(o["direct_mat"], c["direct_mat"]),
                "correct_placement_pct_point": _delta(o["direct_correct"], c["direct_correct"]),
                "repair_edit_distance_delta": _delta(o["direct_repair"], c["direct_repair"]),
            },
            "structured_openai_minus_claude": {
                "iou_pct_point": _delta(o["structured_iou"], c["structured_iou"]),
                "f1_pct_point": _delta(o["structured_f1"], c["structured_f1"]),
                "material_pct_point": _delta(o["structured_mat"], c["structured_mat"]),
                "correct_placement_pct_point": _delta(o["structured_correct"], c["structured_correct"]),
                "repair_edit_distance_delta": _delta(o["structured_repair"], c["structured_repair"]),
            },
        },
        "difficulty_breakdown": {
            args.openai_tag: openai.get("difficulty_breakdown", {}),
            args.claude_tag: claude.get("difficulty_breakdown", {}),
        },
        "caution": {
            "sample_size": 10,
            "note": "Interpret as suggestive diagnostic evidence, not definitive benchmark ranking.",
        },
    }

    lines = [
        "# LLM-authored 10-case Diagnostic (Shared-source) Summary",
        "",
        "- この結果は診断用セット（10件）で、Main/Supplementaryベンチとは別です。",
        f"- source author: `{dataset_manifest.get('source_author', {}).get('provider','?')}/{dataset_manifest.get('source_author', {}).get('model','?')}`",
        "",
        "## 1) OpenAI: direct vs structured",
        f"- direct IoU/F1/material/correct: {_fmt(o['direct_iou'])}% / {_fmt(o['direct_f1'])}% / {_fmt(o['direct_mat'])}% / {_fmt(o['direct_correct'])}%",
        f"- structured IoU/F1/material/correct: {_fmt(o['structured_iou'])}% / {_fmt(o['structured_f1'])}% / {_fmt(o['structured_mat'])}% / {_fmt(o['structured_correct'])}%",
        f"- delta(structured-direct): IoU {_fmt(o['structured_iou']-o['direct_iou'])}pt, F1 {_fmt(o['structured_f1']-o['direct_f1'])}pt, material {_fmt(o['structured_mat']-o['direct_mat'])}pt, correct {_fmt(o['structured_correct']-o['direct_correct'])}pt",
        "",
        "## 2) Claude: direct vs structured",
        f"- direct IoU/F1/material/correct: {_fmt(c['direct_iou'])}% / {_fmt(c['direct_f1'])}% / {_fmt(c['direct_mat'])}% / {_fmt(c['direct_correct'])}%",
        f"- structured IoU/F1/material/correct: {_fmt(c['structured_iou'])}% / {_fmt(c['structured_f1'])}% / {_fmt(c['structured_mat'])}% / {_fmt(c['structured_correct'])}%",
        f"- delta(structured-direct): IoU {_fmt(c['structured_iou']-c['direct_iou'])}pt, F1 {_fmt(c['structured_f1']-c['direct_f1'])}pt, material {_fmt(c['structured_mat']-c['direct_mat'])}pt, correct {_fmt(c['structured_correct']-c['direct_correct'])}pt",
        "",
        "## 3) Cross-provider (same shared-source 10 cases)",
        f"- direct OpenAI-Claude: IoU {_fmt(o['direct_iou']-c['direct_iou'])}pt, F1 {_fmt(o['direct_f1']-c['direct_f1'])}pt, material {_fmt(o['direct_mat']-c['direct_mat'])}pt, correct {_fmt(o['direct_correct']-c['direct_correct'])}pt",
        f"- structured OpenAI-Claude: IoU {_fmt(o['structured_iou']-c['structured_iou'])}pt, F1 {_fmt(o['structured_f1']-c['structured_f1'])}pt, material {_fmt(o['structured_mat']-c['structured_mat'])}pt, correct {_fmt(o['structured_correct']-c['structured_correct'])}pt",
        "",
        "## 4) 説明品質",
        f"- OpenAI auto/strict/coarse/dim: {_fmt(o['desc_auto'])}% / {_fmt(o['desc_strict_mat_f1'])}% / {_fmt(o['desc_coarse_mat_f1'])}% / {_fmt(o['desc_dim'])}%",
        f"- Claude auto/strict/coarse/dim: {_fmt(c['desc_auto'])}% / {_fmt(c['desc_strict_mat_f1'])}% / {_fmt(c['desc_coarse_mat_f1'])}% / {_fmt(c['desc_dim'])}%",
        "",
        "## 5) 注意点",
        "- n=10 の診断セットなので、傾向は示唆的です（確定的主張は不可）。",
        "- human study はプロトコルのみで、結果主張はしていません。",
        "",
    ]

    out_json_path = reports_dir / args.out_json
    out_md_path = reports_dir / args.out_summary
    out_json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize_llm_authored_10_shared_source] wrote {out_json_path}")
    print(f"[summarize_llm_authored_10_shared_source] wrote {out_md_path}")


if __name__ == "__main__":
    main()
