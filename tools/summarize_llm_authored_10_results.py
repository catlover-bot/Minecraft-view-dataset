#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize llm_authored_10 diagnostic experiment outputs.")
    p.add_argument("--dataset_root", default="datasets/llm_authored_10")
    p.add_argument("--outputs_root", default="outputs/llm_authored_10")
    p.add_argument("--provider_tag", required=True, help="Tag used by run_llm_authored_diagnostic.py")
    p.add_argument("--description_subdir", default="description_direct")
    p.add_argument("--direct_plan_subdir", default="rebuild_plan_direct")
    p.add_argument("--direct_rebuild_subdir", default="rebuild_world_direct")
    p.add_argument("--structured_ir_subdir", default="structured_intermediate")
    p.add_argument("--structured_plan_subdir", default="rebuild_plan_structured")
    p.add_argument("--structured_rebuild_subdir", default="rebuild_world_structured")
    p.add_argument("--reports_dir", default="reports/final")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(v: float) -> float:
    return float(v) * 100.0


def _safe_mean(values: Iterable[float]) -> float:
    xs = [float(v) for v in values]
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _difficulty_map(dataset_root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for bdir in sorted([p for p in dataset_root.glob("llm_case_*") if p.is_dir()]):
        spec = bdir / "source_spec.json"
        if not spec.is_file():
            continue
        payload = _read_json(spec)
        d = str(payload.get("difficulty", "unknown")).strip().lower() or "unknown"
        out[bdir.name] = d
    return out


def _load_metrics_pair(outputs_root: Path, provider_tag: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    base = outputs_root / "metrics"
    desc = _read_json(base / f"description_{provider_tag}.json")
    direct = _read_json(base / f"rebuild_direct_{provider_tag}.json")
    structured = _read_json(base / f"rebuild_structured_{provider_tag}.json")
    return desc, direct, structured


def _load_repair_pair(outputs_root: Path, provider_tag: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = outputs_root / "metrics"
    r1 = _read_json(base / f"repair_direct_{provider_tag}.json")
    r2 = _read_json(base / f"repair_structured_{provider_tag}.json")
    return r1, r2


def _index_by_building(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        b = str(it.get("building", "")).strip()
        if not b:
            continue
        out[b] = it
    return out


def _difficulty_breakdown(
    items_direct: List[Dict[str, Any]],
    items_structured: List[Dict[str, Any]],
    repair_direct_items: List[Dict[str, Any]],
    repair_structured_items: List[Dict[str, Any]],
    difficulty_by_case: Dict[str, str],
) -> Dict[str, Any]:
    direct_map = _index_by_building(items_direct)
    struct_map = _index_by_building(items_structured)
    repair_direct_map = _index_by_building(repair_direct_items)
    repair_struct_map = _index_by_building(repair_structured_items)

    buckets: Dict[str, List[str]] = {"simple": [], "medium": [], "complex": [], "unknown": []}
    for b, d in difficulty_by_case.items():
        buckets[d if d in buckets else "unknown"].append(b)

    out: Dict[str, Any] = {}
    for d, cases in buckets.items():
        if not cases:
            continue
        direct_rows = [direct_map[c] for c in cases if c in direct_map]
        struct_rows = [struct_map[c] for c in cases if c in struct_map]
        rd_rows = [repair_direct_map[c] for c in cases if c in repair_direct_map]
        rs_rows = [repair_struct_map[c] for c in cases if c in repair_struct_map]

        out[d] = {
            "n_cases": len(cases),
            "direct": {
                "iou": _pct(_safe_mean(r["metrics"]["iou"] for r in direct_rows)),
                "f1": _pct(_safe_mean(r["metrics"]["f1"] for r in direct_rows)),
                "material_match": _pct(_safe_mean(r["metrics"]["material_match"] for r in direct_rows)),
                "correct_placement_rate": _pct(_safe_mean(r["metrics"]["correct_placement_rate"] for r in direct_rows)),
                "normalized_edit_distance": _safe_mean(r["normalized"]["edit_distance_over_gt"] for r in rd_rows),
            },
            "structured": {
                "iou": _pct(_safe_mean(r["metrics"]["iou"] for r in struct_rows)),
                "f1": _pct(_safe_mean(r["metrics"]["f1"] for r in struct_rows)),
                "material_match": _pct(_safe_mean(r["metrics"]["material_match"] for r in struct_rows)),
                "correct_placement_rate": _pct(_safe_mean(r["metrics"]["correct_placement_rate"] for r in struct_rows)),
                "normalized_edit_distance": _safe_mean(r["normalized"]["edit_distance_over_gt"] for r in rs_rows),
            },
        }
    return out


def _write_case_markdowns(
    dataset_root: Path,
    outputs_root: Path,
    provider_tag: str,
    desc_map: Dict[str, Dict[str, Any]],
    direct_map: Dict[str, Dict[str, Any]],
    struct_map: Dict[str, Dict[str, Any]],
    repair_direct_map: Dict[str, Dict[str, Any]],
    repair_struct_map: Dict[str, Dict[str, Any]],
    difficulty_by_case: Dict[str, str],
) -> None:
    out_dir = outputs_root / "case_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    for bdir in sorted([p for p in dataset_root.glob("llm_case_*") if p.is_dir()]):
        case_id = bdir.name
        desc_item = desc_map.get(case_id, {})
        d_item = direct_map.get(case_id, {})
        s_item = struct_map.get(case_id, {})
        rd_item = repair_direct_map.get(case_id, {})
        rs_item = repair_struct_map.get(case_id, {})

        lines = [
            f"# {case_id} ({difficulty_by_case.get(case_id, 'unknown')})",
            "",
            f"- provider_tag: `{provider_tag}`",
            f"- source_spec: `{case_id}/source_spec.json`",
            f"- description: `{case_id}/description_direct/description.json`",
            f"- structured_ir: `{case_id}/structured_intermediate/intermediate.json`",
            "",
            "## Description",
            f"- auto_score: {100.0 * float(desc_item.get('auto_score', 0.0)):.2f}%",
            f"- strict_material_f1: {100.0 * float(desc_item.get('strict_material_metrics', {}).get('f1', 0.0)):.2f}%",
            f"- coarse_material_f1: {100.0 * float(desc_item.get('coarse_material_metrics', {}).get('f1', 0.0)):.2f}%",
            f"- dimension_score: {100.0 * float(desc_item.get('dimension_metrics', {}).get('dim_score', 0.0)):.2f}%",
            "",
            "## Rebuild Comparison",
            f"- direct IoU/F1/material/correct: "
            f"{100.0 * float(d_item.get('metrics', {}).get('iou', 0.0)):.2f}% / "
            f"{100.0 * float(d_item.get('metrics', {}).get('f1', 0.0)):.2f}% / "
            f"{100.0 * float(d_item.get('metrics', {}).get('material_match', 0.0)):.2f}% / "
            f"{100.0 * float(d_item.get('metrics', {}).get('correct_placement_rate', 0.0)):.2f}%",
            f"- structured IoU/F1/material/correct: "
            f"{100.0 * float(s_item.get('metrics', {}).get('iou', 0.0)):.2f}% / "
            f"{100.0 * float(s_item.get('metrics', {}).get('f1', 0.0)):.2f}% / "
            f"{100.0 * float(s_item.get('metrics', {}).get('material_match', 0.0)):.2f}% / "
            f"{100.0 * float(s_item.get('metrics', {}).get('correct_placement_rate', 0.0)):.2f}%",
            "",
            "## Repair Effort",
            f"- direct normalized_edit_distance: {float(rd_item.get('normalized', {}).get('edit_distance_over_gt', 0.0)):.4f}",
            f"- structured normalized_edit_distance: {float(rs_item.get('normalized', {}).get('edit_distance_over_gt', 0.0)):.4f}",
            f"- direct edits(add/del/rep): "
            f"{int(d_item.get('metrics', {}).get('intersection_count', 0.0))} overlap ref, "
            f"{int(rd_item.get('counts', {}).get('additions_needed', 0))}/"
            f"{int(rd_item.get('counts', {}).get('deletions_needed', 0))}/"
            f"{int(rd_item.get('counts', {}).get('replacements_needed', 0))}",
            f"- structured edits(add/del/rep): "
            f"{int(s_item.get('metrics', {}).get('intersection_count', 0.0))} overlap ref, "
            f"{int(rs_item.get('counts', {}).get('additions_needed', 0))}/"
            f"{int(rs_item.get('counts', {}).get('deletions_needed', 0))}/"
            f"{int(rs_item.get('counts', {}).get('replacements_needed', 0))}",
            "",
            "Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.",
            "",
        ]
        (out_dir / f"{case_id}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    desc, direct, structured = _load_metrics_pair(outputs_root, args.provider_tag)
    repair_direct, repair_structured = _load_repair_pair(outputs_root, args.provider_tag)
    difficulty_by_case = _difficulty_map(dataset_root)

    desc_items = desc.get("items", []) if isinstance(desc.get("items"), list) else []
    direct_items = direct.get("items", []) if isinstance(direct.get("items"), list) else []
    structured_items = structured.get("items", []) if isinstance(structured.get("items"), list) else []
    repair_direct_items = repair_direct.get("items", []) if isinstance(repair_direct.get("items"), list) else []
    repair_struct_items = repair_structured.get("items", []) if isinstance(repair_structured.get("items"), list) else []

    desc_map = _index_by_building(desc_items)
    direct_map = _index_by_building(direct_items)
    structured_map = _index_by_building(structured_items)
    repair_direct_map = _index_by_building(repair_direct_items)
    repair_struct_map = _index_by_building(repair_struct_items)

    _write_case_markdowns(
        dataset_root,
        outputs_root,
        args.provider_tag,
        desc_map,
        direct_map,
        structured_map,
        repair_direct_map,
        repair_struct_map,
        difficulty_by_case,
    )

    table_rows: List[Dict[str, Any]] = []
    for case_id in sorted(difficulty_by_case.keys()):
        d = direct_map.get(case_id, {})
        s = structured_map.get(case_id, {})
        rd = repair_direct_map.get(case_id, {})
        rs = repair_struct_map.get(case_id, {})
        table_rows.append(
            {
                "case_id": case_id,
                "difficulty": difficulty_by_case.get(case_id, "unknown"),
                "direct_iou_pct": _pct(float(d.get("metrics", {}).get("iou", 0.0))),
                "structured_iou_pct": _pct(float(s.get("metrics", {}).get("iou", 0.0))),
                "direct_f1_pct": _pct(float(d.get("metrics", {}).get("f1", 0.0))),
                "structured_f1_pct": _pct(float(s.get("metrics", {}).get("f1", 0.0))),
                "direct_material_pct": _pct(float(d.get("metrics", {}).get("material_match", 0.0))),
                "structured_material_pct": _pct(float(s.get("metrics", {}).get("material_match", 0.0))),
                "direct_correct_placement_pct": _pct(float(d.get("metrics", {}).get("correct_placement_rate", 0.0))),
                "structured_correct_placement_pct": _pct(float(s.get("metrics", {}).get("correct_placement_rate", 0.0))),
                "direct_edit_distance": float(rd.get("normalized", {}).get("edit_distance_over_gt", 0.0)),
                "structured_edit_distance": float(rs.get("normalized", {}).get("edit_distance_over_gt", 0.0)),
            }
        )

    difficulty_breakdown = _difficulty_breakdown(
        direct_items,
        structured_items,
        repair_direct_items,
        repair_struct_items,
        difficulty_by_case,
    )

    summary_json = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "llm_authored_10_diagnostic",
        "provider_tag": args.provider_tag,
        "separation_note": "Diagnostic set only. Do not mix with main/supplementary/execution-gap benchmark claims.",
        "source_dataset_root": str(dataset_root),
        "outputs_root": str(outputs_root),
        "description_summary": {
            "evaluated": int(desc.get("summary", {}).get("evaluated_buildings", 0)),
            "auto_score_pct": _pct(float(desc.get("aggregate", {}).get("auto_score_mean", 0.0))),
            "strict_material_f1_pct": _pct(float(desc.get("aggregate", {}).get("strict_material_f1_mean", 0.0))),
            "coarse_material_f1_pct": _pct(float(desc.get("aggregate", {}).get("coarse_material_f1_mean", 0.0))),
            "dimension_score_pct": _pct(float(desc.get("aggregate", {}).get("dimension_score_mean", 0.0))),
        },
        "direct_rebuild_summary": {
            "evaluated": int(direct.get("summary", {}).get("evaluated_buildings", 0)),
            "iou_pct": _pct(float(direct.get("aggregate", {}).get("metrics", {}).get("iou", 0.0))),
            "f1_pct": _pct(float(direct.get("aggregate", {}).get("metrics", {}).get("f1", 0.0))),
            "material_match_pct": _pct(float(direct.get("aggregate", {}).get("metrics", {}).get("material_match", 0.0))),
            "correct_placement_pct": _pct(float(direct.get("aggregate", {}).get("metrics", {}).get("correct_placement_rate", 0.0))),
            "repair_edit_distance": float(repair_direct.get("summary", {}).get("mean_edit_distance_over_gt", 0.0)),
        },
        "structured_rebuild_summary": {
            "evaluated": int(structured.get("summary", {}).get("evaluated_buildings", 0)),
            "iou_pct": _pct(float(structured.get("aggregate", {}).get("metrics", {}).get("iou", 0.0))),
            "f1_pct": _pct(float(structured.get("aggregate", {}).get("metrics", {}).get("f1", 0.0))),
            "material_match_pct": _pct(float(structured.get("aggregate", {}).get("metrics", {}).get("material_match", 0.0))),
            "correct_placement_pct": _pct(float(structured.get("aggregate", {}).get("metrics", {}).get("correct_placement_rate", 0.0))),
            "repair_edit_distance": float(repair_structured.get("summary", {}).get("mean_edit_distance_over_gt", 0.0)),
        },
        "comparison_delta_structured_minus_direct": {
            "iou_pct_point": _pct(float(structured.get("aggregate", {}).get("metrics", {}).get("iou", 0.0)) - float(direct.get("aggregate", {}).get("metrics", {}).get("iou", 0.0)),
            ),
            "f1_pct_point": _pct(float(structured.get("aggregate", {}).get("metrics", {}).get("f1", 0.0)) - float(direct.get("aggregate", {}).get("metrics", {}).get("f1", 0.0))),
            "material_match_pct_point": _pct(float(structured.get("aggregate", {}).get("metrics", {}).get("material_match", 0.0)) - float(direct.get("aggregate", {}).get("metrics", {}).get("material_match", 0.0))),
            "correct_placement_pct_point": _pct(float(structured.get("aggregate", {}).get("metrics", {}).get("correct_placement_rate", 0.0)) - float(direct.get("aggregate", {}).get("metrics", {}).get("correct_placement_rate", 0.0))),
            "repair_edit_distance_delta": float(repair_structured.get("summary", {}).get("mean_edit_distance_over_gt", 0.0)) - float(repair_direct.get("summary", {}).get("mean_edit_distance_over_gt", 0.0)),
        },
        "difficulty_breakdown": difficulty_breakdown,
        "case_table": table_rows,
    }

    md_lines = [
        "# LLM-authored 10-case Diagnostic Summary",
        "",
        "- This is a diagnostic validation set. It is separate from main/supplementary/execution-gap benchmark reports.",
        f"- provider_tag: `{args.provider_tag}`",
        "",
        "## Description",
        f"- auto_score: {summary_json['description_summary']['auto_score_pct']:.2f}%",
        f"- strict_material_f1: {summary_json['description_summary']['strict_material_f1_pct']:.2f}%",
        f"- coarse_material_f1: {summary_json['description_summary']['coarse_material_f1_pct']:.2f}%",
        f"- dimension_score: {summary_json['description_summary']['dimension_score_pct']:.2f}%",
        "",
        "## Rebuild",
        "- Direct (description -> plan -> render):",
        f"  IoU={summary_json['direct_rebuild_summary']['iou_pct']:.2f}%, "
        f"F1={summary_json['direct_rebuild_summary']['f1_pct']:.2f}%, "
        f"material={summary_json['direct_rebuild_summary']['material_match_pct']:.2f}%, "
        f"correct={summary_json['direct_rebuild_summary']['correct_placement_pct']:.2f}%, "
        f"repair_edit={summary_json['direct_rebuild_summary']['repair_edit_distance']:.4f}",
        "- Structured (description -> structured IR -> deterministic plan -> render):",
        f"  IoU={summary_json['structured_rebuild_summary']['iou_pct']:.2f}%, "
        f"F1={summary_json['structured_rebuild_summary']['f1_pct']:.2f}%, "
        f"material={summary_json['structured_rebuild_summary']['material_match_pct']:.2f}%, "
        f"correct={summary_json['structured_rebuild_summary']['correct_placement_pct']:.2f}%, "
        f"repair_edit={summary_json['structured_rebuild_summary']['repair_edit_distance']:.4f}",
        "",
        "## Direct vs Structured delta (structured - direct)",
        f"- IoU: {summary_json['comparison_delta_structured_minus_direct']['iou_pct_point']:+.2f} pt",
        f"- F1: {summary_json['comparison_delta_structured_minus_direct']['f1_pct_point']:+.2f} pt",
        f"- material_match: {summary_json['comparison_delta_structured_minus_direct']['material_match_pct_point']:+.2f} pt",
        f"- correct_placement: {summary_json['comparison_delta_structured_minus_direct']['correct_placement_pct_point']:+.2f} pt",
        f"- repair_edit_distance: {summary_json['comparison_delta_structured_minus_direct']['repair_edit_distance_delta']:+.4f}",
        "",
        "## Notes",
        "- Human kit is protocol-only; no human outcomes are claimed.",
        "- Interpretation should remain diagnostic; do not use these numbers as headline benchmark replacement.",
        "",
    ]

    out_json = reports_dir / f"llm_authored_10_results_{args.provider_tag}.json"
    out_md = reports_dir / f"llm_authored_10_summary_{args.provider_tag}.md"
    out_human = reports_dir / "llm_authored_10_human_protocol.md"
    out_json.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    human_kit_protocol = Path(args.outputs_root).resolve() / "human_kit" / "protocol.md"
    if human_kit_protocol.is_file():
        out_human.write_text(human_kit_protocol.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[summarize_llm_authored_10_results] wrote {out_json}")
    print(f"[summarize_llm_authored_10_results] wrote {out_md}")
    if out_human.is_file():
        print(f"[summarize_llm_authored_10_results] wrote {out_human}")


if __name__ == "__main__":
    main()
