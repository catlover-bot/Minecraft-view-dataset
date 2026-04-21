#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Condition:
    condition_id: str
    model: str
    regime: str
    dataset: str
    gt_root: Path
    pred_root: Path
    rebuild_metrics_path: Path
    repair_metrics_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run and summarize repair-effort diagnostics for original benchmark datasets (buildings_100_v1/v4)."
    )
    p.add_argument("--run_eval", action="store_true", help="Run evaluate_repair_effort.py for all configured conditions.")
    p.add_argument("--max_shift_xy", type=int, default=48)
    p.add_argument("--max_shift_y", type=int, default=8)
    p.add_argument("--top_shift_candidates", type=int, default=24)
    p.add_argument("--reports_dir", default="reports/final")
    p.add_argument("--csv_out", default="reports/final/original_benchmark_repair_effort_cases.csv")
    p.add_argument("--out_json", default="reports/final/original_benchmark_repair_effort_results.json")
    p.add_argument("--out_md", default="reports/final/original_benchmark_repair_effort_summary.md")
    p.add_argument("--out_main_json", default="reports/final/original_benchmark_repair_effort_main.json")
    p.add_argument(
        "--out_supplementary_json",
        default="reports/final/original_benchmark_repair_effort_supplementary.json",
    )
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_conditions() -> List[Condition]:
    base = ROOT / "outputs" / "i2t2b"
    rebuild_metrics = {
        ("openai", "main", "v1"): base
        / "buildings_100_v1"
        / "metrics"
        / "rebuild"
        / "schema_v5_repair_openai_self_refine_common_v8_struct_full.json",
        ("openai", "main", "v4"): base
        / "buildings_100_v4"
        / "metrics"
        / "rebuild"
        / "schema_v5_repair_openai_self_refine_common_v8_struct_full.json",
        ("claude", "main", "v1"): base
        / "buildings_100_v1"
        / "metrics"
        / "rebuild"
        / "schema_v5_repair_claude_self_refine_common_v8_struct_full.json",
        ("claude", "main", "v4"): base
        / "buildings_100_v4"
        / "metrics"
        / "rebuild"
        / "schema_v5_repair_claude_self_refine_common_v8_struct_full.json",
        ("openai", "supplementary", "v1"): base
        / "buildings_100_v1"
        / "metrics"
        / "rebuild"
        / "schema_v5_repair_openai_self_refine_tuned.json",
        ("openai", "supplementary", "v4"): base
        / "buildings_100_v4"
        / "metrics"
        / "rebuild"
        / "schema_v5_repair_openai_self_refine_tuned.json",
        ("claude", "supplementary", "v1"): base
        / "buildings_100_v1"
        / "metrics"
        / "rebuild"
        / "schema_v5_repair_claude_self_refine_tuned_conditional_precboost_full.json",
        ("claude", "supplementary", "v4"): base
        / "buildings_100_v4"
        / "metrics"
        / "rebuild"
        / "schema_v5_repair_claude_self_refine_tuned_conditional_precboost_full.json",
    }

    out: List[Condition] = []
    for (model, regime, dataset), rebuild_path in rebuild_metrics.items():
        ds_dir = f"buildings_100_{dataset}"
        pred_root = base / ds_dir
        gt_root = ROOT / "datasets" / ds_dir
        repair_path = pred_root / "metrics" / "repair" / f"{model}_{regime}.json"
        cid = f"{model}_{regime}_{dataset}"
        out.append(
            Condition(
                condition_id=cid,
                model=model,
                regime=regime,
                dataset=dataset,
                gt_root=gt_root,
                pred_root=pred_root,
                rebuild_metrics_path=rebuild_path,
                repair_metrics_path=repair_path,
            )
        )
    return out


def assert_original_benchmark_artifact(metrics_path: Path, expected_dataset: str) -> Tuple[str, int]:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"rebuild metrics not found: {metrics_path}")
    payload = load_json(metrics_path)
    items = payload.get("items", [])
    if not items:
        raise ValueError(f"no items in rebuild metrics: {metrics_path}")
    first = items[0]
    gt_vox = str(first.get("gt_voxels", ""))
    pred_subdir = str(first.get("pred_source_used", "")).strip()
    if "llm_authored_10" in gt_vox:
        raise ValueError(f"invalid source (llm_authored) detected in {metrics_path}: {gt_vox}")
    expect_token = f"/datasets/buildings_100_{expected_dataset}/"
    if expect_token not in gt_vox:
        raise ValueError(f"unexpected gt path in {metrics_path}: {gt_vox} (expected token {expect_token})")
    evaluated = int(payload.get("summary", {}).get("evaluated_buildings", 0))
    return pred_subdir, evaluated


def run_repair_eval(cond: Condition, pred_subdir: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "evaluate_repair_effort.py"),
        "--gt_root",
        str(cond.gt_root),
        "--pred_root",
        str(cond.pred_root),
        "--pred_subdir",
        pred_subdir,
        "--building_pattern",
        "building_*",
        "--max_shift_xy",
        str(args.max_shift_xy),
        "--max_shift_y",
        str(args.max_shift_y),
        "--top_shift_candidates",
        str(args.top_shift_candidates),
        "--out",
        str(cond.repair_metrics_path),
    ]
    cond.repair_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def _mean(xs: List[float]) -> float:
    return 0.0 if not xs else float(sum(xs) / len(xs))


def _weighted_merge(parts: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total_n = sum(n for n, _ in parts)
    if total_n <= 0:
        return {k: 0.0 for k in (parts[0][1].keys() if parts else [])}
    out: Dict[str, float] = {}
    keys = list(parts[0][1].keys()) if parts else []
    for k in keys:
        out[k] = sum(n * d[k] for n, d in parts) / total_n
    return out


def summarize(conditions: List[Condition], args: argparse.Namespace) -> Dict[str, Any]:
    reports_dir = Path(args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    case_rows: List[Dict[str, Any]] = []
    condition_rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    provenance: List[Dict[str, Any]] = []

    for cond in conditions:
        pred_subdir, rebuild_eval_count = assert_original_benchmark_artifact(cond.rebuild_metrics_path, cond.dataset)
        rebuild = load_json(cond.rebuild_metrics_path)
        repair = load_json(cond.repair_metrics_path)
        r_items = repair.get("items", [])
        m_items = rebuild.get("items", [])
        m_by_building = {str(x.get("building", "")): x for x in m_items}

        additions = float(repair["summary"]["mean_additions_over_gt"])
        deletions = float(repair["summary"]["mean_deletions_over_gt"])
        replacements = float(repair["summary"]["mean_replacements_over_gt"])
        edit = float(repair["summary"]["mean_edit_distance_over_gt"])
        evaluated = int(repair["summary"]["evaluated_buildings"])
        miss = list(repair["summary"].get("missing_predictions", []))

        replacement_share_items: List[float] = []
        for it in r_items:
            c = it.get("counts", {})
            total = int(c.get("total_edit_operations", 0))
            rep = int(c.get("replacements_needed", 0))
            if total > 0:
                replacement_share_items.append(rep / total)

        rebuild_agg = rebuild.get("aggregate", {}).get("metrics", {})
        iou = float(rebuild_agg.get("iou", 0.0))
        f1 = float(rebuild_agg.get("f1", 0.0))
        material = float(rebuild_agg.get("material_match", 0.0))
        coarse_material = float(rebuild_agg.get("coarse_material_match", 0.0))
        correct = float(rebuild_agg.get("correct_placement_rate", 0.0))

        row = {
            "condition_id": cond.condition_id,
            "model": cond.model,
            "regime": cond.regime,
            "dataset": cond.dataset,
            "evaluated_buildings": evaluated,
            "missing_predictions": miss,
            "pred_subdir": pred_subdir,
            "rebuild_metrics_path": str(cond.rebuild_metrics_path),
            "repair_metrics_path": str(cond.repair_metrics_path),
            "rebuild_iou": iou,
            "rebuild_f1": f1,
            "rebuild_material_match": material,
            "rebuild_coarse_material_match": coarse_material,
            "rebuild_correct_placement_rate": correct,
            "repair_edit_distance_over_gt": edit,
            "repair_additions_over_gt": additions,
            "repair_deletions_over_gt": deletions,
            "repair_replacements_over_gt": replacements,
            "repair_replacement_share_mean": _mean(replacement_share_items),
        }
        condition_rows.append(row)
        provenance.append(
            {
                "condition_id": cond.condition_id,
                "dataset": cond.dataset,
                "rebuild_metrics_path": str(cond.rebuild_metrics_path),
                "repair_metrics_path": str(cond.repair_metrics_path),
                "pred_subdir": pred_subdir,
                "verified_gt_path_example": m_items[0].get("gt_voxels") if m_items else None,
            }
        )

        for it in r_items:
            b = str(it.get("building", ""))
            m = m_by_building.get(b, {})
            counts = it.get("counts", {})
            norm = it.get("normalized", {})
            total = int(counts.get("total_edit_operations", 0))
            rep = int(counts.get("replacements_needed", 0))
            case_rows.append(
                {
                    "condition_id": cond.condition_id,
                    "model": cond.model,
                    "regime": cond.regime,
                    "dataset": cond.dataset,
                    "building": b,
                    "iou": float(m.get("metrics", {}).get("iou", 0.0)),
                    "f1": float(m.get("metrics", {}).get("f1", 0.0)),
                    "material_match": float(m.get("metrics", {}).get("material_match", 0.0)),
                    "coarse_material_match": float(m.get("metrics", {}).get("coarse_material_match", 0.0)),
                    "correct_placement_rate": float(m.get("metrics", {}).get("correct_placement_rate", 0.0)),
                    "edit_distance_over_gt": float(norm.get("edit_distance_over_gt", 0.0)),
                    "additions_over_gt": float(norm.get("additions_over_gt", 0.0)),
                    "deletions_over_gt": float(norm.get("deletions_over_gt", 0.0)),
                    "replacements_over_gt": float(norm.get("replacements_over_gt", 0.0)),
                    "total_edit_operations": total,
                    "replacement_share": (rep / total) if total > 0 else 0.0,
                }
            )

        gkey = (cond.model, cond.regime)
        pack = by_key.setdefault(gkey, {"v1": None, "v4": None})
        pack[cond.dataset] = row

    # all_200 combined for each (model, regime)
    combined_rows: List[Dict[str, Any]] = []
    for (model, regime), parts in by_key.items():
        if not parts.get("v1") or not parts.get("v4"):
            continue
        a = parts["v1"]
        b = parts["v4"]
        merged = _weighted_merge(
            [
                (
                    int(a["evaluated_buildings"]),
                    {
                        "rebuild_iou": a["rebuild_iou"],
                        "rebuild_f1": a["rebuild_f1"],
                        "rebuild_material_match": a["rebuild_material_match"],
                        "rebuild_coarse_material_match": a["rebuild_coarse_material_match"],
                        "rebuild_correct_placement_rate": a["rebuild_correct_placement_rate"],
                        "repair_edit_distance_over_gt": a["repair_edit_distance_over_gt"],
                        "repair_additions_over_gt": a["repair_additions_over_gt"],
                        "repair_deletions_over_gt": a["repair_deletions_over_gt"],
                        "repair_replacements_over_gt": a["repair_replacements_over_gt"],
                        "repair_replacement_share_mean": a["repair_replacement_share_mean"],
                    },
                ),
                (
                    int(b["evaluated_buildings"]),
                    {
                        "rebuild_iou": b["rebuild_iou"],
                        "rebuild_f1": b["rebuild_f1"],
                        "rebuild_material_match": b["rebuild_material_match"],
                        "rebuild_coarse_material_match": b["rebuild_coarse_material_match"],
                        "rebuild_correct_placement_rate": b["rebuild_correct_placement_rate"],
                        "repair_edit_distance_over_gt": b["repair_edit_distance_over_gt"],
                        "repair_additions_over_gt": b["repair_additions_over_gt"],
                        "repair_deletions_over_gt": b["repair_deletions_over_gt"],
                        "repair_replacements_over_gt": b["repair_replacements_over_gt"],
                        "repair_replacement_share_mean": b["repair_replacement_share_mean"],
                    },
                ),
            ]
        )
        combined_rows.append(
            {
                "condition_id": f"{model}_{regime}_all_200",
                "model": model,
                "regime": regime,
                "dataset": "all_200",
                "evaluated_buildings": int(a["evaluated_buildings"]) + int(b["evaluated_buildings"]),
                **merged,
            }
        )

    # near-miss analysis
    low_iou_near_miss = []
    for r in case_rows:
        if r["iou"] < 0.20 and r["edit_distance_over_gt"] <= 0.50:
            low_iou_near_miss.append(r)
    low_iou_near_miss = sorted(low_iou_near_miss, key=lambda x: x["edit_distance_over_gt"])[:50]

    # condition-level shares: replacement dominant vs structure dominant
    comp: Dict[str, Dict[str, int]] = {}
    for r in case_rows:
        cid = r["condition_id"]
        c = comp.setdefault(cid, {"replacement_dominant": 0, "add_del_dominant": 0, "n": 0})
        c["n"] += 1
        if r["replacement_share"] >= 0.5:
            c["replacement_dominant"] += 1
        else:
            c["add_del_dominant"] += 1

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": "Supplemental repair-effort diagnostics for ORIGINAL benchmark only (buildings_100_v1/v4). Existing rebuild metrics remain unchanged.",
        "coverage": {
            "models": sorted({r["model"] for r in condition_rows}),
            "regimes": sorted({r["regime"] for r in condition_rows}),
            "datasets": sorted({r["dataset"] for r in condition_rows}),
            "condition_count": len(condition_rows),
            "case_count": len(case_rows),
        },
        "artifact_provenance_checks": provenance,
        "condition_rows": condition_rows,
        "all_200_rows": combined_rows,
        "material_vs_structure_edit_pattern": comp,
        "near_miss_low_iou_but_low_edit_cases": low_iou_near_miss,
    }

    # split files
    main_rows = [r for r in condition_rows if r["regime"] == "main"] + [r for r in combined_rows if r["regime"] == "main"]
    supp_rows = [r for r in condition_rows if r["regime"] == "supplementary"] + [
        r for r in combined_rows if r["regime"] == "supplementary"
    ]
    main_out = {"created_at": out["created_at"], "rows": main_rows}
    supp_out = {"created_at": out["created_at"], "rows": supp_rows}

    # CSV
    csv_path = Path(args.csv_out).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition_id",
        "model",
        "regime",
        "dataset",
        "building",
        "iou",
        "f1",
        "material_match",
        "coarse_material_match",
        "correct_placement_rate",
        "edit_distance_over_gt",
        "additions_over_gt",
        "deletions_over_gt",
        "replacements_over_gt",
        "total_edit_operations",
        "replacement_share",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in case_rows:
            w.writerow(r)

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_main_json).resolve().write_text(json.dumps(main_out, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_supplementary_json).resolve().write_text(
        json.dumps(supp_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = []
    md.append("# Original Benchmark Repair-Effort Summary")
    md.append("")
    md.append("この結果は `datasets/buildings_100_v1` / `datasets/buildings_100_v4` に対する追加診断です。")
    md.append("Main/Supplementary の既存IoU/F1等は変更せず、repair-effort軸を追加しています。")
    md.append("")
    md.append("## Coverage")
    md.append(f"- models: {', '.join(out['coverage']['models'])}")
    md.append(f"- regimes: {', '.join(out['coverage']['regimes'])}")
    md.append(f"- datasets: {', '.join(out['coverage']['datasets'])}")
    md.append(f"- conditions: {out['coverage']['condition_count']}, cases: {out['coverage']['case_count']}")
    md.append("")
    md.append("## Main (shared hparams)")
    for r in sorted(main_rows, key=lambda x: (x["model"], x["dataset"])):
        md.append(
            f"- {r['model']} {r['dataset']}: IoU {100*r['rebuild_iou']:.2f}% / F1 {100*r['rebuild_f1']:.2f}% / "
            f"material {100*r['rebuild_material_match']:.2f}% / correct {100*r['rebuild_correct_placement_rate']:.2f}% / "
            f"edit {r['repair_edit_distance_over_gt']:.3f} "
            f"(add {r['repair_additions_over_gt']:.3f}, del {r['repair_deletions_over_gt']:.3f}, rep {r['repair_replacements_over_gt']:.3f})"
        )
    md.append("")
    md.append("## Supplementary (model-tuned)")
    for r in sorted(supp_rows, key=lambda x: (x["model"], x["dataset"])):
        md.append(
            f"- {r['model']} {r['dataset']}: IoU {100*r['rebuild_iou']:.2f}% / F1 {100*r['rebuild_f1']:.2f}% / "
            f"material {100*r['rebuild_material_match']:.2f}% / correct {100*r['rebuild_correct_placement_rate']:.2f}% / "
            f"edit {r['repair_edit_distance_over_gt']:.3f} "
            f"(add {r['repair_additions_over_gt']:.3f}, del {r['repair_deletions_over_gt']:.3f}, rep {r['repair_replacements_over_gt']:.3f})"
        )
    md.append("")
    md.append("## Near-Miss (low IoU but low edit)")
    md.append("- criterion: IoU < 0.20 and edit_distance_over_gt <= 0.50")
    md.append(f"- matched cases: {len(low_iou_near_miss)}")
    for r in low_iou_near_miss[:10]:
        md.append(
            f"- {r['condition_id']} {r['building']}: IoU {100*r['iou']:.2f}%, edit {r['edit_distance_over_gt']:.3f}, "
            f"add/del/rep=({r['additions_over_gt']:.3f}/{r['deletions_over_gt']:.3f}/{r['replacements_over_gt']:.3f})"
        )
    md.append("")
    md.append("## Interpretation guardrails")
    md.append("- repair-effortは IoU/F1 の置き換えではなく補助指標です。")
    md.append("- Main と Supplementary は混ぜずに解釈してください。")
    md.append("- これは original benchmark 追加診断で、llm_authored_10 とは分離しています。")

    Path(args.out_md).resolve().write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    conditions = build_conditions()

    audit_rows = []
    for cond in conditions:
        pred_subdir, evaluated = assert_original_benchmark_artifact(cond.rebuild_metrics_path, cond.dataset)
        audit_rows.append((cond.condition_id, pred_subdir, evaluated))
        if args.run_eval:
            run_repair_eval(cond, pred_subdir, args)

    out = summarize(conditions, args)
    print("[run_original_benchmark_repair_effort] conditions audited:")
    for cid, subdir, n in audit_rows:
        print(f"  - {cid}: pred_subdir={subdir}, evaluated={n}")
    print(f"[run_original_benchmark_repair_effort] wrote {args.out_json}")
    print(f"[run_original_benchmark_repair_effort] wrote {args.out_md}")
    print(f"[run_original_benchmark_repair_effort] wrote {args.csv_out}")


if __name__ == "__main__":
    main()
