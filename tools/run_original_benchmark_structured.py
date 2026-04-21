#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.llm_config import load_llm_config


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DirectCondition:
    condition_id: str
    model: str
    regime: str
    dataset: str
    rebuild_metrics_path: Path


@dataclass(frozen=True)
class StructuredCondition:
    condition_id: str
    model: str
    regime: str
    dataset: str
    gt_root: Path
    out_root: Path
    description_subdir: str
    intermediate_subdir: str
    plan_subdir: str
    rebuild_subdir: str
    rebuild_metrics_path: Path
    repair_metrics_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run original benchmark structured-intermediate condition and compare against existing direct metrics."
        )
    )
    p.add_argument("--run_structured", action="store_true", help="Run structured pipeline (build IR -> plan -> render -> eval).")
    p.add_argument("--run_repair_eval", action="store_true", help="Force re-run repair evaluation for structured outputs.")
    p.add_argument("--include_supplementary", action="store_true", help="Include direct supplementary rows in comparison tables.")
    p.add_argument("--include_gemini_main", action="store_true", help="Include Gemini main direct/structured rows if artifacts exist.")
    p.add_argument("--gemini_model_tag", default="", help="Optional gemini model tag (e.g., gemini_gemini_3_1_pro_preview).")
    p.add_argument(
        "--structured_models",
        default="openai,claude",
        help="Comma-separated models to run for structured generation (default: openai,claude).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite intermediate/plan/rebuild artifacts.")
    p.add_argument("--building_pattern", default="building_*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_shift_xy", type=int, default=48)
    p.add_argument("--max_shift_y", type=int, default=8)
    p.add_argument("--top_shift_candidates", type=int, default=24)
    p.add_argument("--out_json", default="reports/final/original_benchmark_structured_results.json")
    p.add_argument("--out_md", default="reports/final/original_benchmark_structured_summary.md")
    p.add_argument("--out_repair_json", default="reports/final/original_benchmark_structured_repair_results.json")
    p.add_argument("--out_cases_csv", default="reports/final/original_benchmark_structured_vs_direct_cases.csv")
    p.add_argument("--out_main_json", default="reports/final/original_benchmark_structured_main.json")
    p.add_argument("--out_supp_json", default="reports/final/original_benchmark_structured_supplementary.json")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")


def _mean(xs: List[float]) -> float:
    return 0.0 if not xs else float(sum(xs) / len(xs))


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def resolve_gemini_model_tag(explicit: str) -> str:
    tag = (explicit or "").strip()
    if tag:
        return tag if tag.startswith("gemini_") else f"gemini_{_slugify(tag)}"
    cfg = load_llm_config(None)
    return f"gemini_{_slugify(cfg.gemini_model or 'gemini_model')}"


def parse_model_list(csv_text: str) -> List[str]:
    vals = []
    for x in (csv_text or "").split(","):
        y = x.strip().lower()
        if y:
            vals.append(y)
    return vals


def _weighted_merge(parts: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    if not parts:
        return {}
    total_n = sum(n for n, _ in parts)
    if total_n <= 0:
        return {k: 0.0 for k in parts[0][1].keys()}
    keys = list(parts[0][1].keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = sum(n * d.get(k, 0.0) for n, d in parts) / total_n
    return out


def detect_description_subdir(out_root: Path, model: str) -> str:
    sample = out_root / "building_000"
    if not sample.is_dir():
        raise FileNotFoundError(f"missing sample building dir: {sample}")
    if model == "openai":
        candidates = sorted([p.name for p in sample.glob("description_openai_*") if p.is_dir()])
    elif model == "gemini":
        candidates = sorted([p.name for p in sample.glob("description_gemini_*") if p.is_dir()])
    else:
        candidates = sorted([p.name for p in sample.glob("description_anthropic_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"description subdir not found for model={model} under {sample}")
    return candidates[0]


def build_direct_conditions(include_gemini_main: bool, gemini_model_tag: str) -> List[DirectCondition]:
    base = ROOT / "outputs" / "i2t2b"
    mapping = {
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
    if include_gemini_main:
        mapping[("gemini", "main", "v1")] = (
            base
            / "buildings_100_v1"
            / "metrics"
            / "rebuild"
            / f"schema_v5_repair_{gemini_model_tag}_self_refine_common_v8_struct_full.json"
        )
        mapping[("gemini", "main", "v4")] = (
            base
            / "buildings_100_v4"
            / "metrics"
            / "rebuild"
            / f"schema_v5_repair_{gemini_model_tag}_self_refine_common_v8_struct_full.json"
        )
    out: List[DirectCondition] = []
    for (model, regime, dataset), mpath in mapping.items():
        out.append(
            DirectCondition(
                condition_id=f"{model}_{regime}_{dataset}",
                model=model,
                regime=regime,
                dataset=dataset,
                rebuild_metrics_path=mpath,
            )
        )
    return out


def build_structured_conditions(
    models: List[str],
    include_gemini_main: bool,
    gemini_model_tag: str,
) -> List[StructuredCondition]:
    out: List[StructuredCondition] = []
    model_set = set(models)
    if include_gemini_main:
        model_set.add("gemini")
    for dataset in ("v1", "v4"):
        ds = f"buildings_100_{dataset}"
        out_root = ROOT / "outputs" / "i2t2b" / ds
        gt_root = ROOT / "datasets" / ds
        for model in ("openai", "claude", "gemini"):
            if model not in model_set:
                continue
            if model == "gemini":
                pinned = f"description_{gemini_model_tag}"
                if (out_root / "building_000" / pinned).is_dir():
                    desc_subdir = pinned
                else:
                    desc_subdir = detect_description_subdir(out_root, model)
            else:
                desc_subdir = detect_description_subdir(out_root, model)
            regime = "main"
            model_stem = model
            if model == "gemini":
                model_stem = gemini_model_tag
            stem = f"structured_ir_{model_stem}_{regime}_orig_20260418"
            out.append(
                StructuredCondition(
                    condition_id=f"{model}_{regime}_{dataset}",
                    model=model,
                    regime=regime,
                    dataset=dataset,
                    gt_root=gt_root,
                    out_root=out_root,
                    description_subdir=desc_subdir,
                    intermediate_subdir=f"structured_intermediate_{stem}",
                    plan_subdir=f"rebuild_plan_{stem}",
                    rebuild_subdir=f"rebuild_world_{stem}",
                    rebuild_metrics_path=out_root / "metrics" / "rebuild" / f"{stem}.json",
                    repair_metrics_path=out_root / "metrics" / "repair" / f"{stem}.json",
                )
            )
    return out


def assert_original_benchmark_metrics(metrics_path: Path, dataset: str) -> Tuple[int, int]:
    _require_file(metrics_path, "metrics")
    payload = _load_json(metrics_path)
    items = payload.get("items", [])
    if not items:
        return 0, int(payload.get("summary", {}).get("missing_predictions", 0) or 0)
    first_gt = str(items[0].get("gt_voxels", ""))
    if "llm_authored_10" in first_gt:
        raise ValueError(f"llm_authored artifact mixed in: {metrics_path}")
    token = f"/datasets/buildings_100_{dataset}/"
    if token not in first_gt:
        raise ValueError(f"unexpected gt path token in {metrics_path}: {first_gt}")
    summary = payload.get("summary", {})
    return int(summary.get("evaluated_buildings", 0)), len(summary.get("missing_predictions", []))


def run_structured_pipeline(cond: StructuredCondition, args: argparse.Namespace) -> None:
    py = sys.executable
    common = [
        "--dataset_root",
        str(cond.out_root),
        "--building_pattern",
        args.building_pattern,
    ]
    if args.limit > 0:
        common += ["--limit", str(args.limit)]
    if args.overwrite:
        common += ["--overwrite"]

    _run(
        [
            py,
            str(ROOT / "tools" / "build_structured_intermediate.py"),
            *common,
            "--description_subdir",
            cond.description_subdir,
            "--out_subdir",
            cond.intermediate_subdir,
        ]
    )
    _run(
        [
            py,
            str(ROOT / "tools" / "generate_plan_from_intermediate.py"),
            *common,
            "--intermediate_subdir",
            cond.intermediate_subdir,
            "--out_subdir",
            cond.plan_subdir,
        ]
    )
    _run(
        [
            py,
            str(ROOT / "tools" / "render_rebuild_from_plan.py"),
            *common,
            "--plan_subdir",
            cond.plan_subdir,
            "--out_subdir",
            cond.rebuild_subdir,
        ]
    )
    _run(
        [
            py,
            str(ROOT / "tools" / "evaluate_rebuild_metrics.py"),
            "--gt_root",
            str(cond.gt_root),
            "--pred_root",
            str(cond.out_root),
            "--pred_subdir",
            cond.rebuild_subdir,
            "--pred_source",
            "rebuild_world",
            "--building_pattern",
            args.building_pattern,
            "--out",
            str(cond.rebuild_metrics_path),
            "--fail_on_missing_pred",
        ]
        + (["--limit", str(args.limit)] if args.limit > 0 else [])
    )

    if args.run_repair_eval or not cond.repair_metrics_path.is_file():
        _run(
            [
                py,
                str(ROOT / "tools" / "evaluate_repair_effort.py"),
                "--gt_root",
                str(cond.gt_root),
                "--pred_root",
                str(cond.out_root),
                "--pred_subdir",
                cond.rebuild_subdir,
                "--building_pattern",
                args.building_pattern,
                "--max_shift_xy",
                str(args.max_shift_xy),
                "--max_shift_y",
                str(args.max_shift_y),
                "--top_shift_candidates",
                str(args.top_shift_candidates),
                "--out",
                str(cond.repair_metrics_path),
            ]
            + (["--limit", str(args.limit)] if args.limit > 0 else [])
        )


def _aggregate_row(
    *,
    condition_id: str,
    model: str,
    regime: str,
    dataset: str,
    family: str,
    rebuild_metrics_path: Path,
    repair_metrics_path: Optional[Path],
    pred_subdir: str,
) -> Dict[str, Any]:
    rebuild = _load_json(rebuild_metrics_path)
    summary = rebuild.get("summary", {})
    agg = rebuild.get("aggregate", {}).get("metrics", {})
    row = {
        "condition_id": condition_id,
        "family": family,
        "model": model,
        "regime": regime,
        "dataset": dataset,
        "evaluated_buildings": int(summary.get("evaluated_buildings", 0)),
        "missing_predictions": list(summary.get("missing_predictions", [])),
        "pred_subdir": pred_subdir,
        "rebuild_metrics_path": str(rebuild_metrics_path),
        "repair_metrics_path": str(repair_metrics_path) if repair_metrics_path else "",
        "rebuild_iou": float(agg.get("iou", 0.0)),
        "rebuild_f1": float(agg.get("f1", 0.0)),
        "rebuild_material_match": float(agg.get("material_match", 0.0)),
        "rebuild_coarse_material_match": float(agg.get("coarse_material_match", 0.0)),
        "rebuild_correct_placement_rate": float(agg.get("correct_placement_rate", 0.0)),
    }
    if repair_metrics_path and repair_metrics_path.is_file():
        rep = _load_json(repair_metrics_path)
        rs = rep.get("summary", {})
        row.update(
            {
                "repair_edit_distance_over_gt": float(rs.get("mean_edit_distance_over_gt", 0.0)),
                "repair_additions_over_gt": float(rs.get("mean_additions_over_gt", 0.0)),
                "repair_deletions_over_gt": float(rs.get("mean_deletions_over_gt", 0.0)),
                "repair_replacements_over_gt": float(rs.get("mean_replacements_over_gt", 0.0)),
                "repair_case_count": int(rs.get("evaluated_buildings", 0)),
                "repair_missing_predictions": list(rs.get("missing_predictions", [])),
            }
        )
    else:
        row.update(
            {
                "repair_edit_distance_over_gt": 0.0,
                "repair_additions_over_gt": 0.0,
                "repair_deletions_over_gt": 0.0,
                "repair_replacements_over_gt": 0.0,
                "repair_case_count": 0,
                "repair_missing_predictions": [],
            }
        )
    return row


def _collect_case_rows(cond_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rm = Path(cond_row["rebuild_metrics_path"])
    rp = Path(cond_row["repair_metrics_path"]) if cond_row["repair_metrics_path"] else None
    if not rm.is_file():
        return out
    rebuild = _load_json(rm)
    r_items = rebuild.get("items", [])
    repair_by_building: Dict[str, Dict[str, Any]] = {}
    if rp and rp.is_file():
        rep = _load_json(rp)
        for it in rep.get("items", []):
            repair_by_building[str(it.get("building", ""))] = it
    for it in r_items:
        b = str(it.get("building", ""))
        met = it.get("metrics", {})
        rit = repair_by_building.get(b, {})
        norm = rit.get("normalized", {})
        out.append(
            {
                "family": cond_row["family"],
                "model": cond_row["model"],
                "regime": cond_row["regime"],
                "dataset": cond_row["dataset"],
                "condition_id": cond_row["condition_id"],
                "building": b,
                "iou": float(met.get("iou", 0.0)),
                "f1": float(met.get("f1", 0.0)),
                "material_match": float(met.get("material_match", 0.0)),
                "coarse_material_match": float(met.get("coarse_material_match", 0.0)),
                "correct_placement_rate": float(met.get("correct_placement_rate", 0.0)),
                "edit_distance_over_gt": float(norm.get("edit_distance_over_gt", 0.0)),
                "additions_over_gt": float(norm.get("additions_over_gt", 0.0)),
                "deletions_over_gt": float(norm.get("deletions_over_gt", 0.0)),
                "replacements_over_gt": float(norm.get("replacements_over_gt", 0.0)),
            }
        )
    return out


def summarize(
    direct_rows: List[Dict[str, Any]],
    structured_rows: List[Dict[str, Any]],
    case_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # Comparison rows (direct vs structured) by (model, regime, dataset)
    key = lambda r: (r["model"], r["regime"], r["dataset"])
    dmap = {key(r): r for r in direct_rows}
    smap = {key(r): r for r in structured_rows}
    comparisons = []
    for k, s in sorted(smap.items()):
        d = dmap.get(k)
        if not d:
            continue
        comparisons.append(
            {
                "model": k[0],
                "regime": k[1],
                "dataset": k[2],
                "direct_condition_id": d["condition_id"],
                "structured_condition_id": s["condition_id"],
                "delta_iou": s["rebuild_iou"] - d["rebuild_iou"],
                "delta_f1": s["rebuild_f1"] - d["rebuild_f1"],
                "delta_material": s["rebuild_material_match"] - d["rebuild_material_match"],
                "delta_correct_placement": s["rebuild_correct_placement_rate"] - d["rebuild_correct_placement_rate"],
                "delta_edit_distance_over_gt": s["repair_edit_distance_over_gt"] - d["repair_edit_distance_over_gt"],
                "delta_additions_over_gt": s["repair_additions_over_gt"] - d["repair_additions_over_gt"],
                "delta_deletions_over_gt": s["repair_deletions_over_gt"] - d["repair_deletions_over_gt"],
                "delta_replacements_over_gt": s["repair_replacements_over_gt"] - d["repair_replacements_over_gt"],
            }
        )

    # all_200 merge for direct/structured main
    all200_rows = []
    for family, rows in (("direct", direct_rows), ("structured", structured_rows)):
        for model in ("openai", "claude"):
            main_v1 = next((r for r in rows if r["model"] == model and r["regime"] == "main" and r["dataset"] == "v1"), None)
            main_v4 = next((r for r in rows if r["model"] == model and r["regime"] == "main" and r["dataset"] == "v4"), None)
            if not main_v1 or not main_v4:
                continue
            merged = _weighted_merge(
                [
                    (
                        int(main_v1["evaluated_buildings"]),
                        {
                            "rebuild_iou": float(main_v1["rebuild_iou"]),
                            "rebuild_f1": float(main_v1["rebuild_f1"]),
                            "rebuild_material_match": float(main_v1["rebuild_material_match"]),
                            "rebuild_coarse_material_match": float(main_v1["rebuild_coarse_material_match"]),
                            "rebuild_correct_placement_rate": float(main_v1["rebuild_correct_placement_rate"]),
                            "repair_edit_distance_over_gt": float(main_v1["repair_edit_distance_over_gt"]),
                            "repair_additions_over_gt": float(main_v1["repair_additions_over_gt"]),
                            "repair_deletions_over_gt": float(main_v1["repair_deletions_over_gt"]),
                            "repair_replacements_over_gt": float(main_v1["repair_replacements_over_gt"]),
                        },
                    ),
                    (
                        int(main_v4["evaluated_buildings"]),
                        {
                            "rebuild_iou": float(main_v4["rebuild_iou"]),
                            "rebuild_f1": float(main_v4["rebuild_f1"]),
                            "rebuild_material_match": float(main_v4["rebuild_material_match"]),
                            "rebuild_coarse_material_match": float(main_v4["rebuild_coarse_material_match"]),
                            "rebuild_correct_placement_rate": float(main_v4["rebuild_correct_placement_rate"]),
                            "repair_edit_distance_over_gt": float(main_v4["repair_edit_distance_over_gt"]),
                            "repair_additions_over_gt": float(main_v4["repair_additions_over_gt"]),
                            "repair_deletions_over_gt": float(main_v4["repair_deletions_over_gt"]),
                            "repair_replacements_over_gt": float(main_v4["repair_replacements_over_gt"]),
                        },
                    ),
                ]
            )
            all200_rows.append(
                {
                    "family": family,
                    "model": model,
                    "regime": "main",
                    "dataset": "all_200",
                    "evaluated_buildings": int(main_v1["evaluated_buildings"]) + int(main_v4["evaluated_buildings"]),
                    **merged,
                }
            )

    # near-miss counts (per family/model/regime/dataset)
    near_miss = {}
    for fam in ("direct", "structured"):
        for model in ("openai", "claude"):
            for regime in ("main", "supplementary"):
                for ds in ("v1", "v4"):
                    xs = [
                        r
                        for r in case_rows
                        if r["family"] == fam and r["model"] == model and r["regime"] == regime and r["dataset"] == ds
                    ]
                    if not xs:
                        continue
                    hits = [r for r in xs if r["iou"] < 0.20 and r["edit_distance_over_gt"] <= 0.50]
                    near_miss[f"{fam}_{model}_{regime}_{ds}"] = {
                        "count": len(hits),
                        "total": len(xs),
                        "rate": (len(hits) / len(xs)) if xs else 0.0,
                        "criteria": "iou < 0.20 and edit_distance_over_gt <= 0.50",
                    }

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "Original benchmark structured-intermediate supplemental condition. "
            "Direct Main/Supplementary published results are preserved; this is an added comparison layer."
        ),
        "coverage": {
            "direct_condition_count": len(direct_rows),
            "structured_condition_count": len(structured_rows),
            "comparison_count": len(comparisons),
            "case_rows": len(case_rows),
        },
        "direct_rows": direct_rows,
        "structured_rows": structured_rows,
        "all_200_rows": all200_rows,
        "comparisons": comparisons,
        "near_miss": near_miss,
    }

    # write CSV (direct+structured + pairwise delta per building where both exist)
    csv_path = Path(args.out_cases_csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pair_index: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for r in case_rows:
        k = (r["model"], r["regime"], r["dataset"], r["building"], r["family"])
        pair_index[k] = r
    fieldnames = [
        "model",
        "regime",
        "dataset",
        "building",
        "direct_iou",
        "structured_iou",
        "delta_iou",
        "direct_f1",
        "structured_f1",
        "delta_f1",
        "direct_material_match",
        "structured_material_match",
        "delta_material_match",
        "direct_correct_placement_rate",
        "structured_correct_placement_rate",
        "delta_correct_placement_rate",
        "direct_edit_distance_over_gt",
        "structured_edit_distance_over_gt",
        "delta_edit_distance_over_gt",
        "direct_additions_over_gt",
        "structured_additions_over_gt",
        "delta_additions_over_gt",
        "direct_deletions_over_gt",
        "structured_deletions_over_gt",
        "delta_deletions_over_gt",
        "direct_replacements_over_gt",
        "structured_replacements_over_gt",
        "delta_replacements_over_gt",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        base_keys = sorted({(r["model"], r["regime"], r["dataset"], r["building"]) for r in case_rows})
        for m, rg, ds, b in base_keys:
            d = pair_index.get((m, rg, ds, b, "direct"))
            s = pair_index.get((m, rg, ds, b, "structured"))
            if not d or not s:
                continue
            rec = {
                "model": m,
                "regime": rg,
                "dataset": ds,
                "building": b,
            }
            for k in (
                "iou",
                "f1",
                "material_match",
                "correct_placement_rate",
                "edit_distance_over_gt",
                "additions_over_gt",
                "deletions_over_gt",
                "replacements_over_gt",
            ):
                rec[f"direct_{k}"] = d[k]
                rec[f"structured_{k}"] = s[k]
                rec[f"delta_{k}"] = s[k] - d[k]
            w.writerow(rec)

    # write report files
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    repair_only = {
        "created_at": out["created_at"],
        "direct_repair_rows": [
            {
                "condition_id": r["condition_id"],
                "model": r["model"],
                "regime": r["regime"],
                "dataset": r["dataset"],
                "repair_edit_distance_over_gt": r["repair_edit_distance_over_gt"],
                "repair_additions_over_gt": r["repair_additions_over_gt"],
                "repair_deletions_over_gt": r["repair_deletions_over_gt"],
                "repair_replacements_over_gt": r["repair_replacements_over_gt"],
                "repair_case_count": r["repair_case_count"],
            }
            for r in direct_rows
        ],
        "structured_repair_rows": [
            {
                "condition_id": r["condition_id"],
                "model": r["model"],
                "regime": r["regime"],
                "dataset": r["dataset"],
                "repair_edit_distance_over_gt": r["repair_edit_distance_over_gt"],
                "repair_additions_over_gt": r["repair_additions_over_gt"],
                "repair_deletions_over_gt": r["repair_deletions_over_gt"],
                "repair_replacements_over_gt": r["repair_replacements_over_gt"],
                "repair_case_count": r["repair_case_count"],
            }
            for r in structured_rows
        ],
    }
    Path(args.out_repair_json).resolve().write_text(json.dumps(repair_only, ensure_ascii=False, indent=2), encoding="utf-8")

    main_payload = {
        "created_at": out["created_at"],
        "direct_rows": [r for r in direct_rows if r["regime"] == "main"],
        "structured_rows": [r for r in structured_rows if r["regime"] == "main"],
        "comparisons": [r for r in comparisons if r["regime"] == "main"],
        "all_200_rows": [r for r in all200_rows if r["regime"] == "main"],
    }
    supp_payload = {
        "created_at": out["created_at"],
        "direct_rows": [r for r in direct_rows if r["regime"] == "supplementary"],
        "structured_rows": [r for r in structured_rows if r["regime"] == "supplementary"],
        "comparisons": [r for r in comparisons if r["regime"] == "supplementary"],
        "all_200_rows": [r for r in all200_rows if r["regime"] == "supplementary"],
    }
    Path(args.out_main_json).resolve().write_text(json.dumps(main_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_supp_json).resolve().write_text(json.dumps(supp_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md: List[str] = []
    md.append("# Original Benchmark Structured-Intermediate Summary")
    md.append("")
    md.append("この結果は original benchmark (`buildings_100_v1/v4`) に structured-intermediate 条件を追加した補助分析です。")
    md.append("既存 Main/Supplementary の published direct 結果は上書きせず、比較行のみ追加しています。")
    md.append("")
    md.append("## Coverage")
    md.append(f"- direct conditions: {len(direct_rows)}")
    md.append(f"- structured conditions: {len(structured_rows)}")
    md.append(f"- comparisons: {len(comparisons)}")
    md.append("")
    md.append("## Main: Direct vs Structured")
    for row in [r for r in comparisons if r["regime"] == "main"]:
        md.append(
            f"- {row['model']} {row['dataset']}: "
            f"ΔIoU {row['delta_iou']:+.4f}, ΔF1 {row['delta_f1']:+.4f}, "
            f"Δmaterial {row['delta_material']:+.4f}, Δcorrect {row['delta_correct_placement']:+.4f}, "
            f"Δedit {row['delta_edit_distance_over_gt']:+.4f}"
        )
    md.append("")
    md.append("## all_200 (main only)")
    for model in ("openai", "claude"):
        d = next((r for r in all200_rows if r["family"] == "direct" and r["model"] == model and r["regime"] == "main"), None)
        s = next((r for r in all200_rows if r["family"] == "structured" and r["model"] == model and r["regime"] == "main"), None)
        if not d or not s:
            continue
        md.append(
            f"- {model}: direct IoU {100*d['rebuild_iou']:.2f}% -> structured {100*s['rebuild_iou']:.2f}% "
            f"(Δ {100*(s['rebuild_iou']-d['rebuild_iou']):+.2f}pt), "
            f"direct edit {d['repair_edit_distance_over_gt']:.3f} -> structured {s['repair_edit_distance_over_gt']:.3f} "
            f"(Δ {(s['repair_edit_distance_over_gt']-d['repair_edit_distance_over_gt']):+.3f})"
        )
    md.append("")
    md.append("## Guardrails")
    md.append("- repair-effort は IoU/F1 の置換ではなく追加診断です。")
    md.append("- Main と Supplementary は分離して解釈してください。")
    md.append("- `llm_authored_10` は本集計に含めていません。")
    Path(args.out_md).resolve().write_text("\n".join(md) + "\n", encoding="utf-8")

    return out


def main() -> None:
    args = parse_args()
    gemini_model_tag = resolve_gemini_model_tag(args.gemini_model_tag)
    structured_models = parse_model_list(args.structured_models)

    direct_conditions = build_direct_conditions(
        include_gemini_main=bool(args.include_gemini_main),
        gemini_model_tag=gemini_model_tag,
    )
    structured_conditions = build_structured_conditions(
        models=structured_models,
        include_gemini_main=bool(args.include_gemini_main),
        gemini_model_tag=gemini_model_tag,
    )

    # Run structured pipeline if requested
    if args.run_structured:
        for cond in structured_conditions:
            run_structured_pipeline(cond, args)

    # collect direct rows
    direct_rows: List[Dict[str, Any]] = []
    for cond in direct_conditions:
        if not args.include_supplementary and cond.regime != "main":
            continue
        if not cond.rebuild_metrics_path.is_file():
            continue
        eval_count, _ = assert_original_benchmark_metrics(cond.rebuild_metrics_path, cond.dataset)
        payload = _load_json(cond.rebuild_metrics_path)
        pred_subdir = ""
        items = payload.get("items", [])
        if items:
            pred_subdir = str(items[0].get("pred_source_used", ""))
        repair_path = (
            ROOT
            / "outputs"
            / "i2t2b"
            / f"buildings_100_{cond.dataset}"
            / "metrics"
            / "repair"
            / f"{cond.model}_{cond.regime}.json"
        )
        row = _aggregate_row(
            condition_id=cond.condition_id,
            model=cond.model,
            regime=cond.regime,
            dataset=cond.dataset,
            family="direct",
            rebuild_metrics_path=cond.rebuild_metrics_path,
            repair_metrics_path=repair_path if repair_path.is_file() else None,
            pred_subdir=pred_subdir,
        )
        row["evaluated_buildings"] = eval_count
        direct_rows.append(row)

    # collect structured rows
    structured_rows: List[Dict[str, Any]] = []
    for cond in structured_conditions:
        if not cond.rebuild_metrics_path.is_file():
            continue
        eval_count, _ = assert_original_benchmark_metrics(cond.rebuild_metrics_path, cond.dataset)
        row = _aggregate_row(
            condition_id=cond.condition_id,
            model=cond.model,
            regime=cond.regime,
            dataset=cond.dataset,
            family="structured",
            rebuild_metrics_path=cond.rebuild_metrics_path,
            repair_metrics_path=cond.repair_metrics_path if cond.repair_metrics_path.is_file() else None,
            pred_subdir=cond.rebuild_subdir,
        )
        row["evaluated_buildings"] = eval_count
        row["description_subdir"] = cond.description_subdir
        row["intermediate_subdir"] = cond.intermediate_subdir
        row["plan_subdir"] = cond.plan_subdir
        row["rebuild_subdir"] = cond.rebuild_subdir
        structured_rows.append(row)

    # per-case rows
    case_rows: List[Dict[str, Any]] = []
    for r in direct_rows + structured_rows:
        case_rows.extend(_collect_case_rows(r))

    out = summarize(direct_rows, structured_rows, case_rows, args)
    print(f"[run_original_benchmark_structured] direct rows: {len(direct_rows)}")
    print(f"[run_original_benchmark_structured] structured rows: {len(structured_rows)}")
    print(f"[run_original_benchmark_structured] comparisons: {len(out['comparisons'])}")
    print(f"[run_original_benchmark_structured] wrote {args.out_json}")
    print(f"[run_original_benchmark_structured] wrote {args.out_md}")
    print(f"[run_original_benchmark_structured] wrote {args.out_cases_csv}")


if __name__ == "__main__":
    main()
