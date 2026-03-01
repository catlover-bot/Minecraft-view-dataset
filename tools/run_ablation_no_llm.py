#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4-condition ablation without additional LLM API calls.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root with building_xxx")
    parser.add_argument("--source_plan_subdir", required=True, help="Raw source plan subdir")
    parser.add_argument("--description_subdir", required=True, help="Description subdir for repair/self-refine")
    parser.add_argument("--model_tag", required=True, help="Tag used in output names (e.g., openai_gpt_5_mini)")
    parser.add_argument("--thresholds_json", default="tools/thresholds_levels.example.json")
    parser.add_argument("--building_pattern", default="building_*")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--self_refine_max_dim", type=int, default=192)
    parser.add_argument("--self_refine_max_iterations", type=int, default=2)
    parser.add_argument("--self_refine_min_score_gain", type=float, default=0.01)
    parser.add_argument("--self_refine_max_added_ops_per_iter", type=int, default=64)
    parser.add_argument("--self_refine_roof_search_variants", type=int, default=8)
    parser.add_argument("--self_refine_window_search_variants", type=int, default=8)
    parser.add_argument("--self_refine_max_search_candidates", type=int, default=20)
    parser.add_argument("--self_refine_material_budget_reprojection_strength", type=float, default=0.4)
    parser.add_argument("--self_refine_material_budget_reprojection_min_deficit_ratio", type=float, default=0.025)
    parser.add_argument("--self_refine_material_budget_reprojection_trigger_material_score", type=float, default=0.78)
    parser.add_argument("--self_refine_selection_op_penalty", type=float, default=0.0015)
    parser.add_argument("--material_budget_tolerance", type=float, default=0.35)
    parser.add_argument("--role_fix_min_confidence", type=float, default=0.78)
    parser.add_argument("--max_operations", type=int, default=260)

    return parser.parse_args()


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run_ablation_no_llm] $", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _repair_cmd(
    py: str,
    root: Path,
    *,
    dataset_root: Path,
    source_plan_subdir: str,
    out_plan_subdir: str,
    description_subdir: str,
    building_pattern: str,
    limit: int,
    overwrite: bool,
    strict_schema: bool,
    enforce_role_fixed: bool,
    require_material_budget: bool,
    prefer_description_palette: bool,
    material_budget_tolerance: float,
    role_fix_min_confidence: float,
    max_operations: int,
) -> List[str]:
    cmd = [
        py,
        str(root / "tools" / "repair_rebuild_plans.py"),
        "--dataset_root",
        str(dataset_root),
        "--source_plan_subdir",
        source_plan_subdir,
        "--out_plan_subdir",
        out_plan_subdir,
        "--description_subdir",
        description_subdir,
        "--building_pattern",
        building_pattern,
        "--material_budget_tolerance",
        str(material_budget_tolerance),
        "--role_fix_min_confidence",
        str(role_fix_min_confidence),
        "--max_operations",
        str(max_operations),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if overwrite:
        cmd.append("--overwrite")
    cmd.append("--strict_schema" if strict_schema else "--no_strict_schema")
    cmd.append("--enforce_role_fixed" if enforce_role_fixed else "--no_enforce_role_fixed")
    cmd.append("--require_material_budget" if require_material_budget else "--no_require_material_budget")
    cmd.append("--prefer_description_palette" if prefer_description_palette else "--no_prefer_description_palette")
    return cmd


def _self_refine_cmd(
    py: str,
    root: Path,
    *,
    dataset_root: Path,
    plan_subdir: str,
    description_subdir: str,
    out_plan_subdir: str,
    building_pattern: str,
    limit: int,
    overwrite: bool,
    strict_schema: bool,
    enforce_role_fixed: bool,
    require_material_budget: bool,
    prefer_description_palette: bool,
    enable_material_budget_reprojection: bool,
    max_dim: int,
    max_iterations: int,
    min_score_gain: float,
    max_added_ops_per_iter: int,
    roof_search_variants: int,
    window_search_variants: int,
    max_search_candidates: int,
    material_budget_reprojection_strength: float,
    material_budget_reprojection_min_deficit_ratio: float,
    material_budget_reprojection_trigger_material_score: float,
    selection_op_penalty: float,
    material_budget_tolerance: float,
    role_fix_min_confidence: float,
    max_operations: int,
) -> List[str]:
    cmd = [
        py,
        str(root / "tools" / "self_refine_rebuild_plans_no_gt.py"),
        "--dataset_root",
        str(dataset_root),
        "--plan_subdir",
        plan_subdir,
        "--description_subdir",
        description_subdir,
        "--out_plan_subdir",
        out_plan_subdir,
        "--building_pattern",
        building_pattern,
        "--max_dim",
        str(max_dim),
        "--max_iterations",
        str(max_iterations),
        "--min_score_gain",
        str(min_score_gain),
        "--max_added_ops_per_iter",
        str(max_added_ops_per_iter),
        "--roof_search_variants",
        str(roof_search_variants),
        "--window_search_variants",
        str(window_search_variants),
        "--max_search_candidates",
        str(max_search_candidates),
        "--material_budget_reprojection_strength",
        str(material_budget_reprojection_strength),
        "--material_budget_reprojection_min_deficit_ratio",
        str(material_budget_reprojection_min_deficit_ratio),
        "--material_budget_reprojection_trigger_material_score",
        str(material_budget_reprojection_trigger_material_score),
        "--selection_op_penalty",
        str(selection_op_penalty),
        "--material_budget_tolerance",
        str(material_budget_tolerance),
        "--role_fix_min_confidence",
        str(role_fix_min_confidence),
        "--max_operations",
        str(max_operations),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if overwrite:
        cmd.append("--overwrite")
    cmd.append("--strict_schema" if strict_schema else "--no_strict_schema")
    cmd.append("--enforce_role_fixed" if enforce_role_fixed else "--no_enforce_role_fixed")
    cmd.append("--require_material_budget" if require_material_budget else "--no_require_material_budget")
    cmd.append("--prefer_description_palette" if prefer_description_palette else "--no_prefer_description_palette")
    cmd.append("--enable_material_budget_reprojection" if enable_material_budget_reprojection else "--no_enable_material_budget_reprojection")
    return cmd


def _render_cmd(py: str, root: Path, *, dataset_root: Path, plan_subdir: str, out_subdir: str, building_pattern: str, limit: int, overwrite: bool) -> List[str]:
    cmd = [
        py,
        str(root / "tools" / "render_rebuild_from_plan.py"),
        "--dataset_root",
        str(dataset_root),
        "--plan_subdir",
        plan_subdir,
        "--out_subdir",
        out_subdir,
        "--building_pattern",
        building_pattern,
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def _eval_cmd(
    py: str,
    root: Path,
    *,
    dataset_root: Path,
    pred_subdir: str,
    out_path: Path,
    thresholds_json: str,
    building_pattern: str,
    limit: int,
) -> List[str]:
    cmd = [
        py,
        str(root / "tools" / "evaluate_rebuild_metrics.py"),
        "--gt_root",
        str(dataset_root),
        "--pred_root",
        str(dataset_root),
        "--building_pattern",
        building_pattern,
        "--pred_source",
        "rebuild_world",
        "--pred_subdir",
        pred_subdir,
        "--out",
        str(out_path),
        "--thresholds_json",
        thresholds_json,
        "--fail_on_missing_pred",
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def _aggregate_metric(metrics_path: Path, key: str) -> float:
    obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(obj["aggregate"]["metrics"][key])


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    # Condition naming.
    # A: schema_only
    # B: material_only
    # C: self_refine_only
    # D: full_stack
    tag = args.model_tag
    source_plan = args.source_plan_subdir
    description_subdir = args.description_subdir
    pattern = args.building_pattern

    plan_schema_only = f"rebuild_plan_ablation_schema_only_{tag}"
    plan_material_only = f"rebuild_plan_ablation_material_only_{tag}"
    plan_self_refine_only = f"rebuild_plan_ablation_self_refine_only_{tag}"
    plan_full_pre = f"rebuild_plan_ablation_full_pre_{tag}"
    plan_full = f"rebuild_plan_ablation_full_{tag}"

    world_schema_only = f"rebuild_world_ablation_schema_only_{tag}"
    world_material_only = f"rebuild_world_ablation_material_only_{tag}"
    world_self_refine_only = f"rebuild_world_ablation_self_refine_only_{tag}"
    world_full = f"rebuild_world_ablation_full_{tag}"

    metrics_schema_only = dataset_root / f"metrics_levels_ablation_schema_only_{tag}.json"
    metrics_material_only = dataset_root / f"metrics_levels_ablation_material_only_{tag}.json"
    metrics_self_refine_only = dataset_root / f"metrics_levels_ablation_self_refine_only_{tag}.json"
    metrics_full = dataset_root / f"metrics_levels_ablation_full_{tag}.json"

    # 1) schema_only
    _run(
        _repair_cmd(
            py,
            root,
            dataset_root=dataset_root,
            source_plan_subdir=source_plan,
            out_plan_subdir=plan_schema_only,
            description_subdir=description_subdir,
            building_pattern=pattern,
            limit=args.limit,
            overwrite=args.overwrite,
            strict_schema=True,
            enforce_role_fixed=False,
            require_material_budget=False,
            prefer_description_palette=False,
            material_budget_tolerance=args.material_budget_tolerance,
            role_fix_min_confidence=args.role_fix_min_confidence,
            max_operations=args.max_operations,
        ),
        cwd=root,
    )

    # 2) material_only
    _run(
        _repair_cmd(
            py,
            root,
            dataset_root=dataset_root,
            source_plan_subdir=source_plan,
            out_plan_subdir=plan_material_only,
            description_subdir=description_subdir,
            building_pattern=pattern,
            limit=args.limit,
            overwrite=args.overwrite,
            strict_schema=False,
            enforce_role_fixed=True,
            require_material_budget=True,
            prefer_description_palette=True,
            material_budget_tolerance=args.material_budget_tolerance,
            role_fix_min_confidence=args.role_fix_min_confidence,
            max_operations=args.max_operations,
        ),
        cwd=root,
    )

    # 3) self_refine_only (from source plan, no schema/material constraints).
    _run(
        _self_refine_cmd(
            py,
            root,
            dataset_root=dataset_root,
            plan_subdir=source_plan,
            description_subdir=description_subdir,
            out_plan_subdir=plan_self_refine_only,
            building_pattern=pattern,
            limit=args.limit,
            overwrite=args.overwrite,
            strict_schema=False,
            enforce_role_fixed=False,
            require_material_budget=False,
            prefer_description_palette=False,
            enable_material_budget_reprojection=False,
            max_dim=args.self_refine_max_dim,
            max_iterations=args.self_refine_max_iterations,
            min_score_gain=args.self_refine_min_score_gain,
            max_added_ops_per_iter=args.self_refine_max_added_ops_per_iter,
            roof_search_variants=args.self_refine_roof_search_variants,
            window_search_variants=args.self_refine_window_search_variants,
            max_search_candidates=args.self_refine_max_search_candidates,
            material_budget_reprojection_strength=args.self_refine_material_budget_reprojection_strength,
            material_budget_reprojection_min_deficit_ratio=args.self_refine_material_budget_reprojection_min_deficit_ratio,
            material_budget_reprojection_trigger_material_score=args.self_refine_material_budget_reprojection_trigger_material_score,
            selection_op_penalty=args.self_refine_selection_op_penalty,
            material_budget_tolerance=args.material_budget_tolerance,
            role_fix_min_confidence=args.role_fix_min_confidence,
            max_operations=args.max_operations,
        ),
        cwd=root,
    )

    # 4) full_stack = (schema+material repair) + self_refine(with reprojection)
    _run(
        _repair_cmd(
            py,
            root,
            dataset_root=dataset_root,
            source_plan_subdir=source_plan,
            out_plan_subdir=plan_full_pre,
            description_subdir=description_subdir,
            building_pattern=pattern,
            limit=args.limit,
            overwrite=args.overwrite,
            strict_schema=True,
            enforce_role_fixed=True,
            require_material_budget=True,
            prefer_description_palette=True,
            material_budget_tolerance=args.material_budget_tolerance,
            role_fix_min_confidence=args.role_fix_min_confidence,
            max_operations=args.max_operations,
        ),
        cwd=root,
    )
    _run(
        _self_refine_cmd(
            py,
            root,
            dataset_root=dataset_root,
            plan_subdir=plan_full_pre,
            description_subdir=description_subdir,
            out_plan_subdir=plan_full,
            building_pattern=pattern,
            limit=args.limit,
            overwrite=args.overwrite,
            strict_schema=True,
            enforce_role_fixed=True,
            require_material_budget=True,
            prefer_description_palette=True,
            enable_material_budget_reprojection=True,
            max_dim=args.self_refine_max_dim,
            max_iterations=args.self_refine_max_iterations,
            min_score_gain=args.self_refine_min_score_gain,
            max_added_ops_per_iter=args.self_refine_max_added_ops_per_iter,
            roof_search_variants=args.self_refine_roof_search_variants,
            window_search_variants=args.self_refine_window_search_variants,
            max_search_candidates=args.self_refine_max_search_candidates,
            material_budget_reprojection_strength=args.self_refine_material_budget_reprojection_strength,
            material_budget_reprojection_min_deficit_ratio=args.self_refine_material_budget_reprojection_min_deficit_ratio,
            material_budget_reprojection_trigger_material_score=args.self_refine_material_budget_reprojection_trigger_material_score,
            selection_op_penalty=args.self_refine_selection_op_penalty,
            material_budget_tolerance=args.material_budget_tolerance,
            role_fix_min_confidence=args.role_fix_min_confidence,
            max_operations=args.max_operations,
        ),
        cwd=root,
    )

    # Render all conditions.
    _run(_render_cmd(py, root, dataset_root=dataset_root, plan_subdir=plan_schema_only, out_subdir=world_schema_only, building_pattern=pattern, limit=args.limit, overwrite=args.overwrite), cwd=root)
    _run(_render_cmd(py, root, dataset_root=dataset_root, plan_subdir=plan_material_only, out_subdir=world_material_only, building_pattern=pattern, limit=args.limit, overwrite=args.overwrite), cwd=root)
    _run(_render_cmd(py, root, dataset_root=dataset_root, plan_subdir=plan_self_refine_only, out_subdir=world_self_refine_only, building_pattern=pattern, limit=args.limit, overwrite=args.overwrite), cwd=root)
    _run(_render_cmd(py, root, dataset_root=dataset_root, plan_subdir=plan_full, out_subdir=world_full, building_pattern=pattern, limit=args.limit, overwrite=args.overwrite), cwd=root)

    # Evaluate all conditions.
    _run(_eval_cmd(py, root, dataset_root=dataset_root, pred_subdir=world_schema_only, out_path=metrics_schema_only, thresholds_json=args.thresholds_json, building_pattern=pattern, limit=args.limit), cwd=root)
    _run(_eval_cmd(py, root, dataset_root=dataset_root, pred_subdir=world_material_only, out_path=metrics_material_only, thresholds_json=args.thresholds_json, building_pattern=pattern, limit=args.limit), cwd=root)
    _run(_eval_cmd(py, root, dataset_root=dataset_root, pred_subdir=world_self_refine_only, out_path=metrics_self_refine_only, thresholds_json=args.thresholds_json, building_pattern=pattern, limit=args.limit), cwd=root)
    _run(_eval_cmd(py, root, dataset_root=dataset_root, pred_subdir=world_full, out_path=metrics_full, thresholds_json=args.thresholds_json, building_pattern=pattern, limit=args.limit), cwd=root)

    summary_path = dataset_root / f"ablation_summary_{tag}.json"
    summary: Dict[str, Dict[str, float]] = {
        "schema_only": {
            "iou": _aggregate_metric(metrics_schema_only, "iou"),
            "f1": _aggregate_metric(metrics_schema_only, "f1"),
            "material_match": _aggregate_metric(metrics_schema_only, "material_match"),
        },
        "material_only": {
            "iou": _aggregate_metric(metrics_material_only, "iou"),
            "f1": _aggregate_metric(metrics_material_only, "f1"),
            "material_match": _aggregate_metric(metrics_material_only, "material_match"),
        },
        "self_refine_only": {
            "iou": _aggregate_metric(metrics_self_refine_only, "iou"),
            "f1": _aggregate_metric(metrics_self_refine_only, "f1"),
            "material_match": _aggregate_metric(metrics_self_refine_only, "material_match"),
        },
        "full_stack": {
            "iou": _aggregate_metric(metrics_full, "iou"),
            "f1": _aggregate_metric(metrics_full, "f1"),
            "material_match": _aggregate_metric(metrics_full, "material_match"),
        },
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "model_tag": args.model_tag,
        "source_plan_subdir": source_plan,
        "description_subdir": description_subdir,
        "metrics_paths": {
            "schema_only": str(metrics_schema_only),
            "material_only": str(metrics_material_only),
            "self_refine_only": str(metrics_self_refine_only),
            "full_stack": str(metrics_full),
        },
        "summary": summary,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_ablation_no_llm] wrote summary: {summary_path}")
    for cond, vals in summary.items():
        print(
            f"  {cond}: IoU={vals['iou']:.4f} F1={vals['f1']:.4f} material={vals['material_match']:.4f}"
        )


if __name__ == "__main__":
    main()

