#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from tools.llm_config import load_llm_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image->text->build experiment pipeline.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root with building_xxx")
    parser.add_argument("--dotenv", default="", help="Optional .env path")
    parser.add_argument("--provider", default="", help="openai|anthropic|gemini|mock")
    parser.add_argument(
        "--output_tag",
        default="",
        help="Optional tag appended to default outputs. Example: openai_gpt_5_mini",
    )
    parser.add_argument(
        "--split_by_model",
        dest="split_by_model",
        action="store_true",
        help="Auto-separate outputs by provider+model tag (default: on).",
    )
    parser.add_argument(
        "--no_split_by_model",
        dest="split_by_model",
        action="store_false",
        help="Disable model-based output separation.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max buildings (0=all)")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.set_defaults(split_by_model=True)

    parser.add_argument("--description_subdir", default="description")
    parser.add_argument("--plan_subdir", default="rebuild_plan")
    parser.add_argument("--rebuild_subdir", default="rebuild_world")

    parser.add_argument("--skip_descriptions", action="store_true")
    parser.add_argument("--skip_plan", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--skip_description_eval", action="store_true")
    parser.add_argument("--skip_rebuild_eval", action="store_true")

    parser.add_argument("--desc_temperature", type=float, default=0.2)
    parser.add_argument("--desc_max_tokens", type=int, default=1800)
    parser.add_argument("--desc_max_images", type=int, default=6)
    parser.add_argument("--desc_llm_seed", type=int, default=-1, help="Optional description LLM seed (OpenAI only).")

    parser.add_argument("--plan_temperature", type=float, default=0.2)
    parser.add_argument("--plan_max_tokens", type=int, default=1800)
    parser.add_argument("--plan_llm_seed", type=int, default=-1, help="Optional plan LLM seed (OpenAI only).")
    parser.add_argument("--plan_prompt_profile", default="", help="Optional rebuild-plan prompt profile JSON path.")
    parser.add_argument(
        "--plan_critic_revise",
        action="store_true",
        help="Enable critic-revise second pass in rebuild planning.",
    )
    parser.add_argument("--plan_strict_schema", dest="plan_strict_schema", action="store_true")
    parser.add_argument("--plan_no_strict_schema", dest="plan_strict_schema", action="store_false")
    parser.add_argument("--plan_enforce_role_fixed", dest="plan_enforce_role_fixed", action="store_true")
    parser.add_argument("--plan_no_enforce_role_fixed", dest="plan_enforce_role_fixed", action="store_false")
    parser.add_argument("--plan_require_material_budget", dest="plan_require_material_budget", action="store_true")
    parser.add_argument("--plan_no_require_material_budget", dest="plan_require_material_budget", action="store_false")
    parser.add_argument("--plan_material_budget_tolerance", type=float, default=0.25)
    parser.add_argument("--plan_role_fix_min_confidence", type=float, default=0.7)
    parser.add_argument("--plan_prefer_description_palette", dest="plan_prefer_description_palette", action="store_true")
    parser.add_argument("--plan_no_prefer_description_palette", dest="plan_prefer_description_palette", action="store_false")
    parser.add_argument("--plan_max_operations", type=int, default=260)
    parser.add_argument("--use_heuristic_plan_only", action="store_true")
    parser.add_argument("--no_fallback_heuristic", action="store_true")
    parser.add_argument(
        "--enable_self_refine_no_gt",
        action="store_true",
        help="Run post-plan no-GT self-consistency refinement before rendering.",
    )
    parser.add_argument(
        "--self_refine_plan_subdir",
        default="",
        help="Output plan subdir for self-refined plans (default: <plan_subdir>_self_refine_no_gt).",
    )
    parser.add_argument("--self_refine_max_dim", type=int, default=192)
    parser.add_argument("--self_refine_max_iterations", type=int, default=2)
    parser.add_argument("--self_refine_min_score_gain", type=float, default=0.01)
    parser.add_argument("--self_refine_max_added_ops_per_iter", type=int, default=64)
    parser.add_argument("--self_refine_roof_search_variants", type=int, default=6)
    parser.add_argument("--self_refine_window_search_variants", type=int, default=6)
    parser.add_argument("--self_refine_max_search_candidates", type=int, default=16)
    parser.add_argument(
        "--self_refine_enable_candidate_diversification",
        dest="self_refine_enable_candidate_diversification",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enable_candidate_diversification",
        dest="self_refine_enable_candidate_diversification",
        action="store_false",
    )
    parser.add_argument(
        "--self_refine_candidate_diversification_high_risk_only",
        dest="self_refine_candidate_diversification_high_risk_only",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_candidate_diversification_high_risk_only",
        dest="self_refine_candidate_diversification_high_risk_only",
        action="store_false",
    )
    parser.add_argument("--self_refine_candidate_diversification_risk_threshold", type=float, default=-1.0)
    parser.add_argument("--self_refine_candidate_diversification_underbuild_ratio_threshold", type=float, default=0.92)
    parser.add_argument(
        "--self_refine_wall_balance_shell_high_risk_only",
        dest="self_refine_wall_balance_shell_high_risk_only",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_wall_balance_shell_high_risk_only",
        dest="self_refine_wall_balance_shell_high_risk_only",
        action="store_false",
    )
    parser.add_argument("--self_refine_wall_balance_shell_min_deficit", type=float, default=0.06)
    parser.add_argument("--self_refine_wall_balance_shell_max_shape_drop_forecast", type=float, default=0.06)
    parser.add_argument("--self_refine_wall_balance_shell_shape_drop_scale", type=float, default=0.30)
    parser.add_argument(
        "--self_refine_wall_shell_model_specific",
        dest="self_refine_wall_shell_model_specific",
        action="store_true",
        help="Use model-specific wall-shell thresholds (Claude-relaxed profile) when self-refine runs.",
    )
    parser.add_argument(
        "--self_refine_no_wall_shell_model_specific",
        dest="self_refine_wall_shell_model_specific",
        action="store_false",
        help="Disable model-specific wall-shell thresholds and always use common values.",
    )
    parser.add_argument("--self_refine_wall_shell_claude_min_deficit", type=float, default=0.05)
    parser.add_argument("--self_refine_wall_shell_claude_max_shape_drop_forecast", type=float, default=0.10)
    parser.add_argument("--self_refine_material_budget_reprojection_strength", type=float, default=0.5)
    parser.add_argument("--self_refine_material_budget_reprojection_min_deficit_ratio", type=float, default=0.03)
    parser.add_argument("--self_refine_material_budget_reprojection_trigger_material_score", type=float, default=0.65)
    parser.add_argument("--self_refine_selection_op_penalty", type=float, default=0.0015)
    parser.add_argument("--self_refine_selection_overbuild_penalty", type=float, default=0.35)
    parser.add_argument("--self_refine_selection_underbuild_penalty", type=float, default=0.0)
    parser.add_argument("--self_refine_selection_material_budget_violation_penalty", type=float, default=0.03)
    parser.add_argument("--self_refine_selection_material_budget_count_weight", type=float, default=0.25)
    parser.add_argument("--self_refine_selection_ratio_target_penalty", type=float, default=0.18)
    parser.add_argument("--self_refine_selection_shape_drop_penalty", type=float, default=0.35)
    parser.add_argument("--self_refine_selection_dim_drop_penalty", type=float, default=0.40)
    parser.add_argument("--self_refine_selection_growth_excess_penalty", type=float, default=0.35)
    parser.add_argument("--self_refine_selection_footprint_profile_penalty", type=float, default=0.15)
    parser.add_argument("--self_refine_selection_height_profile_penalty", type=float, default=0.20)
    parser.add_argument("--self_refine_max_pred_target_ratio", type=float, default=1.05)
    parser.add_argument("--self_refine_adaptive_risk_ratio_threshold", type=float, default=1.25)
    parser.add_argument("--self_refine_adaptive_high_risk_max_pred_target_ratio", type=float, default=1.10)
    parser.add_argument("--self_refine_adaptive_high_risk_overbuild_penalty", type=float, default=0.35)
    parser.add_argument("--self_refine_adaptive_normal_max_pred_target_ratio", type=float, default=1.20)
    parser.add_argument("--self_refine_adaptive_normal_overbuild_penalty", type=float, default=0.15)
    parser.add_argument(
        "--self_refine_enable_candidate_growth_guard",
        dest="self_refine_enable_candidate_growth_guard",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enable_candidate_growth_guard",
        dest="self_refine_enable_candidate_growth_guard",
        action="store_false",
    )
    parser.add_argument("--self_refine_candidate_growth_ratio_max", type=float, default=1.18)
    parser.add_argument("--self_refine_candidate_growth_ratio_underbuild_threshold", type=float, default=0.90)
    parser.add_argument("--self_refine_candidate_growth_ratio_underbuild_max", type=float, default=1.45)
    parser.add_argument("--self_refine_max_shape_proxy_drop", type=float, default=0.03)
    parser.add_argument("--self_refine_max_dim_score_drop", type=float, default=0.06)
    parser.add_argument(
        "--self_refine_enable_profile_match_guard",
        dest="self_refine_enable_profile_match_guard",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enable_profile_match_guard",
        dest="self_refine_enable_profile_match_guard",
        action="store_false",
    )
    parser.add_argument("--self_refine_max_footprint_profile_l1", type=float, default=0.22)
    parser.add_argument("--self_refine_max_height_profile_l1", type=float, default=0.25)
    parser.add_argument(
        "--self_refine_enforce_two_stage_generation",
        dest="self_refine_enforce_two_stage_generation",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enforce_two_stage_generation",
        dest="self_refine_enforce_two_stage_generation",
        action="store_false",
    )
    parser.add_argument("--self_refine_two_stage_coarse_ready_threshold", type=float, default=0.70)
    parser.add_argument(
        "--self_refine_enable_material_budget_reprojection",
        dest="self_refine_enable_material_budget_reprojection",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enable_material_budget_reprojection",
        dest="self_refine_enable_material_budget_reprojection",
        action="store_false",
    )
    parser.add_argument(
        "--self_refine_enable_overbuild_guard",
        dest="self_refine_enable_overbuild_guard",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enable_overbuild_guard",
        dest="self_refine_enable_overbuild_guard",
        action="store_false",
    )
    parser.add_argument(
        "--self_refine_enable_adaptive_overbuild_control",
        dest="self_refine_enable_adaptive_overbuild_control",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enable_adaptive_overbuild_control",
        dest="self_refine_enable_adaptive_overbuild_control",
        action="store_false",
    )
    parser.add_argument(
        "--self_refine_enable_shape_degradation_guard",
        dest="self_refine_enable_shape_degradation_guard",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enable_shape_degradation_guard",
        dest="self_refine_enable_shape_degradation_guard",
        action="store_false",
    )
    parser.add_argument(
        "--self_refine_reject_strict_blocking_candidates",
        dest="self_refine_reject_strict_blocking_candidates",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_reject_strict_blocking_candidates",
        dest="self_refine_reject_strict_blocking_candidates",
        action="store_false",
    )
    parser.add_argument(
        "--self_refine_enable_conditional_precboost",
        dest="self_refine_enable_conditional_precboost",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_enable_conditional_precboost",
        dest="self_refine_enable_conditional_precboost",
        action="store_false",
    )
    parser.add_argument(
        "--self_refine_conditional_precboost_require_keyword_match",
        dest="self_refine_conditional_precboost_require_keyword_match",
        action="store_true",
    )
    parser.add_argument(
        "--self_refine_no_conditional_precboost_require_keyword_match",
        dest="self_refine_conditional_precboost_require_keyword_match",
        action="store_false",
    )
    parser.add_argument("--self_refine_conditional_precboost_allow_keywords", type=str, default="bunker,storage,shed,house,residential,cottage")
    parser.add_argument("--self_refine_conditional_precboost_block_keywords", type=str, default="monument,watchtower,fortification,shrine,decorative")
    parser.add_argument("--self_refine_conditional_precboost_max_roof_score", type=float, default=0.92)
    parser.add_argument("--self_refine_conditional_precboost_min_material_score", type=float, default=0.6)
    parser.add_argument("--self_refine_conditional_precboost_max_window_score", type=float, default=1.0)
    parser.add_argument("--self_refine_conditional_precboost_min_raw_score_gain", type=float, default=0.008)
    parser.add_argument("--self_refine_conditional_precboost_max_overbuild_excess", type=float, default=0.12)
    parser.add_argument("--self_refine_conditional_precboost_max_underbuild_excess", type=float, default=0.35)
    parser.add_argument("--self_refine_conditional_precboost_max_budget_violation_rel_increase", type=float, default=0.05)
    parser.add_argument("--self_refine_precboost_selection_op_penalty", type=float, default=0.001)
    parser.add_argument("--self_refine_precboost_selection_overbuild_penalty", type=float, default=0.25)
    parser.add_argument("--self_refine_precboost_selection_underbuild_penalty", type=float, default=0.45)
    parser.add_argument("--self_refine_precboost_max_pred_target_ratio", type=float, default=1.05)
    parser.add_argument("--self_refine_precboost_adaptive_risk_ratio_threshold", type=float, default=1.22)
    parser.add_argument("--self_refine_precboost_adaptive_high_risk_max_pred_target_ratio", type=float, default=1.12)
    parser.add_argument("--self_refine_precboost_adaptive_high_risk_overbuild_penalty", type=float, default=0.30)
    parser.add_argument("--self_refine_precboost_adaptive_normal_max_pred_target_ratio", type=float, default=1.22)
    parser.add_argument("--self_refine_precboost_adaptive_normal_overbuild_penalty", type=float, default=0.12)
    parser.set_defaults(
        plan_strict_schema=True,
        plan_enforce_role_fixed=True,
        plan_require_material_budget=True,
        plan_prefer_description_palette=True,
        self_refine_enable_material_budget_reprojection=True,
        self_refine_enable_overbuild_guard=True,
        self_refine_enable_adaptive_overbuild_control=True,
        self_refine_enable_shape_degradation_guard=True,
        self_refine_enable_profile_match_guard=True,
        self_refine_enforce_two_stage_generation=True,
        self_refine_reject_strict_blocking_candidates=True,
        self_refine_enable_candidate_diversification=False,
        self_refine_candidate_diversification_high_risk_only=True,
        self_refine_wall_balance_shell_high_risk_only=False,
        self_refine_wall_shell_model_specific=True,
        self_refine_enable_conditional_precboost=True,
        self_refine_conditional_precboost_require_keyword_match=True,
        self_refine_enable_candidate_growth_guard=True,
    )

    parser.add_argument("--rebuild_metrics_out", default="metrics_levels.json")
    parser.add_argument("--description_metrics_out", default="description_metrics.json")
    parser.add_argument("--thresholds_json", default="tools/thresholds_levels.example.json")
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[run_i2t2b_experiment] $", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _resolve_provider_model(args: argparse.Namespace) -> Tuple[str, str]:
    cfg = load_llm_config(args.dotenv or None)
    provider = (args.provider or cfg.provider or "openai").strip().lower()
    if provider == "openai":
        model = cfg.openai_model or "openai_model"
    elif provider == "anthropic":
        model = cfg.anthropic_model or "anthropic_model"
    elif provider == "gemini":
        model = cfg.gemini_model or "gemini_model"
    elif provider == "mock":
        model = "mock-model"
    else:
        model = "unknown-model"
    return provider, model


def _resolve_output_names(args: argparse.Namespace) -> Tuple[str, str, str, str, str, str]:
    desc_subdir = args.description_subdir
    plan_subdir = args.plan_subdir
    rebuild_subdir = args.rebuild_subdir
    desc_metrics_out = args.description_metrics_out
    rebuild_metrics_out = args.rebuild_metrics_out

    tag = (args.output_tag or "").strip()
    if not tag and args.split_by_model:
        provider, model = _resolve_provider_model(args)
        tag = f"{_slugify(provider)}_{_slugify(model)}"
    elif tag:
        tag = _slugify(tag)

    if tag:
        if desc_subdir == "description":
            desc_subdir = f"description_{tag}"
        if plan_subdir == "rebuild_plan":
            plan_subdir = f"rebuild_plan_{tag}"
        if rebuild_subdir == "rebuild_world":
            rebuild_subdir = f"rebuild_world_{tag}"
        if desc_metrics_out == "description_metrics.json":
            desc_metrics_out = f"description_metrics_{tag}.json"
        if rebuild_metrics_out == "metrics_levels.json":
            rebuild_metrics_out = f"metrics_levels_{tag}.json"

    return tag, desc_subdir, plan_subdir, rebuild_subdir, desc_metrics_out, rebuild_metrics_out


def _resolve_self_refine_wall_shell_thresholds(args: argparse.Namespace) -> Tuple[float, float, str]:
    min_def = float(args.self_refine_wall_balance_shell_min_deficit)
    max_shape = float(args.self_refine_wall_balance_shell_max_shape_drop_forecast)
    mode = "common"
    if not bool(args.self_refine_wall_shell_model_specific):
        return min_def, max_shape, mode
    provider, model = _resolve_provider_model(args)
    p = str(provider).strip().lower()
    m = str(model).strip().lower()
    if "anthropic" in p or "claude" in m:
        return (
            float(args.self_refine_wall_shell_claude_min_deficit),
            float(args.self_refine_wall_shell_claude_max_shape_drop_forecast),
            "claude_relaxed",
        )
    return min_def, max_shape, mode


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    py = sys.executable
    tag, desc_subdir, plan_subdir, rebuild_subdir, desc_metrics_out, rebuild_metrics_out = _resolve_output_names(args)
    print(
        "[run_i2t2b_experiment] outputs:"
        f" tag={tag or '(none)'}"
        f" description_subdir={desc_subdir}"
        f" plan_subdir={plan_subdir}"
        f" rebuild_subdir={rebuild_subdir}"
        f" description_metrics_out={desc_metrics_out}"
        f" rebuild_metrics_out={rebuild_metrics_out}"
    )

    common = ["--dataset_root", str(dataset_root)]
    common += ["--building_pattern", args.building_pattern]
    if args.limit > 0:
        common += ["--limit", str(args.limit)]
    if args.overwrite:
        common += ["--overwrite"]
    if args.provider:
        common += ["--provider", args.provider]
    if args.dotenv:
        common += ["--dotenv", args.dotenv]

    if not args.skip_descriptions:
        cmd = [
            py,
            str(root / "tools" / "generate_building_descriptions.py"),
            *common,
            "--out_subdir",
            desc_subdir,
            "--max_images",
            str(args.desc_max_images),
            "--temperature",
            str(args.desc_temperature),
            "--max_tokens",
            str(args.desc_max_tokens),
            "--llm_seed",
            str(args.desc_llm_seed),
        ]
        run_cmd(cmd, cwd=root)

    if not args.skip_plan:
        cmd = [
            py,
            str(root / "tools" / "generate_rebuild_plans.py"),
            *common,
            "--description_subdir",
            desc_subdir,
            "--plan_subdir",
            plan_subdir,
            "--temperature",
            str(args.plan_temperature),
            "--max_tokens",
            str(args.plan_max_tokens),
            "--llm_seed",
            str(args.plan_llm_seed),
            "--material_budget_tolerance",
            str(args.plan_material_budget_tolerance),
            "--role_fix_min_confidence",
            str(args.plan_role_fix_min_confidence),
            "--max_operations",
            str(args.plan_max_operations),
        ]
        if args.plan_prompt_profile:
            cmd += ["--prompt_profile", args.plan_prompt_profile]
        if args.plan_critic_revise:
            cmd.append("--critic_revise")
        if args.plan_strict_schema:
            cmd.append("--strict_schema")
        else:
            cmd.append("--no_strict_schema")
        if args.plan_enforce_role_fixed:
            cmd.append("--enforce_role_fixed")
        else:
            cmd.append("--no_enforce_role_fixed")
        if args.plan_require_material_budget:
            cmd.append("--require_material_budget")
        else:
            cmd.append("--no_require_material_budget")
        if args.plan_prefer_description_palette:
            cmd.append("--prefer_description_palette")
        else:
            cmd.append("--no_prefer_description_palette")
        if args.use_heuristic_plan_only:
            cmd.append("--use_heuristic_only")
        if args.no_fallback_heuristic:
            cmd.append("--no_fallback_heuristic")
        run_cmd(cmd, cwd=root)

    effective_plan_subdir = plan_subdir
    if args.enable_self_refine_no_gt:
        self_refine_plan_subdir = (args.self_refine_plan_subdir or "").strip()
        if not self_refine_plan_subdir:
            self_refine_plan_subdir = f"{plan_subdir}_self_refine_no_gt"
        wall_shell_min_deficit, wall_shell_max_shape_drop_forecast, wall_shell_profile = _resolve_self_refine_wall_shell_thresholds(args)
        print(
            "[run_i2t2b_experiment] self_refine wall-shell profile:"
            f" {wall_shell_profile}"
            f" min_deficit={wall_shell_min_deficit:.4f}"
            f" max_shape_drop_forecast={wall_shell_max_shape_drop_forecast:.4f}"
        )
        cmd = [
            py,
            str(root / "tools" / "self_refine_rebuild_plans_no_gt.py"),
            "--dataset_root",
            str(dataset_root),
            "--plan_subdir",
            plan_subdir,
            "--description_subdir",
            desc_subdir,
            "--out_plan_subdir",
            self_refine_plan_subdir,
            "--building_pattern",
            args.building_pattern,
            "--max_dim",
            str(args.self_refine_max_dim),
            "--max_iterations",
            str(args.self_refine_max_iterations),
            "--min_score_gain",
            str(args.self_refine_min_score_gain),
            "--max_added_ops_per_iter",
            str(args.self_refine_max_added_ops_per_iter),
            "--roof_search_variants",
            str(args.self_refine_roof_search_variants),
            "--window_search_variants",
            str(args.self_refine_window_search_variants),
            "--max_search_candidates",
            str(args.self_refine_max_search_candidates),
            "--material_budget_reprojection_strength",
            str(args.self_refine_material_budget_reprojection_strength),
            "--material_budget_reprojection_min_deficit_ratio",
            str(args.self_refine_material_budget_reprojection_min_deficit_ratio),
            "--material_budget_reprojection_trigger_material_score",
            str(args.self_refine_material_budget_reprojection_trigger_material_score),
            "--selection_op_penalty",
            str(args.self_refine_selection_op_penalty),
            "--selection_overbuild_penalty",
            str(args.self_refine_selection_overbuild_penalty),
            "--selection_underbuild_penalty",
            str(args.self_refine_selection_underbuild_penalty),
            "--selection_material_budget_violation_penalty",
            str(args.self_refine_selection_material_budget_violation_penalty),
            "--selection_material_budget_count_weight",
            str(args.self_refine_selection_material_budget_count_weight),
            "--selection_ratio_target_penalty",
            str(args.self_refine_selection_ratio_target_penalty),
            "--selection_shape_drop_penalty",
            str(args.self_refine_selection_shape_drop_penalty),
            "--selection_dim_drop_penalty",
            str(args.self_refine_selection_dim_drop_penalty),
            "--selection_growth_excess_penalty",
            str(args.self_refine_selection_growth_excess_penalty),
            "--selection_footprint_profile_penalty",
            str(args.self_refine_selection_footprint_profile_penalty),
            "--selection_height_profile_penalty",
            str(args.self_refine_selection_height_profile_penalty),
            "--max_pred_target_ratio",
            str(args.self_refine_max_pred_target_ratio),
            "--adaptive_risk_ratio_threshold",
            str(args.self_refine_adaptive_risk_ratio_threshold),
            "--adaptive_high_risk_max_pred_target_ratio",
            str(args.self_refine_adaptive_high_risk_max_pred_target_ratio),
            "--adaptive_high_risk_overbuild_penalty",
            str(args.self_refine_adaptive_high_risk_overbuild_penalty),
            "--adaptive_normal_max_pred_target_ratio",
            str(args.self_refine_adaptive_normal_max_pred_target_ratio),
            "--adaptive_normal_overbuild_penalty",
            str(args.self_refine_adaptive_normal_overbuild_penalty),
            "--candidate_growth_ratio_max",
            str(args.self_refine_candidate_growth_ratio_max),
            "--candidate_growth_ratio_underbuild_threshold",
            str(args.self_refine_candidate_growth_ratio_underbuild_threshold),
            "--candidate_growth_ratio_underbuild_max",
            str(args.self_refine_candidate_growth_ratio_underbuild_max),
            "--candidate_diversification_risk_threshold",
            str(args.self_refine_candidate_diversification_risk_threshold),
            "--candidate_diversification_underbuild_ratio_threshold",
            str(args.self_refine_candidate_diversification_underbuild_ratio_threshold),
            "--wall_balance_shell_min_deficit",
            str(wall_shell_min_deficit),
            "--wall_balance_shell_max_shape_drop_forecast",
            str(wall_shell_max_shape_drop_forecast),
            "--wall_balance_shell_shape_drop_scale",
            str(args.self_refine_wall_balance_shell_shape_drop_scale),
            "--max_shape_proxy_drop",
            str(args.self_refine_max_shape_proxy_drop),
            "--max_dim_score_drop",
            str(args.self_refine_max_dim_score_drop),
            "--max_footprint_profile_l1",
            str(args.self_refine_max_footprint_profile_l1),
            "--max_height_profile_l1",
            str(args.self_refine_max_height_profile_l1),
            "--two_stage_coarse_ready_threshold",
            str(args.self_refine_two_stage_coarse_ready_threshold),
            "--conditional_precboost_allow_keywords",
            str(args.self_refine_conditional_precboost_allow_keywords),
            "--conditional_precboost_block_keywords",
            str(args.self_refine_conditional_precboost_block_keywords),
            "--conditional_precboost_max_roof_score",
            str(args.self_refine_conditional_precboost_max_roof_score),
            "--conditional_precboost_min_material_score",
            str(args.self_refine_conditional_precboost_min_material_score),
            "--conditional_precboost_max_window_score",
            str(args.self_refine_conditional_precboost_max_window_score),
            "--conditional_precboost_min_raw_score_gain",
            str(args.self_refine_conditional_precboost_min_raw_score_gain),
            "--conditional_precboost_max_overbuild_excess",
            str(args.self_refine_conditional_precboost_max_overbuild_excess),
            "--conditional_precboost_max_underbuild_excess",
            str(args.self_refine_conditional_precboost_max_underbuild_excess),
            "--conditional_precboost_max_budget_violation_rel_increase",
            str(args.self_refine_conditional_precboost_max_budget_violation_rel_increase),
            "--precboost_selection_op_penalty",
            str(args.self_refine_precboost_selection_op_penalty),
            "--precboost_selection_overbuild_penalty",
            str(args.self_refine_precboost_selection_overbuild_penalty),
            "--precboost_selection_underbuild_penalty",
            str(args.self_refine_precboost_selection_underbuild_penalty),
            "--precboost_max_pred_target_ratio",
            str(args.self_refine_precboost_max_pred_target_ratio),
            "--precboost_adaptive_risk_ratio_threshold",
            str(args.self_refine_precboost_adaptive_risk_ratio_threshold),
            "--precboost_adaptive_high_risk_max_pred_target_ratio",
            str(args.self_refine_precboost_adaptive_high_risk_max_pred_target_ratio),
            "--precboost_adaptive_high_risk_overbuild_penalty",
            str(args.self_refine_precboost_adaptive_high_risk_overbuild_penalty),
            "--precboost_adaptive_normal_max_pred_target_ratio",
            str(args.self_refine_precboost_adaptive_normal_max_pred_target_ratio),
            "--precboost_adaptive_normal_overbuild_penalty",
            str(args.self_refine_precboost_adaptive_normal_overbuild_penalty),
            "--material_budget_tolerance",
            str(args.plan_material_budget_tolerance),
            "--role_fix_min_confidence",
            str(args.plan_role_fix_min_confidence),
            "--max_operations",
            str(args.plan_max_operations),
        ]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.plan_strict_schema:
            cmd.append("--strict_schema")
        else:
            cmd.append("--no_strict_schema")
        if args.plan_enforce_role_fixed:
            cmd.append("--enforce_role_fixed")
        else:
            cmd.append("--no_enforce_role_fixed")
        if args.plan_require_material_budget:
            cmd.append("--require_material_budget")
        else:
            cmd.append("--no_require_material_budget")
        if args.plan_prefer_description_palette:
            cmd.append("--prefer_description_palette")
        else:
            cmd.append("--no_prefer_description_palette")
        if args.self_refine_enable_material_budget_reprojection:
            cmd.append("--enable_material_budget_reprojection")
        else:
            cmd.append("--no_enable_material_budget_reprojection")
        if args.self_refine_enable_candidate_diversification:
            cmd.append("--enable_candidate_diversification")
        else:
            cmd.append("--no_enable_candidate_diversification")
        if args.self_refine_candidate_diversification_high_risk_only:
            cmd.append("--candidate_diversification_high_risk_only")
        else:
            cmd.append("--no_candidate_diversification_high_risk_only")
        if args.self_refine_wall_balance_shell_high_risk_only:
            cmd.append("--wall_balance_shell_high_risk_only")
        else:
            cmd.append("--no_wall_balance_shell_high_risk_only")
        if args.self_refine_enable_overbuild_guard:
            cmd.append("--enable_overbuild_guard")
        else:
            cmd.append("--no_enable_overbuild_guard")
        if args.self_refine_enable_adaptive_overbuild_control:
            cmd.append("--enable_adaptive_overbuild_control")
        else:
            cmd.append("--no_enable_adaptive_overbuild_control")
        if args.self_refine_enable_candidate_growth_guard:
            cmd.append("--enable_candidate_growth_guard")
        else:
            cmd.append("--no_enable_candidate_growth_guard")
        if args.self_refine_enable_shape_degradation_guard:
            cmd.append("--enable_shape_degradation_guard")
        else:
            cmd.append("--no_enable_shape_degradation_guard")
        if args.self_refine_enable_profile_match_guard:
            cmd.append("--enable_profile_match_guard")
        else:
            cmd.append("--no_enable_profile_match_guard")
        if args.self_refine_enforce_two_stage_generation:
            cmd.append("--enforce_two_stage_generation")
        else:
            cmd.append("--no_enforce_two_stage_generation")
        if args.self_refine_reject_strict_blocking_candidates:
            cmd.append("--reject_strict_blocking_candidates")
        else:
            cmd.append("--no_reject_strict_blocking_candidates")
        if args.self_refine_enable_conditional_precboost:
            cmd.append("--enable_conditional_precboost")
        else:
            cmd.append("--no_enable_conditional_precboost")
        if args.self_refine_conditional_precboost_require_keyword_match:
            cmd.append("--conditional_precboost_require_keyword_match")
        else:
            cmd.append("--no_conditional_precboost_require_keyword_match")
        run_cmd(cmd, cwd=root)
        effective_plan_subdir = self_refine_plan_subdir
        print(
            "[run_i2t2b_experiment] self_refine_no_gt enabled:"
            f" source_plan_subdir={plan_subdir}"
            f" refined_plan_subdir={effective_plan_subdir}"
        )

    if not args.skip_render:
        cmd = [
            py,
            str(root / "tools" / "render_rebuild_from_plan.py"),
            "--dataset_root",
            str(dataset_root),
            "--plan_subdir",
            effective_plan_subdir,
            "--out_subdir",
            rebuild_subdir,
        ]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        if args.overwrite:
            cmd.append("--overwrite")
        run_cmd(cmd, cwd=root)

    if not args.skip_description_eval:
        out_path = Path(desc_metrics_out)
        if not out_path.is_absolute():
            out_path = dataset_root / out_path
        cmd = [
            py,
            str(root / "tools" / "evaluate_description_quality.py"),
            "--dataset_root",
            str(dataset_root),
            "--description_subdir",
            desc_subdir,
            "--building_pattern",
            args.building_pattern,
            "--out",
            str(out_path),
        ]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        run_cmd(cmd, cwd=root)

    if not args.skip_rebuild_eval:
        out_path = Path(rebuild_metrics_out)
        if not out_path.is_absolute():
            out_path = dataset_root / out_path
        cmd = [
            py,
            str(root / "tools" / "evaluate_rebuild_metrics.py"),
            "--gt_root",
            str(dataset_root),
            "--pred_root",
            str(dataset_root),
            "--building_pattern",
            args.building_pattern,
            "--pred_source",
            "rebuild_world",
            "--pred_subdir",
            rebuild_subdir,
            "--out",
            str(out_path),
            "--fail_on_missing_pred",
        ]
        if args.thresholds_json:
            cmd += ["--thresholds_json", args.thresholds_json]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        run_cmd(cmd, cwd=root)

    print("[run_i2t2b_experiment] pipeline finished")


if __name__ == "__main__":
    main()
