#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.generate_rebuild_plans import (
    DEFAULT_ROLE_BLOCKS,
    REQUIRED_PALETTE_ROLES,
    _coerce_plan,
    _expand_roof_template,
    _expand_window_pattern,
    _infer_shape_preferences,
    _int,
    _normalize_block_type,
    _normalize_role,
    _validate_and_repair_plan,
)

NON_FATAL_STRICT_ISSUES = {"material_budget_violation"}
NON_FATAL_STRICT_PREFIXES = ("material_budget_violation",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-GT self-refinement pass after render simulation.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing building_xxx")
    parser.add_argument("--plan_subdir", required=True, help="Input rebuild plan subdir")
    parser.add_argument("--description_subdir", default="description", help="Description subdir")
    parser.add_argument("--out_plan_subdir", required=True, help="Output plan subdir")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern")
    parser.add_argument("--limit", type=int, default=0, help="Max building count (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--max_dim", type=int, default=192, help="Safety bound for voxel axis size")
    parser.add_argument("--max_iterations", type=int, default=2, help="Refinement iterations")
    parser.add_argument("--min_score_gain", type=float, default=0.01, help="Minimum score gain to accept")
    parser.add_argument("--max_added_ops_per_iter", type=int, default=24, help="Max correction ops per iteration")
    parser.add_argument(
        "--roof_search_variants",
        type=int,
        default=6,
        help="Number of roof-template variants to explore per iteration.",
    )
    parser.add_argument(
        "--window_search_variants",
        type=int,
        default=6,
        help="Number of window-pattern variants to explore per iteration.",
    )
    parser.add_argument(
        "--max_search_candidates",
        type=int,
        default=16,
        help="Maximum correction candidates evaluated per iteration.",
    )
    parser.add_argument(
        "--enable_candidate_diversification",
        dest="enable_candidate_diversification",
        action="store_true",
        help="Enable diversified candidate generation (single/pair/full bundles with base variants).",
    )
    parser.add_argument(
        "--no_enable_candidate_diversification",
        dest="enable_candidate_diversification",
        action="store_false",
        help="Disable candidate diversification and keep legacy full-bundle candidate generation.",
    )
    parser.add_argument(
        "--candidate_diversification_high_risk_only",
        dest="candidate_diversification_high_risk_only",
        action="store_true",
        help="Apply candidate diversification only when current plan state is high-risk for overbuild.",
    )
    parser.add_argument(
        "--no_candidate_diversification_high_risk_only",
        dest="candidate_diversification_high_risk_only",
        action="store_false",
        help="Apply candidate diversification regardless of risk when enabled.",
    )
    parser.add_argument(
        "--candidate_diversification_risk_threshold",
        type=float,
        default=-1.0,
        help="Risk-ratio threshold for high-risk diversification. <=0 uses adaptive_risk_ratio_threshold.",
    )
    parser.add_argument(
        "--candidate_diversification_underbuild_ratio_threshold",
        type=float,
        default=0.92,
        help="Enable diversification when pred/target ratio is below this threshold (underbuild-heavy case).",
    )
    parser.add_argument(
        "--wall_balance_shell_high_risk_only",
        dest="wall_balance_shell_high_risk_only",
        action="store_true",
        help="Apply self_refine wall shell balance only when current state is high-risk.",
    )
    parser.add_argument(
        "--no_wall_balance_shell_high_risk_only",
        dest="wall_balance_shell_high_risk_only",
        action="store_false",
        help="Apply self_refine wall shell balance regardless of risk.",
    )
    parser.add_argument(
        "--wall_balance_shell_min_deficit",
        type=float,
        default=0.06,
        help="Enable wall-shell balance only when wall role deficit is above this threshold.",
    )
    parser.add_argument(
        "--wall_balance_shell_max_shape_drop_forecast",
        type=float,
        default=0.06,
        help="Enable wall-shell balance only when forecasted shape-proxy drop is below this threshold.",
    )
    parser.add_argument(
        "--wall_balance_shell_shape_drop_scale",
        type=float,
        default=0.30,
        help="Scale factor for wall-shell shape-drop forecast from overbuild excess.",
    )
    parser.add_argument(
        "--material_budget_reprojection_strength",
        type=float,
        default=0.25,
        help="Strength of render-after material budget reprojection.",
    )
    parser.add_argument(
        "--material_budget_reprojection_min_deficit_ratio",
        type=float,
        default=0.03,
        help="Minimum deficit ratio to trigger budget reprojection for a role.",
    )
    parser.add_argument(
        "--material_budget_reprojection_trigger_material_score",
        type=float,
        default=0.65,
        help="Apply budget reprojection only when material component is below this score.",
    )
    parser.add_argument(
        "--selection_op_penalty",
        type=float,
        default=0.0015,
        help="Candidate selection penalty per added operation to avoid over-correction.",
    )
    parser.add_argument(
        "--selection_overbuild_penalty",
        type=float,
        default=0.35,
        help="Additional candidate penalty per (pred/target - 1.0) overbuild excess.",
    )
    parser.add_argument(
        "--selection_underbuild_penalty",
        type=float,
        default=0.0,
        help="Additional candidate penalty per (1.0 - pred/target) underbuild excess.",
    )
    parser.add_argument(
        "--selection_material_budget_violation_penalty",
        type=float,
        default=0.03,
        help="Penalty applied when strict issues include non-fatal material_budget_violation.",
    )
    parser.add_argument(
        "--selection_material_budget_count_weight",
        type=float,
        default=0.25,
        help="Count weight added per budget violation when computing continuous budget penalty.",
    )
    parser.add_argument(
        "--selection_ratio_target_penalty",
        type=float,
        default=0.18,
        help="Penalty on absolute |pred/target - 1| to keep block count near target.",
    )
    parser.add_argument(
        "--selection_shape_drop_penalty",
        type=float,
        default=0.35,
        help="Penalty applied to shape proxy drop from current plan state.",
    )
    parser.add_argument(
        "--selection_dim_drop_penalty",
        type=float,
        default=0.40,
        help="Penalty applied to dimension score drop from current plan state.",
    )
    parser.add_argument(
        "--selection_growth_excess_penalty",
        type=float,
        default=0.35,
        help="Penalty on per-iteration non-air growth beyond allowed growth ratio.",
    )
    parser.add_argument(
        "--selection_footprint_profile_penalty",
        type=float,
        default=0.15,
        help="Penalty on footprint profile mismatch against current plan state.",
    )
    parser.add_argument(
        "--selection_height_profile_penalty",
        type=float,
        default=0.20,
        help="Penalty on height profile mismatch against current plan state.",
    )
    parser.add_argument(
        "--max_pred_target_ratio",
        type=float,
        default=1.05,
        help="Reject candidate when rendered non-air exceeds this ratio vs target non-air.",
    )
    parser.add_argument(
        "--enable_material_budget_reprojection",
        dest="enable_material_budget_reprojection",
        action="store_true",
        help="Enable material budget reprojection using render feedback.",
    )
    parser.add_argument(
        "--no_enable_material_budget_reprojection",
        dest="enable_material_budget_reprojection",
        action="store_false",
        help="Disable material budget reprojection.",
    )
    parser.add_argument(
        "--enable_overbuild_guard",
        dest="enable_overbuild_guard",
        action="store_true",
        help="Enable hard guard that rejects overbuilt candidates by pred/target ratio.",
    )
    parser.add_argument(
        "--no_enable_overbuild_guard",
        dest="enable_overbuild_guard",
        action="store_false",
        help="Disable hard guard for overbuild ratio.",
    )
    parser.add_argument(
        "--enable_adaptive_overbuild_control",
        dest="enable_adaptive_overbuild_control",
        action="store_true",
        help="Adapt max_pred_target_ratio and overbuild penalty by predicted overbuild risk.",
    )
    parser.add_argument(
        "--no_enable_adaptive_overbuild_control",
        dest="enable_adaptive_overbuild_control",
        action="store_false",
        help="Disable adaptive overbuild control and use fixed ratio/penalty.",
    )
    parser.add_argument(
        "--adaptive_risk_ratio_threshold",
        type=float,
        default=1.25,
        help="Risk threshold on (pred_non_air/target_non_air) for high-risk overbuild control.",
    )
    parser.add_argument(
        "--adaptive_high_risk_max_pred_target_ratio",
        type=float,
        default=1.10,
        help="Applied max_pred_target_ratio when risk is high.",
    )
    parser.add_argument(
        "--adaptive_high_risk_overbuild_penalty",
        type=float,
        default=0.35,
        help="Applied overbuild penalty when risk is high.",
    )
    parser.add_argument(
        "--adaptive_normal_max_pred_target_ratio",
        type=float,
        default=1.20,
        help="Applied max_pred_target_ratio when risk is normal.",
    )
    parser.add_argument(
        "--adaptive_normal_overbuild_penalty",
        type=float,
        default=0.15,
        help="Applied overbuild penalty when risk is normal.",
    )
    parser.add_argument(
        "--enable_candidate_growth_guard",
        dest="enable_candidate_growth_guard",
        action="store_true",
        help="Reject candidates when per-iteration non-air growth ratio exceeds adaptive threshold.",
    )
    parser.add_argument(
        "--no_enable_candidate_growth_guard",
        dest="enable_candidate_growth_guard",
        action="store_false",
        help="Disable per-iteration growth guard.",
    )
    parser.add_argument(
        "--candidate_growth_ratio_max",
        type=float,
        default=1.18,
        help="Max candidate/base non-air ratio in normal mode.",
    )
    parser.add_argument(
        "--candidate_growth_ratio_underbuild_threshold",
        type=float,
        default=0.90,
        help="Treat current plan as underbuild-heavy when current pred/target is below this threshold.",
    )
    parser.add_argument(
        "--candidate_growth_ratio_underbuild_max",
        type=float,
        default=1.45,
        help="Max candidate/base non-air ratio when current plan is underbuild-heavy.",
    )
    parser.add_argument(
        "--enable_shape_degradation_guard",
        dest="enable_shape_degradation_guard",
        action="store_true",
        help="Reject candidates that degrade shape proxy beyond tolerance.",
    )
    parser.add_argument(
        "--no_enable_shape_degradation_guard",
        dest="enable_shape_degradation_guard",
        action="store_false",
        help="Disable shape degradation guard.",
    )
    parser.add_argument(
        "--max_shape_proxy_drop",
        type=float,
        default=0.03,
        help="Maximum allowed drop for shape proxy score from current plan state.",
    )
    parser.add_argument(
        "--max_dim_score_drop",
        type=float,
        default=0.06,
        help="Maximum allowed drop for dimension score from current plan state.",
    )
    parser.add_argument(
        "--enable_profile_match_guard",
        dest="enable_profile_match_guard",
        action="store_true",
        help="Reject candidates when footprint/height profile mismatch is too large.",
    )
    parser.add_argument(
        "--no_enable_profile_match_guard",
        dest="enable_profile_match_guard",
        action="store_false",
        help="Disable profile mismatch hard guard (penalty still applies).",
    )
    parser.add_argument(
        "--max_footprint_profile_l1",
        type=float,
        default=0.22,
        help="Maximum footprint profile L1 mismatch allowed by profile guard.",
    )
    parser.add_argument(
        "--max_height_profile_l1",
        type=float,
        default=0.25,
        help="Maximum height profile L1 mismatch allowed by profile guard.",
    )
    parser.add_argument(
        "--enforce_two_stage_generation",
        dest="enforce_two_stage_generation",
        action="store_true",
        help="Force coarse-shape-first candidate generation before decor candidates.",
    )
    parser.add_argument(
        "--no_enforce_two_stage_generation",
        dest="enforce_two_stage_generation",
        action="store_false",
        help="Disable coarse/decor staged candidate generation.",
    )
    parser.add_argument(
        "--two_stage_coarse_ready_threshold",
        type=float,
        default=0.70,
        help="Decor stage is enabled only after coarse stage score reaches this threshold.",
    )
    parser.add_argument(
        "--reject_strict_blocking_candidates",
        dest="reject_strict_blocking_candidates",
        action="store_true",
        help="Reject candidates when strict_blocking_issues is non-empty.",
    )
    parser.add_argument(
        "--no_reject_strict_blocking_candidates",
        dest="reject_strict_blocking_candidates",
        action="store_false",
        help="Allow candidates even with strict_blocking_issues.",
    )
    parser.add_argument(
        "--enable_conditional_precboost",
        dest="enable_conditional_precboost",
        action="store_true",
        help="Use tuned as default, and activate precboost profile only for eligible building types when it wins candidate-level checks.",
    )
    parser.add_argument(
        "--no_enable_conditional_precboost",
        dest="enable_conditional_precboost",
        action="store_false",
        help="Disable conditional precboost profile switching.",
    )
    parser.add_argument(
        "--conditional_precboost_require_keyword_match",
        dest="conditional_precboost_require_keyword_match",
        action="store_true",
        help="Require description keyword match before considering precboost profile.",
    )
    parser.add_argument(
        "--no_conditional_precboost_require_keyword_match",
        dest="conditional_precboost_require_keyword_match",
        action="store_false",
        help="Allow precboost profile even without keyword match (still blocked by blocked-keyword list and score gates).",
    )
    parser.add_argument(
        "--conditional_precboost_allow_keywords",
        type=str,
        default="bunker,storage,shed,house,residential,cottage",
        help="Comma-separated positive keywords for conditional precboost gating.",
    )
    parser.add_argument(
        "--conditional_precboost_block_keywords",
        type=str,
        default="monument,watchtower,fortification,shrine,decorative",
        help="Comma-separated blocked keywords for conditional precboost gating.",
    )
    parser.add_argument(
        "--conditional_precboost_max_roof_score",
        type=float,
        default=0.92,
        help="Eligible only when baseline roof component is <= this value.",
    )
    parser.add_argument(
        "--conditional_precboost_min_material_score",
        type=float,
        default=0.6,
        help="Eligible only when baseline material component is >= this value.",
    )
    parser.add_argument(
        "--conditional_precboost_max_window_score",
        type=float,
        default=1.0,
        help="Eligible only when baseline window component is <= this value.",
    )
    parser.add_argument(
        "--conditional_precboost_min_raw_score_gain",
        type=float,
        default=0.008,
        help="Switch to precboost only when best_precboost raw self-consistency score exceeds tuned by at least this amount.",
    )
    parser.add_argument(
        "--conditional_precboost_max_overbuild_excess",
        type=float,
        default=0.12,
        help="Switch to precboost only when selected candidate overbuild_excess is <= this value.",
    )
    parser.add_argument(
        "--conditional_precboost_max_underbuild_excess",
        type=float,
        default=0.35,
        help="Switch to precboost only when selected candidate underbuild_excess is <= this value.",
    )
    parser.add_argument(
        "--conditional_precboost_max_budget_violation_rel_increase",
        type=float,
        default=0.05,
        help="Allowed increase in budget_violation_rel_sum vs tuned when switching to precboost.",
    )
    parser.add_argument(
        "--precboost_selection_op_penalty",
        type=float,
        default=0.001,
        help="Precboost profile: operation penalty.",
    )
    parser.add_argument(
        "--precboost_selection_overbuild_penalty",
        type=float,
        default=0.25,
        help="Precboost profile: overbuild penalty.",
    )
    parser.add_argument(
        "--precboost_selection_underbuild_penalty",
        type=float,
        default=0.45,
        help="Precboost profile: underbuild penalty.",
    )
    parser.add_argument(
        "--precboost_max_pred_target_ratio",
        type=float,
        default=1.05,
        help="Precboost profile: hard max pred/target ratio.",
    )
    parser.add_argument(
        "--precboost_adaptive_risk_ratio_threshold",
        type=float,
        default=1.22,
        help="Precboost profile: adaptive overbuild high-risk threshold.",
    )
    parser.add_argument(
        "--precboost_adaptive_high_risk_max_pred_target_ratio",
        type=float,
        default=1.12,
        help="Precboost profile: high-risk max pred/target ratio.",
    )
    parser.add_argument(
        "--precboost_adaptive_high_risk_overbuild_penalty",
        type=float,
        default=0.30,
        help="Precboost profile: high-risk overbuild penalty.",
    )
    parser.add_argument(
        "--precboost_adaptive_normal_max_pred_target_ratio",
        type=float,
        default=1.22,
        help="Precboost profile: normal-risk max pred/target ratio.",
    )
    parser.add_argument(
        "--precboost_adaptive_normal_overbuild_penalty",
        type=float,
        default=0.12,
        help="Precboost profile: normal-risk overbuild penalty.",
    )

    parser.add_argument("--strict_schema", dest="strict_schema", action="store_true")
    parser.add_argument("--no_strict_schema", dest="strict_schema", action="store_false")
    parser.add_argument("--enforce_role_fixed", dest="enforce_role_fixed", action="store_true")
    parser.add_argument("--no_enforce_role_fixed", dest="enforce_role_fixed", action="store_false")
    parser.add_argument("--require_material_budget", dest="require_material_budget", action="store_true")
    parser.add_argument("--no_require_material_budget", dest="require_material_budget", action="store_false")
    parser.add_argument("--prefer_description_palette", dest="prefer_description_palette", action="store_true")
    parser.add_argument("--no_prefer_description_palette", dest="prefer_description_palette", action="store_false")
    parser.add_argument("--material_budget_tolerance", type=float, default=0.35)
    parser.add_argument("--role_fix_min_confidence", type=float, default=0.78)
    parser.add_argument("--max_operations", type=int, default=260)
    parser.add_argument(
        "--required_palette_roles",
        nargs="*",
        default=list(REQUIRED_PALETTE_ROLES),
        help="Required palette roles subset",
    )
    parser.set_defaults(
        strict_schema=True,
        enforce_role_fixed=True,
        require_material_budget=True,
        prefer_description_palette=True,
        enable_material_budget_reprojection=True,
        enable_overbuild_guard=True,
        enable_adaptive_overbuild_control=True,
        reject_strict_blocking_candidates=True,
        enable_shape_degradation_guard=True,
        enable_profile_match_guard=True,
        enforce_two_stage_generation=True,
        enable_candidate_diversification=False,
        candidate_diversification_high_risk_only=True,
        enable_conditional_precboost=True,
        conditional_precboost_require_keyword_match=True,
        enable_candidate_growth_guard=True,
        wall_balance_shell_high_risk_only=False,
    )
    return parser.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in dataset_root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _normalize_required_roles(raw_roles: List[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for role in raw_roles:
        nr = _normalize_role(role)
        if nr in REQUIRED_PALETTE_ROLES and nr not in out:
            out.append(nr)
    return tuple(out) if out else REQUIRED_PALETTE_ROLES


def _resolve_bbox(plan: Dict[str, Any]) -> Dict[str, int]:
    bbox_raw = plan.get("bbox", {}) if isinstance(plan.get("bbox"), dict) else {}
    bbox = {
        "xmin": _int(bbox_raw.get("xmin", 0), 0),
        "xmax": _int(bbox_raw.get("xmax", 15), 15),
        "ymin": _int(bbox_raw.get("ymin", 0), 0),
        "ymax": _int(bbox_raw.get("ymax", 12), 12),
        "zmin": _int(bbox_raw.get("zmin", 0), 0),
        "zmax": _int(bbox_raw.get("zmax", 15), 15),
    }
    if bbox["xmax"] < bbox["xmin"]:
        bbox["xmin"], bbox["xmax"] = bbox["xmax"], bbox["xmin"]
    if bbox["ymax"] < bbox["ymin"]:
        bbox["ymin"], bbox["ymax"] = bbox["ymax"], bbox["ymin"]
    if bbox["zmax"] < bbox["zmin"]:
        bbox["zmin"], bbox["zmax"] = bbox["zmax"], bbox["zmin"]

    for op in plan.get("operations", []) if isinstance(plan.get("operations"), list) else []:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op", "")).strip().lower()
        if kind in {"fill", "carve"}:
            x1 = _int(op.get("x1", 0), 0)
            y1 = _int(op.get("y1", 0), 0)
            z1 = _int(op.get("z1", 0), 0)
            x2 = _int(op.get("x2", x1), x1)
            y2 = _int(op.get("y2", y1), y1)
            z2 = _int(op.get("z2", z1), z1)
            bbox["xmin"] = min(bbox["xmin"], x1, x2)
            bbox["xmax"] = max(bbox["xmax"], x1, x2)
            bbox["ymin"] = min(bbox["ymin"], y1, y2)
            bbox["ymax"] = max(bbox["ymax"], y1, y2)
            bbox["zmin"] = min(bbox["zmin"], z1, z2)
            bbox["zmax"] = max(bbox["zmax"], z1, z2)
        elif kind == "set":
            x = _int(op.get("x", 0), 0)
            y = _int(op.get("y", 0), 0)
            z = _int(op.get("z", 0), 0)
            bbox["xmin"] = min(bbox["xmin"], x)
            bbox["xmax"] = max(bbox["xmax"], x)
            bbox["ymin"] = min(bbox["ymin"], y)
            bbox["ymax"] = max(bbox["ymax"], y)
            bbox["zmin"] = min(bbox["zmin"], z)
            bbox["zmax"] = max(bbox["zmax"], z)
    return bbox


def _apply_fill(vox: np.ndarray, bbox: Dict[str, int], x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, block: str) -> None:
    xmin, ymin, zmin = bbox["xmin"], bbox["ymin"], bbox["zmin"]
    ix1, ix2 = x1 - xmin, x2 - xmin
    iy1, iy2 = y1 - ymin, y2 - ymin
    iz1, iz2 = z1 - zmin, z2 - zmin
    vox[iy1 : iy2 + 1, ix1 : ix2 + 1, iz1 : iz2 + 1] = block


def _render_plan(plan: Dict[str, Any], *, max_dim: int) -> Tuple[np.ndarray, Dict[str, int]]:
    bbox = _resolve_bbox(plan)
    sx = bbox["xmax"] - bbox["xmin"] + 1
    sy = bbox["ymax"] - bbox["ymin"] + 1
    sz = bbox["zmax"] - bbox["zmin"] + 1
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise RuntimeError("Invalid bbox dimensions")
    if max(sx, sy, sz) > int(max_dim):
        raise RuntimeError(f"Bbox too large for self-refine render: {sx}x{sy}x{sz}")

    vox = np.full((sy, sx, sz), "air", dtype="<U40")
    ops = plan.get("operations", []) if isinstance(plan.get("operations"), list) else []
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op", "")).strip().lower()
        if kind == "fill":
            x1 = _int(op.get("x1", 0), 0)
            y1 = _int(op.get("y1", 0), 0)
            z1 = _int(op.get("z1", 0), 0)
            x2 = _int(op.get("x2", x1), x1)
            y2 = _int(op.get("y2", y1), y1)
            z2 = _int(op.get("z2", z1), z1)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if z2 < z1:
                z1, z2 = z2, z1
            block = _normalize_block_type(op.get("block", "stonebrick"))
            _apply_fill(vox, bbox, x1, y1, z1, x2, y2, z2, block)
        elif kind == "carve":
            x1 = _int(op.get("x1", 0), 0)
            y1 = _int(op.get("y1", 0), 0)
            z1 = _int(op.get("z1", 0), 0)
            x2 = _int(op.get("x2", x1), x1)
            y2 = _int(op.get("y2", y1), y1)
            z2 = _int(op.get("z2", z1), z1)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if z2 < z1:
                z1, z2 = z2, z1
            _apply_fill(vox, bbox, x1, y1, z1, x2, y2, z2, "air")
        elif kind == "set":
            x = _int(op.get("x", 0), 0)
            y = _int(op.get("y", 0), 0)
            z = _int(op.get("z", 0), 0)
            block = _normalize_block_type(op.get("block", "stonebrick"))
            _apply_fill(vox, bbox, x, y, z, x, y, z, block)
    return vox, bbox


def _role_block_map(palette: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for role in REQUIRED_PALETTE_ROLES:
        out[role] = _normalize_block_type(palette.get(role, DEFAULT_ROLE_BLOCKS[role]))
    return out


def _counts_by_role(vox: np.ndarray, role_blocks: Dict[str, str]) -> Dict[str, int]:
    counts = {r: 0 for r in REQUIRED_PALETTE_ROLES}
    # Priority prevents double counting when same block is reused across roles.
    priority = ("glass", "light", "roof", "floor", "wall", "trim")
    total = int(np.count_nonzero(vox != "air"))
    for role in priority:
        block = role_blocks[role]
        counts[role] = int(np.count_nonzero(vox == block))

    assigned = sum(counts.values())
    if assigned < total:
        # Remaining non-air goes to wall bucket.
        counts["wall"] += total - assigned
    return counts


def _target_role_counts(plan: Dict[str, Any], desc: Dict[str, Any], total_non_air: int) -> Dict[str, int]:
    budget = plan.get("material_budget", {})
    out = {r: 0 for r in REQUIRED_PALETTE_ROLES}
    if isinstance(budget, dict):
        found = False
        for role in REQUIRED_PALETTE_ROLES:
            item = budget.get(role, {})
            if not isinstance(item, dict):
                continue
            target = max(0, _int(item.get("target_blocks", 0), 0))
            if target > 0:
                found = True
            out[role] = target
        if found:
            return out

    # Fallback priors from description context.
    text = " ".join(
        str(x)
        for x in (
            desc.get("summary", ""),
            " ".join(desc.get("elements", [])) if isinstance(desc.get("elements"), list) else "",
            " ".join(desc.get("rebuild_hints", [])) if isinstance(desc.get("rebuild_hints"), list) else "",
        )
    ).lower()
    priors = {
        "wall": 0.45,
        "roof": 0.20,
        "trim": 0.10,
        "glass": 0.08,
        "light": 0.04,
        "floor": 0.13,
    }
    if "window" in text:
        priors["glass"] += 0.04
        priors["wall"] -= 0.02
        priors["trim"] -= 0.02
    if "tower" in text:
        priors["wall"] += 0.06
        priors["roof"] -= 0.03
        priors["floor"] -= 0.03
    s = max(1e-6, sum(max(0.0, v) for v in priors.values()))
    for role in REQUIRED_PALETTE_ROLES:
        out[role] = int(round(total_non_air * max(0.0, priors[role]) / s))
    return out


def _compute_dim_score(vox: np.ndarray, desc: Dict[str, Any]) -> float:
    dims = desc.get("dimensions_estimate", {}) if isinstance(desc.get("dimensions_estimate"), dict) else {}
    exp_w = max(1, _int(dims.get("width", vox.shape[1]), vox.shape[1]))
    exp_d = max(1, _int(dims.get("depth", vox.shape[2]), vox.shape[2]))
    exp_h = max(1, _int(dims.get("height", vox.shape[0]), vox.shape[0]))
    pred_h, pred_w, pred_d = vox.shape

    def axis_score(pred: int, exp: int) -> float:
        err = abs(pred - exp) / float(max(1, exp))
        return max(0.0, 1.0 - err)

    return (axis_score(pred_w, exp_w) + axis_score(pred_d, exp_d) + axis_score(pred_h, exp_h)) / 3.0


def _compute_window_score(vox: np.ndarray, role_blocks: Dict[str, str], desc_text: str) -> float:
    if "window" not in desc_text and "glass" not in desc_text:
        return 1.0
    sy, sx, sz = vox.shape
    glass = role_blocks["glass"]
    y1 = max(1, int(round(sy * 0.25)))
    y2 = min(sy - 2, int(round(sy * 0.75)))
    if y2 < y1:
        y1, y2 = 1, max(1, sy - 2)
    sub = vox[y1 : y2 + 1, :, :]
    boundary = np.zeros_like(sub, dtype=bool)
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    actual = int(np.count_nonzero((sub == glass) & boundary))
    perimeter = max(8, 2 * (sx + sz))
    expected = max(6, perimeter // 8)
    return min(1.0, actual / float(max(1, expected)))


def _compute_entrance_score(vox: np.ndarray, desc_text: str) -> float:
    if ("entrance" not in desc_text) and ("door" not in desc_text) and ("entry" not in desc_text):
        return 1.0
    sy, sx, sz = vox.shape
    y_bot = 1
    y_top = min(sy - 1, 3)
    if y_top <= y_bot:
        return 0.0

    openings = 0
    # North/south faces
    for x in range(1, sx - 1):
        if np.all(vox[y_bot:y_top, x, 0] == "air"):
            openings += 1
        if np.all(vox[y_bot:y_top, x, sz - 1] == "air"):
            openings += 1
    # West/east faces
    for z in range(1, sz - 1):
        if np.all(vox[y_bot:y_top, 0, z] == "air"):
            openings += 1
        if np.all(vox[y_bot:y_top, sx - 1, z] == "air"):
            openings += 1
    return 1.0 if openings >= 2 else (0.5 if openings == 1 else 0.0)


def _compute_roof_score(vox: np.ndarray, role_blocks: Dict[str, str], roof_type: str) -> float:
    sy = vox.shape[0]
    top_layers = min(3, sy)
    roof_block = role_blocks["roof"]
    top = vox[sy - top_layers : sy, :, :]
    top_non_air = int(np.count_nonzero(top != "air"))
    if top_non_air <= 0:
        return 0.0
    roof_ratio = int(np.count_nonzero(top == roof_block)) / float(top_non_air)
    if roof_type not in {"gable", "hip", "dome"}:
        return roof_ratio

    # Taper preference: upper occupied area should not grow.
    areas: List[int] = []
    for y in range(max(0, sy - 6), sy):
        areas.append(int(np.count_nonzero(vox[y, :, :] != "air")))
    violations = 0
    for i in range(1, len(areas)):
        if areas[i] > areas[i - 1]:
            violations += 1
    taper = 1.0 if len(areas) <= 1 else max(0.0, 1.0 - violations / float(len(areas) - 1))
    return 0.7 * roof_ratio + 0.3 * taper


def _compute_material_score(
    counts: Dict[str, int],
    target: Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    total_actual = float(max(1, sum(counts.values())))
    total_target = float(max(1, sum(target.values())))
    l1 = 0.0
    deficits: Dict[str, float] = {}
    for role in REQUIRED_PALETTE_ROLES:
        ar = counts[role] / total_actual
        tr = target[role] / total_target
        l1 += abs(ar - tr)
        deficits[role] = tr - ar
    score = max(0.0, 1.0 - 0.5 * l1)
    return score, deficits


def _target_non_air_from_metrics(metrics: Dict[str, Any], fallback_non_air: int) -> int:
    target = metrics.get("target_role_counts", {})
    if isinstance(target, dict):
        total = 0
        for role in REQUIRED_PALETTE_ROLES:
            total += max(0, _int(target.get(role, 0), 0))
        if total > 0:
            return int(total)
    return int(max(1, fallback_non_air))


def _pred_target_ratio(pred_non_air: int, target_non_air: int) -> float:
    return float(pred_non_air) / float(max(1, target_non_air))


def _shape_proxy(components: Dict[str, Any]) -> float:
    dim = float(components.get("dim", 0.0) or 0.0)
    roof = float(components.get("roof", 0.0) or 0.0)
    window = float(components.get("window", 0.0) or 0.0)
    entrance = float(components.get("entrance", 0.0) or 0.0)
    # Shape-focused proxy (ignores material component).
    return 0.40 * dim + 0.30 * roof + 0.20 * window + 0.10 * entrance


def _coarse_stage_score(components: Dict[str, Any]) -> float:
    dim = float(components.get("dim", 0.0) or 0.0)
    roof = float(components.get("roof", 0.0) or 0.0)
    entrance = float(components.get("entrance", 0.0) or 0.0)
    return 0.50 * dim + 0.35 * roof + 0.15 * entrance


def _resample_profile(vals: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((target_len,), dtype=np.float64)
    if arr.size == target_len:
        return arr
    if arr.size == 1:
        return np.full((target_len,), float(arr[0]), dtype=np.float64)
    src_x = np.linspace(0.0, 1.0, num=arr.size, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
    return np.interp(dst_x, src_x, arr)


def _profile_l1(a: List[float], b: List[float]) -> float:
    arr_a = np.asarray(a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(b, dtype=np.float64).reshape(-1)
    target_len = int(max(1, arr_a.size, arr_b.size))
    ra = _resample_profile(arr_a, target_len)
    rb = _resample_profile(arr_b, target_len)
    return float(np.mean(np.abs(ra - rb)))


def _spatial_profiles(vox: np.ndarray) -> Dict[str, Any]:
    occ = vox != "air"
    sy, sx, sz = occ.shape
    denom_layer = float(max(1, sx * sz))
    height_profile = occ.reshape(sy, -1).sum(axis=1).astype(np.float64) / denom_layer
    footprint = np.any(occ, axis=0).astype(np.float64)  # (sx, sz)
    footprint_x = np.mean(footprint, axis=1) if sz > 0 else np.zeros((sx,), dtype=np.float64)
    footprint_z = np.mean(footprint, axis=0) if sx > 0 else np.zeros((sz,), dtype=np.float64)
    return {
        "height": height_profile.tolist(),
        "footprint_x": footprint_x.tolist(),
        "footprint_z": footprint_z.tolist(),
    }


def _profile_mismatch(base_profiles: Dict[str, Any], cand_profiles: Dict[str, Any]) -> Tuple[float, float]:
    base_fx = base_profiles.get("footprint_x", [])
    base_fz = base_profiles.get("footprint_z", [])
    base_h = base_profiles.get("height", [])
    cand_fx = cand_profiles.get("footprint_x", [])
    cand_fz = cand_profiles.get("footprint_z", [])
    cand_h = cand_profiles.get("height", [])
    footprint_l1 = 0.5 * (_profile_l1(base_fx, cand_fx) + _profile_l1(base_fz, cand_fz))
    height_l1 = _profile_l1(base_h, cand_h)
    return float(footprint_l1), float(height_l1)


def _material_budget_violation_penalty(
    validation: Dict[str, Any],
    *,
    penalty_scale: float,
    count_weight: float,
) -> Tuple[float, int, float]:
    violations = validation.get("budget_violations", [])
    if not isinstance(violations, list) or not violations:
        return 0.0, 0, 0.0
    rel_sum = 0.0
    for item in violations:
        if not isinstance(item, dict):
            continue
        rel_sum += max(0.0, float(item.get("relative_error", 0.0) or 0.0))
    count = len(violations)
    penalty = float(penalty_scale) * (float(rel_sum) + float(count_weight) * float(count))
    return float(penalty), int(count), float(rel_sum)


def _effective_overbuild_control(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    fixed_ratio = float(args.max_pred_target_ratio)
    fixed_penalty = float(args.selection_overbuild_penalty)
    role_counts = metrics.get("role_counts", {})
    pred_non_air = 0
    if isinstance(role_counts, dict):
        for role in REQUIRED_PALETTE_ROLES:
            pred_non_air += max(0, _int(role_counts.get(role, 0), 0))
    pred_non_air = int(max(1, pred_non_air))
    target_non_air = _target_non_air_from_metrics(metrics, fallback_non_air=pred_non_air)
    risk_ratio = _pred_target_ratio(pred_non_air, target_non_air)

    if not bool(args.enable_adaptive_overbuild_control):
        return {
            "risk_ratio": float(risk_ratio),
            "risk_level": "fixed",
            "max_pred_target_ratio": float(fixed_ratio),
            "selection_overbuild_penalty": float(fixed_penalty),
        }

    threshold = float(args.adaptive_risk_ratio_threshold)
    if risk_ratio > threshold:
        return {
            "risk_ratio": float(risk_ratio),
            "risk_level": "high",
            "max_pred_target_ratio": float(args.adaptive_high_risk_max_pred_target_ratio),
            "selection_overbuild_penalty": float(args.adaptive_high_risk_overbuild_penalty),
        }
    return {
        "risk_ratio": float(risk_ratio),
        "risk_level": "normal",
        "max_pred_target_ratio": float(args.adaptive_normal_max_pred_target_ratio),
        "selection_overbuild_penalty": float(args.adaptive_normal_overbuild_penalty),
    }


def _effective_candidate_diversification(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    enabled_config = bool(args.enable_candidate_diversification)
    overbuild = _effective_overbuild_control(metrics, args)
    risk_level = str(overbuild.get("risk_level", "fixed"))
    risk_ratio = float(overbuild.get("risk_ratio", 1.0))
    underbuild_ratio = float(risk_ratio)
    underbuild_threshold = float(args.candidate_diversification_underbuild_ratio_threshold)
    underbuild_high = underbuild_ratio < underbuild_threshold
    threshold = float(args.candidate_diversification_risk_threshold)
    if threshold <= 0.0:
        threshold = float(args.adaptive_risk_ratio_threshold)
    overbuild_high = risk_level == "high" or risk_ratio > threshold
    high_risk = overbuild_high or underbuild_high
    if overbuild_high and underbuild_high:
        mode = "both"
    elif overbuild_high:
        mode = "overbuild"
    elif underbuild_high:
        mode = "underbuild"
    else:
        mode = "normal"
    if not bool(args.candidate_diversification_high_risk_only):
        return {
            "enabled": enabled_config,
            "high_risk": bool(high_risk),
            "risk_level": risk_level,
            "risk_ratio": float(risk_ratio),
            "risk_threshold": float(threshold),
            "underbuild_high": bool(underbuild_high),
            "underbuild_ratio": float(underbuild_ratio),
            "underbuild_threshold": float(underbuild_threshold),
            "mode": mode,
        }
    return {
        "enabled": bool(enabled_config and high_risk),
        "high_risk": bool(high_risk),
        "risk_level": risk_level,
        "risk_ratio": float(risk_ratio),
        "risk_threshold": float(threshold),
        "underbuild_high": bool(underbuild_high),
        "underbuild_ratio": float(underbuild_ratio),
        "underbuild_threshold": float(underbuild_threshold),
        "mode": mode,
    }


def _wall_shell_activation_state(
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    diversification_control: Dict[str, Any],
) -> Dict[str, Any]:
    deficits = metrics.get("role_deficits", {})
    if not isinstance(deficits, dict):
        deficits = {}
    wall_deficit = float(deficits.get("wall", 0.0) or 0.0)

    components = metrics.get("components", {})
    if not isinstance(components, dict):
        components = {}
    shape_proxy = float(_shape_proxy(components))

    bbox = metrics.get("bbox", {})
    if not isinstance(bbox, dict):
        bbox = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}
    x1 = int(bbox.get("xmin", 0))
    x2 = int(bbox.get("xmax", 0))
    y1 = int(bbox.get("ymin", 0)) + 1
    y2 = max(int(bbox.get("ymin", 0)) + 3, int(bbox.get("ymax", 0)) - 2)
    z1 = int(bbox.get("zmin", 0))
    z2 = int(bbox.get("zmax", 0))
    width = max(0, x2 - x1 + 1)
    depth = max(0, z2 - z1 + 1)
    height = max(0, y2 - y1 + 1)
    shell_cells_est = max(0, (2 * (width + depth) - 4) * height)

    role_counts = metrics.get("role_counts", {})
    if not isinstance(role_counts, dict):
        role_counts = {}
    pred_non_air = 0
    for role in REQUIRED_PALETTE_ROLES:
        pred_non_air += max(0, _int(role_counts.get(role, 0), 0))
    pred_non_air = int(max(1, pred_non_air))
    target_non_air = _target_non_air_from_metrics(metrics, fallback_non_air=pred_non_air)
    current_ratio = _pred_target_ratio(pred_non_air, target_non_air)

    min_deficit = float(args.wall_balance_shell_min_deficit)
    deficit_scale_denom = max(1e-6, min_deficit * 2.5)
    deficit_scale = min(1.0, max(0.0, wall_deficit / deficit_scale_denom))
    shell_add_est = int(round(float(shell_cells_est) * float(deficit_scale)))
    forecast_ratio = _pred_target_ratio(pred_non_air + shell_add_est, target_non_air)
    overbuild_excess_forecast = max(0.0, forecast_ratio - 1.0)
    shape_drop_forecast = (
        overbuild_excess_forecast
        * float(args.wall_balance_shell_shape_drop_scale)
        * (0.6 + 0.4 * float(shape_proxy))
    )

    deficit_ok = wall_deficit >= min_deficit
    shape_ok = shape_drop_forecast <= float(args.wall_balance_shell_max_shape_drop_forecast)
    high_risk = bool(diversification_control.get("high_risk", False))
    high_risk_gate = bool(args.wall_balance_shell_high_risk_only)
    risk_ok = (not high_risk_gate) or high_risk
    enabled = bool(deficit_ok and shape_ok and risk_ok)

    if not deficit_ok:
        reason = "wall_deficit_below_threshold"
    elif not shape_ok:
        reason = "shape_drop_forecast_too_high"
    elif not risk_ok:
        reason = "risk_gate_not_met"
    else:
        reason = "enabled"

    return {
        "enabled": bool(enabled),
        "reason": str(reason),
        "wall_deficit": float(wall_deficit),
        "min_deficit": float(min_deficit),
        "shape_proxy": float(shape_proxy),
        "shape_drop_forecast": float(shape_drop_forecast),
        "max_shape_drop_forecast": float(args.wall_balance_shell_max_shape_drop_forecast),
        "shape_drop_scale": float(args.wall_balance_shell_shape_drop_scale),
        "pred_non_air": int(pred_non_air),
        "target_non_air": int(target_non_air),
        "pred_target_ratio": float(current_ratio),
        "forecast_pred_target_ratio": float(forecast_ratio),
        "shell_cells_est": int(shell_cells_est),
        "shell_add_est": int(shell_add_est),
        "deficit_scale": float(deficit_scale),
        "high_risk": bool(high_risk),
        "high_risk_gate": bool(high_risk_gate),
    }


def _effective_growth_control(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    role_counts = metrics.get("role_counts", {})
    base_pred_non_air = 0
    if isinstance(role_counts, dict):
        for role in REQUIRED_PALETTE_ROLES:
            base_pred_non_air += max(0, _int(role_counts.get(role, 0), 0))
    base_pred_non_air = int(max(1, base_pred_non_air))
    base_target_non_air = _target_non_air_from_metrics(metrics, fallback_non_air=base_pred_non_air)
    base_pred_target_ratio = _pred_target_ratio(base_pred_non_air, base_target_non_air)
    underbuild_threshold = float(args.candidate_growth_ratio_underbuild_threshold)
    if base_pred_target_ratio < underbuild_threshold:
        mode = "underbuild"
        max_growth_ratio = float(args.candidate_growth_ratio_underbuild_max)
    else:
        mode = "normal"
        max_growth_ratio = float(args.candidate_growth_ratio_max)
    return {
        "enabled": bool(args.enable_candidate_growth_guard),
        "mode": mode,
        "base_pred_non_air": int(base_pred_non_air),
        "base_target_non_air": int(base_target_non_air),
        "base_pred_target_ratio": float(base_pred_target_ratio),
        "underbuild_threshold": float(underbuild_threshold),
        "max_growth_ratio": float(max_growth_ratio),
    }


def _split_strict_issues(issues: Any) -> Tuple[List[str], List[str]]:
    fatal: List[str] = []
    non_fatal: List[str] = []
    if not isinstance(issues, list):
        return fatal, non_fatal
    for issue in issues:
        tag = str(issue).strip()
        if not tag:
            continue
        if tag in NON_FATAL_STRICT_ISSUES or any(tag.startswith(prefix) for prefix in NON_FATAL_STRICT_PREFIXES):
            non_fatal.append(tag)
        else:
            fatal.append(tag)
    return fatal, non_fatal


def _parse_csv_keywords(raw: str) -> List[str]:
    parts = []
    for token in str(raw or "").split(","):
        t = token.strip().lower()
        if t:
            parts.append(t)
    return parts


def _description_text(desc: Dict[str, Any]) -> str:
    if not isinstance(desc, dict):
        return ""
    chunks: List[str] = []
    for key in ("building_type", "summary", "shape"):
        value = desc.get(key)
        if isinstance(value, str):
            chunks.append(value)
    elems = desc.get("elements", [])
    if isinstance(elems, list):
        chunks.extend(str(x) for x in elems[:8])
    text = " ".join(chunks).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _conditional_precboost_eligibility(
    *,
    desc: Dict[str, Any],
    base_metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not bool(args.enable_conditional_precboost):
        return {"enabled": False, "eligible": False, "reason": "disabled"}
    components = base_metrics.get("components", {})
    if not isinstance(components, dict):
        components = {}
    roof = float(components.get("roof", 1.0) or 1.0)
    material = float(components.get("material", 0.0) or 0.0)
    window = float(components.get("window", 0.0) or 0.0)
    text = _description_text(desc)
    allow = _parse_csv_keywords(args.conditional_precboost_allow_keywords)
    block = _parse_csv_keywords(args.conditional_precboost_block_keywords)
    allow_hits = [k for k in allow if k in text]
    block_hits = [k for k in block if k in text]

    if block_hits:
        return {
            "enabled": True,
            "eligible": False,
            "reason": "blocked_keyword",
            "allow_hits": allow_hits,
            "block_hits": block_hits,
            "roof_score": roof,
            "material_score": material,
            "window_score": window,
        }
    if bool(args.conditional_precboost_require_keyword_match) and not allow_hits:
        return {
            "enabled": True,
            "eligible": False,
            "reason": "no_allow_keyword",
            "allow_hits": allow_hits,
            "block_hits": block_hits,
            "roof_score": roof,
            "material_score": material,
            "window_score": window,
        }
    if roof > float(args.conditional_precboost_max_roof_score):
        return {
            "enabled": True,
            "eligible": False,
            "reason": "roof_score_too_high",
            "allow_hits": allow_hits,
            "block_hits": block_hits,
            "roof_score": roof,
            "material_score": material,
            "window_score": window,
        }
    if material < float(args.conditional_precboost_min_material_score):
        return {
            "enabled": True,
            "eligible": False,
            "reason": "material_score_too_low",
            "allow_hits": allow_hits,
            "block_hits": block_hits,
            "roof_score": roof,
            "material_score": material,
            "window_score": window,
        }
    if window > float(args.conditional_precboost_max_window_score):
        return {
            "enabled": True,
            "eligible": False,
            "reason": "window_score_too_high",
            "allow_hits": allow_hits,
            "block_hits": block_hits,
            "roof_score": roof,
            "material_score": material,
            "window_score": window,
        }
    return {
        "enabled": True,
        "eligible": True,
        "reason": "eligible",
        "allow_hits": allow_hits,
        "block_hits": block_hits,
        "roof_score": roof,
        "material_score": material,
        "window_score": window,
    }


def _selection_profile_from_args(args: argparse.Namespace, name: str) -> Dict[str, Any]:
    key = str(name or "").strip().lower()
    if key == "precboost":
        return {
            "name": "precboost",
            "selection_op_penalty": float(args.precboost_selection_op_penalty),
            "selection_underbuild_penalty": float(args.precboost_selection_underbuild_penalty),
            "max_pred_target_ratio": float(args.precboost_max_pred_target_ratio),
            "selection_overbuild_penalty": float(args.precboost_selection_overbuild_penalty),
            "enable_overbuild_guard": bool(args.enable_overbuild_guard),
            "enable_adaptive_overbuild_control": bool(args.enable_adaptive_overbuild_control),
            "adaptive_risk_ratio_threshold": float(args.precboost_adaptive_risk_ratio_threshold),
            "adaptive_high_risk_max_pred_target_ratio": float(args.precboost_adaptive_high_risk_max_pred_target_ratio),
            "adaptive_high_risk_overbuild_penalty": float(args.precboost_adaptive_high_risk_overbuild_penalty),
            "adaptive_normal_max_pred_target_ratio": float(args.precboost_adaptive_normal_max_pred_target_ratio),
            "adaptive_normal_overbuild_penalty": float(args.precboost_adaptive_normal_overbuild_penalty),
        }
    return {
        "name": "tuned",
        "selection_op_penalty": float(args.selection_op_penalty),
        "selection_underbuild_penalty": float(args.selection_underbuild_penalty),
        "max_pred_target_ratio": float(args.max_pred_target_ratio),
        "selection_overbuild_penalty": float(args.selection_overbuild_penalty),
        "enable_overbuild_guard": bool(args.enable_overbuild_guard),
        "enable_adaptive_overbuild_control": bool(args.enable_adaptive_overbuild_control),
        "adaptive_risk_ratio_threshold": float(args.adaptive_risk_ratio_threshold),
        "adaptive_high_risk_max_pred_target_ratio": float(args.adaptive_high_risk_max_pred_target_ratio),
        "adaptive_high_risk_overbuild_penalty": float(args.adaptive_high_risk_overbuild_penalty),
        "adaptive_normal_max_pred_target_ratio": float(args.adaptive_normal_max_pred_target_ratio),
        "adaptive_normal_overbuild_penalty": float(args.adaptive_normal_overbuild_penalty),
    }


def _effective_overbuild_control_profile(metrics: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    fixed_ratio = float(profile.get("max_pred_target_ratio", 1.05))
    fixed_penalty = float(profile.get("selection_overbuild_penalty", 0.35))
    role_counts = metrics.get("role_counts", {})
    pred_non_air = 0
    if isinstance(role_counts, dict):
        for role in REQUIRED_PALETTE_ROLES:
            pred_non_air += max(0, _int(role_counts.get(role, 0), 0))
    pred_non_air = int(max(1, pred_non_air))
    target_non_air = _target_non_air_from_metrics(metrics, fallback_non_air=pred_non_air)
    risk_ratio = _pred_target_ratio(pred_non_air, target_non_air)

    if not bool(profile.get("enable_adaptive_overbuild_control", True)):
        return {
            "risk_ratio": float(risk_ratio),
            "risk_level": "fixed",
            "max_pred_target_ratio": float(fixed_ratio),
            "selection_overbuild_penalty": float(fixed_penalty),
        }

    threshold = float(profile.get("adaptive_risk_ratio_threshold", 1.25))
    if risk_ratio > threshold:
        return {
            "risk_ratio": float(risk_ratio),
            "risk_level": "high",
            "max_pred_target_ratio": float(profile.get("adaptive_high_risk_max_pred_target_ratio", fixed_ratio)),
            "selection_overbuild_penalty": float(profile.get("adaptive_high_risk_overbuild_penalty", fixed_penalty)),
        }
    return {
        "risk_ratio": float(risk_ratio),
        "risk_level": "normal",
        "max_pred_target_ratio": float(profile.get("adaptive_normal_max_pred_target_ratio", fixed_ratio)),
        "selection_overbuild_penalty": float(profile.get("adaptive_normal_overbuild_penalty", fixed_penalty)),
    }


def _select_profile_result(
    *,
    tuned_result: Optional[Dict[str, Any]],
    precboost_result: Optional[Dict[str, Any]],
    conditional_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    debug = {
        "conditional_enabled": bool(args.enable_conditional_precboost),
        "conditional_eligible": bool(conditional_meta.get("eligible", False)),
        "eligibility_reason": str(conditional_meta.get("reason", "")),
    }
    if tuned_result is None and precboost_result is None:
        debug["decision"] = "none"
        return "none", None, debug
    if tuned_result is None:
        debug["decision"] = "precboost_no_tuned_candidate"
        return "precboost", precboost_result, debug
    if precboost_result is None:
        debug["decision"] = "tuned_no_precboost_candidate"
        return "tuned", tuned_result, debug
    if not bool(args.enable_conditional_precboost) or not bool(conditional_meta.get("eligible", False)):
        debug["decision"] = "tuned_not_eligible"
        return "tuned", tuned_result, debug

    tuned_raw = float(tuned_result.get("metrics", {}).get("score", 0.0))
    pb_raw = float(precboost_result.get("metrics", {}).get("score", 0.0))
    raw_gain = pb_raw - tuned_raw
    pb_over = float(precboost_result.get("overbuild_excess", 0.0))
    pb_under = float(precboost_result.get("underbuild_excess", 0.0))
    tuned_budget_rel = float(tuned_result.get("budget_violation_rel_sum", 0.0))
    pb_budget_rel = float(precboost_result.get("budget_violation_rel_sum", 0.0))
    budget_rel_increase = pb_budget_rel - tuned_budget_rel

    debug.update(
        {
            "tuned_raw_score": tuned_raw,
            "precboost_raw_score": pb_raw,
            "raw_score_gain": raw_gain,
            "precboost_overbuild_excess": pb_over,
            "precboost_underbuild_excess": pb_under,
            "tuned_budget_violation_rel_sum": tuned_budget_rel,
            "precboost_budget_violation_rel_sum": pb_budget_rel,
            "budget_violation_rel_increase": budget_rel_increase,
        }
    )

    if raw_gain < float(args.conditional_precboost_min_raw_score_gain):
        debug["decision"] = "tuned_raw_gain_too_small"
        return "tuned", tuned_result, debug
    if pb_over > float(args.conditional_precboost_max_overbuild_excess):
        debug["decision"] = "tuned_precboost_overbuild_excess"
        return "tuned", tuned_result, debug
    if pb_under > float(args.conditional_precboost_max_underbuild_excess):
        debug["decision"] = "tuned_precboost_underbuild_excess"
        return "tuned", tuned_result, debug
    if budget_rel_increase > float(args.conditional_precboost_max_budget_violation_rel_increase):
        debug["decision"] = "tuned_precboost_budget_increase"
        return "tuned", tuned_result, debug
    debug["decision"] = "precboost_selected"
    return "precboost", precboost_result, debug


def _self_consistency_score(
    plan: Dict[str, Any],
    vox: np.ndarray,
    bbox: Dict[str, int],
    desc: Dict[str, Any],
) -> Dict[str, Any]:
    palette = plan.get("palette", {}) if isinstance(plan.get("palette"), dict) else {}
    role_blocks = _role_block_map(palette)
    counts = _counts_by_role(vox, role_blocks)
    non_air = int(np.count_nonzero(vox != "air"))
    target = _target_role_counts(plan, desc, non_air)
    material_score, deficits = _compute_material_score(counts, target)
    prefs = _infer_shape_preferences(desc if isinstance(desc, dict) else {})
    desc_text = str(prefs.get("text", ""))

    dim_score = _compute_dim_score(vox, desc)
    window_score = _compute_window_score(vox, role_blocks, desc_text)
    entrance_score = _compute_entrance_score(vox, desc_text)
    roof_score = _compute_roof_score(vox, role_blocks, str(prefs.get("roof_type", "flat")))

    weights = {
        "material": 0.35,
        "dim": 0.20,
        "roof": 0.20,
        "window": 0.15,
        "entrance": 0.10,
    }
    score = (
        weights["material"] * material_score
        + weights["dim"] * dim_score
        + weights["roof"] * roof_score
        + weights["window"] * window_score
        + weights["entrance"] * entrance_score
    )
    return {
        "score": float(score),
        "components": {
            "material": float(material_score),
            "dim": float(dim_score),
            "roof": float(roof_score),
            "window": float(window_score),
            "entrance": float(entrance_score),
        },
        "role_counts": counts,
        "target_role_counts": target,
        "role_deficits": deficits,
        "bbox": bbox,
        "role_blocks": role_blocks,
        "shape_preferences": prefs,
    }


def _add_op_if_room(ops: List[Dict[str, Any]], op: Dict[str, Any], max_operations: int) -> bool:
    if len(ops) >= max_operations:
        return False
    ops.append(op)
    return True


def _tag_ops(ops: List[Dict[str, Any]], *, role: str, reason: str, confidence: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        fixed = dict(op)
        kind = str(fixed.get("op", "")).strip().lower()
        if kind != "carve":
            fixed["role"] = role
            fixed["role_confidence"] = float(confidence)
            fixed["role_reason"] = reason
        out.append(fixed)
    return out


def _trim_perimeter_ops(bbox: Dict[str, int], block: str, band_y: int, purpose: str) -> List[Dict[str, Any]]:
    x1, x2 = bbox["xmin"], bbox["xmax"]
    z1, z2 = bbox["zmin"], bbox["zmax"]
    if x2 <= x1 or z2 <= z1:
        return []
    return [
        {"op": "fill", "x1": x1, "y1": band_y, "z1": z1, "x2": x2, "y2": band_y, "z2": z1, "block": block, "purpose": purpose},
        {"op": "fill", "x1": x1, "y1": band_y, "z1": z2, "x2": x2, "y2": band_y, "z2": z2, "block": block, "purpose": purpose},
        {"op": "fill", "x1": x1, "y1": band_y, "z1": z1, "x2": x1, "y2": band_y, "z2": z2, "block": block, "purpose": purpose},
        {"op": "fill", "x1": x2, "y1": band_y, "z1": z1, "x2": x2, "y2": band_y, "z2": z2, "block": block, "purpose": purpose},
    ]


def _wall_shell_ops(bbox: Dict[str, int], block: str, y1: int, y2: int, purpose: str) -> List[Dict[str, Any]]:
    x1, x2 = bbox["xmin"], bbox["xmax"]
    z1, z2 = bbox["zmin"], bbox["zmax"]
    if x2 <= x1 or z2 <= z1 or y2 < y1:
        return []
    return [
        {"op": "fill", "x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z1, "block": block, "purpose": f"{purpose}_north"},
        {"op": "fill", "x1": x1, "y1": y1, "z1": z2, "x2": x2, "y2": y2, "z2": z2, "block": block, "purpose": f"{purpose}_south"},
        {"op": "fill", "x1": x1, "y1": y1, "z1": z1, "x2": x1, "y2": y2, "z2": z2, "block": block, "purpose": f"{purpose}_west"},
        {"op": "fill", "x1": x2, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2, "block": block, "purpose": f"{purpose}_east"},
    ]


def _op_dedupe_key(op: Dict[str, Any]) -> Tuple[Any, ...]:
    kind = str(op.get("op", "")).strip().lower()
    if kind == "set":
        return (
            kind,
            int(_int(op.get("x", 0), 0)),
            int(_int(op.get("y", 0), 0)),
            int(_int(op.get("z", 0), 0)),
            _normalize_block_type(op.get("block", "")),
            _normalize_role(op.get("role", "")),
        )
    if kind in {"fill", "carve", "replace", "outline"}:
        x1 = _int(op.get("x1", 0), 0)
        y1 = _int(op.get("y1", 0), 0)
        z1 = _int(op.get("z1", 0), 0)
        x2 = _int(op.get("x2", x1), x1)
        y2 = _int(op.get("y2", y1), y1)
        z2 = _int(op.get("z2", z1), z1)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if z2 < z1:
            z1, z2 = z2, z1
        return (
            kind,
            int(x1),
            int(y1),
            int(z1),
            int(x2),
            int(y2),
            int(z2),
            _normalize_block_type(op.get("block", "")),
            _normalize_role(op.get("role", "")),
        )
    return (kind, json.dumps(op, sort_keys=True, ensure_ascii=True, separators=(",", ":")))


def _is_decor_op(op: Dict[str, Any]) -> bool:
    role = _normalize_role(op.get("role", ""))
    if role in {"trim", "glass", "light"}:
        return True
    purpose = str(op.get("purpose", "")).strip().lower()
    return (
        "window" in purpose
        or "glass" in purpose
        or "trim" in purpose
        or "light" in purpose
        or "decor" in purpose
    )


def _looks_like_wall_fill(op: Dict[str, Any]) -> bool:
    kind = str(op.get("op", "")).strip().lower()
    if kind != "fill":
        return False
    role = _normalize_role(op.get("role", ""))
    if role == "wall":
        return True
    purpose = str(op.get("purpose", "")).strip().lower()
    if any(k in purpose for k in ("wall", "shell", "facade", "outer_shell")):
        return True
    return False


def _enforce_wall_fill_shell_policy(ops: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    converted = 0
    preserved = 0
    for op in ops:
        if not isinstance(op, dict):
            continue
        if not _looks_like_wall_fill(op):
            out.append(dict(op))
            preserved += 1
            continue
        x1 = _int(op.get("x1", 0), 0)
        y1 = _int(op.get("y1", 0), 0)
        z1 = _int(op.get("z1", 0), 0)
        x2 = _int(op.get("x2", x1), x1)
        y2 = _int(op.get("y2", y1), y1)
        z2 = _int(op.get("z2", z1), z1)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if z2 < z1:
            z1, z2 = z2, z1
        sx = x2 - x1 + 1
        sy = y2 - y1 + 1
        sz = z2 - z1 + 1
        # Keep already-face-like wall fills untouched.
        if sx <= 1 or sz <= 1 or sy <= 1:
            fixed = dict(op)
            fixed["role"] = "wall"
            out.append(fixed)
            preserved += 1
            continue
        block = _normalize_block_type(op.get("block", "stonebrick"))
        base_purpose = str(op.get("purpose", "")).strip() or "wall_shell_enforced"
        faces = [
            (x1, y1, z1, x2, y2, z1, "north"),
            (x1, y1, z2, x2, y2, z2, "south"),
            (x1, y1, z1, x1, y2, z2, "west"),
            (x2, y1, z1, x2, y2, z2, "east"),
        ]
        for fx1, fy1, fz1, fx2, fy2, fz2, face in faces:
            out.append(
                {
                    "op": "fill",
                    "x1": int(fx1),
                    "y1": int(fy1),
                    "z1": int(fz1),
                    "x2": int(fx2),
                    "y2": int(fy2),
                    "z2": int(fz2),
                    "block": block,
                    "role": "wall",
                    "role_confidence": float(op.get("role_confidence", 0.85) or 0.85),
                    "role_reason": str(op.get("role_reason", "wall_shell_enforced")),
                    "purpose": f"{base_purpose}_face_{face}",
                }
            )
        converted += 1

    deduped: List[Dict[str, Any]] = []
    seen = set()
    dropped_dupes = 0
    for op in out:
        key = _op_dedupe_key(op)
        if key in seen:
            dropped_dupes += 1
            continue
        seen.add(key)
        deduped.append(op)
    return deduped, {
        "converted_wall_fills": int(converted),
        "preserved_ops": int(preserved),
        "dedup_dropped": int(dropped_dupes),
    }


def _propose_base_balance_ops(
    metrics: Dict[str, Any],
    *,
    max_operations: int,
    enable_wall_shell_balance: bool,
    wall_min_deficit: float,
) -> List[Dict[str, Any]]:
    bbox = metrics["bbox"]
    role_blocks = metrics["role_blocks"]
    comps = metrics["components"]
    deficits = metrics["role_deficits"]
    ops: List[Dict[str, Any]] = []

    if comps["entrance"] < 0.65:
        cx = (bbox["xmin"] + bbox["xmax"]) // 2
        _add_op_if_room(
            ops,
            {
                "op": "carve",
                "x1": cx,
                "y1": bbox["ymin"] + 1,
                "z1": bbox["zmin"],
                "x2": cx,
                "y2": min(bbox["ymax"] - 1, bbox["ymin"] + 3),
                "z2": bbox["zmin"],
                "block": "air",
                "purpose": "self_refine_entrance",
            },
            max_operations,
        )

    if enable_wall_shell_balance and deficits.get("wall", 0.0) > float(wall_min_deficit):
        shell_y1 = bbox["ymin"] + 1
        shell_y2 = max(bbox["ymin"] + 3, bbox["ymax"] - 2)
        shell_ops = _wall_shell_ops(
            bbox,
            role_blocks["wall"],
            shell_y1,
            shell_y2,
            "self_refine_wall_balance",
        )
        for op in _tag_ops(shell_ops, role="wall", reason="self_refine_wall_balance", confidence=0.85):
            if not _add_op_if_room(ops, op, max_operations):
                break

    if deficits.get("floor", 0.0) > 0.04:
        _add_op_if_room(
            ops,
            {
                "op": "fill",
                "x1": bbox["xmin"],
                "y1": bbox["ymin"],
                "z1": bbox["zmin"],
                "x2": bbox["xmax"],
                "y2": bbox["ymin"],
                "z2": bbox["zmax"],
                "block": role_blocks["floor"],
                "role": "floor",
                "role_confidence": 0.88,
                "role_reason": "self_refine_floor_balance",
                "purpose": "self_refine_floor_balance",
            },
            max_operations,
        )

    if deficits.get("trim", 0.0) > 0.03:
        band_y = max(bbox["ymin"] + 2, min(bbox["ymax"] - 1, bbox["ymax"] - 3))
        trim_ops = _trim_perimeter_ops(bbox, role_blocks["trim"], band_y, "self_refine_trim_perimeter")
        for op in _tag_ops(trim_ops, role="trim", reason="self_refine_trim_balance", confidence=0.84):
            if not _add_op_if_room(ops, op, max_operations):
                break

    if deficits.get("light", 0.0) > 0.02:
        for x, z in (
            (bbox["xmin"], bbox["zmin"]),
            (bbox["xmax"], bbox["zmin"]),
            (bbox["xmin"], bbox["zmax"]),
            (bbox["xmax"], bbox["zmax"]),
        ):
            light = {
                "op": "set",
                "x": x,
                "y": min(bbox["ymax"], bbox["ymin"] + 2),
                "z": z,
                "block": role_blocks["light"],
                "role": "light",
                "role_confidence": 0.9,
                "role_reason": "self_refine_light_balance",
                "purpose": "self_refine_corner_light",
            }
            if not _add_op_if_room(ops, light, max_operations):
                break
    return ops


def _roof_variant_specs(
    bbox: Dict[str, int],
    prefs: Dict[str, Any],
    *,
    max_variants: int,
) -> List[Dict[str, Any]]:
    pref_type = str(prefs.get("roof_type", "flat")).strip().lower() or "flat"
    roof_types = [pref_type, "gable", "hip", "flat", "dome", "other"]
    roof_types = list(dict.fromkeys(roof_types))
    layers_opts = [3, 4, 5, 2, 6]
    base_opts = [
        max(bbox["ymin"] + 3, bbox["ymax"] - 3),
        max(bbox["ymin"] + 3, bbox["ymax"] - 2),
        max(bbox["ymin"] + 2, bbox["ymax"] - 1),
        max(bbox["ymin"] + 2, bbox["ymax"] - 4),
    ]
    base_opts = list(dict.fromkeys(base_opts))

    w = bbox["xmax"] - bbox["xmin"] + 1
    d = bbox["zmax"] - bbox["zmin"] + 1
    max_pad_x = 2 if w >= 12 else (1 if w >= 8 else 0)
    max_pad_z = 2 if d >= 12 else (1 if d >= 8 else 0)
    pad_opts_x = list(range(0, max_pad_x + 1))
    pad_opts_z = list(range(0, max_pad_z + 1))

    by_type: List[List[Dict[str, Any]]] = []
    for rt in roof_types:
        specs: List[Dict[str, Any]] = []
        for layers in layers_opts:
            for by in base_opts:
                for px in pad_opts_x:
                    for pz in pad_opts_z:
                        x1 = bbox["xmin"] + px
                        x2 = bbox["xmax"] - px
                        z1 = bbox["zmin"] + pz
                        z2 = bbox["zmax"] - pz
                        if x2 <= x1 or z2 <= z1:
                            continue
                        specs.append(
                            {
                                "roof_type": rt,
                                "layers": int(layers),
                                "base_y": int(by),
                                "x1": int(x1),
                                "x2": int(x2),
                                "z1": int(z1),
                                "z2": int(z2),
                            }
                        )
        if specs:
            by_type.append(specs)

    out: List[Dict[str, Any]] = []
    target = max(1, int(max_variants))
    k = 0
    while len(out) < target:
        progressed = False
        for specs in by_type:
            if k < len(specs):
                out.append(specs[k])
                progressed = True
                if len(out) >= target:
                    break
        if not progressed:
            break
        k += 1
    return out


def _window_variant_specs(
    bbox: Dict[str, int],
    deficits: Dict[str, float],
    *,
    max_variants: int,
) -> List[Dict[str, Any]]:
    glass_def = float(deficits.get("glass", 0.0) or 0.0)
    if glass_def > 0.08:
        spacing_opts = [2, 3, 4, 5]
    elif glass_def > 0.04:
        spacing_opts = [3, 2, 4, 5]
    else:
        spacing_opts = [4, 3, 5, 2]
    height_opts = [2, 3, 1]
    y_opts = [
        bbox["ymin"] + 2,
        bbox["ymin"] + 3,
        max(bbox["ymin"] + 1, (bbox["ymin"] + bbox["ymax"]) // 2 - 1),
        max(bbox["ymin"] + 1, bbox["ymax"] - 3),
    ]
    face_opts = ["all", "north", "south", "east", "west"]

    by_face: List[List[Dict[str, Any]]] = []
    for face in face_opts:
        specs: List[Dict[str, Any]] = []
        for spacing in spacing_opts:
            for win_h in height_opts:
                for yv in y_opts:
                    specs.append({"face": face, "spacing": int(spacing), "window_height": int(win_h), "y": int(yv)})
        if specs:
            by_face.append(specs)

    out: List[Dict[str, Any]] = []
    target = max(1, int(max_variants))
    k = 0
    while len(out) < target:
        progressed = False
        for specs in by_face:
            if k < len(specs):
                out.append(specs[k])
                progressed = True
                if len(out) >= target:
                    break
        if not progressed:
            break
        k += 1
    return out


def _bundle_signature(ops: List[Dict[str, Any]]) -> str:
    try:
        return json.dumps(ops, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(len(ops))


def _propose_correction_candidates(
    plan: Dict[str, Any],
    desc: Dict[str, Any],
    metrics: Dict[str, Any],
    *,
    max_operations: int,
    max_added_ops_per_iter: int,
    roof_search_variants: int,
    window_search_variants: int,
    max_search_candidates: int,
    enable_candidate_diversification: bool,
    diversification_mode: str,
    enable_wall_shell_balance: bool,
    wall_min_deficit: float,
    coarse_stage_only: bool,
) -> List[List[Dict[str, Any]]]:
    bbox = metrics["bbox"]
    role_blocks = metrics["role_blocks"]
    comps = metrics["components"]
    deficits = metrics["role_deficits"]
    prefs = metrics["shape_preferences"]

    need_roof = comps["roof"] < 0.62 or deficits.get("roof", 0.0) > 0.05
    need_window = comps["window"] < 0.60 or deficits.get("glass", 0.0) > 0.03

    roof_variants = max(1, int(roof_search_variants))
    window_variants = max(1, int(window_search_variants))
    max_candidates = max(1, int(max_search_candidates))

    roof_ops_list: List[List[Dict[str, Any]]] = [[]]
    if need_roof:
        roof_ops_list = []
        for idx, spec in enumerate(_roof_variant_specs(bbox, prefs, max_variants=roof_variants)):
            rops = _expand_roof_template(
                {**spec, "block": role_blocks["roof"]},
                bbox,
                {k: role_blocks[k] for k in role_blocks},
                "roof",
                f"self_refine_roof_v{idx}",
            )
            rops = _tag_ops(rops, role="roof", reason=f"self_refine_roof_v{idx}", confidence=0.9)
            roof_ops_list.append(rops)
        if not roof_ops_list:
            roof_ops_list = [[]]

    window_ops_list: List[List[Dict[str, Any]]] = [[]]
    if need_window and (not coarse_stage_only):
        window_ops_list = []
        for idx, spec in enumerate(_window_variant_specs(bbox, deficits, max_variants=window_variants)):
            wops = _expand_window_pattern(
                {**spec, "block": role_blocks["glass"]},
                bbox,
                {k: role_blocks[k] for k in role_blocks},
                "glass",
                f"self_refine_window_v{idx}",
            )
            wops = _tag_ops(wops, role="glass", reason=f"self_refine_window_v{idx}", confidence=0.9)
            window_ops_list.append(wops)
        if not window_ops_list:
            window_ops_list = [[]]

    base_ops = _propose_base_balance_ops(
        metrics,
        max_operations=max_operations,
        enable_wall_shell_balance=bool(enable_wall_shell_balance),
        wall_min_deficit=float(wall_min_deficit),
    )
    if coarse_stage_only:
        base_ops = [op for op in base_ops if not _is_decor_op(op)]

    bundles: List[List[Dict[str, Any]]] = []
    seen = set()

    if not bool(enable_candidate_diversification):
        for rops in roof_ops_list:
            for wops in window_ops_list:
                ops: List[Dict[str, Any]] = []
                ops.extend(rops)
                ops.extend(wops)
                ops.extend(base_ops)
                if not ops:
                    continue
                if len(ops) > max_added_ops_per_iter:
                    ops = ops[:max_added_ops_per_iter]
                sig = _bundle_signature(ops)
                if sig in seen:
                    continue
                seen.add(sig)
                bundles.append(ops)
                if len(bundles) >= max_candidates:
                    return bundles
        return bundles

    base_variants: List[List[Dict[str, Any]]] = []
    if base_ops:
        core_ops = [op for op in base_ops if "trim" not in str(op.get("purpose", "")).lower() and "light" not in str(op.get("purpose", "")).lower()]
        decor_ops = [op for op in base_ops if op not in core_ops]
        base_variants.append(base_ops)
        if core_ops:
            base_variants.append(core_ops)
        if decor_ops:
            base_variants.append(decor_ops)
    base_variants.append([])

    def add_bundle(parts: List[List[Dict[str, Any]]]) -> bool:
        ops: List[Dict[str, Any]] = []
        for part in parts:
            ops.extend(part)
        if not ops:
            return False
        if len(ops) > max_added_ops_per_iter:
            ops = ops[:max_added_ops_per_iter]
        sig = _bundle_signature(ops)
        if sig in seen:
            return False
        seen.add(sig)
        bundles.append(ops)
        return len(bundles) >= max_candidates

    def emit_single() -> bool:
        for bops in base_variants:
            if add_bundle([bops]):
                return True
        for rops in roof_ops_list:
            if add_bundle([rops]):
                return True
        for wops in window_ops_list:
            if add_bundle([wops]):
                return True
        return False

    def emit_pair() -> bool:
        for bops in base_variants:
            for rops in roof_ops_list:
                if add_bundle([bops, rops]):
                    return True
            for wops in window_ops_list:
                if add_bundle([bops, wops]):
                    return True
        return False

    def emit_full() -> bool:
        for bops in base_variants:
            for rops in roof_ops_list:
                for wops in window_ops_list:
                    if add_bundle([bops, rops, wops]):
                        return True
        return False

    mode = str(diversification_mode or "normal").strip().lower()
    if mode == "underbuild":
        if emit_full():
            return bundles
        if emit_pair():
            return bundles
        if emit_single():
            return bundles
    elif mode == "both":
        if emit_pair():
            return bundles
        if emit_full():
            return bundles
        if emit_single():
            return bundles
    else:
        # normal / overbuild: prefer conservative edits first.
        if emit_single():
            return bundles
        if emit_pair():
            return bundles
        if emit_full():
            return bundles
    return bundles


def _is_better_result(candidate: Dict[str, Any], best: Optional[Dict[str, Any]], *, eps: float = 1e-9) -> bool:
    if best is None:
        return True

    c_score = float(candidate.get("selection_score", 0.0))
    b_score = float(best.get("selection_score", 0.0))
    if c_score > b_score + eps:
        return True
    if c_score < b_score - eps:
        return False

    # Tie-break priority: pred_target_ratio, then material budget violations.
    c_ratio_err = abs(float(candidate.get("pred_target_ratio", 1.0)) - 1.0)
    b_ratio_err = abs(float(best.get("pred_target_ratio", 1.0)) - 1.0)
    if c_ratio_err < b_ratio_err - eps:
        return True
    if c_ratio_err > b_ratio_err + eps:
        return False

    c_dim_drop = float(candidate.get("dim_score_drop", 0.0))
    b_dim_drop = float(best.get("dim_score_drop", 0.0))
    if c_dim_drop < b_dim_drop - eps:
        return True
    if c_dim_drop > b_dim_drop + eps:
        return False

    c_shape_drop = float(candidate.get("shape_proxy_drop", 0.0))
    b_shape_drop = float(best.get("shape_proxy_drop", 0.0))
    if c_shape_drop < b_shape_drop - eps:
        return True
    if c_shape_drop > b_shape_drop + eps:
        return False

    c_fp = float(candidate.get("footprint_profile_l1", 0.0))
    b_fp = float(best.get("footprint_profile_l1", 0.0))
    if c_fp < b_fp - eps:
        return True
    if c_fp > b_fp + eps:
        return False

    c_hp = float(candidate.get("height_profile_l1", 0.0))
    b_hp = float(best.get("height_profile_l1", 0.0))
    if c_hp < b_hp - eps:
        return True
    if c_hp > b_hp + eps:
        return False

    c_bv_rel = float(candidate.get("budget_violation_rel_sum", 0.0))
    b_bv_rel = float(best.get("budget_violation_rel_sum", 0.0))
    if c_bv_rel < b_bv_rel - eps:
        return True
    if c_bv_rel > b_bv_rel + eps:
        return False

    c_bv_count = int(candidate.get("budget_violation_count", 0))
    b_bv_count = int(best.get("budget_violation_count", 0))
    if c_bv_count < b_bv_count:
        return True
    if c_bv_count > b_bv_count:
        return False

    c_over = float(candidate.get("overbuild_excess", 0.0))
    b_over = float(best.get("overbuild_excess", 0.0))
    if c_over < b_over - eps:
        return True
    if c_over > b_over + eps:
        return False

    c_ops = int(candidate.get("candidate_added_ops", 0))
    b_ops = int(best.get("candidate_added_ops", 0))
    if c_ops < b_ops:
        return True
    if c_ops > b_ops:
        return False

    # Stable fallback.
    return int(candidate.get("candidate_index", 10**9)) < int(best.get("candidate_index", 10**9))


def _sample_block_positions(
    vox: np.ndarray,
    bbox: Dict[str, int],
    block: str,
    *,
    max_points: int,
    prefer_top: bool = False,
) -> List[Tuple[int, int, int]]:
    idx = np.argwhere(vox == block)
    if idx.size == 0:
        return []
    if prefer_top:
        order = np.argsort(idx[:, 0])[::-1]
        idx = idx[order]
    n = int(max(1, min(int(max_points), idx.shape[0])))
    step = max(1, idx.shape[0] // n)
    picked = idx[::step][:n]
    out: List[Tuple[int, int, int]] = []
    for iy, ix, iz in picked:
        out.append((bbox["xmin"] + int(ix), bbox["ymin"] + int(iy), bbox["zmin"] + int(iz)))
    return out


def _propose_material_budget_reprojection_ops(
    plan: Dict[str, Any],
    metrics: Dict[str, Any],
    vox: np.ndarray,
    bbox: Dict[str, int],
    *,
    max_operations: int,
    max_added_ops_per_iter: int,
    strength: float,
    min_deficit_ratio: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    role_blocks = metrics["role_blocks"]
    deficits = metrics["role_deficits"]
    target = metrics["target_role_counts"]
    prefs = metrics["shape_preferences"]
    total_target = int(max(1, sum(int(v) for v in target.values())))
    ops: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"added_by_role": {}, "reduced_by_role": {}}

    def add_many(new_ops: List[Dict[str, Any]]) -> None:
        for op in new_ops:
            if len(ops) >= max_operations or len(ops) >= max_added_ops_per_iter:
                break
            ops.append(op)

    # Positive deficits: add role-constrained ops.
    for role in REQUIRED_PALETTE_ROLES:
        deficit = float(deficits.get(role, 0.0) or 0.0)
        if deficit <= float(min_deficit_ratio):
            continue
        demand_blocks = int(round(deficit * total_target * float(strength)))
        if demand_blocks <= 0:
            continue

        added_before = len(ops)
        if role == "roof":
            layers = max(2, min(6, 2 + demand_blocks // 150))
            rops = _expand_roof_template(
                {
                    "roof_type": str(prefs.get("roof_type", "flat")),
                    "base_y": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                    "layers": layers,
                    "block": role_blocks["roof"],
                },
                bbox,
                {k: role_blocks[k] for k in role_blocks},
                "roof",
                "budget_reproject_roof",
            )
            add_many(_tag_ops(rops, role="roof", reason="budget_reproject_roof", confidence=0.9))
        elif role == "glass":
            spacing = 2 if deficit > 0.08 else (3 if deficit > 0.05 else 4)
            wops = _expand_window_pattern(
                {
                    "face": "all",
                    "spacing": spacing,
                    "window_height": 2 if deficit < 0.1 else 3,
                    "y": bbox["ymin"] + 2,
                    "block": role_blocks["glass"],
                },
                bbox,
                {k: role_blocks[k] for k in role_blocks},
                "glass",
                "budget_reproject_glass",
            )
            add_many(_tag_ops(wops, role="glass", reason="budget_reproject_glass", confidence=0.9))
        elif role == "wall":
            wall_fill = [
                {
                    "op": "fill",
                    "x1": bbox["xmin"],
                    "y1": bbox["ymin"] + 1,
                    "z1": bbox["zmin"],
                    "x2": bbox["xmax"],
                    "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                    "z2": bbox["zmin"],
                    "block": role_blocks["wall"],
                    "purpose": "budget_reproject_wall_north",
                },
                {
                    "op": "fill",
                    "x1": bbox["xmin"],
                    "y1": bbox["ymin"] + 1,
                    "z1": bbox["zmax"],
                    "x2": bbox["xmax"],
                    "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                    "z2": bbox["zmax"],
                    "block": role_blocks["wall"],
                    "purpose": "budget_reproject_wall_south",
                },
            ]
            if deficit > 0.08:
                wall_fill.extend(
                    [
                        {
                            "op": "fill",
                            "x1": bbox["xmin"],
                            "y1": bbox["ymin"] + 1,
                            "z1": bbox["zmin"],
                            "x2": bbox["xmin"],
                            "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                            "z2": bbox["zmax"],
                            "block": role_blocks["wall"],
                            "purpose": "budget_reproject_wall_west",
                        },
                        {
                            "op": "fill",
                            "x1": bbox["xmax"],
                            "y1": bbox["ymin"] + 1,
                            "z1": bbox["zmin"],
                            "x2": bbox["xmax"],
                            "y2": max(bbox["ymin"] + 3, bbox["ymax"] - 2),
                            "z2": bbox["zmax"],
                            "block": role_blocks["wall"],
                            "purpose": "budget_reproject_wall_east",
                        },
                    ]
                )
            add_many(_tag_ops(wall_fill, role="wall", reason="budget_reproject_wall", confidence=0.88))
        elif role == "floor":
            add_many(
                _tag_ops(
                    [
                        {
                            "op": "fill",
                            "x1": bbox["xmin"],
                            "y1": bbox["ymin"],
                            "z1": bbox["zmin"],
                            "x2": bbox["xmax"],
                            "y2": bbox["ymin"],
                            "z2": bbox["zmax"],
                            "block": role_blocks["floor"],
                            "purpose": "budget_reproject_floor",
                        }
                    ],
                    role="floor",
                    reason="budget_reproject_floor",
                    confidence=0.9,
                )
            )
        elif role == "trim":
            band_y = max(bbox["ymin"] + 2, min(bbox["ymax"] - 1, bbox["ymax"] - 3))
            tops = _trim_perimeter_ops(bbox, role_blocks["trim"], band_y, "budget_reproject_trim")
            add_many(_tag_ops(tops, role="trim", reason="budget_reproject_trim", confidence=0.86))
        elif role == "light":
            light_ops = []
            for x, z in (
                (bbox["xmin"], bbox["zmin"]),
                (bbox["xmax"], bbox["zmin"]),
                (bbox["xmin"], bbox["zmax"]),
                (bbox["xmax"], bbox["zmax"]),
            ):
                light_ops.append(
                    {
                        "op": "set",
                        "x": x,
                        "y": min(bbox["ymax"], bbox["ymin"] + 2),
                        "z": z,
                        "block": role_blocks["light"],
                        "purpose": "budget_reproject_light",
                    }
                )
            add_many(_tag_ops(light_ops, role="light", reason="budget_reproject_light", confidence=0.9))
        report["added_by_role"][role] = len(ops) - added_before

    # Negative deficits: reduce oversupplied non-core roles by converting to wall.
    wall_block = role_blocks.get("wall", "stonebrick")
    for role in ("glass", "trim", "light", "roof"):
        deficit = float(deficits.get(role, 0.0) or 0.0)
        over_ratio = -deficit
        if over_ratio <= float(min_deficit_ratio):
            continue
        over_blocks = int(round(over_ratio * total_target * float(strength) * 0.4))
        if over_blocks <= 0:
            continue
        sample_n = int(max(2, min(24, over_blocks)))
        coords = _sample_block_positions(
            vox,
            bbox,
            role_blocks.get(role, ""),
            max_points=sample_n,
            prefer_top=(role == "roof"),
        )
        reduced = 0
        for x, y, z in coords:
            set_op = {
                "op": "set",
                "x": int(x),
                "y": int(y),
                "z": int(z),
                "block": wall_block,
                "role": "wall",
                "role_confidence": 0.82,
                "role_reason": f"budget_reproject_reduce_{role}",
                "purpose": f"budget_reproject_reduce_{role}",
            }
            if not _add_op_if_room(ops, set_op, max_operations):
                break
            if len(ops) >= max_added_ops_per_iter:
                break
            reduced += 1
        report["reduced_by_role"][role] = reduced

    if len(ops) > max_added_ops_per_iter:
        ops = ops[:max_added_ops_per_iter]
    report["total_added_ops"] = len(ops)
    return ops, report


def _merge_with_stage_order(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def rank(op: Dict[str, Any]) -> int:
        kind = str(op.get("op", "")).strip().lower()
        if kind == "carve":
            return 2
        role = _normalize_role(op.get("role", ""))
        if role == "floor":
            return 0
        if role in {"wall", "roof"}:
            return 1
        if role in {"trim", "glass", "light"}:
            return 3
        return 2

    indexed = list(enumerate(ops))
    indexed.sort(key=lambda it: (rank(it[1]), it[0]))
    return [op for _idx, op in indexed]


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    required_roles = _normalize_required_roles(args.required_palette_roles)
    buildings = _list_buildings(dataset_root, args.building_pattern, args.limit)

    for bdir in buildings:
        src_plan_path = bdir / args.plan_subdir / "plan.json"
        desc_path = bdir / args.description_subdir / "description.json"
        if not src_plan_path.is_file():
            print(f"[self_refine_no_gt] skip {bdir.name} (missing plan)")
            continue
        if not desc_path.is_file():
            print(f"[self_refine_no_gt] skip {bdir.name} (missing description)")
            continue

        out_dir = bdir / args.out_plan_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_plan_path = out_dir / "plan.json"
        out_req_path = out_dir / "plan.request.json"
        if out_plan_path.is_file() and not args.overwrite:
            print(f"[self_refine_no_gt] skip {bdir.name} (exists)")
            continue

        raw_plan = json.loads(src_plan_path.read_text(encoding="utf-8"))
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        coerced, coerce_report = _coerce_plan(raw_plan)
        plan, validation = _validate_and_repair_plan(
            coerced,
            desc=desc if isinstance(desc, dict) else {},
            strict_schema=bool(args.strict_schema),
            enforce_role_fixed=bool(args.enforce_role_fixed),
            require_material_budget=bool(args.require_material_budget),
            material_budget_tolerance=float(args.material_budget_tolerance),
            role_fix_min_confidence=float(args.role_fix_min_confidence),
            prefer_description_palette=bool(args.prefer_description_palette),
            max_operations=int(args.max_operations),
            required_palette_roles=required_roles,
        )
        initial_wall_shell_report: Dict[str, int] = {"converted_wall_fills": 0, "preserved_ops": 0, "dedup_dropped": 0}
        init_ops = plan.get("operations", []) if isinstance(plan.get("operations"), list) else []
        fixed_init_ops, initial_wall_shell_report = _enforce_wall_fill_shell_policy([dict(op) for op in init_ops if isinstance(op, dict)])
        if len(fixed_init_ops) != len(init_ops) or initial_wall_shell_report.get("converted_wall_fills", 0) > 0:
            plan = dict(plan)
            plan["operations"] = fixed_init_ops[: int(args.max_operations)]
            plan, validation = _validate_and_repair_plan(
                plan,
                desc=desc if isinstance(desc, dict) else {},
                strict_schema=bool(args.strict_schema),
                enforce_role_fixed=bool(args.enforce_role_fixed),
                require_material_budget=bool(args.require_material_budget),
                material_budget_tolerance=float(args.material_budget_tolerance),
                role_fix_min_confidence=float(args.role_fix_min_confidence),
                prefer_description_palette=bool(args.prefer_description_palette),
                max_operations=int(args.max_operations),
                required_palette_roles=required_roles,
            )
        coerce_report = dict(coerce_report)
        coerce_report["initial_wall_shell_report"] = initial_wall_shell_report

        history: List[Dict[str, Any]] = []
        accepted_iterations = 0
        total_added_ops = 0

        try:
            vox0, bbox0 = _render_plan(plan, max_dim=int(args.max_dim))
            metrics = _self_consistency_score(plan, vox0, bbox0, desc)
            current_vox = vox0
        except Exception as exc:  # noqa: BLE001
            print(f"[self_refine_no_gt] {bdir.name} render failed: {exc}")
            plan["self_refine_no_gt"] = {
                "enabled": True,
                "error": str(exc),
                "accepted_iterations": 0,
                "history": [],
            }
            out_plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            out_req_path.write_text(
                json.dumps(
                    {
                        "building": bdir.name,
                        "source_plan": str(src_plan_path.relative_to(bdir)),
                        "description_path": str(desc_path.relative_to(bdir)),
                        "coerce_report": coerce_report,
                        "validation_report": validation,
                        "self_refine_error": str(exc),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue

        history.append({"iter": 0, "accepted": True, "score": metrics["score"], "components": metrics["components"], "added_ops": 0})

        for it in range(1, max(0, int(args.max_iterations)) + 1):
            diversification_control = _effective_candidate_diversification(metrics, args)
            iter_enable_candidate_diversification = bool(diversification_control["enabled"])
            iter_diversification_mode = str(diversification_control.get("mode", "normal"))
            wall_shell_state = _wall_shell_activation_state(metrics, args, diversification_control)
            iter_enable_wall_shell_balance = bool(wall_shell_state["enabled"])
            coarse_stage_score = _coarse_stage_score(metrics.get("components", {}))
            coarse_stage_only = bool(
                args.enforce_two_stage_generation
                and coarse_stage_score < float(args.two_stage_coarse_ready_threshold)
            )
            corr_candidates = _propose_correction_candidates(
                plan,
                desc,
                metrics,
                max_operations=int(args.max_operations),
                max_added_ops_per_iter=int(args.max_added_ops_per_iter),
                roof_search_variants=int(args.roof_search_variants),
                window_search_variants=int(args.window_search_variants),
                max_search_candidates=int(args.max_search_candidates),
                enable_candidate_diversification=iter_enable_candidate_diversification,
                diversification_mode=iter_diversification_mode,
                enable_wall_shell_balance=bool(iter_enable_wall_shell_balance),
                wall_min_deficit=float(args.wall_balance_shell_min_deficit),
                coarse_stage_only=bool(coarse_stage_only),
            )
            if not corr_candidates:
                history.append(
                    {
                        "iter": it,
                        "accepted": False,
                        "reason": "no_corrections",
                        "candidate_diversification_enabled": bool(iter_enable_candidate_diversification),
                        "candidate_diversification_high_risk": bool(diversification_control["high_risk"]),
                        "candidate_diversification_risk_level": diversification_control["risk_level"],
                        "candidate_diversification_risk_ratio": float(diversification_control["risk_ratio"]),
                        "candidate_diversification_risk_threshold": float(diversification_control["risk_threshold"]),
                        "candidate_diversification_underbuild_high": bool(diversification_control["underbuild_high"]),
                        "candidate_diversification_underbuild_ratio": float(diversification_control["underbuild_ratio"]),
                        "candidate_diversification_underbuild_threshold": float(diversification_control["underbuild_threshold"]),
                        "candidate_diversification_mode": iter_diversification_mode,
                        "coarse_stage_score": float(coarse_stage_score),
                        "coarse_stage_only": bool(coarse_stage_only),
                        "wall_balance_shell_high_risk_only": bool(args.wall_balance_shell_high_risk_only),
                        "wall_balance_shell_enabled": bool(iter_enable_wall_shell_balance),
                        "wall_balance_shell_state": wall_shell_state,
                    }
                )
                break

            conditional_precboost_meta = _conditional_precboost_eligibility(
                desc=desc if isinstance(desc, dict) else {},
                base_metrics=metrics,
                args=args,
            )
            selection_profile_order: List[str] = ["tuned"]
            if bool(conditional_precboost_meta.get("eligible", False)):
                selection_profile_order.append("precboost")
            selection_profiles = {name: _selection_profile_from_args(args, name) for name in selection_profile_order}
            best_result_by_profile: Dict[str, Optional[Dict[str, Any]]] = {name: None for name in selection_profile_order}

            best_result: Optional[Dict[str, Any]] = None
            render_fail_count = 0
            overbuild_reject_count_by_profile: Dict[str, int] = {name: 0 for name in selection_profile_order}
            strict_reject_count = 0
            shape_reject_count = 0
            growth_reject_count = 0
            profile_reject_count = 0
            base_components = metrics.get("components", {})
            if not isinstance(base_components, dict):
                base_components = {}
            base_shape_proxy = _shape_proxy(base_components)
            base_dim_score = float(base_components.get("dim", 0.0) or 0.0)
            base_profiles = _spatial_profiles(current_vox)
            growth_control = _effective_growth_control(metrics, args)
            for cand_idx, corr_ops in enumerate(corr_candidates):
                candidate = dict(plan)
                corr_decor_ops = int(sum(1 for op in corr_ops if isinstance(op, dict) and _is_decor_op(op)))
                if coarse_stage_only and corr_decor_ops > 0:
                    profile_reject_count += 1
                    continue
                c_ops = list(plan.get("operations", [])) + list(corr_ops)
                c_ops = _merge_with_stage_order(c_ops)
                c_ops, wall_shell_report = _enforce_wall_fill_shell_policy([dict(op) for op in c_ops if isinstance(op, dict)])
                if len(c_ops) > int(args.max_operations):
                    c_ops = c_ops[: int(args.max_operations)]
                candidate["operations"] = c_ops

                candidate, cand_validation = _validate_and_repair_plan(
                    candidate,
                    desc=desc if isinstance(desc, dict) else {},
                    strict_schema=bool(args.strict_schema),
                    enforce_role_fixed=bool(args.enforce_role_fixed),
                    require_material_budget=bool(args.require_material_budget),
                    material_budget_tolerance=float(args.material_budget_tolerance),
                    role_fix_min_confidence=float(args.role_fix_min_confidence),
                    prefer_description_palette=bool(args.prefer_description_palette),
                    max_operations=int(args.max_operations),
                    required_palette_roles=required_roles,
                )
                strict_blocking_issues = cand_validation.get("strict_blocking_issues", [])
                fatal_issues, non_fatal_issues = _split_strict_issues(strict_blocking_issues)
                if bool(args.reject_strict_blocking_candidates) and len(fatal_issues) > 0:
                    strict_reject_count += 1
                    continue

                try:
                    c_vox, c_bbox = _render_plan(candidate, max_dim=int(args.max_dim))
                    c_metrics = _self_consistency_score(candidate, c_vox, c_bbox, desc)
                except Exception:  # noqa: BLE001
                    render_fail_count += 1
                    continue
                footprint_profile_l1 = 0.0
                height_profile_l1 = 0.0

                budget_reproject_ops: List[Dict[str, Any]] = []
                budget_reproject_report: Dict[str, Any] = {}
                c_material = float(c_metrics.get("components", {}).get("material", 1.0))
                if bool(args.enable_material_budget_reprojection) and c_material < float(
                    args.material_budget_reprojection_trigger_material_score
                ):
                    budget_reproject_ops, budget_reproject_report = _propose_material_budget_reprojection_ops(
                        candidate,
                        c_metrics,
                        c_vox,
                        c_bbox,
                        max_operations=int(args.max_operations),
                        max_added_ops_per_iter=max(8, int(args.max_added_ops_per_iter) // 2),
                        strength=float(args.material_budget_reprojection_strength),
                        min_deficit_ratio=float(args.material_budget_reprojection_min_deficit_ratio),
                    )
                    if budget_reproject_ops:
                        c2 = dict(candidate)
                        c2_ops = list(candidate.get("operations", [])) + budget_reproject_ops
                        c2_ops = _merge_with_stage_order(c2_ops)
                        c2_ops, _ = _enforce_wall_fill_shell_policy([dict(op) for op in c2_ops if isinstance(op, dict)])
                        if len(c2_ops) > int(args.max_operations):
                            c2_ops = c2_ops[: int(args.max_operations)]
                        c2["operations"] = c2_ops
                        c2, c2_validation = _validate_and_repair_plan(
                            c2,
                            desc=desc if isinstance(desc, dict) else {},
                            strict_schema=bool(args.strict_schema),
                            enforce_role_fixed=bool(args.enforce_role_fixed),
                            require_material_budget=bool(args.require_material_budget),
                            material_budget_tolerance=float(args.material_budget_tolerance),
                            role_fix_min_confidence=float(args.role_fix_min_confidence),
                            prefer_description_palette=bool(args.prefer_description_palette),
                            max_operations=int(args.max_operations),
                            required_palette_roles=required_roles,
                        )
                        c2_blocking_issues = c2_validation.get("strict_blocking_issues", [])
                        c2_fatal_issues, c2_non_fatal_issues = _split_strict_issues(c2_blocking_issues)
                        if bool(args.reject_strict_blocking_candidates) and len(c2_fatal_issues) > 0:
                            budget_reproject_ops = []
                            budget_reproject_report = {
                                **budget_reproject_report,
                                "skipped_due_to_strict_fatal_issues": c2_fatal_issues,
                            }
                        else:
                            if c2_non_fatal_issues:
                                budget_reproject_report = {
                                    **budget_reproject_report,
                                    "strict_non_fatal_issues": c2_non_fatal_issues,
                                }
                            try:
                                c2_vox, c2_bbox = _render_plan(c2, max_dim=int(args.max_dim))
                                c2_metrics = _self_consistency_score(c2, c2_vox, c2_bbox, desc)
                                if float(c2_metrics["score"]) >= float(c_metrics["score"]):
                                    candidate = c2
                                    cand_validation = c2_validation
                                    c_metrics = c2_metrics
                                    c_vox = c2_vox
                                    c_bbox = c2_bbox
                                    fatal_issues = c2_fatal_issues
                                    non_fatal_issues = c2_non_fatal_issues
                            except Exception:  # noqa: BLE001
                                pass

                candidate_added_ops = int(len(corr_ops) + len(budget_reproject_ops))
                cand_profiles = _spatial_profiles(c_vox)
                footprint_profile_l1, height_profile_l1 = _profile_mismatch(base_profiles, cand_profiles)
                if bool(args.enable_profile_match_guard) and (
                    footprint_profile_l1 > float(args.max_footprint_profile_l1)
                    or height_profile_l1 > float(args.max_height_profile_l1)
                ):
                    profile_reject_count += 1
                    continue
                pred_non_air = int(np.count_nonzero(c_vox != "air"))
                target_non_air = _target_non_air_from_metrics(c_metrics, fallback_non_air=pred_non_air)
                pred_target_ratio = _pred_target_ratio(pred_non_air, target_non_air)
                overbuild_excess = max(0.0, pred_target_ratio - 1.0)
                underbuild_excess = max(0.0, 1.0 - pred_target_ratio)
                base_pred_non_air = int(growth_control["base_pred_non_air"])
                candidate_growth_ratio = float(pred_non_air) / float(max(1, base_pred_non_air))
                max_growth_ratio = float(growth_control["max_growth_ratio"])
                growth_excess = max(0.0, candidate_growth_ratio - max_growth_ratio)
                if bool(growth_control["enabled"]) and growth_excess > 0.0:
                    growth_reject_count += 1
                    continue
                c_components = c_metrics.get("components", {})
                if not isinstance(c_components, dict):
                    c_components = {}
                candidate_shape_proxy = _shape_proxy(c_components)
                candidate_dim_score = float(c_components.get("dim", 0.0) or 0.0)
                shape_proxy_drop = max(0.0, float(base_shape_proxy) - float(candidate_shape_proxy))
                dim_score_drop = max(0.0, float(base_dim_score) - float(candidate_dim_score))
                if bool(args.enable_shape_degradation_guard) and (
                    shape_proxy_drop > float(args.max_shape_proxy_drop)
                    or dim_score_drop > float(args.max_dim_score_drop)
                ):
                    shape_reject_count += 1
                    continue

                budget_violation_penalty, budget_violation_count, budget_violation_rel_sum = _material_budget_violation_penalty(
                    cand_validation,
                    penalty_scale=float(args.selection_material_budget_violation_penalty),
                    count_weight=float(args.selection_material_budget_count_weight),
                )
                if budget_violation_penalty <= 0.0 and "material_budget_violation" in non_fatal_issues:
                    budget_violation_penalty = float(args.selection_material_budget_violation_penalty)
                    budget_violation_count = max(1, int(budget_violation_count))
                result_common = {
                    "candidate": candidate,
                    "rendered_vox": c_vox,
                    "metrics": c_metrics,
                    "validation": cand_validation,
                    "corr_ops_count": len(corr_ops),
                    "candidate_index": int(cand_idx),
                    "budget_reproject_ops_count": len(budget_reproject_ops),
                    "budget_reproject_report": budget_reproject_report,
                    "candidate_added_ops": int(candidate_added_ops),
                    "pred_non_air": int(pred_non_air),
                    "target_non_air": int(target_non_air),
                    "pred_target_ratio": float(pred_target_ratio),
                    "overbuild_excess": float(overbuild_excess),
                    "underbuild_excess": float(underbuild_excess),
                    "base_pred_target_ratio": float(growth_control["base_pred_target_ratio"]),
                    "candidate_growth_ratio": float(candidate_growth_ratio),
                    "max_growth_ratio": float(max_growth_ratio),
                    "growth_excess": float(growth_excess),
                    "growth_mode": str(growth_control["mode"]),
                    "strict_fatal_issues": fatal_issues,
                    "strict_non_fatal_issues": non_fatal_issues,
                    "budget_violation_penalty": float(budget_violation_penalty),
                    "budget_violation_count": int(budget_violation_count),
                    "budget_violation_rel_sum": float(budget_violation_rel_sum),
                    "base_shape_proxy": float(base_shape_proxy),
                    "candidate_shape_proxy": float(candidate_shape_proxy),
                    "shape_proxy_drop": float(shape_proxy_drop),
                    "base_dim_score": float(base_dim_score),
                    "candidate_dim_score": float(candidate_dim_score),
                    "dim_score_drop": float(dim_score_drop),
                    "footprint_profile_l1": float(footprint_profile_l1),
                    "height_profile_l1": float(height_profile_l1),
                    "corr_decor_ops": int(corr_decor_ops),
                    "wall_shell_report": wall_shell_report,
                }
                for profile_name in selection_profile_order:
                    profile = selection_profiles[profile_name]
                    overbuild_control = _effective_overbuild_control_profile(c_metrics, profile)
                    effective_max_pred_target_ratio = float(overbuild_control["max_pred_target_ratio"])
                    effective_overbuild_penalty = float(overbuild_control["selection_overbuild_penalty"])
                    risk_level = str(overbuild_control.get("risk_level", "fixed"))
                    risk_ratio = float(overbuild_control.get("risk_ratio", pred_target_ratio))

                    if bool(profile.get("enable_overbuild_guard", True)) and pred_target_ratio > float(effective_max_pred_target_ratio):
                        overbuild_reject_count_by_profile[profile_name] = int(overbuild_reject_count_by_profile.get(profile_name, 0)) + 1
                        continue

                    selection_score = (
                        float(c_metrics["score"])
                        - float(profile["selection_op_penalty"]) * float(candidate_added_ops)
                        - float(effective_overbuild_penalty) * float(overbuild_excess)
                        - float(profile["selection_underbuild_penalty"]) * float(underbuild_excess)
                        - float(args.selection_ratio_target_penalty) * abs(float(pred_target_ratio) - 1.0)
                        - float(args.selection_shape_drop_penalty) * float(shape_proxy_drop)
                        - float(args.selection_dim_drop_penalty) * float(dim_score_drop)
                        - float(args.selection_growth_excess_penalty) * float(growth_excess)
                        - float(args.selection_footprint_profile_penalty) * float(footprint_profile_l1)
                        - float(args.selection_height_profile_penalty) * float(height_profile_l1)
                        - float(budget_violation_penalty)
                    )
                    result = {
                        **result_common,
                        "selection_profile": str(profile_name),
                        "profile_op_penalty": float(profile["selection_op_penalty"]),
                        "profile_underbuild_penalty": float(profile["selection_underbuild_penalty"]),
                        "risk_level": risk_level,
                        "risk_ratio": float(risk_ratio),
                        "effective_max_pred_target_ratio": float(effective_max_pred_target_ratio),
                        "effective_overbuild_penalty": float(effective_overbuild_penalty),
                        "selection_score": float(selection_score),
                    }
                    if _is_better_result(result, best_result_by_profile.get(profile_name)):
                        best_result_by_profile[profile_name] = result

            selected_profile, best_result, profile_selection_debug = _select_profile_result(
                tuned_result=best_result_by_profile.get("tuned"),
                precboost_result=best_result_by_profile.get("precboost"),
                conditional_meta=conditional_precboost_meta,
                args=args,
            )
            overbuild_reject_count = int(overbuild_reject_count_by_profile.get("tuned", 0))

            if best_result is None:
                if strict_reject_count > 0 and overbuild_reject_count == 0 and render_fail_count == 0:
                    reason = f"all_candidates_rejected_strict_blocking:{strict_reject_count}"
                elif overbuild_reject_count > 0 and strict_reject_count == 0 and render_fail_count == 0 and shape_reject_count == 0:
                    reason = f"all_candidates_rejected_overbuild:{overbuild_reject_count}"
                elif growth_reject_count > 0 and strict_reject_count == 0 and overbuild_reject_count == 0 and render_fail_count == 0 and shape_reject_count == 0:
                    reason = f"all_candidates_rejected_growth_guard:{growth_reject_count}"
                elif shape_reject_count > 0 and strict_reject_count == 0 and overbuild_reject_count == 0 and render_fail_count == 0 and growth_reject_count == 0:
                    reason = f"all_candidates_rejected_shape_guard:{shape_reject_count}"
                elif profile_reject_count > 0 and strict_reject_count == 0 and overbuild_reject_count == 0 and render_fail_count == 0 and growth_reject_count == 0 and shape_reject_count == 0:
                    reason = f"all_candidates_rejected_profile_guard:{profile_reject_count}"
                else:
                    reason = (
                        f"all_candidates_failed_render:{render_fail_count}"
                        f"_or_overbuild_rejected:{overbuild_reject_count}"
                        f"_or_growth_rejected:{growth_reject_count}"
                        f"_or_strict_rejected:{strict_reject_count}"
                        f"_or_shape_rejected:{shape_reject_count}"
                        f"_or_profile_rejected:{profile_reject_count}"
                    )
                history.append(
                    {
                        "iter": it,
                        "accepted": False,
                        "reason": reason,
                        "candidate_diversification_enabled": bool(iter_enable_candidate_diversification),
                        "candidate_diversification_high_risk": bool(diversification_control["high_risk"]),
                        "candidate_diversification_risk_level": diversification_control["risk_level"],
                        "candidate_diversification_risk_ratio": float(diversification_control["risk_ratio"]),
                        "candidate_diversification_risk_threshold": float(diversification_control["risk_threshold"]),
                        "candidate_diversification_underbuild_high": bool(diversification_control["underbuild_high"]),
                        "candidate_diversification_underbuild_ratio": float(diversification_control["underbuild_ratio"]),
                        "candidate_diversification_underbuild_threshold": float(diversification_control["underbuild_threshold"]),
                        "candidate_diversification_mode": iter_diversification_mode,
                        "coarse_stage_score": float(coarse_stage_score),
                        "coarse_stage_only": bool(coarse_stage_only),
                        "wall_balance_shell_high_risk_only": bool(args.wall_balance_shell_high_risk_only),
                        "wall_balance_shell_enabled": bool(iter_enable_wall_shell_balance),
                        "wall_balance_shell_state": wall_shell_state,
                        "conditional_precboost": conditional_precboost_meta,
                        "selection_profiles": selection_profile_order,
                        "overbuild_reject_count_by_profile": {k: int(v) for k, v in overbuild_reject_count_by_profile.items()},
                        "growth_control": growth_control,
                        "growth_reject_count": int(growth_reject_count),
                        "profile_reject_count": int(profile_reject_count),
                        "profile_selection_debug": profile_selection_debug,
                    }
                )
                break

            gain = float(best_result["metrics"]["score"]) - float(metrics["score"])
            accepted = gain >= float(args.min_score_gain)
            history.append(
                {
                    "iter": it,
                    "accepted": bool(accepted),
                    "gain": float(gain),
                    "score": best_result["metrics"]["score"],
                    "components": best_result["metrics"]["components"],
                    "added_ops": int(best_result["corr_ops_count"] + best_result["budget_reproject_ops_count"]),
                    "candidate_index": int(best_result["candidate_index"]),
                    "candidates_tested": len(corr_candidates),
                    "candidate_diversification_enabled": bool(iter_enable_candidate_diversification),
                    "candidate_diversification_high_risk": bool(diversification_control["high_risk"]),
                    "candidate_diversification_risk_level": diversification_control["risk_level"],
                    "candidate_diversification_risk_ratio": float(diversification_control["risk_ratio"]),
                    "candidate_diversification_risk_threshold": float(diversification_control["risk_threshold"]),
                    "candidate_diversification_underbuild_high": bool(diversification_control["underbuild_high"]),
                    "candidate_diversification_underbuild_ratio": float(diversification_control["underbuild_ratio"]),
                    "candidate_diversification_underbuild_threshold": float(diversification_control["underbuild_threshold"]),
                    "candidate_diversification_mode": iter_diversification_mode,
                    "coarse_stage_score": float(coarse_stage_score),
                    "coarse_stage_only": bool(coarse_stage_only),
                    "wall_balance_shell_high_risk_only": bool(args.wall_balance_shell_high_risk_only),
                    "wall_balance_shell_enabled": bool(iter_enable_wall_shell_balance),
                    "wall_balance_shell_state": wall_shell_state,
                    "selection_profile": str(selected_profile),
                    "selection_profiles": selection_profile_order,
                    "conditional_precboost": conditional_precboost_meta,
                    "profile_selection_debug": profile_selection_debug,
                    "render_fail_count": int(render_fail_count),
                    "overbuild_reject_count": int(overbuild_reject_count),
                    "overbuild_reject_count_by_profile": {k: int(v) for k, v in overbuild_reject_count_by_profile.items()},
                    "strict_reject_count": int(strict_reject_count),
                    "shape_reject_count": int(shape_reject_count),
                    "growth_reject_count": int(growth_reject_count),
                    "profile_reject_count": int(profile_reject_count),
                    "growth_control": growth_control,
                    "budget_reproject_ops_count": int(best_result["budget_reproject_ops_count"]),
                    "budget_reproject_report": best_result["budget_reproject_report"],
                    "pred_non_air": int(best_result["pred_non_air"]),
                    "target_non_air": int(best_result["target_non_air"]),
                    "pred_target_ratio": float(best_result["pred_target_ratio"]),
                    "overbuild_excess": float(best_result["overbuild_excess"]),
                    "underbuild_excess": float(best_result["underbuild_excess"]),
                    "base_pred_target_ratio": float(best_result["base_pred_target_ratio"]),
                    "candidate_growth_ratio": float(best_result["candidate_growth_ratio"]),
                    "max_growth_ratio": float(best_result["max_growth_ratio"]),
                    "growth_excess": float(best_result["growth_excess"]),
                    "growth_mode": str(best_result["growth_mode"]),
                    "risk_level": best_result["risk_level"],
                    "risk_ratio": float(best_result["risk_ratio"]),
                    "effective_max_pred_target_ratio": float(best_result["effective_max_pred_target_ratio"]),
                    "effective_overbuild_penalty": float(best_result["effective_overbuild_penalty"]),
                    "strict_fatal_issues": best_result["strict_fatal_issues"],
                    "strict_non_fatal_issues": best_result["strict_non_fatal_issues"],
                    "budget_violation_penalty": float(best_result["budget_violation_penalty"]),
                    "budget_violation_count": int(best_result["budget_violation_count"]),
                    "budget_violation_rel_sum": float(best_result["budget_violation_rel_sum"]),
                    "base_shape_proxy": float(best_result["base_shape_proxy"]),
                    "candidate_shape_proxy": float(best_result["candidate_shape_proxy"]),
                    "shape_proxy_drop": float(best_result["shape_proxy_drop"]),
                    "base_dim_score": float(best_result["base_dim_score"]),
                    "candidate_dim_score": float(best_result["candidate_dim_score"]),
                    "dim_score_drop": float(best_result["dim_score_drop"]),
                    "footprint_profile_l1": float(best_result["footprint_profile_l1"]),
                    "height_profile_l1": float(best_result["height_profile_l1"]),
                    "corr_decor_ops": int(best_result["corr_decor_ops"]),
                    "wall_shell_report": best_result["wall_shell_report"],
                    "selection_score": float(best_result["selection_score"]),
                    "validation_report": {
                        "strict_blocking_issues": best_result["validation"].get("strict_blocking_issues", []),
                        "budget_violations_count": len(best_result["validation"].get("budget_violations", [])),
                    },
                }
            )
            if not accepted:
                break

            plan = best_result["candidate"]
            validation = best_result["validation"]
            metrics = best_result["metrics"]
            current_vox = best_result["rendered_vox"]
            accepted_iterations += 1
            total_added_ops += int(best_result["candidate_added_ops"])

        plan["self_refine_no_gt"] = {
            "enabled": True,
            "accepted_iterations": int(accepted_iterations),
            "total_added_ops": int(total_added_ops),
            "final_score": float(metrics["score"]),
            "final_components": metrics["components"],
            "history": history,
            "profile": {
                "max_iterations": int(args.max_iterations),
                "min_score_gain": float(args.min_score_gain),
                "max_added_ops_per_iter": int(args.max_added_ops_per_iter),
                "roof_search_variants": int(args.roof_search_variants),
                "window_search_variants": int(args.window_search_variants),
                "max_search_candidates": int(args.max_search_candidates),
                "enable_candidate_diversification": bool(args.enable_candidate_diversification),
                "candidate_diversification_high_risk_only": bool(args.candidate_diversification_high_risk_only),
                "wall_balance_shell_high_risk_only": bool(args.wall_balance_shell_high_risk_only),
                "wall_balance_shell_min_deficit": float(args.wall_balance_shell_min_deficit),
                "wall_balance_shell_max_shape_drop_forecast": float(args.wall_balance_shell_max_shape_drop_forecast),
                "wall_balance_shell_shape_drop_scale": float(args.wall_balance_shell_shape_drop_scale),
                "candidate_diversification_risk_threshold": float(args.candidate_diversification_risk_threshold),
                "candidate_diversification_underbuild_ratio_threshold": float(args.candidate_diversification_underbuild_ratio_threshold),
                "enable_material_budget_reprojection": bool(args.enable_material_budget_reprojection),
                "material_budget_reprojection_strength": float(args.material_budget_reprojection_strength),
                "material_budget_reprojection_min_deficit_ratio": float(args.material_budget_reprojection_min_deficit_ratio),
                "material_budget_reprojection_trigger_material_score": float(args.material_budget_reprojection_trigger_material_score),
                "selection_op_penalty": float(args.selection_op_penalty),
                "selection_overbuild_penalty": float(args.selection_overbuild_penalty),
                "selection_underbuild_penalty": float(args.selection_underbuild_penalty),
                "selection_material_budget_violation_penalty": float(args.selection_material_budget_violation_penalty),
                "selection_material_budget_count_weight": float(args.selection_material_budget_count_weight),
                "selection_ratio_target_penalty": float(args.selection_ratio_target_penalty),
                "selection_shape_drop_penalty": float(args.selection_shape_drop_penalty),
                "selection_dim_drop_penalty": float(args.selection_dim_drop_penalty),
                "selection_growth_excess_penalty": float(args.selection_growth_excess_penalty),
                "selection_footprint_profile_penalty": float(args.selection_footprint_profile_penalty),
                "selection_height_profile_penalty": float(args.selection_height_profile_penalty),
                "enable_overbuild_guard": bool(args.enable_overbuild_guard),
                "max_pred_target_ratio": float(args.max_pred_target_ratio),
                "enable_adaptive_overbuild_control": bool(args.enable_adaptive_overbuild_control),
                "adaptive_risk_ratio_threshold": float(args.adaptive_risk_ratio_threshold),
                "adaptive_high_risk_max_pred_target_ratio": float(args.adaptive_high_risk_max_pred_target_ratio),
                "adaptive_high_risk_overbuild_penalty": float(args.adaptive_high_risk_overbuild_penalty),
                "adaptive_normal_max_pred_target_ratio": float(args.adaptive_normal_max_pred_target_ratio),
                "adaptive_normal_overbuild_penalty": float(args.adaptive_normal_overbuild_penalty),
                "enable_candidate_growth_guard": bool(args.enable_candidate_growth_guard),
                "candidate_growth_ratio_max": float(args.candidate_growth_ratio_max),
                "candidate_growth_ratio_underbuild_threshold": float(args.candidate_growth_ratio_underbuild_threshold),
                "candidate_growth_ratio_underbuild_max": float(args.candidate_growth_ratio_underbuild_max),
                "enable_shape_degradation_guard": bool(args.enable_shape_degradation_guard),
                "max_shape_proxy_drop": float(args.max_shape_proxy_drop),
                "max_dim_score_drop": float(args.max_dim_score_drop),
                "enable_profile_match_guard": bool(args.enable_profile_match_guard),
                "max_footprint_profile_l1": float(args.max_footprint_profile_l1),
                "max_height_profile_l1": float(args.max_height_profile_l1),
                "enforce_two_stage_generation": bool(args.enforce_two_stage_generation),
                "two_stage_coarse_ready_threshold": float(args.two_stage_coarse_ready_threshold),
                "reject_strict_blocking_candidates": bool(args.reject_strict_blocking_candidates),
                "enable_conditional_precboost": bool(args.enable_conditional_precboost),
                "conditional_precboost_require_keyword_match": bool(args.conditional_precboost_require_keyword_match),
                "conditional_precboost_allow_keywords": _parse_csv_keywords(args.conditional_precboost_allow_keywords),
                "conditional_precboost_block_keywords": _parse_csv_keywords(args.conditional_precboost_block_keywords),
                "conditional_precboost_max_roof_score": float(args.conditional_precboost_max_roof_score),
                "conditional_precboost_min_material_score": float(args.conditional_precboost_min_material_score),
                "conditional_precboost_max_window_score": float(args.conditional_precboost_max_window_score),
                "conditional_precboost_min_raw_score_gain": float(args.conditional_precboost_min_raw_score_gain),
                "conditional_precboost_max_overbuild_excess": float(args.conditional_precboost_max_overbuild_excess),
                "conditional_precboost_max_underbuild_excess": float(args.conditional_precboost_max_underbuild_excess),
                "conditional_precboost_max_budget_violation_rel_increase": float(args.conditional_precboost_max_budget_violation_rel_increase),
                "precboost_selection_op_penalty": float(args.precboost_selection_op_penalty),
                "precboost_selection_overbuild_penalty": float(args.precboost_selection_overbuild_penalty),
                "precboost_selection_underbuild_penalty": float(args.precboost_selection_underbuild_penalty),
                "precboost_max_pred_target_ratio": float(args.precboost_max_pred_target_ratio),
                "precboost_adaptive_risk_ratio_threshold": float(args.precboost_adaptive_risk_ratio_threshold),
                "precboost_adaptive_high_risk_max_pred_target_ratio": float(args.precboost_adaptive_high_risk_max_pred_target_ratio),
                "precboost_adaptive_high_risk_overbuild_penalty": float(args.precboost_adaptive_high_risk_overbuild_penalty),
                "precboost_adaptive_normal_max_pred_target_ratio": float(args.precboost_adaptive_normal_max_pred_target_ratio),
                "precboost_adaptive_normal_overbuild_penalty": float(args.precboost_adaptive_normal_overbuild_penalty),
                "strict_non_fatal_issues": sorted(NON_FATAL_STRICT_ISSUES),
                "strict_schema": bool(args.strict_schema),
                "enforce_role_fixed": bool(args.enforce_role_fixed),
                "require_material_budget": bool(args.require_material_budget),
                "material_budget_tolerance": float(args.material_budget_tolerance),
                "role_fix_min_confidence": float(args.role_fix_min_confidence),
                "prefer_description_palette": bool(args.prefer_description_palette),
                "initial_wall_shell_report": initial_wall_shell_report,
            },
        }
        plan["validation_report"] = validation
        plan["building"] = bdir.name
        plan["created_at"] = datetime.now(timezone.utc).isoformat()

        out_plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        out_req_path.write_text(
            json.dumps(
                {
                    "building": bdir.name,
                    "source_plan": str(src_plan_path.relative_to(bdir)),
                    "description_path": str(desc_path.relative_to(bdir)),
                    "coerce_report": coerce_report,
                    "validation_report": validation,
                    "self_refine_no_gt": plan["self_refine_no_gt"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[self_refine_no_gt] wrote {out_plan_path}")


if __name__ == "__main__":
    main()
