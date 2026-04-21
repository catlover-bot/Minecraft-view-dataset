#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.llm_config import load_llm_config


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    dataset_name: str
    provider: str
    model_tag: str
    model_short_tag: str
    outputs_root: Path
    gt_root: Path
    description_subdir: str
    base_plan_subdir: str
    baseline_renderer_subdir: str
    baseline_agent_subdir: str


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    mode: str  # baseline | rerender | mission_only
    refine_flags: Tuple[str, ...] = ()
    agentexec_flags: Tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real-agent intervention A/B: failure-type interventions "
            "(overbuild/underbuild/material/mission) and two-stage ON/OFF."
        )
    )
    parser.add_argument("--limit", type=int, default=10, help="Max buildings per case.")
    parser.add_argument("--building_pattern", default="building_*")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument(
        "--placement_mode",
        choices=["chat_commands", "hand_place"],
        default="chat_commands",
        help="Agent execution mode for generate_agentexec_world_real.py",
    )
    parser.add_argument("--thresholds_json", default="tools/thresholds_levels.example.json")
    parser.add_argument("--dotenv", default="", help="Optional .env path used to resolve default Gemini model tag.")
    parser.add_argument(
        "--gemini_model_tag",
        default="",
        help=(
            "Model tag used in directory names for Gemini cases. "
            "Example: gemini_gemini_3_1_pro_preview"
        ),
    )
    parser.add_argument(
        "--cases",
        default="v1_openai,v1_claude,v4_openai,v4_claude",
        help="Comma-separated case keys.",
    )
    parser.add_argument(
        "--variants",
        default="baseline,twostage_off,overbuild_guard,underbuild_relax,material_reproject,mission_stable_exec",
        help="Comma-separated variant keys.",
    )
    parser.add_argument("--overwrite_variants", action="store_true", help="Overwrite rerender outputs.")
    parser.add_argument("--overwrite_agentexec", action="store_true", help="Overwrite agentexec outputs.")
    parser.add_argument("--agentexec_retry_attempts", type=int, default=3, help="Retries for agentexec generation.")
    parser.add_argument("--low_iou_threshold", type=float, default=0.20)
    parser.add_argument("--overbuild_ratio_threshold", type=float, default=1.15)
    parser.add_argument("--underbuild_ratio_threshold", type=float, default=0.85)
    parser.add_argument("--material_match_threshold", type=float, default=0.20)
    parser.add_argument(
        "--out_json",
        default="reports/final/intervention_ab_real_limit10.json",
    )
    parser.add_argument(
        "--out_md",
        default="reports/final/intervention_ab_real_limit10.md",
    )
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Skip all generation/evaluation and build report from existing metric files.",
    )
    return parser.parse_args()


def _run_cmd(cmd: List[str]) -> None:
    print("[run_real_failure_intervention_ab] $", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _resolve_gemini_model_tag(explicit_tag: str, dotenv: str) -> str:
    if explicit_tag.strip():
        tag = _slugify(explicit_tag)
        return tag if tag.startswith("gemini_") else f"gemini_{tag}"
    cfg = load_llm_config(dotenv or None)
    model = cfg.gemini_model or "gemini_model"
    return f"gemini_{_slugify(model)}"


def _case_specs(gemini_model_tag: str) -> Dict[str, CaseSpec]:
    out_root = ROOT / "outputs" / "i2t2b"
    gt_root = ROOT / "datasets"
    specs = {
        "v1_openai": CaseSpec(
            key="v1_openai",
            label="v1/OpenAI",
            dataset_name="buildings_100_v1",
            provider="openai",
            model_tag="openai_gpt_5_mini",
            model_short_tag="openai_gpt5mini",
            outputs_root=out_root / "buildings_100_v1",
            gt_root=gt_root / "buildings_100_v1",
            description_subdir="description_openai_gpt_5_mini",
            base_plan_subdir="rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini",
            baseline_renderer_subdir="rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned",
            baseline_agent_subdir="rebuild_world_agentexec_real_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned",
        ),
        "v1_claude": CaseSpec(
            key="v1_claude",
            label="v1/Claude",
            dataset_name="buildings_100_v1",
            provider="anthropic",
            model_tag="anthropic_claude_haiku_4_5_20251001",
            model_short_tag="claude_haiku45",
            outputs_root=out_root / "buildings_100_v1",
            gt_root=gt_root / "buildings_100_v1",
            description_subdir="description_anthropic_claude_haiku_4_5_20251001",
            base_plan_subdir="rebuild_plan_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001",
            baseline_renderer_subdir="rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned",
            baseline_agent_subdir="rebuild_world_agentexec_real_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned",
        ),
        "v4_openai": CaseSpec(
            key="v4_openai",
            label="v4/OpenAI",
            dataset_name="buildings_100_v4",
            provider="openai",
            model_tag="openai_gpt_5_mini",
            model_short_tag="openai_gpt5mini",
            outputs_root=out_root / "buildings_100_v4",
            gt_root=gt_root / "buildings_100_v4",
            description_subdir="description_openai_gpt_5_mini",
            base_plan_subdir="rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini",
            baseline_renderer_subdir="rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned",
            baseline_agent_subdir="rebuild_world_agentexec_real_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned",
        ),
        "v4_claude": CaseSpec(
            key="v4_claude",
            label="v4/Claude",
            dataset_name="buildings_100_v4",
            provider="anthropic",
            model_tag="anthropic_claude_haiku_4_5_20251001",
            model_short_tag="claude_haiku45",
            outputs_root=out_root / "buildings_100_v4",
            gt_root=gt_root / "buildings_100_v4",
            description_subdir="description_anthropic_claude_haiku_4_5_20251001",
            base_plan_subdir="rebuild_plan_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001",
            baseline_renderer_subdir="rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned",
            baseline_agent_subdir="rebuild_world_agentexec_real_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned",
        ),
    }
    gemini_short = _slugify(gemini_model_tag).replace("gemini_", "gm_", 1)
    specs["v1_gemini"] = CaseSpec(
        key="v1_gemini",
        label="v1/Gemini",
        dataset_name="buildings_100_v1",
        provider="gemini",
        model_tag=gemini_model_tag,
        model_short_tag=gemini_short,
        outputs_root=out_root / "buildings_100_v1",
        gt_root=gt_root / "buildings_100_v1",
        description_subdir=f"description_{gemini_model_tag}",
        base_plan_subdir=f"rebuild_plan_schema_material_v5_repair_{gemini_model_tag}",
        baseline_renderer_subdir=f"rebuild_world_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned",
        baseline_agent_subdir=f"rebuild_world_agentexec_real_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned",
    )
    specs["v4_gemini"] = CaseSpec(
        key="v4_gemini",
        label="v4/Gemini",
        dataset_name="buildings_100_v4",
        provider="gemini",
        model_tag=gemini_model_tag,
        model_short_tag=gemini_short,
        outputs_root=out_root / "buildings_100_v4",
        gt_root=gt_root / "buildings_100_v4",
        description_subdir=f"description_{gemini_model_tag}",
        base_plan_subdir=f"rebuild_plan_schema_material_v5_repair_{gemini_model_tag}",
        baseline_renderer_subdir=f"rebuild_world_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned",
        baseline_agent_subdir=f"rebuild_world_agentexec_real_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned",
    )
    return specs


def _list_buildings(root: Path, pattern: str, limit: int) -> List[Path]:
    xs = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if limit > 0:
        xs = xs[:limit]
    return xs


def _variant_specs() -> Dict[str, VariantSpec]:
    return {
        "baseline": VariantSpec(
            key="baseline",
            label="Baseline tuned (existing)",
            mode="baseline",
        ),
        "twostage_off": VariantSpec(
            key="twostage_off",
            label="Two-stage OFF",
            mode="rerender",
            refine_flags=(
                "--self_refine_no_enforce_two_stage_generation",
            ),
        ),
        "overbuild_guard": VariantSpec(
            key="overbuild_guard",
            label="Overbuild intervention",
            mode="rerender",
            refine_flags=(
                "--self_refine_enforce_two_stage_generation",
                "--self_refine_max_pred_target_ratio",
                "1.02",
                "--self_refine_selection_overbuild_penalty",
                "0.50",
                "--self_refine_selection_growth_excess_penalty",
                "0.45",
                "--self_refine_adaptive_high_risk_max_pred_target_ratio",
                "1.05",
                "--self_refine_adaptive_high_risk_overbuild_penalty",
                "0.45",
                "--self_refine_adaptive_normal_max_pred_target_ratio",
                "1.12",
                "--self_refine_adaptive_normal_overbuild_penalty",
                "0.20",
            ),
        ),
        "underbuild_relax": VariantSpec(
            key="underbuild_relax",
            label="Underbuild intervention",
            mode="rerender",
            refine_flags=(
                "--self_refine_enforce_two_stage_generation",
                "--self_refine_candidate_growth_ratio_max",
                "1.24",
                "--self_refine_candidate_growth_ratio_underbuild_max",
                "1.60",
                "--self_refine_selection_growth_excess_penalty",
                "0.20",
                "--self_refine_max_pred_target_ratio",
                "1.10",
            ),
        ),
        "material_reproject": VariantSpec(
            key="material_reproject",
            label="Material intervention",
            mode="rerender",
            refine_flags=(
                "--self_refine_enforce_two_stage_generation",
                "--self_refine_material_budget_reprojection_strength",
                "0.80",
                "--self_refine_material_budget_reprojection_trigger_material_score",
                "0.75",
                "--self_refine_selection_material_budget_violation_penalty",
                "0.08",
                "--self_refine_selection_material_budget_count_weight",
                "0.45",
            ),
        ),
        "mission_stable_exec": VariantSpec(
            key="mission_stable_exec",
            label="Mission stability intervention",
            mode="mission_only",
            agentexec_flags=(
                "--command_interval_sec",
                "0.08",
                "--post_command_wait_sec",
                "2.5",
                "--stability_max_seconds",
                "120.0",
                "--stability_max_samples",
                "240",
            ),
        ),
    }


def _renderer_subdir(case: CaseSpec, variant: VariantSpec, limit: int) -> str:
    if variant.mode in {"baseline", "mission_only"}:
        return case.baseline_renderer_subdir
    return f"rebuild_world_ab_{variant.key}_{case.model_short_tag}_l{limit:03d}"


def _refined_plan_subdir(case: CaseSpec, variant: VariantSpec, limit: int) -> str:
    return f"rebuild_plan_ab_{variant.key}_{case.model_short_tag}_l{limit:03d}"


def _agent_subdir(case: CaseSpec, variant: VariantSpec, limit: int) -> str:
    return _agent_subdir_for_mode(case=case, variant=variant, limit=limit, placement_mode="chat_commands")


def _agent_subdir_for_mode(case: CaseSpec, variant: VariantSpec, limit: int, placement_mode: str) -> str:
    hand = placement_mode == "hand_place"
    if variant.mode == "baseline":
        if hand:
            return case.baseline_agent_subdir.replace("rebuild_world_agentexec_real_", "rebuild_world_agentexec_hand_")
        return case.baseline_agent_subdir
    prefix = "rebuild_world_agentexec_hand_ab_" if hand else "rebuild_world_agentexec_real_ab_"
    return f"{prefix}{variant.key}_{case.model_short_tag}_l{limit:03d}"


def _safe_div(a: float, b: float) -> float:
    if b <= 0.0:
        return 0.0
    return a / b


def _agentexec_report(case: CaseSpec, building: str, agent_subdir: str) -> Dict[str, Any]:
    p = case.outputs_root / building / agent_subdir / "agentexec_real_report.json"
    if not p.is_file():
        return {"missing_report": True}
    try:
        obj = _load_json(p)
        obj["_path"] = str(p)
        return obj
    except Exception:
        return {"missing_report": True, "invalid_report": True, "_path": str(p)}


def _is_mission_failure(report: Dict[str, Any]) -> bool:
    if bool(report.get("missing_report")):
        return True
    mode = str(report.get("agentexec_mode", "")).strip().lower()
    reason = str(report.get("reason", "")).strip().lower()
    if "fallback_source_copy_after_real_failure" in mode:
        return True
    if "malmo" in reason or "command_channel_closed" in reason:
        return True
    return False


def _evaluate_metrics(
    *,
    py: str,
    gt_root: Path,
    pred_root: Path,
    pred_subdir: str,
    out_path: Path,
    building_pattern: str,
    limit: int,
    thresholds_json: str,
) -> Dict[str, Any]:
    cmd = [
        py,
        str(ROOT / "tools" / "evaluate_rebuild_metrics.py"),
        "--gt_root",
        str(gt_root),
        "--pred_root",
        str(pred_root),
        "--pred_source",
        "rebuild_world",
        "--pred_subdir",
        pred_subdir,
        "--out",
        str(out_path),
        "--building_pattern",
        building_pattern,
        "--fail_on_missing_pred",
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if thresholds_json:
        cmd += ["--thresholds_json", thresholds_json]
    _run_cmd(cmd)
    return _load_json(out_path)


def _summarize_with_failure_types(
    *,
    case: CaseSpec,
    agent_metrics: Dict[str, Any],
    agent_subdir: str,
    low_iou_threshold: float,
    overbuild_ratio_threshold: float,
    underbuild_ratio_threshold: float,
    material_match_threshold: float,
) -> Dict[str, Any]:
    items = agent_metrics.get("items", [])
    if not isinstance(items, list):
        items = []
    total = len(items)
    over_cnt = 0
    under_cnt = 0
    low_iou_cnt = 0
    mission_fail_cnt = 0
    cause_counts: Dict[str, int] = {
        "overbuild": 0,
        "underbuild": 0,
        "material_mismatch": 0,
        "mission_failure": 0,
    }
    low_iou_rows: List[Dict[str, Any]] = []

    for it in items:
        if not isinstance(it, dict):
            continue
        building = str(it.get("building", "")).strip()
        counts = it.get("counts", {})
        metrics = it.get("metrics", {})
        if not isinstance(counts, dict):
            counts = {}
        if not isinstance(metrics, dict):
            metrics = {}
        gt_non_air = int(counts.get("gt_non_air") or 0)
        pred_non_air = int(counts.get("pred_non_air_after_shift") or counts.get("pred_non_air") or 0)
        ratio = _safe_div(float(pred_non_air), float(max(1, gt_non_air)))
        iou = float(metrics.get("iou") or 0.0)
        mat = float(metrics.get("material_match") or 0.0)
        report = _agentexec_report(case, building, agent_subdir)
        mission_fail = _is_mission_failure(report)

        if ratio > overbuild_ratio_threshold:
            over_cnt += 1
        if ratio < underbuild_ratio_threshold:
            under_cnt += 1
        if mission_fail:
            mission_fail_cnt += 1

        if iou < low_iou_threshold:
            low_iou_cnt += 1
            if mission_fail:
                cause = "mission_failure"
            elif ratio > overbuild_ratio_threshold:
                cause = "overbuild"
            elif ratio < underbuild_ratio_threshold:
                cause = "underbuild"
            elif mat < material_match_threshold:
                cause = "material_mismatch"
            else:
                cause = "overbuild" if ratio >= 1.0 else "underbuild"
            cause_counts[cause] = int(cause_counts.get(cause, 0)) + 1
            low_iou_rows.append(
                {
                    "building": building,
                    "iou": iou,
                    "f1": float(metrics.get("f1") or 0.0),
                    "material_match": mat,
                    "pred_gt_ratio": ratio,
                    "cause": cause,
                    "mission_failure": mission_fail,
                }
            )

    low_iou_rows.sort(key=lambda r: (float(r["iou"]), float(r["material_match"])))
    return {
        "total_evaluated": total,
        "overbuild_rate": _safe_div(float(over_cnt), float(max(1, total))),
        "underbuild_rate": _safe_div(float(under_cnt), float(max(1, total))),
        "low_iou_rate": _safe_div(float(low_iou_cnt), float(max(1, total))),
        "mission_failure_rate": _safe_div(float(mission_fail_cnt), float(max(1, total))),
        "failure_cause_counts": cause_counts,
        "low_iou_examples": low_iou_rows[:10],
    }


def _run_rerender_variant(
    *,
    py: str,
    case: CaseSpec,
    variant: VariantSpec,
    renderer_subdir: str,
    refined_plan_subdir: str,
    args: argparse.Namespace,
) -> None:
    cmd = [
        py,
        str(ROOT / "tools" / "run_i2t2b_experiment.py"),
        "--dataset_root",
        str(case.outputs_root),
        "--provider",
        case.provider,
        "--description_subdir",
        case.description_subdir,
        "--plan_subdir",
        case.base_plan_subdir,
        "--rebuild_subdir",
        renderer_subdir,
        "--enable_self_refine_no_gt",
        "--self_refine_plan_subdir",
        refined_plan_subdir,
        "--skip_descriptions",
        "--skip_plan",
        "--skip_description_eval",
        "--skip_rebuild_eval",
        "--building_pattern",
        args.building_pattern,
        "--limit",
        str(args.limit),
    ]
    if args.overwrite_variants:
        cmd.append("--overwrite")
    cmd.extend(list(variant.refine_flags))
    _run_cmd(cmd)


def _renderer_outputs_ready(case: CaseSpec, renderer_subdir: str, pattern: str, limit: int) -> bool:
    buildings = _list_buildings(case.outputs_root, pattern, limit)
    if not buildings:
        return False
    for b in buildings:
        if not (b / renderer_subdir / "voxels.npy").is_file():
            return False
    return True


def _run_agentexec_variant(
    *,
    py: str,
    case: CaseSpec,
    renderer_subdir: str,
    agent_subdir: str,
    args: argparse.Namespace,
    variant: VariantSpec,
) -> None:
    cmd = [
        py,
        str(ROOT / "tools" / "generate_agentexec_world_real.py"),
        "--dataset_root",
        str(case.outputs_root),
        "--source_subdir",
        renderer_subdir,
        "--out_subdir",
        agent_subdir,
        "--port",
        str(args.port),
        "--building_pattern",
        args.building_pattern,
        "--limit",
        str(args.limit),
    ]
    if args.overwrite_agentexec:
        cmd.append("--overwrite")
    if str(args.placement_mode).strip().lower() == "hand_place":
        cmd.extend(["--placement_mode", "hand_place"])
    cmd.extend(list(variant.agentexec_flags))

    attempts = max(1, int(args.agentexec_retry_attempts))
    for i in range(1, attempts + 1):
        try:
            _run_cmd(cmd)
            return
        except subprocess.CalledProcessError:
            if i >= attempts:
                raise
            print(
                "[run_real_failure_intervention_ab] agentexec failed;"
                f" retrying ({i}/{attempts - 1}) after cooldown."
            )
            time.sleep(6.0 * i)
            if "--command_interval_sec" not in cmd:
                cmd.extend(["--command_interval_sec", "0.08"])
            if "--post_command_wait_sec" not in cmd:
                cmd.extend(["--post_command_wait_sec", "2.5"])
            if "--stability_max_seconds" not in cmd:
                cmd.extend(["--stability_max_seconds", "120.0"])
            if "--stability_max_samples" not in cmd:
                cmd.extend(["--stability_max_samples", "240"])


def _float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _extract_aggregate_metrics(metrics_obj: Dict[str, Any]) -> Dict[str, Any]:
    agg = metrics_obj.get("aggregate", {})
    if isinstance(agg, dict):
        nested = agg.get("metrics")
        if isinstance(nested, dict):
            return nested
        return agg
    return {}


def _build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    placement_mode = str(payload.get("settings", {}).get("placement_mode", "chat_commands"))
    mode_label = "Hand Place (use)" if placement_mode == "hand_place" else "Real Chat Commands"
    lines.append("# Real Agent 実験: 失敗タイプ介入 + 2段生成A/B（limit=10）")
    lines.append("")
    lines.append(f"- 作成時刻: `{payload['created_at']}`")
    lines.append(f"- placement_mode: `{placement_mode}` ({mode_label})")
    lines.append(f"- building_pattern: `{payload['settings']['building_pattern']}`")
    lines.append(f"- limit: `{payload['settings']['limit']}`")
    lines.append(f"- low_iou_threshold: `{payload['settings']['low_iou_threshold']:.2f}`")
    lines.append("")

    for case in payload.get("cases", []):
        lines.append(f"## {case['label']}")
        lines.append("")
        lines.append("| variant | IoU | F1 | material | placement | overbuild率 | underbuild率 | mission失敗率 | low-IoU率 | ΔIoU | ΔF1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        base = case.get("baseline_agent", {})
        base_iou = _float(base.get("iou"))
        base_f1 = _float(base.get("f1"))
        for row in case.get("variants", []):
            am = row.get("agent_metrics", {})
            an = row.get("agent_analysis", {})
            status = str(row.get("agentexec_status", "ok"))
            iou = _float(am.get("iou")) if status == "ok" else float("nan")
            f1 = _float(am.get("f1")) if status == "ok" else float("nan")
            mat = _float(am.get("material_match_relaxed_id") or am.get("material_match")) if status == "ok" else float("nan")
            cpr = _float(am.get("correct_placement_rate_relaxed_id") or am.get("correct_placement_rate")) if status == "ok" else float("nan")
            over = _float(an.get("overbuild_rate")) if status == "ok" else float("nan")
            under = _float(an.get("underbuild_rate")) if status == "ok" else float("nan")
            miss = _float(an.get("mission_failure_rate")) if status == "ok" else float("nan")
            lowr = _float(an.get("low_iou_rate")) if status == "ok" else float("nan")
            diou = iou - base_iou
            df1 = f1 - base_f1
            if status != "ok":
                lines.append(
                    f"| {row['label']} | - | - | - | - | - | - | - | - | - | - |"
                )
                continue
            lines.append(
                f"| {row['label']} | {iou:.4f} | {f1:.4f} | {mat:.4f} | {cpr:.4f} | {over:.2%} | {under:.2%} | {miss:.2%} | {lowr:.2%} | {diou:+.4f} | {df1:+.4f} |"
            )
        lines.append("")
        lines.append("| variant | overbuild | underbuild | material_mismatch | mission_failure |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in case.get("variants", []):
            cc = row.get("agent_analysis", {}).get("failure_cause_counts", {})
            lines.append(
                "| {v} | {o} | {u} | {m} | {ms} |".format(
                    v=row["label"],
                    o=int(cc.get("overbuild", 0)),
                    u=int(cc.get("underbuild", 0)),
                    m=int(cc.get("material_mismatch", 0)),
                    ms=int(cc.get("mission_failure", 0)),
                )
            )
        lines.append("")

    lines.append("## メモ")
    lines.append("- 失敗分類は low-IoU建物だけを対象。")
    lines.append("- 優先順位は mission_failure -> overbuild/underbuild -> material_mismatch。")
    lines.append("- `mission_stable_exec` は plan/renderを変えず、実行条件だけ変更。")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    py = sys.executable

    gemini_model_tag = _resolve_gemini_model_tag(str(args.gemini_model_tag), str(args.dotenv))
    all_cases = _case_specs(gemini_model_tag)
    all_variants = _variant_specs()
    selected_cases = [c.strip() for c in str(args.cases).split(",") if c.strip()]
    selected_variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]

    for c in selected_cases:
        if c not in all_cases:
            raise SystemExit(f"Unknown case key: {c}")
    for v in selected_variants:
        if v not in all_variants:
            raise SystemExit(f"Unknown variant key: {v}")

    out_json = (ROOT / args.out_json).resolve() if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_md = (ROOT / args.out_md).resolve() if not Path(args.out_md).is_absolute() else Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    mode_suffix = "hand" if str(args.placement_mode).strip().lower() == "hand_place" else "real"
    metrics_dir = out_json.parent / f"intervention_metrics_{mode_suffix}"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    payload_cases: List[Dict[str, Any]] = []
    for ck in selected_cases:
        case = all_cases[ck]
        if not case.outputs_root.is_dir():
            raise SystemExit(f"outputs root not found: {case.outputs_root}")
        if not case.gt_root.is_dir():
            raise SystemExit(f"gt root not found: {case.gt_root}")

        rows: List[Dict[str, Any]] = []
        baseline_agent_metrics: Optional[Dict[str, Any]] = None

        for vk in selected_variants:
            variant = all_variants[vk]
            renderer_subdir = _renderer_subdir(case, variant, args.limit)
            refined_plan_subdir = _refined_plan_subdir(case, variant, args.limit)
            agent_subdir = _agent_subdir_for_mode(
                case=case,
                variant=variant,
                limit=args.limit,
                placement_mode=str(args.placement_mode).strip().lower(),
            )

            agentexec_status = "ok"
            agentexec_error = ""

            if not args.summarize_only:
                if variant.mode == "rerender":
                    need_rerender = True
                    if (not args.overwrite_variants) and _renderer_outputs_ready(
                        case=case,
                        renderer_subdir=renderer_subdir,
                        pattern=args.building_pattern,
                        limit=int(args.limit),
                    ):
                        need_rerender = False
                        print(
                            "[run_real_failure_intervention_ab] skip rerender:"
                            f" case={case.key} variant={variant.key}"
                            " (outputs already present)."
                        )
                    if need_rerender:
                        _run_rerender_variant(
                            py=py,
                            case=case,
                            variant=variant,
                            renderer_subdir=renderer_subdir,
                            refined_plan_subdir=refined_plan_subdir,
                            args=args,
                        )
                    try:
                        _run_agentexec_variant(
                            py=py,
                            case=case,
                            renderer_subdir=renderer_subdir,
                            agent_subdir=agent_subdir,
                            args=args,
                            variant=variant,
                        )
                    except subprocess.CalledProcessError as exc:
                        agentexec_status = "failed"
                        agentexec_error = str(exc)
                elif variant.mode == "mission_only":
                    try:
                        _run_agentexec_variant(
                            py=py,
                            case=case,
                            renderer_subdir=renderer_subdir,
                            agent_subdir=agent_subdir,
                            args=args,
                            variant=variant,
                        )
                    except subprocess.CalledProcessError as exc:
                        agentexec_status = "failed"
                        agentexec_error = str(exc)

            renderer_metrics_path = metrics_dir / f"{case.key}.{variant.key}.renderer.json"
            agent_metrics_path = metrics_dir / f"{case.key}.{variant.key}.agent.json"
            if args.summarize_only:
                if renderer_metrics_path.is_file():
                    renderer_metrics_obj = _load_json(renderer_metrics_path)
                else:
                    raise SystemExit(
                        f"renderer metrics file missing in summarize_only mode: {renderer_metrics_path}"
                    )
            else:
                renderer_metrics_obj = _evaluate_metrics(
                    py=py,
                    gt_root=case.gt_root,
                    pred_root=case.outputs_root,
                    pred_subdir=renderer_subdir,
                    out_path=renderer_metrics_path,
                    building_pattern=args.building_pattern,
                    limit=args.limit,
                    thresholds_json=args.thresholds_json,
                )
            agent_metrics_obj: Dict[str, Any] = {}
            agent_analysis: Dict[str, Any] = {}
            if args.summarize_only and not agent_metrics_path.is_file():
                agentexec_status = "failed"
                agentexec_error = "agent metrics missing in summarize_only mode"
            if agentexec_status == "ok":
                try:
                    if args.summarize_only:
                        agent_metrics_obj = _load_json(agent_metrics_path)
                    else:
                        agent_metrics_obj = _evaluate_metrics(
                            py=py,
                            gt_root=case.gt_root,
                            pred_root=case.outputs_root,
                            pred_subdir=agent_subdir,
                            out_path=agent_metrics_path,
                            building_pattern=args.building_pattern,
                            limit=args.limit,
                            thresholds_json=args.thresholds_json,
                        )
                    agent_analysis = _summarize_with_failure_types(
                        case=case,
                        agent_metrics=agent_metrics_obj,
                        agent_subdir=agent_subdir,
                        low_iou_threshold=float(args.low_iou_threshold),
                        overbuild_ratio_threshold=float(args.overbuild_ratio_threshold),
                        underbuild_ratio_threshold=float(args.underbuild_ratio_threshold),
                        material_match_threshold=float(args.material_match_threshold),
                    )
                except subprocess.CalledProcessError as exc:
                    agentexec_status = "failed"
                    agentexec_error = str(exc)
                    agent_metrics_obj = {}
                    agent_analysis = {}

            row = {
                "key": variant.key,
                "label": variant.label,
                "mode": variant.mode,
                "renderer_subdir": renderer_subdir,
                "agent_subdir": agent_subdir,
                "renderer_metrics_path": str(renderer_metrics_path),
                "agent_metrics_path": str(agent_metrics_path),
                "renderer_metrics": _extract_aggregate_metrics(renderer_metrics_obj),
                "agent_metrics": _extract_aggregate_metrics(agent_metrics_obj),
                "agent_analysis": agent_analysis,
                "agentexec_status": agentexec_status,
                "agentexec_error": agentexec_error,
            }
            rows.append(row)
            if variant.key == "baseline":
                baseline_agent_metrics = row["agent_metrics"]

        if baseline_agent_metrics is None and rows:
            baseline_agent_metrics = rows[0].get("agent_metrics", {})

        payload_cases.append(
            {
                "key": case.key,
                "label": case.label,
                "dataset_name": case.dataset_name,
                "provider": case.provider,
                "model_tag": case.model_tag,
                "baseline_agent": baseline_agent_metrics or {},
                "variants": rows,
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "limit": int(args.limit),
            "building_pattern": str(args.building_pattern),
            "port": int(args.port),
            "placement_mode": str(args.placement_mode),
            "thresholds_json": str(args.thresholds_json),
            "low_iou_threshold": float(args.low_iou_threshold),
            "overbuild_ratio_threshold": float(args.overbuild_ratio_threshold),
            "underbuild_ratio_threshold": float(args.underbuild_ratio_threshold),
            "material_match_threshold": float(args.material_match_threshold),
            "cases": selected_cases,
            "variants": selected_variants,
        },
        "cases": payload_cases,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_build_markdown(payload), encoding="utf-8")

    print(f"[run_real_failure_intervention_ab] wrote: {out_json}")
    print(f"[run_real_failure_intervention_ab] wrote: {out_md}")


if __name__ == "__main__":
    main()
