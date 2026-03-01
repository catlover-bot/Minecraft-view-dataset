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
    parser.add_argument("--provider", default="", help="openai|anthropic|mock")
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

    parser.add_argument("--plan_temperature", type=float, default=0.2)
    parser.add_argument("--plan_max_tokens", type=int, default=1800)
    parser.add_argument("--plan_prompt_profile", default="", help="Optional rebuild-plan prompt profile JSON path.")
    parser.add_argument(
        "--plan_critic_revise",
        action="store_true",
        help="Enable critic-revise second pass in rebuild planning.",
    )
    parser.add_argument("--use_heuristic_plan_only", action="store_true")
    parser.add_argument("--no_fallback_heuristic", action="store_true")

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
        ]
        if args.plan_prompt_profile:
            cmd += ["--prompt_profile", args.plan_prompt_profile]
        if args.plan_critic_revise:
            cmd.append("--critic_revise")
        if args.use_heuristic_plan_only:
            cmd.append("--use_heuristic_only")
        if args.no_fallback_heuristic:
            cmd.append("--no_fallback_heuristic")
        run_cmd(cmd, cwd=root)

    if not args.skip_render:
        cmd = [
            py,
            str(root / "tools" / "render_rebuild_from_plan.py"),
            "--dataset_root",
            str(dataset_root),
            "--plan_subdir",
            plan_subdir,
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
