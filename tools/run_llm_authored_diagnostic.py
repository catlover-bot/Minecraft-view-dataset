#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from tools.llm_config import load_llm_config, model_for_provider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run llm_authored_10 diagnostic suite end-to-end.")
    p.add_argument("--dataset_root", default="datasets/llm_authored_10")
    p.add_argument("--outputs_root", default="outputs/llm_authored_10")
    p.add_argument("--reports_dir", default="reports/final")
    p.add_argument("--provider", default="", help="openai|anthropic|mock")
    p.add_argument("--dotenv", default="")
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--views", type=int, default=10)
    p.add_argument("--image_size", nargs=2, type=int, default=[960, 540], metavar=("W", "H"))
    p.add_argument("--max_images", type=int, default=6)
    p.add_argument("--desc_temperature", type=float, default=0.2)
    p.add_argument("--desc_max_tokens", type=int, default=1800)
    p.add_argument("--plan_temperature", type=float, default=0.2)
    p.add_argument("--plan_max_tokens", type=int, default=1800)
    p.add_argument("--llm_seed", type=int, default=-1)
    p.add_argument("--strict_schema", action="store_true")
    p.add_argument("--allow_template_fallback", action="store_true")
    p.add_argument("--skip_source_generation", action="store_true")
    p.add_argument("--skip_dataset_build", action="store_true")
    p.add_argument("--skip_descriptions", action="store_true")
    p.add_argument("--skip_direct_plan", action="store_true")
    p.add_argument("--skip_structured_ir", action="store_true")
    p.add_argument("--skip_human_kit", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--source_capture_mode",
        choices=("synthetic", "minecraft"),
        default="synthetic",
        help="Source image creation mode for llm_authored cases.",
    )
    p.add_argument(
        "--rebuild_capture_mode",
        choices=("synthetic", "minecraft"),
        default="synthetic",
        help="Rebuild image creation mode for direct/structured branches.",
    )
    p.add_argument("--port", type=int, default=10000, help="Malmo client port (minecraft mode only).")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print("[run_llm_authored_diagnostic] $ " + " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _tag(provider: str, model: str) -> str:
    raw = f"{provider}_{model}".lower().replace("/", "_").replace("-", "_").replace(".", "_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw.strip("_")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)
    (outputs_root / "metrics").mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_llm_config(args.dotenv or None)
    provider = (args.provider or cfg.provider).strip().lower()
    if args.provider:
        cfg.provider = provider
    model = model_for_provider(cfg, provider)
    provider_tag = _tag(provider, model)

    spec_out_dir = dataset_root / "source_specs"
    spec_json = spec_out_dir / "source_specs.json"

    if not args.skip_source_generation:
        cmd = [
            sys.executable,
            str(root / "tools" / "generate_llm_authored_specs.py"),
            "--out_dir",
            str(spec_out_dir),
            "--provider",
            provider,
            "--temperature",
            "0.2",
            "--max_tokens",
            "2600",
        ]
        if args.dotenv:
            cmd += ["--dotenv", args.dotenv]
        if args.llm_seed >= 0:
            cmd += ["--llm_seed", str(args.llm_seed)]
        if args.allow_template_fallback:
            cmd += ["--allow_template_fallback"]
        _run(cmd)

    if not args.skip_dataset_build:
        if args.source_capture_mode == "minecraft":
            cmd = [
                sys.executable,
                str(root / "tools" / "capture_llm_authored_minecraft.py"),
                "--spec_json",
                str(spec_json),
                "--out_root",
                str(dataset_root),
                "--port",
                str(args.port),
                "--views",
                str(args.views),
                "--image_size",
                str(args.image_size[0]),
                str(args.image_size[1]),
                "--fov",
                "70.0",
                "--limit",
                str(args.limit),
            ]
            if args.overwrite:
                cmd += ["--overwrite"]
            _run(cmd)
        else:
            cmd = [
                sys.executable,
                str(root / "tools" / "create_llm_authored_dataset.py"),
                "--spec_json",
                str(spec_json),
                "--out_root",
                str(dataset_root),
                "--views",
                str(args.views),
                "--image_size",
                str(args.image_size[0]),
                str(args.image_size[1]),
            ]
            if args.overwrite:
                cmd += ["--overwrite"]
            _run(cmd)

    description_subdir = "description_direct"
    direct_plan_subdir = "rebuild_plan_direct"
    direct_rebuild_subdir = "rebuild_world_direct"
    direct_rebuild_images_subdir = "rebuild_world_direct_images"
    structured_ir_subdir = "structured_intermediate"
    structured_plan_subdir = "rebuild_plan_structured"
    structured_rebuild_subdir = "rebuild_world_structured"
    structured_rebuild_images_subdir = "rebuild_world_structured_images"

    if not args.skip_descriptions:
        cmd = [
            sys.executable,
            str(root / "tools" / "generate_building_descriptions.py"),
            "--dataset_root",
            str(dataset_root),
            "--out_subdir",
            description_subdir,
            "--building_pattern",
            args.building_pattern,
            "--limit",
            str(args.limit),
            "--max_images",
            str(args.max_images),
            "--temperature",
            str(args.desc_temperature),
            "--max_tokens",
            str(args.desc_max_tokens),
            "--provider",
            provider,
        ]
        if args.dotenv:
            cmd += ["--dotenv", args.dotenv]
        if args.llm_seed >= 0:
            cmd += ["--llm_seed", str(args.llm_seed)]
        if args.overwrite:
            cmd += ["--overwrite"]
        _run(cmd)

    if not args.skip_direct_plan:
        cmd = [
            sys.executable,
            str(root / "tools" / "generate_rebuild_plans.py"),
            "--dataset_root",
            str(dataset_root),
            "--description_subdir",
            description_subdir,
            "--plan_subdir",
            direct_plan_subdir,
            "--building_pattern",
            args.building_pattern,
            "--limit",
            str(args.limit),
            "--temperature",
            str(args.plan_temperature),
            "--max_tokens",
            str(args.plan_max_tokens),
            "--provider",
            provider,
        ]
        if args.dotenv:
            cmd += ["--dotenv", args.dotenv]
        if args.llm_seed >= 0:
            cmd += ["--llm_seed", str(args.llm_seed)]
        if args.strict_schema:
            cmd += ["--strict_schema"]
        if args.overwrite:
            cmd += ["--overwrite"]
        _run(cmd)

    # Structured IR from description
    if not args.skip_structured_ir:
        cmd = [
            sys.executable,
            str(root / "tools" / "build_structured_intermediate.py"),
            "--dataset_root",
            str(dataset_root),
            "--description_subdir",
            description_subdir,
            "--out_subdir",
            structured_ir_subdir,
            "--building_pattern",
            args.building_pattern,
            "--limit",
            str(args.limit),
        ]
        if args.overwrite:
            cmd += ["--overwrite"]
        _run(cmd)

        cmd = [
            sys.executable,
            str(root / "tools" / "generate_plan_from_intermediate.py"),
            "--dataset_root",
            str(dataset_root),
            "--intermediate_subdir",
            structured_ir_subdir,
            "--out_subdir",
            structured_plan_subdir,
            "--building_pattern",
            args.building_pattern,
            "--limit",
            str(args.limit),
        ]
        if args.overwrite:
            cmd += ["--overwrite"]
        _run(cmd)

    # Render both conditions
    for plan_subdir, out_subdir in (
        (direct_plan_subdir, direct_rebuild_subdir),
        (structured_plan_subdir, structured_rebuild_subdir),
    ):
        cmd = [
            sys.executable,
            str(root / "tools" / "render_rebuild_from_plan.py"),
            "--dataset_root",
            str(dataset_root),
            "--plan_subdir",
            plan_subdir,
            "--out_subdir",
            out_subdir,
            "--building_pattern",
            args.building_pattern,
            "--limit",
            str(args.limit),
        ]
        if args.overwrite:
            cmd += ["--overwrite"]
        _run(cmd)

    # Render rebuild images for both conditions (synthetic preview or Minecraft capture).
    if args.rebuild_capture_mode == "minecraft":
        for rebuild_subdir, out_subdir, prov_prefix in (
            (direct_rebuild_subdir, "direct_rebuild_images_minecraft_capture", "direct_rebuild"),
            (structured_rebuild_subdir, "structured_rebuild_images_minecraft_capture", "structured_rebuild"),
        ):
            cmd = [
                sys.executable,
                str(root / "tools" / "capture_llm_authored_rebuild_minecraft.py"),
                "--dataset_root",
                str(dataset_root),
                "--rebuild_subdir",
                rebuild_subdir,
                "--out_subdir",
                out_subdir,
                "--provenance_prefix",
                prov_prefix,
                "--building_pattern",
                args.building_pattern,
                "--limit",
                str(args.limit),
                "--port",
                str(args.port),
                "--views",
                "8",
                "--image_size",
                str(args.image_size[0]),
                str(args.image_size[1]),
                "--fov",
                "70.0",
            ]
            if args.overwrite:
                cmd += ["--overwrite"]
            _run(cmd)
    else:
        for rebuild_subdir, out_subdir in (
            (direct_rebuild_subdir, direct_rebuild_images_subdir),
            (structured_rebuild_subdir, structured_rebuild_images_subdir),
        ):
            cmd = [
                sys.executable,
                str(root / "tools" / "capture_rebuild_views.py"),
                "--dataset_root",
                str(dataset_root),
                "--rebuild_subdir",
                rebuild_subdir,
                "--out_subdir",
                out_subdir,
                "--building_pattern",
                args.building_pattern,
                "--limit",
                str(args.limit),
                "--views",
                "8",
                "--image_size",
                str(args.image_size[0]),
                str(args.image_size[1]),
            ]
            if args.overwrite:
                cmd += ["--overwrite"]
            _run(cmd)

    # Metrics: description + rebuild(direct/structured) + repair(direct/structured)
    cmd = [
        sys.executable,
        str(root / "tools" / "evaluate_description_quality.py"),
        "--dataset_root",
        str(dataset_root),
        "--description_subdir",
        description_subdir,
        "--building_pattern",
        args.building_pattern,
        "--limit",
        str(args.limit),
        "--out",
        str(outputs_root / "metrics" / f"description_{provider_tag}.json"),
    ]
    _run(cmd)

    for pred_subdir, out_name in (
        (direct_rebuild_subdir, f"rebuild_direct_{provider_tag}.json"),
        (structured_rebuild_subdir, f"rebuild_structured_{provider_tag}.json"),
    ):
        cmd = [
            sys.executable,
            str(root / "tools" / "evaluate_rebuild_metrics.py"),
            "--gt_root",
            str(dataset_root),
            "--pred_root",
            str(dataset_root),
            "--pred_subdir",
            pred_subdir,
            "--building_pattern",
            args.building_pattern,
            "--limit",
            str(args.limit),
            "--out",
            str(outputs_root / "metrics" / out_name),
        ]
        _run(cmd)

    for pred_subdir, out_name in (
        (direct_rebuild_subdir, f"repair_direct_{provider_tag}.json"),
        (structured_rebuild_subdir, f"repair_structured_{provider_tag}.json"),
    ):
        cmd = [
            sys.executable,
            str(root / "tools" / "evaluate_repair_effort.py"),
            "--gt_root",
            str(dataset_root),
            "--pred_root",
            str(dataset_root),
            "--pred_subdir",
            pred_subdir,
            "--building_pattern",
            args.building_pattern,
            "--limit",
            str(args.limit),
            "--out",
            str(outputs_root / "metrics" / out_name),
        ]
        _run(cmd)

    if not args.skip_human_kit:
        cmd = [
            sys.executable,
            str(root / "tools" / "prepare_llm_authored_human_kit.py"),
            "--dataset_root",
            str(dataset_root),
            "--out_root",
            str(outputs_root / "human_kit"),
            "--building_pattern",
            args.building_pattern,
            "--limit",
            str(args.limit),
        ]
        _run(cmd)

    cmd = [
        sys.executable,
        str(root / "tools" / "summarize_llm_authored_10_results.py"),
        "--dataset_root",
        str(dataset_root),
        "--outputs_root",
        str(outputs_root),
        "--provider_tag",
        provider_tag,
        "--description_subdir",
        description_subdir,
        "--direct_plan_subdir",
        direct_plan_subdir,
        "--direct_rebuild_subdir",
        direct_rebuild_subdir,
        "--structured_ir_subdir",
        structured_ir_subdir,
        "--structured_plan_subdir",
        structured_plan_subdir,
        "--structured_rebuild_subdir",
        structured_rebuild_subdir,
        "--reports_dir",
        str(reports_dir),
    ]
    _run(cmd)

    print("[run_llm_authored_diagnostic] complete")
    print(f"[run_llm_authored_diagnostic] provider_tag={provider_tag}")
    print(f"[run_llm_authored_diagnostic] dataset_root={dataset_root}")
    print(f"[run_llm_authored_diagnostic] outputs_root={outputs_root}")


if __name__ == "__main__":
    main()
