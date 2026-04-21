#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools.llm_config import load_llm_config
from tools.plot_experiment_figures import _draw_grouped_bars_svg


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate agent-exec worlds, evaluate renderer-vs-agent execution gap "
            "for v1/v4 x OpenAI/Claude, and update figures/reports."
        )
    )
    parser.add_argument("--limit", type=int, default=0, help="Max buildings per case (0=all).")
    parser.add_argument("--building_pattern", default="building_*")
    parser.add_argument("--thresholds_json", default=str(ROOT / "tools/thresholds_levels.example.json"))
    parser.add_argument("--overwrite_agentexec", action="store_true")
    parser.add_argument(
        "--agentexec_mode",
        choices=["proxy", "real", "hand"],
        default="proxy",
        help="proxy: sanitize-only approximation, real: chat-command placement, hand: creative hand-placement.",
    )
    parser.add_argument("--port", type=int, default=10000, help="Malmo client port for real mode.")
    parser.add_argument("--malmo_wait_timeout", type=int, default=240)
    parser.add_argument(
        "--start_malmo_if_needed",
        dest="start_malmo_if_needed",
        action="store_true",
        help="Auto-launch Malmo client if port is not LISTEN (real mode only).",
    )
    parser.add_argument(
        "--no_start_malmo_if_needed",
        dest="start_malmo_if_needed",
        action="store_false",
    )
    parser.add_argument("--skip_agentexec_generation", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--skip_report", action="store_true")
    parser.add_argument("--dotenv", default="", help="Optional .env path used to resolve Gemini model tag.")
    parser.add_argument(
        "--include_gemini_cases",
        action="store_true",
        help="Also run v1/v4 Gemini cases using schema_material_v5_repair directories.",
    )
    parser.add_argument(
        "--gemini_model_tag",
        default="",
        help=(
            "Model tag used in output directory names for Gemini. "
            "Example: gemini_gemini_3_1_pro_preview"
        ),
    )
    parser.add_argument("--date_tag", default=date.today().isoformat(), help="Date tag for figure data filename.")
    parser.set_defaults(start_malmo_if_needed=True)
    return parser.parse_args()


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


def _cases(
    agentexec_mode: str,
    *,
    include_gemini_cases: bool,
    gemini_model_tag: str,
) -> List[Dict[str, str]]:
    if agentexec_mode == "real":
        agent_openai = "rebuild_world_agentexec_real_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned"
        agent_claude = "rebuild_world_agentexec_real_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned"
        agent_gemini = f"rebuild_world_agentexec_real_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned"
        metrics_openai = "execution_gap_openai_tuned_real_agentexec.json"
        metrics_claude = "execution_gap_claude_tuned_real_agentexec.json"
        metrics_gemini = "execution_gap_gemini_tuned_real_agentexec.json"
    elif agentexec_mode == "hand":
        agent_openai = "rebuild_world_agentexec_hand_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned"
        agent_claude = "rebuild_world_agentexec_hand_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned"
        agent_gemini = f"rebuild_world_agentexec_hand_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned"
        metrics_openai = "execution_gap_openai_tuned_hand_agentexec.json"
        metrics_claude = "execution_gap_claude_tuned_hand_agentexec.json"
        metrics_gemini = "execution_gap_gemini_tuned_hand_agentexec.json"
    else:
        agent_openai = "rebuild_world_agentexec_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned"
        agent_claude = "rebuild_world_agentexec_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned"
        agent_gemini = f"rebuild_world_agentexec_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned"
        metrics_openai = "execution_gap_openai_tuned_proxy_agentexec.json"
        metrics_claude = "execution_gap_claude_tuned_proxy_agentexec.json"
        metrics_gemini = "execution_gap_gemini_tuned_proxy_agentexec.json"

    rows = [
        {
            "case_key": "v1_openai",
            "label": "v1/OpenAI",
            "dataset_name": "buildings_100_v1",
            "model_tag": "openai_gpt_5_mini",
            "renderer_subdir": "rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned",
            "agent_subdir": agent_openai,
            "metrics_name": metrics_openai,
        },
        {
            "case_key": "v1_claude",
            "label": "v1/Claude",
            "dataset_name": "buildings_100_v1",
            "model_tag": "anthropic_claude_haiku_4_5_20251001",
            "renderer_subdir": "rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned",
            "agent_subdir": agent_claude,
            "metrics_name": metrics_claude,
        },
        {
            "case_key": "v4_openai",
            "label": "v4/OpenAI",
            "dataset_name": "buildings_100_v4",
            "model_tag": "openai_gpt_5_mini",
            "renderer_subdir": "rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned",
            "agent_subdir": agent_openai,
            "metrics_name": metrics_openai,
        },
        {
            "case_key": "v4_claude",
            "label": "v4/Claude",
            "dataset_name": "buildings_100_v4",
            "model_tag": "anthropic_claude_haiku_4_5_20251001",
            "renderer_subdir": "rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned",
            "agent_subdir": agent_claude,
            "metrics_name": metrics_claude,
        },
    ]
    if include_gemini_cases:
        rows.extend(
            [
                {
                    "case_key": "v1_gemini",
                    "label": "v1/Gemini",
                    "dataset_name": "buildings_100_v1",
                    "model_tag": gemini_model_tag,
                    "renderer_subdir": f"rebuild_world_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned",
                    "agent_subdir": agent_gemini,
                    "metrics_name": metrics_gemini,
                },
                {
                    "case_key": "v4_gemini",
                    "label": "v4/Gemini",
                    "dataset_name": "buildings_100_v4",
                    "model_tag": gemini_model_tag,
                    "renderer_subdir": f"rebuild_world_schema_material_v5_repair_{gemini_model_tag}_self_refine_no_gt_tuned",
                    "agent_subdir": agent_gemini,
                    "metrics_name": metrics_gemini,
                },
            ]
        )
    return rows


def _run(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    print("[run_execution_gap_suite] $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _pct(v: float) -> str:
    return f"{v * 100.0:.2f}%"


def _port_listening(port: int) -> bool:
    for host in ("127.0.0.1", "::1"):
        family = socket.AF_INET6 if host == "::1" else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.35)
                if s.connect_ex((host, int(port))) == 0:
                    return True
        except OSError:
            continue
    return False


def _prepare_malmo_env() -> Dict[str, str]:
    env = os.environ.copy()
    if not env.get("MALMO_DIR"):
        candidate = ROOT / "MalmoPlatform"
        if candidate.is_dir():
            env["MALMO_DIR"] = str(candidate.resolve())
    malmo_dir = Path(env.get("MALMO_DIR", "")).expanduser()
    if malmo_dir.is_dir():
        xsd_candidates = [
            malmo_dir / "Schemas",
            malmo_dir / "Malmo" / "Schemas",
        ]
        current_xsd = Path(env.get("MALMO_XSD_PATH", "")).expanduser() if env.get("MALMO_XSD_PATH") else None
        current_ok = bool(current_xsd and (current_xsd / "Mission.xsd").is_file())
        if not current_ok:
            for c in xsd_candidates:
                if (c / "Mission.xsd").is_file():
                    env["MALMO_XSD_PATH"] = str(c.resolve())
                    break
    return env


def _ensure_malmo_client(port: int, wait_timeout: int, start_if_needed: bool, env: Dict[str, str]) -> None:
    if _port_listening(port):
        print(f"[run_execution_gap_suite] Malmo port already LISTEN: {port}")
        return
    if not start_if_needed:
        raise SystemExit(
            f"Malmo port {port} is not LISTEN and auto-start is disabled. "
            "Use --start_malmo_if_needed or start client manually."
        )
    _run([str(ROOT / "scripts" / "start_malmo_client_mac.sh"), "--port", str(int(port))], env=env)
    _run(
        [
            str(ROOT / "scripts" / "wait_for_malmo_port.sh"),
            "--host",
            "127.0.0.1",
            "--port",
            str(int(port)),
            "--timeout",
            str(int(wait_timeout)),
        ],
        env=env,
    )


def _make_figures(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    ja_font = "Hiragino Sans, Yu Gothic, Meiryo, Noto Sans CJK JP, sans-serif"
    categories = [r["label"] for r in rows]

    renderer_iou = [_f(r["renderer"]["iou"]) for r in rows]
    agent_iou = [_f(r["agent"]["iou"]) for r in rows]
    renderer_f1 = [_f(r["renderer"]["f1"]) for r in rows]
    agent_f1 = [_f(r["agent"]["f1"]) for r in rows]

    mat_r = [_f(r["renderer"]["material_match"]) for r in rows]
    mat_a = [_f(r["agent"]["material_match"]) for r in rows]
    cpr_r = [_f(r["renderer"]["correct_placement_rate"]) for r in rows]
    cpr_a = [_f(r["agent"]["correct_placement_rate"]) for r in rows]

    keep_iou = [_f(r["retention"]["iou"]) for r in rows]
    keep_f1 = [_f(r["retention"]["f1"]) for r in rows]
    keep_mat = [_f(r["retention"]["material_match"]) for r in rows]
    keep_cpr = [_f(r["retention"]["correct_placement_rate"]) for r in rows]

    gap_iou = [_f(r["gap"]["iou"]) for r in rows]
    gap_f1 = [_f(r["gap"]["f1"]) for r in rows]
    gap_mat = [_f(r["gap"]["material_match"]) for r in rows]
    gap_cpr = [_f(r["gap"]["correct_placement_rate"]) for r in rows]

    p1 = out_dir / "execution_gap_iou_f1_ja.svg"
    _draw_grouped_bars_svg(
        out_path=p1,
        title="Renderer上限 vs Agent実運用（IoU/F1）",
        categories=categories,
        series_names=["Renderer IoU", "Agent IoU", "Renderer F1", "Agent F1"],
        values=[renderer_iou, agent_iou, renderer_f1, agent_f1],
        y_min=0.0,
        y_max=0.7,
        font_family=ja_font,
    )

    p2 = out_dir / "execution_gap_material_placement_ja.svg"
    _draw_grouped_bars_svg(
        out_path=p2,
        title="Renderer上限 vs Agent実運用（材質/配置）",
        categories=categories,
        series_names=["Renderer 材質一致", "Agent 材質一致", "Renderer 正配置率", "Agent 正配置率"],
        values=[mat_r, mat_a, cpr_r, cpr_a],
        y_min=0.0,
        y_max=0.7,
        font_family=ja_font,
    )

    p3 = out_dir / "execution_gap_retention_ja.svg"
    _draw_grouped_bars_svg(
        out_path=p3,
        title="Agentの保持率（Agent / Renderer）",
        categories=categories,
        series_names=["IoU保持率", "F1保持率", "材質保持率", "正配置率保持率"],
        values=[keep_iou, keep_f1, keep_mat, keep_cpr],
        y_min=0.0,
        y_max=1.1,
        font_family=ja_font,
    )

    p4 = out_dir / "execution_gap_absolute_ja.svg"
    _draw_grouped_bars_svg(
        out_path=p4,
        title="Execution gap（Renderer - Agent）",
        categories=categories,
        series_names=["IoU gap", "F1 gap", "材質 gap", "正配置率 gap"],
        values=[gap_iou, gap_f1, gap_mat, gap_cpr],
        y_min=0.0,
        y_max=0.35,
        font_family=ja_font,
    )
    return [p1, p2, p3, p4]


def _write_report(
    rows: List[Dict[str, Any]],
    md_path: Path,
    date_tag: str,
    figure_paths: List[Path],
    bundle_path: Path,
    agentexec_mode: str,
) -> None:
    mean_renderer_iou = sum(_f(r["renderer"]["iou"]) for r in rows) / max(1, len(rows))
    mean_agent_iou = sum(_f(r["agent"]["iou"]) for r in rows) / max(1, len(rows))
    mean_renderer_f1 = sum(_f(r["renderer"]["f1"]) for r in rows) / max(1, len(rows))
    mean_agent_f1 = sum(_f(r["agent"]["f1"]) for r in rows) / max(1, len(rows))
    mean_renderer_mat = sum(_f(r["renderer"]["material_match"]) for r in rows) / max(1, len(rows))
    mean_agent_mat = sum(_f(r["agent"]["material_match"]) for r in rows) / max(1, len(rows))
    mean_renderer_cpr = sum(_f(r["renderer"]["correct_placement_rate"]) for r in rows) / max(1, len(rows))
    mean_agent_cpr = sum(_f(r["agent"]["correct_placement_rate"]) for r in rows) / max(1, len(rows))

    lines: List[str] = []
    lines.append("# Execution Gap まとめ（Renderer上限 vs Agent実運用）")
    lines.append("")
    lines.append(f"更新日: {date_tag}")
    lines.append("")
    lines.append("今回は `rebuild_world_*` をRenderer上限、`rebuild_world_agentexec_*` をAgent実運用として比較しました。")
    if agentexec_mode == "real":
        lines.append("※ 今回の `agentexec` は、Malmo上で `chat /setblock` / `chat /fill` を実行した real placement です。")
    elif agentexec_mode == "hand":
        lines.append("※ 今回の `agentexec` は、Malmo上で Creative 手置き（`use`）で配置した実行結果です。")
    else:
        lines.append("※ 今回の `agentexec` は、Malmo実行時のブロック正規化を反映した proxy です。")
    lines.append("")
    lines.append(f"## 全体（{len(rows)}条件平均）")
    lines.append("")
    lines.append(f"- IoU: Renderer `{_pct(mean_renderer_iou)}` -> Agent `{_pct(mean_agent_iou)}`（gap `{_pct(mean_renderer_iou - mean_agent_iou)}`）")
    lines.append(f"- F1: Renderer `{_pct(mean_renderer_f1)}` -> Agent `{_pct(mean_agent_f1)}`（gap `{_pct(mean_renderer_f1 - mean_agent_f1)}`）")
    lines.append(f"- material_match: Renderer `{_pct(mean_renderer_mat)}` -> Agent `{_pct(mean_agent_mat)}`（gap `{_pct(mean_renderer_mat - mean_agent_mat)}`）")
    lines.append(f"- correct_placement_rate: Renderer `{_pct(mean_renderer_cpr)}` -> Agent `{_pct(mean_agent_cpr)}`（gap `{_pct(mean_renderer_cpr - mean_agent_cpr)}`）")
    lines.append("")
    lines.append("## 条件別")
    lines.append("")
    lines.append("| 条件 | Renderer IoU | Agent IoU | IoU保持率 | Renderer F1 | Agent F1 | F1保持率 | Renderer材質 | Agent材質 | Renderer配置率 | Agent配置率 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {label} | {ri} | {ai} | {ki} | {rf} | {af} | {kf} | {rm} | {am} | {rc} | {ac} |".format(
                label=r["label"],
                ri=_pct(_f(r["renderer"]["iou"])),
                ai=_pct(_f(r["agent"]["iou"])),
                ki=_pct(_f(r["retention"]["iou"])),
                rf=_pct(_f(r["renderer"]["f1"])),
                af=_pct(_f(r["agent"]["f1"])),
                kf=_pct(_f(r["retention"]["f1"])),
                rm=_pct(_f(r["renderer"]["material_match"])),
                am=_pct(_f(r["agent"]["material_match"])),
                rc=_pct(_f(r["renderer"]["correct_placement_rate"])),
                ac=_pct(_f(r["agent"]["correct_placement_rate"])),
            )
        )
    lines.append("")
    lines.append("## 図")
    lines.append("")
    for p in figure_paths:
        rel = p.relative_to(ROOT / "reports")
        lines.append(f"- `reports/{rel}`")
    lines.append("")
    lines.append("## 元データ")
    lines.append("")
    lines.append(f"- `{bundle_path.relative_to(ROOT)}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    py = sys.executable
    gemini_model_tag = _resolve_gemini_model_tag(str(args.gemini_model_tag), str(args.dotenv))
    cases = _cases(
        args.agentexec_mode,
        include_gemini_cases=bool(args.include_gemini_cases),
        gemini_model_tag=gemini_model_tag,
    )
    malmo_env = _prepare_malmo_env()
    rows: List[Dict[str, Any]] = []

    if args.agentexec_mode in {"real", "hand"} and not args.skip_agentexec_generation:
        _ensure_malmo_client(
            port=int(args.port),
            wait_timeout=int(args.malmo_wait_timeout),
            start_if_needed=bool(args.start_malmo_if_needed),
            env=malmo_env,
        )

    for case in cases:
        dataset_name = case["dataset_name"]
        gt_root = ROOT / "datasets" / dataset_name
        pred_root = ROOT / "outputs" / "i2t2b" / dataset_name
        metrics_out = pred_root / "metrics" / "rebuild" / case["metrics_name"]

        if not args.skip_agentexec_generation:
            if args.agentexec_mode in {"real", "hand"}:
                cmd = [
                    py,
                    str(ROOT / "tools" / "generate_agentexec_world_real.py"),
                    "--dataset_root",
                    str(pred_root),
                    "--source_subdir",
                    case["renderer_subdir"],
                    "--out_subdir",
                    case["agent_subdir"],
                    "--port",
                    str(int(args.port)),
                    "--building_pattern",
                    str(args.building_pattern),
                ]
                if args.agentexec_mode == "hand":
                    cmd += ["--placement_mode", "hand_place"]
            else:
                cmd = [
                    py,
                    str(ROOT / "tools" / "generate_agentexec_world_proxy.py"),
                    "--dataset_root",
                    str(pred_root),
                    "--source_subdir",
                    case["renderer_subdir"],
                    "--out_subdir",
                    case["agent_subdir"],
                    "--building_pattern",
                    str(args.building_pattern),
                ]
            if args.limit > 0:
                cmd += ["--limit", str(int(args.limit))]
            if args.overwrite_agentexec:
                cmd += ["--overwrite"]
            _run(cmd, env=malmo_env)

        if not args.skip_eval:
            cmd = [
                py,
                str(ROOT / "tools" / "evaluate_execution_gap.py"),
                "--gt_root",
                str(gt_root),
                "--pred_root",
                str(pred_root),
                "--renderer_pred_subdir",
                case["renderer_subdir"],
                "--agent_pred_subdir",
                case["agent_subdir"],
                "--out",
                str(metrics_out),
                "--building_pattern",
                str(args.building_pattern),
                "--thresholds_json",
                str(Path(args.thresholds_json).resolve()),
            ]
            if args.limit > 0:
                cmd += ["--limit", str(int(args.limit))]
            _run(cmd, env=malmo_env)

        payload = _load(metrics_out)
        agg = payload.get("aggregate", {})
        renderer = agg.get("renderer", {}).get("metrics", {})
        agent = agg.get("agent", {}).get("metrics", {})
        gap = agg.get("execution_gap", {}).get("metrics", {})
        retention = agg.get("execution_gap", {}).get("metrics_retention_ratio", {})
        rows.append(
            {
                "case_key": case["case_key"],
                "label": case["label"],
                "dataset_name": dataset_name,
                "model_tag": case["model_tag"],
                "renderer_subdir": case["renderer_subdir"],
                "agent_subdir": case["agent_subdir"],
                "metrics_path": str(metrics_out),
                "renderer": renderer,
                "agent": agent,
                "gap": gap,
                "retention": retention,
                "summary": payload.get("summary", {}),
            }
        )

    report_dir = ROOT / "reports" / "final"
    figures_dir = ROOT / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "date_tag": args.date_tag,
        "agentexec_mode": args.agentexec_mode,
        "note": (
            "agentexec is real_chat_command_placement in current run."
            if args.agentexec_mode == "real"
            else (
                "agentexec is creative_hand_place in current run."
                if args.agentexec_mode == "hand"
                else "agentexec is proxy_sanitize_only in current run."
            )
        ),
        "rows": rows,
    }
    bundle_path = figures_dir / f"execution_gap_data_{args.date_tag}.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_paths: List[Path] = []
    if not args.skip_figures:
        figure_paths = _make_figures(rows=rows, out_dir=figures_dir)
        print("[run_execution_gap_suite] wrote figures:")
        for p in figure_paths:
            print(f"  - {p}")

    if not args.skip_report:
        md_path = report_dir / "execution_gap_summary_ja.md"
        _write_report(
            rows=rows,
            md_path=md_path,
            date_tag=args.date_tag,
            figure_paths=figure_paths,
            bundle_path=bundle_path,
            agentexec_mode=args.agentexec_mode,
        )
        print(f"[run_execution_gap_suite] wrote report: {md_path}")

    print(f"[run_execution_gap_suite] wrote bundle: {bundle_path}")
    print("[run_execution_gap_suite] done.")


if __name__ == "__main__":
    main()
