#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare Minecraft-native human image->rebuild pilot package from existing original-benchmark pilot cases."
    )
    p.add_argument(
        "--source_manifest",
        default="reports/final/original_benchmark_human_image_rebuild_cases.json",
        help="Existing voxel-based pilot manifest used as source case selection and baseline metadata.",
    )
    p.add_argument(
        "--source_case_packages",
        default="outputs/human_image_rebuild/case_packages",
        help="Existing case package directory to reuse source images/assets.",
    )
    p.add_argument("--out_root", default="outputs/human_minecraft_rebuild")
    p.add_argument("--reports_dir", default="reports/final")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_results_template(path: Path, cases: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "condition",
                "case_id",
                "submission_path",
                "export_type",
                "start_time_iso",
                "end_time_iso",
                "elapsed_minutes",
                "notes",
            ],
        )
        writer.writeheader()
        for case in cases:
            cid = str(case["case_id"])
            conds = case.get("conditions_supported", ["image_only"])
            for cond in conds:
                writer.writerow(
                    {
                        "participant_id": "",
                        "condition": cond,
                        "case_id": cid,
                        "submission_path": f"outputs/human_minecraft_rebuild/submissions/<participant_id>/{cond}/{cid}",
                        "export_type": "structure.nbt",
                        "start_time_iso": "",
                        "end_time_iso": "",
                        "elapsed_minutes": "",
                        "notes": "",
                    }
                )


def _write_comparison_template(path: Path, cases: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "dataset_split",
        "building_id",
        "difficulty",
        "selection_reason",
        "openai_direct_iou",
        "openai_structured_iou",
        "openai_direct_f1",
        "openai_structured_f1",
        "openai_direct_material_match",
        "openai_structured_material_match",
        "openai_direct_correct_placement_rate",
        "openai_structured_correct_placement_rate",
        "openai_direct_edit_distance_over_gt",
        "openai_structured_edit_distance_over_gt",
        "claude_direct_iou",
        "claude_structured_iou",
        "claude_direct_f1",
        "claude_structured_f1",
        "claude_direct_material_match",
        "claude_structured_material_match",
        "claude_direct_correct_placement_rate",
        "claude_structured_correct_placement_rate",
        "claude_direct_edit_distance_over_gt",
        "claude_structured_edit_distance_over_gt",
        "human_image_only_iou",
        "human_image_only_f1",
        "human_image_only_material_match",
        "human_image_only_correct_placement_rate",
        "human_image_only_edit_distance_over_gt",
        "human_image_plus_description_iou",
        "human_image_plus_description_f1",
        "human_image_plus_description_material_match",
        "human_image_plus_description_correct_placement_rate",
        "human_image_plus_description_edit_distance_over_gt",
        "human_image_plus_description_plus_structured_ir_iou",
        "human_image_plus_description_plus_structured_ir_f1",
        "human_image_plus_description_plus_structured_ir_material_match",
        "human_image_plus_description_plus_structured_ir_correct_placement_rate",
        "human_image_plus_description_plus_structured_ir_edit_distance_over_gt",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            b = case.get("llm_baselines", {}) if isinstance(case.get("llm_baselines"), dict) else {}
            o = b.get("openai_main", {}) if isinstance(b.get("openai_main"), dict) else {}
            c = b.get("claude_main", {}) if isinstance(b.get("claude_main"), dict) else {}
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "dataset_split": case.get("dataset_split", ""),
                    "building_id": case.get("building_id", ""),
                    "difficulty": case.get("difficulty", ""),
                    "selection_reason": case.get("selection_reason", ""),
                    "openai_direct_iou": o.get("direct_iou", ""),
                    "openai_structured_iou": o.get("structured_iou", ""),
                    "openai_direct_f1": o.get("direct_f1", ""),
                    "openai_structured_f1": o.get("structured_f1", ""),
                    "openai_direct_material_match": o.get("direct_material_match", ""),
                    "openai_structured_material_match": o.get("structured_material_match", ""),
                    "openai_direct_correct_placement_rate": o.get("direct_correct_placement_rate", ""),
                    "openai_structured_correct_placement_rate": o.get("structured_correct_placement_rate", ""),
                    "openai_direct_edit_distance_over_gt": o.get("direct_edit_distance_over_gt", ""),
                    "openai_structured_edit_distance_over_gt": o.get("structured_edit_distance_over_gt", ""),
                    "claude_direct_iou": c.get("direct_iou", ""),
                    "claude_structured_iou": c.get("structured_iou", ""),
                    "claude_direct_f1": c.get("direct_f1", ""),
                    "claude_structured_f1": c.get("structured_f1", ""),
                    "claude_direct_material_match": c.get("direct_material_match", ""),
                    "claude_structured_material_match": c.get("structured_material_match", ""),
                    "claude_direct_correct_placement_rate": c.get("direct_correct_placement_rate", ""),
                    "claude_structured_correct_placement_rate": c.get("structured_correct_placement_rate", ""),
                    "claude_direct_edit_distance_over_gt": c.get("direct_edit_distance_over_gt", ""),
                    "claude_structured_edit_distance_over_gt": c.get("structured_edit_distance_over_gt", ""),
                }
            )


def _write_docs(
    out_root: Path,
    reports_dir: Path,
    manifest_path: Path,
    case_count: int,
    difficulty_counts: Dict[str, int],
) -> None:
    docs_dir = out_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    participant_protocol = f"""# Original Benchmark Human Minecraft Rebuild Protocol

この文書は **実験実施インフラ** の定義です。人間成績は含みません。

## 目的
提示画像を見て、参加者がMinecraft内で建築を再現し、Minecraftネイティブ提出物（Structure Block `.nbt`）を提出する。

## 条件
- `image_only`
- `image_plus_description`
- `image_plus_description_plus_structured_ir`（任意）

## 参加者向け手順（要約）
1. `outputs/human_minecraft_rebuild/case_packages/<case_id>/source_images/` を参照。
2. 条件に応じて `condition_assets/description` / `condition_assets/structured_intermediate` を使用。
3. クリエイティブモードで、`build_constraints.json` のローカル座標サイズに合わせて再構築。
4. Structure Block で構造をエクスポートし、`structure.nbt` を提出。
5. `submission_meta.json` を同梱して提出。

## 提出先
`outputs/human_minecraft_rebuild/submissions/<participant_id>/<condition>/<case_id>/`

必須:
- `structure.nbt`
- `submission_meta.json`

任意:
- `structure.zip`（`structure.nbt` を含むzip）

## 評価
提出物は `bbox.json + voxels.npy` に変換後、既存LLM系と整合する同系列指標で採点します。
- IoU, F1
- material_match, coarse_material_match
- correct_placement_rate
- repair-effort（additions/deletions/replacements/edit_distance）

## 注意
- 本タスクはインフラ整備のみであり、人間性能の主張は行いません。
- 検証用プレースホルダ提出は研究結果に含めません。
""".strip()

    experimenter_protocol = """# Experimenter Protocol (Human Minecraft Rebuild Pilot)

## 推奨デザイン
- 参加者: 6-10名（パイロット）
- ケース: 8ケース（v1=4, v4=4）
- デザイン: within-subject（条件順はカウンターバランス）

## 推奨時間
- easy: 20分
- medium: 30分
- hard: 40分

## 実施手順
1. ケース配布: `outputs/human_minecraft_rebuild/case_packages/`
2. 提出回収: `outputs/human_minecraft_rebuild/submissions/`
3. 変換: `tools/convert_human_minecraft_submissions.py`
4. 採点: `tools/score_human_image_rebuild_submissions.py`（変換出力を入力）

## ガードレール
- 既存ベンチ（Main/Supplementary/Execution-gap）とは別管理。
- placeholder結果を人間結果として扱わない。
""".strip()

    submission_spec = """# Submission Format Specification (Minecraft-native)

## Primary format
提出ディレクトリ:
`outputs/human_minecraft_rebuild/submissions/<participant_id>/<condition>/<case_id>/`

必須ファイル:
- `structure.nbt`
- `submission_meta.json`

`structure.nbt` はStructure Blockエクスポート（Java Edition）を想定。

## Secondary format
- `structure.zip`（zip内に`.nbt`が1つ以上。最初に見つかった`.nbt`を使用）

## submission_meta.json 最低項目
- `participant_id`
- `case_id`
- `condition`
- `minecraft_version`
- `notes`

## 変換後の内部形式
採点前に以下へ変換:
- `bbox.json`
- `voxels.npy`（軸順Y,X,Z）
""".strip()

    scoring_readme = f"""# Scoring README (Minecraft-native human submissions)

## 1) Convert Minecraft-native submission artifacts
```bash
python3 tools/convert_human_minecraft_submissions.py \
  --cases_manifest {manifest_path} \
  --submission_root outputs/human_minecraft_rebuild/submissions \
  --out_root outputs/human_minecraft_rebuild/converted_submissions
```

## 2) Score converted submissions
```bash
python3 tools/score_human_image_rebuild_submissions.py \
  --cases_manifest {manifest_path} \
  --submission_root outputs/human_minecraft_rebuild/converted_submissions \
  --out_root outputs/human_minecraft_rebuild/scored_submissions
```

出力はインフラ検証目的。人間成績の主張には直接使いません。
""".strip()

    setup_summary = (
        "# Original Benchmark Human Minecraft Rebuild Setup Summary\n\n"
        "この文書は人間実験実施基盤のまとめです（人間成績は未収集）。\n\n"
        "## Scope\n"
        "- datasets: `buildings_100_v1`, `buildings_100_v4`\n"
        f"- selected cases: `{case_count}` (easy={difficulty_counts.get('easy', 0)}, medium={difficulty_counts.get('medium', 0)}, hard={difficulty_counts.get('hard', 0)})\n"
        "- conditions: image_only / image+description / image+description+structured_ir\n\n"
        "## Output namespace\n"
        "- `outputs/human_minecraft_rebuild/`\n"
        "- `reports/final/original_benchmark_human_minecraft_rebuild_*`\n\n"
        "## Notes\n"
        "- 既存ベンチ結果を上書きしない分離運用。\n"
        "- Minecraftネイティブ提出物（structure.nbt）を評価用voxelへ変換して採点。\n"
    )

    (reports_dir / "original_benchmark_human_minecraft_rebuild_protocol.md").write_text(
        participant_protocol + "\n", encoding="utf-8"
    )
    (reports_dir / "original_benchmark_human_minecraft_rebuild_setup_summary.md").write_text(
        setup_summary, encoding="utf-8"
    )
    (docs_dir / "experimenter_protocol.md").write_text(experimenter_protocol + "\n", encoding="utf-8")
    (docs_dir / "submission_format_spec.md").write_text(submission_spec + "\n", encoding="utf-8")
    (docs_dir / "scoring_readme.md").write_text(scoring_readme + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    source_manifest = Path(args.source_manifest).resolve()
    source_case_packages = Path(args.source_case_packages).resolve()
    out_root = Path(args.out_root).resolve()
    reports_dir = Path(args.reports_dir).resolve()

    if not source_manifest.is_file():
        raise SystemExit(f"source_manifest not found: {source_manifest}")
    if not source_case_packages.is_dir():
        raise SystemExit(f"source_case_packages not found: {source_case_packages}")

    src = _load_json(source_manifest)
    src_cases = src.get("cases", []) if isinstance(src.get("cases"), list) else []
    if not src_cases:
        raise SystemExit(f"No cases in source_manifest: {source_manifest}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "case_packages").mkdir(parents=True, exist_ok=True)
    (out_root / "submissions").mkdir(parents=True, exist_ok=True)
    (out_root / "converted_submissions").mkdir(parents=True, exist_ok=True)
    (out_root / "scored_submissions").mkdir(parents=True, exist_ok=True)

    packaged_cases: List[Dict[str, Any]] = []

    for case in src_cases:
        cid = str(case.get("case_id", "")).strip()
        if not cid:
            continue
        src_case_dir = source_case_packages / cid
        if not src_case_dir.is_dir():
            raise SystemExit(f"source case package missing: {src_case_dir}")

        dst_case_dir = out_root / "case_packages" / cid
        if dst_case_dir.exists() and args.overwrite:
            shutil.rmtree(dst_case_dir)
        if not dst_case_dir.exists():
            shutil.copytree(src_case_dir, dst_case_dir)

        # Enrich task metadata with Minecraft-native submission instructions.
        task_path = dst_case_dir / "task.json"
        task = _load_json(task_path) if task_path.is_file() else {}

        local_bbox = {}
        bc_path = dst_case_dir / "build_constraints.json"
        if bc_path.is_file():
            bc = _load_json(bc_path)
            if isinstance(bc.get("local_bbox_template"), dict):
                local_bbox = bc["local_bbox_template"]

        task["submission_format_minecraft"] = {
            "primary": {
                "type": "minecraft_structure_nbt",
                "required_files": ["structure.nbt", "submission_meta.json"],
            },
            "secondary": {
                "type": "zip_with_structure_nbt",
                "required_files": ["structure.zip", "submission_meta.json"],
            },
            "path_template": f"outputs/human_minecraft_rebuild/submissions/<participant_id>/<condition>/{cid}/",
            "expected_local_bbox": local_bbox,
            "expected_voxel_axis_order": "Y,X,Z",
        }
        task["submission_format"] = task.get("submission_format", {
            "primary_required": ["bbox.json", "voxels.npy"],
            "secondary_optional": ["plan.json"],
        })
        _write_json(task_path, task)

        minecraft_template_dir = dst_case_dir / "minecraft_submission_template"
        minecraft_template_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            minecraft_template_dir / "submission_meta.json",
            {
                "participant_id": "<fill_me>",
                "case_id": cid,
                "condition": "image_only",
                "minecraft_version": "<fill_me>",
                "export_format": "structure.nbt",
                "notes": "",
                "infrastructure_validation_only": False,
            },
        )
        (minecraft_template_dir / "README.md").write_text(
            (
                "# Minecraft submission template\n\n"
                "Primary files to submit:\n"
                "- structure.nbt\n"
                "- submission_meta.json\n\n"
                "Optional secondary:\n"
                "- structure.zip (must include .nbt file)\n"
            ),
            encoding="utf-8",
        )

        task_minecraft_md = (
            f"# {cid} (Minecraft-native)\n\n"
            "1. source_images を参照して、Minecraft内で再建築してください。\n"
            "2. build_constraints.json のサイズに合わせ、ローカル座標原点(0,0,0)基準で作業してください。\n"
            "3. Structure Blockで `structure.nbt` を書き出してください。\n"
            "4. submission_meta.json と合わせて提出してください。\n"
        )
        (dst_case_dir / "task_minecraft.md").write_text(task_minecraft_md, encoding="utf-8")

        packaged = dict(case)
        packaged["minecraft_submission"] = {
            "primary_format": "minecraft_structure_nbt",
            "secondary_format": "zip_with_structure_nbt",
            "path_template": f"outputs/human_minecraft_rebuild/submissions/<participant_id>/<condition>/{cid}/",
            "required_files_primary": ["structure.nbt", "submission_meta.json"],
            "required_files_secondary": ["structure.zip", "submission_meta.json"],
            "expected_local_bbox": local_bbox,
            "conversion_target": {
                "bbox": "bbox.json",
                "voxels": "voxels.npy",
                "voxel_axis_order": "Y,X,Z",
            },
        }
        packaged["package_dir"] = str(dst_case_dir)
        packaged_cases.append(packaged)

    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_type": "human_minecraft_rebuild_pilot_infrastructure",
        "scope": "original benchmark only (buildings_100_v1 / buildings_100_v4)",
        "note": "Protocol/setup only. No human performance claims included.",
        "source_manifest": str(source_manifest),
        "outputs_root": str(out_root),
        "conditions": src.get(
            "conditions",
            ["image_only", "image_plus_description", "image_plus_description_plus_structured_ir"],
        ),
        "cases": packaged_cases,
    }

    reports_manifest = reports_dir / "original_benchmark_human_minecraft_rebuild_cases.json"
    _write_json(reports_manifest, manifest)
    _write_json(out_root / "case_packages" / "manifest.json", manifest)

    results_template_path = reports_dir / "original_benchmark_human_minecraft_rebuild_results_template.csv"
    comparison_template_path = reports_dir / "original_benchmark_human_minecraft_rebuild_comparison_template.csv"
    _write_results_template(results_template_path, packaged_cases)
    _write_comparison_template(comparison_template_path, packaged_cases)
    shutil.copy2(results_template_path, out_root / "results_template.csv")

    diffs: Dict[str, int] = {}
    for c in packaged_cases:
        d = str(c.get("difficulty", "unknown"))
        diffs[d] = diffs.get(d, 0) + 1

    _write_docs(
        out_root=out_root,
        reports_dir=reports_dir,
        manifest_path=reports_manifest,
        case_count=len(packaged_cases),
        difficulty_counts=diffs,
    )

    print(f"[prepare_original_benchmark_human_minecraft_rebuild_pilot] wrote {reports_manifest}")
    print(f"[prepare_original_benchmark_human_minecraft_rebuild_pilot] wrote {results_template_path}")
    print(f"[prepare_original_benchmark_human_minecraft_rebuild_pilot] wrote {comparison_template_path}")
    print(f"[prepare_original_benchmark_human_minecraft_rebuild_pilot] case packages: {out_root / 'case_packages'}")


if __name__ == "__main__":
    main()
