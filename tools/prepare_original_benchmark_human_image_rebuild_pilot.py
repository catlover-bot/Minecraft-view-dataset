#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.evaluate_rebuild_metrics import load_voxels, normalize_block_type


@dataclass(frozen=True)
class PilotCase:
    dataset_split: str
    building_id: str
    difficulty: str
    reason: str

    @property
    def case_id(self) -> str:
        return f"{self.dataset_split}_{self.building_id}"


DEFAULT_PILOT_CASES: Tuple[PilotCase, ...] = (
    PilotCase(
        dataset_split="v1",
        building_id="building_087",
        difficulty="easy",
        reason="v1でdirect IoUが高い単純構造（flat/1F）をeasy代表として採用。",
    ),
    PilotCase(
        dataset_split="v1",
        building_id="building_029",
        difficulty="medium",
        reason="v1の中位帯で2階建てgable系。easy/hardの中間難度を代表。",
    ),
    PilotCase(
        dataset_split="v1",
        building_id="building_014",
        difficulty="medium",
        reason="v1の中位帯でroof向きが異なるケース。v1内の形状バリエーション確保。",
    ),
    PilotCase(
        dataset_split="v1",
        building_id="building_017",
        difficulty="hard",
        reason="v1の低IoU帯。簡素カテゴリでも失敗しやすいケースをhard代表として採用。",
    ),
    PilotCase(
        dataset_split="v4",
        building_id="building_012",
        difficulty="easy",
        reason="v4内で相対的に高IoU。複雑側データでも再建築しやすい基準ケース。",
    ),
    PilotCase(
        dataset_split="v4",
        building_id="building_007",
        difficulty="medium",
        reason="H-shape + tower要素を持つ中位難度。構造複雑性の中間代表。",
    ),
    PilotCase(
        dataset_split="v4",
        building_id="building_076",
        difficulty="hard",
        reason="v4低IoU帯の複雑形状。困難ケース代表として採用。",
    ),
    PilotCase(
        dataset_split="v4",
        building_id="building_086",
        difficulty="hard",
        reason="v4の大規模（高block数）ケース。高複雑・高修復負荷を想定して採用。",
    ),
)


TIME_LIMIT_BY_DIFFICULTY = {
    "easy": 20,
    "medium": 30,
    "hard": 40,
}


CONDITIONS = [
    "image_only",
    "image_plus_description",
    "image_plus_description_plus_structured_ir",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare original benchmark human image->rebuild pilot infrastructure.")
    p.add_argument("--datasets_root", default="datasets")
    p.add_argument("--benchmark_outputs_root", default="outputs/i2t2b")
    p.add_argument("--out_root", default="outputs/human_image_rebuild")
    p.add_argument("--reports_dir", default="reports/final")
    p.add_argument(
        "--comparison_cases_csv",
        default="reports/final/original_benchmark_structured_vs_direct_cases.csv",
        help="Per-case direct/structured metrics used for explainable case selection and comparison templates.",
    )
    p.add_argument("--description_subdir_primary", default="description_openai_gpt_5_mini")
    p.add_argument("--description_subdir_secondary", default="description_anthropic_claude_haiku_4_5_20251001")
    p.add_argument(
        "--structured_subdir_primary",
        default="structured_intermediate_structured_ir_openai_main_orig_20260418",
    )
    p.add_argument(
        "--structured_subdir_secondary",
        default="structured_intermediate_structured_ir_claude_main_orig_20260418",
    )
    p.add_argument("--copy_max_images", type=int, default=0, help="0 means copy all images.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_case_metrics(csv_path: Path) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    if not csv_path.is_file():
        raise SystemExit(f"comparison csv not found: {csv_path}")
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["building"], row["model"])
            out[key] = {
                "direct_iou": float(row["direct_iou"]),
                "structured_iou": float(row["structured_iou"]),
                "direct_f1": float(row["direct_f1"]),
                "structured_f1": float(row["structured_f1"]),
                "direct_material_match": float(row["direct_material_match"]),
                "structured_material_match": float(row["structured_material_match"]),
                "direct_correct_placement_rate": float(row["direct_correct_placement_rate"]),
                "structured_correct_placement_rate": float(row["structured_correct_placement_rate"]),
                "direct_edit_distance_over_gt": float(row["direct_edit_distance_over_gt"]),
                "structured_edit_distance_over_gt": float(row["structured_edit_distance_over_gt"]),
                "delta_iou": float(row["delta_iou"]),
                "delta_edit_distance_over_gt": float(row["delta_edit_distance_over_gt"]),
            }
    return out


def _copy_images(src_dir: Path, dst_dir: Path, copy_max_images: int) -> List[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(src_dir.glob("*.png"))
    if copy_max_images > 0:
        pngs = pngs[:copy_max_images]
    copied: List[str] = []
    for p in pngs:
        out = dst_dir / p.name
        shutil.copy2(p, out)
        copied.append(str(Path("source_images") / p.name))
    return copied


def _gt_allowed_blocks(gt_vox_path: Path) -> List[str]:
    arr = load_voxels(gt_vox_path)
    vals = sorted({normalize_block_type(x) for x in arr.flatten().tolist()})
    vals = [x for x in vals if x != "air"] + ["air"]
    return vals


def _dims_from_bbox(bbox: Dict[str, int]) -> Dict[str, int]:
    return {
        "width": int(bbox["xmax"]) - int(bbox["xmin"]) + 1,
        "height": int(bbox["ymax"]) - int(bbox["ymin"]) + 1,
        "depth": int(bbox["zmax"]) - int(bbox["zmin"]) + 1,
    }


def _task_markdown(case: Dict[str, Any]) -> str:
    cond_lines = "\n".join(f"- `{c}`" for c in case["conditions_supported"])
    return (
        f"# {case['case_id']}\n\n"
        f"- Dataset split: `{case['dataset_split']}`\n"
        f"- Building: `{case['building_id']}`\n"
        f"- Difficulty: `{case['difficulty']}`\n"
        f"- Recommended time limit: `{case['recommended_time_limit_min']} min`\n\n"
        f"## Conditions\n{cond_lines}\n\n"
        f"## Submission (Primary)\n"
        f"`bbox.json` + `voxels.npy` を `submissions/<participant_id>/<condition>/{case['case_id']}/` に保存してください。\n\n"
        f"## Submission (Secondary)\n"
        f"`plan.json` でも提出できます（採点時に自動レンダリングして評価します）。\n"
    )


def _write_docs(out_root: Path, reports_dir: Path, manifest_path: Path, case_rows: List[Dict[str, Any]]) -> None:
    docs_dir = out_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    participant_protocol = """# Original Benchmark Human Image->Rebuild Pilot Protocol

この文書は **実験実施用プロトコル** です。ここには人間被験者の結果は含みません。

## 目的
画像からMinecraft建築を再構成し、GTと比較可能な形式で提出してもらうための小規模パイロットを実施する。

## 条件
- `image_only`
- `image_plus_description`
- `image_plus_description_plus_structured_ir`（任意）

## 参加者向け手順
1. `outputs/human_image_rebuild/case_packages/<case_id>/source_images/` を見る。
2. 条件に応じて `condition_assets/description/` と `condition_assets/structured_intermediate/` を使う。
3. 指定の許可ブロック・制約内で再構成する。
4. 提出先: `outputs/human_image_rebuild/submissions/<participant_id>/<condition>/<case_id>/`

## 提出形式
Primary:
- `bbox.json`
- `voxels.npy`

Secondary:
- `plan.json`（採点時に `voxels.npy` へ変換して評価）

## 評価指標
LLM評価と整合する形で次を算出:
- IoU, F1
- material_match, coarse_material_match
- correct_placement_rate
- repair-effort（additions/deletions/replacements/edit_distance）

## 注意
- このパイロット構築タスクでは人間成績を主張しない。
- プレースホルダ提出は配線確認専用で、研究結果に含めない。
""".strip()

    experimenter_protocol = """# Experimenter Protocol (Human Image->Rebuild Pilot)

## 推奨実施デザイン
- 小規模パイロット: 6-10名
- ケース数: 8ケース（v1=4, v4=4）
- デザイン: 参加者内比較（within-subject）
- セッション分割例:
  - Session A: image_only
  - Session B: image_plus_description
  - Session C (optional): image_plus_description_plus_structured_ir

## 推奨時間
- easy: 20分
- medium: 30分
- hard: 40分

## 実施手順
1. ケース配布: `case_packages/`
2. 提出回収: `submissions/`
3. 採点実行: `tools/score_human_image_rebuild_submissions.py`
4. 比較表更新: `reports/final/original_benchmark_human_image_rebuild_comparison_template.csv`

## 重要ガードレール
- 人間結果と既存ベンチ結果を混ぜない。
- プレースホルダ提出は必ず別ラベルで管理する。
""".strip()

    submission_spec = """# Submission Format Specification

## Primary format (推奨)
必須ファイル:
- `bbox.json`
- `voxels.npy`

`voxels.npy` は軸順 `Y,X,Z`。ブロック名文字列配列。

## Secondary format
- `plan.json`

`plan.json` を提出した場合、採点側で `fill/carve/set` をレンダリングし `voxels.npy` に変換して評価します。

## Path convention
`outputs/human_image_rebuild/submissions/<participant_id>/<condition>/<case_id>/`
""".strip()

    scoring_readme = f"""# Scoring README

## Manifest
`{manifest_path}`

## Main command
```bash
python3 tools/score_human_image_rebuild_submissions.py \\
  --cases_manifest {manifest_path} \\
  --submission_root outputs/human_image_rebuild/submissions \\
  --out_root outputs/human_image_rebuild/scored_submissions
```

## Outputs
- `human_scores.json`
- `human_scores.csv`
- `human_scores_summary.md`
- `human_vs_llm_case_table.csv`

※ このREADMEは評価インフラ説明であり、人間結果の主張ではありません。
""".strip()

    (reports_dir / "original_benchmark_human_image_rebuild_protocol.md").write_text(
        participant_protocol + "\n", encoding="utf-8"
    )
    (docs_dir / "experimenter_protocol.md").write_text(experimenter_protocol + "\n", encoding="utf-8")
    (docs_dir / "submission_format_spec.md").write_text(submission_spec + "\n", encoding="utf-8")
    (docs_dir / "scoring_readme.md").write_text(scoring_readme + "\n", encoding="utf-8")

    easy = sum(1 for x in case_rows if x["difficulty"] == "easy")
    medium = sum(1 for x in case_rows if x["difficulty"] == "medium")
    hard = sum(1 for x in case_rows if x["difficulty"] == "hard")

    setup_summary = (
        "# Original Benchmark Human Image->Rebuild Pilot Setup Summary\n\n"
        "この文書は人手実験の**実施基盤**のまとめです。人間成績の報告は含みません。\n\n"
        "## Scope\n"
        "- datasets: `buildings_100_v1`, `buildings_100_v4`\n"
        f"- selected cases: `{len(case_rows)}` (easy={easy}, medium={medium}, hard={hard})\n"
        "- conditions: image_only / image+description / image+description+structured_ir\n\n"
        "## Output namespace\n"
        "- `outputs/human_image_rebuild/`\n"
        "- `reports/final/original_benchmark_human_image_rebuild_*`\n\n"
        "## Notes\n"
        "- 既存Main/Supplementaryベンチ結果は上書きしていません。\n"
        "- 提出スコアはLLM評価と整合する同系指標で計算します。\n"
    )
    (reports_dir / "original_benchmark_human_image_rebuild_setup_summary.md").write_text(
        setup_summary + "\n", encoding="utf-8"
    )


def _build_case_rows(
    args: argparse.Namespace,
    case_metrics: Dict[Tuple[str, str, str], Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in DEFAULT_PILOT_CASES:
        ds_root = Path(args.datasets_root).resolve() / f"buildings_100_{case.dataset_split}" / case.building_id
        out_root = Path(args.benchmark_outputs_root).resolve() / f"buildings_100_{case.dataset_split}" / case.building_id
        meta_path = ds_root / "meta.json"
        gt_bbox_path = ds_root / "gt" / "bbox.json"
        gt_vox_path = ds_root / "gt" / "voxels.npy"
        img_dir = ds_root / "images"
        if not (meta_path.is_file() and gt_bbox_path.is_file() and gt_vox_path.is_file() and img_dir.is_dir()):
            raise SystemExit(f"missing benchmark assets for {case.case_id}: {ds_root}")

        meta = _load_json(meta_path)
        gt_bbox = _load_json(gt_bbox_path)
        dims = _dims_from_bbox(gt_bbox)

        desc_primary = out_root / args.description_subdir_primary / "description.json"
        desc_secondary = out_root / args.description_subdir_secondary / "description.json"
        ir_primary = out_root / args.structured_subdir_primary / "intermediate.json"
        ir_secondary = out_root / args.structured_subdir_secondary / "intermediate.json"

        conds = ["image_only"]
        if desc_primary.is_file():
            conds.append("image_plus_description")
        if desc_primary.is_file() and ir_primary.is_file():
            conds.append("image_plus_description_plus_structured_ir")

        recommended = TIME_LIMIT_BY_DIFFICULTY.get(case.difficulty, 30)

        openai_metrics = case_metrics.get((case.dataset_split, case.building_id, "openai"))
        claude_metrics = case_metrics.get((case.dataset_split, case.building_id, "claude"))
        if openai_metrics is None or claude_metrics is None:
            raise SystemExit(f"missing direct/structured comparison metrics for {case.case_id}")

        rows.append(
            {
                "case_id": case.case_id,
                "dataset_split": case.dataset_split,
                "building_id": case.building_id,
                "difficulty": case.difficulty,
                "selection_reason": case.reason,
                "style": str(meta.get("style", "")),
                "profile": str(meta.get("profile", "")),
                "num_blocks": int(meta.get("generation", {}).get("num_blocks", 0)),
                "bbox": gt_bbox,
                "dimensions": dims,
                "local_bbox_template": {
                    "xmin": 0,
                    "xmax": dims["width"] - 1,
                    "ymin": 0,
                    "ymax": dims["height"] - 1,
                    "zmin": 0,
                    "zmax": dims["depth"] - 1,
                    "order": "xmin,xmax,ymin,ymax,zmin,zmax",
                    "voxel_axis_order": "Y,X,Z",
                },
                "images_dir": str(img_dir),
                "gt_bbox_path": str(gt_bbox_path),
                "gt_voxels_path": str(gt_vox_path),
                "description_primary_path": str(desc_primary) if desc_primary.is_file() else "",
                "description_secondary_path": str(desc_secondary) if desc_secondary.is_file() else "",
                "structured_primary_path": str(ir_primary) if ir_primary.is_file() else "",
                "structured_secondary_path": str(ir_secondary) if ir_secondary.is_file() else "",
                "conditions_supported": conds,
                "recommended_time_limit_min": recommended,
                "llm_baselines": {
                    "openai_main": openai_metrics,
                    "claude_main": claude_metrics,
                },
            }
        )
    return rows


def _copy_optional(src: str, dst: Path) -> bool:
    if not src:
        return False
    p = Path(src)
    if not p.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    return True


def _write_results_template(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "condition",
                "case_id",
                "submission_path",
                "start_time_iso",
                "end_time_iso",
                "elapsed_minutes",
                "notes",
            ],
        )
        writer.writeheader()
        for r in rows:
            for cond in r["conditions_supported"]:
                writer.writerow(
                    {
                        "participant_id": "",
                        "condition": cond,
                        "case_id": r["case_id"],
                        "submission_path": f"outputs/human_image_rebuild/submissions/<participant_id>/{cond}/{r['case_id']}",
                        "start_time_iso": "",
                        "end_time_iso": "",
                        "elapsed_minutes": "",
                        "notes": "",
                    }
                )


def _write_comparison_template(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            o = r["llm_baselines"]["openai_main"]
            c = r["llm_baselines"]["claude_main"]
            writer.writerow(
                {
                    "case_id": r["case_id"],
                    "dataset_split": r["dataset_split"],
                    "building_id": r["building_id"],
                    "difficulty": r["difficulty"],
                    "selection_reason": r["selection_reason"],
                    "openai_direct_iou": o["direct_iou"],
                    "openai_structured_iou": o["structured_iou"],
                    "openai_direct_f1": o["direct_f1"],
                    "openai_structured_f1": o["structured_f1"],
                    "openai_direct_material_match": o["direct_material_match"],
                    "openai_structured_material_match": o["structured_material_match"],
                    "openai_direct_correct_placement_rate": o["direct_correct_placement_rate"],
                    "openai_structured_correct_placement_rate": o["structured_correct_placement_rate"],
                    "openai_direct_edit_distance_over_gt": o["direct_edit_distance_over_gt"],
                    "openai_structured_edit_distance_over_gt": o["structured_edit_distance_over_gt"],
                    "claude_direct_iou": c["direct_iou"],
                    "claude_structured_iou": c["structured_iou"],
                    "claude_direct_f1": c["direct_f1"],
                    "claude_structured_f1": c["structured_f1"],
                    "claude_direct_material_match": c["direct_material_match"],
                    "claude_structured_material_match": c["structured_material_match"],
                    "claude_direct_correct_placement_rate": c["direct_correct_placement_rate"],
                    "claude_structured_correct_placement_rate": c["structured_correct_placement_rate"],
                    "claude_direct_edit_distance_over_gt": c["direct_edit_distance_over_gt"],
                    "claude_structured_edit_distance_over_gt": c["structured_edit_distance_over_gt"],
                }
            )


def main() -> None:
    args = parse_args()

    datasets_root = Path(args.datasets_root).resolve()
    benchmark_outputs_root = Path(args.benchmark_outputs_root).resolve()
    out_root = Path(args.out_root).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    comparison_cases_csv = Path(args.comparison_cases_csv).resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "case_packages").mkdir(parents=True, exist_ok=True)
    (out_root / "submissions").mkdir(parents=True, exist_ok=True)
    (out_root / "scored_submissions").mkdir(parents=True, exist_ok=True)

    if not datasets_root.is_dir():
        raise SystemExit(f"datasets_root not found: {datasets_root}")
    if not benchmark_outputs_root.is_dir():
        raise SystemExit(f"benchmark_outputs_root not found: {benchmark_outputs_root}")

    case_metrics = _load_case_metrics(comparison_cases_csv)
    case_rows = _build_case_rows(args, case_metrics)

    packaged_rows: List[Dict[str, Any]] = []
    for case in case_rows:
        case_dir = out_root / "case_packages" / case["case_id"]
        if case_dir.exists() and args.overwrite:
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        source_images_dir = case_dir / "source_images"
        copied_images = _copy_images(Path(case["images_dir"]), source_images_dir, args.copy_max_images)

        allowed_blocks = _gt_allowed_blocks(Path(case["gt_voxels_path"]))
        (case_dir / "allowed_blocks.txt").write_text("\n".join(allowed_blocks) + "\n", encoding="utf-8")

        build_constraints = {
            "case_id": case["case_id"],
            "dataset_split": case["dataset_split"],
            "building_id": case["building_id"],
            "difficulty": case["difficulty"],
            "recommended_time_limit_min": case["recommended_time_limit_min"],
            "gt_bbox": case["bbox"],
            "dimensions": case["dimensions"],
            "local_bbox_template": case["local_bbox_template"],
            "voxel_axis_order": "Y,X,Z",
            "allowed_blocks": allowed_blocks,
        }
        _write_json(case_dir / "build_constraints.json", build_constraints)

        # Optional assets
        desc_dir = case_dir / "condition_assets" / "description"
        ir_dir = case_dir / "condition_assets" / "structured_intermediate"
        desc_primary_ok = _copy_optional(case["description_primary_path"], desc_dir / "description_openai_gpt_5_mini.json")
        _copy_optional(case["description_secondary_path"], desc_dir / "description_claude_haiku_4_5.json")
        ir_primary_ok = _copy_optional(case["structured_primary_path"], ir_dir / "intermediate_openai_main.json")
        _copy_optional(case["structured_secondary_path"], ir_dir / "intermediate_claude_main.json")

        submission_template_dir = case_dir / "submission_template"
        submission_template_dir.mkdir(parents=True, exist_ok=True)
        _write_json(submission_template_dir / "bbox.json", case["local_bbox_template"])
        (submission_template_dir / "README.md").write_text(
            (
                "# Submission template\n\n"
                "Primary submit files:\n"
                "- bbox.json\n"
                "- voxels.npy\n\n"
                "Secondary submit file:\n"
                "- plan.json (fill/carve/set).\n"
            ),
            encoding="utf-8",
        )

        task_payload = {
            "case_id": case["case_id"],
            "dataset_split": case["dataset_split"],
            "building_id": case["building_id"],
            "difficulty": case["difficulty"],
            "selection_reason": case["selection_reason"],
            "images": copied_images,
            "conditions_supported": case["conditions_supported"],
            "recommended_time_limit_min": case["recommended_time_limit_min"],
            "allowed_blocks_file": "allowed_blocks.txt",
            "build_constraints_file": "build_constraints.json",
            "submission_format": {
                "primary_required": ["bbox.json", "voxels.npy"],
                "secondary_optional": ["plan.json"],
                "path_template": f"outputs/human_image_rebuild/submissions/<participant_id>/<condition>/{case['case_id']}/",
            },
            "default_condition_assets": {
                "image_plus_description": "condition_assets/description/description_openai_gpt_5_mini.json" if desc_primary_ok else "",
                "image_plus_description_plus_structured_ir": {
                    "description": "condition_assets/description/description_openai_gpt_5_mini.json" if desc_primary_ok else "",
                    "structured_intermediate": "condition_assets/structured_intermediate/intermediate_openai_main.json" if ir_primary_ok else "",
                },
            },
            "llm_baselines_main": case["llm_baselines"],
        }
        _write_json(case_dir / "task.json", task_payload)
        (case_dir / "task.md").write_text(_task_markdown(task_payload) + "\n", encoding="utf-8")

        packaged_rows.append(
            {
                **case,
                "images": copied_images,
                "allowed_blocks": allowed_blocks,
                "package_dir": str(case_dir),
            }
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    cases_manifest_path = reports_dir / "original_benchmark_human_image_rebuild_cases.json"
    results_template_path = reports_dir / "original_benchmark_human_image_rebuild_results_template.csv"
    comparison_template_path = reports_dir / "original_benchmark_human_image_rebuild_comparison_template.csv"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_type": "human_image_rebuild_pilot_infrastructure",
        "scope": "original benchmark only (buildings_100_v1 / buildings_100_v4)",
        "note": "Protocol/setup only. No human performance claims included.",
        "outputs_root": str(out_root),
        "cases": packaged_rows,
        "conditions": CONDITIONS,
        "recommended_design": {
            "participant_count": "6-10 (recommended)",
            "design": "within-subject with counterbalanced condition order",
            "time_limit_by_difficulty_min": TIME_LIMIT_BY_DIFFICULTY,
        },
    }
    _write_json(cases_manifest_path, manifest)
    _write_json(out_root / "case_packages" / "manifest.json", manifest)

    _write_results_template(results_template_path, packaged_rows)
    shutil.copy2(results_template_path, out_root / "results_template.csv")

    _write_comparison_template(comparison_template_path, packaged_rows)

    _write_docs(out_root, reports_dir, cases_manifest_path, packaged_rows)

    print(f"[prepare_original_benchmark_human_image_rebuild_pilot] wrote {cases_manifest_path}")
    print(f"[prepare_original_benchmark_human_image_rebuild_pilot] wrote {results_template_path}")
    print(f"[prepare_original_benchmark_human_image_rebuild_pilot] wrote {comparison_template_path}")
    print(f"[prepare_original_benchmark_human_image_rebuild_pilot] case packages: {out_root / 'case_packages'}")


if __name__ == "__main__":
    main()
