#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare human reconstruction protocol kit for llm_authored_10.")
    p.add_argument("--dataset_root", default="datasets/llm_authored_10")
    p.add_argument("--out_root", default="outputs/llm_authored_10/human_kit")
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--time_limit_min", type=int, default=25)
    return p.parse_args()


def _list_buildings(root: Path, pattern: str, limit: int) -> List[Path]:
    xs = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if limit > 0:
        xs = xs[:limit]
    return xs


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_images(src_dir: Path, dst_dir: Path) -> List[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for p in sorted(src_dir.glob("*.png")):
        out = dst_dir / p.name
        out.write_bytes(p.read_bytes())
        paths.append(str(out))
    return paths


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    cases_dir = out_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    submissions_root = out_root / "submissions"
    submissions_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for bdir in _list_buildings(dataset_root, args.building_pattern, args.limit):
        case_id = bdir.name
        meta = _load_json(bdir / "meta.json") if (bdir / "meta.json").is_file() else {}
        spec = _load_json(bdir / "source_spec.json") if (bdir / "source_spec.json").is_file() else {}
        bbox = _load_json(bdir / "gt" / "bbox.json") if (bdir / "gt" / "bbox.json").is_file() else {}

        case_out = cases_dir / case_id
        images_out = case_out / "images"
        copied_images = _copy_images(bdir / "images", images_out)

        allowed_blocks = spec.get("materials", {}) if isinstance(spec.get("materials"), dict) else {}
        allowed = sorted(set(str(v) for v in allowed_blocks.values() if str(v).strip()))
        if "air" not in allowed:
            allowed.append("air")

        payload = {
            "case_id": case_id,
            "difficulty": meta.get("difficulty", "unknown"),
            "title": spec.get("title", case_id),
            "images": [str(Path("images") / Path(p).name) for p in copied_images],
            "allowed_blocks": allowed,
            "build_area": bbox,
            "submission_format": {
                "required": ["bbox.json", "voxels.npy"],
                "path_template": "submissions/<participant_id>/<condition>/<case_id>/",
            },
            "conditions": [
                "image_only",
                "image_plus_description",
                "image_plus_description_plus_structured_ir",
            ],
            "recommended_time_limit_min": int(args.time_limit_min),
        }
        (case_out / "task.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        rows.append(
            {
                "participant_id": "",
                "condition": "image_only",
                "case_id": case_id,
                "submission_path": f"submissions/<participant_id>/image_only/{case_id}",
                "elapsed_minutes": "",
                "notes": "",
            }
        )

    protocol = """
# Human Reconstruction Protocol (LLM-authored 10-case diagnostic)

This package is protocol/toolkit only. No human results are claimed here.

## Task
Reconstruct each source building from provided images under one of the conditions:
1. image_only
2. image_plus_description
3. image_plus_description_plus_structured_ir

## Constraints
- Use provided allowed block list per case.
- Build inside provided build_area (bbox).
- Recommended time limit: {time_limit} minutes per case.

## Submission format
For each case, submit:
- bbox.json
- voxels.npy
Path template:
- submissions/<participant_id>/<condition>/<case_id>/

## Scoring
Use `tools/evaluate_human_rebuild_submissions.py` to score with the same rebuild metrics framework.
""".strip().format(time_limit=int(args.time_limit_min))
    (out_root / "protocol.md").write_text(protocol + "\n", encoding="utf-8")

    csv_path = out_root / "results_template.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["participant_id", "condition", "case_id", "submission_path", "elapsed_minutes", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "out_root": str(out_root),
        "cases_dir": str(cases_dir),
        "submissions_root": str(submissions_root),
        "protocol": str(out_root / "protocol.md"),
        "results_template": str(csv_path),
        "note": "Protocol only. No human outcomes included.",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare_llm_authored_human_kit] wrote {out_root}")


if __name__ == "__main__":
    main()
