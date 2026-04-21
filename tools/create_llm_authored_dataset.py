#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from building.gt_export import export_ground_truth
from building.llm_authored import LLMAuthoredCase, build_spec_to_building
from tools.voxel_render_utils import render_views_to_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create datasets/llm_authored_10 from source spec JSON.")
    p.add_argument("--spec_json", default="datasets/llm_authored_10/source_specs/source_specs.json")
    p.add_argument("--out_root", default="datasets/llm_authored_10")
    p.add_argument("--views", type=int, default=10)
    p.add_argument("--image_size", nargs=2, type=int, default=[960, 540], metavar=("W", "H"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_specs(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    spec_json = Path(args.spec_json).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    payload = _load_specs(spec_json)
    source_provider = str(payload.get("author_provider", "")).strip()
    source_model = str(payload.get("author_model", "")).strip()
    source_condition = str(payload.get("source_condition", "shared_source")).strip() or "shared_source"

    cases = payload.get("cases", []) if isinstance(payload.get("cases"), list) else []
    if len(cases) < 1:
        raise SystemExit(f"No cases in {spec_json}")

    index_rows: List[Dict[str, Any]] = []
    difficulty_count = {"simple": 0, "medium": 0, "complex": 0}

    for i, raw in enumerate(cases, start=1):
        if not isinstance(raw, dict):
            continue
        case_id = str(raw.get("case_id", f"llm_case_{i:03d}")).strip() or f"llm_case_{i:03d}"
        case_dir = out_root / case_id

        if case_dir.exists() and not args.overwrite:
            if (case_dir / "meta.json").is_file() and (case_dir / "gt" / "voxels.npy").is_file():
                print(f"[create_llm_authored_dataset] skip {case_id} (exists)")
                continue

        authored_case = LLMAuthoredCase(
            case_id=case_id,
            title=str(raw.get("title", case_id)).strip() or case_id,
            difficulty=str(raw.get("difficulty", "medium")).strip().lower(),
            author_provider=source_provider,
            author_model=source_model,
            source_condition=source_condition,
            spec=raw,
        )
        building = build_spec_to_building(authored_case, origin=(0, 4, 0), style_id=i)

        images_dir = case_dir / "images"
        gt_dir = case_dir / "gt"
        logs_dir = case_dir / "logs"
        images_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        export_ground_truth(gt_dir, building.bbox, building.blocks)
        views = render_views_to_dir(
            building.blocks,
            building.bbox,
            out_dir=images_dir,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            views=int(args.views),
            prefix="rgb",
        )

        spec_payload = {
            **raw,
            "case_id": case_id,
            "author_provider": source_provider,
            "author_model": source_model,
            "source_condition": source_condition,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(case_dir / "source_spec.json", spec_payload)

        block_items = [
            {"x": int(x), "y": int(y), "z": int(z), "block": block}
            for x, y, z, block in building.blocks
        ]
        _write_json(
            case_dir / "source_blocks.json",
            {
                "case_id": case_id,
                "bbox": building.bbox,
                "blocks": block_items,
                "count": len(block_items),
                "generated_by": "tools/create_llm_authored_dataset.py",
            },
        )

        meta_payload = {
            "case_id": case_id,
            "seed": i,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "bbox": building.bbox,
            "style": building.style,
            "palette": building.palette_name,
            "difficulty": authored_case.difficulty,
            "source_author": {
                "provider": source_provider,
                "model": source_model,
                "condition": source_condition,
            },
            "views": views,
        }
        _write_json(case_dir / "meta.json", meta_payload)

        difficulty = authored_case.difficulty if authored_case.difficulty in difficulty_count else "medium"
        difficulty_count[difficulty] += 1
        index_rows.append(
            {
                "case_id": case_id,
                "difficulty": difficulty,
                "title": authored_case.title,
                "path": str(case_dir),
                "bbox": building.bbox,
                "num_blocks": len(building.blocks),
            }
        )
        print(f"[create_llm_authored_dataset] wrote {case_id}")

    _write_json(
        out_root / "dataset_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "study": "llm_authored_10_diagnostic",
            "dataset_root": str(out_root),
            "source_spec_json": str(spec_json),
            "source_author": {
                "provider": source_provider,
                "model": source_model,
                "condition": source_condition,
            },
            "difficulty_count": difficulty_count,
            "cases": index_rows,
        },
    )

    print(f"[create_llm_authored_dataset] done: {out_root}")


if __name__ == "__main__":
    main()
