#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from tools.evaluate_rebuild_metrics import load_voxels
from tools.nbt_structure_utils import write_structure_nbt_from_voxels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a non-human placeholder Minecraft-native submission for infrastructure validation only."
    )
    p.add_argument(
        "--cases_manifest",
        default="reports/final/original_benchmark_human_minecraft_rebuild_cases.json",
    )
    p.add_argument("--submission_root", default="outputs/human_minecraft_rebuild/submissions")
    p.add_argument("--participant_id", default="validation_placeholder_minecraft")
    p.add_argument("--condition", default="image_only")
    p.add_argument("--case_id", default="", help="If omitted, the first case in manifest is used.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.cases_manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"cases_manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path)
    cases = manifest.get("cases", []) if isinstance(manifest.get("cases"), list) else []
    if not cases:
        raise SystemExit(f"No cases in manifest: {manifest_path}")

    case_map = {str(c.get("case_id")): c for c in cases if str(c.get("case_id", "")).strip()}

    case_id = args.case_id.strip()
    if not case_id:
        case_id = str(cases[0].get("case_id"))
    case = case_map.get(case_id)
    if case is None:
        raise SystemExit(f"case_id not found in manifest: {case_id}")

    gt_vox_path = Path(str(case.get("gt_voxels_path", ""))).resolve()
    if not gt_vox_path.is_file():
        raise SystemExit(f"gt_voxels_path missing for case {case_id}: {gt_vox_path}")

    vox = load_voxels(gt_vox_path)

    out_dir = Path(args.submission_root).resolve() / args.participant_id / args.condition / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    nbt_path = out_dir / "structure.nbt"
    if nbt_path.exists() and not args.overwrite:
        raise SystemExit(f"structure.nbt already exists (use --overwrite): {nbt_path}")

    write_structure_nbt_from_voxels(nbt_path, voxels=vox, compress_gzip=True)

    meta = {
        "participant_id": args.participant_id,
        "case_id": case_id,
        "condition": args.condition,
        "minecraft_version": "validation_only",
        "export_format": "structure.nbt",
        "notes": "Infrastructure validation only. This is GT-derived synthetic placeholder, not human output.",
        "infrastructure_validation_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_gt_voxels": str(gt_vox_path),
    }
    _write_json(out_dir / "submission_meta.json", meta)

    (out_dir / "README.md").write_text(
        (
            "# Placeholder submission\n\n"
            "This directory was generated for conversion/scoring pipeline validation only.\n"
            "It is NOT a human participant result.\n"
        ),
        encoding="utf-8",
    )

    print(f"[make_placeholder_human_minecraft_submission] wrote {nbt_path}")
    print(f"[make_placeholder_human_minecraft_submission] wrote {out_dir / 'submission_meta.json'}")


if __name__ == "__main__":
    main()
