#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.nbt_structure_utils import extract_structure_to_voxels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Minecraft-native human submissions (structure.nbt / zip) into canonical bbox.json + voxels.npy"
    )
    p.add_argument(
        "--cases_manifest",
        default="reports/final/original_benchmark_human_minecraft_rebuild_cases.json",
        help="Human Minecraft pilot manifest.",
    )
    p.add_argument("--submission_root", default="outputs/human_minecraft_rebuild/submissions")
    p.add_argument("--out_root", default="outputs/human_minecraft_rebuild/converted_submissions")
    p.add_argument("--participant_glob", default="*")
    p.add_argument("--condition_glob", default="*")
    p.add_argument("--case_glob", default="*")
    p.add_argument("--allow_zip", action="store_true", help="Allow zip fallback when structure.nbt is absent.")
    p.add_argument(
        "--strict_expected_dims",
        action="store_true",
        help="Mark conversion invalid when submitted structure dimensions do not match expected case dimensions.",
    )
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_structure_nbt(case_dir: Path) -> Optional[Path]:
    direct = case_dir / "structure.nbt"
    if direct.is_file():
        return direct
    candidates = sorted([p for p in case_dir.glob("*.nbt") if p.is_file()])
    return candidates[0] if candidates else None


def _extract_nbt_from_zip(case_dir: Path, work_dir: Path) -> Tuple[Optional[Path], str]:
    zips = []
    for name in ("structure.zip",):
        p = case_dir / name
        if p.is_file():
            zips.append(p)
    zips.extend(sorted([p for p in case_dir.glob("*.zip") if p.is_file() and p.name != "structure.zip"]))

    for zpath in zips:
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".nbt")]
                if not names:
                    continue
                pick = sorted(names)[0]
                out_nbt = work_dir / "extracted" / zpath.stem / Path(pick).name
                out_nbt.parent.mkdir(parents=True, exist_ok=True)
                out_nbt.write_bytes(zf.read(pick))
                return out_nbt, str(zpath)
        except Exception:
            continue
    return None, ""


def _expected_dims(case: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    dims = case.get("dimensions") if isinstance(case.get("dimensions"), dict) else None
    if not dims:
        return None
    try:
        return (int(dims["width"]), int(dims["height"]), int(dims["depth"]))
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    cases_manifest = Path(args.cases_manifest).resolve()
    submission_root = Path(args.submission_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not cases_manifest.is_file():
        raise SystemExit(f"cases_manifest not found: {cases_manifest}")
    if not submission_root.is_dir():
        raise SystemExit(f"submission_root not found: {submission_root}")

    manifest = _load_json(cases_manifest)
    cases = manifest.get("cases", []) if isinstance(manifest.get("cases"), list) else []
    case_map = {str(c.get("case_id")): c for c in cases if str(c.get("case_id", "")).strip()}
    if not case_map:
        raise SystemExit(f"No valid case entries in manifest: {cases_manifest}")

    out_root.mkdir(parents=True, exist_ok=True)

    converted_items: List[Dict[str, Any]] = []
    invalid_items: List[Dict[str, Any]] = []

    participant_dirs = sorted([p for p in submission_root.glob(args.participant_glob) if p.is_dir()])
    for pdir in participant_dirs:
        participant_id = pdir.name
        condition_dirs = sorted([c for c in pdir.glob(args.condition_glob) if c.is_dir()])
        for cdir in condition_dirs:
            condition = cdir.name
            case_dirs = sorted([d for d in cdir.glob(args.case_glob) if d.is_dir()])
            for case_dir in case_dirs:
                case_id = case_dir.name
                case = case_map.get(case_id)
                if case is None:
                    invalid_items.append(
                        {
                            "participant_id": participant_id,
                            "condition": condition,
                            "case_id": case_id,
                            "submission_path": str(case_dir),
                            "status": "invalid_case_id",
                            "reason": "case_id not found in manifest",
                        }
                    )
                    continue

                chosen_nbt: Optional[Path] = _find_structure_nbt(case_dir)
                chosen_from_zip = ""
                source_format = "minecraft_structure_nbt"

                if chosen_nbt is None and args.allow_zip:
                    chosen_nbt, chosen_from_zip = _extract_nbt_from_zip(case_dir, out_root)
                    if chosen_nbt is not None:
                        source_format = "zip_with_structure_nbt"

                if chosen_nbt is None:
                    invalid_items.append(
                        {
                            "participant_id": participant_id,
                            "condition": condition,
                            "case_id": case_id,
                            "submission_path": str(case_dir),
                            "status": "missing_minecraft_artifact",
                            "reason": "structure.nbt not found (and zip fallback not available or empty)",
                        }
                    )
                    continue

                out_case_dir = out_root / participant_id / condition / case_id
                out_case_dir.mkdir(parents=True, exist_ok=True)

                try:
                    extracted = extract_structure_to_voxels(chosen_nbt)
                except Exception as exc:  # noqa: BLE001
                    invalid_items.append(
                        {
                            "participant_id": participant_id,
                            "condition": condition,
                            "case_id": case_id,
                            "submission_path": str(case_dir),
                            "status": "parse_failed",
                            "reason": str(exc),
                            "artifact": str(chosen_nbt),
                        }
                    )
                    continue

                voxels = extracted.voxels
                bbox = extracted.bbox
                sy, sx, sz = int(voxels.shape[0]), int(voxels.shape[1]), int(voxels.shape[2])

                expected = _expected_dims(case)
                dims_match = True
                dims_reason = ""
                if expected is not None:
                    ew, eh, ed = expected
                    dims_match = (sx == ew and sy == eh and sz == ed)
                    if not dims_match:
                        dims_reason = f"submitted_dims=({sx},{sy},{sz}) expected=({ew},{eh},{ed})"

                if args.strict_expected_dims and not dims_match:
                    invalid_items.append(
                        {
                            "participant_id": participant_id,
                            "condition": condition,
                            "case_id": case_id,
                            "submission_path": str(case_dir),
                            "status": "dimension_mismatch",
                            "reason": dims_reason,
                            "artifact": str(chosen_nbt),
                        }
                    )
                    continue

                bbox_path = out_case_dir / "bbox.json"
                vox_path = out_case_dir / "voxels.npy"
                np.save(vox_path, voxels)
                _write_json(bbox_path, bbox)

                submission_meta_path = case_dir / "submission_meta.json"
                submission_meta: Dict[str, Any] = {}
                if submission_meta_path.is_file():
                    try:
                        submission_meta = _load_json(submission_meta_path)
                    except Exception:
                        submission_meta = {"_raw_text": submission_meta_path.read_text(encoding="utf-8", errors="ignore")}

                conversion_meta = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "participant_id": participant_id,
                    "condition": condition,
                    "case_id": case_id,
                    "submission_path": str(case_dir),
                    "source_format": source_format,
                    "source_artifact": str(chosen_nbt),
                    "source_zip": chosen_from_zip,
                    "source_build_origin": "minecraft_human_submission",
                    "source_image_origin": "minecraft_capture_case_package",
                    "converted_canonical_format": {
                        "bbox": str(bbox_path),
                        "voxels": str(vox_path),
                        "voxel_axis_order": "Y,X,Z",
                    },
                    "shape": {"y": sy, "x": sx, "z": sz},
                    "expected_dims": (
                        {"width": expected[0], "height": expected[1], "depth": expected[2]} if expected else {}
                    ),
                    "dimensions_match_expected": dims_match,
                    "dimension_check_note": dims_reason,
                    "nbt_extract_metadata": extracted.metadata,
                    "submission_meta": submission_meta,
                    "any_fallback_used": source_format != "minecraft_structure_nbt",
                }
                _write_json(out_case_dir / "conversion_meta.json", conversion_meta)

                converted_items.append(
                    {
                        "participant_id": participant_id,
                        "condition": condition,
                        "case_id": case_id,
                        "submission_path": str(case_dir),
                        "converted_path": str(out_case_dir),
                        "source_format": source_format,
                        "dimensions_match_expected": dims_match,
                        "status": "ok",
                    }
                )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases_manifest": str(cases_manifest),
        "submission_root": str(submission_root),
        "converted_root": str(out_root),
        "note": "Infrastructure conversion output only. Not a human study result claim.",
        "coverage": {
            "participants_detected": len(participant_dirs),
            "converted_submissions": len(converted_items),
            "invalid_submissions": len(invalid_items),
        },
        "converted_items": converted_items,
        "invalid_items": invalid_items,
    }
    summary_path = out_root / "conversion_summary.json"
    _write_json(summary_path, summary)

    csv_path = out_root / "conversion_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "participant_id",
            "condition",
            "case_id",
            "status",
            "source_format",
            "dimensions_match_expected",
            "submission_path",
            "converted_path",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in converted_items:
            writer.writerow(
                {
                    "participant_id": row.get("participant_id", ""),
                    "condition": row.get("condition", ""),
                    "case_id": row.get("case_id", ""),
                    "status": row.get("status", ""),
                    "source_format": row.get("source_format", ""),
                    "dimensions_match_expected": row.get("dimensions_match_expected", ""),
                    "submission_path": row.get("submission_path", ""),
                    "converted_path": row.get("converted_path", ""),
                    "reason": "",
                }
            )
        for row in invalid_items:
            writer.writerow(
                {
                    "participant_id": row.get("participant_id", ""),
                    "condition": row.get("condition", ""),
                    "case_id": row.get("case_id", ""),
                    "status": row.get("status", ""),
                    "source_format": row.get("source_format", ""),
                    "dimensions_match_expected": row.get("dimensions_match_expected", ""),
                    "submission_path": row.get("submission_path", ""),
                    "converted_path": row.get("converted_path", ""),
                    "reason": row.get("reason", ""),
                }
            )

    print(f"[convert_human_minecraft_submissions] wrote {summary_path}")
    print(f"[convert_human_minecraft_submissions] wrote {csv_path}")


if __name__ == "__main__":
    main()
