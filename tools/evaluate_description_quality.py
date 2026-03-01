#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np

from tools.evaluate_rebuild_metrics import coarse_material, normalize_block_type, safe_div


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate description quality against GT voxels/bbox.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing building_xxx dirs")
    parser.add_argument("--description_subdir", default="description", help="Subdir containing description.json")
    parser.add_argument("--out", default="description_metrics.json", help="Output JSON path")
    parser.add_argument("--building_pattern", default="building_*", help="Building glob pattern")
    parser.add_argument("--limit", type=int, default=0, help="Max building count (0=all)")
    parser.add_argument("--gt_material_top_k", type=int, default=6, help="Top-K GT materials for matching")
    parser.add_argument("--gt_material_min_ratio", type=float, default=0.03, help="Minimum GT material ratio")
    return parser.parse_args()


def _list_buildings(dataset_root: Path, pattern: str, limit: int) -> List[Path]:
    dirs = [p for p in dataset_root.glob(pattern) if p.is_dir()]
    dirs.sort()
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_metrics(pred: Set[str], gt: Set[str]) -> Dict[str, float]:
    inter = len(pred & gt)
    precision = safe_div(inter, len(pred))
    recall = safe_div(inter, len(gt))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    jaccard = safe_div(inter, len(pred | gt))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "intersection": float(inter),
        "pred_count": float(len(pred)),
        "gt_count": float(len(gt)),
    }


def _extract_desc_materials(desc: Dict[str, Any]) -> Set[str]:
    mats = desc.get("materials", [])
    out: Set[str] = set()
    if isinstance(mats, list):
        for m in mats:
            if isinstance(m, dict):
                name = m.get("name", "")
            else:
                name = m
            norm = normalize_block_type(name)
            if norm and norm != "air":
                out.add(norm)
    return out


def _gt_materials(voxels: np.ndarray, top_k: int, min_ratio: float) -> Set[str]:
    flat = voxels.reshape(-1)
    counts: Counter[str] = Counter()
    for raw in flat:
        norm = normalize_block_type(raw)
        if norm == "air":
            continue
        counts[norm] += 1

    total = sum(counts.values())
    if total == 0:
        return set()

    kept = []
    for mat, c in counts.most_common(max(1, top_k * 2)):
        if safe_div(c, total) < min_ratio:
            continue
        kept.append(mat)
        if len(kept) >= top_k:
            break

    if not kept:
        kept = [m for m, _ in counts.most_common(top_k)]
    return set(kept)


def _dimension_metrics(desc: Dict[str, Any], bbox: Dict[str, Any]) -> Dict[str, float]:
    w_gt = int(bbox["xmax"]) - int(bbox["xmin"]) + 1
    d_gt = int(bbox["zmax"]) - int(bbox["zmin"]) + 1
    h_gt = int(bbox["ymax"]) - int(bbox["ymin"]) + 1

    dims = desc.get("dimensions_estimate", {}) if isinstance(desc.get("dimensions_estimate"), dict) else {}

    def get_dim(k: str, fallback: int) -> int:
        try:
            return max(1, int(round(float(dims.get(k, fallback)))))
        except Exception:
            return fallback

    w_pr = get_dim("width", w_gt)
    d_pr = get_dim("depth", d_gt)
    h_pr = get_dim("height", h_gt)

    err_w = abs(w_pr - w_gt)
    err_d = abs(d_pr - d_gt)
    err_h = abs(h_pr - h_gt)

    rel_w = safe_div(err_w, w_gt)
    rel_d = safe_div(err_d, d_gt)
    rel_h = safe_div(err_h, h_gt)

    mean_rel = (rel_w + rel_d + rel_h) / 3.0
    dim_score = max(0.0, 1.0 - min(1.0, mean_rel))

    return {
        "gt_width": float(w_gt),
        "gt_depth": float(d_gt),
        "gt_height": float(h_gt),
        "pred_width": float(w_pr),
        "pred_depth": float(d_pr),
        "pred_height": float(h_pr),
        "abs_err_width": float(err_w),
        "abs_err_depth": float(err_d),
        "abs_err_height": float(err_h),
        "rel_err_width": rel_w,
        "rel_err_depth": rel_d,
        "rel_err_height": rel_h,
        "rel_err_mean": mean_rel,
        "dim_score": dim_score,
    }


def _completeness(desc: Dict[str, Any]) -> Dict[str, float]:
    summary = str(desc.get("summary", "")).strip()
    mats = desc.get("materials", [])
    dims = desc.get("dimensions_estimate", {})
    elements = desc.get("elements", [])
    hints = desc.get("rebuild_hints", [])

    has_summary = 1.0 if len(summary) >= 30 else 0.0
    has_materials = 1.0 if isinstance(mats, list) and len(mats) >= 2 else 0.0
    has_dims = 1.0 if isinstance(dims, dict) and all(str(dims.get(k, "")).strip() for k in ("width", "depth", "height")) else 0.0
    has_elements = 1.0 if isinstance(elements, list) and len(elements) >= 2 else 0.0
    has_hints = 1.0 if isinstance(hints, list) and len(hints) >= 2 else 0.0

    score = (has_summary + has_materials + has_dims + has_elements + has_hints) / 5.0
    return {
        "has_summary": has_summary,
        "has_materials": has_materials,
        "has_dimensions": has_dims,
        "has_elements": has_elements,
        "has_rebuild_hints": has_hints,
        "completeness_score": score,
    }


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals)) / float(len(vals))


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    items: List[Dict[str, Any]] = []
    missing_desc: List[str] = []

    for bdir in _list_buildings(dataset_root, args.building_pattern, args.limit):
        desc_path = bdir / args.description_subdir / "description.json"
        gt_bbox_path = bdir / "gt" / "bbox.json"
        gt_vox_path = bdir / "gt" / "voxels.npy"
        if not gt_bbox_path.is_file() or not gt_vox_path.is_file():
            continue
        if not desc_path.is_file():
            missing_desc.append(bdir.name)
            continue

        desc = _load_json(desc_path)
        bbox = _load_json(gt_bbox_path)
        vox = np.load(gt_vox_path, allow_pickle=True)

        gt_mats = _gt_materials(vox, top_k=max(1, args.gt_material_top_k), min_ratio=max(0.0, args.gt_material_min_ratio))
        pr_mats = _extract_desc_materials(desc)

        strict_m = _set_metrics(pr_mats, gt_mats)
        gt_coarse = {coarse_material(x) for x in gt_mats if x != "air"}
        pr_coarse = {coarse_material(x) for x in pr_mats if x != "air"}
        coarse_m = _set_metrics(pr_coarse, gt_coarse)

        dim_m = _dimension_metrics(desc, bbox)
        comp = _completeness(desc)

        auto_score = (
            0.30 * strict_m["f1"]
            + 0.25 * coarse_m["f1"]
            + 0.25 * dim_m["dim_score"]
            + 0.20 * comp["completeness_score"]
        )

        item = {
            "building": bdir.name,
            "description_path": str(desc_path),
            "gt_materials": sorted(gt_mats),
            "pred_materials": sorted(pr_mats),
            "gt_coarse_materials": sorted(gt_coarse),
            "pred_coarse_materials": sorted(pr_coarse),
            "strict_material_metrics": strict_m,
            "coarse_material_metrics": coarse_m,
            "dimension_metrics": dim_m,
            "completeness": comp,
            "auto_score": auto_score,
        }
        items.append(item)

    aggregate = {
        "auto_score_mean": mean(item["auto_score"] for item in items),
        "strict_material_f1_mean": mean(item["strict_material_metrics"]["f1"] for item in items),
        "coarse_material_f1_mean": mean(item["coarse_material_metrics"]["f1"] for item in items),
        "dimension_score_mean": mean(item["dimension_metrics"]["dim_score"] for item in items),
        "completeness_score_mean": mean(item["completeness"]["completeness_score"] for item in items),
    }

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "description_subdir": args.description_subdir,
        "settings": {
            "gt_material_top_k": int(args.gt_material_top_k),
            "gt_material_min_ratio": float(args.gt_material_min_ratio),
        },
        "summary": {
            "evaluated_buildings": len(items),
            "missing_descriptions": missing_desc,
        },
        "aggregate": aggregate,
        "items": items,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate_description_quality] wrote: {out_path}")
    print(
        "[evaluate_description_quality] "
        f"evaluated_buildings={len(items)} missing_descriptions={len(missing_desc)} "
        f"auto_score_mean={aggregate['auto_score_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
