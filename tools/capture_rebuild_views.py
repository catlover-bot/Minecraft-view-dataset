#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.voxel_render_utils import blocks_from_voxels, render_views_to_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render preview images from rebuilt voxels.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--rebuild_subdir", required=True)
    p.add_argument("--out_subdir", default="")
    p.add_argument("--building_pattern", default="llm_case_*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--image_size", nargs=2, type=int, default=[960, 540], metavar=("W", "H"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _list_buildings(root: Path, pattern: str, limit: int) -> List[Path]:
    xs = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if limit > 0:
        xs = xs[:limit]
    return xs


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    out_subdir = args.out_subdir.strip() or f"{args.rebuild_subdir}_images"
    rendered = 0
    for bdir in _list_buildings(dataset_root, args.building_pattern, args.limit):
        rebuild_dir = bdir / args.rebuild_subdir
        vox_path = rebuild_dir / "voxels.npy"
        bbox_path = rebuild_dir / "bbox.json"
        if not (vox_path.is_file() and bbox_path.is_file()):
            print(f"[capture_rebuild_views] skip {bdir.name} (missing voxels/bbox)")
            continue
        out_dir = bdir / out_subdir
        meta_out = out_dir / "capture_meta.json"
        if meta_out.is_file() and not args.overwrite:
            print(f"[capture_rebuild_views] skip {bdir.name} (exists)")
            continue

        voxels = np.load(vox_path, allow_pickle=True)
        bbox = _load_json(bbox_path)
        blocks = blocks_from_voxels(voxels, bbox)
        views = render_views_to_dir(
            blocks,
            bbox,
            out_dir=out_dir,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            views=int(args.views),
            prefix="rebuild_rgb",
        )
        meta_out.write_text(
            json.dumps(
                {
                    "building": bdir.name,
                    "rebuild_subdir": args.rebuild_subdir,
                    "out_subdir": out_subdir,
                    "views": views,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        rendered += 1
        print(f"[capture_rebuild_views] wrote {bdir.name}/{out_subdir}")

    print(f"[capture_rebuild_views] done rendered={rendered}")


if __name__ == "__main__":
    main()
