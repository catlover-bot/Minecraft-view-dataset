from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

Block = Tuple[int, int, int, str]


def build_voxels_array(bbox: Dict[str, int], blocks: Iterable[Block]) -> np.ndarray:
    """Build [Y, X, Z] block-name array inside bbox."""
    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    zmin, zmax = bbox["zmin"], bbox["zmax"]

    size_x = xmax - xmin + 1
    size_y = ymax - ymin + 1
    size_z = zmax - zmin + 1

    voxels = np.full((size_y, size_x, size_z), "air", dtype="<U32")
    for x, y, z, block in blocks:
        if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
            continue
        voxels[y - ymin, x - xmin, z - zmin] = block
    return voxels


def export_ground_truth(out_dir: Path, bbox: Dict[str, int], blocks: Iterable[Block]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox_payload = {
        **bbox,
        "order": "xmin,xmax,ymin,ymax,zmin,zmax",
        "voxel_axis_order": "Y,X,Z",
    }
    bbox_path = out_dir / "bbox.json"
    bbox_path.write_text(json.dumps(bbox_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    voxels = build_voxels_array(bbox, blocks)
    voxels_path = out_dir / "voxels.npy"
    np.save(voxels_path, voxels)

    return {
        "bbox_json": str(bbox_path),
        "voxels_npy": str(voxels_path),
    }

