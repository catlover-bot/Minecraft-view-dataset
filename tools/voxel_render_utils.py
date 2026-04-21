#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from tools.evaluate_rebuild_metrics import normalize_block_type

Block = Tuple[int, int, int, str]


BLOCK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "stone": (135, 135, 135),
    "stonebrick": (122, 125, 130),
    "cobblestone": (110, 110, 110),
    "planks": (164, 129, 79),
    "wood": (164, 129, 79),
    "log": (120, 88, 55),
    "sandstone": (215, 202, 146),
    "brick_block": (155, 74, 68),
    "nether_brick": (86, 41, 40),
    "quartz_block": (232, 232, 228),
    "glass": (166, 217, 233),
    "glowstone": (224, 197, 98),
    "sea_lantern": (180, 235, 224),
    "stone_slab": (144, 144, 146),
    "slab_stone": (144, 144, 146),
    "fence": (128, 96, 58),
    "air": (0, 0, 0),
}


def _shade(rgb: Tuple[int, int, int], gain: float) -> Tuple[int, int, int]:
    return (
        max(0, min(255, int(rgb[0] * gain))),
        max(0, min(255, int(rgb[1] * gain))),
        max(0, min(255, int(rgb[2] * gain))),
    )


def blocks_from_voxels(voxels: np.ndarray, bbox: Dict[str, int]) -> List[Block]:
    ymin = int(bbox["ymin"])
    xmin = int(bbox["xmin"])
    zmin = int(bbox["zmin"])
    sy, sx, sz = voxels.shape
    blocks: List[Block] = []
    for yi in range(sy):
        for xi in range(sx):
            for zi in range(sz):
                b = normalize_block_type(voxels[yi, xi, zi])
                if b == "air":
                    continue
                blocks.append((xmin + xi, ymin + yi, zmin + zi, b))
    return blocks


def _project(
    x: float,
    y: float,
    z: float,
    *,
    yaw_rad: float,
    pitch_rad: float,
) -> Tuple[float, float, float]:
    cx = math.cos(yaw_rad)
    sx = math.sin(yaw_rad)

    xr = cx * x - sx * z
    zr = sx * x + cx * z

    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)

    yp = cp * y - sp * zr
    zp = sp * y + cp * zr
    return xr, yp, zp


def render_block_view(
    blocks: Sequence[Block],
    bbox: Dict[str, int],
    *,
    yaw_deg: float,
    pitch_deg: float,
    image_size: Tuple[int, int],
    background: Tuple[int, int, int] = (242, 246, 250),
) -> np.ndarray:
    width, height = image_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = int(background[0])
    img[:, :, 1] = int(background[1])
    img[:, :, 2] = int(background[2])

    if not blocks:
        return img

    cx = (bbox["xmin"] + bbox["xmax"]) / 2.0
    cy = (bbox["ymin"] + bbox["ymax"]) / 2.0
    cz = (bbox["zmin"] + bbox["zmax"]) / 2.0

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    proj: List[Tuple[float, float, float, Tuple[int, int, int]]] = []
    xmin2 = float("inf")
    xmax2 = float("-inf")
    ymin2 = float("inf")
    ymax2 = float("-inf")

    for x, y, z, block in blocks:
        xr, yp, zp = _project(
            x - cx,
            y - cy,
            z - cz,
            yaw_rad=yaw,
            pitch_rad=pitch,
        )
        proj.append((xr, yp, zp, BLOCK_COLORS.get(block, (150, 150, 150))))
        xmin2 = min(xmin2, xr)
        xmax2 = max(xmax2, xr)
        ymin2 = min(ymin2, yp)
        ymax2 = max(ymax2, yp)

    span_x = max(1e-6, xmax2 - xmin2)
    span_y = max(1e-6, ymax2 - ymin2)
    margin = 28
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    px = max(2, int(scale * 0.85))

    proj.sort(key=lambda t: (t[2], t[1]))

    for xr, yp, zp, base in proj:
        sx = (xr - xmin2) * scale + margin
        sy = (ymax2 - yp) * scale + margin
        x0 = max(0, int(round(sx - px / 2)))
        y0 = max(0, int(round(sy - px / 2)))
        x1 = min(width - 1, int(round(sx + px / 2)))
        y1 = min(height - 1, int(round(sy + px / 2)))
        if x1 <= x0 or y1 <= y0:
            continue
        img[y0 : y1 + 1, x0 : x1 + 1, :] = np.array(base, dtype=np.uint8)
        outline = np.array(_shade(base, 0.75), dtype=np.uint8)
        img[y0, x0 : x1 + 1, :] = outline
        img[y1, x0 : x1 + 1, :] = outline
        img[y0 : y1 + 1, x0, :] = outline
        img[y0 : y1 + 1, x1, :] = outline

    return img


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def save_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    compressed = zlib.compress(raw, level=6)
    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def generate_view_schedule(views: int) -> List[Tuple[float, float]]:
    n_ring = max(4, views - 2)
    out: List[Tuple[float, float]] = []
    for i in range(n_ring):
        out.append((i * 360.0 / n_ring, 22.0))
    out.append((45.0, 58.0))
    out.append((225.0, 62.0))
    return out[:views]


def render_views_to_dir(
    blocks: Sequence[Block],
    bbox: Dict[str, int],
    *,
    out_dir: Path,
    image_size: Tuple[int, int],
    views: int,
    prefix: str,
) -> List[Dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float]] = []
    schedule = generate_view_schedule(max(1, views))
    cx = (bbox["xmin"] + bbox["xmax"]) / 2.0 + 0.5
    cy = (bbox["ymin"] + bbox["ymax"]) / 2.0 + 0.5
    cz = (bbox["zmin"] + bbox["zmax"]) / 2.0 + 0.5

    for i, (yaw, pitch) in enumerate(schedule):
        img = render_block_view(
            blocks,
            bbox,
            yaw_deg=yaw,
            pitch_deg=pitch,
            image_size=image_size,
        )
        name = f"{prefix}_view{i:02d}_yaw{yaw:+07.2f}_pitch{pitch:+07.2f}.png"
        out_path = out_dir / name
        save_png(out_path, img)
        records.append(
            {
                "path": str(Path("images") / name),
                "x": cx,
                "y": cy,
                "z": cz,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "fov": 70.0,
                "width": int(image_size[0]),
                "height": int(image_size[1]),
            }
        )
    return records
