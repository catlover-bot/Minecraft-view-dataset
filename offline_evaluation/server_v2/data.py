"""Safe readers for the repository's Y,X,Z Unicode NPY voxel arrays.

Storage bbox endpoints are inclusive, including padded air cells. A storage
index [yi,xi,zi] maps to world (xmin+xi, ymin+yi, zmin+zi). The storage origin
is bbox minimum, which differs from meta.generation.origin (a generation
anchor). Exact material IDs are strings; no numeric palette is inferred.

These readers reject object dtype and never enable pickle. Header dimensions,
dtype, byte counts, bbox shape and coordinate limits are checked before load.
The caller records trusted repository/ref/blob provenance separately.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import Grid, Limits, ValidationError, _integer, material_id, strict_json


@dataclass
class Scene:
    scene_id: str
    grid: Grid
    bbox: dict
    header: dict
    paths: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_npy(path: Path, limits: Limits = Limits()) -> dict:
    """Read a bounded NPY header only; refuse executable object payloads."""
    path = Path(path)
    if path.stat().st_size > limits.max_numpy_bytes:
        raise ValidationError("NPY file exceeds byte limit")
    with path.open("rb") as stream:
        try:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10_000)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10_000)
            else:
                raise ValidationError(f"unsupported NPY version {version}")
        except (ValueError, EOFError) as exc:
            raise ValidationError("invalid NPY header") from exc
        offset = stream.tell()
    if dtype.hasobject:
        raise ValidationError("object dtype requires pickle and is forbidden")
    if dtype.kind not in {"U", "S"} or dtype.itemsize > 512 or dtype.itemsize == 0:
        raise ValidationError("only bounded string material NPY arrays are supported")
    if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValidationError("NPY array must have three positive Y,X,Z dimensions")
    cells = math.prod(shape)
    if cells > limits.max_voxels:
        raise ValidationError("NPY storage cell count exceeds voxel limit")
    expected_bytes = offset + cells * dtype.itemsize
    if expected_bytes > limits.max_numpy_bytes or path.stat().st_size != expected_bytes:
        raise ValidationError("NPY payload byte count is truncated, oversized, or has trailing data")
    return {"version": list(version), "shape_yxz": list(shape), "dtype": dtype.str,
            "fortran_order": bool(fortran), "header_bytes": offset,
            "storage_cells": cells, "file_bytes": expected_bytes, "allow_pickle": False}


def _load_bbox(path: Path, limits: Limits) -> dict:
    with Path(path).open("rb") as stream:
        raw = stream.read(limits.max_json_bytes + 1)
    if len(raw) > limits.max_json_bytes:
        raise ValidationError("bbox JSON exceeds byte limit")
    try:
        bbox = strict_json(raw.decode("utf-8"), limits)
    except UnicodeDecodeError as exc:
        raise ValidationError("bbox JSON is not UTF-8") from exc
    required = {a + end for a in "xyz" for end in ("min", "max")}
    if not isinstance(bbox, dict) or not required <= set(bbox):
        raise ValidationError("bbox is missing inclusive XYZ endpoints")
    if bbox.get("voxel_axis_order", "Y,X,Z") != "Y,X,Z":
        raise ValidationError("bbox declares unsupported voxel axis order")
    for key in required:
        _integer(bbox[key], key, limits.max_coordinate_abs)
    if any(bbox[a + "min"] > bbox[a + "max"] for a in "xyz"):
        raise ValidationError("bbox endpoints must be ordered")
    return bbox


def _load_grid_with_header(voxels_path: Path, bbox_path: Path, limits: Limits):
    voxels_path, bbox_path = Path(voxels_path), Path(bbox_path)
    header = inspect_npy(voxels_path, limits)
    bbox = _load_bbox(bbox_path, limits)
    expected_shape = [bbox[a + "max"] - bbox[a + "min"] + 1 for a in "yxz"]
    if header["shape_yxz"] != expected_shape:
        raise ValidationError("NPY Y,X,Z dimensions do not match inclusive bbox")
    array = np.load(voxels_path, allow_pickle=False, mmap_mode="r")
    voxels = {}
    xmin, ymin, zmin = (bbox[a + "min"] for a in "xyz")
    for yi, xi, zi in np.ndindex(array.shape):
        raw = array[yi, xi, zi]
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValidationError("non-ASCII byte material identifier") from exc
        material = material_id(str(raw))
        if material != "air":
            voxels[(xmin + xi, ymin + yi, zmin + zi)] = material
    return Grid(voxels, origin=(xmin, ymin, zmin)), bbox, header


def load_grid(voxels_path: Path, bbox_path: Path, limits: Limits = Limits()) -> Grid:
    """Safely load GT or saved prediction arrays under the same coordinate rules."""
    return _load_grid_with_header(voxels_path, bbox_path, limits)[0]


def load_scene(building_dir: Path, limits: Limits = Limits()) -> Scene:
    building_dir = Path(building_dir)
    voxels_path, bbox_path = building_dir / "gt/voxels.npy", building_dir / "gt/bbox.json"
    grid, bbox, header = _load_grid_with_header(voxels_path, bbox_path, limits)
    return Scene(f"{building_dir.parent.name}_{building_dir.name}", grid, bbox, header,
                 {"voxels": str(voxels_path), "bbox": str(bbox_path)})
