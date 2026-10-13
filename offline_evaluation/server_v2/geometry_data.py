"""Read preserved legacy occupancy without manufacturing valid material labels.

Geometry validity and material validity are independent. Exact air labels are
absence; a descriptive label mentioning air, or an empty label, has unknown
presence and prevents complete occupancy scoring. Other descriptive labels
remain occupied but have UNKNOWN material. No GT is consulted by this reader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

from .core import AIR, Grid, Limits, ValidationError, material_id
from .data import _load_bbox, inspect_npy


def _unknown_presence(label):
    return not label.strip() or bool(re.search(
        r"(?:^|[^a-z0-9])(air|cave_air|void_air)(?:$|[^a-z0-9])", label.lower()))


def storage_truncation_evidence(read: "GeometryRead", actions: dict) -> list[dict]:
    """Compare existing saved normalized action labels, never execute them.

    A prefix is evidence of lossy storage, not permission to infer which full
    material applies to an individual coordinate. Full labels remain separate.
    """
    capacity = int(np.dtype(read.header["dtype"]).itemsize / (4 if np.dtype(read.header["dtype"]).kind == "U" else 1))
    operations = actions.get("operations", []) if isinstance(actions, dict) else []
    full_labels = {op["block"] for op in operations if isinstance(op, dict) and isinstance(op.get("block"), str)}
    result = []
    for label, count in read.raw_labels.items():
        candidates = sorted(full for full in full_labels if len(full) > capacity and full[:capacity] == label)
        if len(label) == capacity and candidates:
            result.append({"stored_label": label, "voxel_count": count,
                           "saved_action_labels": candidates, "capacity": capacity,
                           "classification": "CONFIRMED_STORAGE_TRUNCATION", "material": "UNKNOWN",
                           "evidence": "saved normalized action string is longer than storage capacity and has identical stored prefix"})
            if any(_unknown_presence(full) for full in candidates):
                read.geometry_valid = False
                read.coordinates = read.coordinates - read.label_coordinates.get(label, frozenset())
                result[-1]["geometry_presence"] = "UNKNOWN"
            else:
                result[-1]["geometry_presence"] = "occupied"
    if result:
        read.material_valid = False
        read.material_grid = None
    return result


@dataclass
class GeometryRead:
    coordinates: frozenset[tuple[int, int, int]]
    material_grid: Grid | None
    geometry_valid: bool
    material_valid: bool
    invalid_labels: list[dict]
    bbox: dict
    header: dict
    raw_labels: dict[str, int]
    label_coordinates: dict[str, frozenset] = field(default_factory=dict)


def _label_text(raw):
    if isinstance(raw, bytes):
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError:
            raise ValidationError("non-ASCII byte material identifier") from None
    return str(raw)


def load_geometry(voxels_path, bbox_path, limits: Limits = Limits()) -> GeometryRead:
    """NPY/header/coordinate checks are identical to the strict source reader."""
    header = inspect_npy(Path(voxels_path), limits)
    bbox = _load_bbox(Path(bbox_path), limits)
    if header["shape_yxz"] != [bbox[a + "max"] - bbox[a + "min"] + 1 for a in "yxz"]:
        raise ValidationError("NPY Y,X,Z dimensions do not match inclusive bbox")
    array = np.load(voxels_path, allow_pickle=False, mmap_mode="r")
    labels, counts = np.unique(array, return_counts=True)
    raw_labels = {_label_text(label): int(count) for label, count in zip(labels, counts)}
    normalized, invalid, unknown_labels = {}, [], set()
    geometry_valid = True
    for label, count in raw_labels.items():
        try:
            normalized[label] = material_id(label)
            # Legacy descriptive alternatives can also match identifier syntax.
            # Their syntax does not establish a unique material or air presence.
            if normalized[label] != 'air' and (_unknown_presence(label) or '_or_' in label.lower()):
                raise ValidationError('descriptive material alternative')
        except ValidationError:
            unknown_presence = _unknown_presence(label)
            geometry_valid &= not unknown_presence
            if unknown_presence:
                unknown_labels.add(label)
            classification = ("AMBIGUOUS_PRESENCE" if unknown_presence else
                              "MULTIPLE_OR_UNCERTAIN_MATERIAL" if "|" in label or "?" in label or "or_" in label else
                              "DESCRIPTIVE_NON_IDENTIFIER")
            invalid.append({"stored_label": label, "voxel_count": count,
                            "classification": classification, "material": "UNKNOWN",
                            "geometry_presence": "UNKNOWN" if unknown_presence else "occupied",
                            "stored_character_count": len(label),
                            "storage_character_capacity": header["dtype"]})
            normalized[label] = None
    # Retain definitely occupied coordinates only. A caller must check the
    # geometry_valid flag before using this set as a complete prediction.
    occupied, materials, label_coordinates = set(), {}, {}
    capacity = np.dtype(header["dtype"]).itemsize // (4 if np.dtype(header["dtype"]).kind == "U" else 1)
    xmin, ymin, zmin = (bbox[a + "min"] for a in "xyz")
    for yi, xi, zi in np.ndindex(array.shape):
        label = _label_text(array[yi, xi, zi])
        material = normalized[label]
        if material == "air" or label in unknown_labels:
            continue
        coordinate = (xmin + xi, ymin + yi, zmin + zi)
        occupied.add(coordinate)
        if len(label) == capacity:
            label_coordinates.setdefault(label, set()).add(coordinate)
        if material is not None:
            materials[coordinate] = material
    material_valid = not invalid
    grid = Grid(materials, origin=(xmin, ymin, zmin)) if material_valid else None
    return GeometryRead(frozenset(occupied), grid, bool(geometry_valid), material_valid,
                        invalid, bbox, header, raw_labels,
                        {label: frozenset(coords) for label, coords in label_coordinates.items()})


def dimensions(coordinates):
    return tuple(max(p[i] for p in coordinates) - min(p[i] for p in coordinates) + 1
                 for i in range(3)) if coordinates else (0, 0, 0)


def geometry_metrics(gt_coordinates, prediction_coordinates, shift=(0, 0, 0)):
    """Occupancy and dimensions only; undefined material/repair scores are null."""
    gt = set(gt_coordinates)
    pred = {tuple(p[i] + shift[i] for i in range(3)) for p in prediction_coordinates}
    overlap, union = len(gt & pred), len(gt | pred)
    dims_gt, dims_pred = dimensions(gt), dimensions(pred)
    return {"occupancy_iou": overlap / union if union else 1.0,
            "occupancy_f1": 2 * overlap / (len(gt) + len(pred)) if gt or pred else 1.0,
            "dimension_mae": sum(abs(a-b) for a, b in zip(dims_gt, dims_pred)) / 3,
            "gt_dimensions_xyz": list(dims_gt), "pred_dimensions_xyz": list(dims_pred),
            "gt_voxel_count": len(gt), "pred_voxel_count": len(pred),
            "occupancy_intersection": overlap, "occupancy_union": union,
            "material_set_f1": None, "material_aware_iou": None,
            "material_match_count": None, "add_count": None, "delete_count": None, "replace_count": None,
            "add_normalized_cost": None, "delete_normalized_cost": None,
            "replace_normalized_cost": None, "total_normalized_repair_cost": None}
