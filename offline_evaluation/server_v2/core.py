"""Strict bounded JSON executor and sparse voxel metrics.

Coordinates use XYZ; Y is vertical. Cuboid endpoints are inclusive and may be
reversed. Operations run in order: the last write wins, air/carve delete.
The executor accepts world coordinates or explicit local coordinates offset by
origin, and always returns world coordinates. There is no rotation, scaling,
clamping, material synonym merging, or GT-dependent output repair.

Material IDs are exact identifiers after stripping an optional ``minecraft:``
namespace. ``air``, ``cave_air`` and ``void_air`` denote absence. Block states
are deliberately outside this new action schema and are not silently dropped.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

Coord = tuple[int, int, int]
AIR = {"air", "cave_air", "void_air"}
MATERIAL_RE = re.compile(r"(?:minecraft:)?[a-z0-9_][a-z0-9_./-]{0,127}\Z")


class ValidationError(ValueError):
    """Rejected input; no partial execution is returned."""


@dataclass(frozen=True)
class Limits:
    max_operations: int = 50_000
    max_coordinate_abs: int = 100_000
    max_touched_voxels: int = 2_000_000
    max_voxels: int = 1_000_000
    max_json_bytes: int = 8_000_000
    max_numpy_bytes: int = 128_000_000
    max_alignment_radius: int = 8
    max_alignment_work: int = 50_000_000
    max_alignment_sparse_pairs: int = 100_000
    max_alignment_transform_cells: int = 2_097_152
    max_alignment_transform_work: int = 150_000_000
    max_alignment_memory_bytes: int = 268_435_456

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 0:
                raise ValidationError(f"{name} must be a nonnegative integer")


def _integer(value: Any, label: str, bound: int | None = None) -> int:
    if type(value) is not int:
        raise ValidationError(f"{label} must be an integer (booleans are invalid)")
    if bound is not None and abs(value) > bound:
        raise ValidationError(f"{label} exceeds coordinate bound {bound}")
    return value


def material_id(value: Any) -> str:
    if not isinstance(value, str) or not MATERIAL_RE.fullmatch(value):
        raise ValidationError("block must be a lower-case Minecraft material identifier")
    value = value.removeprefix("minecraft:")
    return "air" if value in AIR else value


def strict_json(text: str, limits: Limits = Limits()) -> Any:
    if not isinstance(text, str) or len(text.encode("utf-8")) > limits.max_json_bytes:
        raise ValidationError("JSON input exceeds byte limit or is not text")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(_):
        raise ValidationError("non-finite JSON numbers are invalid")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("invalid JSON") from exc


def parse_actions(text: str, limits: Limits = Limits()) -> list[dict]:
    """Parse one strict object {operations: [...]} or a bare operation list."""
    parsed = strict_json(text, limits)
    if isinstance(parsed, dict):
        if set(parsed) != {"operations"}:
            raise ValidationError("action object must contain only operations")
        parsed = parsed["operations"]
    return validate_actions(parsed, limits)


def validate_actions(actions: Any, limits: Limits = Limits()) -> list[dict]:
    """Validate all operations and volume bounds before enumerating any voxels.

    The touch budget counts cuboid visits including repeated writes/deletes.
    Each material fill must also fit the live voxel limit. Accumulated live
    occupancy is checked before each insertion in execute, without exceeding
    that limit. Invalid input never produces a partial success.
    """
    if not isinstance(actions, list) or len(actions) > limits.max_operations:
        raise ValidationError("operations must be a list within operation limit")
    result, touches = [], 0
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValidationError(f"operation {index} must be an object")
        op = action.get("op")
        if op == "set":
            required = {"op", "x", "y", "z", "block"}
        elif op == "fill":
            required = {"op", "x1", "y1", "z1", "x2", "y2", "z2", "block"}
        elif op == "carve":
            required = {"op", "x1", "y1", "z1", "x2", "y2", "z2"}
        else:
            raise ValidationError(f"operation {index} has unsupported op")
        if set(action) != required:
            raise ValidationError(f"operation {index} fields do not match {op} schema")
        out = {"op": op}
        if op == "set":
            for axis in "xyz":
                out[axis] = _integer(action[axis], axis, limits.max_coordinate_abs)
            volume = 1
        else:
            volume = 1
            for axis in "xyz":
                a = _integer(action[axis + "1"], axis + "1", limits.max_coordinate_abs)
                b = _integer(action[axis + "2"], axis + "2", limits.max_coordinate_abs)
                out[axis + "1"], out[axis + "2"] = min(a, b), max(a, b)
                volume *= abs(a - b) + 1
        if op != "carve":
            out["block"] = material_id(action["block"])
            if out["block"] != "air" and volume > limits.max_voxels:
                raise ValidationError(f"operation {index} fill exceeds live voxel limit")
        touches += volume
        if touches > limits.max_touched_voxels:
            raise ValidationError(f"operation {index} exceeds total voxel touch limit")
        result.append(out)
    return result


@dataclass
class Grid:
    voxels: dict[Coord, str] = field(default_factory=dict)
    coordinate_frame: str = "world"
    origin: Coord = (0, 0, 0)

    def __post_init__(self):
        if self.coordinate_frame not in {"world", "local"}:
            raise ValidationError("coordinate_frame must be world or local")
        if len(self.origin) != 3:
            raise ValidationError("origin must have three XYZ integers")
        self.origin = tuple(_integer(v, "origin") for v in self.origin)
        normalized = {}
        for coord, material in self.voxels.items():
            if len(coord) != 3:
                raise ValidationError("voxel coordinate must have three XYZ integers")
            coord = tuple(_integer(v, "voxel coordinate") for v in coord)
            material = material_id(material)
            if material != "air":
                normalized[coord] = material
        self.voxels = normalized

    def as_world(self) -> Grid:
        if self.coordinate_frame == "world":
            return self
        return Grid({tuple(c[i] + self.origin[i] for i in range(3)): m
                     for c, m in self.voxels.items()}, origin=self.origin)

    def shifted(self, shift: Coord) -> Grid:
        if len(shift) != 3:
            raise ValidationError("shift must have three XYZ integers")
        shift = tuple(_integer(v, "shift") for v in shift)
        world = self.as_world()
        return Grid({tuple(c[i] + shift[i] for i in range(3)): m
                     for c, m in world.voxels.items()}, origin=world.origin)

    def bbox(self) -> dict[str, int] | None:
        if not self.voxels:
            return None
        return {axis + end: fn(c[i] for c in self.voxels)
                for i, axis in enumerate("xyz") for end, fn in (("min", min), ("max", max))}

    def dimensions(self) -> Coord:
        bbox = self.bbox()
        return tuple(bbox[a + "max"] - bbox[a + "min"] + 1 for a in "xyz") if bbox else (0, 0, 0)

    def to_dict(self) -> dict:
        return grid_to_payload(self)

    @classmethod
    def from_dict(cls, payload: dict, limits: Limits = Limits()) -> Grid:
        return payload_to_grid(payload, limits)


def grid_to_payload(grid: Grid) -> dict:
    return {"schema": "server_v2_grid_v1", "provenance": "new_implementation",
            "axis_order": "XYZ", "coordinate_frame": grid.coordinate_frame,
            "origin_xyz": list(grid.origin),
            "voxels": [[*c, m] for c, m in sorted(grid.voxels.items())]}


def payload_to_grid(payload: dict, limits: Limits = Limits()) -> Grid:
    keys = {"schema", "provenance", "axis_order", "coordinate_frame", "origin_xyz", "voxels"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValidationError("invalid grid payload fields")
    if (payload["schema"] != "server_v2_grid_v1" or payload["axis_order"] != "XYZ"
            or payload["provenance"] != "new_implementation"):
        raise ValidationError("unsupported grid payload schema/provenance/axes")
    rows = payload["voxels"]
    origin = payload["origin_xyz"]
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValidationError("grid origin must be a list of three XYZ integers")
    origin = tuple(_integer(v, "origin", limits.max_coordinate_abs) for v in origin)
    if not isinstance(rows, list) or len(rows) > limits.max_voxels:
        raise ValidationError("grid payload exceeds voxel limit")
    voxels = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 4:
            raise ValidationError("voxel row must contain XYZ and material")
        c = tuple(_integer(v, "coordinate", limits.max_coordinate_abs) for v in row[:3])
        if c in voxels:
            raise ValidationError("duplicate voxel coordinate")
        m = material_id(row[3])
        if m == "air":
            raise ValidationError("sparse payload cannot explicitly store air")
        voxels[c] = m
    return Grid(voxels, payload["coordinate_frame"], origin)


def execute(actions: list[dict], limits: Limits = Limits(), *,
            coordinate_frame: str = "world", origin: Coord = (0, 0, 0)) -> Grid:
    operations = validate_actions(actions, limits)
    if coordinate_frame not in {"world", "local"} or len(origin) != 3:
        raise ValidationError("invalid coordinate frame/origin")
    origin = tuple(_integer(v, "origin", limits.max_coordinate_abs) for v in origin)
    offset = origin if coordinate_frame == "local" else (0, 0, 0)
    # Validate translated endpoints for every operation before execution.
    for operation in operations:
        for i, axis in enumerate("xyz"):
            for key in ([axis] if operation["op"] == "set" else [axis + "1", axis + "2"]):
                operation[key] = _integer(operation[key] + offset[i], key, limits.max_coordinate_abs)
    voxels = {}
    for operation in operations:
        if operation["op"] == "set":
            coords = [tuple(operation[a] for a in "xyz")]
        else:
            coords = itertools.product(*(range(operation[a + "1"], operation[a + "2"] + 1)
                                         for a in "xyz"))
        material = operation.get("block", "air")
        for coord in coords:
            if material == "air":
                voxels.pop(coord, None)
            else:
                if coord not in voxels and len(voxels) >= limits.max_voxels:
                    raise ValidationError("accumulated live occupancy exceeds voxel limit")
                voxels[coord] = material
    return Grid(voxels, origin=origin)


def metrics(gt: Grid, pred: Grid) -> dict:
    """Metrics at the supplied coordinates; no internal alignment.

    Material-aware IoU treats (coordinate, exact material ID) as a set element,
    so a replacement contributes one unmatched element on each side. Repair
    charges one unit for each add/delete/replace, divided by max(1, GT count).
    Dimensions use tight occupied bounds, not padded storage-array bounds.
    Both-empty overlap/F1 values are one; dimension MAE and repair are zero.
    """
    g, p = gt.as_world().voxels, pred.as_world().voxels
    gs, ps = set(g), set(p)
    intersection = gs & ps
    union = gs | ps
    matched = sum(g[c] == p[c] for c in intersection)
    materials_g, materials_p = set(g.values()), set(p.values())
    material_common = len(materials_g & materials_p)
    add, delete, replace = len(gs - ps), len(ps - gs), len(intersection) - matched
    denom = len(g) + len(p)
    material_denom = len(materials_g) + len(materials_p)
    gd, pd = gt.dimensions(), pred.dimensions()
    return {"gt_voxel_count": len(g), "pred_voxel_count": len(p),
            "occupancy_intersection": len(intersection), "occupancy_union": len(union),
            "occupancy_iou": len(intersection) / len(union) if union else 1.0,
            "occupancy_f1": 2 * len(intersection) / denom if denom else 1.0,
            "dimension_mae": sum(abs(a - b) for a, b in zip(gd, pd)) / 3,
            "gt_dimensions_xyz": list(gd), "pred_dimensions_xyz": list(pd),
            "material_set_f1": 2 * material_common / material_denom if material_denom else 1.0,
            "material_aware_iou": matched / (denom - matched) if denom else 1.0,
            "material_match_count": matched,
            "add_count": add, "delete_count": delete, "replace_count": replace,
            "add_normalized_cost": add / max(1, len(g)),
            "delete_normalized_cost": delete / max(1, len(g)),
            "replace_normalized_cost": replace / max(1, len(g)),
            "total_normalized_repair_cost": (add + delete + replace) / max(1, len(g))}


def evaluate_pair(gt: Grid, pred: Grid, alignment_radius: int = 2,
                  limits: Limits = Limits(), *, alignment_mode: str = "bounded") -> dict:
    """Evaluate with explicitly bounded or globally exact XYZ translation.

    Select the shift maximizing occupancy intersection (and hence occupancy
    IoU). Ties use smallest L1 shift magnitude, then lexicographic XYZ. All
    aligned material and repair metrics use this same single selected shift.
    No coordinate truncation occurs when prediction extends outside GT bounds.
    The historical default remains bounded; strict_translation is a separately
    versioned primary evaluation over all integer translations.
    """
    from .alignment import select_translation

    gt, pred = gt.as_world(), pred.as_world()
    alignment = select_translation(gt.voxels, pred.voxels, mode=alignment_mode,
                                   radius=alignment_radius, limits=limits)
    version = "server_v2.1" if alignment_mode == "bounded" else "server_v2.2_strict_translation"
    return {"provenance": "new_implementation", "metric_version": version,
            "absolute": metrics(gt, pred),
            "aligned": metrics(gt, pred.shifted(tuple(alignment["shift_xyz"]))),
            "alignment": alignment}
