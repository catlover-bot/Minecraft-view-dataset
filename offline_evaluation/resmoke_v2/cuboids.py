"""Lossless E diagnostic: deterministic, disjoint, same-material cuboids.

This version never changes the old Z-run encoder. Start at the first remaining
cell in Y, X, Z order, extend through equal remaining material along +Z, then
+X, then +Y, and remove that whole cuboid. Repeat in the same fixed seed order.
This is one deterministic partition, with no claim of a minimum partition.

Reference actions are CPU controls, never model outputs or replacement outputs.
"""
from __future__ import annotations

import json
from itertools import chain

from offline_evaluation.server_v2.core import (
    Grid, Limits, ValidationError, _integer, material_id, strict_json,
    validate_actions,
)


SEMANTICS = {
    "schema": "resmoke_v2_exact_cuboids_v1",
    "provenance": "new_implementation",
    "condition": "exact_discrete_spec_to_actions_diagnostic",
    "source": "ground_truth",
    "partition_rule": "first_remaining_YXZ_extend_Z_then_X_then_Y_v1",
    "cuboid_fields": ["x1", "y1", "z1", "x2", "y2", "z2", "palette_index"],
    "cuboid_order": "ascending_Y1_X1_Z1",
    "endpoints": "inclusive",
    "absent_coordinates": "air",
    "palette_semantics": "zero_based_exact_non_air_material_identifiers",
    "world_mapping": "world: XYZ unchanged; local: add origin_xyz to XYZ",
}


def compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _check_bytes(spec: dict, limits: Limits):
    if len(compact_json(spec).encode("utf-8")) > limits.max_json_bytes:
        raise ValidationError("cuboid exact-spec exceeds JSON byte limit; truncation forbidden")


def encode_exact(grid: Grid, limits: Limits = Limits()) -> dict:
    """Encode all occupied cells without filling holes or crossing materials."""
    if len(grid.voxels) > min(limits.max_voxels, limits.max_touched_voxels):
        raise ValidationError("cuboid exact-spec exceeds voxel limit")
    for coordinate in chain(grid.origin, chain.from_iterable(grid.voxels)):
        _integer(coordinate, "cuboid coordinate", limits.max_coordinate_abs)
    palette = sorted(set(grid.voxels.values()))
    indices = {material: index for index, material in enumerate(palette)}
    remaining = dict(grid.voxels)
    cuboids = []
    for x1, y1, z1 in sorted(remaining, key=lambda c: (c[1], c[0], c[2])):
        material = remaining.get((x1, y1, z1))
        if material is None:
            continue
        x2, y2, z2 = x1, y1, z1
        while remaining.get((x1, y1, z2 + 1)) == material:
            z2 += 1
        while all(remaining.get((x2 + 1, y1, z)) == material for z in range(z1, z2 + 1)):
            x2 += 1
        while all(remaining.get((x, y2 + 1, z)) == material
                  for x in range(x1, x2 + 1) for z in range(z1, z2 + 1)):
            y2 += 1
        cuboids.append([x1, y1, z1, x2, y2, z2, indices[material]])
        if len(cuboids) > limits.max_operations:
            raise ValidationError("cuboid exact-spec exceeds operation limit; truncation forbidden")
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                for z in range(z1, z2 + 1):
                    del remaining[(x, y, z)]
    metadata = {key: list(value) if isinstance(value, list) else value for key, value in SEMANTICS.items()}
    spec = {**metadata, "coordinate_frame": grid.coordinate_frame,
            "origin_xyz": list(grid.origin), "palette": palette, "cuboids": cuboids}
    _check_bytes(spec, limits)
    return spec


def decode_exact(spec: dict | str, limits: Limits = Limits()) -> Grid:
    """Reject invalid bounds, ambiguous metadata, and all overlap before use."""
    if isinstance(spec, str):
        spec = strict_json(spec, limits)
    keys = set(SEMANTICS) | {"coordinate_frame", "origin_xyz", "palette", "cuboids"}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise ValidationError("invalid cuboid exact-spec fields")
    if any(spec[key] != value for key, value in SEMANTICS.items()):
        raise ValidationError("unsupported cuboid exact-spec semantics")
    _check_bytes(spec, limits)
    if spec["coordinate_frame"] not in ("world", "local"):
        raise ValidationError("invalid cuboid coordinate frame")
    origin = spec["origin_xyz"]
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValidationError("cuboid origin must contain XYZ integers")
    origin = tuple(_integer(value, "origin", limits.max_coordinate_abs) for value in origin)
    palette = spec["palette"]
    if not isinstance(palette, list) or len(palette) > limits.max_voxels:
        raise ValidationError("invalid cuboid palette")
    normalized = [material_id(material) for material in palette]
    if normalized != palette or palette != sorted(set(palette)) or "air" in palette:
        raise ValidationError("palette must contain sorted unique exact non-air identifiers")
    cuboids = spec["cuboids"]
    if not isinstance(cuboids, list) or len(cuboids) > limits.max_operations:
        raise ValidationError("cuboid count exceeds operation limit")
    total, previous = 0, None
    checked = []
    for row in cuboids:
        if not isinstance(row, list) or len(row) != 7:
            raise ValidationError("cuboid must have six XYZ bounds and a palette index")
        x1, y1, z1, x2, y2, z2 = [
            _integer(value, "cuboid coordinate", limits.max_coordinate_abs) for value in row[:6]
        ]
        index = _integer(row[6], "palette index")
        if not 0 <= index < len(palette) or x1 > x2 or y1 > y2 or z1 > z2:
            raise ValidationError("invalid cuboid bounds or palette index")
        key = (y1, x1, z1)
        if previous is not None and key <= previous:
            raise ValidationError("cuboids must be strictly sorted by Y1,X1,Z1")
        previous = key
        total += (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
        if total > min(limits.max_voxels, limits.max_touched_voxels):
            raise ValidationError("cuboid expanded voxels exceed limit")
        checked.append((x1, y1, z1, x2, y2, z2, index))
    voxels = {}
    for x1, y1, z1, x2, y2, z2, index in checked:
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                for z in range(z1, z2 + 1):
                    if (x, y, z) in voxels:
                        raise ValidationError("overlapping cuboids are forbidden, including same material")
                    voxels[(x, y, z)] = palette[index]
    if set(voxels.values()) != set(palette):
        raise ValidationError("cuboid palette must contain exactly the used materials")
    return Grid(voxels, spec["coordinate_frame"], origin)


def exact_builder_text(spec: dict) -> str:
    """The complete E input; no reference action list is supplied to the model."""
    decode_exact(spec)
    return (
        "This diagnostic supplies a complete discrete building specification. "
        "Every cuboid row is [x1,y1,z1,x2,y2,z2,palette_index] with inclusive bounds; "
        "palette indices start at zero and identify exact non-air materials. "
        "Cuboids do not overlap; every omitted coordinate is air. "
        "For world coordinates use XYZ unchanged; for local coordinates add origin_xyz to each endpoint. "
        "Produce the specified building using only the common set/fill/carve actions in world coordinates. "
        "A fill can represent one entire cuboid without expanding it into cells or Z-runs. "
        "Do not omit cuboids. This is a discrete-specification-to-actions diagnostic, "
        "not a natural-language information ceiling.\n" + compact_json(spec)
    )


def reference_actions(spec: dict, limits: Limits = Limits()) -> dict:
    """One feasible CPU reference sequence, not a minimum or LLM completion."""
    grid = decode_exact(spec, limits)
    offset = grid.origin if grid.coordinate_frame == "local" else (0, 0, 0)
    actions = []
    for x1, y1, z1, x2, y2, z2, index in spec["cuboids"]:
        actions.append({"op": "fill", "x1": x1 + offset[0], "y1": y1 + offset[1],
                        "z1": z1 + offset[2], "x2": x2 + offset[0], "y2": y2 + offset[1],
                        "z2": z2 + offset[2], "block": spec["palette"][index]})
    return {"operations": validate_actions(actions, limits)}
