"""Explicit GT-derived diagnostic conditions and deterministic executor control.

Exact-spec compression groups equal-material contiguous Z cells at fixed Y,X.
Runs are [y,x,z_first,z_last,palette_index], inclusive and sorted by (y,x,z).
Sorted palette strings preserve exact material IDs. Missing coordinates are
air; no dimensions-only representation is called Exact-spec. These CPU
round-trips and GT-derived actions are implementation checks, not LLM success.
"""
from __future__ import annotations

import json

from .core import Grid, Limits, ValidationError, _integer, execute, material_id, strict_json, validate_actions


def _check_size(spec: dict, limits: Limits):
    size = len(json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    if size > limits.max_json_bytes:
        raise ValidationError("exact-spec input exceeds JSON byte limit; truncation is forbidden")


def encode_exact_spec(grid: Grid, limits: Limits = Limits()) -> dict:
    if len(grid.voxels) > limits.max_voxels:
        raise ValidationError("exact-spec exceeds voxel limit")
    palette = sorted(set(grid.voxels.values()))
    palette_index = {m: i for i, m in enumerate(palette)}
    runs = []
    for (x, y, z), material in sorted(grid.voxels.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])):
        for value in (x, y, z):
            _integer(value, "exact-spec coordinate", limits.max_coordinate_abs)
        index = palette_index[material]
        if runs and runs[-1][0:2] == [y, x] and runs[-1][3] + 1 == z and runs[-1][4] == index:
            runs[-1][3] = z
        else:
            runs.append([y, x, z, z, index])
    if len(runs) > limits.max_operations:
        raise ValidationError("exact-spec run count exceeds operation limit; truncation is forbidden")
    spec = {"schema": "server_v2_exact_spec_v1", "provenance": "new_implementation",
            "condition": "exact_spec_oracle", "source": "ground_truth",
            "coordinate_frame": grid.coordinate_frame, "origin_xyz": list(grid.origin),
            "run_order": "Y,X,Z_FIRST,Z_LAST,PALETTE_INDEX", "endpoints": "inclusive",
            "absent_coordinates": "air", "palette": palette, "runs": runs}
    _check_size(spec, limits)
    return spec


def decode_exact_spec(spec: dict | str, limits: Limits = Limits()) -> Grid:
    if isinstance(spec, str):
        spec = strict_json(spec, limits)
    keys = {"schema", "provenance", "condition", "source", "coordinate_frame", "origin_xyz",
            "run_order", "endpoints", "absent_coordinates", "palette", "runs"}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise ValidationError("invalid exact-spec fields")
    expected = {"schema": "server_v2_exact_spec_v1", "provenance": "new_implementation",
                "condition": "exact_spec_oracle", "source": "ground_truth",
                "run_order": "Y,X,Z_FIRST,Z_LAST,PALETTE_INDEX", "endpoints": "inclusive",
                "absent_coordinates": "air"}
    if any(spec[k] != v for k, v in expected.items()):
        raise ValidationError("unsupported exact-spec semantics")
    _check_size(spec, limits)
    palette, runs = spec["palette"], spec["runs"]
    if not isinstance(palette, list) or len(palette) > limits.max_voxels:
        raise ValidationError("invalid exact-spec palette")
    normalized = [material_id(m) for m in palette]
    if normalized != palette or palette != sorted(set(palette)) or "air" in palette:
        raise ValidationError("palette must be sorted, unique, exact non-air identifiers")
    if not isinstance(runs, list) or len(runs) > limits.max_operations:
        raise ValidationError("exact-spec exceeds run count limit")
    origin = spec["origin_xyz"]
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValidationError("exact-spec origin must be XYZ integers")
    origin = tuple(_integer(v, "origin", limits.max_coordinate_abs) for v in origin)
    if spec["coordinate_frame"] not in {"world", "local"}:
        raise ValidationError("invalid exact-spec coordinate frame")
    count, previous = 0, None
    for run in runs:
        if not isinstance(run, list) or len(run) != 5:
            raise ValidationError("invalid exact-spec run")
        y, x, z1, z2 = [_integer(v, "run coordinate", limits.max_coordinate_abs) for v in run[:4]]
        index = _integer(run[4], "palette index")
        if not 0 <= index < len(palette) or z1 > z2:
            raise ValidationError("invalid palette index or run endpoints")
        if previous is not None:
            py, px, _, pz2, pi = previous
            if (y, x, z1) <= (py, px, pz2):
                raise ValidationError("runs must be sorted and non-overlapping")
            if (y, x) == (py, px) and z1 == pz2 + 1 and index == pi:
                raise ValidationError("adjacent equal-material runs must be merged")
        count += z2 - z1 + 1
        if count > min(limits.max_voxels, limits.max_touched_voxels):
            raise ValidationError("exact-spec expanded voxels exceed limit")
        previous = run
    voxels = {(x, y, z): palette[index] for y, x, z1, z2, index in runs for z in range(z1, z2 + 1)}
    return Grid(voxels, spec["coordinate_frame"], origin)


def gt_to_actions(grid: Grid, limits: Limits = Limits()) -> list[dict]:
    """GT-derived world-coordinate runs; no LLM invocation."""
    spec = encode_exact_spec(grid.as_world(), limits)
    return validate_actions([
        {"op": "fill", "x1": x, "y1": y, "z1": z1, "x2": x, "y2": y, "z2": z2,
         "block": spec["palette"][index]} for y, x, z1, z2, index in spec["runs"]
    ], limits)


def executor_control(grid: Grid, limits: Limits = Limits()) -> Grid:
    return execute(gt_to_actions(grid, limits), limits, origin=grid.origin)


def partial_oracle(grid: Grid) -> dict:
    """Diagnostic attributes from GT: dimensions, bbox, materials and counts.

    This incomplete specification deliberately exposes GT information and
    must never be mixed into original/canonicalized ordinary inputs.
    """
    grid = grid.as_world()
    return {"condition": "partial_oracle", "provenance": "new_implementation",
            "source": "ground_truth", "diagnostic_only": True,
            "provided_attributes": ["occupied_dimensions_xyz", "world_bbox", "exact_material_set", "voxel_count"],
            "occupied_dimensions_xyz": list(grid.dimensions()), "world_bbox": grid.bbox(),
            "exact_material_set": sorted(set(grid.voxels.values())), "voxel_count": len(grid.voxels),
            "not_provided": ["individual_voxel_coordinates", "material_at_each_coordinate"]}
