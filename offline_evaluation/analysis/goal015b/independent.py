"""Independent bounded XYZ executor for Goal015B diagnostics.

This module imports no previous executor, validator, provider, or evaluator.
It implements the frozen world-coordinate semantics with explicit finite loops.
Trace rows contain counts, indices and hashes, never coordinate arrays or grids.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re


@dataclass(frozen=True)
class IndependentLimits:
    max_operations: int = 50_000
    max_coordinate_abs: int = 100_000
    max_touched_voxels: int = 2_000_000
    max_voxels: int = 1_000_000


class IndependentExecutionError(ValueError):
    """Invalid input or exceeded bound; no partial grid is returned."""
    def __init__(self, reason, operation_index=None, phase='validation'):
        self.reason = reason
        self.operation_index = operation_index
        self.phase = phase
        super().__init__(reason if operation_index is None else f'operation {operation_index}: {reason}')


_MATERIAL = re.compile(r'(?:minecraft:)?[a-z0-9_][a-z0-9_./-]{0,127}\Z')
_AIR_NAMES = frozenset(('air', 'cave_air', 'void_air'))
_FIELDS = {
    'set': frozenset(('op', 'x', 'y', 'z', 'block')),
    'fill': frozenset(('op', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'block')),
    'carve': frozenset(('op', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2')),
}


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def grid_sha256(grid):
    """SHA256 of compact UTF8 JSON sorted [[x,y,z,material],...].

    This is a diagnostic hash, distinct from the frozen full grid-payload hash.
    Stream entries into the digest to avoid constructing another full grid.
    """
    digest = hashlib.sha256(b'[')
    for index, (coordinate, material) in enumerate(sorted(grid.items())):
        if index:
            digest.update(b',')
        digest.update(json.dumps([*coordinate, material], ensure_ascii=False,
                                 separators=(',', ':'), allow_nan=False).encode('utf-8'))
    digest.update(b']')
    return digest.hexdigest()


def _limits(supplied):
    supplied = IndependentLimits() if supplied is None else supplied
    result = {}
    for name in IndependentLimits.__dataclass_fields__:
        value = getattr(supplied, name, None)
        if type(value) is not int or value < 0:
            raise IndependentExecutionError(f'{name} must be a nonnegative integer')
        result[name] = value
    return IndependentLimits(**result)


def _validated(operations, limits):
    if not isinstance(operations, list) or len(operations) > limits.max_operations:
        raise IndependentExecutionError('operations must be a list within operation limit')
    normalized = []
    total_touches = 0
    for index, action in enumerate(operations):
        def reject(reason):
            raise IndependentExecutionError(reason, index)
        if not isinstance(action, dict):
            reject('operation must be an object')
        name = action.get('op')
        if not isinstance(name, str) or name not in _FIELDS:
            reject('unsupported op')
        if set(action) != _FIELDS[name]:
            reject('fields do not match operation schema')
        def integer(key):
            value = action[key]
            if type(value) is not int:
                reject(f'{key} must be an integer; booleans are invalid')
            if abs(value) > limits.max_coordinate_abs:
                reject(f'{key} exceeds coordinate bound')
            return value
        if name == 'set':
            lower = upper = tuple(integer(axis) for axis in 'xyz')
        else:
            first = tuple(integer(axis + '1') for axis in 'xyz')
            second = tuple(integer(axis + '2') for axis in 'xyz')
            lower = tuple(min(a, b) for a, b in zip(first, second))
            upper = tuple(max(a, b) for a, b in zip(first, second))
        volume = 1
        for low, high in zip(lower, upper):
            volume *= high - low + 1
        material = 'air'
        if name != 'carve':
            raw_material = action['block']
            if not isinstance(raw_material, str) or not _MATERIAL.fullmatch(raw_material):
                reject('block must be a lower-case Minecraft material identifier')
            material = raw_material.removeprefix('minecraft:')
            if material in _AIR_NAMES:
                material = 'air'
            if material != 'air' and volume > limits.max_voxels:
                reject('fill exceeds live voxel limit')
        total_touches += volume
        if total_touches > limits.max_touched_voxels:
            reject('total voxel touch limit exceeded')
        normalized.append((name, lower, upper, material, volume, _json_sha(action)))
    return normalized


def execute_independent(operations, limits=None):
    """Return (sparse world XYZ dictionary, compact diagnostic trace).

    ``limits`` accepts the frozen core.Limits object, but none of its methods or
    validators are invoked. All input operations are validated before execution.
    Indices are zero-based. Air write visits include carve visits; actual deletion
    counts only occupied voxels removed. No GT clipping or output repair occurs.
    """
    limits = _limits(limits)
    normalized = _validated(operations, limits)
    grid = {}
    trace = {'schema': 'goal015b_independent_trace_v1', 'operation_index_base': 0,
             'coordinate_frame': 'world', 'axis_order': 'XYZ',
             'operation_count': len(normalized), 'operation_counts': {'set': 0, 'fill': 0, 'carve': 0},
             'touched_voxel_visits': 0, 'air_write_visits': 0, 'non_air_write_visits': 0,
             'carve_delete_visits': 0, 'deleted_occupied_voxels': 0,
             'new_occupied_voxels': 0, 'material_replacements': 0,
             'first_occupied_operation_index': None, 'max_occupied_voxels': 0,
             'max_occupied_operation_index': None, 'last_zero_transition_operation_index': None,
             'zero_transition_count': 0, 'operation_trace': [],
             'grid_hash_encoding': 'SHA256 compact UTF8 JSON of sorted [x,y,z,material] rows'}
    for index, (name, lower, upper, material, volume, operation_hash) in enumerate(normalized):
        before = len(grid)
        inserted = removed = replacements = 0
        # XYZ loops are intentionally separate from the old itertools executor.
        for x in range(lower[0], upper[0] + 1):
            for y in range(lower[1], upper[1] + 1):
                for z in range(lower[2], upper[2] + 1):
                    coordinate = (x, y, z)
                    if material == 'air':
                        if coordinate in grid:
                            del grid[coordinate]
                            removed += 1
                    else:
                        old = grid.get(coordinate)
                        if old is None:
                            if len(grid) >= limits.max_voxels:
                                raise IndependentExecutionError('accumulated live occupancy exceeds voxel limit',
                                                                index, phase='execution')
                            inserted += 1
                        elif old != material:
                            replacements += 1
                        grid[coordinate] = material
        after = len(grid)
        if after and trace['first_occupied_operation_index'] is None:
            trace['first_occupied_operation_index'] = index
        if after > trace['max_occupied_voxels']:
            trace['max_occupied_voxels'] = after
            trace['max_occupied_operation_index'] = index
        if before and not after:
            trace['last_zero_transition_operation_index'] = index
            trace['zero_transition_count'] += 1
        air, nonair = (volume, 0) if material == 'air' else (0, volume)
        trace['operation_counts'][name] += 1
        trace['touched_voxel_visits'] += volume
        trace['air_write_visits'] += air
        trace['non_air_write_visits'] += nonair
        trace['carve_delete_visits'] += volume if name == 'carve' else 0
        trace['deleted_occupied_voxels'] += removed
        trace['new_occupied_voxels'] += inserted
        trace['material_replacements'] += replacements
        trace['operation_trace'].append({'operation_index': index, 'op': name,
            'original_operation_sha256': operation_hash, 'occupied_before': before, 'occupied_after': after,
            'touched_voxel_visits': volume, 'air_write_visits': air, 'non_air_write_visits': nonair,
            'deleted_occupied_voxels': removed, 'new_occupied_voxels': inserted,
            'material_replacements': replacements})
    trace['final_occupied_voxels'] = len(grid)
    trace['final_grid_sha256'] = grid_sha256(grid)
    trace['empty_classification'] = ('NONEMPTY' if grid else 'EMPTY_OPERATIONS' if not normalized else
        'AIR_OR_CARVE_ONLY' if not trace['non_air_write_visits'] else 'PLACED_THEN_DELETED_ALL')
    return grid, trace
