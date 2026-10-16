"""Independent exact box/shell occupancy checks by 3D summed-volume queries.

This implementation never calls the production evaluator or translation search.
All positive-intersection translations are enumerated; outside this cube the
intersection is zero. Material labels do not enter any arithmetic.
"""
from __future__ import annotations

import math
import numpy as np

from ...server_v2.core import Grid, Limits


class IndependentGeometryResourceLimit(ValueError):
    def __init__(self, reason, metadata):
        super().__init__('INDEPENDENT_GEOMETRY_RESOURCE_LIMIT: ' + reason)
        self.metadata = dict(metadata, status='INDEPENDENT_GEOMETRY_RESOURCE_LIMIT',
                             reason=reason, search_complete=False, approximate_result_returned=False)


def _bounds(coords):
    return (tuple(min(point[i] for point in coords) for i in range(3)),
            tuple(max(point[i] for point in coords) for i in range(3)))


def _validate_grid(grid, limits):
    if not isinstance(grid, Grid) or grid.coordinate_frame != 'world':
        raise ValueError('independent check requires explicit world XYZ Grid inputs')
    if len(grid.voxels) > limits.max_voxels:
        raise IndependentGeometryResourceLimit('occupied count exceeds voxel limit', {})
    for coord in grid.voxels:
        if len(coord) != 3 or any(type(v) is not int for v in coord):
            raise ValueError('integer world XYZ coordinates required')
        if any(abs(v) > limits.max_coordinate_abs for v in coord):
            raise IndependentGeometryResourceLimit('input coordinate exceeds configured bound', {})


def _shape(pred, structure):
    if structure not in {'filled_box', 'six_face_shell'}:
        raise ValueError('fixed filled_box or six_face_shell structure required')
    if not pred:
        return None, None, None
    low, high = _bounds(pred)
    dims = tuple(high[i] - low[i] + 1 for i in range(3))
    outer = math.prod(dims)
    inner = math.prod(max(d - 2, 0) for d in dims)
    expected = outer if structure == 'filled_box' else outer - inner
    if len(pred) != expected:
        raise ValueError('prediction occupancy is not the declared complete box/shell')
    if structure == 'six_face_shell' and any(not any(c[i] in (low[i], high[i]) for i in range(3)) for c in pred):
        raise ValueError('six-face shell prediction contains an interior voxel')
    return low, high, dims


def _iou(count, gt_count, pred_count):
    union = gt_count + pred_count - count
    return count / union if union else 1.0


def _queries(prefix, x, y, z, dims, gt_dims):
    """Count half-open cuboids at one X and broadcast Y/Z anchor arrays."""
    x0 = max(0, min(gt_dims[0], x)); x1 = max(0, min(gt_dims[0], x + dims[0]))
    y0 = np.clip(y, 0, gt_dims[1])[:, None]; y1 = np.clip(y + dims[1], 0, gt_dims[1])[:, None]
    z0 = np.clip(z, 0, gt_dims[2])[None, :]; z1 = np.clip(z + dims[2], 0, gt_dims[2])[None, :]
    return (prefix[x1, y1, z1] - prefix[x0, y1, z1] - prefix[x1, y0, z1] - prefix[x1, y1, z0]
            + prefix[x0, y0, z1] + prefix[x0, y1, z0] + prefix[x1, y0, z0] - prefix[x0, y0, z0])


def check_box(gt_grid, pred_grid, structure, limits=Limits()):
    """Return exact absolute/aligned occupancy intersections, IoUs and shift.

    Empty sets follow the evaluator convention (zero shift; both-empty IoU=1).
    Nonempty predictions must equal their full bounding box or six-face shell.
    The origin stored on a world Grid is metadata and is never applied twice.
    """
    _validate_grid(gt_grid, limits); _validate_grid(pred_grid, limits)
    gt, pred = gt_grid.voxels, pred_grid.voxels
    pred_min, pred_max, pred_dims = _shape(pred, structure)
    absolute = sum(c in pred for c in gt)
    common = {'absolute_intersection': absolute, 'absolute_occupancy_iou': _iou(absolute, len(gt), len(pred)),
              'structure': structure, 'gt_occupied_count': len(gt), 'pred_occupied_count': len(pred),
              'tie_break': 'minimum_L1_then_lexicographic_XYZ', 'material_labels_used': False,
              'rotation': False, 'scale': False, 'approximation': False}
    if not gt or not pred:
        return dict(common, aligned_intersection=0, aligned_occupancy_iou=_iou(0, len(gt), len(pred)),
                    shift_xyz=[0, 0, 0], method='independent_exact_empty_sets', search_complete=True,
                    candidates_evaluated=1, empty_prediction=not bool(pred))
    gt_min, gt_max = _bounds(gt)
    gt_dims = tuple(gt_max[i] - gt_min[i] + 1 for i in range(3))
    prefix_shape = tuple(d + 1 for d in gt_dims)
    search_shape = tuple(gt_dims[i] + pred_dims[i] - 1 for i in range(3))
    prefix_cells, candidate_count = math.prod(prefix_shape), math.prod(search_shape)
    slab_cells = search_shape[1] * search_shape[2]
    has_inner = structure == 'six_face_shell' and all(d > 2 for d in pred_dims)
    # Prefix and cumsum scratch plus a conservative allowance for query arrays,
    # masks, L1 tie distances and broadcast indices. Input Grids already exist.
    estimated_bytes = 16 * prefix_cells + 96 * slab_cells + 128 * sum(search_shape)
    estimated_work = 3 * prefix_cells + (16 if has_inner else 8) * candidate_count + len(gt) + len(pred)
    bounds = [[gt_min[i] - pred_max[i], gt_max[i] - pred_min[i]] for i in range(3)]
    resource = {'prefix_cells': prefix_cells, 'candidate_count': candidate_count,
                'search_shape_xyz': list(search_shape), 'prefix_shape_xyz': list(prefix_shape),
                'estimated_scratch_bytes': estimated_bytes, 'estimated_work': estimated_work,
                'xyz_search_bounds': bounds}
    if prefix_cells > limits.max_alignment_transform_cells or candidate_count > limits.max_alignment_transform_cells:
        raise IndependentGeometryResourceLimit('prefix or search cube exceeds cell limit', resource)
    if estimated_bytes > min(limits.max_numpy_bytes, limits.max_alignment_memory_bytes):
        raise IndependentGeometryResourceLimit('summed-volume scratch exceeds memory limit', resource)
    if estimated_work > min(limits.max_alignment_work, limits.max_alignment_transform_work):
        raise IndependentGeometryResourceLimit('summed-volume queries exceed work limit', resource)
    # Every dense allocation is below the checks above. No padded predictions or
    # dense world-origin-to-origin canvas is constructed.
    prefix = np.zeros(prefix_shape, dtype=np.int64)
    for c in gt:
        prefix[tuple(c[i] - gt_min[i] + 1 for i in range(3))] = 1
    for axis in range(3): np.cumsum(prefix, axis=axis, dtype=np.int64, out=prefix)
    y = np.arange(1 - pred_dims[1], gt_dims[1], dtype=np.int64)
    z = np.arange(1 - pred_dims[2], gt_dims[2], dtype=np.int64)
    shifts_y = y + gt_min[1] - pred_min[1]; shifts_z = z + gt_min[2] - pred_min[2]
    l1_yz = np.abs(shifts_y)[:, None] + np.abs(shifts_z)[None, :]
    best_count, best_shift, best_key = -1, None, None
    for x in range(1 - pred_dims[0], gt_dims[0]):
        counts = _queries(prefix, x, y, z, pred_dims, gt_dims)
        if has_inner:
            counts -= _queries(prefix, x + 1, y + 1, z + 1, tuple(d - 2 for d in pred_dims), gt_dims)
        count = int(counts.max())
        if count < best_count: continue
        mask = counts == count
        minimum_yz = int(np.min(l1_yz, where=mask, initial=np.iinfo(np.int64).max))
        flat = int(np.argmax(mask & (l1_yz == minimum_yz)))
        shift = (x + gt_min[0] - pred_min[0], int(shifts_y[flat // len(z)]), int(shifts_z[flat % len(z)]))
        key = (abs(shift[0]) + minimum_yz, shift)
        if count > best_count or best_key is None or key < best_key:
            best_count, best_shift, best_key = count, shift, key
    if not 1 <= best_count <= min(len(gt), len(pred)):
        raise ValueError('independent summed-volume maximum is inconsistent')
    # Independently recheck the selected shift by sparse coordinate membership.
    selected = sum(tuple(c[i] + best_shift[i] for i in range(3)) in gt for c in pred)
    if selected != best_count:
        raise ValueError('summed-volume selected-shift sparse verification failed')
    return dict(common, aligned_intersection=best_count,
                aligned_occupancy_iou=_iou(best_count, len(gt), len(pred)), shift_xyz=list(best_shift),
                method='independent_int64_3d_prefix_sum_outer_minus_inner' if has_inner else 'independent_int64_3d_prefix_sum_box',
                search_complete=True, candidates_evaluated=candidate_count,
                selected_shift_sparse_verified=True, **resource)
