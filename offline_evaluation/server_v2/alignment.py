"""Exact occupancy translation alignment, independent of material labels.

For nonempty finite sets, every translation with positive overlap is a GT
coordinate minus a prediction coordinate. Sparse difference counting resolves
that set directly. Larger compact shapes use integer polynomial convolution
over the prime 998244353: each coefficient is at most min(|GT|,|prediction|),
strictly below the prime, so modular coefficients are the exact integer overlap.
No floating-point FFT, numerical rounding, rotation, scaling or clipping occurs.

Only occupied bounding boxes enter the convolution. With joint output strides,
the flattened XYZ polynomial has no coordinate carries. Reversing the local
prediction turns convolution indices into translations; stored minima restore
the exact world shift even when both world origins are distant.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Iterable

import numpy as np

from .core import Coord, Limits, ValidationError, _integer

MODULUS = 998_244_353
PRIMITIVE_ROOT = 3
MAX_TRANSFORM_CELLS = 1 << 23


class AlignmentResourceLimit(ValidationError):
    """An exact search was refused before its configured resource ceiling."""

    def __init__(self, reason: str, metadata: dict):
        super().__init__("ALIGNMENT_RESOURCE_LIMIT: " + reason)
        self.metadata = dict(metadata, status="ALIGNMENT_RESOURCE_LIMIT",
                             search_complete=False, resource_limit_reached=True,
                             reason=reason)


def _coordinates(values: Iterable[Coord], limits: Limits) -> set[Coord]:
    result = set()
    for coordinate in values:
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 3:
            raise ValidationError("alignment coordinate must contain three XYZ integers")
        item = tuple(_integer(v, "alignment coordinate") for v in coordinate)
        result.add(item)
        if len(result) > limits.max_voxels:
            raise AlignmentResourceLimit("occupied coordinate count exceeds voxel limit", {})
    return result


def _bounds(coords: set[Coord]) -> tuple[Coord, Coord]:
    return (tuple(min(c[i] for c in coords) for i in range(3)),
            tuple(max(c[i] for c in coords) for i in range(3)))


def _overlap(gt: set[Coord], pred: set[Coord], shift: Coord) -> int:
    dx, dy, dz = shift
    if len(pred) <= len(gt):
        return sum((x + dx, y + dy, z + dz) in gt for x, y, z in pred)
    return sum((x - dx, y - dy, z - dz) in pred for x, y, z in gt)


def _tie(shift: Coord):
    return sum(abs(v) for v in shift), shift


def _result(metadata: dict, shift: Coord, overlap: int, **extra) -> dict:
    return dict(metadata, status="PASS", search_complete=True,
                resource_limit_reached=False, shift_xyz=list(shift),
                maximum_occupancy_intersection=overlap, **extra)


def _bit_reversed_indices(length: int) -> np.ndarray:
    source = np.arange(length, dtype=np.int64)
    result = np.zeros(length, dtype=np.int64)
    for _ in range(length.bit_length() - 1):
        result <<= 1
        result |= source & 1
        source >>= 1
    return result


def _root_powers(root: int, count: int) -> np.ndarray:
    """Modular powers by exact doubling; no overflowing exponentiation."""
    powers = np.ones(count, dtype=np.int64)
    done = 1
    while done < count:
        powers[done:2 * done] = powers[:done] * pow(root, done, MODULUS) % MODULUS
        done *= 2
    return powers


def _ntt_inplace(values: np.ndarray, reverse: np.ndarray, *, inverse=False):
    values[:] = values[reverse]
    size = 2
    while size <= len(values):
        root = pow(PRIMITIVE_ROOT, (MODULUS - 1) // size, MODULUS)
        if inverse:
            root = pow(root, MODULUS - 2, MODULUS)
        half = size // 2
        twiddle = _root_powers(root, half)
        view = values.reshape(-1, size)
        left = view[:, :half].copy()
        right = view[:, half:] * twiddle % MODULUS
        view[:, :half] = (left + right) % MODULUS
        view[:, half:] = (left - right) % MODULUS
        size *= 2
    if inverse:
        values *= pow(len(values), MODULUS - 2, MODULUS)
        values %= MODULUS


def _integer_correlation(gt: set[Coord], pred: set[Coord], gt_min: Coord,
                         pred_min: Coord, gt_dims: Coord, pred_dims: Coord,
                         output_dims: Coord, transform_cells: int) -> np.ndarray:
    g = np.zeros(transform_cells, dtype=np.int64)
    p = np.zeros(transform_cells, dtype=np.int64)
    stride_x = output_dims[1] * output_dims[2]
    stride_y = output_dims[2]
    gi = np.fromiter(((x - gt_min[0]) * stride_x + (y - gt_min[1]) * stride_y
                      + z - gt_min[2] for x, y, z in gt), dtype=np.int64, count=len(gt))
    pi = np.fromiter(((pred_dims[0] - 1 - (x - pred_min[0])) * stride_x
                      + (pred_dims[1] - 1 - (y - pred_min[1])) * stride_y
                      + pred_dims[2] - 1 - (z - pred_min[2]) for x, y, z in pred),
                     dtype=np.int64, count=len(pred))
    g[gi] = 1
    p[pi] = 1
    del gi, pi
    reverse = _bit_reversed_indices(transform_cells)
    _ntt_inplace(g, reverse)
    _ntt_inplace(p, reverse)
    g *= p
    g %= MODULUS
    del p
    _ntt_inplace(g, reverse, inverse=True)
    return g[:math.prod(output_dims)]


def select_translation(gt_coordinates: Iterable[Coord], pred_coordinates: Iterable[Coord], *,
                       mode: str = "strict_translation", radius: int = 2,
                       limits: Limits = Limits()) -> dict:
    """Select one globally exact or explicitly bounded occupancy-optimal shift.

    Coordinate-only callers can safely score geometry without inventing material
    identifiers. Coordinates are world XYZ integers. Duplicate coordinates are
    set members, not repeated observations. All ties use L1 then XYZ lexicographic
    order; for either empty set every shift ties and the selected shift is zero.

    Resource errors carry structured metadata and no approximate successful
    result. ``candidates_evaluated`` counts actual coefficient/shift processing;
    a verified upper-bound certificate can resolve the full domain without
    enumerating every candidate, and is explicitly identified in the result.
    """
    if mode not in {"bounded", "strict_translation"}:
        raise ValidationError("alignment mode must be bounded or strict_translation")
    metadata = {"mode": mode, "objective": "maximum_occupancy_intersection",
                "tie_break": "minimum_L1_then_lexicographic_XYZ",
                "all_aligned_metrics_share_shift": True,
                "rotation": False, "scale": False, "clipping": False}
    try:
        gt = _coordinates(gt_coordinates, limits)
        pred = _coordinates(pred_coordinates, limits)
    except AlignmentResourceLimit as exc:
        exc.metadata = dict(metadata, **exc.metadata)
        raise
    if mode == "bounded":
        radius = _integer(radius, "alignment_radius")
        if not 0 <= radius <= limits.max_alignment_radius:
            raise ValidationError("alignment radius outside configured bounds")
        candidates = (2 * radius + 1) ** 3
        metadata.update(method="exact_exhaustive_translation", radius=radius,
                        xyz_search_bounds=[[-radius, radius] for _ in range(3)],
                        globally_optimal_unrestricted=False,
                        candidate_domain="bounded_integer_xyz_cube",
                        candidates_total=candidates)
        if candidates * min(len(gt), len(pred)) > limits.max_alignment_work:
            raise AlignmentResourceLimit("exact bounded alignment exceeds comparison work limit", metadata)
        best_shift, best_count = (0, 0, 0), -1
        for shift in sorted(itertools.product(range(-radius, radius + 1), repeat=3), key=_tie):
            count = _overlap(gt, pred, shift)
            if count > best_count:
                best_shift, best_count = shift, count
        return _result(metadata, best_shift, best_count, candidates_evaluated=candidates,
                       all_candidates_processed=True, all_candidates_resolved=True,
                       optimality_certificate="exhaustive_bounded_cube")

    metadata.update(radius=None, globally_optimal_unrestricted=True,
                    candidate_domain="all_integer_xyz_translations")
    if not gt or not pred:
        return _result(metadata, (0, 0, 0), 0, method="empty_set_exact",
                       xyz_search_bounds=None, candidates_total=None, candidates_evaluated=1,
                       all_candidates_processed=False, all_candidates_resolved=True,
                       optimality_certificate="all_shifts_have_zero_overlap_minimum_L1_is_zero")

    gt_min, gt_max = _bounds(gt)
    pred_min, pred_max = _bounds(pred)
    low = tuple(gt_min[i] - pred_max[i] for i in range(3))
    high = tuple(gt_max[i] - pred_min[i] for i in range(3))
    gt_dims = tuple(gt_max[i] - gt_min[i] + 1 for i in range(3))
    pred_dims = tuple(pred_max[i] - pred_min[i] + 1 for i in range(3))
    output_dims = tuple(gt_dims[i] + pred_dims[i] - 1 for i in range(3))
    candidates = math.prod(output_dims)
    metadata.update(xyz_search_bounds=[[low[i], high[i]] for i in range(3)],
                    positive_overlap_support="complete_bbox_difference_cube; outside overlap is zero",
                    candidates_total=candidates, gt_occupied_origin_xyz=list(gt_min),
                    pred_occupied_origin_xyz=list(pred_min))
    zero_overlap = _overlap(gt, pred, (0, 0, 0))
    if zero_overlap == min(len(gt), len(pred)):
        return _result(metadata, (0, 0, 0), zero_overlap, method="verified_overlap_upper_bound",
                       candidates_evaluated=1, all_candidates_processed=False,
                       all_candidates_resolved=True,
                       optimality_certificate="minimum_count_upper_bound_at_unique_minimum_L1_shift")
    translated_minima = tuple(gt_min[i] - pred_min[i] for i in range(3))
    if len(gt) == len(pred) and gt_dims == pred_dims:
        count = _overlap(gt, pred, translated_minima)
        if count == len(gt):
            return _result(metadata, translated_minima, count,
                           method="verified_identical_shape_upper_bound", candidates_evaluated=2,
                           all_candidates_processed=False, all_candidates_resolved=True,
                           optimality_certificate="finite_nonempty_equal_sets_have_unique_full_overlap_translation")

    pair_count = len(gt) * len(pred)
    if pair_count <= limits.max_alignment_sparse_pairs:
        estimated_memory = pair_count * 320 + 160 * (len(gt) + len(pred))
        if pair_count > limits.max_alignment_work or estimated_memory > limits.max_alignment_memory_bytes:
            raise AlignmentResourceLimit("exact sparse difference counting exceeds work or memory limit", metadata)
        differences = Counter((gx - px, gy - py, gz - pz)
                              for gx, gy, gz in gt for px, py, pz in pred)
        best_count = max(differences.values())
        best_shift = min((shift for shift, count in differences.items() if count == best_count), key=_tie)
        return _result(metadata, best_shift, best_count, method="exact_sparse_relative_difference_count",
                       candidates_evaluated=len(differences), coordinate_pairs_processed=pair_count,
                       all_candidates_processed=False, all_candidates_resolved=True,
                       all_coordinate_pairs_processed=True,
                       estimated_workspace_bytes=estimated_memory,
                       optimality_certificate="all_coordinate_differences_counted_outside_support_zero")

    transform_cells = 1 << (candidates - 1).bit_length()
    transform_work = 3 * transform_cells * (transform_cells.bit_length() - 1)
    # Includes both int64 operands, bit reversal, butterfly intermediates and
    # local index arrays and normalized coordinate-set containers. The caller's
    # existing material grids are outside this additional-workspace ceiling.
    estimated_memory = 80 * transform_cells + 160 * (len(gt) + len(pred))
    metadata.update(transform_cells=transform_cells, transform_work=transform_work,
                    estimated_workspace_bytes=estimated_memory)
    if (transform_cells > min(MAX_TRANSFORM_CELLS, limits.max_alignment_transform_cells)
            or transform_work > limits.max_alignment_transform_work
            or estimated_memory > limits.max_alignment_memory_bytes
            or min(len(gt), len(pred)) >= MODULUS):
        raise AlignmentResourceLimit("exact integer correlation exceeds transform, work, memory or modulus limit", metadata)
    correlation = _integer_correlation(gt, pred, gt_min, pred_min, gt_dims, pred_dims,
                                       output_dims, transform_cells)
    best_count = int(correlation.max())
    if not 1 <= best_count <= min(len(gt), len(pred)):
        raise ValidationError("exact integer correlation failed overlap range verification")
    maxima = np.flatnonzero(correlation == best_count)

    def shift_from_index(value):
        value = int(value)
        x, rest = divmod(value, output_dims[1] * output_dims[2])
        y, z = divmod(rest, output_dims[2])
        return (low[0] + x, low[1] + y, low[2] + z)

    best_shift = min((shift_from_index(value) for value in maxima), key=_tie)
    if _overlap(gt, pred, best_shift) != best_count:
        raise ValidationError("exact integer correlation failed selected-shift verification")
    return _result(metadata, best_shift, best_count, method="exact_integer_ntt_correlation",
                   candidates_evaluated=candidates, all_candidates_processed=True,
                   all_candidates_resolved=True, correlation_output_shape_xyz=list(output_dims),
                   modular_prime=MODULUS, primitive_root=PRIMITIVE_ROOT,
                   max_possible_overlap=min(len(gt), len(pred)), selected_shift_verified=True,
                   arithmetic="exact_int64_modular_no_floating_point",
                   local_to_world_shift="gt_min-pred_min+correlation_xyz-(pred_shape_xyz-1)",
                   optimality_certificate="every_integer_overlap_coefficient_strictly_below_modular_prime")
