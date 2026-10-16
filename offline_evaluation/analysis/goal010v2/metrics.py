"""Exact, CPU-only diagnostic metrics at an externally verified fixed shift.

No alignment search, label correction, file loading, network, or failure imputation
occurs here. Input dictionaries must already be validated canonical world grids.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from ...server_v2.core import Limits, material_id


PROTOCOL_ID = 'goal010v2_fixed_shift_metrics_v1'
COORDINATE_SYSTEM = 'world_XYZ'
ALIGNMENT_PROTOCOL = 'server_v2.2_strict_translation'
MAX_COORDINATE_ABS = min(1_000_000, Limits().max_coordinate_abs)
MAX_OCCUPIED = 2_000_000
MAX_SURFACE_POINTS = 2_000_000
MAX_ESTIMATED_SCRATCH_BYTES = 512 * 1024 * 1024
CPU_BUDGET_SECONDS = 120.0
NEIGHBORS6 = ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
              (0, 1, 0), (0, 0, -1), (0, 0, 1))
SCALES = ((1, '0'), (2, '0'), (2, 'half'), (4, '0'),
          (4, 'half'), (8, '0'), (8, 'half'))
SURFACE_METRIC_IDS = tuple(f'surface_{term}_at_{k}' for k in (1, 2)
                           for term in ('precision', 'recall', 'f1'))
MULTISCALE_METRIC_IDS = tuple(f'multiscale_iou_k{k}_anchor{a}' for k, a in SCALES)
MATERIAL_METRIC_IDS = ('exact_material_match_count', 'conditional_exact_material_accuracy',
                       'gt_material_recall', 'macro_gt_material_recall', 'material_aware_iou')
CONNECTIVITY_METRIC_IDS = ('gt_components6', 'pred_components6',
                          'gt_largest_component_fraction6', 'pred_largest_component_fraction6')
CORE_METRICS = ('occupancy_precision', 'occupancy_recall', 'footprint_iou',
                'conditional_exact_material_accuracy', 'surface_f1_at_1', 'surface_f1_at_2')
FRAME_METRIC_IDS = (
    'gt_occupied_count', 'pred_occupied_count', 'occupancy_intersection', 'occupancy_union',
    'occupancy_precision', 'occupancy_recall', 'occupancy_f1', 'occupancy_iou',
    'excess_count', 'omission_count', 'volume_ratio', 'relative_volume_error',
    'gt_footprint_columns', 'pred_footprint_columns', 'common_footprint_columns',
    'union_footprint_columns', 'footprint_iou', 'column_count_mae', 'common_top_y_mae',
    'common_footprint_gt_coverage', 'common_footprint_pred_coverage',
    *MATERIAL_METRIC_IDS, 'gt_surface_voxels', 'pred_surface_voxels',
    *SURFACE_METRIC_IDS, *MULTISCALE_METRIC_IDS, *CONNECTIVITY_METRIC_IDS,
)
BOUNDED_FAILURE0_METRICS = frozenset((
    'occupancy_precision', 'occupancy_recall', 'occupancy_f1', 'occupancy_iou',
    'footprint_iou', 'gt_material_recall', 'macro_gt_material_recall', 'material_aware_iou',
    *SURFACE_METRIC_IDS, *MULTISCALE_METRIC_IDS,
))
# Counts, diagnostic fractions, volume, and errors remain useful comparisons, but
# never acquire a success score or failure0 merely because they happen to be bounded.
BASELINE_GEOMETRY_METRICS = frozenset(FRAME_METRIC_IDS) - frozenset(MATERIAL_METRIC_IDS)


class MetricResourceLimit(RuntimeError):
    """No approximate or zero-valued replacement is permitted."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason
        self.status = 'METRIC_RESOURCE_LIMIT'


class _Budget:
    def __init__(self):
        self.started = time.process_time()

    def check(self):
        if time.process_time() - self.started > CPU_BUDGET_SECONDS:
            raise MetricResourceLimit('CPU_BUDGET_EXCEEDED')


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_world(world, label, budget):
    if type(world) is not dict:
        raise ValueError(f'{label}: canonical world dictionary required')
    if len(world) > MAX_OCCUPIED:
        raise MetricResourceLimit('OCCUPIED_COUNT_LIMIT')
    for index, (point, material) in enumerate(world.items()):
        if (type(point) is not tuple or len(point) != 3
                or any(type(x) is not int for x in point)):
            raise ValueError(f'{label}: XYZ tuple of three integers required')
        if any(abs(x) > MAX_COORDINATE_ABS for x in point):
            raise ValueError(f'{label}: input coordinate bound exceeded')
        # Frozen readers remove minecraft: and air. Do not silently normalize here.
        if not isinstance(material, str) or material_id(material) != material or material == 'air':
            raise ValueError(f'{label}: canonical non-air material ID required')
        if index % 8192 == 0:
            budget.check()


def _grid_sha256(world, budget):
    digest = hashlib.sha256(b'goal010v2_world_XYZ_canonical_nonair_v1\n')
    for index, point in enumerate(sorted(world)):
        digest.update(json.dumps([*point, world[point]], separators=(',', ':'),
                                 ensure_ascii=True).encode('ascii') + b'\n')
        if index % 8192 == 0:
            budget.check()
    return digest.hexdigest()


def _ratio(numerator, denominator, metric, reasons, reason):
    if denominator == 0:
        reasons[metric] = reason
        return None
    return numerator / denominator


def _columns(world):
    counts, tops = Counter(), {}
    for x, y, z in world:
        key = (x, z)
        counts[key] += 1
        tops[key] = max(y, tops.get(key, y))
    return counts, tops


def _core(gt, pred, budget):
    """All scalar computations here are sparse and exact (up to final ratios)."""
    g, p = len(gt), len(pred)
    common = gt.keys() & pred.keys()
    c, m = len(common), sum(gt[x] == pred[x] for x in common)
    reasons = {}
    values = {
        'gt_occupied_count': g, 'pred_occupied_count': p,
        'occupancy_intersection': c, 'occupancy_union': g + p - c,
        'occupancy_precision': _ratio(c, p, 'occupancy_precision', reasons, 'EMPTY_PREDICTION'),
        'occupancy_recall': _ratio(c, g, 'occupancy_recall', reasons, 'EMPTY_GT'),
        'occupancy_f1': 2 * c / (g + p) if g + p else 1.0,
        'occupancy_iou': c / (g + p - c) if g + p - c else 1.0,
        'excess_count': p - c, 'omission_count': g - c,
        'volume_ratio': _ratio(p, g, 'volume_ratio', reasons, 'EMPTY_GT'),
        'relative_volume_error': _ratio(p - g, g, 'relative_volume_error', reasons, 'EMPTY_GT'),
        'exact_material_match_count': m,
        'conditional_exact_material_accuracy': _ratio(m, c, 'conditional_exact_material_accuracy',
                                                       reasons, 'NO_OCCUPANCY_INTERSECTION'),
        'gt_material_recall': _ratio(m, g, 'gt_material_recall', reasons, 'EMPTY_GT'),
        'material_aware_iou': m / (g + p - m) if g + p else 1.0,
    }
    budget.check()
    gc, gy = _columns(gt)
    pc, py = _columns(pred)
    footprint_common, footprint_union = gc.keys() & pc.keys(), gc.keys() | pc.keys()
    cf, uf = len(footprint_common), len(footprint_union)
    values.update({
        'gt_footprint_columns': len(gc), 'pred_footprint_columns': len(pc),
        'common_footprint_columns': cf, 'union_footprint_columns': uf,
        'footprint_iou': cf / uf if uf else 1.0,
        'column_count_mae': _ratio(sum(abs(gc[x] - pc[x]) for x in footprint_union), uf,
                                   'column_count_mae', reasons, 'NO_FOOTPRINT_COLUMNS'),
        'common_top_y_mae': _ratio(sum(abs(gy[x] - py[x]) for x in footprint_common), cf,
                                   'common_top_y_mae', reasons, 'NO_COMMON_FOOTPRINT'),
        'common_footprint_gt_coverage': _ratio(cf, len(gc), 'common_footprint_gt_coverage',
                                               reasons, 'EMPTY_GT_FOOTPRINT'),
        'common_footprint_pred_coverage': _ratio(cf, len(pc), 'common_footprint_pred_coverage',
                                                 reasons, 'EMPTY_PREDICTION_FOOTPRINT'),
    })
    material_count, matched_by_material, confusion = Counter(gt.values()), Counter(), Counter()
    for index, (point, material) in enumerate(gt.items()):
        other = pred.get(point)
        confusion[(material, other)] += 1
        if material == other:
            matched_by_material[material] += 1
        if index % 8192 == 0:
            budget.check()
    for point, material in pred.items():
        if point not in gt:
            confusion[(None, material)] += 1
    recalls = [{'material_id': material, 'gt_count': count,
                'exact_match_count': matched_by_material[material],
                'recall': matched_by_material[material] / count}
               for material, count in sorted(material_count.items())]
    values['macro_gt_material_recall'] = _ratio(sum(x['recall'] for x in recalls), len(recalls),
                                                'macro_gt_material_recall', reasons, 'NO_GT_MATERIALS')
    multiscale = {}
    for k, anchor in SCALES:
        budget.check()
        a = 0 if anchor == '0' else k // 2
        # Python // is mathematical floor, including negative integers.
        gs = {tuple((v - a) // k for v in point) for point in gt}
        ps = {tuple((v - a) // k for v in point) for point in pred}
        intersection = len(gs & ps)
        union = len(gs) + len(ps) - intersection
        metric = f'multiscale_iou_k{k}_anchor{anchor}'
        values[metric] = intersection / union if union else 1.0
        multiscale[metric] = {'k': k, 'anchor_xyz': [a, a, a], 'gt_cells': len(gs),
                             'pred_cells': len(ps), 'intersection': intersection, 'union': union,
                             'diagnostic_only': k == 8}
    details = {
        'null_reasons': reasons,
        'confusion': [{'gt_material_id': a, 'pred_material_id': b, 'count': count}
                      for (a, b), count in sorted(confusion.items(),
                                                 key=lambda x: (x[0][0] or '', x[0][1] or ''))],
        'material_recalls': recalls,
        'material_vocabulary': sorted(set(gt.values()) | set(pred.values())),
        'column_counts': {'gt_only': len(gc) - cf, 'pred_only': len(pc) - cf,
                          'common': cf, 'union': uf, 'missing_side_occupancy_count': 0},
        'boundary_counts': {}, 'multiscale_counts': multiscale,
        'flags': {'empty_gt': not bool(gt), 'empty_prediction': not bool(pred),
                  'both_empty_fixture_only': not gt and not pred,
                  'failure0_applied': False, 'approximation': False},
    }
    budget.check()
    return values, details


def _boundary(world, budget):
    result = []
    for index, (x, y, z) in enumerate(world):
        if any((x + dx, y + dy, z + dz) not in world for dx, dy, dz in NEIGHBORS6):
            result.append((x, y, z))
            if len(result) > MAX_SURFACE_POINTS:
                raise MetricResourceLimit('SURFACE_POINT_LIMIT')
        if index % 8192 == 0:
            budget.check()
    return result


def _surface(gt_boundary, pred_boundary, budget):
    import numpy as np
    from scipy.spatial import cKDTree

    g, p = len(gt_boundary), len(pred_boundary)
    matched = {k: [0, 0] for k in (1, 2)}
    budget.check()
    if g and p:
        ga, pa = np.asarray(gt_boundary, dtype=np.float64), np.asarray(pred_boundary, dtype=np.float64)
        # eps=0, workers=1; no pairwise dense distance matrix or approximate NN.
        pred_distance = cKDTree(ga).query(pa, k=1, eps=0, workers=1)[0]
        budget.check()
        gt_distance = cKDTree(pa).query(ga, k=1, eps=0, workers=1)[0]
        for k in (1, 2):
            matched[k] = [int(np.count_nonzero(pred_distance <= k)),
                          int(np.count_nonzero(gt_distance <= k))]
    budget.check()
    values, reasons, counts = {}, {}, {}
    for k in (1, 2):
        pm, gm = matched[k]
        precision_id, recall_id, f1_id = (f'surface_{x}_at_{k}' for x in ('precision', 'recall', 'f1'))
        precision = _ratio(pm, p, precision_id, reasons, 'EMPTY_PREDICTION_BOUNDARY')
        recall = _ratio(gm, g, recall_id, reasons, 'EMPTY_GT_BOUNDARY')
        if not g and not p:
            f1 = 1.0
        elif not g or not p or not precision or not recall:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        values.update({precision_id: precision, recall_id: recall, f1_id: f1})
        counts[str(k)] = {'threshold_blocks': k, 'pred_within_threshold': pm,
                          'gt_within_threshold': gm, 'pred_boundary': p, 'gt_boundary': g}
    return values, reasons, counts


def _components(world, budget):
    remaining = set(world)
    components, largest, visited = 0, 0, 0
    while remaining:
        stack = [remaining.pop()]
        components += 1
        size = 0
        while stack:
            x, y, z = stack.pop()
            size += 1
            visited += 1
            for dx, dy, dz in NEIGHBORS6:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
            if visited % 8192 == 0:
                budget.check()
        largest = max(largest, size)
    budget.check()
    return components, largest / len(world) if world else None


def _mark_null(values, details, ids, reason):
    for metric in ids:
        values[metric] = None
        details['null_reasons'][metric] = reason


def _optional_reason(exc):
    if isinstance(exc, MetricResourceLimit):
        return 'METRIC_RESOURCE_LIMIT:' + exc.reason
    if isinstance(exc, MemoryError):
        return 'METRIC_RESOURCE_LIMIT:MEMORY_ERROR'
    return 'DEPENDENCY_UNAVAILABLE:SCIPY_OR_NUMPY'


def compute_metrics(gt_world_dict, pred_world_dict, shift_xyz):
    """Measure absolute and saved strict-aligned grids, without optimizing shift.

    Invalid input raises ValueError; core resource excess raises MetricResourceLimit.
    Optional surface/connectivity limits yield per-metric NULL plus safe reason.
    Caller retains all failed/unknown slots with NULL, never treats them as empty.
    """
    budget = _Budget()
    if (not isinstance(shift_xyz, (tuple, list)) or len(shift_xyz) != 3
            or any(type(x) is not int for x in shift_xyz)):
        raise ValueError('saved shift must contain exactly three XYZ integers')
    shift = tuple(shift_xyz)
    if any(abs(x) > 2 * MAX_COORDINATE_ABS for x in shift):
        raise ValueError('saved shift exceeds possible input coordinate difference')
    if type(gt_world_dict) is not dict or type(pred_world_dict) is not dict:
        raise ValueError('canonical world dictionaries required')
    estimated = 320 * (len(gt_world_dict) + len(pred_world_dict))
    if estimated > MAX_ESTIMATED_SCRATCH_BYTES:
        raise MetricResourceLimit('ESTIMATED_SCRATCH_LIMIT')
    _validate_world(gt_world_dict, 'GT', budget)
    _validate_world(pred_world_dict, 'prediction', budget)
    gt_hash, pred_hash = (_grid_sha256(x, budget) for x in (gt_world_dict, pred_world_dict))
    gt, pred = gt_world_dict, pred_world_dict
    shifted = {tuple(point[i] + shift[i] for i in range(3)): material
               for point, material in pred.items()} if any(shift) else pred
    absolute, absolute_details = _core(gt, pred, budget)
    aligned, aligned_details = _core(gt, shifted, budget)
    frames = {'absolute': absolute, 'aligned': aligned}
    details = {'absolute': absolute_details, 'aligned': aligned_details}
    try:
        gb, pb = _boundary(gt, budget), _boundary(pred, budget)
    except (MetricResourceLimit, MemoryError) as exc:
        for frame in frames:
            _mark_null(frames[frame], details[frame],
                       ('gt_surface_voxels', 'pred_surface_voxels', *SURFACE_METRIC_IDS), _optional_reason(exc))
    else:
        for frame in frames:
            frames[frame].update(gt_surface_voxels=len(gb), pred_surface_voxels=len(pb))
            details[frame]['boundary_counts'] = {'gt_boundary': len(gb), 'pred_boundary': len(pb)}
            try:
                current = [tuple(point[i] + shift[i] for i in range(3)) for point in pb] if frame == 'aligned' and any(shift) else pb
                values, reasons, counts = _surface(gb, current, budget)
            except (MetricResourceLimit, MemoryError, ImportError) as exc:
                _mark_null(frames[frame], details[frame], SURFACE_METRIC_IDS, _optional_reason(exc))
            else:
                frames[frame].update(values)
                details[frame]['null_reasons'].update(reasons)
                details[frame]['boundary_counts']['thresholds'] = counts
    # Translation preserves 6-neighbor components, so compute exactly once/grid.
    try:
        gc, gf = _components(gt, budget)
        pc, pf = _components(pred, budget)
    except (MetricResourceLimit, MemoryError) as exc:
        for frame in frames:
            _mark_null(frames[frame], details[frame], CONNECTIVITY_METRIC_IDS, _optional_reason(exc))
    else:
        for frame in frames:
            frames[frame].update(gt_components6=gc, pred_components6=pc,
                                 gt_largest_component_fraction6=gf, pred_largest_component_fraction6=pf)
            for metric, value in [('gt_largest_component_fraction6', gf), ('pred_largest_component_fraction6', pf)]:
                if value is None:
                    details[frame]['null_reasons'][metric] = 'EMPTY_OCCUPANCY'
    dimensions = [max(c[i] for c in gt) - min(c[i] for c in gt) + 1 for i in range(3)] if gt else [0, 0, 0]
    l1, l2 = sum(abs(v) for v in shift), math.sqrt(sum(v * v for v in shift))
    displacement = {'shift_xyz': list(shift), 'shift_l1': l1, 'shift_l2': l2,
                    'gt_occupied_dimensions_xyz': dimensions,
                    'shift_abs_relative_xyz': [abs(shift[i]) / dimensions[i] for i in range(3)] if gt else None,
                    'shift_l1_relative_extent_sum': l1 / sum(dimensions) if gt else None,
                    'shift_l2_relative_bbox_diagonal': l2 / math.sqrt(sum(v * v for v in dimensions)) if gt else None,
                    'occupancy_iou_gain': aligned['occupancy_iou'] - absolute['occupancy_iou']}
    details['displacement'] = {'null_reasons': {key: 'EMPTY_GT_OCCUPIED_BBOX'
                                               for key, value in displacement.items() if value is None}}
    for frame, values in frames.items():
        assert set(values) == set(FRAME_METRIC_IDS)
        assert {key for key, value in values.items() if value is None} == set(details[frame]['null_reasons'])
        assert values['multiscale_iou_k1_anchor0'] == values['occupancy_iou']
    shift_hash = hashlib.sha256(json.dumps(list(shift), separators=(',', ':')).encode()).hexdigest()
    binding = hashlib.sha256((gt_hash + ':' + pred_hash + ':' + shift_hash).encode()).hexdigest()
    return {**frames, 'displacement': displacement, 'details': details, 'provenance': {
        'protocol_id': PROTOCOL_ID, 'coordinate_system': COORDINATE_SYSTEM,
        'alignment_protocol': ALIGNMENT_PROTOCOL, 'stored_alignment_shift_xyz': list(shift),
        'alignment_searched': False, 'shift_optimality_checked': False,
        'gt_grid_sha256': gt_hash, 'pred_grid_sha256': pred_hash, 'shift_sha256': shift_hash,
        'inputs_and_shift_sha256': binding, 'implementation_sha256': implementation_sha256(),
        'estimated_scratch_bytes': estimated, 'CPU_budget_seconds': CPU_BUDGET_SECONDS,
        'failure0_applied': False, 'input_source_file_hashes_must_be_bound_by_caller': True,
    }}


def metric_registry():
    """Machine-readable registry; emitted to a frozen file before real runs."""
    formulas = {
        'gt_occupied_count': 'g=|G|', 'pred_occupied_count': 'p=|P|',
        'occupancy_intersection': 'C=|G intersect (P+tau)|', 'occupancy_union': 'g+p-C',
        'occupancy_precision': 'C/p', 'occupancy_recall': 'C/g',
        'occupancy_f1': '2*C/(g+p)', 'occupancy_iou': 'C/(g+p-C)',
        'excess_count': 'p-C', 'omission_count': 'g-C', 'volume_ratio': 'p/g',
        'relative_volume_error': '(p-g)/g',
        'gt_footprint_columns': '|F(G)|', 'pred_footprint_columns': '|F(P+tau)|',
        'common_footprint_columns': '|F(G) intersect F(P+tau)|',
        'union_footprint_columns': '|F(G) union F(P+tau)|',
        'footprint_iou': '|F(G) intersect F(P+tau)| / |F(G) union F(P+tau)|',
        'column_count_mae': 'sum_union_columns |n_G(x,z)-n_(P+tau)(x,z)| / union_footprint_columns',
        'common_top_y_mae': 'sum_common_columns |topY_G(x,z)-topY_(P+tau)(x,z)| / common_footprint_columns',
        'common_footprint_gt_coverage': 'common_footprint_columns/gt_footprint_columns',
        'common_footprint_pred_coverage': 'common_footprint_columns/pred_footprint_columns',
        'exact_material_match_count': 'M=count of common coordinates with identical canonical material ID',
        'conditional_exact_material_accuracy': 'M/C', 'gt_material_recall': 'M/g',
        'macro_gt_material_recall': 'mean over GT material IDs of exact matches(ID)/GT count(ID)',
        'material_aware_iou': 'M/(g+p-M)',
        'gt_surface_voxels': '|B_G|', 'pred_surface_voxels': '|B_(P+tau)|',
        'gt_components6': 'number of 6-neighbor components of G',
        'pred_components6': 'number of 6-neighbor components of P',
        'gt_largest_component_fraction6': 'largest 6-neighbor component(G)/g',
        'pred_largest_component_fraction6': 'largest 6-neighbor component(P)/p',
    }
    denominators = {
        'occupancy_precision': 'p', 'occupancy_recall': 'g', 'occupancy_f1': 'g+p',
        'occupancy_iou': 'g+p-C', 'volume_ratio': 'g', 'relative_volume_error': 'g',
        'footprint_iou': 'union_footprint_columns', 'column_count_mae': 'union_footprint_columns',
        'common_top_y_mae': 'common_footprint_columns',
        'common_footprint_gt_coverage': 'gt_footprint_columns',
        'common_footprint_pred_coverage': 'pred_footprint_columns',
        'conditional_exact_material_accuracy': 'C (common occupancy, not all GT)',
        'gt_material_recall': 'g', 'macro_gt_material_recall': 'number of distinct GT material IDs',
        'material_aware_iou': 'g+p-M', 'gt_largest_component_fraction6': 'g',
        'pred_largest_component_fraction6': 'p',
    }
    for k in (1, 2):
        for part, bound in [('precision', 'B_(P+tau)'), ('recall', 'B_G')]:
            metric = f'surface_{part}_at_{k}'
            formulas[metric] = f'fraction of {bound} with Euclidean nearest boundary-center distance <= {k}'
            denominators[metric] = '|' + bound + '|'
        formulas[f'surface_f1_at_{k}'] = f'harmonic mean of surface_precision_at_{k} and surface_recall_at_{k}'
        denominators[f'surface_f1_at_{k}'] = 'surface precision + recall (defined empty conventions)'
    for k, anchor in SCALES:
        metric = f'multiscale_iou_k{k}_anchor{anchor}'
        a = 0 if anchor == '0' else k // 2
        formulas[metric] = f'IoU of OR-pooled integer cells floor((XYZ-({a},{a},{a}))/{k})'
        denominators[metric] = 'union of pooled occupied cells'
    null_policy = ('NULL if its explicit denominator is zero, or resource/dependency failure; '
                   'both-empty fixture IoU/F1=1; one-empty F1=0; P/R keep NULL for empty denominator. '
                   'Invalid/unknown slots never enter compute_metrics and remain NULL. '
                   'Successful NULL values are never failure0-imputed.')
    lower = {'excess_count', 'omission_count', 'column_count_mae', 'common_top_y_mae'}
    fractions = {'conditional_exact_material_accuracy', 'common_footprint_gt_coverage',
                 'common_footprint_pred_coverage', 'gt_largest_component_fraction6',
                 'pred_largest_component_fraction6'}
    implementation = implementation_sha256()
    entries = []
    for metric in FRAME_METRIC_IDS:
        higher = metric in BOUNDED_FAILURE0_METRICS or metric == 'conditional_exact_material_accuracy'
        bounded = higher or metric in fractions
        is_count = metric.endswith('_count') or metric.endswith('_columns') or metric.endswith('_voxels') or metric in {
            'occupancy_intersection', 'occupancy_union', 'gt_components6', 'pred_components6'}
        entries.append({
            'metric_id': metric, 'section': 'frame', 'formula': formulas[metric],
            'unit': 'blocks/column' if metric == 'column_count_mae' else 'blocks' if metric == 'common_top_y_mae' else 'count' if is_count else 'ratio',
            'value_range': [0, 1] if bounded else [-1, None] if metric == 'relative_volume_error' else [0, None],
            'direction': 'higher' if higher else 'lower' if metric in lower else 'descriptive',
            'coordinate_system': COORDINATE_SYSTEM,
            'alignment': {'absolute': [0, 0, 0], 'aligned': 'same saved strict occupancy-optimal tau; no new optimization'},
            'denominator': denominators.get(metric, 'none (integer count)'),
            'null_policy': null_policy,
            'baseline_comparable': metric in BASELINE_GEOMETRY_METRICS,
            'failure0_allowed': metric in BOUNDED_FAILURE0_METRICS,
            'failure0_scope': 'only independently identified completed model-output failure; never successful NULL, unknown, resource, GT missing, or baseline missing',
            'priority': 'core' if metric in CORE_METRICS else 'diagnostic' if '_k8_' in metric or not higher and metric not in lower else 'additional',
            'limitations': ['No semantic rooms, floors, visibility, human effort, causal effect or perceptual-quality claim.',
                            'Surface boundary includes cavity walls; integer voxel centers with <= thresholds.',
                            'OR pooling may hide errors and depends on anchor; k8 is diagnostic only.',
                            'Conditional material accuracy selects overlapping positions; accompany M/g and material IoU.',
                            'Material IDs are exact; artificial baseline material is not interpreted.'],
            'implementation_sha256': implementation,
            'sample_hash_fields': ['provenance.gt_grid_sha256', 'provenance.pred_grid_sha256',
                                   'provenance.shift_sha256', 'provenance.inputs_and_shift_sha256'],
        })
    displacement_formulas = {
        'shift_xyz': 'saved tau=[dx,dy,dz]', 'shift_l1': '|dx|+|dy|+|dz|',
        'shift_l2': 'sqrt(dx^2+dy^2+dz^2)', 'gt_occupied_dimensions_xyz': 'maxXYZ(G)-minXYZ(G)+1, or [0,0,0] for empty fixture',
        'shift_abs_relative_xyz': '|tau_axis| / inclusive occupied GT extent_axis',
        'shift_l1_relative_extent_sum': 'L1(tau)/sum(inclusive occupied GT extents)',
        'shift_l2_relative_bbox_diagonal': 'L2(tau)/L2(inclusive occupied GT extents)',
        'occupancy_iou_gain': 'aligned occupancy IoU - absolute occupancy IoU',
    }
    for metric, formula in displacement_formulas.items():
        entries.append({'metric_id': metric, 'section': 'displacement', 'formula': formula,
                        'unit': 'ratio' if 'relative' in metric or metric.endswith('gain') else 'blocks',
                        'value_range': [-1, 1] if metric.endswith('gain') else [None, None] if metric == 'shift_xyz' else [0, None],
                        'direction': 'descriptive', 'coordinate_system': COORDINATE_SYSTEM,
                        'alignment': 'provided saved tau only; no new optimization',
                        'denominator': 'inclusive occupied GT extent(s)' if 'relative' in metric else 'none',
                        'null_policy': 'relative displacement NULL for empty GT; no failure0',
                        'baseline_comparable': True, 'failure0_allowed': False, 'priority': 'diagnostic',
                        'limitations': ['World origin may be unidentifiable from images; shift magnitude is not a quality ranking.',
                                        'Storage bbox is not used as the occupied bbox.'],
                        'implementation_sha256': implementation,
                        'sample_hash_fields': ['provenance.inputs_and_shift_sha256']})
    return {'schema': 'goal010v2_metric_registry_v1', 'protocol_id': PROTOCOL_ID,
            'analysis_label': 'POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC',
            'specification_type': 'post-hoc pre-computation specification', 'metrics': entries,
            'resource_limits': {'occupied_per_grid': MAX_OCCUPIED, 'boundary_per_grid': MAX_SURFACE_POINTS,
                                'input_coordinate_abs': MAX_COORDINATE_ABS,
                                'shift_coordinate_abs': 2 * MAX_COORDINATE_ABS,
                                'derived_aligned_coordinate_abs': 3 * MAX_COORDINATE_ABS,
                                'estimated_scratch_bytes': MAX_ESTIMATED_SCRATCH_BYTES,
                                'estimated_scratch_formula': '320*(g+p), conservative planning guard, not measured RSS',
                                'CPU_budget_seconds': CPU_BUDGET_SECONDS},
            'connectivity26': {'status': 'OMITTED_OPTIONAL', 'reason': '6-neighbor required diagnostic fixed; avoid extra neighborhood-dependent metric and cost'},
            'failure0_note': 'engine never imputes; caller enforces eligibility in an explicitly separate table',
            'artifact_hash_note': 'Caller additionally binds actual source/input/GT/pred file SHA and selected attempt provenance.'}
