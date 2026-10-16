"""Synthetic counterexamples and independent tiny-set oracles only."""
import builtins
import copy
import hashlib
import itertools
import json
import math
import random

import pytest

from offline_evaluation.analysis.goal010v2 import metrics as m
from offline_evaluation.server_v2.core import Grid, metrics as frozen_metrics


def world(points, material='stone'):
    return {tuple(p): material for p in points}


def box(shape, origin=(0, 0, 0), shell=False):
    return world(tuple(p[i] + origin[i] for i in range(3))
                 for p in itertools.product(*(range(n) for n in shape))
                 if not shell or any(p[i] in (0, shape[i] - 1) for i in range(3)))


def brute_boundary(points):
    # Independent all-pairs Manhattan neighbor count rather than six offsets.
    return [p for p in points if sum(sum(abs(p[i] - q[i]) for i in range(3)) == 1
                                    for q in points) < 6]


def brute_surface(gt, pred, threshold):
    gb, pb = brute_boundary(gt), brute_boundary(pred)
    def hits(source, target):
        return sum(any(sum((p[i] - q[i]) ** 2 for i in range(3)) <= threshold ** 2
                       for q in target) for p in source)
    return len(gb), len(pb), hits(pb, gb), hits(gb, pb)


def brute_components(points):
    # All-pairs disjoint sets; deliberately unlike the production sparse flood.
    points = list(points)
    parents = list(range(len(points)))
    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i
    for i, p in enumerate(points):
        for j, q in enumerate(points[:i]):
            if sum(abs(p[k] - q[k]) for k in range(3)) == 1:
                parents[find(i)] = find(j)
    sizes = {}
    for i in range(len(points)):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    return len(sizes), max(sizes.values()) / len(points) if points else None


@pytest.mark.parametrize('shape', [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 4, 2)])
def test_identical_grids_exact_scores_and_identity(shape):
    g = box(shape, (-2, 4, -1))
    before = copy.deepcopy(g)
    out = m.compute_metrics(g, g, (0, 0, 0))
    assert g == before
    for frame in ('absolute', 'aligned'):
        values = out[frame]
        assert set(values) == set(m.FRAME_METRIC_IDS)
        for metric in m.BOUNDED_FAILURE0_METRICS | {'conditional_exact_material_accuracy'}:
            assert values[metric] == 1
        assert values['column_count_mae'] == values['common_top_y_mae'] == 0
        assert values['excess_count'] == values['omission_count'] == 0
        assert values['gt_components6'] == values['pred_components6'] == 1
        assert values['relative_volume_error'] == 0
        assert not out['details'][frame]['null_reasons']
    assert out['provenance']['gt_grid_sha256'] == out['provenance']['pred_grid_sha256']
    assert out['provenance']['alignment_searched'] is False
    assert out['provenance']['implementation_sha256'] == hashlib.sha256(m.Path(m.__file__).read_bytes()).hexdigest()


def test_fixed_saved_shift_recovers_without_reoptimization():
    g = box((2, 3, 2), (-4, 7, -8))
    shift = (-8, 2, -3)
    p = {tuple(x[i] - shift[i] for i in range(3)): material for x, material in g.items()}
    out = m.compute_metrics(g, p, shift)
    assert out['absolute']['occupancy_iou'] == 0
    assert out['aligned']['occupancy_iou'] == 1
    assert out['aligned']['material_aware_iou'] == 1
    assert out['displacement'] == {
        'shift_xyz': list(shift), 'shift_l1': 13, 'shift_l2': math.sqrt(77),
        'gt_occupied_dimensions_xyz': [2, 3, 2],
        'shift_abs_relative_xyz': [4, 2 / 3, 1.5],
        'shift_l1_relative_extent_sum': 13 / 7,
        'shift_l2_relative_bbox_diagonal': math.sqrt(77) / math.sqrt(17),
        'occupancy_iou_gain': 1,
    }
    wrong = m.compute_metrics(g, p, (0, 0, 0))
    assert wrong['aligned']['occupancy_iou'] == 0
    assert wrong['aligned']['surface_f1_at_2'] < 1


def test_alignment_can_leave_input_coordinate_box_without_clipping():
    g = world([(m.MAX_COORDINATE_ABS, 0, 0)])
    p = world([(m.MAX_COORDINATE_ABS, 0, 0)])
    out = m.compute_metrics(g, p, (2 * m.MAX_COORDINATE_ABS, 0, 0))
    assert out['aligned']['pred_occupied_count'] == 1
    assert out['aligned']['occupancy_iou'] == 0
    assert out['displacement']['occupancy_iou_gain'] == -1  # no fake optimality check


def test_frozen_strict_translation_selected_shift_and_exact_overlap_metrics_match():
    from offline_evaluation.server_v2.core import evaluate_pair
    g = {(0, 0, 0): 'stone', (1, 0, 0): 'glass', (1, 1, 0): 'stone', (1, 2, 0): 'wood'}
    p = {(7, -3, 4): 'stone', (8, -3, 4): 'dirt', (8, -2, 4): 'stone'}
    frozen = evaluate_pair(Grid(g), Grid(p), alignment_mode='strict_translation')
    selected = frozen['alignment']['shift_xyz']
    out = m.compute_metrics(g, p, selected)
    assert out['provenance']['stored_alignment_shift_xyz'] == selected
    for frame in ('absolute', 'aligned'):
        for metric in ('occupancy_iou', 'occupancy_f1', 'material_aware_iou',
                       'occupancy_intersection', 'occupancy_union'):
            assert out[frame][metric] == frozen[frame][metric]
        assert out[frame]['multiscale_iou_k1_anchor0'] == frozen[frame]['occupancy_iou']
        c = frozen[frame]['occupancy_intersection']
        assert out[frame]['conditional_exact_material_accuracy'] == (
            frozen[frame]['material_match_count'] / c if c else None)


def test_exact_material_ids_and_conditional_zero_without_aliases():
    g = {(0, 0, 0): 'stonebrick', (1, 0, 0): 'brick_block'}
    p = {(0, 0, 0): 'brick_block', (1, 0, 0): 'stonebrick'}
    out = m.compute_metrics(g, p, [0, 0, 0])['aligned']
    assert out['occupancy_iou'] == 1
    assert out['conditional_exact_material_accuracy'] == 0
    assert out['material_aware_iou'] == 0
    assert out['gt_material_recall'] == out['macro_gt_material_recall'] == 0
    assert frozen_metrics(Grid(g), Grid(p))['material_set_f1'] == 1


def test_no_intersection_conditional_null_even_successful_input():
    out = m.compute_metrics(world([(0, 0, 0)]), world([(8, 0, 0)]), [0, 0, 0])
    for frame in ('absolute', 'aligned'):
        assert out[frame]['occupancy_iou'] == out[frame]['gt_material_recall'] == 0
        assert out[frame]['conditional_exact_material_accuracy'] is None
        assert out['details'][frame]['null_reasons']['conditional_exact_material_accuracy'] == 'NO_OCCUPANCY_INTERSECTION'
        assert out[frame]['common_top_y_mae'] is None


def test_macro_material_recall_and_confusion_include_missing_and_extra():
    g = {(0, 0, 0): 'stone', (1, 0, 0): 'stone', (2, 0, 0): 'stone', (3, 0, 0): 'glass'}
    p = {(0, 0, 0): 'stone', (1, 0, 0): 'dirt', (3, 0, 0): 'glass', (4, 0, 0): 'wood'}
    out = m.compute_metrics(g, p, (0, 0, 0))
    v, details = out['aligned'], out['details']['aligned']
    assert v['gt_material_recall'] == 0.5
    assert v['macro_gt_material_recall'] == pytest.approx(2 / 3)
    assert v['conditional_exact_material_accuracy'] == 2 / 3
    counts = {(r['gt_material_id'], r['pred_material_id']): r['count'] for r in details['confusion']}
    assert counts == {('stone', 'stone'): 1, ('stone', 'dirt'): 1, ('stone', None): 1,
                      ('glass', 'glass'): 1, (None, 'wood'): 1}
    assert sum(counts.values()) == v['occupancy_union']
    assert [r['material_id'] for r in details['material_recalls']] == ['glass', 'stone']


def test_column_union_and_common_top_have_different_denominators():
    g = world([(0, 0, 0), (0, 3, 0), (1, 4, 0)])
    p = world([(0, 5, 0), (2, -2, 0), (2, 0, 0)])
    out = m.compute_metrics(g, p, (0, 0, 0))
    v = out['aligned']
    assert v['union_footprint_columns'] == 3
    assert v['common_footprint_columns'] == 1
    assert v['column_count_mae'] == 4 / 3
    assert v['common_top_y_mae'] == 2
    assert v['common_footprint_gt_coverage'] == v['common_footprint_pred_coverage'] == 0.5
    assert v['footprint_iou'] == 1 / 3
    assert out['details']['aligned']['column_counts'] == {
        'gt_only': 1, 'pred_only': 1, 'common': 1, 'union': 3, 'missing_side_occupancy_count': 0}


@pytest.mark.parametrize('offset,expected1,expected2', [((1, 0, 0), 1, 1), ((2, 0, 0), 0, 1),
                                                      ((1, 1, 0), 0, 1), ((2, 1, 0), 0, 0)])
def test_surface_euclidean_inclusive_threshold(offset, expected1, expected2):
    out = m.compute_metrics(world([(0, 0, 0)]), world([offset]), (0, 0, 0))['absolute']
    assert out['surface_f1_at_1'] == expected1
    assert out['surface_f1_at_2'] == expected2


def test_filled_box_vs_shell_surface_saturates_but_occupancy_differs():
    filled, shell = box((3, 3, 3)), box((3, 3, 3), shell=True)
    out = m.compute_metrics(filled, shell, (0, 0, 0))['absolute']
    assert out['gt_surface_voxels'] == out['pred_surface_voxels'] == 26
    assert out['occupancy_precision'] == 1
    assert out['occupancy_recall'] == out['occupancy_iou'] == 26 / 27
    assert out['surface_f1_at_1'] == out['surface_f1_at_2'] == out['footprint_iou'] == 1
    assert out['column_count_mae'] == 1 / 9


@pytest.mark.parametrize('shape', [(1, 5, 5), (2, 5, 5), (5, 1, 5), (5, 2, 5)])
def test_one_or_two_cell_width_all_occupied_are_boundary(shape):
    g = box(shape)
    out = m.compute_metrics(g, g, (0, 0, 0))['absolute']
    assert out['gt_surface_voxels'] == math.prod(shape)


def test_cavity_boundary_counts_internal_walls():
    g = box((5, 5, 5))
    g.pop((2, 2, 2))
    out = m.compute_metrics(g, g, (0, 0, 0))['absolute']
    assert out['gt_surface_voxels'] == 98 + 6
    assert out['gt_components6'] == 1


def test_negative_floor_and_fixed_half_anchor():
    negative = m.compute_metrics(world([(-1, 0, 0)]), world([(0, 0, 0)]), (0, 0, 0))
    assert negative['aligned']['multiscale_iou_k2_anchor0'] == 0
    result = m.compute_metrics(world([(0, 0, 0)]), world([(1, 0, 0)]), (0, 0, 0))
    assert result['aligned']['multiscale_iou_k2_anchor0'] == 1
    assert result['aligned']['multiscale_iou_k2_anchorhalf'] == 0
    assert result['details']['aligned']['multiscale_counts']['multiscale_iou_k4_anchorhalf']['anchor_xyz'] == [2, 2, 2]


@pytest.mark.parametrize('gt_nonempty,pred_nonempty', [(True, False), (False, True), (False, False)])
def test_empty_conventions_preserve_denominator_nulls(gt_nonempty, pred_nonempty):
    g, p = world([(0, 0, 0)]) if gt_nonempty else {}, world([(0, 0, 0)]) if pred_nonempty else {}
    out = m.compute_metrics(g, p, (0, 0, 0))
    both_empty = not gt_nonempty and not pred_nonempty
    for frame in ('absolute', 'aligned'):
        v = out[frame]
        assert v['occupancy_iou'] == v['occupancy_f1'] == v['surface_f1_at_1'] == (1 if both_empty else 0)
        assert v['occupancy_precision'] == (0 if pred_nonempty else None)
        assert v['occupancy_recall'] == (0 if gt_nonempty else None)
        assert v['surface_precision_at_1'] == (0 if pred_nonempty else None)
        assert v['surface_recall_at_1'] == (0 if gt_nonempty else None)
        assert v['conditional_exact_material_accuracy'] is None
        assert v['common_top_y_mae'] is None
        assert v['column_count_mae'] == (None if both_empty else 1)
        assert v['pred_components6'] == int(pred_nonempty)
        assert v['pred_largest_component_fraction6'] == (1 if pred_nonempty else None)
        assert out['details'][frame]['flags']['empty_prediction'] == (not pred_nonempty)
        assert {k for k, value in v.items() if value is None} == set(out['details'][frame]['null_reasons'])
    assert out['displacement']['shift_abs_relative_xyz'] == ([0, 0, 0] if gt_nonempty else None)


def test_six_neighbor_components_do_not_connect_corner_touching_cells():
    p = world([(0, 0, 0), (1, 0, 0), (2, 1, 1), (4, 4, 4)])
    out = m.compute_metrics(p, p, (9, -2, 4))
    for frame in ('absolute', 'aligned'):
        assert out[frame]['pred_components6'] == 3
        assert out[frame]['pred_largest_component_fraction6'] == 0.5


@pytest.mark.parametrize('seed', range(6))
def test_random_tiny_sets_match_brute_surface_floor_components_columns_and_frozen(seed):
    rng = random.Random(seed)
    universe = list(itertools.product(range(-2, 3), range(-1, 3), range(-2, 2)))
    g = {p: rng.choice(['stone', 'glass']) for p in rng.sample(universe, 15)}
    p = {p: rng.choice(['stone', 'glass']) for p in rng.sample(universe, 14)}
    shift = (seed % 3 - 1, 1, -1)
    result = m.compute_metrics(g, p, shift)
    for frame, delta in [('absolute', (0, 0, 0)), ('aligned', shift)]:
        moved = {tuple(c[i] + delta[i] for i in range(3)): material for c, material in p.items()}
        v, details = result[frame], result['details'][frame]
        reference = frozen_metrics(Grid(g), Grid(moved))
        for name in ('occupancy_iou', 'occupancy_f1', 'occupancy_intersection', 'occupancy_union', 'material_aware_iou'):
            assert v[name] == reference[name]
        c, ng, np = reference['occupancy_intersection'], len(g), len(p)
        assert v['occupancy_precision'] * np == pytest.approx(c)
        assert v['occupancy_recall'] * ng == pytest.approx(c)
        assert v['excess_count'] + c == np
        assert v['omission_count'] + c == ng
        assert v['occupancy_f1'] == pytest.approx(2 * v['occupancy_iou'] / (1 + v['occupancy_iou']))
        assert v['gt_material_recall'] * ng == reference['material_match_count']
        for threshold in (1, 2):
            gb, pb, ph, gh = brute_surface(g, moved, threshold)
            counts = details['boundary_counts']['thresholds'][str(threshold)]
            assert (counts['gt_boundary'], counts['pred_boundary'], counts['pred_within_threshold'], counts['gt_within_threshold']) == (gb, pb, ph, gh)
        for k, a in m.SCALES:
            anchor = 0 if a == '0' else k // 2
            gs = {tuple(math.floor((float(x) - anchor) / k) for x in c) for c in g}
            ps = {tuple(math.floor((float(x) - anchor) / k) for x in c) for c in moved}
            expected = len(gs & ps) / len(gs | ps)
            assert v[f'multiscale_iou_k{k}_anchor{a}'] == expected
        assert (v['gt_components6'], v['gt_largest_component_fraction6']) == brute_components(g)
        assert (v['pred_components6'], v['pred_largest_component_fraction6']) == brute_components(moved)
        columns = {(x, z) for x, y, z in set(g) | set(moved)}
        total = sum(abs(sum(c[0] == x and c[2] == z for c in g)
                        - sum(c[0] == x and c[2] == z for c in moved)) for x, z in columns)
        assert v['column_count_mae'] == total / len(columns)


@pytest.mark.parametrize('bad', [
    {(True, 0, 0): 'stone'}, {(0.0, 0, 0): 'stone'}, {(0, 0): 'stone'},
    {(0, 0, 0): 'air'}, {(0, 0, 0): 'cave_air'}, {(0, 0, 0): 'void_air'},
    {(0, 0, 0): 'minecraft:air'}, {(0, 0, 0): 'minecraft:stone'},
    {(0, 0, 0): ''}, {(0, 0, 0): True}, {(0, 0, 0): 'Stone'},
    {(100_001, 0, 0): 'stone'}, [(0, 0, 0, 'stone')],
])
def test_invalid_world_input_fails_instead_of_repairing(bad):
    with pytest.raises(ValueError):
        m.compute_metrics(world([(0, 0, 0)]), bad, (0, 0, 0))


@pytest.mark.parametrize('shift', [(True, 0, 0), (0.0, 0, 0), [0, 0], '0,0,0', (200_001, 0, 0)])
def test_invalid_shift_fails(shift):
    with pytest.raises(ValueError):
        m.compute_metrics({}, {}, shift)


def test_count_and_planned_scratch_caps_fail_closed(monkeypatch):
    g = box((3, 1, 1))
    monkeypatch.setattr(m, 'MAX_OCCUPIED', 2)
    with pytest.raises(m.MetricResourceLimit, match='OCCUPIED_COUNT_LIMIT'):
        m.compute_metrics(g, {}, (0, 0, 0))
    monkeypatch.setattr(m, 'MAX_OCCUPIED', 20)
    monkeypatch.setattr(m, 'MAX_ESTIMATED_SCRATCH_BYTES', 100)
    with pytest.raises(m.MetricResourceLimit, match='ESTIMATED_SCRATCH_LIMIT'):
        m.compute_metrics(g, {}, (0, 0, 0))


def test_cpu_budget_core_failure_raises_no_partial_or_zero(monkeypatch):
    monkeypatch.setattr(m, 'CPU_BUDGET_SECONDS', -1)
    with pytest.raises(m.MetricResourceLimit, match='CPU_BUDGET_EXCEEDED'):
        m.compute_metrics(world([(0, 0, 0)]), {}, (0, 0, 0))


@pytest.mark.parametrize('failure', ['surface_limit', 'surface_memory', 'surface_import', 'surface_cpu', 'connectivity_memory'])
def test_optional_resource_and_dependency_failures_are_null_with_reason(monkeypatch, failure):
    g = box((2, 2, 2))
    if failure == 'surface_limit':
        monkeypatch.setattr(m, 'MAX_SURFACE_POINTS', 1)
    else:
        def fail(*args):
            if failure == 'surface_import':
                raise ImportError('fixture dependency unavailable')
            if failure == 'surface_cpu':
                raise m.MetricResourceLimit('CPU_BUDGET_EXCEEDED')
            raise MemoryError
        monkeypatch.setattr(m, '_components' if failure == 'connectivity_memory' else '_surface', fail)
    out = m.compute_metrics(g, g, (0, 0, 0))
    ids = m.CONNECTIVITY_METRIC_IDS if failure == 'connectivity_memory' else m.SURFACE_METRIC_IDS
    for frame in ('absolute', 'aligned'):
        assert out[frame]['occupancy_iou'] == 1
        for metric in ids:
            assert out[frame][metric] is None
            assert metric in out['details'][frame]['null_reasons']
        assert out['details'][frame]['flags']['failure0_applied'] is False


def test_hash_binds_material_coordinates_shift_and_is_insertion_order_independent():
    g = world([(0, 0, 0), (1, 0, 0)])
    reordered = dict(reversed(list(g.items())))
    a = m.compute_metrics(g, reordered, (0, 0, 0))
    assert a['provenance']['gt_grid_sha256'] == a['provenance']['pred_grid_sha256']
    changed = dict(g);changed[(0, 0, 0)] = 'glass'
    b = m.compute_metrics(g, changed, (0, 0, 0))
    c = m.compute_metrics(g, g, (1, 0, 0))
    assert len({x['provenance']['inputs_and_shift_sha256'] for x in [a, b, c]}) == 3
    json.dumps(a, allow_nan=False)


def test_registry_is_complete_and_failure0_strictly_bounded_and_material_baselines_excluded():
    registry = m.metric_registry()
    entries = registry['metrics']
    assert len({x['metric_id'] for x in entries}) == len(entries)
    frame = {x['metric_id']: x for x in entries if x['section'] == 'frame'}
    assert set(frame) == set(m.FRAME_METRIC_IDS)
    assert set(m.CORE_METRICS) == {k for k, x in frame.items() if x['priority'] == 'core'}
    assert m.BOUNDED_FAILURE0_METRICS == {k for k, x in frame.items() if x['failure0_allowed']}
    for entry in entries:
        assert entry['implementation_sha256'] == m.implementation_sha256()
        assert entry['formula'] and entry['null_policy'] and entry['denominator'] and entry['sample_hash_fields']
        if entry['failure0_allowed']:
            assert entry['direction'] == 'higher' and entry['value_range'] == [0, 1]
        if entry['metric_id'] in m.MATERIAL_METRIC_IDS:
            assert entry['baseline_comparable'] is False
    assert frame['conditional_exact_material_accuracy']['failure0_allowed'] is False
    for name in ['column_count_mae', 'common_top_y_mae', 'volume_ratio', 'relative_volume_error',
                 'common_footprint_gt_coverage', 'pred_largest_component_fraction6']:
        assert frame[name]['failure0_allowed'] is False
    assert all(frame[x]['priority'] == 'diagnostic' for x in frame if '_k8_' in x)
    assert registry['connectivity26']['status'] == 'OMITTED_OPTIONAL'
    json.dumps(registry, allow_nan=False)
