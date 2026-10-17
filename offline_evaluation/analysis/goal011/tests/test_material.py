"""Synthetic-only checks; assignment oracle is exhaustive, not Hungarian."""
import copy
import itertools
import json
import random

import pytest

from offline_evaluation.analysis.goal011 import material as m
from offline_evaluation.resmoke_v2 import cuboids
from offline_evaluation.server_v2.core import Grid, Limits, execute


def brute_mapping(gt_labels, pred_labels, matrix, injective):
    candidates = []
    # Integer target positions before the unmatched sentinel: true lex witness.
    for targets in itertools.product(range(len(gt_labels) + 1), repeat=len(pred_labels)):
        active = [x for x in targets if x < len(gt_labels)]
        if injective and len(active) != len(set(active)):
            continue
        score = sum(matrix[gt][pred] for pred, gt in enumerate(targets) if gt < len(gt_labels))
        candidates.append((score, targets))
    best = max(x[0] for x in candidates)
    witnesses = [x[1] for x in candidates if x[0] == best]
    chosen = min(witnesses)
    possibilities = [{x[i] for x in witnesses} for i in range(len(pred_labels))]
    return best, chosen, possibilities


def bounds(matrix, gl=None, pl=None):
    gl = gl or ['g' + str(i) for i in range(len(matrix))]
    pl = pl or ['p' + str(i) for i in range(len(matrix[0]))]
    c = sum(map(sum, matrix))
    original = sum(matrix[i][j] for i, a in enumerate(gl) for j, b in enumerate(pl) if a == b)
    return m.mapping_upper_bounds(gl, pl, matrix, g=c, p=c, original_matches=original)


def worlds_from_matrix(matrix, gl=None, pl=None):
    gl = gl or ['g' + str(i) for i in range(len(matrix))]
    pl = pl or ['p' + str(i) for i in range(len(matrix[0]))]
    gt, pred, position = {}, {}, 0
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            for _ in range(count):
                gt[position, 0, 0] = gl[i]
                pred[position, 0, 0] = pl[j]
                position += 1
    return gt, pred


def fill(x1, x2, label='stone', y1=0, y2=0, z1=0, z2=0):
    return {'op': 'fill', 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
            'z1': z1, 'z2': z2, 'block': label}


def test_perfect_labels_are_unique_and_no_mismatch_fraction_is_undefined():
    result = bounds([[10, 0], [0, 1]], ['a', 'b'], ['a', 'b'])
    for kind in ('one_to_one', 'many_to_one'):
        r = result[kind]
        assert r['maximum_exact_matches'] == 11
        assert r['conditional_accuracy_upper_bound'] == r['gt_material_recall_upper_bound'] == r['material_iou_upper_bound'] == 1
        assert r['ambiguous_mapping'] is False
        assert r['recoverable_fraction_of_original_common_mismatches'] is None
        assert r['null_reasons']['recoverable_fraction_of_original_common_mismatches'] == 'NO_ORIGINAL_COMMON_MATERIAL_MISMATCHES'


def test_transposition_can_be_explained_without_changing_prediction():
    gt, pred = worlds_from_matrix([[0, 7], [5, 0]], ['a', 'b'], ['a', 'b'])
    saved = copy.deepcopy((gt, pred))
    out = m.analyze_case(gt, pred, (0, 0, 0))
    assert (gt, pred) == saved
    assert out['M'] == 0 and out['C'] == 12
    assert out['strict']['occupancy_iou'] == 1
    assert out['strict']['material_aware_iou'] == 0
    assert out['one_to_one']['maximum_exact_matches'] == 12
    assert [(r['pred_label'], r['gt_label']) for r in out['one_to_one']['mapping']] == [('a', 'b'), ('b', 'a')]
    assert out['one_to_one']['recoverable_fraction_of_original_common_mismatches'] == 1
    assert out['provenance']['prediction_artifact_modified'] is False
    assert out['reference_action_audit']['status'] == 'NOT_VERIFIED'


def test_many_to_one_is_a_distinct_larger_bound_and_one_to_one_keeps_unmatched():
    result = bounds([[5, 7]], ['stone'], ['glass', 'wood'])
    one, many = result['one_to_one'], result['many_to_one']
    assert one['maximum_exact_matches'] == 7
    assert [r['gt_label'] for r in one['mapping']] == [None, 'stone']
    assert many['maximum_exact_matches'] == 12
    assert one['material_iou_upper_bound'] == 7 / 17
    assert many['material_iou_upper_bound'] == 1


def test_one_pred_label_inconsistently_used_cannot_match_two_gt_ids():
    out = bounds([[100], [1]], ['a', 'b'], ['x'])
    assert out['one_to_one']['maximum_exact_matches'] == out['many_to_one']['maximum_exact_matches'] == 100
    assert out['many_to_one']['residual_common_material_mismatches'] == 1
    assert out['many_to_one']['mapping'][0]['gt_label'] == 'a'


def test_rare_class_can_determine_the_global_one_to_one_assignment():
    # Both predicted IDs mostly denote a; the single rare b cell resolves the
    # assignment. Greedy independent column-majority selection is not injective.
    out = bounds([[1000, 1000], [1, 0]], ['a', 'b'], ['x', 'y'])
    assert out['one_to_one']['maximum_exact_matches'] == 1001
    assert [row['gt_label'] for row in out['one_to_one']['mapping']] == ['b', 'a']
    assert out['one_to_one']['ambiguous_mapping'] is False
    assert out['many_to_one']['maximum_exact_matches'] == 2000


def test_equal_weight_ties_global_possibilities_and_deterministic_lex_witness():
    out = bounds([[2, 2], [2, 2]], ['a', 'b'], ['x', 'y'])
    assert [r['gt_label'] for r in out['one_to_one']['mapping']] == ['a', 'b']
    assert out['one_to_one']['maximum_exact_matches'] == 4
    for row in out['one_to_one']['mapping']:
        assert row['possible_gt_labels_in_any_optimum'] == ['a', 'b']
    assert out['one_to_one']['ambiguous_observed_pred_labels'] == ['x', 'y']
    assert [r['gt_label'] for r in out['many_to_one']['mapping']] == ['a', 'a']


def test_zero_support_label_ambiguity_is_not_observed_binding_ambiguity():
    gt = {(0, 0, 0): 'a', (2, 0, 0): 'b'}
    pred = {(0, 0, 0): 'z', (4, 0, 0): 'x'}
    out = m.analyze_case(gt, pred, (0, 0, 0))
    assert out['missing_occupied_count'] == out['extra_occupied_count'] == 1
    one = out['one_to_one']
    assert one['maximum_exact_matches'] == 1
    assert one['ambiguous_mapping'] is True
    assert one['ambiguous_observed_pred_labels'] == []
    assert one['unobserved_pred_labels'] == ['x']
    assert one['mapping'][0]['possible_gt_labels_in_any_optimum'] == ['b', None]
    assert out['confusion']['missing_by_gt_label'] == [
        {'gt_material_id': 'a', 'pred_material_id': None, 'category': 'missing_geometry', 'count': 0},
        {'gt_material_id': 'b', 'pred_material_id': None, 'category': 'missing_geometry', 'count': 1}]
    assert out['confusion']['extra_by_pred_label'] == [
        {'gt_material_id': None, 'pred_material_id': 'x', 'category': 'extra_geometry', 'count': 1},
        {'gt_material_id': None, 'pred_material_id': 'z', 'category': 'extra_geometry', 'count': 0}]
    assert one['gt_material_recall_upper_bound'] == 0.5
    assert one['material_iou_upper_bound'] == 1 / 3


@pytest.mark.parametrize('gt,pred', [({}, {}), ({(0, 0, 0): 'a'}, {}),
                                     ({}, {(0, 0, 0): 'a'}),
                                     ({(0, 0, 0): 'a'}, {(4, 0, 0): 'b'})])
def test_empty_intersection_count_zero_and_mapping_unidentifiable(gt, pred):
    out = m.analyze_case(gt, pred, (0, 0, 0))
    assert out['C'] == out['M'] == 0
    assert out['strict']['conditional_exact_material_accuracy'] is None
    for kind in ('one_to_one', 'many_to_one'):
        assert out[kind]['status'] == 'UNIDENTIFIABLE_NO_COMMON_OCCUPANCY'
        assert out[kind]['mapping'] is None and out[kind]['ambiguous_mapping'] is None
        assert out[kind]['maximum_exact_matches'] == 0
        assert out[kind]['conditional_accuracy_upper_bound'] is None
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize('seed', range(16))
def test_random_small_assignment_matches_exhaustive_all_optima(seed):
    rng = random.Random(seed)
    ng, np = rng.randrange(1, 4), rng.randrange(1, 5)
    gl, pl = ['g' + str(i) for i in range(ng)], ['p' + str(i) for i in range(np)]
    matrix = [[rng.randrange(4) for _ in range(np)] for _ in range(ng)]
    matrix[0][0] = 1
    result = bounds(matrix, gl, pl)
    for name, injective in [('one_to_one', True), ('many_to_one', False)]:
        maximum, selected, possibilities = brute_mapping(gl, pl, matrix, injective)
        actual = result[name]
        assert actual['maximum_exact_matches'] == maximum
        expected_labels = [gl[i] if i < ng else None for i in selected]
        assert [r['gt_label'] for r in actual['mapping']] == expected_labels
        for row, opts in zip(actual['mapping'], possibilities):
            assert set(row['possible_gt_labels_in_any_optimum']) == {gl[i] if i < ng else None for i in opts}


def test_fixed_shift_and_source_hashes_bind_geometry_without_reoptimizing():
    g = {(0, 0, 0): 'stone', (1, 0, 0): 'glass'}
    p = {(5, 0, 0): 'glass', (6, 0, 0): 'stone'}
    binding = {'input_sha256': 'a' * 64}
    out = m.analyze_case(g, p, (-5, 0, 0), source_binding=binding)
    assert out['C'] == 2 and out['M'] == 0
    assert out['one_to_one']['maximum_exact_matches'] == 2
    wrong = m.analyze_case(g, p, (0, 0, 0))
    assert wrong['C'] == 0
    assert out['provenance']['shift_sha256'] != wrong['provenance']['shift_sha256']
    binding['input_sha256'] = 'b' * 64
    assert out['provenance']['source_binding']['input_sha256'] == 'a' * 64
    assert out['provenance']['geometry_optimized'] is False


def test_unique_inclusive_bounds_transcription_correct_and_wrong_labels():
    refs = [fill(-2, 0, 'stone'), fill(2, 2, 'glass')]
    predicted = [fill(0, -2, 'wood'), {'op': 'set', 'x': 2, 'y': 0, 'z': 0, 'block': 'glass'}]
    before = copy.deepcopy((refs, predicted))
    out = m.audit_reference_actions(refs, predicted)
    assert out['unique_literal_pairs'] == 2
    assert out['unique_correct_label_pairs'] == out['unique_wrong_label_pairs'] == 1
    assert out['ambiguous_reference_actions'] == out['unclassified_predicted_actions'] == 0
    assert out['references'][0]['bounds_xyz_inclusive'] == [-2, 0, 0, 0, 0, 0]
    assert (refs, predicted) == before


def test_supplied_id_spellings_survive_frozen_namespace_normalization():
    reference = [fill(0, 1, 'stonebrick')]
    predicted = [fill(1, 0, 'minecraft:stonebrick')]
    out = m.audit_reference_actions(reference, predicted)
    pair = out['references'][0]
    assert pair['reference_material'] == pair['predicted_material'] == 'stonebrick'
    assert pair['reference_material_as_supplied'] == 'stonebrick'
    assert pair['predicted_material_as_supplied'] == 'minecraft:stonebrick'
    assert pair['label_agreement'] is True
    assert out['outputs'][0]['material_as_supplied'] == 'minecraft:stonebrick'
    assert out['predicted_actions_as_supplied_sha256'] != out['predicted_actions_sha256']
    # Historical distinct spellings remain distinct canonical IDs.
    synonym = m.audit_reference_actions(reference, [fill(0, 1, 'stone_bricks')])
    assert synonym['unique_wrong_label_pairs'] == 1


@pytest.mark.parametrize('kind', ['duplicate', 'overwrite', 'split', 'merged', 'shifted', 'carve', 'air'])
def test_ambiguous_actions_never_guess_intent(kind):
    refs = [fill(0, 1)]
    if kind == 'duplicate':
        predicted = [fill(0, 1), fill(0, 1, 'glass')]
    elif kind == 'overwrite':
        predicted = [fill(0, 1), fill(1, 2, 'glass')]
    elif kind == 'split':
        predicted = [fill(0, 0), fill(1, 1)]
        assert execute(refs).voxels == execute(predicted).voxels
    elif kind == 'merged':
        refs = [fill(0, 0), fill(1, 1)]
        predicted = [fill(0, 1)]
        assert execute(refs).voxels == execute(predicted).voxels
    elif kind == 'shifted':
        predicted = [fill(1, 2)]
    elif kind == 'carve':
        predicted = [{k: v for k, v in fill(0, 1).items() if k != 'block'}]
        predicted[0]['op'] = 'carve'
    else:
        predicted = [fill(0, 1, 'air')]
    out = m.audit_reference_actions(refs, predicted)
    assert out['unique_literal_pairs'] == 0
    assert all(r['status'] == 'AMBIGUOUS_NOT_CLASSIFIED' for r in out['references'])
    assert all(r['label_agreement'] is None for r in out['references'])
    assert out['model_internal_mechanism_inferred'] is False


def test_duplicate_reference_bounds_are_ambiguous_not_overwritten_in_index():
    refs = [fill(0, 1), fill(0, 1)]
    result = m.audit_reference_actions(refs, [fill(0, 1)])
    assert result['reference_actions'] == result['ambiguous_reference_actions'] == 2
    assert result['unique_literal_pairs'] == 0
    assert all('DUPLICATE_REFERENCE_BOUNDS' in r['ambiguity_reasons'] for r in result['references'])


def test_frozen_cuboid_control_world_origin_and_action_literal_matching():
    local = Grid({(0, 0, 0): 'stone', (1, 0, 0): 'stone', (2, 2, 0): 'glass'},
                 coordinate_frame='local', origin=(-5, 7, 2))
    spec = cuboids.encode_exact(local)
    reference = cuboids.reference_actions(spec)['operations']
    g = local.as_world().voxels
    assert execute(reference).voxels == g
    prediction = copy.deepcopy(reference)
    prediction[0]['block'] = 'dirt'
    pred = execute(prediction).voxels
    result = m.analyze_case(g, pred, (0, 0, 0), reference_actions=reference, predicted_actions=prediction)
    assert result['C'] == len(g)
    assert result['reference_action_audit']['unique_wrong_label_pairs'] == 1
    assert result['reference_action_audit']['unique_literal_pairs'] == len(reference)
    assert result['reference_action_audit']['alignment_applied_to_actions'] is False
    assert result['one_to_one']['maximum_exact_matches'] == len(g)


@pytest.mark.parametrize('invalid', [
    {(True, 0, 0): 'stone'}, {(0.0, 0, 0): 'stone'}, {(100001, 0, 0): 'stone'},
    {(0, 0, 0): 'air'}, {(0, 0, 0): 'cave_air'}, {(0, 0, 0): 'minecraft:stone'},
    {(0, 0, 0): 'Stone'}, {(0, 0, 0): ''}, [(0, 0, 0, 'stone')],
])
def test_invalid_grids_rejected_before_diagnostic(invalid):
    with pytest.raises(ValueError):
        m.analyze_case({}, invalid, (0, 0, 0))


@pytest.mark.parametrize('shift', [(True, 0, 0), (0.0, 0, 0), (200001, 0, 0), [0, 0]])
def test_invalid_shift_rejected(shift):
    with pytest.raises(ValueError):
        m.analyze_case({}, {}, shift)


def test_invalid_count_contract_not_tolerated():
    with pytest.raises(ValueError, match='diagonal'):
        m.mapping_upper_bounds(['a'], ['a'], [[1]], g=1, p=1, original_matches=0)
    with pytest.raises(ValueError, match='integer'):
        m.mapping_upper_bounds(['a'], ['a'], [[True]], g=1, p=1, original_matches=1)
    with pytest.raises(ValueError, match='disagree'):
        m.mapping_upper_bounds(['a'], ['a'], [[2]], g=1, p=1, original_matches=2)


def test_label_resource_limit_preserves_strict_confusion_without_large_matrix(monkeypatch):
    monkeypatch.setattr(m, 'MAX_LABELS', 2)
    g = {(i, 0, 0): 'm' + str(i) for i in range(3)}
    out = m.analyze_case(g, g, (0, 0, 0))
    assert out['M'] == 3 and out['strict']['material_aware_iou'] == 1
    assert out['confusion']['common_counts'] is None
    assert len(out['confusion']['common_nonzero']) == 3
    assert out['one_to_one']['maximum_exact_matches'] is None
    assert out['one_to_one']['reason'] == 'LABEL_LIMIT'


def test_assignment_call_cap_is_explicit_no_zero_substitution(monkeypatch):
    monkeypatch.setattr(m, 'MAX_ASSIGNMENT_CALLS', 0)
    g = {(0, 0, 0): 'stone'}
    out = m.analyze_case(g, g, (0, 0, 0))
    assert out['M'] == 1
    assert out['one_to_one']['reason'] == 'ASSIGNMENT_CALL_LIMIT'
    assert out['one_to_one']['maximum_exact_matches'] is None


def test_actions_resource_limit_and_invalid_reference_are_explicit(monkeypatch):
    class Exhausted:
        def check(self):
            raise m.MaterialResourceLimit('CPU_BUDGET_EXCEEDED')
    with pytest.raises(m.MaterialResourceLimit):
        m.audit_reference_actions([fill(0, 1)], [fill(0, 1)], budget=Exhausted())
    with pytest.raises(ValueError):
        m.audit_reference_actions([fill(0, 1, 'air')], [fill(0, 1)])


def test_network_is_not_used_and_source_dicts_remain_unchanged(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('research network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    g, p = worlds_from_matrix([[0, 2], [3, 1]])
    before = copy.deepcopy((g, p))
    r = m.analyze_case(g, p, (0, 0, 0))
    assert (g, p) == before
    assert r['provenance']['research_api_requests_sent'] == r['provenance']['api_key_queries'] == 0
    json.dumps(r, allow_nan=False)
