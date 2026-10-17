"""Synthetic-only coverage of the fixed twelve estimands and their denominators."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from offline_evaluation.analysis.goal014 import statistics as s


def fixture(n=3):
    scenes = [f'scene_{i:03}' for i in range(n)]
    scores, outcomes, qa = [], [], []
    for i, scene in enumerate(scenes):
        for j, provider in enumerate(s.PROVIDERS):
            key = {'scene_id': scene, 'description_provider': provider}
            scores.append({**key, 'score_id': s.SCORE_ID, 'score_status': 'COMPUTATION_VALID',
                           'score': (i * 3 + j + 1) / (n * 3), 'token_coverage_status': 'VERIFIED',
                           'summary_token_count_untruncated': 20 + i * 3 + j,
                           'token_diagnostics': {'truncated': j == 0}})
            outcomes.append({**key, 'outcomes': {
                builder: {'status': 'success', 'aligned_occupancy_iou': ((i + j * 3 + k) % 9) / 10,
                          'aligned_surface_f1_at_1': ((i * 2 + j + k) % 8) / 10}
                for k, builder in enumerate(s.BUILDERS)}})
            qa.append({**key, 'D_i': i / 10 + j / 20, 'W_i': (i + j) / (n + 3), 'null_reason': None})
    return scores, outcomes, qa, scenes


def fixed_draws(row):
    return np.tile(row, (s.REPLICATES, 1)).astype(np.int64)


def test_twelve_estimands_shared_frame_qc_counts_and_no_input_writes():
    inputs = fixture()
    scores, outcomes, qa, scenes = inputs
    # One missing success scalar affects Q-B common set, not Q-C's IoU set.
    outcomes[1]['outcomes']['openai']['aligned_surface_f1_at_1'] = None
    # Missing source is never imputed as a score or failure0.
    scores[-1].update(score=None, score_status='MISSING_SUMMARY', token_coverage_status='MISSING_SUMMARY')
    qa[-1].update(D_i=None, W_i=None, null_reason='MISSING_SUMMARY')
    original = deepcopy(inputs)
    result = s.analyze(*inputs, fixture=True)
    assert inputs == original
    assert [len(result[key]) for key in ('q_a', 'q_b', 'q_c')] == [2, 6, 4]
    matrix = s.bootstrap_indices(scenes)
    assert result['inference']['index_matrix_sha256_little_endian_int64'] == hashlib.sha256(matrix.astype('<i8').tobytes()).hexdigest()
    assert result['planned_buildings'] == 3
    assert result['truncation_groups']['false']['description_rows'] == 5
    assert result['truncation_groups']['true']['description_rows'] == 3
    assert result['truncation_groups']['true']['provider_counts'] == {'openai': 3, 'anthropic': 0, 'gemini': 0}
    qb = [row for row in result['q_b'] if row['builder'] == 'openai']
    assert {row['paired_description_rows'] for row in qb} == {7}
    assert {row['original_valid_iou_rows'] for row in qb} == {8}
    assert all(not row['common_set_equals_original_iou_set'] for row in qb)
    assert len({row['row_ids_sha256'] for row in qb}) == 1
    qc = {(row['builder'], row['original_truncated']): row for row in result['q_c']}
    assert qc['openai', False]['paired_description_rows'] == 5
    assert qc['openai', True]['paired_description_rows'] == 3
    assert all(len(row['bootstrap_draws']) == 2000 for family in ('q_a', 'q_b', 'q_c') for row in result[family])


def test_qa_equal_building_weight_and_bootstrap_multiplicity():
    scenes = ['a', 'b', 'c']
    rows = [{'scene_id': 'a', 'description_provider': provider, 'D_i': 0.0} for provider in s.PROVIDERS]
    rows += [{'scene_id': 'b', 'description_provider': 'openai', 'D_i': 3.0}]
    result = s._qa_result(rows, scenes, fixed_draws([0, 1, 1]), 'D_i')
    assert result['estimate'] == 1.5  # Pair weighting would give 0.75.
    assert result['bootstrap_draws'][0] == 2.0  # Duplicate b retains multiplicity.
    assert result['planned_buildings'] == 3
    assert result['contributing_buildings'] == 2
    assert result['paired_description_rows'] == 4
    assert result['per_building'][-1]['estimate'] is None


def test_missing_buildings_stay_in_frame_and_undefined_draws_not_replaced():
    rows = [{'scene_id': 'a', 'description_provider': 'openai', 'D_i': 1.0}]
    result = s._qa_result(rows, ['a', 'b', 'c'], fixed_draws([1, 2, 2]), 'D_i')
    assert result['estimate'] == 1
    assert result['contributing_buildings'] == 1 and result['planned_buildings'] == 3
    assert result['bootstrap_finite_draws'] == 0 and result['bootstrap_undefined_draws'] == 2000
    assert result['ci95'] is None
    assert result['ci_status'] == 'INSUFFICIENT_BOOTSTRAP_VALIDITY'


def test_paired_delta_uses_common_observations_and_same_resampled_rows():
    scenes = ['a', 'b', 'c']
    rows = [
        {'scene_id': 'a', 'score': 1., 'iou': 0.1, 'surface': 0.8},
        {'scene_id': 'a', 'score': 2., 'iou': 0.7, 'surface': 0.3},
        {'scene_id': 'b', 'score': 4., 'iou': 0.4, 'surface': 0.9},
        {'scene_id': 'c', 'score': 3., 'iou': 0.6, 'surface': 0.5},
    ]
    points, samples = s._paired_correlations(rows, scenes, fixed_draws([0, 0, 1]))
    assert points[2] == pytest.approx(points[1] - points[0])
    duplicated = rows[:2] + rows[:2] + rows[2:3]
    expected_iou = s.spearman([r['score'] for r in duplicated], [r['iou'] for r in duplicated])
    expected_surface = s.spearman([r['score'] for r in duplicated], [r['surface'] for r in duplicated])
    assert samples[0][0] == expected_iou
    assert samples[1][0] == expected_surface
    assert samples[2][0] == expected_surface - expected_iou
    assert samples[2][0] != pytest.approx(points[2])


def test_constant_one_outcome_keeps_other_rho_but_delta_null():
    rows = [{'scene_id': scene, 'score': float(i), 'iou': .5, 'surface': i / 10}
            for i, scene in enumerate(('a', 'b', 'c'))]
    points, samples = s._paired_correlations(rows, ['a', 'b', 'c'], fixed_draws([0, 1, 2]))
    assert points == (None, 1., None)
    assert all(value is None for value in samples[0] + samples[2])
    assert all(value == 1. for value in samples[1])


@pytest.mark.parametrize('finite_count,buildings,point,valid', [
    (1900, 2, .2, True), (1899, 2, .2, False), (2000, 1, .2, False), (2000, 2, None, False)])
def test_ci_gate_without_replacement_and_type7(finite_count, buildings, point, valid):
    draws = [float(i) for i in range(finite_count)] + [None] * (2000 - finite_count)
    result = s._result(point, draws, buildings)
    assert result['bootstrap_finite_draws'] == finite_count
    assert (result['ci95'] is not None) == valid
    if valid:
        assert result['ci95'] == pytest.approx([(finite_count - 1) * .025, (finite_count - 1) * .975])


@pytest.mark.parametrize('mutation', ['truncated_string', 'missing_flag', 'score_nan', 'bad_surface',
                                      'failed_finite', 'missing_row', 'duplicate_row', 'qa_missing_reason'])
def test_malformed_source_flags_frame_and_nonfinite_values_rejected(mutation):
    scores, outcomes, qa, scenes = fixture()
    if mutation == 'truncated_string': scores[0]['token_diagnostics']['truncated'] = 'false'
    elif mutation == 'missing_flag': scores[0]['token_diagnostics'] = {}
    elif mutation == 'score_nan': scores[0]['score'] = float('nan')
    elif mutation == 'bad_surface': outcomes[0]['outcomes']['openai']['aligned_surface_f1_at_1'] = float('inf')
    elif mutation == 'failed_finite': outcomes[0]['outcomes']['openai']['status'] = 'failed'
    elif mutation == 'missing_row': outcomes.pop()
    elif mutation == 'duplicate_row': outcomes[-1] = deepcopy(outcomes[0])
    elif mutation == 'qa_missing_reason': qa[0].update(D_i=None, W_i=None)
    with pytest.raises(ValueError):
        s.analyze(scores, outcomes, qa, scenes, fixture=True)


def test_production_frame_cannot_shrink():
    with pytest.raises(ValueError, match='60 planned'):
        s.analyze(*fixture())


def test_input_permutation_does_not_change_masks_or_results():
    inputs = fixture(2)
    result = s.analyze(*inputs, fixture=True)
    reordered = tuple(list(reversed(values)) for values in inputs)
    second = s.analyze(*reordered, fixture=True)
    assert json.dumps(result, sort_keys=True) == json.dumps(second, sort_keys=True)
