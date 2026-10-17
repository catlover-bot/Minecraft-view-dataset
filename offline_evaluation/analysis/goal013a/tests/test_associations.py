"""Small synthetic fixtures only: no saved research outcome/score reads."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
import pytest

from offline_evaluation.analysis.goal013a import associations as a
from offline_evaluation.analysis.goal013a import outcome_binding as b


def phase2_gate():
    return {'status': 'PASS', 'score_freeze_verified': True, 'representativeness_status': 'PASS',
            'outcome_binding_status': 'PASS'}


def fixture(n=3):
    scores, sources, slots = [], [], []
    for scene_index in range(n):
        scene = f'scene_{scene_index:03}'
        for provider_index, provider in enumerate(a.PROVIDERS):
            identity = {'scene_id': scene, 'description_provider': provider,
                'O_input_id': f'{scene}:{provider}', 'O_input_sha256': '1' * 64,
                'description_source_sha256': '1' * 64, 'summary_sha256': '2' * 64}
            source = {**identity, 'binding_status': 'PASS', 'summary_status': 'VALID_ORIGINAL_SUMMARY',
                'images_ordered': [{'path': f'/synthetic/{scene}/view{v}.png', 'sha256': str(v) * 64,
                                   'bytes': 100 + v} for v in range(6)]}
            sources.append(source)
            scores.append({**deepcopy(source), 'score_id': a.SCORE_ID, 'score_status': 'COMPUTATION_VALID',
                'score': (3 * scene_index + provider_index + 1) / (n * 3),
                'token_coverage_status': 'VERIFIED', 'summary_token_count_untruncated': provider_index + 1 + scene_index})
            for builder_index, builder in enumerate(a.BUILDERS):
                index = len(slots)
                slots.append({'scene_id': scene, 'description_model': provider, 'provider': builder,
                    'condition': 'O', 'global_schedule_index': index, 'status': 'success',
                    'input_id': identity['O_input_id'], 'input_sha256': identity['O_input_sha256'],
                    'attempted': True, 'attempt_id': str(index), 'selected_attempt_id': str(index),
                    'selected_final_observation': True, 'usage_known': True, 'completed_request_outcome': True,
                    'failure_kind': None, 'metrics': {'metric_version': 'server_v2.2_strict_translation',
                        'aligned': {'occupancy_iou': ((scene_index + provider_index * 3 + builder_index) % 9) / 10}}})
    return scores, sources, slots


def joined(n=3):
    scores, sources, slots = fixture(n)
    rows, scenes = b.bind_selected_outcomes(scores, sources, slots, fixture=True)
    return rows, scenes


def independent_spearman(x, y):
    # Enumerate each observation's lower/equal counts, without sorting/rank code.
    def ranks(values):
        return [1 + sum(other < value for other in values) + (sum(other == value for other in values) - 1) / 2
                for value in values]
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    dx, dy = [v - mean(rx) for v in rx], [v - mean(ry) for v in ry]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(u * v for u, v in zip(dx, dy)) / denominator if denominator else None


@pytest.mark.parametrize('x,y', [([4, 1, 1, 3], [0, 2, 1, 3]), ([1, 2, 3], [9, 6, 1]),
    ([1, 1, 1], [1, 2, 3]), ([1], [2]), ([], []), ([1, np.nextafter(1., 2.), 2], [3, 2, 1])])
def test_spearman_matches_independent_enumeration_with_exact_ties(x, y):
    expected = independent_spearman(x, y)
    actual = a.spearman(x, y)
    assert actual is None if expected is None else actual == pytest.approx(expected, abs=1e-15)


def test_average_rank_exact_ties_and_nonnumeric_rejection():
    assert a.average_ranks([4, 1, 1, 3]).tolist() == [4, 1.5, 1.5, 3]
    with pytest.raises(ValueError, match='finite'):
        a.average_ranks([1, float('nan')])


def test_bootstrap_uses_all_planned_clusters_and_is_deterministic():
    scenes = [f's{i:02}' for i in range(60)]
    actual = a.bootstrap_indices(scenes)
    expected = np.random.Generator(np.random.PCG64(20261017)).integers(0, 60, size=(2000, 60), dtype=np.int64)
    assert actual.shape == (2000, 60)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, a.bootstrap_indices(scenes))
    assert len(set(actual[0])) < 60


def test_pooled_bootstrap_keeps_duplicate_clusters_and_reranks():
    pairs = [('a', 'openai', 1., .1, 0), ('a', 'gemini', 3., .3, 0),
             ('b', 'openai', 2., .9, 0), ('b', 'gemini', 4., .2, 0)]
    draws = np.array([[0, 0, 1]] * 2000)
    result = a._pooled_result(pairs, ['a', 'b', 'missing'], draws)
    expected = independent_spearman([1, 3, 1, 3, 2, 4], [.1, .3, .1, .3, .9, .2])
    assert result['bootstrap_draws'][0] == pytest.approx(expected)
    assert result['bootstrap_draws'][0] != pytest.approx(result['estimate'])
    assert result['paired_description_rows'] == 4
    assert result['contributing_buildings'] == 2


@pytest.mark.parametrize('finite_draws,clusters,passes', [(1900, 2, True), (1899, 2, False), (2000, 1, False)])
def test_ci_gate_and_type7_without_replacing_undefined_draws(finite_draws, clusters, passes):
    values = list(map(float, range(finite_draws))) + [None] * (2000 - finite_draws)
    result = a.bootstrap_interval(2., values, clusters)
    assert result['bootstrap_finite_draws'] == finite_draws
    assert result['bootstrap_undefined_draws'] == 2000 - finite_draws
    if passes:
        # Equally spaced values make type-7 quantiles exactly (n-1)*p.
        assert result['ci95'] == pytest.approx([(finite_draws - 1) * .025, (finite_draws - 1) * .975])
    else:
        assert result['ci95'] is None


def test_concordance_is_building_equal_not_pair_weighted_and_keeps_multiplicity():
    rows, scenes = joined()
    for row in rows:
        provider = a.PROVIDERS.index(row['description_provider'])
        scene = scenes.index(row['scene_id'])
        row['score'] = provider + 1.
        row['outcomes']['openai']['aligned_occupancy_iou'] = (provider + 1) / 4 if scene == 0 else (3 - provider) / 4
        if scene == 2 or (scene == 1 and provider == 2):
            row['score_status'], row['score'] = 'MISSING', None
    draws = np.array([[0, 0, 1]] * 2000)
    result = a._concordance_result(rows, scenes, draws, 'openai', 'valid_only')
    assert result['estimate'] == .5  # C_a=1, C_b=0; pooled pairs would be 3/4.
    assert result['bootstrap_draws'][0] == pytest.approx(2 / 3)
    assert result['planned_candidate_pairs'] == 9
    assert result['missing_pairs'] == 5
    assert result['comparable_pairs'] == 4
    assert result['concordant_pairs'] == 3
    assert result['discordant_pairs'] == 1
    assert result['paired_description_rows'] == 5
    assert result['contributing_buildings'] == 2


def test_missing_before_mutually_exclusive_ties_partition_all_pairs():
    rows, scenes = joined()
    for row in rows:
        provider = a.PROVIDERS.index(row['description_provider'])
        scene = scenes.index(row['scene_id'])
        row['score'] = [1., 1., 2.][provider]
        row['outcomes']['openai']['aligned_occupancy_iou'] = ([.1, .2, .2] if scene == 0 else [.1, .1, .2])[provider]
        if scene == 2:
            row['token_coverage_status'] = 'TOKEN_COVERAGE_NOT_VERIFIED'
    result = a._concordance_result(rows, scenes, a.bootstrap_indices(scenes), 'openai', 'valid_only')
    assert [result[k] for k in ('missing_pairs', 'score_only_tie_pairs', 'outcome_only_tie_pairs', 'both_tie_pairs', 'comparable_pairs')] == [3, 1, 1, 1, 3]
    assert result['estimate'] == 1


@pytest.mark.parametrize('kind,complete,known,status,http,expected', [
    ('parse_failure', True, True, 'failed', None, True),
    ('validation_failure', True, True, 'failed', None, True),
    ('execution_failure', True, True, 'failed', None, True),
    ('truncation', True, True, 'failed', None, True),
    ('http_failure', False, False, 'operational_unmeasurable', 520, False),
    ('truncation', True, False, 'failed', None, False),
    ('truncation', False, True, 'failed', None, False),
    ('context_length', True, True, 'failed', None, False),
    ('parse_failure', True, True, 'failed', 401, False),
    (None, True, True, 'success', None, False),
])
def test_only_known_completed_model_failures_are_failure0(kind, complete, known, status, http, expected):
    _, _, slots = fixture(1)
    slot = slots[0]
    slot.update(failure_kind=kind, completed_request_outcome=complete, usage_known=known, status=status,
                http_status=http, metrics=None)
    result = b._slot_outcome(slot)
    assert result['known_failure0_eligible'] is expected
    assert result['aligned_occupancy_iou'] is None


def test_normal_empty_success_zero_and_undefined_success_are_not_failures():
    _, _, slots = fixture(1)
    slots[0]['valid_nonempty_reconstruction'] = False
    slots[0]['metrics']['aligned']['occupancy_iou'] = 0.
    out = b._slot_outcome(slots[0])
    assert out['aligned_occupancy_iou'] == 0 and not out['known_failure0_eligible']
    slots[0]['metrics']['aligned']['occupancy_iou'] = None
    out = b._slot_outcome(slots[0])
    assert out['aligned_occupancy_iou'] is None and not out['known_failure0_eligible']


def test_fixed_analysis_targets_actual_denominators_and_token_diagnostics():
    rows, scenes = joined()
    rows[0]['outcomes']['openai'].update(status='failed', aligned_occupancy_iou=None, known_failure0_eligible=True)
    rows[1]['outcomes']['anthropic'].update(aligned_occupancy_iou=None)
    rows[2].update(score=None, score_status='MISSING_EMPTY_SUMMARY')
    rows[3].update(token_coverage_status='TOKEN_COVERAGE_NOT_VERIFIED')
    result = a.analyze(rows, scenes, phase2_gate=phase2_gate(), fixture=True)
    assert len(result['pooled_spearman']) == 6
    assert len(result['within_building_concordance']) == 4
    assert len(result['token_length_diagnostics']) == 4
    assert result['score_eligible_description_rows'] == 7
    entries = {(r['policy'], r['target']): r for r in result['pooled_spearman']}
    assert entries['valid_only', 'openai']['paired_description_rows'] == 6
    assert entries['known_failure0', 'openai']['paired_description_rows'] == 7
    assert entries['valid_only', 'anthropic']['paired_description_rows'] == 6
    assert entries['known_failure0', 'anthropic']['paired_description_rows'] == 6
    assert entries['valid_only', 'both_builders_mean']['paired_description_rows'] == 5
    assert entries['known_failure0', 'both_builders_mean']['paired_description_rows'] == 6
    assert entries['known_failure0', 'both_builders_mean']['failure0_imputed_builder_slots'] == 1
    assert result['token_length_diagnostics'][0]['paired_description_rows'] == 7
    assert [r['paired_description_rows'] for r in result['token_length_diagnostics'][1:]] == [2, 2, 3]
    assert result['inference']['p_values_computed'] is False


@pytest.mark.parametrize('mutation', ['duplicate', 'wrong_input', 'wrong_hash', 'wrong_source', 'wrong_summary',
    'wrong_scorer', 'missing_filled', 'image_order', 'image_hash', 'image_path', 'image_size', 'missing_view'])
def test_binding_rejects_identity_or_missingness_changes(mutation):
    scores, sources, slots = fixture(1)
    if mutation == 'duplicate':
        slots.append(deepcopy(slots[0]))
    elif mutation == 'wrong_input':
        slots[0]['input_id'] = 'other'
    elif mutation == 'wrong_hash':
        slots[0]['input_sha256'] = '8' * 64
    elif mutation == 'wrong_source':
        scores[0]['description_source_sha256'] = '8' * 64
    elif mutation == 'wrong_summary':
        scores[0]['summary_sha256'] = '8' * 64
    elif mutation == 'wrong_scorer':
        scores[0]['score_id'] = 'alternative'
    elif mutation == 'image_order':
        scores[0]['images_ordered'].reverse()
    elif mutation == 'image_hash':
        scores[0]['images_ordered'][0]['sha256'] = '9' * 64
    elif mutation == 'image_path':
        scores[0]['images_ordered'][0]['path'] = '/synthetic/other.png'
    elif mutation == 'image_size':
        scores[0]['images_ordered'][0]['bytes'] += 1
    elif mutation == 'missing_view':
        scores[0]['images_ordered'].pop()
    else:
        sources[0]['summary_status'] = 'MISSING_EMPTY_SUMMARY'
    with pytest.raises(ValueError):
        b.bind_selected_outcomes(scores, sources, slots, fixture=True)


def test_retry_stays_one_selected_observation_and_GT_does_not_enter_output():
    scores, sources, slots = fixture(1)
    slots[0].update(original_attempt_id='original_401', attempt_history_ids=['original_401', slots[0]['attempt_id']],
                    gt_grid={'forbidden': True}, artifacts={'voxels': 'forbidden'}, response='forbidden')
    rows, _ = b.bind_selected_outcomes(scores, sources, slots, fixture=True)
    assert len(rows) == 3
    selected = next(row for row in rows if row['description_provider'] == 'openai')
    assert selected['outcomes']['openai']['original_attempt_id'] == 'original_401'
    text = json.dumps(rows)
    assert 'forbidden' not in text and 'gt_grid' not in text and 'attempt_history_ids' not in text


def write_bound(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def frozen_fixture(tmp_path):
    scores, sources, _ = fixture(60)
    gate = {'status': 'PASS', 'overall': {'status': 'PASS'},
            'providers': {p: {'status': 'PASS'} for p in a.PROVIDERS}}
    freeze = {'status': 'PASS', 'source_binding_status': 'PASS', 'scorer_validation_status': 'PASS',
        'score_id': a.SCORE_ID, 'code_commit': '1' * 40, 'spec_commit': '2' * 40,
        'score_artifact': write_bound(tmp_path / 'scores.json', scores),
        'source_binding_artifact': write_bound(tmp_path / 'source.json', sources),
        'representativeness_artifact': write_bound(tmp_path / 'gate.json', gate)}
    return freeze


@pytest.mark.parametrize('mutation', ['freeze_hash', 'source_gate', 'scorer_gate', 'code_commit', 'same_commits',
    'overall_gate', 'provider_gate', 'scores_hash', 'source_hash', 'score_identity'])
def test_outcome_file_not_opened_before_every_gate_passes(tmp_path, monkeypatch, mutation):
    freeze = frozen_fixture(tmp_path)
    if mutation == 'source_gate':
        freeze['source_binding_status'] = 'FAIL'
    elif mutation == 'scorer_gate':
        freeze['scorer_validation_status'] = 'FAIL'
    elif mutation == 'code_commit':
        freeze['code_commit'] = None
    elif mutation == 'same_commits':
        freeze['spec_commit'] = freeze['code_commit']
    elif mutation in ('overall_gate', 'provider_gate'):
        gate_path = tmp_path / 'gate.json'
        gate = json.loads(gate_path.read_bytes())
        if mutation == 'overall_gate':
            gate['overall']['status'] = 'FAIL'
        else:
            gate['providers']['gemini']['status'] = 'FAIL'
        freeze['representativeness_artifact'] = write_bound(gate_path, gate)
    elif mutation in ('scores_hash', 'source_hash'):
        key = 'score_artifact' if mutation == 'scores_hash' else 'source_binding_artifact'
        freeze[key]['sha256'] = '9' * 64
    elif mutation == 'score_identity':
        score_path = tmp_path / 'scores.json'
        scores = json.loads(score_path.read_bytes())
        scores[0]['O_input_id'] = 'changed'
        freeze['score_artifact'] = write_bound(score_path, scores)
    descriptor = write_bound(tmp_path / 'freeze.json', freeze)
    reader = b._read_bound
    reads = []
    def guarded(path, *args, **kwargs):
        reads.append(Path(path))
        if Path(path) == b.GOAL007_MANIFEST or str(path).endswith(b.GOAL007_SLOTS_PRODUCT):
            raise AssertionError('old outcome source opened before gates')
        return reader(path, *args, **kwargs)
    monkeypatch.setattr(b, '_read_bound', guarded)
    expected = '0' * 64 if mutation == 'freeze_hash' else descriptor['sha256']
    with pytest.raises(ValueError):
        b.load_frozen_outcomes(descriptor['path'], expected_freeze_sha256=expected)
    assert b.GOAL007_MANIFEST not in reads


def test_arithmetic_gate_is_fail_closed():
    rows, scenes = joined()
    gate = phase2_gate()
    gate['representativeness_status'] = 'FAIL'
    with pytest.raises(ValueError, match='gates'):
        a.analyze(rows, scenes, phase2_gate=gate, fixture=True)


def test_production_loader_resolves_manifest_canonical_path_after_gates(tmp_path, monkeypatch):
    freeze = frozen_fixture(tmp_path)
    _, _, slots = fixture(60)
    for index in range(360, 960):
        # Unused synthetic non-O entries preserve the frozen 960/934/25/1 scope.
        slots.append({'global_schedule_index': index, 'condition': 'J',
                      'status': 'failed' if index < 385 else 'operational_unmeasurable' if index == 942 else 'success'})
    canonical = tmp_path / 'canonical'
    (canonical / 'data').mkdir(parents=True)
    slot_descriptor = write_bound(canonical / b.GOAL007_SLOTS_PRODUCT, slots)
    manifest = {'output_root': str(canonical), 'source_inventories_before_after_equal': True,
                'products': {b.GOAL007_SLOTS_PRODUCT: slot_descriptor}}
    mirror = tmp_path / 'mirror.json'
    manifest_descriptor = write_bound(mirror, manifest)
    write_bound(canonical / 'MANIFEST.json', manifest)
    monkeypatch.setattr(b, 'GOAL007_MANIFEST', mirror)
    monkeypatch.setattr(b, 'GOAL007_MANIFEST_SHA256', manifest_descriptor['sha256'])
    monkeypatch.setattr(b, 'GOAL007_SLOTS_SHA256', slot_descriptor['sha256'])
    descriptor = write_bound(tmp_path / 'freeze.json', freeze)
    rows, scenes, receipt = b.load_frozen_outcomes(descriptor['path'], expected_freeze_sha256=descriptor['sha256'])
    assert len(rows) == 180 and len(scenes) == 60
    assert receipt['bound_selected_O_slots'] == 360
    assert receipt['selected_slots_artifact']['path'] == str(canonical / b.GOAL007_SLOTS_PRODUCT)
    assert receipt['GT_files_opened'] == receipt['source_writes'] == 0
    assert receipt['score_freeze_verified'] is True


def test_production_frame_cannot_silently_shrink():
    rows, scenes = joined()
    with pytest.raises(ValueError, match='60'):
        a.analyze(rows, scenes, phase2_gate=phase2_gate())
