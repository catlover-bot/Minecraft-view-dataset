"""Synthetic Goal007 denominator, missingness and frozen-inference regression tests."""
from copy import deepcopy
import json
from statistics import mean
import socket

import pytest

from offline_evaluation.analysis.goal007 import statistics as stats
from offline_evaluation.server_v2 import analysis_v4 as legacy
from offline_evaluation.server_v2.core import Grid, evaluate_pair
from offline_evaluation.server_v2.statistics import holm_adjust


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Goal007 statistics fixtures may not access the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


def row(index=0, scene='fixture-0', provider='openai', condition='O', model='openai', value=.4,
        status='success', failure_kind=None, **extra):
    block = {'occupancy_iou': value, 'dimension_mae': 2.0, 'material_set_f1': .7,
             'material_aware_iou': value / 2, 'add_count': 2, 'delete_count': 1, 'replace_count': 3,
             'total_normalized_repair_cost': .6}
    result = {'global_schedule_index': index, 'schedule_index': index, 'scene_id': scene,
              'provider': provider, 'condition': condition,
              'description_model': model if condition in ('O', 'J') else None,
              'input_id': f'fixture-{scene}-{condition}-{model}', 'attempt_id': f'fixture-attempt-{index}',
              'attempted': True, 'completed_request_outcome': status != 'operational_unmeasurable',
              'usage_known': status != 'operational_unmeasurable', 'status': status,
              'failure_kind': failure_kind, 'usage': {'input_tokens': 1, 'output_tokens': 2} if status != 'operational_unmeasurable' else None, 'response': None,
              'metrics': {'metric_version': 'server_v2.2_strict_translation',
                          'aligned': dict(block), 'absolute': dict(block)} if status == 'success' else None,
              'http_status_code': 520 if status == 'operational_unmeasurable' else None}
    result.update(extra)
    return result


def fixed_rows():
    values = []
    for building in range(60):
        for condition in ('O', 'J', 'P', 'E'):
            for model in stats.DESCRIPTION_PROVIDERS if condition in ('O', 'J') else (None,):
                for provider in legacy.PROVIDERS:
                    values.append(row(len(values), f'fixture-building-{building:02}', provider, condition, model))
    target = next(r for r in values if r['scene_id'] == 'fixture-building-58' and r['provider'] == 'openai'
                  and r['condition'] == 'J' and r['description_model'] == 'openai')
    other = values[942]; displaced = target['global_schedule_index']
    target.update(global_schedule_index=942, schedule_index=942)
    other.update(global_schedule_index=displaced, schedule_index=displaced)
    target.update(status='operational_unmeasurable', metrics=None, usage=None, completed_request_outcome=False,
                  usage_known=False, failure_kind='http_failure', http_status_code=520)
    return sorted(values, key=lambda r: r['global_schedule_index'])


def comparisons(result):
    return {r['id']: r for r in result['comparisons']}


@pytest.mark.parametrize('gate', [None, {}, {'status': 'FAIL', 'input_kind': 'verified_sources'},
                                   {'status': 'PASS', 'input_kind': 'unaudited'}])
def test_failed_or_missing_audit_gate_prevents_statistics(gate, monkeypatch):
    monkeypatch.setattr(legacy, 'summarize_v4', lambda *_a, **_k: pytest.fail('statistics ran before audit'))
    with pytest.raises(ValueError, match='audit gate'):
        stats.analyze_slots([], audit_gate=gate)


def test_fixed960_grid_missingness_and_description_builder_denominators():
    rows = fixed_rows()
    rows[0].update(status='failed', metrics=None, failure_kind='truncation')
    checked = stats.validate_rows(rows)
    assert len(checked) == 960
    counts = stats._counts(checked)
    assert counts['planned_slots'] == 960 and counts['independent_buildings'] == 60
    assert counts['completed_request_outcome_slots'] == 959 and counts['unknown_usage_slots'] == 1
    assert counts['model_output_failure_slots'] == 1 and counts['valid_only_aligned_iou_slots'] == 958
    assert counts['bounded_iou_eligible_slots'] == 959 and counts['operational_unmeasurable_slots'] == 1
    groups, denominators = stats.descriptive_groups(checked)
    assert len(denominators) == 20 and len(groups) == 20 * len(stats.METRICS)
    overall = [g for g in denominators if g['group_level'] == 'builder_condition']
    assert sum(g['planned_slots'] for g in overall) == 960
    assert {(g['condition'], g['planned_slots']) for g in overall} == {('O', 180), ('J', 180), ('P', 60), ('E', 60)}
    crossed = [g for g in denominators if g['group_level'] == 'description_builder_condition']
    assert len(crossed) == 12 and all(g['planned_slots'] == 60 for g in crossed)
    assert all(g['condition'] in ('O', 'J') for g in crossed)
    missing = next(g for g in crossed if (g['provider'], g['condition'], g['description_model']) == ('openai', 'J', 'openai'))
    assert missing['planned_slots'] == 60 and missing['bounded_iou_eligible_slots'] == 59
    assert missing['valid_only_aligned_iou_slots'] == 59


def test_integration_http_status_field_and_conflicting_aliases():
    values = fixed_rows()
    for value in values:
        value['http_status'] = value.pop('http_status_code')
    stats.validate_rows(values)
    values[942]['http_status_code'] = 401
    with pytest.raises(ValueError, match='aliases disagree'):
        stats.validate_rows(values)


@pytest.mark.parametrize('mutation,match', [
    (lambda rows: rows.append(deepcopy(rows[0])), 'duplicate final'),
    (lambda rows: rows[0].update(global_schedule_index=1), 'schedule index|global_schedule'),
    (lambda rows: rows[942].update(metrics={'aligned': {'occupancy_iou': 0.}}), 'metrics must be null'),
    (lambda rows: rows[942].update(usage_known=True), 'operational missing'),
    (lambda rows: rows[942].update(completed_request_outcome=True), 'known usage'),
    (lambda rows: rows[942].update(http_status_code=401), 'global942'),
    (lambda rows: rows[0]['metrics'].update(metric_version='server_v2.1'), 'strict_translation'),
])
def test_invalid_fixed_slot_or_missing_imputation_rejected(mutation, match):
    values = fixed_rows(); mutation(values)
    with pytest.raises(ValueError, match=match):
        stats.validate_rows(values)


@pytest.mark.parametrize('failure', ['parse_failure', 'validation_failure', 'execution_failure', 'truncation'])
def test_known_model_failure_has_only_bounded_iou_failure0(failure):
    values = stats.validate_rows([row(status='failed', failure_kind=failure)], fixed_design=False)
    groups, _ = stats.descriptive_groups(values)
    relevant = [g for g in groups if g['group_level'] == 'builder_condition' and g['provider'] == 'openai' and g['condition'] == 'O']
    for record in relevant:
        assert record['valid_rows'] == 0 and record['equal_building_mean'] is None
        if record['metric'] in legacy.BOUNDED_IOU:
            assert record['failure0_eligible_rows'] == 1 and record['failure0_equal_building_mean'] == 0
        else:
            assert record['failure0_eligible_rows'] is None and record['failure0_equal_building_mean'] is None


@pytest.mark.parametrize('status', [401, 520])
def test_http_status_alone_cannot_create_failure0(status):
    invalid = row(status='failed', failure_kind='parse_failure', http_status_code=status)
    with pytest.raises(ValueError, match='operational error'):
        stats.validate_rows([invalid], fixed_design=False)
    missing = row(942, condition='J', status='operational_unmeasurable', failure_kind='http_failure', http_status_code=status)
    assert legacy.bounded_all_attempted_value(missing) is None


def test_other_known_failure_remains_separate_and_is_not_failure0():
    values = stats.validate_rows([row(status='failed', failure_kind='refusal')], fixed_design=False)
    counts = stats._counts(values)
    assert counts['model_output_failure_slots'] == counts['known_completed_other_failure_slots'] == 1
    assert counts['failure0_definitive_model_failure_slots'] == counts['bounded_iou_eligible_slots'] == 0


def test_frozen_inference_exactly_reused_and_all_twenty_holm_tests_retained():
    values = []
    for scene in ('fixture-a', 'fixture-b', 'fixture-c'):
        values.append(row(len(values), scene, value=.2))
        values.append(row(len(values), scene, provider='anthropic', value=.8))
    before = deepcopy(values)
    result = stats.analyze_slots(values, audit_gate={'status': 'PASS', 'input_kind': 'fixture'})
    original = legacy.summarize_v4(values, seed=stats.SEED, bootstrap_replicates=2000, signflip_replicates=10000, smoke=False)
    assert result['summary'] == original and values == before
    assert [(f['id'], f['predeclared_size']) for f in result['summary']['families']] == [(f['id'], f['size']) for f in legacy.comparison_families()]
    assert len(result['comparisons']) == 20
    assert result['inference'] == {'seed': 20261013, 'bootstrap_replicates': 2000, 'signflip_replicates': 10000}
    by_id = comparisons(result)
    contrast = by_id['anthropic-minus-openai:O:aligned_occupancy_iou']
    assert contrast['buildings'] == 3 and contrast['common_valid_pair_count'] == 3
    assert contrast['mean_delta'] == pytest.approx(.6) and contrast['p_two_sided'] == .25
    assert contrast['signflip_method'] == 'exact' and contrast['p_holm'] == 1.0
    for family in result['summary']['families']:
        expected = holm_adjust([r['p_two_sided'] if r['p_two_sided'] is not None else 1.0 for r in family['results']])
        assert [r['p_holm'] for r in family['results']] == [p if r['p_two_sided'] is not None else None for r, p in zip(family['results'], expected)]


def test_metric_specific_pairs_and_equal_building_weight_do_not_count_descriptions_as_buildings():
    values = []
    for model in stats.DESCRIPTION_PROVIDERS:
        values.append(row(len(values), 'fixture-a', model=model, value=0.))
        values.append(row(len(values), 'fixture-a', provider='anthropic', model=model, value=1.))
    values += [row(6, 'fixture-b', value=1.), row(7, 'fixture-b', provider='anthropic', value=0.)]
    values[1]['metrics']['aligned']['dimension_mae'] = None
    result = stats.analyze_slots(values, audit_gate={'status': 'PASS', 'input_kind': 'fixture'})
    by_id = comparisons(result)
    iou = by_id['anthropic-minus-openai:O:aligned_occupancy_iou']
    mae = by_id['anthropic-minus-openai:O:aligned_dimension_mae']
    assert iou['buildings'] == mae['buildings'] == 2
    assert iou['common_valid_pair_count'] == 4 and mae['common_valid_pair_count'] == 3
    assert iou['mean_delta'] == 0.0  # Row-weighted mean would be +0.5.
    assert iou['p_two_sided'] == 1.0


def test_frozen_monte_carlo_seed_and_plus_one_signflip_are_unchanged():
    values = []
    for scene in range(17):
        values += [row(len(values), f'fixture-{scene:02}', value=.2),
                   row(len(values) + 1, f'fixture-{scene:02}', provider='anthropic', value=.8)]
    result = stats.analyze_slots(values, audit_gate={'status': 'PASS', 'input_kind': 'fixture'})
    contrast = comparisons(result)['anthropic-minus-openai:O:aligned_occupancy_iou']
    pairs = legacy._paired(values, contrast)
    expected = legacy.paired_building_inference(pairs, seed=stats.SEED, bootstrap_replicates=2000,
        signflip_replicates=10000, domain='primary_builder_valid_pair' + contrast['id'])
    assert all(contrast[key] == value for key, value in expected.items())
    assert contrast['signflip_method'] == 'Monte Carlo plus-one'
    assert contrast['p_two_sided'] >= 1 / 10001


@pytest.mark.parametrize('status', ['ALIGNMENT_RESOURCE_LIMIT', 'GT_MISSING'])
def test_unavailable_strict_metric_keeps_null_without_failure0(status):
    value = row(status=status, failure_kind=status)
    checked = stats.validate_rows([value], fixed_design=False)
    counts = stats._counts(checked)
    assert counts['completed_request_outcome_slots'] == 1 and counts['unknown_usage_slots'] == 0
    assert counts['valid_only_aligned_iou_slots'] == counts['bounded_iou_eligible_slots'] == 0
    assert counts['model_output_failure_slots'] == 0
    assert legacy.bounded_all_attempted_value(checked[0]) is None


def test_missing942_sensitivity_has_only_descriptive_bounds_and_exact_pair_loss():
    values = []
    for scene in ('fixture-a', 'fixture-b'):
        for condition in ('O', 'J'):
            for model in stats.DESCRIPTION_PROVIDERS:
                for provider in legacy.PROVIDERS:
                    values.append(row(len(values), scene, provider, condition, model, value=.25 if provider == 'openai' else .75))
    missing = next(r for r in values if (r['scene_id'], r['provider'], r['condition'], r['description_model']) == ('fixture-a', 'openai', 'J', 'openai'))
    missing.update(global_schedule_index=942, schedule_index=942, status='operational_unmeasurable',
                   metrics=None, usage=None, usage_known=False, completed_request_outcome=False, failure_kind='http_failure', http_status_code=520)
    before = deepcopy(values)
    result = stats.sensitivity_missing942(stats.validate_rows(values, fixed_design=False))
    assert values == before and result['exploratory'] is True
    affected = {r['id']: r for r in result['affected_comparisons']}
    assert len(affected) == 4
    entry = affected['anthropic-minus-openai:J:aligned_occupancy_iou']
    assert entry['main_common_valid']['pairs'] == 5 and entry['main_common_valid']['buildings'] == 2
    assert entry['possible_additional_pairs'] == 1 and entry['possible_additional_buildings'] == 0
    assert entry['mean_delta_bounds'] == pytest.approx([.375, .5416666666666666])
    assert all(x['common_valid']['pairs'] == 6 for x in entry['endpoint_scenarios'])
    mae = affected['anthropic-minus-openai:J:aligned_dimension_mae']
    assert mae['endpoint_scenarios'] is None and mae['bound_status'] == 'NO_FINITE_0_TO1_BOUND_FOR_MAE'
    assert mae['possible_additional_pairs'] == 1 and mae['possible_additional_buildings'] == 0
    assert 'p_two_sided' not in json.dumps(result) and 'p_holm' not in json.dumps(result)
    assert 'ci95' not in json.dumps(result)


def test_missing_counterpart_already_invalid_does_not_invent_a_recovered_pair():
    values = [row(942, condition='J', status='operational_unmeasurable', failure_kind='http_failure'),
              row(943, provider='anthropic', condition='J', status='failed', failure_kind='truncation')]
    result = stats.sensitivity_missing942(stats.validate_rows(values, fixed_design=False))
    entry = next(r for r in result['affected_comparisons'] if r['id'] == 'anthropic-minus-openai:J:aligned_occupancy_iou')
    assert entry['possible_additional_pairs'] == entry['possible_additional_buildings'] == 0
    assert entry['mean_delta_bounds'] is None
    assert entry['bounded_failure0_mean_delta_bounds'] == [-1., 0.]


def test_strict_evaluator_preserves_gt_direction_and_material_identity():
    gt = Grid({(0, 0, 0): 'stone', (1, 0, 0): 'oak_planks', (2, 0, 0): 'stone'})
    pred = Grid({(0, 0, 0): 'cobblestone', (1, 0, 0): 'oak_planks'})
    metrics = evaluate_pair(gt, pred, alignment_mode='strict_translation')
    assert metrics['metric_version'] == 'server_v2.2_strict_translation'
    assert metrics['absolute']['add_count'] == 1 and metrics['absolute']['delete_count'] == 0
    assert metrics['absolute']['replace_count'] == 1
    assert metrics['absolute']['total_normalized_repair_cost'] == pytest.approx(2 / 3)
    reversed_metrics = evaluate_pair(pred, gt, alignment_mode='strict_translation')
    assert reversed_metrics['absolute']['add_count'] == 0 and reversed_metrics['absolute']['delete_count'] == 1
    assert reversed_metrics['absolute']['total_normalized_repair_cost'] == 1.0
    assert metrics['absolute']['material_aware_iou'] == pytest.approx(1 / 4)
    value = row(); value['metrics'] = metrics
    groups, _ = stats.descriptive_groups(stats.validate_rows([value], fixed_design=False))
    record = next(r for r in groups if r['group_level'] == 'builder_condition' and r['provider'] == 'openai'
                  and r['condition'] == 'O' and r['metric'] == 'absolute_total_normalized_repair_cost')
    assert record['equal_building_mean'] == pytest.approx(2 / 3)
