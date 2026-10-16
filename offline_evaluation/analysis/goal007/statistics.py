"""Read-only Goal007 statistics using the frozen v4 estimands and inference.

This module receives already audited final slots, never ledgers or live API
credentials. It does not call aggregate_verified or relax its complete guard.
Operationally unmeasurable slots remain present with null metrics throughout.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from statistics import mean

from offline_evaluation.server_v2 import analysis_v4 as legacy
from offline_evaluation.server_v2.cohort_v4 import SEED

ANALYSIS_LABEL = 'fixed60 analysis with one operational missing observation'
METRICS = (*legacy.PRIMARY_METRICS, *legacy.SECONDARY_METRICS, 'absolute_dimension_mae')
DESCRIPTION_PROVIDERS = ('openai', 'anthropic', 'gemini')
INFERENCE = {'seed': SEED, 'bootstrap_replicates': 2000, 'signflip_replicates': 10000}


def _http_status(row):
    if 'http_status' in row and 'http_status_code' in row and row['http_status'] != row['http_status_code']:
        raise ValueError('HTTP status aliases disagree')
    return row.get('http_status', row.get('http_status_code'))


def validate_rows(rows, *, fixed_design=True):
    """Audit missingness and eligibility before entering frozen statistics."""
    rows = legacy._validated(rows)
    indices, pairs = set(), set()
    for row in rows:
        index = row.get('global_schedule_index')
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError('unique nonnegative global_schedule_index required')
        indices.add(index)
        if 'schedule_index' in row and row['schedule_index'] != index:
            raise ValueError('global and source schedule index disagree')
        if type(row.get('usage_known')) is not bool or type(row.get('completed_request_outcome')) is not bool:
            raise ValueError('explicit usage_known and completed_request_outcome booleans required')
        if row.get('attempted') is not True or not isinstance(row.get('attempt_id'), str) or not row['attempt_id']:
            raise ValueError('each final fixed slot requires its selected attempt identity')
        if row['completed_request_outcome'] and not row['usage_known']:
            raise ValueError('completed request outcome requires known usage')
        if row['status'] == 'success' and (not row['completed_request_outcome'] or not isinstance(row.get('metrics'), dict)):
            raise ValueError('scored success requires identified complete outcome and strict metric object')
        if row['status'] == 'operational_unmeasurable':
            if (row['completed_request_outcome'] or row['usage_known'] or row.get('metrics') is not None
                    or row.get('usage') is not None or row.get('response') is not None
                    or row.get('response_id') is not None or row.get('response_model') is not None):
                raise ValueError('operational missing observation must retain null usage response and metrics')
        if legacy.attempted_failure(row) and not row['usage_known']:
            raise ValueError('failure0 requires known usage and identified completed output')
        if row['status'] not in {'success', 'failed', 'operational_unmeasurable', 'ALIGNMENT_RESOURCE_LIMIT', 'GT_MISSING'}:
            raise ValueError('unexpected final-slot analysis status')
        if row['status'] == 'failed' and (not row['completed_request_outcome'] or not row['usage_known']
                or not row.get('failure_kind') or _http_status(row) in {401, 520}):
            raise ValueError('failed final observation must be a known completed model output, not an operational error')
        pair = (row['provider'], row.get('input_id'))
        if fixed_design:
            if not isinstance(pair[1], str) or not pair[1] or pair in pairs:
                raise ValueError('unique provider/input_id required')
            pairs.add(pair)
        # Validate every reported value, including metrics not used in a test.
        for metric in METRICS:
            legacy._metric(row, metric)
        _http_status(row)
    scenes = {row['scene_id'] for row in rows}
    if len(scenes) > 60:
        raise ValueError('buildings are independent units; maximum 60')
    if fixed_design:
        if len(rows) != 960 or indices != set(range(960)) or len(scenes) != 60:
            raise ValueError('fixed60 analysis requires exactly960 unique global slots and60 buildings')
        expected = Counter({('O', p, d): 1 for p in legacy.PROVIDERS for d in DESCRIPTION_PROVIDERS})
        expected.update({('J', p, d): 1 for p in legacy.PROVIDERS for d in DESCRIPTION_PROVIDERS})
        expected.update({(c, p, None): 1 for c in ('P', 'E') for p in legacy.PROVIDERS})
        for scene in scenes:
            actual = Counter((r['condition'], r['provider'], r.get('description_model')) for r in rows if r['scene_id'] == scene)
            if actual != expected:
                raise ValueError('each building must have O6 J6 P2 E2 without oracle replication')
        missing = [r for r in rows if r['status'] == 'operational_unmeasurable']
        if (len(missing) != 1 or missing[0]['global_schedule_index'] != 942
                or missing[0]['condition'] != 'J' or missing[0]['provider'] != 'openai'
                or _http_status(missing[0]) != 520):
            raise ValueError('the operational missing slot must be global942 J OpenAI HTTP520')
    return rows


def _counts(rows):
    valid = [r for r in rows if legacy._metric(r, 'aligned_occupancy_iou') is not None]
    failures = [r for r in rows if legacy.attempted_failure(r)]
    bounded = [r for r in rows if legacy.bounded_all_attempted_value(r) is not None]
    return {'planned_slots': len(rows), 'independent_buildings': len({r['scene_id'] for r in rows}),
            'selected_attempts': sum(r.get('attempted') is True for r in rows),
            'completed_request_outcome_slots': sum(r['completed_request_outcome'] for r in rows),
            'model_output_failure_slots': sum(r['status'] == 'failed' for r in rows),
            'failure0_definitive_model_failure_slots': len(failures),
            'known_completed_other_failure_slots': sum(r['status'] == 'failed' and not legacy.attempted_failure(r) for r in rows),
            'operational_unmeasurable_slots': sum(r['status'] == 'operational_unmeasurable' for r in rows),
            'unknown_usage_slots': sum(not r['usage_known'] for r in rows),
            'valid_only_aligned_iou_slots': len(valid), 'bounded_iou_eligible_slots': len(bounded),
            'alignment_resource_limit_slots': sum(r['status'] == 'ALIGNMENT_RESOURCE_LIMIT' for r in rows),
            'gt_missing_slots': sum(r['status'] == 'GT_MISSING' for r in rows),
            'status_counts': dict(sorted(Counter(r['status'] for r in rows).items())),
            'failure_kind_counts': dict(sorted(Counter(r['failure_kind'] for r in rows if r.get('failure_kind')).items()))}


def _equal_building(values):
    grouped = {}
    for row, value in values:
        if value is not None:
            grouped.setdefault(row['scene_id'], []).append(value)
    return {'rows': sum(len(values) for values in grouped.values()), 'buildings': len(grouped),
            'mean': mean(mean(values) for values in grouped.values()) if grouped else None}


def descriptive_groups(rows):
    """Flat metric-long rows; description providers are never independent buildings."""
    result, denominators = [], []
    for provider in legacy.PROVIDERS:
        for condition in legacy.CONDITIONS:
            groupings = [('builder_condition', None)]
            if condition in ('O', 'J'):
                groupings += [('description_builder_condition', model) for model in DESCRIPTION_PROVIDERS]
            for level, model in groupings:
                selected = [r for r in rows if r['provider'] == provider and r['condition'] == condition
                            and (level == 'builder_condition' or r.get('description_model') == model)]
                identity = {'group_level': level, 'provider': provider, 'condition': condition, 'description_model': model}
                counts = _counts(selected)
                denominators.append({**identity, **counts})
                scalar_counts = {k: v for k, v in counts.items() if not isinstance(v, dict)}
                for metric in METRICS:
                    valid = legacy._descriptive_metric(selected, metric)
                    companion = _equal_building((row, legacy.bounded_all_attempted_value(row, metric)) for row in selected) if metric in legacy.BOUNDED_IOU else None
                    result.append({**identity, 'metric': metric, **scalar_counts, **valid,
                        'failure0_eligible_rows': companion['rows'] if companion else None,
                        'failure0_buildings': companion['buildings'] if companion else None,
                        'failure0_equal_building_mean': companion['mean'] if companion else None})
    return result, denominators


def _direction(comparison):
    name_a = comparison['a_provider'] if comparison['a_provider'] != comparison['b_provider'] else comparison['a_condition']
    name_b = comparison['b_provider'] if comparison['a_provider'] != comparison['b_provider'] else comparison['b_condition']
    higher_better = comparison['metric'] in legacy.BOUNDED_IOU or comparison['metric'].endswith('material_set_f1')
    return {'effect_direction': name_b + ' minus ' + name_a,
            'metric_preference': 'higher is better' if higher_better else 'lower is better',
            'positive_delta_means': ('higher ' if higher_better else 'worse ') + comparison['metric'] + ' for ' + name_b}


def _comparison_records(rows, summary):
    records = []
    for family in summary['families']:
        for result in family['results']:
            pairs = legacy._paired(rows, result, True) if result['metric'] in legacy.BOUNDED_IOU else None
            records.append({**deepcopy(result), 'family_id': family['id'],
                'family_size': family['predeclared_size'], **_direction(result),
                'common_failure0_pair_count': len(pairs) if pairs is not None else None,
                'common_failure0_buildings': len({p['scene_id'] for p in pairs}) if pairs is not None else None})
    return records


def _pair_description(pairs):
    grouped = {}
    for pair in pairs:
        grouped.setdefault(pair['scene_id'], []).append(pair['b'] - pair['a'])
    return {'pairs': len(pairs), 'buildings': len(grouped),
            'mean_delta': mean(mean(values) for values in grouped.values()) if grouped else None}


def sensitivity_missing942(rows):
    """Exploratory deterministic IoU bounds only; no new inference or p-values.

    The null source observation is never changed. Temporary endpoint scenarios
    only answer what the paired descriptive mean could be if its IoU were known.
    """
    missing = [r for r in rows if r['global_schedule_index'] == 942 and r['status'] == 'operational_unmeasurable']
    if not missing:
        return {'status': 'NO_MISSING942_IN_FIXTURE', 'exploratory': True, 'affected_comparisons': []}
    if len(missing) != 1:
        raise ValueError('exactly one missing942 observation required')
    row = missing[0]
    result = {'status': 'EXPLORATORY_BOUNDS_ONLY', 'exploratory': True,
        'missing_slot': {'global_schedule_index': 942, 'scene_id': row['scene_id'], 'provider': row['provider'],
                         'condition': row['condition'], 'description_model': row.get('description_model'),
                         'status': row['status'], 'metrics': None},
        'endpoint_assumption': 'Hypothetical aligned occupancy IoU in [0,1], not observed outcomes or failure0 imputation.',
        'inference': 'No p-values, confidence intervals, or post-hoc significance claims computed for endpoint scenarios.',
        'affected_comparisons': [], 'group_bounds': []}
    scenarios = {}
    for endpoint in (0.0, 1.0):
        scenario = deepcopy(rows)
        hypothetical = next(r for r in scenario if r['global_schedule_index'] == 942)
        hypothetical.update(status='success', metrics={'metric_version': 'server_v2.2_strict_translation',
                            'aligned': {'occupancy_iou': endpoint}})
        scenarios[endpoint] = scenario
    for family in legacy.comparison_families():
        for comparison in family['comparisons']:
            if not any(row['provider'] == comparison[p + '_provider'] and row['condition'] == comparison[p + '_condition'] for p in ('a', 'b')):
                continue
            baseline_pairs = legacy._paired(rows, comparison)
            baseline = _pair_description(baseline_pairs)
            missing_side = next(p for p in ('a', 'b') if row['provider'] == comparison[p + '_provider']
                                and row['condition'] == comparison[p + '_condition'])
            other_side = 'b' if missing_side == 'a' else 'a'
            eligible_counterpart = any(r['scene_id'] == row['scene_id']
                and r.get('description_model') == row.get('description_model')
                and r['provider'] == comparison[other_side + '_provider']
                and r['condition'] == comparison[other_side + '_condition']
                and legacy._metric(r, comparison['metric']) is not None for r in rows)
            entry = {'family_id': family['id'], **comparison, **_direction(comparison),
                     'main_common_valid': baseline, 'main_missing_excluded': True,
                     'possible_additional_pairs': int(eligible_counterpart),
                     'possible_additional_buildings': int(eligible_counterpart and row['scene_id'] not in {p['scene_id'] for p in baseline_pairs})}
            if comparison['metric'] == 'aligned_occupancy_iou':
                bounds = []
                for endpoint, scenario in scenarios.items():
                    bounds.append({'hypothetical_missing_iou': endpoint,
                        'common_valid': _pair_description(legacy._paired(scenario, comparison)),
                        'bounded_failure0_companion': _pair_description(legacy._paired(scenario, comparison, True))})
                deltas = [e['common_valid']['mean_delta'] for e in bounds if e['common_valid']['mean_delta'] is not None]
                companions = [e['bounded_failure0_companion']['mean_delta'] for e in bounds if e['bounded_failure0_companion']['mean_delta'] is not None]
                entry.update(endpoint_scenarios=bounds, mean_delta_bounds=[min(deltas), max(deltas)] if deltas else None,
                    bounded_failure0_mean_delta_bounds=[min(companions), max(companions)] if companions else None,
                    possible_additional_pairs=bounds[0]['common_valid']['pairs'] - baseline['pairs'],
                    possible_additional_buildings=bounds[0]['common_valid']['buildings'] - baseline['buildings'])
            else:
                entry['endpoint_scenarios'] = None
                entry['bound_status'] = 'NO_FINITE_0_TO1_BOUND_FOR_MAE'
            result['affected_comparisons'].append(entry)
    for level, model in (('builder_condition', None), ('description_builder_condition', row.get('description_model'))):
        def select(values):
            return [r for r in values if r['provider'] == row['provider'] and r['condition'] == row['condition']
                    and (level == 'builder_condition' or r.get('description_model') == model)]
        entry = {'group_level': level, 'provider': row['provider'], 'condition': row['condition'], 'description_model': model,
                 'main_valid_only': legacy._descriptive_metric(select(rows), 'aligned_occupancy_iou')}
        endpoints = []
        for endpoint, scenario in scenarios.items():
            selected = select(scenario)
            endpoints.append({'hypothetical_missing_iou': endpoint,
                'valid_only': legacy._descriptive_metric(selected, 'aligned_occupancy_iou'),
                'bounded_failure0_companion': _equal_building((r, legacy.bounded_all_attempted_value(r)) for r in selected)})
        entry['endpoint_scenarios'] = endpoints
        result['group_bounds'].append(entry)
    return result


def analyze_slots(rows, *, audit_gate):
    """Run only after a verified source gate, or explicitly marked CPU fixtures."""
    if (not isinstance(audit_gate, dict) or audit_gate.get('status') != 'PASS'
            or audit_gate.get('input_kind') not in {'fixture', 'verified_sources'}):
        raise ValueError('passing source audit gate required before statistical analysis')
    rows = validate_rows(rows, fixed_design=audit_gate['input_kind'] == 'verified_sources')
    # No alteration of seed, replicates, families, Holm scope, or complete guard.
    summary = legacy.summarize_v4(rows, **INFERENCE, smoke=False)
    if [(f['id'], f['predeclared_size']) for f in summary['families']] != [
        (f['id'], f['size']) for f in legacy.comparison_families()]:
        raise ValueError('frozen comparison family mismatch')
    per_group, group_denominators = descriptive_groups(rows)
    comparisons = _comparison_records(rows, summary)
    pair_denominators = [{k: result[k] for k in ('family_id', 'id', 'metric', 'buildings', 'common_valid_pair_count',
                         'common_failure0_pair_count', 'common_failure0_buildings')} for result in comparisons]
    return {'analysis_label': ANALYSIS_LABEL, 'analysis_version': 'goal007_frozen_v4_statistics_v1',
            'input_kind': audit_gate['input_kind'], 'independent_unit': 'building', 'inference': INFERENCE.copy(),
            'summary': summary, 'comparisons': comparisons, 'per_group': per_group,
            'description_provider_role': 'saved image-to-text description model; distinct from Builder provider',
            'denominators': {'overall': _counts(rows), 'groups': group_denominators, 'comparisons': pair_denominators,
                'attempt_history_counts': 'Reported separately from the verified attempt audit; never inferred by counting final slots.'},
            'sensitivity_missing942': sensitivity_missing942(rows), 'research_api_requests_sent': 0}
