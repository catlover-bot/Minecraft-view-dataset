"""Separate post-hoc metric denominators and planned-building paired summaries.

This module accepts source-bound scalar rows only. It performs no voxel, model,
credential, network or file I/O, and never modifies historical metric objects.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

import numpy as np

LABEL = 'POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC'
PROTOCOL = 'goal010v2_metric_building_statistics_v1'
SEED = 20261017
REPLICATES = 2000
MIN_VALID_DRAWS = 1900
BUILDERS = ('openai', 'anthropic')
DESCRIPTIONS = ('openai', 'anthropic', 'gemini')
CONDITIONS = ('O', 'J', 'P', 'E')
FRAMES = ('absolute', 'aligned')
STRUCTURES = ('filled_box', 'six_face_shell')
DEFINITIVE_FAILURES = frozenset(('parse_failure', 'validation_failure', 'execution_failure', 'truncation'))
KNOWN_STATUSES = frozenset(('success', 'failed', 'operational_unmeasurable', 'not_run', 'NOT_RUN',
                            'ALIGNMENT_RESOURCE_LIMIT', 'GT_MISSING', 'BASELINE_INPUT_OR_RESOURCE_REJECTED'))
CORE_GEOMETRY = frozenset(('occupancy_precision', 'occupancy_recall', 'occupancy_iou',
                          'surface_f1_at_1', 'surface_f1_at_2', 'footprint_iou'))
BINDING_FIELDS = ('input_id', 'input_sha256', 'gt_grid_sha256', 'coordinate_system', 'alignment_protocol')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _gate(gate, specification_sha256):
    if (not isinstance(gate, dict) or gate.get('status') != 'PASS'
            or gate.get('input_kind') not in ('fixture', 'verified_sources')
            or gate.get('phase0_binding') != 'PASS' or gate.get('definitions_frozen') is not True
            or not re.fullmatch('[0-9a-f]{64}', specification_sha256 or '')
            or gate.get('specification_sha256') != specification_sha256):
        raise ValueError('passing Phase0 and frozen specification binding required')
    scenes = gate.get('planned_scene_ids')
    if (not isinstance(scenes, list) or not scenes or any(not isinstance(s, str) or not s for s in scenes)
            or scenes != sorted(set(scenes)) or len(scenes) > 60
            or gate['input_kind'] == 'verified_sources' and len(scenes) != 60):
        raise ValueError('sorted unique planned building frame required')
    return scenes


def _registry(registry):
    entries = registry.get('metrics') if isinstance(registry, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError('nonempty frozen metric registry required')
    result, seen = {}, set()
    for entry in entries:
        name = entry.get('metric_id')
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError('unique metric IDs required')
        seen.add(name)
        if entry.get('section') != 'frame':
            if entry.get('section') != 'displacement': raise ValueError('unknown registry section')
            continue
        interval = entry.get('value_range')
        if (entry.get('direction') not in ('higher', 'lower', 'descriptive')
                or type(entry.get('failure0_allowed')) is not bool
                or type(entry.get('baseline_comparable')) is not bool
                or not isinstance(interval, list) or len(interval) != 2):
            raise ValueError('explicit metric bounds, direction and eligibility required')
        if entry['failure0_allowed'] and (interval != [0, 1] or entry['direction'] != 'higher'):
            raise ValueError('failure0 allowed only for bounded [0,1] higher-is-better metrics')
        if entry['baseline_comparable'] and 'material' in name:
            raise ValueError('artificial baseline material comparisons forbidden')
        if name == 'conditional_exact_material_accuracy' and entry['failure0_allowed']:
            raise ValueError('undefined conditional material accuracy must not be failure0')
        result[name] = entry
    if not result or not CORE_GEOMETRY <= set(result) or any(not result[k]['baseline_comparable'] for k in CORE_GEOMETRY):
        raise ValueError('complete fixed core geometry metrics required')
    return result


def _number(value, entry):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('metric must be a finite JSON number or explicit null')
    lo, hi = entry['value_range']
    if lo is not None and value < lo or hi is not None and value > hi:
        raise ValueError('metric outside frozen range: ' + entry['metric_id'])


def _row(row, registry, *, baseline):
    if (row.get('condition') not in CONDITIONS or not isinstance(row.get('scene_id'), str)
            or row.get('status') not in KNOWN_STATUSES):
        raise ValueError('explicit scene, condition and outcome status required')
    description = row.get('description_model')
    if (row['condition'] in ('O', 'J') and description not in DESCRIPTIONS
            or row['condition'] in ('P', 'E') and description is not None):
        raise ValueError('description provider belongs only to O/J')
    for field in BINDING_FIELDS:
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError('source/input/GT/coordinate/protocol binding required: ' + field)
    for field in ('input_sha256', 'gt_grid_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', row[field]): raise ValueError('typed SHA256 required')
    if row['coordinate_system'] != 'XYZ_WORLD_Y_UP_INTEGER' or row['alignment_protocol'] != 'server_v2.2_strict_translation':
        raise ValueError('fixed world axes and strict translation protocol required')
    if baseline:
        if row['condition'] not in ('O', 'P') or row.get('structure') not in STRUCTURES:
            raise ValueError('only two frozen O/P baseline structures permitted')
        if not isinstance(row.get('prediction_id'), str) or not row['prediction_id']:
            raise ValueError('baseline prediction identity required')
        if row.get('failure0_eligible', False) is not False:
            raise ValueError('baseline failures cannot become model failure0')
    else:
        if row.get('provider') not in BUILDERS or type(row.get('global_schedule_index')) is not int:
            raise ValueError('unique selected Builder slot identity required')
        if type(row.get('failure0_eligible')) is not bool:
            raise ValueError('explicit failure0 eligibility required')
        if type(row.get('usage_known')) is not bool or type(row.get('completed_request_outcome')) is not bool:
            raise ValueError('identified usage and completion booleans required')
        http = row.get('http_status_code', row.get('http_status'))
        if 'http_status_code' in row and 'http_status' in row and row['http_status_code'] != row['http_status']:
            raise ValueError('HTTP status aliases disagree')
        expected = (row['status'] == 'failed' and row.get('failure_kind') in DEFINITIVE_FAILURES
                    and row['usage_known'] and row['completed_request_outcome'] and http not in (401, 520))
        if row['failure0_eligible'] != expected:
            raise ValueError('failure0 flag conflicts with identified completed output failure')
        if row['status'] == 'success' and not (row['usage_known'] and row['completed_request_outcome']):
            raise ValueError('success requires known completed model output')
        if row['status'] == 'operational_unmeasurable' and (row['usage_known'] or row['completed_request_outcome']):
            raise ValueError('operational missing cannot be a known completed output')
    metrics = row.get('metrics'); reasons = row.get('null_reasons')
    if not isinstance(metrics, dict) or set(metrics) != set(FRAMES) or not isinstance(reasons, dict):
        raise ValueError('both metric frames and scalar null reasons required')
    for frame in FRAMES:
        if not isinstance(metrics[frame], dict) or set(metrics[frame]) != set(registry):
            raise ValueError('every registry frame metric must remain explicit, including nulls')
        if not isinstance(reasons.get(frame), dict): raise ValueError('frame null reasons required')
        for name, entry in registry.items():
            value = metrics[frame][name]
            if value is None:
                if not isinstance(reasons[frame].get(name), str) or not reasons[frame][name]:
                    raise ValueError('null metric requires explicit reason')
            else:
                _number(value, entry)
                if row['status'] != 'success': raise ValueError('non-success geometry must remain null')
                if name in reasons[frame] and reasons[frame][name] is not None:
                    raise ValueError('measured scalar cannot simultaneously have a null reason')
    return row


def validate_inputs(metric_rows, baseline_rows, registry, scenes, *, fixed):
    slots, baselines, indices, prediction_ids = {}, {}, set(), set()
    for row in metric_rows:
        _row(row, registry, baseline=False)
        key = row['scene_id'], row['condition'], row.get('description_model'), row['provider']
        index = row['global_schedule_index']
        if key in slots or index in indices or index < 0 or row['scene_id'] not in scenes:
            raise ValueError('duplicate, unexpected or negative selected slot')
        slots[key] = row; indices.add(index)
    expected = {(scene, condition, description, builder) for scene in scenes for builder in BUILDERS
                for condition in CONDITIONS for description in (DESCRIPTIONS if condition in ('O', 'J') else (None,))}
    if set(slots) != expected:
        raise ValueError('complete fixed16 slots per building required, preserving explicit missing rows')
    for row in baseline_rows:
        _row(row, registry, baseline=True)
        key = row['scene_id'], row['condition'], row.get('description_model'), row['structure']
        if key in baselines or row['prediction_id'] in prediction_ids or row['scene_id'] not in scenes:
            raise ValueError('duplicate or unexpected fixed baseline prediction')
        baselines[key] = row; prediction_ids.add(row['prediction_id'])
        for builder in BUILDERS:
            slot = slots[row['scene_id'], row['condition'], row.get('description_model'), builder]
            if any(row[field] != slot[field] for field in BINDING_FIELDS):
                raise ValueError('baseline/Builder input, GT or coordinate/shift protocol mismatch')
    expected_baselines = {(scene, condition, description, structure) for scene in scenes for structure in STRUCTURES
                          for condition in ('O', 'P') for description in (DESCRIPTIONS if condition == 'O' else (None,))}
    if set(baselines) != expected_baselines:
        raise ValueError('complete fixed eight baseline predictions per building required')
    # All observations of a building must reference the same immutable GT.
    for scene in scenes:
        if len({r['gt_grid_sha256'] for r in metric_rows if r['scene_id'] == scene}) != 1:
            raise ValueError('GT identity changed within building')
    if fixed:
        if (len(metric_rows) != 960 or len(baseline_rows) != 480 or indices != set(range(960))
                or Counter(r['status'] for r in metric_rows) != {'success': 934, 'failed': 25, 'operational_unmeasurable': 1}
                or Counter(r.get('failure_kind') for r in metric_rows if r['status'] == 'failed') != {'truncation': 19, 'validation_failure': 6}):
            raise ValueError('frozen960/934/25/1 status and failure taxonomy changed')
        missing = next(r for r in metric_rows if r['status'] == 'operational_unmeasurable')
        if missing['global_schedule_index'] != 942 or missing['condition'] != 'J' or missing['provider'] != 'openai' or missing['failure0_eligible']:
            raise ValueError('global942 operational missing identity changed')
    return slots, baselines


def draw_counts(size):
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = rng.integers(0, size, size=(REPLICATES, size))
    counts = np.zeros((REPLICATES, size), dtype=np.int64)
    np.add.at(counts, (np.arange(REPLICATES)[:, None], draws), 1)
    return counts


def building_interval(values, scenes, counts=None):
    if not set(values) <= set(scenes) or any(not math.isfinite(v) for v in values.values()):
        raise ValueError('finite values on the planned building frame required')
    present = np.array([s in values for s in scenes], dtype=bool)
    array = np.array([values.get(s, 0.) for s in scenes], dtype=float)
    result = {'mean_delta': float(array[present].mean()) if present.any() else None,
        'ci95': None, 'independent_buildings': int(present.sum()), 'planned_buildings': len(scenes),
        'bootstrap_planned': REPLICATES, 'bootstrap_attempted': 0, 'bootstrap_valid': 0,
        'bootstrap_invalid': 0, 'status': 'INSUFFICIENT_BUILDINGS'}
    if present.sum() < 2: return result
    counts = draw_counts(len(scenes)) if counts is None else counts
    denominator = counts[:, present].sum(axis=1)
    finite = denominator > 0
    samples = (counts[:, present] * array[present]).sum(axis=1)[finite] / denominator[finite]
    samples = samples[np.isfinite(samples)]
    result.update(bootstrap_attempted=REPLICATES, bootstrap_valid=len(samples), bootstrap_invalid=REPLICATES-len(samples))
    if len(samples) < MIN_VALID_DRAWS: return dict(result, status='ESTIMATE_WITHOUT_STABLE_CI')
    return dict(result, status='ESTIMABLE', ci95=np.quantile(samples, [.025, .975], method='linear').tolist())


def _value(row, frame, name, entry, policy):
    value = row['metrics'][frame][name]
    if value is not None: return float(value), False, None
    if policy == 'known_failure0' and entry['failure0_allowed'] and row.get('failure0_eligible') is True:
        return 0., True, None
    return None, False, row['null_reasons'][frame][name]


def _policies(entry):
    return ('common_valid', 'known_failure0') if entry['failure0_allowed'] else ('common_valid',)


def _building_mean(pairs, key):
    by_scene = defaultdict(list)
    for row in pairs:
        if row[key] is not None: by_scene[row['scene_id']].append(row[key])
    return {scene: float(np.mean(values)) for scene, values in by_scene.items()}


def _ordinal(delta, direction):
    if delta is None: return 'NOT_ESTIMABLE'
    if direction == 'descriptive': return 'NO_FAVORABLE_ORDER'
    signed = delta if direction == 'higher' else -delta
    return 'BUILDER_HIGHER_FAVORABLE' if signed > 0 else 'BASELINE_HIGHER_FAVORABLE' if signed < 0 else 'NUMERIC_TIE'


def _evidence(row):
    """Compact exact row reference plus stored per-prediction hash/shift proof."""
    engine = row.get('metric_provenance') or row.get('provenance') or {}
    return {'scalar_row_sha256': digest(row),
        **{k: row.get(k) for k in ('attempt_id', 'source_label', 'source_plan_identity', 'source_entry_hash',
            'gt_voxels_sha256', 'gt_bbox_sha256', 'original_metric_object_sha256')},
        **{k: engine.get(k) for k in ('gt_grid_sha256', 'pred_grid_sha256', 'shift_sha256',
            'inputs_and_shift_sha256', 'implementation_sha256', 'stored_alignment_shift_xyz')}}


def paired_statistics(slots, baselines, registry, scenes):
    pairs, grouped = [], defaultdict(list)
    geometry = {k: v for k, v in registry.items() if v['baseline_comparable']}
    counts = draw_counts(len(scenes))
    evidence = {id(row): _evidence(row) for row in (*slots.values(), *baselines.values())}
    for key in sorted(baselines, key=lambda x: tuple('' if v is None else v for v in x)):
        base = baselines[key]
        for builder in BUILDERS:
            row = slots[base['scene_id'], base['condition'], base.get('description_model'), builder]
            for frame in FRAMES:
                for name, entry in geometry.items():
                    for policy in _policies(entry):
                        value, imputed, reason = _value(row, frame, name, entry, policy)
                        bv, _, br = _value(base, frame, name, entry, 'common_valid')
                        eligible = value is not None and bv is not None
                        exclusion = ('both_unmeasured' if value is None and bv is None else 'builder_unmeasured' if value is None else 'baseline_unmeasured') if not eligible else None
                        pair = {**{k:base[k] for k in ('prediction_id', 'scene_id', 'condition', 'description_model', 'structure', *BINDING_FIELDS)},
                            'builder': builder, 'global_schedule_index': row['global_schedule_index'],
                            'frame': frame, 'metric_id': name, 'policy': policy,
                            'builder_status': row['status'], 'baseline_status': base['status'],
                            'builder_failure_kind': row.get('failure_kind'), 'failure0_applied': imputed,
                            'builder_null_reason': reason, 'baseline_null_reason': br, 'exclusion_reason': exclusion,
                            'builder_value': value, 'baseline_value': bv, 'delta': value-bv if eligible else None,
                            'builder_provenance': evidence[id(row)], 'baseline_provenance': evidence[id(base)]}
                        pairs.append(pair)
                        for scope in ('all', base['description_model']) if base['condition'] == 'O' else ('all',):
                            grouped[base['condition'], scope, base['structure'], builder, frame, name, policy].append(pair)
    summaries = []
    for key, selected in grouped.items():
        condition, scope, structure, builder, frame, name, policy = key
        eligible = [r for r in selected if r['delta'] is not None]
        interval = building_interval(_building_mean(eligible, 'delta'), scenes, counts)
        means = {name_: float(np.mean(list(_building_mean(eligible, name_).values()))) if eligible else None
                 for name_ in ('builder_value', 'baseline_value')}
        summaries.append({'id': ':'.join(key), 'analysis_label': LABEL, 'condition': condition,
            'description_provider': scope, 'structure': structure, 'builder': builder, 'frame': frame,
            'metric_id': name, 'priority': 'core_geometry' if name in CORE_GEOMETRY else 'additional_geometry',
            'policy': policy, 'direction': registry[name]['direction'], 'difference_direction': 'Builder minus baseline',
            'planned_pairs': len(selected), 'eligible_pairs': len(eligible), 'excluded_pairs': len(selected)-len(eligible),
            'failure0_imputed_pairs': sum(r['failure0_applied'] for r in eligible),
            'exclusion_counts': dict(Counter(r['exclusion_reason'] for r in selected if r['exclusion_reason'])),
            'builder_null_reason_counts': dict(Counter(r['builder_null_reason'] for r in selected if r['builder_null_reason'])),
            'baseline_null_reason_counts': dict(Counter(r['baseline_null_reason'] for r in selected if r['baseline_null_reason'])),
            'estimator': 'equal-building mean after averaging eligible description-provider pairs within each building',
            'bootstrap_frame': 'all planned buildings, including buildings without eligible pairs',
            'ordinal_preference': _ordinal(interval['mean_delta'], registry[name]['direction']),
            **means, **interval, 'p_values_computed': False})
    expected = 40 * sum(len(_policies(e)) for e in geometry.values())
    if len(summaries) != expected: raise AssertionError('incomplete paired summary grid')
    return summaries, pairs, grouped


def reliability(rows):
    result = []
    for builder in BUILDERS:
        for condition in CONDITIONS:
            selected = [r for r in rows if r['provider'] == builder and r['condition'] == condition]
            statuses = Counter(r['status'] for r in selected)
            failure_types = Counter(r.get('failure_kind') or 'UNSPECIFIED' for r in selected if r['status'] == 'failed')
            result.append({'builder': builder, 'condition': condition, 'planned_slots': len(selected),
                'independent_buildings': len({r['scene_id'] for r in selected}), 'valid_outputs': statuses['success'],
                'identified_output_failures': sum(r['failure0_eligible'] for r in selected),
                'failure_kind_counts': dict(failure_types), 'operational_unknown': statuses['operational_unmeasurable'],
                'not_run': statuses['not_run'] + statuses['NOT_RUN'], 'status_counts': dict(statuses),
                'valid_outputs_with_any_null_metric': sum(r['status'] == 'success' and any(
                    v is None for frame in FRAMES for v in r['metrics'][frame].values()) for r in selected),
                'metric_status_counts': dict(Counter(r.get('metric_status', 'NOT_SEPARATELY_RECORDED') for r in selected)),
                'valid_fraction': statuses['success']/len(selected) if selected else None,
                'success_fraction_not_multiplied_into_metric_scores': True})
    return result


def metric_denominators(rows, registry):
    result = []
    for builder in BUILDERS:
        for condition in CONDITIONS:
            scopes = ('all', *DESCRIPTIONS) if condition in ('O', 'J') else ('all',)
            for scope in scopes:
                selected = [r for r in rows if r['provider'] == builder and r['condition'] == condition
                            and (scope == 'all' or r.get('description_model') == scope)]
                for frame in FRAMES:
                    for name, entry in registry.items():
                        for policy in _policies(entry):
                            measured, imputed, reasons = [], 0, []
                            for row in selected:
                                value, zero, reason = _value(row, frame, name, entry, policy)
                                if value is None: reasons.append(reason)
                                else: measured.append({'scene_id': row['scene_id'], 'value': value}); imputed += zero
                            by_building = _building_mean(measured, 'value')
                            result.append({'builder': builder, 'condition': condition, 'description_provider': scope,
                                'frame': frame, 'metric_id': name, 'policy': policy, 'direction': entry['direction'],
                                'planned_slots': len(selected), 'planned_buildings': len({r['scene_id'] for r in selected}),
                                'eligible_slots': len(measured), 'excluded_slots': len(reasons),
                                'independent_buildings': len(by_building), 'failure0_imputed_slots': imputed,
                                'null_reason_counts': dict(Counter(reasons)),
                                'mean': float(np.mean(list(by_building.values()))) if by_building else None,
                                'estimator': 'equal-building mean after eligible descriptions within building',
                                'p_values_computed': False})
    return result


def rank_reversals(grouped, registry):
    """Descriptive order changes on identical eligible pairs, not extra tests."""
    output = []
    for key, selected in grouped.items():
        condition, scope, structure, builder, frame, name, policy = key
        if name == 'occupancy_iou' or policy != 'common_valid': continue
        reference = grouped[condition, scope, structure, builder, frame, 'occupancy_iou', policy]
        index = {r['global_schedule_index']: r for r in reference if r['delta'] is not None}
        common = [r for r in selected if r['delta'] is not None and r['global_schedule_index'] in index]
        iou_pairs = [index[r['global_schedule_index']] for r in common]
        means = {label: float(np.mean(list(_building_mean(values, 'delta').values()))) if values else None
                 for label, values in (('reference_iou_delta', iou_pairs), ('metric_delta', common))}
        iou_order = _ordinal(means['reference_iou_delta'], 'higher')
        metric_order = _ordinal(means['metric_delta'], registry[name]['direction'])
        favorable = {'BUILDER_HIGHER_FAVORABLE', 'BASELINE_HIGHER_FAVORABLE'}
        reversal = iou_order != metric_order if iou_order in favorable and metric_order in favorable else None
        output.append({'condition': condition, 'description_provider': scope, 'structure': structure, 'builder': builder,
            'frame': frame, 'metric_id': name, 'policy': policy, 'reference_metric': 'occupancy_iou',
            'common_pairs': len(common), 'independent_buildings': len({r['scene_id'] for r in common}),
            'same_pair_set_for_both_metrics': True,
            'pair_set_sha256': digest(sorted(r['global_schedule_index'] for r in common)),
            **means, 'reference_ordinal_preference': iou_order, 'metric_ordinal_preference': metric_order,
            'directional_reversal': reversal, 'new_inference_test': False,
            'interpretation': 'descriptive metric-dependent ordering only; no causal or significance claim'})
    return output


def summarize(metric_rows, baseline_rows, *, gate, specification_sha256, registry):
    scenes = _gate(gate, specification_sha256)
    registered = _registry(registry)
    before = digest([metric_rows, baseline_rows])
    slots, baselines = validate_inputs(metric_rows, baseline_rows, registered, scenes,
                                      fixed=gate['input_kind'] == 'verified_sources')
    summary, pairs, grouped = paired_statistics(slots, baselines, registered, scenes)
    result = {'paired_summary': summary, 'paired_rows': pairs,
        'exclusions': [r for r in pairs if r['exclusion_reason'] is not None],
        'metric_denominators': metric_denominators(metric_rows, registered),
        'reliability': reliability(metric_rows), 'rank_reversals': rank_reversals(grouped, registered),
        'metadata': {'version': PROTOCOL, 'analysis_label': LABEL, 'input_kind': gate['input_kind'],
            'specification_sha256': specification_sha256, 'metric_registry_sha256': digest(registry),
            'input_scalar_rows_sha256': before, 'planned_buildings': len(scenes), 'planned_scene_ids': scenes,
            'metric_rows': len(metric_rows), 'baseline_rows': len(baseline_rows),
            'paired_summary_cells': len(summary), 'candidate_metric_pairs': len(pairs),
            'geometry_metric_ids': [k for k, v in registered.items() if v['baseline_comparable']],
            'core_geometry_metric_ids': sorted(CORE_GEOMETRY),
            'failure0_metric_ids': [k for k, v in registered.items() if v['failure0_allowed']],
            'inference': {'independent_unit': 'building', 'seed': SEED, 'generator': 'NumPy PCG64',
                'bootstrap_replicates': REPLICATES, 'minimum_valid_draws': MIN_VALID_DRAWS,
                'percentile_interval': [.025, .975], 'quantile_method': 'linear/type7',
                'frame': 'all planned buildings; missing buildings retained in resampling',
                'p_values': False, 'holm_family_changes': False},
            'known_failure0_is_separate_auxiliary_policy': True,
            'normal_empty_or_undefined_success_metrics_never_imputed': True,
            'baseline_artificial_material_comparisons': False,
            'rank_reversal_is_descriptive_on_intersected_pairs': True,
            'old_scores_or_families_changed': False, 'research_api_requests_sent': 0, 'api_key_queries': 0}}
    if digest([metric_rows, baseline_rows]) != before: raise AssertionError('input scalar rows mutated')
    return result
