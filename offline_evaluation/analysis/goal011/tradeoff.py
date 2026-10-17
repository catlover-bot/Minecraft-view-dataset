"""Finite saved-metric audit; no geometry, network, bootstrap or file writes."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import math
import re
from statistics import stdev

from scipy.stats import t

LABEL = 'POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC'
CORE = ('occupancy_iou', 'occupancy_precision', 'occupancy_recall', 'surface_f1_at_1')
METRICS = CORE + ('surface_f1_at_2', 'footprint_iou')
BUILDERS = ('openai', 'anthropic')
PROVIDERS = ('openai', 'anthropic', 'gemini')
STRUCTURES = ('filled_box', 'six_face_shell')
FRAMES = ('absolute', 'aligned')
POLICIES = ('common_valid', 'known_failure0')
COORDINATES = 'XYZ_WORLD_Y_UP_INTEGER'
ALIGNMENT = 'server_v2.2_strict_translation'
TOLERANCE = 1e-12


class AuditFailure(ValueError):
    """A controlled, exportable proof; the caller must halt without correction."""
    def __init__(self, code, evidence=None):
        self.report = {'status': 'FAIL', 'critical': True, 'code': code,
                       'evidence': evidence or {}, 'historical_changes_permitted': False,
                       'correction_proposal': 'Inspect the bounded discrepancy; preserve all historical artifacts.'}
        super().__init__(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _fail(code, **evidence):
    raise AuditFailure(code, evidence)


def _gate(gate, plan_sha256):
    if (not isinstance(gate, dict) or gate.get('status') != 'PASS'
            or gate.get('input_kind') not in ('fixture', 'verified_sources')
            or gate.get('plan_sha256') != plan_sha256
            or not re.fullmatch('[0-9a-f]{64}', plan_sha256 or '')):
        _fail('PASSING_FROZEN_PLAN_GATE_REQUIRED')
    for key, length in [('question_register_sha256', 64), ('source_package_sha256', 64),
                        ('code_commit', 40), ('specification_commit', 40)]:
        if not re.fullmatch('[0-9a-f]{'+str(length)+'}', str(gate.get(key, ''))):
            _fail('INCOMPLETE_FROZEN_GATE', field=key)


def _index(rows, keys, code):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            _fail(code, key=list(key))
        result[key] = row
    return result


def _close(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return (type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=TOLERANCE))


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def _equal_building(pairs):
    groups = defaultdict(list)
    for row in pairs:
        groups[row['scene_id']].append(row)
    means = {s: {k: _mean([r[k] for r in rs])
                 for k in ('builder_value', 'baseline_value', 'delta')}
             for s, rs in sorted(groups.items())}
    return {'builder_value': _mean([r['builder_value'] for r in means.values()]),
            'baseline_value': _mean([r['baseline_value'] for r in means.values()]),
            'mean_delta': _mean([r['delta'] for r in means.values()]),
            'independent_buildings': len(means), 'eligible_pairs': len(pairs)}, means


def _source_value(row, frame, metric, policy):
    value = row['metrics'][frame][metric]
    if value is not None:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            _fail('INVALID_BOUNDED_METRIC', metric=metric)
        if row['status'] != 'success':
            _fail('NONSUCCESS_HAS_MEASURED_METRIC')
        return float(value), False
    if not row['null_reasons'][frame].get(metric):
        _fail('NULL_WITHOUT_REASON', metric=metric)
    if policy == 'known_failure0' and row.get('failure0_eligible') is True:
        return 0., True
    return None, False


def _json_input(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail('DUPLICATE_INPUT_JSON_KEY')
            result[key] = value
        return result
    return json.loads(text, parse_float=Decimal, object_pairs_hook=pairs,
                      parse_constant=lambda value: _fail('NONFINITE_INPUT_JSON'))


def _numeric_tree(value):
    if type(value) in (int, float, Decimal):
        number = Decimal(str(value))
        if not number.is_finite():
            _fail('NONFINITE_BASELINE_INPUT')
        return ('number', number)
    if isinstance(value, dict):
        return {k: _numeric_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_numeric_tree(v) for v in value]
    return value


def _fixed_shift_check(row, original_metrics, *, identity):
    """Bind scalar frames to the original occupancy-optimal shift, without searching."""
    if not isinstance(original_metrics, dict) or original_metrics.get('metric_version') != ALIGNMENT:
        _fail('ORIGINAL_ALIGNMENT_VERSION_MISMATCH', identity=identity)
    certificate = original_metrics.get('alignment', {})
    shift = certificate.get('shift_xyz')
    if (not isinstance(shift, list) or len(shift) != 3 or any(type(v) is not int for v in shift)
            or certificate.get('mode') != 'strict_translation'
            or any(certificate.get(k) is not True for k in
                   ('search_complete', 'globally_optimal_unrestricted', 'all_aligned_metrics_share_shift'))
            or any(certificate.get(k) is not False for k in
                   ('rotation', 'scale', 'clipping', 'resource_limit_reached'))):
        _fail('ORIGINAL_STRICT_ALIGNMENT_CERTIFICATE_INVALID', identity=identity)
    provenance = row.get('metric_provenance') or {}
    if (row.get('shift_xyz') != shift or (row.get('displacement') or {}).get('shift_xyz') != shift
            or provenance.get('stored_alignment_shift_xyz') != shift
            or provenance.get('alignment_protocol') != ALIGNMENT
            or provenance.get('alignment_searched') is not False
            or provenance.get('failure0_applied') is not False):
        _fail('SAVED_OCCUPANCY_SHIFT_NOT_PRESERVED', identity=identity)
    return list(shift)


def audit_construction(data):
    """Verify saved input-only box construction through an independent formula."""
    manifest = data['baseline_manifest']
    if (manifest.get('status') != 'ALL_PREDICTIONS_FROZEN_BEFORE_GT_EVALUATION'
            or manifest.get('GT_voxels_read_during_construction') is not False
            or manifest.get('Builder_replication_of_predictions') is not False):
        _fail('BASELINE_CONSTRUCTION_MANIFEST_INVALID')
    inputs = {}
    for row in data['input_records']:
        if row['condition'] not in ('O', 'P'):
            continue
        key = row['scene_id'], row['condition'], row.get('description_model')
        item = {k: row.get(k) for k in ('input_id', 'input_sha256', 'input_text')}
        if hashlib.sha256(item['input_text'].encode()).hexdigest() != item['input_sha256']:
            _fail('ORIGINAL_INPUT_TEXT_HASH_MISMATCH', input_id=item['input_id'])
        if key in inputs and item != inputs[key]:
            _fail('TWO_BUILDERS_DIFFERENT_BASELINE_INPUT', scene_id=key[0])
        inputs[key] = item
    predictions = _index(manifest['predictions'], ('prediction_id',), 'DUPLICATE_FROZEN_PREDICTION')
    evaluations = _index(data['baseline_evaluations'], ('prediction_id',), 'DUPLICATE_OLD_BASELINE_EVALUATION')
    rows = []
    for baseline in data['baseline_rows']:
        pid = baseline['prediction_id']
        if (pid,) not in predictions or (pid,) not in evaluations:
            _fail('MISSING_FROZEN_BASELINE_REFERENCE', prediction_id=pid)
        saved = predictions[pid,]; old = evaluations[pid,]
        key = baseline['scene_id'], baseline['condition'], baseline.get('description_model')
        source = inputs.get(key)
        if source is None:
            _fail('MISSING_BASELINE_INPUT', prediction_id=pid)
        for item in (saved, old, baseline):
            for field in ('scene_id', 'condition', 'description_model', 'input_id', 'input_sha256', 'structure'):
                if item.get(field) != baseline.get(field):
                    _fail('BASELINE_REFERENCE_JOIN_MISMATCH', prediction_id=pid, field=field)
        if any(baseline[k] != source[k] for k in ('input_id', 'input_sha256')):
            _fail('BASELINE_ORIGINAL_INPUT_JOIN_MISMATCH', prediction_id=pid)
        payload = _json_input(source['input_text'])
        if baseline['condition'] == 'O':
            permitted = payload.get('dimensions_estimate')
            if not isinstance(permitted, dict):
                _fail('O_DIMENSIONS_MISSING')
            dimensions = []
            for name in ('width', 'height', 'depth'):
                v = permitted.get(name)
                if type(v) not in (int, Decimal) or v <= 0 or v > 100001:
                    _fail('O_DIMENSIONS_INVALID')
                dimensions.append(max(1, round(v)))
            bbox = {a+end: 0 if end == 'min' else dimensions[i]-1
                    for i, a in enumerate('xyz') for end in ('min', 'max')}
            if saved.get('source_sha256') != source['input_sha256']:
                _fail('O_DESCRIPTION_SOURCE_HASH_MISMATCH')
        else:
            permitted = payload.get('world_bbox')
            if not isinstance(permitted, dict):
                _fail('P_DISCLOSED_BBOX_MISSING')
            bbox = {a+end: permitted.get(a+end) for a in 'xyz' for end in ('min', 'max')}
            if any(type(v) is not int or abs(v) > 100000 for v in bbox.values()):
                _fail('P_DISCLOSED_BBOX_INVALID')
            dimensions = [bbox[a+'max']-bbox[a+'min']+1 for a in 'xyz']
            if min(dimensions) <= 0:
                _fail('P_DISCLOSED_BBOX_REVERSED')
        if (_numeric_tree(saved['permitted_input']) != _numeric_tree(permitted)
                or _numeric_tree(old['permitted_input']) != _numeric_tree(permitted)):
            _fail('BASELINE_CONSTRUCTION_USED_DIFFERENT_INPUT', prediction_id=pid)
        count = math.prod(dimensions)
        if baseline['structure'] == 'six_face_shell':
            count -= math.prod(max(d-2, 0) for d in dimensions)
        elif baseline['structure'] != 'filled_box':
            _fail('UNPLANNED_BASELINE_STRUCTURE')
        for item in (saved, old):
            if (item.get('prediction_bbox') != bbox or item.get('prediction_dimensions_xyz') != dimensions
                    or item.get('prediction_voxels') != count
                    or item.get('material_metrics_used') is not False or item.get('artificial_material') != 'stone'):
                _fail('BASELINE_SAVED_CONSTRUCTION_MISMATCH', prediction_id=pid)
        if saved.get('status') != 'PREDICTION_SAVED' or old.get('status') != 'success' or baseline.get('status') != 'success':
            _fail('BASELINE_UNEXPECTED_STATUS', prediction_id=pid)
        if old.get('independent_check_status') != 'PASS' or old.get('gt_grid_sha256') != baseline.get('gt_grid_sha256'):
            _fail('BASELINE_GT_OR_PRIOR_PROOF_MISMATCH', prediction_id=pid)
        shift = _fixed_shift_check(baseline, {'metric_version': old.get('metric_version'),
                                             'alignment': old.get('alignment')}, identity=pid)
        if (old.get('independent_check') or {}).get('shift_xyz') != shift:
            _fail('BASELINE_INDEPENDENT_SHIFT_MISMATCH', prediction_id=pid)
        for frame in FRAMES:
            if (baseline['metrics'][frame].get('pred_occupied_count') != count
                    or not _close(baseline['metrics'][frame]['occupancy_iou'], old[frame+'_occupancy_iou'])):
                _fail('BASELINE_NEW_VS_OLD_METRIC_MISMATCH', prediction_id=pid, frame=frame)
        rows.append({'prediction_id': pid, 'condition': baseline['condition'],
                     'input_id': baseline['input_id'], 'input_sha256': baseline['input_sha256'],
                     'gt_grid_sha256': baseline['gt_grid_sha256'], 'construction': 'PASS',
                     'origin_rule': 'FIXED_ZERO' if baseline['condition'] == 'O' else 'SUPPLIED_P_WORLD_BBOX',
                     'original_occupancy_shift_xyz': shift, 'fixed_shift_check': 'PASS',
                     'expected_occupied_count': count, 'frozen_prediction_sha256': digest(saved),
                     'old_evaluation_sha256': digest(old), 'new_scalar_sha256': digest(baseline)})
    if len(predictions) != len(rows) or len(evaluations) != len(rows):
        _fail('BASELINE_REFERENCE_COVERAGE_MISMATCH')
    return rows


def _validate_rows(data, gate):
    rows = data['metric_rows']; bases = data['baseline_rows']
    cohort = _index(data['cohort_rows'], ('scene_id',), 'DUPLICATE_COHORT_SCENE')
    scenes = sorted(k[0] for k in cohort)
    if not scenes or len(scenes) > 60:
        _fail('INVALID_COHORT_SIZE')
    if any(type(r['size_rank_stratum']) is not int or r['size_rank_stratum'] not in range(5) for r in cohort.values()):
        _fail('FROZEN_FIVE_QUINTILES_REQUIRED')
    if set(r['size_rank_stratum'] for r in cohort.values()) != set(range(5)):
        _fail('MISSING_FROZEN_QUINTILE')
    index = _index(rows, ('scene_id', 'condition', 'description_model', 'provider'), 'DUPLICATE_SELECTED_SLOT')
    _index(rows, ('global_schedule_index',), 'DUPLICATE_GLOBAL_SLOT')
    original = _index(data['original_slots'], ('global_schedule_index',), 'DUPLICATE_ORIGINAL_SLOT')
    expected = {(s, c, d, b) for s in scenes for c in ('O', 'J', 'P', 'E')
                for d in (PROVIDERS if c in ('O', 'J') else (None,)) for b in BUILDERS}
    if set(index) != expected or len(original) != len(rows):
        _fail('SELECTED_FIXED_DESIGN_MISMATCH')
    for r in rows:
        original_row = original.get((r['global_schedule_index'],))
        if original_row is None or any(original_row.get(k) != r.get(k) for k in
                ('global_schedule_index', 'scene_id', 'condition', 'description_model', 'provider',
                 'input_id', 'input_sha256', 'status', 'failure_kind', 'usage_known', 'completed_request_outcome',
                 'attempt_id', 'original_attempt_id', 'selection_reason', 'http_status_code', 'empty_output',
                 'source_label', 'source_plan_identity', 'source_entry_hash', 'source_gt_descriptor_hash', 'gt_grid_sha256')):
            _fail('ORIGINAL_SELECTED_SLOT_CHANGED', global_schedule_index=r['global_schedule_index'])
        if r.get('original_metric_object_sha256') != digest(original_row.get('metrics')):
            _fail('ORIGINAL_METRIC_OBJECT_CHANGED', global_schedule_index=r['global_schedule_index'])
        if r['status'] == 'success':
            _fixed_shift_check(r, original_row.get('metrics'), identity=r['global_schedule_index'])
            if r.get('empty_output') is not (r['metrics']['aligned'].get('pred_occupied_count') == 0):
                _fail('NORMAL_EMPTY_CLASSIFICATION_CHANGED', global_schedule_index=r['global_schedule_index'])
        elif any(r.get(k) is not None for k in ('shift_xyz', 'displacement', 'metric_provenance')):
            _fail('NON_SUCCESS_HAS_GEOMETRIC_PROVENANCE', global_schedule_index=r['global_schedule_index'])
        failure0 = (r['status'] == 'failed' and r.get('failure_kind') in
                    ('parse_failure', 'validation_failure', 'execution_failure', 'truncation')
                    and r.get('usage_known') is True and r.get('completed_request_outcome') is True
                    and r.get('http_status_code') not in (401, 520))
        if r.get('failure0_eligible') is not failure0:
            _fail('FAILURE0_STATUS_MISMATCH', global_schedule_index=r['global_schedule_index'])
        for frame in FRAMES:
            for metric in METRICS:
                _source_value(r, frame, metric, 'common_valid')
    bindex = _index(bases, ('scene_id', 'condition', 'description_model', 'structure'), 'DUPLICATE_BASELINE_ROW')
    bexpected = {(s, c, d, shape) for s in scenes for c in ('O', 'P')
                 for d in (PROVIDERS if c == 'O' else (None,)) for shape in STRUCTURES}
    if set(bindex) != bexpected:
        _fail('BASELINE_FIXED_DESIGN_MISMATCH')
    for key, b in bindex.items():
        for builder in BUILDERS:
            r = index[key[:3]+(builder,)]
            for field in ('input_id', 'input_sha256', 'gt_grid_sha256', 'coordinate_system', 'alignment_protocol'):
                if r.get(field) != b.get(field) or not r.get(field):
                    _fail('BUILDER_BASELINE_BINDING_MISMATCH', field=field, scene_id=key[0])
            if b['coordinate_system'] != COORDINATES or b['alignment_protocol'] != ALIGNMENT:
                _fail('UNEXPECTED_COORDINATE_OR_ALIGNMENT_PROTOCOL')
    if gate['input_kind'] == 'verified_sources':
        counts = Counter(r['status'] for r in rows)
        if (len(scenes) != 60 or counts != {'success': 934, 'failed': 25, 'operational_unmeasurable': 1}
                or Counter(r['failure_kind'] for r in rows if r['status'] == 'failed') != {'validation_failure': 6, 'truncation': 19}
                or Counter(r['size_rank_stratum'] for r in cohort.values()) != {q: 12 for q in range(5)}
                or {r['global_schedule_index'] for r in rows} != set(range(960))):
            _fail('HISTORICAL_COUNTS_OR_QUINTILES_CHANGED')
        selected = original[940,]; missing = original[942,]
        if (selected['status'] != 'success' or selected.get('original_attempt_id') in (None, selected['attempt_id'])
                or selected.get('source_attempt_id') != selected['original_attempt_id']
                or selected.get('selection_reason') != 'approved_manual_auth_retry_after_401'):
            _fail('GLOBAL940_APPROVED_RETRY_PROVENANCE_CHANGED')
        if (missing['status'] != 'operational_unmeasurable' or missing.get('usage_known') is not False
                or missing.get('completed_request_outcome') is not False or missing.get('http_status_code') != 520
                or any(missing.get(k) is not None for k in ('metrics', 'usage', 'response'))
                or index[missing['scene_id'], missing['condition'], missing['description_model'], missing['provider']]['failure0_eligible']):
            _fail('GLOBAL942_MISSING_RECLASSIFIED')
        ref = data['integrity_reference']
        if (ref.get('historical_calls') != 991 or ref.get('fixed_scope_attempts') != 961
                or ref.get('unknown_usage_holds_released') is not False
                or len(ref.get('unknown_usage_holds', [])) != 2):
            _fail('HISTORICAL_CALLS_OR_HOLDS_CHANGED')
        hold_identity = {(r.get('global_schedule_index'), r.get('attempt_id'), r.get('http_status'))
                         for r in ref['unknown_usage_holds']}
        if hold_identity != {(940, selected['original_attempt_id'], 401),
                             (942, missing['attempt_id'], 520)}:
            _fail('UNKNOWN_USAGE_HOLD_IDENTITIES_CHANGED')
        if sum(r['status'] == 'success' and r.get('empty_output') is True for r in rows) != 1:
            _fail('NORMAL_EMPTY_SUCCESS_COUNT_CHANGED')
    return index, bindex, cohort


def _pair_rows(index, bases, cohort):
    result = []
    model_hashes = {r['global_schedule_index']: digest(r) for r in index.values()}
    base_hashes = {r['prediction_id']: digest(r) for r in bases.values()}
    for (scene, condition, provider, shape), base in sorted(bases.items(), key=str):
        if condition != 'O':
            continue
        for builder in BUILDERS:
            model = index[scene, condition, provider, builder]
            for frame in FRAMES:
                for metric in METRICS:
                    for policy in POLICIES:
                        mv, imputed = _source_value(model, frame, metric, policy)
                        bv, _ = _source_value(base, frame, metric, 'common_valid')
                        result.append({'scene_id': scene, 'size_rank_stratum': cohort[scene,]['size_rank_stratum'],
                            'condition': 'O', 'description_provider': provider, 'builder': builder, 'structure': shape,
                            'frame': frame, 'metric_id': metric, 'policy': policy,
                            'global_schedule_index': model['global_schedule_index'], 'prediction_id': base['prediction_id'],
                            'input_id': base['input_id'], 'input_sha256': base['input_sha256'],
                            'gt_grid_sha256': base['gt_grid_sha256'], 'builder_value': mv, 'baseline_value': bv,
                            'delta': mv-bv if mv is not None and bv is not None else None,
                            'failure0_imputed': imputed, 'builder_scalar_sha256': model_hashes[model['global_schedule_index']],
                            'baseline_scalar_sha256': base_hashes[base['prediction_id']]})
    return result


def _effect_audit(pairs, data, scenes):
    keys = ('condition', 'description_provider', 'structure', 'builder', 'frame', 'metric_id', 'policy')
    saved = _index(data['saved_summaries'], keys, 'DUPLICATE_SAVED_SUMMARY')
    old = _index(data['old_summaries'], ('condition', 'description_provider', 'structure', 'builder', 'metric', 'policy'),
                 'DUPLICATE_OLD_SUMMARY')
    effects = []
    for builder in BUILDERS:
        for shape in STRUCTURES:
            for frame in FRAMES:
                for scope in ('all',)+PROVIDERS:
                    for metric in METRICS:
                        for policy in POLICIES:
                            candidates = [p for p in pairs if p['builder'] == builder and p['structure'] == shape
                                and p['frame'] == frame and p['metric_id'] == metric and p['policy'] == policy
                                and (scope == 'all' or p['description_provider'] == scope)]
                            eligible = [p for p in candidates if p['delta'] is not None]
                            actual, _ = _equal_building(eligible)
                            key = ('O', scope, shape, builder, frame, metric, policy)
                            target = saved.get(key)
                            if target is None:
                                _fail('MISSING_SAVED_CORE_SUMMARY', key=list(key))
                            exact = {'eligible_pairs': len(eligible), 'independent_buildings': actual['independent_buildings'],
                                     'planned_pairs': len(candidates), 'planned_buildings': len(scenes),
                                     'excluded_pairs': len(candidates)-len(eligible),
                                     'failure0_imputed_pairs': sum(p['failure0_imputed'] for p in eligible)}
                            for name, v in exact.items():
                                if target.get(name) != v:
                                    _fail('SAVED_SUMMARY_DENOMINATOR_CHANGED', key=list(key), field=name,
                                          saved=target.get(name), independent=v)
                            for name in ('builder_value', 'baseline_value', 'mean_delta'):
                                if not _close(actual[name], target.get(name)):
                                    _fail('SAVED_SUMMARY_VALUE_CHANGED', key=list(key), field=name,
                                          saved=target.get(name), independent=actual[name])
                            old_iou = None
                            if metric == 'occupancy_iou':
                                previous = old.get(('O', scope, shape, builder, frame+'_occupancy_iou', policy))
                                if previous is None:
                                    _fail('MISSING_OLD_IOU_SUMMARY')
                                for name, v in [('common_pairs', exact['eligible_pairs']), ('independent_buildings', actual['independent_buildings'])]:
                                    if previous.get(name) != v:
                                        _fail('OLD_IOU_DENOMINATOR_CHANGED', field=name)
                                if (previous.get('ci95') != target.get('ci95')
                                        or not _close(previous.get('mean_delta'), target.get('mean_delta'))):
                                    _fail('OLD_IOU_EFFECT_OR_SAVED_CI_CHANGED', key=list(key))
                                old_iou = {'mean_delta': previous.get('mean_delta'),
                                    'mean_delta_bit_identical': previous.get('mean_delta') == target.get('mean_delta'),
                                    'saved_CI_bit_identical': True}
                            if target.get('ci95') is not None and (len(target['ci95']) != 2
                                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in target['ci95'])
                                    or target['ci95'][0] > target['ci95'][1]):
                                _fail('INVALID_SAVED_CI')
                            effects.append({**target, 'saved_summary_sha256': digest(target), 'CI_source': 'UNCHANGED_GOAL010V2',
                                            'independent_point_verification_tolerance': TOLERANCE,
                                            'old_goal010_iou_reconciliation': old_iou,
                                            'new_CI_computed': False})
    return effects


def _masks_and_quintiles(pairs, cohort):
    cv = [p for p in pairs if p['policy'] == 'common_valid']
    audits = []; summaries = []; deltas = []
    for builder in BUILDERS:
        for shape in STRUCTURES:
            for frame in FRAMES:
                for scope in ('all',)+PROVIDERS:
                    for quintile in range(5):
                        scenes = {s[0] for s, c in cohort.items() if c['size_rank_stratum'] == quintile}
                        selected = [p for p in cv if p['builder'] == builder and p['structure'] == shape and p['frame'] == frame
                            and p['scene_id'] in scenes and (scope == 'all' or p['description_provider'] == scope)]
                        masks = {m: {p['global_schedule_index'] for p in selected if p['metric_id'] == m and p['delta'] is not None}
                                 for m in METRICS}
                        common = set.intersection(*masks.values())
                        core_common = set.intersection(*(masks[m] for m in CORE))
                        ctx = {'condition': 'O', 'builder': builder, 'structure': shape, 'frame': frame,
                               'description_provider': scope, 'size_rank_stratum': quintile, 'policy': 'common_valid'}
                        audit = {**ctx, 'core_masks_identical': len({tuple(sorted(masks[m])) for m in CORE}) == 1,
                                 'all_six_masks_identical': len({tuple(sorted(v)) for v in masks.values()}) == 1,
                                 'metric_specific_pairs': {m: len(v) for m, v in masks.items()},
                                 'common_pairs': len(common), 'common_global_indices': sorted(common),
                                 'core_common_pairs': len(core_common), 'core_common_global_indices': sorted(core_common),
                                 'extra_excluded_from_four_core_mask': sorted(core_common-common),
                                 'extra_excluded_by_common_mask': {m: sorted(v-common) for m, v in masks.items()}}
                        audits.append(audit)
                        for metric in METRICS:
                            use = [p for p in selected if p['metric_id'] == metric and p['global_schedule_index'] in common]
                            result, building = _equal_building(use)
                            building_values = [v['delta'] for v in building.values()]
                            summaries.append({**ctx, 'metric_id': metric, **result, 'planned_buildings': len(scenes),
                                'planned_pairs': len(scenes)*(3 if scope == 'all' else 1), 'CI': None,
                                'building_delta_sd_ddof1': stdev(building_values) if len(building_values) > 1 else None,
                                'building_delta_sign_counts': {'positive': sum(v > 0 for v in building_values),
                                    'negative': sum(v < 0 for v in building_values), 'zero': sum(v == 0 for v in building_values)},
                                'status': 'DESCRIPTIVE_ONLY_SMALL_STRATUM' if building else 'NOT_ESTIMABLE',
                                'p_value': None, 'new_inference': False, 'valid_mask_sha256': digest(sorted(common))})
                            for scene, values in building.items():
                                deltas.append({**ctx, 'metric_id': metric, 'scene_id': scene, **values,
                                               'independent_unit': 'building'})
    return audits, summaries, deltas


def precision_design_inputs(building_deltas):
    """Planning SD and approximate half-widths; no power, p value or new CI."""
    results = []
    for provider in PROVIDERS:
        for builder in BUILDERS:
            for metric in ('occupancy_iou', 'surface_f1_at_1'):
                rows = [r for r in building_deltas if r['description_provider'] == provider and r['builder'] == builder
                        and r['metric_id'] == metric and r['structure'] == 'filled_box' and r['frame'] == 'aligned']
                if len({r['scene_id'] for r in rows}) != len(rows):
                    _fail('DUPLICATE_PRECISION_BUILDING')
                values = [r['delta'] for r in sorted(rows, key=lambda r: r['scene_id'])]
                if len(values) > 60 or any(type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1 for v in values):
                    _fail('INVALID_PRECISION_DESIGN_INPUT')
                sd = stdev(values) if len(values) >= 2 else None
                designs = []
                for n in (30, 60):
                    critical = float(t.ppf(.975, n-1))
                    for repeat in (1, 3):
                        upper = critical*sd/math.sqrt(n) if sd is not None else None
                        designs.append({'target_buildings': n, 'repeat_generations_per_building': repeat,
                            't_critical_975': critical, 't_degrees_of_freedom': n-1,
                            'approximate_halfwidth_range': [upper/math.sqrt(repeat), upper] if upper is not None else None,
                            'target_halfwidth_planning_tolerance': .03, 'validated_MCID': False,
                            'repeat_variance_identified': False, 'independent_N_is_buildings': True,
                            'status': 'ASSUMPTION_DEPENDENT_DESIGN_HEURISTIC' if sd is not None else 'INSUFFICIENT_BUILDINGS'})
                results.append({'description_provider': provider, 'builder': builder, 'metric_id': metric,
                    'frame': 'aligned', 'structure': 'filled_box', 'policy': 'common_valid',
                    'observed_valid_buildings': len(values), 'building_delta_sample_sd_ddof1': sd,
                    'building_delta_vector_sha256': digest([{'scene_id': r['scene_id'], 'delta': r['delta']}
                        for r in sorted(rows, key=lambda r: r['scene_id'])]), 'designs': designs,
                    'assumptions': ['eval60 valid-only variation transports to the later cohort',
                        'nonnegative between-building and within-building variance components',
                        'conditionally independent, identical repeat errors within a building',
                        'no additional selection change from validity or missingness'],
                    'limitations': 'Repeat variance is unidentified. Bounds are planning scenarios, not confidence bounds; '
                        'correlated repeats, validity selection or cohort shift can invalidate them. No power or sufficiency claim.',
                    'preferred_candidate_provider_fixed_before_computation': provider == 'openai',
                    'no_outcome_based_provider_selection': True})
    return results


def render_report(result):
    effects = result['global_saved_effects']
    lines = ['# O条件：coverage と selectivity のトレードオフ監査', '', LABEL, '',
        '既存の6指標だけを使用し、全60の95% CIはGoal010v2から変更せず再利用した。五分位の新集計は記述のみで、新CI・p値・bootstrapを計算しない。', '',
        '差はBuilder−対照。building内の適格な説明生成元ペアを平均した後、建築を等重みで平均する。2 Buildersと3説明を独立Nとして足さない。alignedはGTに対するoccupancy最適シフト後の評価であり、世界原点の推定精度ではない。', '',
        '| Builder | 対照 | frame | 指標 | ペアN / 建築N | 差 [保存済み95% CI] |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in effects:
        if r['description_provider'] != 'all' or r['policy'] != 'common_valid':
            continue
        ci = r['ci95']; text = 'NULL' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'
        delta = 'NULL' if r['mean_delta'] is None else f"{r['mean_delta']:+.6f}"
        lines.append(f"| {r['builder']} | {r['structure']} | {r['frame']} | {r['metric_id']} | "
                     f"{r['eligible_pairs']} / {r['independent_buildings']} | {delta} {text} |")
    deviations = [r for r in result['valid_mask_audit'] if not r['all_six_masks_identical']]
    core_deviations = [r for r in result['valid_mask_audit'] if not r['core_masks_identical']]
    lines += ['', f"五分位×Builder×生成元scope×shape×frameのvalid-mask監査は{len(result['valid_mask_audit'])}群。"
        f"4 core指標間で異なる群は{len(core_deviations)}。"
        f"6指標間で異なる群は{len(deviations)}。新しい五分位表では常に6指標の共通valid集合を使い、追加除外を記録した。", '',
        ('保存済みの全体表は旧estimandのmetric別maskを保持する。core-mask差異があるため、その比較を同一分母と呼ばない。'
         if core_deviations else '全体表でも4 core指標の適格pair ID集合は一致する。Builder別・説明scope別の適格Nは保持し、Builder間で同じNとは仮定しない。'), '',
        'filled boxはGT占有を広く覆う一方で余分な占有を作る。Builderは占有を選別し、その分作り漏らしも残り得る。これらはPrecision・Recall・IoUを併記して検討する。boundary-neighbor F1は6近傍境界セル中心の距離で、空洞内壁を含み、撮影画像の可視面・見た目・建築機能の忠実度ではない。', '',
        '五分位は最大12建築で、細分した生成元比較の不確実性をこの記述表だけで解消できない。正負の件数を独立の有意な発見数と数えず、全生成元・両Builder・両shape・両frameを保持する。正常emptyや未定義scalarを既知失敗0に混ぜず、HTTP520は未計測のまま残す。', '',
        '将来設計のSDとt近似半幅はeval60のvalid-only建築差から得た設計入力である。N30/N60とr1/r3の範囲は反復内・建築間分散の非負分解と独立反復を仮定する。反復分散は未同定で、十分な標本数・検出力・将来CIを保証しない。目標半幅0.03は計画上の許容幅でMCIDではない。実行承認はなく、研究API送信0。', '']
    lines += ['| 条件 | Builder | 固定slot N | 成功 | 既知出力失敗 | 運用欠測 | 正常empty | 既知失敗率 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in result['claim_audit']['reliability_outcomes']:
        counts = r['status_counts']
        lines.append(f"| {r['condition']} | {r['builder']} | {r['planned_slots']} | {counts.get('success', 0)} | "
                     f"{counts.get('failed', 0)} | {counts.get('operational_unmeasurable', 0)} | "
                     f"{r['normal_empty_success_count']} | {r['known_output_failure_fraction']:.6f} |")
    lines += ['', '失敗率の分母は固定slotであり、failed attemptを独立建築として追加しない。global940は承認済みretryの成功を1slotとして選択し、元401とglobal942のHTTP520に対するunknown-usage holdは維持する。費用やusageの推定をしない。', '',
        'O baselineの構築入力は保存説明のdimensions_estimateと固定原点(0,0,0)、Pは開示world_bboxだけである。いずれも人工材質stoneを材質比較に用いない。構築時の入力制約と、Builder同様にGTを参照するaligned評価を区別する。', '',
        '旧Goal010とv2のIoU差に許容内の浮動小数点丸め差がある場合はclaim auditに保存する。数値の近似一致をbit完全一致と呼ばず、過去表を書き換えない。相関する指標と探索的CIを独立した有意な発見として数えない。', '']
    return '\n'.join(lines)


def analyze_tradeoff(data, *, gate, plan_sha256):
    _gate(gate, plan_sha256)
    before = digest(data)
    index, bases, cohort = _validate_rows(data, gate)
    construction = audit_construction(data)
    pairs = _pair_rows(index, bases, cohort)
    effects = _effect_audit(pairs, data, sorted(s[0] for s in cohort))
    masks, quintiles, deltas = _masks_and_quintiles(pairs, cohort)
    precision = precision_design_inputs(deltas)
    if digest(data) != before:
        _fail('INPUT_MUTATION_DETECTED')
    reliability = []
    for condition in ('O', 'J', 'P', 'E'):
        for builder in BUILDERS:
            selected = [r for r in data['metric_rows'] if r['condition'] == condition and r['provider'] == builder]
            reliability.append({'condition': condition, 'builder': builder, 'planned_slots': len(selected),
                'independent_buildings': len({r['scene_id'] for r in selected}),
                'status_counts': dict(Counter(r['status'] for r in selected)),
                'identified_failure_kind_counts': dict(Counter(r['failure_kind'] for r in selected if r['status'] == 'failed')),
                'known_failure0_eligible_slots': sum(r['failure0_eligible'] for r in selected),
                'normal_empty_success_count': sum(r['status'] == 'success' and r['metrics']['aligned'].get('pred_occupied_count') == 0
                                                   for r in selected),
                'failure_rate_denominator': len(selected),
                'known_output_failure_fraction': sum(r['status'] == 'failed' for r in selected)/len(selected),
                'operational_missing_fraction': sum(r['status'] == 'operational_unmeasurable' for r in selected)/len(selected),
                'not_multiplied_into_metric_scores': True})
    result = {'claim_audit': {'status': 'PASS', 'critical_inconsistencies': [],
        'construction_checks': construction, 'all_original_classifications_retained': True,
        'reliability_outcomes': reliability,
        'historical_calls_holds_reference_sha256': digest(data['integrity_reference']),
        'prior_saved_point_effects_independently_reconciled': True, 'saved_CIs_unchanged': True,
        'absolute_numeric_reconciliation_tolerance': TOLERANCE,
        'normal_empty_success_count': sum(r['status'] == 'success' and r['metrics']['aligned'].get('pred_occupied_count') == 0
                                         for r in data['metric_rows']),
        'fixed_original_occupancy_shifts_preserved': True,
        'core_valid_masks_identical': all(r['core_masks_identical'] for r in masks),
        'core_valid_mask_deviation_groups': sum(not r['core_masks_identical'] for r in masks),
        'valid_mask_deviation_groups': sum(not r['all_six_masks_identical'] for r in masks),
        'prior_iou_roundoff_findings': [
            {**{k: r[k] for k in ('condition', 'description_provider', 'structure', 'builder', 'frame', 'policy')},
             'old_mean_delta': r['old_goal010_iou_reconciliation']['mean_delta'],
             'v2_mean_delta': r['mean_delta'], 'status': 'WITHIN_TOLERANCE_RETAINED_NOT_CORRECTED'}
            for r in effects if r['old_goal010_iou_reconciliation'] is not None
            and not r['old_goal010_iou_reconciliation']['mean_delta_bit_identical']],
        'limitations': ['Input artifact and frozen implementation byte verification is performed by the pipeline source guard.',
                        'No causal information-loss inference; O is structured JSON and J preserves its typed tree.',
                        'No visible-image, semantic or architectural fidelity validation.'],
        'historical_results_modified': False}, 'global_saved_effects': effects,
        'quintile_effects': quintiles, 'valid_mask_audit': masks, 'building_deltas': deltas,
        'precision_design_inputs': precision,
        'metadata': {'analysis_label': LABEL, 'gate': dict(gate), 'input_bundle_sha256': before,
            'core_metrics': list(CORE), 'sensitivity_metrics': list(METRICS[4:]),
            'input_kind': gate['input_kind'], 'input_unchanged': True,
            'global_saved_effects': len(effects), 'quintile_effects': len(quintiles),
            'precision_design_cells': len(precision), 'new_CIs': 0, 'new_p_values': 0,
            'new_bootstraps': 0, 'new_geometry_metrics': 0, 'research_api_requests_sent': 0,
            'model_queries': 0, 'api_key_queries': 0, 'token_count_api_requests': 0}}
    result['report_markdown'] = render_report(result)
    return result
