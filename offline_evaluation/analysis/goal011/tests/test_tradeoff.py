import copy
import hashlib
import json
import math
from statistics import mean, stdev

import pytest

from offline_evaluation.analysis.goal011 import tradeoff as m


PLAN = '1'*64


def gate():
    return {'status': 'PASS', 'input_kind': 'fixture', 'plan_sha256': PLAN,
            'question_register_sha256': '2'*64, 'source_package_sha256': '3'*64,
            'code_commit': '4'*40, 'specification_commit': '5'*40}


def fixture_alignment(shift):
    return {'metric_version': m.ALIGNMENT, 'alignment': {
        'shift_xyz': list(shift), 'mode': 'strict_translation', 'search_complete': True,
        'globally_optimal_unrestricted': True, 'all_aligned_metrics_share_shift': True,
        'rotation': False, 'scale': False, 'clipping': False, 'resource_limit_reached': False}}


def fixture_provenance(shift):
    return {'shift_xyz': list(shift), 'displacement': {'shift_xyz': list(shift)},
            'metric_provenance': {'stored_alignment_shift_xyz': list(shift),
                'alignment_protocol': m.ALIGNMENT, 'alignment_searched': False, 'failure0_applied': False}}


def fixture_data():
    data = {'metric_rows': [], 'baseline_rows': [], 'original_slots': [], 'input_records': [],
            'cohort_rows': [], 'baseline_evaluations': [], 'baseline_manifest': {
                'status': 'ALL_PREDICTIONS_FROZEN_BEFORE_GT_EVALUATION', 'GT_voxels_read_during_construction': False,
                'Builder_replication_of_predictions': False, 'predictions': []}, 'integrity_reference': {}}
    number = 0
    for q in range(5):
        scene = f'scene_{q}'
        data['cohort_rows'].append({'scene_id': scene, 'size_rank_stratum': q})
        for condition in ('O', 'J', 'P', 'E'):
            for provider in m.PROVIDERS if condition in ('O', 'J') else (None,):
                dimensions = {'width': 3+q, 'height': 4, 'depth': 5}
                bbox = {'xmin': 10, 'xmax': 12+q, 'ymin': -2, 'ymax': 1, 'zmin': -4, 'zmax': 0}
                payload = {'dimensions_estimate': dimensions, 'world_bbox': bbox}
                text = json.dumps(payload, sort_keys=True)
                input_sha = hashlib.sha256(text.encode()).hexdigest()
                identity = {'scene_id': scene, 'condition': condition, 'description_model': provider,
                            'input_id': f'{scene}:{condition}:{provider}', 'input_sha256': input_sha,
                            'gt_grid_sha256': str(q)*64, 'coordinate_system': m.COORDINATES,
                            'alignment_protocol': m.ALIGNMENT}
                data['input_records'].append({**identity, 'input_text': text})
                for builder in m.BUILDERS:
                    values = {metric: .35 + .02*q + (.03 if builder == 'anthropic' else 0)
                              + (.01 if provider == 'gemini' else 0) for metric in m.METRICS}
                    row = {**identity, 'global_schedule_index': number, 'provider': builder, 'status': 'success',
                           'failure_kind': None, 'failure0_eligible': False, 'usage_known': True,
                           'completed_request_outcome': True, 'attempt_id': f'attempt{number}',
                           'empty_output': False, **fixture_provenance([1, 2, 3]),
                           'metrics': {f: {**values, 'pred_occupied_count': 20} for f in m.FRAMES},
                           'null_reasons': {f: {} for f in m.FRAMES}}
                    original = {**copy.deepcopy(row), 'metrics': fixture_alignment([1, 2, 3])}
                    row['original_metric_object_sha256'] = m.digest(original['metrics'])
                    number += 1
                    data['metric_rows'].append(row); data['original_slots'].append(original)
                if condition not in ('O', 'P'):
                    continue
                for shape in m.STRUCTURES:
                    width = 3+q
                    count = width*20 if shape == 'filled_box' else width*20-(width-2)*6
                    use_bbox = bbox if condition == 'P' else {'xmin': 0, 'xmax': width-1, 'ymin': 0,
                                                           'ymax': 3, 'zmin': 0, 'zmax': 4}
                    prediction = {**identity, 'prediction_id': f'{scene}:{condition}:{provider}:{shape}',
                                  'structure': shape, 'permitted_input': dimensions if condition == 'O' else bbox,
                                  'prediction_bbox': use_bbox, 'prediction_dimensions_xyz': [width, 4, 5],
                                  'prediction_voxels': count, 'material_metrics_used': False,
                                  'artificial_material': 'stone', 'source_sha256': input_sha,
                                  'status': 'PREDICTION_SAVED'}
                    data['baseline_manifest']['predictions'].append(prediction)
                    value = .2 + .01*q + (.01 if shape == 'six_face_shell' else 0)
                    evaluation = {**copy.deepcopy(prediction), 'status': 'success', 'independent_check_status': 'PASS',
                                  'absolute_occupancy_iou': value, 'aligned_occupancy_iou': value,
                                  **fixture_alignment([4, 5, 6]), 'independent_check': {'shift_xyz': [4, 5, 6]}}
                    data['baseline_evaluations'].append(evaluation)
                    row = {**identity, 'prediction_id': prediction['prediction_id'], 'structure': shape,
                           'status': 'success', 'failure0_eligible': False,
                           **fixture_provenance([4, 5, 6]),
                           'metrics': {f: {**{metric: value for metric in m.METRICS}, 'pred_occupied_count': count}
                                       for f in m.FRAMES}, 'null_reasons': {f: {} for f in m.FRAMES}}
                    data['baseline_rows'].append(row)
    refresh_summaries(data)
    return data


def refresh_summaries(data):
    # Independent fixture generator: explicit scene/provider lists and statistics.mean.
    rows = data['metric_rows']; bases = data['baseline_rows']; summaries = []; old = []
    for scope in ('all',)+m.PROVIDERS:
        for shape in m.STRUCTURES:
            for builder in m.BUILDERS:
                for frame in m.FRAMES:
                    for metric in m.METRICS:
                        for policy in m.POLICIES:
                            group = {}; n = imputed = planned = 0
                            for r in rows:
                                if r['condition'] != 'O' or r['provider'] != builder or (scope != 'all' and r['description_model'] != scope):
                                    continue
                                planned += 1
                                base = next(b for b in bases if b['scene_id'] == r['scene_id'] and b['condition'] == 'O'
                                            and b['description_model'] == r['description_model'] and b['structure'] == shape)
                                mv = r['metrics'][frame][metric]; bv = base['metrics'][frame][metric]
                                fill = mv is None and policy == 'known_failure0' and r['failure0_eligible']
                                if fill: mv = 0
                                if mv is None or bv is None: continue
                                group.setdefault(r['scene_id'], []).append((mv, bv, mv-bv))
                                n += 1; imputed += fill
                            values = [tuple(mean(t[i] for t in ps) for i in range(3)) for ps in group.values()]
                            point = mean(t[2] for t in values) if values else None
                            summary = {'condition': 'O', 'description_provider': scope, 'structure': shape,
                                'builder': builder, 'frame': frame, 'metric_id': metric, 'policy': policy,
                                'builder_value': mean(t[0] for t in values) if values else None,
                                'baseline_value': mean(t[1] for t in values) if values else None,
                                'mean_delta': point, 'eligible_pairs': n, 'planned_pairs': planned,
                                'independent_buildings': len(group), 'planned_buildings': 5,
                                'excluded_pairs': planned-n, 'failure0_imputed_pairs': imputed,
                                'ci95': [point-.1, point+.1] if point is not None else None, 'status': 'FIXTURE_SAVED'}
                            summaries.append(summary)
                            if metric == 'occupancy_iou':
                                old.append({**summary, 'metric': frame+'_occupancy_iou', 'common_pairs': n})
    data['saved_summaries'] = summaries; data['old_summaries'] = old


def run(data):
    return m.analyze_tradeoff(data, gate=gate(), plan_sha256=PLAN)


def test_full_finite_fixture_and_immutability():
    data = fixture_data(); before = copy.deepcopy(data); r = run(data)
    assert data == before
    assert r['claim_audit']['status'] == 'PASS'
    assert len(r['global_saved_effects']) == 384
    assert len(r['quintile_effects']) == 960
    assert len(r['valid_mask_audit']) == 160
    assert len(r['precision_design_inputs']) == 12
    assert all(v['all_six_masks_identical'] for v in r['valid_mask_audit'])
    assert all(v['CI'] is None and v['p_value'] is None for v in r['quintile_effects'])
    assert all(v['CI_source'] == 'UNCHANGED_GOAL010V2' and not v['new_CI_computed'] for v in r['global_saved_effects'])
    assert all(v['building_delta_sd_ddof1'] is None for v in r['quintile_effects'])
    assert len(r['claim_audit']['reliability_outcomes']) == 8
    assert sum(r['claim_audit']['reliability_outcomes'][i]['planned_slots'] for i in range(8)) == 80
    assert r['metadata']['research_api_requests_sent'] == r['metadata']['new_bootstraps'] == 0
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('field,value', [('status', 'FAIL'), ('input_kind', 'unknown'), ('plan_sha256', 'a'*64),
    ('question_register_sha256', ''), ('code_commit', 'abc'), ('specification_commit', 'none'), ('source_package_sha256', '')])
def test_gate_precedes_any_input_access(field, value):
    g = gate(); g[field] = value
    with pytest.raises(m.AuditFailure): m.analyze_tradeoff(None, gate=g, plan_sha256=PLAN)


@pytest.mark.parametrize('field,value', [('input_id', 'changed'), ('input_sha256', 'f'*64),
    ('gt_grid_sha256', 'f'*64), ('coordinate_system', 'YZX'), ('alignment_protocol', 'new_metric_shift')])
def test_binding_tamper_stops(field, value):
    d = fixture_data(); d['baseline_rows'][0][field] = value
    with pytest.raises(m.AuditFailure):run(d)


@pytest.mark.parametrize('target', ['metric_rows', 'baseline_rows', 'cohort_rows', 'original_slots'])
def test_duplicate_join_rejected(target):
    d = fixture_data(); d[target].append(copy.deepcopy(d[target][0]))
    with pytest.raises(m.AuditFailure):run(d)


def test_description_only_origin_and_p_disclosed_bbox_audit():
    d = fixture_data(); r = run(d)
    proof = r['claim_audit']['construction_checks']
    assert {p['origin_rule'] for p in proof} == {'FIXED_ZERO', 'SUPPLIED_P_WORLD_BBOX'}
    assert all(p['fixed_shift_check'] == 'PASS' and p['original_occupancy_shift_xyz'] == [4, 5, 6] for p in proof)
    first = proof[0]; assert first['expected_occupied_count'] == 60
    shell = next(p for p in proof if p['prediction_id'].endswith('six_face_shell'))
    assert shell['expected_occupied_count'] == 54
    d['baseline_manifest']['predictions'][0]['prediction_bbox']['xmin'] = 10
    with pytest.raises(m.AuditFailure, match='BASELINE_SAVED_CONSTRUCTION_MISMATCH'):run(d)


@pytest.mark.parametrize('mutation', ['gt_during_construction', 'permitted', 'old_gt', 'count', 'input_text', 'material'])
def test_construction_leakage_or_artifact_drift(mutation):
    d = fixture_data()
    if mutation == 'gt_during_construction':d['baseline_manifest']['GT_voxels_read_during_construction'] = True
    if mutation == 'permitted':d['baseline_manifest']['predictions'][0]['permitted_input']['width'] += 1
    if mutation == 'old_gt':d['baseline_evaluations'][0]['gt_grid_sha256'] = 'f'*64
    if mutation == 'count':d['baseline_manifest']['predictions'][0]['prediction_voxels'] += 1
    if mutation == 'input_text':d['input_records'][0]['input_text'] += ' '
    if mutation == 'material':d['baseline_manifest']['predictions'][0]['material_metrics_used'] = True
    with pytest.raises(m.AuditFailure):run(d)


def test_core_masks_and_stratum_common_intersection_keep_null():
    d = fixture_data(); r = d['metric_rows'][0]
    r['metrics']['absolute']['occupancy_precision'] = None
    r['null_reasons']['absolute']['occupancy_precision'] = 'EMPTY_PREDICTION'
    refresh_summaries(d); result = run(d)
    assert result['claim_audit']['core_valid_masks_identical'] is False
    assert result['claim_audit']['core_valid_mask_deviation_groups'] == 4
    assert 'その比較を同一分母と呼ばない' in result['report_markdown']
    audits = [r for r in result['valid_mask_audit'] if not r['all_six_masks_identical']]
    assert len(audits) == 4  # both shapes, all/one provider, one frame/stratum
    for a in audits:
        assert a['extra_excluded_by_common_mask']['occupancy_iou'] == [0]
        assert a['extra_excluded_by_common_mask']['occupancy_precision'] == []
    q = [r for r in result['quintile_effects'] if r['builder'] == 'openai' and r['structure'] == 'filled_box'
         and r['frame'] == 'absolute' and r['description_provider'] == 'all' and r['size_rank_stratum'] == 0]
    assert len(q) == 6 and all(x['eligible_pairs'] == 2 for x in q)
    assert len({r['valid_mask_sha256'] for r in q}) == 1
    f0 = next(x for x in result['global_saved_effects'] if x['builder'] == 'openai' and x['structure'] == 'filled_box'
              and x['frame'] == 'absolute' and x['description_provider'] == 'all' and x['policy'] == 'known_failure0'
              and x['metric_id'] == 'occupancy_precision')
    assert f0['failure0_imputed_pairs'] == 0


def test_known_failure0_and_unknown_are_separate():
    d = fixture_data()
    for number, status, failure, known in [(0, 'failed', 'validation_failure', True),
                                         (16, 'operational_unmeasurable', None, False)]:
        r = d['metric_rows'][number]
        r.update(status=status, failure_kind=failure, usage_known=known, completed_request_outcome=known,
                 failure0_eligible=known, shift_xyz=None, displacement=None, metric_provenance=None,
                 original_metric_object_sha256=m.digest(None))
        for frame in m.FRAMES:
            for metric in m.METRICS:
                r['metrics'][frame][metric] = None; r['null_reasons'][frame][metric] = status
        d['original_slots'][number] = copy.deepcopy(r)
        d['original_slots'][number]['metrics'] = None
    refresh_summaries(d); result = run(d)
    s = next(r for r in result['global_saved_effects'] if r['builder'] == 'openai' and r['structure'] == 'filled_box'
             and r['frame'] == 'aligned' and r['metric_id'] == 'occupancy_iou'
             and r['description_provider'] == 'all' and r['policy'] == 'known_failure0')
    assert s['eligible_pairs'] == 14 and s['failure0_imputed_pairs'] == 1
    rel = next(x for x in result['claim_audit']['reliability_outcomes'] if x['condition'] == 'O' and x['builder'] == 'openai')
    assert rel['known_output_failure_fraction'] == pytest.approx(1/15)
    assert rel['operational_missing_fraction'] == pytest.approx(1/15)
    d['metric_rows'][16]['failure0_eligible'] = True
    with pytest.raises(m.AuditFailure, match='FAILURE0_STATUS_MISMATCH'):run(d)


@pytest.mark.parametrize('value', [True, float('inf'), float('nan'), -0.1, 1.1])
def test_invalid_metric_rejected(value):
    d = fixture_data(); d['metric_rows'][0]['metrics']['absolute']['occupancy_iou'] = value
    with pytest.raises((m.AuditFailure, ValueError)):run(d)


def test_saved_ci_reused_and_critical_effect_proof():
    d = fixture_data(); d['saved_summaries'][0]['mean_delta'] += .01
    with pytest.raises(m.AuditFailure, match='SAVED_SUMMARY_VALUE_CHANGED') as e:run(d)
    assert e.value.report['critical'] is True and 'independent' in e.value.report['evidence']
    assert e.value.report['historical_changes_permitted'] is False


def test_old_iou_ci_cannot_silently_change():
    d = fixture_data(); d['old_summaries'][0] = copy.deepcopy(d['old_summaries'][0])
    d['old_summaries'][0]['ci95'][0] -= .01
    with pytest.raises(m.AuditFailure, match='OLD_IOU_EFFECT_OR_SAVED_CI_CHANGED'):run(d)


def test_old_iou_roundoff_is_disclosed_without_rewriting():
    d = fixture_data(); original = d['old_summaries'][0]['mean_delta']
    d['old_summaries'][0]['mean_delta'] = math.nextafter(original, math.inf)
    before = copy.deepcopy(d); result = run(d)
    assert d == before
    findings = result['claim_audit']['prior_iou_roundoff_findings']
    assert len(findings) == 1
    assert findings[0]['old_mean_delta'] != findings[0]['v2_mean_delta']
    assert findings[0]['status'] == 'WITHIN_TOLERANCE_RETAINED_NOT_CORRECTED'


@pytest.mark.parametrize('target', ['metric_rows', 'baseline_rows'])
@pytest.mark.parametrize('field', ['shift_xyz', 'displacement', 'stored_shift', 'new_search', 'failure0'])
def test_saved_gt_shift_is_preserved_for_builder_and_baseline(target, field):
    d = fixture_data(); row = d[target][0]
    if field == 'shift_xyz': row['shift_xyz'][0] += 1
    if field == 'displacement': row['displacement']['shift_xyz'][0] += 1
    if field == 'stored_shift': row['metric_provenance']['stored_alignment_shift_xyz'][0] += 1
    if field == 'new_search': row['metric_provenance']['alignment_searched'] = True
    if field == 'failure0': row['metric_provenance']['failure0_applied'] = True
    with pytest.raises(m.AuditFailure, match='SAVED_OCCUPANCY_SHIFT_NOT_PRESERVED'): run(d)


@pytest.mark.parametrize('field', ['rotation', 'scale', 'clipping', 'resource_limit_reached', 'search_complete'])
def test_original_alignment_certificate_is_required(field):
    d = fixture_data(); orig = d['original_slots'][0]['metrics']
    orig['alignment'][field] = not orig['alignment'][field]
    d['metric_rows'][0]['original_metric_object_sha256'] = m.digest(orig)
    with pytest.raises(m.AuditFailure, match='ORIGINAL_STRICT_ALIGNMENT_CERTIFICATE_INVALID'): run(d)


def test_baseline_prior_independent_shift_must_match():
    d = fixture_data(); d['baseline_evaluations'][0]['independent_check']['shift_xyz'] = [0, 0, 0]
    with pytest.raises(m.AuditFailure, match='BASELINE_INDEPENDENT_SHIFT_MISMATCH'): run(d)


@pytest.mark.parametrize('field,value', [('http_status_code', 520), ('selection_reason', 'unapproved_retry'),
                                      ('original_attempt_id', 'changed'), ('empty_output', True)])
def test_original_attempt_and_empty_classifications_are_preserved(field, value):
    d = fixture_data(); d['metric_rows'][0][field] = value
    with pytest.raises(m.AuditFailure, match='ORIGINAL_SELECTED_SLOT_CHANGED'): run(d)


def test_normal_empty_success_keeps_undefined_precision_and_valid_status():
    d = fixture_data(); row = next(r for r in d['metric_rows'] if r['condition'] == 'J')
    row['empty_output'] = True
    d['original_slots'][row['global_schedule_index']]['empty_output'] = True
    for frame in m.FRAMES:
        row['metrics'][frame]['pred_occupied_count'] = 0
        for metric in m.METRICS:
            row['metrics'][frame][metric] = 0
        row['metrics'][frame]['occupancy_precision'] = None
        row['null_reasons'][frame]['occupancy_precision'] = 'EMPTY_PREDICTION'
    result = run(d)
    assert result['claim_audit']['normal_empty_success_count'] == 1
    rel = next(r for r in result['claim_audit']['reliability_outcomes']
               if r['condition'] == row['condition'] and r['builder'] == row['provider'])
    assert rel['normal_empty_success_count'] == 1 and rel['known_output_failure_fraction'] == 0
    assert row['metrics']['aligned']['occupancy_precision'] is None


def test_building_weighting_does_not_count_provider_pairs_independently():
    values = [{'scene_id': 'a', 'builder_value': 1., 'baseline_value': 0., 'delta': 1.}]*3 + [
        {'scene_id': 'b', 'builder_value': 0., 'baseline_value': 0., 'delta': 0.}]
    summary, building = m._equal_building(values)
    assert summary['mean_delta'] == .5 and summary['eligible_pairs'] == 4
    assert summary['independent_buildings'] == 2 and len(building) == 2


def test_precision_only_sd_and_assumption_dependent_halfwidth():
    rows = [{'description_provider': 'openai', 'builder': 'openai', 'metric_id': 'occupancy_iou',
             'structure': 'filled_box', 'frame': 'aligned', 'scene_id': str(i), 'delta': x}
            for i, x in enumerate([-.1, .1, .3])]
    result = m.precision_design_inputs(rows)[0]
    assert result['building_delta_sample_sd_ddof1'] == pytest.approx(.2)
    n30r1, n30r3, n60r1, n60r3 = result['designs']
    h30 = 2.045229642132703*.2/math.sqrt(30)
    h60 = 2.0009953780882674*.2/math.sqrt(60)
    assert n30r1['approximate_halfwidth_range'] == pytest.approx([h30, h30])
    assert n30r3['approximate_halfwidth_range'] == pytest.approx([h30/math.sqrt(3), h30])
    assert n60r1['approximate_halfwidth_range'] == pytest.approx([h60, h60])
    assert n60r3['target_buildings'] == 60 and n30r3['target_buildings'] == 30
    assert all(not r['repeat_variance_identified'] for r in result['designs'])
    assert all(not r['validated_MCID'] for r in result['designs'])


def test_single_building_precision_is_not_verified():
    rows = [{'description_provider': 'openai', 'builder': 'openai', 'metric_id': 'occupancy_iou',
             'structure': 'filled_box', 'frame': 'aligned', 'scene_id': 'a', 'delta': .1}]
    result = m.precision_design_inputs(rows)[0]
    assert result['building_delta_sample_sd_ddof1'] is None
    assert all(r['approximate_halfwidth_range'] is None and r['status'] == 'INSUFFICIENT_BUILDINGS' for r in result['designs'])


def test_sensitivity_only_null_reports_four_core_to_six_mask_difference():
    d = fixture_data(); r = d['metric_rows'][0]
    r['metrics']['absolute']['footprint_iou'] = None
    r['null_reasons']['absolute']['footprint_iou'] = 'FIXTURE_RESOURCE_NULL'
    refresh_summaries(d); result = run(d)
    changed = [x for x in result['valid_mask_audit'] if not x['all_six_masks_identical']]
    assert len(changed) == 4
    assert all(x['core_masks_identical'] and x['extra_excluded_from_four_core_mask'] == [0] for x in changed)


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), 1.01])
def test_precision_rejects_nonfinite_or_invalid_delta(value):
    rows = [{'description_provider': 'openai', 'builder': 'openai', 'metric_id': 'occupancy_iou',
             'structure': 'filled_box', 'frame': 'aligned', 'scene_id': 'a', 'delta': value}]
    with pytest.raises(m.AuditFailure, match='INVALID_PRECISION_DESIGN_INPUT'):m.precision_design_inputs(rows)


def test_no_file_or_socket_access(monkeypatch):
    import builtins
    import socket
    d = fixture_data()
    def denied(*args, **kwargs): raise AssertionError('unexpected IO')
    monkeypatch.setattr(builtins, 'open', denied)
    monkeypatch.setattr(socket, 'socket', denied)
    result = run(d)
    assert result['metadata']['research_api_requests_sent'] == 0
