"""Synthetic bound sources only: never open the production E corpus."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import signal
import socket

import pytest

from offline_evaluation.analysis.goal010.tests.test_exact_oracle import fixture as source_fixture
from offline_evaluation.analysis.goal011 import material_batch as batch


def rebind_fixture_artifacts(src):
    """Rebind newly constructed synthetic bytes, never production artifacts."""
    manifest = batch.old.read_json(src['manifest'])
    ledger_path = Path(manifest['sources']['fixture']['ledger_path'])
    ledger = batch.old.read_json(ledger_path)
    for row in src['slots']:
        for descriptor in row['artifacts'].values():
            path = Path(row['source_run']) / descriptor['path']
            descriptor.update(sha256=batch.old.sha256_file(path), bytes=path.stat().st_size)
        entry = next(e for e in ledger['entries'] if e['attempt_id'] == row['attempt_id'])
        entry['artifacts'] = deepcopy(row['artifacts'])
        row['source_entry_hash'] = batch.old.object_hash(entry)
    ledger_path.write_bytes(batch.old.json_bytes(ledger))
    manifest['sources']['fixture']['ledger_sha256'] = batch.old.sha256_file(ledger_path)
    slots_path = src['manifest'].parent / 'data/final_slots_960.json'
    slots_path.write_bytes(batch.old.json_bytes(src['slots']))
    manifest['products']['data/final_slots_960.json'].update(
        sha256=batch.old.sha256_file(slots_path), bytes=slots_path.stat().st_size)
    src['manifest'].write_bytes(batch.old.json_bytes(manifest))
    src['gate']['goal007_manifest_sha256'] = batch.old.sha256_file(src['manifest'])


@pytest.fixture
def bound(source_fixture):
    src = source_fixture
    for row in src['slots']:
        if row['status'] != 'success':
            continue
        root = Path(row['source_run'])
        parsed = batch.old.read_json(root / row['artifacts']['parsed.json']['path'])
        text = json.dumps(parsed)
        raw = ({'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]}
               if row['provider'] == 'openai' else
               {'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': text}]})
        (root / row['artifacts']['raw.json']['path']).write_bytes(batch.old.json_bytes(raw))
    rebind_fixture_artifacts(src)
    controls = []
    for scene in src['scenes'].values():
        row = next(r for r in src['slots'] if r['scene_id'] == scene.scene_id)
        plan = batch.old.read_json(row['source_plan_path'])
        record = next(r for r in plan['identity']['schedule'] if r['schedule_index'] == row['global_schedule_index'])
        _, control = batch.old.reference_control(scene, record['input_text'], batch.Limits())
        controls.append(control)
    path = src['root'] / 'old_controls.json'
    path.write_bytes(batch.old.json_bytes(controls))
    package = {'slots': src['slots'], 'goal007_manifest_path': str(src['manifest']),
               'goal007_manifest': batch.old.read_json(src['manifest']),
               'old_e_controls_descriptor': {'path': str(path), 'sha256': batch.old.sha256_file(path), 'bytes': path.stat().st_size}}
    gate = {**src['gate'], 'plan_sha256': src['gate']['specification_sha256'],
            'question_register_sha256': 'd'*64, 'source_package_sha256': 'e'*64,
            'specification_commit': 'f'*40}
    return package, gate, src


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('network is forbidden in CPU fixtures')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


def snapshots(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_end_to_end_bound_sources_reexecute_original_metrics_and_preserve_all_outcomes(bound):
    package, gate, src = bound
    before, package_before = snapshots(src['root']), deepcopy(package)
    result = batch.run_material(package, gate)
    assert snapshots(src['root']) == before and package == package_before
    assert result['audit']['status'] == 'PASS'
    assert result['audit']['outcomes'] == 4 and result['audit']['controls'] == 2
    assert result['audit']['original_successes'] == 3 and result['audit']['original_truncations'] == 1
    assert result['audit']['outcome_counts'] == {'openai': {'success': 1, 'truncation': 1}, 'anthropic': {'success': 2}}
    assert len(result['cases']) == 3 and len(result['mapping']) == 12
    assert len(result['audit']['building034_focus']) == 2
    assert result['audit']['historical_control_equality_verified']
    assert result['audit']['source_before_after_unchanged']
    assert result['audit']['research_api_requests_sent'] == result['audit']['api_key_queries'] == 0
    assert result['provenance']['material_implementation_sha256'] == batch.material.implementation_sha256()
    assert all(r['complete_confusion_independently_verified'] is True
               and r['raw_actions_to_saved_parsed_exact'] is True
               for r in result['audit']['outcome_records'] if r['original_status'] == 'success')
    assert str(src['manifest']) in result['source_inventory']
    failure = next(r for r in result['audit']['outcome_records'] if r['original_status'] == 'failed')
    assert failure['diagnostic_status'] == 'NOT_EVALUATED_ORIGINAL_FAILURE' and failure['frames'] == {}
    assert failure['capacity']['output_token_cap'] == failure['capacity']['output_tokens_observed'] == 32768
    assert failure['capacity']['completion_counterfactual'] is None


def test_fixed034_focus_complete_confusion_and_oracle_do_not_repair_model(bound):
    package, gate, _ = bound
    result = batch.run_material(package, gate)
    focus = next(r for r in result['audit']['building034_focus'] if r['provider'] == 'openai')
    value = focus['frames']['aligned']
    assert value['strict']['occupancy_iou'] == 1 and value['M'] < value['C']
    assert value['one_to_one']['maximum_exact_matches'] == value['C']
    assert value['provenance']['prediction_artifact_modified'] is False
    rows = [r for r in result['confusion'] if r['global_schedule_index'] == focus['global_schedule_index'] and r['frame'] == 'aligned']
    common = [r for r in rows if r['category'] == 'shared_occupancy']
    assert len(common) == len(value['confusion']['gt_labels']) * len(value['confusion']['pred_labels'])
    assert any(r['voxel_count'] == 0 for r in common)
    assert sum(r['voxel_count'] for r in common) == value['C']
    assert all(r['pred_material_id'] is None for r in rows if r['category'] == 'missing_geometry')
    assert all(r['gt_material_id'] is None for r in rows if r['category'] == 'extra_geometry')
    assert 'GTを見て最大化' in result['report_markdown']
    assert '両Builderの成功集合差' in result['report_markdown']


def test_absolute_and_saved_alignment_differ_and_actions_stay_absolute(bound):
    package, gate, _ = bound
    value = batch.run_material(package, gate)
    row = next(r for r in value['audit']['outcome_records'] if r['global_schedule_index'] == 3)
    absolute, aligned = row['frames']['absolute'], row['frames']['aligned']
    assert absolute['strict']['occupancy_iou'] < 1 == aligned['strict']['occupancy_iou']
    assert row['stored_alignment_shift_xyz'] == [-1, 0, 0]
    assert absolute['provenance']['saved_shift_xyz'] == [0, 0, 0]
    assert aligned['provenance']['saved_shift_xyz'] == [-1, 0, 0]
    assert absolute['reference_action_audit'] == aligned['reference_action_audit']
    assert aligned['reference_action_audit']['alignment_applied_to_actions'] is False


def test_summaries_use_one_building_per_success_not_voxel_weight(bound):
    package, gate, _ = bound
    value = batch.run_material(package, gate)
    for row in value['summary']:
        selected = [r['frames'][row['frame']] for r in value['audit']['outcome_records']
                    if r['provider'] == row['provider'] and r['original_status'] == 'success']
        expected = sum(r['strict']['material_aware_iou'] for r in selected)/len(selected)
        assert row['material_aware_iou']['mean'] == expected
        assert row['material_aware_iou']['valid_buildings'] == len(selected)
        assert row['pooled_voxel_counts']['M'] == sum(r['M'] for r in selected)
        assert row['equal_building_weight_within_builder'] is True
    for frame in batch.FRAMES:
        for builder in batch.old.BUILDERS:
            a = sum(r['voxel_count'] for r in value['confusion'] if r['frame'] == frame and r['provider'] == builder)
            b = sum(r['voxel_count'] for r in value['global_confusion'] if r['frame'] == frame and r['provider'] == builder)
            assert a == b


@pytest.mark.parametrize('field,value', [('status','FAIL'), ('phase0_binding','FAIL'),
    ('definitions_frozen',False), ('code_commit','bad'), ('plan_sha256','c'*64), ('specification_commit','bad')])
def test_incomplete_gate_prevents_any_source_binding(bound, monkeypatch, field, value):
    package, gate, _ = bound
    gate[field] = value
    monkeypatch.setattr(batch.old, 'bind_sources', lambda *a, **k: pytest.fail('source read before gate'))
    with pytest.raises(ValueError): batch.run_material(package, gate)


def test_controls_are_bound_and_recomputed_not_trusted_by_declaration(bound):
    package, gate, _ = bound
    descriptor = package['old_e_controls_descriptor']
    path = Path(descriptor['path'])
    control = batch.old.read_json(path)
    control[0]['reference_operations'] += 1
    path.write_bytes(batch.old.json_bytes(control))
    descriptor.update(sha256=batch.old.sha256_file(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match='MATERIAL_SOURCE_CONTROL_OR_COUNT_AUDIT_FAILED'):
        batch.run_material(package, gate)


@pytest.mark.parametrize('target', ['parsed.json', 'voxels.json', 'raw.json'])
def test_original_artifact_bytes_tamper_stops_before_diagnostic(bound, monkeypatch, target):
    package, gate, _ = bound
    row = package['slots'][0]
    path = Path(row['source_run'])/row['artifacts'][target]['path']
    path.write_bytes(path.read_bytes() + b' ')
    monkeypatch.setattr(batch.material, 'analyze_case', lambda *a, **k: pytest.fail('unbound artifact reached diagnostic'))
    with pytest.raises(ValueError, match='MATERIAL_SOURCE_CONTROL_OR_COUNT_AUDIT_FAILED'): batch.run_material(package, gate)


def test_original_cached_metric_disagreement_halts_without_replacement(bound, monkeypatch):
    package, gate, _ = bound
    bind = batch.old.bind_sources
    def changed(*args, **kwargs):
        rows, inventory = bind(*args, **kwargs)
        rows = deepcopy(rows)
        rows[0]['slot']['metrics']['aligned']['material_aware_iou'] = .9
        return rows, inventory
    monkeypatch.setattr(batch.old, 'bind_sources', changed)
    with pytest.raises(ValueError, match='MATERIAL_SOURCE_CONTROL_OR_COUNT_AUDIT_FAILED'): batch.run_material(package, gate)


def test_new_engine_disagreement_halts_without_changing_original(bound, monkeypatch):
    package, gate, _ = bound
    original = batch.material.analyze_case
    def wrong(*args, **kwargs):
        result = original(*args, **kwargs)
        result['M'] += 1
        return result
    monkeypatch.setattr(batch.material, 'analyze_case', wrong)
    with pytest.raises(ValueError, match='MATERIAL_SOURCE_CONTROL_OR_COUNT_AUDIT_FAILED'): batch.run_material(package, gate)


def test_off_diagonal_confusion_corruption_cannot_hide_behind_unchanged_C_and_M():
    gt = {(0, 0, 0): 'a', (1, 0, 0): 'a', (2, 0, 0): 'b'}
    pred = dict.fromkeys(gt, 'x')
    result = batch.material.analyze_case(gt, pred, [0, 0, 0])
    independent, pairs = batch.old.independent_metrics(gt, pred)
    rows = [{'GT_material': a, 'prediction_material': b, 'voxel_count': n}
            for (a, b), n in pairs.items()]
    batch._independent_check(result, independent, rows)
    result['confusion']['common_counts'] = [[1], [2]]
    result['confusion']['common_nonzero'][0]['count'] = 1
    result['confusion']['common_nonzero'][1]['count'] = 2
    assert result['C'] == 3 and result['M'] == 0
    with pytest.raises(ValueError, match='independent per-coordinate counts'):
        batch._independent_check(result, independent, rows)


@pytest.mark.parametrize('gt,pred', [({}, {}), ({(0, 0, 0): 'a'}, {}),
    ({}, {(0, 0, 0): 'b'}), ({(0, 0, 0): 'a'}, {(2, 0, 0): 'b'})])
def test_independent_confusion_empty_and_missing_side_semantics(gt, pred):
    result = batch.material.analyze_case(gt, pred, [0, 0, 0])
    independent, pairs = batch.old.independent_metrics(gt, pred)
    rows = [{'GT_material': a, 'prediction_material': b, 'voxel_count': n}
            for (a, b), n in pairs.items()]
    batch._independent_check(result, independent, rows)


def test_hash_bound_raw_response_must_reproduce_parsed_actions(bound):
    package, gate, src = bound
    row = package['slots'][0]
    path = Path(row['source_run']) / row['artifacts']['raw.json']['path']
    raw = batch.old.read_json(path)
    raw['output'][0]['content'][0]['text'] = '{"operations":[]}'
    path.write_bytes(batch.old.json_bytes(raw))
    rebind_fixture_artifacts(src)
    package['goal007_manifest'] = batch.old.read_json(src['manifest'])
    gate['goal007_manifest_sha256'] = src['gate']['goal007_manifest_sha256']
    with pytest.raises(batch.MaterialBatchAuditFailure) as error:
        batch.run_material(package, gate)
    assert error.value.report['stage'] == 'case'
    assert 'differs from saved parsed' in str(error.value.__cause__)


def test_raw_block_spelling_is_saved_separately_from_exact_canonical_metric_id(bound):
    package, gate, src = bound
    row = package['slots'][0]
    path = Path(row['source_run']) / row['artifacts']['raw.json']['path']
    raw = batch.old.read_json(path)
    document = json.loads(raw['output'][0]['content'][0]['text'])
    original = document['operations'][0]['block']
    document['operations'][0]['block'] = 'minecraft:' + original
    raw['output'][0]['content'][0]['text'] = json.dumps(document)
    path.write_bytes(batch.old.json_bytes(raw))
    rebind_fixture_artifacts(src)
    package['goal007_manifest'] = batch.old.read_json(src['manifest'])
    gate['goal007_manifest_sha256'] = src['gate']['goal007_manifest_sha256']
    result = batch.run_material(package, gate)
    case = next(r for r in result['audit']['outcome_records'] if r['global_schedule_index'] == 0)
    assert case['raw_actions_to_saved_parsed_exact'] is True
    published_action = case['frames']['absolute']['reference_action_audit']
    assert 'outputs' not in published_action and 'references' not in published_action
    assert any(r['material_as_supplied'] == 'minecraft:' + original
               and r['canonical_material'] == original for r in published_action['all_output_label_counts'])
    assert sum(r['action_count'] for r in published_action['all_output_label_counts']) == published_action['predicted_actions']
    assert sum(r['action_pairs'] for r in published_action['unique_literal_label_pair_counts']) == published_action['unique_literal_pairs']
    assert 'bounds_xyz_inclusive' not in json.dumps(result['audit'])


def test_source_after_hash_detects_change_during_computation(bound, monkeypatch):
    package, gate, _ = bound
    original = batch.material.analyze_case
    path = Path(package['old_e_controls_descriptor']['path'])
    def tamper(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b' ')
        return result
    monkeypatch.setattr(batch.material, 'analyze_case', tamper)
    with pytest.raises(ValueError, match='MATERIAL_SOURCE_CONTROL_OR_COUNT_AUDIT_FAILED'): batch.run_material(package, gate)


def test_resource_missing_preserves_success_rows_and_null_bounds(bound, monkeypatch):
    package, gate, _ = bound
    def limited(*args, **kwargs):
        raise batch.CaseResourceLimit('fixture')
    monkeypatch.setattr(batch.material, 'analyze_case', limited)
    result = batch.run_material(package, gate)
    assert len(result['cases']) == 3 and len(result['mapping']) == 12
    assert result['audit']['original_successes'] == result['audit']['diagnostic_resource_missing_successes'] == 3
    assert result['audit']['status'] == 'PASS_WITH_DIAGNOSTIC_RESOURCE_MISSING'
    assert result['audit']['original_truncations'] == 1
    assert result['confusion'] == []
    assert all(r['aligned_M'] is None and r['original_status'] == 'success' for r in result['cases'])
    assert all(r['maximum_exact_matches'] is None for r in result['mapping'])


def test_case_budget_installs_both_limits_and_restores_prior_stage_timers(monkeypatch):
    installed = []; handlers = {signal.SIGALRM: object(), signal.SIGPROF: object()}
    monkeypatch.setattr(signal, 'getsignal', lambda s: handlers[s])
    monkeypatch.setattr(signal, 'signal', lambda *args: installed.append(('handler', *args)))
    monkeypatch.setattr(signal, 'getitimer', lambda timer: (300, 0))
    monkeypatch.setattr(signal, 'setitimer', lambda *args: installed.append(('timer', *args)))
    with batch.case_budget(): pass
    assert ('timer', signal.ITIMER_REAL, 180) in installed
    assert ('timer', signal.ITIMER_PROF, 120) in installed
    assert ('handler', signal.SIGALRM, handlers[signal.SIGALRM]) in installed
    assert ('handler', signal.SIGPROF, handlers[signal.SIGPROF]) in installed
    assert any(x[0:2] == ('timer', signal.ITIMER_REAL) and 299 < x[2] <= 300 for x in installed)


def test_truncated_output_never_enters_material_engine(bound, monkeypatch):
    package, gate, _ = bound
    original = batch.material.analyze_case
    seen = []
    def checked(*args, **kwargs):
        index = kwargs['source_binding']['global_schedule_index']
        assert index != 2
        seen.append(index)
        return original(*args, **kwargs)
    monkeypatch.setattr(batch.material, 'analyze_case', checked)
    batch.run_material(package, gate)
    assert set(seen) == {0, 1, 3}


def test_optional_missing_cannot_claim_all_success_diagnostics_complete(bound, monkeypatch):
    package, gate, _ = bound
    monkeypatch.setattr(batch.material, 'MAX_LABELS', 1)
    result = batch.run_material(package, gate)
    assert len(result['cases']) == 3
    assert result['audit']['status'] == 'PASS_WITH_DIAGNOSTIC_RESOURCE_MISSING'
    assert result['audit']['optional_missing_successes'] == 3
    assert result['audit']['diagnostic_computed_successes'] == 0
    assert all(r['aligned_M'] is not None for r in result['cases'])
    assert all(r['aligned_one_to_one_maximum_exact_matches'] is None for r in result['cases'])


def test_critical_late_disagreement_retains_safe_completed_proof(bound, monkeypatch):
    package, gate, _ = bound
    original = batch.old.audit_outcome
    def late(item, *args):
        if item['slot']['global_schedule_index'] == 3:
            raise ValueError('fixture secret raw response must not be in safe proof')
        return original(item, *args)
    monkeypatch.setattr(batch.old, 'audit_outcome', late)
    with pytest.raises(batch.MaterialBatchAuditFailure) as error:
        batch.run_material(package, gate)
    report = error.value.report
    assert report['stage'] == 'case' and report['current_identity']['global_schedule_index'] == 3
    # Sorted providers: first scene anthropic+openai, then second anthropic failure.
    assert len(report['completed_outcome_summaries']) == 2
    assert 'secret raw response' not in str(report) and 'secret raw response' not in str(error.value)
    assert report['historical_changes_permitted'] is False


def test_outer_stage_timeout_retains_completed_proof_without_claiming_count_error(bound, monkeypatch):
    package, gate, _ = bound
    original = batch.old.audit_outcome
    def timeout(item, *args):
        if item['slot']['global_schedule_index'] == 3:
            raise batch.SampleResourceLimit('synthetic stage timer')
        return original(item, *args)
    monkeypatch.setattr(batch.old, 'audit_outcome', timeout)
    with pytest.raises(batch.MaterialBatchAuditFailure) as error:
        batch.run_material(package, gate)
    report = error.value.report
    assert report['status'] == 'NOT_VERIFIED_RESOURCE_LIMIT'
    assert report['critical'] is False
    assert report['code'] == 'MATERIAL_RESOURCE_LIMIT_PARTIAL_PROOF'
    assert len(report['completed_outcome_summaries']) == 2
    assert report['current_identity']['global_schedule_index'] == 3


def test_progress_is_bounded_metadata_only(bound, monkeypatch, capsys):
    package, gate, _ = bound
    monkeypatch.setattr(batch, 'PROGRESS_EVERY', 2)
    batch.run_material(package, gate)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert all('goal011_material_progress' in line and 'input_text' not in line for line in lines)


def test_public_audit_retains_action_evidence_without_geometry_or_raw_response(bound):
    package, gate, _ = bound
    result = batch.run_material(package, gate)
    encoded = batch.old.json_bytes(result['audit'])
    assert b'bounds_xyz_inclusive' not in encoded
    assert b'"input_text"' not in encoded and b'"voxels"' not in encoded
    for outcome in result['audit']['outcome_records']:
        for frame in outcome['frames'].values():
            action = frame['reference_action_audit']
            assert 'references' not in action and 'outputs' not in action
            assert action['action_geometry_arrays_published'] is False
            assert len(action['reference_actions_sha256']) == len(action['predicted_actions_sha256']) == 64
            assert sum(r['operations'] for r in action['unique_literal_material_pairs']) == action['unique_literal_pairs']
    assert result['audit']['summary'] == result['summary']
    assert result['audit']['provenance'] == result['provenance']
    assert result['audit']['pooled_global_confusion'] == result['global_confusion']
