"""Synthetic source binding, staged execution, and truthful metric QA checks."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import socket
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from offline_evaluation.analysis.goal010v2 import pipeline as p
from offline_evaluation.analysis.goal010v2 import metrics
from offline_evaluation.analysis.goal007.audit import AuditError, no_python_network
from offline_evaluation.server_v2.core import Grid, Limits, evaluate_pair, grid_to_payload
from offline_evaluation.server_v2.data import load_scene
from offline_evaluation.server_v2.generation import GenerationError, object_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def descriptor(path, relative_to=None):
    return {'path': str(path.relative_to(relative_to)) if relative_to else str(path),
            'bytes': path.stat().st_size, 'sha256': p.sha(path)}


def scene_package():
    all_scenes = [f'buildings_100_v4_building_{i:03d}' for i in range(100)]
    selected = all_scenes[:60]
    return {'cohort': {'scenes': [{'scene_id': s} for s in all_scenes]},
            'full_manifest': {'identity': {'evaluation_scene_ids': selected}},
            'slots': [{'scene_id': s} for s in selected],
            'descriptions': [{'scene_id': s} for s in selected]}


def test_pilot_selection_uses_bound60_not_full100_cohort():
    package = scene_package()
    assert len(package['cohort']['scenes']) == 100
    order = p.scene_order(package)
    assert len(order) == 60 and order[0] == 'buildings_100_v4_building_034'
    assert set(order) == set(package['full_manifest']['identity']['evaluation_scene_ids'])
    assert order[1:] == sorted(order[1:])


@pytest.mark.parametrize('field', ['slots', 'descriptions'])
def test_pilot_rejects_disagreeing_source_scene_sets(field):
    package = scene_package(); package[field].pop()
    with pytest.raises(ValueError): p.scene_order(package)


@pytest.fixture
def grids():
    g = {(0, 0, 0): 'stone', (0, 1, 0): 'glass', (2, 0, -1): 'stone'}
    pred = {(x + 4, y - 3, z + 2): m for (x, y, z), m in g.items()}
    return g, pred, [-4, 3, -2]


def test_independent_checks_bind_original_strict_metrics_and_fixed_shift(grids):
    g, pred, shift = grids
    original = evaluate_pair(Grid(g), Grid(pred), alignment_mode='strict_translation')
    result = metrics.compute_metrics(g, pred, shift)
    before = deepcopy((g, pred, original))
    proof = p.independent_checks(g, pred, result, shift, original)
    assert proof['aligned']['strict_original_metrics_verified'] is True
    assert proof['aligned']['optional_checks_not_available'] == []
    assert (g, pred, original) == before


@pytest.mark.parametrize('metric', [
    'occupancy_precision', 'conditional_exact_material_accuracy', 'macro_gt_material_recall',
    'footprint_iou', 'column_count_mae', 'common_top_y_mae', 'surface_f1_at_1',
    'surface_recall_at_2', 'multiscale_iou_k2_anchorhalf', 'pred_components6'])
def test_independent_checks_detect_corrupted_metric_families(grids, metric):
    g, pred, shift = grids; result = metrics.compute_metrics(g, pred, shift)
    result['aligned'][metric] += .125
    with pytest.raises(ValueError): p.independent_checks(g, pred, result, shift)


@pytest.mark.parametrize('metric', ['shift_l1', 'shift_l2', 'shift_l1_relative_extent_sum',
                                   'shift_l2_relative_bbox_diagonal', 'occupancy_iou_gain'])
def test_independent_checks_detect_displacement_corruption(grids, metric):
    g, pred, shift = grids; result = metrics.compute_metrics(g, pred, shift)
    result['displacement'][metric] += .25
    with pytest.raises(ValueError): p.independent_checks(g, pred, result, shift)


def test_independent_checks_rejects_unexplained_optional_null(grids):
    g, pred, shift = grids; result = metrics.compute_metrics(g, pred, shift)
    result['aligned']['surface_f1_at_1'] = None
    with pytest.raises(ValueError, match='unexplained'): p.independent_checks(g, pred, result, shift)


def test_independent_checks_reports_resource_null_without_imputation(grids):
    g, pred, shift = grids; result = metrics.compute_metrics(g, pred, shift)
    result['aligned']['surface_f1_at_1'] = None
    result['details']['aligned']['null_reasons']['surface_f1_at_1'] = 'METRIC_RESOURCE_LIMIT:FIXTURE'
    proof = p.independent_checks(g, pred, result, shift)
    assert proof['aligned']['optional_checks_not_available'] == ['surface_f1_at_1']
    assert result['aligned']['surface_f1_at_1'] is None


def test_independent_checks_cannot_relabel_worse_shift_as_strict():
    g = {(0, 0, 0): 'stone'}; shift = [5, 0, 0]
    result = metrics.compute_metrics(g, g, shift)
    with pytest.raises(ValueError, match='worse'): p.independent_checks(g, g, result, shift)


def receipt(root, number, bound, product=None):
    body = {'status': 'PASS', 'stage': number, 'binding': bound,
            'source_package_sha256': p.SOURCE_SHA, 'products': {},
            'result': {'status': 'PASS', 'buildings': {1: 1, 2: 10, 3: 60}.get(number),
                       'phase0_binding': 'PASS', 'definitions_frozen': True}}
    if number in (1, 2, 3):
        body['result'].update(slots={1: 16, 2: 160, 3: 960}[number],
                              baseline_predictions={1: 8, 2: 80, 3: 480}[number])
    if product: body['products'][product.name] = descriptor(product)
    write(root / f'{number}_{p.STAGES[number]}_RECEIPT.json', body)
    return body


def test_previous_enforces_order_products_and_binding(tmp_path):
    bound = {'fixture': 'bound'}
    with pytest.raises(GenerationError): p.previous(tmp_path, 1, bound)
    product = tmp_path / 'source.json'; write(product, {'fixture': 1})
    receipt(tmp_path, 0, bound, product)
    assert len(p.previous(tmp_path, 1, bound)) == 1
    with pytest.raises(GenerationError): p.previous(tmp_path, 2, bound)
    with pytest.raises(ValueError): p.previous(tmp_path, 1, {'fixture': 'different'})
    write(product, {'fixture': 2})
    with pytest.raises(ValueError): p.previous(tmp_path, 1, bound)


def test_previous_rejects_receipt_with_wrong_stage_number(tmp_path):
    bound = {'fixture': 'bound'}; body = receipt(tmp_path, 0, bound)
    body['stage'] = 2; write(tmp_path / '0_source_RECEIPT.json', body)
    with pytest.raises(ValueError): p.previous(tmp_path, 1, bound)


def test_artifact_rejects_traversal_symlink_hash_and_size(tmp_path):
    path = tmp_path / 'safe.json'; write(path, {'safe': 1}); d = descriptor(path, tmp_path)
    assert p.checked_artifact(tmp_path, d) == path
    for change in ({'path': '../safe.json'}, {'bytes': d['bytes'] + 1}, {'sha256': '0' * 64}):
        with pytest.raises(ValueError): p.checked_artifact(tmp_path, {**d, **change})
    link = tmp_path / 'link.json'; link.symlink_to(path)
    with pytest.raises(AuditError): p.checked_artifact(tmp_path, {**d, 'path': link.name})


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    """One complete16-slot/8-baseline source, only fabricated files and voxels."""
    directory = tmp_path / 'buildings_100_v4/building_034'; (directory / 'gt').mkdir(parents=True)
    bbox = {'xmin': -3, 'xmax': 2, 'ymin': 2, 'ymax': 5, 'zmin': 3, 'zmax': 7,
            'voxel_axis_order': 'Y,X,Z'}
    array = np.full((4, 6, 5), 'air', dtype='<U10')
    array[1, 1, 1] = 'stone'; array[1, 3, 1] = 'glass'
    np.save(directory / 'gt/voxels.npy', array, allow_pickle=False); write(directory / 'gt/bbox.json', bbox)
    scene = load_scene(directory); scene_id = scene.scene_id
    gt = {'building_directory': str(directory), 'voxels_sha256': p.sha(directory / 'gt/voxels.npy'),
          'bbox_sha256': p.sha(directory / 'gt/bbox.json'), 'grid_sha256': object_hash(scene.grid.to_dict())}
    pred = Grid(dict(scene.grid.voxels)); ops = [{'op': 'set', 'x': c[0], 'y': c[1], 'z': c[2], 'block': m}
                                               for c, m in sorted(pred.voxels.items())]
    old_metrics = evaluate_pair(scene.grid, pred, alignment_mode='strict_translation')
    enabled = tmp_path / 'enabled'; enabled.mkdir()
    schedule = []; slots = []; entries = []
    for condition in ('O', 'J', 'P', 'E'):
        for desc in (('openai', 'anthropic', 'gemini') if condition in ('O', 'J') else (None,)):
            text = json.dumps({'fixture': [condition, desc]}); digest = hashlib.sha256(text.encode()).hexdigest()
            for builder in ('openai', 'anthropic'):
                idx = len(schedule); attempt = f'fixture{idx}'; status = 'failed' if idx == 0 else 'operational_unmeasurable' if idx == 6 else 'success'
                record = {'schedule_index': idx, 'scene_id': scene_id, 'provider': builder,
                          'condition': condition, 'description_model': desc, 'input_id': digest,
                          'input_text': text, 'input_sha256': digest}
                schedule.append(record); artifacts = {}
                if status == 'success':
                    for name, value in [('parsed.json', {'operations': ops}), ('voxels.json', grid_to_payload(pred))]:
                        path = enabled / 'attempts' / attempt / name; write(path, value); artifacts[name] = descriptor(path, enabled)
                entry = {'attempt_id': attempt, 'schedule_index': idx, 'provider': builder, 'input_id': digest,
                         'status': 'success' if status == 'success' else 'failed', 'artifacts': artifacts}
                entries.append(entry)
                slots.append({**record, 'global_schedule_index': idx, 'attempt_id': attempt,
                    'original_attempt_id': attempt, 'selection_reason': 'fixture', 'source_label': 'fixture',
                    'source_run': str(enabled), 'source_entry_hash': object_hash(entry),
                    'source_gt_descriptor_hash': object_hash(gt), 'gt_grid_sha256': gt['grid_sha256'],
                    'status': status, 'generation_status': entry['status'], 'artifacts': artifacts,
                    'metrics': deepcopy(old_metrics) if status == 'success' else None,
                    'failure_kind': 'truncation' if idx == 0 else None, 'usage_known': idx != 6,
                    'completed_request_outcome': idx != 6, 'http_status_code': 520 if idx == 6 else 200,
                    'final_voxel_count': len(pred.voxels) if status == 'success' else None})
    identity = {'schedule': schedule, 'enabled_directory': str(enabled), 'limits': asdict(Limits()), 'gt_by_scene': {scene_id: gt}}
    plan = {'identity': identity, 'identity_hash': object_hash(identity)}
    for row in slots: row['source_plan_identity'] = plan['identity_hash']
    ledger = enabled / 'ledger.json'; write(ledger, {'entries': entries})
    package = scene_package(); package.update(slots=slots, descriptions=[{'scene_id': scene_id, 'gt_descriptor': gt}],
        plans={'fixture': plan}, goal007_manifest={'sources': {'fixture': {'ledger_path': str(ledger), 'enabled_directory': str(enabled)}}})
    package['full_manifest']['identity']['full_schedule'] = schedule
    oldrun = tmp_path / 'oldrun'; prediction_root = oldrun / 'phase1_baselines/predictions'; prediction_root.mkdir(parents=True)
    bases = []
    for condition in ('O', 'P'):
        for desc in (('openai', 'anthropic', 'gemini') if condition == 'O' else (None,)):
            record = next(x for x in schedule if x['condition'] == condition and x['description_model'] == desc)
            for shape in ('filled_box', 'six_face_shell'):
                i = len(bases); bpred = Grid({(0, 0, 0): 'stone', (1, 0, 0): 'stone'})
                actions = {'operations': [{'op': 'fill', 'x1': 0, 'y1': 0, 'z1': 0, 'x2': 1, 'y2': 0, 'z2': 0, 'block': 'stone'}]}
                ap = prediction_root / f'b{i}.actions.json'; cp = prediction_root / f'b{i}.xyz.npy'; write(ap, actions)
                np.save(cp, np.asarray(sorted(bpred.voxels), dtype=np.int32), allow_pickle=False)
                scores = evaluate_pair(scene.grid, bpred, alignment_mode='strict_translation')
                bases.append({'prediction_id': f'base{i}', 'scene_id': scene_id, 'condition': condition,
                    'description_model': desc, 'input_id': record['input_id'], 'input_sha256': record['input_sha256'],
                    'source_sha256': record['input_sha256'], 'structure': shape, 'gt_grid_sha256': gt['grid_sha256'],
                    'gt_bbox_sha256': gt['bbox_sha256'], 'gt_voxels_sha256': gt['voxels_sha256'], 'size_rank_stratum': 0,
                    'status': 'success', 'prediction_voxels': len(bpred.voxels), 'alignment': scores['alignment'],
                    'absolute_occupancy_iou': scores['absolute']['occupancy_iou'], 'aligned_occupancy_iou': scores['aligned']['occupancy_iou'],
                    'artifacts': {'actions': descriptor(ap, prediction_root), 'coordinates': descriptor(cp, prediction_root)}})
    write(oldrun / 'phase1_baselines/baseline_per_building.json', bases)
    report = tmp_path / 'newreports'; report.mkdir(); write(report / 'METRIC_REGISTRY.json', metrics.metric_registry())
    monkeypatch.setattr(p, 'REPORT', report)
    monkeypatch.setattr(p, 'scene_order', lambda package: [scene_id])
    return package, oldrun, {'hashes': {'fixture.py': 'a' * 64}, 'specification_sha256': 'b' * 64}, tmp_path


@pytest.mark.parametrize('field,value', [('input_sha256', 'f' * 64), ('source_entry_hash', 'f' * 64),
                                       ('source_plan_identity', 'f' * 64), ('source_gt_descriptor_hash', 'f' * 64)])
def test_slot_source_rejects_provenance_corruption(source_fixture, field, value):
    package, _, _, _ = source_fixture; row = deepcopy(package['slots'][1]); row[field] = value
    with pytest.raises(ValueError): p.slot_source(package, row)


def test_complete_synthetic_stage_preserves16_slots8_baselines_and_sources(source_fixture):
    package, oldrun, bound, tmp_path = source_fixture
    before = deepcopy(package); inventory = {str(f): p.sha(f) for f in tmp_path.rglob('*') if f.is_file()}
    out = tmp_path / 'output'; out.mkdir(); report = p.evaluate_stage(package, oldrun, out, 1, bound)
    rows = p.guard.read_json(out / 'pilot_metric_values.json'); bases = p.guard.read_json(out / 'baseline_new_metric_values.json')
    assert len(rows) == 16 and len(bases) == 8
    assert report['original_valid_count'] == report['original_valid_metrics_checked'] == 14
    assert report['baseline_original_IoUs_checked'] == 8
    assert report['all_surface_footprint_multiscale_independently_checked'] is True
    assert rows[0]['failure0_eligible'] is True and rows[6]['failure0_eligible'] is False
    assert all(v is None for i in (0, 6) for frame in rows[i]['metrics'].values() for v in frame.values())
    for row in bases:
        for frame in ('absolute', 'aligned'):
            assert all(row['metrics'][frame][k] is None for k in metrics.MATERIAL_METRIC_IDS)
    assert package == before
    assert {name: p.sha(Path(name)) for name in inventory} == inventory


def test_resource_limited_stage_keeps_nulls_and_cannot_claim_all_checks(source_fixture, monkeypatch):
    package, oldrun, bound, tmp_path = source_fixture
    def limited(*args, **kwargs): raise metrics.MetricResourceLimit('FIXTURE_LIMIT')
    monkeypatch.setattr(p, 'evaluate', limited)
    out = tmp_path / 'resource_output'; out.mkdir(); report = p.evaluate_stage(package, oldrun, out, 1, bound)
    assert report['original_valid_metrics_checked'] == report['baseline_original_IoUs_checked'] == 0
    assert report['all_original_valid_metrics_checked'] is False
    assert report['all_baseline_original_IoUs_identical'] is False
    assert report['all_surface_footprint_multiscale_independently_checked'] is False
    rows = p.guard.read_json(out / 'pilot_metric_values.json')
    assert rows[1]['status'] == 'success' and rows[1]['metric_status'] == 'RESOURCE_LIMIT'
    assert rows[1]['failure0_eligible'] is False
    assert all(value is None for frame in rows[1]['metrics'].values() for value in frame.values())


def capacity_fixture(tmp_path, *, cap=32768, count=120):
    source = tmp_path / 'old/phase3_exact_oracle/E_capacity_diagnostics.csv'; source.parent.mkdir(parents=True)
    fields = ['provider', 'gt_occupied_voxels', 'reference_operations', 'spec_utf8_bytes',
              'output_tokens_observed', 'output_token_cap', 'outcome_category']
    with source.open('w') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for i in range(count):
            writer.writerow(dict(provider='openai' if i < 60 else 'anthropic', gt_occupied_voxels=100 + i,
                reference_operations=5 + i, spec_utf8_bytes=50 + i, output_tokens_observed=cap if i % 6 == 0 else 1000,
                output_token_cap=cap, outcome_category='truncation' if i % 6 == 0 else 'success'))
    out = tmp_path / 'out'; out.mkdir()
    return source.parent.parent, out


def test_capacity_uses_observed120_and_never_rewrites_csv(tmp_path):
    oldrun, out = capacity_fixture(tmp_path); source = oldrun / 'phase3_exact_oracle/E_capacity_diagnostics.csv'; before = p.sha(source)
    p.capacity_diagnostic(oldrun, out)
    tree = ET.parse(out / 'E_capacity_diagnostic.svg')
    assert len(tree.findall('.//{http://www.w3.org/2000/svg}circle')) == 360
    receipt = p.guard.read_json(out / 'E_capacity_plot_receipt.json')
    assert receipt['outcomes'] == 120 and receipt['model_queries'] == 0
    assert receipt['completion_counterfactuals_computed'] is False and p.sha(source) == before


@pytest.mark.parametrize('kwargs', [{'count': 119}, {'cap': 16384}])
def test_capacity_rejects_changed_count_or_output_cap(tmp_path, kwargs):
    oldrun, out = capacity_fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError): p.capacity_diagnostic(oldrun, out)


def test_previous_retains_every_prior_receipt_hash(tmp_path):
    bound = {'fixture': 'bound'}
    for number in range(4): receipt(tmp_path, number, bound)
    expected = {f'{i}_{p.STAGES[i]}_RECEIPT.json': p.sha(tmp_path / f'{i}_{p.STAGES[i]}_RECEIPT.json')
                for i in range(4)}
    assert p.previous(tmp_path, 4, bound) == expected


@pytest.mark.parametrize('number', [1, 2, 3])
def test_previous_rejects_wrong_pilot_or_full_scope(tmp_path, number):
    bound = {'fixture': 'bound'}
    for i in range(number + 1): receipt(tmp_path, i, bound)
    path = tmp_path / f'{number}_{p.STAGES[number]}_RECEIPT.json'
    value = p.guard.read_json(path); value['result']['slots'] -= 1; write(path, value)
    with pytest.raises(ValueError, match='scope'): p.previous(tmp_path, number + 1, bound)


@pytest.fixture
def orchestration_fixture(tmp_path, monkeypatch):
    """No source reads: run real orchestration with fabricated bound packages."""
    target = tmp_path / 'newrun'; package = scene_package()
    bound = {'specification_sha256': 'b' * 64, 'hashes': {'fixture': 'a' * 64}}
    state = {'active': False, 'events': [], 'exit_failure': False, 'source_mismatch': False}
    monkeypatch.setattr(p, 'safe_root', lambda value, stage: target)
    monkeypatch.setattr(p, 'binding', lambda *args: bound)
    monkeypatch.setattr(p.resource, 'setrlimit', lambda *args: None)
    monkeypatch.setattr(p, 'SOURCE_SHA', p.oldpipe.digest(package))

    def assert_active():
        assert state['active']
        with pytest.raises(AuditError, match='python_network_disabled'):
            socket.socket.connect(None, ('fixture.invalid', 1))

    @contextmanager
    def locked():
        with no_python_network():
            state['active'] = True; state['events'].append('process_audit_enter')
            try:
                yield ['fixture-readonly-lock']
                state['events'].append('process_audit_exit')
                assert not (target / '0_source_RECEIPT.json').exists()
                if state['exit_failure']: raise ValueError('fixture exit process audit failed')
            finally: state['active'] = False

    def load():
        assert_active(); state['events'].append('load_verified')
        return {} if state['source_mismatch'] else deepcopy(package)

    def verify_old():
        assert_active(); state['events'].append('verify_old'); return tmp_path / 'immutable_old'

    def package_receipt(value):
        assert_active(); assert value == package
        state['events'].append('compute_phase0_receipt'); return {'fixture': True}

    monkeypatch.setattr(p.guard, 'locked_sources', locked)
    monkeypatch.setattr(p.guard, 'load_verified_sources', load)
    monkeypatch.setattr(p, 'verify_old', verify_old)
    monkeypatch.setattr(p.oldpipe, 'package_receipt', package_receipt)
    return target, state


def test_run_phase0_holds_network_and_locks_until_exit_audit(orchestration_fixture):
    target, state = orchestration_fixture
    result = p.run(0, str(target), 'fixture-code', 'fixture-spec')
    assert result['status'] == 'PASS' and result['source_lock_count'] == 1
    assert result['OS_level_isolation_guaranteed'] is False
    assert state['events'] == ['process_audit_enter', 'load_verified', 'verify_old',
        'compute_phase0_receipt', 'load_verified', 'verify_old', 'process_audit_exit']
    assert not state['active'] and (target / '0_source_RECEIPT.json').is_file()
    assert not (target / '0_STOP.json').exists()


def test_run_source_mismatch_creates_no_output(orchestration_fixture):
    target, state = orchestration_fixture; state['source_mismatch'] = True
    with pytest.raises(ValueError, match='source package changed'):
        p.run(0, str(target), 'fixture-code', 'fixture-spec')
    assert not target.exists() and not state['active']


def test_run_exit_process_audit_failure_never_writes_pass(orchestration_fixture):
    target, state = orchestration_fixture; state['exit_failure'] = True
    with pytest.raises(ValueError, match='exit process audit'):
        p.run(0, str(target), 'fixture-code', 'fixture-spec')
    assert not (target / '0_source_RECEIPT.json').exists()
    assert p.guard.read_json(target / '0_STOP.json')['status'] == 'STOP'
    assert (target / '0_source/SOURCE_BINDINGS.json').is_file() and not state['active']


def test_run_never_overwrites_partial_stage(orchestration_fixture):
    target, state = orchestration_fixture
    partial = target / '0_source/retained.json'; write(partial, {'retained': True}); before = p.sha(partial)
    with pytest.raises(FileExistsError): p.run(0, str(target), 'fixture-code', 'fixture-spec')
    assert p.sha(partial) == before and state['events'] == []


def test_nested_resource_budget_preserves_outer_remaining_time(monkeypatch):
    timers = {p.signal.ITIMER_REAL: (0, 0), p.signal.ITIMER_PROF: (0, 0)}
    handlers = {p.signal.SIGALRM: 'original-wall', p.signal.SIGPROF: 'original-cpu'}
    clock = {'wall': 100., 'cpu': 10.}
    monkeypatch.setattr(p.signal, 'getsignal', lambda key: handlers[key])
    monkeypatch.setattr(p.signal, 'signal', lambda key, value: handlers.__setitem__(key, value))
    monkeypatch.setattr(p.signal, 'getitimer', lambda key: timers[key])
    monkeypatch.setattr(p.signal, 'setitimer', lambda key, value, interval=0: timers.__setitem__(key, (value, interval)))
    monkeypatch.setattr(p.time, 'monotonic', lambda: clock['wall'])
    monkeypatch.setattr(p.time, 'process_time', lambda: clock['cpu'])
    with p.sample_budget(3600, 7200):
        outer_handlers = dict(handlers)
        with pytest.raises(RuntimeError):
            with p.sample_budget(180, 240):
                assert timers[p.signal.ITIMER_REAL][0] == 240
                assert timers[p.signal.ITIMER_PROF][0] == 180
                clock.update(wall=106., cpu=12.)
                raise RuntimeError('fixture body error')
        assert timers[p.signal.ITIMER_REAL][0] == 7194
        assert timers[p.signal.ITIMER_PROF][0] == 3598
        assert handlers == outer_handlers
    assert all(value[0] == 0 for value in timers.values())
    assert handlers == {p.signal.SIGALRM: 'original-wall', p.signal.SIGPROF: 'original-cpu'}
