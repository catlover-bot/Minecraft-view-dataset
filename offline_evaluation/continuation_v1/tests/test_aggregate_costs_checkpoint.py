"""Offline-only denominators, incremental metrics and immutable save/retry guards."""
from copy import deepcopy
from datetime import datetime, timezone
import base64
import json
from pathlib import Path
import socket
import subprocess

import pytest

from offline_evaluation.continuation_v1 import aggregate, checkpoint, costs
from offline_evaluation.server_v2 import analysis_v4
from offline_evaluation.server_v2.core import Grid, grid_to_payload
from offline_evaluation.server_v2.generation import GenerationError, object_hash, digest, write_json, read_json
from offline_evaluation.server_v2.data import sha256_file
from offline_evaluation.server_v2.io_utils import ROOT


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('live network forbidden'))


@pytest.fixture
def joined_fixture(tmp_path):
    prices = read_json(ROOT / 'reports/server_only/goal005/official_pricing.json')
    prices = {p: prices[p] for p in ('openai', 'anthropic')}
    scenes = [f'fixture_{n:03d}' for n in range(60)]
    gt = Grid({(0, 0, 0): 'stone'})
    descriptors = {s: {'grid_sha256': object_hash(grid_to_payload(gt))} for s in scenes}
    full = []
    for scene in scenes:
        for condition, description in [(c, d) for c in 'OJ' for d in ('openai', 'anthropic', 'gemini')] + [('P', None), ('E', None)]:
            for provider in ('openai', 'anthropic'):
                text = scene + condition + str(description)
                full.append({'provider': provider, 'input_id': text, 'scene_id': scene, 'condition': condition,
                             'description_model': description, 'schedule_index': len(full), 'input_text': text,
                             'input_sha256': digest(text.encode()), 'protocol_id': 'fixed_semantic',
                             'prompt_sha256': 'prompt', 'schema_sha256': 'schema', 'model': prices[provider]['model']})
    mi = {'protocol_id': 'fixed_semantic', 'full_schedule': full, 'evaluation_scene_ids': scenes,
          'gt_by_scene': descriptors, 'prompt_sha256': 'prompt', 'schema_sha256': 'schema'}
    manifest = {'identity': mi, 'identity_hash': object_hash(mi)}
    def make_plan(schedule, protocol):
        identity = {'protocol_id': protocol, 'semantic_protocol_id': 'fixed_semantic', 'schedule': schedule,
                    'manifest_identity_hash': manifest['identity_hash'], 'prompt_sha256': 'prompt', 'schema_sha256': 'schema',
                    'gt_by_scene': {r['scene_id']: descriptors[r['scene_id']] for r in schedule}, 'pricing': prices}
        return {'identity': identity, 'identity_hash': object_hash(identity)}
    parent = make_plan(full[:160], 'old_operational'); plan = make_plan(full[160:], 'new_operational')
    plan['identity']['parent'] = {'plan_identity_hash': parent['identity_hash']}
    plan['identity_hash'] = object_hash(plan['identity'])
    def outcomes(p):
        ledger, rows = {'entries': []}, []
        for rec in p['identity']['schedule']:
            entry = {'attempt_id': f"{rec['schedule_index']:032x}", 'schedule_index': rec['schedule_index'],
                     'provider': rec['provider'], 'input_id': rec['input_id'], 'status': 'success', 'failure_kind': None,
                     'response_id': 'fixture', 'response_model': prices[rec['provider']]['accepted_response_model_ids'][0],
                     'usage': {'input_tokens': 1, 'output_tokens': 2}, 'artifacts': {'raw.json': {}},
                     'started_at_utc': '2026-10-14T00:00:00Z', 'finished_at_utc': '2026-10-14T00:00:01Z'}
            ledger['entries'].append(entry)
            row = aggregate._row(p['identity'], rec, entry)
            row['metrics'] = {'metric_version': 'server_v2.2_strict_translation',
                              'absolute': {'occupancy_iou': .5, 'material_aware_iou': .25, 'dimension_mae': 0},
                              'aligned': {'occupancy_iou': .5, 'material_aware_iou': .25, 'dimension_mae': 0}}
            rows.append(row)
        return ledger, {'plan_identity_hash': p['identity_hash'], 'rows': rows}
    pl, pe = outcomes(parent); ledger, evaluation = outcomes(plan)
    return {'plan': plan, 'parent': parent, 'manifest': manifest, 'pl': pl, 'pe': pe,
            'ledger': ledger, 'evaluation': evaluation, 'gt': gt}


def joined(f, ledger=None, evaluation=None):
    return aggregate.aggregate_verified(f['plan'], f['parent'], f['pe'],
        f['evaluation'] if evaluation is None else evaluation, f['pl'],
        f['ledger'] if ledger is None else ledger, f['manifest'])


def test_interim160_preserves800_not_run_and_no_inference(joined_fixture):
    f = joined_fixture
    result = joined(f, {'entries': []}, {'plan_identity_hash': f['plan']['identity_hash'], 'rows': []})
    assert len(result['rows']) == 960 and result['summary']['not_run_inputs'] == 800
    assert result['summary']['families'] == [] and not result['summary']['fixed60_complete']
    assert sum(analysis_v4.bounded_all_attempted_value(r) is None for r in result['rows']) == 800
    assert len(result['summary']['E_common_valid_scenes']) == 10
    assert {r['source_run_protocol_id'] for r in result['rows']} == {'old_operational', 'new_operational'}


def test_complete960_uses_frozen_families_building_denominators(joined_fixture, monkeypatch):
    original = analysis_v4.summarize_v4
    monkeypatch.setattr(analysis_v4, 'summarize_v4', lambda rows, smoke: original(rows, smoke=smoke, bootstrap_replicates=3, signflip_replicates=8))
    result = joined(joined_fixture)['summary']
    assert result['fixed60_complete'] and [f['predeclared_size'] for f in result['families']] == [8, 4, 8]
    assert all(r['buildings'] == 60 for f in result['families'] for r in f['results'])
    assert all(g['independent_buildings'] == 60 for g in result['groups'])


@pytest.mark.parametrize('field,value', [('provider', 'wrong'), ('input_id', 'wrong'), ('gt_grid_sha256', 'wrong'),
                                        ('prompt_sha256', 'wrong'), ('schema_sha256', 'wrong'), ('semantic_protocol_id', 'wrong')])
def test_wrong_join_identity_rejected(joined_fixture, field, value):
    f = joined_fixture; f['evaluation']['rows'][0][field] = value
    with pytest.raises(GenerationError):
        joined(f)


def test_unknown_outcomes_never_failure0_and_failed_mae_null(joined_fixture):
    f = joined_fixture
    for idx, failure in ((0, 'truncation'), (1, 'timeout')):
        e = f['ledger']['entries'][idx]; r = f['evaluation']['rows'][idx]
        e.update(status='failed', failure_kind=failure)
        r.update(status='failed', failure_kind=failure, metrics=None)
        if failure == 'timeout':
            e['usage'] = None; r['completed_request_outcome'] = False
    result = joined(f)
    a, b = result['rows'][160:162]
    assert analysis_v4.bounded_all_attempted_value(a) == 0
    assert analysis_v4.bounded_all_attempted_value(b) is None
    assert analysis_v4._metric(a, 'aligned_dimension_mae') is None
    assert not result['summary']['fixed60_complete'] and result['summary']['families'] == []
    bounded = [r for r in result['summary']['bounded_IoU_denominators'] if r['condition'] == 'O' and r['metric'] == 'aligned_material_aware_iou']
    assert {r['provider']: r['all_attempted_failure0_rows'] for r in bounded} == {'openai': 180, 'anthropic': 179}
    assert {r['provider']: r['valid_only']['valid_rows'] for r in bounded} == {'openai': 179, 'anthropic': 179}


def test_duplicate_provider_input_row_rejected(joined_fixture):
    f = joined_fixture; f['evaluation']['rows'].append(deepcopy(f['evaluation']['rows'][0]))
    with pytest.raises(GenerationError, match='duplicate_provider_input'):
        joined(f)


def test_incremental_metric_cache_does_not_repeat_alignment(joined_fixture, tmp_path, monkeypatch):
    from offline_evaluation.continuation_v1 import runner
    f = joined_fixture; plan = f['plan']; directory = tmp_path / 'enabled'; directory.mkdir()
    record = plan['identity']['schedule'][0]; entry = deepcopy(f['ledger']['entries'][0])
    data = grid_to_payload(f['gt']); path = directory / 'voxels.json'; write_json(path, data)
    entry['artifacts']['voxels.json'] = {'path': 'voxels.json', 'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    ledger = {'entries': [entry]}
    monkeypatch.setattr(runner, 'execution_summary', lambda p, l: {'attempted_inputs': len(l['entries'])})
    calls = []
    original = aggregate.evaluate_pair
    monkeypatch.setattr(aggregate, 'evaluate_pair', lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    grids = {record['scene_id']: f['gt']}
    first = aggregate.evaluate_verified(plan, directory, ledger, grids)
    second = aggregate.evaluate_verified(plan, directory, ledger, grids)
    assert len(calls) == 1 and first['metrics_computed_this_call'] == 1
    assert second['metrics_computed_this_call'] == 0 and second['metrics_reused_this_call'] == 1
    ledger['entries'][0]['response_model'] = 'changed'
    with pytest.raises(GenerationError, match='previous_evaluated_attempt_changed'):
        aggregate.evaluate_verified(plan, directory, ledger, grids)


def test_forecast_weighting_and_pause_exclusion():
    schedule = [{'provider': 'openai', 'condition': 'O', 'description_model': 'openai', 'input_id': str(i), 'schedule_index': i} for i in range(2)]
    es = [{'provider': 'openai', 'input_id': str(i), 'schedule_index': i, 'status': 'success', 'usage': {},
           'charged_input_tokens': 10, 'charged_output_tokens': 20, 'charged_cost_usd': .01,
           'started_at_utc': f'2026-10-14T0{hour}:00:00Z', 'finished_at_utc': f'2026-10-14T0{hour}:00:10Z'} for i, hour in enumerate((0, 9))]
    suffix = [{**schedule[0], 'input_id': 'new' + str(i)} for i in range(6)]
    pricing = {'openai': {'context_window_tokens': 400000, 'input_usd_per_million': .25, 'output_usd_per_million': 2}}
    report = costs.forecast(schedule, {'entries': es}, suffix, pricing, {'tokens': 100, 'cost_usd_upper_bound': 4})
    assert report['projected_remaining_cost_usd'] == .06 and report['projected_total_cost_usd'] == 4.06
    assert report['projected_remaining_active_seconds'] == 60
    assert report['observed_time']['classified_pause_seconds'] == 32390
    assert report['reservation']['cumulative_max_tokens'] == 100 + 6 * 432768
    with pytest.raises(GenerationError, match='overlaps'):
        costs.forecast(schedule, {'entries': es}, schedule, pricing, {'tokens': 100, 'cost_usd_upper_bound': 4})


def _ciphertext_with_pattern():
    head = base64.urlsafe_b64encode(b'\x80' + int(datetime(2026, 10, 14, tzinfo=timezone.utc).timestamp()).to_bytes(8, 'big') + b'\x00' * 160).decode().rstrip('=')
    value = head[:100] + '_sk-' + 'A' * 40 + head[144:]
    return value + '=' * (-len(value) % 4)


def test_reasoning_ciphertext_scanner_narrow_exception_preserves_raw(tmp_path):
    path = tmp_path / 'raw.json'
    obj = {'model': 'gpt-5-mini-2025-08-07', 'output': [{'type': 'reasoning', 'encrypted_content': _ciphertext_with_pattern()}]}
    write_json(path, obj); before = path.read_bytes()
    scope = {'provider': 'openai', 'sha256': sha256_file(path), 'accepted_models': [obj['model']]}
    exceptions = checkpoint.scan_public_file(path, scope)
    assert len(exceptions) == 1 and exceptions[0]['field'] == 'output[0].encrypted_content' and path.read_bytes() == before
    obj['bad'] = 'sk-' + 'z' * 40; write_json(path, obj); scope['sha256'] = sha256_file(path)
    with pytest.raises(ValueError, match='outside'):
        checkpoint.scan_public_file(path, scope)
    scope['provider'] = 'anthropic'
    with pytest.raises(ValueError):
        checkpoint.scan_public_file(path, scope)


def test_checkpoint_archive_roundtrip_and_secret_rejection(tmp_path):
    source = tmp_path / 'source.json'; write_json(source, {'raw': 'fixture'})
    target = tmp_path / 'save'; target.mkdir()
    archive = target / 'fixture.tar.gz'
    manifest = checkpoint._archive(archive, [('continuation/source.json', source)], {'scope': 'fixture'}, {})
    result = checkpoint.restore_archive(archive, tmp_path / 'restored', sha256_file(archive))
    assert result['files'] == 1 and manifest['files'][0]['sha256'] == sha256_file(source)
    assert (tmp_path / 'restored/continuation/source.json').read_bytes() == source.read_bytes()
    secret = tmp_path / 'secret.json'; write_json(secret, {'bad': 'sk-' + 'z' * 40})
    with pytest.raises(ValueError):
        checkpoint._archive(target / 'bad.tar.gz', [('secret.json', secret)], {}, {})


def test_upload_retry_never_calls_generator_on_changed_package(tmp_path, monkeypatch):
    from offline_evaluation.continuation_v1 import runner
    monkeypatch.setattr(runner, 'run', lambda *a, **k: pytest.fail('upload retry called generator'))
    monkeypatch.setattr(checkpoint, '_gh', lambda *a, **k: pytest.fail('tampered package reached network'))
    asset = tmp_path / 'asset'; asset.write_bytes(b'changed')
    write_json(tmp_path / 'checkpoint.json', {'version': 'goal006_checkpoint_package_v1', 'repository': checkpoint.REPOSITORY,
        'tag': 'checkpoint-goal006-fixture', 'target_commitish': 'a' * 40,
        'assets': [{'name': 'asset', 'sha256': 'wrong', 'bytes': 7}]})
    with pytest.raises(GenerationError, match='package_changed'):
        checkpoint.upload(tmp_path)


def _upload_package(tmp_path):
    package_dir = tmp_path / 'package'; package_dir.mkdir()
    planfile = tmp_path / 'plan.json'
    write_json(planfile, {'identity': {'plan_path': '/root/minecraft-runs/fixture-canonical/plan.json'}})
    archive = package_dir / 'continuation-point-in-time.tar.gz'
    checkpoint._archive(archive, [('continuation/plan.json', planfile)], {'fixture_only': True}, {})
    for name in ('snapshot.json', 'SOURCE_REFERENCE.json'):
        write_json(package_dir / name, {'fixture_only': True})
    (package_dir / 'RESTORE.md').write_text('Offline fixture restore.')
    names = [archive.name, 'FILES.json', 'snapshot.json', 'SOURCE_REFERENCE.json', 'RESTORE.md']
    (package_dir / 'SHA256SUMS').write_text(''.join(f'{sha256_file(package_dir / n)}  {n}\n' for n in names))
    names.append('SHA256SUMS')
    write_json(package_dir / 'checkpoint.json', {'version': 'goal006_checkpoint_package_v1', 'repository': checkpoint.REPOSITORY,
        'tag': 'checkpoint-goal006-fixture', 'checkpoint_id': 'goal006-fixture', 'target_commitish': 'a' * 40,
        'archive': archive.name, 'assets': [{'name': n, 'sha256': sha256_file(package_dir / n),
                                           'bytes': (package_dir / n).stat().st_size} for n in names]})
    return package_dir


def test_successful_upload_retry_preserves_assets_and_never_generates(tmp_path, monkeypatch):
    from offline_evaluation.continuation_v1 import runner
    monkeypatch.setattr(runner, 'run', lambda *a, **k: pytest.fail('GitHub retry invoked generation'))
    package = _upload_package(tmp_path)
    state = {'release': None, 'uploads': [], 'bytes': {}, 'failed': False}
    def gh(args, check=True):
        result = lambda value: subprocess.CompletedProcess(args, 0, json.dumps(value), '')
        if args[0] == 'api' and '/compare/' in args[-1]:
            return result({'status': 'ahead', 'behind_by': 0})
        if args[:3] == ['api', '--method', 'POST']:
            state['release'] = {'id': 17, 'draft': True, 'target_commitish': 'a' * 40,
                                'html_url': 'https://github.com/fixture/draft', 'assets': []}
            return result(state['release'])
        if args[0] == 'api':
            if state['release'] is None:
                return subprocess.CompletedProcess(args, 1, '', 'HTTP 404')
            return result(state['release'])
        if args[:2] == ['release', 'upload']:
            path = Path(args[3])
            if state['uploads'] and not state['failed']:
                state['failed'] = True
                raise GenerationError('fixture GitHub interruption; no generation')
            state['uploads'].append(path.name); state['bytes'][path.name] = path.read_bytes()
            state['release']['assets'].append({'id': len(state['uploads']), 'name': path.name,
                                               'size': path.stat().st_size, 'digest': 'sha256:' + sha256_file(path)})
            return result({})
        if args[:2] == ['release', 'download']:
            name = args[args.index('--pattern') + 1]; directory = Path(args[args.index('--dir') + 1])
            (directory / name).write_bytes(state['bytes'][name]); return result({})
        pytest.fail('Unexpected fixture GitHub command')
    monkeypatch.setattr(checkpoint, '_gh', gh)
    with pytest.raises(GenerationError, match='interruption'):
        checkpoint.upload(package)
    result = checkpoint.upload(package)
    assert result['status'] == 'GITHUB_VERIFIED' and result['release_id'] == 17
    assert len(state['uploads']) == 6 and len(set(state['uploads'])) == 6
    assert result['target_commit_on_dedicated_branch'] and 'canonical' in result['copied_plan_load_rejection']
    assert result['research_api_requests_sent_by_upload'] == 0


def test_unpushed_checkpoint_commit_blocks_draft_creation(tmp_path, monkeypatch):
    package = _upload_package(tmp_path); commands = []
    def gh(args, check=True):
        commands.append(args)
        assert '/compare/' in args[-1]
        return subprocess.CompletedProcess(args, 0, json.dumps({'status': 'diverged', 'behind_by': 1}), '')
    monkeypatch.setattr(checkpoint, '_gh', gh)
    with pytest.raises(GenerationError, match='not_pushed'):
        checkpoint.upload(package)
    assert len(commands) == 1


def test_saved_stage_and_completed_outcome_cannot_override_ledger(joined_fixture):
    f = joined_fixture
    f['evaluation']['rows'][0]['schema_success'] = True
    with pytest.raises(GenerationError, match='stage_flag'):
        joined(f)
    f['evaluation']['rows'][0]['schema_success'] = None
    f['evaluation']['rows'][0]['completed_request_outcome'] = False
    with pytest.raises(GenerationError, match='completed_response'):
        joined(f)


def test_disabled_collect_loads_string_manifest_and_parent_paths(joined_fixture, tmp_path, monkeypatch):
    """Exercise production JSON path types instead of supplying in-memory manifest."""
    from contextlib import nullcontext
    from offline_evaluation.continuation_v1 import plan as planning, runner
    f = joined_fixture
    manifest_path = tmp_path / 'manifest.json'; write_json(manifest_path, f['manifest'])
    parent_dir = tmp_path / 'parent'; parent_dir.mkdir()
    parent_path = parent_dir / 'plan.json'
    f['parent']['identity']['enabled_directory'] = str(parent_dir)
    f['parent']['identity_hash'] = object_hash(f['parent']['identity'])
    f['pe']['plan_identity_hash'] = f['parent']['identity_hash']
    write_json(parent_path, f['parent']); write_json(parent_dir / 'evaluation.json', f['pe'])
    write_json(parent_dir / 'ledger.json', f['pl'])
    enabled = tmp_path / 'not-enabled'
    f['plan']['identity'].update(manifest_path=str(manifest_path), enabled_directory=str(enabled))
    f['plan']['identity']['parent'].update(plan_path=str(parent_path), enabled_directory=str(parent_dir),
                                           plan_identity_hash=f['parent']['identity_hash'])
    f['plan']['identity_hash'] = object_hash(f['plan']['identity'])
    monkeypatch.setattr(planning, 'load_plan', lambda _: f['plan'])
    monkeypatch.setattr(runner, '_parent_locked', lambda _: nullcontext())
    output = tmp_path / 'aggregate'
    result = aggregate.collect(tmp_path / 'plan.json', output)
    assert result['attempted_inputs'] == 160 and result['not_run_inputs'] == 800
    assert result['families'] == [] and not enabled.exists()
    assert read_json(output / 'aggregate.json')['manifest_identity_hash'] == f['manifest']['identity_hash']


def test_retry_command_uses_actual_python_and_repository_from_any_cwd(tmp_path):
    result = subprocess.run(checkpoint.retry_command(tmp_path / 'missing checkpoint'), shell=True,
                            cwd='/tmp', capture_output=True, text=True)
    assert result.returncode == 5
    assert 'STOPPED:' in result.stderr and 'command not found' not in result.stderr
    assert 'No module named' not in result.stderr
