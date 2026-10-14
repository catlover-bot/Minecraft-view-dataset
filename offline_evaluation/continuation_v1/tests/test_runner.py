"""Continuation fixtures block live HTTP and never activate a production plan."""
from copy import deepcopy
import datetime
from pathlib import Path
import socket

import pytest

from offline_evaluation.batch_v1.tests.test_runner import response, transports
from offline_evaluation.continuation_v1 import plan as planning, runner, cli
from offline_evaluation.continuation_v1.inputs import suffix
from offline_evaluation.resmoke_v2 import adapter
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.core import Grid, grid_to_payload
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json
from offline_evaluation.server_v2.io_utils import ROOT


@pytest.fixture
def fixture_plan(tmp_path, monkeypatch):
    for name in providers.KEY_NAMES.values():
        monkeypatch.setenv(name, 'fixture_only_never_live')
    def forbidden(*a, **k):
        pytest.fail('live research/network transport forbidden')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(adapter, 'http_transport', forbidden)
    monkeypatch.setattr(providers, 'http_transport', forbidden)
    parent = tmp_path / 'completed_parent'
    parent.mkdir()
    (parent / 'run.lock').write_bytes(b'unchanged')
    write_json(parent / 'ledger.json', planning.PARENT_USAGE)
    before = {str(p): p.read_bytes() for p in parent.iterdir()}
    def verify(identity):
        if {str(p): p.read_bytes() for p in parent.iterdir()} != before:
            raise GenerationError('parent_changed')
    monkeypatch.setattr(runner, '_verify_parent', verify)
    monkeypatch.setattr(planning, 'verify_parent', lambda p: verify({'parent': p}))
    document = read_json(ROOT / 'reports/server_only/goal005/official_pricing.json')
    document['verified_on'] = runner.utc_now()[:10]
    document['verified_at_utc'] = runner.utc_now()
    document['retrieval'] = 'OFFLINE_FIXTURE_ONLY_NOT_DOCUMENTATION_EVIDENCE'
    for p in adapter.MODELS:
        document[p]['verified_on'] = document['verified_on']
    pricing = {p: document[p] for p in adapter.MODELS}
    scenes = [f'fixture_{i:03d}' for i in range(60)]
    kinds = [(c, d) for d in ('openai', 'anthropic', 'gemini') for c in 'OJ'] + [('P', None), ('E', None)]
    full = []
    for scene in scenes:
        for condition, description in kinds:
            for provider, model in adapter.MODELS.items():
                full.append({'input_id': scene + condition + str(description), 'scene_id': scene,
                    'condition': condition, 'description_model': description, 'provider': provider,
                    'model': model, 'input_text': scene + condition, 'input_sha256': 'fixture',
                    'schedule_index': len(full), 'protocol_id': 'fixed_semantic_fixture'})
    manifest_identity = {'full_schedule': full, 'batch_schedule': full[:160],
                         'evaluation_scene_ids': scenes, 'protocol_id': 'fixed_semantic_fixture'}
    manifest = {'identity': manifest_identity, 'identity_hash': object_hash(manifest_identity)}
    schedule, remaining, segments = suffix(manifest, {'identity': {'schedule': full[:160]}})
    grid = Grid({(0, 0, 0): 'stone'})
    path = tmp_path / 'plan.json'
    identity = {'protocol_id': planning.PROTOCOL_ID, 'semantic_protocol_id': 'fixed_semantic_fixture',
        'source_run_protocol_id': 'parent_run_fixture', 'provenance': 'fixture_only',
        'plan_path': str(path), 'enabled_directory': str(tmp_path / 'enabled'),
        'authority_path': str(tmp_path / 'authority.json'),
        'parent': {'usage': dict(planning.PARENT_USAGE), 'lock_paths': [str(parent / 'run.lock')]},
        'schedule': schedule, 'batch_scene_ids': remaining, 'evaluation_scene_ids': scenes,
        'segments': segments, 'global_schedule_offset': 160, 'scope': planning.SCOPE,
        'prompt': 'fixture common prompt', 'pricing': pricing, 'pricing_document': document,
        'pricing_document_hash': object_hash(document), 'budgets': planning.budgets(pricing),
        'gt_by_scene': {s: {'grid_sha256': object_hash(grid_to_payload(grid))} for s in remaining}}
    plan = {'identity': identity, 'identity_hash': object_hash(identity), 'api_allowed': False, 'status': 'FIXTURE_DISABLED'}
    write_json(path, plan)
    def load(candidate):
        candidate = Path(candidate)
        value = read_json(candidate)
        if object_hash(value['identity']) != value['identity_hash']:
            raise GenerationError('fixture_plan_hash_mismatch')
        if str(candidate) != value['identity']['plan_path']:
            raise GenerationError('canonical_plan_path_mismatch')
        return value
    monkeypatch.setattr(runner, '_plan', load)
    monkeypatch.setattr(planning, 'load_plan', load)
    monkeypatch.setattr(runner, '_progress', lambda *a: None)
    return {'path': path, 'plan': plan, 'manifest': manifest, 'parent_plan': {'identity': {'schedule': full[:160]}},
            'before': before, 'parent': parent, 'grids': {s: grid for s in remaining}}


def enable(fixture):
    plan = fixture['plan']
    return runner.initialize_enabled(fixture['path'], approval={
        'manual_confirmed': True, 'plan_identity_hash': plan['identity_hash'],
        'enabled_directory': plan['identity']['enabled_directory'], 'approved_at_utc': runner.utc_now(),
        'confirmation_channel': '/dev/tty', 'scope': runner.APPROVAL_SCOPE})


def saved(*a, **k):
    return {'status': 'FIXTURE_SAVED'}


def test_exact_suffix_global_local_boundary_and_semantic_duplicate_rules(fixture_plan):
    manifest, parent = fixture_plan['manifest'], fixture_plan['parent_plan']
    rows, scenes, segments = suffix(manifest, parent)
    assert rows == manifest['identity']['full_schedule'][160:]
    assert [rows[i]['schedule_index'] for i in (0, 159, 160, 799)] == [160, 319, 320, 959]
    assert len({r['input_id'] for r in rows}) == 400
    assert len({(r['provider'], r['input_id']) for r in rows}) == 800
    assert len(scenes) == 50 and len(segments) == 5
    bad = deepcopy(manifest)
    bad['identity']['full_schedule'][161]['input_id'] = bad['identity']['full_schedule'][163]['input_id']
    bad['identity_hash'] = object_hash(bad['identity'])
    with pytest.raises(GenerationError, match='duplicate_provider_input'):
        suffix(bad, parent)
    bad = deepcopy(manifest)
    bad['identity']['full_schedule'][160]['schedule_index'] = 159
    bad['identity_hash'] = object_hash(bad['identity'])
    with pytest.raises(GenerationError, match='global_index'):
        suffix(bad, parent)


def test_complete800_global_indices_parent_once_payload_same_and_five_checkpoints(fixture_plan):
    directory = enable(fixture_plan)
    calls, checkpoints = [], []
    identity = fixture_plan['plan']['identity']
    def inspect(provider, payload, count):
        row = identity['schedule'][count - 1]
        entry = read_json(directory / 'ledger.json')['entries'][-1]
        assert entry['local_index'] == count - 1 and entry['schedule_index'] == count + 159
        assert entry['status'] == 'reserved'
        assert payload == runner.base._payload(identity, row)
        assert payload == adapter.build_payload(provider, row['model'], identity['prompt'], row['input_text'], 32768)
        return response(provider)
    def checkpoint(plan, directory, ledger, reason, **kw):
        checkpoints.append((len(ledger['entries']), reason))
        assert all(e['status'] != 'reserved' for e in ledger['entries'])
        return saved()
    wire = transports(calls, inspect)
    result = runner.run(fixture_plan['path'], permission=True, transports=wire, checkpoint_callback=checkpoint)
    assert result['status'] == 'COMPLETE_CONTINUATION800'
    assert result['used']['combined']['calls'] == 990 and result['used']['combined']['retries'] == 4
    assert result['used']['combined']['tokens'] == 1828481 + 24000
    assert result['used']['prior']['cost_usd_upper_bound'] == 4.048922
    assert result['used']['combined']['cost_usd_upper_bound'] == pytest.approx(4.109922)
    assert checkpoints == [(n, 'segment_complete') for n in (160, 320, 480, 640, 800)]
    runner.run(fixture_plan['path'], permission=True, transports=wire, checkpoint_callback=checkpoint)
    assert len(calls) == 800
    assert {str(p): p.read_bytes() for p in fixture_plan['parent'].iterdir()} == fixture_plan['before']


@pytest.mark.parametrize('kind', ['timeout', 'auth_failure', 'response_identity_failure'])
def test_uncertain_or_critical_halt_never_resends(fixture_plan, kind):
    directory = enable(fixture_plan)
    calls = []
    def fail(*a):
        if kind == 'response_identity_failure':
            return response(a[0], model='wrong_snapshot')
        raise providers.ProviderError(kind)
    wire = transports(calls, fail)
    result = runner.run(fixture_plan['path'], permission=True, transports=wire, checkpoint_callback=saved)
    assert result['attempted_inputs'] == 1 and result['not_run_inputs'] == 799
    assert result['halt_reason'] == ('response_identity_failure' if kind == 'response_identity_failure' else 'unknown_usage:' + kind)
    before = (directory / 'ledger.json').read_bytes()
    runner.run(fixture_plan['path'], permission=True, transports=wire, checkpoint_callback=saved)
    assert len(calls) == 1 and (directory / 'ledger.json').read_bytes() == before


def test_budget_reserves_before_send_and_cannot_reset(fixture_plan):
    directory = enable(fixture_plan)
    identity = fixture_plan['plan']['identity']
    assert identity['budgets']['max_tokens'] == 266214400
    assert identity['budgets']['max_total_tokens'] == 268042881
    assert identity['budgets']['max_cost_usd'] == 30
    ledger = {'entries': [], 'halt_reason': None}
    reservation = runner._reservation(identity, identity['schedule'][0])
    modified = deepcopy(identity)
    modified['budgets']['max_cost_usd'] = 4.048922
    assert not runner._can_reserve(modified, ledger, reservation)
    copied = fixture_plan['path'].with_name('copied.json')
    copied.write_bytes(fixture_plan['path'].read_bytes())
    with pytest.raises(GenerationError, match='canonical_plan_path'):
        runner.run(copied, permission=True)
    other = deepcopy(fixture_plan['plan'])
    other['identity']['plan_path'] = str(copied)
    other['identity']['enabled_directory'] += '-other'
    other['identity_hash'] = object_hash(other['identity'])
    write_json(copied, other)
    with pytest.raises(GenerationError, match='lineage_already_claimed'):
        enable({'path': copied, 'plan': other})
    assert read_json(directory / 'ledger.json')['entries'] == []


def test_pricing_midrun_stop_before_reservation_then_append_refresh_same_approval(fixture_plan, monkeypatch):
    directory = enable(fixture_plan)
    calls, checkpoints = [], []
    old_day = fixture_plan['plan']['identity']['pricing']['openai']['verified_on']
    new_day = (datetime.date.fromisoformat(old_day) + datetime.timedelta(days=1)).isoformat()
    day = [old_day]
    real_validate = planning.validate_pricing
    def validate(pricing, *, fresh=False):
        real_validate(pricing, fresh=False)
        if fresh and any(r['verified_on'] != day[0] for r in pricing.values()):
            raise GenerationError('pricing_documentation_requires_today_verification_and_replan')
    monkeypatch.setattr(planning, 'validate_pricing', validate)
    def send(provider, payload, count):
        if count == 2:
            day[0] = new_day
        return response(provider)
    def checkpoint(plan, directory, ledger, reason, **kw):
        checkpoints.append((len(ledger['entries']), reason))
        return saved()
    wire = transports(calls, send)
    result = runner.run(fixture_plan['path'], permission=True, transports=wire, checkpoint_callback=checkpoint)
    assert result['attempted_inputs'] == 2 and result['not_run_inputs'] == 798
    assert result['halt_reason'] is None and 'today_verification' in result['operational_stop']
    assert checkpoints == [(2, 'pricing_freshness_stop')]
    assert 'launch.py' in result['resume_command']
    before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    approval = (directory / 'approval.json').read_bytes()
    assert runner.status(fixture_plan['path'])['not_run_inputs'] == 798
    document = deepcopy(fixture_plan['plan']['identity']['pricing_document'])
    document['verified_on'] = new_day
    document['verified_at_utc'] = new_day + 'T10:00:00Z'
    for provider in adapter.MODELS:
        document[provider]['verified_on'] = new_day
    price = fixture_plan['path'].with_name('actual_fixture_reverification.json')
    write_json(price, document)
    monkeypatch.setattr(planning, 'utc_now', lambda: new_day + 'T11:00:00Z')
    planning.refresh_pricing(fixture_plan['path'], pricing_file=price)
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    planning.current_pricing(fixture_plan['plan']['identity'], fresh=True)
    def stop_saved(*args, **kw):
        return {'status': 'GITHUB_SAVE_FAILED', 'checkpoint': 'fixture_saved_on_disk'}
    result = runner.run(fixture_plan['path'], permission=True, transports=wire, checkpoint_callback=stop_saved)
    assert result['attempted_inputs'] == 160 and len(calls) == 160
    assert result['halt_reason'] is None and result['operational_stop'] == 'checkpoint_save_failed'
    assert (directory / 'approval.json').read_bytes() == approval
    document['openai']['context_window_tokens'] += 1
    write_json(price, document)
    with pytest.raises(GenerationError, match='scope_reapproval'):
        planning.refresh_pricing(fixture_plan['path'], pricing_file=price)


def test_disabled_status_preview_evaluate_and_tty_refusal_create_nothing(fixture_plan, monkeypatch):
    before = {str(p): p.read_bytes() for p in fixture_plan['path'].parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(cli, '_confirm_from_tty', lambda text: False)
    assert cli.launch(fixture_plan['path'], preview_only=True) == 0
    assert cli.launch(fixture_plan['path']) == 2
    assert runner.status(fixture_plan['path'])['approved'] is False
    with pytest.raises(GenerationError, match='run_not_enabled'):
        runner.evaluate_and_summarize(fixture_plan['path'])
    assert before == {str(p): p.read_bytes() for p in fixture_plan['path'].parent.rglob('*') if p.is_file()}
    text = cli.confirmation_text(fixture_plan['plan'])
    assert '800 SENDS 0 RETRIES USD30 TOTAL' in text
    assert fixture_plan['plan']['identity_hash'][:12] in text


def test_immutable_claim_detects_rollback_and_both_indices(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def fail(*a):
        raise providers.ProviderError('timeout')
    runner.run(fixture_plan['path'], permission=True, transports=transports(calls, fail), checkpoint_callback=saved)
    ledger = read_json(directory / 'ledger.json')
    ledger['entries'][0]['local_index'] = 160
    write_json(directory / 'ledger.json', ledger)
    with pytest.raises(GenerationError, match='invalid_ledger_attempt_scope'):
        runner._load_enabled(fixture_plan['plan'])
    ledger['entries'] = []
    write_json(directory / 'ledger.json', ledger)
    with pytest.raises(GenerationError, match='reservation_claim_ledger_mismatch'):
        runner._load_enabled(fixture_plan['plan'])
    assert len(calls) == 1


def test_interrupted_request_keeps_reservation_and_never_freezes_inflight(fixture_plan):
    directory = enable(fixture_plan)
    calls, checkpoints = [], []
    def interrupted(*a):
        raise KeyboardInterrupt
    def checkpoint(*a, **k):
        checkpoints.append('safe_after_transport_returned')
        return saved()
    wire = transports(calls, interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run(fixture_plan['path'], permission=True, transports=wire, checkpoint_callback=checkpoint)
    assert checkpoints == []
    assert read_json(directory / 'ledger.json')['entries'][0]['status'] == 'reserved'
    result = runner.run(fixture_plan['path'], permission=True, transports=wire, checkpoint_callback=checkpoint)
    assert result['halt_reason'] == 'unreconciled_prior_reservation'
    assert len(calls) == 1 and checkpoints == ['safe_after_transport_returned']


def test_known_usage_validation_and_truncation_continue_to_next_unattempted(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def answer(provider, payload, count):
        if count == 1:
            return response(provider, '{"operations":[{"op":"set","x":0,"y":0,"z":0,"block":"quartz (white)"}]}')
        if count == 2:
            return response(provider, stop_reason='max_tokens')
        return response(provider)
    def checkpoint(*a, **k):
        return {'status': 'GITHUB_SAVE_FAILED'}
    result = runner.run(fixture_plan['path'], permission=True,
                        transports=transports(calls, answer), checkpoint_callback=checkpoint)
    entries = read_json(directory / 'ledger.json')['entries']
    assert len(entries) == 160 and len(calls) == 160
    assert [r['failure_kind'] for r in entries[:2]] == ['validation_failure', 'truncation']
    assert all(r['usage'] is not None and not r['retry'] for r in entries)
    assert result['halt_reason'] is None and result['not_run_inputs'] == 640
