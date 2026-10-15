"""Resume fixtures use the frozen runner and block every live network path."""
from pathlib import Path

import pytest

from offline_evaluation.continuation_v1 import aggregate, checkpoint, runner
from offline_evaluation.continuation_v1.tests.test_runner import fixture_plan, enable, response, transports
from offline_evaluation.operations.goal006a import archive, resume
from offline_evaluation.server_v2.data import sha256_file
from offline_evaluation.server_v2.generation import GenerationError, read_json, write_json


def package(fixture, root, count, suffix='fixture'):
    plan = fixture['plan']
    directory = root / f"goal006-{plan['identity_hash'][:12]}-{count:04d}-{suffix}"
    directory.mkdir()
    ledger_path = Path(plan['identity']['enabled_directory']) / 'ledger.json'
    write_json(directory / 'checkpoint.json', {
        'plan_identity_hash': plan['identity_hash'], 'archive': 'fixture.tar.gz',
        'assets': [{'name': 'fixture.tar.gz', 'sha256': 'fixture-archive'}]})
    write_json(directory / 'snapshot.json', {
        'plan_identity_hash': plan['identity_hash'],
        'canonical_plan_sha256': sha256_file(fixture['path']),
        'original_sha256': {str(ledger_path): sha256_file(ledger_path)},
        'execution': {'attempted_inputs': count}})
    return directory


def report(checkpoint_directory):
    snapshot = read_json(Path(checkpoint_directory) / 'snapshot.json')
    return {'status': 'GITHUB_VERIFIED', 'plan_identity_hash': snapshot['plan_identity_hash'],
            'verifiedledger_sha256': next(iter(snapshot['original_sha256'].values())),
            'attempted_inputs': snapshot['execution']['attempted_inputs'], 'release_id': 123,
            'assets': [{'name': 'fixture.tar.gz', 'asset_id': 456}]}


def test_preview_disabled_does_not_create_state_or_call_recovery(fixture_plan, monkeypatch):
    before = {str(p): p.read_bytes() for p in fixture_plan['path'].parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(archive, 'recover', lambda *a, **k: pytest.fail('preview called network/recovery'))
    result = resume.preview(fixture_plan['path'], checkpoint_root=fixture_plan['path'].parent)
    assert not result['api_allowed'] and not result['approved']
    assert result['research_api_requests_sent_by_preview'] == 0
    assert before == {str(p): p.read_bytes() for p in fixture_plan['path'].parent.rglob('*') if p.is_file()}


def test_frozen_runner_resume_320_to480_save_failure_then_no_resend(fixture_plan, monkeypatch):
    directory = enable(fixture_plan)
    calls = []
    def answer(provider, payload, count):
        if count == 1:
            return response(provider, '{"operations":[{"op":"set","x":0,"y":0,"z":0,"block":"quartz (white)"}]}')
        if count == 2:
            return response(provider, stop_reason='max_tokens')
        return response(provider)
    wire = transports(calls, answer)
    def initial_checkpoint(p, d, ledger, reason, **kw):
        return {'status': 'FIXTURE_SAVED' if len(ledger['entries']) == 160 else 'GITHUB_SAVE_FAILED'}
    initial = runner.run(fixture_plan['path'], permission=True, transports=wire,
                         checkpoint_callback=initial_checkpoint)
    assert initial['attempted_inputs'] == 320 and len(calls) == 320
    original = read_json(directory / 'ledger.json')
    assert [entry['failure_kind'] for entry in original['entries'][:2]] == ['validation_failure', 'truncation']
    approval = (directory / 'approval.json').read_bytes()
    authority = Path(fixture_plan['plan']['identity']['authority_path']).read_bytes()
    root = fixture_plan['path'].parent / 'checkpoints'; root.mkdir()
    anchor = package(fixture_plan, root, 320)
    events = []
    def recover(checkpoint_directory, verification_directory, *, release_id=None, allow_create=False):
        count = report(checkpoint_directory)['attempted_inputs']
        events.append(('verify', count, release_id, allow_create))
        if count == 480:
            raise archive.RecoveryError('asset_download', 'fixture_failure', returncode=1, http_status=503)
        return report(checkpoint_directory)
    monkeypatch.setattr(archive, 'recover', recover)
    monkeypatch.setattr(aggregate, 'evaluate_verified', lambda p, d, ledger: events.append(('evaluate', len(ledger['entries']))))
    def save(p, d, ledger, reason, *, github=False):
        assert github is False
        events.append(('local', len(ledger['entries']), reason))
        target = package(fixture_plan, root, len(ledger['entries']))
        return {'status': 'LOCAL_VERIFIED', 'checkpoint_directory': str(target)}
    monkeypatch.setattr(checkpoint, 'safe_checkpoint', save)
    kwargs = dict(checkpoint_root=root, verification_root=root / 'operations', anchor_checkpoint=anchor,
                  anchor_release_id=123, anchor_archive_asset_id=456,
                  anchor_archive_sha256='fixture-archive', transports=wire)
    result = resume.resume(fixture_plan['path'], **kwargs)
    assert result['attempted_inputs'] == 480 and result['not_run_inputs'] == 320
    assert result['operational_stop'] == 'checkpoint_save_failed' and result['halt_reason'] is None
    assert result['checkpoint']['error'] == {'stage': 'asset_download', 'code': 'fixture_failure', 'returncode': 1, 'http_status': 503}
    assert len(calls) == 480 and result['used']['combined']['calls'] == 670
    assert result['used']['combined']['retries'] == 4
    assert result['used']['prior']['cost_usd_upper_bound'] == 4.048922
    assert read_json(directory / 'ledger.json')['entries'][:320] == original['entries']
    assert (directory / 'approval.json').read_bytes() == approval
    assert Path(fixture_plan['plan']['identity']['authority_path']).read_bytes() == authority
    assert events == [('verify', 320, 123, False), ('evaluate', 480), ('local', 480, 'segment_complete'), ('verify', 480, None, True)]
    before = (directory / 'ledger.json').read_bytes()
    with pytest.raises(archive.RecoveryError, match='asset_download'):
        resume.resume(fixture_plan['path'], **kwargs)
    assert len(calls) == 480 and (directory / 'ledger.json').read_bytes() == before
    assert events[-2:] == [('verify', 320, 123, False), ('verify', 480, None, False)]
    assert all(not entry['retry'] for entry in read_json(directory / 'ledger.json')['entries'])
    assert fixture_plan['plan']['identity']['budgets']['max_cost_usd'] == 30
    assert fixture_plan['plan']['identity']['budgets']['max_output_tokens'] == 32768


def test_unapproved_or_unknown_run_never_reaches_recovery(fixture_plan, monkeypatch):
    monkeypatch.setattr(archive, 'recover', lambda *a, **k: pytest.fail('unsafe run reached recovery'))
    with pytest.raises(GenerationError, match='original_human_approval'):
        resume.resume(fixture_plan['path'], checkpoint_root=fixture_plan['path'].parent)
    directory = enable(fixture_plan)
    def interrupted(*a):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        runner.run(fixture_plan['path'], permission=True,
                   transports=transports([], interrupted), checkpoint_callback=lambda *a, **k: {})
    before = (directory / 'ledger.json').read_bytes()
    with pytest.raises(GenerationError, match='unknown_usage'):
        resume.resume(fixture_plan['path'], checkpoint_root=fixture_plan['path'].parent)
    assert (directory / 'ledger.json').read_bytes() == before


def test_current_checkpoint_missing_cannot_reuse_older_anchor(fixture_plan):
    directory = enable(fixture_plan)
    root = fixture_plan['path'].parent / 'checkpoints'; root.mkdir()
    package(fixture_plan, root, 0)
    ledger = read_json(directory / 'ledger.json')
    ledger['halt_reason'] = 'changed_after_saved_checkpoint'
    write_json(directory / 'ledger.json', ledger)
    with pytest.raises(GenerationError, match='current_ledger_checkpoint_missing'):
        resume.matching_checkpoint(fixture_plan['plan'], root)


def test_helper_outside_frozen_hash_scan():
    hashes = planning_hashes()
    assert not any('/operations/' in path for path in hashes)


def planning_hashes():
    from offline_evaluation.continuation_v1 import plan
    return plan.implementation_hashes()
