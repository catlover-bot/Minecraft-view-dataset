"""CPU-only Goal006C first-attempt scheduling and preserved-history accounting."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from offline_evaluation.batch_v1.tests.test_runner import response, transports
from offline_evaluation.operations.goal006c import planning, runner
from offline_evaluation.operations.goal006a.archive import RecoveryError
from offline_evaluation.resmoke_v2 import adapter
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json


def approval(plan):
    return {'manual_confirmed': True, 'plan_identity_hash': plan['identity_hash'],
            'enabled_directory': plan['identity']['enabled_directory'], 'approved_at_utc': runner.utc_now(),
            'confirmation_channel': '/dev/tty', 'scope': runner.APPROVAL_SCOPE,
            'acknowledges_both_unknown_holds': True}


def verified_checkpoints(plan):
    identity = plan['identity']
    reports = {}
    for name, reference, snapshot, filename in (
        ('original781', 'original_checkpoint_reference', 'original_source_snapshot', 'continuation-point-in-time.tar.gz'),
        ('supplement3', 'checkpoint_reference', 'source_snapshot', 'supplement-point-in-time.tar.gz'),
    ):
        ref, source = identity[reference], identity[snapshot]
        reports[name] = {'status': 'GITHUB_VERIFIED', 'draft': True,
            'release_id': ref['release_id'], 'plan_identity_hash': source['plan_identity_hash'],
            'verifiedledger_sha256': source['ledger_sha256'], 'attempted_inputs': source['attempted_inputs'],
            'assets': [{'name': filename, 'asset_id': ref['archive_asset_id'], 'sha256': ref['archive_sha256']}]}
    return reports


def saved(*_args, **_kwargs):
    return {'status': 'GITHUB_VERIFIED', 'fixture_only': True}


def enable(fixture):
    return runner.initialize_enabled(fixture['path'], approval=approval(fixture['plan']))


def execute(fixture, calls, handler=None, **kwargs):
    return runner.run(fixture['path'], permission=True, transports=transports(calls, handler),
        checkpoint_verifier=kwargs.pop('checkpoint_verifier', verified_checkpoints),
        checkpoint_callback=kwargs.pop('checkpoint_callback', saved), **kwargs)


def assert_originals_unchanged(fixture):
    assert all(Path(path).read_bytes() == content for path, content in fixture['source_before'].items())


def test_exact17_global943_to959_payloads_no_retries_and_no_repeat(fixture_plan):
    directory = enable(fixture_plan)
    calls, checkpoints = [], []
    original = fixture_plan['source_plan']['identity']
    expected = original['schedule'][3:]
    assert len(expected) == 17
    old_pairs = {(entry['provider'], entry['input_id']) for entry in fixture_plan['source_ledger']['entries']}
    def inspect(provider, payload, count):
        record = expected[count - 1]
        entry = read_json(directory / 'ledger.json')['entries'][-1]
        assert entry['status'] == 'reserved' and entry['usage'] is None
        assert entry['schedule_index'] == record['schedule_index'] == 942 + count
        assert entry['provider'] == provider and entry['input_id'] == record['input_id']
        assert entry['retry'] is False and entry['intent'] == 'first_attempt' and entry['source_attempt_id'] is None
        assert entry['charged_output_tokens'] == 32768
        assert (provider, entry['input_id']) not in old_pairs
        assert payload == adapter.build_payload(provider, record['model'], original['prompt'], record['input_text'], 32768)
        assert entry['request_payload_hash'] == object_hash(payload)
        return response(provider)
    def checkpoint(_plan, _directory, ledger, reason):
        checkpoints.append((len(ledger['entries']), reason))
        assert all(entry['status'] != 'reserved' for entry in ledger['entries'])
        return saved()
    result = execute(fixture_plan, calls, inspect, checkpoint_callback=checkpoint)
    assert len(calls) == 17 and result['supplemental_attempted'] == 17 and result['supplemental_not_run'] == 0
    assert result['combined_calls'] == 991 and result['combined_retries'] == 5
    assert result['manual_auth_retry_attempted'] == result['automatic_retries'] == 0
    assert result['original_attempts_preserved'] == 781 and result['source_supplemental_attempts_preserved'] == 3
    assert result['historical_unknown_usage_attempts_preserved'] == 2
    assert result['historical_unknown_usage_reservation_reconciled'] is False
    assert result['http520_retried'] is False and result['actual_billed_usd'] is None
    assert result['source_halt_reason_unchanged'] == 'unknown_usage:http_failure'
    assert result['original_halt_reason_unchanged'] == 'unknown_usage:auth_failure'
    entries = read_json(directory / 'ledger.json')['entries']
    assert [e['schedule_index'] for e in entries] == list(range(943, 960))
    assert len({(e['provider'], e['input_id']) for e in entries}) == 17
    before = (directory / 'ledger.json').read_bytes()
    execute(fixture_plan, calls)
    assert len(calls) == 17 and (directory / 'ledger.json').read_bytes() == before
    assert checkpoints == [(17, 'supplement_finished')]
    assert_originals_unchanged(fixture_plan)


def test_permission_and_unapproved_plan_send_zero(fixture_plan):
    calls = []
    with pytest.raises(GenerationError):
        runner.run(fixture_plan['path'])
    with pytest.raises(GenerationError):
        execute(fixture_plan, calls)
    assert calls == []
    assert not Path(fixture_plan['plan']['identity']['enabled_directory']).exists()
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize('field,value', [
    ('scope', 'goal006b_19_unattempted_plus_1_explicit_auth_retry_cumulative_usd30_calls991'),
    ('acknowledges_both_unknown_holds', False), ('manual_confirmed', False),
    ('confirmation_channel', 'stdin'), ('approved_at_utc', '2020-01-01T00:00:00Z'),
])
def test_separate_exact_approval_and_both_holds_required(fixture_plan, field, value):
    invalid = approval(fixture_plan['plan']); invalid[field] = value
    with pytest.raises(GenerationError):
        runner.initialize_enabled(fixture_plan['path'], approval=invalid)
    assert not Path(fixture_plan['plan']['identity']['enabled_directory']).exists()
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize('name,field', [('original781', 'verifiedledger_sha256'), ('supplement3', 'release_id')])
def test_each_checkpoint_must_verify_before_first_reservation(fixture_plan, name, field):
    directory = enable(fixture_plan); calls = []
    def invalid(plan):
        reports = verified_checkpoints(plan)
        reports[name][field] = 'mismatched'
        return reports
    with pytest.raises(GenerationError, match='_checkpoint_verification_mismatch'):
        execute(fixture_plan, calls, checkpoint_verifier=invalid)
    assert calls == [] and read_json(directory / 'ledger.json')['entries'] == []
    assert not (directory / 'reservations').exists()
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize('kind,status_code', [('http_failure', 520), ('auth_failure', 401), ('timeout', None)])
def test_new_unknown_usage_stops_after_one_and_never_resends(fixture_plan, kind, status_code):
    directory = enable(fixture_plan); calls = []
    def fail(*_args):
        raise providers.ProviderError(kind, status_code)
    result = execute(fixture_plan, calls, fail)
    assert result['halt_reason'] == 'unknown_usage:' + kind
    assert result['supplemental_attempted'] == 1 and result['supplemental_not_run'] == 16
    assert result['historical_unknown_usage_attempts_preserved'] == 2
    assert result['supplemental_unknown_usage_attempts'] == 1
    entry = read_json(directory / 'ledger.json')['entries'][0]
    assert entry['usage'] is None and entry['billing'] == 'worst_case_reservation' and entry['artifacts'] == {}
    assert entry['retry'] is False and entry['source_attempt_id'] is None
    assert tuple(entry[k] for k in ('charged_input_tokens', 'charged_output_tokens', 'charged_cost_usd')) == runner.base._reservation(
        fixture_plan['plan']['identity'], fixture_plan['plan']['identity']['schedule'][0])
    before = (directory / 'ledger.json').read_bytes()
    execute(fixture_plan, calls)
    assert len(calls) == 1 and (directory / 'ledger.json').read_bytes() == before
    assert_originals_unchanged(fixture_plan)


def test_interrupted_reserved_attempt_is_never_resent(fixture_plan):
    directory = enable(fixture_plan); calls = []
    def interrupt(*_args):
        raise KeyboardInterrupt('fixture interruption after durable reservation')
    with pytest.raises(KeyboardInterrupt):
        execute(fixture_plan, calls, interrupt)
    ledger = read_json(directory / 'ledger.json')
    assert len(ledger['entries']) == 1 and ledger['entries'][0]['status'] == 'reserved'
    execute(fixture_plan, calls)
    recovered = read_json(directory / 'ledger.json')
    assert len(calls) == 1 and recovered['entries'] == ledger['entries']
    assert recovered['halt_reason'] == 'unreconciled_supplemental_reservation'
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize('kind', ['truncation', 'schema'])
def test_known_model_failure_kept_once_then_next_input_runs(fixture_plan, kind):
    directory = enable(fixture_plan); calls = []
    def model_output(provider, _payload, count):
        if count == 1:
            return response(provider, stop_reason='max_tokens') if kind == 'truncation' else response(provider, text='{"not_operations": []}')
        return response(provider)
    result = execute(fixture_plan, calls, model_output)
    entries = read_json(directory / 'ledger.json')['entries']
    assert len(calls) == 17 and result['halt_reason'] is None
    assert entries[0]['status'] == 'failed' and entries[0]['usage'] is not None
    if kind == 'schema':
        assert entries[0]['failure_kind'] == 'validation_failure'
        assert entries[0]['schema_success'] is False and entries[0]['failure_stage'] == 'schema'
    else:
        assert entries[0]['failure_kind'] == 'truncation'
    assert all(e['status'] == 'success' for e in entries[1:])
    execute(fixture_plan, calls)
    assert len(calls) == 17
    assert_originals_unchanged(fixture_plan)


def test_save_failure_keeps_terminal_ledger_and_recovery_sends_zero(fixture_plan):
    directory = enable(fixture_plan); calls = []
    result = execute(fixture_plan, calls, checkpoint_callback=lambda *_a: {'status': 'LOCAL_VERIFIED'})
    assert result['operational_stop'] == 'checkpoint_save_failed' and result['halt_reason'] is None
    before = (directory / 'ledger.json').read_bytes()
    again = execute(fixture_plan, calls, checkpoint_callback=lambda *_a: {'status': 'SAVE_FAILED'})
    assert again['operational_stop'] == 'checkpoint_save_failed'
    recovered = execute(fixture_plan, calls)
    assert recovered['operational_stop'] is None
    assert len(calls) == 17 and (directory / 'ledger.json').read_bytes() == before
    assert_originals_unchanged(fixture_plan)


def test_pricing_stop_and_failed_save_block_next_send_until_prefix_verified(fixture_plan, monkeypatch):
    from offline_evaluation.resmoke_v2 import plan as semantic_plan
    from offline_evaluation.server_v2 import generation
    directory = enable(fixture_plan); calls, checkpoints = [], []
    document = deepcopy(fixture_plan['plan']['identity']['pricing_document'])
    tomorrow = (datetime.fromisoformat(document['verified_at_utc'].replace('Z', '+00:00')) + timedelta(days=1)).isoformat().replace('+00:00', 'Z')
    class FixtureDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.fromisoformat(tomorrow.replace('Z', '+00:00'))
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    def advance(provider, _payload, count):
        assert count == 1
        monkeypatch.setattr(semantic_plan, 'utc_now', lambda: tomorrow)
        monkeypatch.setattr(planning, 'utc_now', lambda: tomorrow)
        monkeypatch.setattr(generation, 'dt', SimpleNamespace(date=date, datetime=FixtureDateTime, timezone=timezone))
        return response(provider)
    def failed_save(_plan, _directory, ledger, reason):
        checkpoints.append((len(ledger['entries']), reason))
        return {'status': 'LOCAL_VERIFIED'}
    result = execute(fixture_plan, calls, advance, checkpoint_callback=failed_save)
    assert result['operational_stop'] == 'pricing_freshness_stop' and result['checkpoint']['status'] == 'SAVE_FAILED'
    assert len(calls) == 1 and result['halt_reason'] is None
    before = (directory / 'ledger.json').read_bytes()
    document['verified_at_utc'], document['verified_on'] = tomorrow, tomorrow[:10]
    for provider in adapter.MODELS:
        document[provider]['verified_on'] = tomorrow[:10]
    price_path = fixture_plan['root'].parent / 'next-date-pricing-fixture.json'
    write_json(price_path, document)
    planning.refresh_pricing(fixture_plan['path'], pricing_file=price_path)
    blocked = execute(fixture_plan, calls, checkpoint_callback=failed_save)
    assert blocked['operational_stop'] == 'checkpoint_save_failed'
    assert len(calls) == 1 and (directory / 'ledger.json').read_bytes() == before
    def successful_save(_plan, _directory, ledger, reason):
        checkpoints.append((len(ledger['entries']), reason)); return saved()
    resumed = execute(fixture_plan, calls, checkpoint_callback=successful_save)
    assert len(calls) == 17 and resumed['supplemental_not_run'] == 0
    assert read_json(directory / 'ledger.json')['entries'][0] == json.loads(before)['entries'][0]
    assert checkpoints == [(1, 'pricing_freshness_stop'), (1, 'before_partial_resume'),
                           (1, 'before_partial_resume'), (17, 'supplement_finished')]
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize('corruption', ['payload', 'retry', 'source_attempt', 'duplicate', 'cost'])
def test_modified_child_ledger_rejected_without_resend(fixture_plan, corruption):
    directory = enable(fixture_plan); calls = []
    execute(fixture_plan, calls)
    ledger = read_json(directory / 'ledger.json')
    if corruption == 'payload': ledger['entries'][0]['request_payload_hash'] = '0' * 64
    elif corruption == 'retry': ledger['entries'][0]['retry'] = True
    elif corruption == 'source_attempt': ledger['entries'][0]['source_attempt_id'] = '0' * 32
    elif corruption == 'duplicate': ledger['entries'][1] = deepcopy(ledger['entries'][0])
    else: ledger['entries'][0]['charged_cost_usd'] = 0
    write_json(directory / 'ledger.json', ledger)
    with pytest.raises(GenerationError):
        execute(fixture_plan, calls)
    assert len(calls) == 17
    assert_originals_unchanged(fixture_plan)


def test_copied_child_plan_cannot_run(fixture_plan):
    copied = fixture_plan['root'] / 'goal006c-copy'; copied.mkdir()
    (copied / 'plan.json').write_bytes(fixture_plan['path'].read_bytes())
    with pytest.raises(GenerationError):
        runner.run(copied / 'plan.json', permission=True, transports=transports([]),
                   checkpoint_verifier=verified_checkpoints, checkpoint_callback=saved)
    assert_originals_unchanged(fixture_plan)


def test_production_provenance_rejects_fixture_injections(fixture_plan, monkeypatch):
    plan = deepcopy(fixture_plan['plan']); plan['identity']['provenance'] = 'new_implementation'
    plan['identity_hash'] = object_hash(plan['identity']); write_json(fixture_plan['path'], plan)
    monkeypatch.setattr(planning, 'PROVENANCE', 'new_implementation')
    with pytest.raises(GenerationError, match='injections_require_fixture_only'):
        execute(fixture_plan, [])
    assert_originals_unchanged(fixture_plan)


def test_both_unknown_holds_count_toward_usd30_and_calls991(fixture_plan):
    identity = fixture_plan['plan']['identity']; prior = identity['parent']['usage']
    holds = identity['source_accounting']['unknown_usage_holds']
    assert prior['calls'] == 974 and prior['retries'] == 5 and prior['unknown_usage_attempts'] == 2
    assert [hold['tokens'] for hold in holds] == [432768, 432768]
    assert [hold['cost_usd_upper_bound'] for hold in holds] == [0.165536, 0.165536]
    assert identity['budgets']['max_cost_usd'] == 30 and identity['budgets']['max_total_calls'] == 991
    assert identity['budgets']['max_new_calls'] == 17
    assert identity['budgets']['max_manual_auth_retries'] == identity['budgets']['max_automatic_retries'] == 0
    reservation = runner.base._reservation(identity, identity['schedule'][0]); empty = {'entries': [], 'halt_reason': None}
    assert runner.base._can_reserve(identity, empty, reservation)
    exhausted = deepcopy(identity); exhausted['parent']['usage']['cost_usd_upper_bound'] = 30 - reservation[2] + .000001
    assert not runner.base._can_reserve(exhausted, empty, reservation)
    exhausted = deepcopy(identity); exhausted['parent']['usage']['calls'] = 991
    assert not runner.base._can_reserve(exhausted, empty, reservation)
    assert_originals_unchanged(fixture_plan)


def test_double_launch_blocks_other_process_and_cannot_reinitialize(fixture_plan):
    directory = enable(fixture_plan); calls = []
    with pytest.raises(GenerationError, match='already_claimed'):
        enable(fixture_plan)
    def competing(provider, _payload, count):
        if count == 1:
            reader, writer = os.pipe(); pid = os.fork()
            if pid == 0:
                os.close(reader); child_calls = []
                try:
                    execute(fixture_plan, child_calls)
                    report = {'unexpected': True, 'calls': len(child_calls)}
                except GenerationError as exc:
                    report = {'code': str(exc), 'calls': len(child_calls)}
                os.write(writer, json.dumps(report).encode()); os.close(writer); os._exit(0)
            os.close(writer)
            try:
                result = json.loads(os.read(reader, 4096))
            finally:
                os.close(reader)
            _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0
            assert result == {'code': 'original_run_active', 'calls': 0}
        return response(provider)
    execute(fixture_plan, calls, competing)
    assert len(calls) == 17 and len(read_json(directory / 'ledger.json')['entries']) == 17
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize('error_source', ['original', 'supplemental'])
def test_default_save_retains_local_path_and_safe_github_diagnostic(fixture_plan, monkeypatch, error_source):
    from offline_evaluation.operations.goal006c import archive, checkpoint
    directory = enable(fixture_plan)
    local = str(fixture_plan['root'].parent / 'local-checkpoint-fixture')
    monkeypatch.setattr(checkpoint, 'save_local', lambda *_a: {'status': 'LOCAL_VERIFIED', 'checkpoint_directory': local})
    def fail(*_args, **_kwargs):
        error = RecoveryError if error_source == 'original' else archive.RecoveryError
        raise error('asset_download', 'github_command_failed', returncode=1, http_status=503)
    monkeypatch.setattr(archive, 'recover', fail)
    ledger = read_json(directory / 'ledger.json')
    before = (directory / 'ledger.json').read_bytes()
    result = runner._save(fixture_plan['plan'], directory, ledger, 'fixture_test', None)
    assert result['status'] == 'SAVE_FAILED' and result['checkpoint_directory'] == local
    assert result['github_error'] == {'stage': 'asset_download', 'code': 'github_command_failed', 'returncode': 1, 'http_status': 503}
    import shlex
    retry_argv = shlex.split(result['upload_only_retry_command'])
    assert retry_argv[-3:] == ['--save-checkpoint', local, '--allow-create']
    assert '--execute' not in retry_argv
    assert (directory / 'ledger.json').read_bytes() == before
    assert_originals_unchanged(fixture_plan)


def test_default_pre_send_dispatch_restores_both_pinned_archives(fixture_plan, monkeypatch):
    from offline_evaluation.operations.goal006a import archive as original_archive
    from offline_evaluation.operations.goal006c import archive as supplement_archive
    enable(fixture_plan)
    reports, events, calls = verified_checkpoints(fixture_plan['plan']), [], []
    identity = fixture_plan['plan']['identity']
    def recovery(name, reference):
        def recover(checkpoint_directory, verification_directory, *, release_id, allow_create):
            assert checkpoint_directory == identity[reference]['checkpoint_directory']
            assert release_id == identity[reference]['release_id'] and allow_create is False
            assert name in Path(verification_directory).name
            events.append(name)
            return reports[name]
        return recover
    monkeypatch.setattr(original_archive, 'recover', recovery('original781', 'original_checkpoint_reference'))
    monkeypatch.setattr(supplement_archive, 'recover', recovery('supplement3', 'checkpoint_reference'))
    def stop(*_args):
        assert events == ['original781', 'supplement3']
        events.append('fake_provider_attempt')
        raise providers.ProviderError('timeout')
    result = execute(fixture_plan, calls, stop, checkpoint_verifier=None)
    assert result['halt_reason'] == 'unknown_usage:timeout' and len(calls) == 1
    assert events == ['original781', 'supplement3', 'fake_provider_attempt']
    assert_originals_unchanged(fixture_plan)
