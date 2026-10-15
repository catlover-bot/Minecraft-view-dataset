"""Real local archive/restore fixtures; no live provider or GitHub requests."""
import base64
from datetime import datetime, timezone
from pathlib import Path
import socket
import tarfile

import pytest

from offline_evaluation.batch_v1.tests.test_runner import transports
from offline_evaluation.operations.goal006b import checkpoint, runner
from offline_evaluation.server_v2.data import sha256_file
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json


@pytest.fixture
def enabled(fixture_plan, tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, 'CHECKPOINT_ROOT', tmp_path / 'checkpoints')
    def forbidden(*args, **kwargs):
        pytest.fail('network forbidden in local checkpoint fixture')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    plan = fixture_plan['plan']
    approval = {'manual_confirmed': True, 'plan_identity_hash': plan['identity_hash'],
        'enabled_directory': plan['identity']['enabled_directory'], 'approved_at_utc': runner.utc_now(),
        'confirmation_channel': '/dev/tty', 'scope': runner.APPROVAL_SCOPE, 'acknowledges_unknown_hold': True}
    directory = runner.initialize_enabled(fixture_plan['path'], approval=approval)
    fixture_plan['directory'] = directory
    return fixture_plan


def ledger(fixture):
    return read_json(fixture['directory'] / 'ledger.json')


def source_unchanged(fixture):
    assert all(Path(path).read_bytes() == value for path, value in fixture['source_before'].items())


def reserved(fixture, raw=None):
    plan = fixture['plan']; identity = plan['identity']; directory = fixture['directory']
    record, intent = identity['schedule'][0], identity['intents'][0]
    charge = runner.base._reservation(identity, record)
    entry = {'attempt_id': 'e' * 32, 'local_index': 0, 'schedule_index': record['schedule_index'],
        'input_id': record['input_id'], 'provider': record['provider'], 'retry': True,
        'intent': intent['kind'], 'source_attempt_id': intent['source_attempt_id'],
        'request_payload_hash': object_hash(runner.base._payload(identity, record)),
        'status': 'reserved', 'usage': None, 'charged_input_tokens': charge[0],
        'charged_output_tokens': charge[1], 'charged_cost_usd': charge[2],
        'billing': 'worst_case_reservation', 'artifacts': {}}
    if raw is not None:
        relative = f"attempts/{entry['attempt_id']}/raw.json"
        path = directory / relative
        write_json(path, raw)
        entry['artifacts']['raw.json'] = {'path': relative, 'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    state = ledger(fixture); state['entries'] = [entry]
    write_json(directory / 'reservations' / (entry['attempt_id'] + '.json'), runner._claim(plan, entry))
    write_json(directory / 'ledger.json', state)
    return state


def save(fixture, state=None):
    return checkpoint.save_local(fixture['plan'], fixture['directory'], state or ledger(fixture), 'fixture_safe_checkpoint')


def test_empty_enabled_snapshot_real_archive_restore_and_source_reference(enabled):
    before = {str(p): p.read_bytes() for p in enabled['directory'].rglob('*') if p.is_file()}
    result = save(enabled)
    assert result['status'] == 'LOCAL_VERIFIED' and result['github_status'] == 'NOT_UPLOADED'
    assert result['research_api_requests_sent_by_checkpoint'] == 0
    assert result['execution']['supplemental_attempted'] == 0
    assert 'canonical' in result['copied_plan_load_rejection']
    destination = Path(result['checkpoint_directory'])
    assert len(result['assets']) == 6
    for asset in result['assets']:
        assert sha256_file(destination / asset['name']) == asset['sha256']
    with tarfile.open(destination / result['archive']) as tar:
        names = tar.getnames()
    assert all(name.startswith('supplement/') or name in {'FILES.json', 'SOURCE_REFERENCE.json'} for name in names)
    assert not any(name.startswith(('originals/', 'continuation/')) for name in names)
    reference = read_json(destination / 'SOURCE_REFERENCE.json')
    assert reference['original_checkpoint'] == enabled['plan']['identity']['checkpoint_reference']
    assert reference['original_attempts'] == 781
    assert reference['original_unknown_usage_reservation_reconciled'] is False
    assert len(reference['source_controls_sha256']) == 4
    assert all(Path(path).read_bytes() == value for path, value in before.items())
    source_unchanged(enabled)


def test_completed_fixture_twenty_snapshot_preserves_claims_and_responses(enabled):
    calls = []
    def verified(plan):
        ref = plan['identity']['checkpoint_reference']
        return {'status': 'GITHUB_VERIFIED', 'release_id': ref['release_id'],
            'verifiedledger_sha256': plan['identity']['source_snapshot']['ledger_sha256'],
            'attempted_inputs': 781, 'assets': [{'name': 'continuation-point-in-time.tar.gz',
               'asset_id': ref['archive_asset_id'], 'sha256': ref['archive_sha256']}]}
    runner.run(enabled['path'], permission=True, transports=transports(calls), checkpoint_verifier=verified,
               checkpoint_callback=lambda *_: {'status': 'LOCAL_VERIFIED'})
    assert len(calls) == 20
    state = ledger(enabled)
    result = save(enabled, state)
    assert result['execution']['supplemental_attempted'] == 20
    assert result['execution']['manual_auth_retry_attempted'] == 1
    restored = Path(result['checkpoint_directory']) / 'restored-verification/supplement/enabled'
    assert read_json(restored / 'ledger.json') == state
    assert len(list((restored / 'reservations').glob('*.json'))) == 20
    assert len(list((restored / 'attempts').glob('*/raw.json'))) == 20
    assert len(list((restored / 'attempts').glob('*/parsed.json'))) == 20
    source_unchanged(enabled)


def test_reserved_crash_snapshot_retains_unknown_budget_and_immutable_claim(enabled):
    state = reserved(enabled)
    before = (enabled['directory'] / 'ledger.json').read_bytes()
    result = save(enabled, state)
    destination = Path(result['checkpoint_directory'])
    snapshot = read_json(destination / 'snapshot.json')
    assert snapshot['terminal_reserved_attempts'] == 1
    assert snapshot['supplemental_unknown_usage_attempts'] == 1
    assert result['execution']['supplemental_unknown_usage_reservation_usd'] > 0
    assert (destination / 'restored-verification/supplement/enabled/ledger.json').read_bytes() == before
    assert (enabled['directory'] / 'ledger.json').read_bytes() == before
    source_unchanged(enabled)


@pytest.mark.parametrize('fault', ['extra_file', 'extra_directory', 'symlink', 'missing_artifact', 'changed_claim'])
def test_unexpected_or_changed_files_rejected(enabled, fault):
    state = reserved(enabled, {'model': 'fixture_response'})
    directory = enabled['directory']
    if fault == 'extra_file': (directory / 'unapproved.txt').write_text('fixture')
    if fault == 'extra_directory': (directory / 'unexpected').mkdir()
    if fault == 'symlink': (directory / 'unapproved-link').symlink_to(directory / 'ledger.json')
    if fault == 'missing_artifact': next((directory / 'attempts').glob('*/raw.json')).unlink()
    if fault == 'changed_claim': write_json(next((directory / 'reservations').glob('*.json')), {'changed': True})
    with pytest.raises((GenerationError, ValueError)):
        save(enabled, state)
    source_unchanged(enabled)


def _ciphertext():
    stamp = int(datetime(2026, 10, 15, tzinfo=timezone.utc).timestamp())
    value = base64.urlsafe_b64encode(b'\x80' + stamp.to_bytes(8, 'big') + b'\x00' * 160).decode().rstrip('=')
    value = value[:100] + '_sk-' + 'A' * 40 + value[144:]
    return value + '=' * (-len(value) % 4)


def test_provider_encrypted_reasoning_exception_preserves_raw_but_not_other_secrets(enabled):
    model = enabled['plan']['identity']['pricing']['openai']['accepted_response_model_ids'][0]
    raw = {'model': model, 'output': [{'type': 'reasoning', 'encrypted_content': _ciphertext()}]}
    state = reserved(enabled, raw)
    original_raw = next((enabled['directory'] / 'attempts').glob('*/raw.json'))
    raw_bytes = original_raw.read_bytes()
    result = save(enabled, state)
    destination = Path(result['checkpoint_directory'])
    manifest = read_json(destination / 'FILES.json')
    exceptions = manifest['metadata']['public_safety_scanner_exceptions']
    assert len(exceptions) == 1 and exceptions[0]['field'] == 'output[0].encrypted_content'
    restored_raw = next((destination / 'restored-verification/supplement/enabled/attempts').glob('*/raw.json'))
    assert original_raw.read_bytes() == restored_raw.read_bytes() == raw_bytes
    raw['unapproved_field'] = 'sk-' + 'z' * 40
    state = reserved(enabled, raw)
    with pytest.raises(ValueError, match='outside'):
        save(enabled, state)
    source_unchanged(enabled)


def test_secret_like_summary_is_never_archived(enabled):
    write_json(enabled['directory'] / 'summary.json', {'synthetic_only': 'sk-' + 'z' * 40})
    with pytest.raises(ValueError, match='Secret-like'):
        save(enabled)
    source_unchanged(enabled)
