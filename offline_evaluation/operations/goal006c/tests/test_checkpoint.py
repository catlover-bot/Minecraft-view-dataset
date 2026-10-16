"""Actual Goal006C local archives and restored provenance, without live HTTP."""
from pathlib import Path

import pytest

from offline_evaluation.operations.goal006c import archive, checkpoint, runner
from offline_evaluation.operations.goal006c.tests.test_archive import Github
from offline_evaluation.operations.goal006c.tests.test_runner import enable, execute
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.generation import GenerationError, read_json, write_json


@pytest.fixture
def enabled(fixture_plan, tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, 'CHECKPOINT_ROOT', tmp_path / 'checkpoints')
    fixture_plan['directory'] = enable(fixture_plan)
    return fixture_plan


def save(fixture):
    directory = fixture['directory']
    return checkpoint.save_local(fixture['plan'], directory, read_json(directory / 'ledger.json'), 'fixture_safe_state')


def unchanged(fixture):
    assert all(Path(p).read_bytes() == data for p, data in fixture['source_before'].items())


def test_empty_c_snapshot_references_both_halts_and_all_source_controls(enabled):
    result = save(enabled)
    assert result['status'] == 'LOCAL_VERIFIED' and result['github_status'] == 'NOT_UPLOADED'
    assert len(result['assets']) == 6 and result['source_responses_duplicated'] is False
    path = Path(result['checkpoint_directory'])
    reference = read_json(path / 'SOURCE_REFERENCE.json')
    assert reference['original_attempts'] == 781 and reference['source_attempts'] == 3
    assert reference['original_halt_reason'] == 'unknown_usage:auth_failure'
    assert reference['source_halt_reason'] == 'unknown_usage:http_failure'
    assert reference['original_unknown_usage_reservation_reconciled'] is False
    assert reference['source_unknown_usage_reservation_reconciled'] is False
    assert len(reference['source_controls_sha256']) == 8
    package = archive._package(path)
    proof = archive.verify_restored_prefix(path / 'restored-verification', package, path)
    assert proof['package_kind'] == 'goal006c' and proof['attempted_inputs'] == 0
    assert 'canonical' in proof['copied_plan_load_rejection']
    unchanged(enabled)


def test_completed_c17_real_local_and_remote_fixture_restore(enabled, monkeypatch, tmp_path):
    calls = []; execute(enabled, calls)
    assert len(calls) == 17
    result = save(enabled)
    path = Path(result['checkpoint_directory'])
    fixture = {'path': path, 'package': archive._package(path)}
    github = Github(fixture); monkeypatch.setattr(archive, '_gh', github)
    verified = archive.recover(path, tmp_path / 'remote-restored', release_id=100)
    assert verified['status'] == 'GITHUB_VERIFIED' and verified['package_kind'] == 'goal006c'
    assert verified['attempted_inputs'] == verified['claims'] == 17
    assert verified['artifacts'] == 51 and len(verified['assets']) == 6
    assert result['execution']['combined_calls'] == 991
    unchanged(enabled)


def test_unknown_third_hold_is_saved_without_releasing_old_two(enabled):
    calls = []
    def unknown(*args):
        raise providers.ProviderError('http_failure', 520)
    execute(enabled, calls, unknown)
    assert len(calls) == 1
    before = (enabled['directory'] / 'ledger.json').read_bytes()
    result = save(enabled)
    restored = Path(result['checkpoint_directory']) / 'restored-verification/supplement/enabled/ledger.json'
    assert restored.read_bytes() == before
    assert read_json(restored)['entries'][0]['usage'] is None
    assert result['execution']['historical_unknown_usage_reservation_reconciled'] is False
    assert result['execution']['parent_budget_accounting']['unknown_usage_attempts'] == 2
    unchanged(enabled)


@pytest.mark.parametrize('fault', ['unexpected', 'symlink', 'secret'])
def test_checkpoint_refuses_unapproved_paths_and_secret_fields(enabled, fault):
    if fault == 'unexpected': (enabled['directory'] / 'extra.txt').write_text('fixture')
    if fault == 'symlink': (enabled['directory'] / 'extra').symlink_to(enabled['directory'] / 'ledger.json')
    if fault == 'secret': write_json(enabled['directory'] / 'summary.json', {'synthetic_key': 'sk-' + 'z' * 40})
    with pytest.raises((GenerationError, ValueError)):
        save(enabled)
    unchanged(enabled)
