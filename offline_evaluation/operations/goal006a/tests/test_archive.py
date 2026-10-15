"""Fixture-only archive recovery: no real HTTP or research calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from offline_evaluation.operations.goal006a import archive as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def package(tmp_path, monkeypatch):
    folder = tmp_path / 'package'
    folder.mkdir()
    for name in a.ASSET_NAMES - {'SHA256SUMS'}:
        (folder / name).write_bytes(b'fixture archive' if name.endswith('.gz') else b'{}\n')
    (folder / 'SHA256SUMS').write_text(''.join(f'{sha(folder / name)}  {name}\n' for name in sorted(a.ASSET_NAMES - {'SHA256SUMS'})))
    assets = [{'name': name, 'sha256': sha(folder / name), 'bytes': (folder / name).stat().st_size}
              for name in sorted(a.ASSET_NAMES)]
    doc = {'repository': a.REPOSITORY, 'version': 'goal006_checkpoint_package_v1',
           'checkpoint_id': 'goal006-fixture-0320', 'tag': 'checkpoint-goal006-fixture-0320',
           'target_commitish': 'a' * 40, 'plan_identity_hash': 'b' * 64,
           'archive': 'continuation-point-in-time.tar.gz', 'assets': assets}
    write(folder / 'checkpoint.json', doc)
    monkeypatch.setattr(a, 'restore_archive', lambda *args: {'status': 'VERIFIED', 'files': 8})
    monkeypatch.setattr(a.frozen, '_verify_restored_public', lambda *args: 'canonical_copy_rejected')
    monkeypatch.setattr(a, 'verify_restored_prefix', lambda *args: {
        'verifiedledger_sha256': 'c' * 64, 'attempted_inputs': 320,
        'plan_identity_hash': doc['plan_identity_hash']})
    return folder, doc


class GitHubFixture:
    def __init__(self, folder, doc):
        self.folder, self.doc = folder, doc
        self.release = {'id': 100, 'draft': True, 'tag_name': doc['tag'],
                        'target_commitish': doc['target_commitish']}
        self.assets = [{**asset, 'id': index + 200, 'size': asset['bytes'],
                        'digest': 'sha256:' + asset['sha256'], 'state': 'uploaded'}
                       for index, asset in enumerate(doc['assets'])]
        self.calls = []
        self.release_pages = {1: [self.release], 2: []}
        self.missing = set()
        self.corrupt_download = False

    def __call__(self, stage, args, *, output=None):
        self.calls.append((stage, args, output))
        endpoint = next((x for x in args if x.startswith(('repos/', 'https://uploads.github.com/repos/'))), '')
        if output is not None:
            assert '/releases/assets/' in endpoint
            asset_id = int(endpoint.rsplit('/', 1)[1])
            asset = next(x for x in self.assets if x['id'] == asset_id)
            Path(output).write_bytes(b'bad' if self.corrupt_download else (self.folder / asset['name']).read_bytes())
            asset['download_count'] = asset.get('download_count', 0) + 1
            asset['updated_at'] = 'legitimately changed after download'
            return
        if '/compare/' in endpoint:
            return {'status': 'ahead', 'behind_by': 0}
        if stage == 'asset_upload':
            assert '--hostname' not in args and endpoint.startswith('https://uploads.github.com/')
            assert '/releases/100/assets?name=' in endpoint
            name = endpoint.split('name=')[1]
            assert f'Content-Length: {(self.folder / name).stat().st_size}' in args
            assert 'Content-Type: application/octet-stream' in args
            self.missing.discard(name)
            return next(x for x in self.assets if x['name'] == name)
        if stage == 'release_create':
            return deepcopy(self.release)
        if '/assets?per_page=' in endpoint:
            return deepcopy([x for x in self.assets if x['name'] not in self.missing]) if endpoint.endswith('page=1') else []
        if '/releases?per_page=' in endpoint:
            return deepcopy(self.release_pages[int(endpoint.rsplit('=', 1)[1])])
        if endpoint.endswith('/releases/100'):
            return deepcopy(self.release)
        raise AssertionError((stage, args))


def test_existing_six_assets_numeric_ids_no_reupload(package, tmp_path, monkeypatch):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    monkeypatch.setattr(a, '_gh', gh)
    report = a.recover(folder, tmp_path / 'verified', release_id=100)
    assert report['status'] == 'GITHUB_VERIFIED'
    assert report['attempted_inputs'] == 320 and report['uploaded_assets'] == []
    downloads = [args for stage, args, _ in gh.calls if stage == 'asset_download']
    assert len(downloads) == 6
    assert all('/releases/assets/' in ' '.join(args) for args in downloads)
    assert not any(stage in {'release_create', 'asset_upload', 'release_list'} for stage, *_ in gh.calls)
    assert (tmp_path / 'verified/verification.json').exists()


def test_paginated_draft_lookup_second_page_never_tags(package, tmp_path, monkeypatch):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    gh.release_pages = {1: [{'id': 9, 'tag_name': 'unrelated'}], 2: [gh.release], 3: []}
    monkeypatch.setattr(a, '_gh', gh)
    report = a.recover(folder, tmp_path / 'verified', allow_create=True)
    assert report['release_id'] == 100
    calls = [' '.join(args) for _, args, _ in gh.calls]
    assert any('page=3' in x and '/releases?' in x for x in calls)
    assert not any('/tags/' in x or 'release download' in x or 'release upload' in x for x in calls)
    assert not any(stage == 'release_create' for stage, *_ in gh.calls)


def test_ambiguous_draft_refuses_creation(package, tmp_path, monkeypatch):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    gh.release_pages[1].append({**gh.release, 'id': 101})
    monkeypatch.setattr(a, '_gh', gh)
    with pytest.raises(a.RecoveryError, match='ambiguous_exact_tag'):
        a.recover(folder, tmp_path / 'verified', allow_create=True)
    assert not any(stage == 'release_create' for stage, *_ in gh.calls)


def test_missing_asset_strict_recovery_never_uploads(package, tmp_path, monkeypatch):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    gh.missing.add('RESTORE.md')
    monkeypatch.setattr(a, '_gh', gh)
    with pytest.raises(a.RecoveryError, match='missing_assets_no_upload_authorized'):
        a.recover(folder, tmp_path / 'verified', release_id=100)
    assert not any(stage == 'asset_upload' for stage, *_ in gh.calls)


def test_future_missing_asset_upload_by_release_id(package, tmp_path, monkeypatch):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    gh.missing.add('RESTORE.md')
    monkeypatch.setattr(a, '_gh', gh)
    report = a.recover(folder, tmp_path / 'verified', allow_create=True)
    assert [x['name'] for x in report['uploaded_assets']] == ['RESTORE.md']
    assert len([1 for stage, *_ in gh.calls if stage == 'asset_upload']) == 1


def test_future_create_only_after_complete_absence(package, tmp_path, monkeypatch):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    gh.release_pages = {1: []}
    gh.missing = set(a.ASSET_NAMES)
    monkeypatch.setattr(a, '_gh', gh)
    report = a.recover(folder, tmp_path / 'verified', allow_create=True)
    assert len(report['uploaded_assets']) == 6
    stages = [stage for stage, *_ in gh.calls]
    assert stages.index('release_list') < stages.index('release_create')


@pytest.mark.parametrize('fault', ['digest', 'size', 'extra', 'published', 'target', 'tag'])
def test_collisions_reject_without_remote_write(package, tmp_path, monkeypatch, fault):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    if fault == 'digest': gh.assets[0]['digest'] = 'sha256:' + '0' * 64
    if fault == 'size': gh.assets[0]['size'] += 1
    if fault == 'extra': gh.assets.append({**gh.assets[0], 'id': 999, 'name': 'unexpected'})
    if fault == 'published': gh.release['draft'] = False
    if fault == 'target': gh.release['target_commitish'] = '0' * 40
    if fault == 'tag': gh.release['tag_name'] += '-wrong'
    monkeypatch.setattr(a, '_gh', gh)
    with pytest.raises(a.RecoveryError):
        a.recover(folder, tmp_path / 'verified', release_id=100, allow_create=True)
    assert not any(stage in {'release_create', 'asset_upload'} for stage, *_ in gh.calls)
    assert json.loads((tmp_path / 'verified/failure.json').read_text())['status'] == 'FAILED'


def test_download_mismatch_never_reports_success(package, tmp_path, monkeypatch):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    gh.corrupt_download = True
    monkeypatch.setattr(a, '_gh', gh)
    with pytest.raises(a.RecoveryError, match='redownload_hash_or_size_mismatch'):
        a.recover(folder, tmp_path / 'verified', release_id=100)
    assert not (tmp_path / 'verified/verification.json').exists()


def test_safe_failure_records_stage_exit_http_not_provider_text(monkeypatch):
    secret = ''.join(('Author', 'ization: ', 'token ', 'synthetic-fixture-value-', 'never-a-credential'))
    monkeypatch.setattr(a.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 1, b'', (secret + ' (HTTP 404)').encode()))
    with pytest.raises(a.RecoveryError) as result:
        a._gh('asset_download', ['api', 'fixture'])
    assert result.value.report == {'stage': 'asset_download', 'code': 'github_command_failed',
                                   'returncode': 1, 'http_status': 404}
    assert secret not in str(result.value)


def test_uncertain_lookup_never_creates(package, tmp_path, monkeypatch):
    folder, doc = package
    gh = GitHubFixture(folder, doc)
    def fail(stage, args, **kwargs):
        if stage == 'release_list':
            raise a.RecoveryError(stage, 'github_command_failed', returncode=1, http_status=503)
        return gh(stage, args, **kwargs)
    monkeypatch.setattr(a, '_gh', fail)
    with pytest.raises(a.RecoveryError, match='503'):
        a.recover(folder, tmp_path / 'verified', allow_create=True)
    assert not any(stage == 'release_create' for stage, *_ in gh.calls)


def test_existing_verification_directory_never_overwritten(package, tmp_path):
    folder, _ = package
    target = tmp_path / 'existing'
    target.mkdir()
    (target / 'preserved').write_text('original')
    with pytest.raises(a.RecoveryError, match='new_directory_required'):
        a.recover(folder, target, release_id=100)
    assert (target / 'preserved').read_text() == 'original'


@pytest.fixture
def provenance(tmp_path, monkeypatch):
    """Small real filesystem ledger: only upstream model payload construction stubbed."""
    from offline_evaluation.continuation_v1 import plan as planning, runner
    import shutil
    live = tmp_path / 'live'
    enabled = live / 'enabled'
    canonical = live / 'plan.json'
    authority = live / 'authority.json'
    record = {'schedule_index': 160, 'input_id': 'fixture-input', 'provider': 'openai'}
    identity = {'plan_path': str(canonical), 'enabled_directory': str(enabled),
                'authority_path': str(authority), 'protocol_id': 'fixture_protocol',
                'schedule': [record]}
    plan = {'identity': identity, 'identity_hash': a.object_hash(identity)}
    write(canonical, plan)
    monkeypatch.setattr(planning, 'load_plan', lambda path: a.read_json(path))
    monkeypatch.setattr(runner, '_payload', lambda identity, record: {'input': record['input_id']})
    monkeypatch.setattr(runner, 'execution_summary', lambda plan, ledger: {'attempted_inputs': len(ledger['entries'])})
    approval = {'manual_confirmed': True, 'plan_identity_hash': plan['identity_hash'],
                'enabled_directory': str(enabled), 'confirmation_channel': '/dev/tty',
                'scope': runner.APPROVAL_SCOPE, 'approved_at_utc': '2026-10-14T00:00:00Z'}
    write(enabled / 'approval.json', approval)
    write(enabled / 'enabled.json', {'api_allowed': True, 'plan_identity_hash': plan['identity_hash'],
          'approval_hash': a.object_hash(approval), 'enabled_directory': str(enabled), 'protocol_id': identity['protocol_id']})
    write(authority, runner._authority(plan, approval))
    aid = 'd' * 32
    raw = enabled / 'attempts' / aid / 'raw.json'
    write(raw, {'model': 'fixture', 'output': 'fixture'})
    entry = {'attempt_id': aid, 'local_index': 0, **record, 'retry': False,
             'request_payload_hash': a.object_hash({'input': record['input_id']}),
             'artifacts': {'raw.json': {'path': f'attempts/{aid}/raw.json', 'sha256': sha(raw), 'bytes': raw.stat().st_size}}}
    write(enabled / 'reservations' / (aid + '.json'), runner._claim(plan, entry))
    ledger = {'version': 'continuation_v1_ledger_v1', 'plan_identity_hash': plan['identity_hash'],
              'entries': [entry], 'halt_reason': None}
    write(enabled / 'ledger.json', ledger)
    restored = tmp_path / 'restored'
    shutil.copytree(live, restored / 'continuation')
    sidecars = tmp_path / 'sidecars'
    sidecars.mkdir()
    manifest = {'files': [{'path': 'continuation/plan.json'}]}
    write(sidecars / 'FILES.json', manifest)
    shutil.copyfile(sidecars / 'FILES.json', restored / 'FILES.json')
    package = {'plan_identity_hash': plan['identity_hash'], 'checkpoint_id': 'fixture',
               'archive': 'continuation-point-in-time.tar.gz',
               'assets': [{'name': 'continuation-point-in-time.tar.gz', 'sha256': '0' * 64}]}
    write(sidecars / 'snapshot.json', {'checkpoint_id': 'fixture', 'plan_identity_hash': plan['identity_hash'],
          'canonical_plan_sha256': sha(canonical), 'archive_sha256': '0' * 64,
          'file_count': 1, 'execution': {'attempted_inputs': 1}})
    return restored, package, sidecars, live, ledger


def test_restored_prefix_accepts_current_immutable_originals(provenance):
    restored, package, sidecars, live, ledger = provenance
    result = a.verify_restored_prefix(restored, package, sidecars)
    assert result['attempted_inputs'] == 1 and result['claims'] == 1 and result['artifacts'] == 1
    assert result['verifiedledger_sha256'] == sha(live / 'enabled/ledger.json')


def test_restored_prefix_allows_only_live_extension(provenance):
    restored, package, sidecars, live, ledger = provenance
    ledger['entries'].append({'later': 'immutable fixture entry'})
    write(live / 'enabled/ledger.json', ledger)
    result = a.verify_restored_prefix(restored, package, sidecars)
    assert result['attempted_inputs'] == 1 and result['current_attempted_inputs'] == 2


@pytest.mark.parametrize('fault', ['shrink', 'replace', 'claim', 'raw', 'approval', 'authority', 'extra_artifact', 'extra_claim'])
def test_restored_prefix_rejects_missing_or_replaced_originals(provenance, fault):
    restored, package, sidecars, live, ledger = provenance
    base = restored / 'continuation'
    aid = 'd' * 32
    if fault == 'shrink':
        ledger['entries'] = []
        write(live / 'enabled/ledger.json', ledger)
    if fault == 'replace':
        ledger['entries'][0]['input_id'] = 'replaced'
        write(live / 'enabled/ledger.json', ledger)
    if fault == 'claim': write(base / 'enabled/reservations' / (aid + '.json'), {'changed': True})
    if fault == 'raw': write(base / 'enabled/attempts' / aid / 'raw.json', {'changed': True})
    if fault == 'approval': write(live / 'enabled/approval.json', {'changed': True})
    if fault == 'authority': write(base / 'authority.json', {'changed': True})
    if fault == 'extra_artifact': write(base / 'enabled/attempts/extra/raw.json', {})
    if fault == 'extra_claim': write(base / 'enabled/reservations/extra.json', {})
    with pytest.raises(a.RecoveryError):
        a.verify_restored_prefix(restored, package, sidecars)
