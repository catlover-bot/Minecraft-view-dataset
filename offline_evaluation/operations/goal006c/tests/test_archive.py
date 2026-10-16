"""Supplement schema, immutable remote identity and actual archive restore fixtures."""
from copy import deepcopy
import io
from pathlib import Path
import tarfile
from urllib.parse import parse_qs, urlsplit

import pytest

from offline_evaluation.batch_v1.tests.test_runner import response, transports
from offline_evaluation.operations.goal006b import checkpoint as old_checkpoint, runner as old_runner
from offline_evaluation.operations.goal006b.tests.test_runner import approval, verified_checkpoint
from offline_evaluation.operations.goal006c import archive
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.generation import read_json, write_json


@pytest.fixture
def b_package(source_fixture, tmp_path, monkeypatch):
    source = source_fixture
    directory = old_runner.initialize_enabled(source['path'], approval=approval(source['plan']))
    calls = []
    def three(provider, payload, count):
        if count == 3:
            raise providers.ProviderError('http_failure', 520)
        return response(provider)
    old_runner.run(source['path'], permission=True, transports=transports(calls, three),
                   checkpoint_verifier=verified_checkpoint, checkpoint_callback=lambda *_: {'status': 'LOCAL_VERIFIED'})
    assert len(calls) == 3
    monkeypatch.setattr(old_checkpoint, 'CHECKPOINT_ROOT', tmp_path / 'checkpoints')
    saved = old_checkpoint.save_local(source['plan'], directory, read_json(directory / 'ledger.json'), 'fixture_520_halt')
    path = Path(saved['checkpoint_directory'])
    before = {str(p): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    return {'path': path, 'source': source, 'package': archive._package(path), 'before': before}


class Github:
    def __init__(self, fixture):
        self.fixture, self.package = fixture, fixture['package']
        self.release = {'id': 100, 'draft': True, 'published_at': None,
            'tag_name': self.package['tag'], 'target_commitish': self.package['target_commitish'],
            'body': archive._release_body(self.package)}
        self.releases = [[self.release], []]
        self.assets = [{**self.row(asset, index + 200)} for index, asset in enumerate(self.package['assets'])]
        self.calls = []
        self.corrupt_download = None
        self.fail_stage = None
        self.ancestry = {'status': 'ahead', 'behind_by': 0, 'base_commit': {'sha': self.package['target_commitish']}}

    @staticmethod
    def row(asset, identifier):
        return {'id': identifier, 'name': asset['name'], 'size': asset['bytes'],
                'digest': 'sha256:' + asset['sha256'], 'state': 'uploaded'}

    def __call__(self, stage, args, *, output=None):
        self.calls.append((stage, args))
        if stage == self.fail_stage:
            raise archive.RecoveryError(stage, 'fixture_http_failure', returncode=1, http_status=503)
        if stage == 'commit_ancestry':
            return self.ancestry
        if stage == 'release_list':
            page = int(parse_qs(urlsplit(args[1]).query)['page'][0])
            return deepcopy(self.releases[page - 1] if page <= len(self.releases) else [])
        if stage == 'release_lookup':
            return deepcopy(self.release)
        if stage == 'release_create':
            request = read_json(Path(args[-1]))
            assert request['draft'] is True and request['tag_name'] == self.package['tag']
            assert request['target_commitish'] == self.package['target_commitish']
            assert request['body'] == archive._release_body(self.package)
            return deepcopy(self.release)
        if stage == 'asset_list':
            page = int(parse_qs(urlsplit(args[1]).query)['page'][0])
            return deepcopy(self.assets if page == 1 else [])
        if stage == 'asset_upload':
            url = next(a for a in args if a.startswith('https://uploads.github.com/'))
            name = parse_qs(urlsplit(url).query)['name'][0]
            assert name not in {a['name'] for a in self.assets}
            desired = next(a for a in self.package['assets'] if a['name'] == name)
            assert f"Content-Length: {desired['bytes']}" in args
            assert Path(args[-1]).read_bytes() == (self.fixture['path'] / name).read_bytes()
            row = self.row(desired, 300 + len(self.assets)); self.assets.append(row)
            return deepcopy(row)
        if stage == 'asset_download':
            identifier = int(args[1].rsplit('/', 1)[-1])
            row = next(a for a in self.assets if a['id'] == identifier)
            data = (self.fixture['path'] / row['name']).read_bytes()
            if row['name'] == self.corrupt_download:
                data += b'corruption'
            with Path(output).open('xb') as stream:
                stream.write(data)
            return None
        raise AssertionError(stage)


def recover(fixture, monkeypatch, tmp_path, *, remote=None, **options):
    remote = remote or Github(fixture)
    monkeypatch.setattr(archive, '_gh', remote)
    result = archive.recover(fixture['path'], tmp_path / 'verification', **options)
    return result, remote


def test_b3_six_numeric_downloads_real_safe_restore_and_unknown_hold(b_package, monkeypatch, tmp_path):
    result, remote = recover(b_package, monkeypatch, tmp_path, release_id=100)
    assert result['status'] == 'GITHUB_VERIFIED' and result['package_kind'] == 'goal006b'
    assert result['archive'] == 'supplement-point-in-time.tar.gz'
    assert result['attempted_inputs'] == result['claims'] == 3 and result['artifacts'] == 6
    assert len(result['assets']) == 6 and result['uploaded_assets'] == []
    assert result['source_references_verified'] and result['public_safety_scan_verified']
    assert 'canonical' in result['copied_plan_load_rejection']
    assert sum(stage == 'asset_download' for stage, _ in remote.calls) == 6
    assert not any(stage in {'release_create', 'asset_upload'} for stage, _ in remote.calls)
    restored = Path(result['verification_directory']) / 'restored/supplement/enabled/ledger.json'
    ledger = read_json(restored)
    assert ledger['halt_reason'] == 'unknown_usage:http_failure'
    assert ledger['entries'][-1]['usage'] is None and ledger['entries'][-1]['artifacts'] == {}
    assert ledger['entries'][-1]['charged_cost_usd'] == .165536
    assert all(Path(p).read_bytes() == data for p, data in b_package['before'].items())


def test_paginated_draft_reuse_never_creates_duplicate(b_package, monkeypatch, tmp_path):
    remote = Github(b_package)
    remote.releases = [[{**remote.release, 'id': 99, 'tag_name': 'other'}], [remote.release], []]
    result, remote = recover(b_package, monkeypatch, tmp_path, remote=remote, allow_create=True)
    assert result['release_id'] == 100 and not result['uploaded_assets']
    assert sum(stage == 'release_list' for stage, _ in remote.calls) == 3
    assert not any(stage == 'release_create' for stage, _ in remote.calls)


@pytest.mark.parametrize('fault', ['duplicate_tag', 'duplicate_id_page', 'target', 'published', 'digest', 'extra_asset', 'wrong_full_identity_empty_assets'])
def test_remote_collisions_fail_closed_before_any_mutation(b_package, monkeypatch, tmp_path, fault):
    remote = Github(b_package)
    if fault == 'duplicate_tag': remote.releases = [[remote.release], [{**remote.release, 'id': 101}], []]
    if fault == 'duplicate_id_page': remote.releases = [[remote.release], [remote.release], []]
    if fault == 'target': remote.release['target_commitish'] = 'f' * 40
    if fault == 'published': remote.release['draft'] = False
    if fault == 'digest': remote.assets[0]['digest'] = 'sha256:' + '0' * 64
    if fault == 'extra_asset': remote.assets.append({**remote.assets[0], 'id': 999, 'name': 'unexpected'})
    if fault == 'wrong_full_identity_empty_assets':
        remote.release['body'] = remote.release['body'].replace(b_package['package']['plan_identity_hash'], 'f' * 64)
        remote.assets = []
    monkeypatch.setattr(archive, '_gh', remote)
    with pytest.raises(archive.RecoveryError):
        archive.recover(b_package['path'], tmp_path / 'verification', allow_create=True)
    assert not any(stage in {'release_create', 'asset_upload'} for stage, _ in remote.calls)


def test_create_only_after_complete_absence_and_uploads_all_six(b_package, monkeypatch, tmp_path):
    remote = Github(b_package); remote.releases = [[]]; remote.assets = []
    result, remote = recover(b_package, monkeypatch, tmp_path, remote=remote, allow_create=True)
    assert len(result['uploaded_assets']) == 6
    assert sum(stage == 'release_create' for stage, _ in remote.calls) == 1
    assert sum(stage == 'asset_download' for stage, _ in remote.calls) == 6


@pytest.mark.parametrize('allow_create', [False, True])
def test_partial_upload_resumes_only_missing_without_overwrite(b_package, monkeypatch, tmp_path, allow_create):
    remote = Github(b_package); missing = remote.assets.pop()['name']
    if allow_create:
        result, remote = recover(b_package, monkeypatch, tmp_path, remote=remote, allow_create=True)
        assert [a['name'] for a in result['uploaded_assets']] == [missing]
    else:
        monkeypatch.setattr(archive, '_gh', remote)
        with pytest.raises(archive.RecoveryError, match='missing_assets'):
            archive.recover(b_package['path'], tmp_path / 'verification')
        assert not any(stage == 'asset_upload' for stage, _ in remote.calls)


@pytest.mark.parametrize('fault', ['download_digest', 'download_503', 'ancestry'])
def test_save_failure_never_claims_github_verified(b_package, monkeypatch, tmp_path, fault):
    remote = Github(b_package)
    if fault == 'download_digest': remote.corrupt_download = 'FILES.json'
    if fault == 'download_503': remote.fail_stage = 'asset_download'
    if fault == 'ancestry': remote.ancestry['behind_by'] = 1
    monkeypatch.setattr(archive, '_gh', remote)
    with pytest.raises(archive.RecoveryError):
        archive.recover(b_package['path'], tmp_path / 'verification', release_id=100)
    assert not (tmp_path / 'verification/verification.json').exists()
    assert read_json(tmp_path / 'verification/failure.json')['status'] == 'FAILED'
    assert all(Path(p).read_bytes() == data for p, data in b_package['before'].items())


@pytest.mark.parametrize('fault', ['claim', 'ledger_halt', 'authority', 'source_reference', 'snapshot', 'extra_file'])
def test_restored_prefix_control_corruption_rejected(b_package, tmp_path, fault):
    restored = tmp_path / 'restored'
    package = b_package['package']
    archive.restore_archive(b_package['path'] / package['archive'], restored, package['archive_sha256'])
    if fault == 'claim': path = next((restored / 'supplement/enabled/reservations').glob('*.json'))
    if fault == 'ledger_halt': path = restored / 'supplement/enabled/ledger.json'
    if fault == 'authority': path = restored / 'supplement/authority.json'
    if fault == 'source_reference': path = restored / 'SOURCE_REFERENCE.json'
    if fault == 'extra_file': path = restored / 'unapproved.json'
    if fault != 'snapshot': write_json(path, {'corruption': True})
    else:
        snapshot = read_json(b_package['path'] / 'snapshot.json')
        snapshot['original_sha256'] = {}
        write_json(b_package['path'] / 'snapshot.json', snapshot)
    with pytest.raises((archive.RecoveryError, KeyError)):
        archive.verify_restored_prefix(restored, package, b_package['path'])


def test_archive_traversal_rejected_before_any_extraction(tmp_path):
    path = tmp_path / 'hostile.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        data = b'{}'; info = tarfile.TarInfo('../outside.json'); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    from offline_evaluation.server_v2.data import sha256_file
    with pytest.raises(ValueError):
        archive.restore_archive(path, tmp_path / 'restore', sha256_file(path))
    assert not (tmp_path / 'outside.json').exists()


def test_oversized_asset_is_rejected_before_github(b_package, monkeypatch, tmp_path):
    remote = Github(b_package)
    monkeypatch.setattr(archive, '_gh', remote)
    monkeypatch.setattr(archive, 'MAX_ASSET_BYTES', 1)
    with pytest.raises(archive.RecoveryError, match='1GiB'):
        archive.recover(b_package['path'], tmp_path / 'verification', allow_create=True)
    assert remote.calls == []
