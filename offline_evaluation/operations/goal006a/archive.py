"""Save-only GitHub recovery using numeric release and asset identities.

This operations directory is outside every frozen implementation hash glob.
No function calls generation, edits live state, overwrites assets, or publishes.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

from ...continuation_v1 import checkpoint as frozen
from ...server_v2.checkpoint import restore_archive, safe_relative, scan_file
from ...server_v2.data import sha256_file
from ...server_v2.generation import object_hash, read_json, utc_now, write_json
from ...server_v2.io_utils import writable

REPOSITORY = 'catlover-bot/Minecraft-view-dataset'
BRANCH = 'research/server-only-20261013'
ASSET_NAMES = {'continuation-point-in-time.tar.gz', 'FILES.json', 'snapshot.json',
               'SOURCE_REFERENCE.json', 'RESTORE.md', 'SHA256SUMS'}


class RecoveryError(RuntimeError):
    """Safe diagnostics: never include gh stdout/stderr, URLs, or headers."""
    def __init__(self, stage, code, *, returncode=None, http_status=None):
        self.report = {'stage': stage, 'code': code, 'returncode': returncode,
                       'http_status': http_status}
        super().__init__(f'{stage}: {code}; exit={returncode}; http={http_status}')


def _fail(stage, code):
    raise RecoveryError(stage, code)


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        _fail('local_package', 'regular_nonsymlink_file_required')
    return path


def _gh(stage, args, *, output=None):
    """Keep raw provider output private, including possible signed redirect URLs."""
    try:
        if output is None:
            result = subprocess.run(['gh', *args], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, check=False)
        else:
            with Path(output).open('xb') as stream:
                result = subprocess.run(['gh', *args], stdout=stream,
                                        stderr=subprocess.PIPE, check=False)
    except (OSError, subprocess.SubprocessError):
        raise RecoveryError(stage, 'github_process_unavailable') from None
    if result.returncode:
        stderr = result.stderr.decode('utf-8', 'replace')
        matches = re.findall(r'\bHTTP(?:/\d(?:\.\d)?)?\s+(\d{3})\b', stderr)
        status = int(matches[-1]) if matches else None
        raise RecoveryError(stage, 'github_command_failed', returncode=result.returncode,
                            http_status=status)
    if output is not None:
        return None
    try:
        return json.loads(result.stdout)
    except (TypeError, ValueError):
        raise RecoveryError(stage, 'github_invalid_json') from None


def _pages(endpoint, stage):
    """Fetch until an empty page; never mistake a first-page miss for absence."""
    rows, identifiers, page = [], set(), 1
    while True:
        batch = _gh(stage, ['api', f'{endpoint}?per_page=100&page={page}'])
        if not isinstance(batch, list):
            _fail(stage, 'github_list_required')
        if not batch:
            return rows
        for row in batch:
            if not isinstance(row, dict) or not isinstance(row.get('id'), int) or row['id'] in identifiers:
                _fail(stage, 'github_list_duplicate_or_invalid_id')
            identifiers.add(row['id'])
        rows.extend(batch)
        page += 1
        if page > 10000:
            _fail(stage, 'github_pagination_limit')


def _package(directory):
    package = read_json(_regular(directory / 'checkpoint.json'))
    if (package.get('repository') != REPOSITORY
            or package.get('version') != 'goal006_checkpoint_package_v1'
            or not re.fullmatch(r'checkpoint-goal006-[A-Za-z0-9-]+', package.get('tag', ''))
            or not re.fullmatch(r'[0-9a-f]{40}', package.get('target_commitish', ''))
            or package.get('archive') != 'continuation-point-in-time.tar.gz'):
        _fail('local_package', 'invalid_package_identity')
    assets = package.get('assets', [])
    if (len(assets) != 6 or {a.get('name') for a in assets} != ASSET_NAMES):
        _fail('local_package', 'exact_six_assets_required')
    for asset in assets:
        path = _regular(directory / asset['name'])
        if sha256_file(path) != asset['sha256'] or path.stat().st_size != asset['bytes']:
            _fail('local_package', 'immutable_asset_hash_or_size_changed')
        if asset['name'] != package['archive']:
            scan_file(path)
    _checksums(directory, package)
    return package


def _checksums(directory, package):
    expected = {a['name']: a['sha256'] for a in package['assets'] if a['name'] != 'SHA256SUMS'}
    actual = {}
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        fields = line.split('  ', 1)
        if len(fields) != 2 or not re.fullmatch(r'[0-9a-f]{64}', fields[0]):
            _fail('checksums', 'invalid_checksum_format')
        value, name = fields
        safe_relative(name)
        if name in actual or name not in expected or sha256_file(directory / name) != value:
            _fail('checksums', 'checksum_duplicate_or_mismatch')
        actual[name] = value
    if actual != expected:
        _fail('checksums', 'checksum_manifest_mismatch')


def _release(package, attempt, release_id, allow_create):
    if release_id is not None:
        if not isinstance(release_id, int) or release_id <= 0:
            _fail('release_lookup', 'positive_numeric_release_id_required')
        release = _gh('release_lookup', ['api', f'repos/{REPOSITORY}/releases/{release_id}'])
        if release.get('id') != release_id:
            _fail('release_lookup', 'release_id_mismatch')
    else:
        releases = _pages(f'repos/{REPOSITORY}/releases', 'release_list')
        matches = [r for r in releases if r.get('tag_name') == package['tag']]
        if len(matches) > 1:
            _fail('release_lookup', 'ambiguous_exact_tag')
        if matches:
            release = matches[0]
        elif not allow_create:
            _fail('release_lookup', 'release_absent_no_creation_authorized')
        else:
            payload = {'tag_name': package['tag'], 'target_commitish': package['target_commitish'],
                       'draft': True, 'name': package['checkpoint_id'],
                       'body': 'Immutable Goal006 continuation checkpoint. Save-only operation; no generation requests.'}
            path = attempt / 'release-request.json'
            write_json(path, payload)
            release = _gh('release_create', ['api', '--method', 'POST',
                         f'repos/{REPOSITORY}/releases', '--input', str(path)])
    if (release.get('draft') is not True or release.get('tag_name') != package['tag']
            or release.get('target_commitish') != package['target_commitish']
            or not isinstance(release.get('id'), int)):
        _fail('release_identity', 'release_tag_target_or_draft_mismatch')
    return release


def _assets(release_id, package, *, require_complete):
    rows = _pages(f'repos/{REPOSITORY}/releases/{release_id}/assets', 'asset_list')
    current = {}
    expected = {a['name']: a for a in package['assets']}
    for row in rows:
        name = row.get('name')
        if name in current or name not in expected:
            _fail('asset_identity', 'unexpected_or_duplicate_asset')
        desired = expected[name]
        if (row.get('size') != desired['bytes'] or row.get('digest') != 'sha256:' + desired['sha256']
                or row.get('state') != 'uploaded'):
            _fail('asset_identity', 'existing_asset_size_digest_or_state_mismatch')
        current[name] = row
    if require_complete and set(current) != ASSET_NAMES:
        _fail('asset_identity', 'missing_assets_no_upload_authorized')
    return current


def verify_restored_prefix(restored, package, sidecars):
    """Prove archived controls and each original still belong to this live run.

    A later live ledger may extend the archived prefix, never shrink or replace
    it. This is byte/provenance validation; it does not re-evaluate old outputs.
    """
    from ...continuation_v1 import plan as planning, runner
    base = Path(restored) / 'continuation'
    copied_path = base / 'plan.json'
    copied = read_json(copied_path)
    identity = copied['identity']
    canonical = _regular(identity['plan_path'])
    live_plan = planning.load_plan(canonical)
    if (sha256_file(copied_path) != sha256_file(canonical) or copied != live_plan
            or copied['identity_hash'] != package['plan_identity_hash']
            or object_hash(identity) != copied['identity_hash']):
        _fail('restore_provenance', 'canonical_plan_mismatch')
    enabled = Path(identity['enabled_directory'])
    for name in ('approval.json', 'enabled.json'):
        if sha256_file(base / 'enabled' / name) != sha256_file(_regular(enabled / name)):
            _fail('restore_provenance', 'immutable_approval_or_enabled_mismatch')
    if sha256_file(base / 'authority.json') != sha256_file(_regular(identity['authority_path'])):
        _fail('restore_provenance', 'immutable_authority_mismatch')
    approval = read_json(base / 'enabled/approval.json')
    expected_enabled = {'api_allowed': True, 'plan_identity_hash': copied['identity_hash'],
        'approval_hash': object_hash(approval), 'enabled_directory': str(enabled),
        'protocol_id': identity['protocol_id']}
    if (read_json(base / 'enabled/enabled.json') != expected_enabled
            or read_json(base / 'authority.json') != runner._authority(copied, approval)):
        _fail('restore_provenance', 'authority_approval_binding_mismatch')
    if (approval.get('manual_confirmed') is not True or approval.get('plan_identity_hash') != copied['identity_hash']
            or approval.get('enabled_directory') != str(enabled) or approval.get('confirmation_channel') != '/dev/tty'
            or approval.get('scope') != runner.APPROVAL_SCOPE):
        _fail('restore_provenance', 'approval_scope_mismatch')
    ledger = read_json(base / 'enabled/ledger.json')
    live = read_json(_regular(enabled / 'ledger.json'))
    entries = ledger.get('entries')
    if (not isinstance(entries, list) or len(entries) > len(live.get('entries', []))
            or entries != live['entries'][:len(entries)]
            or ledger.get('version') != 'continuation_v1_ledger_v1'
            or ledger.get('plan_identity_hash') != copied['identity_hash']
            or live.get('version') != ledger['version'] or live.get('plan_identity_hash') != copied['identity_hash']):
        _fail('restore_provenance', 'restored_ledger_not_current_prefix')
    claim_names = set()
    artifact_names = set()
    for index, entry in enumerate(entries):
        record = identity['schedule'][index]
        if (entry.get('local_index') != index or entry.get('schedule_index') != record['schedule_index']
                or entry.get('input_id') != record['input_id'] or entry.get('provider') != record['provider']
                or entry.get('request_payload_hash') != object_hash(runner._payload(identity, record))
                or entry.get('retry') is not False or not re.fullmatch(r'[0-9a-f]{32}', entry.get('attempt_id', ''))):
            _fail('restore_provenance', 'attempt_schedule_or_payload_mismatch')
        claim_name = entry['attempt_id'] + '.json'
        if claim_name in claim_names:
            _fail('restore_provenance', 'duplicate_attempt_id')
        claim_names.add(claim_name)
        claim = base / 'enabled/reservations' / claim_name
        if read_json(claim) != runner._claim(copied, entry) or sha256_file(claim) != sha256_file(_regular(enabled / 'reservations' / claim_name)):
            _fail('restore_provenance', 'reservation_claim_mismatch')
        for name, descriptor in entry['artifacts'].items():
            relative = f"attempts/{entry['attempt_id']}/{name}"
            if name not in {'raw.json', 'parsed.json', 'voxels.json'} or descriptor['path'] != relative:
                _fail('restore_provenance', 'unexpected_artifact')
            artifact_names.add(relative)
            for root in (base / 'enabled', enabled):
                artifact = _regular(root / relative)
                if sha256_file(artifact) != descriptor['sha256'] or artifact.stat().st_size != descriptor['bytes']:
                    _fail('restore_provenance', 'artifact_hash_or_size_mismatch')
    if {p.name for p in (base / 'enabled/reservations').glob('*')} != claim_names:
        _fail('restore_provenance', 'restored_claim_set_mismatch')
    if {p.relative_to(base / 'enabled').as_posix() for p in (base / 'enabled/attempts').rglob('*') if p.is_file()} != artifact_names:
        _fail('restore_provenance', 'restored_artifact_set_mismatch')
    for path in (base / 'pricing-revisions').glob('*.json'):
        if sha256_file(path) != sha256_file(_regular(canonical.parent / 'pricing-revisions' / path.name)):
            _fail('restore_provenance', 'pricing_revision_changed')
    if (Path(restored) / 'FILES.json').read_bytes() != (sidecars / 'FILES.json').read_bytes():
        _fail('restore_provenance', 'archive_and_sidecar_manifest_differ')
    snapshot = read_json(sidecars / 'snapshot.json')
    manifest = read_json(sidecars / 'FILES.json')
    if (snapshot['checkpoint_id'] != package['checkpoint_id']
            or snapshot['plan_identity_hash'] != copied['identity_hash']
            or snapshot['canonical_plan_sha256'] != sha256_file(canonical)
            or snapshot['archive_sha256'] != next(a['sha256'] for a in package['assets'] if a['name'] == package['archive'])
            or snapshot['file_count'] != len(manifest['files'])
            or snapshot['execution'] != runner.execution_summary(copied, ledger)):
        _fail('restore_provenance', 'snapshot_metadata_mismatch')
    return {'verifiedledger_sha256': sha256_file(base / 'enabled/ledger.json'),
            'attempted_inputs': len(entries), 'current_attempted_inputs': len(live['entries']),
            'current_ledger_extends_restored_prefix': True, 'claims': len(claim_names),
            'artifacts': len(artifact_names), 'plan_identity_hash': copied['identity_hash']}


def recover(checkpoint_directory, verification_directory, *, release_id=None, allow_create=False):
    """Verify existing assets, or finish a future draft only when authorized.

    verification_directory must be new. allow_create=False is strictly read-only
    against GitHub, including when an existing release has missing assets.
    """
    directory = Path(checkpoint_directory).absolute()
    attempt = writable(verification_directory)
    if attempt.exists():
        _fail('verification_directory', 'new_directory_required')
    attempt.mkdir(parents=True, exist_ok=False)
    stage = 'local_package'
    try:
        package = _package(directory)
        archive = next(a for a in package['assets'] if a['name'] == package['archive'])
        stage = 'preflight_restore'
        preflight = attempt / 'preflight-restored'
        restore_archive(directory / archive['name'], preflight, archive['sha256'])
        frozen._verify_restored_public(preflight)
        verify_restored_prefix(preflight, package, directory)
        stage = 'commit_ancestry'
        ancestry = _gh(stage, ['api', f"repos/{REPOSITORY}/compare/{package['target_commitish']}...{quote(BRANCH, safe='')}"])
        if ancestry.get('status') not in {'ahead', 'identical'} or ancestry.get('behind_by') != 0:
            _fail(stage, 'checkpoint_commit_not_on_dedicated_branch')
        release = _release(package, attempt, release_id, allow_create)
        current = _assets(release['id'], package, require_complete=not allow_create)
        uploaded = []
        for asset in package['assets']:
            if asset['name'] in current:
                continue
            stage = 'asset_upload'
            result = _gh(stage, ['api', '--method', 'POST',
                f"https://uploads.github.com/repos/{REPOSITORY}/releases/{release['id']}/assets?name={quote(asset['name'], safe='')}",
                '-H', 'Content-Type: application/octet-stream',
                '-H', f"Content-Length: {asset['bytes']}", '--input', str(directory / asset['name'])])
            if not isinstance(result.get('id'), int):
                _fail(stage, 'uploaded_asset_id_missing')
            uploaded.append({'name': asset['name'], 'asset_id': result['id']})
        current = _assets(release['id'], package, require_complete=True)
        download = attempt / 'redownload'
        download.mkdir()
        for asset in package['assets']:
            live = current[asset['name']]
            stage = 'asset_download'
            _gh(stage, ['api', f"repos/{REPOSITORY}/releases/assets/{live['id']}",
                        '-H', 'Accept: application/octet-stream'], output=download / asset['name'])
            if sha256_file(download / asset['name']) != asset['sha256'] or (download / asset['name']).stat().st_size != asset['bytes']:
                _fail(stage, 'redownload_hash_or_size_mismatch')
        stage = 'download_checksums'
        _checksums(download, package)
        stage = 'download_restore'
        restored = attempt / 'restored'
        restoration = restore_archive(download / archive['name'], restored, archive['sha256'])
        copied_rejection = frozen._verify_restored_public(restored)
        prefix = verify_restored_prefix(restored, package, download)
        # Prove GitHub did not change release identity or any asset during verify.
        final_release = _release(package, attempt, release['id'], False)
        final_assets = _assets(release['id'], package, require_complete=True)
        # Downloads legitimately update download_count/updated_at. Only immutable
        # identity and payload properties define whether the saved assets changed.
        stable = lambda rows: {name: {k: row.get(k) for k in ('id', 'name', 'size', 'digest', 'state')}
                               for name, row in rows.items()}
        if stable(final_assets) != stable(current):
            _fail('final_asset_identity', 'github_assets_changed_during_verification')
        result = {'status': 'GITHUB_VERIFIED', 'verified_at_utc': utc_now(),
            'checkpoint_directory': str(directory), 'verification_directory': str(attempt),
            'release_id': release['id'], 'tag': final_release['tag_name'], 'draft': True,
            'target_commitish': package['target_commitish'],
            'assets': [{**a, 'asset_id': current[a['name']]['id'], 'github_digest': current[a['name']]['digest']} for a in package['assets']],
            'uploaded_assets': uploaded, 'restoration': restoration, **prefix,
            'copied_plan_load_rejection': copied_rejection,
            'target_commit_on_dedicated_branch': True, 'research_api_requests_sent': 0}
        write_json(attempt / 'verification.json', result)
        return result
    except RecoveryError as exc:
        write_json(attempt / 'failure.json', {'status': 'FAILED', **exc.report, 'research_api_requests_sent': 0})
        raise
    except Exception as exc:
        error = RecoveryError(stage, 'local_verification_failed_' + type(exc).__name__)
        write_json(attempt / 'failure.json', {'status': 'FAILED', **error.report, 'research_api_requests_sent': 0})
        raise error from None
