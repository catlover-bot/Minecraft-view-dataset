"""Save-only supplemental GitHub recovery with explicit Goal006B / Goal006C schemas.

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
ASSET_NAMES = {'supplement-point-in-time.tar.gz', 'FILES.json', 'snapshot.json',
               'SOURCE_REFERENCE.json', 'RESTORE.md', 'SHA256SUMS'}
MAX_ASSET_BYTES = 1024**3


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
    source = read_json(_regular(directory / 'checkpoint.json'))
    version = source.get('version')
    kind = {'goal006b_local_checkpoint_package_v1': 'goal006b',
            'goal006c_local_checkpoint_package_v1': 'goal006c'}.get(version)
    if (kind is None or source.get('archive') != 'supplement-point-in-time.tar.gz'
            or source.get('status') != 'LOCAL_VERIFIED' or source.get('github_status') != 'NOT_UPLOADED'
            or source.get('checkpoint_directory') != str(directory)
            or not re.fullmatch(kind + r'-[0-9a-f]{12}-[0-9]{4}-[A-Za-z0-9-]+', source.get('checkpoint_id', ''))):
        _fail('local_package', 'invalid_supplemental_package_identity')
    snapshot = read_json(_regular(directory / 'snapshot.json'))
    target = snapshot.get('code_commit', '')
    if (snapshot.get('schema') != kind + '_local_checkpoint_v1'
            or snapshot.get('checkpoint_id') != source['checkpoint_id']
            or snapshot.get('plan_identity_hash') != source.get('plan_identity_hash')
            or not re.fullmatch(r'[0-9a-f]{40}', target)):
        _fail('local_package', 'invalid_snapshot_identity')
    package = {**source, 'repository': REPOSITORY, 'tag': 'checkpoint-' + source['checkpoint_id'],
               'target_commitish': target, 'package_kind': kind}
    assets = package.get('assets', [])
    if len(assets) != 6 or {a.get('name') for a in assets} != ASSET_NAMES:
        _fail('local_package', 'exact_six_assets_required')
    for asset in assets:
        path = _regular(directory / asset['name'])
        if not isinstance(asset.get('bytes'), int) or not 0 < asset['bytes'] <= MAX_ASSET_BYTES:
            _fail('local_package', 'asset_exceeds_1GiB_or_invalid_size')
        if sha256_file(path) != asset['sha256'] or path.stat().st_size != asset['bytes']:
            _fail('local_package', 'immutable_asset_hash_or_size_changed')
        if asset['name'] != package['archive']:
            scan_file(path)
    archive_sha = next(a['sha256'] for a in assets if a['name'] == package['archive'])
    if source.get('archive_sha256') != archive_sha or snapshot.get('archive_sha256') != archive_sha:
        _fail('local_package', 'archive_metadata_digest_mismatch')
    scan_file(directory / 'checkpoint.json')
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


def _release_body(package):
    return ('Immutable supplemental checkpoint. Save-only operation; no generation requests.\n'
            f"checkpoint_id: {package['checkpoint_id']}\n"
            f"plan_identity_hash: {package['plan_identity_hash']}\n"
            f"archive_sha256: {package['archive_sha256']}")


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
                       'body': _release_body(package)}
            path = attempt / 'release-request.json'
            write_json(path, payload)
            release = _gh('release_create', ['api', '--method', 'POST',
                         f'repos/{REPOSITORY}/releases', '--input', str(path)])
    if (release.get('draft') is not True or release.get('published_at') is not None
            or release.get('tag_name') != package['tag']
            or release.get('target_commitish') != package['target_commitish']
            or release.get('body') != _release_body(package)
            or not isinstance(release.get('id'), int)):
        _fail('release_identity', 'release_tag_target_draft_or_full_identity_mismatch')
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


def _modules(kind):
    if kind == 'goal006b':
        from ..goal006b import planning, runner
    elif kind == 'goal006c':
        from . import planning, runner
    else:
        _fail('restore_provenance', 'unsupported_supplement_schema')
    return planning, runner


def _source_reference(identity, kind):
    snapshots = [identity['source_snapshot']]
    if kind == 'goal006c':
        snapshots.append(identity['original_source_snapshot'])
    controls = {}
    for snapshot in snapshots:
        source = read_json(_regular(snapshot['plan_path']))
        source_directory = Path(source['identity']['enabled_directory'])
        expected = [(snapshot['plan_path'], snapshot['plan_sha256']),
                    (source_directory / 'ledger.json', snapshot['ledger_sha256']),
                    (source_directory / 'approval.json', snapshot['approval_sha256']),
                    (source['identity']['authority_path'], snapshot['authority_sha256'])]
        for path, digest in expected:
            if sha256_file(_regular(path)) != digest:
                _fail('restore_provenance', 'referenced_source_control_changed')
            controls[str(Path(path).absolute())] = digest
    original = snapshots[-1]
    result = {'schema': kind + '_preserved_source_reference_v1', 'repository': REPOSITORY,
        'original_checkpoint': identity.get('original_checkpoint_reference', identity['checkpoint_reference']),
        'source_plan_identity_hash': identity['source_snapshot']['plan_identity_hash'],
        'source_controls_sha256': controls, 'original_attempts': original['attempted_inputs'],
        'original_halt_reason': original['halt_reason'],
        'original_unknown_usage_reservation_reconciled': False,
        'scope': 'Original 781 attempts and earlier sources remain immutable referenced assets; no response bytes duplicated.'}
    if kind == 'goal006c':
        result.update(source_checkpoint=identity['checkpoint_reference'],
            original_plan_identity_hash=original['plan_identity_hash'],
            source_attempts=identity['source_snapshot']['attempted_inputs'],
            source_halt_reason=identity['source_snapshot']['halt_reason'],
            source_unknown_usage_reservation_reconciled=False,
            scope='Original 781 and supplemental 3 attempts remain immutable referenced assets; no response bytes duplicated.')
    return result


def verify_restored_prefix(restored, package, sidecars):
    """Validate exact supplemental schema, controls, claims and actual response bytes.

    Original 781 and earlier results are hash references; only the archived
    supplemental responses are parsed and checked, without historical metrics.
    """
    kind = package['package_kind']
    planning, runner = _modules(kind)
    restored, sidecars = Path(restored), Path(sidecars)
    base = restored / 'supplement'
    copied_path = base / 'plan.json'
    copied = read_json(_regular(copied_path)); identity = copied['identity']
    canonical = _regular(identity['plan_path'])
    live_plan = planning.load_plan(canonical)
    if (copied != live_plan or sha256_file(copied_path) != sha256_file(canonical)
            or copied['identity_hash'] != package['plan_identity_hash']
            or object_hash(identity) != copied['identity_hash']):
        _fail('restore_provenance', 'canonical_plan_mismatch')
    try:
        planning.load_plan(copied_path)
    except Exception as exc:
        from ...server_v2.generation import GenerationError
        if not isinstance(exc, GenerationError) or 'canonical' not in str(exc):
            _fail('restore_provenance', 'copied_plan_rejected_for_unexpected_reason')
        copied_rejection = str(exc)
    else:
        _fail('restore_provenance', 'copied_plan_accepted')
    enabled, live = runner.load_enabled(live_plan)
    if Path(enabled) != Path(identity['enabled_directory']):
        _fail('restore_provenance', 'canonical_enabled_directory_mismatch')
    originals = {'supplement/plan.json': canonical,
                 'supplement/authority.json': _regular(identity['authority_path'])}
    for name in ('approval.json', 'enabled.json'):
        originals['supplement/enabled/' + name] = _regular(enabled / name)
        if sha256_file(base / 'enabled' / name) != sha256_file(originals['supplement/enabled/' + name]):
            _fail('restore_provenance', 'immutable_approval_or_enabled_mismatch')
    if sha256_file(base / 'authority.json') != sha256_file(originals['supplement/authority.json']):
        _fail('restore_provenance', 'immutable_authority_mismatch')
    approval = read_json(base / 'enabled/approval.json')
    runner._approval(copied, approval, fresh=False)
    if (read_json(base / 'enabled/enabled.json') != {'api_allowed': True,
            'plan_identity_hash': copied['identity_hash'], 'approval_hash': object_hash(approval),
            'enabled_directory': str(enabled)}
            or read_json(base / 'authority.json') != runner._authority(copied, approval)):
        _fail('restore_provenance', 'authority_approval_binding_mismatch')
    ledger_path = base / 'enabled/ledger.json'; ledger = read_json(ledger_path)
    entries = ledger.get('entries')
    if (ledger.get('version') != kind + '_ledger_v1' or ledger.get('plan_identity_hash') != copied['identity_hash']
            or not isinstance(entries, list) or len(entries) > len(live['entries'])
            or entries != live['entries'][:len(entries)]
            or (len(entries) == len(live['entries']) and ledger != live)
            or (len(entries) < len(live['entries']) and ledger.get('halt_reason') is not None)):
        _fail('restore_provenance', 'restored_ledger_not_current_prefix')
    originals['supplement/enabled/ledger.json'] = enabled / 'ledger.json'
    allowed = set(originals)
    for name in ('run.lock', 'summary.json'):
        path = base / 'enabled' / name
        if path.exists():
            allowed.add('supplement/enabled/' + name)
            originals['supplement/enabled/' + name] = _regular(enabled / name)
    claim_names, artifact_names, raw_scopes = set(), set(), {}
    for index, entry in enumerate(entries):
        record, intent = identity['schedule'][index], identity['intents'][index]
        if (entry.get('local_index') != index or entry.get('schedule_index') != record['schedule_index']
                or entry.get('input_id') != record['input_id'] or entry.get('provider') != record['provider']
                or entry.get('request_payload_hash') != object_hash(runner.base._payload(identity, record))
                or entry.get('intent') != intent['kind'] or entry.get('source_attempt_id') != intent['source_attempt_id']
                or entry.get('retry') is not (intent['kind'] == 'manual_auth_retry')
                or not re.fullmatch(r'[0-9a-f]{32}', entry.get('attempt_id', ''))):
            _fail('restore_provenance', 'attempt_schedule_or_payload_mismatch')
        claim_name = entry['attempt_id'] + '.json'
        if claim_name in claim_names:
            _fail('restore_provenance', 'duplicate_attempt_id')
        claim_names.add(claim_name)
        relative = 'supplement/enabled/reservations/' + claim_name
        claim = restored / relative
        original = _regular(enabled / 'reservations' / claim_name)
        if read_json(claim) != runner._claim(copied, entry) or sha256_file(claim) != sha256_file(original):
            _fail('restore_provenance', 'reservation_claim_mismatch')
        originals[relative] = original; allowed.add(relative)
        for name, descriptor in entry['artifacts'].items():
            relative = f"attempts/{entry['attempt_id']}/{name}"
            if name not in {'raw.json', 'parsed.json', 'voxels.json'} or descriptor['path'] != relative:
                _fail('restore_provenance', 'unexpected_artifact')
            artifact_names.add(relative)
            for root in (base / 'enabled', enabled):
                artifact = _regular(root / relative)
                if sha256_file(artifact) != descriptor['sha256'] or artifact.stat().st_size != descriptor['bytes']:
                    _fail('restore_provenance', 'artifact_hash_or_size_mismatch')
            archive_name = 'supplement/enabled/' + relative
            originals[archive_name] = enabled / relative; allowed.add(archive_name)
            if name == 'raw.json':
                raw_scopes[archive_name] = {'provider': entry['provider'], 'sha256': descriptor['sha256'],
                    'accepted_models': identity['pricing'][entry['provider']]['accepted_response_model_ids']}
        runner.base._validate_response_artifacts(base / 'enabled', identity, record, entry)
    for index, path in enumerate(sorted((base / 'pricing-revisions').glob('*')), 1):
        if path.name != f'{index:04d}.json':
            _fail('restore_provenance', 'pricing_revision_sequence_mismatch')
        relative = 'supplement/pricing-revisions/' + path.name
        original = _regular(canonical.parent / 'pricing-revisions' / path.name)
        if sha256_file(path) != sha256_file(original):
            _fail('restore_provenance', 'pricing_revision_changed')
        originals[relative] = original; allowed.add(relative)
    allowed.add('SOURCE_REFERENCE.json')
    if {p.relative_to(restored).as_posix() for p in restored.rglob('*') if p.is_file()} != allowed | {'FILES.json'}:
        _fail('restore_provenance', 'restored_exact_allowlist_mismatch')
    if (restored / 'FILES.json').read_bytes() != (sidecars / 'FILES.json').read_bytes():
        _fail('restore_provenance', 'archive_and_sidecar_manifest_differ')
    snapshot = read_json(sidecars / 'snapshot.json'); manifest = read_json(sidecars / 'FILES.json')
    if (len(manifest['files']) != len(allowed)
            or {row['path'] for row in manifest['files']} != allowed):
        _fail('restore_provenance', 'manifest_exact_allowlist_mismatch')
    reference = read_json(restored / 'SOURCE_REFERENCE.json')
    expected_reference = _source_reference(identity, kind)
    if (reference != expected_reference or reference != read_json(sidecars / 'SOURCE_REFERENCE.json')
            or snapshot.get('source_reference') != reference):
        _fail('restore_provenance', 'source_reference_mismatch')
    original_hashes = {str(path): sha256_file(restored / name) for name, path in originals.items()}
    if snapshot.get('original_sha256') != original_hashes:
        _fail('restore_provenance', 'snapshot_original_hash_map_mismatch')
    execution = runner.summary(copied, ledger)
    if (snapshot['checkpoint_id'] != package['checkpoint_id']
            or snapshot['plan_identity_hash'] != copied['identity_hash']
            or snapshot['canonical_plan_sha256'] != sha256_file(canonical)
            or snapshot['code_commit'] != identity['commit']
            or snapshot['file_count'] != len(manifest['files'])
            or snapshot['execution'] != execution or package['execution'] != execution):
        _fail('restore_provenance', 'snapshot_metadata_mismatch')
    exceptions = []
    for row in manifest['files']:
        exceptions.extend(frozen.scan_public_file(restored / row['path'], raw_scopes.get(row['path'])))
    scan_file(restored / 'FILES.json')
    metadata = manifest['metadata']
    if (metadata.get('public_safety_scanner_exceptions') != exceptions
            or any(snapshot.get(key) != value for key, value in metadata.items() if key != 'public_safety_scanner_exceptions')):
        _fail('restore_provenance', 'archive_metadata_or_scanner_exception_mismatch')
    return {'verifiedledger_sha256': sha256_file(ledger_path), 'attempted_inputs': len(entries),
        'current_attempted_inputs': len(live['entries']), 'current_ledger_extends_restored_prefix': True,
        'claims': len(claim_names), 'artifacts': len(artifact_names), 'plan_identity_hash': copied['identity_hash'],
        'copied_plan_load_rejection': copied_rejection, 'package_kind': kind,
        'source_references_verified': True, 'public_safety_scan_verified': True}


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
        originals_before = {name: sha256_file(directory / name) for name in ASSET_NAMES | {'checkpoint.json'}}
        archive = next(a for a in package['assets'] if a['name'] == package['archive'])
        stage = 'preflight_restore'
        preflight = attempt / 'preflight-restored'
        restore_archive(directory / archive['name'], preflight, archive['sha256'])
        verify_restored_prefix(preflight, package, directory)
        stage = 'commit_ancestry'
        ancestry = _gh(stage, ['api', f"repos/{REPOSITORY}/compare/{package['target_commitish']}...{quote(BRANCH, safe='')}"])
        if (ancestry.get('status') not in {'ahead', 'identical'} or ancestry.get('behind_by') != 0
                or ancestry.get('base_commit', {}).get('sha') != package['target_commitish']):
            _fail(stage, 'checkpoint_commit_not_on_dedicated_branch')
        release = _release(package, attempt, release_id, allow_create)
        current = _assets(release['id'], package, require_complete=not allow_create)
        uploaded = []
        for asset in package['assets']:
            if asset['name'] in current:
                continue
            stage = 'asset_upload'
            if (sha256_file(_regular(directory / asset['name'])) != asset['sha256']
                    or (directory / asset['name']).stat().st_size != asset['bytes']):
                _fail(stage, 'local_asset_changed_before_upload')
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
        if originals_before != {name: sha256_file(directory / name) for name in originals_before}:
            _fail('final_local_identity', 'source_package_changed_during_verification')
        result = {'status': 'GITHUB_VERIFIED', 'verified_at_utc': utc_now(),
            'checkpoint_directory': str(directory), 'verification_directory': str(attempt),
            'release_id': release['id'], 'tag': final_release['tag_name'], 'draft': True,
            'target_commitish': package['target_commitish'],
            'assets': [{**a, 'asset_id': current[a['name']]['id'], 'github_digest': current[a['name']]['digest']} for a in package['assets']],
            'uploaded_assets': uploaded, 'restoration': restoration, **prefix,
            'archive': package['archive'],
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
